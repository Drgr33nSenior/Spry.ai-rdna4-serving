"""Adapter lifecycle and graph dependency tests with declared CPU tensor stand-ins.

No Torch, Triton, SGLang, native arithmetic or GPU timing is exercised here.
The original adapter is the trace oracle; simulated graphs replay recorded
operations against mutable input buffers, without rerunning adapter Python.
"""
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(ROOT / 'prototype/serving/verify_decode_adapter.py', 'reuse_candidate')
baseline_path = ROOT / 'tests/fixtures/verify-metadata/original_adapter.py'
baseline = load(baseline_path, 'reuse_original')


class Tensor:
    def __init__(self, torch, shape, dtype='torch.int32', *, values=None,
                 storage=None, positions=None):
        self.torch, self.dtype, self.device = torch, dtype, 'cpu-stand-in'
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.storage = (list(values) if values is not None else
                        storage if storage is not None else
                        [-1] * self.shape[0] if len(self.shape) == 1 else [])
        self.positions = (list(positions) if positions is not None else
                          list(range(self.shape[0])) if len(self.shape) == 1 else [])
        self.trace = []

    def values(self):
        return [self.storage[p] for p in self.positions]

    def put(self, values):
        assert len(values) == len(self.positions)
        for p, value in zip(self.positions, values):
            self.storage[p] = value

    def __getitem__(self, key):
        positions = self.positions[key]
        if isinstance(key, int):
            positions = [positions]
        return Tensor(self.torch, len(positions), self.dtype,
                      storage=self.storage, positions=positions)

    def __sub__(self, other):
        result = Tensor(self.torch, 1, self.dtype)
        self.torch.submit(lambda: result.put([self.values()[0] - other.values()[0]]))
        return result

    def expand(self, count):
        assert len(self.positions) == 1
        return Tensor(self.torch, count, self.dtype, storage=self.storage,
                      positions=self.positions * count)

    def copy_(self, other):
        self.torch.submit(lambda: self.put(other.values()))
        return self

    def numel(self):
        return math.prod(self.shape)


class Torch:
    int32, int64, float32 = 'torch.int32', 'torch.int64', 'torch.float32'

    def __init__(self):
        self.capturing, self.operations, self.stream = False, [], 7
        self.cuda = SimpleNamespace(
            is_current_stream_capturing=lambda: self.capturing,
            current_stream=lambda device: SimpleNamespace(cuda_stream=self.stream))

    def submit(self, operation):
        operation()
        if self.capturing:
            self.operations.append(operation)

    def empty(self, shape, *, device, dtype):
        return Tensor(self, shape, dtype)

    def zeros(self, shape, *, device, dtype):
        return Tensor(self, shape, dtype, values=[0] * shape)

    def arange(self, start, stop, *, device, dtype):
        return Tensor(self, stop - start, dtype, values=range(start, stop))

    def add(self, left, right, *, out):
        self.submit(lambda: out.put([left.values()[0] + n for n in right.values()]))

    def cumsum(self, source, *, dim, out):
        assert dim == 0
        self.submit(lambda: out.put(list(itertools.accumulate(source.values()))))


def singleton_split(length):
    # CPU policy stand-in, not a proof of native log2 rounding. Native evidence
    # independently established that batching the eight rows is not equivalent.
    cdiv = lambda a, b: (a + b - 1) // b
    extended = int(32 * max(math.log2(length / 64), 1))
    chunk = cdiv(length, min(cdiv(extended, 2), 16))
    return max(1, cdiv(length, chunk))


class Translator:
    reads_are_translated = False

    def __init__(self, torch):
        self.torch, self.calls = torch, 0
        self.table = [list(range(1, 65536, 2)), list(range(65535, 0, -2))]

    def fill_packed_read_stream(self, *, req_pool_indices, seq_lens, indptr,
                                total_tokens, out):
        self.calls += 1
        def fill():
            pointers = indptr.values()
            assert pointers[-1] <= total_tokens == out.numel()
            for row, (request, length) in enumerate(zip(req_pool_indices.values(), seq_lens.values())):
                out.storage[pointers[row]:pointers[row + 1]] = self.table[request][:length]
        self.torch.submit(fill)


def decode(q, kb, vb, out, indptr, indices, partial, lse, splits, maximum,
           scale, ks, vs, **kwargs):
    assert (maximum, scale, ks, vs, kwargs) == (
        16, 0.0625, 1.0, 1.0, {'enable_lean': False, 'page_size': 1})
    def consume():
        pointers = indptr.values()
        out.trace.append((tuple(pointers), tuple(splits.values()), tuple(
            (indices.storage[pointers[row]], indices.storage[pointers[row + 1] - 1])
            for row in range(8))))
    q.torch.submit(consume)


class Base:
    def init_forward_metadata_in_graph(self, forward_batch):
        pass


class FakeBackend(Base):
    def init_forward_metadata(self, forward_batch):
        if getattr(self, 'initializer_failure', False):
            raise RuntimeError('CPU initializer failure')
        self.forward_metadata = SimpleNamespace()

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        if in_capture:
            self.forward_metadata = SimpleNamespace()

    def get_num_kv_splits(self, splits, lengths):
        assert splits.numel() == lengths.numel() == 1
        self.selector_calls += 1
        self.torch.submit(lambda: splits.put([singleton_split(lengths.values()[0])]))

    def forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
        self.torch.submit(lambda: self.events.append(('write', layer.layer_id)))
        result = self.extend_attention_fwd(*self.arguments, **kwargs)
        self.torch.submit(lambda: self.events.append(('attention', layer.layer_id)))
        return result


class Fixture:
    def __init__(self, module):
        self.torch = Torch()
        self.cls = type('FakeBackend', (Base,), {
            k: v for k, v in vars(FakeBackend).items() if k not in ('__dict__', '__weakref__')})
        source = dict(path=str(Path(__file__).resolve()),
                      sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.adapter = module.Adapter(dict(arm='mtp', sources={
            'triton_backend': source,
            'decode_attention_fwd': {'module': 'metadata_cpu_decode'}}), {'rank': 0}, self.cls, self.torch)
        self.backend = b = self.cls()
        b.torch, b.events, b.selector_calls = self.torch, [], 0
        b.forward_metadata = SimpleNamespace()
        b.decode_attention_fwd = decode
        b.extend_attention_fwd = lambda *a, **kw: 'native-fallback'
        b.kv_index_translator = Translator(self.torch)
        for key, value in dict(page_size=1, dcp_size=1, use_mla=False,
            enable_deterministic=False, static_kv_splits=False, split_tile_size=None,
            max_kv_splits=16, device_core_count=32, num_head=12, num_kv_head=2,
            max_context_len=32768).items():
            setattr(b, key, value)
        self.batch = SimpleNamespace(batch_size=1,
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
            req_pool_indices=Tensor(self.torch, 1, values=[0]),
            out_cache_loc=Tensor(self.torch, 8, values=range(1, 9)))
        bf16 = 'torch.bfloat16'
        self.q, self.k, self.v, self.out = [Tensor(self.torch, shape, bf16) for shape in
            ((8, 12, 256), (8, 2, 256), (8, 2, 256), (8, 12, 256))]
        self.ptr = Tensor(self.torch, 2, values=[0, 0])
        kb = Tensor(self.torch, (65536, 2, 256), bf16)
        b.arguments = (self.q, self.k, self.v, self.out, kb, kb,
            Tensor(self.torch, 2, values=[0, 8]), self.ptr,
            Tensor(self.torch, 32768), None, True, None, 8, 1.0, 1.0, 0.0625)

    def state(self, prefix, request):
        self.ptr.put([0, prefix])
        self.batch.req_pool_indices.put([request])

    def forward(self, *, initialize=True, layers=range(3, 64, 4)):
        if initialize:
            self.backend.init_forward_metadata(self.batch)
        for layer in layers:
            self.backend.forward_extend(self.q, self.k, self.v,
                SimpleNamespace(layer_id=layer), self.batch)

    def metadata(self):
        b = self.adapter.buffers[id(self.backend)]
        return tuple(b['indptr'].values()), tuple(b['splits'].values()), tuple(b['indices'].values())


class MetadataTests(unittest.TestCase):
    def setUp(self):
        native = ModuleType('metadata_cpu_decode')
        native.decode_attention_fwd = decode
        self.patch = patch.dict(sys.modules, metadata_cpu_decode=native)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_original_fixture_identity(self):
        self.assertEqual(hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                         'ef340d6ed335b50638033d93834d194db3da6c2568ab3f5fac10beee1d83c465')

    def test_changing_prefixes_and_request_slots_match_original(self):
        old, new = Fixture(baseline), Fixture(candidate)
        for i, prefix in enumerate((0, 31, 127, 2040, 15360, 32760, 2040, 31, 15360)):
            for f in (old, new):
                f.state(prefix, i % 2)
                f.forward()
            self.assertEqual(new.metadata(), old.metadata())
            self.assertEqual(new.out.trace, old.out.trace)
            self.assertEqual(new.backend.events, old.backend.events)
        self.assertEqual(old.backend.selector_calls, 9 * 16 * 8)
        self.assertEqual(new.backend.selector_calls, 9 * 8)
        self.assertEqual(new.backend.kv_index_translator.calls, 9)
        self.assertEqual(new.adapter.metadata_reuse_host_calls, 9 * 15)

    def test_replay_uses_live_inputs_and_preserves_buffer_addresses(self):
        old, new = Fixture(baseline), Fixture(candidate)
        graphs = []
        identities = []
        for f in (old, new):
            f.forward()  # Allocate before capture, as native warm-up does.
            f.backend.init_forward_metadata_out_graph(f.batch, in_capture=True)
            for _ in range(2):
                f.backend.init_forward_metadata_in_graph(f.batch)
                f.forward(initialize=False)
            f.torch.operations = []
            f.torch.capturing = True
            f.backend.init_forward_metadata_in_graph(f.batch)
            f.forward(initialize=False)
            f.torch.capturing = False
            graphs.append(list(f.torch.operations))
            identities.append(tuple(id(t) for t in f.adapter.buffers[id(f.backend)].values()))
        self.assertLess(len(graphs[1]), len(graphs[0]))
        self.assertEqual(new.adapter.metadata_capture_build_host_calls, 1)
        for state, prefix in enumerate((0, 127, 32760, 31, 2040)):
            for f, operations, addresses in zip((old, new), graphs, identities):
                f.state(prefix, state % 2)
                f.backend.init_forward_metadata_out_graph(f.batch)
                for _ in range(2):
                    for operation in operations:
                        operation()
                self.assertEqual(tuple(id(t) for t in f.adapter.buffers[id(f.backend)].values()), addresses)
            self.assertEqual(new.metadata(), old.metadata())
            self.assertEqual(new.out.trace[-32:], old.out.trace[-32:])
        self.assertEqual(new.adapter.metadata_capture_build_host_calls, 1)

    def test_no_initializer_means_no_reuse(self):
        f = Fixture(candidate)
        f.forward(initialize=False)
        self.assertEqual(f.backend.selector_calls, 128)
        self.assertEqual(f.adapter.metadata_reuse_host_calls, 0)

    def test_capture_transition_never_reuses_eager_only_preparation(self):
        f = Fixture(candidate)
        f.forward()
        f.torch.operations = []
        f.torch.capturing = True
        # Even a caller that omits the in-graph hook must record preparation.
        f.forward(initialize=False)
        f.torch.capturing = False
        self.assertEqual(f.adapter.metadata_capture_build_host_calls, 1)
        f.state(127, 1)
        for operation in f.torch.operations:
            operation()
        self.assertEqual(f.adapter.buffers[id(f.backend)]['lengths'].values(), list(range(128, 136)))

    def test_initializer_failure_clears_previous_scope(self):
        f = Fixture(candidate)
        f.forward()
        f.backend.initializer_failure = True
        with self.assertRaisesRegex(RuntimeError, 'CPU initializer failure'):
            f.backend.init_forward_metadata(f.batch)
        self.assertNotIn(id(f.backend), f.adapter.metadata_scopes)
        f.state(127, 1)
        f.forward(initialize=False, layers=[3])
        self.assertEqual(f.backend.selector_calls, 16)

    def test_native_initializer_source_mutation_refuses(self):
        f = Fixture(candidate)
        f.cls.init_forward_metadata = lambda *a: None
        source = {'path': str(Path(__file__).resolve())}
        with self.assertRaisesRegex(ValueError, 'native method changed'):
            candidate.checked_method(f.cls, 'init_forward_metadata', source)

    def test_changed_inputs_and_stream_refresh_metadata(self):
        for change in ('ptr', 'requests', 'translator', 'metadata', 'stream', 'cores'):
            f = Fixture(candidate)
            f.forward(layers=[3])
            if change == 'ptr':
                f.ptr = Tensor(f.torch, 2, values=[0, 127])
                args = list(f.backend.arguments); args[7] = f.ptr; f.backend.arguments = tuple(args)
            elif change == 'requests':
                f.batch.req_pool_indices = Tensor(f.torch, 1, values=[1])
            elif change == 'translator':
                f.backend.kv_index_translator = Translator(f.torch)
            elif change == 'metadata':
                f.backend.forward_metadata = SimpleNamespace()
            elif change == 'stream':
                f.torch.stream = 9
            else:
                f.backend.device_core_count = 16
            f.forward(initialize=False, layers=[7])
            self.assertEqual(f.backend.selector_calls, 16, change)

    def test_unsupported_layer_invalidates_scope_and_uses_native(self):
        f = Fixture(candidate)
        f.forward(layers=[3])
        f.batch.batch_size = 2
        f.forward(initialize=False, layers=[7])
        f.batch.batch_size = 1
        f.forward(initialize=False, layers=[11])
        self.assertEqual(f.backend.selector_calls, 16)
        self.assertEqual(f.adapter.metadata_reuse_host_calls, 0)

    def test_interleaved_unscoped_batch_cannot_resurrect_old_metadata(self):
        f = Fixture(candidate)
        f.forward(layers=[3])
        original = f.batch
        f.batch = SimpleNamespace(**vars(original))
        f.batch.req_pool_indices = Tensor(f.torch, 1, values=[1])
        f.state(127, 1)
        f.forward(initialize=False, layers=[7])
        f.batch = original
        f.state(31, 0)
        f.forward(initialize=False, layers=[11])
        self.assertEqual(f.backend.selector_calls, 24)
        self.assertEqual(f.adapter.buffers[id(f.backend)]['lengths'].values(), list(range(32, 40)))

    def test_noop_guard_rejects_effects_and_accepts_only_none(self):
        candidate.checked_noop(Base.init_forward_metadata_in_graph)
        def returns_value(self, batch):
            return 1
        def has_effect(self, batch):
            batch.changed = True
        async def asynchronous(self, batch):
            pass
        for method in (returns_value, has_effect, asynchronous):
            with self.assertRaises(ValueError):
                candidate.checked_noop(method)

    def test_restore_removes_inherited_hook_override(self):
        f = Fixture(candidate)
        f.forward()
        f.adapter.restore()
        self.assertNotIn('init_forward_metadata_in_graph', vars(f.cls))
        self.assertIs(f.cls.init_forward_metadata_in_graph, Base.init_forward_metadata_in_graph)
        self.assertIs(f.cls.init_forward_metadata, FakeBackend.init_forward_metadata)
        self.assertIs(f.cls.forward_extend, FakeBackend.forward_extend)
        self.assertFalse(f.adapter.metadata_scopes)
        f.adapter.restore()


if __name__ == '__main__':
    unittest.main(verbosity=2)
