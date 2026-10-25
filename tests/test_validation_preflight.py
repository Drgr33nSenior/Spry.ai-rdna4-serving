"""Source-only compiler and startup producer/reader regression gate.

No Torch, Triton, SGLang or native/device operations are imported or executed.
The width corpus is frozen failure-derived source, not a pass for a live packet.
Optionally exercise an actual worker producer with SPRY_PREFLIGHT_WORKER_SOURCE
and its independently supplied SPRY_PREFLIGHT_WORKER_SHA256 under the declared
runtime Python. The reader corpus and all negative cases remain unchanged.
"""
import ast
import builtins
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load(ROOT / 'prototype/serving/verify_decode_adapter.py', 'preflight_adapter')
corpus = load(ROOT / 'tests/fixtures/validation_preflight.py', 'preflight_corpus')


def functions(raw, names, namespace, filename):
    """Execute the exact named source bodies with declared dependency stubs."""
    tree = ast.parse(raw, filename=filename)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in nodes) != sorted(names):
        raise ValueError('missing or duplicate actual producer/helper function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec',
                 dont_inherit=True), namespace)
    return namespace


def compiler_fixture(raw, filename):
    # Only the import boundary is stubbed. Compile and execute the WHOLE module,
    # including its future statement, global import and complete class body.
    torch = ModuleType('torch')
    torch.Tensor = object
    torch.add = lambda value, increment: value + increment
    standard_import = builtins.__import__
    def declared_import(name, *args, **kwargs):
        return torch if name == 'torch' else standard_import(name, *args, **kwargs)
    namespace = {'__builtins__': dict(vars(builtins), __import__=declared_import)}
    exec(compile(raw, filename, 'exec', dont_inherit=True), namespace)
    return namespace['NativeBackend']


class CompilerContextTests(unittest.TestCase):
    RAW = (b'from __future__ import annotations\n'
           b'import torch\n\n'
           b'class NativeBackend:\n'
           b'    def forward_extend(self, value: torch.Tensor):\n'
           b'        return torch.add(value, 2)\n')

    def test_actual_guard_accepts_whole_module_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native_backend.py'
            path.write_bytes(self.RAW)
            cls = compiler_fixture(self.RAW, str(path))
            self.assertIs(adapter.checked_method(cls, 'forward_extend', {'path': str(path)}),
                          cls.forward_extend)
            # This reproduces the discarded compiler context: even preserving
            # locations cannot replace compiling the consumed complete module.
            class_node = ast.parse(self.RAW).body[-1]
            isolated = compile(ast.Module(body=[class_node], type_ignores=[]),
                               str(path), 'exec', dont_inherit=True)
            isolated_class = next(c for c in isolated.co_consts
                                  if isinstance(c, type(isolated)) and c.co_name == cls.__name__)
            isolated_method = next(c for c in isolated_class.co_consts
                                   if isinstance(c, type(isolated)) and c.co_name == 'forward_extend')
            self.assertNotEqual(isolated_method, cls.forward_extend.__code__)

    def test_actual_guard_rejects_body_mutation_compiled_from_full_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native_backend.py'
            path.write_bytes(self.RAW)
            # Compile the mutation. Do not assume small integer 2 is a member
            # of co_consts: Python 3.14 can emit LOAD_SMALL_INT instead.
            changed = self.RAW.replace(b'torch.add(value, 2)', b'torch.add(value, 3)')
            cls = compiler_fixture(changed, str(path))
            with self.assertRaisesRegex(ValueError, 'native method changed'):
                adapter.checked_method(cls, 'forward_extend', {'path': str(path)})

    def test_actual_guard_rejects_missing_or_duplicate_source_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native_backend.py'
            cls = compiler_fixture(self.RAW, str(path))
            for raw in (self.RAW.replace(b'class NativeBackend:', b'class OtherBackend:'),
                        self.RAW + b'\nclass NativeBackend:\n    pass\n'):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(ValueError, 'native method source'):
                        adapter.checked_method(cls, 'forward_extend', {'path': str(path)})


class TensorMetadata:
    """Descriptors only; all device observations are explicit CPU substitutes."""
    def __init__(self, count, device, pointer=0x1000000):
        self.count, self.device, self.pointer = count, device, pointer
        self.shape, self.dtype, self.is_cuda = (count,), 'torch.int64', True
        self.strides, self.bytes, self.contiguous = (1,), 8, True

    def numel(self): return self.count
    def data_ptr(self): return self.pointer
    def stride(self): return self.strides
    def element_size(self): return self.bytes
    def is_contiguous(self): return self.contiguous


class WidthReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = ROOT / 'tests/fixtures/validation_preflight.py'
        source = os.environ.get('SPRY_PREFLIGHT_WORKER_SOURCE')
        pin = os.environ.get('SPRY_PREFLIGHT_WORKER_SHA256')
        if bool(source) != bool(pin):
            raise ValueError('actual worker source and independent SHA256 must be supplied together')
        if source:
            path = Path(source)
            if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
                raise ValueError('actual worker source must be a canonical regular file')
            cls.source_path = path
            cls.raw = path.read_bytes()
            if hashlib.sha256(cls.raw).hexdigest() != pin:
                raise ValueError('actual worker source SHA256 differs')
        else:
            cls.raw = cls.source_path.read_bytes()

    def produce(self, width, rank, context=32768, *, original=False, mutate=None):
        arm = 'control' if width == 8 else 'mtp'
        arm_contract = dict(enabled=True, speculative_num_draft_tokens=width,
                            speculative_dflash_block_size=width, speculative_algorithm='DFLASH')
        contract = NS(precision_profile=lambda: dict(precision='fp8', context_length=context),
                      full_enabled_benchmark=lambda: True,
                      contract_for_arm=lambda selected: arm_contract if selected == arm else None)
        manifest = dict(mode='declared-source-fixture', arm=arm, session_sha256='a' * 64,
                        spec_sha256='b' * 64, source_manifest_sha256='c' * 64,
                        evidence_dir='/DECLARED-SOURCE-FIXTURE-NO-WRITES',
                        resolved_contract=dict(context_length=context),
                        sources={name: dict(module='fixture_' + name, sha256='d' * 64)
                                 for name in ('head_linear', 'fp8_config', 'verify_decode_adapter')})
        worker = dict(rank=rank, pid=100 + rank, start_ticks=200 + rank,
                      boot_id='declared-source-fixture', identity_sha256='e' * 64)
        indices = TensorMetadata(width * context, 'cuda:' + str(rank))
        captured = NS(buffers={1: dict(indices=indices)}, buffer_widths={1: width}, enabled=True,
                      eligible_host_calls=16, capture_host_calls=16, layers=set(range(16)),
                      metadata_build_host_calls=1, metadata_reuse_host_calls=15,
                      metadata_capture_build_host_calls=1)
        if mutate:
            mutate(captured, indices)
        linear_name = manifest['sources']['head_linear']['module']
        method_name = manifest['sources']['fp8_config']['module']
        linear = type('RowParallelLinear', (), {'__module__': linear_name})()
        linear.quant_method = type('Fp8LinearMethod', (), {'__module__': method_name})()
        linear.weight = NS(dtype='torch.float8_e4m3fn', shape=(1, 1), device='cuda:' + str(rank),
                           is_cuda=True, is_contiguous=lambda: True)
        modules = {linear_name: NS(RowParallelLinear=type(linear)),
                   method_name: NS(Fp8LinearMethod=type(linear.quant_method))}
        runner = NS(model=NS(model=NS(layers=[NS(mlp=NS(down_proj=linear))])),
                    model_config=NS(quantization='fp8'))
        outputs = {}
        namespace = dict(Path=Path, WorkerEntryError=ValueError,
                         _native_module=lambda source: modules[source['module']],
                         _source_module=lambda source: NS(CURRENT=captured),
                         _precision_loading_policy=lambda *args: {'declared_cpu_stub': True},
                         _atomic_json=lambda path, value: outputs.__setitem__(path.name, copy.deepcopy(value)))
        functions(self.raw, ['_require', '_full_enabled_indices', '_precision_weights'],
                  namespace, str(self.source_path))
        if original:
            functions(corpus.ORIGINAL_PRECISION_WEIGHTS, ['_precision_weights'],
                      namespace, 'retained-original-worker_entry.py')
        namespace['_precision_weights'](NS(tp_worker=NS(model_runner=runner)), manifest, worker,
                                        NS(validate_precision_weights=lambda *args, **kwargs: None), contract, {})
        proof = outputs[f'FULL-ENABLED-rank-{rank}.json']
        reader = functions((ROOT / 'tests/fixtures/validation_preflight.py').read_bytes(),
                           ['_need', 'validate_full_enabled_record'],
                           {'_mtp_contract': lambda: contract}, 'retained-serving_integration.py')
        return proof, manifest, worker, reader['validate_full_enabled_record']

    def test_retained_original_reproduces_width4_startup_refusal(self):
        self.produce(8, 0, original=True)
        with self.assertRaisesRegex(ValueError, '64K split-KV indices geometry differs'):
            self.produce(4, 0, original=True)

    def test_actual_startup_producer_to_strict_reader_all_widths_contexts_ranks(self):
        for width in (8, 4):
            for context in (32768, 65536):
                for rank in (0, 1):
                    with self.subTest(width=width, context=context, rank=rank):
                        proof, manifest, worker, consume = self.produce(width, rank, context)
                        self.assertEqual(proof['indices_elements'], [width * context])
                        self.assertIs(consume(proof, manifest, worker), proof)

    def test_producer_rejects_wrong_width_geometry_dtype_device_and_alias(self):
        changes = {
            'width': lambda a, t: a.buffer_widths.update({1: 8}),
            'count': lambda a, t: setattr(t, 'count', 8 * 32768),
            'shape': lambda a, t: setattr(t, 'shape', (2, t.count // 2)),
            'stride': lambda a, t: setattr(t, 'strides', (2,)),
            'dtype': lambda a, t: setattr(t, 'dtype', 'torch.int32'),
            'bytes': lambda a, t: setattr(t, 'bytes', 4),
            'device': lambda a, t: setattr(t, 'device', 'cuda:1'),
            'cpu': lambda a, t: setattr(t, 'is_cuda', False),
            'contiguous': lambda a, t: setattr(t, 'contiguous', False),
            'bool-count': lambda a, t: setattr(t, 'count', True),
            'empty': lambda a, t: a.buffers.clear(),
            'unaligned': lambda a, t: setattr(t, 'pointer', t.pointer + 1),
            'alias': lambda a, t: (a.buffers.update({2: dict(indices=t)}), a.buffer_widths.update({2: 4})),
        }
        for label, mutate in changes.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.produce(4, 0, mutate=mutate)

    def test_reader_rejects_wrong_arm_width_and_worker_identity(self):
        for field, value in (('indices_elements', [8 * 32768]), ('block_size', 8),
                             ('worker_identity_sha256', 'f' * 64), ('rank', 1)):
            with self.subTest(field=field):
                proof, manifest, worker, consume = self.produce(4, 0)
                proof[field] = value
                with self.assertRaises(ValueError):
                    consume(proof, manifest, worker)


if __name__ == '__main__':
    unittest.main(verbosity=2)
