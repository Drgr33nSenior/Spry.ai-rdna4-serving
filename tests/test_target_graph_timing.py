"""Declared CPU model/event fixtures; no native runtime or device imports."""
import hashlib
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hooks = load(ROOT/'prototype/serving/target_graph_timing.py', 'target_hooks_fixture')
events = load(ROOT/'prototype/serving/device_phase_timing.py', 'target_events_fixture')
event_fixture = load(ROOT/'tests/test_device_phase_timing.py', 'target_cuda_fixture')


class Timer:
    def __init__(self, model, sources):
        self.cuda = event_fixture.Cuda()
        self.events = events.DevicePhaseEvents(self.cuda)
        self.manifest = dict(sources=sources)
        self.bindings = [dict(cycles=list(range(9, 41))) for _ in range(4)]
        self.model, self.saved, self.emitted = model, [], []
        self.observer = NS()
        self.capture_wrapper = self.capture_forward
        self.observer.capture_forward = self.capture_wrapper

    def capture_forward(self, backend, shape, phase, warmup_step=None):
        role = 'draft' if backend._cuda_graph_runner.model_runner.is_draft_worker else 'target'
        if phase == 'warmup' and warmup_step == 0:
            self.prepare_backend(role, backend, shape)
        if phase == 'capture':
            self.events.arm_capture(role, shape.size)

    def prepare_backend(self, role, backend, shape):
        self.events.prepare_graph(role, shape.size)

    def bind(self, obj, name, wrapper, *, instance=False):
        assert name not in obj.__dict__
        self.saved.append((obj, name, wrapper, getattr(obj, name), instance))
        setattr(obj, name, wrapper)

    def restore_rows(self, rows):
        for obj, name, wrapper, original, instance in reversed(rows):
            if obj.__dict__.get(name) is not wrapper:
                raise ValueError('foreign hook')
            delattr(obj, name)

    def finish_capture(self):
        self.events.finish_capture()

    def emit(self, kind, row):
        self.emitted.append((kind, row))


class TargetGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sources, self.modules = {}, {}

        def native(name, text):
            path = Path(self.tmp.name)/(name+'.py')
            path.write_text(text)
            module = load(path, 'target_native_'+name)
            pin = dict(module=module.__name__, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.sources[name], self.modules[name] = pin, module
            return module

        linear = native('head_linear', 'class LinearBase:pass\nclass ColumnParallelLinear(LinearBase):\n def forward(self,x):return x\n')
        comm = native('comm', 'class GroupCoordinator:\n world_size=2\n failure=None\n def all_reduce(self,x):\n  if self.failure is not None:raise self.failure\n  return x\n')
        mlp = native('gdn_mlp', 'class Qwen2MoeMLP:\n def forward(self,x):return self.group.all_reduce(self.b.forward(self.a.forward(x)))\n')
        model = native('head_model', 'class Qwen3_5GatedDeltaNet:\n def forward(self,x):return self.proj.forward(x)\nclass Qwen3_5AttentionDecoderLayer:\n def self_attention(self,x):return self.proj.forward(x)\ndef get_parallel():return PARALLEL\n')
        group = comm.GroupCoordinator()
        model.PARALLEL = NS(tp_group=group, attn_tp_group=group)
        self.objects = []
        self.family = dict(mlp=[], gdn=[], attention=[])
        for family, cls, count in [('mlp', mlp.Qwen2MoeMLP, 64), ('gdn', model.Qwen3_5GatedDeltaNet, 48),
                                   ('attention', model.Qwen3_5AttentionDecoderLayer, 16)]:
            for i in range(count):
                obj = cls()
                self.family[family].append(obj)
                name = family+'.'+str(i)
                self.objects.append((name, obj))
                if family == 'mlp':
                    obj.a, obj.b, obj.group = linear.ColumnParallelLinear(), linear.ColumnParallelLinear(), group
                    self.objects += [(name+'.a', obj.a), (name+'.b', obj.b)]
                else:
                    obj.proj = linear.ColumnParallelLinear()
                    self.objects.append((name+'.proj', obj.proj))
        self.model = NS(named_modules=lambda: iter(self.objects))
        self.timer = Timer(self.model, self.sources)
        self.hook = hooks.TargetGraphHooks.__new__(hooks.TargetGraphHooks)
        self.hook.COMM_SOURCE = self.sources['comm']
        hooks.TargetGraphHooks.__init__(self.hook, self.timer, self.source_module, self.check_function)
        self.backends = {role: NS(_cuda_graph_runner=NS(model_runner=NS(
            is_draft_worker=role == 'draft', model=self.model))) for role in ('draft', 'target')}

    def source_module(self, source):
        module = sys.modules[source['module']]
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
        assert module.__file__ == source['path']
        return module

    def check_function(self, function, source, qualname):
        module = self.source_module(source)
        assert inspect.isfunction(function) and function.__globals__ is module.__dict__
        assert function.__qualname__ == qualname
        actual = module
        for part in qualname.split('.'):
            actual = getattr(actual, part)
        assert actual is function

    def forward(self):
        sentinel = object()
        for family, objects in self.family.items():
            method = 'self_attention' if family == 'attention' else 'forward'
            for obj in objects:
                self.assertIs(getattr(obj, method)(sentinel), sentinel)

    def prepare(self):
        for role in ('target', 'draft'):
            backend = self.backends[role]
            for size in (2, 1):
                shape = NS(size=size)
                self.timer.observer.capture_forward(backend, shape, 'warmup', 0)
                if role == 'target':
                    self.forward()
                self.timer.cuda.capture = True
                try:
                    self.timer.observer.capture_forward(backend, shape, 'capture')
                    if role == 'target':
                        self.forward()
                        parts = ('head', 'gather')
                    else:
                        parts = self.timer.events.GRAPH_PARTS['draft']
                    for part in parts:
                        self.timer.events.capture_end(self.timer.events.capture_begin(part))
                finally:
                    self.timer.cuda.capture = False
        self.timer.finish_capture()
        for name, pair in self.timer.events.pairs.items():
            part = name[-1] if isinstance(name, tuple) else name
            pair[0].elapsed = {'cycle': 2000, 'graph': 1000}.get(part, .01)

    def cycle(self, ordinal, cycle):
        timer = self.timer.events
        timer.start_cycle(dict(ordinal=ordinal, cycle=cycle, selected=True))
        for role in ('draft', 'target'):
            timer.end_graph(timer.begin_graph(role, 1))
        for part in timer.EAGER_PARTS:
            if part != 'cycle':
                timer.end(timer.begin(part))
        timer.finish_cycle(True)

    def test_actual_model_boundaries_return_values_and_both_bucket_capture(self):
        self.prepare()
        proof = self.timer.emitted[-1]
        self.assertEqual(proof[0], 'target_groups')
        self.assertTrue(proof[1]['capture_hooks_restored'])
        counts = {family: sum(row['family'] == family for row in self.hook.parts.values())
                  for family in ('mlp', 'gdn', 'attention', 'projection', 'comm')}
        self.assertEqual(counts, dict(mlp=64, gdn=48, attention=16, projection=192, comm=64))
        self.assertEqual(self.timer.saved, [])
        self.assertTrue(all('forward' not in obj.__dict__ for _, obj in self.objects))

    def test_only_eight_frozen_cycles_are_retained(self):
        self.prepare()
        for ordinal in range(4):
            for cycle in range(1, 41):
                self.cycle(ordinal, cycle)
        rows = self.timer.events.snapshot()['records']
        self.assertEqual(len(rows), 8)
        self.assertEqual([(r['ordinal'], r['cycle']) for r in rows],
                         [(i, n) for i in range(4) for n in (17, 33)])
        self.assertTrue(all(r['status'] == 'COMPLETE' for r in rows))

    def test_wrong_dense_layer_count_is_refused(self):
        self.objects[:] = [row for row in self.objects if row[0] != 'mlp.63']
        with self.assertRaisesRegex(ValueError, 'dense target layer counts'):
            self.prepare()

    def test_source_substitution_is_refused(self):
        self.sources['head_model'] = dict(self.sources['head_model'], sha256='0'*64)
        with self.assertRaises(AssertionError):
            self.prepare()

    def test_missing_bucket_discovery_is_refused(self):
        self.prepare()
        self.hook.seen.pop(next(k for k in self.hook.seen if k[0] == 1))
        with self.assertRaisesRegex(ValueError, 'bucket discovery coverage'):
            self.hook.check_coverage()

    def test_authentic_wrong_method_from_same_module_is_refused(self):
        module = self.modules['head_model']
        module.Qwen3_5GatedDeltaNet.forward = module.Qwen3_5AttentionDecoderLayer.self_attention
        with self.assertRaises(AssertionError):
            self.prepare()

    def test_reordered_capture_is_refused(self):
        original = self.forward
        def reordered():
            if self.timer.cuda.capture:
                self.family['mlp'].reverse()
            original()
        self.forward = reordered
        with self.assertRaisesRegex(ValueError, 'ordered target call transcript'):
            self.prepare()

    def test_changed_communication_parent_is_refused(self):
        self.timer.observer.capture_forward(self.backends['target'], NS(size=2), 'warmup', 0)
        self.forward()
        self.timer.cuda.capture = True
        self.timer.observer.capture_forward(self.backends['target'], NS(size=2), 'capture')
        first = self.family['mlp'][0]
        original = first.a.forward
        def moved(x):
            first.group.all_reduce(x)
            return original(x)
        first.a.forward = moved
        with self.assertRaisesRegex(ValueError, 'ordered target call transcript'):
            self.forward()

    def test_native_exception_identity_is_preserved(self):
        self.timer.observer.capture_forward(self.backends['target'], NS(size=2), 'warmup', 0)
        failure = RuntimeError('native failure')
        group = self.modules['head_model'].PARALLEL.tp_group
        group.failure = failure
        with self.assertRaises(RuntimeError) as caught:
            group.all_reduce(object())
        self.assertIs(caught.exception, failure)


if __name__ == '__main__':
    unittest.main(verbosity=2)
