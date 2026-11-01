"""Exercise complete observer modules with named CPU source/tensor substitutes."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_lifecycle_safety import SOURCE, identities, integration, probe
import serving_graph_observer as graph
import worker_entry


def loaded(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(path, name):
    return dict(path=str(path), module=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def contract(limit=8):
    return SimpleNamespace(settings=lambda: dict(phase='cpu', allocator_snapshot=False),
        BATCH_PHASES={'cpu'}, commit_observation_limit=lambda: limit,
        host_path_cleanup_enabled=lambda arm: True, replay_serving_enabled=lambda arm: False)


@contextlib.contextmanager
def observed_graph(limit=8, rank=0):
    with tempfile.TemporaryDirectory(prefix='cleanup-graph-') as temporary:
        root = Path(temporary).resolve()
        backend_path = root / 'backend.py'
        backend_path.write_text('class FullCudaGraphBackend:\n    pass\n')
        backend_module = loaded(graph.MODULE, backend_path)
        backend = backend_module.FullCudaGraphBackend()
        runner = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False,
            attn_backend=object(), spec_algorithm=SimpleNamespace(is_none=lambda: False, is_dflash=lambda: True)),
            capture_bs=[1, 2], ragged_verify_mode=False, require_mlp_tp_gather=False, captured_req_width=8, raw_bs=1)
        backend._cuda_graph_runner = runner
        helper_path = root / 'commit.py'
        helper_path.write_text('def scatter_mamba_states_after_mtp_verify():\n    pass\n')
        helper_module = loaded('cleanup_native_commit', helper_path)
        helper = helper_module.scatter_mamba_states_after_mtp_verify
        manifest, worker = identities(root, rank)
        manifest['preweight'] = dict(graph_observer=source(SOURCE / 'serving_graph_observer.py', 'serving_graph_observer'))
        manifest['sources']['graph_observer'] = manifest['preweight']['graph_observer']
        log = io.StringIO()
        with patch.dict(sys.modules, {graph.MODULE: backend_module, 'cleanup_native_commit': helper_module,
                                      'mtp_contract': contract(limit)}), contextlib.redirect_stdout(log):
            observer = graph.Observer(rank, root, source(backend_path, graph.MODULE)['sha256'], 'mtp')
            try:
                yield observer, backend, helper, manifest, worker, root, log
            finally:
                observer.finish(manifest, worker)


def capture(observer, backend):
    for size in (1, 2):
        shape = SimpleNamespace(size=size)
        observer.capture_forward(backend, shape, 'warmup', 0)
        observer.capture_forward(backend, shape, 'warmup', 1)
        observer.capture_forward(backend, shape, 'capture')


def bind_commit(observer, helper):
    observer.sources = {'replay_scatter': source(Path(helper.__code__.co_filename), helper.__module__)}


class GraphSafety(unittest.TestCase):
    def test_healthy_capture_replay_commit_and_final_health_match(self):
        with observed_graph() as (observer, backend, helper, manifest, worker, root, log):
            capture(observer, backend)
            bind_commit(observer, helper)
            for size in (1, 2):
                backend._cuda_graph_runner.raw_bs = size
                observer.backend_replayed(backend, SimpleNamespace(size=size), SimpleNamespace(batch_size=size))
                observer.commit_complete(backend._cuda_graph_runner.model_runner.attn_backend, helper)
            observer.finish(manifest, worker)
            health = json.loads((root / 'OBSERVER-HEALTH-graph-rank-0.json').read_bytes())
            counters = graph.validate_health(health, binding=manifest, worker=worker)
            self.assertEqual(counters, dict(captures=2, replays=2, commits=2, bs1_replays=1, bs2_replays=1))
            self.assertEqual(struct.unpack('<QQQQ', (root / 'graph-counter-rank-0.bin').read_bytes()), (2, 2, 1, 1))
            self.assertFalse(observer.health.failed)

    def test_capture_budget_saturates_without_raising(self):
        with observed_graph() as (observer, backend, helper, manifest, worker, root, log):
            capture(observer, backend)
            for _ in range(100):
                observer.capture_forward(backend, SimpleNamespace(size=1), 'warmup', 0)
            self.assertEqual(len(observer.events), 6)
            self.assertTrue(observer.health.saturated)
            self.assertEqual(log.getvalue().count('SPRY_OBSERVER_FAILED'), 1)

    def test_replay_and_commit_caps_leave_bounded_counters_and_failed_health(self):
        for kind in ('replay', 'commit'):
            with self.subTest(kind=kind), observed_graph(limit=2) as fixture:
                observer, backend, helper, manifest, worker, root, log = fixture
                capture(observer, backend)
                bind_commit(observer, helper)
                for _ in range(30):
                    if kind == 'replay':
                        observer.backend_replayed(backend, SimpleNamespace(size=1), SimpleNamespace(batch_size=1))
                    else:
                        observer.commit_complete(backend._cuda_graph_runner.model_runner.attn_backend, helper)
                self.assertEqual(observer.replays if kind == 'replay' else observer.commits, 2)
                self.assertTrue(observer.health.saturated)
                self.assertEqual(observer.bs1_replays, observer.replays)
                observer.finish(manifest, worker)
                health = json.loads((root / 'OBSERVER-HEALTH-graph-rank-0.json').read_bytes())
                with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                    graph.validate_health(health, binding=manifest, worker=worker)

    def test_each_pure_callback_contains_io_errors_and_disables_once(self):
        for method, args in (('capture_forward', ('warmup', 0)), ('backend_replayed', ()), ('commit_complete', ())):
            with self.subTest(method=method), observed_graph() as fixture:
                observer, backend, helper, manifest, worker, root, log = fixture
                with patch.object(observer, '_' + method, side_effect=OSError('private-error-text')) as callback:
                    for _ in range(2):
                        if method == 'capture_forward':
                            observer.capture_forward(backend, SimpleNamespace(size=1), *args)
                        elif method == 'backend_replayed':
                            observer.backend_replayed(backend, SimpleNamespace(size=1), SimpleNamespace(batch_size=1))
                        else:
                            observer.commit_complete(backend, helper)
                    callback.assert_called_once()
                self.assertTrue(observer.health.failed)
                self.assertNotIn('private-error-text', log.getvalue())

    def test_wrong_replay_identity_is_failed_evidence_not_scheduler_exception(self):
        with observed_graph() as (observer, backend, helper, manifest, worker, root, log):
            capture(observer, backend)
            observer.backend_replayed(backend, SimpleNamespace(size=2), SimpleNamespace(batch_size=1))
            self.assertTrue(observer.health.failed)
            self.assertEqual(observer.replays, 0)

    def test_initial_counter_io_failure_is_contained(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, mtp_contract=contract()), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch.object(graph.mmap, 'mmap', side_effect=OSError('mapping')):
                observer = graph.Observer(0, Path(temporary), '0' * 64, 'mtp')
            self.assertTrue(observer.health.failed)
            observer.backend_replayed(None, None, None)
            manifest, worker = identities(Path(temporary))
            manifest['preweight'] = dict(graph_observer={})
            observer.finish(manifest, worker)
            self.assertTrue(json.loads((Path(temporary) / 'OBSERVER-HEALTH-graph-rank-0.json').read_bytes())['observer_failed'])

    def test_original_control_signals_are_not_swallowed(self):
        with observed_graph() as fixture:
            observer = fixture[0]
            with patch.object(observer, '_backend_replayed', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    observer.backend_replayed(None, None, None)


class Subject:
    def __init__(self):
        self.calls = 0
        self.error = None
        self.result = object()

    def target_verify(self, b, ssm_states, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def gdn_fixture():
    # Native startup source/class/tensor discovery is outside this wrapper test.
    # begin() still checks and wraps a real bound method from its complete module.
    observer = graph.GDNObservation.__new__(graph.GDNObservation)
    observer.health = graph._health.Health('graph', 0)
    observer.subject = Subject()
    observer.fold = False
    observer.alignment = None
    observer.sources = dict(gdn_backend=dict(path=Subject.target_verify.__code__.co_filename))
    observer.saved = {}
    observer.active = None
    observer.record = dict(warmups=[], speculation=dict(algorithm='DFLASH'),
                           pool=dict(offsets=[0, 8], layer_ids=[0, 1]))
    observer.begin(1)
    return observer


class GdnSafety(unittest.TestCase):
    def test_healthy_forward_delegates_once_and_restores_method(self):
        observer = gdn_fixture()
        with patch.object(graph, 'tensor_metadata', side_effect=lambda value: dict(offset=value, dtype='bf16')):
            for offset in (0, 8):
                result = observer.subject.target_verify(0, offset, cache_steps=8)
                self.assertIs(result, observer.subject.result)
        self.assertEqual(observer.subject.calls, 2)
        self.assertEqual(observer.active['calls'], 2)
        observer.finish()
        self.assertNotIn('target_verify', observer.subject.__dict__)

    def test_metadata_failure_never_retries_or_skips_forward(self):
        observer = gdn_fixture()
        with patch.object(graph, 'tensor_metadata', side_effect=ValueError('metadata')) as metadata, \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(30):
                self.assertIs(observer.subject.target_verify(0, 0), observer.subject.result)
            metadata.assert_called_once()
        self.assertEqual(observer.subject.calls, 30)
        self.assertTrue(observer.health.failed)
        observer.restore()

    def test_original_forward_exception_propagates_identically(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled), contextlib.redirect_stdout(io.StringIO()):
                observer = gdn_fixture()
                if disabled:
                    observer.health.fail(ValueError('observation'))
                original_error = RuntimeError('forward')
                observer.subject.error = original_error
                with patch.object(graph, 'tensor_metadata', side_effect=lambda value: dict(offset=value)), \
                        self.assertRaises(RuntimeError) as caught:
                    observer.subject.target_verify(0, 0)
                self.assertIs(caught.exception, original_error)
                self.assertEqual(observer.subject.calls, 1)
                self.assertEqual(observer.health.failed, disabled)
                observer.restore()

    def test_layer_cap_saturates_without_stopping_forward(self):
        observer = gdn_fixture()
        with patch.object(graph, 'tensor_metadata', side_effect=lambda value: dict(offset=value)), \
                contextlib.redirect_stdout(io.StringIO()):
            for offset in range(30):
                observer.subject.target_verify(0, offset)
        self.assertEqual(observer.active['calls'], 2)
        self.assertEqual(observer.subject.calls, 30)
        self.assertTrue(observer.health.saturated)
        observer.restore()


@contextlib.contextmanager
def target_kv(rank=0):
    with tempfile.TemporaryDirectory(prefix='cleanup-kv-') as temporary:
        root = Path(temporary).resolve()
        backend_path = root / 'backend.py'
        backend_path.write_text('_FP8_TARGET_KV_OBSERVER = None\n')
        module = loaded('cleanup_kv_backend', backend_path)
        manifest, worker = identities(root, rank)
        manifest['sources']['triton_backend'] = source(backend_path, 'cleanup_kv_backend')
        log = io.StringIO()
        with patch.dict(sys.modules, cleanup_kv_backend=module), contextlib.redirect_stdout(log):
            restore = probe.install_target_kv_witness(manifest, worker)
            def call(layer=0):
                module._FP8_TARGET_KV_OBSERVER(None, SimpleNamespace(shape=[8]), None, None,
                                               SimpleNamespace(layer_id=layer), 'extend', False)
            try:
                yield call, restore, manifest, worker, module, root, log
            finally:
                restore()


class TargetKvSafety(unittest.TestCase):
    def test_routes_are_buffered_then_flushed_and_bound_to_health(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', return_value=dict(route='fixture')):
                call(0)
                call(0)
                call(1)
            self.assertFalse(list(root.glob('TARGET-KV-ROUTES*')))
            restore()
            raw = (root / 'TARGET-KV-ROUTES-rank-0.jsonl').read_bytes()
            record = json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())
            self.assertEqual(probe.validate_target_kv_health(record, binding=manifest, worker=worker),
                             dict(routes=2, bytes=len(raw)))
            self.assertEqual(len(raw.splitlines()), 2)
            self.assertIsNone(module._FP8_TARGET_KV_OBSERVER)

    def test_route_cap_saturates_and_later_callbacks_do_no_metadata_work(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', return_value={}) as metadata:
                for layer in range(200):
                    call(layer)
                self.assertEqual(metadata.call_count, 128)
            restore()
            health = json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())
            self.assertTrue(health['saturated'])
            self.assertEqual(health['counters']['routes'], 128)
            self.assertEqual(log.getvalue().count('SPRY_OBSERVER_FAILED'), 1)
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                probe.validate_target_kv_health(health, binding=manifest, worker=worker)

    def test_metadata_failure_disables_only_the_observer(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', side_effect=OverflowError) as metadata:
                for layer in range(50):
                    call(layer)
                metadata.assert_called_once()
            restore()
            self.assertTrue(json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())['observer_failed'])

    def test_record_byte_cap_is_saturating(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', return_value=dict(data='x' * 8192)):
                call()
            restore()
            record = json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())
            self.assertTrue(record['saturated'])
            self.assertEqual(record['counters'], dict(routes=0, bytes=0))

    def test_failed_route_flush_is_contained_and_health_fails(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', return_value={}):
                call()
            with patch.object(probe.os, 'fsync', side_effect=[OSError('disk'), None]):
                restore()
            record = json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())
            self.assertTrue(record['observer_failed'])

    def test_replaced_callback_is_preserved_and_health_fails(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            replacement = object()
            module._FP8_TARGET_KV_OBSERVER = replacement
            restore()
            self.assertIs(module._FP8_TARGET_KV_OBSERVER, replacement)
            self.assertTrue(json.loads((root / 'OBSERVER-HEALTH-target_kv-rank-0.json').read_bytes())['observer_failed'])

    def test_backend_observer_is_optional_static_call_boundary(self):
        tree = ast.parse((SOURCE / 'triton_backend_candidate.py').read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TritonAttnBackend')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_record_fp8_target_read')
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(method)))
        guard = method.body[0].body[1]
        self.assertEqual(ast.unparse(guard.test), 'observer is not None')
        self.assertEqual(ast.unparse(guard.body[0].value.func), 'observer')


class WorkerFinalization(unittest.TestCase):
    def test_real_worker_finally_finishes_graph_after_other_cleanup(self):
        for instrumented in (False, True):
            for fails in (False, True):
                with self.subTest(instrumented=instrumented, fails=fails), tempfile.TemporaryDirectory() as temporary:
                    manifest, worker = identities(Path(temporary))
                    manifest['expected_ranks'] = [0, 1]
                    manifest['sources'].update(mtp_contract=dict(module='mtp_contract'),
                        serving_parity=dict(module='serving_parity'), gemm={}, scheduler={})
                    manifest['preweight'] = dict(graph_observer=dict(module='serving_graph_observer'))
                    calls = []
                    observer = SimpleNamespace(finish=lambda m, w: calls.append('graph-finish'))
                    cfg = SimpleNamespace(instrumentation_enabled=lambda: instrumented,
                        dense_mxfp4_enabled=lambda: False, settings=lambda: {}, verify_attention_observed=lambda arm: False)
                    modules = dict(mtp_contract=cfg, serving_graph_observer=SimpleNamespace(CURRENT=observer),
                        j01_probe=SimpleNamespace(install_target_kv_witness=lambda *a: lambda: calls.append('kv-finish')),
                        serving_parity=SimpleNamespace(wait_before_weights=lambda *a: None))
                    error = RuntimeError('scheduler')
                    target = Mock(side_effect=error) if fails else Mock(return_value='served')
                    stubs = dict(_read_manifest=lambda: manifest, _validate_manifest=lambda m: m,
                        _worker_record=lambda *a: worker, _source_module=lambda s: modules[s['module']],
                        _native_module=lambda *a: None, _preweight_before_scheduler=lambda *a, **kw: {},
                        _source_selection=lambda *a: target)
                    with patch.multiple(worker_entry, **stubs):
                        if fails:
                            with self.assertRaises(RuntimeError) as caught:
                                worker_entry.run_scheduler_process(None, None, 0, 0, 0, None, None)
                            self.assertIs(caught.exception, error)
                        else:
                            self.assertEqual(worker_entry.run_scheduler_process(None, None, 0, 0, 0, None, None), 'served')
                    target.assert_called_once()
                    self.assertEqual(calls, ['kv-finish', 'graph-finish'] if instrumented else [])


class ObserverReadout(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cleanup-health-readout-')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.binding, _ = identities(root)
        self.binding['sources']['graph_observer'] = dict(
            source(SOURCE / 'serving_graph_observer.py', 'serving_graph_observer'),
            path='/mnt/review/serving_graph_observer.py')
        self.binding['source_manifest_sha256'] = hashlib.sha256(
            probe.canonical(self.binding['sources']).rstrip(b'\n')).hexdigest()
        self.workers = []
        self.observations = {}
        for rank in (0, 1):
            _, worker = identities(root, rank)
            self.workers.append(worker)
            for name, counters, role in (
                ('graph', dict(captures=2, replays=2, commits=2, bs1_replays=1, bs2_replays=1), 'graph_observer'),
                ('target_kv', dict(routes=0, bytes=0), 'j01_probe')):
                health = graph._health.Health(name, rank)
                health.finish(self.binding, worker, self.binding['sources'][role], counters)
                filename = graph._health.filename(name, rank)
                self.observations[filename] = (root / filename).read_bytes()
            self.observations[f'graph-counter-rank-{rank}.bin'] = struct.pack('<QQQQ', 2, 2, 1, 1)
            self.observations[f'replay-commit-counter-rank-{rank}.bin'] = struct.pack('<Q', 2)

    def validate(self):
        with patch.dict(sys.modules, mtp_contract=contract()):
            validator = integration._gdn_observer(self.binding['sources'], require_gdn=False)
            integration._validate_observer_health(self.observations, binding=self.binding,
                                                  workers=self.workers, graph=validator)

    def test_real_receipts_pass_source_bound_validator_and_counter_join(self):
        self.validate()

    def test_each_missing_receipt_fails(self):
        for name in list(self.observations):
            if not name.endswith('.json'):
                continue
            with self.subTest(name=name):
                raw = self.observations.pop(name)
                with self.assertRaisesRegex(ValueError, 'missing or oversized'):
                    self.validate()
                self.observations[name] = raw

    def test_failed_misbound_or_malformed_health_cannot_pass(self):
        for name in ('graph', 'target_kv'):
            filename = f'OBSERVER-HEALTH-{name}-rank-0.json'
            original = self.observations[filename]
            for key, value in dict(observer_failed=True, saturated=True, complete=False,
                    failure_type='ValueError', rank=True, pid=999, boot_id='other',
                    session_sha256='f' * 64, helper_source={}, counters={}, unexpected=True).items():
                with self.subTest(observer=name, field=key):
                    record = json.loads(original)
                    record[key] = value
                    self.observations[filename] = probe.canonical(record)
                    with self.assertRaises(ValueError):
                        self.validate()
            for raw in (b'{', b' ' + original, b'x' * 4097,
                        original.replace(b'"schema":1', b'"schema":1,"schema":1')):
                self.observations[filename] = raw
                with self.assertRaises(ValueError):
                    self.validate()
            self.observations[filename] = original

    def test_stale_graph_or_commit_counters_fail(self):
        for filename, replacement in (
            ('graph-counter-rank-0.bin', struct.pack('<QQQQ', 2, 3, 1, 2)),
            ('replay-commit-counter-rank-0.bin', struct.pack('<Q', 3))):
            with self.subTest(file=filename):
                original = self.observations[filename]
                self.observations[filename] = replacement
                with self.assertRaisesRegex(ValueError, 'counter.*differ'):
                    self.validate()
                self.observations[filename] = original

    def test_unaccounted_target_kv_records_fail(self):
        self.observations['TARGET-KV-ROUTES-rank-0.jsonl'] = b'{}\n'
        with self.assertRaisesRegex(ValueError, 'journal differs'):
            self.validate()

    def test_health_module_substitution_is_refused_by_both_importers(self):
        with tempfile.TemporaryDirectory(prefix='cleanup-health-substitution-') as temporary:
            root = Path(temporary).resolve()
            (root / 'observer_health.py').write_text('raise RuntimeError("must not execute")\n')
            for name in ('j01_probe.py', 'serving_graph_observer.py'):
                path = root / name
                path.write_bytes((SOURCE / name).read_bytes())
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'health source binding'):
                    loaded('cleanup_substituted', path)

    def test_actual_native_reader_runs_health_validation(self):
        from test_lifecycle_safety import HealthReadout
        prior = HealthReadout()
        prior.setUp()
        cfg, observations = prior.native_fixture()
        cfg.commit_observation_limit = lambda: 8
        cfg.validate_recurrent_capacity_cli = Mock(side_effect=RuntimeError('later runtime validation'))
        observations.update(self.observations)
        for name in ('SERVING-BINDING.json', 'WORKER-MANIFEST.json'):
            observations[name] = probe.canonical(self.binding)
        with patch.dict(sys.modules, mtp_contract=cfg), \
                patch.object(integration, '_mtp_contract', return_value=cfg), \
                patch.object(integration, '_private_reader', return_value=(SimpleNamespace(hip_sampling_arm=lambda *a: False), None)), \
                patch.object(integration, '_validate_lifecycle_health'):
            with self.assertRaisesRegex(RuntimeError, 'later runtime validation'):
                integration._validate_native_observations(observations, expected=self.binding, gate={}, full_workers=self.workers)
            bad = json.loads(observations['OBSERVER-HEALTH-graph-rank-1.json'])
            bad['observer_failed'] = True
            observations['OBSERVER-HEALTH-graph-rank-1.json'] = probe.canonical(bad)
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                integration._validate_native_observations(observations, expected=self.binding, gate={}, full_workers=self.workers)


if __name__ == '__main__':
    unittest.main()
