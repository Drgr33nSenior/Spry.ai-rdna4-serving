"""CPU regressions through complete source modules; no native runtime imports."""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'legacy' / 'review'
sys.path.insert(0, str(SOURCE))
probe = importlib.import_module('j01_probe')
integration = importlib.import_module('serving_integration')


class Tensor:
    dtype = 'torch.int64'

    def __init__(self, values):
        self.values = values

    def numel(self):
        return len(self.values)

    def tolist(self):
        return self.values[:]


class Allocator:
    size = 19

    def __init__(self):
        self.free_slots = Tensor(list(range(1, 20)))
        self.calls = []

    def _do_alloc(self, count=1):
        self.calls.append(('alloc', count))
        return Tensor([1])

    def free(self, values):
        self.calls.append(('free', values))
        return values


class Pool:
    def __init__(self, allocator):
        self.mamba_allocator = allocator
        self.mamba_pool = SimpleNamespace(size=19)
        self.free_slots = list(range(2))
        self.calls = []

    def alloc(self, requests):
        self.calls.append(('alloc', requests))
        return requests

    def free(self, request):
        self.calls.append(('free', request))
        return request


class Cache:
    def __init__(self):
        self.calls = []

    def match_prefix(self, params):
        self.calls.append(params)
        return SimpleNamespace(device_indices=[], mamba_branching_seqlen=0)

    def mamba_evictable_size(self):
        return 0

    def mamba_protected_size(self):
        return 0


def identities(root, rank=0):
    source = dict(module='j01_probe', path='/mnt/review/j01_probe.py',
                  sha256=hashlib.sha256((SOURCE / 'j01_probe.py').read_bytes()).hexdigest())
    sources = {'j01_probe': source, 'memory_pool': {}, 'mamba_allocator': {}, 'cache_tree': {}}
    bare = json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()
    manifest = dict(mode=integration.MODE, arm='mtp', session_sha256='1' * 64,
                    spec_sha256='2' * 64, source_manifest_sha256=hashlib.sha256(bare).hexdigest(),
                    evidence_dir=str(root), sources=sources)
    worker = dict(rank=rank, pid=100 + rank, start_ticks=500 + rank,
                  boot_id='cpu-fixture-boot', identity_sha256=str(3 + rank) * 64)
    return manifest, worker


@contextlib.contextmanager
def installed(rank=0):
    with tempfile.TemporaryDirectory(prefix='cleanup-lifecycle-') as temporary:
        root = Path(temporary).resolve()
        manifest, worker = identities(root, rank)
        allocator = Allocator()
        pool = Pool(allocator)
        cache = Cache()
        scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
            req_to_token_pool=pool)), tree_cache=cache)
        output = io.StringIO()
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        # Native source identities and integer tensors are the only constructor seams.
        with patch.dict(sys.modules, torch=torch_stub), contextlib.redirect_stdout(output), \
                patch.object(probe.LifecycleSlots, 'check_source') as source_check:
            observer = probe.install_lifecycle_slots(scheduler, manifest, worker)
            assert source_check.call_count == 7
            try:
                yield observer, allocator, pool, cache, root, output
            finally:
                observer.restore()


def request():
    return SimpleNamespace(rid='fixture', origin_input_ids=[1, 2], cache_salt='fixture',
                           kv=SimpleNamespace(req_pool_idx=0, mamba_pool_idx=None))


def receipt(root, rank=0):
    return json.loads((root / f'LIFECYCLE-HEALTH-rank-{rank}.json').read_bytes())


class LifecycleSafety(unittest.TestCase):
    def test_healthy_callbacks_preserve_results_arguments_and_restoration(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            req = request()
            requests = [req]
            tensor = allocator._do_alloc(3)
            self.assertIs(allocator.free(tensor), tensor)
            self.assertIs(pool.alloc(requests), requests)
            self.assertIs(pool.free(req), req)
            params = SimpleNamespace(req=req)
            self.assertEqual(cache.match_prefix(params).device_indices, [])
            self.assertEqual(allocator.calls, [('alloc', 3), ('free', tensor)])
            self.assertEqual(pool.calls, [('alloc', requests), ('free', req)])
            self.assertEqual(cache.calls, [params])
            self.assertEqual(observer.seq, 6)
            observer.restore()
            self.assertNotIn('_do_alloc', allocator.__dict__)
            self.assertNotIn('free', pool.__dict__)
            self.assertNotIn('match_prefix', cache.__dict__)
            probe.validate_lifecycle_health(receipt(root), binding=observer.manifest, worker=observer.worker)

    def test_real_event_limit_saturates_and_scheduler_finishes_larger_menu(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            for _ in range(9000):
                self.assertEqual(allocator._do_alloc().tolist(), [1])
            self.assertEqual(len(allocator.calls), 9000)
            self.assertEqual(observer.seq, 8192)
            self.assertTrue(observer.observer_failed)
            self.assertTrue(observer.saturated)
            reads = observer.scalar_reads
            allocator._do_alloc()
            self.assertEqual(observer.scalar_reads, reads)
            self.assertEqual(output.getvalue().count('SPRY_LIFECYCLE_OBSERVER_FAILED '), 1)
            observer.restore()
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                probe.validate_lifecycle_health(receipt(root), binding=observer.manifest, worker=observer.worker)

    def test_small_cap_disables_once_and_keeps_event_count_bounded(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            observer.event_limit = 1
            for _ in range(100):
                allocator._do_alloc()
            self.assertEqual(observer.seq, 1)
            self.assertEqual(len(allocator.calls), 100)
            self.assertEqual(output.getvalue().count('SPRY_LIFECYCLE_OBSERVER_FAILED '), 1)

    def test_each_callback_contains_observer_errors_and_delegates_once(self):
        cases = (
            ('record_index_alloc', lambda a, p, c: a._do_alloc(), 'allocator'),
            ('indices', lambda a, p, c: a.free(Tensor([1])), 'allocator'),
            ('record_request_alloc', lambda a, p, c: p.alloc([request()]), 'pool'),
            ('request', lambda a, p, c: p.free(request()), 'pool'),
            ('record_request_free', lambda a, p, c: p.free(request()), 'pool'),
            ('record_cache_match', lambda a, p, c: c.match_prefix(SimpleNamespace(req=request())), 'cache'),
        )
        for method, call, counter in cases:
            for error in (ValueError('injected'), OverflowError('injected'), OSError('injected')):
                with self.subTest(method=method, error=type(error).__name__), installed() as fixture:
                    observer, allocator, pool, cache, root, output = fixture
                    with patch.object(observer, method, side_effect=error) as observation:
                        call(allocator, pool, cache)
                        call(allocator, pool, cache)
                    observation.assert_called_once()
                    self.assertEqual(len(dict(allocator=allocator, pool=pool, cache=cache)[counter].calls), 2)
                    self.assertTrue(observer.observer_failed)
                    self.assertFalse(observer.saturated)
                    self.assertEqual(output.getvalue().count('SPRY_LIFECYCLE_OBSERVER_FAILED '), 1)

    def test_original_errors_propagate_without_retry_before_and_after_disable(self):
        callbacks = (
            ('on_index_alloc', ()), ('on_index_free', (Tensor([1]),)),
            ('on_request_alloc', ([request()],)), ('on_request_free', (request(),)),
            ('on_cache_match', (SimpleNamespace(req=request()),)),
        )
        for name, arguments in callbacks:
            for disabled in (False, True):
                with self.subTest(callback=name, disabled=disabled), installed() as fixture:
                    observer = fixture[0]
                    if disabled:
                        observer.fail(ValueError('injected'))
                    error = RuntimeError('original serving error')
                    original = unittest.mock.Mock(side_effect=error)
                    with self.assertRaises(RuntimeError) as caught:
                        getattr(observer, name)(original, *arguments)
                    self.assertIs(caught.exception, error)
                    original.assert_called_once_with(*arguments)
                    self.assertEqual(observer.observer_failed, disabled)

    def test_failure_logging_does_not_raise_or_leak_exception_text(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            with patch.object(observer, 'indices', side_effect=ValueError('sensitive-details')), \
                    patch('builtins.print', side_effect=BrokenPipeError) as logger:
                allocator._do_alloc()
                allocator._do_alloc()
            logger.assert_called_once()
            self.assertNotIn('sensitive-details', str(logger.call_args))
            self.assertEqual(len(allocator.calls), 2)
            observer.restore()
            self.assertTrue(receipt(root)['observer_failed'])

    def test_keyboard_interrupt_from_observation_is_not_swallowed(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            with patch.object(observer, 'record_index_alloc', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    allocator._do_alloc()
            self.assertEqual(len(allocator.calls), 1)

    def test_startup_source_invariant_remains_fatal(self):
        with installed() as fixture:
            observer, allocator, pool, cache, root, output = fixture
            scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                req_to_token_pool=pool)), tree_cache=cache)
            with patch.object(probe.LifecycleSlots, 'check_source', side_effect=ValueError('source mismatch')):
                with self.assertRaisesRegex(ValueError, 'source mismatch'):
                    probe.install_lifecycle_slots(scheduler, observer.manifest, observer.worker)

    def test_replaced_wrapper_is_retained_and_health_fails(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            replacement = lambda *args: 'replacement'
            allocator._do_alloc = replacement
            observer.restore()
            self.assertIs(allocator._do_alloc, replacement)
            self.assertFalse(receipt(root)['restored'])
            self.assertTrue(receipt(root)['observer_failed'])

    def test_failed_health_write_retains_pending_and_never_publishes_success(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            with patch.object(probe.os, 'fsync', side_effect=OSError('disk error')):
                observer.restore()
            self.assertTrue((root / 'LIFECYCLE-HEALTH-rank-0.pending').exists())
            self.assertFalse((root / 'LIFECYCLE-HEALTH-rank-0.json').exists())
            self.assertTrue(observer.observer_failed)

    def test_failed_final_link_retains_pending_and_no_success_receipt(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            with patch.object(probe.os, 'link', side_effect=OSError('link error')):
                observer.restore()
            self.assertTrue((root / 'LIFECYCLE-HEALTH-rank-0.pending').exists())
            self.assertFalse((root / 'LIFECYCLE-HEALTH-rank-0.json').exists())
            self.assertTrue(observer.observer_failed)

    def test_failed_pending_removal_leaves_multiple_links_for_collector_refusal(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            with patch.object(Path, 'unlink', side_effect=OSError('unlink error')):
                observer.restore()
            self.assertTrue((root / 'LIFECYCLE-HEALTH-rank-0.pending').exists())
            self.assertEqual((root / 'LIFECYCLE-HEALTH-rank-0.json').stat().st_nlink, 2)
            self.assertTrue(observer.observer_failed)

    def test_health_collision_retains_existing_file_and_pending_failure(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            existing = root / 'LIFECYCLE-HEALTH-rank-0.json'
            existing.write_bytes(b'prior evidence\n')
            observer.restore()
            self.assertEqual(existing.read_bytes(), b'prior evidence\n')
            self.assertTrue((root / 'LIFECYCLE-HEALTH-rank-0.pending').exists())
            self.assertTrue(observer.observer_failed)

    def test_restore_is_idempotent_and_closed_observer_delegates(self):
        with installed() as (observer, allocator, pool, cache, root, output):
            saved_callback = allocator._do_alloc
            observer.restore()
            before = (root / 'LIFECYCLE-HEALTH-rank-0.json').read_bytes()
            observer.restore()
            self.assertEqual(saved_callback().tolist(), [1])
            self.assertEqual(observer.seq, 1)
            self.assertEqual((root / 'LIFECYCLE-HEALTH-rank-0.json').read_bytes(), before)


class HealthReadout(unittest.TestCase):
    def setUp(self):
        self.observations = {}
        self.workers = []
        for rank in (0, 1):
            with installed(rank) as (observer, allocator, pool, cache, root, output):
                observer.restore()
                name = f'LIFECYCLE-HEALTH-rank-{rank}.json'
                self.observations[name] = (root / name).read_bytes()
                self.workers.append(observer.worker)
                self.binding = {key: value for key, value in observer.manifest.items() if key != 'evidence_dir'}

    def validate(self):
        integration._validate_lifecycle_health(self.observations, binding=self.binding, workers=self.workers)

    def test_real_producer_receipts_pass_real_reader(self):
        self.validate()

    def test_missing_rank_fails(self):
        self.observations.pop('LIFECYCLE-HEALTH-rank-1.json')
        with self.assertRaisesRegex(ValueError, 'missing or oversized'):
            self.validate()

    def test_failed_incomplete_or_misbound_receipt_is_rejected(self):
        name = 'LIFECYCLE-HEALTH-rank-0.json'
        healthy = self.observations[name]
        changes = dict(observer_failed=True, saturated=True, complete=False, restored=False,
                       failure_type='ValueError', events_emitted=True, event_limit=8193,
                       rank=1, pid=101, start_ticks=501, boot_id='other',
                       worker_identity_sha256='f' * 64, session_sha256='f' * 64,
                       source_manifest_sha256='f' * 64, helper_source={}, unknown=1)
        for key, value in changes.items():
            with self.subTest(field=key):
                record = json.loads(healthy)
                record[key] = value
                self.observations[name] = probe.canonical(record)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_malformed_noncanonical_duplicate_or_oversized_receipt_is_rejected(self):
        name = 'LIFECYCLE-HEALTH-rank-0.json'
        healthy = self.observations[name]
        bad_records = [b'{', b' ' + healthy, healthy.replace(b'"schema":1', b'"schema":1,"schema":1'),
                       b' ' * 4097, healthy.decode()]
        for value in bad_records:
            with self.subTest(value_type=type(value).__name__, size=len(value)):
                self.observations[name] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_reader_authenticates_helper_source(self):
        self.binding['sources']['j01_probe']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'helper source binding'):
            self.validate()

    def native_fixture(self):
        contract = SimpleNamespace(
            phase_config=lambda: (131040, 0, 0), cache_profile=lambda arm: dict(target='bf16'),
            settings=lambda: dict(phase='cpu-fixture'), kv_composition_enabled=lambda: True,
            precision_reference_enabled=lambda: False, head_input_capture_enabled=lambda: False,
            final_norm_capture_enabled=lambda: False, midpoint_capture_enabled=lambda: False,
            raw_attention_capture_enabled=lambda: False, full_enabled_benchmark=lambda: False)
        names = {'NATIVE-READINESS-CALLBACK.json', 'SERVING-BINDING.json', 'WORKER-MANIFEST.json',
                 'MTP-PREFLIGHT.json', 'NATIVE-CHAT-PARITY.json', 'NATIVE-CHAT-PARITY.ready'}
        names.update({f'OBSERVER-HEALTH-{observer}-rank-{rank}.json'
                      for observer in ('graph', 'target_kv') for rank in (0, 1)})
        for rank in (0, 1):
            names.update({f'worker-rank-{rank}-preweight.json', f'MTP-PREFLIGHT-rank-{rank}.json',
                          f'graph-captures-rank-{rank}.json', f'graph-counter-rank-{rank}.bin',
                          f'replay-commit-counter-rank-{rank}.bin', f'TARGET-KV-STARTUP-rank-{rank}.json'})
        observations = {name: b'{}\n' for name in names}
        observations.update(self.observations)
        for name in ('SERVING-BINDING.json', 'WORKER-MANIFEST.json'):
            observations[name] = probe.canonical(self.binding)
        return contract, observations

    def test_actual_native_acceptance_path_requires_health_before_later_checks(self):
        contract, observations = self.native_fixture()
        marker = RuntimeError('reached later graph validation')
        with patch.object(integration, '_mtp_contract', return_value=contract), \
                patch.object(integration, '_private_reader', return_value=(SimpleNamespace(hip_sampling_arm=lambda *a: False), None)), \
                patch.object(integration, '_gdn_observer', side_effect=marker):
            with self.assertRaises(RuntimeError) as caught:
                integration._validate_native_observations(observations, expected=self.binding, gate={}, full_workers=self.workers)
            self.assertIs(caught.exception, marker)
            record = json.loads(observations['LIFECYCLE-HEALTH-rank-1.json'])
            record['observer_failed'] = True
            observations['LIFECYCLE-HEALTH-rank-1.json'] = probe.canonical(record)
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                integration._validate_native_observations(observations, expected=self.binding, gate={}, full_workers=self.workers)

    def test_pending_or_missing_receipts_fail_real_membership_check(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                contract, observations = self.native_fixture()
                if missing:
                    del observations['LIFECYCLE-HEALTH-rank-0.json']
                else:
                    observations['LIFECYCLE-HEALTH-rank-0.pending'] = b'pending'
                with patch.object(integration, '_mtp_contract', return_value=contract), \
                        patch.object(integration, '_private_reader', return_value=(SimpleNamespace(hip_sampling_arm=lambda *a: False), None)):
                    with self.assertRaisesRegex(ValueError, 'membership differs'):
                        integration._validate_native_observations(observations, expected=self.binding, gate={}, full_workers=self.workers)


if __name__ == '__main__':
    unittest.main()
