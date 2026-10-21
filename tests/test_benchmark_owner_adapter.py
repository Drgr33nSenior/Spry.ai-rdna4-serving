"""Composition fixtures only; platform, bwrap and GPU observations are synthetic."""
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


adapter = load('adapter_under_test', ROOT / 'tools/benchmark_owner_adapter.py')
fixtures = load('readout_cpu_fixtures', ROOT / 'tests/test_benchmark_readout.py')
owner = adapter.owner


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.base.chmod(0o700)
        self.root = self.base / 'attempt'
        self.root.mkdir(mode=0o700)
        self.lock = self.base / 'diagnostic.lock'
        self.lock.touch(mode=0o600)
        self.script = self.root / 'model_benchmark.py'
        self.script.write_text('raise RuntimeError("inert fixture must never run")\n')
        self.script_sha = hashlib.sha256(self.script.read_bytes()).hexdigest()
        self.spec = fixtures.fixtures.fixture()
        self.spec['worker_seconds'] = 690
        self.spec_path = self.root / 'spec.json'
        self.spec_path.write_text(json.dumps(self.spec))
        self.spec_sha = hashlib.sha256(self.spec_path.read_bytes()).hexdigest()
        self.collector = 'c' * 64
        self.lease = owner.CanonicalLease(self.lock, self.root, 'd' * 64, 'fixture-boot',
            expected_uid=os.getuid(), expected_gid=os.getgid(), owner_identity=self.identity(os.getpid()))
        self.results, self.calls = [], []
        self.arms = tuple(adapter.FrozenArm(arm,
            (sys.executable, '-I', '-B', '-S', str(self.script), arm),
            str(self.script), self.script_sha) for arm in ('bf16', 'kv4'))

    def tearDown(self):
        if self.lease.fd is not None:
            self.lease.close()
        self.temp.cleanup()

    def test_physical_projection_has_no_writable_host_results_mount(self):
        kwargs = dict(retained_root=str(self.base / 'retained'), composition=str(self.base / 'composition'),
            review=str(self.root / 'review'), model=str(self.base / 'model'),
            output=str(self.root / 'arms/bf16'), entry=str(self.root / 'review/model_benchmark.py'),
            spec_sha256='a' * 64, bounded_scratch=True, physical_artifacts=True,
            session_sha256='b' * 64, artifacts_sha256='c' * 64)
        argv = adapter.bwrap_argv('bf16', **kwargs)
        self.assertNotIn('--bind', argv)
        self.assertNotIn('/mnt/results', argv)
        self.assertEqual(argv[argv.index('--output') + 1], '/mnt/output/benchmark')
        self.assertEqual(argv[-5:], ('--physical-artifacts', '--session-sha256', 'b' * 64,
                                    '--artifacts-sha256', 'c' * 64))
        self.assertEqual(sum(int(argv[i + 1]) for i, value in enumerate(argv) if value == '--size'),
                         3584 * 1024**2)
        for changes in ({'bounded_scratch': False}, {'session_sha256': None}, {'artifacts_sha256': None}):
            with self.assertRaises(owner.OwnerError):
                adapter.bwrap_argv('bf16', **{**kwargs, **changes})

    def identity(self, pid, namespace='pid:[host-fixture]', parent=1):
        return owner.ProcessIdentity('fixture-boot', pid, parent, pid, pid,
            10, os.getuid(), os.getgid(), namespace, 'a' * 64, 'S')

    def supervise(self, argv, **kwargs):
        arm = argv[-1]
        self.calls.append(arm)
        # Exercise actual immutable journal binding, substituting only the child
        # and platform observations. Real POSIX children are in the core suite.
        row = self.lease.record['arms'][-1]
        self.assertEqual(row['argv'], list(argv))
        self.assertEqual(row['script'], kwargs['script_path'])
        self.assertEqual(row['script_sha256'], kwargs['script_sha256'])
        self.assertEqual(self.lease.fd, 9)
        self.assertEqual(kwargs['deadlines'].term_end - kwargs['deadlines'].work_end, 20)
        child = self.identity(12345 + len(self.calls))
        private = self.identity(child.pid + 10, f'pid:[workload-{arm}]', child.pid)
        kwargs['on_child'](child)
        result = owner.SupervisionResult(child, (child, private), 0, None, (), (), True, None)
        self.results.append(result)
        return result

    def snapshot(self, results, *, arm=None):
        record = self.lease.record
        value = {'schema': 1,
            'scope': 'host-benchmark-release-v1' if arm else 'host-benchmark-independent-release-v1',
            'root': str(self.root), 'session_sha256': self.lease.session, 'boot_id': self.lease.boot,
            'collector_source_sha256': self.collector,
            'lock_device': record['lock_device'], 'lock_inode': record['lock_inode'],
            'observed_monotonic': time.monotonic(),
            'checked_processes': [list(row) for row in sorted({(p.boot_id, p.pid, p.start_ticks)
                for result in results for p in result.observed_processes})],
            'authenticated_processes_present': [],
            'pid_namespace_members': {name: [] for name in sorted({p.pid_namespace
                for result in results for p in result.observed_processes
                if p.pid_namespace != result.root_process.pid_namespace})},
            'holders': {'visibility': 'all-users-platform', 'exit_code': 1, 'stdout': '', 'stderr': ''},
            'allocation': {'scope': 'all-namespaces-active-workloads', 'gpu_requests': 0},
            'target': {'hostname': 'threadripper-ai', 'uid': 1000, 'gid': 1000,
                'boot_id': self.lease.boot, 'swap_active': False, 'node_ready': True,
                'pressure_free': True, 'expected_policy_unchanged': True},
            'devices': [{'bdf': bdf, 'uuid': uuid, 'render': '/dev/dri/renderD' + str(128 + index),
                         'sysfs_bdf': bdf} for index, (bdf, uuid) in enumerate(adapter.DEVICES.items())]}
        if arm:
            value.update(arm=arm, child_pid=results[0].root_process.pid,
                         child_start_ticks=results[0].root_process.start_ticks)
        return value

    def collect(self, lease, arm, result, deadline):
        self.assertEqual(lease.fd, 9)
        return self.snapshot([result], arm=arm)

    def admission(self, lease, *, deadline_monotonic):
        self.assertEqual(lease.fd, 9)
        self.assertEqual(lease.record['state'], 'reserved')
        self.assertEqual(self.calls, [])
        value = self.snapshot([])
        value.update(scope='host-benchmark-locked-admission-v1', status='OBSERVATION_COMPLETE', failures=[],
            journal={'authenticated': True, 'state': 'reserved', 'sha256': lease.journal_identity[2]})
        return value

    def independent(self, lease, deadline):
        self.assertIsNone(lease.fd)
        self.assertTrue(lease.journal.exists(), 'journal must survive until independent acceptance')
        return self.snapshot(self.results)

    def inner(self, arm):
        result = fixtures.arm_result(arm)
        result['spec_sha256'] = self.spec_sha
        return result

    def campaign(self, **changes):
        kwargs = dict(lease=self.lease, arms=self.arms, spec_path=self.spec_path,
            spec_sha256=self.spec_sha, collector_sha256=self.collector,
            active_deadline=time.monotonic() + 1800, collect_release=self.collect,
            read_inner_result=self.inner, independent_postcheck=self.independent,
            streams={arm: (subprocess.DEVNULL, subprocess.DEVNULL) for arm in ('bf16', 'kv4')},
            resource_check=lambda: None, supervisor=self.supervise)
        kwargs.update(changes)
        return adapter.run_campaign(**kwargs)

    def test_complete_composition_binds_both_arms_and_never_qualifies(self):
        report = self.campaign()
        self.assertEqual(self.calls, ['bf16', 'kv4'])
        self.assertTrue(report['adapter_complete'], report)
        self.assertTrue(report['independent_release_verified'])
        self.assertFalse(report['unresolved'])
        self.assertFalse(report['physical_campaign_complete'])
        self.assertFalse(report['qualification'])
        self.assertFalse(self.lease.journal.exists())

    def test_failed_arm_is_safely_closed_without_second_or_replacement(self):
        def failed(*args, **kwargs):
            result = self.supervise(*args, **kwargs)
            result = replace(result, exit_code=2, failure='synthetic child nonzero')
            self.results[-1] = result
            return result
        report = self.campaign(supervisor=failed)
        self.assertEqual(self.calls, ['bf16'])
        self.assertFalse(report['adapter_complete'])
        self.assertTrue(report['independent_release_verified'])
        self.assertFalse(report['unresolved'])

    def test_explicit_observer_envelope_preserves_work_reap_and_collection_bounds(self):
        deadlines = {}
        def supervised(argv, **kwargs):
            deadlines[argv[-1]] = kwargs['deadlines']
            return self.supervise(argv, **kwargs)
        def collected(lease, arm, result, deadline):
            self.assertAlmostEqual(deadline - deadlines[arm].work_end, 105)
            self.assertAlmostEqual(deadline - deadlines[arm].reap_end, 75)
            return self.collect(lease, arm, result, deadline)
        report = self.campaign(arm_envelope_seconds=795, collection_reserve_seconds=330,
            active_deadline=time.monotonic() + 2000, supervisor=supervised, collect_release=collected,
            admission_check=self.admission)
        self.assertTrue(report['adapter_complete'], report)
        self.assertEqual(self.calls, ['bf16', 'kv4'])
        self.assertEqual(report['arm_envelope_seconds'], 795)
        self.assertEqual(report['collection_reserve_seconds'], 330)
        self.assertFalse(report['physical_campaign_complete'])
        self.assertFalse(report['qualification'])

    def test_bound_locked_admission_failure_never_reaches_first_arm(self):
        def wrong_journal(*args, **kwargs):
            value = self.admission(*args, **kwargs)
            value['journal']['sha256'] = 'e' * 64
            return value
        report = self.campaign(arm_envelope_seconds=795, collection_reserve_seconds=330,
            active_deadline=time.monotonic() + 2000, admission_check=wrong_journal)
        self.assertFalse(report['adapter_complete'])
        self.assertEqual(self.calls, [])
        self.assertFalse(report['unresolved'])
        self.assertFalse(report['qualification'])

    def test_unreviewed_or_insufficient_observer_envelope_refuses_before_lease(self):
        cases = [(795, 180), (750, 330), (794, 330), (795.0, 330), (True, 330)]
        for arm_seconds, reserve in cases:
            with self.subTest(bounds=(arm_seconds, reserve)), self.assertRaises(owner.OwnerError):
                self.campaign(arm_envelope_seconds=arm_seconds, collection_reserve_seconds=reserve,
                    active_deadline=time.monotonic() + 2000)
        with self.assertRaises(owner.OwnerError):
            self.campaign(arm_envelope_seconds=795, collection_reserve_seconds=330,
                active_deadline=time.monotonic() + 1994, admission_check=self.admission)
        with self.assertRaises(owner.OwnerError):
            self.campaign(arm_envelope_seconds=795, collection_reserve_seconds=330,
                active_deadline=time.monotonic() + 2000)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.lease.fd)
        self.assertFalse(self.lease.journal.exists())

    def test_uncertain_release_preserves_journal_and_stops_second_arm(self):
        def uncertain(*args):
            value = self.collect(*args)
            value['holders']['stderr'] = 'permission denied'
            return value
        report = self.campaign(collect_release=uncertain)
        self.assertEqual(self.calls, ['bf16'])
        self.assertTrue(report['unresolved'])
        self.assertFalse(report['adapter_complete'])
        self.assertTrue(self.lease.journal.exists())

    def test_failed_independent_gate_retains_durable_refusal(self):
        def bad(*args):
            value = self.independent(*args)
            value['allocation']['gpu_requests'] = 1
            return value
        report = self.campaign(independent_postcheck=bad)
        self.assertTrue(report['unresolved'])
        self.assertFalse(report['independent_release_verified'])
        self.assertEqual(self.lease.record['state'], 'pending-independent')
        with self.assertRaises(owner.OwnerError):
            self.lease.acquire()

    def test_missing_namespace_source_device_and_stale_snapshot_fail_closed(self):
        mutations = [lambda v: v['pid_namespace_members'].clear(),
            lambda v: v.update(collector_source_sha256='e' * 64),
            lambda v: v['devices'][0].update(sysfs_bdf='0000:aa:00.0'),
            lambda v: v.update(observed_monotonic=0),
            lambda v: v['checked_processes'].pop(),
            lambda v: v['holders'].update(visibility='current-user-only')]
        self.lease.acquire()
        arm = self.arms[0]
        self.lease.begin_arm(arm.arm, arm.argv, arm.script_path, arm.script_sha256)
        result = self.supervise(arm.argv, script_path=arm.script_path,
            script_sha256=arm.script_sha256, deadlines=owner.Deadlines(1, 21, 26, 31),
            on_child=self.lease.record_child)
        for index, mutate in enumerate(mutations):
            snapshot = self.snapshot([result], arm='bf16')
            mutate(snapshot)
            with self.subTest(index=index), self.assertRaises(owner.OwnerError):
                adapter.release_evidence(snapshot, lease=self.lease, arm='bf16', result=result,
                    collector_sha256=self.collector, deadline=time.monotonic() + 30,
                    not_before=time.monotonic() - 1)

    def test_deadline_and_spec_substitution_refuse_before_lock_or_spawn(self):
        for changes in ({'active_deadline': time.monotonic() + 1679},
                        {'spec_sha256': 'f' * 64}, {'collector_sha256': None},
                        {'arms': tuple(reversed(self.arms))}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, owner.OwnerError)):
                self.campaign(**changes)
            self.assertEqual(self.calls, [])
            self.assertIsNone(self.lease.fd)
            self.assertFalse(self.lease.journal.exists())

    def test_bwrap_projection_keeps_isolation_paths_and_only_declared_arm_differences(self):
        common = dict(retained_root=self.base / 'retained', composition=self.base / 'composition',
            review=self.root, model=self.base / 'model', output=self.base / 'out',
            entry=self.root / 'model_benchmark.py', spec_sha256=self.spec_sha)
        left = adapter.bwrap_argv('bf16', **common)
        right = adapter.bwrap_argv('kv4', **common)
        self.assertEqual(left[:-1], right[:-1])
        self.assertEqual(left[-1], 'bf16')
        self.assertIn('--die-with-parent', left)
        self.assertIn('--unshare-all', left)
        self.assertIn('--share-net', left)  # Explicit: not network-denied Pod isolation.
        self.assertIn('--clearenv', left)
        self.assertNotIn('sudo', left)
        self.assertEqual(left.count('--dev-bind'), 3)
        self.assertIn(str(self.script), left)  # Actual host source is gate-bound.

    def test_bounded_scratch_projection_preserves_runtime_and_both_arms(self):
        common = dict(retained_root=self.base / 'retained', composition=self.base / 'composition',
            review=self.root, model=self.base / 'model', output=self.base / 'out',
            entry=self.root / 'model_benchmark.py', spec_sha256=self.spec_sha)
        old = adapter.bwrap_argv('bf16', **common)
        left = adapter.bwrap_argv('bf16', bounded_scratch=True, **common)
        right = adapter.bwrap_argv('kv4', bounded_scratch=True, **common)
        self.assertEqual(left[:-1], right[:-1])
        self.assertEqual(left[-1], 'bf16')
        sizes = {left[i + 3]: int(left[i + 1]) for i in range(len(left) - 3)
                 if left[i] == '--size' and left[i + 2] == '--tmpfs'}
        self.assertEqual(sizes, {'/mnt/output': 2816 * 1024**2, '/tmp': 512 * 1024**2,
                                '/dev/shm': 256 * 1024**2})
        self.assertEqual(sum(sizes.values()) + 512 * 1024**2, 4 * 1024**3)
        self.assertEqual(left[left.index('--output') + 1], '/mnt/results/benchmark')
        self.assertEqual(old[old.index('--output') + 1], '/mnt/output/benchmark')
        self.assertIn('/mnt/output/cache', left)
        self.assertEqual(left.count('--dev-bind'), old.count('--dev-bind'))
        self.assertEqual(left.count('--ro-bind'), old.count('--ro-bind'))
        with self.assertRaises(owner.OwnerError):
            adapter.bwrap_argv('bf16', bounded_scratch=1, **common)


if __name__ == '__main__':
    unittest.main(verbosity=2)
