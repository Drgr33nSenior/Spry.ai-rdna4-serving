"""CPU-only cache owner composition: synthetic device/process observations."""
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import importlib.util
import io
import json
import os
import struct
from pathlib import Path
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cache_owner_test', ROOT / 'tools/cache_gpu_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
entry, transport, resources = controller.entry, controller.transport, controller.resources
reader_spec = importlib.util.spec_from_file_location('cache_fixture_reader_cpu', ROOT / 'tests/test_cache_gpu_fixture.py')
reader = importlib.util.module_from_spec(reader_spec)
reader_spec.loader.exec_module(reader)


def native_spec(balanced=False):
    value = {'schema': 1, 'experiment_mode': entry.PHASE_PROFILE_MODE if balanced == 'profile' else entry.COUNTERBALANCED_MODE if balanced else entry.MODE, 'session_sha256': 'a' * 64,
        'native_files': {'fixture': 'b' * 64, 'baseline.so': controller.NATIVE_BASELINE_SHA256, 'candidate.so': 'c' * 64},
        'devices': [{'device': i, 'bdf': bdf, 'uuid': uuid} for i, (bdf, uuid) in enumerate(entry.DEVICES.items())],
        'timing_repeats': 5, 'native_processes': 16 if balanced else 8, 'model_starts': 0, 'arm_order': ['bf16', 'kv4'],
        'entry_sha256': 'd' * 64, 'reader_sha256': 'e' * 64, 'owner_sha256': 'f' * 64,
        'artifacts_sha256': '1' * 64}
    if balanced:
        value.update(timing_protocol=entry.TIMING_PROTOCOL, timing_warmups=2,
                     variant_order=['baseline', 'candidate', 'candidate', 'baseline'])
    if balanced == 'profile':
        value['phase_profile'] = {'schema': 1, 'variant': 'candidate', 'boundary': 'host-wall-three-phase-v1',
            'source_sha256': entry.PROFILE_SOURCE_SHA256,
            'unprofiled_candidate_sha256': entry.UNPROFILED_CANDIDATE_SHA256,
            'projected_source_sha256': '6' * 64, 'projector_sha256': '7' * 64, 'profile_header_sha256': '8' * 64}
    return value


def package(balanced=False):
    spec = native_spec(balanced)
    return {'schema': 3, 'scope': 'native-cache-regression-package-v1', 'experiment_mode': spec['experiment_mode'],
        'target_root': '/home/developer/fp8-kv-benchmark.cachefixture',
        'package_root': '/home/developer/fp8-kv-benchmark.cachefixture',
        'boot_id': '12345678-1234-1234-1234-123456789abc', 'session_sha256': spec['session_sha256'],
        'package_files': {name: '2' * 64 for name in controller.PACKAGE_NAMES},
        'retained_root': '/home/developer/retained.fixture', 'retained_base': controller.base.retained_verifier.proof_record(),
        'native_root': '/home/developer/native.fixture', 'native_files': spec['native_files'],
        'spec_sha256': entry.sha(entry.canonical(spec)), 'resource_manifest_sha256': '3' * 64,
        'launch_authorized': False, 'proposed_complete_seconds': 3000 if balanced else 2400}


def native_observation(arm, device, balanced=False, variant='baseline'):
    members = reader.expected_records(arm, timing_protocol=entry.TIMING_PROTOCOL if balanced else 'legacy-v1',
                                      timing_repeats=5)
    value = {'schema': 1, 'status': 'NATIVE_FIXTURE_PASSED', 'qualification': False, 'cache_mode': arm,
        **device, 'visible_devices': 2, 'architecture': 'gfx1201',
        'decoder_comparisons': 8192, 'attention_rows': (18 if arm == 'bf16' else 19) + (24 if balanced else 0),
        'refusals': 35 if arm == 'bf16' else 51, 'records': len(members),
        'output_bytes': 8 + sum(12 + len(name) + size for name, size in members),
        'peak_explicit_device_buffer_bytes': 32 << 20, 'device_buffer_free_errors': 0,
        'timing_repeats': 5, 'timings': [{'protected': arm == 'bf16', 'length': length,
            'repetition': i, 'synchronized_call_ms': 1.0} for length in (128, 1024, 3966) for i in range(5)]}
    if balanced:
        value.update(timing_protocol=entry.TIMING_PROTOCOL, timing_warmups=2)
    if balanced == 'profile' and variant == 'candidate':
        value['phase_profile'] = True
        for index, row in enumerate(value['timings']):
            row['phase_wall'] = {'schema': 1, 'sequence': 4 + index + 3 * (index // 5),
                'host_checks_ns': 300, 'validation_status_ns': 200000,
                'reader_sync_ns': 700000, 'total_ns': 900300}
    return value


def complete_result(spec, arm):
    balanced = 'profile' if spec['experiment_mode'] == entry.PHASE_PROFILE_MODE else spec['experiment_mode'] == entry.COUNTERBALANCED_MODE
    variants = spec['variant_order'] if balanced else ('baseline', 'candidate')
    members = reader.expected_records(arm, timing_protocol=spec.get('timing_protocol', 'legacy-v1'), timing_repeats=5)
    rows = []
    for device in spec['devices']:
        for ordinal, variant in enumerate(variants):
            row = {'device': device['device'], 'variant': variant, 'exit_code': 0,
                'output_sha256': '4' * 64, 'output_bytes': native_observation(arm, device, balanced, variant)['output_bytes'],
                'native': native_observation(arm, device, balanced, variant),
                'supervision': {'exit_code': 0, 'failure': None, 'child_reaped': True,
                    'cleanup_errors': [], 'survivors': [], 'owner_signal': None},
                'capture': {'capture_complete': True}}
            if balanced:
                row['ordinal'] = ordinal
            rows.append(row)
    return {'schema': 1, 'experiment_mode': spec['experiment_mode'], 'arm': arm,
        'spec_sha256': entry.sha(entry.canonical(spec)), 'session_sha256': spec['session_sha256'],
        'qualification': False, 'model_starts': 0, 'status': 'NATIVE_CACHE_ARM_COMPLETE',
        'processes': rows, 'comparisons': [{'schema': 1, 'status': 'NATIVE_OLD_CANDIDATE_BITS_IDENTICAL',
            'qualification': False, 'bdf': device['bdf'], 'scope': 'synthetic-cache-ABI-only',
            'records': len(members), 'compared_bytes': sum(size for _, size in members),
            **({'reference_ordinal': 0, 'compared_ordinal': ordinal} if balanced else {})}
            for device in spec['devices'] for ordinal in range(1, len(variants))], 'timings_usable': True}


class CacheOwnerTests(unittest.TestCase):
    def test_cache_comparison_pin_does_not_promote_the_model_runtime(self):
        self.assertEqual(controller.NATIVE_BASELINE_SHA256,
                         '3bf86042ba225abc279245b41e1baa35c2549def0e8547eaf60199c4b55d8849')
        self.assertEqual(controller.base.BINARY_SHA256,
                         'd09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171')
        self.assertEqual(native_spec()['native_files']['baseline.so'], controller.NATIVE_BASELINE_SHA256)

    def test_actual_package_validator_checks_final_bytes_and_projection(self, balanced=False):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            root, native, retained = [folder / name for name in ('package', 'native', 'retained')]
            (root / 'review').mkdir(parents=True); native.mkdir(); retained.mkdir()
            value, spec = package(balanced), native_spec(balanced)
            value.update(native_root=str(native), retained_root=str(retained))
            value['package_files'] = {}
            for name in controller.SOURCE_NAMES:
                source = ROOT / ('tests/test_cache_gpu_fixture.py' if name == 'cache_gpu_fixture.py' else 'tools/' + name)
                raw = source.read_bytes()
                (root / 'review' / name).write_bytes(raw)
                value['package_files']['review/' + name] = entry.sha(raw)
            for name in spec['native_files']:
                raw = b'inert CPU asset ' + name.encode()
                (native / name).write_bytes(raw)
                spec['native_files'][name] = entry.sha(raw)
            (native / 'fixture').chmod(0o700)
            value['native_files'] = dict(spec['native_files'])
            for key, name in (('entry_sha256', 'cache_gpu_entry.py'), ('reader_sha256', 'cache_gpu_fixture.py'),
                              ('owner_sha256', 'host_process_owner.py'), ('artifacts_sha256', 'benchmark_artifacts.py')):
                spec[key] = value['package_files']['review/' + name]
            value['spec_sha256'] = entry.sha(entry.canonical(spec))
            resource = controller.resource_manifest(value, '5' * 64)
            value['resource_manifest_sha256'] = entry.sha(entry.canonical(resource))
            facts = {'declared': 'CPU retained-tree filesystem observation substitute'}
            for name, payload in (('NATIVE-SPEC.json', spec), ('RESOURCE-MANIFEST.json', resource), ('COMPOSITION-FACTS.json', facts)):
                raw = entry.canonical(payload)
                (root / 'review' / name).write_bytes(raw)
                value['package_files']['review/' + name] = entry.sha(raw)
            manifest = root / 'MANIFEST.json'
            manifest.write_bytes(entry.canonical(value))
            canonical = controller.base._canonical
            def paths(path):
                return root if path == value['target_root'] else canonical(path)
            with patch.object(controller, '__file__', str(root / 'review/cache_gpu_controller.py')), \
                 patch.object(controller.base, '_canonical', side_effect=paths), \
                 patch.object(controller, 'NATIVE_BASELINE_SHA256', spec['native_files']['baseline.so']), \
                 patch.object(controller.base, 'FACTS_SHA256', value['package_files']['review/COMPOSITION-FACTS.json']), \
                 patch.object(controller.base.retained_verifier, 'verify_retained', return_value={'stamps': {}, 'scope': 'CPU filesystem substitute'}) as retained_check:
                prepared = controller.validate_manifest(manifest, entry.sha(manifest.read_bytes()), time.monotonic() + 30)
                self.assertEqual(prepared['spec'], spec)
                retained_check.assert_called_once()
                controller.base.verify_unchanged(prepared)
                # The actual validator must reject a self-consistent package whose
                # library hash is not the separately approved comparison baseline.
                with patch.object(controller, 'NATIVE_BASELINE_SHA256', '0' * 64), \
                     self.assertRaisesRegex(ValueError, 'accepted baseline library required'):
                    controller.validate_manifest(manifest, entry.sha(manifest.read_bytes()), time.monotonic() + 30)
                self.assertEqual(retained_check.call_count, 1)
                candidate = native / 'candidate.so'
                old = candidate.read_bytes()
                candidate.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    controller.validate_manifest(manifest, entry.sha(manifest.read_bytes()), time.monotonic() + 30)
                candidate.write_bytes(old)
                (native / 'extra').write_bytes(b'not allowed')
                with self.assertRaises(ValueError):
                    controller.validate_manifest(manifest, entry.sha(manifest.read_bytes()), time.monotonic() + 30)
                (native / 'extra').unlink()
                (root / 'review/cache_gpu_entry.py').write_bytes(b'changed entry')
                with self.assertRaises(ValueError):
                    controller.validate_manifest(manifest, entry.sha(manifest.read_bytes()), time.monotonic() + 30)

    def test_counterbalanced_package_validator_checks_final_bytes_and_projection(self):
        self.test_actual_package_validator_checks_final_bytes_and_projection(True)

    def test_actual_entry_composition_retains_supervision_on_capture_failure(self):
        original_path = Path
        for balanced, failure in ((b, f) for b in (False, True, 'profile') for f in (None, 'capture', 'mismatch', 'timing', 'cleanup')):
            with self.subTest(balanced=balanced, failure=failure), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory).resolve()
                spec = native_spec(balanced)
                spec['entry_sha256'] = entry.sha(Path(entry.__file__).read_bytes())
                for name in spec['native_files']:
                    raw = b'inert never-executed CPU native source ' + name.encode()
                    (folder / name).write_bytes(raw)
                    spec['native_files'][name] = entry.sha(raw)
                calls = []
                binary = bytearray(b'UQBFV2G1')
                for name, size in reader.expected_records('bf16', timing_protocol=spec.get('timing_protocol', 'legacy-v1'),
                                                         timing_repeats=5):
                    encoded = name.encode()
                    binary.extend(struct.pack('<IQ', len(encoded), size) + encoded + bytes(size))

                def path(value):
                    value = str(value)
                    if value == '/mnt/output/cache': return folder / 'output'
                    if value.startswith('/mnt/native/'): return folder / value.removeprefix('/mnt/native/')
                    return original_path(value)

                def supervise(argv, **kwargs):
                    calls.append(list(argv))
                    limits = kwargs['deadlines']
                    self.assertEqual(limits.term_end - limits.work_end, 5)
                    self.assertEqual(limits.kill_end - limits.work_end, 8)
                    self.assertEqual(limits.reap_end - limits.work_end, 10)
                    device = int(argv[argv.index('--device') + 1])
                    observed = native_observation('bf16', spec['devices'][device], balanced, argv[argv.index('--variant') + 1])
                    target = original_path(argv[argv.index('--output') + 1])
                    self.assertLessEqual(len(list(target.parent.glob('*.bin'))), 1)
                    payload = binary[:-1] + b'\x01' if failure == 'mismatch' and len(calls) == 2 else binary
                    target.write_bytes(payload)
                    if failure == 'timing': observed['timings'].pop()
                    child = controller.adapter.owner.ProcessIdentity('12345678-1234-1234-1234-123456789abc',
                        1000 + len(calls), 1, 1000 + len(calls), 1000 + len(calls), 10,
                        1000, 1000, 'pid:[200]', 'c' * 64, 'S')
                    kwargs['on_child'](child)
                    kwargs['stdout'].write(entry.canonical(observed)); kwargs['stdout'].flush()
                    return controller.adapter.owner.SupervisionResult(child, (child,), 0, None,
                        ('declared cleanup failure',) if failure == 'cleanup' else (), (), True, None)

                owner = SimpleNamespace(Deadlines=controller.adapter.owner.Deadlines, supervise=supervise)
                modules = {'host_process_owner.py': owner, 'benchmark_artifacts.py': controller.artifacts,
                           'cache_gpu_fixture.py': reader}
                original_finish = controller.artifacts.CapturePipes.finish
                real_uid = os.getuid()
                class CaptureOS:
                    def getuid(self): return real_uid
                    def geteuid(self): return real_uid
                    def __getattr__(self, name): return getattr(os, name)
                def finish(cap):
                    result = original_finish(cap)
                    if failure == 'capture': raise ValueError('declared native capture finalization failure')
                    return result
                with patch.object(entry, 'Path', side_effect=path), \
                     patch.object(entry, 'adjacent', side_effect=lambda name, pin: modules[name]), \
                     patch.object(entry.os, 'getuid', return_value=1000), patch.object(entry.os, 'geteuid', return_value=1000), \
                     patch.object(entry.os, 'getgid', return_value=1000), patch.object(entry.os, 'getegid', return_value=1000), \
                     patch.object(controller.artifacts, 'os', CaptureOS()), \
                     patch.object(controller.artifacts.CapturePipes, 'finish', finish), patch('builtins.print'):
                    result = entry.run(spec, entry.sha(entry.canonical(spec)), 'bf16')
                self.assertFalse(result['qualification'])
                self.assertTrue(result['processes'], result)
                self.assertEqual(result['processes'][0]['supervision']['exit_code'], 0)
                self.assertTrue(result['processes'][0]['supervision']['child_reaped'])
                if failure:
                    self.assertEqual(len(calls), 2 if failure == 'mismatch' else 1)
                    self.assertEqual(result['status'], 'INCOMPLETE')
                    self.assertFalse(result['timings_usable'])
                    self.assertTrue(list((folder / 'output').glob('*.bin')))
                else:
                    variants = spec.get('variant_order', ['baseline', 'candidate'])
                    self.assertEqual(len(calls), 2 * len(variants))
                    self.assertEqual(result['status'], 'NATIVE_CACHE_ARM_COMPLETE', result)
                    self.assertEqual([row['variant'] for row in result['processes']], variants * 2)
                    self.assertEqual(len(result['comparisons']), 2 * (len(variants) - 1))
                    retained = entry.decode(entry.canonical(result))
                    self.assertEqual(controller.validate_result(retained, spec, 'bf16', reader=reader), retained)
                    self.assertEqual(len({argv[argv.index('--output') + 1] for argv in calls}), len(calls))
                    self.assertEqual(list((folder / 'output').glob('*.bin')), [])

    def test_actual_campaign_completion_failure_and_cleanup_with_cpu_process_observations(self):
        # The actual campaign and all result/release/resource validators run.
        # Only Linux lease effects, native supervision and root collection are
        # declared CPU substitutes; no controller CLI or GPU is invoked.
        for balanced, failure in ((b, f) for b in (False, True, 'profile') for f in (None, 'native', 'resource', 'release', 'finish', 'counts')):
            with self.subTest(balanced=balanced, failure=failure), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory).resolve()
                value, spec = package(balanced), native_spec(balanced)
                value['target_root'] = str(folder)
                resource = controller.resource_manifest(value, '5' * 64)
                prepared = {'manifest': value, 'spec': spec, 'resource_manifest': resource,
                    'resource_path': str(folder / 'resource.json'), 'stamps': {}}
                (folder / 'diagnostic.lock').write_bytes(b'')

                class Lease:
                    def __init__(self):
                        self.root, self.session, self.boot = folder, value['session_sha256'], value['boot_id']
                        self.lock_path = folder / 'diagnostic.lock'
                        self.journal = folder / 'unresolved.json'
                        info = self.lock_path.stat()
                        self.record = {'state': 'reserved', 'arms': [], 'lock_device': info.st_dev, 'lock_inode': info.st_ino}
                        self.fd = None
                    def save(self):
                        self.journal.write_bytes(entry.canonical(self.record))
                        info = self.journal.stat()
                        self.journal_identity = (info.st_dev, info.st_ino, entry.sha(self.journal.read_bytes()))
                    def acquire(self): self.fd = 9; self.save()
                    def _verify_journal(self):
                        if entry.sha(self.journal.read_bytes()) != self.journal_identity[2]: raise ValueError('changed fixture journal')
                    def begin_arm(self, arm, argv, path, pin):
                        self.record['arms'].append({'arm': arm, 'intent_monotonic': time.monotonic()})
                        self.record['state'] = 'launch-intent'; self.save()
                    def record_child(self, child):
                        self.record['arms'][-1]['child'] = asdict(child)
                        self.record['state'] = 'running'; self.save()
                    def finish_arm(self, result, evidence): self.record['state'] = 'released'; self.save()
                    def close(self):
                        self.fd = None; self.closed_monotonic = time.monotonic()
                        if self.record['state'] == 'released': self.record['state'] = 'pending-independent'
                        self.save()
                    def confirm_closed(self, evidence):
                        if self.record['state'] != 'pending-independent': raise ValueError('unreleased fixture arm')
                        self.journal.unlink()
                lease = Lease()
                observations, supervised = [], []

                def identity(pid, namespace):
                    return controller.adapter.owner.ProcessIdentity(value['boot_id'], pid, 1, pid, pid,
                        10, 1000, 1000, namespace, 'c' * 64, 'S')

                def resource_records(arm, gate):
                    initial = {'schema': 1, 'scope': resources.SCOPE, 'phase': 'initial', 'arm': arm,
                        'root': str(folder), 'session_sha256': value['session_sha256'],
                        **{key: resource[key] for key in ('gate_sha256', 'entry_sha256', 'bwrap_sha256', 'benchmark_spec_sha256')},
                        'workload_argv_sha256': resource['workload_argv_sha256'][arm],
                        'gate_process': {'pid': gate.pid, 'start_ticks': gate.start_ticks},
                        'cgroup_identity': {'device': 20, 'inode': 30}, 'actual_cgroup_verified': True,
                        'scratch_quota_verified': False, 'rlimits': {'core': [0, 0], 'file_size': [128 << 20] * 2},
                        'cgroup_path': resources.USER_MANAGER + '/app.slice/' + resource['units'][arm],
                        'cpu_max': [2400000, 100000], 'limits': {'memory.max': 38 << 30, 'memory.swap.max': 0, 'pids.max': 512},
                        'memory_events': {'max': 0, 'oom': 0, 'oom_kill': 0}, 'pids_events': {'max': 0},
                        'initial_usage': {'memory.current': 100, 'pids.current': 4}}
                    final = {key: initial[key] for key in ('schema', 'scope', 'arm', 'root', 'session_sha256',
                        'benchmark_spec_sha256', 'gate_sha256', 'entry_sha256', 'bwrap_sha256',
                        'workload_argv_sha256', 'gate_process', 'cgroup_identity')}
                    final.update(phase='final', complete=True, child_exit=0, child_reaped=True, owner_signals=[],
                        qualification=False, only_wrapper_remains=True, cgroup_release_proven=False,
                        resource_boundaries_clear=True, memory_events={'max': 0, 'oom': 0, 'oom_kill': 0},
                        pids_events={'max': 0}, cgroup_events={'populated': 1, 'frozen': 0},
                        final_usage={'memory.current': 100, 'memory.peak': 200, 'pids.current': 1})
                    snapshots = [dict(ordinal=i, started_monotonic=10.0 + i * .25,
                        finished_monotonic=10.01 + i * .25, complete=True,
                        **{key: initial[key] for key in ('gate_process', 'cgroup_identity', 'limits', 'cpu_max')},
                        **{key: final[key] for key in ('memory_events', 'pids_events', 'cgroup_events')},
                        usage=final['final_usage'], processes=[str(gate.pid)], threads=[str(gate.pid)],
                        only_wrapper_remains=True) for i in (0, 1)]
                    final['cgroup_observation'] = {'snapshots': snapshots, 'quiescence': {
                        'schema': 1, 'started_monotonic': 10., 'deadline_monotonic': 12., 'finished_monotonic': 10.26,
                        'maximum_seconds': 2., 'sample_interval_seconds': .2, 'sample_limit': 12,
                        'sample_bytes_limit': 4096, 'required_consecutive': 2, 'consecutive': 2, 'complete': True}}
                    if failure == 'resource': final['memory_events']['oom'] = 1
                    return initial, final

                def supervise(argv, **kwargs):
                    arm = lease.record['arms'][-1]['arm']
                    limits = kwargs['deadlines']
                    work_seconds = 900 if balanced else 480
                    self.assertAlmostEqual(limits.work_end - lease.record['arms'][-1]['intent_monotonic'],
                                           work_seconds, delta=1)
                    self.assertEqual(limits.kill_end - limits.work_end, 25)
                    self.assertEqual(limits.reap_end - limits.work_end, 30)
                    self.assertEqual(limits.term_end - limits.work_end, 20)
                    supervised.append(arm)
                    gate, child = identity(1000 + len(supervised), 'pid:[100]'), identity(2000 + len(supervised), 'pid:[200]')
                    kwargs['on_child'](gate)
                    native = complete_result(spec, arm)
                    if failure == 'native': native['processes'][0]['output_sha256'] = '8' * 64
                    if failure == 'counts': native['processes'][0]['native']['decoder_comparisons'] = 0
                    initial, final = resource_records(arm, gate)
                    for prefix, record in ((resources.RESOURCE_PREFIX, initial), (entry.PREFIX, native),
                                           (resources.RESOURCE_PREFIX, final)):
                        kwargs['stdout'].write(prefix.encode() + entry.canonical(record)); kwargs['stdout'].flush()
                    return controller.adapter.owner.SupervisionResult(gate, (gate, child), 0, None, (), (), True, None)

                def observe(request, **kwargs):
                    observations.append(request['mode'])
                    rows, mode = request['observed_processes'], request['mode']
                    snapshot = {key: request[key] for key in ('root', 'session_sha256', 'boot_id', 'lock_device', 'lock_inode')}
                    snapshot.update(schema=1, scope={'locked-admission': 'host-benchmark-locked-admission-v1',
                        'release': 'host-benchmark-release-v1', 'independent': 'host-benchmark-independent-release-v1'}[mode],
                        status='OBSERVATION_COMPLETE', failures=[], collector_source_sha256=kwargs['expected_observer_sha256'],
                        metadata_source_sha256=controller.base.METADATA_SHA256, observed_monotonic=time.monotonic(),
                        checked_processes=[list(x) for x in sorted({(r['boot_id'], r['pid'], r['start_ticks']) for r in rows})],
                        authenticated_processes_present=[], pid_namespace_members={r['pid_namespace']: [] for r in rows if r['pid_namespace'] != 'pid:[100]'},
                        holders={'visibility': 'all-users-platform', 'exit_code': 1, 'stdout': '', 'stderr': ''},
                        allocation={'scope': 'all-namespaces-active-workloads', 'gpu_requests': 0},
                        target={'hostname': 'threadripper-ai', 'uid': 1000, 'gid': 1000, 'boot_id': value['boot_id'],
                            'swap_active': False, 'node_ready': True, 'pressure_free': True, 'expected_policy_unchanged': True},
                        devices=[dict(bdf=d['bdf'], uuid=d['uuid'], render='/dev/dri/renderD' + str(128 + i),
                            sysfs_bdf=d['bdf']) for i, d in enumerate(spec['devices'])],
                        journal={'authenticated': True, 'state': lease.record['state'], 'sha256': request['journal_sha256']})
                    if mode != 'locked-admission':
                        arms = [request['arm']] if mode == 'release' else [r['arm'] for r in lease.record['arms']]
                        snapshot['root_resource_release'] = {'scopes': [dict(arm=arm, absent=True,
                            cgroup_path=transport.USER_CGROUP + resource['units'][arm], observed_monotonic=time.monotonic()) for arm in arms]}
                    if mode == 'release':
                        snapshot.update(arm=request['arm'], child_pid=request['root_process']['pid'],
                                        child_start_ticks=request['root_process']['start_ticks'])
                        if failure == 'release': snapshot['holders']['exit_code'] = 0
                    return snapshot

                original_finish = controller.artifacts.CapturePipes.finish
                def finish(cap):
                    receipt = original_finish(cap)
                    if failure == 'finish': raise ValueError('declared capture finalization failure')
                    return receipt
                with patch.object(controller.adapter.owner, 'CanonicalLease', return_value=lease), \
                     patch.object(resources, 'scope_argv', return_value=('/fixture/frozen-command',)), \
                     patch.object(entry, 'adjacent', return_value=reader), \
                     patch.object(controller.artifacts.CapturePipes, 'finish', finish):
                    result = controller.run_campaign(prepared, {'received_monotonic': time.monotonic()},
                        SimpleNamespace(observe=observe), supervisor=supervise)
                self.assertFalse(result['qualification'])
                self.assertEqual(result['model_starts'], 0)
                self.assertEqual(observations[0], 'locked-admission')
                self.assertEqual(observations[-1], 'independent')
                if failure is None:
                    self.assertEqual(result['status'], 'NATIVE_CACHE_REGRESSION_COMPLETE', result)
                    self.assertEqual(supervised, ['bf16', 'kv4'])
                    self.assertTrue(result['timings_usable'])
                    self.assertFalse(result['unresolved'])
                else:
                    self.assertEqual(result['status'], 'INCOMPLETE')
                    self.assertEqual(supervised, ['bf16'])
                    self.assertFalse(result['timings_usable'])
                    self.assertEqual(result['unresolved'], failure == 'release')

    def test_exact_spec_and_native_command(self):
        value = native_spec()
        self.assertEqual(entry.validate_spec(value), value)
        argv = entry.native_argv(value, 'kv4', value['devices'][0], 'candidate', '/mnt/output/cache/0-candidate.bin')
        self.assertEqual(argv[:4], ['/mnt/native/fixture', '--execute-gpu', '--library', '/mnt/native/candidate.so'])
        self.assertEqual(argv[argv.index('--expected-uuid') + 1], value['devices'][0]['uuid'])
        for key, changed in (('native_processes', 9), ('model_starts', 1), ('timing_repeats', 6),
                             ('experiment_mode', 'model-benchmark-v1'), ('arm_order', ['kv4', 'bf16'])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_spec(dict(value, **{key: changed}))
        bad = copy.deepcopy(value)
        bad['devices'][1] = bad['devices'][0]
        with self.assertRaises(ValueError):
            entry.validate_spec(bad)

    def test_counterbalanced_spec_requires_exact_order_warmup_and_process_budget(self):
        value = native_spec(True)
        self.assertEqual(entry.validate_spec(value), value)
        argv = entry.native_argv(value, 'kv4', value['devices'][0], 'candidate', '/mnt/output/cache/0-1-candidate.bin')
        self.assertEqual(argv[-4:], ['--timing-protocol', entry.TIMING_PROTOCOL, '--timing-warmups', '2'])
        legacy = native_spec()
        self.assertNotIn('--timing-protocol', entry.native_argv(legacy, 'bf16', legacy['devices'][0], 'baseline', '/unused'))
        changes = [('native_processes', 8), ('timing_warmups', 0), ('timing_warmups', True),
                   ('timing_protocol', 'legacy-v1'), ('variant_order', ['baseline', 'candidate'] * 2),
                   ('variant_order', ['candidate', 'baseline', 'baseline', 'candidate'])]
        for key, changed in changes:
            with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                entry.validate_spec(dict(value, **{key: changed}))
        for key in entry.COUNTERBALANCED_FIELDS:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                entry.validate_spec({k: v for k, v in value.items() if k != key})
        with self.assertRaises(ValueError):
            entry.validate_spec(dict(value, experiment_mode=entry.MODE))

    def test_native_child_binds_every_occurrence_to_variant_and_output(self):
        for balanced in (False, True, 'profile'):
            spec = native_spec(balanced)
            variants = spec.get('variant_order', ['baseline', 'candidate'])
            for ordinal, variant in enumerate(variants):
                name = '0-' + (str(ordinal) + '-' if balanced else '') + variant
                argv = ['--native-child', '--spec-sha256', entry.sha(entry.canonical(spec)),
                        '--arm', 'kv4', '--device', '0', '--variant', variant,
                        '--output', '/mnt/output/cache/' + name + '.bin']
                if balanced:
                    argv.extend(('--ordinal', str(ordinal)))
                with self.subTest(balanced=balanced, ordinal=ordinal), \
                     patch.object(entry, 'read', return_value=entry.canonical(spec)), \
                     patch.object(entry.os, 'execv', side_effect=RuntimeError('declared native exec substitute')) as execute:
                    with self.assertRaisesRegex(RuntimeError, 'declared native exec substitute'):
                        entry.main(argv)
                    called = execute.call_args.args[1]
                    self.assertEqual(called[called.index('--library') + 1], '/mnt/native/' + variant + '.so')
                    self.assertEqual(called[called.index('--output') + 1], '/mnt/output/cache/' + name + '.bin')
                    execute.reset_mock()
                    bad = list(argv)
                    bad[bad.index('--output') + 1] = '/mnt/output/cache/other.bin'
                    with self.assertRaises(ValueError):
                        entry.main(bad)
                    execute.assert_not_called()
                    if balanced:
                        bad = list(argv)
                        bad[bad.index('--ordinal') + 1] = str((ordinal + 2) % 4)
                        with self.assertRaises(ValueError):
                            entry.main(bad)
                        execute.assert_not_called()

    def test_projection_is_no_model_no_network_no_host_writes_and_same_resources(self, balanced=False):
        manifest = package(balanced)
        resource = controller.resource_manifest(manifest, '5' * 64)
        self.assertEqual(resources.validate_manifest(resource), resource)
        for arm in ('bf16', 'kv4'):
            argv = resource['workload_argv'][arm]
            self.assertNotIn('--share-net', argv)
            self.assertNotIn('--bind', argv)
            self.assertNotIn('/mnt/models', argv)
            self.assertNotIn('model_benchmark.py', argv)
            self.assertEqual(argv[-2:], ['--arm', arm])
            self.assertEqual(argv.count('--execute-native-cache'), 1)
            self.assertIn(str(64 << 20), argv)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'resource.json'
            path.write_bytes(entry.canonical(resource))
            argv = resources.scope_argv(path, entry.sha(path.read_bytes()), 'bf16')
            self.assertIn('--property=MemoryMax=' + str(38 << 30), argv)
            self.assertIn('--property=MemorySwapMax=0', argv)
            self.assertIn('--property=TasksMax=512', argv)

    def test_counterbalanced_projection_preserves_resource_limits(self):
        self.test_projection_is_no_model_no_network_no_host_writes_and_same_resources(True)

    def test_resource_mode_substitution_and_mutated_tail_refuse(self):
        original = controller.resource_manifest(package(), '5' * 64)
        for change in ('mode', 'entry', 'arm', 'network', 'host-write'):
            value = copy.deepcopy(original)
            if change == 'mode': value.pop('experiment_mode')
            if change == 'entry': value['entry_path'] = value['entry_path'].replace('cache_gpu_entry', 'model_benchmark')
            if change == 'arm': value['workload_argv']['bf16'][-1] = 'kv4'
            if change == 'network': value['workload_argv']['bf16'].insert(1, '--share-net')
            if change == 'host-write': value['workload_argv']['bf16'].insert(1, '--bind')
            value['workload_argv_sha256']['bf16'] = resources.argv_sha256(value['workload_argv']['bf16'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                resources.validate_manifest(value)

    def test_native_grant_has_zero_models_and_strict_original_clock(self, balanced=False):
        value, pin, now = package(balanced), '6' * 64, time.monotonic()
        parent = {'pid': 42, 'start_ticks': 100, 'uid': 0, 'boot_id': value['boot_id']}
        instant = datetime.now(timezone.utc)
        grant = transport.new_grant(value, pin, received_utc=instant.isoformat().replace('+00:00', 'Z'),
                                    received_monotonic=now, parent=parent)
        controller.validate_grant(grant, value, pin, parent, clock=lambda: now, utc_now=lambda: instant)
        self.assertFalse(grant['model_start_authorized'])
        self.assertEqual((grant['maximum_starts'], grant['maximum_native_processes']), (0, 16 if balanced else 8))
        self.assertEqual(grant['cumulative_ceiling_seconds'], 3000 if balanced else 2400)
        self.assertEqual(grant['fixed_reserve_seconds'], 300)
        for change in ({'maximum_starts': 1}, {'model_start_authorized': True}, {'maximum_native_processes': 9},
                       {'scope': 'one-matched-fp8-kv-benchmark'}, {'manifest_sha256': '7' * 64},
                       {'cumulative_ceiling_seconds': 2400 if balanced else 3000},
                       {'maximum_native_processes': 8 if balanced else 16},
                       {'received_monotonic': now - 106}, {'received_monotonic': now + 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                controller.validate_grant(dict(grant, **change), value, pin, parent,
                                          clock=lambda: now, utc_now=lambda: instant)

    def test_counterbalanced_grant_has_zero_models_and_strict_original_clock(self):
        self.test_native_grant_has_zero_models_and_strict_original_clock(True)

    def test_native_allocations_timings_and_complete_result_refuse_substitution(self):
        for balanced, arm in ((b, a) for b in (False, True, 'profile') for a in ('bf16', 'kv4')):
            spec = native_spec(balanced)
            value = complete_result(spec, arm)
            self.assertEqual(controller.validate_result(value, spec, arm, reader=reader), value)
            changes = [lambda v: v.update(qualification=True), lambda v: v.update(model_starts=1),
                lambda v: v['processes'].pop(), lambda v: v['comparisons'].pop(),
                lambda v: v['processes'][0]['native'].update(uuid='unknown'),
                lambda v: v['processes'][0]['native'].update(visible_devices=1),
                lambda v: v['processes'][0]['native']['timings'][0].update(synchronized_call_ms=float('nan')),
                lambda v: v['processes'][0]['supervision'].update(survivors=[{'pid': 1}]),
                lambda v: v['processes'][0].update(output_sha256='8' * 64),
                lambda v: v['processes'][0].update(variant='candidate'),
                lambda v: v.update(spec_sha256='9' * 64),
                lambda v: v['processes'][0]['native'].update(status='INCOMPLETE'),
                lambda v: v['processes'][0]['native'].update(decoder_comparisons=0),
                lambda v: v['processes'][0]['native'].update(device_buffer_free_errors=1),
                lambda v: v['processes'][0]['native'].update(qualification=True),
                lambda v: v['comparisons'][0].update(compared_bytes=0)]
            if balanced:
                changes.extend([lambda v: v['processes'][2].update(ordinal=1),
                    lambda v: v['processes'][2].update(variant='baseline'),
                    lambda v: v['processes'][2]['native']['timings'].pop(),
                    lambda v: v['processes'][2]['native'].update(timing_warmups=0),
                    lambda v: v['processes'][2]['native'].update(timing_protocol='legacy-v1'),
                    lambda v: v['comparisons'][1].update(reference_ordinal=1),
                    lambda v: v['comparisons'][1].update(compared_ordinal=1),
                    lambda v: v['processes'][3].update(output_sha256='8' * 64)])
            for change in changes:
                bad = copy.deepcopy(value)
                change(bad)
                with self.assertRaises(ValueError):
                    controller.validate_result(bad, spec, arm, reader=reader)

    def test_root_transport_archive_requires_explicit_native_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'payload.tar'
            for mode in (entry.MODE, *entry.BALANCED_MODES, 'unknown'):
                data = package('profile' if mode == entry.PHASE_PROFILE_MODE else mode == entry.COUNTERBALANCED_MODE)
                data['experiment_mode'] = mode
                member = b'CPU-only inert source\n'
                data['package_files'] = {'review/inert.py': entry.sha(member)}
                raw = entry.canonical(data)
                with tarfile.open(path, 'w') as archive:
                    for name, contents in (('MANIFEST.json', raw), ('review/inert.py', member)):
                        item = tarfile.TarInfo(name)
                        item.size = len(contents)
                        archive.addfile(item, io.BytesIO(contents))
                if mode in (entry.MODE, *entry.BALANCED_MODES):
                    self.assertEqual(transport.inspect_archive(path, entry.sha(path.read_bytes()), entry.sha(raw))[0], data)
                else:
                    with self.assertRaises(ValueError):
                        transport.inspect_archive(path, entry.sha(path.read_bytes()), entry.sha(raw))

    def test_complete_envelopes_include_inner_cleanup(self):
        self.assertLessEqual(4 * 105 + 30, controller.ARM_WORK)
        self.assertEqual(controller.ARM_ENVELOPE, controller.ARM_WORK + 30 + 75)
        self.assertLessEqual(180 + 2 * controller.ARM_ENVELOPE + 330 + 300, 2400)
        self.assertLessEqual(8 * 105 + 30, controller.COUNTERBALANCED_ARM_WORK)
        self.assertEqual(controller.COUNTERBALANCED_ARM_ENVELOPE, controller.COUNTERBALANCED_ARM_WORK + 30 + 75)
        self.assertLessEqual(180 + 2 * controller.COUNTERBALANCED_ARM_ENVELOPE + 330 + 300, 3000)
        for arm in ('bf16', 'kv4'):
            members = reader.expected_records(arm, timing_protocol=entry.TIMING_PROTOCOL, timing_repeats=5)
            full_bytes = 8 + sum(12 + len(name) + size for name, size in members)
            self.assertLess(full_bytes, 32 << 20)
            self.assertLess(2 * full_bytes + 8 * 2 * (64 << 10), 64 << 20)
        self.assertNotIn('torch', entry.run.__code__.co_names)

    def test_actual_main_forwards_mode_bound_seal_budget(self):
        for balanced in (False, True, 'profile'):
            value = package(balanced)
            root = Path(value['target_root'])
            grant = {'received_monotonic': 100., 'received_utc': '2026-10-15T00:00:00Z'}
            rpc = SimpleNamespace(authenticate_parent=lambda: {}, close=lambda: None)
            report = {'status': 'NATIVE_CACHE_REGRESSION_COMPLETE', 'qualification': False}
            args = ['--execute-owner-cache', '--manifest', str(root / 'MANIFEST.json'),
                    '--manifest-sha256', 'a' * 64, '--grant-json', str(root / 'GRANT.json'),
                    '--grant-sha256', 'b' * 64, '--observer-fd', '3']
            # Substitute external effects only; exercise real CLI and final-budget routing.
            with self.subTest(balanced=balanced), \
                 patch.object(controller.os, 'getuid', return_value=1000), \
                 patch.object(controller.os, 'geteuid', return_value=1000), \
                 patch.object(controller.os, 'getgid', return_value=1000), \
                 patch.object(controller.os, 'getegid', return_value=1000), \
                 patch.object(entry, 'read', side_effect=[entry.canonical(grant), entry.canonical(value)]), \
                 patch.object(transport, 'RpcClient', return_value=rpc), \
                 patch.object(controller, 'validate_grant'), \
                 patch.object(controller, 'validate_manifest', return_value={'declared': 'CPU preparation substitute'}) as validate, \
                 patch.object(controller, 'run_campaign', return_value=report), \
                 patch.object(controller.base, '_publish') as publish, \
                 patch.object(controller.base, 'seal_result') as seal:
                self.assertEqual(controller.main(args), 0)
                self.assertEqual(validate.call_args.args[-1], 205.)
                publish.assert_called_once_with(root / 'CACHE-RESULT.json', report)
                ceiling = 3000 if balanced else 2400
                seal.assert_called_once_with(root, deadline=100. + ceiling, received_monotonic=100.,
                                             received_utc=grant['received_utc'], complete_seconds=ceiling)

    def test_profile_package_binding_and_existing_baseline_refusal(self):
        self.test_actual_package_validator_checks_final_bytes_and_projection('profile')

    def test_profile_projection_preserves_resources(self):
        self.test_projection_is_no_model_no_network_no_host_writes_and_same_resources('profile')

    def test_profile_grant_retains_zero_models_and_full_cleanup(self):
        self.test_native_grant_has_zero_models_and_strict_original_clock('profile')

    def test_profile_selection_cannot_leak_into_other_modes(self):
        value = native_spec('profile')
        self.assertEqual(entry.validate_spec(value), value)
        for variant in ('baseline', 'candidate'):
            argv = entry.native_argv(value, 'kv4', value['devices'][0], variant, '/unused')
            self.assertEqual('--phase-profile' in argv, variant == 'candidate')
        for mode in (entry.MODE, entry.COUNTERBALANCED_MODE, 'native-cache-phase-profile-v2'):
            with self.assertRaises(ValueError):
                entry.validate_spec(dict(value, experiment_mode=mode))
        for field, bad in (('schema', True), ('variant', 'baseline'), ('boundary', 'gpu-kernel'),
                           ('source_sha256', 'f' * 64), ('unprofiled_candidate_sha256', '0' * 64),
                           ('projected_source_sha256', entry.PROFILE_SOURCE_SHA256),
                           ('projector_sha256', None), ('profile_header_sha256', 'invalid')):
            mutated = copy.deepcopy(value)
            mutated['phase_profile'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                entry.validate_spec(mutated)
        mutated = copy.deepcopy(value)
        mutated['native_files']['candidate.so'] = entry.UNPROFILED_CANDIDATE_SHA256
        with self.assertRaises(ValueError):
            entry.validate_spec(mutated)

    def test_profile_full_reader_rejects_missing_stale_and_substituted_phases(self):
        spec = native_spec('profile')
        value = complete_result(spec, 'kv4')
        controller.validate_result(value, spec, 'kv4', reader=reader)
        changes = [
            lambda v: v['processes'][1]['native'].pop('phase_profile'),
            lambda v: v['processes'][1]['native']['timings'][0].pop('phase_wall'),
            lambda v: v['processes'][0]['native'].update(phase_profile=True),
            lambda v: v['processes'][1]['native']['timings'][0]['phase_wall'].update(total_ns=1),
            lambda v: v['processes'][1]['native']['timings'][0]['phase_wall'].update(sequence=True),
            lambda v: v['processes'][1]['native']['timings'][1]['phase_wall'].update(sequence=4),
            lambda v: v['processes'][1]['native']['timings'][0]['phase_wall'].update(reader_sync_ns=-1),
            lambda v: v['processes'][1]['native']['timings'][0]['phase_wall'].update(extra=1),
            lambda v: v['processes'][1]['native']['timings'][0]['phase_wall'].update(validation_status_ns=float('nan')),
            lambda v: v['processes'][1]['native'].update(phase_profile=1),
        ]
        for change in changes:
            bad = copy.deepcopy(value)
            change(bad)
            with self.assertRaises(ValueError):
                controller.validate_result(bad, spec, 'kv4', reader=reader)


if __name__ == '__main__':
    unittest.main()
