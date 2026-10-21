"""One human-authorized native cache regression, using existing host ownership."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time


def adjacent(name):
    path = Path(__file__).resolve().with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('spry_cache_controller_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = adjacent('benchmark_physical_controller')
entry = adjacent('cache_gpu_entry')
transport = adjacent('benchmark_host_transport')
artifacts = adjacent('benchmark_artifacts')
adapter, resources, need = base.adapter, base.resources, base.need
SOURCE_NAMES = base.LIVE_SOURCE_NAMES + ('cache_gpu_controller.py', 'cache_gpu_entry.py', 'cache_gpu_fixture.py')
PACKAGE_NAMES = {'review/' + name for name in SOURCE_NAMES} | {
    'review/NATIVE-SPEC.json', 'review/RESOURCE-MANIFEST.json', 'review/COMPOSITION-FACTS.json'}
FIELDS = {'schema', 'scope', 'experiment_mode', 'target_root', 'package_root', 'boot_id', 'session_sha256',
          'package_files', 'retained_root', 'retained_base', 'native_root', 'native_files',
          'spec_sha256', 'resource_manifest_sha256', 'launch_authorized', 'proposed_complete_seconds'}
ARM_WORK, ARM_ENVELOPE = 480, 585
COUNTERBALANCED_ARM_WORK, COUNTERBALANCED_ARM_ENVELOPE = 900, 1005
# Measured decoder/validator library; independent of the installed model runtime.
NATIVE_BASELINE_SHA256 = '3bf86042ba225abc279245b41e1baa35c2549def0e8547eaf60199c4b55d8849'


def bwrap_argv(value, arm):
    need(arm in ('bf16', 'kv4'), 'cache arm')
    root, retained = Path(value['target_root']), Path(value['retained_root'])
    review = root / 'review'
    argv = ['/usr/bin/bwrap', '--die-with-parent', '--new-session', '--unshare-all',
        '--uid', '1000', '--gid', '1000', '--ro-bind', str(retained / 'base-rootfs'), '/',
        '--tmpfs', '/mnt', '--ro-bind', str(review), '/mnt/review',
        '--ro-bind', str(review / 'cache_gpu_entry.py'), '/mnt/review/cache_gpu_entry.py',
        '--ro-bind', str(retained / 'build-output3/build-venv'), '/mnt/work/build-venv',
        '--ro-bind', value['native_root'], '/mnt/native',
        '--size', str(64 << 20), '--tmpfs', '/mnt/output',
        '--proc', '/proc', '--dev', '/dev', '--size', str(64 << 20), '--tmpfs', '/dev/shm',
        '--dev-bind', '/dev/kfd', '/dev/kfd',
        '--dev-bind', '/dev/dri/renderD128', '/dev/dri/renderD128',
        '--dev-bind', '/dev/dri/renderD129', '/dev/dri/renderD129', '--ro-bind', '/sys', '/sys',
        '--size', str(128 << 20), '--tmpfs', '/tmp', '--clearenv']
    for key, item in {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'HOME': '/tmp', 'TMPDIR': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'LD_LIBRARY_PATH': '/usr/local/lib/python3.14/dist-packages/_rocm_sdk_core/lib'}.items():
        argv.extend(('--setenv', key, item))
    argv.extend(('/mnt/work/build-venv/bin/python3', '-I', '-B', '-S', '/mnt/review/cache_gpu_entry.py',
                 '--execute-native-cache', '--spec-sha256', value['spec_sha256'], '--arm', arm))
    return argv


def resource_manifest(value, bwrap_sha256):
    root = Path(value['target_root'])
    files = value['package_files']
    work = {arm: bwrap_argv(value, arm) for arm in ('bf16', 'kv4')}
    return {'schema': 1, 'scope': resources.SCOPE, 'experiment_mode': value['experiment_mode'],
        'root': str(root), 'session_sha256': value['session_sha256'], 'benchmark_spec_sha256': value['spec_sha256'],
        'gate_path': str(root / 'review/benchmark_resource_gate.py'),
        'gate_sha256': files['review/benchmark_resource_gate.py'],
        'entry_path': str(root / 'review/cache_gpu_entry.py'), 'entry_sha256': files['review/cache_gpu_entry.py'],
        'artifacts_sha256': files['review/benchmark_artifacts.py'], 'bwrap_sha256': bwrap_sha256,
        'units': {arm: 'spry-fp8-benchmark-' + value['session_sha256'][:16] + '-' + arm + '.scope'
                  for arm in ('bf16', 'kv4')}, 'workload_argv': work,
        'workload_argv_sha256': {arm: resources.argv_sha256(argv) for arm, argv in work.items()}}


def validate_manifest(path, expected, deadline, *, clock=time.monotonic):
    value = entry.decode(entry.read(path, 256 << 10, expected))
    balanced = value.get('experiment_mode') in entry.BALANCED_MODES if type(value) is dict else False
    need(type(value) is dict and set(value) == FIELDS and value['schema'] == 3 and
         value['scope'] == 'native-cache-regression-package-v1' and
         value['experiment_mode'] in (entry.MODE, *entry.BALANCED_MODES)
         and value['launch_authorized'] is False and value['proposed_complete_seconds'] == (3000 if balanced else 2400),
         'exact native package contract')
    need(value['target_root'] == value['package_root'] and
         re.fullmatch(r'/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}', value['target_root'])
         and entry.digest(value['session_sha256']) and
         re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value['boot_id']), 'target/session binding')
    root = base._canonical(value['target_root'])
    need(Path(path) == root / 'MANIFEST.json' and Path(__file__).resolve() == root / 'review/cache_gpu_controller.py',
         'fresh exact controller/package path')
    need(set(value['package_files']) == PACKAGE_NAMES, 'exact cache package inventory')
    stamps = base._tree(root / 'review', {name.removeprefix('review/'): pin
        for name, pin in value['package_files'].items()}, deadline, clock)
    for name in SOURCE_NAMES:
        need(base._hash(Path(__file__).resolve().with_name(name), deadline, clock)[0] ==
             value['package_files']['review/' + name], 'executed controller input differs')
    spec = entry.validate_spec(entry.decode(entry.read(root / 'review/NATIVE-SPEC.json', 64 << 10, value['spec_sha256'])))
    need(value['spec_sha256'] == value['package_files']['review/NATIVE-SPEC.json'] and
         spec['experiment_mode'] == value['experiment_mode'] and
         spec['native_files'] == value['native_files'] and spec['session_sha256'] == value['session_sha256'],
         'spec/source identity binding')
    for key, name in (('entry_sha256', 'cache_gpu_entry.py'), ('reader_sha256', 'cache_gpu_fixture.py'),
                      ('owner_sha256', 'host_process_owner.py'), ('artifacts_sha256', 'benchmark_artifacts.py')):
        need(spec[key] == value['package_files']['review/' + name], 'entry helper source binding')
    native = base._canonical(value['native_root'])
    stamps.update(base._tree(native, value['native_files'], deadline, clock))
    need(os.access(native / 'fixture', os.X_OK) and value['native_files']['baseline.so'] == NATIVE_BASELINE_SHA256,
         'executable fixture and accepted baseline library required')
    resource_path = root / 'review/RESOURCE-MANIFEST.json'
    resource = resources.read_manifest(resource_path, value['resource_manifest_sha256'])
    need(value['resource_manifest_sha256'] == value['package_files']['review/RESOURCE-MANIFEST.json'] and
         resource == resource_manifest(value, resource['bwrap_sha256']), 'exact native resource projection')
    base.require_retained_base_binding(value['retained_base'])
    need(value['package_files']['review/COMPOSITION-FACTS.json'] == base.FACTS_SHA256,
         'retained runtime facts identity')
    facts = entry.decode(entry.read(root / 'review/COMPOSITION-FACTS.json', 16 << 20, base.FACTS_SHA256))
    retained = base.retained_verifier.verify_retained(base._canonical(value['retained_root']), facts, deadline)
    return {'manifest': value, 'resource_manifest': resource, 'resource_path': str(resource_path),
            'spec': spec, 'stamps': stamps, 'retained_identity': retained}


def validate_grant(grant, manifest, pin, parent, *, clock=time.monotonic, utc_now=lambda: datetime.now(timezone.utc)):
    expected = transport.new_grant(manifest, pin, received_utc=grant.get('received_utc'),
        received_monotonic=grant.get('received_monotonic'), parent=parent)
    need(type(grant) is dict and grant == expected and all(type(grant[key]) is type(item) for key, item in expected.items())
         and grant['maximum_starts'] == 0 and
         grant['maximum_native_processes'] == (16 if manifest['experiment_mode'] in entry.BALANCED_MODES else 8)
         and grant['model_start_authorized'] is False,
         'explicit native-only human grant required')
    received = grant['received_monotonic']
    need(type(received) in (int, float) and math.isfinite(received) and received <= clock() < received + 105,
         'original native receipt/admission window')
    instant = datetime.fromisoformat(grant['received_utc'].replace('Z', '+00:00'))
    need(instant.tzinfo is not None and abs((utc_now() - instant).total_seconds() - (clock() - received)) <= 3,
         'UTC/monotonic receipt differs')


def validate_result(value, spec, arm, *, reader=None):
    entry.validate_spec(spec)
    reader = entry.adjacent('cache_gpu_fixture.py', spec['reader_sha256']) if reader is None else reader
    balanced = spec['experiment_mode'] in entry.BALANCED_MODES
    variants = spec['variant_order'] if balanced else ('baseline', 'candidate')
    count = len(variants)
    need(type(value) is dict and value.get('schema') == 1 and value.get('experiment_mode') == spec['experiment_mode'] and
         value.get('arm') == arm and value.get('spec_sha256') == entry.sha(entry.canonical(spec)) and
         value.get('session_sha256') == spec['session_sha256'] and value.get('qualification') is False and
         value.get('model_starts') == 0 and value.get('status') == 'NATIVE_CACHE_ARM_COMPLETE' and
         value.get('timings_usable') is True and len(value.get('processes', [])) == 2 * count and
         len(value.get('comparisons', [])) == 2 * (count - 1) and 'failure' not in value, 'complete native arm required')
    for ordinal, row in enumerate(value['processes']):
        device, variant = spec['devices'][ordinal // count], variants[ordinal % count]
        need(row['device'] == device['device'] and row['variant'] == variant and row['exit_code'] == 0
             and entry.digest(row['output_sha256']) and 0 < row['output_bytes'] <= 32 << 20,
             'native child order/count/output')
        if balanced:
            need(type(row.get('ordinal')) is int and row['ordinal'] == ordinal % count,
                 'native child occurrence order')
        entry.validate_native(row['native'], spec, arm, device, variant)
        reader.summary(row['native'], **entry.reader_options(spec, variant))
        need(row['output_bytes'] == row['native']['output_bytes'], 'native report/output size mismatch')
        supervision = row['supervision']
        need(supervision['exit_code'] == 0 and supervision['failure'] is None and supervision['child_reaped'] is True
             and supervision['cleanup_errors'] == [] and supervision['survivors'] == []
             and supervision['owner_signal'] is None and row['capture']['capture_complete'] is True,
             'native child release incomplete')
    for index, comparison in enumerate(value['comparisons']):
        device_index, compared_ordinal = divmod(index, count - 1)
        compared_ordinal += 1
        members = (reader.expected_records(arm, timing_protocol=spec['timing_protocol'],
                                           timing_repeats=spec['timing_repeats']) if balanced
                   else reader.expected_records(arm))
        expected = {'schema': 1, 'status': 'NATIVE_OLD_CANDIDATE_BITS_IDENTICAL', 'qualification': False,
                    'records': len(members), 'compared_bytes': sum(size for _, size in members),
                    'bdf': spec['devices'][device_index]['bdf'], 'scope': 'synthetic-cache-ABI-only'}
        if balanced:
            expected.update(reference_ordinal=0, compared_ordinal=compared_ordinal)
        full_bytes = 8 + sum(12 + len(name) + size for name, size in members)
        reference = value['processes'][device_index * count]
        compared = value['processes'][device_index * count + compared_ordinal]
        need(type(comparison) is dict and comparison == expected and
             all(type(comparison[key]) is type(item) for key, item in expected.items()) and
             all(row['output_bytes'] == full_bytes for row in (reference, compared))
             and reference['output_sha256'] == compared['output_sha256'],
             'exact old/candidate native comparison failed')
    return value


def run_campaign(prepared, grant, rpc, *, clock=time.monotonic, supervisor=None):
    supervisor = adapter.owner.supervise if supervisor is None else supervisor
    value, spec = prepared['manifest'], prepared['spec']
    balanced = spec['experiment_mode'] in entry.BALANCED_MODES
    arm_work, arm_envelope = ((COUNTERBALANCED_ARM_WORK, COUNTERBALANCED_ARM_ENVELOPE) if balanced
                             else (ARM_WORK, ARM_ENVELOPE))
    root = Path(value['target_root'])
    output = root / 'results'
    output.mkdir(mode=0o700)
    lease = adapter.owner.CanonicalLease(adapter.CANONICAL_LOCK, root, value['session_sha256'], value['boot_id'])
    end = grant['received_monotonic'] + (2700 if balanced else 2100)
    report = {'schema': 1, 'experiment_mode': spec['experiment_mode'], 'status': 'INCOMPLETE', 'qualification': False,
        'model_starts': 0, 'arms': [], 'cleanup_errors': [], 'independent_release_verified': False,
        'timings_usable': False, 'observer_requests': []}
    results, acquired = [], False
    observer_pin = value['package_files']['review/benchmark_release_observer.py']

    def observe(mode, arm=None, result=None, deadline=None):
        now = clock()
        until = min(end, now + 75, deadline if deadline is not None else end)
        need(until - now >= 60, 'complete root observation envelope does not fit')
        history = tuple(item for one in results for item in one.observed_processes) if mode == 'independent' else (
            result.observed_processes if result is not None else ())
        info = Path(lease.lock_path).lstat()
        request = {'schema': 1, 'mode': mode, 'root': str(root), 'session_sha256': lease.session,
            'boot_id': lease.boot, 'lock_device': info.st_dev, 'lock_inode': info.st_ino, 'arm': arm,
            'root_process': asdict(result.root_process) if result is not None else None,
            'observed_processes': base.release_process_inventory(history, result.root_process if result else None),
            'journal_sha256': entry.sha(entry.read(lease.journal, 256 << 10)) if lease.journal.exists() else None,
            'issued_monotonic': until - 75, 'deadline_monotonic': until, 'native_ssh_host_contract': True}
        report['observer_requests'].append(request)
        snapshot = rpc.observe(request, expected_observer_sha256=observer_pin, deadline_monotonic=until)
        need(clock() <= until and snapshot.get('status') == 'OBSERVATION_COMPLETE' and snapshot.get('failures') == []
             and snapshot.get('collector_source_sha256') == observer_pin and
             snapshot.get('metadata_source_sha256') == base.METADATA_SHA256, 'root observation failed')
        if mode != 'locked-admission':
            arms = [arm] if mode == 'release' else [row['arm'] for row in lease.record['arms']]
            scopes = snapshot.get('root_resource_release', {}).get('scopes', [])
            need([row.get('arm') for row in scopes] == arms, 'complete root cgroup absence required')
            for row in scopes:
                need(row.get('absent') is True and row.get('cgroup_path') == transport.USER_CGROUP +
                     prepared['resource_manifest']['units'][row['arm']] and
                     now <= row.get('observed_monotonic', -1) <= clock(), 'root cgroup release differs')
        return snapshot, until

    try:
        need(clock() + 75 + 2 * arm_envelope + 330 <= end, 'complete native campaign does not fit')
        lease.acquire()
        acquired = True
        started = clock()
        snapshot, deadline = observe('locked-admission', deadline=grant['received_monotonic'] + 180)
        report['admission'] = snapshot
        adapter.locked_admission_evidence(snapshot, lease=lease, collector_sha256=observer_pin,
            not_before=started, deadline=deadline, clock=clock)
        base.verify_unchanged(prepared)
        for arm in ('bf16', 'kv4'):
            start = clock()
            need(start + (2 - len(results)) * arm_envelope + 330 <= end, 'remaining native envelope does not fit')
            argv = resources.scope_argv(prepared['resource_path'], value['resource_manifest_sha256'], arm)
            path, pin = prepared['resource_manifest']['gate_path'], prepared['resource_manifest']['gate_sha256']
            lease.begin_arm(arm, argv, path, pin)
            row = {'arm': arm, 'release_verified': False, 'complete': False}
            report['arms'].append(row)
            cap = artifacts.CapturePipes(output / (arm + '-capture'), stream_limit=2 << 20)
            result = None
            try:
                result = supervisor(argv, script_path=path, script_sha256=pin,
                    deadlines=adapter.owner.Deadlines(start + arm_work, start + arm_work + 20,
                                                      start + arm_work + 25, start + arm_work + 30),
                    on_child=lease.record_child, stdout=cap.stdout, stderr=cap.stderr, resource_check=cap.check)
                row['supervision'] = asdict(result)
                results.append(result)
            finally:
                try:
                    row['capture'] = cap.finish()
                except BaseException as error:
                    row['capture'] = {'capture_complete': False, 'failure': type(error).__name__ + ': ' + str(error)[:512]}
            need(result is not None, 'supervisor returned no owned-process result')
            post = clock()
            snapshot, deadline = observe('release', arm, result, start + arm_envelope)
            row['release_snapshot'] = snapshot
            release = adapter.release_evidence(snapshot, lease=lease, arm=arm, result=result,
                collector_sha256=observer_pin, not_before=post, deadline=deadline, clock=clock)
            lease.finish_arm(result, release)
            row['release_verified'] = True
            need(result.process_cleanup_observed and result.exit_code == 0 and result.failure is None and
                 result.owner_signal is None and row['capture']['capture_complete'] is True, 'native arm failed')
            stdout = entry.read(output / (arm + '-capture/stdout.raw'), 2 << 20)
            native = [line[len(entry.PREFIX):] for line in stdout.decode().splitlines() if line.startswith(entry.PREFIX)]
            need(len(native) == 1, 'one complete native result required')
            row['native_result'] = validate_result(entry.decode(native[0]), spec, arm)
            observed = [entry.decode(line[len(resources.RESOURCE_PREFIX):]) for line in stdout.decode().splitlines()
                        if line.startswith(resources.RESOURCE_PREFIX)]
            need(len(observed) == 2 and [v.get('phase') for v in observed] == ['initial', 'final'],
                 'initial/final resource observations required')
            row['resource_receipts'] = {'initial': observed[0], 'final': observed[1]}
            base.checked_physical_resources(row['resource_receipts'], prepared, arm, result,
                                             expected_spec_sha256=value['spec_sha256'])
            row['complete'] = True
        base.verify_unchanged(prepared)
        report['status'] = 'NATIVE_CACHE_REGRESSION_COMPLETE'
    except BaseException as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)[:1024]
    finally:
        if acquired:
            try:
                lease.close()
            except BaseException as error:
                report['cleanup_errors'].append('close: ' + str(error)[:512])
            if results:
                try:
                    snapshot, deadline = observe('independent', result=results[-1])
                    report['independent_postcheck'] = snapshot
                    evidence = adapter.independent_evidence(snapshot, lease=lease, results=results,
                        collector_sha256=observer_pin, deadline=deadline, clock=clock)
                    lease.confirm_closed(evidence)
                    report['independent_release_verified'] = True
                except BaseException as error:
                    report['cleanup_errors'].append('independent: ' + str(error)[:512])
        report['unresolved'] = lease.journal.exists() or lease.journal.is_symlink()
        if report['cleanup_errors'] or report['unresolved'] or not report['independent_release_verified']:
            report['status'] = 'INCOMPLETE'
        report['timings_usable'] = report['status'] == 'NATIVE_CACHE_REGRESSION_COMPLETE'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-owner-cache', action='store_true')
    for name in ('manifest', 'manifest-sha256', 'grant-json', 'grant-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--observer-fd', required=True, type=int)
    args = parser.parse_args(argv)
    need(args.execute_owner_cache and (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (1000,) * 4,
         'explicit UID1000 cache owner required')
    grant = entry.decode(entry.read(args.grant_json, 64 << 10, args.grant_sha256))
    manifest = entry.decode(entry.read(args.manifest, 256 << 10, args.manifest_sha256))
    rpc = transport.RpcClient(args.observer_fd, grant, args.manifest_sha256)
    validate_grant(grant, manifest, args.manifest_sha256, rpc.authenticate_parent())
    root = Path(manifest['target_root'])
    need(Path(args.grant_json) == root / 'GRANT.json', 'exact physical grant path')
    report = {'status': 'INCOMPLETE', 'qualification': False, 'model_starts': 0,
              'experiment_mode': manifest['experiment_mode']}
    try:
        prepared = validate_manifest(args.manifest, args.manifest_sha256, grant['received_monotonic'] + 105)
        report = run_campaign(prepared, grant, rpc)
    except BaseException as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)[:1024]
    finally:
        rpc.close()
    base._publish(root / 'CACHE-RESULT.json', report)
    base.seal_result(root, deadline=grant['received_monotonic'] + manifest['proposed_complete_seconds'],
        received_monotonic=grant['received_monotonic'], received_utc=grant['received_utc'],
        complete_seconds=transport.package_ceiling(manifest))
    return 0 if report['status'] == 'NATIVE_CACHE_REGRESSION_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
