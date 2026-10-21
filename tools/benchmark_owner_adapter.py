"""Compose the matched benchmark with host-process ownership, never Pod disposal.

This is a locally tested adapter, not an execution grant or platform observer.
An authenticated physical controller must supply current platform observations.
There is deliberately no remote or GPU-launching CLI in this module.
"""
from dataclasses import asdict, dataclass
import importlib.util
import math
import os
from pathlib import Path
import time


def _adjacent(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


owner = _adjacent('spry_host_process_owner', 'host_process_owner.py')
readout = _adjacent('spry_benchmark_readout', 'analyze_model_benchmark.py')
CANONICAL_LOCK = '/home/developer/model-serving.izR2m1O9/diagnostic.lock'
DEVICES = {
    '0000:c3:00.0': '39373864-6465-3038-6163-303837666533',
    '0000:23:00.0': '34663466-6561-3362-3832-346261366439',
}
ARM_WORK = 690
ARM_ENVELOPE = 750
COLLECTION_RESERVE = 180


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _need(value, message):
    if not value:
        raise owner.OwnerError(message)


@dataclass(frozen=True)
class FrozenArm:
    arm: str
    argv: tuple
    script_path: str
    script_sha256: str


def bwrap_argv(arm, *, retained_root, composition, review, model, output, entry,
               spec_sha256, bounded_scratch=False, physical_artifacts=False,
               session_sha256=None, artifacts_sha256=None):
    """Pure projection of the reviewed bwrap isolation; creates no paths/processes.

    The physical controller must authenticate every mount and complete source,
    binary/build and 81-file model inventories before calling the owner adapter.
    Network sharing permits native TP rendezvous; this is NOT Pod deny-policy
    isolation. Private output/cache roots must be different for the two arms.
    """
    _need(arm in ('bf16', 'kv4') and _digest(spec_sha256), 'arm/spec binding')
    _need(type(bounded_scratch) is bool, 'explicit scratch projection required')
    _need(type(physical_artifacts) is bool and (not physical_artifacts or
        bounded_scratch and _digest(session_sha256) and _digest(artifacts_sha256)),
        'physical artifact source/session binding required')
    _need(physical_artifacts or session_sha256 is artifacts_sha256 is None,
        'artifact bindings require physical mode')
    paths = [Path(value) for value in (retained_root, composition, review, model, output, entry)]
    # This is a remote POSIX projection, not resolution through the generating
    # workstation's filesystem (macOS aliases /home). The physical controller
    # independently resolves and authenticates the real target paths.
    _need(all(path.is_absolute() and os.path.normpath(str(path)) == str(path)
        and '..' not in path.parts for path in paths), 'canonical mount paths')
    retained_root, composition, review, model, output, entry = paths
    _need(entry == review / 'model_benchmark.py', 'exact benchmark entry mount')
    argv = ['/usr/bin/bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--share-net',
        '--uid', '1000', '--gid', '1000', '--ro-bind', str(retained_root / 'base-rootfs'), '/',
        '--tmpfs', '/mnt', '--ro-bind', str(composition), '/mnt/composition',
        '--ro-bind', str(review), '/mnt/review',
        '--ro-bind', str(entry), '/mnt/review/model_benchmark.py',
        '--ro-bind', str(retained_root / 'build-output3/build-venv'), '/mnt/work/build-venv',
        '--ro-bind', str(model), '/mnt/models', '--bind', str(output), '/mnt/output',
        '--ro-bind', str(output / 'hosts'), '/etc/hosts', '--proc', '/proc', '--dev', '/dev',
        '--tmpfs', '/dev/shm', '--dev-bind', '/dev/kfd', '/dev/kfd',
        '--dev-bind', '/dev/dri/renderD128', '/dev/dri/renderD128',
        '--dev-bind', '/dev/dri/renderD129', '/dev/dri/renderD129',
        '--ro-bind', '/sys', '/sys', '--tmpfs', '/tmp', '--clearenv']
    env = {
        'PATH': '/mnt/work/build-venv/bin:/usr/local/bin:/usr/bin:/bin',
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'HOME': '/mnt/output/home', 'TMPDIR': '/mnt/output/tmp',
        'XDG_CACHE_HOME': '/mnt/output/cache', 'TRITON_CACHE_DIR': '/mnt/output/cache/triton',
        'TORCH_EXTENSIONS_DIR': '/mnt/output/cache/torch',
        'ROCM_HOME': '/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel',
        'ROCM_PATH': '/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel',
        'HIP_CLANG_PATH': '/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel/lib/llvm/bin',
        'HIP_PLATFORM': 'amd', 'HIP_COMPILER': 'clang', 'MAX_JOBS': '2', 'HIPCC_JOBS': '2',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'NCCL_SOCKET_IFNAME': 'lo',
        'GLOO_SOCKET_IFNAME': 'lo', 'NCCL_IB_DISABLE': '1', 'NCCL_DEBUG': 'WARN',
        'SGLANG_USE_AITER': 'false', 'SGLANG_USE_AITER_MHA': 'false', 'SGLANG_USE_AITER_MOE': 'false',
    }
    for key, value in env.items():
        argv.extend(('--setenv', key, value))
    argv.extend(('/mnt/work/build-venv/bin/python3', '-I', '-B', '/mnt/review/model_benchmark.py',
        '--execute-reviewed-benchmark', '--spec', '/mnt/review/BENCHMARK-SPEC2.json',
        '--spec-sha256', spec_sha256, '--output', '/mnt/output/benchmark', '--arm', arm))
    if bounded_scratch:
        # Keep one immutable projection per mode. Native mount selection remains
        # unrun; these tmpfs sizes are shared by all users within each mount.
        # 2.75 GiB cache/home/temp + .5 GiB /tmp + .25 GiB /dev/shm = 3.5 GiB.
        # The future owner must separately bound the durable .5 GiB allowance;
        # per-file RLIMIT_FSIZE alone is not an aggregate output quota.
        bind = argv.index('--bind')
        _need(argv[bind:bind + 3] == ['--bind', str(output), '/mnt/output'], 'output bind differs')
        argv[bind:bind + 3] = ['--size', str(2816 * 1024**2), '--tmpfs', '/mnt/output',
            '--dir', '/mnt/output/home', '--dir', '/mnt/output/tmp', '--dir', '/mnt/output/cache']
        if not physical_artifacts:
            # Preserve the inactive, locally reviewed candidate projection.
            # Only the physical export mode removes every host-writable bind.
            argv[bind + 10:bind + 10] = ['--bind', str(output), '/mnt/results']
        for target, size in (('/tmp', 512 * 1024**2), ('/dev/shm', 256 * 1024**2)):
            location = next(i for i in range(len(argv) - 1)
                if argv[i:i + 2] == ['--tmpfs', target])
            argv[location:location] = ['--size', str(size)]
        if not physical_artifacts:
            argv[argv.index('--output') + 1] = '/mnt/results/benchmark'
    if physical_artifacts:
        argv.extend(('--physical-artifacts', '--session-sha256', session_sha256,
            '--artifacts-sha256', artifacts_sha256))
    return tuple(argv)


def release_evidence(snapshot, *, lease, arm, result, collector_sha256, deadline,
                     not_before, clock=time.monotonic):
    """Validate bound raw observations; never promote a bare cleanup Boolean.

    collector_sha256 must come from the physical controller's authenticated
    observer source, not from the snapshot being validated. That observer must
    actually collect all-user holders, allocations and target/policy state.
    No such non-escalating observer has yet been selected in retained evidence.
    """
    _need(_digest(collector_sha256) and type(snapshot) is dict, 'authenticated platform observer required')
    child = result.root_process
    _need(child is not None and result.process_cleanup_observed, 'owned process cleanup is incomplete')
    expected = {'schema': 1, 'scope': 'host-benchmark-release-v1',
        'root': str(lease.root), 'session_sha256': lease.session, 'boot_id': lease.boot,
        'arm': arm, 'child_pid': child.pid, 'child_start_ticks': child.start_ticks,
        'collector_source_sha256': collector_sha256,
        'lock_device': lease.record['lock_device'], 'lock_inode': lease.record['lock_inode']}
    _need(all(type(snapshot.get(key)) is type(value) and snapshot.get(key) == value
              for key, value in expected.items()), 'release snapshot identity mismatch')
    now, observed = clock(), snapshot.get('observed_monotonic')
    intent = lease.record['arms'][-1]['intent_monotonic']
    _need(type(observed) in (int, float) and math.isfinite(observed)
          and intent <= not_before <= observed <= now <= deadline,
          'release observation deadline/freshness')
    _platform_facts(snapshot, [result])
    return owner.ReleaseEvidence(str(lease.root), lease.session, lease.boot, arm,
        child.pid, child.start_ticks, lease.record['lock_device'], lease.record['lock_inode'],
        observed, True, True, True, True, True)


def _platform_facts(snapshot, results):
    """Require checks for every recorded process and actual private namespace."""
    processes = {(value.boot_id, value.pid, value.start_ticks)
                 for result in results for value in result.observed_processes}
    namespaces = {value.pid_namespace for result in results for value in result.observed_processes
                  if value.pid_namespace != result.root_process.pid_namespace}
    _need(bool(processes) and bool(namespaces), 'private workload namespace was not observed')
    _need(snapshot.get('checked_processes') == [list(value) for value in sorted(processes)]
          and snapshot.get('authenticated_processes_present') == [], 'exact process checks missing')
    _need(snapshot.get('pid_namespace_members') == {name: [] for name in sorted(namespaces)},
          'exact private namespace absence checks missing')
    _availability_facts(snapshot)


def _availability_facts(snapshot):
    """Common live target/device checks, without inventing process absence."""
    holders = snapshot.get('holders', {})
    _need(type(holders) is dict and holders.get('visibility') == 'all-users-platform'
          and type(holders.get('exit_code')) is int and holders['exit_code'] == 1
          and holders.get('stdout') == '' and holders.get('stderr') == '', 'holder release is uncertain')
    allocation = snapshot.get('allocation', {})
    _need(type(allocation) is dict and allocation.get('scope') == 'all-namespaces-active-workloads'
          and type(allocation.get('gpu_requests')) is int and allocation['gpu_requests'] == 0,
          'GPU allocation release is uncertain')
    target = snapshot.get('target', {})
    _need(type(target) is dict and target.get('hostname') == 'threadripper-ai'
          and type(target.get('uid')) is int and target['uid'] == 1000
          and type(target.get('gid')) is int and target['gid'] == 1000
          and target.get('boot_id') == snapshot['boot_id'] and target.get('swap_active') is False
          and target.get('node_ready') is True and target.get('pressure_free') is True
          and target.get('expected_policy_unchanged') is True, 'target/policy postcheck failed')
    devices = snapshot.get('devices')
    _need(type(devices) is list and len(devices) == 2, 'exact two-device inventory required')
    _need(all(type(row) is dict and set(row) == {'bdf', 'uuid', 'render', 'sysfs_bdf'} for row in devices),
          'device record shape')
    _need({row['bdf']: row['uuid'] for row in devices} == DEVICES
          and {row['render'] for row in devices} == {'/dev/dri/renderD128', '/dev/dri/renderD129'}
          and all(row['sysfs_bdf'] == row['bdf'] for row in devices), 'device/sysfs binding differs')


def locked_admission_evidence(snapshot, *, lease, collector_sha256, not_before, deadline,
                              clock=time.monotonic):
    _need(type(snapshot) is dict and _digest(collector_sha256), 'bound locked admission required')
    expected = {'schema': 1, 'scope': 'host-benchmark-locked-admission-v1',
        'root': str(lease.root), 'session_sha256': lease.session, 'boot_id': lease.boot,
        'collector_source_sha256': collector_sha256,
        'lock_device': lease.record['lock_device'], 'lock_inode': lease.record['lock_inode']}
    _need(all(type(snapshot.get(key)) is type(value) and snapshot.get(key) == value
              for key, value in expected.items()), 'locked admission identity differs')
    observed = snapshot.get('observed_monotonic')
    _need(type(observed) in (int, float) and math.isfinite(observed)
          and not_before <= observed <= clock() <= deadline, 'locked admission is stale or exhausted')
    _need(snapshot.get('status') == 'OBSERVATION_COMPLETE' and snapshot.get('failures') == []
          and snapshot.get('checked_processes') == []
          and snapshot.get('authenticated_processes_present') == []
          and snapshot.get('pid_namespace_members') == {}, 'locked admission is incomplete')
    lease._verify_journal()
    journal = snapshot.get('journal', {})
    _need(lease.fd == 9 and lease.record['state'] == 'reserved' and lease.record['arms'] == []
          and journal.get('authenticated') is True and journal.get('state') == 'reserved'
          and journal.get('sha256') == lease.journal_identity[2], 'owned reserved journal differs')
    _availability_facts(snapshot)


def independent_evidence(snapshot, *, lease, results, collector_sha256, deadline, clock=time.monotonic):
    _need(type(snapshot) is dict and bool(results), 'independent observed release required')
    expected = {'schema': 1, 'scope': 'host-benchmark-independent-release-v1',
        'root': str(lease.root), 'session_sha256': lease.session, 'boot_id': lease.boot,
        'collector_source_sha256': collector_sha256,
        'lock_device': lease.record['lock_device'], 'lock_inode': lease.record['lock_inode']}
    _need(all(type(snapshot.get(key)) is type(value) and snapshot.get(key) == value
              for key, value in expected.items()), 'independent release binding differs')
    observed = snapshot.get('observed_monotonic')
    _need(type(observed) in (int, float) and math.isfinite(observed)
          and lease.closed_monotonic <= observed <= clock() <= deadline,
          'independent release timestamp/deadline')
    _platform_facts(snapshot, results)
    # The core, not a snapshot Boolean, performs the independent canonical-lock
    # acquisition and journal authentication/removal before releasing it once.
    return owner.IndependentReleaseEvidence(str(lease.root), lease.session, lease.boot,
        lease.record['lock_device'], lease.record['lock_inode'], observed,
        True, True, True, True, True, True)


def run_campaign(*, lease, arms, spec_path, spec_sha256, collector_sha256, active_deadline,
                 collect_release, read_inner_result, independent_postcheck,
                 streams, resource_check, supervisor=owner.supervise, clock=time.monotonic,
                 arm_envelope_seconds=ARM_ENVELOPE, collection_reserve_seconds=COLLECTION_RESERVE,
                 admission_check=None, prepare_streams=None):
    """Single matched screen with explicit observed effects supplied by its owner.

    No observer defaults to success. CPU substitutions must be labelled by the
    caller. The future physical controller still owns current admission, source,
    model/build/resource validation, grant accounting and bounded collection.
    """
    spec, _ = readout.read_json(spec_path, spec_sha256)
    readout.bench.validate_spec(spec)
    _need(_digest(spec_sha256) and _digest(collector_sha256), 'frozen source/spec bindings required')
    _need(type(arms) is tuple and len(arms) == 2
          and all(isinstance(arm, FrozenArm) for arm in arms)
          and [arm.arm for arm in arms] == ['bf16', 'kv4'], 'exact two-arm order required')
    _need(spec['worker_seconds'] == ARM_WORK, 'fixed 690-second inner bound')
    # Both are preparation contracts, never grants. Keep the original default;
    # the explicit observer projection reserves 75 seconds after owned reaping.
    _need(type(arm_envelope_seconds) is int and type(collection_reserve_seconds) is int
          and (arm_envelope_seconds, collection_reserve_seconds) in ((750, 180), (795, 330)),
          'unreviewed owner envelope')
    locked_admission_seconds = 75 if arm_envelope_seconds == 795 else 0
    _need(not locked_admission_seconds or callable(admission_check), 'live locked admission callback required')
    _need(type(active_deadline) in (int, float) and math.isfinite(active_deadline)
          and clock() + locked_admission_seconds + 2 * arm_envelope_seconds
          + collection_reserve_seconds <= active_deadline,
          'complete remaining owner envelope does not fit')
    report = {'schema': 1, 'scope': 'matched-host-owner-adapter-v1', 'arms': [],
        'qualification': False, 'serving_quality_qualified': False,
        'independent_release_verified': False, 'physical_campaign_complete': False,
        'adapter_complete': False, 'arm_envelope_seconds': arm_envelope_seconds,
        'collection_reserve_seconds': collection_reserve_seconds,
        'failure': None, 'cleanup_errors': [], 'unresolved': True}
    acquired, results = False, []
    try:
        lease.acquire()
        acquired = True
        if prepare_streams is not None:
            # The canonical owner must obtain FD9 before stream pipes can use
            # that descriptor. The callback only supplies bounded local sinks.
            _need(callable(prepare_streams), 'bounded stream preparation callback required')
            prepare_streams()
        if locked_admission_seconds:
            beginning = clock()
            deadline = beginning + locked_admission_seconds
            raw = admission_check(lease, deadline_monotonic=deadline)
            report['locked_admission'] = raw
            locked_admission_evidence(raw, lease=lease, collector_sha256=collector_sha256,
                not_before=beginning, deadline=deadline, clock=clock)
        for frozen in arms:
            start = clock()
            remaining_arms = 2 - len(report['arms'])
            _need(start + remaining_arms * arm_envelope_seconds + collection_reserve_seconds <= active_deadline,
                  'remaining complete arm envelopes do not fit')
            lease.begin_arm(frozen.arm, frozen.argv, frozen.script_path, frozen.script_sha256)
            row = {'arm': frozen.arm, 'release_verified': False, 'inner_complete': False}
            report['arms'].append(row)
            result = supervisor(frozen.argv, script_path=frozen.script_path,
                script_sha256=frozen.script_sha256,
                deadlines=owner.Deadlines(start + ARM_WORK, start + ARM_WORK + 20,
                    start + ARM_WORK + 25, start + ARM_WORK + 30),
                on_child=lease.record_child, stdout=streams[frozen.arm][0], stderr=streams[frozen.arm][1],
                resource_check=resource_check)
            row['supervision'] = asdict(result)
            results.append(result)
            cleanup_returned = clock()
            # External evidence must be collected even on a failed model arm.
            raw = collect_release(lease, frozen.arm, result, start + arm_envelope_seconds)
            row['release_snapshot'] = raw
            observed = release_evidence(raw, lease=lease, arm=frozen.arm, result=result,
                collector_sha256=collector_sha256, deadline=start + arm_envelope_seconds,
                not_before=cleanup_returned, clock=clock)
            lease.finish_arm(result, observed)
            row['release_verified'] = True
            _need(result.exit_code == 0 and result.failure is None and result.owner_signal is None,
                  'arm failed; no replacement or second arm')
            inner = read_inner_result(frozen.arm)
            readout.checked_arm(inner, frozen.arm, spec_sha256,
                expected_profile=spec.get('diagnostic_profile') is not None,
                session_sha256=lease.session)
            row['inner_complete'] = True
        report['status'] = 'HOST_OWNER_ADAPTER_COMPLETE'
    except BaseException as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)[:1024]
        report['status'] = 'HOST_OWNER_ADAPTER_INCOMPLETE'
    finally:
        if acquired:
            try:
                lease.close()
            except BaseException as error:
                report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error)[:512])
        report['unresolved'] = lease.journal.exists() or lease.journal.is_symlink()
        if acquired:
            try:
                # This callback must observe release after FD9 is closed and must
                # immediately release any lock acquired solely for verification.
                final = independent_postcheck(lease, active_deadline)
                report['independent_postcheck'] = final
                evidence = independent_evidence(final, lease=lease, results=results,
                    collector_sha256=collector_sha256, deadline=active_deadline, clock=clock)
                lease.confirm_closed(evidence)
                report['independent_release_verified'] = True
            except BaseException as error:
                report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error)[:512])
        report['unresolved'] = lease.journal.exists() or lease.journal.is_symlink()
        if (report['failure'] is not None or report['cleanup_errors'] or report['unresolved']
                or not report['independent_release_verified']):
            report['status'] = 'HOST_OWNER_ADAPTER_INCOMPLETE'
        report['adapter_complete'] = report['status'] == 'HOST_OWNER_ADAPTER_COMPLETE'
    return report


if __name__ == '__main__':
    raise SystemExit('NOT_RUN: local adapter only; authenticated physical controller and observer required')
