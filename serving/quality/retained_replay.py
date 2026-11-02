"""Offline successful two-arm replay; no lease, observer, or launch operations."""
from contextlib import contextmanager
from datetime import datetime
from dataclasses import asdict
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tarfile

SEALER_SHA256 = '082835dabf0bf1e96d9bba6f08918914b01a1306a5a2cd939a0c9309ad334e18'
PIN_FIELDS = {'manifest_sha256','archive_sha256','package_sha256','source_manifest_sha256','session_sha256'}


def need(ok, reason):
    if not ok:
        raise ValueError('retained serving replay: ' + reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def held(path, pin=None, maximum=128 << 20):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'canonical retained path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= maximum,
             'bounded unlinked regular evidence')
        pieces, remaining = [], before.st_size
        while remaining:
            piece = os.read(fd, min(remaining, 1 << 20))
            need(piece, 'short retained read')
            pieces.append(piece)
            remaining -= len(piece)
        raw = b''.join(pieces)
        after, named = os.fstat(fd), path.lstat()
        fields = ('st_dev','st_ino','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        need(all(getattr(before,k) == getattr(after,k) == getattr(named,k) for k in fields), 'retained input changed')
        need(pin is None or sha(raw) == pin, 'retained source/hash mismatch: ' + path.name)
        return raw
    finally:
        os.close(fd)


def read(path):
    return strict(held(path, maximum=8 << 20))


def source_module(path, name):
    # Execute authenticated source bytes without a bytecode writer.
    raw=held(path, maximum=4 << 20)
    class Loader(SourceFileLoader):
        def get_code(self,fullname):return self.source_to_code(raw,str(path))
    spec=importlib.util.spec_from_loader(name,Loader(name,str(path)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def modules(review):
    previous = dict(sys.modules)
    old_path = list(sys.path)
    sys.path.insert(0, str(review))
    try:
        yield source_module(review / 'benchmark_physical_controller.py', '_retained_controller')
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name not in previous:
                del sys.modules[name]
        sys.modules.update(previous)


def supervision(value, owner):
    need(type(value) is dict, 'actual supervision object')
    result = owner.SupervisionResult(root_process=owner.ProcessIdentity(**value['root_process']),
        observed_processes=tuple(owner.ProcessIdentity(**x) for x in value['observed_processes']),
        **{k:tuple(v) if k in {'cleanup_errors','survivors','escalation_receipts'} else v
           for k,v in value.items() if k not in {'root_process','observed_processes'}})
    need(type(result.exit_code) is int and result.exit_code == 0 and result.child_reaped is True and
         result.failure is None and not result.cleanup_errors and not result.survivors and
         result.owner_signal is None and not result.escalation_receipts and result.process_cleanup_observed,
         'successful reaped actual supervised process required')
    return result


def capture(value, local, remote):
    need(value['capture_complete'] is value['reader_thread_absent'] is True and value['failure'] is None and
         set(value['streams']) == {'stdout','stderr'}, 'complete actual pipe capture')
    result = {}
    for channel, row in value['streams'].items():
        need(row['path'] == str(remote(channel)), 'exact capture path mapping')
        raw = held(local(channel), row['sha256'])
        need(type(row['bytes']) is int and len(raw) == row['bytes'], 'actual pipe stream extent')
        result[channel] = raw
    return result


def local_receipts(operation):
    result = {}
    for name in ('upload-once','native-launch','observe-root-closeout','collect-native-results'):
        value = read(operation / (name + '.json'))
        need(type(value['exit_code']) is int and value['exit_code'] == 0 and
             value['child_reaped'] is value['owned_group_absent'] is True and
             value.get('failure') is None and value.get('signals', []) == [], 'successful finite local transport')
        for channel in ('stdout','stderr'):
            row = value[channel]
            raw = held(operation / (name + '.' + channel), row['sha256'])
            need(len(raw) == row['bytes'], 'local transport captured extent')
        result[name] = dict(sha256=sha(held(operation / (name + '.json'))),
            elapsed_seconds=value['elapsed_seconds'])
    return result


def release_phase(controller, campaign, results, request, evidence, protected, bindings, manifest):
    mode, arm = request['mode'], request['arm']
    phase = 'independent' if mode == 'independent' else 'locked-admission' if mode == 'locked-admission' else 'release-' + arm
    value = read(evidence / (phase + '.json'))
    actual_owner = read(evidence / (phase + '-owner.json'))
    pin = sha(canonical(request))
    need(actual_owner['request_sha256'] == value['request_file_sha256'] == value['request_canonical_sha256'] == pin,
         'canonical observer request/raw retained identity')
    need(all(value[k] == request[k] for k in ('root','session_sha256','boot_id','lock_device','lock_inode')) and
         request['root'] == manifest['target_root'] and request['session_sha256'] == manifest['session_sha256'] and
         request['boot_id'] == manifest['boot_id'], 'observer target/session/lock identity')
    need(all(type(request[k]) in (int,float) and math.isfinite(request[k]) for k in ('issued_monotonic','deadline_monotonic')) and
         request['issued_monotonic'] <= value['observed_monotonic'] <= request['deadline_monotonic'], 'observer deadline')
    journal = value['journal']
    state = 'reserved' if mode == 'locked-admission' else 'pending-independent' if mode == 'independent' else 'running'
    need(journal['authenticated'] is True and journal['sha256'] == request['journal_sha256'] and
         journal['state'] == state and value['collector_source_sha256'] == sha(held(protected / 'review/benchmark_release_observer.py')) and
         value['metadata_source_sha256'] == controller.METADATA_SHA256 and
         value['status'] == 'OBSERVATION_COMPLETE' and value['failures'] == [], 'authenticated observer source/journal')
    checked = sorted([p['boot_id'],p['pid'],p['start_ticks']] for p in request['observed_processes'])
    need(value['checked_processes'] == checked, 'exact observer process request')
    observed_supervision = supervision(actual_owner['supervision'], controller.adapter.owner)
    need(observed_supervision.root_process.uid == 0 and
         re.fullmatch(r'/var/tmp/fp8-benchmark-observer\.[0-9a-f]{16}', actual_owner['evidence_path']),
         'root observer exact private capture directory')
    remote = Path(actual_owner['evidence_path']) / 'capture'
    capture(actual_owner['capture'], lambda channel:evidence/(phase+'-'+channel+'.raw'),
            lambda channel:remote/(channel+'.raw'))
    if mode == 'locked-admission':
        controller.adapter._availability_facts(value)
        need(value == campaign['locked_admission'], 'actual locked admission campaign join')
        return dict(phase=phase, request_sha256=pin, snapshot_sha256=sha(held(evidence/(phase+'.json'))))
    selected = list(results.values()) if arm is None else [results[arm]]
    controller.adapter._platform_facts(value, selected)
    cgroups = read(evidence / (phase + '-cgroup.json'))
    snapshot = campaign['independent_postcheck'] if arm is None else next(a['release_snapshot'] for a in campaign['arms'] if a['arm'] == arm)
    need(all(snapshot[k] == item for k,item in value.items()) and set(snapshot)-set(value) == {'root_resource_release'} and
         snapshot['root_resource_release'] == cgroups, 'original platform/owner/root-resource release joins')
    expected_arms = list(results) if arm is None else [arm]
    need([row['arm'] for row in cgroups['scopes']] == expected_arms, 'exact released scope inventory')
    for row in cgroups['scopes']:
        expected = controller.resources.USER_MANAGER + '/app.slice/spry-fp8-benchmark-' + manifest['session_sha256'][:16] + '-' + row['arm'] + '.scope'
        need(row['absent'] is True and row['cgroup_path'] == '/sys/fs/cgroup' + expected and
             value['observed_monotonic'] <= row['observed_monotonic'] <= request['deadline_monotonic'], 'actual scoped absence/deadline')
    return dict(phase=phase, request_sha256=pin, snapshot_sha256=sha(held(evidence/(phase+'.json'))),
                processes_checked=len(checked), original_platform_validator_replayed=True)


def replay_retained(operation_root, *, operation_seal_sha256, workload_seal_sha256,
                    protected_seal_sha256, package_root, expected_package_pins, destination,
                    sealer_path):
    """Return authenticated arm results only for a complete successful operation.

    All five expected package pins must come from the newly authored operation.
    Old successful fixtures cannot satisfy another manifest's admission.
    Failure evidence is retained upstream; it is never a quality-input success.
    """
    operation, package_root, destination = map(lambda p:Path(p).resolve(), (operation_root,package_root,destination))
    need(sys.flags.optimize == 0, 'original assert-based seal validator requires unoptimized Python')
    need(type(expected_package_pins) is dict and set(expected_package_pins) == PIN_FIELDS and
         all(type(v) is str and len(v) == 64 for v in expected_package_pins.values()), 'explicit authored operation pins')
    need(not destination.exists() and destination.parent.is_dir() and
         not destination.is_relative_to(operation) and not destination.is_relative_to(package_root), 'fresh extraction outside all sealed evidence')
    held(Path(sealer_path), SEALER_SHA256, 64 << 10)
    sealer = source_module(Path(sealer_path), '_retained_exact_sealer')
    operation_seal = sealer.verify(operation, operation_seal_sha256)
    bindings, package = read(operation/'LAUNCHER-BINDINGS.json'), read(operation/'PACKAGE-IDENTITIES.json')
    workload = operation/'collected/workload'/Path(bindings['target_root']).name
    protected = operation/'collected/protected'/Path(bindings['protected_root']).name
    seals = dict(operation=operation_seal, workload=sealer.verify(workload,workload_seal_sha256),
                 protected=sealer.verify(protected,protected_seal_sha256))
    manifest_raw = held(workload/'MANIFEST.json', expected_package_pins['manifest_sha256'], 4 << 20)
    manifest = strict(manifest_raw)
    need(held(package_root/'MANIFEST.json') == manifest_raw, 'authored/current retained manifest equality')
    computed = dict(manifest_sha256=sha(manifest_raw), archive_sha256=sha(held(protected/'payload.tar')),
        package_sha256=sha(canonical(manifest['package_files'])),
        source_manifest_sha256=sha(canonical({n:p for n,p in manifest['package_files'].items() if n.endswith('.py')})),
        session_sha256=manifest['session_sha256'])
    need(computed == expected_package_pins and all(package.get(k) == v for k,v in computed.items()), 'exact authored package/source/session identity')
    need(bindings['archive_sha256'] == computed['archive_sha256'] and bindings['manifest_sha256'] == computed['manifest_sha256'] and
         bindings['target_root'] == manifest['target_root'], 'launcher package/target mapping')
    for name, pin in manifest['package_files'].items():
        need(held(workload/name,pin,16<<20) == held(package_root/name,pin,16<<20), 'complete actual/authored source closure')
    registration, authorization = read(operation/'REGISTRATION.json'), read(operation/'RUN-AUTHORIZATION.json')
    need(authorization.pop('registration_sha256') == sha(held(operation/'REGISTRATION.json')) and authorization == registration,
         'exact registered single operation')
    source_root = Path(registration['source_root'])
    need(package_root == source_root/'package-v2', 'exact authored source root')
    seals['source_author'] = sealer.verify(source_root, registration['source_seal']['manifest_sha256'])
    need(seals['source_author'] == registration['source_seal'] and registration['manifest_sha256'] == computed['manifest_sha256'] and
         registration['archive_sha256'] == computed['archive_sha256'] and registration['owner_invocations_allowed'] == 1 and
         registration['model_starts_allowed'] == 2 and registration['no_automatic_retry'] is True and
         registration['promotion_authorized'] is False, 'sealed authored two-start/no-retry registration')
    for name,pin in bindings['upload_inventory'].items():
        authored = held(source_root/'owner-upload'/name,pin,16<<20)
        if name != 'RUN-ON-WORKSTATION.sh':
            need(held(protected/name,pin,16<<20) == authored, 'protected uploaded source copy')
        if name.startswith('review/'):
            need(manifest['package_files'][name] == pin, 'protected/worker same helper bytes')
    upload_raw = held(operation/'upload.tar', maximum=32<<20)
    need(read(operation/'upload-once.json')['stdin_sha256'] == sha(upload_raw), 'actual uploaded stdin archive identity')
    with tarfile.open(fileobj=io.BytesIO(upload_raw),mode='r:') as archive:
        members=archive.getmembers()
        need(len(members) == len(bindings['upload_inventory']) and
             {m.name for m in members} == set(bindings['upload_inventory']) and
             all(m.isfile() and not m.issym() and not m.islnk() and 0 <= m.size <= 16<<20 for m in members),
             'exact ordinary upload archive membership/type/extent')
        for member in members:
            raw=archive.extractfile(member).read(member.size+1)
            need(len(raw) == member.size and sha(raw) == bindings['upload_inventory'][member.name], 'actual uploaded member source identity')
    receipts = local_receipts(operation)
    observed = read(operation/'observe-root-closeout.stdout')
    closeout_path = protected.with_name(protected.name+'.closeout.json')
    closeout_raw = held(closeout_path, observed['sha256'], 1<<20)
    closeout = strict(closeout_raw)
    need(observed['value'] == closeout and observed['uid'] == observed['gid'] == 0 and observed['mode'] == 0o444 and
         observed['path'] == bindings['protected_root']+'.closeout.json' and closeout['physical_success'] is True and
         closeout['root'] == bindings['protected_root'] and closeout['workload_evidence_root'] == manifest['target_root'] and
         closeout['manifest_sha256'] == protected_seal_sha256 and closeout['member_count'] == seals['protected']['members'],
         'observed root-owned successful protected closeout')
    evidence = protected/'evidence'
    grant, receipt = read(evidence/'GRANT.json'), read(evidence/'RECEIPT.json')
    transport_helper=source_module(workload/'review/benchmark_host_transport.py','_retained_transport')
    configuration_pins={key:registration[key] for key in ('run_config_sha256','host_sha256','protected_set_sha256')}
    configured='run_config' in manifest
    config_root=workload if configured else source_root
    if configured:need(all(grant.get(key)==pin for key,pin in configuration_pins.items()),'registered configuration/grant binding')
    run_raw=held(config_root/manifest.get('run_config','config/runs/w5-v2/run.json'),configuration_pins['run_config_sha256'],256<<10)
    run=strict(run_raw)
    held(config_root/run['host']['file'],configuration_pins['host_sha256'],64<<10)
    held(config_root/'PROTECTED-SHA256.json',configuration_pins['protected_set_sha256'],64<<10)
    if configured:
        timing=transport_helper.SessionTiming.from_limits(run['limits'])
        active_seconds,total_seconds=timing.active_seconds,timing.total_seconds
        manifest=transport_helper.bind_manifest_boot(manifest,grant['boot_id'])
    else:
        active_seconds,total_seconds=transport_helper.TOTAL_SECONDS-transport_helper.SEAL_RESERVE,transport_helper.TOTAL_SECONDS
        need(manifest['boot_id']==grant['boot_id']==registration['boot_id'],'legacy control boot differs')
    need(registration['owner_seconds']==total_seconds==bindings['total_seconds'],'approved owner duration differs')
    need(receipt['grant_sha256'] == sha(held(evidence/'GRANT.json')) and receipt['archive_sha256'] == computed['archive_sha256'] and
         receipt['active_deadline_monotonic'] == grant['received_monotonic']+active_seconds and
         receipt['seal_deadline_monotonic'] == grant['received_monotonic']+total_seconds and receipt['root_performs_gpu_work'] is False,
         'actual root grant receipt/deadlines')
    transport, report = read(evidence/'ROOT-TRANSPORT.json'), read(workload/'results/CONTROLLER-RESULT.json')
    need(transport['physical_success'] is transport['all_observation_phases_complete'] is True and
         type(transport['controller_exit']) is int and transport['controller_exit'] == 0 and
         all(transport[k] == [] for k in ('errors','survivors','owner_signals','escalation_receipts')),
         'successful actual root controller transport')
    remote_evidence = Path(bindings['protected_root'])/'evidence'
    capture(transport['controller_capture'], lambda ch:evidence/'controller-capture'/(ch+'.raw'),
            lambda ch:remote_evidence/'controller-capture'/(ch+'.raw'))
    identity = read(evidence/'CONTROLLER-IDENTITY.json')
    observed_controllers = [row for row in transport['observed_processes']
        if all(row[k] == identity[k] for k in ('boot_id','pid','start_ticks'))]
    need(len(observed_controllers) == 1 and identity['uid'] == identity['gid'] == 1000 and
         identity['ppid'] == grant['transport_parent']['pid'] and identity['boot_id'] == manifest['boot_id'] and
         all(observed_controllers[0][k] == identity[k] for k in ('uid','gid','ppid','pgid','sid','pid_namespace')),
         'actual root-supervised controller identity/parent')
    need(report['status'] == 'MATCHED_PHYSICAL_BENCHMARK_COMPLETE' and report['failure'] is None and
         report['remaining_gaps'] == [], 'complete actual controller result')
    campaign = report['campaign']
    need(campaign['adapter_complete'] is campaign['independent_release_verified'] is True and campaign['unresolved'] is False and
         campaign['failure'] is None and campaign['cleanup_errors'] == [] and campaign['qualification'] is False,
         'complete actual owner campaign and independent release; no qualification')
    destination.mkdir(mode=0o700)
    with modules(workload/'review') as controller:
        archive_manifest, archive_members = transport_helper.inspect_archive(protected/'payload.tar', computed['archive_sha256'], computed['manifest_sha256'],
            **({'configuration_pins':configuration_pins} if configured else {}))
        need(archive_members['MANIFEST.json']==manifest_raw and
             (transport_helper.bind_manifest_boot(archive_manifest,grant['boot_id']) if configured else archive_manifest)==manifest and
             all(archive_members[n] == held(workload/n) for n in archive_members), 'original protected archive validator/full source bytes')
        # Replay the original grant's historical schema/time relation; no current
        # admission or parent-process observation is performed here.
        controller.validate_physical_grant(grant,dict(manifest,package_root=str(workload)) if configured else manifest,computed['manifest_sha256'],grant['transport_parent'],
            clock=lambda:grant['received_monotonic'], utc_now=lambda:datetime.fromisoformat(grant['received_utc'].replace('Z','+00:00')))
        arms = campaign['arms']
        need([a['arm'] for a in arms] == ['bf16','kv4'] and all(a['inner_complete'] is a['release_verified'] is True for a in arms), 'exact two complete physical arms')
        results = {a['arm']:supervision(a['supervision'],controller.adapter.owner) for a in arms}
        requests = report['observer_requests']
        need([(q['mode'],q['arm']) for q in requests] == [('locked-admission',None),('release','bf16'),('release','kv4'),('independent',None)], 'exact admission/per-arm/independent observer schedule')
        locks = {(q['lock_device'],q['lock_inode']) for q in requests}
        need(len(locks) == 1 and all(q['deadline_monotonic'] <= receipt['seal_deadline_monotonic'] for q in requests), 'one retained lock and finite owner deadlines')
        phases = [release_phase(controller,campaign,results,q,evidence,protected,bindings,manifest) for q in requests]
        resource_manifest = read(workload/'review/RESOURCE-MANIFEST.json')
        need(sha(held(workload/'review/RESOURCE-MANIFEST.json')) == manifest['resource_manifest_sha256'], 'resource source identity')
        resource_rows = {r['initial']['arm']:r for r in report['resource_receipts']}
        need(len(report['resource_receipts']) == 2 and set(resource_rows) == {'bf16','kv4'} and
             set(report['stream_captures']) == set(report['artifact_receipts']) == set(report['live_namespace_joins']) == {'bf16','kv4'},
             'exact two resource/capture/artifact/namespace inventories')
        spec = read(workload/'review/BENCHMARK-SPEC2.json')
        arm_results, artifact_proofs = {}, {}
        for arm in ('bf16','kv4'):
            raw = capture(report['stream_captures'][arm], lambda ch:workload/'results'/(arm+'-capture')/(ch+'.raw'),
                lambda ch:Path(manifest['target_root'])/'results'/(arm+'-capture')/(ch+'.raw'))
            records = [strict(line[len(b'SPRY_RESOURCE_V1 '):]) for line in raw['stdout'].splitlines() if line.startswith(b'SPRY_RESOURCE_V1 ')]
            need(len(records) == 2 and [r['phase'] for r in records] == ['initial','final'] and
                 dict(zip(('initial','final'),records)) == resource_rows[arm], 'raw pipe/resource receipt equality')
            controller.checked_physical_resources(resource_rows[arm],dict(manifest=manifest,resource_manifest=resource_manifest,
                spec_sha256=manifest['spec_sha256']),arm,results[arm])
            artifacts = source_module(workload/'review/benchmark_artifacts.py', '_retained_artifacts')
            artifact_limits=artifacts.HTTPArtifactLimits.from_limits(run['limits']) if configured else None
            binding = dict(session_sha256=manifest['session_sha256'], arm=arm, spec_sha256=manifest['spec_sha256'], experiment_mode=manifest['experiment_mode'])
            extracted = destination/(arm+'-artifacts')
            artifact_receipt = artifacts.receive_bundle(raw['stdout'],raw['stderr'],extracted,binding,**({'limits':artifact_limits} if configured else {}))
            need(artifact_receipt == report['artifact_receipts'][arm], 'original captured artifact transport receipt')
            controller.require_complete_artifact_members(artifact_receipt,spec,extracted,read(extracted/'result.json'),artifacts)
            expected = controller.http_expected_binding(workload,spec,owner_arm=arm,session_sha256=manifest['session_sha256'],
                **({'spec_sha256':manifest['spec_sha256']} if configured else {}))
            need(expected == campaign['http_bindings'][arm], 'actual source-derived physical/logical mapping')
            joins = [record for pair in report['live_namespace_joins'][arm] for record in pair.values()]
            actual = controller.http_integration.read_http_artifacts(extracted,
                extraction_root=destination/(arm+'-http-client-tree'),expected=expected,owner_arm=arm,
                joins=joins,supervision=results[arm],**({'limits':asdict(artifact_limits)} if configured else {}))
            accepted = controller.http_integration.accept_with_owner(actual,campaign,owner_arm=arm)
            need(report.get('http_serving_report_format') == 'retained-complete-http-wire-reference-v1' and
                 report.get('http_serving',{}).get(arm) == controller._http_wire_reference(accepted,owner_arm=arm),
                 'published complete post-validation HTTP wire reference')
            arm_results[arm] = actual['wire']
            artifact_proofs[arm] = dict(owner_arm=arm,logical_arm=actual['logical_arm'],binding=actual['binding'],
                client_bundle_sha256=actual['client_bundle_sha256'],native_bundle_sha256=actual['native_bundle_sha256'],
                result_sha256=actual['result_sha256'],complete_readout_sha256=sha(canonical(actual)))
        # Reverify every closed input after replay and extraction.
        need(sealer.verify(operation,operation_seal_sha256) == seals['operation'] and
             sealer.verify(workload,workload_seal_sha256) == seals['workload'] and
             sealer.verify(protected,protected_seal_sha256) == seals['protected'] and
             sealer.verify(source_root,registration['source_seal']['manifest_sha256']) == seals['source_author'], 'closed retained inputs stable after replay')
    return dict(schema=1,status='RETAINED_TWO_ARM_OWNER_RESOURCE_RELEASE_HTTP_REPLAY_COMPLETE',
        arm_results=arm_results,provenance=dict(package=computed,seals=seals,local_receipts=receipts,
            controller_sha256=sha(held(workload/'results/CONTROLLER-RESULT.json')),
            root_transport_sha256=sha(held(evidence/'ROOT-TRANSPORT.json')),phases=phases,
            protected_closeout_sha256=sha(closeout_raw),artifacts=artifact_proofs),qualification=False,precision_accepted=False,
        current_gpu_admission=False,native_quality_grade=False)
