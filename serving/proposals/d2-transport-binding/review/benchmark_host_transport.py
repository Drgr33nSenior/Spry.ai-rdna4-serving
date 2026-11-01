"""Human-started native-host observer transport; never runs a GPU task as root.

The root process accepts only four source-bound observation phases over an
inherited anonymous socket. It cannot execute commands requested by its peer.
Importing this module performs no host, credential, or GPU operation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time


PACKET_LIMIT = 2 * 1024 * 1024
ARCHIVE_LIMIT = 16 * 1024 * 1024
TOTAL_SECONDS = 6360
SEAL_RESERVE = 300
LOCK = Path('/home/developer/model-serving.izR2m1O9/diagnostic.lock')
USER_CGROUP = '/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read_file(path, maximum):
    path = Path(path)
    need(path.resolve() == path, 'canonical file required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
            'bounded singly linked regular file required')
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        stamp = lambda value: tuple(getattr(value, key) for key in
            ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
        need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
            'file changed during read')
        return bytes(raw)
    finally:
        os.close(fd)


def exclusive(path, raw, mode=0o600, *, maximum=256 << 10):
    need(type(raw) is bytes and len(raw) <= maximum, 'bounded control/evidence output required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            need(count > 0, 'short exclusive write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def root_source(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and digest(expected), 'root source path/pin')
    for directory in path.parents:
        info = directory.lstat()
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and
            (not info.st_mode & 0o022 or directory == Path('/var/tmp') and info.st_mode & stat.S_ISVTX),
            'root source parent is writable')
    info = path.lstat()
    need(info.st_uid == 0 and not info.st_mode & 0o022, 'root source file is writable')
    raw = read_file(path, 256 << 10)
    need(sha(raw) == expected, 'root source digest differs')
    return raw


def protected_stage(source_root):
    source_root = Path(source_root)
    stage = source_root.parent
    need(source_root.name == 'review' and stage.parent == Path('/var/tmp')
        and re.fullmatch(r'fp8-benchmark-source\.[A-Za-z0-9]{8,64}', stage.name)
        and stage.resolve() == stage, 'fixed protected source-stage layout required')
    info = stage.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and stat.S_IMODE(info.st_mode) == 0o700,
        'root-private source stage required')
    return stage


def exclusive_beneath(directory_fd, name, raw, mode=0o444):
    """Root writes through held no-follow directories, never developer paths."""
    parts = PurePosixPath(name).parts
    need(parts and not name.startswith('/') and str(PurePosixPath(name)) == name
        and not any(part in ('.', '..') for part in parts), 'unsafe relative output')
    held = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=held)
            os.close(held)
            held = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=held)
        try:
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                need(count > 0, 'short owned write')
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(held)


def stage_members(root_fd, members):
    """Stage authenticated review/config files through held no-follow directories."""
    runs = [name for name in members if re.fullmatch(r'config/runs/[a-z0-9][a-z0-9-]{0,63}/run.json', name)]
    hosts = [name for name in members if re.fullmatch(r'config/host/[a-z0-9][a-z0-9-]{0,63}\.json', name)]
    need(len(runs) == len(hosts) == 1, 'one run and host required for staging')
    run_directory = str(PurePosixPath(runs[0]).parent)
    need({name for name in members if name.startswith('config/')} ==
         {runs[0], hosts[0], run_directory + '/menu.json'}, 'exact configuration staging members required')
    held = {'': root_fd}
    try:
        for name in ('config', 'config/host', 'config/runs', run_directory):
            path = PurePosixPath(name)
            parent = held['' if str(path.parent) == '.' else str(path.parent)]
            os.mkdir(path.name, mode=0o755, dir_fd=parent)
            held[name] = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        for name, raw in members.items():
            exclusive_beneath(root_fd, name, raw)
        for name, fd in held.items():
            if name:
                os.fchmod(fd, 0o555)
    finally:
        for name, fd in reversed(tuple(held.items())):
            if name:
                os.close(fd)


def peer_identity(channel):
    need(hasattr(socket, 'SO_PEERCRED'), 'Linux peer credentials required')
    need(channel.family == socket.AF_UNIX and channel.type & 0xf == socket.SOCK_SEQPACKET,
        'anonymous sequenced-packet socket required')
    pid, uid, gid = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    return {'pid': pid, 'uid': uid, 'gid': gid}


def process_start(pid):
    raw = Path('/proc', str(pid), 'stat').read_bytes()
    need(len(raw) <= 65536 and int(raw.split(b'(', 1)[0]) == pid, 'process identity malformed')
    return int(raw[raw.rfind(b')') + 2:].split()[19])


def receive(channel, deadline):
    remaining = deadline - time.monotonic()
    need(remaining > 0, 'socket deadline exhausted')
    channel.settimeout(remaining)
    raw, _, flags, _ = channel.recvmsg(PACKET_LIMIT + 1)
    need(raw and not flags & socket.MSG_TRUNC and len(raw) <= PACKET_LIMIT, 'closed or oversized packet')
    return decode(raw)


def send(channel, value, deadline):
    raw = canonical(value)
    need(len(raw) <= PACKET_LIMIT and time.monotonic() < deadline, 'packet bound/deadline')
    channel.settimeout(max(.001, deadline - time.monotonic()))
    need(channel.send(raw) == len(raw), 'incomplete socket packet')


class RpcClient:
    """UID1000 side. The inherited socket's creator must be the pinned root parent."""
    def __init__(self, fd, grant, manifest_sha256):
        self.channel = socket.socket(fileno=os.dup(fd))
        self.grant, self.manifest_sha256 = grant, manifest_sha256
        self.authenticate_parent()

    def authenticate_parent(self):
        peer = peer_identity(self.channel)
        expected = self.grant['transport_parent']
        need(peer['pid'] == expected['pid'] and peer['uid'] == peer['gid'] == expected['uid'] == 0
            and process_start(peer['pid']) == expected['start_ticks']
            and expected['boot_id'] == self.grant['boot_id']
            and self.grant['manifest_sha256'] == self.manifest_sha256, 'root transport parent differs')
        return dict(expected)

    def observe(self, request, *, expected_observer_sha256, deadline_monotonic):
        self.authenticate_parent()
        send(self.channel, {'schema': 1, 'operation': 'observe', 'request': request,
            'expected_observer_sha256': expected_observer_sha256,
            'deadline_monotonic': deadline_monotonic}, deadline_monotonic)
        response = receive(self.channel, deadline_monotonic)
        need(type(response) is dict and response.get('schema') == 1 and response.get('ok') is True
            and set(response) == {'schema', 'ok', 'snapshot'}, 'root observation failed')
        return response['snapshot']

    def close(self):
        self.channel.close()


MXFP4_FILES = {'review/__init__.py': '39cd59ab506e69df4cb77664d67392e6c510174be5eeb61c172e291b47c59539',
 'review/isolated_scheme.py': 'f396186d36fa83812d7d3d4ec538e47588d3513a06e9996539e1481bebe8c185',
 'review/kernels.py': '5571b10654697561455081455b6cd28d43314702f56c80160ef2e33e535b3693',
 'review/ops.py': '66d25968516c6c9509923ad363a291d57d192918f1e7121f96481202391c3776',
 'review/prefill_reference.py': 'cd5841650fc59da7f1537e82ffebf16fb07ca0db7d191cdfec59540575bbd887',
 'review/private_selection.py': 'e0120a933d0848de1fd82135274deac4ca850af82715112a2a1bfa94107c52ed'}
P1_FILES = {
 "review/p1_fp8_kernels.py": "0d2919277f8cd0f2196e3b93f89444f434310841b7cedc935d02c5cd2ebbfcb1",
 "review/p1_prefill.py": "c2aec66e246119f879c3afcf10c39ce0aa9f9e36e2baa596e56a8f0bc572d5fc"
}

D3B_TARGET_FILES = {
 "review/capture_model_runner.py": "5e2325bb08930cdffa55cc9d944cb1dc589b662c42de821eb391c82c8bdf37de",
 "review/d3b_head_fp8_kernels.py": "32522a45ccb6bf11b887427257853f6af38ced5b1fb9e188fd49ed4867055c51",
 "review/d3b_head_fp8_ops.py": "4814926eb6294f76d29516ad68f78271d1cf879c0de3815554d8044cd8251770",
 "review/d3b_head_launches.py": "e88da9ee318a5e0fac1282a757e7c4282cfddd0b97a8c160419732e42e15a442",
 "review/d3b_target_head_adapter.py": "88721f70837b6775bdabcca9bf9b9f44cac5c84da804699f864e0b080108bee4"
}


def _archive_member_limit(manifest, members):
    """INERT PROPOSAL: exact M8 source closure; other experiment limits stay fixed."""
    expected = manifest['package_files']
    raw = members.get('review/PRECISION-REFERENCE.json')
    profile = decode(raw) if raw is not None else {}
    candidate = profile.get('precision') == 'mxfp4' or bool(set(expected) & set(MXFP4_FILES))
    if candidate:
        need(manifest.get('experiment_mode') == 'native-http-fp8-sprint01-v1' and
             all(expected.get(name) == pin and name in members and sha(members[name]) == pin
                 for name, pin in MXFP4_FILES.items()) and
             profile.get('repository') == 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4' and
             profile.get('revision') == '5233554c5fa56afda40150556b95573c2d7d29c0' and
             profile.get('config_sha256') == '4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55' and
             profile.get('quantization') == 'quark' and
             profile.get('cache_profile') == {'target': 'bf16', 'draft': 'bf16'} and
             not ({'review/attention_capture.py', 'review/raw_attention.py'} & set(expected)),
             'exact MXFP4 archive source/profile extension required')
        need(expected.get('review/layernorm_p3_candidate.py') == 'f1855c99386c39802b07dd1848b0d9197994f67181cd1265dbf85b02da513b42' and
             sha(members.get('review/layernorm_p3_candidate.py', b'')) == 'f1855c99386c39802b07dd1848b0d9197994f67181cd1265dbf85b02da513b42',
             'exact P3 norm overlay source required')
        need(all(expected.get(name) == pin and sha(members.get(name, b'')) == pin for name, pin in P1_FILES.items()),
             'exact P1 prefill route source closure required')
        need(all(expected.get(name) == pin and sha(members.get(name, b'')) == pin
                 for name, pin in D3B_TARGET_FILES.items()), 'exact D3b target head source closure required')
        need(expected.get('review/qwen3_5_candidate.py') == '2a76ddea9a380dd8e8fd71a8919fc4aadae702ec5c5f35bd052c614ed07a78e8' and
             sha(members.get('review/qwen3_5_candidate.py', b'')) == '2a76ddea9a380dd8e8fd71a8919fc4aadae702ec5c5f35bd052c614ed07a78e8',
             'exact GDN copy model source closure required')
        need(expected.get('review/d3b_head_adapter.py') == 'd0eb6f467f29b24079254d5b5cdd0101291ae6e0a5e6f92ad2ed9798123ca942' and
             sha(members.get('review/d3b_head_adapter.py', b'')) == 'd0eb6f467f29b24079254d5b5cdd0101291ae6e0a5e6f92ad2ed9798123ca942',
             'exact current drafter-head source closure required')
        need(expected.get('review/d3b_combined_model_runner.py') == 'da81d4c393cf5910753c48036bedbe593e851ffe0985997f4eca02d79737df76' and
             sha(members.get('review/d3b_combined_model_runner.py', b'')) == 'da81d4c393cf5910753c48036bedbe593e851ffe0985997f4eca02d79737df76',
             'exact current drafter-head source closure required')
        return 80
    return 66 if 'review/raw_attention.py' in expected else 65 if 'review/attention_capture.py' in expected else 64


CONFIGURATION_PIN_FIELDS = ('run_config_sha256', 'host_sha256', 'protected_set_sha256')
PROTECTED_SOURCE_NAMES = (
    'benchmark_artifacts.py', 'benchmark_gpu_identity.py', 'benchmark_host_transport.py',
    'benchmark_owner_adapter.py', 'benchmark_physical_controller.py',
    'benchmark_release_observer.py', 'benchmark_resource_gate.py',
    'collect_benchmark_platform.py', 'host_process_owner.py')


def configuration_members(manifest, members, configuration_pins):
    """Authenticate configuration bytes against independently supplied owner pins."""
    need(type(configuration_pins) is dict and set(configuration_pins) == set(CONFIGURATION_PIN_FIELDS)
         and all(digest(pin) for pin in configuration_pins.values()), 'three configuration approval pins required')

    def read(name, pin, maximum):
        raw = members.get(name)
        need(type(raw) is bytes and 0 < len(raw) <= maximum and digest(pin) and sha(raw) == pin,
             'configuration member digest/size differs')
        value = decode(raw)
        need(type(value) is dict and canonical(value) == raw, 'canonical configuration JSON object required')
        return value

    protected = read('PROTECTED-SHA256.json', configuration_pins['protected_set_sha256'], 64 << 10)
    need(set(protected) == {'review/' + name for name in PROTECTED_SOURCE_NAMES}
         and all(digest(pin) for pin in protected.values()), 'exact protected file set required')
    for name, pin in protected.items():
        raw = members.get(name)
        need(manifest['package_files'].get(name) == pin and type(raw) is bytes
             and 0 < len(raw) <= 1 << 20 and sha(raw) == pin, 'protected set/package binding differs')
    run_name = manifest.get('run_config')
    need(type(run_name) is str and re.fullmatch(r'config/runs/[a-z0-9][a-z0-9-]{0,63}/run.json', run_name),
         'exact run configuration location required')
    run = read(run_name, configuration_pins['run_config_sha256'], 256 << 10)
    need(type(run.get('schema')) is int and run['schema'] == 1
         and run.get('run_id') == PurePosixPath(run_name).parent.name, 'run configuration identity differs')
    host, menu = run.get('host'), run.get('menu')
    need(type(host) is dict and set(host) == {'file', 'sha256'} and type(host['file']) is str
         and re.fullmatch(r'config/host/[a-z0-9][a-z0-9-]{0,63}\.json', host['file'])
         and host['sha256'] == configuration_pins['host_sha256'], 'run/host approval binding differs')
    need(type(menu) is dict and set(menu) == {'file', 'sha256'} and menu['file'] == 'menu.json'
         and digest(menu['sha256']), 'run/menu binding differs')
    host_value = read(host['file'], configuration_pins['host_sha256'], 64 << 10)
    need(set(host_value) == {'schema', 'devices', 'rank_indices', 'power_cap_watts'}
         and type(host_value['schema']) is int and host_value['schema'] == 1,
         'host configuration fields differ; boot ID belongs in the approval')
    menu_name = str(PurePosixPath(run_name).parent / menu['file'])
    read(menu_name, menu['sha256'], 1 << 20)
    return {'PROTECTED-SHA256.json', run_name, host['file'], menu_name}


def inspect_archive(path, expected_sha256, manifest_sha256, *, configuration_pins):
    """Read authenticated regular members without extracting an archive path."""
    need(digest(expected_sha256) and digest(manifest_sha256), 'independent archive/manifest pins required')
    raw = read_file(path, ARCHIVE_LIMIT)
    need(sha(raw) == expected_sha256, 'archive digest differs')
    import io
    members, size = {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        for item in archive:
            name = item.name
            need(item.isfile() and not item.issym() and not item.islnk()
                and str(PurePosixPath(name)) == name and not name.startswith('/')
                and not any(part in ('.', '..', '.idea', 'keeper-env') for part in name.split('/'))
                and (name in ('MANIFEST.json', 'PROTECTED-SHA256.json')
                     or re.fullmatch(r'review/[A-Za-z0-9_.-]+', name) or name in MXFP4_FILES
                     or re.fullmatch(r'config/runs/[a-z0-9][a-z0-9-]{0,63}/(?:run|menu)\.json', name)
                     or re.fullmatch(r'config/host/[a-z0-9][a-z0-9-]{0,63}\.json', name))
                and name not in members and 0 <= item.size <= 16 * 1024 * 1024,
                'unsafe, linked, duplicate or oversized archive member')
            size += item.size
            need(size <= ARCHIVE_LIMIT and len(members) < 80, 'archive expanded size/member limit')
            value = archive.extractfile(item).read(item.size + 1)
            need(len(value) == item.size, 'short archive member')
            members[name] = value
    need('MANIFEST.json' in members and sha(members['MANIFEST.json']) == manifest_sha256,
        'manifest identity differs')
    manifest = decode(members['MANIFEST.json'])
    expected = manifest['package_files']
    need(len(members) <= _archive_member_limit(manifest, members),
         'archive mode-specific member limit')
    configuration = configuration_members(manifest, members, configuration_pins)
    need(not configuration & set(expected), 'configuration must be independently approved')
    need(set(members) == {'MANIFEST.json'} | set(expected) | configuration and all(sha(members[name]) == pin
        and digest(pin) for name, pin in expected.items()), 'archive exact source inventory differs')
    need(manifest.get('schema') == 2 and manifest.get('scope') == 'matched-benchmark-physical-package-v2'
        and manifest.get('launch_authorized') is False
        and manifest.get('proposed_complete_seconds') == TOTAL_SECONDS
        and manifest.get('package_root') == manifest.get('target_root')
        and re.fullmatch(r'/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}', manifest['target_root']),
        'physical package contract differs')
    return manifest, members


def new_grant(manifest, manifest_sha256, *, received_utc, received_monotonic, parent, configuration_pins):
    need(parent['uid'] == 0 and parent['boot_id'] == manifest['boot_id'], 'native root parent required')
    need(type(configuration_pins) is dict and set(configuration_pins) == set(CONFIGURATION_PIN_FIELDS)
         and all(digest(pin) for pin in configuration_pins.values()), 'three configuration approval pins required')
    files = manifest['package_files']
    return {'schema': 2, 'scope': 'one-matched-fp8-kv-benchmark', 'model_start_authorized': True,
        'maximum_starts': 2, 'arm_order': ['bf16', 'kv4'], 'cumulative_ceiling_seconds': TOTAL_SECONDS,
        'fixed_reserve_seconds': SEAL_RESERVE, 'received_utc': received_utc,
        'received_monotonic': received_monotonic, 'boot_id': manifest['boot_id'],
        'manifest_sha256': manifest_sha256, 'package_sha256': sha(canonical(files)),
        'source_manifest_sha256': sha(canonical({name: pin for name, pin in files.items() if name.endswith('.py')})),
        'transport_parent': parent, **configuration_pins}


class PhaseSequence:
    def __init__(self, manifest, observer_sha256, active_end):
        self.manifest, self.observer_sha256, self.active_end = manifest, observer_sha256, active_end
        self.admitted, self.releases, self.closed = False, [], False

    def check(self, packet):
        need(type(packet) is dict and set(packet) == {'schema', 'operation', 'request',
            'expected_observer_sha256', 'deadline_monotonic'} and packet['schema'] == 1
            and packet['operation'] == 'observe' and packet['expected_observer_sha256'] == self.observer_sha256,
            'only the frozen observation operation is permitted')
        request = packet['request']
        need(type(request) is dict and request['root'] == self.manifest['target_root']
            and request['session_sha256'] == self.manifest['session_sha256']
            and request['boot_id'] == self.manifest['boot_id']
            and request['deadline_monotonic'] == packet['deadline_monotonic']
            and type(packet['deadline_monotonic']) in (float, int)
            and math.isfinite(packet['deadline_monotonic'])
            and time.monotonic() < packet['deadline_monotonic'] <= self.active_end, 'RPC binding/deadline differs')
        mode, arm = request['mode'], request['arm']
        need(not self.closed and (mode == 'locked-admission' and not self.admitted and not self.releases
            or mode == 'release' and self.admitted and len(self.releases) < 2
                and arm == ['bf16', 'kv4'][len(self.releases)]
            or mode == 'independent' and self.admitted and bool(self.releases)), 'observation phase order differs')
        return request

    def finish(self, request):
        if request['mode'] == 'locked-admission':
            self.admitted = True
        elif request['mode'] == 'release':
            self.releases.append(request['arm'])
        else:
            self.closed = True


def resource_absence(arms, session):
    records = []
    for arm in arms:
        need(arm in ('bf16', 'kv4') and digest(session), 'resource arm/session binding')
        path = USER_CGROUP + 'spry-fp8-benchmark-' + session[:16] + '-' + arm + '.scope'
        try:
            Path(path).lstat()
        except FileNotFoundError:
            records.append({'arm': arm, 'cgroup_path': path, 'absent': True,
                'observed_monotonic': time.monotonic()})
        else:
            raise ValueError('owned resource scope remains; no root cleanup fallback')
    return {'scopes': records}


def run_observer(request, source_root, pin, *, core, artifacts, evidence_root):
    """One fixed root-owned observer child with a complete 75-second envelope."""
    folder = Path('/var/tmp/fp8-benchmark-observer.' + secrets.token_hex(8))
    folder.mkdir(mode=0o700)
    raw = canonical(request)
    exclusive(folder / 'REQUEST.json', raw, 0o444)
    end = request['deadline_monotonic']
    argv = ['/usr/bin/python3', '-I', '-B', '-S', str(source_root / 'benchmark_release_observer.py'),
        '--collect-owner-snapshot', '--request', str(folder / 'REQUEST.json'),
        '--request-sha256', sha(raw), '--expected-self-sha256', pin,
        '--output', str(folder / 'SNAPSHOT.json')]
    # Core reaps and signals authenticated child instances, never an arbitrary
    # PID/group. Its own root descendants are the read-only collector children.
    capture = artifacts.CapturePipes(folder / 'capture', stream_limit=64 << 10)
    try:
        result = core.supervise(argv, script_path=argv[4], script_sha256=pin,
            deadlines=core.Deadlines(end - 15, end - 5, end - 2, end),
            on_child=lambda child: exclusive(folder / 'CHILD.json', canonical(vars(child)), maximum=64 << 10),
            stdout=capture.stdout, stderr=capture.stderr, resource_check=capture.check)
    finally:
        capture_receipt = capture.finish()
    receipt = {'supervision': vars(result), 'request_sha256': sha(raw), 'evidence_path': str(folder)}
    from dataclasses import asdict
    receipt['supervision'] = asdict(result)
    receipt['capture'] = capture_receipt
    exclusive(folder / 'RECEIPT.json', canonical(receipt))
    name = request['mode'] + ('-' + request['arm'] if request['arm'] else '')
    # Preserve bounded failure evidence before refusing. A failed observation
    # must never be promoted to a release snapshot or advance the phase.
    exclusive(evidence_root / (name + '-owner.json'), canonical(receipt))
    for stream in ('stdout', 'stderr'):
        exclusive(evidence_root / (name + '-' + stream + '.raw'),
            read_file(folder / 'capture' / (stream + '.raw'), 64 << 10))
    need(result.process_cleanup_observed and result.exit_code == 0 and result.failure is None
        and capture_receipt.get('capture_complete') is True,
        'root observer failed or release is uncertain')
    snapshot = decode(read_file(folder / 'SNAPSHOT.json', PACKET_LIMIT))
    exclusive(evidence_root / (name + '.json'), canonical(snapshot), maximum=PACKET_LIMIT)
    return snapshot


def seal(root, *, deadline):
    """Exact regular-file membership; seal remains external to its own inventory."""
    def files():
        found, total = {}, 0
        for path in sorted(root.rglob('*')):
            need(time.monotonic() < deadline and not path.is_symlink(), 'seal deadline/link refusal')
            if path.is_dir():
                continue
            need(path.is_file(), 'seal refuses special evidence')
            if path.name == 'FINAL-ARTIFACTS.sha256' and path.parent == root:
                continue
            raw = read_file(path, 128 * 1024 * 1024)
            total += len(raw)
            need(total <= 96 << 20, 'protected source/observer/outer-output partition exceeded')
            found[str(path.relative_to(root))] = sha(raw)
        return found
    members = files()
    manifest = ''.join(pin + '  ' + name + '\n' for name, pin in members.items()).encode()
    path = root / 'FINAL-ARTIFACTS.sha256'
    exclusive(path, manifest, 0o444)
    # Directory timestamp changes on creating the seal. Put the seal timestamp
    # last, then verify every directory (including root), every member and count.
    os.utime(path, None)
    need(files() == members, 'seal hash/membership changed')
    timestamp = path.stat().st_mtime_ns
    need(root.stat().st_mtime_ns <= timestamp and all(value.lstat().st_mtime_ns <= timestamp
        for value in root.rglob('*')), 'entry newer than seal')
    return {'manifest_sha256': sha(manifest), 'member_count': len(members)}


def human_main(args):
    need(os.getuid() == os.geteuid() == os.getgid() == os.getegid() == 0,
        'human must start the reviewed root wrapper; agent execution is not permitted')
    beginning, received = args.original_receipt
    active_end, seal_end = beginning + TOTAL_SECONDS - SEAL_RESERVE, beginning + TOTAL_SECONDS
    source_root = Path(__file__).resolve().parent
    protected_stage(source_root)
    # Root imports only authenticated files under its private staged source.
    root_source(Path(__file__).resolve(), args.expected_self_sha256)
    configuration_pins = {name: getattr(args, name) for name in CONFIGURATION_PIN_FIELDS}
    manifest, members = inspect_archive(Path(args.archive).resolve(), args.archive_sha256, args.manifest_sha256,
        configuration_pins=configuration_pins)
    need(manifest['package_files']['review/benchmark_host_transport.py'] == args.expected_self_sha256,
        'root transport/package identity differs')
    trusted = {name: root_source(source_root / name, manifest['package_files']['review/' + name])
        for name in ('host_process_owner.py', 'benchmark_release_observer.py',
                     'collect_benchmark_platform.py', 'benchmark_artifacts.py')}
    expected_stage = {str(source_root / name) for name in trusted} | {str(Path(__file__).resolve()),
        str(source_root.parent / 'payload.tar')}
    need(Path(args.archive) == source_root.parent / 'payload.tar' and
        {str(path) for path in source_root.parent.rglob('*') if not path.is_dir()} == expected_stage,
        'protected stage must contain only the reviewed source subset and archive')
    need(Path(args.archive).stat().st_size + sum(map(len, members.values()))
        + sum(map(len, trusted.values())) + len(read_file(Path(__file__).resolve(), 256 << 10))
        <= 32 << 20, 'combined protected archive/source and workload source limit')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(boot == manifest['boot_id'], 'new boot requires reviewed package rebinding')
    root = Path(manifest['target_root'])
    root.mkdir(mode=0o700)  # Exclusive: never reuse/remove an earlier remote root.
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    held = os.fstat(root_fd)
    need(held.st_uid == 0 and stat.S_IMODE(held.st_mode) == 0o700, 'new root directory substituted')
    os.mkdir('review', mode=0o755, dir_fd=root_fd)
    stage_members(root_fd, members)
    os.mkdir('arms', mode=0o755, dir_fd=root_fd)
    arms_fd = os.open('arms', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        for arm in ('bf16', 'kv4'):
            os.mkdir(arm, mode=0o755, dir_fd=arms_fd)
            exclusive_beneath(arms_fd, arm + '/hosts', b'127.0.0.1 localhost\n')
    finally:
        os.close(arms_fd)
    review_fd = os.open('review', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    os.fchmod(review_fd, 0o555)
    os.close(review_fd)
    review = root / 'review'
    parent = {'pid': os.getpid(), 'uid': 0, 'start_ticks': process_start(os.getpid()), 'boot_id': boot}
    grant = new_grant(manifest, args.manifest_sha256, received_utc=received,
        received_monotonic=beginning, parent=parent, configuration_pins=configuration_pins)
    raw = canonical(grant)
    exclusive_beneath(root_fd, 'GRANT.json', raw)
    receipt = canonical({'schema': 1, 'received_utc': received,
        'grant_sha256': sha(raw), 'archive_sha256': args.archive_sha256,
        'active_deadline_monotonic': active_end, 'seal_deadline_monotonic': seal_end,
        'gpu_uid': 1000, 'root_performs_gpu_work': False})
    exclusive_beneath(root_fd, 'RECEIPT.json', receipt)
    named = root.lstat()
    need((named.st_dev, named.st_ino) == (held.st_dev, held.st_ino), 'private directory renamed before handoff')
    os.fchown(root_fd, 1000, 1000)
    os.close(root_fd)
    # No root process writes or seals developer-controlled paths after handoff.
    # The unprivileged controller seals its own tree. Root observations, bounded
    # output and independent closeout remain under the root-private source stage.
    evidence = source_root.parent / 'evidence'
    evidence.mkdir(mode=0o700)
    exclusive(evidence / 'GRANT.json', raw, 0o444)
    exclusive(evidence / 'RECEIPT.json', receipt, 0o444)
    spec = importlib.util.spec_from_file_location('spry_root_owned_core', source_root / 'host_process_owner.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    exec(compile(trusted['host_process_owner.py'], str(source_root / 'host_process_owner.py'), 'exec'), core.__dict__)
    spec = importlib.util.spec_from_file_location('spry_root_bounded_capture', source_root / 'benchmark_artifacts.py')
    artifacts = importlib.util.module_from_spec(spec)
    exec(compile(trusted['benchmark_artifacts.py'], str(source_root / 'benchmark_artifacts.py'), 'exec'), artifacts.__dict__)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    # FD9 belongs exclusively to the later UID1000 canonical lock owner.
    import fcntl
    moved = fcntl.fcntl(right.fileno(), fcntl.F_DUPFD_CLOEXEC, 32)
    right.close()
    right = socket.socket(fileno=moved)
    sequence = PhaseSequence(manifest, manifest['package_files']['review/benchmark_release_observer.py'], active_end)
    argv = ['/usr/bin/python3', '-I', '-B', '-S', str(review / 'benchmark_physical_controller.py'),
        '--execute-owner-benchmark', '--manifest', str(root / 'MANIFEST.json'),
        '--manifest-sha256', args.manifest_sha256, '--grant-json', str(root / 'GRANT.json'),
        '--grant-sha256', sha(raw), '--observer-fd', str(right.fileno())]
    env = {'PATH': '/usr/bin:/bin', 'HOME': '/home/developer', 'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'XDG_RUNTIME_DIR': '/run/user/1000',
        'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus'}
    import pwd
    account = pwd.getpwnam('developer')
    need(account.pw_uid == account.pw_gid == 1000 and time.monotonic() + 1995 <= active_end,
        'complete preparation/owner envelope does not fit')
    known, errors, owner_signals, escalations = [], [], [], []
    previous = {number: signal.signal(number, lambda number, frame: owner_signals.append(number))
        for number in (signal.SIGINT, signal.SIGTERM)}
    child = None
    try:
        capture = artifacts.CapturePipes(evidence / 'controller-capture', stream_limit=8 << 20)
        try:
            child = subprocess.Popen(argv, user=1000, group=1000,
                extra_groups=os.getgrouplist('developer', 1000), pass_fds=(right.fileno(),),
                close_fds=True, stdin=subprocess.DEVNULL, stdout=capture.stdout, stderr=capture.stderr,
                start_new_session=True, env=env, cwd=root)
            right.close()
            identity = core.read_process(child.pid)
            need(identity is not None and identity.uid == identity.gid == 1000,
                'unprivileged controller identity was not observed')
            known.append(identity)
            exclusive(evidence / 'CONTROLLER-IDENTITY.json', canonical(vars(identity)))
            selector = selectors.DefaultSelector()
            selector.register(left, selectors.EVENT_READ)
            signaled = False
            try:
                while child.poll() is None:
                    capture.check()
                    for value in core.descendants(known):
                        if not any(value.same_instance(old) for old in known):
                            known.append(value)
                    need(len(known) <= 512, 'controller descendant bound exceeded')
                    now = time.monotonic()
                    if owner_signals and not signaled:
                        escalations.append(core.signal_authenticated(identity, signal.SIGTERM))
                        signaled = True
                    if now >= active_end - 30:
                        raise TimeoutError('outer active deadline; preserving final kill/reap reserve')
                    for _, _ in selector.select(.05):
                        packet = receive(left, min(active_end - 30, time.monotonic() + 1))
                        try:
                            request = sequence.check(packet)
                            snapshot = run_observer(request, source_root, sequence.observer_sha256,
                                core=core, artifacts=artifacts, evidence_root=evidence)
                            arms = [request['arm']] if request['mode'] == 'release' else sequence.releases
                            if request['mode'] != 'locked-admission':
                                snapshot['root_resource_release'] = resource_absence(arms, manifest['session_sha256'])
                                exclusive(evidence / (request['mode'] + ('-' + request['arm'] if request['arm'] else '')
                                    + '-cgroup.json'), canonical(snapshot['root_resource_release']), maximum=64 << 10)
                            sequence.finish(request)
                            send(left, {'schema': 1, 'ok': True, 'snapshot': snapshot}, request['deadline_monotonic'])
                            if sequence.closed:
                                # The child seals after its final reply and may
                                # close RPC before exiting. EOF is not a fifth
                                # observation or a transport failure.
                                selector.unregister(left)
                        except Exception as error:
                            errors.append(type(error).__name__ + ': ' + str(error)[:512])
                            send(left, {'schema': 1, 'ok': False, 'error': errors[-1]},
                                min(active_end - 30, time.monotonic() + 1))
            finally:
                selector.close()
            child.wait(timeout=max(.001, active_end - time.monotonic()))
        finally:
            # On abnormal exit the cleanup below must run before pipe finality
            # can be asserted. finish is therefore performed after that cleanup.
            pass
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error)[:512])
    finally:
        right.close()
        left.close()
        if child is not None:
            cleanup_end = min(active_end, time.monotonic() + 30)
            # Only authenticated UID1000 descendants can receive a signal.
            for number, end in ((signal.SIGTERM, cleanup_end - 10), (signal.SIGKILL, cleanup_end - 2)):
                for value in reversed(known):
                    try:
                        receipt = core.signal_authenticated(value, number)
                        if receipt:
                            escalations.append(receipt)
                    except Exception as error:
                        errors.append('termination: ' + type(error).__name__)
                while time.monotonic() < end and child.poll() is None:
                    time.sleep(.02)
            try:
                child.wait(timeout=max(.001, cleanup_end - time.monotonic()))
            except subprocess.TimeoutExpired:
                errors.append('controller reaping uncertain')
        for number, handler in previous.items():
            signal.signal(number, handler)
    survivors = []
    for value in known:
        actual = core.read_process(value.pid)
        if actual is not None and value.same_instance(actual):
            survivors.append(vars(actual))
    try:
        capture_receipt = capture.finish()
        need(capture_receipt.get('capture_complete') is True, 'controller capture incomplete')
    except Exception as error:
        capture_receipt = {'capture_complete': False, 'failure': type(error).__name__}
        errors.append('controller output: ' + type(error).__name__)
    summary = {'schema': 1, 'scope': 'human-root-matched-benchmark-transport-v1',
        'controller_exit': child.returncode if child else None, 'errors': errors,
        'owner_signals': owner_signals, 'escalation_receipts': escalations,
        'controller_capture': capture_receipt,
        'observed_processes': [vars(value) for value in known], 'survivors': survivors,
        'root_performed_gpu_work': False, 'qualification': False,
        'all_observation_phases_complete': sequence.closed and sequence.releases == ['bf16', 'kv4'],
        'physical_success': bool(child and child.returncode == 0 and not errors and not survivors
            and not owner_signals and sequence.closed and sequence.releases == ['bf16', 'kv4'])}
    exclusive(evidence / 'ROOT-TRANSPORT.json', canonical(summary), maximum=4 << 20)
    result = seal(source_root.parent, deadline=seal_end)
    completed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    elapsed = time.monotonic() - beginning
    need(elapsed <= TOTAL_SECONDS, 'verified seal deadline exceeded')
    closeout = {**result, 'root': str(source_root.parent), 'workload_evidence_root': str(root),
        'verified_completion_utc': completed,
        'elapsed_seconds': elapsed, 'charged_seconds': math.ceil(elapsed),
        'physical_success': summary['physical_success'], 'qualification': False}
    exclusive(Path(str(source_root.parent) + '.closeout.json'), canonical(closeout), 0o444)
    print(canonical(closeout).decode(), end='', flush=True)
    return 0 if summary['physical_success'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-root-benchmark', action='store_true')
    for name in ('archive', 'archive-sha256', 'manifest-sha256', 'expected-self-sha256',
                 'run-config-sha256', 'host-sha256', 'protected-set-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    need(args.human_root_benchmark, 'explicit human physical invocation required')
    args.original_receipt = (time.monotonic(), datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    try:
        return human_main(args)
    except BaseException as error:
        # Bootstrap refusals also consume the human's one use. Do not touch an
        # existing workload root, retry, or infer device release from this error.
        root_source(Path(__file__).resolve(), args.expected_self_sha256)
        stage = protected_stage(Path(__file__).resolve().parent)
        failure = {'schema': 1, 'status': 'BLOCKED_INCOMPLETE', 'retry_authorized': False,
            'qualification': False, 'failure': type(error).__name__ + ': ' + str(error)[:512],
            'received_utc': args.original_receipt[1], 'release_authority': False}
        if not os.path.lexists(stage / 'FINAL-ARTIFACTS.sha256'):
            exclusive(stage / 'BOOTSTRAP-FAILURE.json', canonical(failure))
            result = seal(stage, deadline=args.original_receipt[0] + TOTAL_SECONDS)
        else:
            result = {'seal_verification_failed_or_already_closed': True}
        elapsed = time.monotonic() - args.original_receipt[0]
        closeout = {**failure, **result, 'elapsed_seconds': elapsed, 'charged_seconds': math.ceil(elapsed),
            'verified_completion_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}
        exclusive(Path(str(stage) + '.refusal.json'), canonical(closeout), 0o444)
        print(canonical(closeout).decode(), end='', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
