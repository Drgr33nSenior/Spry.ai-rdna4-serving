"""Bounded pre-launch or post-closeout host read: boot id, ownership record, power caps (approved wake).

The wake opens each card's render node inside the read and closes it before returning, as the approved
Screen 7 power guard does; no power or runtime-PM setting is changed.
Uses the existing authorized observer (sudo -n python3 -I -B -S -), 15 seconds, no GPU work.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from connection import SSH, check
from local_commands import run

ROOT = Path(__file__).resolve().parent
MAXIMUM_POWER_CAP_WATTS = 235
CODE = '''import json,os,socket,stat
from pathlib import Path
assert os.getuid()==os.getgid()==0 and socket.gethostname()=="threadripper-ai"
boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
journal=Path("/home/developer/model-serving.izR2m1O9/diagnostic.lock.unresolved.json")
fds,wake,power=[],[],{}
try:
    for bdf,node in NODES.items():
        device=Path("/sys/bus/pci/devices")/bdf
        assert sorted(p.name for p in (device/"drm").iterdir() if p.name.startswith("renderD"))==[node]
        fd=os.open("/dev/dri/"+node,os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW);fds.append(fd)
        assert stat.S_ISCHR(os.fstat(fd).st_mode)
        wake.append(dict(bdf=bdf,node="/dev/dri/"+node,runtime_status_while_open=(device/"power/runtime_status").read_text()[:16].strip()))
    for bdf in NODES:
        paths=sorted((Path("/sys/bus/pci/devices")/bdf/"hwmon").glob("hwmon*/power1_cap"))
        assert len(paths)==1
        raw=paths[0].read_bytes()[:33]
        assert raw.strip().isdigit()
        power[bdf]=int(raw)
finally:
    for fd in fds:
        os.close(fd)
roots={p:(Path(p).exists() or Path(p).is_symlink()) for p in ROOTS}
print(json.dumps(dict(phase=PHASE,boot_id=boot,owner_journal_absent=not journal.exists() and not journal.is_symlink(),
    power_microwatts=power,wake=wake,descriptors_closed=True,campaign_roots_present=roots,model_or_GPU_operations=False),sort_keys=True))
'''


def read_host(binding):
    relative, pin = binding.get('host_config'), binding.get('host_sha256')
    if not (type(relative) is str and re.fullmatch(r'config/host/[a-z0-9][a-z0-9-]{0,63}\.json', relative)
            and type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin)):
        raise ValueError('explicit host configuration binding required')
    path = ROOT / relative
    if path.resolve(strict=True) != path:
        raise ValueError('canonical host configuration required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not (stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 64 << 10):
            raise ValueError('bounded regular host configuration required')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read((64 << 10) + 1)
        stamp = lambda row: (row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        if (len(raw) != before.st_size or stamp(before) != stamp(os.fstat(fd))
                or stamp(before) != stamp(path.lstat()) or hashlib.sha256(raw).hexdigest() != pin):
            raise ValueError('host configuration changed or hash differs')
    finally:
        os.close(fd)
    value = json.loads(raw)
    canonical = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if not (type(value) is dict and set(value) == {'schema','devices','rank_indices','power_cap_watts'}
            and type(value['schema']) is int and value['schema'] == 1 and raw == canonical
            and type(value['power_cap_watts']) is int and 0 < value['power_cap_watts'] <= MAXIMUM_POWER_CAP_WATTS
            and type(value['rank_indices']) is list and len(value['rank_indices']) == 2
            and all(type(index) is int for index in value['rank_indices']) and set(value['rank_indices']) == {0,1}
            and type(value['devices']) is dict and len(value['devices']) == 2
            and all(type(bdf) is str and re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf)
                    and type(row) is dict and set(row) == {'uuid','render'}
                    and type(row['uuid']) is str and row['uuid'] and type(row['render']) is str
                    and re.fullmatch(r'/dev/dri/renderD[0-9]+', row['render']) for bdf,row in value['devices'].items())
            and len({row['render'] for row in value['devices'].values()}) == 2):
        raise ValueError('closed bounded host configuration required')
    return value


def observation_code(binding, host, phase):
    roots = [binding['target_root'], binding['protected_root']]
    nodes = {bdf: Path(row['render']).name for bdf,row in host['devices'].items()}
    prefix = 'ROOTS = ' + repr(roots) + '\nPHASE = ' + repr(phase) + '\nNODES = ' + repr(nodes) + '\n'
    code = prefix + CODE
    ast.parse(code, filename='power-observe')
    return code


def main(argv):
    if argv not in (['pre'], ['post']):
        raise ValueError('expected exactly pre or post')
    phase = argv[0]
    binding = json.loads((ROOT / 'LAUNCHER-BINDINGS.json').read_bytes())
    host = read_host(binding)
    cap = host['power_cap_watts'] * 1000000
    code = observation_code(binding, host, phase)
    check()
    name = 'power-observe-' + phase
    receipt = run(name, SSH + ['/usr/bin/sudo -n /usr/bin/python3 -I -B -S -'], 15, stdin=code.encode(), output_limit=32768)
    assert receipt['exit_code'] == 0 and receipt['child_reaped'] and receipt['owned_group_absent']
    record = json.loads((ROOT / (name + '.stdout')).read_bytes())
    assert record['boot_id'] == binding['boot_id'] and record['owner_journal_absent'] is True and record['descriptors_closed'] is True
    assert record['power_microwatts'] == {bdf: cap for bdf in host['devices']}, 'observed caps must match the authenticated host configuration'
    if phase == 'pre':
        assert not any(record['campaign_roots_present'].values()), 'fresh campaign roots required'
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1:])
