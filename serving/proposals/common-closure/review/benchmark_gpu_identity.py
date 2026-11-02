"""Future physical identity gate, using the actual pinned HIP declaration.

No library is loaded or device queried when this module is imported.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import stat
import time

LIBRARY = Path('/usr/local/lib/python3.14/dist-packages/_rocm_sdk_core/lib/libamdhip64.so.7')
LIBRARY_BYTES = 6636089
LIBRARY_SHA256 = '32c8b5104d0fead0e09a9c18bb135f5b4bbcc70de59e0d47ce4c9d7d852aaf34'
HEADER_SHA256 = '06d7c70c870d4306087291d5559f3077a78719863470e6348e309fc62028bfed'


def _load_spry_common(caller, pin, error):
    """Authenticate common source before any shared guard can execute."""
    import hashlib
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import os
    from pathlib import Path
    import stat

    def check(value, message):
        if not value:
            raise error('shared common source: ' + message)

    origin = Path(caller)
    path = origin.with_name('spry_common_canon.py')
    if origin.name.startswith('common_') and origin.parent.name in ('spry_serving', 'spry_evidence'):
        path = origin.resolve().parents[1] / 'spry_common/selected.py'
    elif (origin.parent.name == 'review' and origin.parent.parent.name == 'common-closure'
          and origin.parent.parent.parent.name == 'proposals'):
        path = origin.resolve().parents[3] / 'spry_common/selected.py'
    check(path.is_absolute() and path.resolve(strict=True) == path, 'canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 4096, 'bounded regular file required')
        raw = os.read(descriptor, 4097)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        check(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(descriptor))
              == stamp(path.lstat()), 'file changed during read')
    finally:
        os.close(descriptor)
    check(hashlib.sha256(raw).hexdigest() == pin, 'digest differs')
    # Use the checked bytes, ignoring ambient modules and cached bytecode.
    loader = SourceFileLoader('_spry_common', str(path))
    loader.get_code = lambda name: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader('_spry_common', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_SPRY_COMMON_SHA256 = 'c2cb7702d6ae482b0f9d70b6b838af3d0b9eb312140ff09d001df0880a9e6bed'
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, RuntimeError)


def need(value, message):
    return _SPRY_COMMON.need(value, message, error=RuntimeError, prefix='')


def canonical_bdf(raw):
    need(type(raw) is bytes and re.fullmatch(rb'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', raw),
         'HIP returned malformed PCI identity')
    return raw.decode('ascii').lower()


def validate_devices(records, expected):
    need(len(records) == 2 and [row['index'] for row in records] == [0, 1] and
         all(type(row['index']) is int for row in records) and
         len({row['bdf'] for row in records}) == 2 and
         {row['bdf']: row['uuid'] for row in records} == expected and
         all(row['architecture'].split(':')[0] == 'gfx1201' and type(row['vram_bytes']) is int and
             20 * 1024 ** 3 <= row['vram_bytes'] < 35 * 1024 ** 3 for row in records),
         'wrong target allocation or stock capacity-default branch')


def validate_proof(proof):
    need(proof['library'] == str(LIBRARY) and proof['sha256'] == LIBRARY_SHA256 and
         proof['bytes'] == LIBRARY_BYTES and proof['no_load_handle'] is True and
         proof['symbol'] == 'hipDeviceGetPCIBusId' and proof['header_sha256'] == HEADER_SHA256 and
         type(proof['loaded_inode']) is int and proof['loaded_inode'] > 0 and
         type(proof['loaded_device']) is int and proof['loaded_device'] > 0, 'actual loaded HIP identity proof differs')


def file_identity(deadline):
    info = LIBRARY.lstat()
    need(LIBRARY.resolve() == LIBRARY and stat.S_ISREG(info.st_mode) and info.st_nlink >= 1 and
         info.st_size == LIBRARY_BYTES and os.statvfs(LIBRARY).f_flag & os.ST_RDONLY,
         'exact immutable stock HIP library required')
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    stable = lambda value: tuple(getattr(value, key) for key in fields)
    digest = hashlib.sha256()
    with LIBRARY.open('rb') as stream:
        need(stable(info) == stable(os.fstat(stream.fileno())), 'HIP open identity changed')
        while True:
            need(time.monotonic() < deadline, 'HIP identity deadline')
            raw = stream.read(1024 ** 2)
            if not raw: break
            digest.update(raw)
        need(stable(info) == stable(os.fstat(stream.fileno())) == stable(LIBRARY.lstat()), 'HIP file changed')
    need(digest.hexdigest() == LIBRARY_SHA256, 'HIP library hash differs')
    return info


def mapped_identity(info):
    raw = Path('/proc/self/maps').read_text()
    need(len(raw) <= 8 * 1024 ** 2, 'bounded loaded-library mappings')
    rows = []
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or 'libamdhip64.so' not in Path(fields[5]).name: continue
        major, minor = (int(part, 16) for part in fields[3].split(':'))
        need(not fields[5].endswith(' (deleted)') and Path(fields[5]).resolve() == LIBRARY and
             int(fields[4]) == info.st_ino and os.makedev(major, minor) == info.st_dev,
             'loaded HIP path/inode/device differs')
        rows.append(fields[5])
    need(rows, 'Torch did not load the pinned HIP library')


def physical_devices(torch, expected, deadline):
    need(torch.__version__ == '2.14.0+git2e66fe81.rocm10.2.0' and torch.version.hip == '7.17.26416', 'wrong physical Torch/HIP tuple')
    info = file_identity(deadline); mapped_identity(info)
    library = ctypes.CDLL(str(LIBRARY), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    function = library.hipDeviceGetPCIBusId
    function.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    class DlInfo(ctypes.Structure):
        _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    lookup = ctypes.CDLL(None).dladdr
    lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]; lookup.restype = ctypes.c_int
    origin = DlInfo()
    need(lookup(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(origin)) != 0 and origin.filename and
         Path(os.fsdecode(origin.filename)).resolve() == LIBRARY, 'HIP PCI function origin differs')
    need(torch.cuda.device_count() == 2, 'exact two-device allocation required')
    records = []
    for index in range(2):
        need(time.monotonic() < deadline, 'physical identity deadline')
        output = ctypes.create_string_buffer(32)
        need(function(output, len(output), index) == 0, 'HIP PCI identity query failed')
        props = torch.cuda.get_device_properties(index)
        records.append(dict(index=index, bdf=canonical_bdf(output.value), uuid=str(props.uuid),
                            architecture=props.gcnArchName, vram_bytes=props.total_memory))
    validate_devices(records, expected)
    mapped_identity(info)
    return records, dict(library=str(LIBRARY), sha256=LIBRARY_SHA256, bytes=LIBRARY_BYTES,
                         loaded_inode=info.st_ino, loaded_device=info.st_dev, no_load_handle=True,
                         symbol='hipDeviceGetPCIBusId', header_sha256=HEADER_SHA256)

import json,zipfile
from pathlib import PurePosixPath

canonical = _SPRY_COMMON.canonical_line

sha = _SPRY_COMMON.digest

def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key');result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def binding(value, *, allow_empty=False):
    need(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'} and
         type(value['path']) is str and PurePosixPath(value['path']).is_absolute() and
         str(PurePosixPath(value['path'])) == value['path'] and '..' not in PurePosixPath(value['path']).parts and
         type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']) and
         type(value['bytes']) is int and value['bytes'] >= (0 if allow_empty else 1) and
         (value['bytes'] != 0 or value['sha256'] == sha(b'')), 'file binding schema')
    return value

def validate_candidate(value):
    fields = {'root', 'wheel', 'version', 'metadata_directory', 'content_inventory_sha256', 'sources', 'source_commit'}
    need(type(value) is dict and set(value) in (fields, fields | {'hip_version', 'sdk'}), 'candidate fields')
    sdk102 = value['version'] == '2.14.0+git2e66fe81.rocm10.2.0'
    need((sdk102 and set(value) == fields | {'hip_version', 'sdk'} or
         not sdk102 and set(value) == fields and value['version'] == '2.14.0+git2e66fe81.rocm10.1.0') and
         value['source_commit'] == '2e66fe8153890424e76464545bc5de6391bf9d8c' and
         type(value['root']) is str and re.fullmatch(r'/home/developer/pytorch-214-sdk' + ('102' if sdk102 else '101') + r'\.[0-9a-f]{16}/candidate', value['root']) and
         value['metadata_directory'] == 'torch-' + value['version'] + '.dist-info' and
         type(value['content_inventory_sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['content_inventory_sha256']) and
         type(value['sources']) is dict and 2 <= len(value['sources']) <= 10000, 'candidate Torch wheel identity')
    wheel = binding(value['wheel'])
    need(str(PurePosixPath(wheel['path']).parent) == str(PurePosixPath(value['root']).parent / 'artifact') and
         wheel['path'].endswith('.whl') and wheel['bytes'] <= 8 << 30, 'candidate wheel path/size')
    for relative, row in value['sources'].items():
        need(type(relative) is str and relative.startswith('torch/') and
             str(PurePosixPath(relative)) == relative and '..' not in PurePosixPath(relative).parts and
             (relative.endswith('.py') or re.search(r'\.so(?:\.\d+)*$', relative)) and
             type(row) is dict and set(row) == {'sha256', 'bytes'}, 'candidate Torch source row')
        binding(dict(path='/usr/local/lib/python3.14/dist-packages/' + relative, **row), allow_empty=True)
    if sdk102:
        need(type(value['hip_version']) is str and re.fullmatch(r'7\.17\.\d+(?:[-+].*)?', value['hip_version']), 'candidate HIP development version')
        sdk = value['sdk'];parent = PurePosixPath(value['root']).parent
        need(type(sdk) is dict and set(sdk) == {'root', 'manifest', 'version'} and
             sdk['root'] == str(parent / 'runtime-sitepackages') and
             sdk['version'] == '10.2.0.dev20261020+git535d2990', 'source SDK projection identity')
        record = binding(sdk['manifest'])
        need(record['path'] == str(parent / 'SDK-RUNTIME-MANIFEST.json') and record['bytes'] <= 16 << 20,
             'source SDK inventory binding')
    return value

def candidate_inventory(value):
    """Reconstruct all immutable candidate file identities from its pinned wheel."""
    validate_candidate(value);path = Path(value['wheel']['path'])
    need(path.resolve(strict=True) == path and path.stat().st_nlink == 1, 'candidate wheel canonical file')
    with path.open('rb') as stream:
        need(path.stat().st_size == value['wheel']['bytes'] and
             hashlib.file_digest(stream, 'sha256').hexdigest() == value['wheel']['sha256'], 'candidate wheel hash')
    rows = {};total = 0
    with zipfile.ZipFile(path) as archive:
        need(len(archive.infolist()) <= 32768, 'candidate wheel member bound')
        for item in archive.infolist():
            if item.is_dir(): continue
            relative = PurePosixPath(item.filename)
            need(not relative.is_absolute() and str(relative) == item.filename and '..' not in relative.parts and
                 item.filename not in rows and not stat.S_ISLNK(item.external_attr >> 16), 'candidate wheel member path/type')
            total += item.file_size;need(total <= 12 << 30, 'candidate unpacked size bound')
            with archive.open(item) as stream: pin = hashlib.file_digest(stream, 'sha256').hexdigest()
            rows[item.filename] = dict(sha256=pin, bytes=item.file_size)
    need(sha(canonical(rows)) == value['content_inventory_sha256'], 'candidate content inventory hash')
    need({name: row for name, row in rows.items() if name.startswith('torch/') and
         (name.endswith('.py') or re.search(r'\.so(?:\.\d+)*$', name))} == value['sources'], 'candidate source inventory join')
    return rows

def checked_candidate_tree(value):
    rows = candidate_inventory(value);root = Path(value['root'])
    need(root.resolve(strict=True) == root and not root.is_symlink(), 'candidate canonical directory')
    seen = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'candidate tree symlink')
        if path.is_dir(): continue
        name = path.relative_to(root).as_posix();need(name in rows, 'extra candidate file')
        before = path.stat();need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
            before.st_size == rows[name]['bytes'], 'candidate file type/size')
        with path.open('rb') as stream: pin = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        need(all(getattr(before, key) == getattr(after, key) for key in
            ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink')) and
            pin == rows[name]['sha256'], 'candidate file hash/stability')
        seen.add(name)
    need(seen == set(rows), 'missing candidate file')
    return rows

def sdk_inventory(value, torch_rows=None):
    """Read the pinned SDK/runtime inventory and join it to the candidate wheel."""
    validate_candidate(value);sdk = value['sdk']
    data = strict_json(checked(sdk['manifest']))
    need(type(data) is dict and set(data) == {'schema', 'root', 'version', 'torch_wheel_sha256', 'identity', 'files'} and
         type(data['schema']) is int and data['schema'] == 1 and data['root'] == sdk['root'] and
         data['version'] == sdk['version'] and data['torch_wheel_sha256'] == value['wheel']['sha256'], 'SDK inventory identity')
    rows = data['files'];need(type(rows) is dict and 1 <= len(rows) <= 32768, 'bounded SDK runtime inventory')
    total = 0
    for name, row in rows.items():
        relative = PurePosixPath(name)
        need(type(name) is str and not relative.is_absolute() and str(relative) == name and '..' not in relative.parts and
             type(row) is dict and set(row) == {'sha256', 'bytes'}, 'SDK runtime file row')
        binding(dict(path=sdk['root'] + '/' + name, **row), allow_empty=True)
        total += row['bytes'];need(total <= 64 << 30, 'SDK runtime byte bound')
    if torch_rows is None: torch_rows = candidate_inventory(value)
    need(all(rows.get(name) == row for name, row in torch_rows.items()), 'merged runtime Torch wheel join')
    identity = data['identity']
    need(type(identity) is dict and set(identity) == {'therock_commit', 'hip_version', 'hip_library', 'hip_header_sha256',
         'rccl_version', 'rccl_library'} and identity['therock_commit'] == '535d29907838d00afdb1382b700a0b1223ca942f' and
         identity['hip_version'] == value['hip_version'] and identity['rccl_version'] == '2.32.3' and
         type(identity['hip_header_sha256']) is str and re.fullmatch('[0-9a-f]{64}', identity['hip_header_sha256']), 'SDK component identities')
    prefix = '/usr/local/lib/python3.14/dist-packages/'
    for key in ('hip_library', 'rccl_library'):
        record = binding(identity[key]);need(record['path'].startswith(prefix), 'SDK library virtual path')
        need(rows.get(record['path'][len(prefix):]) == {k: record[k] for k in ('sha256', 'bytes')}, 'SDK library content join')
    return rows

def checked_sdk_tree(value, torch_rows=None):
    rows = sdk_inventory(value, torch_rows);root = Path(value['sdk']['root'])
    need(root.resolve(strict=True) == root and not root.is_symlink(), 'SDK canonical runtime directory')
    seen = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'SDK runtime symlink')
        if path.is_dir(): continue
        name = path.relative_to(root).as_posix();need(name in rows, 'extra SDK runtime file')
        before = path.stat();need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
             before.st_size == rows[name]['bytes'], 'SDK runtime file type/size')
        with path.open('rb') as stream: pin = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        need(all(getattr(before, key) == getattr(after, key) for key in
             ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink')) and
             pin == rows[name]['sha256'], 'SDK runtime file hash/stability')
        seen.add(name)
    need(seen == set(rows), 'missing SDK runtime file')
    return rows

def checked(source, maximum=16 << 20, *, allow_empty=False):
    source = binding(source, allow_empty=allow_empty);path = Path(source['path'])
    need(path.resolve(strict=True) == path and not path.is_symlink(), 'canonical input file')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_size == source['bytes'] <= maximum and before.st_nlink == 1, 'input size/link bound')
        digest = hashlib.sha256();parts = []
        while True:
            raw = os.read(descriptor, 1 << 20)
            if not raw: break
            digest.update(raw);parts.append(raw)
        after, named = os.fstat(descriptor), path.lstat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')
        need(all(getattr(before, key) == getattr(after, key) == getattr(named, key) for key in fields) and
             digest.hexdigest() == source['sha256'], 'input hash/stability: ' + path.name)
        return b''.join(parts)
    finally: os.close(descriptor)
