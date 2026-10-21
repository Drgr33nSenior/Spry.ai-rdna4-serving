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
LIBRARY_BYTES = 28275305
LIBRARY_SHA256 = '6d1dbee8cb3ad924e4c1bca7734e22f2c93026eb3bc871a7f64a916778d55e36'
HEADER_SHA256 = 'e68c693da4838dc6e616e029583a6a812c7a8030246d3608dcd49812a9e4a836'


def need(value, message):
    if not value: raise RuntimeError(message)


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
    need(torch.__version__ == '2.13.0+rocm10.1.0' and torch.version.hip == '7.16.26385', 'wrong physical Torch/HIP tuple')
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
