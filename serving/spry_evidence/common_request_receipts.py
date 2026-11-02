"""Bounded HTTP request and group receipts; preserves each original byte field.

This changes the private receipt layout, not the protected exporter. Each block
still has <=256 logical receipt fields. No expanded files are written by readers.
"""
import base64
import json
import os
from pathlib import Path
import re
import stat
import threading
import zlib
import mtp_contract

MEMBER_LIMIT=4<<20
TREE_LIMIT=24<<20
FILE_LIMIT=512
LOGICAL_LIMIT=4096  # W5 v2: 7 receipt fields per request
CONTROL_FIELD_LIMIT=512
CHUNK_REQUESTS=3
CHUNK_FIELD_LIMIT=CHUNK_REQUESTS*16
CHUNK_MAGIC=b'SPRY-HTTP-RECEIPTS-V3\n'
SUFFIXES=('raw','intent.json','send-entered.json','send-returned.json','headers.json',
          'partial.raw','record.json','failure.json','stream.json','history.json','prepare-failure.json')
LOCK=threading.Lock()


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
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, ValueError)


need = _SPRY_COMMON.need


canonical = mtp_contract.json_line
sha = mtp_contract.digest_bytes


def is_receipt(path):
    path=Path(path)
    return path.parent.name in ('requests','responses') or (
        path.parent.name=='controls' and re.fullmatch(r'group-[A-Za-z0-9_-]+\.json',path.name) is not None)


def mapping(path,*,writing=False):
    path=Path(path)
    need(path.is_absolute() and is_receipt(path),'explicit request/response receipt')
    root=path.parent.parent
    need(root.name in ('paired','context','lifecycle'),'explicit composition block')
    if path.parent.name=='controls':
        return root,root/'receipts'/'control-groups.json','controls/'+path.name
    match=re.fullmatch(r'([0-9]{2,3}-[A-Za-z0-9_-]+)\.(.+)',path.name)
    need(match is not None and match[2] in SUFFIXES,'bounded receipt name/suffix')
    legacy=root/'receipts'/(match[1]+'.json')
    ordinal=int(match[1].split('-',1)[0])
    chunk=root/'receipts'/('chunk-%03d.json.zlib'%(ordinal//CHUNK_REQUESTS))
    need(not (legacy.exists() and chunk.exists()),'mixed legacy/chunk receipt ownership')
    need(not writing or not legacy.exists(),'new chunks cannot append to legacy receipts')
    return root,chunk if writing or not legacy.exists() else legacy,path.parent.name+'/'+path.name


def regular(path,maximum=MEMBER_LIMIT):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path,'canonical receipt member')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=maximum,'regular bounded receipt')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(maximum+1)
        after,named=os.fstat(fd),path.lstat()
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink')
        need(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k)==getattr(named,k) for k in fields),
             'receipt changed during read')
        return raw
    finally:os.close(fd)


def decode(raw):
    if raw.startswith(CHUNK_MAGIC):
        decoder=zlib.decompressobj()
        raw=decoder.decompress(raw[len(CHUNK_MAGIC):],MEMBER_LIMIT+1)
        need(len(raw)<=MEMBER_LIMIT and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
             'bounded complete compressed receipt envelope')
    value=json.loads(raw)
    need(type(value) is dict and canonical(value)==raw and set(value)=={'schema','kind','fields'} and
         type(value['schema']) is int and value['schema'] in (1,2,3) and type(value['fields']) is dict,
         'exact canonical receipt envelope')
    control=value['kind']=='native-http-control-receipts-v2'
    need((value['schema']==1 and value['kind']=='native-http-request-receipts-v1') or
         (value['schema']==2 and value['kind'] in ('native-http-request-receipts-v2','native-http-control-receipts-v2')) or
         (value['schema']==3 and value['kind']=='native-http-chunk-receipts-v3'),
         'exact canonical receipt envelope')
    need(1<=len(value['fields'])<=(CONTROL_FIELD_LIMIT if control else CHUNK_FIELD_LIMIT if value['schema']==3 else 12),'bounded receipt fields')
    fields={};decoded_bytes=0
    for name,row in value['fields'].items():
        expected={'bytes','sha256','base64'}|({'codec'} if value['schema'] in (2,3) else set())
        need(type(row) is dict and set(row)==expected and type(row['bytes']) is int and 0<=row['bytes']<=MEMBER_LIMIT,
             'receipt field metadata')
        need(type(name) is str and (re.fullmatch(r'controls/group-[A-Za-z0-9_-]+\.json',name) is not None if control
             else name.startswith(('requests/','responses/'))),'receipt field ownership')
        decoded_bytes+=row['bytes'];need(decoded_bytes<=MEMBER_LIMIT,'bounded decoded receipt member')
        data=base64.b64decode(row['base64'],validate=True)
        codec=row.get('codec','base64')
        need(codec in ('base64','zlib+base64'),'receipt field codec')
        if codec=='zlib+base64':
            decoder=zlib.decompressobj()
            data=decoder.decompress(data,row['bytes']+1)
            need(len(data)==row['bytes'] and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                 'bounded complete receipt compression')
        need(len(data)==row['bytes'] and sha(data)==row['sha256'],'receipt field bytes/hash')
        fields[name]=data
    return value,fields


def read(path,maximum=MEMBER_LIMIT):
    _,bundle,name=mapping(path)
    need(not Path(path).exists(),'mixed raw/enveloped receipt')
    _,fields=decode(regular(bundle))
    need(name in fields and len(fields[name])<=maximum,'bounded required receipt field')
    return fields[name]


def logical_names(directory):
    directory=Path(directory)
    paths=list(directory.rglob('*'))
    need(len(paths)<=FILE_LIMIT+16 and all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths),
         'unchanged physical receipt inventory cap')
    names=set();logical_bytes=0
    for path in paths:
        if not path.is_file():continue
        if path.parent.name=='receipts':
            _,fields=decode(regular(path))
            for name,data in fields.items():
                _,expected,valid=mapping(directory/name)
                need(expected==path and valid==name and name not in names,'receipt envelope member ownership')
                names.add(name);logical_bytes+=len(data)
        else:
            name=path.relative_to(directory).as_posix()
            need(not name.startswith(('requests/','responses/')) and name not in names,'unbound raw receipt')
            names.add(name);logical_bytes+=path.stat().st_size
    need(len(names)<=LOGICAL_LIMIT and logical_bytes<=TREE_LIMIT,'unchanged per-block decoded receipt bounds')
    return names


def write(path,raw):
    need(type(raw) is bytes and len(raw)<=MEMBER_LIMIT,'bounded raw receipt field')
    root,bundle,name=mapping(path,writing=True)
    with LOCK:
        root.mkdir(parents=True,exist_ok=True)
        need(root.resolve()==root and not root.is_symlink(),'owned canonical receipt block')
        bundle.parent.mkdir(exist_ok=True)
        fields={}
        if bundle.exists():_,fields=decode(regular(bundle))
        need(name not in fields and not Path(path).exists(),'write-once receipt field')
        fields[name]=raw
        control=path.parent.name=='controls'
        need(len(fields)<=(CONTROL_FIELD_LIMIT if control else CHUNK_FIELD_LIMIT) and sum(map(len,fields.values()))<=MEMBER_LIMIT,
             'bounded decoded receipt member')
        value=dict(schema=2 if control else 3,kind='native-http-control-receipts-v2' if control else 'native-http-chunk-receipts-v3',fields={})
        for field,data in fields.items():
            compressed=zlib.compress(data)
            packed,codec=(compressed,'zlib+base64') if len(compressed)<len(data) else (data,'base64')
            value['fields'][field]=dict(bytes=len(data),sha256=sha(data),codec=codec,base64=base64.b64encode(packed).decode('ascii'))
        encoded=canonical(value);need(len(encoded)<=MEMBER_LIMIT,'unchanged physical member cap')
        if not control:encoded=CHUNK_MAGIC+zlib.compress(encoded,9)
        need(len(encoded)<=MEMBER_LIMIT,'unchanged physical member cap')
        # All three blocks share the original 24MiB/256-file client partition.
        client=root.parent;paths=list(client.rglob('*'))
        need(len(paths)<=FILE_LIMIT+16 and all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths),
             'unchanged global client inventory')
        files=[p for p in paths if p.is_file()]
        need(len(files)+1<=FILE_LIMIT and sum(p.stat().st_size for p in files)+len(encoded)<=TREE_LIMIT,
             'unchanged global client live bytes/files including atomic temporary')
        temporary=bundle.with_suffix('.next')
        with temporary.open('xb') as stream:
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,bundle)
    return bundle
