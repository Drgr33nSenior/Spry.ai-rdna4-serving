"""Execute one registered configured owner packet through retained finite commands."""
import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
HELPERS=('benchmark_host_transport.py','host_process_owner.py','benchmark_release_observer.py',
         'collect_benchmark_platform.py','benchmark_artifacts.py')


def need(value,reason):
    if not value:raise ValueError('configured caller: '+reason)


def held(path,pin=None,maximum=32<<20):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path,'canonical source required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<=before.st_size<=maximum,'bounded regular source required')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(maximum+1)
        stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        need(len(raw)==before.st_size and stamp(before)==stamp(os.fstat(fd))==stamp(path.lstat()),'source changed during read')
        need(pin is None or hashlib.sha256(raw).hexdigest()==pin,'source hash differs: '+path.name)
        return raw
    finally:os.close(fd)


def strict(raw):
    def pairs(items):
        result={}
        for name,value in items:need(name not in result,'duplicate JSON key');result[name]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def checked_module(root,name,pin):
    path=root/(name+'.py');raw=held(path,pin,1<<20)
    class Loader(SourceFileLoader):
        def get_code(self,fullname):return self.source_to_code(raw,str(path))
    spec=importlib.util.spec_from_loader(name,Loader(name,str(path)))
    value=importlib.util.module_from_spec(spec)
    previous=sys.modules.get(name);sys.modules[name]=value
    try:spec.loader.exec_module(value)
    except BaseException:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous
        raise
    return value


def validate_context(root,authorization_sha256):
    need(type(authorization_sha256) is str and re.fullmatch('[0-9a-f]{64}',authorization_sha256),'independent authorization SHA256 required')
    root=Path(root)
    auth=strict(held(root/'RUN-AUTHORIZATION.json',authorization_sha256,1<<20))
    need(str(auth.get('status','')).startswith('REGISTERED_') and auth.get('owner_invocations_allowed')==1
         and type(auth['owner_invocations_allowed']) is int and auth.get('model_starts_allowed')==2
         and type(auth['model_starts_allowed']) is int and auth.get('no_automatic_retry') is True
         and auth.get('deployment_authorized') is auth.get('promotion_authorized') is False,
         'registered two-start one-use authority required')
    pins=auth.get('caller_inventory')
    need(type(pins) is dict and 'run_native.py' in pins and
         all(re.fullmatch(r'[a-z_]+\.py',name) and re.fullmatch('[0-9a-f]{64}',pin) for name,pin in pins.items()),
         'exact caller code inventory required')
    need({p.name for p in root.glob('*.py')}==set(pins),'caller source membership differs')
    for name,pin in pins.items():held(root/name,pin,1<<20)
    source=Path(auth['source_root'])
    template=strict(held(source/'LAUNCHER-BINDINGS-TEMPLATE.json',auth['binding_template_sha256'],64<<10))
    binding=strict(held(root/'LAUNCHER-BINDINGS.json',auth['binding_sha256'],64<<10))
    boot=auth.get('boot_id')
    need(type(boot) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',boot)
         and 'boot_id' not in template and binding==dict(template,boot_id=boot),'approval-only boot binding differs')
    for name in ('run_config_sha256','host_sha256','protected_set_sha256','manifest_sha256','archive_sha256','nonce'):
        need(auth.get(name)==binding.get(name),'approval/source binding differs: '+name)
    manifest=strict(held(source/'package-v2/MANIFEST.json',binding['manifest_sha256'],1<<20))
    need((manifest.get('schema')==3 and 'boot_id' not in manifest) or
         (manifest.get('schema')==2 and manifest.get('boot_id')==boot),'source archive/approval boot differs')
    need(auth.get('owner_seconds')==binding['total_seconds'] and type(auth['owner_seconds']) is int
         and auth.get('caller_seconds')==binding['caller_seconds'] and type(auth['caller_seconds']) is int,
         'configured owner/caller duration differs')
    need(type(binding['nonce']) is str and re.fullmatch('[0-9a-f]{16}',binding['nonce']) and
         binding['target_root']=='/home/developer/fp8-kv-benchmark.'+binding['nonce'] and
         binding['protected_root']=='/var/tmp/fp8-benchmark-source.'+binding['nonce'] and
         binding['remote_upload_root']=='/home/developer/fp8-benchmark-upload.'+binding['nonce'],'fresh configured paths differ')
    inventory=binding['upload_inventory'];expected={'payload.tar','RUN-ON-WORKSTATION.sh'}|{'review/'+name for name in HELPERS}
    if 'bootstrap_common_sha256' in binding:
        pin=binding['bootstrap_common_sha256']
        need(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin),'bootstrap common digest required')
        name='review/spry_common_canon.py'
        protected=strict(held(source/'package-v2/PROTECTED-SHA256.json',binding['protected_set_sha256'],64<<10))
        need(manifest.get('package_files',{}).get(name)==protected.get(name)==inventory.get(name)==pin,
             'bootstrap common manifest/inventory binding differs')
        expected.add(name)
    need(set(inventory)==expected,'exact retained upload subset required')
    upload=source/'owner-upload'
    need({p.relative_to(upload).as_posix() for p in upload.rglob('*') if not p.is_dir()}==expected,'upload source membership differs')
    for name,pin in inventory.items():held(upload/name,pin)
    need(inventory['payload.tar']==binding['archive_sha256'],'archive identity differs')
    return auth,binding,source


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('upload','launch'))
    parser.add_argument('--authorization-sha256',required=True)
    args=parser.parse_args(argv)
    need(sys.flags.optimize==0,'unoptimized source verifier required')
    auth,binding,source=validate_context(ROOT,args.authorization_sha256)
    pins=auth['caller_inventory']
    sealer=checked_module(ROOT,'seal_evidence',pins['seal_evidence.py'])
    need(sealer.verify(source,auth['source_seal']['manifest_sha256'])==auth['source_seal'],'complete sealed author differs')
    connection=checked_module(ROOT,'connection',pins['connection.py'])
    commands=checked_module(ROOT,'local_commands',pins['local_commands.py'])
    connection.check()
    if args.phase=='upload':
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w') as archive:
            for name,pin in sorted(binding['upload_inventory'].items()):
                raw=held(source/'owner-upload'/name,pin);info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444
                archive.addfile(info,io.BytesIO(raw))
        raw=payload.getvalue();need(len(raw)<=32<<20,'retained upload allowance exceeded')
        with (ROOT/'upload.tar').open('xb') as output:output.write(raw)
        remote=binding['remote_upload_root']
        command='set -eu\numask 077\n/usr/bin/mkdir -m 0700 '+shlex.quote(remote)+'\n/usr/bin/tar -C '+shlex.quote(remote)+' -xf -\n'
        receipt=commands.run('upload-once',connection.SSH+[command],60,stdin=raw)
    else:
        uploaded=strict(held(ROOT/'upload-once.json',maximum=1<<20))
        need(uploaded['exit_code']==0 and uploaded['child_reaped'] is uploaded['owned_group_absent'] is True,
             'completed reaped upload required before launch')
        need(uploaded['stdin_sha256']==hashlib.sha256(held(ROOT/'upload.tar')).hexdigest(),'uploaded input differs')
        with (ROOT/'LAUNCH-INVOKED.json').open('x') as output:
            json.dump(dict(authorization_sha256=args.authorization_sha256,maximum_invocations=1),output,sort_keys=True)
        command=shlex.join(['bash',binding['remote_upload_root']+'/RUN-ON-WORKSTATION.sh',binding['boot_id']])
        receipt=commands.run('native-launch',connection.SSH+[command],binding['total_seconds']+80)
    print(json.dumps({key:receipt[key] for key in ('exit_code','elapsed_seconds','child_reaped','owned_group_absent')},sort_keys=True))
    return 0 if (receipt['exit_code']==0 and receipt['child_reaped'] is receipt['owned_group_absent'] is True
                 and receipt.get('failure') is None) else 1


if __name__=='__main__':raise SystemExit(main())
