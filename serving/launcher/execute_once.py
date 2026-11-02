"""Run one registered production campaign; retain failures and always park afterward."""
import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def need(value,reason):
    if not value:raise ValueError('one-use caller: '+reason)


def dump(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


def load_caller(pin):
    raw=(ROOT/'RUN-AUTHORIZATION.json').read_bytes()
    need(hashlib.sha256(raw).hexdigest()==pin,'independent authorization identity differs')
    auth=json.loads(raw);path=ROOT/'run_native.py';source=path.read_bytes()
    need(hashlib.sha256(source).hexdigest()==auth['caller_inventory']['run_native.py'],'caller bootstrap source differs')
    class Loader(SourceFileLoader):
        def get_code(self,fullname):return self.source_to_code(source,str(path))
    spec=importlib.util.spec_from_loader('configured_native_caller',Loader('configured_native_caller',str(path)))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def validate_approval(caller,root,grant,source):
    registration=caller.strict(caller.held(root/'REGISTRATION.json',grant['registration_sha256'],1<<20))
    need({k:v for k,v in grant.items() if k!='registration_sha256'}==registration,'registered operation differs')
    need(type(grant.get('global_gpu_attempt_index')) is int and grant['global_gpu_attempt_index']>0
         and grant.get('registered') is True and grant.get('proposal') is False and grant.get('gpu_allowance') is True,
         'registered global operation required')
    record=grant['approval_record'];approval=caller.strict(caller.held(Path(record['path']),record['sha256'],1<<20))
    need(type(grant.get('owner_answer')) is str and grant['owner_answer'].strip()
         and approval['owner_answer']==grant['owner_answer'] and grant['registration_authority']==record
         and approval['approval_request_sha256']==grant['approval_request_sha256']
         and approval['source_seal']==grant['source_seal'],'exact owner approval record required')
    request=caller.strict(caller.held(root/'APPROVAL-REQUEST.json',grant['approval_request_sha256'],1<<20))
    for key in ('nonce','run_config_sha256','host_sha256','protected_set_sha256','manifest_sha256',
                'archive_sha256','binding_template_sha256','caller_inventory'):
        need(request[key]==grant[key],'approved source differs: '+key)
    for reviewed,registered in (('requested_owner_seconds','owner_seconds'),('requested_caller_seconds','caller_seconds'),
            ('requested_model_starts','model_starts_allowed'),('requested_generations','generations'),
            ('requested_cpu_grading_seconds','cpu_grading_seconds'),('observed_boot_id','boot_id')):
        need(type(request[reviewed]) is type(grant[registered]) and request[reviewed]==grant[registered],
             'approved work/boot differs: '+reviewed)
    caller.held(source/'PROTECTED-COMPLETE.patch',request['protected_patch_sha256'],4<<20)
    need(caller.strict(caller.held(source/'CALLER-SOURCES.json',maximum=64<<10))==grant['caller_inventory'],
         'sealed caller source inventory differs')
    return request


def run(grant,root,runner,sampler,*,authorization_sha256,clock=time.monotonic):
    """Preserve the retained sequence, including collection after launch failure."""
    steps=[];start=clock();deadline=start+grant['caller_seconds'];launched=False;success=False;stop=None
    def step(name,argv,seconds):
        try:row=runner(name,argv,seconds)
        except BaseException as error:
            row=dict(name=name,exit_code=None,failure=type(error).__name__+': '+str(error)[:256])
        steps.append(row);return row.get('exit_code')==0 and not row.get('failure')
    if not step('power-pre',['power_observe.py','pre'],60):status='STOPPED_BEFORE_LAUNCH_PRE_CHECK'
    elif not step('upload',['run_native.py','upload','--authorization-sha256',authorization_sha256],90):status='STOPPED_BEFORE_LAUNCH_UPLOAD'
    elif deadline-clock()<grant['owner_seconds']+100+520:status='STOPPED_BEFORE_LAUNCH_CALLER_RESERVE'
    else:
        try:
            stop=sampler()
            launched=step('launch',['run_native.py','launch','--authorization-sha256',authorization_sha256],grant['owner_seconds']+100)
        except BaseException as error:
            steps.append(dict(name='sampler-start',exit_code=None,failure=type(error).__name__))
        finally:
            if stop is not None:
                try:steps.append(stop())
                except BaseException as error:steps.append(dict(name='sampler-stop',exit_code=None,failure=type(error).__name__))
        collected=step('collect',['collect_results.py'],160)
        try:
            path=root/'collected/protected'/('fp8-benchmark-source.'+grant['nonce']+'.closeout.json')
            success=json.loads(path.read_bytes())['physical_success'] is True if collected else False
        except (OSError,ValueError,KeyError):success=False
        observer='observe_closeout.py' if success or launched and not collected else 'observe_failed_closeout.py'
        step('closeout',[observer],45)
        step('power-post',['power_observe.py','post'],35)
        step('host-faults',['host_faults.py'],45)
        status='COMPLETE' if launched and success and all(s.get('exit_code')==0 and not s.get('failure') for s in steps) else 'INCOMPLETE_RETAINED'
    report=dict(schema=1,status=status,steps=steps,global_operation=grant['global_gpu_attempt_index'],
        automatic_retry=False,run_root=str(root),physical_success=success,qualification=False,selection=False,
        elapsed_seconds=clock()-start)
    dump(root/'EXECUTION-SUMMARY.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--authorization-sha256',required=True)
    args=parser.parse_args(argv);need(sys.flags.optimize==0,'unoptimized source verifier required')
    caller=load_caller(args.authorization_sha256)
    grant,binding,source=caller.validate_context(ROOT,args.authorization_sha256)
    validate_approval(caller,ROOT,grant,source)
    need(not list(ROOT.rglob('*.pyc')) and not list(ROOT.rglob('__pycache__')),'caller bytecode cache prohibited')
    pins=grant['caller_inventory'];sealer=caller.checked_module(ROOT,'seal_evidence',pins['seal_evidence.py'])
    need(sealer.verify(source,grant['source_seal']['manifest_sha256'])==grant['source_seal'],'sealed author differs')
    dump(ROOT/'CALLER-INVOKED.json',dict(authorization_sha256=args.authorization_sha256,maximum_invocations=1,automatic_retry=False))
    steps=ROOT/'steps';steps.mkdir(mode=0o700)
    commands=caller.checked_module(ROOT,'local_commands',pins['local_commands.py']);commands.ROOT=steps
    bootstrap='import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    def child(argv):return [sys.executable,'-I','-B','-S','-c',bootstrap,str(ROOT),str(ROOT/argv[0]),*argv[1:]]
    def runner(name,argv,seconds):
        receipt_name='collect-native-results' if name=='collect' else name
        if name=='collect':commands.ROOT=ROOT
        try:receipt=commands.run(receipt_name,child(argv),seconds,cwd=ROOT)
        finally:commands.ROOT=steps
        return dict(name=name,exit_code=receipt['exit_code'],elapsed_seconds=receipt['elapsed_seconds'],
            failure=receipt.get('failure') or (None if receipt['child_reaped'] is receipt['owned_group_absent'] is True
                                              else 'local process cleanup incomplete'))
    def sampler():
        out=(steps/'host-sample.stdout').open('xb');err=(steps/'host-sample.stderr').open('xb')
        try:
            process=subprocess.Popen(child(['host_sample.py','run',str(grant['owner_seconds']+260)]),cwd=ROOT,
                stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        except BaseException:
            out.close();err.close();raise
        def stop():
            try:
                directory=ROOT/'host-samples'
                if directory.is_dir():(directory/'STOP').touch(exist_ok=False)
                try:code=process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    for sig in (signal.SIGTERM,signal.SIGKILL):
                        os.killpg(process.pid,sig)
                        try:process.wait(timeout=5);break
                        except subprocess.TimeoutExpired:continue
                    code=process.poll()
                return dict(name='host-sample',exit_code=code,child_reaped=process.poll() is not None)
            finally:out.close();err.close()
        return stop
    report=run(grant,ROOT,runner,sampler,authorization_sha256=args.authorization_sha256)
    seal=sealer.seal(ROOT)
    print(json.dumps(dict(status=report['status'],seal=seal['manifest_sha256']),sort_keys=True))
    return 0 if report['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
