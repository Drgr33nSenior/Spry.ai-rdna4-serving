"""Fresh CPU tokenizer entry with SDK dependencies and unchanged CPU resource checks."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent

def load(name,pin):
    path=ROOT/(name+'.py');raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('private source identity changed: '+name)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;exec(compile(raw,str(path),'exec'),module.__dict__);return module

def run(deadline,package_pin,admission_pin):
    started=time.monotonic()
    if not(sys.flags.isolated==1 and sys.flags.no_site==1 and sys.flags.dont_write_bytecode==1 and 'site'not in sys.modules):raise ValueError('isolated no-site no-bytecode entry required')
    startup_paths=list(sys.path)
    if any('site-packages'in path or 'dist-packages'in path for path in startup_paths):raise ValueError('unguarded startup dependency path')
    if not(type(deadline)in(int,float)and math.isfinite(deadline)and started<deadline<=started+210):raise ValueError('shared210second CPU work deadline required')
    if ROOT!=Path('/mnt/review')or(os.getuid(),os.geteuid(),os.getgid(),os.getegid())!=(1000,1000,1000,1000):raise ValueError('owned CPU tokenizer namespace required')
    if Path('/dev/kfd').exists()or list(Path('/dev/dri').glob('renderD*')):raise ValueError('GPU devices forbidden')
    package_raw=(ROOT/'PACKAGE.json').read_bytes()
    if hashlib.sha256(package_raw).hexdigest()!=package_pin:raise ValueError('package identity changed')
    package=json.loads(package_raw);common=load('common',package['sources']['common.py']);common.validate_package(ROOT/'PACKAGE.json',package_pin)
    witness=load('tokenizer_witness',package['sources']['tokenizer_witness.py']);core=load('tokenizer_core',package['sources']['tokenizer_core.py'])
    q=load('quality_contract',package['sources']['quality_contract.py']);load('thinking_segments',package['sources']['thinking_segments.py']);quality=load('quality_witness',package['sources']['quality_witness.py'])
    report=dict(schema=1,mode=quality.MODE,status='FINAL_DECODE_INCOMPLETE',complete=False,qualification=False,native_tokenization_performed=False,model_loaded=False,gpu_execution_performed=False,gpu_nodes_absent=True,
      package_sha256=package_pin,input_sha256=package['sources']['CPU-TOKENIZER-INPUT.json'],model_root='/mnt/tokenizer-model',cases=[],render_options=witness.RENDER_OPTIONS,encode_options=witness.ENCODE_OPTIONS,decode_options=witness.DECODE_OPTIONS,parsers=dict(reasoning=None,tool_call=None))
    failure=None;publication=None;finder=None
    try:
        common.checked_file(Path('/mnt/admission.json'),admission_pin,maximum=1<<20);admission=json.loads(Path('/mnt/admission.json').read_bytes())
        witness.need(admission['package_sha256']==package_pin and admission['work_deadline_monotonic']==deadline and admission['boot_id']==package['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'exact shared CPU admission')
        frozen=json.loads((ROOT/'CPU-TOKENIZER-INPUT.json').read_bytes());config=json.loads((ROOT/'TOKENIZER-CONFIG.json').read_bytes())
        value=q.strict((ROOT/'QUALITY-CPU-INPUT.json').read_bytes());expected_quality=q.strict((ROOT/'EXPECTED-QUALITY.json').read_bytes())
        q.validate_input(value,expected_quality)
        common.checked_file(Path('/mnt/stack-proof.json'),admission['stack_proof_sha256'],maximum=1<<20);proof=json.loads(Path('/mnt/stack-proof.json').read_bytes())
        common.checked_file(Path('/mnt/tokenizer-runtime-inventory.zlib'),admission['tokenizer_runtime_inventory_sha256'],maximum=8<<20)
        runtime=core.inflate(Path('/mnt/tokenizer-runtime-inventory.zlib').read_bytes(),8<<20)
        witness.need(runtime['sdk_manifest_sha256']==config['sdk_manifest_sha256']==proof['result']['sdk_manifest_sha256'],'SDK complete preauthentication manifest join')
        prefix_raw=(ROOT/'TOKENIZER-PREFIX-INVENTORY.zlib').read_bytes();witness.need(witness.sha(prefix_raw)==config['prefix_compressed_sha256'],'retained prefix inventory pin')
        prefix=core.inflate(prefix_raw,4<<20);witness.need(witness.sha(witness.encoded(prefix))==config['prefix_raw_sha256']and len(prefix)==config['prefix_members'],'complete retained dependency inventory')
        runtime['prefix_files']=prefix
        model=Path('/mnt/tokenizer-model');model_assets=witness.assets(frozen);witness.need({p.name for p in model.iterdir()}==set(model_assets),'only seven model/tokenizer assets mounted')
        authenticated={name:core.held(model/name,row['sha256'],32<<20)for name,row in model_assets.items()}
        witness.need(all(authenticated[name]['bytes']==row['bytes']for name,row in model_assets.items()),'exact tokenizer asset sizes')
        report['render_source_binding']=dict(path='/mnt/review/SERVING-CHAT-SOURCE.py',sha256=frozen['source_serving_chat']['sha256'],bytes=frozen['source_serving_chat']['bytes'],serving_path_executed=False)
        report.update(model_revision=frozen['model_revision'],assets=authenticated,asset_authentication_before_imports=True,admission_sha256=admission_pin,host_stack_proof_sha256=witness.sha(witness.encoded(proof)),runtime_inventory_sha256=admission['tokenizer_runtime_inventory_sha256'])
        witness.need(not any(name.split('.')[0]in {'sglang','tensorflow','jax'}for name in sys.modules),'forbidden runtime imported before tokenizer entry')
        report['forbidden_namespaces']=['jax','sglang','tensorflow']
        report['resources_before_imports']=load('cpu_resources',package['sources']['cpu_resources.py']).verify_resources(admission,load('review/benchmark_resource_gate',package['sources']['review/benchmark_resource_gate.py']))
        base=json.loads((ROOT/'TOKENIZER-BASE-INVENTORY.json').read_bytes());runtime['base_inventory']=base
        report['base_dependency_authentication']=core.authenticate_base(base,package['sources']['TOKENIZER-BASE-INVENTORY.json'],deadline)
        finder=core.DependencyFinder(runtime['sdk_files'],prefix,deadline,base=base);sys.meta_path.insert(0,finder)
        stack=json.loads((ROOT/'STACK.json').read_bytes());compat=load('compat_bootstrap',package['sources']['compat_bootstrap.py'])
        report['compatibility_authentication']=compat.authenticate(stack,common.checked_file,deadline,root=Path('/mnt/compat'))
        sys.meta_path.insert(0,compat.Finder(stack['compatibility']['files'],Path('/mnt/compat')))
        cache=Path(__import__('tempfile').mkdtemp(prefix='cpu-tokenizer-pycache-',dir='/tmp'));sys.pycache_prefix=str(cache);sys.dont_write_bytecode=True
        report['pycache_isolation']=dict(prefix=str(cache),created_exclusive=True,dont_write_bytecode=True,source_adjacent_cache_reads=False)
        sys.path[:0]=[str(core.SDK),str(core.PREFIX),str(core.BASE)]
        report['startup_isolation']=dict(isolated=sys.flags.isolated,no_site=sys.flags.no_site,dont_write_bytecode=sys.flags.dont_write_bytecode,site_absent=True,startup_paths=startup_paths,guard_installed_before_runtime_paths=True,guarded_runtime_paths=[str(core.SDK),str(core.PREFIX),str(core.BASE)],addsitedir_used=False)
        torch=__import__('importlib').import_module('torch');triton=__import__('importlib').import_module('triton')
        witness.need(str(torch.__file__)==str(core.SDK/'torch/__init__.py')and str(triton.__file__)==str(core.SDK/'triton/__init__.py'),'SDK-first actual Torch/Triton origins')
        report['runtime_dependencies']=dict(torch_version=str(torch.__version__),hip_version=str(torch.version.hip),triton_version=str(triton.__version__),torch_origin=str(torch.__file__),triton_origin=str(triton.__file__),
          modules={name:finder.module(name,sys.modules[name])for name in ('torch','torch._C','triton')})
        sys.path.insert(0,'/mnt/compat')
        core.deadline(deadline);tokenizer,identity=core.native_factory(model,finder,frozen)
        witness.publish_identity(report,identity)
        from tokenizers import Tokenizer
        native=Tokenizer.from_file('/mnt/tokenizer-model/tokenizer.json')
        report['quality_input_sha256']=package['sources']['QUALITY-CPU-INPUT.json']
        report['answers']=q.decode_rows(value,expected_quality,native)
        cls=type(tokenizer);finder.tokenizer_class=cls
        final_identity=dict(identity);final_identity['loaded_sources']=finder.loaded();final_identity['compatibility_aliases']=finder.alias_witness;final_identity['requests_same_object_aliases']=finder.requests_alias_witness;final_identity['qwen2_fast_alias']=finder.qwen2_alias_witness;final_identity['class_source']=final_identity['loaded_sources'][cls.__module__]
        witness.publish_identity(report,final_identity,final=True)
        report['compatibility_loaded']=compat.loaded_closure(stack['compatibility']['files'],common.checked_file,deadline,root=Path('/mnt/compat'),require_top=False)
        report['resources_after_decoding']=load('cpu_resources',package['sources']['cpu_resources.py']).verify_resources(admission,load('review/benchmark_resource_gate',package['sources']['review/benchmark_resource_gate.py']))
        witness.need(not any(name.split('.')[0]in {'sglang','tensorflow','jax'}for name in sys.modules),'forbidden model namespace loaded')
        report['forbidden_namespaces_absent_before_and_after']=True
        core.deadline(deadline);report.update(native_tokenization_performed=False,native_decoding_performed=True,complete=True,status='NATIVE_CPU_FINAL_DECODE_COMPLETE',elapsed_seconds=time.monotonic()-started)
        quality.validate_decoder(report,frozen=frozen,input_sha256=package['sources']['CPU-TOKENIZER-INPUT.json'],package=package,admission=admission,proof=proof,runtime_inventory=runtime,stack=stack,raw=witness.encoded(report))
    except BaseException as error:
        failure=error;report.update(complete=False,status='FINAL_DECODE_INCOMPLETE',failure=dict(error_type=type(error).__name__,reason=str(error)[:2048],notes=getattr(error,'__notes__',[])))
        if finder is not None and finder.mismatches['total']:
            report['module_mismatches']=witness.validate_module_mismatches(finder.mismatches)
            report.pop('tokenizer_identity',None)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        try:
            raw=witness.encoded(report);witness.need(len(raw)<=quality.REPORT_LIMIT,'1MiB final decoder report cap')
            directory=Path('/mnt/output/quality');directory.mkdir(mode=0o700)
            with (directory/'DECODE-RESULT.json').open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
            print('CPU_TOKENIZER_RESULT='+json.dumps(dict(status=report['status'],complete=report['complete'],qualification=False)),flush=True)
        except BaseException as error:
            publication=error
            if failure is not None:failure.add_note('Secondary tokenizer publication failure: '+type(error).__name__+': '+str(error)[:1024])
    if failure is not None:raise failure
    if publication is not None:raise publication
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deadline-monotonic',type=float,required=True);parser.add_argument('--package-sha256',required=True);parser.add_argument('--admission-sha256',required=True)
    args=parser.parse_args();run(args.deadline_monotonic,args.package_sha256,args.admission_sha256)
