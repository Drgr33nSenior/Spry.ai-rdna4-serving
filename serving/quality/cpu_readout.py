"""Authenticate native CPU output, actual transport, supervisors and release."""
import base64
import json
from pathlib import Path
import host_cpu as host
ROOT=Path(__file__).resolve().parent


def transport(directory,plan,plan_raw,prepared,core):
    host.need(not (ROOT/'TRANSPORT-FAILURE.json').exists(),'failed transport cannot publish completion')
    receipt_raw=host.held(ROOT/'cpu-import-once.json',maximum=256<<10);receipt=host.strict_json(receipt_raw)
    host.need(receipt['argv']==prepared['argv']and receipt['cwd']==str(ROOT)and receipt['stdin_sha256']==prepared['recipe_sha256']and receipt.get('failure')is None and receipt.get('signals',[])==[]and receipt['exit_code']==0 and type(receipt['exit_code'])is int and receipt['child_reaped']is receipt['owned_group_absent']is True,'actual reviewed successful transport invocation/reaping')
    host.need(receipt['work_seconds']==310 and receipt['term_seconds']==receipt['reap_seconds']==5 and receipt['output_limit_bytes']==16<<20 and type(receipt['pid'])is int and receipt['pid']>0 and receipt['pgid']==receipt['pid']and 0<receipt['elapsed_seconds']<=320,'unchanged actual local320 transport envelope')
    streams={}
    for label in ('stdout','stderr'):
        raw=host.held(ROOT/('cpu-import-once.'+label),receipt[label]['sha256'],16<<20)
        host.need(type(receipt[label]['bytes'])is int and len(raw)==receipt[label]['bytes'],'actual transport captured stream hashes/bytes');streams[label]=raw
    lines=[line[len(b'CPU_IMPORT_BUNDLE_V1 '):]for line in streams['stdout'].splitlines()if line.startswith(b'CPU_IMPORT_BUNDLE_V1 ')]
    host.need(len(lines)==1,'one actual complete transport bundle')
    body=core.inflate(base64.b64decode(lines[0],validate=True),48<<20)
    host.need(body['schema']==1 and type(body['schema'])is int and body['plan_sha256']==host.sha(plan_raw)and body['package_sha256']==plan['package_sha256']and body['qualification']is False and body['failure']is None and type(body['exit_code'])is int and body['exit_code']==0,'actual transport plan/source/success joins')
    host.need(type(body['files'])is dict and 1<=len(body['files'])<=128,'transport retained member cap')
    pins={};total=0
    for name,row in body['files'].items():
        path=Path(name);host.need(not path.is_absolute()and str(path)==name and path.parts[0]in ('host','output')and all(part not in ('','.','..')for part in name.split('/')),'transport retained canonical member')
        raw=base64.b64decode(row['data'],validate=True);total+=len(raw)
        host.need(type(row['bytes'])is int and len(raw)==row['bytes']<=16<<20 and row['sha256']==host.sha(raw),'actual transported file extent/hash')
        host.need(host.held(directory/name,row['sha256'],16<<20)==raw,'actual retained bytes equal transported bytes')
        pins[name]=dict(bytes=len(raw),sha256=row['sha256'])
    host.need(total+len(streams['stdout'])<=31<<20,'unchanged transport plus retained aggregate')
    actual={str(p.relative_to(directory))for p in directory.rglob('*')if p.is_file()}
    host.need(actual==set(pins)|{'BUNDLE-IDENTITY.json'},'closed retained bundle membership')
    identity=host.strict_json(host.held(directory/'BUNDLE-IDENTITY.json'))
    expected=dict(plan_sha256=host.sha(plan_raw),package_sha256=plan['package_sha256'],files=len(pins),bytes=total,remote_exit_code=0,remote_failure=None,qualification=False,file_identities=pins,transport_receipt_sha256=host.sha(receipt_raw),transport_stdout_sha256=host.sha(streams['stdout']),preparation_sha256=host.sha(host.held(ROOT/'PREPARATION.json')),recipe_sha256=prepared['recipe_sha256'])
    host.need(identity==expected,'actual decoder/transport/retention provenance hashes')
    return pins


def validate_retained(directory):
    directory=Path(directory).resolve();source=ROOT/'source'
    prepared=host.strict_json(host.held(ROOT/'PREPARATION.json'))
    binder=host.load(ROOT/'bind_future.py',prepared['binder_sha256'],'final_quality_serving_input_binding')
    binder.require_replay_binding(ROOT)
    for name,key in [('workstation.py','workstation_sha256'),('local_commands.py','local_commands_sha256'),('readout.py','readout_sha256'),('REMOTE-RECIPE.py','recipe_sha256')]:host.held(ROOT/name,prepared[key],(80<<20) if name=='REMOTE-RECIPE.py' else (16<<20))
    host.need(prepared['execution']=='NOT RUN'and prepared['source_author']==str(source)and prepared['native_imports']is prepared['gpu_observations']is False,'exact prospective packet provenance')
    plan_raw=host.held(ROOT/'PLAN.json',prepared['plan_sha256']);plan=host.validate_plan(host.strict_json(plan_raw))
    host.held(ROOT/'host_cpu.py',plan['launcher']['host_cpu.py'],2<<20)
    package=host.strict_json(host.held(source/'PACKAGE.json',plan['package_sha256']))
    host.need(package['sources']==plan['sources']and package['remote_root']==plan['source_root']and package['boot_id']==plan['target']['boot_id'],'exact current package/plan/source identity')
    for name,pin in plan['sources'].items():host.held(source/name,pin,(53<<20) if name=='QUALITY-CPU-INPUT.json' else (2<<20))
    host.load(source/'tokenizer_witness.py',plan['sources']['tokenizer_witness.py'],'tokenizer_witness')
    host.load(source/'thinking_segments.py',plan['sources']['thinking_segments.py'],'thinking_segments')
    q=host.load(source/'quality_contract.py',plan['sources']['quality_contract.py'],'quality_contract')
    witness=host.load(source/'quality_witness.py',plan['sources']['quality_witness.py'],'quality_witness');core=host.load(source/'tokenizer_core.py',plan['sources']['tokenizer_core.py'],'cpu_tokenizer_readout_core')
    pins=transport(directory,plan,plan_raw,prepared,core)
    host.need(host.held(directory/'host/PLAN.json')==plan_raw,'actual retained current host plan bytes')
    owner=host.strict_json(host.held(directory/'host/CPU-OWNER.json'))
    host.need(owner['schema']==1 and type(owner['schema'])is int and owner['mode']==plan['mode']and owner['qualification']is owner['gpu_grant']is owner['gpu_observations']is False,'actual owner CPU source/mode')
    inventory=owner['output_inventory'];seen={};total=0
    for row in inventory['files']:
        name=None
        for remote,label in [(plan['run_root'],'host'),(plan['source_root']+'/output','output')]:
            if row['path'].startswith(remote+'/'):name=label+'/'+row['path'][len(remote)+1:];break
        host.need(name is not None and name not in seen and type(row['bytes'])is int and row['bytes']>=0,'actual output inventory canonical unique extent')
        target=directory/name;host.need(target.exists()and row['bytes']==(target.stat().st_size if target.is_file()else 0),'actual output inventory retained extent join')
        total+=row['bytes'];seen[name]=row['bytes']
    expected={str(p.relative_to(directory))for label in ('host','output')for p in (directory/label).rglob('*')if str(p.relative_to(directory))!='host/CPU-OWNER.json'}
    host.need(set(seen)==expected and type(inventory['members'])is int and inventory['members']==len(seen)<=128 and type(inventory['bytes'])is int and inventory['bytes']==total<=32<<20,'closed actual host/output resource inventory')
    if owner.get('complete')is not True:return dict(status='CPU_FINAL_QUALITY_INCOMPLETE',qualification=False,owner_failure=owner.get('failures'),retry_authorized=False)
    admission=host.strict_json(host.held(directory/'host/ADMISSION.json'));proof=host.strict_json(host.held(directory/'host/STACK-PROOF.json',admission['stack_proof_sha256']))
    runtime=core.inflate(host.held(directory/'host/TOKENIZER-RUNTIME-INVENTORY.zlib',admission['tokenizer_runtime_inventory_sha256'],8<<20),8<<20)
    runtime['prefix_files']=core.inflate(host.held(source/'TOKENIZER-PREFIX-INVENTORY.zlib',plan['sources']['TOKENIZER-PREFIX-INVENTORY.zlib'],2<<20),4<<20)
    runtime['base_inventory']=host.strict_json(host.held(source/'TOKENIZER-BASE-INVENTORY.json',plan['sources']['TOKENIZER-BASE-INVENTORY.json']))
    frozen=host.strict_json(host.held(source/'CPU-TOKENIZER-INPUT.json',plan['sources']['CPU-TOKENIZER-INPUT.json']));stack=host.strict_json(host.held(source/'STACK.json',plan['sources']['STACK.json']))
    common=host.load(source/'common.py',plan['sources']['common.py'],'cpu_tokenizer_reader_common');common.validate_contract(package)
    identities=host.strict_json(host.held(directory/'host/RUNTIME-IDENTITIES.json'))
    host.need(set(identities)==set(package['runtime_files']),'actual runtime-file closure')
    for name,pin in package['runtime_files'].items():
        row=identities[name];host.need(row['sha256']==pin and type(row['bytes'])is int and row['bytes']>0 and row['path']==common.BASE+'/base-rootfs/'+name and all(type(row[k])is int and row[k]>=(0 if k=='device'else 1)for k in ('device','inode','nlink')),'actual current runtime-file source/hash/extent')
    raw=host.held(directory/'output/quality/QUALITY-RESULT.json',owner['quality_result_sha256'],4<<20);result=host.strict_json(raw)
    decoder_raw=host.held(directory/'output/quality/DECODE-RESULT.json',result['decode_result_sha256'],4<<20);decoded=host.strict_json(decoder_raw)
    witness.validate_decoder(decoded,frozen=frozen,input_sha256=plan['sources']['CPU-TOKENIZER-INPUT.json'],package=package,admission=admission,proof=proof,runtime_inventory=runtime,stack=stack,raw=decoder_raw)
    actual=q.strict(host.held(source/'QUALITY-CPU-INPUT.json',plan['sources']['QUALITY-CPU-INPUT.json'],q.INPUT_LIMIT))
    expected=q.strict(host.held(source/'EXPECTED-QUALITY.json',plan['sources']['EXPECTED-QUALITY.json']))
    stages=host.load(source/'quality_stages.py',plan['sources']['quality_stages.py'],'quality_stages')
    stages.validate_final(result,plan=plan,run=directory/'host',source=directory,decoded=decoded,value=actual,expected=expected,host=host,admission=admission)
    summary=result['grade_summary']
    release=host.strict_json(host.held(directory/'host/INDEPENDENT-RELEASE.json',owner['independent_release_sha256']));scope=host.strict_json(host.held(directory/'host/scope.json'))
    expected_inner=common.namespace_argv(package,admission['work_deadline_monotonic'],admission=admission,admission_file=plan['run_root']+'/ADMISSION.json',admission_sha256=host.sha(host.encoded(admission)),stack_proof_file=plan['run_root']+'/STACK-PROOF.json',cgroup_directory='/sys/fs/cgroup'+admission['host_cgroup_path'])
    rows={name:host.strict_json(host.held(directory/'host'/name))for name in ('preflight.json','independent.json','SCOPE-ARGV.json','INNER-ARGV.json')}
    witness.validate_supervisors(plan=plan,plan_sha256=host.sha(plan_raw),admission=admission,scope=scope,preflight=rows['preflight.json'],independent=rows['independent.json'],scope_argv=rows['SCOPE-ARGV.json'],inner_argv=rows['INNER-ARGV.json'],release=release,expected_inner=expected_inner)
    witness.validate_release(release,plan=plan,admission=admission,scope=scope,owner=owner)
    return dict(summary,status='NATIVE_CPU_FINAL_QUALITY_AND_RELEASE_VALIDATED',qualification=False,independent_release_verified=True,retry_authorized=False,native_generation_performed=False)

if __name__=='__main__':print(json.dumps(validate_retained(ROOT/'retained-cpu'),sort_keys=True))
