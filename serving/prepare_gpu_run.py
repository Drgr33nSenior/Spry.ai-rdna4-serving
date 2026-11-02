"""Assemble fresh production GPU campaign sources and unfunded approval requests."""
import argparse
import ast
import difflib
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build as builder
import prepare_measurement
import propose_packet
from launcher import run_native as caller
from spry_common.canon import canonical_line as canonical,digest,need,parse_json
from spry_common.limits import derive


def put(root,name,raw):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw)


def archive_bytes(members,*,compressed,expected_members=80):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as archive:
        for name,raw in sorted(members.items()):
            builder.relative_name(name)
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444
            archive.addfile(info,io.BytesIO(raw))
    raw=output.getvalue()
    need(type(expected_members) is int and expected_members in (78,79,80)
         and len(raw)<=16<<20 and len(members)==expected_members,
         'retained source archive size/member ceiling')
    packed=gzip.compress(raw,mtime=0) if compressed else raw
    need(len(packed)<=16<<20,'retained physical archive ceiling')
    return packed


def launch_script(authenticated,binding,*,configured):
    prior=parse_json(authenticated['author/LAUNCHER-BINDINGS.json'])
    raw=authenticated['author/owner-upload/RUN-ON-WORKSTATION.sh'].decode()
    need(digest(raw.encode())==prior['upload_inventory']['RUN-ON-WORKSTATION.sh'],'retained shell pin differs')
    def replace(old,new,count):
        nonlocal raw
        need(raw.count(old)==count,'retained shell template differs: '+old[:40]);raw=raw.replace(old,new)
    replace(prior['nonce'],binding['nonce'],8)
    replace(prior['archive_sha256'],binding['archive_sha256'],2)
    replace(prior['manifest_sha256'],binding['manifest_sha256'],1)
    for name in caller.HELPERS:
        count=3 if name=='benchmark_host_transport.py' else 2
        key='review/'+name
        replace(prior['upload_inventory'][key],binding['upload_inventory'][key],count)
    replace('set -euo pipefail\n','set -euo pipefail\n[[ "$#" == 1 ]]\nreadonly approved_boot="$1"\n[[ "$approved_boot" =~ ^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]]\n',1)
    replace(prior['boot_id'],'"$approved_boot"',1)
    if configured:
        pin=binding['upload_inventory']['review/benchmark_host_transport.py']
        replace('    --expected-self-sha256 '+pin+'\n',
            '    --expected-self-sha256 '+pin+' \\\n    --run-config-sha256 '+binding['run_config_sha256']+' \\\n    --host-sha256 '+binding['host_sha256']+' \\\n    --protected-set-sha256 '+binding['protected_set_sha256']+' \\\n    --approved-boot-id "$approved_boot"\n',1)
    if 'bootstrap_common_sha256' in binding:
        pin=binding['bootstrap_common_sha256'];name='spry_common_canon.py'
        replace('\nPINS\n','\n'+pin+'  review/'+name+'\nPINS\n',1)
        replace('\nPROTECTED_PINS\n','\n'+pin+'  '+binding['protected_root']+'/review/'+name+'\nPROTECTED_PINS\n',1)
        group='for member in '+' '.join(caller.HELPERS)+'; do'
        replace(group,'for member in '+' '.join((*caller.HELPERS,name))+'; do',1)
    return raw.encode()


def quality_template(authenticated,destination,request_rows,*,source_plan=None):
    prefix='quality-template-w5/'
    for name,raw in authenticated.items():
        if name.startswith(prefix) and name!=prefix+'PROJECTION-INVENTORY.json':
            put(destination,name.removeprefix(prefix),raw)
    for relative,source in (('retained_replay.py','retained_replay.py'),('bind_future.py','bind_future.py'),('prepare_packet.py','prepare_packet.py'),
            ('producer/final_quality_inputs.py','final_quality_inputs.py'),('source/quality_contract.py','quality_contract.py'),
            ('source/thinking_segments.py','thinking_segments.py'),('source/tokenizer_entry.py','tokenizer_entry.py'),
            ('source/quality_stages.py','quality_stages.py'),('readout.py','cpu_readout.py')):
        (destination/relative).write_bytes(builder.read_regular(HERE/'quality'/source,1<<20))
    (destination/'source/common.py').write_bytes(builder.read_regular(HERE/'quality/common.py',1<<20))
    q=caller.checked_module(destination/'source','quality_contract',digest((destination/'source/quality_contract.py').read_bytes()))
    producer=caller.checked_module(destination/'producer','final_quality_inputs',digest((destination/'producer/final_quality_inputs.py').read_bytes()))
    private=parse_json((destination/'producer/inputs/QUALITY-MENU.json').read_bytes())['rows']
    expected=producer.project_menu(private,request_rows,source_plan=source_plan)
    (destination/'source/EXPECTED-QUALITY.json').write_bytes(q.encoded(expected))
    names=('retained_replay.py','seal_evidence.py','producer/final_quality_inputs.py',
           'producer/inputs/QUALITY-MENU.json','source/quality_contract.py','source/EXPECTED-QUALITY.json')
    put(destination,'SOURCE-PINS.json',canonical({name:digest((destination/name).read_bytes()) for name in names}))
    inventory=parse_json(authenticated[prefix+'PROJECTION-INVENTORY.json'])
    inventory['files']={p.relative_to(destination).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes()))
                        for p in sorted(destination.rglob('*')) if p.is_file()}
    need(len(inventory['files'])<=64,'retained quality-template source closure bound')
    put(destination,'PROJECTION-INVENTORY.json',canonical(inventory))
    binder=caller.checked_module(destination,'bind_future',digest((destination/'bind_future.py').read_bytes()))
    binder.authenticate_projection(destination,digest(canonical(inventory)))
    return dict(projection_inventory_sha256=digest(canonical(inventory)),rows=len(expected),
                native_cpu_seconds=320,execution_authorized=False)


def prepare(donor,output,*,packet_id,observed_boot_id,proposal=None,baseline_proposal=None,
            run_dir=HERE/'config/runs/w5-v2',qualification=False):
    output,donor=Path(output),Path(donor)
    need(output.is_absolute() and output.resolve()==output and output.parent.is_dir()
         and not output.exists() and not output.is_symlink(),'fresh canonical GPU preparation output required')
    need(not output.is_relative_to(HERE) and not output.is_relative_to(donor),'output must be outside source and donor')
    need(type(qualification) is bool and (not qualification or baseline_proposal is not None),
         'instrumented qualification requires an explicit matched baseline proposal')
    import re
    need(type(observed_boot_id) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',observed_boot_id),
         'explicit observed target boot identity required for baseline preparation')
    metadata=parse_json(builder.read_regular(HERE/'SOURCE.json',1<<20))
    authenticated=builder.authenticated_packet(donor,metadata)
    grading_tree=ast.parse(authenticated['quality-template-w5/host_cpu.py'])
    grading_boot=[node.value.value for node in grading_tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=='BOOT' for target in node.targets)
                  and isinstance(node.value,ast.Constant)]
    need(grading_boot==[observed_boot_id],'retained CPU grading template requires its reviewed boot; prepare a new template after reboot')
    seed=parse_json(authenticated['author/package-v2/MANIFEST.json'])
    before={Path(name).name:authenticated['author/package-v2/'+name] for name in seed['package_files']}
    with tempfile.TemporaryDirectory(prefix='gpu-candidate-review-') as temporary:
        review=Path(temporary).resolve()/'review'
        selected=HERE/'proposals/evidence-capacity' if proposal is None else Path(proposal)
        report=propose_packet.build(donor,review,packet_id=packet_id,proposal=selected,run_dir=run_dir)
        after={p.name:builder.read_regular(p) for p in (review/'package/review').iterdir()}
        run_raw=builder.read_regular(review/'package/config/runs/w5-v2/run.json');run=parse_json(run_raw)
        menu=parse_json(builder.read_regular(review/'package/config/runs/w5-v2/menu.json'))
        canonical_names=set(metadata['source_files'])|set(metadata['source_extractions'])|{Path(p).name for p in metadata['new_sources']}
        baseline_id=digest(canonical(dict(scope='cleanup-baseline-review',candidate_id=packet_id)))[:16]
        baseline_review=None
        if baseline_proposal is not None:
            baseline_review=Path(temporary).resolve()/'baseline'
            baseline_report=propose_packet.build(donor,baseline_review,packet_id=baseline_id,
                proposal=Path(baseline_proposal),run_dir=run_dir)
            before={p.name:builder.read_regular(p) for p in (baseline_review/'package/review').iterdir()}
            seed=parse_json(builder.read_regular(baseline_review/'package/MANIFEST-PROPOSAL.json'))
        comparison=prepare_measurement.compare_inputs(before,after,run=run,canonical_sources=canonical_names,
            protected_sources=metadata['protected_source_files'],paired_cleanup=baseline_proposal is not None,
            qualification=qualification)
        limits=derive(run['limits']);run_pin=digest(run_raw)
        config={p.relative_to(review/'package').as_posix():builder.read_regular(p) for p in (review/'package/config').rglob('*') if p.is_file()}
        baseline_files=dict(before)
        if baseline_proposal is None:
            baseline_identity=builder.render_session_identity(baseline_files,dict(pins=dict(run=run_pin)),baseline_id)
            baseline_manifest=dict(seed,boot_id=observed_boot_id,target_root=baseline_identity['target_root'],
                package_root=baseline_identity['target_root'],session_sha256=baseline_identity['session_sha256'],
                package_files={'review/'+name:digest(raw) for name,raw in baseline_files.items()},
                resource_manifest_sha256=digest(baseline_files['RESOURCE-MANIFEST.json']))
        else:
            baseline_manifest=seed
        candidate_manifest=parse_json(builder.read_regular(review/'package/MANIFEST-PROPOSAL.json'))
        output.mkdir(mode=0o700);campaigns={}
        for variant,files,manifest in (('baseline',baseline_files,baseline_manifest),('cleanup',after,candidate_manifest)):
            root=output/variant;author=root/'author';work=root/'caller';author.mkdir(parents=True,mode=0o700);work.mkdir(mode=0o700)
            configured=variant=='cleanup' or baseline_proposal is not None
            members={'review/'+name:raw for name,raw in files.items()};members['MANIFEST.json']=canonical(manifest)
            protected={'review/'+name:digest(files[name]) for name in metadata['protected_source_files']}
            package_review=review if variant=='cleanup' else baseline_review
            expected_members=80
            if package_review is not None:
                protected=parse_json(builder.read_regular(package_review/'package/PROTECTED-SHA256.json'))
                expected_members=report['proposed_package_members'] if variant=='cleanup' else baseline_report['proposed_package_members']
            if configured:members.update(config);members['PROTECTED-SHA256.json']=canonical(protected)
            payload=archive_bytes(members,compressed=configured,expected_members=expected_members)
            for name,raw in members.items():put(author/'package-v2',name,raw)
            put(author/'package-v2','payload.tar',payload)
            put(author,'PROTECTED-SHA256.json',canonical(protected))
            for name,raw in config.items():put(author,name,raw)
            upload={'payload.tar':payload,**{'review/'+name:files[name] for name in caller.HELPERS}}
            common_pin=protected.get('review/spry_common_canon.py')
            if common_pin is not None:
                need(digest(files['spry_common_canon.py'])==common_pin,'protected bootstrap common source differs')
                upload['review/spry_common_canon.py']=files['spry_common_canon.py']
            nonce=Path(manifest['target_root']).name.rsplit('.',1)[1]
            binding=dict(schema=2,scope='configured-owner-caller-v1',nonce=nonce,
                target_root=manifest['target_root'],protected_root='/var/tmp/fp8-benchmark-source.'+nonce,
                remote_upload_root='/home/developer/fp8-benchmark-upload.'+nonce,
                run_config_sha256=run_pin,host_config=run['host']['file'],host_sha256=run['host']['sha256'],
                protected_set_sha256=digest(canonical(protected)),manifest_sha256=digest(members['MANIFEST.json']),
                archive_sha256=digest(payload),total_seconds=limits['total_seconds'],caller_seconds=limits['caller_seconds'],
                upload_inventory={name:digest(raw) for name,raw in upload.items()},
                native_configuration_binding=configured,source_archive_boot_independent=configured)
            if common_pin is not None:binding['bootstrap_common_sha256']=common_pin
            script=launch_script(authenticated,binding,configured=configured);upload['RUN-ON-WORKSTATION.sh']=script
            binding['upload_inventory']['RUN-ON-WORKSTATION.sh']=digest(script)
            for name,raw in upload.items():put(author/'owner-upload',name,raw)
            put(author,'LAUNCHER-BINDINGS-TEMPLATE.json',canonical(binding))
            identities=dict(nonce=nonce,target_root=manifest['target_root'],manifest_sha256=binding['manifest_sha256'],
                archive_sha256=binding['archive_sha256'],package_sha256=digest(canonical(manifest['package_files'])),
                source_manifest_sha256=digest(canonical({n:p for n,p in manifest['package_files'].items() if n.endswith('.py')})),
                session_sha256=manifest['session_sha256'])
            put(author,'PACKAGE-IDENTITIES.json',canonical(identities));put(work,'PACKAGE-IDENTITIES.json',canonical(identities))
            for name,raw in authenticated.items():
                if name.startswith('launcher/caller-preview/') and name.endswith('.py'):
                    filename=Path(name).name
                    if filename in ('run_native.py','power_observe.py'):raw=builder.read_regular(HERE/'launcher'/filename,1<<20)
                    put(work,filename,raw);put(author/'caller-sources',filename,raw)
            driver=builder.read_regular(HERE/'launcher/execute_once.py',1<<20)
            put(work,'execute_once.py',driver);put(author/'caller-sources','execute_once.py',driver)
            put(work,run['host']['file'],config[run['host']['file']])
            caller_pins={p.name:digest(p.read_bytes()) for p in work.glob('*.py')}
            put(author,'CALLER-SOURCES.json',canonical(caller_pins))
            grading=quality_template(authenticated,author/'quality-template',menu['rows'])
            delta=''
            for key in sorted(protected):
                name=Path(key).name
                delta+=''.join(difflib.unified_diff(before.get(name,b'').decode().splitlines(True),files[name].decode().splitlines(True),
                    fromfile='a/review/'+name,tofile='b/review/'+name))
            put(author,'PROTECTED-COMPLETE.patch',delta.encode())
            approval=dict(schema=1,status='UNFUNDED_GPU_RUN_REQUEST',variant=variant,nonce=nonce,
                run_config_sha256=run_pin,host_sha256=binding['host_sha256'],protected_set_sha256=binding['protected_set_sha256'],
                manifest_sha256=binding['manifest_sha256'],archive_sha256=binding['archive_sha256'],
                protected_patch_sha256=digest(delta.encode()),caller_inventory=caller_pins,
                binding_template_sha256=digest(canonical(binding)),observed_boot_id=observed_boot_id,
                requested_owner_seconds=limits['total_seconds'],requested_caller_seconds=limits['caller_seconds'],
                requested_model_starts=2,requested_generations=2*len(menu['rows']),requested_cpu_grading_seconds=320,
                model_start_authorized=False,maximum_starts=0,cpu_execution_authorized=False,
                retry_authorized=False,promotion_authorized=False,deployment_authorized=False,publication_authorized=False)
            put(root,'APPROVAL-REQUEST.json',canonical(approval))
            put(work,'APPROVAL-REQUEST.json',canonical(approval))
            campaigns[variant]=dict(approval_request_sha256=digest(canonical(approval)),package=identities,
                protected_set_sha256=binding['protected_set_sha256'],source_archive_bytes=len(payload),
                source_archive_boot_independent=configured,grading=grading,
                native_configuration_binding=configured,legacy_control_retains_original_gates=not configured)
        result=dict(schema=1,status='GPU_PREPARATION_PENDING_PREFLIGHT_AND_APPROVAL',campaigns=campaigns,
            matched_inputs=comparison,pilot_order=['baseline','cleanup'],launch_authorized=False,model_starts_executed=0,
            qualification_only=qualification,
            automatic_sequence=False,hyperloom_launch_delegation=False,profilers_enabled=False,
            remaining_gates=['CPU caller/quality integration checks','native interpreter preflight',
                'exact protected checkpoint review','source seal and fresh one-use approval for the first campaign'])
        put(output,'GPU-PREPARATION.json',canonical(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--packet-id',required=True);parser.add_argument('--observed-boot-id',required=True)
    parser.add_argument('--proposal',type=Path);parser.add_argument('--baseline-proposal',type=Path)
    parser.add_argument('--run-dir',type=Path,default=HERE/'config/runs/w5-v2')
    parser.add_argument('--qualification',action='store_true')
    args=parser.parse_args();result=prepare(args.donor,args.output,packet_id=args.packet_id,
        observed_boot_id=args.observed_boot_id,proposal=args.proposal,baseline_proposal=args.baseline_proposal,
        run_dir=args.run_dir,qualification=args.qualification)
    print(json.dumps(dict(status=result['status'],variants=list(result['campaigns']),launch_authorized=False),sort_keys=True))


if __name__=='__main__':main()
