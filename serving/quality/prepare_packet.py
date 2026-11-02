"""Materialize the existing CPU packet after a separate complete serving/release readout.

This function validates the declared paired-menu data contract and creates an inert packet.
It does NOT authenticate physical serving release, authorize execution, register an
operation or contact the host. Root must bind actual readout/release provenance
before using the existing execution branch. Synthetic fixture packets are never
accepted as actual evidence.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import sys

ROOT=Path(__file__).resolve().parent
PROGRAMME=Path('/Users/uk-gr9yjx0l0y/Projects/ArchLinuxThreadripperAI/scratchpad/fp8-remaining-optimizations.w3m3xpg7')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def validate_source_members(root,inventory,*,materialized=False):
    """Exact frozen source tree; only a completed packet has both generated files."""
    def need(value,reason):
        if not value:raise ValueError('CPU packet source membership: '+reason)
    source=Path(root)/'source';info=source.lstat()
    need(stat.S_ISDIR(info.st_mode) and source.resolve(strict=True)==source,'canonical real source directory')
    expected={name[len('source/'):] for name in inventory['files'] if name.startswith('source/')}
    generated={'QUALITY-CPU-INPUT.json','PACKAGE.json'}
    need(type(materialized) is bool and expected and not expected.intersection(generated),'generated files cannot be frozen template members')
    files=expected|(generated if materialized else set())
    directories={p.as_posix() for name in files for p in Path(name).parents if p!=Path('.')}
    actual_files=set();actual_directories=set();count=0
    for path in source.rglob('*'):
        count+=1;need(count<=len(files)+len(directories),'extra source-subtree member')
        info=path.lstat();name=path.relative_to(source).as_posix()
        need(not stat.S_ISLNK(info.st_mode) and path.resolve(strict=True)==path,'source links forbidden')
        if stat.S_ISDIR(info.st_mode):actual_directories.add(name)
        else:
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular single-link source members required')
            actual_files.add(name)
    need(actual_files==files and actual_directories==directories,'exact source-subtree membership required')


def materialize(input_raw,destination,*,projection_inventory_sha256,fixture=False):
    inventory_raw=(ROOT/'PROJECTION-INVENTORY.json').read_bytes()
    if hashlib.sha256(inventory_raw).hexdigest()!=projection_inventory_sha256:raise ValueError('exact reviewed projection before materialization')
    inventory=json.loads(inventory_raw)
    validate_source_members(ROOT,inventory,materialized=False)
    for name,row in inventory['files'].items():
        if name.startswith('source/') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('frozen source identity before materialization')
    q=load(ROOT/'source/quality_contract.py','quality_contract')
    value=q.strict(input_raw);expected=q.strict((ROOT/'source/EXPECTED-QUALITY.json').read_bytes())
    q.validate_input(value,expected)
    q.need(input_raw==q.encoded(value),'canonical complete input; no reserialization/truncation')
    destination=Path(destination)
    q.need(destination.is_absolute() and destination.parent.resolve()==destination.parent and not os.path.lexists(destination),'fresh canonical exclusive packet root')
    source=destination/'source'
    destination.mkdir(mode=0o700);shutil.copytree(ROOT/'source',source)
    validate_source_members(destination,inventory,materialized=False)
    files={name[len('source/'):]:row['sha256'] for name,row in inventory['files'].items() if name.startswith('source/')}
    for name,pin in files.items():q.need(q.sha((source/name).read_bytes())==pin,'copied frozen source identity')
    with (source/'QUALITY-CPU-INPUT.json').open('xb') as stream:stream.write(input_raw)
    files['QUALITY-CPU-INPUT.json']=q.sha(input_raw)
    donor=q.strict((ROOT/'donor/PACKAGE.json').read_bytes())
    package=dict(donor,mode=q.MODE,payload='tokenizer_entry.py',
        final_quality=dict(decoder_runs=1,rows=len(value['rows']),grader_runs=0,input_bytes=q.INPUT_LIMIT,report_bytes=q.REPORT_LIMIT),
        remote_root='/home/developer/fp8-kv-benchmark.'+secrets.token_hex(8),sources=files)
    package.pop('tokenization',None)
    with (source/'PACKAGE.json').open('xb') as stream:stream.write(q.encoded(package))
    validate_source_members(destination,inventory,materialized=True)
    top_bytes=0
    for name in inventory['files']:
        if name.startswith('source/'):continue
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);top_bytes+=target.stat().st_size
    shutil.copyfile(ROOT/'PROJECTION-INVENTORY.json',destination/'PROJECTION-INVENTORY.json')
    top_bytes+=(destination/'PROJECTION-INVENTORY.json').stat().st_size
    q.need(top_bytes<=1<<20,'explicit top-level ordinary source allowance')
    workstation=load(destination/'workstation.py','quality_packet_workstation_'+secrets.token_hex(4))
    workstation.PROGRAMME=PROGRAMME
    plan=workstation.prepare(source)
    proof=dict(schema=1,status='SOURCE_PACKET_ONLY_PENDING_COMPLETE_SERVING_RELEASE_BINDING',
        fixture=fixture,input_sha256=q.sha(input_raw),input_bytes=len(input_raw),rows=len(value['rows']),cancellations=0,
        template_inventory_sha256=q.sha((ROOT/'TEMPLATE-INVENTORY.json').read_bytes()) if (ROOT/'TEMPLATE-INVENTORY.json').exists() else None,
        input_semantics_validated=True,physical_release_authenticated=False,
        native_decode='NOT RUN',quality='NOT RUN',precision_accepted=False,execution_authorized=False,
        owner_review_required=True,registered=False,package_sha256=q.sha((source/'PACKAGE.json').read_bytes()),plan=plan)
    pending=q.encoded(proof);q.need(len(pending)<=1<<20,'bounded preparation proof metadata')
    with (destination/'INPUT-BINDING-PENDING.json').open('xb') as stream:stream.write(pending)
    return proof


if __name__=='__main__':
    raise SystemExit('No direct execution/binding CLI: complete serving/release authenticator must bind actual outputs first.')
