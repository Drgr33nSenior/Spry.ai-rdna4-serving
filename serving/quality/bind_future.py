"""Exact full owner replay -> producer -> declared CPU input. No execution or authority."""
import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(__file__).resolve().parent
SOURCE_PINS={}
PIN_FIELDS={'manifest_sha256','archive_sha256','package_sha256','source_manifest_sha256','session_sha256'}
RESULT_FIELDS={'schema','status','arm_results','provenance','qualification','precision_accepted','current_gpu_admission','native_quality_grade'}
PROVENANCE_FIELDS={'package','seals','local_receipts','controller_sha256','root_transport_sha256','phases','protected_closeout_sha256','artifacts'}


def need(value,reason):
 if not value:raise ValueError('final CPU input binding: '+reason)
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def strict(raw):
 def pairs(items):
  result={}
  for key,value in items:need(key not in result,'duplicate JSON key');result[key]=value
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def held(path,pin=None,maximum=1<<20):
 path=Path(path);need(path.is_absolute() and path.resolve(strict=True)==path,'canonical source/evidence path')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=maximum,'bounded single-link regular file')
  with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(maximum+1)
  fields=('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns');after,named=os.fstat(fd),path.lstat()
  need(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k)==getattr(named,k) for k in fields),'held file changed')
  need(pin is None or sha(raw)==pin,'exact authenticated bytes '+path.name)
  return raw
 finally:os.close(fd)


def module(path,name,pin):
 raw=held(path,pin,4<<20)
 class Loader(SourceFileLoader):
  def get_code(self,fullname):return self.source_to_code(raw,str(path))
 spec=importlib.util.spec_from_loader(name,Loader(name,str(path)));result=importlib.util.module_from_spec(spec)
 sys.modules[name]=result;spec.loader.exec_module(result);return result


def authenticate_projection(root,pin,*,materialized=False):
 global SOURCE_PINS
 need(sys.flags.optimize==0,'unoptimized original seal verifier required')
 inventory=strict(held(root/'PROJECTION-INVENTORY.json',pin));need(type(inventory['files']) is dict and len(inventory['files'])<=64,'exact small projection closure')
 for name,row in inventory['files'].items():
  path=Path(name);need(not path.is_absolute() and str(path)==name and all(x not in ('','.','..') for x in name.split('/')),'canonical projection source name')
  raw=held(root/name,row['sha256'],4<<20);need(len(raw)==row['bytes'],'exact source extent')
 source_pins=strict(held(root/'SOURCE-PINS.json',inventory['files']['SOURCE-PINS.json']['sha256']))
 need(set(source_pins)=={'retained_replay.py','seal_evidence.py','producer/final_quality_inputs.py',
  'producer/inputs/QUALITY-MENU.json','source/quality_contract.py','source/EXPECTED-QUALITY.json'},'complete frozen quality dependencies')
 for name,pin in source_pins.items():need(inventory['files'][name]['sha256']==pin,'frozen decoder/reader/producer dependency')
 SOURCE_PINS=source_pins
 preparer=module(root/'prepare_packet.py','exact_source_membership_validator',inventory['files']['prepare_packet.py']['sha256'])
 preparer.validate_source_members(root,inventory,materialized=materialized)
 return inventory


def validate_replay(result,request):
 need(type(result) is dict and set(result)==RESULT_FIELDS and type(result['schema']) is int and result['schema']==1 and
  result['status']=='RETAINED_TWO_ARM_OWNER_RESOURCE_RELEASE_HTTP_REPLAY_COMPLETE' and
  all(result[k] is False for k in ('qualification','precision_accepted','current_gpu_admission','native_quality_grade')),'exact complete unqualified replay')
 proof=result['provenance'];need(type(proof) is dict and set(proof)==PROVENANCE_FIELDS and proof['package']==request['expected_package_pins'],'exact authored package/source/session pins')
 seals=proof['seals'];need(type(seals) is dict and set(seals)=={'operation','workload','protected','source_author'},'four original closed evidence trees')
 for label,row in seals.items():
  need(type(row) is dict and set(row)=={'root','manifest_sha256','members'} and digest(row['manifest_sha256']) and type(row['members']) is int and row['members']>0,'original seal result')
  if label!='source_author':need(row['manifest_sha256']==request[label+'_seal_sha256'],'requested '+label+' seal identity')
 need(seals['operation']['root']==request['operation_root'] and Path(seals['source_author']['root'])/'package-v2'==Path(request['package_root']),'operation/source-author mapping')
 need(type(result['arm_results']) is dict and set(result['arm_results'])=={'bf16','kv4'} and set(proof['artifacts'])=={'bf16','kv4'},'exact two physical arms')
 for arm,logical in (('bf16','control'),('kv4','mtp')):
  row=proof['artifacts'][arm]
  need(row['owner_arm']==arm and row['logical_arm']==logical and result['arm_results'][arm]['arm']==logical,'selected physical/logical arm mapping')
 return result


def bind(*,operation_root,operation_seal_sha256,workload_seal_sha256,protected_seal_sha256,
 package_root,expected_package_pins,destination,projection_inventory_sha256):
 inventory=authenticate_projection(ROOT,projection_inventory_sha256)
 need(type(expected_package_pins) is dict and set(expected_package_pins)==PIN_FIELDS and all(digest(p) for p in expected_package_pins.values()),'five authored pins, not per-arm HTTP source maps')
 operation_root,package_root,destination=map(Path,(operation_root,package_root,destination))
 for path in (operation_root,package_root):need(path.is_absolute() and path.resolve(strict=True)==path and path.is_dir(),'canonical existing evidence/source roots')
 need(destination.is_absolute() and destination.parent.resolve(strict=True)==destination.parent and not os.path.lexists(destination) and
  not destination.is_relative_to(operation_root) and not destination.is_relative_to(package_root.parent),'fresh binding root outside sealed trees')
 need(all(digest(pin) for pin in (operation_seal_sha256,workload_seal_sha256,protected_seal_sha256)),'three explicit requested seals')
 request=dict(operation_root=str(operation_root),operation_seal_sha256=operation_seal_sha256,workload_seal_sha256=workload_seal_sha256,
  protected_seal_sha256=protected_seal_sha256,package_root=str(package_root),expected_package_pins=expected_package_pins)
 reader=module(ROOT/'retained_replay.py','exact_retained_input_replay',SOURCE_PINS['retained_replay.py'])
 destination.mkdir(mode=0o700)
 replay=reader.replay_retained(**request,destination=destination/'serving-replay',sealer_path=ROOT/'seal_evidence.py')
 validate_replay(replay,request)
 q=module(ROOT/'source/quality_contract.py','quality_contract',SOURCE_PINS['source/quality_contract.py'])
 producer=module(ROOT/'producer/final_quality_inputs.py','exact_quality_producer',SOURCE_PINS['producer/final_quality_inputs.py'])
 expected=q.strict(held(ROOT/'source/EXPECTED-QUALITY.json',SOURCE_PINS['source/EXPECTED-QUALITY.json']))
 value=producer.build_input(replay['arm_results'],menu=expected);q.validate_input(value,expected);raw=q.encoded(value)
 prepare=module(ROOT/'prepare_packet.py','exact_quality_materializer',inventory['files']['prepare_packet.py']['sha256'])
 packet=destination/'cpu';prepare.materialize(raw,packet,projection_inventory_sha256=projection_inventory_sha256)
 proof=dict(schema=2,status='ACTUAL_PAIRED_INPUT_BOUND_BY_RECOMPUTED_OWNER_RESOURCE_RELEASE_HTTP',request=request,
  source_pins=SOURCE_PINS,projection_inventory_sha256=projection_inventory_sha256,replay_provenance=replay['provenance'],replay_result_sha256=sha(canonical(replay)),
  input_sha256=sha(raw),input_bytes=len(raw),rows=len(value['rows']),cancellations=0,cpu_package_sha256=sha(held(packet/'source/PACKAGE.json')),
  cpu_plan_sha256=sha(held(packet/'PLAN.json')),cpu_recipe_sha256=sha(held(packet/'REMOTE-RECIPE.py',maximum=80<<20)),
  precision_accepted=False,qualification=False,native_quality_grade=False,execution_authorized=False)
 proof_raw=canonical(proof);need(len(proof_raw)<=1<<20,'bounded binding metadata')
 with (packet/'ACTUAL-INPUT-BINDING.json').open('xb') as stream:stream.write(proof_raw)
 old=held(packet/'PREPARATION.json');metadata=strict(old);metadata.update(serving_input_binding_sha256=sha(proof_raw),projection_inventory_sha256=projection_inventory_sha256)
 with (packet/'PREPARATION-BEFORE-BINDING.json').open('xb') as stream:stream.write(old)
 updated=canonical(metadata);need(len(updated)<=1<<20,'bounded preparation metadata');(packet/'PREPARATION.json').write_bytes(updated)
 require_replay_binding(packet)
 return proof


def require_replay_binding(packet_root):
 """Consume the exact pinned replay product; reverify all four original seals."""
 root=Path(packet_root);prepared=strict(held(root/'PREPARATION.json'))
 need(digest(prepared.get('projection_inventory_sha256')) and digest(prepared.get('serving_input_binding_sha256')),'full recomputed serving input binding required')
 authenticate_projection(root,prepared['projection_inventory_sha256'],materialized=True)
 proof=strict(held(root/'ACTUAL-INPUT-BINDING.json',prepared['serving_input_binding_sha256']))
 legacy=proof.get('schema')==1 and proof.get('status')=='ACTUAL_352_INPUT_BOUND_BY_RECOMPUTED_OWNER_RESOURCE_RELEASE_HTTP' and proof.get('rows')==352
 configured=proof.get('schema')==2 and proof.get('status')=='ACTUAL_PAIRED_INPUT_BOUND_BY_RECOMPUTED_OWNER_RESOURCE_RELEASE_HTTP'
 need(type(proof['schema']) is int and (legacy or configured) and
  proof['source_pins']==SOURCE_PINS and proof['projection_inventory_sha256']==prepared['projection_inventory_sha256'] and
  all(proof[k] is False for k in ('precision_accepted','qualification','native_quality_grade','execution_authorized')),'source-pinned actual input binding')
 provenance=proof['replay_provenance'];request=proof['request']
 need(provenance['package']==request['expected_package_pins'] and set(provenance['seals'])=={'operation','workload','protected','source_author'},'exact package/four-seal provenance')
 sealer=module(root/'seal_evidence.py','exact_bound_quality_sealer',SOURCE_PINS['seal_evidence.py'])
 for label,row in provenance['seals'].items():need(sealer.verify(Path(row['root']),row['manifest_sha256'])==row,'closed '+label+' evidence remains byte-identical')
 q=module(root/'source/quality_contract.py','quality_contract',SOURCE_PINS['source/quality_contract.py'])
 producer=module(root/'producer/final_quality_inputs.py','bound_quality_producer',SOURCE_PINS['producer/final_quality_inputs.py'])
 expected=q.strict(held(root/'source/EXPECTED-QUALITY.json',SOURCE_PINS['source/EXPECTED-QUALITY.json']))
 raw=held(root/'source/QUALITY-CPU-INPUT.json',proof['input_sha256'],q.INPUT_LIMIT);value=q.strict(raw);q.validate_input(value,expected)
 need(raw==q.encoded(value) and len(raw)==proof['input_bytes'] and type(proof['rows']) is int and
  proof['rows']==len(value['rows'])==len(expected) and proof['cancellations']==0,'exact input bytes/declared menu')
 package=strict(held(root/'source/PACKAGE.json',proof['cpu_package_sha256']));plan=strict(held(root/'PLAN.json',proof['cpu_plan_sha256']))
 held(root/'REMOTE-RECIPE.py',proof['cpu_recipe_sha256'],80<<20)
 need(package['sources']['QUALITY-CPU-INPUT.json']==proof['input_sha256'] and plan['package_sha256']==proof['cpu_package_sha256'] and
  prepared['plan_sha256']==proof['cpu_plan_sha256'] and prepared['recipe_sha256']==proof['cpu_recipe_sha256'],'CPU input/package/plan/recipe joins')
 return proof


if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for name in ('operation-root','operation-seal-sha256','workload-seal-sha256','protected-seal-sha256','package-root','destination','projection-inventory-sha256'):
  parser.add_argument('--'+name,required=True)
 parser.add_argument('--expected-package-pins',type=Path,required=True)
 args=vars(parser.parse_args());args['expected_package_pins']=strict(held(args['expected_package_pins']))
 proof=bind(**args)
 print(json.dumps(dict(status=proof['status'],input_sha256=proof['input_sha256'],input_bytes=proof['input_bytes'],native_quality_grade=False,execution_authorized=False),sort_keys=True))
