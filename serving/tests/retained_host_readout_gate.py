"""Retained host-chain CPU regression gate; never accepts or grades operation206."""
import argparse, ast, copy, hashlib, importlib.util, json, marshal, stat, sys, tarfile, tempfile, types
from pathlib import Path

PINS = {
 '206': ('target-replay-scope-v1/instrumented-pair-v2/cleanup','e2011102000000a1',
 ['0f4e971490fab34a91b5b514a7be067fc97d7bfde305585fa627b889b64aeb1c','61b953231578e33202261bdad193c985accd47175fa9be5c72db60fc67449e17','bb408cc43fa400f4515dc463f2872a741b2b1b7f8872f245c18c50fbfc4afd02']),
 '203': ('pre-replay-control-v1/pair-v1/baseline','ca3598712e150f21',
 ['87be8af76d41709a67a67cc8c72036d3d6e8ca64c0aff1f303c6d8414442386c','c95a0831de8b9147d7e6bccddd4b714038b4829e34bf726e3c9352e24c2eb36a','8ac5eaf9c50ec7907494d304d00fb7caecefbf9cc7d56f6bd3ed10f7e98ac5d5'])}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(path):return json.loads(path.read_bytes())
def verify(root,pin):
 seal=root/'FINAL-ARTIFACTS.sha256';assert sha(seal.read_bytes())==pin
 expected={}
 for line in seal.read_text().splitlines():
  digest,name=line.split('  ',1);p=Path(name);assert not p.is_absolute() and '..' not in p.parts and name not in expected;expected[name]=digest
 actual={}
 for p in (root,*root.rglob('*')):
  info=p.lstat();assert p.resolve()==p and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode));assert info.st_mtime_ns<=seal.stat().st_mtime_ns
  if p.is_file() and p!=seal:
   assert info.st_nlink==1
   with p.open('rb') as f:actual[p.relative_to(root).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==expected
 return dict(root=str(root),manifest_sha256=pin,members=len(actual))
def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
class Hostile(types.ModuleType):
 def __getattr__(self,name):raise AssertionError('ambient contract accessed: '+name)
def must_refuse(label,call,kind,contains,events):
 try:call()
 except kind as e:
  assert contains in str(e),(label,str(e));events.append(dict(case=label,exception=type(e).__name__,message=str(e)))
 else:raise AssertionError('negative accepted: '+label)
def context(root,label,report):
 relative,nonce,pins=PINS[label];base=root/'.test-build/serving-cleanup'/relative
 for name,pin in zip(('author','caller','execution-v1'),pins):report['seals'].append(verify(base/name,pin))
 workload=base/'caller/collected/workload'/('fp8-kv-benchmark.'+nonce)
 protected=base/'caller/collected/protected'/('fp8-benchmark-source.'+nonce)
 # These exact nested manifests have already been authenticated by the caller's pinned seal.
 for nested in (workload,protected):report['seals'].append(verify(nested,sha((nested/'FINAL-ARTIFACTS.sha256').read_bytes())))
 author=base/'author/package-v2/review'
 review=Path(report['scratch_root'])/('protected-payload-'+label)/'review';review.mkdir(parents=True)
 with tarfile.open(protected/'payload.tar','r:*') as archive:
  names=set()
  for entry in archive.getmembers():
   name=Path(entry.name);assert entry.isfile() and not name.is_absolute() and '..' not in name.parts and entry.size<=16<<20 and entry.name not in names;names.add(entry.name)
   if entry.name.startswith('review/'):
    raw=archive.extractfile(entry).read();assert raw==(author/name.relative_to('review')).read_bytes()
    target=review/name.relative_to('review');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 for name in ('benchmark_physical_controller.py','serving_integration.py','mtp_contract.py'):
  assert (review/name).read_bytes()==(author/name).read_bytes()
 controller=load(review/'benchmark_physical_controller.py','astra_controller_'+label)
 manifest=read(workload/'MANIFEST.json');out=workload/'results';result=read(out/'CONTROLLER-RESULT.json')
 prepared=dict(manifest=manifest,resource_manifest=read(review/'RESOURCE-MANIFEST.json'),spec_sha256=manifest['spec_sha256'])
 return dict(controller=controller,reader=controller.http_integration,review=review,out=out,result=result,prepared=prepared)
def arm_inputs(c,arm):
 raw=read(c['out']/(arm+'-artifacts/result.json'));expected={k:raw[k] for k in ('source_manifest_sha256','session_sha256','spec_sha256')};expected['arm']=raw['logical_arm']
 owner=c['controller'].adapter.owner;fields=dict(next(r['supervision'] for r in c['result']['campaign']['arms'] if r['arm']==arm))
 for key in ('observed_processes','survivors'):fields[key]=tuple(owner.ProcessIdentity(**r) for r in fields[key])
 fields['root_process']=owner.ProcessIdentity(**fields['root_process'])
 for key in ('cleanup_errors','escalation_receipts'):fields[key]=tuple(fields[key])
 supervision=owner.SupervisionResult(**fields)
 joins=[r for pair in c['result']['live_namespace_joins'][arm] for r in pair.values()]
 partial=c['result'].get('partial_resource_records',{}).get(arm)
 if partial is not None:receipts={k:partial[k] for k in ('initial','final')}
 else:
  receipts=next(r for r in c['result']['resource_receipts'] if r['initial']['arm']==arm)
 return expected,supervision,joins,receipts
def chain(c,arm,work):
 expected,supervision,joins,receipts=arm_inputs(c,arm)
 with tempfile.TemporaryDirectory(prefix='readback-',dir=work) as tmp:
  value=c['reader'].read_http_artifacts(c['out']/(arm+'-artifacts'),extraction_root=Path(tmp).resolve()/'client',expected=expected,owner_arm=arm,joins=joins,supervision=supervision)
  c['controller'].checked_physical_resources(receipts,c['prepared'],arm,supervision)
  assert c['controller'].adapter.readout.checked_http_arm(value,arm,c['prepared']['spec_sha256']) is value
  assert value['status']=='HTTP_ARM_COMPLETE' and len(value['wire']['rows'])==176
 return dict(arm=arm,rows=176,read_http_artifacts=True,checked_physical_resources=True,checked_http_arm=True)
def negatives(c,work,events):
 m=c['reader'];expected,sup,joins,receipts=arm_inputs(c,'bf16')
 _,_,workers,observations,*_=m._native((c['out']/'bf16-artifacts/native-http-bundle.json').read_bytes(),expected)
 observations,_=m._bb_expand_observations(observations);binding=m._observation_json(observations,'SERVING-BINDING.json');graph=m._gdn_observer(binding['sources'])
 def validate(obs=observations,b= binding):return m._validate_observer_health(obs,binding=b,workers=workers,graph=graph)
 for name in ('graph','target_kv'):
  filename=f'OBSERVER-HEALTH-{name}-rank-0.json'
  for field in ('observer_failed','complete'):
   obs=dict(observations);record=json.loads(obs[filename]);record[field]=field=='observer_failed';obs[filename]=graph.canonical(record)
   must_refuse(name+'_'+field,lambda:validate(obs),ValueError,'incomplete or failed',events)
  obs=dict(observations);del obs[filename]
  must_refuse(name+'_missing',lambda:validate(obs),ValueError,'missing or oversized',events)
 for filename in ('graph-counter-rank-0.bin','replay-commit-counter-rank-0.bin'):
  obs=dict(observations);raw=bytearray(obs[filename]);raw[0]^=1;obs[filename]=bytes(raw)
  must_refuse(filename,lambda:validate(obs),ValueError,'counter',events)
 for role,fragment in (('mtp_contract','contract source changed'),('j01_probe','helper'),('graph_observer','graph observer source changed')):
  b=copy.deepcopy(binding);b['sources'][role]['sha256']='0'*64
  call=(lambda:m._gdn_observer(b['sources'])) if role=='graph_observer' else (lambda:validate(b=b))
  must_refuse(role+'_digest',call,ValueError,fragment,events)
 bad=copy.deepcopy(receipts);bad['final']['complete']=False
 must_refuse('resource_final_incomplete',lambda:c['controller'].checked_physical_resources(bad,c['prepared'],'bf16',sup),ValueError,'final resource monitor incomplete',events)
 # Authenticate a deliberately failing scratch contract, exercising real scope exec/finally.
 # Only the existing scope function's __file__ location is relocated; no loader mock.
 with tempfile.TemporaryDirectory(prefix='bad-contract-',dir=work) as tmp:
  directory=Path(tmp).resolve();raw=b'raise RuntimeError("authenticated contract execution failure")\n';(directory/'mtp_contract.py').write_bytes(raw)
  globals_copy=dict(m.__dict__,__file__=str(directory/'serving_integration.py'))
  scope=types.FunctionType(m._gdn_contract_scope.__wrapped__.__code__,globals_copy,'_gdn_contract_scope')
  from contextlib import contextmanager
  scope=contextmanager(scope);sources=copy.deepcopy(binding['sources']);sources['mtp_contract']['sha256']=sha(raw)
  def failing_load():
   with scope(sources):raise AssertionError('failing contract yielded')
  must_refuse('authenticated_contract_loading_exception',failing_load,RuntimeError,'authenticated contract execution failure',events)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 assert sys.version_info[:2]==(3,14) and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
 root=args.repo.resolve();work=args.output.resolve();work.mkdir(exist_ok=False)
 report=dict(scratch_root=str(work),schema=1,status='RUNNING',seals=[],runs=[],negatives=[],original206_accepted=False,native_work=False,quality_grading=False)
 try:
  source=root/'serving/spry_evidence/common_host_scope_integration.py';raw=source.read_bytes();compiled=compile(raw,str(source),'exec',dont_inherit=True)
  code=next(v for v in compiled.co_consts if isinstance(v,types.CodeType) and v.co_name=='_validate_observer_health')
  report['candidate']=dict(path=str(source),source_sha256=sha(raw),complete_module_code_sha256=sha(marshal.dumps(compiled)),function_code_sha256=sha(marshal.dumps(code)),function_ast_sha256=sha(functions(raw)['_validate_observer_health'].encode()),substitution='Only function code from completely compiled candidate module installed in original sealed reader globals; original validators and artifact pins unchanged')
  c=context(root,'206',report);old=(c['review']/'serving_integration.py').read_bytes();fa,fb=functions(raw),functions(old)
  changed=[name for name in fa.keys()|fb.keys() if fa.get(name)!=fb.get(name)];assert changed==['_validate_observer_health'],changed
  # Require all other module-level syntax to remain identical, too.
  ta,tb=ast.parse(raw),ast.parse(old)
  for t in (ta,tb):t.body=[n for n in t.body if not(isinstance(n,ast.FunctionDef) and n.name=='_validate_observer_health')]
  assert ast.dump(ta)==ast.dump(tb)
  report['candidate']['changed_functions']=changed
  assert 'mtp_contract' not in sys.modules
  must_refuse('original206_exact_failure',lambda:chain(c,'bf16',work),ModuleNotFoundError,"No module named 'mtp_contract'",report['negatives'])
  c['reader']._validate_observer_health=types.FunctionType(code,c['reader'].__dict__,'_validate_observer_health')
  for hostile in (False,True):
   trap=Hostile('mtp_contract') if hostile else None
   if hostile:sys.modules['mtp_contract']=trap
   row=chain(c,'bf16',work);assert sys.modules.get('mtp_contract') is trap
   events=[];negatives(c,work,events);assert sys.modules.get('mtp_contract') is trap
   report['negatives'].extend(dict(e,hostile_ambient=hostile) for e in events)
   report['runs'].append(dict(row,operation=206,hostile_ambient=hostile,ambient_restored=True,prospective_function=True));print(json.dumps(report['runs'][-1]),flush=True)
   sys.modules.pop('mtp_contract',None)
  p=context(root,'203',report);old203=(p['review']/'serving_integration.py').read_bytes();fc=functions(old203)
  report['production_function_equivalence']={name:fa[name]==fc[name] for name in ('_production_readback','read_http_artifacts')};assert all(report['production_function_equivalence'].values())
  for hostile in (False,True):
   trap=Hostile('mtp_contract') if hostile else None
   if hostile:sys.modules['mtp_contract']=trap
   for arm in ('bf16','kv4'):
    row=chain(p,arm,work);assert sys.modules.get('mtp_contract') is trap
    report['runs'].append(dict(row,operation=203,hostile_ambient=hostile,ambient_restored=True,prospective_function=False));print(json.dumps(report['runs'][-1]),flush=True)
   sys.modules.pop('mtp_contract',None)
  assert not any(name.split('.')[0] in ('torch','triton','sglang','numpy') for name in sys.modules)
  report.update(status='PASS',script_sha256=sha(Path(__file__).read_bytes()),limitations=['206 remains failed partial176/352; prospective CPU validation is not retroactive operation acceptance.','203 retains original production evidence; no new native production candidate evidence or grading.','Production branch equivalence is AST identity of entry/readback bodies; instrumented host health remains separately tested.','Authenticated-loading negative uses deliberately failing contract bytes only in fresh scratch with explicit scope relocation.'])
 except BaseException as error:
  report.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
 finally:
  (work/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
