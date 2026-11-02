"""Source-only graph simulation; no Torch, native arithmetic or GPU proof."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
standins=load('pre_replay_standins',ROOT/'tests/test_verify_metadata_reuse.py')
baseline=load('pre_replay_control',ROOT/'serving/legacy/review/verify_decode_adapter.py')
candidate=load('pre_replay_candidate',ROOT/'serving/spry_serving/pre_replay_adapter.py')
Tensor,Torch,Translator,Base=standins.Tensor,standins.Torch,standins.Translator,standins.Base

class FakeBackend(Base):
 def init_forward_metadata(self,batch):
  if getattr(self,'initializer_failure',False):raise RuntimeError('initializer failure')
  self.forward_metadata=SimpleNamespace(kv_indptr=self.arguments[7],max_extend_len=8)
 def init_forward_metadata_out_graph(self,batch,in_capture=False):
  if getattr(self,'initializer_failure',False):raise RuntimeError('initializer failure')
  if in_capture:self.forward_metadata=SimpleNamespace(kv_indptr=self.arguments[7],max_extend_len=8)
 def get_num_kv_splits(self,splits,lengths):
  assert splits.numel()==lengths.numel()==1
  self.selector_calls+=1
  self.torch.submit(lambda:splits.put([standins.singleton_split(lengths.values()[0])]))
 def forward_extend(self,q,k,v,layer,batch,*args,**kwargs):
  self.torch.submit(lambda:self.events.append(('write',layer.layer_id)))
  result=self.extend_attention_fwd(*self.arguments,**kwargs)
  self.torch.submit(lambda:self.events.append(('attention',layer.layer_id)))
  return result

def decode(q,kb,vb,out,indptr,indices,partial,lse,splits,maximum,scale,ks,vs,**kwargs):
 assert (maximum,scale,ks,vs,kwargs)==(16,0.0625,1.0,1.0,dict(enable_lean=False,page_size=1))
 def consume():
  ptr=indptr.values();out.trace.append((tuple(ptr),tuple(splits.values()),tuple((indices.storage[ptr[row]],indices.storage[ptr[row+1]-1]) for row in range(q.shape[0]))))
 q.torch.submit(consume)

class Fixture:
 def __init__(self,module,bs=1):
  self.torch=Torch();self.cls=type('FakeBackend',(Base,),{k:v for k,v in vars(FakeBackend).items() if k not in ('__dict__','__weakref__')});path=Path(__file__).resolve()
  self.adapter=module.Adapter(dict(arm='mtp',sources=dict(triton_backend=dict(module='pre_replay_native',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),decode_attention_fwd=dict(module='pre_replay_decode'))),dict(rank=0),self.cls,self.torch)
  self.backend=b=self.cls();b.torch,b.events,b.selector_calls=self.torch,[],0;b.forward_metadata=SimpleNamespace();b.decode_attention_fwd=decode;b.extend_attention_fwd=lambda *a,**kw:'native-fallback';b.kv_index_translator=Translator(self.torch)
  for key,value in dict(page_size=1,dcp_size=1,use_mla=False,enable_deterministic=False,static_kv_splits=False,split_tile_size=None,max_kv_splits=16,device_core_count=32,num_head=12,num_kv_head=2,max_context_len=32768).items():setattr(b,key,value)
  self.batch=SimpleNamespace(batch_size=bs,forward_mode=SimpleNamespace(is_target_verify=lambda:True),req_pool_indices=Tensor(self.torch,bs,values=list(range(bs))),out_cache_loc=Tensor(self.torch,8*bs,values=range(1,8*bs+1)))
  bf16='torch.bfloat16';rows=8*bs
  self.q,self.k,self.v,self.out=[Tensor(self.torch,shape,bf16) for shape in ((rows,12,256),(rows,2,256),(rows,2,256),(rows,12,256))];self.ptr=Tensor(self.torch,bs+1,values=[0]*(bs+1));kb=Tensor(self.torch,(65536,2,256),bf16)
  b.arguments=(self.q,self.k,self.v,self.out,kb,kb,Tensor(self.torch,bs+1,values=range(0,rows+1,8)),self.ptr,Tensor(self.torch,65536),None,True,None,8,1.0,1.0,0.0625)
 def state(self,prefixes,requests=None):
  total=0;ptr=[0]
  for prefix in prefixes:total+=prefix;ptr.append(total)
  self.ptr.put(ptr)
  if requests is not None:self.batch.req_pool_indices.put(requests)
 def forward(self,initialize=True,layers=range(3,64,4)):
  if initialize:self.backend.init_forward_metadata(self.batch)
  for layer in layers:self.backend.forward_extend(self.q,self.k,self.v,SimpleNamespace(layer_id=layer),self.batch)
 def metadata(self):
  b=self.adapter.buffers[id(self.backend),8*self.batch.batch_size];end=b['indptr'].values()[-1];return tuple(b['indptr'].values()),tuple(b['splits'].values()),tuple(b['indices'].storage[:end])
 def capture(self):
  self.forward();self.backend.init_forward_metadata_out_graph(self.batch,in_capture=True)
  for _ in range(2):self.backend.init_forward_metadata_in_graph(self.batch);self.forward(False)
  self.torch.operations=[];self.torch.capturing=True;self.backend.init_forward_metadata_in_graph(self.batch);self.forward(False);self.torch.capturing=False
  return list(self.torch.operations)

class PreReplayMetadata(unittest.TestCase):
 def setUp(self):
  native=ModuleType('pre_replay_decode');native.decode_attention_fwd=decode;self.context=patch.dict(sys.modules,pre_replay_decode=native);self.context.start();self.addCleanup(self.context.stop)
 def test_eager_metadata_and_attention_inputs_match_at_both_batch_sizes(self):
  for bs in (1,2):
   a,b=Fixture(baseline,bs),Fixture(candidate,bs)
   for prefix in (0,31,127,2040,32760):
    values=[prefix] if bs==1 else [prefix,max(0,prefix-23)]
    for f in (a,b):f.state(values);f.forward()
    self.assertEqual(a.metadata(),b.metadata());self.assertEqual(a.out.trace,b.out.trace);self.assertEqual(a.backend.events,b.backend.events)
   self.assertEqual(b.adapter.metadata_capture_build_host_calls,0)
 def test_graph_reads_prepared_buffers_without_recording_metadata_work(self):
  for bs in (1,2):
   a,b=Fixture(baseline,bs),Fixture(candidate,bs);graphs=[a.capture(),b.capture()]
   self.assertEqual(len(graphs[1]),48);self.assertGreater(len(graphs[0]),len(graphs[1]));self.assertEqual(b.adapter.metadata_capture_build_host_calls,0)
   addresses=[tuple(id(t) for t in f.adapter.buffers[id(f.backend),8*bs].values()) for f in (a,b)]
   for prefix in (0,127,32760,31):
    for f,graph,identity in zip((a,b),graphs,addresses):
     f.state([prefix] if bs==1 else [prefix,max(0,prefix-11)],list(range(bs)) if prefix%2==0 else list(reversed(range(bs))))
     f.backend.init_forward_metadata_out_graph(f.batch)
     for operation in graph:operation()
     self.assertEqual(tuple(id(t) for t in f.adapter.buffers[id(f.backend),8*bs].values()),identity)
    self.assertEqual(a.metadata(),b.metadata());self.assertEqual(a.out.trace[-16:],b.out.trace[-16:])
 def test_replay_preparation_updates_before_the_graph_runs(self):
  f=Fixture(candidate);graph=f.capture();f.state([127],[1]);builds=f.adapter.metadata_build_host_calls;f.backend.init_forward_metadata_out_graph(f.batch)
  self.assertEqual(f.adapter.metadata_build_host_calls,builds+1);self.assertEqual(f.adapter.buffers[id(f.backend),8]['lengths'].values(),list(range(128,136)))
  for operation in graph:operation()
  self.assertEqual(f.adapter.metadata_build_host_calls,builds+1)
 def test_missing_outside_preparation_refuses_capture(self):
  f=Fixture(candidate);f.state([127]);f.backend.forward_metadata=SimpleNamespace(kv_indptr=f.ptr,max_extend_len=8);f.torch.capturing=True
  with self.assertRaisesRegex(ValueError,'outside the forward graph'):f.forward(False)
 def test_initializer_failure_invalidates_stale_preparation(self):
  f=Fixture(candidate);f.forward();f.backend.initializer_failure=True
  with self.assertRaisesRegex(RuntimeError,'initializer failure'):f.backend.init_forward_metadata_out_graph(f.batch)
  self.assertNotIn(id(f.backend),f.adapter.metadata_scopes)
 def test_stream_and_reference_changes_refresh_only_before_eager_forward(self):
  for change in ('stream','requests','translator'):
   f=Fixture(candidate);f.forward(layers=[3]);builds=f.adapter.metadata_build_host_calls
   if change=='stream':f.torch.stream=99
   elif change=='requests':f.batch.req_pool_indices=Tensor(f.torch,1,values=[1])
   else:f.backend.kv_index_translator=Translator(f.torch)
   f.forward(False,layers=[7]);self.assertEqual(f.adapter.metadata_build_host_calls,builds+1)
 def test_in_graph_hook_preserves_outside_preparation(self):
  f=Fixture(candidate);f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True);scope=f.adapter.metadata_scopes[id(f.backend)];f.backend.init_forward_metadata_in_graph(f.batch);self.assertIs(f.adapter.metadata_scopes[id(f.backend)],scope)
 def test_restore_removes_the_inherited_hook_and_restores_native_methods(self):
  f=Fixture(candidate);f.forward();f.adapter.restore();self.assertNotIn('init_forward_metadata_in_graph',f.cls.__dict__);self.assertFalse(f.adapter.metadata_scopes)



class PreReplayWitness(unittest.TestCase):
 def setUp(self):
  import tempfile
  temporary=tempfile.TemporaryDirectory(prefix='pre-replay-reader-');self.addCleanup(temporary.cleanup)
  directory=Path(temporary.name).resolve()
  (directory/'kv_graph_join.py').write_bytes((ROOT/'serving/legacy/review/kv_graph_join.py').read_bytes())
  (directory/'serving_integration.py').write_bytes((ROOT/'serving/spry_evidence/pre_replay_integration.py').read_bytes())
  self.reader=load('pre_replay_witness_reader',directory/'serving_integration.py')
  self.enabled=True;self.binding=dict(mode='native-http-fp8-sprint01-v1',arm='mtp',session_sha256='a'*64,spec_sha256='b'*64,source_manifest_sha256='c'*64,sources=dict(verify_decode_adapter=dict(path='/mnt/review/verify_decode_adapter.py',module='verify_decode_adapter',sha256='d'*64)),resolved_contract=dict(context_length=32768));self.worker=dict(rank=0,pid=42,identity_sha256='e'*64)
  self.record={key:self.binding[key] for key in ('mode','arm','session_sha256','spec_sha256','source_manifest_sha256')};self.record.update(schema=1,rank=0,pid=42,worker_identity_sha256='e'*64,enabled=True,block_size=8,before_external_generations=True,extra_tensor_operations=0,extra_model_forwards=0,qualification=False,adapter_source=self.binding['sources']['verify_decode_adapter'],eligible_host_calls=96,capture_host_calls=32,layers=list(range(3,64,4)),buffers=2,metadata_build_host_calls=6,metadata_reuse_host_calls=96,metadata_capture_build_host_calls=0,indices_elements=[8*(32768+8),16*(32768+8)],metadata_preparation_policy='outside-forward-graph-v1')
  self.context=patch.object(self.reader,'_mtp_contract',return_value=SimpleNamespace(contract_for_arm=lambda arm:dict(enabled=self.enabled)));self.context.start();self.addCleanup(self.context.stop)
 def validate(self):return self.reader.validate_full_enabled_record(self.record,self.binding,self.worker)
 def test_new_preparation_witness_is_accepted(self):self.assertIs(self.validate(),self.record)
 def test_legacy_captured_preparation_record_refuses(self):
  self.record['metadata_capture_build_host_calls']=2;self.record['metadata_reuse_host_calls']=90
  with self.assertRaisesRegex(ValueError,'outside the forward graph'):self.validate()
 def test_missing_or_wrong_policy_refuses(self):
  del self.record['metadata_preparation_policy']
  with self.assertRaisesRegex(ValueError,'witness schema'):self.validate()
  self.record['metadata_preparation_policy']='native singleton per causal row'
  with self.assertRaisesRegex(ValueError,'outside the forward graph'):self.validate()
 def test_incomplete_capture_or_preparation_count_refuses(self):
  for name,value in [('capture_host_calls',16),('metadata_build_host_calls',1),('metadata_reuse_host_calls',95),('metadata_capture_build_host_calls',1)]:
   previous=self.record[name];self.record[name]=value
   with self.subTest(field=name),self.assertRaisesRegex(ValueError,'outside the forward graph'):self.validate()
   self.record[name]=previous
 def test_negative_and_boolean_metadata_counts_refuse(self):
  for name in ('metadata_build_host_calls','metadata_reuse_host_calls','metadata_capture_build_host_calls'):
   previous=self.record[name]
   for value in (-1,True):
    self.record[name]=value
    with self.subTest(field=name,value=value),self.assertRaisesRegex(ValueError,'metadata reuse witness counts'):self.validate()
   self.record[name]=previous
 def test_bound_session_source_and_worker_checks_still_refuse(self):
  for name,value in [('spec_sha256','f'*64),('worker_identity_sha256','f'*64),('adapter_source',{}),('indices_elements',[8*32768,16*32768])]:
   previous=self.record[name];self.record[name]=value
   with self.subTest(field=name),self.assertRaises(ValueError):self.validate()
   self.record[name]=previous
 def test_disabled_reference_has_no_preparation_claim(self):
  self.enabled=False;self.record.update(enabled=False,block_size=None,adapter_source=None,layers=[],buffers=0,eligible_host_calls=0,capture_host_calls=0,metadata_build_host_calls=0,metadata_reuse_host_calls=0,metadata_capture_build_host_calls=0,indices_elements=[],metadata_preparation_policy=None);self.assertIs(self.validate(),self.record)
  self.record['metadata_preparation_policy']='outside-forward-graph-v1'
  with self.assertRaisesRegex(ValueError,'ordinary reference reported'):self.validate()



class PreservedFrontends(unittest.TestCase):
 def test_native_decode_frontends_and_fallback_eligibility_are_unchanged(self):
  import ast
  old=ast.parse((ROOT/'serving/legacy/review/verify_decode_adapter.py').read_bytes())
  new=ast.parse((ROOT/'serving/spry_serving/pre_replay_adapter.py').read_bytes())
  for name in ('eligible','source_module','checked_method','checked_noop'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
  method=lambda tree,name:next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Adapter' for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
  for name in ('snapshot','restore'):self.assertEqual(ast.dump(method(old,name)),ast.dump(method(new,name)),name)
  calls=lambda tree:[ast.dump(n) for n in ast.walk(method(tree,'decode')) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='backend.decode_attention_fwd']
  self.assertEqual(calls(old),calls(new));self.assertEqual(len(calls(new)),2)
 def test_prepare_refuses_to_record_metadata_into_a_captured_graph(self):
  # The only Tensor preparation entry has an explicit capture refusal; the
  # graph simulation also proves captured replay contains attention work only.
  import ast
  tree=ast.parse((ROOT/'serving/spry_serving/pre_replay_adapter.py').read_bytes())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Adapter')
  decode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode')
  calls=[ast.unparse(n.func) for n in ast.walk(decode) if isinstance(n,ast.Call)]
  for name in ('torch.add','torch.cumsum','torch.empty','backend.get_num_kv_splits','backend.kv_index_translator.fill_packed_read_stream'):
   self.assertNotIn(name,calls)

if __name__=='__main__':unittest.main()
