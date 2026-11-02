"""Source-only target/draft startup hook and authenticated worker checks."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('spry_post_load',ROOT/'spry_serving/post_load.py')
hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)

class PostLoad(unittest.TestCase):
 def setUp(self):
  self.events=[];self.settings=dict(d3b_target_head={'mode':'selected'},d3b_draft_head_arm='both',gdn_layout_both=True)
  contract=ModuleType('mtp_contract');contract.settings=lambda:self.settings;contract.dense_mxfp4_enabled=lambda:True
  target=ModuleType('d3b_target_head_adapter');target.install=lambda model,**kw:self.install('target',model,kw)
  draft=ModuleType('d3b_head_adapter');draft.install=lambda model,**kw:self.install('draft',model,kw)
  self.context=patch.dict(sys.modules,dict(mtp_contract=contract,d3b_target_head_adapter=target,d3b_head_adapter=draft));self.context.start();self.addCleanup(self.context.stop)
 def install(self,role,model,kw):
  self.events.append((role,model,kw));return {'installed':role}
 def runner(self,draft=False,count=48):
  test=self
  class Qwen3_5GatedDeltaNet:
   def __init__(self,index):self.index=index
   def finalize_packed_mxfp4_in_proj(self):test.events.append(('pack',self.index));return True
  model=SimpleNamespace(modules=lambda:[Qwen3_5GatedDeltaNet(n) for n in range(count)])
  return SimpleNamespace(is_draft_worker=draft,model=model,startup_weight_load=None,gpu_id=1,tp_rank=1)
 def test_target_installs_once_then_packs_all_layers(self):
  r=self.runner();hook.run(r);self.assertEqual(r.d3b_target_head_inventory,{'installed':'target'});self.assertEqual(self.events[0],('target',r.model,dict(expected_device=1,expected_rank=1)));self.assertEqual(self.events[1:],[('pack',n) for n in range(48)])
 def test_draft_installs_only_the_attached_head(self):
  r=self.runner(True);hook.run(r);self.assertEqual(r.d3b_draft_head_inventory,{'installed':'draft'});self.assertEqual(self.events,[('draft',r.model,dict(expected_device=1,expected_rank=1))])
 def test_repeat_install_refuses(self):
  for draft in (False,True):
   with self.subTest(draft=draft):
    r=self.runner(draft);hook.run(r);before=list(self.events)
    with self.assertRaisesRegex(RuntimeError,'single-use'):hook.run(r)
    self.assertEqual(self.events,before)
 def test_incomplete_target_load_refuses_before_adapter(self):
  r=self.runner();r.startup_weight_load=object()
  with self.assertRaisesRegex(RuntimeError,'synchronous'):hook.run(r)
  self.assertEqual(self.events,[])
 def test_missing_or_rejected_pack_aborts_startup(self):
  r=self.runner(count=47)
  with self.assertRaisesRegex(RuntimeError,'all 48'):hook.run(r)
  self.assertEqual(len(self.events),48)
  r=self.runner();r.model.modules=lambda:[SimpleNamespace()]
  with self.assertRaisesRegex(RuntimeError,'all 48'):hook.run(r)
 def test_adapter_failure_propagates_without_publish_or_pack(self):
  for draft,name in ((False,'d3b_target_head_adapter'),(True,'d3b_head_adapter')):
   r=self.runner(draft)
   with patch.object(sys.modules[name],'install',side_effect=RuntimeError('adapter failure')):
    with self.assertRaisesRegex(RuntimeError,'adapter failure'):hook.run(r)
   self.assertFalse(hasattr(r,'d3b_draft_head_inventory' if draft else 'd3b_target_head_inventory'))
  self.assertEqual(self.events,[])
 def test_inconsistent_selection_refuses_before_adapter(self):
  for key,value in [('d3b_target_head',None),('d3b_draft_head_arm','control'),('gdn_layout_both',False)]:
   with patch.dict(self.settings,{key:value}),self.assertRaisesRegex(ValueError,'selected W5'):hook.run(self.runner())
  self.assertEqual(self.events,[])
 def test_runner_keeps_other_methods_and_hook_precedes_capture(self):
  old=ast.parse((ROOT/'legacy/review/d3b_combined_model_runner.py').read_bytes());new=ast.parse((ROOT/'spry_serving/hook_runner.py').read_bytes())
  cls=lambda t:next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='ModelRunner')
  a,b=cls(old),cls(new)
  methods=lambda c:{x.name:ast.dump(x) for x in c.body if isinstance(x,ast.FunctionDef) and x.name!='init_cuda_graphs'}
  self.assertEqual(methods(a),methods(b))
  f=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name=='init_cuda_graphs');calls=[(x.lineno,ast.unparse(x.func)) for x in ast.walk(f) if isinstance(x,ast.Call)]
  hookline=next(n for n,name in calls if name=='spry_post_load');captureline=next(n for n,name in calls if name=='capture_cuda_graphs');self.assertLess(hookline,captureline)

class PostLoadSource(unittest.TestCase):
 def setUp(self):
  import hashlib,tempfile
  self.temporary=tempfile.TemporaryDirectory(prefix='post-load-source-');self.addCleanup(self.temporary.cleanup)
  self.path=Path(self.temporary.name).resolve()/'spry_post_load.py';self.path.write_bytes((ROOT/'spry_serving/post_load.py').read_bytes())
  self.source=dict(path=str(self.path),module='spry_post_load',sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
  self.contract=SimpleNamespace(POST_LOAD_SOURCE=dict(self.source));self.manifest=dict(sources=dict(post_load=dict(self.source)))
  spec=importlib.util.spec_from_file_location('post_load_worker_test',ROOT/'spry_serving/hook_worker.py');self.worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=self.worker;spec.loader.exec_module(self.worker);self.addCleanup(sys.modules.pop,spec.name,None)
  self.modules=patch.dict(sys.modules);self.modules.start();self.addCleanup(self.modules.stop);sys.modules.pop('spry_post_load',None)
 def test_worker_registers_the_source_before_scheduler_without_native_imports(self):
  module=self.worker._load_post_load_hook(self.manifest,self.contract)
  self.assertIs(sys.modules['spry_post_load'],module);self.assertEqual(module.__file__,str(self.path));self.assertTrue(callable(module.run))
  self.assertIs(self.worker._load_post_load_hook(self.manifest,self.contract),module)
 def test_missing_changed_or_unbound_source_refuses(self):
  with self.assertRaisesRegex(self.worker.WorkerEntryError,'binding differs'):
   self.worker._load_post_load_hook(dict(sources={}),self.contract)
  self.path.write_text('raise AssertionError("changed source executed")\n')
  with self.assertRaisesRegex(self.worker.WorkerEntryError,'source changed'):
   self.worker._load_post_load_hook(self.manifest,self.contract)
  self.assertNotIn('spry_post_load',sys.modules)
 def test_preloaded_ambient_or_rebound_callable_refuses(self):
  sys.modules['spry_post_load']=ModuleType('spry_post_load')
  with self.assertRaisesRegex(self.worker.WorkerEntryError,'origin differs'):
   self.worker._load_post_load_hook(self.manifest,self.contract)
  del sys.modules['spry_post_load'];module=self.worker._load_post_load_hook(self.manifest,self.contract)
  module.run=lambda runner:None
  with self.assertRaisesRegex(self.worker.WorkerEntryError,'callable/code/globals changed'):
   self.worker._load_post_load_hook(self.manifest,self.contract)

if __name__=='__main__':unittest.main()
