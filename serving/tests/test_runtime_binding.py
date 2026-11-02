"""Startup-bound runtime resolution with inert modules and tensor descriptors."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spry_serving import ops
from spry_serving import scheme as scheme_source
import spry_serving


class Tensor:
    def __init__(self,shape,dtype,device,pointer,stride=None):
        self.shape=shape;self.dtype=dtype;self.device=SimpleNamespace(type='cuda',index=device)
        self.pointer=pointer;self.layout=stride or (shape[1],1)
    def stride(self,axis=None):return self.layout if axis is None else self.layout[axis]
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.pointer)


class Kernel:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def launch(*args,**kwargs):self.calls.append((grid,args,kwargs));return 'launch-result'
        return launch


class RuntimeBinding(unittest.TestCase):
    def setUp(self):
        ops._BOUND_RUNTIMES.clear();self.addCleanup(ops._BOUND_RUNTIMES.clear)
        self.device=0
        self.torch=ModuleType('torch');self.torch.cuda=SimpleNamespace(current_device=lambda:self.device)
        self.torch.bfloat16=object();self.torch.uint8=object()
        self.torch.get_default_dtype=lambda:self.torch.bfloat16
        self.target=Mock(return_value=SimpleNamespace(backend='hip',arch='gfx1201'))
        self.triton=ModuleType('triton');self.triton.runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(get_current_target=self.target)))
        self.triton.cdiv=lambda a,b:(a+b-1)//b
        self.kernels=ModuleType('spry_serving.kernels')
        self.kernels._dynamic_mxfp4_quant_kernel=Kernel();self.kernels._gemm_afp4wfp4_kernel=Kernel()
        self.patch=patch.dict(sys.modules,torch=self.torch,triton=self.triton,**{'spry_serving.kernels':self.kernels})
        self.patch.start();self.addCleanup(self.patch.stop)
        attr=patch.object(spry_serving,'kernels',self.kernels,create=True);attr.start();self.addCleanup(attr.stop)

    def operands(self,m=8,n=8192,k=5120,device=0):
        t=lambda shape,dtype,pointer,stride=None:Tensor(shape,dtype,device,pointer,stride)
        return (t((m,k),self.torch.bfloat16,1),t((m,k//2),self.torch.uint8,2),
            t((m,k//32),self.torch.uint8,3,(1,m)),t((n,k//2),self.torch.uint8,4),
            t((n,k//32),self.torch.uint8,5),t((m,n),self.torch.bfloat16,6))

    def test_cold_forward_refuses_without_resolving_or_launching(self):
        x,packed,scales,*_=self.operands()
        with self.assertRaisesRegex(ValueError,'bound before serving'):
            ops.quantize_preallocated(x,packed,scales,expected_device=0)
        self.target.assert_not_called();self.assertEqual(self.kernels._dynamic_mxfp4_quant_kernel.calls,[])

    def test_repeated_calls_use_one_admitted_runtime_without_requerying_the_driver(self):
        values=ops.bind_runtime(0);self.assertEqual(values,(self.torch,self.triton,self.kernels))
        x,packed,scales,w,ws,y=self.operands()
        with patch.object(ops,'_resolve_runtime',side_effect=AssertionError('forward attempted runtime setup')):
            for _ in range(3):
                self.assertEqual(ops.quantize_preallocated(x,packed,scales,expected_device=0),'launch-result')
                self.assertEqual(ops.gemm_preallocated(packed,w,scales,ws,y,expected_device=0),'launch-result')
                self.assertIs(ops.bind_runtime(0),values)
        self.target.assert_called_once();self.assertEqual(len(ops._BOUND_RUNTIMES),1)

    def test_each_device_is_bound_once_and_current_card_is_checked_on_every_call(self):
        first=ops.bind_runtime(0);self.device=1;second=ops.bind_runtime(1)
        self.assertEqual(len(ops._BOUND_RUNTIMES),2);self.assertEqual(self.target.call_count,2)
        self.assertIs(ops.runtime(1),second)
        with self.assertRaisesRegex(ValueError,'wrong current card'):ops.runtime(0)
        self.device=0;self.assertIs(ops.runtime(0),first)
        with self.assertRaisesRegex(ValueError,'wrong current card'):ops.bind_runtime(1)

    def test_failed_architecture_and_device_checks_do_not_publish_a_binding(self):
        for target in (SimpleNamespace(backend='cuda',arch='gfx1201'),SimpleNamespace(backend='hip',arch='gfx1100')):
            self.target.return_value=target
            with self.assertRaisesRegex(ValueError,'gfx1201 HIP'):ops.bind_runtime(0)
            self.assertEqual(ops._BOUND_RUNTIMES,{})
        for device in (True,-1,2,'0',0.0):
            with self.subTest(device=device),self.assertRaises(ValueError):ops.bind_runtime(device)
        self.device=1
        with self.assertRaisesRegex(ValueError,'wrong current card'):ops.bind_runtime(0)
        self.assertEqual(ops._BOUND_RUNTIMES,{})

    def test_tensor_card_layout_and_storage_refusals_still_precede_launch(self):
        ops.bind_runtime(0)
        for kind in ('device','dtype','stride','alias'):
            x,packed,scales,*_=self.operands()
            if kind=='device':packed.device.index=1
            if kind=='dtype':packed.dtype=self.torch.bfloat16
            if kind=='stride':scales.layout=(scales.shape[1],1)
            if kind=='alias':packed.pointer=x.pointer
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                ops.quantize_preallocated(x,packed,scales,expected_device=0)
        self.assertEqual(self.kernels._dynamic_mxfp4_quant_kernel.calls,[])

    def test_kernel_frontend_bodies_and_tile_policy_are_byte_identical_to_the_control(self):
        before=ast.parse((ROOT/'legacy/review/ops.py').read_bytes());after=ast.parse((ROOT/'spry_serving/ops.py').read_bytes())
        for name in ('validate_geometry','_tensor','_distinct_storage','quantize_preallocated','gemm_preallocated'):
            left=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
            right=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
            with self.subTest(name=name):self.assertEqual(ast.dump(left),ast.dump(right))
        config=lambda tree:next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
        self.assertEqual(ast.dump(config(before)),ast.dump(config(after)))

    def test_actual_worker_authenticates_operator_without_binding_the_parent_device(self):
        temporary=tempfile.TemporaryDirectory(prefix='runtime-binding-worker-');self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name).resolve()
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module;spec.loader.exec_module(module);return module
        worker=load('runtime_binding_worker',ROOT/'spry_serving/worker_entry.py')
        methods=lambda names:''.join('    def '+name+'(self,*args):pass\n' for name in names)
        texts={
            'loader_weight_utils':'def get_quant_config(*args):pass\n',
            'model_loader':'# External native loader seam.\n',
            'quark_config':"class QuarkConfig:\n"+methods(('from_config','get_linear_scheme','get_quant_method'))+
                "class QuarkLinearMethod:\n"+methods(('create_weights','process_weights_after_loading','apply')),
            'quark_mxfp4_scheme':"class QuarkW4A4MXFP4:\n"+methods(('create_weights',)),
            'quark_parameter':'# External native parameter seam.\n',
            'mxfp4_package':'# External package seam.\n',
            'mxfp4_kernels':'# No native kernels execute in this fixture.\n',
            'mxfp4_scheme':'def build_scheme_class(*args):raise Stopped(args)\n',
            'mxfp4_selection':'def build_config_class(*args):pass\nclass Selection:\n'+methods(('__init__','check','__enter__','__exit__'))}
        modules={}
        for role,text in texts.items():
            path=directory/(role+'.py');path.write_text(text);modules[role]=load('binding_fixture_'+role,path)
        modules['model_loader'].get_quant_config=modules['loader_weight_utils'].get_quant_config
        modules['mxfp4_ops']=ops;modules['mxfp4_prefill_reference']=sys.modules['spry_serving.prefill_reference']
        class Stopped(Exception):pass
        modules['mxfp4_scheme'].Stopped=Stopped
        sources={role:dict(module=module.__name__,path=module.__file__,
                           sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()) for role,module in modules.items()}
        by_name={module.__name__:module for module in modules.values()}
        contract=SimpleNamespace(dense_mxfp4_enabled=lambda:True,mxfp4_sources=Mock())
        config=dict(quantization_config=dict(global_quant_config=dict(weight={},input_tensors={})))
        with patch.object(worker,'_validate_precision_config',return_value=config),\
                patch.object(worker,'_source_module',side_effect=lambda source:by_name[source['module']]),\
                patch.object(worker,'_native_module',side_effect=lambda source:by_name[source['module']]):
            with self.assertRaises(Stopped) as caught:worker._install_mxfp4_selection(dict(sources=sources),contract)
            self.assertIs(caught.exception.args[0][2],ops)
            self.assertEqual(ops._BOUND_RUNTIMES,{})
            self.target.assert_not_called()

    def test_scheme_construction_binds_rank_one_after_native_device_selection(self):
        with patch.object(scheme_source,'_p1_route',return_value=None):
            cls=scheme_source.build_scheme_class(object,self.torch,ops,{'weight':'pinned'},{'input':'pinned'})
        self.assertEqual(ops._BOUND_RUNTIMES,{})
        self.target.assert_not_called()
        # ModelRunner selects rank one's device after the worker's class
        # factory and before the native loader constructs scheme instances.
        self.device=1
        first=cls({'weight':'pinned'},{'input':'pinned'})
        second=cls({'weight':'pinned'},{'input':'pinned'})
        self.assertEqual(set(ops._BOUND_RUNTIMES),{1});self.target.assert_called_once()
        self.assertIs(first.out_dtype,self.torch.bfloat16);self.assertIs(second.out_dtype,first.out_dtype)
        x,packed,scales,w,ws,y=self.operands(device=1)
        ops.quantize_preallocated(x,packed,scales,expected_device=1)
        ops.gemm_preallocated(packed,w,scales,ws,y,expected_device=1)
        self.target.assert_called_once()

    def test_scheme_spec_dtype_and_architecture_refusals_cannot_publish_a_binding(self):
        with patch.object(scheme_source,'_p1_route',return_value=None):
            cls=scheme_source.build_scheme_class(object,self.torch,ops,{'weight':'pinned'},{'input':'pinned'})
        with self.assertRaisesRegex(ValueError,'exact serialized'):cls({}, {})
        self.target.assert_not_called()
        with patch.object(self.torch,'get_default_dtype',return_value=self.torch.uint8),self.assertRaisesRegex(ValueError,'BF16 target'):
            cls({'weight':'pinned'},{'input':'pinned'})
        self.target.assert_not_called()
        self.target.return_value=SimpleNamespace(backend='hip',arch='gfx1100')
        with self.assertRaisesRegex(ValueError,'gfx1201 HIP'):cls({'weight':'pinned'},{'input':'pinned'})
        self.assertEqual(ops._BOUND_RUNTIMES,{})


if __name__=='__main__':unittest.main()
