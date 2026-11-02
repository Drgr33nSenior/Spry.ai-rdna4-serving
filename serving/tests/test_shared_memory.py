"""Memory source contracts with sparse, standard-library tensor stand-ins.

These checks do not establish real tensor aliasing or native arithmetic.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spry_serving import fp8_head as module, memory_scheme, memory_p1


class Weight:
    def __init__(self):
        self.shape = (module.N, module.K)
        self.dtype = 'bf16'
        self.device = SimpleNamespace(type='cuda', index=1)
        self.requires_grad, self._version, self.pointer = False, 0, 256
        self.bad_rows = set()
    def data_ptr(self): return self.pointer
    def is_contiguous(self): return True
    def __getitem__(self, key): return not any(key.start <= row < key.stop for row in self.bad_rows)


class Scalar:
    def __init__(self, torch, value): self.torch, self.value = torch, value
    def all(self): return self
    def item(self):
        self.torch.scalar_reads += 1
        return self.value


class Torch:
    bool, bfloat16, float8_e4m3fn, float32 = 'bool', 'bf16', 'fp8', 'fp32'
    uint8 = 'uint8'
    def __init__(self):
        self.scalar_reads = self.scans = self.allocations = 0
        self.capturing = False
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)
    def ones(self, shape, **kwargs): return Scalar(self, True)
    def isfinite(self, value):
        self.scans += 1
        return Scalar(self, value)
    def logical_and(self, a, b, *, out): out.value = a.value and b.value
    def empty(self, shape, **kwargs):
        self.allocations += 1
        return SimpleNamespace(shape=shape, **kwargs)


class Ops:
    def __init__(self): self.calls, self.fail = 0, False
    def quantize_weight_preallocated(self, weight, b8, scales, **kwargs):
        self.calls += 1
        if self.fail: raise RuntimeError('quantizer failure')


class SharedHead(unittest.TestCase):
    def setUp(self): self.head, self.torch, self.ops = SimpleNamespace(weight=Weight()), Torch(), Ops()
    def call(self, ops=None):
        return module.shared_fp8_head(self.head, self.torch, self.ops if ops is None else ops, expected_device=1)
    def test_two_adapters_reuse_exact_objects_and_quantize_once(self):
        b8, scales, created = self.call()
        other = Ops()
        b8b, scalesb, createdb = self.call(other)
        self.assertTrue(created); self.assertFalse(createdb)
        self.assertIs(b8, b8b); self.assertIs(scales, scalesb)
        self.assertEqual((self.ops.calls, other.calls, self.torch.allocations), (1, 0, 2))
    def test_scan_is_bounded_and_reads_one_scalar(self):
        self.call(); self.call()
        self.assertEqual((self.torch.scans, self.torch.scalar_reads), (module.N // 128, 1))
    def test_nonfinite_tail_refuses_before_allocating_or_publishing(self):
        self.head.weight.bad_rows = {module.N - 1}
        with self.assertRaisesRegex(ValueError, 'nonfinite'): self.call()
        self.assertEqual((self.ops.calls, self.torch.allocations, self.torch.scalar_reads), (0, 0, 1))
        self.assertFalse(hasattr(self.head, module._CACHE))
    def test_pointer_version_and_object_changes_refuse(self):
        for kind in ('version', 'pointer', 'object'):
            self.setUp(); self.call()
            if kind == 'version': self.head.weight._version += 1
            elif kind == 'pointer': self.head.weight.pointer += 256
            else: self.head.weight = Weight()
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'binding changed'): self.call()
            self.assertEqual(self.ops.calls, 1)
    def test_quantizer_failure_does_not_publish_cache(self):
        self.ops.fail = True
        with self.assertRaisesRegex(RuntimeError, 'quantizer failure'): self.call()
        self.assertFalse(hasattr(self.head, module._CACHE))
    def test_capture_and_bad_device_refuse_before_scanning(self):
        self.torch.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'precede graph'): self.call()
        self.torch.capturing = False
        for device in (True, -1, 2, '1'):
            with self.assertRaises(ValueError):
                module.shared_fp8_head(self.head, self.torch, self.ops, expected_device=device)
        self.assertEqual(self.torch.scans, 0)
    def test_other_quantizer_or_corrupt_cache_refuses(self):
        self.call()
        class OtherOps(Ops): pass
        with self.assertRaisesRegex(RuntimeError, 'binding changed'): self.call(OtherOps())
        setattr(self.head, module._CACHE, ())
        with self.assertRaisesRegex(RuntimeError, 'binding changed'): self.call()


class Tensor:
    """Sparse storage descriptors; values are tags, never native tensor bytes."""
    serial = 100
    def __init__(self, shape, *, tag='equal', pointer=None):
        self.shape, self.tag, self.dtype = shape, tag, 'uint8'
        self.device, self._version = SimpleNamespace(type='cuda', index=1), 0
        Tensor.serial += 1
        self.pointer = Tensor.serial * (1 << 28) if pointer is None else pointer
    def data_ptr(self): return self.pointer
    def is_contiguous(self): return True
    def numel(self): return self.shape[0] * self.shape[1]
    def element_size(self): return 1
    def contiguous(self): return self
    def detach(self): return copy.copy(self)
    def __getitem__(self, key):
        return Tensor((key.stop - key.start, self.shape[1]), tag=self.tag,
                      pointer=self.pointer + key.start * self.shape[1])
    def set_(self, view):
        self.pointer, self.shape, self.tag = view.pointer, view.shape, view.tag
        self._version += 1


class PackedViews(unittest.TestCase):
    def setUp(self):
        self.torch = Torch(); self.route = memory_p1.PrefillRoute(self.torch, memory_p1.SETTING)
        self.torch.cat = lambda values, dim: Tensor((sum(t.shape[0] for t in values), values[0].shape[1]))
        self.torch.equal = lambda a, b: a.shape == b.shape and a.tag == b.tag
        from contextlib import nullcontext
        self.torch.no_grad = nullcontext
        self.layers = [SimpleNamespace(weight=Tensor((n, 2560)), weight_scale=Tensor((n, 160)),
                      _isolated_mxfp4_loaded=True) for n in (8192, 48)]
        self.route.validated[id(self.layers[0])] = self.route._identity(self.layers[0])
        self.ops = SimpleNamespace(validate_geometry=lambda *args: None)
        contract = ModuleType('mtp_contract'); contract.settings = lambda: dict(instrumentation='production')
        self.context = patch.dict(sys.modules, mtp_contract=contract)
        self.context.start(); self.addCleanup(self.context.stop)
        with patch.object(memory_scheme, '_p1_route', return_value=self.route):
            cls = memory_scheme.build_scheme_class(object, self.torch, self.ops, {}, {})
        self.scheme = object.__new__(cls); self.scheme.input_size_per_partition = 5120
    def test_parameters_retain_objects_classes_and_metadata_and_p1_rebinds(self):
        originals = [(layer.weight, layer.weight_scale) for layer in self.layers]
        self.layers[0].weight.loader_metadata = object()
        metadata = self.layers[0].weight.loader_metadata
        packed, scales, widths, k = self.scheme.pack_rows(self.layers)
        self.assertEqual((widths, k), ((8192, 48), 5120))
        for index, layer in enumerate(self.layers):
            offset = 0 if index == 0 else 8192
            self.assertIs(layer.weight, originals[index][0]); self.assertIs(layer.weight_scale, originals[index][1])
            self.assertEqual(layer.weight.data_ptr(), packed.data_ptr() + offset * 2560)
            self.assertEqual(layer.weight_scale.data_ptr(), scales.data_ptr() + offset * 160)
            self.assertIs(layer._spry_packed_view['p1_identity_rebound'], index == 0)
            self.assertNotIn('weight_sha256', layer._spry_packed_view)
        self.assertIs(self.layers[0].weight.loader_metadata, metadata)
        self.assertEqual(self.route.validated[id(self.layers[0])], self.route._identity(self.layers[0]))
    def test_missing_validation_stale_identity_and_byte_change_refuse(self):
        for kind in ('missing', 'version', 'bytes'):
            self.setUp()
            if kind == 'missing': self.route.validated.clear()
            elif kind == 'version': self.layers[0].weight._version += 1
            else: self.layers[0].weight.tag = 'different'
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                self.scheme.pack_rows(self.layers)
    def test_capture_refuses_p1_rebind(self):
        self.torch.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'precede graph'): self.scheme.pack_rows(self.layers)
    def test_apply_accepts_rebound_identity_and_rejects_later_change(self):
        self.scheme.pack_rows(self.layers)
        self.route._runtime = lambda device: (None, None)
        self.route._views = lambda *args: (None,) * 5
        self.route._check = lambda *args: None
        self.route._expand = self.route._quantize = self.route._gemm = lambda *args: None
        x = SimpleNamespace(shape=(1024, 5120), device=SimpleNamespace(index=1))
        self.route.apply(self.layers[0], x, 8192, 5120)
        self.layers[0].weight._version += 1
        with self.assertRaisesRegex(RuntimeError, 'not validated'): self.route.apply(self.layers[0], x, 8192, 5120)
    def test_compute_frontends_and_binding_guards_are_unchanged(self):
        for before, after, cls, names in (
            ('p1_prefill.py','memory_p1.py','PrefillRoute',('_expand','_quantize','_gemm','apply')),
            ('d3b_target_head_adapter.py','memory_target_head.py','TargetHeadFP8',('_bound','_eligible','__call__')),
            ('d3b_head_adapter.py','memory_draft_head.py','D3bDraftHeadMethod',('_bound','_eligible','apply')),
            ('isolated_scheme.py','memory_scheme.py','IsolatedPackedMXFP4',('apply_weights','apply_packed'))):
            paths = (ROOT/'legacy/review'/before, ROOT/'spry_serving'/after)
            methods = []
            for path in paths:
                tree = ast.parse(path.read_bytes())
                owner = next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name==cls)
                methods.append({node.name:ast.dump(node) for node in owner.body if isinstance(node,ast.FunctionDef)})
            for name in names: self.assertEqual(methods[0][name], methods[1][name])


def memory_record(rank=1):
    rows = []
    for n in range(48):
        pair = {}
        for name, width, offset in (('qkvz',8192,0),('ba',48,8192)):
            base = (n + 1) * (1 << 28)
            pair[name] = dict(weight_offset_bytes=offset*2560, scale_offset_bytes=offset*160,
                weight_base_pointer=base, scale_base_pointer=base+(1<<26),
                weight_pointer=base+offset*2560, scale_pointer=base+(1<<26)+offset*160,
                p1_identity_rebound=name=='qkvz', equal_at_startup=True,
                rebound_payload_bytes=width*(2560+160),weight_sha256='a'*64,scale_sha256='b'*64)
        rows.append(pair)
    sharing=dict(fp8_id=1,scales_id=2,fp8_pointer=256,scales_pointer=512)
    return dict(schema=1,policy='shared-head-packed-views-v1',device=rank,
        head=dict(equal=True,transient_quantizations=1,fp8_sha256='c'*64,scales_sha256='d'*64),
        sharing=dict(target=sharing,draft=dict(sharing)),packed=rows,before_graph_capture=True,
        predicted_released_payload_bytes=1711668800)


class MemoryWitness(unittest.TestCase):
    def test_full_witness_and_predicted_bytes(self):
        record=memory_record();self.assertIs(module.validate_memory_recovery(record,rank=1),record)
        self.assertEqual(48*8240*(2560+160)+module.N*module.K+4*(module.N//128)*(module.K//128),1711668800)
    def test_incomplete_or_forged_memory_witnesses_refuse(self):
        mutations=[lambda r:r['packed'].pop(), lambda r:r['head'].update(equal=False),
            lambda r:r['head'].update(transient_quantizations=True),lambda r:r['sharing']['draft'].update(fp8_id=3),
            lambda r:r['packed'][0]['qkvz'].update(p1_identity_rebound=False),
            lambda r:r['packed'][0]['ba'].update(weight_offset_bytes=1),
            lambda r:r['packed'][0]['ba'].update(equal_at_startup=False),
            lambda r:r['packed'][0]['ba'].update(weight_sha256='bad'),
            lambda r:r['packed'].__setitem__(1,copy.deepcopy(r['packed'][0])),
            lambda r:r.update(predicted_released_payload_bytes=1711668801),
            lambda r:r.update(device=True),lambda r:r.pop('sharing')]
        for index, change in enumerate(mutations):
            record=memory_record();change(record)
            with self.subTest(index=index),self.assertRaises(ValueError):module.validate_memory_recovery(record,rank=1)


class ByteTensor:
    def __init__(self,data):self.data=bytes(data)
    def is_contiguous(self):return True
    def detach(self):return self
    def view(self,*args):return self
    def reshape(self,*args):return self
    def numel(self):return len(self.data)
    def __getitem__(self,key):return ByteTensor(self.data[key])
    def cpu(self):return self
    def numpy(self):return self.data


class StartupByteProof(unittest.TestCase):
    def setUp(self):
        self.b8,self.scales=ByteTensor(b'fp8-bytes'),ByteTensor(b'scale-bytes')
        self.torch=Torch();self.transient=[]
        def empty_like(value):
            result=ByteTensor(b'');self.transient.append(result);return result
        self.torch.empty_like=empty_like
        self.torch.equal=lambda a,b:a.data==b.data
        self.changed=False
        def quantize(weight,b8,scales,**kwargs):
            b8.data=b'changed' if self.changed else self.b8.data;scales.data=self.scales.data
        self.ops=SimpleNamespace(quantize_weight_preallocated=quantize)
    def test_chunked_hash_and_transient_byte_equality_use_actual_helpers(self):
        with patch.object(module,'HASH_CHUNK_BYTES',3), \
             patch.object(module,'shared_fp8_head',return_value=(self.b8,self.scales,False)):
            proof=module.verify_shared_fp8_head(SimpleNamespace(weight=object()),self.torch,self.ops,expected_device=1)
        self.assertEqual(proof,dict(equal=True,transient_quantizations=1,
            fp8_sha256=hashlib.sha256(self.b8.data).hexdigest(),scales_sha256=hashlib.sha256(self.scales.data).hexdigest()))
        self.assertEqual(len(self.transient),2)
    def test_mismatched_transient_or_capture_hash_refuses(self):
        self.changed=True
        with patch.object(module,'shared_fp8_head',return_value=(self.b8,self.scales,False)):
            with self.assertRaisesRegex(ValueError,'quantization bytes differ'):
                module.verify_shared_fp8_head(SimpleNamespace(weight=object()),self.torch,self.ops,expected_device=1)
        self.torch.capturing=True
        with self.assertRaisesRegex(ValueError,'outside capture'):module.tensor_sha256(self.b8,self.torch)


class HeadAdapters(unittest.TestCase):
    def setUp(self):
        SharedHead.setUp(self)
        self.context=patch.dict(sys.modules,spry_fp8_head=module)
        self.context.start();self.addCleanup(self.context.stop)
        def load(name):
            spec=importlib.util.spec_from_file_location(name,ROOT/('spry_serving/'+name+'.py'))
            value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
        self.target_module=load('memory_target_head');self.draft_module=load('memory_draft_head')
    def test_actual_adapter_constructors_share_allocation_and_keep_both_guards(self):
        processor=SimpleNamespace()
        target=self.target_module.TargetHeadFP8(processor,self.head,self.torch,self.ops,1,lambda *a:None)
        processor._compute_lm_head=target
        other=Ops();draft=self.draft_module.D3bDraftHeadMethod(self.head,self.torch,other,1,module.N,1,lambda *a:None)
        proxy=SimpleNamespace(quant_method=draft);draft.proxy_id=id(proxy)
        self.assertIs(target.b8,draft.b8);self.assertIs(target.scales,draft.scales)
        self.assertEqual((self.ops.calls,other.calls),(1,0))
        target._bound();draft._bound(proxy);self.head.weight._version+=1
        with self.assertRaisesRegex(RuntimeError,'binding changed'):target._bound()
        with self.assertRaisesRegex(RuntimeError,'binding changed'):draft._bound(proxy)
        self.assertEqual(self.draft_module.tensor_bytes()['added_persistent_bytes_per_rank'],0)
        self.assertEqual(self.target_module.tensor_bytes()['added_persistent_bytes_per_rank'],635854400)
    def test_draft_first_refuses_without_scanning_or_quantizing(self):
        with self.assertRaisesRegex(RuntimeError,'installed target allocation'):
            self.draft_module.D3bDraftHeadMethod(self.head,self.torch,self.ops,1,module.N,1,lambda *a:None)
        self.assertEqual((self.torch.scans,self.ops.calls,self.torch.allocations),(0,0,0))


class StartupHook(unittest.TestCase):
    def setUp(self):
        from spry_serving import memory_post_load
        self.hook=memory_post_load;self.events=[]
        contract=ModuleType('mtp_contract');contract.settings=lambda:dict(instrumentation='production',d3b_target_head=True,d3b_draft_head_arm='both',gdn_layout_both=True);contract.dense_mxfp4_enabled=lambda:True
        target=ModuleType('d3b_target_head_adapter');target.install=lambda *a,**k:self.events.append('target')
        draft=ModuleType('d3b_head_adapter');draft.install=lambda *a,**k:self.events.append('draft')
        torch=ModuleType('torch');torch.cuda=SimpleNamespace(empty_cache=lambda:self.events.append('release-cache'))
        self.context=patch.dict(sys.modules,mtp_contract=contract,d3b_target_head_adapter=target,d3b_head_adapter=draft,torch=torch)
        self.context.start();self.addCleanup(self.context.stop)
    def runner(self,draft=False,count=48):
        events=self.events
        class Qwen3_5GatedDeltaNet:
            def finalize_packed_mxfp4_in_proj(self):events.append('pack');return True
        return SimpleNamespace(is_draft_worker=draft,startup_weight_load=None,gpu_id=1,tp_rank=1,
            model=SimpleNamespace(modules=lambda:[Qwen3_5GatedDeltaNet() for _ in range(count)]))
    def test_pack_all_layers_then_target_then_return_cache_without_diagnostics(self):
        runner=self.runner();self.hook.run(runner)
        self.assertEqual(self.events,['pack']*48+['target','release-cache'])
        self.assertFalse(hasattr(runner,'_spry_memory_recovery'))
        with self.assertRaisesRegex(RuntimeError,'single-use'):self.hook.run(runner)
    def test_draft_only_and_incomplete_load_refusal(self):
        runner=self.runner(True);self.hook.run(runner);self.assertEqual(self.events,['draft'])
        runner=self.runner();runner.startup_weight_load=object()
        with self.assertRaisesRegex(RuntimeError,'synchronous'):self.hook.run(runner)
        self.assertEqual(self.events,['draft'])
    def test_missing_layer_refuses_before_any_pack_or_head(self):
        with self.assertRaisesRegex(RuntimeError,'all 48'):self.hook.run(self.runner(count=47))
        self.assertEqual(self.events,[])


if __name__ == '__main__': unittest.main()
