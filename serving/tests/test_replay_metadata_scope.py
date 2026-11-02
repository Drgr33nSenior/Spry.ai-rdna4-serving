"""Captured verify maps must survive native eager metadata object replacement."""
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_pre_replay_metadata as support
fixed=support.load('fixed_replay_scope',HERE.parent/'spry_serving/common_replay_scope_adapter.py')


class ReplayMetadataScope(unittest.TestCase):
    def setUp(self):
        native=ModuleType('pre_replay_decode');native.decode_attention_fwd=support.decode
        context=patch.dict(sys.modules,pre_replay_decode=native);context.start();self.addCleanup(context.stop)
    def test_retained_candidate_leaves_stale_map_after_eager_metadata_replacement(self):
        for bs in (1,2):
            fixture=support.Fixture(support.candidate,bs);graph=fixture.capture();before=fixture.metadata()
            fixture.backend.forward_metadata=SimpleNamespace(max_extend_len=1453,kv_indptr=fixture.ptr)
            fixture.state([127]if bs==1 else [127,104]);fixture.backend.init_forward_metadata_out_graph(fixture.batch)
            for operation in graph:operation()
            self.assertEqual(fixture.metadata(),before)
    def test_fixed_maps_match_original_graph_after_eager_metadata_replacement(self):
        for bs in (1,2):
            control=support.Fixture(support.baseline,bs);candidate=support.Fixture(fixed,bs)
            graphs=[control.capture(),candidate.capture()]
            for prefixes in ([127]if bs==1 else [127,104],[31]if bs==1 else [31,8]):
                for fixture,graph in zip((control,candidate),graphs):
                    fixture.backend.forward_metadata=SimpleNamespace(max_extend_len=1453,kv_indptr=fixture.ptr)
                    fixture.state(prefixes);fixture.backend.init_forward_metadata_out_graph(fixture.batch)
                    for operation in graph:operation()
                self.assertEqual(control.metadata(),candidate.metadata());self.assertEqual(control.out.trace,candidate.out.trace)
            self.assertEqual(candidate.adapter.metadata_capture_build_host_calls,0)
            self.assertEqual(len(graphs[1]),48)
    def test_replay_without_capture_refuses(self):
        f=support.Fixture(fixed);f.state([31]);f.backend.init_forward_metadata(f.batch)
        with self.assertRaisesRegex(ValueError,'captured verify metadata is unavailable'):f.backend.init_forward_metadata_out_graph(f.batch)
    def test_invalid_captured_metadata_refuses(self):
        f=support.Fixture(fixed);f.capture();f.adapter.graph_metadata[id(f.backend),8].max_extend_len=7
        with self.assertRaisesRegex(ValueError,'captured verify metadata is incompatible'):f.backend.init_forward_metadata_out_graph(f.batch)
    def test_failed_recapture_cannot_reuse_previous_metadata(self):
        f=support.Fixture(fixed);f.capture();f.backend.initializer_failure=True
        with self.assertRaisesRegex(RuntimeError,'initializer failure'):f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
        self.assertNotIn((id(f.backend),8),f.adapter.graph_metadata)
        f.backend.initializer_failure=False
        with self.assertRaisesRegex(ValueError,'captured verify metadata is unavailable'):f.backend.init_forward_metadata_out_graph(f.batch)
    def test_each_capture_size_uses_its_own_view_after_eager_rebinding(self):
        f=support.Fixture(fixed);graphs={1:f.capture()}
        names=('batch','q','k','v','out','ptr')
        views={1:({name:getattr(f,name)for name in names},f.backend.arguments)}
        f.batch=SimpleNamespace(batch_size=2,forward_mode=SimpleNamespace(is_target_verify=lambda:True),
            req_pool_indices=support.Tensor(f.torch,2,values=[0,1]),out_cache_loc=support.Tensor(f.torch,16,values=range(1,17)))
        f.q,f.k,f.v,f.out=[support.Tensor(f.torch,shape,'torch.bfloat16')for shape in ((16,12,256),(16,2,256),(16,2,256),(16,12,256))]
        f.ptr=support.Tensor(f.torch,3,values=[0,0,0]);old=f.backend.arguments
        f.backend.arguments=(f.q,f.k,f.v,f.out,old[4],old[5],support.Tensor(f.torch,3,values=[0,8,16]),f.ptr,*old[8:])
        graphs[2]=f.capture();views[2]=({name:getattr(f,name)for name in names},f.backend.arguments)
        self.assertEqual(set(f.adapter.graph_metadata),{(id(f.backend),8),(id(f.backend),16)})
        for bs in (1,2,1,2):
            values,args=views[bs]
            for name,value in values.items():setattr(f,name,value)
            f.backend.arguments=args;f.backend.forward_metadata=SimpleNamespace(max_extend_len=1453,kv_indptr=None)
            prefixes=[127]if bs==1 else[127,104];f.state(prefixes);f.backend.init_forward_metadata_out_graph(f.batch)
            for operation in graphs[bs]:operation()
            lengths=[prefix+n for prefix in prefixes for n in range(1,9)];ptr=[0]
            for length in lengths:ptr.append(ptr[-1]+length)
            self.assertEqual(f.metadata()[0],tuple(ptr))
    def test_non_verify_native_initialization_keeps_its_fallback(self):
        f=support.Fixture(fixed);f.batch.forward_mode.is_target_verify=lambda:False
        f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
        self.assertEqual(f.adapter.graph_metadata,{})
        self.assertEqual(f.adapter.metadata_scopes,{})
    def test_failed_capture_preparation_leaves_no_registered_view(self):
        f=support.Fixture(fixed);f.capture()
        with patch.object(f.adapter,'prepare',side_effect=RuntimeError('preparation failure')):
            with self.assertRaisesRegex(RuntimeError,'preparation failure'):f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
        self.assertNotIn((id(f.backend),8),f.adapter.graph_metadata)
    def test_restore_clears_capture_references_and_preserves_hooks(self):
        f=support.Fixture(fixed);f.capture();self.assertTrue(f.adapter.graph_metadata);f.adapter.restore();self.assertFalse(f.adapter.graph_metadata)


if __name__=='__main__':unittest.main()
