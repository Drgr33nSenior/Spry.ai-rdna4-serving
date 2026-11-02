"""The class-wide replay hook must preserve native draft attention metadata."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_replay_metadata_scope as scope

fixed=scope.support.load('fixed_target_replay_scope',HERE.parent/'spry_serving/common_target_replay_scope_adapter.py')
retained=scope.fixed


class TargetReplayScope(scope.ReplayMetadataScope):
    def setUp(self):
        super().setUp()
        context=patch.object(scope,'fixed',fixed);context.start();self.addCleanup(context.stop)

    def test_retained_class_hook_refuses_draft_capture(self):
        for bs in (1,2):
            f=scope.support.Fixture(retained,bs)
            f.backend.num_head,f.backend.num_kv_head=16,4
            with self.assertRaisesRegex(ValueError,'captured verify metadata is incompatible'):
                f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)

    def test_draft_capture_and_replay_delegate_without_adapter_state(self):
        for bs in (1,2):
            f=scope.support.Fixture(fixed,bs)
            f.backend.num_head,f.backend.num_kv_head=16,4
            for capture in (True,False,True,False):
                f.backend.init_forward_metadata_out_graph(f.batch,in_capture=capture)
                self.assertIs(f.backend.forward_metadata.kv_indptr,f.ptr)
                self.assertEqual(f.adapter.graph_metadata,{})
                self.assertEqual(f.adapter.metadata_scopes,{})
                self.assertEqual(f.adapter.buffers,{})
            self.assertEqual(f.adapter.metadata_build_host_calls,0)

    def test_draft_graph_forward_keeps_native_attention(self):
        f=scope.support.Fixture(fixed,2)
        f.backend.num_head,f.backend.num_kv_head=16,4
        f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
        f.torch.capturing=True
        f.backend.init_forward_metadata_in_graph(f.batch)
        f.forward(False,layers=[0])
        f.torch.capturing=False
        self.assertEqual(f.adapter.capture_host_calls,0)
        self.assertEqual(f.adapter.metadata_build_host_calls,0)
        self.assertEqual(f.adapter.graph_metadata,{})

    def test_draft_instance_of_the_same_class_preserves_target_capture(self):
        f=scope.support.Fixture(fixed);graph=f.capture()
        draft=f.cls();draft.__dict__.update(vars(f.backend));draft.num_head,draft.num_kv_head=16,4
        retained_metadata=dict(f.adapter.graph_metadata)
        for capture in (True,False):draft.init_forward_metadata_out_graph(f.batch,in_capture=capture)
        self.assertEqual(f.adapter.graph_metadata,retained_metadata)
        self.assertNotIn(id(draft),f.adapter.metadata_scopes)
        f.state([127]);f.backend.forward_metadata=scope.SimpleNamespace(max_extend_len=1453,kv_indptr=None)
        f.backend.init_forward_metadata_out_graph(f.batch)
        for operation in graph:operation()
        self.assertEqual(f.metadata()[0],(0,128,257,387,518,650,783,917,1052))

    def test_other_head_geometries_preserve_the_native_fallback(self):
        for heads,kv_heads in ((16,2),(12,4),(8,1)):
            f=scope.support.Fixture(fixed);f.backend.num_head,f.backend.num_kv_head=heads,kv_heads
            f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
            f.backend.init_forward_metadata_out_graph(f.batch)
            self.assertEqual(f.adapter.graph_metadata,{})
            self.assertEqual(f.adapter.metadata_scopes,{})

    def test_native_draft_initializer_exception_is_not_suppressed(self):
        f=scope.support.Fixture(fixed);f.backend.num_head,f.backend.num_kv_head=16,4
        f.backend.initializer_failure=True
        with self.assertRaisesRegex(RuntimeError,'initializer failure'):
            f.backend.init_forward_metadata_out_graph(f.batch,in_capture=True)
        self.assertEqual(f.adapter.graph_metadata,{})


if __name__=='__main__':unittest.main()
