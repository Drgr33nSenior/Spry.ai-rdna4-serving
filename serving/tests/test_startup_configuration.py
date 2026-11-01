"""Effective startup flags and runner selection fail before native identity work."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
from test_config_binding_proposal import sha


class StartupConfiguration(unittest.TestCase):
    def setUp(self):
        self.worker=load('cleanup_startup_worker',HERE.parent/'legacy/review/worker_entry.py')
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-startup-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.runner=self.root/'model_runner.py';self.runner.write_bytes(b'# source-pinned runner\n')
        self.row=dict(key='model_runner',module='sglang.srt.model_executor.model_runner',
            target=str(self.runner),file='model_runner.py',sha256=sha(self.runner.read_bytes()),original_sha256='a'*64)
        self.plan=dict(speculative_algorithm='DFLASH',speculative_num_steps=1,speculative_eagle_topk=1,
            speculative_num_draft_tokens=8,speculative_dflash_block_size=8,enable_multi_layer_eagle=False,
            target_cuda_graph_backend_decode='full',target_cuda_graph_backend_prefill='disabled',
            target_cuda_graph_capture_bs=[1,2],target_cuda_graph_max_bs=2)
        self.expected=dict(speculative_algorithm='DFLASH',speculative_num_steps=1,speculative_eagle_topk=1,
            speculative_num_draft_tokens=8,speculative_dflash_block_size=8,enable_multi_layer_eagle=False,
            tp_size=2,pp_size=1,disable_overlap_schedule=True,cuda_graph_backend_decode='full',
            cuda_graph_backend_prefill='disabled',cuda_graph_bs_decode=[1,2],cuda_graph_max_bs_decode=2)
        self.cfg=SimpleNamespace(**copy.deepcopy(self.expected))
        self.contract=SimpleNamespace(contract_for_arm=Mock(return_value=self.plan),
            validate_contract=lambda value:value,cache_config=lambda:dict(disable_overlap_schedule=True),
            runtime_overlays=Mock(return_value=[self.row]))
        self.resolver=Mock(return_value=self.cfg)
        self.native=patch.object(self.worker,'_native_module',return_value=SimpleNamespace(resolving_view=self.resolver))
        self.native.start();self.addCleanup(self.native.stop)
        self.manifest=dict(arm='control',expected_ranks=[0,1],sources=dict(model_config={'module':'pinned.model_config'},
            model_runner=dict(module=self.row['module'],path=self.row['target'],sha256=self.row['sha256'],stock_sha256=self.row['original_sha256'])),
            preweight=dict(deadline_monotonic=200,gpu_identity={'module':'gpu'},graph_observer={'module':'observer'}))

    def validate(self): return self.worker._validate_serving_startup(self.manifest,object(),self.contract)

    def test_both_arms_use_effective_config_and_authenticate_the_selected_runner(self):
        for arm in ('control','mtp'):
            self.manifest['arm']=arm
            self.validate()
            self.contract.contract_for_arm.assert_called_with(arm)
            self.contract.runtime_overlays.assert_called_with(arm)
        self.assertEqual(self.resolver.call_count,2)

    def test_every_resolved_field_mismatch_refuses(self):
        for name,value in self.expected.items():
            changed = ('wrong' if type(value) is str else False if type(value) is bool
                       else [1] if type(value) is list else value+1)
            if type(value) is bool and value is False: changed=True
            setattr(self.cfg,name,changed)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'startup field differs'):
                self.validate()
            setattr(self.cfg,name,copy.deepcopy(value))

    def test_boolean_integer_aliases_and_graph_batch_aliases_cannot_pass(self):
        for name,value in (('pp_size',True),('speculative_num_steps',True),
                           ('disable_overlap_schedule',1),('cuda_graph_bs_decode',[True,2])):
            original=getattr(self.cfg,name);setattr(self.cfg,name,value)
            with self.subTest(name=name),self.assertRaises(ValueError):self.validate()
            setattr(self.cfg,name,original)

    def test_parsed_flags_do_not_override_a_wrong_effective_resolution(self):
        parsed=SimpleNamespace(**self.expected)
        self.cfg.speculative_num_draft_tokens=4
        with self.assertRaisesRegex(ValueError,'speculative_num_draft_tokens'):
            self.worker._validate_serving_startup(self.manifest,parsed,self.contract)
        self.resolver.assert_called_once_with(parsed)

    def test_missing_or_changed_runner_binding_refuses(self):
        for key,value in (('module','wrong'),('path',str(self.root/'other.py')),('sha256','b'*64),('stock_sha256','c'*64)):
            source=self.manifest['sources']['model_runner'];original=source[key];source[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'overlay binding differs'):self.validate()
            source[key]=original
        self.contract.runtime_overlays.return_value=[]
        with self.assertRaisesRegex(ValueError,'runner overlay required'):self.validate()

    def test_changed_or_linked_runner_file_refuses_under_the_original_binding(self):
        self.runner.write_bytes(b'# substituted runner\n')
        with self.assertRaisesRegex(ValueError,'overlay source differs'):self.validate()
        self.runner.unlink();self.runner.symlink_to(self.root/'missing.py')
        with self.assertRaises((ValueError,FileNotFoundError)):self.validate()

    def test_missing_native_resolver_refuses(self):
        with patch.object(self.worker,'_native_module',return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ValueError,'resolved-config constructor missing'):self.validate()

    def test_production_and_instrumented_bad_flags_refuse_before_gpu_queries_or_writes(self):
        self.cfg.disable_overlap_schedule=False
        for instrumented in (False,True):
            with self.subTest(instrumented=instrumented), \
                    patch.object(self.worker.time,'monotonic',return_value=100), \
                    patch.object(self.worker,'_source_module') as source, \
                    patch.object(self.worker,'_atomic_json') as write:
                with self.assertRaisesRegex(ValueError,'disable_overlap_schedule'):
                    self.worker._preweight_before_scheduler(self.manifest,0,object(),gpu_id=0,
                        contract=self.contract,instrumented=instrumented)
                source.assert_not_called();write.assert_not_called()

    def test_configuration_work_cannot_reopen_an_expired_preweight_deadline(self):
        with patch.object(self.worker.time,'monotonic',side_effect=[100,200]), \
                patch.object(self.worker,'_source_module') as source:
            with self.assertRaisesRegex(ValueError,'elapsed during configuration'):
                self.worker._preweight_before_scheduler(self.manifest,0,object(),gpu_id=0,
                    contract=self.contract,instrumented=False)
            source.assert_not_called()


if __name__=='__main__':unittest.main()
