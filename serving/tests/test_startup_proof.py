"""Startup proof must consume module policy and pass the unchanged real reader."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_startup_proof as checks


class StartupProof(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='startup-proof-source-');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name).resolve()
        for name,source in [('mtp_contract.py','spry_serving/common_mtp_contract.py'),
                            ('worker_entry.py','spry_serving/common_worker_entry.py'),
                            ('verify_decode_adapter.py','spry_serving/common_verify_decode_adapter.py'),
                            ('serving_integration.py','spry_evidence/common_serving_integration.py'),
                            ('kv_graph_join.py','spry_evidence/common_kv_graph_join.py'),
                            ('spry_common_canon.py','spry_common/selected.py')]:
            shutil.copyfile(HERE.parent/source,root/name)
        for path in (HERE/'fixtures/w5').glob('*.json'):shutil.copyfile(path,root/path.name)
        self.context=checks.loaded(root);self.values=self.context.__enter__();self.addCleanup(self.context.__exit__,None,None,None)
        settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());self.values[0].settings=lambda:copy.deepcopy(settings)
        reader_contract=patch.object(self.values[-1],'_mtp_contract',return_value=self.values[0])
        reader_contract.start();self.addCleanup(reader_contract.stop)
    def run_case(self,**kwargs):return checks.run_case(*self.values,**kwargs)
    def test_both_arms_and_ranks_export_through_real_reader_without_instance_policy(self):
        for arm in ('control','mtp'):
            for rank in (0,1):
                with self.subTest(arm=arm,rank=rank):self.assertTrue(self.run_case(arm=arm,rank=rank)['real_full_enabled_reader_passed'])
    def test_instance_attribute_cannot_replace_module_policy(self):
        self.assertTrue(self.run_case(mutate_adapter=lambda adapter:setattr(adapter,'METADATA_PREPARATION_POLICY','wrong-instance-value'))['real_full_enabled_reader_passed'])
    def test_wrong_indices_geometry_preserves_precision_witness_and_refuses_full_proof(self):
        from types import SimpleNamespace
        self.run_case(mutate_adapter=lambda adapter:adapter.buffers.update({8:dict(indices=SimpleNamespace(numel=lambda:1))}),
            expected_error='split-KV indices geometry differs')
    def test_missing_capture_refuses_full_proof(self):
        self.run_case(mutate_adapter=lambda adapter:setattr(adapter,'capture_host_calls',31),expected_error='adapter not captured')
    def test_missing_layer_refuses_full_proof(self):
        self.run_case(mutate_adapter=lambda adapter:adapter.layers.remove(15),expected_error='adapter not captured')
    def test_real_reader_refuses_metadata_built_during_capture(self):
        self.run_case(mutate_adapter=lambda adapter:setattr(adapter,'metadata_capture_build_host_calls',1),reader_error='not prepared outside')
    def test_real_reader_refuses_changed_module_policy(self):
        self.run_case(module_policy='inside-forward-graph',reader_error='not prepared outside')
    def test_missing_module_policy_has_no_fallback(self):
        self.run_case(remove_module_policy=True,expected_error='METADATA_PREPARATION_POLICY')


if __name__=='__main__':unittest.main()
