"""Selected draft preflight must reach real metadata validation before Scheduler."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_draft_preflight as checks


class DraftPreflight(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='selected-draft-');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name).resolve()
        for name,source in [('mtp_contract.py','spry_serving/common_mtp_contract.py'),
                            ('worker_entry.py','spry_serving/common_worker_entry.py'),
                            ('spry_common_canon.py','spry_common/selected.py')]:
            shutil.copyfile(HERE.parent/source,root/name)
        for path in (HERE/'fixtures/w5').glob('*.json'):shutil.copyfile(path,root/path.name)
        self.context=checks.loaded(root);self.contract,self.worker=self.context.__enter__();self.addCleanup(self.context.__exit__,None,None,None)
        settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes())
        self.contract.settings=lambda:copy.deepcopy(settings)
    def run_case(self,**kwargs):return checks.run_case(self.worker,self.contract,**kwargs)
    def test_both_arms_and_ranks_retain_only_real_enabled_draft_metadata(self):
        for arm in ('control','mtp'):
            for rank in (0,1):
                with self.subTest(arm=arm,rank=rank):self.assertTrue(self.run_case(arm=arm,rank=rank)['complete_preflight'])
    def test_retired_summary_kinds_remain_refused(self):
        for kind in ('none','mtp','eagle',None):
            with self.subTest(kind=kind),self.assertRaisesRegex(self.contract.MtpContractError,'selected DFlash2'):
                self.contract.expected_draft_summary(kind)
    def test_wrong_architecture_and_hidden_size_refuse_before_writing(self):
        for name,value in [('architectures',['OtherDraft']),('hidden_size',1)]:
            with self.subTest(field=name):
                self.run_case(mutate_draft=lambda draft:setattr(draft.hf_config,name,value),expected_error='native DFlash2 draft config differs')
    def test_wrong_dtype_and_quantization_refuse_before_writing(self):
        for name,value in [('dtype','torch.float32'),('quantization','fp8')]:
            with self.subTest(field=name):
                self.run_case(mutate_draft=lambda draft:setattr(draft,name,value),expected_error='native DFlash2 draft config differs')
    def test_false_draft_identity_refuses_before_writing(self):
        self.run_case(mutate_draft=lambda draft:setattr(draft,'is_draft_model',False),expected_error='native DFlash2 draft config differs')
    def test_changed_draft_width_refuses_before_metadata_constructor(self):
        result=self.run_case(mutate_config=lambda cfg:setattr(cfg,'speculative_num_draft_tokens',4),expected_error='native MTP field differs')
        self.assertEqual(result['native_metadata_constructor_calls'],0)
    def test_changed_draft_config_digest_refuses_before_metadata_constructor(self):
        result=self.run_case(wrong_file_digest=True,expected_error='DFlash2 draft config file differs')
        self.assertEqual(result['native_metadata_constructor_calls'],0)
    def test_changed_explicit_quantization_intent_refuses_before_metadata_constructor(self):
        result=self.run_case(mutate_config=lambda cfg:setattr(cfg,'_speculative_draft_quantization_explicitly_set',False),expected_error='native draft quantization intent differs')
        self.assertEqual(result['native_metadata_constructor_calls'],0)


if __name__=='__main__':unittest.main()
