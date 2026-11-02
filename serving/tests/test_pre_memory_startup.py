"""The pre-memory control retains compute code and uses corrected startup paths."""
from contextlib import contextmanager
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import propose_packet
import check_draft_preflight as draft
import check_startup_proof as proof

CORRECTION=dict(original_sha256='6b4622b92c2f759b9346909fbb87f56b55eb3b22c7c9a6c0ec2fac918d8da1b0',
    corrected_sha256='d46fd8f749e26af4e4bb761a5f835fb9aac20ce2796d9bf23e3d36a470c3444f')
ORIGINAL=(HERE.parent/'spry_serving/pre_replay_worker.py').read_bytes()


@contextmanager
def fixture(*,corrected=True,post_load=False):
    with tempfile.TemporaryDirectory(prefix='pre-memory-startup-') as temporary:
        root=Path(temporary).resolve()
        for name,source in [('mtp_contract.py','spry_serving/hook_contract.py'),
                ('verify_decode_adapter.py','spry_serving/pre_replay_adapter.py'),
                ('serving_integration.py','spry_evidence/pre_replay_integration.py'),
                ('kv_graph_join.py','legacy/review/kv_graph_join.py'),
                ('spry_common_canon.py','spry_common/canon.py')]:
            if post_load and name=='verify_decode_adapter.py':source='legacy/review/verify_decode_adapter.py'
            if post_load and name=='serving_integration.py':source='spry_evidence/closed_integration.py'
            shutil.copyfile(HERE.parent/source,root/name)
        original=(HERE.parent/'spry_serving/hook_worker.py').read_bytes() if post_load else ORIGINAL
        correction=POST_LOAD_CORRECTION if post_load else CORRECTION
        (root/'worker_entry.py').write_bytes(propose_packet.correct_pre_memory_worker(original,correction) if corrected else original)
        for path in (HERE/'fixtures/w5').glob('*.json'):shutil.copyfile(path,root/path.name)
        yield root


class PreMemoryStartup(unittest.TestCase):
    def test_only_preflight_and_proof_producer_change(self):
        corrected=propose_packet.correct_pre_memory_worker(ORIGINAL,CORRECTION)
        self.assertEqual(propose_packet.digest(corrected),CORRECTION['corrected_sha256'])
        def nodes(raw):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(raw).body if hasattr(node,'name')}
        before,after=nodes(ORIGINAL),nodes(corrected)
        self.assertEqual(before.keys(),after.keys())
        self.assertEqual({name for name in before if before[name]!=after[name]},{'_mtp_preflight','_precision_weights'})
    def test_correction_is_explicit_and_other_proposals_keep_original_bytes(self):
        self.assertIs(propose_packet.correct_pre_memory_worker(ORIGINAL,None),ORIGINAL)
    def test_changed_input_refuses(self):
        with self.assertRaisesRegex(ValueError,'input differs'):
            propose_packet.correct_pre_memory_worker(ORIGINAL+b'\n',CORRECTION)
    def test_unreviewed_correction_refuses(self):
        with self.assertRaisesRegex(ValueError,'exact pre-memory startup correction'):
            propose_packet.correct_pre_memory_worker(ORIGINAL,dict(CORRECTION,corrected_sha256='a'*64))
    def test_second_application_refuses(self):
        with self.assertRaisesRegex(ValueError,'input differs'):
            propose_packet.correct_pre_memory_worker(propose_packet.correct_pre_memory_worker(ORIGINAL,CORRECTION),CORRECTION)
    def test_real_draft_preflight_uses_only_selected_metadata_in_both_modes(self):
        with fixture() as root,draft.loaded(root) as (contract,worker):
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());contract.settings=lambda:copy.deepcopy(settings)
            original=contract.expected_draft_summary
            with patch.object(contract,'expected_draft_summary',wraps=original) as summary:
                for mode in ('on','production'):
                    settings['instrumentation']=mode
                    for arm in ('control','mtp'):
                        for rank in (0,1):
                            self.assertTrue(draft.run_case(worker,contract,arm=arm,rank=rank)['complete_preflight'])
                self.assertTrue(summary.call_args_list)
                self.assertEqual({call.args[0] for call in summary.call_args_list},{'dflash2'})
    def test_real_pre_memory_producer_and_reader_pass_both_modes_and_ranks(self):
        with fixture() as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                for mode in ('on','production'):
                    settings['instrumentation']=mode
                    for arm in ('control','mtp'):
                        for rank in (0,1):
                            self.assertTrue(proof.run_case(*values,arm=arm,rank=rank)['real_full_enabled_reader_passed'])
    def test_original_producer_reproduces_instance_policy_failure(self):
        with fixture(corrected=False) as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                self.assertTrue(proof.run_case(*values,expected_error='METADATA_PREPARATION_POLICY')['precision_witness_retained'])
    def test_original_reader_still_rejects_bad_metadata_policy(self):
        with fixture() as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                proof.run_case(*values,module_policy='inside-forward-graph',reader_error='not prepared outside')


POST_LOAD_CORRECTION=dict(original_sha256='5ea61d216dcd91c0d7f3de8d9816d5e44643385776cd712124424eb2fee2e330',
    corrected_sha256='f9b9f962da347c31289f4ea31998bbdcff2d936da6c888ecf1b64bcdc3bcf0f7')


class PostLoadStartup(unittest.TestCase):
    def test_only_eager_preflight_summary_changes(self):
        original=(HERE.parent/'spry_serving/hook_worker.py').read_bytes()
        corrected=propose_packet.correct_pre_memory_worker(original,POST_LOAD_CORRECTION)
        expected=original.replace(b"    draft_summary: dict[str, Any] = contract_module.expected_draft_summary('none')\n",b'')
        self.assertEqual(corrected,expected)
        self.assertEqual(propose_packet.digest(corrected),POST_LOAD_CORRECTION['corrected_sha256'])
    def test_post_load_correction_rejects_changed_input(self):
        with self.assertRaisesRegex(ValueError,'input differs'):
            propose_packet.correct_pre_memory_worker((HERE.parent/'spry_serving/hook_worker.py').read_bytes()+b'\n',POST_LOAD_CORRECTION)
    def test_complete_post_load_draft_preflight_both_modes_arms_and_ranks(self):
        with fixture(post_load=True) as root,draft.loaded(root) as (contract,worker):
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());contract.settings=lambda:copy.deepcopy(settings)
            for mode in ('on','production'):
                settings['instrumentation']=mode
                for arm in ('control','mtp'):
                    for rank in (0,1):self.assertTrue(draft.run_case(worker,contract,arm=arm,rank=rank)['complete_preflight'])
    def test_original_post_load_reproduces_retired_summary_failure(self):
        with fixture(post_load=True,corrected=False) as root,draft.loaded(root) as (contract,worker):
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());contract.settings=lambda:copy.deepcopy(settings)
            self.assertTrue(draft.run_case(worker,contract,expected_error='selected DFlash2')['refused_before_evidence'])
    def test_complete_post_load_producer_and_original_reader_both_modes_arms_and_ranks(self):
        with fixture(post_load=True) as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                for mode in ('on','production'):
                    settings['instrumentation']=mode
                    for arm in ('control','mtp'):
                        for rank in (0,1):self.assertTrue(proof.run_case(*values,arm=arm,rank=rank,expected_policy=None)['real_full_enabled_reader_passed'])
    def test_post_load_producer_still_rejects_invalid_geometry(self):
        from types import SimpleNamespace
        with fixture(post_load=True) as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                proof.run_case(*values,expected_policy=None,mutate_adapter=lambda adapter:adapter.buffers.update({8:dict(indices=SimpleNamespace(numel=lambda:1))}),expected_error='split-KV indices geometry differs')
    def test_post_load_reader_still_requires_two_capture_metadata_builds(self):
        with fixture(post_load=True) as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with patch.object(values[-1],'_mtp_contract',return_value=values[0]):
                proof.run_case(*values,expected_policy=None,mutate_adapter=lambda adapter:setattr(adapter,'metadata_capture_build_host_calls',0),reader_error='metadata not reused')
    def test_legacy_policy_declaration_cannot_downgrade_the_current_fixture(self):
        with fixture() as root,proof.loaded(root) as values:
            settings=json.loads((root/'FINAL-SETTINGS.json').read_bytes());values[0].settings=lambda:copy.deepcopy(settings)
            with self.assertRaisesRegex(AssertionError,'declared startup metadata policy differs'):
                proof.run_case(*values,expected_policy=None)


if __name__=='__main__':unittest.main()
