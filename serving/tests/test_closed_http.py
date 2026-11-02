"""Differential checks for the admitted HTTP source domain; standard library only."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load

RETIRED={'prompt_minimization_enabled','verify_attention_ab_enabled','verify_attention_chain_enabled',
    'gdn_capture_enabled','norm_capture_enabled','norm_candidate_enabled','greedy_quality_draft_ab',
    'j01_probe_enabled','head_input_capture_enabled','final_norm_capture_enabled','midpoint_capture_enabled',
    'raw_attention_capture_enabled','kv_composition_enabled','norm_source_for_arm'}

class ClosedHttp(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='closed-http-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.modules=[]
        for label,source in [('control','spry_serving/shared_contract.py'),('candidate','spry_serving/closed_contract.py')]:
            directory=self.root/label;directory.mkdir()
            (directory/'mtp_contract.py').write_bytes((ROOT/source).read_bytes())
            (directory/'spry_common_canon.py').write_bytes((ROOT/'spry_common/canon.py').read_bytes())
            self.modules.append(load('closed_http_'+label,directory/'mtp_contract.py'))

    def test_retired_feature_apis_are_absent_and_active_policy_views_remain(self):
        old,new=self.modules
        for name in RETIRED:
            self.assertTrue(hasattr(old,name),name);self.assertFalse(hasattr(new,name),name)
        for name in ('settings','precision_reference_enabled','full_enabled_benchmark','paired_serving_enabled',
                     'contract_for_arm','runtime_overlays','expected_decode_log_interval'):
            self.assertTrue(callable(getattr(new,name)))

    def test_attention_route_keeps_values_validation_and_short_circuit_order(self):
        for enabled in (True,False):
            for arm in ('control','mtp','wrong',None):
                values=[];calls=[]
                for module in self.modules:
                    settings=Mock(return_value={'full_enabled_benchmark':'dflash8-splitkv-aligned-gdn-v1' if enabled else 'wrong'})
                    with patch.object(module,'settings',settings):values.append(module.verify_attention_observed(arm))
                    calls.append(settings.call_count)
                self.assertEqual(values[0],values[1]);self.assertEqual(calls,[2,2])
        for module in self.modules:
            with patch.object(module,'settings',side_effect=ValueError('invalid source settings')),self.assertRaisesRegex(ValueError,'invalid source settings'):
                module.verify_attention_observed('control')

    def test_all_installed_consumers_have_no_retired_feature_calls(self):
        paths=[*ROOT.glob('spry_serving/closed*.py'),*ROOT.glob('spry_evidence/closed*.py'),
               ROOT/'proposals/closed-http/review/benchmark_physical_controller.py',
               ROOT/'proposals/closed-http/review/benchmark_owner_adapter.py']
        for path in paths:
            for n in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(n,ast.Call):
                    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else None
                    self.assertNotIn(name,RETIRED,(str(path),getattr(n,'lineno',None),name))
        # The installed owner/controller must not retain calls to APIs removed
        # from the HTTP benchmark and readout, including unreachable diagnostics.
        def declarations(path):
            tree=ast.parse(path.read_bytes());names=set()
            for n in tree.body:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names.add(n.name)
                elif isinstance(n,(ast.Import,ast.ImportFrom)):
                    names.update(a.asname or a.name.split('.')[0] for a in n.names)
                elif isinstance(n,ast.Assign):
                    names.update(t.id for t in n.targets if isinstance(t,ast.Name))
            return names
        readout=declarations(ROOT/'spry_evidence/closed_readout.py')
        benchmark=declarations(ROOT/'spry_serving/closed_benchmark.py')
        for path in paths[-2:]:
            for n in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(n,ast.Attribute):
                    base=ast.unparse(n.value)
                    if base in ('readout','adapter.readout'):
                        self.assertIn(n.attr,readout,(str(path),n.lineno,n.attr))
                    elif base in ('readout.bench','adapter.readout.bench'):
                        self.assertIn(n.attr,benchmark,(str(path),n.lineno,n.attr))

    def test_runtime_provenance_reference_data_is_exactly_the_control(self):
        # The original comparison helper is authenticated by the entry's retained pin.
        candidate=load('closed_http_entry',ROOT/'spry_serving/closed_benchmark.py')
        expected=dict(schema=1,mode='ordered-score-v4-model-v1',entry='uq_bf16_attention_scores_v3',
            workspace_limit_bytes=268435456,binary_sha256='1ca7efb0cf9220adb16a56e30e4cd2931b5d49edbe3277fc15d68e629500cd7b',
            device_sha256='48d2c77e69df1e8b352fbafbf2f1ff832d8cc6717b1a1ffc47970a4f59ee64df',
            source_sha256='661441715312cba400f04bc0168818092d031fc6799affbcbe78fa0e8fa2557b',reader_events=False)
        self.assertEqual(candidate.HTTP_REFERENCE,expected)

    def test_retired_modes_and_optional_experiment_fields_refuse_at_entry(self):
        candidate=load('closed_http_entry_refusals',ROOT/'spry_serving/closed_benchmark.py')
        for mode in ('native-model-context-retrieval-32k-v1','independent-bf16-reference-v1','v4-repeatability-control-v1',None):
            with self.assertRaisesRegex(ValueError,'only the W5'):candidate.validate_spec({'experiment_mode':mode})
        for name in ('independent_reference','validation_comparison','cache_profile','callback_profile','stall_trace'):
            with self.assertRaisesRegex(ValueError,'retired experiment fields'):
                candidate.validate_spec({'experiment_mode':'native-http-fp8-sprint01-v1',name:{}})

    def test_owner_readout_preserves_the_actual_http_acceptance_body(self):
        def body(path,name):
            return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('_object','_constant','read_json','checked_http_arm'):
            self.assertEqual(ast.dump(body(ROOT/'legacy/review/analyze_model_benchmark.py',name)),
                             ast.dump(body(ROOT/'spry_evidence/closed_readout.py',name)))
        names={n.name for n in ast.parse((ROOT/'spry_evidence/closed_readout.py').read_bytes()).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(names,{'_object','_constant','read_json','checked_http_arm'})

    def test_worker_kernel_geometry_and_weight_checks_are_unchanged(self):
        old=ast.parse((ROOT/'spry_serving/startup_worker.py').read_bytes())
        new=ast.parse((ROOT/'spry_serving/closed_worker.py').read_bytes())
        for name in ('_precision_weights','_validate_precision_config','_precision_loading_policy','_mxfp4_loaded_weights','_install_mxfp4_selection'):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)

if __name__=='__main__':unittest.main()
