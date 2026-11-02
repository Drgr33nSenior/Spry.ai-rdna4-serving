"""Selected-phase consumers preserve active behavior and reject retired modes."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
MARKERS={'GDN_PAIR_PHASE','LIFECYCLE_PHASE','PRECISION_REFERENCE_PHASES','CLOSURE_PHASES',
    'BATCH_PHASES','CACHE_PHASES','HISTORY_PHASES','TAIL_PHASES','FAIL_FAST_PHASES','CLOSURE_PHASE',
    'COMBINED_PHASE','SAMPLING_PHASE','HIP_SAMPLING_PHASE','CONFIRMATION_PHASE','HIP_PHASE_ARMS','SAMPLING_PHASES'}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class PhaseClosure(unittest.TestCase):
    def test_all_consumers_and_the_contract_remove_inactive_marker_definitions_and_reads(self):
        record=json.loads((ROOT/'proposals/phase-closure/PROPOSAL.json').read_bytes())
        paths=[ROOT/row['file'] for row in record['source_files'].values()]
        paths += list((ROOT/'proposals/phase-closure/review').glob('*.py'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_bytes())):
                name=node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else None
                with self.subTest(path=path.name,name=name):self.assertNotIn(name,MARKERS)
    def test_selected_contract_still_validates_settings_before_constant_phase_answers(self):
        contract=load('selected_phase_contract',ROOT/'spry_serving/selected_contract.py')
        with patch.object(contract,'settings',return_value={'phase':contract.PAIRED_PHASE}) as settings:
            self.assertTrue(contract.precision_reference_enabled());settings.assert_called_once()
        with patch.object(contract,'settings',side_effect=ValueError('invalid selected settings')):
            with self.assertRaisesRegex(ValueError,'invalid selected settings'):contract.precision_reference_enabled()
        self.assertTrue(MARKERS.isdisjoint(vars(contract)))
    def runtime(self,candidate=True):
        contract=ModuleType('mtp_contract');contract.settings=lambda:{'phase':'sprint01-paired-serving'}
        if not candidate:contract.LIFECYCLE_PHASE='sprint01-gdn-aligned-lifecycle'
        serving=ModuleType('serving_contract')
        if not candidate:serving.HIP_PHASE_ARMS={'sprint01-hip-sampling-forward':'mtp'}
        names=dict(mtp_contract=contract,serving_contract=serving,
            serving_client=ModuleType('serving_client'),serving_entry=ModuleType('serving_entry'))
        names['serving_client'].ServingClient=object
        names['serving_entry'].run_bound=lambda *a,**k:None
        source='spry_serving/selected_runtime.py' if candidate else 'spry_serving/closed_runtime.py'
        with patch.dict(sys.modules,names):return load('phase_runtime_'+str(candidate),ROOT/source)
    def test_real_runtime_refuses_unexpected_hip_records_and_returns_none_for_both_selected_arms(self):
        with tempfile.TemporaryDirectory(prefix='phase-closure-') as temporary:
            directory=Path(temporary).resolve()
            for candidate in (False,True):
                runtime=self.runtime(candidate)
                for arm in ('control','mtp'):
                    material=dict(entry_binding=dict(arm=arm),native_dir=directory)
                    self.assertIsNone(runtime._hip_sampling_ready(material,{}))
                    path=directory/'HIP-SAMPLING-COMPONENT-rank-0.json';path.write_text('{}\n')
                    try:
                        with self.assertRaisesRegex(runtime.ServingRuntimeError,'outside the candidate phase'):
                            runtime._hip_sampling_ready(material,{})
                    finally:path.unlink()
    def test_complete_parity_module_retains_unsupported_component_refusal(self):
        with patch.dict(sys.modules,serving_contract=ModuleType('serving_contract')):
            module=load('phase_parity',ROOT/'spry_evidence/selected_parity.py')
        with self.assertRaisesRegex(ValueError,'HIP component trial phase'):
            module.validate_hip_sampling_components([],binding={},workers=[],phase='sprint01-paired-serving')


if __name__=='__main__':unittest.main()
