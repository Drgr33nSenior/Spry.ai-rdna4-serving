"""Selected-phase contract checks with retained settings and synthetic CPU assets."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'legacy' / 'review'
FIXTURES = HERE / 'fixtures' / 'w5'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, set):
        value = sorted(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class SelectedContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-contract-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for path in FIXTURES.glob('*.json'):
            shutil.copyfile(path, self.root / path.name)
        shutil.copyfile(SOURCE / 'mtp_contract.py', self.root / 'mtp_contract.py')
        self.contract = load('cleanup_selected_contract', self.root / 'mtp_contract.py')
        self.settings = json.loads((self.root / 'FINAL-SETTINGS.json').read_bytes())
        # Exercise shape and hash validation without shipping the private prompt corpus.
        self.assets = {}
        for row in self.settings['requests']:
            for field, pin in (('path', 'sha256'), ('expected', 'expected_sha256')):
                raw = canonical(dict(case=row['case'], field=field, cpu_fixture=True))
                self.assets[row[field]] = raw
                row[pin] = hashlib.sha256(raw).hexdigest()
        self.write_settings()
        patched = patch.object(self.contract, '_assets', return_value=self.assets)
        patched.start()
        self.addCleanup(patched.stop)

    def write_settings(self):
        (self.root / 'FINAL-SETTINGS.json').write_bytes(canonical(self.settings))

    def test_only_selected_phase_is_registered(self):
        self.assertEqual(set(self.contract.PHASES), {'sprint01-paired-serving'})
        self.assertEqual(self.contract.phase_config(),
            (131040, 2040, 176, ('dflash2', 'dflash2'), ('tail', 'tail')))

    def test_other_phases_refuse_before_assets(self):
        for phase in ('sprint01-gdn-aligned-lifecycle', 'main-forward', 'sprint01-width-forward', '', None):
            self.settings['phase'] = phase
            self.write_settings()
            with self.subTest(phase=phase), patch.object(self.contract, '_assets') as assets, \
                 self.assertRaisesRegex(self.contract.MtpContractError, 'selected paired-serving phase'):
                self.contract.settings()
            assets.assert_not_called()

    def test_both_arms_keep_dflash8_bf16_caches_and_capacity(self):
        for arm in ('control', 'mtp'):
            with self.subTest(arm=arm):
                value = self.contract.contract_for_arm(arm)
                self.assertEqual(self.contract.validate_contract(value), value)
                self.assertEqual(value['speculative_algorithm'], 'DFLASH')
                self.assertEqual(value['speculative_num_draft_tokens'], 8)
                self.assertIsNone(value['speculative_draft_model_quantization'])
                self.assertEqual(value['target_cuda_graph_capture_bs'], [1, 2])
                self.assertEqual(self.contract.memory_pool_config(arm), dict(context_length=131040,
                    max_running_requests=2, max_total_tokens=262144, kv_cache_dtype='bf16', max_mamba_cache_size=19))
                self.assertEqual(self.contract.cache_profile(arm), dict(target='bf16', draft='bf16'))
                self.assertEqual(self.contract.expected_environment(arm), {'SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH': '1'})
        self.assertEqual(self.contract.commit_observation_limit(), 1593344)
        self.assertEqual(self.contract.expected_recurrent_slots(), 19)

    def test_instrumentation_requires_explicit_evidence_mode(self):
        for mode, expected in [('on', True), ('production', False), (None, False)]:
            if mode is None:
                self.settings.pop('instrumentation')
            else:
                self.settings['instrumentation'] = mode
            self.write_settings()
            self.assertIs(self.contract.instrumentation_enabled(), expected)

    def test_stop_policy_keeps_natural_rows_strict_and_admits_exact_declared_caps(self):
        normal=dict(status='NORMAL_STOP_COMPLETE',finish_reason='stop')
        cap=dict(status='OUTPUT_CAP_REACHED',finish_reason='length',metric_eligible=False,
                 reported_completion_tokens=4,response_id_count=4)
        self.assertTrue(self.contract.completion_allowed(normal,{'stop_policy':'natural'}))
        self.assertFalse(self.contract.completion_allowed(cap,{'stop_policy':'natural','maximum_output_tokens':4}))
        self.assertTrue(self.contract.completion_allowed(normal,{'stop_policy':'cap','maximum_output_tokens':4}))
        self.assertTrue(self.contract.completion_allowed(cap,{'stop_policy':'cap','maximum_output_tokens':4}))
        for key in ('reported_completion_tokens','response_id_count'):
            for value in (True,4.0,3,5,None):
                with self.subTest(key=key,value=value):
                    self.assertFalse(self.contract.completion_allowed(dict(cap,**{key:value}),
                        {'stop_policy':'cap','maximum_output_tokens':4}))
        self.assertFalse(self.contract.completion_allowed(dict(cap,metric_eligible=True),
                        {'stop_policy':'cap','maximum_output_tokens':4}))

    def test_invalid_row_stop_policy_refuses_before_asset_access(self):
        for policy in (None,True,'length',''):
            self.settings['requests'][0]['stop_policy']=policy
            self.write_settings()
            with self.subTest(policy=policy),patch.object(self.contract,'_assets') as assets, \
                    self.assertRaisesRegex(self.contract.MtpContractError,'stop policy'):
                self.contract.settings()
            assets.assert_not_called()

    def test_explicit_cap_policy_preserves_the_request_asset(self):
        before=copy.deepcopy(self.assets)
        self.settings['requests'][0]['stop_policy']='cap'
        self.write_settings()
        self.assertEqual(self.contract.settings()['requests'][0]['stop_policy'],'cap')
        self.assertEqual(self.assets,before)

    def test_legacy_fields_and_generated_history_are_rejected(self):
        original = copy.deepcopy(self.settings)
        for key in ('phase_timing', 'target_timing', 'performance_benchmark', 'j01_probe',
                    'midpoint_capture', 'diagnostic_kernel_serialization'):
            self.settings = copy.deepcopy(original)
            self.settings[key] = True
            self.write_settings()
            with self.subTest(field=key), self.assertRaises(self.contract.MtpContractError):
                self.contract.settings()
        self.settings = original
        self.settings['requests'][5]['generated_history'] = {}
        self.write_settings()
        with self.assertRaisesRegex(self.contract.MtpContractError, 'generated history'):
            self.contract.settings()

    def test_request_shapes_and_assets_remain_mandatory(self):
        original = copy.deepcopy(self.settings)
        for field, value in [('maximum_output_tokens', 1), ('input_tokens', 1), ('ordinal', True),
                              ('sha256', '0' * 64), ('path', '../FINAL-REQUEST-00.json')]:
            self.settings = copy.deepcopy(original)
            self.settings['requests'][0][field] = value
            self.write_settings()
            with self.subTest(field=field), self.assertRaises(self.contract.MtpContractError):
                self.contract.settings()
        self.settings = original
        self.write_settings()
        self.assets['unbound'] = b'unbound'
        with self.assertRaisesRegex(self.contract.MtpContractError, 'exact compressed asset inventory'):
            self.contract.settings()

    def test_speculation_drift_and_retired_kinds_refuse(self):
        for field, value in [('speculative_kind', 'mtp'), ('speculative_algorithm', 'EAGLE'),
                              ('speculative_num_draft_tokens', 4), ('enabled', False)]:
            contract = self.contract.contract_for_arm('mtp')
            contract[field] = value
            with self.subTest(field=field), self.assertRaises(self.contract.MtpContractError):
                self.contract.validate_contract(contract)
        for kind in ('none', 'mtp'):
            with self.assertRaises(self.contract.MtpContractError):
                self.contract.expected_draft_summary(kind)

    def test_unknown_arm_and_changed_precision_refuse(self):
        for name in ('contract_for_arm', 'memory_pool_config', 'cache_profile', 'expected_environment',
                     'gemm_for_arm', 'gdn_source_for_arm', 'prefill_chunk_size', 'pool_token_capacity'):
            with self.subTest(function=name), self.assertRaises(self.contract.MtpContractError):
                getattr(self.contract, name)('unknown')
        path = self.root / 'PRECISION-REFERENCE.json'
        profile = json.loads(path.read_bytes())
        profile['cache_profile']['draft'] = 'fp8_e4m3'
        path.write_bytes(canonical(profile))
        with self.assertRaisesRegex(self.contract.MtpContractError, 'identity/cache profile'):
            self.contract.precision_profile()

    def test_retained_api_markers_cannot_enable_legacy_paths(self):
        for name in ('prompt_minimization_enabled', 'verify_attention_ab_enabled', 'gdn_capture_enabled',
                     'norm_capture_enabled', 'norm_candidate_enabled', 'greedy_quality_draft_ab',
                     'j01_probe_enabled', 'head_input_capture_enabled', 'final_norm_capture_enabled',
                     'midpoint_capture_enabled', 'raw_attention_capture_enabled'):
            self.assertIs(getattr(self.contract, name)(), False, name)
        for name in ('CLOSURE_PHASES', 'HISTORY_PHASES', 'TAIL_PHASES', 'FAIL_FAST_PHASES'):
            self.assertEqual(getattr(self.contract, name), frozenset())
        self.assertFalse(hasattr(self.contract, 'betterbench_checkpoint_enabled'))

    def test_fixture_bytes_match_recorded_retained_sources(self):
        metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        for name, record in metadata['contract_test_fixtures'].items():
            raw = (FIXTURES / name).read_bytes()
            self.assertEqual(len(raw), record['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])


if __name__ == '__main__':
    unittest.main()
