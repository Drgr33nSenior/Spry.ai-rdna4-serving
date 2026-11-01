"""Owner timing bounds, immutable inputs and exact isolated source changes."""
import ast
from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import test_spec_binding_proposal as spec_tests
from test_config_binding_proposal import canonical, sha
from spry_common.limits import derive

PROPOSAL = HERE.parent / 'proposals/d2-owner-timing'
CONTROLLER = 'review/benchmark_physical_controller.py'
ADAPTER = 'review/benchmark_owner_adapter.py'


def timing_module():
    tree = ast.parse((PROPOSAL / ADAPTER).read_bytes())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'dataclasses':
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == '_need':
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'CampaignTiming':
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('INNER_WORK', 'ARM_WORK', 'ARM_ENVELOPE') for t in node.targets):
            selected.append(node)
    module = ModuleType('owner_timing_fixture')
    module.owner = SimpleNamespace(OwnerError=ValueError)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PROPOSAL / ADAPTER), 'exec'), module.__dict__)
    return module


def baseline_timing(adapter):
    run = json.loads((HERE.parent / 'config/runs/w5-v2/run.json').read_bytes())
    limits = derive(run['limits'])
    return adapter.CampaignTiming(worker_seconds=limits['worker_seconds'], **{
        key: run['limits'][key] for key in ('owner_cleanup_seconds', 'arm_release_seconds',
                                          'locked_admission_seconds', 'collection_reserve_seconds')})


class TimingBounds(unittest.TestCase):
    def setUp(self):
        self.adapter = timing_module()
        self.timing = baseline_timing(self.adapter)

    def test_baseline_matches_canonical_limit_arithmetic(self):
        run = json.loads((HERE.parent / 'config/runs/w5-v2/run.json').read_bytes())
        limits = derive(run['limits'])
        self.assertEqual(self.timing.worker_seconds, limits['worker_seconds'])
        self.assertEqual(self.timing.arm_work_seconds, limits['arm_work_seconds'])
        self.assertEqual(self.timing.arm_envelope_seconds, limits['arm_envelope_seconds'])
        self.assertEqual((self.timing.arm_work_seconds, self.timing.arm_envelope_seconds), (2430, 2490))

    def test_shorter_worker_keeps_owner_cleanup_and_release_reserves(self):
        timing = self.adapter.CampaignTiming(**dict(asdict(self.timing), worker_seconds=2180))
        self.assertEqual((timing.arm_work_seconds, timing.arm_envelope_seconds), (2330, 2390))
        self.assertEqual((timing.owner_cleanup_seconds, timing.arm_release_seconds), (150, 60))

    def test_invalid_types_higher_ceilings_and_shorter_reserves_refuse(self):
        cases = [('worker_seconds', True), ('worker_seconds', 0), ('worker_seconds', 2281),
                 ('worker_seconds', 2280.0), ('owner_cleanup_seconds', 149), ('owner_cleanup_seconds', 151),
                 ('arm_release_seconds', 59), ('arm_release_seconds', 61),
                 ('locked_admission_seconds', 74), ('locked_admission_seconds', 76),
                 ('collection_reserve_seconds', 329), ('collection_reserve_seconds', 331)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.adapter.CampaignTiming(**dict(asdict(self.timing), **{key: value}))

    def test_plan_is_immutable_and_derived_values_cannot_be_supplied(self):
        with self.assertRaises(FrozenInstanceError):
            self.timing.worker_seconds = 3000
        with self.assertRaises(TypeError):
            self.adapter.CampaignTiming(**dict(asdict(self.timing), arm_work_seconds=1))


class ConfiguredTiming(spec_tests.SpecBinding):
    def setUp(self):
        with patch.object(spec_tests, 'PROPOSAL', PROPOSAL):
            super().setUp()
        self.adapter = timing_module()
        self.controller.adapter.CampaignTiming = self.adapter.CampaignTiming

    def test_verified_config_returns_the_actual_owner_timing(self):
        verified = self.verify()
        expected = baseline_timing(self.adapter)
        self.assertEqual(verified['owner_timing'], expected)
        self.assertEqual(verified['spec']['worker_seconds'], expected.worker_seconds)


class ExactOwnerProposal(unittest.TestCase):
    def test_source_hashes_patch_and_protected_set_are_exact(self):
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        patch_raw = (PROPOSAL / 'D2.patch').read_bytes()
        self.assertEqual(sha(patch_raw), record['patch_sha256'])
        prior = json.loads((HERE.parent / 'proposals/d2-spec-binding/PROPOSAL.json').read_bytes())
        pins = dict(prior['protected_files'])
        for name, change in record['changes'].items():
            self.assertEqual(sha((PROPOSAL / name).read_bytes()), change['proposed_sha256'])
            self.assertEqual(change['baseline_sha256'], prior['protected_files'][Path(name).name])
            pins[Path(name).name] = change['proposed_sha256']
        self.assertEqual(pins, record['protected_files'])
        self.assertEqual(sha(canonical({'review/' + name: pin for name, pin in pins.items()})),
                         record['protected_set_sha256'])
        for key in ('owner_ceilings_changed', 'whole_run_config_mutations_supported', 'consumed_by_builder',
                    'owner_approved', 'launch_authorized', 'entire_D2_complete'):
            self.assertFalse(record[key])

    def test_controller_only_binds_passes_and_reports_the_plan(self):
        before = ast.parse((HERE.parent / 'proposals/d2-spec-binding' / CONTROLLER).read_bytes())
        after = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
        allowed = {'validate_manifest', 'verify_configured_spec', '_run_bound_controller'}
        for tree in (before, after):
            tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name not in allowed]
        self.assertEqual(ast.dump(before), ast.dump(after))
        tree = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'run_campaign')
        kwargs = {item.arg: item.value for item in call.keywords}
        self.assertEqual(ast.dump(kwargs['timing']), ast.dump(ast.Name(id='timing', ctx=ast.Load())))
        self.assertNotIn('arm_envelope_seconds', kwargs)
        self.assertNotIn('collection_reserve_seconds', kwargs)


if __name__ == '__main__':
    unittest.main()
