"""Exercise configured owner timing on authenticated complete CPU source copies."""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict, replace
import difflib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_spec_binding_packet as spec_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_owner_timing_proposal import PROPOSAL, ADAPTER, CONTROLLER, baseline_timing, canonical, sha


class OwnerTimingPacket(spec_checks.SpecBindingPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        with patch.object(spec_checks, 'PROPOSAL', PROPOSAL):
            super().setUpClass()
        raw = (PROPOSAL / ADAPTER).read_bytes()
        if sha(raw) != cls.record['protected_files'][Path(ADAPTER).name]:
            raise AssertionError('proposed owner source pin differs')
        (cls.package / ADAPTER).write_bytes(raw)
        path = cls.package / 'MANIFEST.json'
        manifest = json.loads(path.read_bytes())
        manifest['package_files'][ADAPTER] = sha(raw)
        path.write_bytes(canonical(manifest))
        cls.members.update({ADAPTER: raw, 'MANIFEST.json': path.read_bytes()})
        cls.owner_fixture = load('cleanup_owner_fixture', HERE.parents[1] / 'tests/test_benchmark_owner_adapter.py')
        cls.evidence.update(owner_adapter_sha256=sha(raw), package_inventory=manifest['package_files'],
            owner_fixture_sha256=sha(Path(cls.owner_fixture.__file__).read_bytes()),
            inherited_spec_checker_sha256=sha((HERE / 'check_spec_binding_packet.py').read_bytes()))

    @contextmanager
    def owner_case(self, adapter):
        # Use the retained composition fixture with actual proposed owner and
        # lease modules. Only child/platform observations and the clock are fake.
        with patch.object(self.owner_fixture, 'adapter', adapter), \
                patch.object(self.owner_fixture, 'owner', adapter.owner), \
                patch.object(time, 'monotonic', lambda: 1000):
            case = self.owner_fixture.AdapterTests()
            case.setUp()
            try:
                case.spec = json.loads(self.members['review/BENCHMARK-SPEC2.json'])
                case.spec_path.write_bytes(canonical(case.spec))
                case.spec_sha = sha(case.spec_path.read_bytes())
                yield case
            finally:
                case.tearDown()

    def run_owner(self, controller, scenario='complete', *, proposed=True, worker_seconds=2280, timing=None):
        adapter = controller.adapter
        with self.owner_case(adapter) as case:
            if worker_seconds != case.spec['worker_seconds']:
                case.spec['http_serving']['client_seconds'] += worker_seconds - case.spec['worker_seconds']
                case.spec['worker_seconds'] = worker_seconds
                case.spec_path.write_bytes(canonical(case.spec))
                case.spec_sha = sha(case.spec_path.read_bytes())
            observed = {'supervision': {}, 'release': {}}

            def supervise(argv, **kwargs):
                observed['supervision'][argv[-1]] = asdict(kwargs['deadlines'])
                result = case.supervise(argv, **kwargs)
                if scenario == 'failed_arm':
                    result = replace(result, exit_code=2, failure='synthetic child nonzero')
                    case.results[-1] = result
                return result

            def collect(lease, arm, result, deadline):
                observed['release'][arm] = deadline
                value = case.collect(lease, arm, result, deadline)
                if scenario == 'uncertain_release':
                    value['holders']['stderr'] = 'synthetic visibility failure'
                return value

            def admission(lease, *, deadline_monotonic):
                observed['admission'] = deadline_monotonic
                value = case.admission(lease, deadline_monotonic=deadline_monotonic)
                if scenario == 'bad_admission':
                    value['journal']['sha256'] = 'e' * 64
                return value

            def independent(lease, deadline):
                observed['independent'] = deadline
                value = case.independent(lease, deadline)
                if scenario == 'bad_independent':
                    value['holders']['stderr'] = 'synthetic visibility failure'
                return value

            def inner(arm):
                return dict(schema=1, mode='native-http-fp8-sprint01-v1', status='HTTP_ARM_COMPLETE',
                    owner_arm=arm, logical_arm=controller.mtp_contract.owner_arm_map()[arm], qualification=False,
                    binding={'spec_sha256': case.spec_sha}, wire=dict(status='HTTP_WIRE_COMPLETE',
                    wire_complete=True, speculation={'spec_verify_ct': 1}))

            kwargs = dict(active_deadline=7060, clock=lambda: 1000, supervisor=supervise,
                collect_release=collect, admission_check=admission, independent_postcheck=independent,
                read_inner_result=inner)
            if proposed:
                kwargs['timing'] = (timing if timing is not None else
                                    replace(baseline_timing(adapter), worker_seconds=worker_seconds))
            else:
                kwargs.update(arm_envelope_seconds=2490, collection_reserve_seconds=330)
            report = case.campaign(**kwargs)
            keys = ('status', 'adapter_complete', 'independent_release_verified', 'unresolved',
                    'qualification', 'physical_campaign_complete', 'arm_envelope_seconds',
                    'collection_reserve_seconds', 'failure', 'cleanup_errors')
            projected = {key: report[key] for key in keys}
            projected.update(calls=case.calls, journal_retained=case.lease.journal.exists(),
                fd_closed=case.lease.fd is None, deadlines=observed,
                arms=[{key: row[key] for key in ('arm', 'release_verified', 'inner_complete')}
                      for row in report['arms']])
            return projected

    def test_exact_diff_and_unmodified_release_validators(self):
        baseline = {ADAPTER: self.files['author/package-v2/' + ADAPTER],
                    CONTROLLER: (HERE.parent / 'proposals/d2-spec-binding' / CONTROLLER).read_bytes()}
        expected = ''.join(''.join(difflib.unified_diff(baseline[name].decode().splitlines(True),
            (PROPOSAL / name).read_text().splitlines(True), fromfile='a/' + name, tofile='b/' + name))
            for name in (CONTROLLER, ADAPTER))
        self.assertEqual((PROPOSAL / 'D2.patch').read_text(), expected)
        before, after = ast.parse(baseline[ADAPTER]), ast.parse((PROPOSAL / ADAPTER).read_bytes())
        for tree in (before, after):
            tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
                         or n.name not in ('CampaignTiming', 'run_campaign')]
        self.assertEqual(ast.dump(before), ast.dump(after))
        self.evidence['exact_patch_verified'] = True
        self.evidence['release_validators_unchanged'] = True

    def test_verified_configuration_supplies_complete_owner_plan(self):
        with self.controller() as (controller, path, _, _, _):
            verified = controller.verify_configured_spec(self.pins, json.loads(path.read_bytes()),
                                                         deadline=time.monotonic() + 105)
            self.assertEqual(verified['owner_timing'], baseline_timing(controller.adapter))
            report = self.run_owner(controller, timing=verified['owner_timing'])
            self.assertTrue(report['adapter_complete'], report)
            self.evidence['verified_owner_timing'] = asdict(verified['owner_timing'])
            self.evidence['verified_plan_consumed_by_actual_owner'] = True

    def test_complete_campaign_matches_original_deadlines_and_decisions(self):
        with isolated_modules(), self.binding.controller_on_mirror(self.packages['baseline']) as value:
            old = self.run_owner(value[0], proposed=False)
        with self.controller() as (controller, *_):
            new = self.run_owner(controller)
        self.assertEqual(new, old)
        self.assertTrue(new['adapter_complete'], new)
        self.assertEqual(new['calls'], ['bf16', 'kv4'])
        self.assertTrue(new['independent_release_verified'])
        self.assertFalse(new['journal_retained'])
        self.assertFalse(new['qualification'])
        for arm in new['calls']:
            self.assertEqual(new['deadlines']['supervision'][arm],
                             dict(work_end=3430, term_end=3450, kill_end=3455, reap_end=3460))
            self.assertEqual(new['deadlines']['release'][arm], 3490)
        self.assertEqual(new['deadlines']['admission'], 1075)
        self.assertEqual(new['deadlines']['independent'], 7060)
        self.evidence['baseline_owner_campaign'] = new

    def test_failures_keep_original_second_arm_and_journal_decisions(self):
        results = {}
        for scenario in ('failed_arm', 'uncertain_release', 'bad_admission', 'bad_independent'):
            with self.subTest(scenario=scenario):
                with isolated_modules(), self.binding.controller_on_mirror(self.packages['baseline']) as value:
                    old = self.run_owner(value[0], scenario, proposed=False)
                with self.controller() as (controller, *_):
                    new = self.run_owner(controller, scenario)
                self.assertEqual(new, old)
                self.assertFalse(new['adapter_complete'])
                self.assertFalse(new['qualification'])
                self.assertTrue(new['fd_closed'])
                self.assertEqual(new['calls'], [] if scenario == 'bad_admission' else
                                 ['bf16', 'kv4'] if scenario == 'bad_independent' else ['bf16'])
                self.assertEqual(new['journal_retained'], scenario in ('uncertain_release', 'bad_independent'))
                results[scenario] = new
        self.evidence['owner_failure_cases'] = results

    def test_shorter_plan_reaches_real_owner_deadlines_under_existing_ceilings(self):
        # Adapter-only consumption check: the whole-run verifier continues to
        # reject a reapproved nonbaseline run. No validator is replaced here.
        with self.controller() as (controller, *_):
            report = self.run_owner(controller, worker_seconds=2180)
        self.assertTrue(report['adapter_complete'], report)
        self.assertEqual(report['arm_envelope_seconds'], 2390)
        for arm in report['calls']:
            self.assertEqual(report['deadlines']['supervision'][arm],
                             dict(work_end=3330, term_end=3350, kill_end=3355, reap_end=3360))
            self.assertEqual(report['deadlines']['release'][arm], 3390)
        self.evidence['adapter_only_shorter_worker'] = report
        self.evidence['whole_run_config_mutations_supported'] = False

    def test_invalid_timing_refuses_before_lease_or_observers(self):
        cases = ('missing_plan', 'wrong_type', 'corrupted_plan', 'worker_mismatch', 'short_active', 'no_admission')
        for scenario in cases:
            with self.subTest(scenario=scenario), self.controller() as (controller, *_):
                adapter = controller.adapter
                with self.owner_case(adapter) as case:
                    kwargs = dict(timing=baseline_timing(adapter), active_deadline=7060,
                                  clock=lambda: 1000, admission_check=case.admission)
                    if scenario == 'missing_plan':
                        del kwargs['timing']
                    elif scenario == 'wrong_type':
                        kwargs['timing'] = asdict(kwargs['timing'])
                    elif scenario == 'corrupted_plan':
                        object.__setattr__(kwargs['timing'], 'owner_cleanup_seconds', 149)
                    elif scenario == 'worker_mismatch':
                        kwargs['timing'] = replace(kwargs['timing'], worker_seconds=2180)
                    elif scenario == 'short_active':
                        kwargs['active_deadline'] = 6384  # 1000 + 75 + 2*2490 + 330 - 1
                    elif scenario == 'no_admission':
                        kwargs['admission_check'] = None
                    with self.assertRaises((adapter.owner.OwnerError, TypeError)):
                        case.campaign(**kwargs)
                    self.assertIsNone(case.lease.fd)
                    self.assertFalse(case.lease.journal.exists())
                    self.assertEqual(case.calls, [])
        self.evidence['prelease_timing_refusals'] = list(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    spec_checks.inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(OwnerTimingPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
