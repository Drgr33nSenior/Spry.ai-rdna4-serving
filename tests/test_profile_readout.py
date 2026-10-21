"""Actual attribution readers/controller with declared CPU process/GPU fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profiles = load('readout_profile_fixtures', ROOT / 'tests/test_model_benchmark_profile.py')
ordinary = load('readout_original_fixtures', ROOT / 'tests/test_benchmark_readout.py')
controllers = load('readout_controller_fixtures', ROOT / 'tests/test_benchmark_physical_controller.py')
readout = ordinary.readout


def profiled_arm(arm, *, digest='b' * 64, session='a' * 64):
    rows = profiles.profile_fixture_records(arm, session_sha256=session, spec_sha256=digest)
    for row in rows:
        row.update(qualification=False, serving_quality_qualified=False)
    result = ordinary.arm_result(arm)
    result.update(spec_sha256=digest, ranks=rows,
        samples=readout.bench.summarize(rows, arm, expected_profile=True),
        instrumented_timings=True, profile_method=readout.bench.PROFILE_METHOD,
        diagnostic_profile_complete=True, diagnostic_profile=dict(readout.bench.PROFILE_CONFIG))
    return result


class ProfileReadoutTests(unittest.TestCase):
    def test_complete_raw_profiles_pass_independent_reader_with_diagnostic_labels(self):
        spec = ordinary.fixtures.fixture()
        spec['diagnostic_profile'] = dict(readout.bench.PROFILE_CONFIG)
        report = readout.analyze(spec, 'b' * 64, profiled_arm('bf16'), profiled_arm('kv4'))
        self.assertEqual(report['status'], 'OFFLINE_ATTRIBUTION_COMPLETE')
        self.assertTrue(report['instrumented_timings'])
        self.assertFalse(report['qualification'])
        self.assertEqual(len(report['diagnostic_profiles']['kv4']), 2)
        self.assertEqual(len(report['attribution']['kv4']), 8)
        self.assertEqual(report['attribution']['kv4'][0]['gpu_interval_union_us'], 200)
        self.assertFalse(report['attribution']['kv4'][0]['gpu_occupancy_or_bandwidth_measured'])
        self.assertEqual(len(report['output_comparisons']), 42)

    def test_missing_or_substituted_profile_cannot_pass_complete_reader(self):
        spec = ordinary.fixtures.fixture()
        spec['diagnostic_profile'] = dict(readout.bench.PROFILE_CONFIG)
        changes = (
            lambda r: r.pop('instrumented_timings'),
            lambda r: r.update(diagnostic_profile_complete=False),
            lambda r: r.update(diagnostic_profile={'schema': 2, 'mode': 'other'}),
            lambda r: r['ranks'][0].pop('diagnostic_profile'),
            lambda r: r['ranks'][1]['diagnostic_profile']['windows'].pop(),
            lambda r: r['ranks'][0]['diagnostic_profile']['windows'][0].update(pid=9999),
            lambda r: r['ranks'][0]['diagnostic_profile']['windows'][0].update(spec_sha256='c' * 64),
            lambda r: r['ranks'][0]['diagnostic_profile']['windows'][0]['gpu'].update(intervals=[]),
            lambda r: r['ranks'][0]['diagnostic_profile']['windows'][0]['gpu']['categories'].clear(),
            lambda r: r['ranks'][0]['diagnostic_profile']['sources'][0].update(sha256='d' * 64),
            lambda r: r['ranks'][0]['diagnostic_profile']['windows'][0].update(cpu=[
                row for row in r['ranks'][0]['diagnostic_profile']['windows'][0]['cpu']
                if row['qualname'] != 'Native.call']),
            lambda r: r.update(groups_observed_absent=[True, False]),
        )
        for change in changes:
            left, right = profiled_arm('bf16'), profiled_arm('kv4')
            change(right)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError, TypeError)):
                readout.analyze(spec, 'b' * 64, left, right)
        with self.assertRaises(ValueError):
            readout.analyze(ordinary.fixtures.fixture(), 'b' * 64,
                            profiled_arm('bf16'), profiled_arm('kv4'))
        with self.assertRaises(ValueError):
            readout.checked_arm(profiled_arm('bf16'), 'bf16', 'b' * 64,
                                expected_profile=True, session_sha256='f' * 64)

    def test_interval_union_does_not_double_count_overlapping_kernels(self):
        self.assertEqual(readout._interval_union([(0, 10), (5, 10), (20, 5), (22, 1)]), 20)
        self.assertEqual(readout._interval_union([(10, 0), (10, 0)]), 0)

    def make_controller(self):
        fixture = controllers.ControllerTests('test_actual_controller_positive_is_inert_and_never_launch_ready')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        c = controllers.controller
        spec = json.loads(fixture.spec_raw)
        spec['diagnostic_profile'] = dict(c.adapter.readout.bench.PROFILE_CONFIG)
        fixture.spec_raw = controllers.raw(spec)
        digest = controllers.sha(fixture.spec_raw)
        (fixture.package / 'review/BENCHMARK-SPEC2.json').write_bytes(fixture.spec_raw)
        resource_path = fixture.package / 'review/RESOURCE-MANIFEST.json'
        resource = json.loads(resource_path.read_bytes())
        resource['benchmark_spec_sha256'] = digest
        for arm in ('bf16', 'kv4'):
            argv = resource['workload_argv'][arm]
            argv[argv.index('--spec-sha256') + 1] = digest
            resource['workload_argv_sha256'][arm] = c.resources.argv_sha256(argv)
        resource_path.write_bytes(controllers.raw(resource))
        fixture.value.update(spec_sha256=digest,
            resource_manifest_sha256=controllers.sha(resource_path.read_bytes()),
            package_files=fixture.files(fixture.package))
        fixture.refresh()
        fixture.boundaries.enter_context(patch.object(c, 'SPEC_SHA256', digest))
        fixture.prepared = c.validate_manifest(fixture.manifest, fixture.pin,
                                               deadline=c.time.monotonic() + 30)
        return fixture, digest

    def test_actual_controller_accepts_both_profiles_and_keeps_inert_authority(self):
        fixture, digest = self.make_controller()
        report = fixture.invoke(read_inner_result=lambda arm: profiled_arm(arm, digest=digest))
        self.assertEqual(report['status'], 'LOCAL_CONTROLLER_COMPOSITION_PASSED', report)
        self.assertTrue(report['instrumented_timings'])
        self.assertFalse(report['qualification'])
        self.assertFalse(report['launch_authorized'])
        self.assertEqual(fixture.supervised, ['bf16', 'kv4'])
        self.assertEqual(fixture.observed, ['locked-admission', 'release', 'release', 'independent'])

    def test_actual_controller_refuses_unprofiled_arm_and_still_observes_release(self):
        fixture, digest = self.make_controller()
        def missing_profile(arm):
            result = ordinary.arm_result(arm)
            result['spec_sha256'] = digest
            return result
        report = fixture.invoke(read_inner_result=missing_profile)
        self.assertEqual(report['status'], 'INCOMPLETE', report)
        self.assertIn('incomplete', report['failure'].lower())
        self.assertFalse(report['qualification'])
        self.assertEqual(fixture.supervised, ['bf16'])
        self.assertEqual(fixture.observed, ['locked-admission', 'release', 'independent'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
