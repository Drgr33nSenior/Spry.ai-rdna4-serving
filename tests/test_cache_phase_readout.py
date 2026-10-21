"""CPU synthetic observations; no native library or remote execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module(ROOT / 'tests/test_cache_gpu_owner.py', 'phase_owner_fixtures')
readout = module(ROOT / 'tools/analyze_cache_phases.py', 'phase_readout')
old_readout = module(ROOT / 'tools/analyze_cache_benchmark.py', 'old_readout')


def observations():
    spec = fixtures.native_spec('profile')
    result = dict(experiment_mode=spec['experiment_mode'], status='NATIVE_CACHE_REGRESSION_COMPLETE',
                  qualification=False, model_starts=0, timings_usable=True, cleanup_errors=[],
                  unresolved=False, independent_release_verified=True,
                  arms=[dict(arm=arm, complete=True, release_verified=True,
                             native_result=fixtures.complete_result(spec, arm)) for arm in ('bf16', 'kv4')])
    return result, spec


class PhaseReadoutTests(unittest.TestCase):
    def analyze(self, result, spec):
        return readout.analyze(result, spec, controller=fixtures.controller, reader=fixtures.reader)

    def test_exact_samples_not_a_speed_comparison(self):
        result, spec = observations()
        value = self.analyze(result, spec)
        self.assertFalse(value['qualification'])
        self.assertEqual(len(value['cells']), 12)
        self.assertEqual(value['profiled_samples'], 120)
        self.assertEqual(value['phase_projection'], spec['phase_profile'])
        for cell in value['cells']:
            self.assertEqual([row['ordinal'] for row in cell['candidate_occurrences']], [1, 2])
            for occurrence in cell['candidate_occurrences']:
                self.assertEqual(len(occurrence['samples']), 5)
                self.assertEqual(occurrence['median_phase_ms']['total_ns'], 0.9003)
                self.assertEqual(occurrence['median_outer_call_ms'], 1.0)
                for shares in occurrence['phase_share_of_profiled_call_percent']:
                    self.assertAlmostEqual(sum(shares.values()), 100)
        self.assertNotIn('speed_ratio', repr(value))
        self.assertNotIn('latency_reduction_percent', repr(value))

    def test_old_benchmark_reader_refuses_profile_and_reverse(self):
        result, spec = observations()
        with self.assertRaisesRegex(ValueError, 'counterbalanced records required'):
            old_readout.analyze(result, spec, controller=fixtures.controller, reader=fixtures.reader)
        with self.assertRaisesRegex(ValueError, 'explicit phase-profile records required'):
            self.analyze(result, fixtures.native_spec(True))

    def test_complete_identity_and_cleanup_required(self):
        result, spec = observations()
        changes = [lambda r: r.update(qualification=True), lambda r: r.update(unresolved=True),
            lambda r: r.update(independent_release_verified=False), lambda r: r['arms'].reverse(),
            lambda r: r['arms'][0].update(release_verified=False),
            lambda r: r['arms'][0]['native_result']['processes'][1]['native']['timings'][0].pop('phase_wall'),
            lambda r: r['arms'][0]['native_result']['processes'][1]['native']['timings'][0]['phase_wall'].update(total_ns=1),
            lambda r: r['arms'][0]['native_result']['processes'][1]['supervision'].update(survivors=[{'pid': 99}])]
        for change in changes:
            mutated = copy.deepcopy(result)
            change(mutated)
            with self.assertRaises(ValueError):
                self.analyze(mutated, spec)


if __name__ == '__main__':
    unittest.main()
