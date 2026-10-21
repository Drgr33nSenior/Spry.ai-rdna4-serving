"""Synthetic records exercise the actual v2 reader and descriptive readout."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


owner = load(ROOT / 'tests/test_cache_gpu_owner.py', 'counterbalanced_cpu_owner')
readout = load(ROOT / 'tools/analyze_cache_benchmark.py', 'counterbalanced_readout')


class CacheReadoutTests(unittest.TestCase):
    def records(self):
        spec = owner.native_spec()
        spec.update(experiment_mode=owner.entry.COUNTERBALANCED_MODE,
                    timing_protocol=owner.entry.TIMING_PROTOCOL, timing_warmups=2,
                    variant_order=['baseline', 'candidate', 'candidate', 'baseline'], native_processes=16)
        arms = []
        for arm in ('bf16', 'kv4'):
            native = owner.complete_result(spec, arm)
            for process in native['processes']:
                # A0=4 B1=2 B2=3 A3=6: both AB and BA ratios must equal2,
                # with1.5 later/earlier shifts; these are not GPU observations.
                for cell in process['native']['timings']:
                    cell['synchronized_call_ms'] = (4., 2., 3., 6.)[process['ordinal']]
            arms.append({'arm': arm, 'complete': True, 'release_verified': True, 'native_result': native})
        result = {'experiment_mode': spec['experiment_mode'], 'status': 'NATIVE_CACHE_REGRESSION_COMPLETE',
            'qualification': False, 'model_starts': 0, 'timings_usable': True, 'cleanup_errors': [],
            'unresolved': False, 'independent_release_verified': True, 'arms': arms}
        return result, spec

    def evaluate(self, result, spec):
        return readout.analyze(result, spec, controller=owner.controller, reader=owner.reader)

    def test_preserves_all_occurrences_and_both_order_directions(self):
        result, spec = self.records()
        output = self.evaluate(result, spec)
        self.assertFalse(output['qualification'])
        self.assertEqual(output['timing_samples'], 240)
        self.assertEqual(len(output['cells']), 12)
        for cell in output['cells']:
            self.assertEqual([row['variant'] for row in cell['occurrences']], spec['variant_order'])
            self.assertEqual([row['speed_ratio'] for row in cell['adjacent_pairs']], [2., 2.])
            self.assertEqual(cell['within_variant_later_over_earlier'], {'baseline': 1.5, 'candidate': 1.5})
            self.assertEqual(sum(len(row['samples_ms']) for row in cell['occurrences']), 20)

    def test_uncertain_cleanup_and_incomplete_campaign_cannot_emit_timing_readout(self):
        result, spec = self.records()
        for change in ({'qualification': True}, {'unresolved': True}, {'cleanup_errors': ['uncertain']},
                       {'independent_release_verified': False}, {'timings_usable': False}, {'arms': []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evaluate(dict(result, **change), spec)

    def test_actual_reader_rejects_order_source_counts_and_nonfinite(self):
        original, spec = self.records()
        for mutation in ('order', 'source', 'count', 'nan', 'output'):
            result = copy.deepcopy(original)
            native = result['arms'][0]['native_result']
            if mutation == 'order':
                native['processes'][0], native['processes'][1] = native['processes'][1], native['processes'][0]
            elif mutation == 'source':
                native['spec_sha256'] = '0' * 64
            elif mutation == 'count':
                native['processes'].pop()
            elif mutation == 'nan':
                native['processes'][0]['native']['timings'][0]['synchronized_call_ms'] = float('nan')
            else:
                native['processes'][1]['output_sha256'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.evaluate(result, spec)


if __name__ == '__main__':
    unittest.main(verbosity=2)
