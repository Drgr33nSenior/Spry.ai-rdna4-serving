"""Negative joins for declared synthetic authenticated-reader inputs."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase_readout', ROOT/'prototype/serving/phase_timing_readout.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(rank=0, pid=123, source_manifest_sha256='a'*64)
        self.bindings = [dict(case='synthetic', request_sha256='b'*64, cycles=[9], input_tokens=2)]
        self.pending = dict(self.identity, ordinal=0, case='synthetic', cycle=9, rid='r', request_sha256='b'*64,
                            host_request_prefix_tokens=10, selected=True, replays_before=8, commits_before=8, event_generation=9)
        values = dict(cycle=100, accept=20, target_probs=10, mamba_commit=1, draft_append=1, next_input=1)
        values.update({'draft.1.graph': 10, 'draft.1.candidate_total': 3, 'draft.1.projection': 1,
                       'draft.1.local_topk': 1, 'draft.1.selector_walk': 1, 'target.1.graph': 10,
                       'target.1.head': 1, 'target.1.gather': 1})
        self.collected = dict(self.pending, status='COMPLETE', graph_generations=dict(draft=9, target=9),
                              device_intervals_ms=values, extra_synchronizations=0, extra_model_forwards=0, qualification=False)
        self.host = {k: self.pending[k] for k in ('ordinal', 'case', 'cycle', 'rid', 'request_sha256', 'host_request_prefix_tokens')}
        self.host.update(status='RETURNED', observer_before_native_ns=10, native_accept_host_ns=10,
                         observer_after_native_ns=10, host_total_ns=30)

    def validate(self, collected=None, host=None, pending=None):
        return module.validate_phase_pair(self.pending if pending is None else pending,
            self.collected if collected is None else collected, self.host if host is None else host,
            identity=self.identity, bindings=self.bindings)

    def test_complete_and_explicit_incomplete_samples(self):
        self.assertEqual(self.validate()['status'], 'COMPLETE')
        for status in ('NOT_READY', 'FINAL_UNCOLLECTED'):
            row = copy.deepcopy(self.collected)
            del row['device_intervals_ms']
            row['status'] = status
            self.assertEqual(self.validate(row)['status'], status)

    def test_wrong_rank_source_request_cycle_or_generation_is_refused(self):
        for key, value in [('rank', 1), ('source_manifest_sha256', 'c'*64), ('rid', 'other'),
                           ('request_sha256', 'c'*64), ('cycle', 10), ('event_generation', 10)]:
            row = copy.deepcopy(self.collected)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(row)

    def test_unscheduled_or_boolean_cycle_is_refused(self):
        for value in (True, 10):
            row = dict(self.pending, cycle=value)
            with self.assertRaises(ValueError):
                self.validate(pending=row)

    def test_host_mismatch_failure_and_invalid_partition_are_refused(self):
        for key, value in [('rid', 'other'), ('status', 'NATIVE_EXCEPTION'), ('host_total_ns', 29)]:
            row = dict(self.host, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(host=row)

    def test_stale_graph_generation_and_invalid_device_intervals_are_refused(self):
        row = copy.deepcopy(self.collected)
        row['graph_generations']['target'] = 8
        with self.assertRaises(ValueError):
            self.validate(row)
        for value in (float('nan'), float('inf'), -1, True, 20):
            row = copy.deepcopy(self.collected)
            row['device_intervals_ms']['target.1.head'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(row)

    def test_target_families_and_nested_projection_communication(self):
        parts = ('gdn_000', 'mlp_000', 'projection_000', 'comm_000')
        self.collected['device_intervals_ms'].update({
            'target.1.gdn_000': 3, 'target.1.mlp_000': 4,
            'target.1.projection_000': 6, 'target.1.comm_000': 3})
        result = module.validate_phase_pair(self.pending, self.collected, self.host,
            identity=self.identity, bindings=self.bindings, target_parts=parts)
        self.assertEqual(result['status'], 'COMPLETE')
        for name, value in [('gdn_000', 5), ('projection_000', 11), ('comm_000', 11)]:
            row = copy.deepcopy(self.collected)
            row['device_intervals_ms']['target.1.' + name] = value
            with self.assertRaises(ValueError):
                module.validate_phase_pair(self.pending, row, self.host,
                    identity=self.identity, bindings=self.bindings, target_parts=parts)

    def test_unbound_duplicate_or_unknown_target_parts_are_refused(self):
        for parts in (['gdn_000'], ('gdn_000', 'gdn_000'), ('unknown_000',), (True,)):
            with self.assertRaises(ValueError):
                module.validate_phase_pair(self.pending, self.collected, self.host,
                    identity=self.identity, bindings=self.bindings, target_parts=parts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
