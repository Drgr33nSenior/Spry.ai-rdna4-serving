"""CPU substitutes for clock/request metadata; no model or device evidence."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sampling_host_timing', ROOT/'prototype/serving/sampling_host_timing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SamplingHostTiming = module.SamplingHostTiming


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 10
        return self.now


def bindings():
    return [dict(case='synthetic-'+str(i), request_sha256=str(i+1)*64,
                 input_ids_sha256=str(i+5)*64, cache_salt='salt-'+str(i), input_tokens=2,
                 reference_verifier_cycles=count, cycles=[9+j*(count-10)//31 for j in range(32)])
            for i, count in enumerate((62, 188, 357, 386))]


class SamplingHostTimingTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.source = bindings()
        self.timer = SamplingHostTiming(self.source, clock=self.clock)
        self.timer.bind_route(dict(rank=0, pid=123, source_sha256='a'*64))

    def request(self, ordinal=0):
        row = self.source[ordinal]
        return dict(rid='request-'+str(ordinal), input_ids_sha256=row['input_ids_sha256'],
                    cache_salt=row['cache_salt'])

    def call(self, ordinal=0, *, failed=False, prefix=2):
        ticket = self.timer.before_native(self.clock(), self.request(ordinal), prefix)
        self.timer.finish(ticket, self.clock(), native_exception=failed)

    def test_full_reference_windows_retain_exactly_128_rows_without_limiting_native_calls(self):
        for ordinal, row in enumerate(self.source):
            for cycle in range(1, row['reference_verifier_cycles']+1):
                self.call(ordinal, prefix=cycle+2)
        snapshot = self.timer.snapshot()
        self.assertEqual(len(snapshot['records']), 128)
        self.assertEqual(sum(x['successful_calls'] for x in snapshot['coverage']), 993)
        for ordinal, coverage in enumerate(snapshot['coverage']):
            self.assertEqual(coverage['sampled_cycles'], self.source[ordinal]['cycles'])
            self.assertEqual(coverage['missing_scheduled_cycles'], [])
            self.assertEqual(coverage['uncovered_tail_cycles'], 1)

    def test_host_intervals_partition_elapsed_and_do_not_claim_gpu_time(self):
        for _ in range(9):
            self.call()
        row = self.timer.snapshot()['records'][0]
        self.assertEqual(row['cycle'], 9)
        self.assertEqual([row[k] for k in ('observer_before_native_ns', 'native_accept_host_ns',
                                          'observer_after_native_ns', 'host_total_ns')], [10, 10, 10, 30])
        self.assertEqual(row['scope'], 'HOST_ELAPSED_NOT_GPU_TIME')
        self.assertTrue(row['excludes_record_assembly'])
        self.assertFalse(self.timer.snapshot()['device_time_measured'])

    def test_early_stop_retains_missing_schedule_and_actual_prefix_range(self):
        for cycle in range(1, 11):
            self.call(prefix=cycle+2)
        snapshot = self.timer.snapshot()
        coverage = snapshot['coverage'][0]
        self.assertEqual(coverage['sampled_cycles'], [9, 10])
        self.assertEqual(coverage['sampled_prefix_range'], [11, 12])
        self.assertEqual(len(coverage['missing_scheduled_cycles']), 30)
        self.assertFalse(snapshot['whole_request_bottleneck_established'])

    def test_unknown_readiness_traffic_cannot_consume_samples(self):
        request = self.request()
        request['cache_salt'] = 'readiness'
        ticket = self.timer.before_native(self.clock(), request, 2)
        self.assertIsNone(ticket)
        self.timer.finish(ticket, self.clock())
        self.assertEqual(self.timer.snapshot()['records'], [])
        self.assertEqual(sum(self.timer.attempts.values()), 0)

    def test_native_exception_is_retained_without_success_credit(self):
        for _ in range(8):
            self.call()
        self.call(failed=True)
        snapshot = self.timer.snapshot()
        self.assertEqual(snapshot['records'][0]['status'], 'NATIVE_EXCEPTION')
        self.assertEqual(snapshot['coverage'][0]['successful_calls'], 8)
        self.assertEqual(snapshot['coverage'][0]['attempted_cycles'], 9)

    def test_active_duplicate_and_stale_tickets_are_refused(self):
        ticket = self.timer.before_native(self.clock(), self.request(), 2)
        with self.assertRaisesRegex(ValueError, 'active call ownership'):
            self.timer.before_native(self.clock(), self.request(), 2)
        with self.assertRaisesRegex(ValueError, 'unfinished native call'):
            self.timer.snapshot()
        self.timer.finish(ticket, self.clock())
        with self.assertRaisesRegex(ValueError, 'stale or duplicate'):
            self.timer.finish(ticket, self.clock())

    def test_bindings_and_snapshots_are_detached_and_route_binds_once(self):
        self.source[0]['cycles'].clear()
        self.assertEqual(len(self.timer.bindings[0]['cycles']), 32)
        snapshot = self.timer.snapshot()
        snapshot['route_identity']['pid'] = 999
        self.assertEqual(self.timer.snapshot()['route_identity']['pid'], 123)
        with self.assertRaisesRegex(ValueError, 'already bound'):
            self.timer.bind_route({})

    def test_request_replay_or_cross_case_id_reuse_is_refused(self):
        self.call()
        request = self.request()
        request['rid'] = 'another-request'
        with self.assertRaisesRegex(ValueError, 'another ID'):
            self.timer.before_native(self.clock(), request, 2)
        request = self.request(1)
        request['rid'] = 'request-0'
        with self.assertRaisesRegex(ValueError, 'ID reused'):
            self.timer.before_native(self.clock(), request, 2)

    def test_schedule_hash_duplicates_and_boolean_indices_are_refused(self):
        for change in ('cycles', 'hash', 'duplicate', 'bool'):
            bad = bindings()
            if change == 'cycles':
                bad[0]['cycles'][-1] -= 1
            elif change == 'hash':
                bad[0]['request_sha256'] = 'invalid'
            elif change == 'duplicate':
                bad[1]['input_ids_sha256'] = bad[0]['input_ids_sha256']
                bad[1]['cache_salt'] = bad[0]['cache_salt']
            else:
                bad[0]['cycles'][0] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                SamplingHostTiming(bad)

    def test_call_bound_is_separate_from_retained_row_bound(self):
        self.timer.attempts[0] = self.timer.CALL_LIMIT
        with self.assertRaisesRegex(ValueError, 'finite native-call'):
            self.timer.before_native(self.clock(), self.request(), 2)
        self.assertEqual(self.timer.records, [])

    def test_clock_regression_is_refused(self):
        ticket = self.timer.before_native(self.clock(), self.request(), 2)
        with self.assertRaisesRegex(ValueError, 'monotonic interval'):
            self.timer.finish(ticket, ticket['native_enter_ns']-1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
