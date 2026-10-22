"""Declared CPU event substitutes; no device timing or kernel evidence."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('device_phase_timing', ROOT/'prototype/serving/device_phase_timing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Cuda:
    def __init__(self):
        self.capture = False
        self.events = []
        self.device_index = 0

    def current_device(self):
        return self.device_index

    def is_current_stream_capturing(self):
        return self.capture

    def Event(self, **kwargs):
        if self.capture or kwargs != dict(enable_timing=True, external=True):
            raise AssertionError('event initialization boundary')
        owner = self

        class Event:
            def __init__(self):
                self.ready, self.elapsed, self.reads, self.queries, self.records = True, 1.0, 0, 0, 0
                self.device_index = None

            def record(self):
                if self.device_index is None:
                    self.device_index = owner.current_device()
                if self.device_index != owner.current_device():
                    raise RuntimeError('Event device')
                self.records += 1

            def query(self):
                self.queries += 1
                return self.ready

            def elapsed_time(self, end):
                self.reads += 1
                return self.elapsed

        event = Event()
        owner.events.append(event)
        return event


class DevicePhaseTests(unittest.TestCase):
    def setUp(self):
        self.cuda, self.output = Cuda(), []
        self.timer = module.DevicePhaseEvents(self.cuda, emit=lambda kind, row: self.output.append((kind, row)))
        self.capture_all()

    def capture_all(self):
        for role in ('draft', 'target'):
            for size in (1, 2):
                self.timer.prepare_graph(role, size)
                self.cuda.capture = True
                self.timer.arm_capture(role, size)
                for part in self.timer.GRAPH_PARTS[role]:
                    ticket = self.timer.capture_begin(part)
                    self.timer.capture_end(ticket)
                self.cuda.capture = False
        self.timer.finish_capture()
        for name, pair in self.timer.pairs.items():
            part = name[-1] if isinstance(name, tuple) else name
            pair[0].elapsed = {'cycle': 100, 'graph': 10, 'candidate_total': 3,
                               'accept': 20, 'target_probs': 10}.get(part, 1)

    def cycle(self, *, selected=True, cycle=9, joined=True):
        self.timer.start_cycle(dict(case='synthetic', cycle=cycle, ordinal=0, rid='r', selected=selected))
        for role in ('draft', 'target'):
            ticket = self.timer.begin_graph(role, 1)
            self.timer.end_graph(ticket)
        for name in self.timer.EAGER_PARTS:
            if name != 'cycle':
                ticket = self.timer.begin(name)
                self.timer.end(ticket)
        self.timer.finish_cycle(joined)

    def test_roles_buckets_and_native_generations_are_separate(self):
        self.assertIsNot(self.timer.pairs[('draft', 1, 'graph')][0], self.timer.pairs[('target', 1, 'graph')][0])
        self.cycle()
        self.timer.collect()
        row = self.timer.records[0]
        self.assertEqual(row['graph_generations'], dict(draft=1, target=1))
        self.assertEqual(row['status'], 'COMPLETE')
        self.assertEqual(row['device_intervals_ms']['draft.1.local_topk'], 1)

    def test_rank_device_is_selected_after_observer_construction(self):
        self.cuda, self.output = Cuda(), []
        self.timer = module.DevicePhaseEvents(self.cuda, emit=lambda kind, row: self.output.append((kind, row)))
        self.assertEqual(self.cuda.events, [])
        self.cuda.device_index = 1  # Native ModelRunner selects rank 1 after the hook initializes.
        self.capture_all()
        self.cycle()
        self.timer.collect()
        self.assertEqual(self.timer.records[0]['status'], 'COMPLETE')
        self.assertTrue(all(event.device_index == 1 for event in self.cuda.events))

    def test_runtime_device_switch_is_refused_before_event_reuse(self):
        self.cycle()
        before = sum(event.records for event in self.cuda.events)
        self.cuda.device_index = 1
        with self.assertRaisesRegex(ValueError, 'event device changed'):
            self.timer.start_cycle(dict(selected=True))
        self.assertEqual(sum(event.records for event in self.cuda.events), before)
        self.assertTrue(all(event.reads == 0 for event in self.cuda.events))

    def test_unsampled_cycle_collects_before_reusing_events(self):
        self.cycle()
        self.cycle(selected=False, cycle=10)
        self.assertEqual(len(self.timer.records), 1)
        self.assertEqual(self.timer.records[0]['graph_generations'], dict(draft=1, target=1))
        self.assertEqual(self.timer.generations[('target', 1)], 2)

    def test_fresh_outer_completion_blocks_old_ready_inner_events(self):
        self.cycle()
        self.timer.pairs['cycle'][1].ready = False
        self.timer.collect()
        self.assertEqual(self.timer.records[0]['status'], 'NOT_READY')
        self.assertTrue(all(e.reads == 0 for e in self.cuda.events))
        self.assertNotIn('device_intervals_ms', self.timer.records[0])

    def test_overwritten_generation_refused_before_elapsed_reads(self):
        self.cycle()
        self.timer.generations[('draft', 1)] += 1
        with self.assertRaisesRegex(ValueError, 'overwritten event generation'):
            self.timer.collect()
        self.assertTrue(all(e.reads == 0 for e in self.cuda.events))

    def test_native_calls_are_not_limited_to_128_retained_rows(self):
        for cycle in range(1, 513):
            self.cycle(selected=cycle in (9, 100, 300, 500), cycle=cycle)
        self.assertEqual(self.timer.sequence, 512)
        self.assertEqual(len(self.timer.records), 4)
        self.timer.sequence = 36864
        with self.assertRaisesRegex(ValueError, 'native-call observation bound'):
            self.timer.start_cycle(None)

    def test_final_pending_sample_can_remain_uncollected_without_sync(self):
        self.cycle()
        self.timer.pairs[('target', 1, 'graph')][1].ready = False
        row = self.timer.snapshot()['records'][0]
        self.assertEqual(row['status'], 'FINAL_UNCOLLECTED')
        self.assertEqual(row['extra_synchronizations'], 0)
        self.assertEqual([kind for kind, _ in self.output], ['pending', 'collected'])

    def test_duplicate_capture_and_recapture_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'unexpected recapture'):
            self.timer.prepare_graph('draft', 1)
        self.cuda.capture = True
        self.timer.arm_capture('draft', 1)
        with self.assertRaisesRegex(ValueError, 'duplicate captured span'):
            self.timer.capture_begin('projection')

    def test_all_required_spans_and_join_are_required(self):
        self.timer.start_cycle(dict(selected=True))
        with self.assertRaisesRegex(ValueError, 'request/replay/commit join'):
            self.timer.finish_cycle(False)
        for role in ('draft', 'target'):
            self.timer.end_graph(self.timer.begin_graph(role, 1))
        with self.assertRaisesRegex(ValueError, 'incomplete phase coverage'):
            self.timer.finish_cycle(True)

    def test_nested_interval_violation_is_not_accepted(self):
        self.cycle()
        self.timer.pairs[('target', 1, 'head')][0].elapsed = 20
        with self.assertRaisesRegex(ValueError, 'nested interval containment'):
            self.timer.collect()

    def test_incomplete_startup_and_allocation_inside_capture_are_refused(self):
        cuda = Cuda()
        other = module.DevicePhaseEvents(cuda)
        with self.assertRaisesRegex(ValueError, 'both roles/buckets'):
            other.finish_capture()
        cuda.capture = True
        with self.assertRaisesRegex(ValueError, 'initialize outside capture'):
            module.DevicePhaseEvents(cuda)


if __name__ == '__main__':
    unittest.main(verbosity=2)
