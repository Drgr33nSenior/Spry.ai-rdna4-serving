"""Configured accounting and exclusive seals on actual temporary CPU trees."""
import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
import test_session_timing_proposal as session_tests

PROPOSAL = HERE.parent / 'proposals/d2-durable-limits'
ARTIFACTS = 'review/benchmark_artifacts.py'
TRANSPORT = 'review/benchmark_host_transport.py'
CONTROLLER = 'review/benchmark_physical_controller.py'


class DurableLimits(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-durable-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workload = self.root / 'workload'; self.workload.mkdir()
        self.protected = self.root / 'protected'; self.protected.mkdir()
        self.artifacts = load('cleanup_durable_artifacts', PROPOSAL / ARTIFACTS)
        self.transport = load('cleanup_durable_transport', PROPOSAL / TRANSPORT)
        self.plan = replace(self.artifacts.HTTPArtifactLimits(),
            result_json_bytes=128, client_bundle_bytes=2048, native_bundle_bytes=1024,
            http_stream_bytes=16384, client_tree_bytes=1024, native_tree_bytes=1024,
            arm_metadata_bytes=1024, source_copies_bytes=4096, observer_copies_bytes=4096,
            outer_stdio_bytes=1024, summary_closeout_bytes=8192, small_metadata_bytes=4096,
            session_bytes=1 << 20, http_member_read_bytes=512, client_tree_files=2)

    def put(self, root, name, size):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * size)
        return path

    def audit(self, **kwargs):
        return self.artifacts.audit_durable_trees(self.workload, protected_root=self.protected,
            limits=self.plan, **kwargs)

    def test_real_files_charge_once_to_total_and_conservative_report_partitions(self):
        entries = [(self.workload, 'review/source.py', 10),
            (self.protected, 'payload.tar', 11), (self.protected, 'review/helper.py', 12),
            (self.protected, 'evidence/release-bf16.json', 13),
            (self.workload, 'results/CONTROLLER-RESULT.json', 14),
            (self.protected, 'evidence/ROOT-TRANSPORT.json', 15),
            (self.protected, 'evidence/controller-capture/stdout.raw', 16),
            (self.workload, 'results/bf16-capture/stderr.raw', 17),
            (self.workload, 'results/kv4-artifacts/result.json', 18),
            (self.workload, 'results/bf16-http-client-tree/request.json', 19),
            (self.workload, 'results/bf16-diagnostics.json', 20),
            (self.workload, 'GRANT.json', 21)]
        for root, name, size in entries: self.put(root, name, size)
        self.workload.with_name('workload.closeout.json').write_bytes(b'{}')
        report = self.audit(pending=(('protected', 'FINAL-ARTIFACTS.sha256', 22),))
        self.assertEqual(report['total_bytes'], sum(row[2] for row in entries) + 24)
        self.assertEqual(report['files'], len(entries) + 2)
        used = report['partition_bytes']
        self.assertEqual(used['source_copies_bytes'], 33)
        self.assertEqual(used['observer_copies_bytes'], 42)
        self.assertEqual(used['summary_closeout_bytes'], 31)
        self.assertEqual(used['small_metadata_bytes'], 43)
        self.assertFalse(report['qualification'])

    def test_each_outer_partition_refuses_in_isolation(self):
        cases = [(self.workload, 'review/source.py', 'source_copies_bytes'),
            (self.protected, 'evidence/snapshot.json', 'observer_copies_bytes'),
            (self.workload, 'results/CONTROLLER-RESULT.json', 'summary_closeout_bytes'),
            (self.workload, 'unclassified.json', 'small_metadata_bytes')]
        for root, name, key in cases:
            with self.subTest(key=key):
                path = self.put(root, name, getattr(self.plan, key) + 1)
                with self.assertRaisesRegex(ValueError, 'configured durable partition'):
                    self.audit()
                path.unlink()
        self.put(self.protected, 'evidence/controller-capture/stdout.raw', 513)
        with self.assertRaisesRegex(ValueError, 'outer stream limit'): self.audit()

    def test_each_arm_member_stream_tree_and_metadata_partition_is_enforced(self):
        for arm in ('bf16', 'kv4'):
            cases = [(f'results/{arm}-artifacts/{name}', maximum + 1)
                     for name, maximum in self.plan.members.items()]
            cases += [(f'results/{arm}-capture/{name}.raw', 16385) for name in ('stdout', 'stderr')]
            cases += [(f'results/{arm}-diagnostics.json', 1025)]
            for name, size in cases:
                with self.subTest(name=name):
                    path = self.put(self.workload, name, size)
                    with self.assertRaisesRegex(ValueError, 'configured durable partition'):
                        self.audit()
                    path.unlink()

    def test_client_file_count_individual_size_and_aggregate_refuse(self):
        base = 'results/bf16-http-client-tree/'
        path = self.put(self.workload, base + 'one', 513)
        with self.assertRaisesRegex(ValueError, 'decoded member bound'): self.audit()
        path.write_bytes(b'x' * 512)
        self.put(self.workload, base + 'two', 512)
        self.put(self.workload, base + 'three', 0)
        with self.assertRaisesRegex(ValueError, 'file count'): self.audit()
        self.plan = replace(self.plan, client_tree_files=3)
        self.put(self.workload, base + 'three', 1)
        with self.assertRaisesRegex(ValueError, 'durable partition'): self.audit()

    def test_unknown_members_and_duplicate_or_unsafe_pending_seals_refuse(self):
        path = self.put(self.workload, 'results/kv4-artifacts/unknown', 0)
        with self.assertRaisesRegex(ValueError, 'unexpected configured HTTP'): self.audit()
        path.unlink()
        self.put(self.workload, 'FINAL-ARTIFACTS.sha256', 1)
        with self.assertRaisesRegex(ValueError, 'duplicate durable'):
            self.audit(pending=(('workload', 'FINAL-ARTIFACTS.sha256', 1),))
        for row in (('workload', '../outside', 1), ('workload', '/outside', 1),
                    ('unknown', 'file', 1), ('workload', None, 1), ('workload', 'file', True)):
            with self.subTest(row=row), self.assertRaises(ValueError): self.audit(pending=(row,))

    def test_links_overlapping_roots_and_empty_root_selection_refuse(self):
        target = self.put(self.workload, 'file', 1)
        linked = self.workload / 'link'; linked.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'link refused'): self.audit()
        linked.unlink(); os.link(target, linked)
        with self.assertRaisesRegex(ValueError, 'singly linked'): self.audit()
        linked.unlink()
        with self.assertRaisesRegex(ValueError, 'independent durable'):
            self.artifacts.audit_durable_trees(self.workload, protected_root=self.workload, limits=self.plan)
        with self.assertRaisesRegex(ValueError, 'root required'):
            self.artifacts.audit_durable_trees(None, limits=self.plan)

    def test_pending_seal_is_counted_before_any_file_is_written(self):
        self.plan = replace(self.plan, small_metadata_bytes=1)
        with self.assertRaisesRegex(ValueError, 'small_metadata_bytes'):
            self.audit(pending=(('protected', 'FINAL-ARTIFACTS.sha256', 2),))
        self.assertFalse((self.protected / 'FINAL-ARTIFACTS.sha256').exists())

    def test_closeout_and_refusal_files_share_the_summary_partition(self):
        self.protected.with_name('protected.closeout.json').write_bytes(b'x' * 4096)
        self.protected.with_name('protected.refusal.json').write_bytes(b'x' * 4097)
        with self.assertRaisesRegex(ValueError, 'summary_closeout_bytes'): self.audit()

    def test_root_seal_audits_combined_source_copies_and_its_own_manifest(self):
        self.put(self.workload, 'review/work.py', 100)
        self.put(self.protected, 'review/root.py', 100)
        self.put(self.protected, 'payload.tar', 100)
        result = self.transport.seal(self.protected, deadline=float('inf'), artifacts=self.artifacts,
                                     limits=self.plan, workload_root=self.workload)
        self.assertEqual(result['durable_budget']['total_bytes'],
            300 + (self.protected / 'FINAL-ARTIFACTS.sha256').stat().st_size)
        self.assertEqual(result['durable_budget']['partition_bytes']['source_copies_bytes'], 300)
        self.assertEqual(self.audit()['total_bytes'], result['durable_budget']['total_bytes'])

    def test_root_seal_refuses_oversize_source_and_pending_manifest_before_publication(self):
        self.put(self.workload, 'review/work.py', 2048)
        path = self.put(self.protected, 'payload.tar', 2049)
        with self.assertRaisesRegex(ValueError, 'source_copies_bytes'):
            self.transport.seal(self.protected, deadline=float('inf'), artifacts=self.artifacts,
                                limits=self.plan, workload_root=self.workload)
        path.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'small_metadata_bytes'):
            self.transport.seal(self.protected, deadline=float('inf'), artifacts=self.artifacts,
                limits=replace(self.plan, small_metadata_bytes=1), workload_root=self.workload)
        self.assertFalse((self.protected / 'FINAL-ARTIFACTS.sha256').exists())

    def test_root_outer_streams_and_closeout_use_remaining_budgets(self):
        self.assertEqual(self.transport.outer_stream_limit(self.plan), 512)
        with self.assertRaisesRegex(ValueError, 'two streams'):
            self.transport.outer_stream_limit(replace(self.plan, outer_stdio_bytes=1))
        budget = dict(remaining_summary_bytes=10, remaining_session_bytes=10)
        self.assertEqual(self.transport.closeout_bytes({}, budget=budget), b'{}\n')
        for key in budget:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'closeout partition'):
                self.transport.closeout_bytes({}, budget=dict(budget, **{key: 1}))

    def test_root_loader_uses_checked_bytes_with_dataclasses_and_restores_failed_registration(self):
        path = self.put(self.protected, 'module.py', 0)
        path.write_text('raise RuntimeError("untrusted replacement")\n')
        name = 'cleanup_checked_root_fixture'
        self.addCleanup(sys.modules.pop, name, None)
        raw = b'from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Value:\n    count: int = 3\n'
        loaded = self.transport.authenticated_module(name, path, raw)
        self.assertEqual(loaded.Value().count, 3)
        self.assertIs(sys.modules[name], loaded)
        with self.assertRaises(RuntimeError):
            self.transport.authenticated_module(name, path, b'raise RuntimeError("fixture")\n')
        self.assertIs(sys.modules[name], loaded)
        sys.modules.pop(name)
        with self.assertRaises(SyntaxError): self.transport.authenticated_module(name, path, b'broken !')
        self.assertNotIn(name, sys.modules)

    def test_shipped_root_source_has_no_direct_exec_or_compile(self):
        tree = ast.parse((PROPOSAL / TRANSPORT).read_bytes())
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id in ('exec', 'compile') for node in ast.walk(tree)))


class ControllerDurableSeal(session_tests.ControllerSeal):
    def setUp(self):
        with patch.object(session_tests, 'PROPOSAL', PROPOSAL): super().setUp()
        self.artifacts = load('cleanup_controller_seal_artifacts', PROPOSAL / ARTIFACTS)
        self.controller._adjacent = lambda *args: self.artifacts

    def seal(self, plan):
        return self.controller.seal_result(self.root, timing=session_tests.baseline_timing(),
            received_monotonic=100, received_utc='2026-01-01T00:00:00Z', clock=lambda: 101,
            artifact_limits=asdict(plan))

    def test_configured_seal_includes_manifest_and_bounds_the_outside_closeout(self):
        result = self.seal(self.artifacts.HTTPArtifactLimits())
        budget = result['durable_budget']
        self.assertEqual(budget['total_bytes'], 3 + (self.root / 'FINAL-ARTIFACTS.sha256').stat().st_size)
        raw = self.root.with_name('result.closeout.json').read_bytes()
        self.assertEqual(json.loads(raw), result)
        self.assertLessEqual(len(raw), budget['remaining_summary_bytes'])

    def test_bad_plan_and_pending_manifest_overflow_refuse_before_the_seal(self):
        for plan in (replace(self.artifacts.HTTPArtifactLimits(), small_metadata_bytes=3),):
            with self.assertRaisesRegex(ValueError, 'small_metadata_bytes'): self.seal(plan)
        self.assertFalse((self.root / 'FINAL-ARTIFACTS.sha256').exists())

    def test_configured_summary_exhaustion_cannot_publish_an_oversized_closeout(self):
        with self.assertRaisesRegex(ValueError, 'bounded controller result'):
            self.seal(replace(self.artifacts.HTTPArtifactLimits(), summary_closeout_bytes=1))
        self.assertFalse(self.root.with_name('result.closeout.json').exists())


if __name__ == '__main__': unittest.main()
