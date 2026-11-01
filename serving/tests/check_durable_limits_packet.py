"""Authenticate the full CPU composition and configured cross-root seals."""
import argparse
from dataclasses import asdict, replace
import difflib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_artifact_binding_packet as binding_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical, sha

PROPOSAL = HERE.parent / 'proposals/d2-durable-limits'


class DurableLimitsPacket(binding_checks.ArtifactBindingPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        with patch.object(binding_checks, 'PROPOSAL', PROPOSAL): super().setUpClass()

    def test_exact_diff_and_unmodified_release_validators(self):
        baselines = {'benchmark_artifacts.py': 'd2-artifact-binding',
            'benchmark_physical_controller.py': 'd2-artifact-binding',
            'benchmark_host_transport.py': 'd2-session-timing'}
        expected = ''
        for name, base in baselines.items():
            relative = 'review/' + name
            before = HERE.parent / 'proposals' / base / relative
            after = PROPOSAL / relative
            expected += ''.join(difflib.unified_diff(before.read_text().splitlines(True),
                after.read_text().splitlines(True), fromfile='a/' + relative, tofile='b/' + relative))
            self.assertEqual(sha(before.read_bytes()), self.record['changes'][relative]['baseline_sha256'])
        self.assertEqual(expected, (PROPOSAL / 'D2.patch').read_text())
        for name, pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/' + name]), pin)
        for name, row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent / name).read_bytes()), row['proposed_sha256'])
        self.evidence.update(exact_patch_verified=True, release_validators_unchanged=True,
            owner_adapter_unchanged=True, transport_unchanged=False, configured_durable_partitions=True)

    def test_workload_and_root_seals_charge_the_actual_archive_sources_and_both_closeouts(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-protected-durable-')
        self.addCleanup(temporary.cleanup)
        protected = Path(temporary.name).resolve() / 'protected'
        (protected / 'review').mkdir(parents=True)
        for name in ('host_process_owner.py', 'benchmark_release_observer.py',
                     'collect_benchmark_platform.py', 'benchmark_artifacts.py', 'benchmark_host_transport.py'):
            (protected / 'review' / name).write_bytes(self.members['review/' + name])
        archive = self.archive('durable-input.tar', self.members)
        shutil.copyfile(archive, protected / 'payload.tar')
        with self.controller() as (controller, _, _, _, root):
            artifacts = controller._adjacent('cleanup_durable_packet_artifacts', 'benchmark_artifacts.py')
            plan = artifacts.HTTPArtifactLimits()
            timing = controller.transport.SessionTiming.from_limits(
                json.loads(self.members['config/runs/w5-v2/run.json'])['limits'])
            workload_result = controller.seal_result(root, timing=timing,
                received_monotonic=time.monotonic(), received_utc='2026-01-01T00:00:00Z',
                artifact_limits=asdict(plan))
            result = controller.transport.seal(protected, deadline=time.monotonic() + 300,
                artifacts=artifacts, limits=plan, workload_root=root)
            closeout = controller.transport.closeout_bytes(result, budget=result['durable_budget'])
            protected.with_name('protected.closeout.json').write_bytes(closeout)
            audited = artifacts.audit_durable_trees(root, protected_root=protected, limits=plan)
            actual = sum(p.stat().st_size for folder in (root, protected)
                         for p in folder.rglob('*') if p.is_file())
            actual += root.with_name(root.name + '.closeout.json').stat().st_size + len(closeout)
            self.assertEqual(audited['total_bytes'], actual)
            self.assertEqual(audited['partition_bytes']['summary_closeout_bytes'],
                root.with_name(root.name + '.closeout.json').stat().st_size + len(closeout))
            self.assertFalse(workload_result['qualification'])
            self.assertFalse(audited['qualification'])
            self.evidence['composed_durable_seals'] = dict(actual_bytes=actual,
                source_copies_bytes=audited['partition_bytes']['source_copies_bytes'],
                both_manifests_and_closeouts_counted=True, native_or_owner_execution=False)

    def test_actual_controller_seal_refuses_tight_source_budget_before_writing(self):
        with self.controller() as (controller, _, _, _, root):
            artifacts = controller._adjacent('cleanup_durable_refusal_artifacts', 'benchmark_artifacts.py')
            plan = replace(artifacts.HTTPArtifactLimits(), source_copies_bytes=1)
            timing = controller.transport.SessionTiming.from_limits(
                json.loads(self.members['config/runs/w5-v2/run.json'])['limits'])
            with self.assertRaisesRegex(ValueError, 'source_copies_bytes'):
                controller.seal_result(root, timing=timing, received_monotonic=time.monotonic(),
                    received_utc='2026-01-01T00:00:00Z', artifact_limits=asdict(plan))
            self.assertFalse((root / 'FINAL-ARTIFACTS.sha256').exists())
        self.evidence['configured_source_refusal_before_seal'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    binding_checks.host_checks.stop_checks.artifact_checks.worker_checks.session_checks.inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report:
        report = dict(DurableLimitsPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream: stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
