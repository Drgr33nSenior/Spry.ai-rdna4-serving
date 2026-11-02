"""Exercise boot-independent archives through actual transport and controller APIs."""
import argparse
from datetime import datetime, timezone, timedelta
import difflib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_quality_policy_packet as quality_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical, sha
from test_contract_cleanup import load
from test_approval_boot import BOOT_A, BOOT_B

PROPOSAL = HERE.parent / 'proposals/d2-approval-boot'


class ApprovalBootPacket(quality_checks.QualityPolicyPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        with patch.object(quality_checks.durable_checks, 'PROPOSAL', PROPOSAL): super().setUpClass()

    def test_exact_diff_and_unmodified_release_validators(self):
        expected = ''
        for name in ('benchmark_host_transport.py', 'benchmark_physical_controller.py'):
            relative = 'review/' + name
            before = HERE.parent / 'proposals/d2-durable-limits' / relative
            after = PROPOSAL / relative
            expected += ''.join(difflib.unified_diff(before.read_text().splitlines(True),
                after.read_text().splitlines(True), fromfile='a/' + relative, tofile='b/' + relative))
            self.assertEqual(sha(before.read_bytes()), self.record['changes'][relative]['baseline_sha256'])
        self.assertEqual(expected, (PROPOSAL / 'D2.patch').read_text())
        for name, pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/' + name]), pin)
        self.evidence.update(exact_patch_verified=True, release_validators_unchanged=True,
            owner_adapter_unchanged=True, transport_unchanged=False, approval_boot_binding=True)

    def test_two_fresh_boot_approvals_validate_against_the_same_manifest_bytes(self):
        records = []
        with self.controller() as (controller, path, _, _, _):
            manifest = json.loads(path.read_bytes())
            manifest.update(schema=3, scope='matched-benchmark-physical-package-v3'); manifest.pop('boot_id')
            raw = canonical(manifest); path.write_bytes(raw); pin = sha(raw)
            timing = controller.transport.SessionTiming.from_limits(
                json.loads(self.members['config/runs/w5-v2/run.json'])['limits'])
            for boot in (BOOT_A, BOOT_B):
                parent = dict(pid=100, uid=0, start_ticks=200, boot_id=boot)
                grant = controller.transport.new_grant(manifest, pin, received_utc='2026-01-01T00:00:00Z',
                    received_monotonic=100, parent=parent, configuration_pins=self.pins,
                    timing=timing, approved_boot_id=boot)
                utc = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1)
                self.assertEqual(controller.validate_physical_grant(grant, manifest, pin, parent,
                    clock=lambda:101, utc_now=utc), grant)
                self.assertEqual(path.read_bytes(), raw)
                stale = dict(parent, boot_id=BOOT_B if boot == BOOT_A else BOOT_A)
                with self.assertRaisesRegex(ValueError, 'authenticated human-root parent'):
                    controller.validate_physical_grant(grant, manifest, pin, stale, clock=lambda:101, utc_now=utc)
                records.append(dict(boot=boot, manifest_sha256=pin, maximum_starts=grant['maximum_starts']))
        self.evidence['same_manifest_fresh_boot_approvals'] = dict(cases=records, stale_parent_refused=True,
            grants='CPU objects only; no approval or invocation was issued')

    def test_boot_independent_archive_keeps_the_80_member_ceiling(self):
        members = dict(self.members); manifest = json.loads(members['MANIFEST.json'])
        manifest.update(schema=3, scope='matched-benchmark-physical-package-v3'); manifest.pop('boot_id')
        members['MANIFEST.json'] = canonical(manifest)
        archive = self.archive('boot-independent.tar', members)
        transport = load('cleanup_boot_archive', self.package / 'review/benchmark_host_transport.py')
        actual, payload = transport.inspect_archive(archive, sha(archive.read_bytes()),
            sha(members['MANIFEST.json']), configuration_pins=self.pins)
        self.assertEqual(payload, members); self.assertNotIn('boot_id', actual)
        self.assertEqual(transport._archive_member_limit(actual, payload), 80)
        self.evidence['boot_independent_archive'] = dict(members=len(payload), ceiling=80, boot_id_absent=True)

    def test_v3_prelaunch_requires_approval_boot_and_retains_the_external_proof_gate(self):
        with self.controller() as (controller, path, _, _, root):
            manifest = json.loads(path.read_bytes())
            manifest.update(schema=3, scope='matched-benchmark-physical-package-v3'); manifest.pop('boot_id')
            path.write_bytes(canonical(manifest)); pin = sha(path.read_bytes())
            error, frames = self.prelaunch(controller, path, pin, manifest, root)
            self.assertIsInstance(error, ValueError)
            self.assertIn('explicit approval boot identity', str(error))
            self.assertFalse(any('proof, _ = ' in (frame.line or '') for frame in frames))
        with self.controller() as (controller, path, _, _, root):
            manifest = json.loads(path.read_bytes())
            manifest.update(schema=3, scope='matched-benchmark-physical-package-v3'); manifest.pop('boot_id')
            path.write_bytes(canonical(manifest)); pin = sha(path.read_bytes())
            with patch.object(self, 'pins', dict(self.pins, boot_id=BOOT_B)):
                error, frames = self.prelaunch(controller, path, pin, manifest, root)
            self.assertIsInstance(error, FileNotFoundError)
            self.assertTrue(any('proof, _ = adapter.readout.read_json(runtime / proof_name' in
                                (frame.line or '') for frame in frames))
        self.evidence['boot_independent_prelaunch'] = dict(missing_boot_refused_before_external_proof=True,
            approved_boot_preserves_external_proof_gate=True, full_native_prelaunch_completed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    quality_checks.durable_checks.binding_checks.host_checks.stop_checks.artifact_checks.worker_checks.session_checks.inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report:
        report = dict(ApprovalBootPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream: stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
