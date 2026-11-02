"""Boot-independent archive bytes and explicit per-invocation boot approval."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
from test_config_binding_proposal import canonical, sha
import test_transport_binding_proposal as transport_tests

PROPOSAL = HERE.parent / 'proposals/d2-approval-boot'
BOOT_A = '12345678-1234-1234-1234-123456789abc'
BOOT_B = '87654321-4321-4321-4321-cba987654321'


class ApprovalBoot(unittest.TestCase):
    def setUp(self):
        self.transport = load('cleanup_approval_boot', PROPOSAL / 'review/benchmark_host_transport.py')
        for item in (patch.object(transport_tests, 'transport', self.transport),
                     patch.object(transport_tests, 'PROPOSAL', PROPOSAL)):
            item.start(); self.addCleanup(item.stop)
        self.fixture = transport_tests.TransportBinding()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.manifest = dict(self.fixture.manifest, schema=3, scope='matched-benchmark-physical-package-v3')
        self.manifest.pop('boot_id')
        self.timing = self.transport.SessionTiming.from_limits(
            json.loads(self.fixture.members[self.fixture.run_name])['limits'])

    def grant(self, boot=BOOT_A, **changes):
        options = dict(received_utc='2026-01-01T00:00:00Z', received_monotonic=100,
            parent=dict(self.fixture.parent, boot_id=boot), configuration_pins=self.fixture.pins,
            timing=self.timing, approved_boot_id=boot)
        options.update(changes)
        return self.transport.new_grant(self.manifest, sha(canonical(self.manifest)), **options)

    def test_reboots_change_only_runtime_binding_and_grant_not_manifest_bytes(self):
        before = canonical(self.manifest)
        for boot in (BOOT_A, BOOT_B):
            bound = self.transport.bind_manifest_boot(self.manifest, boot)
            self.assertEqual(bound['schema'], 2); self.assertEqual(bound['boot_id'], boot)
            grant = self.grant(boot)
            self.assertEqual(grant['manifest_sha256'], sha(before))
            self.assertEqual(grant['boot_id'], boot)
            self.assertEqual(grant['maximum_starts'], 2)
            self.assertEqual(grant['arm_order'], ['bf16', 'kv4'])
        self.assertEqual(canonical(self.manifest), before)
        self.assertNotIn('boot_id', self.manifest)

    def test_new_manifest_requires_explicit_well_formed_approval_boot(self):
        for boot in (None, True, '', 'not-a-boot', BOOT_A.upper()):
            with self.subTest(boot=boot), self.assertRaises(ValueError):
                self.transport.bind_manifest_boot(self.manifest, boot)
        with self.assertRaises(ValueError): self.grant(approved_boot_id=None)

    def test_embedded_boot_and_scope_or_version_drift_refuse(self):
        for changes in ({'boot_id': BOOT_A}, {'schema': 3.0}, {'schema': True},
                        {'scope': 'matched-benchmark-physical-package-v2'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.transport.bind_manifest_boot(dict(self.manifest, **changes), BOOT_A)

    def test_legacy_manifest_keeps_its_pinned_boot_requirement(self):
        original = self.fixture.manifest
        self.assertEqual(self.transport.bind_manifest_boot(original, BOOT_A), original)
        with self.assertRaisesRegex(ValueError, 'legacy manifest/approval boot'):
            self.transport.bind_manifest_boot(original, BOOT_B)

    def test_grant_still_requires_matching_parent_and_configuration_pins(self):
        for changes in ({'parent': dict(self.fixture.parent, boot_id=BOOT_B)},
                        {'configuration_pins': dict(self.fixture.pins, extra='a'*64)},
                        {'configuration_pins': {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.grant(**changes)

    def test_archive_authentication_preserves_boot_independent_bytes(self):
        self.fixture.manifest = copy.deepcopy(self.manifest); self.fixture.refresh_manifest()
        actual, members = self.fixture.inspect()
        self.assertEqual(canonical(actual), canonical(self.manifest))
        self.assertEqual(members['MANIFEST.json'], canonical(self.manifest))
        self.assertNotIn('boot_id', actual)

    def test_archive_rejects_a_v3_embedded_boot_even_when_repinned(self):
        self.fixture.manifest = dict(self.manifest, boot_id=BOOT_A); self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'physical package contract'):
            self.fixture.inspect()

    def test_human_entry_checks_approval_against_observed_boot_before_workload_creation(self):
        fixture = self.fixture
        artifact = 'review/benchmark_artifacts.py'
        fixture.members[artifact] = (PROPOSAL / artifact).read_bytes()
        files = dict(self.manifest['package_files']); files[artifact] = sha(fixture.members[artifact])
        fixture.members['PROTECTED-SHA256.json'] = canonical(files)
        fixture.pins['protected_set_sha256'] = sha(fixture.members['PROTECTED-SHA256.json'])
        fixture.manifest = dict(self.manifest, package_files=files); fixture.refresh_manifest()
        archive, pin = fixture.archive()
        stage = fixture.root / 'root-stage'; review = stage / 'review'; review.mkdir(parents=True)
        for name in ('benchmark_host_transport.py', 'benchmark_artifacts.py', 'host_process_owner.py',
                     'benchmark_release_observer.py', 'collect_benchmark_platform.py'):
            (review / name).write_bytes(fixture.members['review/' + name])
        payload = stage / 'payload.tar'; shutil.copyfile(archive, payload)
        args = SimpleNamespace(original_receipt=(100, '2026-01-01T00:00:00Z'), archive=str(payload),
            archive_sha256=pin, manifest_sha256=fixture.manifest_pin, approved_boot_id=BOOT_A,
            expected_self_sha256=sha(fixture.members['review/benchmark_host_transport.py']), **fixture.pins)
        original_read_text = Path.read_text
        def read_boot(path, *args, **kwargs):
            return BOOT_B if str(path) == '/proc/sys/kernel/random/boot_id' else original_read_text(path, *args, **kwargs)
        with patch.dict(sys.modules), patch.object(self.transport, '__file__', str(review / 'benchmark_host_transport.py')), \
                patch.multiple(self.transport.os, getuid=lambda:0, geteuid=lambda:0, getgid=lambda:0, getegid=lambda:0), \
                patch.object(self.transport, 'protected_stage'), \
                patch.object(self.transport, 'root_source', side_effect=lambda path, pin:path.read_bytes()), \
                patch.object(Path, 'read_text', read_boot), patch.object(Path, 'mkdir') as mkdir, \
                patch.object(self.transport.time, 'monotonic', return_value=100), \
                patch.object(self.transport, 'stage_members') as stage_members, \
                patch.object(self.transport.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'observed boot differs'):
                self.transport.human_main(args)
            mkdir.assert_not_called(); stage_members.assert_not_called(); popen.assert_not_called()
            args.approved_boot_id = BOOT_B
            mkdir.side_effect = RuntimeError('synthetic stop at workload creation')
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop at workload creation'):
                self.transport.human_main(args)
            self.assertEqual(mkdir.call_count, 1)
            stage_members.assert_not_called(); popen.assert_not_called()


if __name__ == '__main__': unittest.main()
