"""Check the fresh authoring renderer with complete authenticated donor inputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_approval_boot_packet as boot_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_config_binding_proposal import sha
from test_source_builder import tree_bytes
from test_packet_projection import renderer


class PacketProjectionPacket(boot_checks.ApprovalBootPacket):
    evidence = {}

    def test_fresh_proposal_is_deterministic_and_never_publishes_launch_inputs(self):
        outputs = [self.root / name for name in ('render-first', 'render-second')]
        reports = [renderer.build(inventory_checks.PACKET.resolve(),
            output, packet_id='e201110100000002', proposal=getattr(self,'projection_proposal',boot_checks.PROPOSAL)) for output in outputs]
        self.assertEqual(tree_bytes(outputs[0]), tree_bytes(outputs[1]))
        self.assertEqual(reports[0], reports[1])
        root = outputs[0]
        for forbidden in ('MANIFEST.json', 'LAUNCHER-BINDINGS.json', 'RUN-AUTHORIZATION.json', 'RUN-ON-WORKSTATION.sh'):
            self.assertFalse(list(root.rglob(forbidden)))
        manifest = json.loads((root / 'package/MANIFEST-PROPOSAL.json').read_bytes())
        binding = json.loads((root / 'caller/LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
        approval = json.loads((root / 'APPROVAL-TEMPLATE.json').read_bytes())
        self.assertEqual(manifest['schema'], 3); self.assertNotIn('boot_id', manifest)
        self.assertNotIn('boot_id', binding); self.assertIsNone(approval['boot_id'])
        self.assertFalse(approval['model_start_authorized']); self.assertEqual(approval['maximum_starts'], 0)
        self.assertEqual(binding['manifest_sha256'], sha((root / 'package/MANIFEST-PROPOSAL.json').read_bytes()))
        self.assertEqual(binding['protected_set_sha256'], sha((root / 'package/PROTECTED-SHA256.json').read_bytes()))
        self.assertEqual(reports[0]['proposed_package_members'], 80)
        self.assertTrue(reports[0]['source_builder_protected_sources_unchanged'])
        self.evidence['fresh_manifest_caller_projection'] = dict(
            deterministic=True, content_sha256=reports[0]['content_sha256'], proposed_package_members=80,
            no_launchable_manifest_or_authorization=True, caller_activation_complete=False)

    def test_projected_package_authenticates_and_the_staged_caller_reads_its_actual_host_config(self):
        output = self.root / 'render-authentication'
        renderer.build(inventory_checks.PACKET.resolve(),
            output, packet_id='e201110100000003', proposal=getattr(self,'projection_proposal',boot_checks.PROPOSAL))
        package = output / 'package'
        members = tree_bytes(package)
        members['MANIFEST.json'] = members.pop('MANIFEST-PROPOSAL.json')
        binding = json.loads((output / 'caller/LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
        pins = {name: binding[name] for name in ('run_config_sha256', 'host_sha256', 'protected_set_sha256')}
        archive = self.archive('projected-review-cpu-only.tar', members)
        transport = load('cleanup_projected_transport', package / 'review/benchmark_host_transport.py')
        manifest, payload = transport.inspect_archive(archive, sha(archive.read_bytes()),
            sha(members['MANIFEST.json']), configuration_pins=pins)
        self.assertEqual(payload, members)
        self.assertEqual(manifest['target_root'], '/home/developer/fp8-kv-benchmark.e201110100000003')
        mirror_source = self.root / 'projected-controller-fixture'
        shutil.copytree(package, mirror_source)
        (mirror_source / 'MANIFEST-PROPOSAL.json').rename(mirror_source / 'MANIFEST.json')
        with self.controller(mirror_source) as (controller, path, digest, value, root):
            configured = controller.verify_configured_spec(pins, json.loads(path.read_bytes()),
                deadline=time.monotonic() + 105)
            self.assertEqual(configured['spec_sha256'], manifest['spec_sha256'])
            with patch.object(self, 'pins', dict(pins, boot_id=boot_checks.BOOT_B)):
                error, frames = self.prelaunch(controller, path, digest, value, root)
            self.assertIsInstance(error, FileNotFoundError)
            self.assertTrue(any('proof, _ = adapter.readout.read_json(runtime / proof_name' in
                                (frame.line or '') for frame in frames))
        caller = output / 'caller'
        with patch.dict(sys.modules):
            for name in ('connection', 'local_commands'): sys.modules.pop(name, None)
            sys.path.insert(0, str(caller))
            try:
                connection = __import__('connection'); commands = __import__('local_commands')
                with patch.object(connection, 'check') as check, patch.object(commands, 'run') as run:
                    power = load('cleanup_projected_power_observer', caller / 'power_observe.py')
                    host = power.read_host(binding)
                    self.assertEqual(host['power_cap_watts'], 235)
                    check.assert_not_called(); run.assert_not_called()
            finally: sys.path.remove(str(caller))
        self.evidence['projected_source_authentication'] = dict(archive_members=len(payload),
            actual_configuration_and_spec_verified=True, external_runtime_proof_still_required=True,
            configured_caller_host_read=True, SSH_or_native_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report:
        from test_config_binding_proposal import canonical
        report = dict(PacketProjectionPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream: stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
