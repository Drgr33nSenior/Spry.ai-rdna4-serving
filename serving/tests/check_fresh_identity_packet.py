"""Validate fresh configured identities with the actual resource gate and owner."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_contract_packet as contract_checks
from test_source_builder import builder,tree_bytes


class FreshIdentityPacket(contract_checks.ContractPacket):
    evidence={}

    def test_fresh_configured_identity_is_deterministic_and_matches_actual_owner_arguments(self):
        outputs=[self.bundles[0].parent/name for name in ('fresh-one','fresh-two','fresh-other')]
        reports=[builder.build(contract_checks.PACKET.resolve(),output,
            run_dir=HERE.parent/'config/runs/w5-v2',packet_id=packet)
            for output,packet in zip(outputs,('fedcba0987654321','fedcba0987654321','0000000000000001'),strict=True)]
        self.assertEqual(reports[0],reports[1])
        self.assertEqual(tree_bytes(outputs[0]),tree_bytes(outputs[1]))
        self.assertNotEqual(reports[0]['content_sha256'],reports[2]['content_sha256'])
        for name,raw in tree_bytes(outputs[0]).items():
            if name not in ('SOURCE-BUILD.json','review/RESOURCE-MANIFEST.json'):
                self.assertEqual(raw,(outputs[2]/name).read_bytes(),name)
        for output,report in zip(outputs,reports,strict=True):
            identity=report['session_identity']
            self.assertFalse(report['launch_authorized'])
            self.assertFalse(identity['launch_authorized'])
            self.assertFalse((output/'MANIFEST.json').exists())
            manifest=dict(self.manifest,target_root=identity['target_root'],session_sha256=identity['session_sha256'])
            with patch.object(self,'configured',[output]),patch.object(self,'configured_reports',[report]), \
                    patch.object(self,'manifest',manifest):
                contract_checks.ContractPacket.test_generated_resource_manifest_passes_actual_gate_and_owner_projection(self)
        self.evidence.update(tests_on_fresh_identities=3,identical_rebuild=True,
            only_identity_artifacts_changed=True,actual_resource_gate_and_owner_projection=True,
            source_bundle_only=True,launch_manifest_created=False,
            content_sha256=[report['content_sha256'] for report in reports],
            identities=[report['session_identity'] for report in reports])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    contract_checks.PACKET=options.packet
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        value=dict(FreshIdentityPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            builder_sha256=builder.digest((HERE.parent/'build.py').read_bytes()),
            test_source_sha256=builder.digest(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(builder.canonical(value))
    raise SystemExit(0 if result.wasSuccessful() else 1)
