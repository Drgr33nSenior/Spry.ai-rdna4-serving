"""Reproduce all active W5 overlays and validate the resulting review package."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_fresh_identity_packet as fresh_checks
from test_source_builder import builder,tree_bytes


class PatchSeriesPacket(fresh_checks.FreshIdentityPacket):
    evidence={}

    def test_all_active_overlay_patches_match_retained_bytes_and_actual_resource_gate(self):
        outputs=[self.bundles[0].parent/name for name in ('series-one','series-two','without-series')]
        reports=[builder.build(fresh_checks.contract_checks.PACKET.resolve(),output,
            run_dir=HERE.parent/'config/runs/w5-v2',packet_id='fedcba0987654321',
            upstream_root=UPSTREAM if index<2 else None) for index,output in enumerate(outputs)]
        self.assertEqual(reports[0],reports[1])
        self.assertEqual(tree_bytes(outputs[0]),tree_bytes(outputs[1]))
        self.assertEqual(reports[0]['content_sha256'],reports[2]['content_sha256'])
        for name,raw in tree_bytes(outputs[0]).items():
            if name!='SOURCE-BUILD.json':self.assertEqual(raw,(outputs[2]/name).read_bytes(),name)
        self.assertEqual(len(reports[0]['overlay_series']['overlays']),11)
        identity=reports[0]['session_identity']
        manifest=dict(self.manifest,target_root=identity['target_root'],session_sha256=identity['session_sha256'])
        with patch.object(self,'configured',[outputs[0]]),patch.object(self,'configured_reports',[reports[0]]), \
                patch.object(self,'manifest',manifest):
            fresh_checks.contract_checks.ContractPacket.test_generated_resource_manifest_passes_actual_gate_and_owner_projection(self)
        self.evidence.update(active_overlay_patches=11,all_payload_bytes_identical=True,
            repeated_build_identical=True,actual_resource_gate_and_owner_projection=True,
            overlay_series=reports[0]['overlay_series'],content_sha256=reports[0]['content_sha256'],
            protected_sources_unchanged=reports[0]['protected_sources_unchanged'],gpu_run=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--upstream-root',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    fresh_checks.contract_checks.PACKET=options.packet
    UPSTREAM=options.upstream_root.resolve(strict=True)
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(PatchSeriesPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            builder_sha256=builder.digest((HERE.parent/'build.py').read_bytes()),
            test_source_sha256=builder.digest(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(builder.canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
