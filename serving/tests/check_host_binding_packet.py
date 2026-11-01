"""Verify the observed-host startup increment on authenticated CPU copies."""
import argparse
import difflib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_stop_policy_packet as stop_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha

PROPOSAL=HERE.parent/'proposals/d2-host-binding'


class HostBindingPacket(stop_checks.StopPolicyPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(stop_checks,'PROPOSAL',PROPOSAL):
            super().setUpClass()
        cls.evidence['inherited_stop_checker_sha256']=sha((HERE/'check_stop_policy_packet.py').read_bytes())

    def test_exact_diff_and_unmodified_release_validators(self):
        name='review/benchmark_physical_controller.py'
        before=HERE.parent/'proposals/d2-stop-policy'/name
        expected=''.join(difflib.unified_diff(before.read_text().splitlines(True),
            (PROPOSAL/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        for name,pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.evidence.update(exact_patch_verified=True,release_validators_unchanged=True,
            owner_adapter_unchanged=True,transport_unchanged=True)

    def test_verified_host_is_the_controller_device_authority(self):
        with self.controller() as (controller,path,_,_,_):
            verified=controller.verify_configured_spec(self.pins,json.loads(path.read_bytes()),
                deadline=time.monotonic()+105)
            self.assertEqual(verified['host'],json.loads(self.members['config/host/r9700-pair.json']))
            self.assertEqual(verified['host']['devices'],verified['spec']['http_serving']['expected_devices'])
            self.assertEqual(verified['host']['rank_indices'],verified['spec']['gpu_ids'])
        self.evidence['verified_host_device_and_rank_projection']=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    stop_checks.artifact_checks.worker_checks.session_checks.inventory_checks.PACKET=options.packet
    with isolated_modules():
        result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(HostBindingPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
