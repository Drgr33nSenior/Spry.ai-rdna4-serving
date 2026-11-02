"""Verify lossless compact evidence and source-upload margins on the actual W5 path."""
import argparse
import difflib
import gzip
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_production_isolation_packet as isolation_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha

PROPOSAL=HERE.parent/'proposals/evidence-capacity'


class EvidenceCapacityPacket(isolation_checks.ProductionIsolationPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(isolation_checks,'PROPOSAL',PROPOSAL):super().setUpClass()

    def archive(self,name,members):
        path=super().archive(name,members)
        path.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
        return path

    def test_exact_diff_and_unmodified_release_validators(self):
        prior=HERE.parent/'proposals/production-isolation'
        before=json.loads((prior/'PROPOSAL.json').read_bytes())
        name='review/benchmark_host_transport.py';old=(prior/name).read_bytes();new=(PROPOSAL/name).read_bytes()
        expected=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        self.assertEqual(sha(old),self.record['changes'][name]['baseline_sha256'])
        self.assertEqual(sha(new),self.record['changes'][name]['proposed_sha256'])
        self.assertEqual(sha((PROPOSAL/'D2.patch').read_bytes()),self.record['patch_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),self.record['source_patch_sha256'])
        for name,pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/'+name]),pin)
            if name!='benchmark_host_transport.py':self.assertEqual(pin,before['protected_files'][name])
        for name,row in self.record['source_changes'].items():self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.assertEqual(sha(canonical({'review/'+name:pin for name,pin in self.record['protected_files'].items()})),self.record['protected_set_sha256'])
        self.evidence.update(exact_patch_verified=True,owner_ceilings_changed=False,release_validators_unchanged=True,
            controller_unchanged=True,source_upload_codec='gzip; independently bounded 16MiB expanded tar')

    def test_workload_and_root_seals_charge_the_actual_archive_sources_and_both_closeouts(self):
        super().test_workload_and_root_seals_charge_the_actual_archive_sources_and_both_closeouts()
        size=self.evidence['composed_durable_seals']['source_copies_bytes']
        self.assertLessEqual(2*size,32<<20,'twofold configured source-copy margin required')
        self.evidence['composed_durable_seals']['source_copy_margin']=(32<<20)/size

    def check_postrun(self,shape):
        super().check_postrun(shape)
        rows=self.evidence['postrun_'+shape]['margins']
        for row in rows:self.assertGreaterEqual(row['margin'],2.0,row['row'])
        self.evidence['postrun_'+shape]['twofold_capacity_margin_verified']=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args();inventory_checks.PACKET=options.packet
    with isolated_modules():result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(EvidenceCapacityPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
