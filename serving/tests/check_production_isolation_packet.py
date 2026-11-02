"""Authenticate production masks, owner projection and the complete W5 composition."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_source_fingerprints_packet as fingerprint_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha

PROPOSAL=HERE.parent/'proposals/production-isolation'


class ProductionIsolationPacket(fingerprint_checks.SourceFingerprintsPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(fingerprint_checks,'PROPOSAL',PROPOSAL):super().setUpClass()
        cls.projection_proposal=PROPOSAL

    def test_exact_diff_and_unmodified_release_validators(self):
        before=json.loads((HERE.parent/'proposals/source-fingerprints/PROPOSAL.json').read_bytes())
        baselines={'review/benchmark_owner_adapter.py':HERE.parent/'proposals/d2-owner-timing/review/benchmark_owner_adapter.py',
            'review/benchmark_physical_controller.py':HERE.parent/'proposals/source-fingerprints/review/benchmark_physical_controller.py'}
        expected=''
        for name,path in baselines.items():
            self.assertEqual(sha(path.read_bytes()),self.record['changes'][name]['baseline_sha256'])
            expected+=''.join(difflib.unified_diff(path.read_text().splitlines(True),(PROPOSAL/name).read_text().splitlines(True),
                fromfile='a/'+name,tofile='b/'+name))
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        self.assertEqual(sha((PROPOSAL/'D2.patch').read_bytes()),self.record['patch_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),self.record['source_patch_sha256'])
        self.assertEqual(sha(canonical({'review/'+name:pin for name,pin in self.record['protected_files'].items()})),
            self.record['protected_set_sha256'])
        for name,pin in self.record['protected_files'].items():self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        for name in set(self.record['protected_files'])-{'benchmark_owner_adapter.py','benchmark_physical_controller.py'}:
            self.assertEqual(self.record['protected_files'][name],before['protected_files'][name])
        self.evidence.update(exact_patch_verified=True,owner_adapter_changed_only_for_masks=True,
            protected_controller_changed_for_mask_creation_and_stamp=True,release_validators_unchanged=True,
            source_validators_unchanged=True,owner_ceilings_changed=False)

    def test_actual_owner_mount_projection_matches_both_frozen_production_arms(self):
        cases=[]
        with self.controller() as (controller,path,_,_,root):
            manifest=json.loads(path.read_bytes())
            configured=controller.verify_configured_spec(self.pins,manifest,deadline=time.monotonic()+105)
            spec=configured['spec'];frozen=json.loads((root/'review/RESOURCE-MANIFEST.json').read_bytes())
            contract=controller.mtp_contract
            for arm in ('bf16','kv4'):
                target=Path(manifest['target_root'])
                argv=controller.adapter.bwrap_argv(arm,retained_root=Path(manifest['retained_root']),
                    composition=Path(manifest['runtime_root']),review=target/'review',model=Path(manifest['model_root']),
                    output=target/'arms'/arm,entry=target/'review/model_benchmark.py',
                    spec_sha256=configured['spec_sha256'],bounded_scratch=True,physical_artifacts=True,
                    session_sha256=manifest['session_sha256'],artifacts_sha256=manifest['package_files']['review/benchmark_artifacts.py'],
                    graph_contract=spec['graph_projection'],http_contract=controller.http_projection_contract(spec))
                actual=contract.replay_serving_sandbox_argv(argv,manifest['package_files'])
                self.assertEqual(list(actual),frozen['workload_argv'][arm]);self.assertLessEqual(len(actual),256)
                for name in contract.PRODUCTION_EVIDENCE_FILES:
                    group=['--ro-bind',str(target/contract.PRODUCTION_EVIDENCE_MASK_NAME),'/mnt/review/'+name]
                    self.assertEqual(sum(list(actual[i:i+3])==group for i in range(len(actual)-2)),1)
                self.assertIn('pycache_prefix=/mnt/capture-bytecode',actual)
                cases.append(dict(arm=arm,argv_items=len(actual),evidence_masks=6))
        self.evidence['actual_owner_production_projection']=dict(cases=cases,existing_argv_ceiling=256,
            frozen_projection_byte_identical=True,native_bwrap_executed=False)

    def test_actual_controller_creates_a_bound_mask_and_its_seal_accounts_for_it(self):
        with self.controller() as (controller,_,_,_,root):
            path=controller.prepare_production_evidence_mask(root)
            raw=controller.mtp_contract.PRODUCTION_EVIDENCE_MASK_BYTES
            self.assertEqual(path.read_bytes(),raw)
            stamps={}
            controller._configuration_file(path,hashlib.sha256(raw).hexdigest(),len(raw),
                deadline=200,clock=lambda:100,stamps=stamps,json_object=False)
            controller.verify_unchanged(dict(stamps=stamps))
            with self.assertRaises(FileExistsError):controller.prepare_production_evidence_mask(root)
            artifacts=controller._adjacent('production_mask_accounting','benchmark_artifacts.py')
            budget=artifacts.audit_durable_trees(root,limits=artifacts.HTTPArtifactLimits())
            self.assertGreaterEqual(budget['partition_bytes']['small_metadata_bytes'],len(raw))
            self.evidence['actual_production_mask_file']=dict(bytes=len(raw),sha256=sha(raw),exclusive=True,
                source_stamp_verified=True,durable_partition_accounted=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args();inventory_checks.PACKET=options.packet
    with isolated_modules():result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(ProductionIsolationPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
