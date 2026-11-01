"""Check source-only fingerprint changes and the complete proposed Python AST gate."""
import argparse
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_production_imports_packet as imports_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha

PROPOSAL=HERE.parent/'proposals/source-fingerprints'


class SourceFingerprintsPacket(imports_checks.ProductionImportsPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(imports_checks,'PROPOSAL',PROPOSAL):super().setUpClass()

    def test_exact_diff_and_unmodified_release_validators(self):
        before=json.loads((HERE.parent/'proposals/production-imports/PROPOSAL.json').read_bytes())
        self.assertEqual(self.record['protected_files'],before['protected_files'])
        self.assertEqual(self.record['protected_set_sha256'],before['protected_set_sha256'])
        for name,pin in self.record['protected_files'].items():self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),self.record['source_patch_sha256'])
        worker='legacy/review/worker_entry.py'
        self.assertEqual(self.record['direct_source_changes'][worker]['baseline_sha256'],before['source_changes'][worker]['proposed_sha256'])
        self.assertEqual(self.record['direct_source_changes']['legacy/review/verify_decode_adapter.py']['baseline_sha256'],
            sha(self.files['author/package-v2/review/verify_decode_adapter.py']))
        self.assertEqual(len(self.report['canonical_sources']),21)
        self.evidence.update(protected_helpers_unchanged_from_import_stage=True,
            exact_source_pins_verified=True,canonical_sources=21,release_validators_unchanged=True)

    def test_complete_proposed_python_source_set_has_no_direct_exec_or_compile(self):
        paths=[name for name in self.members if name.endswith('.py')]
        for name in paths:
            for node in ast.walk(ast.parse(self.members[name],filename=name)):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,('exec','compile'),name)
        self.evidence['proposed_python_ast_gate']=dict(python_files=len(paths),
            direct_exec_or_compile_calls=0,includes_proposed_protected_sources=True,
            canonical_builder_protected_sources_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args();inventory_checks.PACKET=options.packet
    with isolated_modules():result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(SourceFingerprintsPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
