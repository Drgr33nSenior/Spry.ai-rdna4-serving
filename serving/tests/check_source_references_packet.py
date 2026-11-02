"""Prove protected-source references produce the identical post-load package."""
import argparse
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_post_load_packet as post_load
from spry_common.canon import canonical_line as canonical,digest
configured=post_load.configured
PROPOSAL=HERE.parent/'proposals/source-references'
configured.PROPOSAL=PROPOSAL

class SourceReferencePacket(post_load.PostLoadPacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/post-load').resolve())
        parent=output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('source-reference-parity')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate)
        self.assertEqual(before,after)
        self.assertEqual(len(after)-1,78)
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        self.assertEqual(len(record['protected_source_references']),3)
        self.assertFalse((PROPOSAL/'review').exists())
        self.evidence['source_reference_payload_parity']=dict(all_package_members_byte_identical=True,
            actual_members=78,protected_references=3,protected_source_bytes_changed=0,
            runtime_source_bytes_changed=0,new_review_copies=0,native_execution=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=SourceReferencePacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
