"""Check the closed HTTP package against the authenticated common-helper control."""
import argparse
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_common_helpers_packet as common
from spry_common.canon import canonical_line as canonical,digest
configured=common.configured
PROPOSAL=HERE.parent/'proposals/closed-http'
configured.PROPOSAL=PROPOSAL

class ClosedHttpPacket(common.CommonHelperPacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/common-helpers').resolve())
        parent=output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('closed-http-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate)
        removed={'review/graph_screen.py','review/context_tokenizer.py','review/validation_comparison.py'}
        self.assertEqual(set(before)-set(after),removed)
        self.assertEqual(set(after)-set(before),set())
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        code={'review/'+name for name,row in record['runtime_changes'].items() if row['baseline_sha256']!=row['proposed_sha256']}
        protected={name for name,row in record['changes'].items() if row['baseline_sha256']!=row['proposed_sha256']}
        derived={'MANIFEST.json','MANIFEST-PROPOSAL.json','PROTECTED-SHA256.json','review/RESOURCE-MANIFEST.json',
                 'review/BENCHMARK-SPEC2.json','review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'}
        changed={name for name in set(before)&set(after) if before[name]!=after[name]}
        self.assertEqual(changed,code|protected|derived)
        self.assertEqual(len(after)-1,77)  # Exclude the test-only activated manifest copy.
        self.evidence['closed_http_payload_delta']=dict(changed=sorted(changed),removed=sorted(removed),
            archive_members=77,owner_archive_ceiling=80,requests_settings_model_stack_kernels_and_active_overlays_byte_identical=True,
            native_execution=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=ClosedHttpPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
