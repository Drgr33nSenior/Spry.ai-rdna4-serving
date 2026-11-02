"""Verify the selected-phase closure through actual package producers/readers."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_shared_memory_packet as memory
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest
configured=memory.configured
PROPOSAL=HERE.parent/'proposals/phase-closure'
configured.PROPOSAL=PROPOSAL


class PhaseClosurePacket(memory.SharedMemoryPacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/shared-memory').resolve())
        parent=output/'package';(parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('phase-closure-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate);self.assertEqual(set(before),set(after))
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        code={'review/'+name for name in record['runtime_changes']};protected=set(record['changes'])
        derived={'MANIFEST.json','MANIFEST-PROPOSAL.json','PROTECTED-SHA256.json','review/RESOURCE-MANIFEST.json',
                 'review/BENCHMARK-SPEC2.json','review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'}
        changed={name for name in before if before[name]!=after[name]}
        self.assertEqual(changed,code|protected|derived);self.assertEqual(len(after)-1,79)
        self.evidence['phase_payload_delta']=dict(changed=sorted(changed),archive_members=79,
            kernel_head_projection_request_settings_and_pool_bytes_unchanged=True,native_execution=False)

    def test_packaged_selected_contract_and_consumers_have_no_legacy_phase_dependency(self):
        package,_,_=self.package('phase-selected-contract');review=package/'review'
        contract=load('phase_packet_contract',review/'mtp_contract.py')
        with patch.dict(sys.modules,mtp_contract=contract):
            serving=load('phase_packet_serving_contract',review/'serving_contract.py')
            for arm in ('control','mtp'):
                self.assertFalse(serving.hip_sampling_arm(contract.PAIRED_PHASE,arm))
                self.assertEqual(contract.contract_for_arm(arm)['speculative_dflash_block_size'],8)
            self.assertEqual(len(serving.requests()),176)
            for action in (lambda:serving.materialize_history(None,None,None),
                           lambda:serving.compare_closure(None,None),
                           lambda:serving.inspect_stream_response({},b'{}',0.0)):
                with self.assertRaises(ValueError):action()
            with patch.dict(sys.modules,serving_contract=serving):
                observer=load('phase_packet_observer',review/'serving_graph_observer.py')
            self.assertTrue(observer.closure_phase())
            self.assertEqual([observer.expected_alignment(arm) for arm in ('control','mtp')],['beta-bf16']*2)
        cases=load('phase_packet_cases',HERE/'test_phase_closure.py')
        self.assertTrue(cases.MARKERS.isdisjoint(vars(contract)))
        self.evidence['phase_contract']=dict(actual_packaged_contract_and_observer=True,request_rows=176,
            width=8,alignment='beta-bf16',unsupported_stream_history_closure_refused=True,
            inactive_marker_names_removed=len(cases.MARKERS),native_execution=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args();configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=PhaseClosurePacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
