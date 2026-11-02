"""Exercise the pre-replay adapter and strict reader through the actual package."""
import argparse
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_source_references_packet as references
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest
configured=references.configured
PROPOSAL=HERE.parent/'proposals/pre-replay'
configured.PROPOSAL=PROPOSAL

class PreReplayPacket(references.SourceReferencePacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/source-references').resolve())
        parent=output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('pre-replay-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate)
        self.assertEqual(set(before),set(after));self.assertEqual(len(after)-1,78)
        changed={name for name in before if before[name]!=after[name]}
        self.assertEqual(changed,{'review/verify_decode_adapter.py','review/worker_entry.py','review/serving_integration.py',
            'MANIFEST.json','MANIFEST-PROPOSAL.json','review/BENCHMARK-SPEC2.json','review/RESOURCE-MANIFEST.json',
            'review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'})
        self.evidence['pre_replay_payload_delta']=dict(changed=sorted(changed),archive_members=78,
            protected_sources_kernels_heads_active_overlays_requests_and_settings_byte_identical=True,
            native_execution=False)

    def test_packaged_adapter_prepares_before_capture_and_strict_witness_refuses_old_counts(self):
        package,_,_=self.package('pre-replay-source');review=package/'review'
        cases=load('packaged_pre_replay_cases',HERE/'test_pre_replay_metadata.py')
        adapter=load('packaged_pre_replay_adapter',review/'verify_decode_adapter.py')
        native=ModuleType('pre_replay_decode');native.decode_attention_fwd=cases.decode
        with patch.dict(sys.modules,pre_replay_decode=native):
            for bs in (1,2):
                fixture=cases.Fixture(adapter,bs);graph=fixture.capture()
                self.assertEqual(len(graph),48)
                self.assertEqual(fixture.adapter.metadata_capture_build_host_calls,0)
                fixture.state([127]*bs)
                fixture.backend.init_forward_metadata_out_graph(fixture.batch)
                self.assertEqual(fixture.adapter.buffers[id(fixture.backend),8*bs]['lengths'].values(),list(range(128,136))*bs)
                for operation in graph:operation()
        reader=load('packaged_pre_replay_reader',review/'serving_integration.py')
        witness=cases.PreReplayWitness();witness.setUp();self.addCleanup(witness.doCleanups)
        with patch.object(reader,'_mtp_contract',return_value=SimpleNamespace(contract_for_arm=lambda arm:dict(enabled=True))):
            record=witness.record
            self.assertEqual(record['metadata_preparation_policy'],adapter.METADATA_PREPARATION_POLICY)
            self.assertIs(reader.validate_full_enabled_record(record,witness.binding,witness.worker),record)
            record['metadata_capture_build_host_calls']=2
            with self.assertRaisesRegex(ValueError,'outside the forward graph'):
                reader.validate_full_enabled_record(record,witness.binding,witness.worker)
        self.evidence['pre_replay_source']=dict(batch_sizes=[1,2],actual_packaged_adapter_and_reader=True,
            preparation_recorded_in_forward_graph=False,old_witness_refused=True,
            source_only_tensor_standins=True,native_arithmetic=False,native_execution=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=PreReplayPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
