"""Replay natural and declared-cap rows through the actual CPU serving chain."""
import argparse
from contextlib import contextmanager
import difflib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_artifact_limits_packet as artifact_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha

PROPOSAL=HERE.parent/'proposals/d2-stop-policy'


class StopPolicyPacket(artifact_checks.ArtifactLimitsPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(artifact_checks,'PROPOSAL',PROPOSAL):
            super().setUpClass()
        cls.evidence['inherited_artifact_checker_sha256']=sha((HERE/'check_artifact_limits_packet.py').read_bytes())

    def test_exact_diff_and_unmodified_release_validators(self):
        name='review/benchmark_physical_controller.py'
        before=HERE.parent/'proposals/d2-artifact-limits'/name
        expected=''.join(difflib.unified_diff(before.read_text().splitlines(True),
            (PROPOSAL/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        for name,pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.evidence.update(exact_patch_verified=True,release_validators_unchanged=True,
            owner_adapter_unchanged=True,transport_unchanged=True)

    def test_declared_cap_row_completes_actual_client_reader_summary_and_protected_report(self):
        original_mirror=self.postrun.mirror
        original_responder=self.postrun.responder
        original_quality=self.postrun.quality_input_bytes
        @contextmanager
        def declared_cap(package):
            with original_mirror(package) as value:
                controller,manifest,root=value
                settings_path=root/'review/FINAL-SETTINGS.json'
                settings=json.loads(settings_path.read_bytes())
                settings['requests'][0]['stop_policy']='cap'
                settings_path.write_bytes(canonical(settings))
                # This is a consumer-boundary fixture, with a synthetic declared
                # row. The complete pinned run verifier must still reject it.
                yield value
        def capped_responder(contract,shape,arm):
            ordinary,sizes=original_responder(contract,shape,arm)
            at_cap,_=original_responder(contract,'worst',arm)
            case=contract.requests()[0]
            def respond(body):
                if body['rid']!=case['body']['rid']:
                    return ordinary(body)
                value=json.loads(at_cap(body))
                value['choices'][0]['finish_reason']='length'
                value['choices'][0]['matched_stop']=None
                value['choices'][0]['meta_info']['finish_reason']={'type':'length','length':body['max_completion_tokens']}
                raw=json.dumps(value,separators=(',',':'),ensure_ascii=False).encode()
                sizes[case['case']]=len(raw)
                return raw
            return respond,sizes
        # Reuse the synchronized stand-in for real client concurrency, while
        # omitting only the whole-run baseline verifier for this new row policy.
        measured=[]
        def ungraded_cap(wires):
            for wire in wires.values():
                row=wire['rows'][0]
                self.assertEqual((row['status'],row['finish_reason'],row['metric_eligible']),
                                 ('OUTPUT_CAP_REACHED','length',False))
            # The separate retained quality packet declares only natural
            # oracles. It must refuse this new cap work, not grade it as an
            # answer. Its config/menu migration remains a separate gate.
            with self.assertRaisesRegex(ValueError,'terminal and independent canonical-response join'):
                original_quality(wires)
            return 0
        def consumer_only(_self,shape):
            sizes,references,refusals=self.postrun.run(self.package,shape)
            self.assertEqual(set(references),{'bf16','kv4'})
            self.assertTrue(all(row['request_count']==row['response_count']==176 for row in references.values()))
            self.assertTrue(all('natural-stop response menu' in message for message in refusals.values()))
            measured.append(dict(shape=shape,arms=sorted(references),requests_per_arm=176,
                                 declared_cap_ordinal=0,whole_run_mutation_supported=False))
            self.evidence.setdefault('postrun_'+shape,{})
        with patch.object(self.postrun,'mirror',declared_cap), \
                patch.object(self.postrun,'responder',capped_responder), \
                patch.object(self.postrun,'quality_input_bytes',ungraded_cap), \
                patch.object(artifact_checks.worker_checks.session_checks.owner_checks.spec_checks.SpecBindingPacket,
                             'check_postrun',consumer_only):
            artifact_checks.ArtifactLimitsPacket.check_postrun(self,'expected')
        self.evidence['declared_cap_consumer_replay']=measured
        self.evidence['cap_work_retained_quality_packet_refused']=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    artifact_checks.worker_checks.session_checks.inventory_checks.PACKET=options.packet
    with isolated_modules():
        result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(StopPolicyPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
