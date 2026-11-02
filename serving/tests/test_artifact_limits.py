"""Exercise configured artifact bounds with actual pack/export/receive code."""
from dataclasses import FrozenInstanceError, asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load

PROPOSAL = HERE.parent / 'proposals/d2-artifact-limits'


class ArtifactLimits(unittest.TestCase):
    def setUp(self):
        self.module = load('cleanup_artifact_limits', PROPOSAL/'review/benchmark_artifacts.py')
        self.plan = self.module.HTTPArtifactLimits()
        temporary = tempfile.TemporaryDirectory(prefix='artifact-limits-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.binding = dict(session_sha256='a'*64, arm='bf16', spec_sha256='b'*64,
                            experiment_mode=self.module.HTTP_MODE)

    def test_baseline_matches_protected_literals_and_common_arithmetic(self):
        self.assertEqual(self.plan.members, self.module.HTTP_MEMBER_LIMITS)
        self.assertEqual(self.plan.http_session_max_bytes, self.module.HTTP_SESSION_MAX_BYTES)
        sys.path.insert(0, str(HERE.parent))
        self.addCleanup(sys.path.remove, str(HERE.parent))
        from spry_common.limits import derive
        primaries=json.loads((HERE.parent/'config/runs/w5-v2/run.json').read_bytes())['limits']
        self.assertEqual(self.module.HTTPArtifactLimits.from_limits(primaries), self.plan)
        self.assertEqual(derive(primaries)['http_session_max_bytes'],self.plan.http_session_max_bytes)

    def test_plan_is_frozen_and_does_not_expose_mutable_members(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan.client_tree_files = 1
        self.plan.members.clear()
        self.assertEqual(len(self.plan.members),3)

    def test_every_ceiling_and_strict_type_is_enforced(self):
        for name,ceiling in asdict(self.plan).items():
            for value in (True,0,-1,float(ceiling),ceiling+1):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    replace(self.plan,**{name:value})

    def test_primary_shape_and_derived_injection_refuse(self):
        values=asdict(self.plan)
        for name in values:
            malformed=dict(values);del malformed[name]
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.module.HTTPArtifactLimits.from_limits(malformed)
        for key in ('http_payload_bytes','http_session_max_bytes'):
            with self.assertRaises(ValueError):
                self.module.HTTPArtifactLimits.from_limits(dict(values,**{key:1}))

    def test_inconsistent_partitions_refuse(self):
        for changes in ({'session_bytes':self.plan.http_session_max_bytes},
                        {'http_stream_bytes':self.plan.encoded_bytes-1}):
            with self.assertRaises(ValueError):
                replace(self.plan,**changes)

    def test_corrupted_or_wrong_plan_refuses_before_staging(self):
        for policy in ({},self.module.HTTPArtifactLimits()):
            if type(policy) is self.module.HTTPArtifactLimits:
                object.__setattr__(policy,'client_tree_files',513)
            with self.assertRaises(ValueError):
                self.module.stage_artifacts(self.root,self.binding,limits=policy)
            self.assertFalse((self.root/'export').exists())

    def test_http_plan_cannot_change_other_transports(self):
        with self.assertRaises(ValueError):
            self.module.member_limits({'experiment_mode':self.module.INDEPENDENT_MODE},limits=self.plan)
        self.assertEqual(self.module.member_limits({'experiment_mode':self.module.INDEPENDENT_MODE}),
                         self.module.INDEPENDENT_MEMBER_LIMITS)

    def produce(self, plan=None):
        source=self.root/'source';source.mkdir()
        (source/'client').mkdir();(source/'client/a.json').write_bytes(b'{"a":1}\n')
        (source/'native').mkdir()
        (source/'result.json').write_bytes(b'{}\n')
        kwargs={} if plan is None else {'limits':plan}
        self.module.pack_http_artifacts(source,self.binding,**kwargs)
        staged=self.module.stage_artifacts(source,self.binding,**kwargs)
        sink=io.BytesIO()
        self.module.export_bundle(staged,self.binding,sink,**kwargs)
        return source,sink.getvalue()

    def test_explicit_baseline_and_legacy_calls_emit_identical_bytes(self):
        _,original=self.produce()
        self.root=self.root/'explicit';self.root.mkdir()
        _,configured=self.produce(self.plan)
        self.assertEqual(original,configured)
        receipt=self.module.receive_bundle(configured,b'',self.root/'received',self.binding,limits=self.plan)
        self.assertTrue(receipt['all_members_present'])

    def test_tightened_plan_roundtrips_the_actual_trees(self):
        plan=replace(self.plan,result_json_bytes=128,client_bundle_bytes=2048,native_bundle_bytes=1024,
            http_stream_bytes=16384,client_tree_bytes=1024,native_tree_bytes=1024,
            client_tree_files=2,native_tree_files=2,http_member_read_bytes=512)
        source,wire=self.produce(plan)
        receipt=self.module.receive_bundle(wire,b'',self.root/'received',self.binding,limits=plan)
        self.assertTrue(receipt['transport_valid'])
        for name in plan.members:
            self.assertEqual((source/name).read_bytes(),(self.root/'received'/name).read_bytes())

    def test_tree_file_and_member_byte_limits_are_consumed(self):
        client=self.root/'client';client.mkdir()
        (client/'one').write_bytes(b'1234');(client/'two').write_bytes(b'x')
        for changes in ({'client_tree_files':1},{'http_member_read_bytes':3},{'client_tree_bytes':4}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.module.pack_http_artifacts(self.root,self.binding,limits=replace(self.plan,**changes))
            self.assertFalse((self.root/'http-client-bundle.json').exists())

    def test_tightened_receiver_refuses_before_destination(self):
        _,wire=self.produce()
        plan=replace(self.plan,result_json_bytes=1)
        destination=self.root/'refused'
        with self.assertRaisesRegex(ValueError,'artifact shape/size/hash'):
            self.module.receive_bundle(wire,b'',destination,self.binding,limits=plan)
        self.assertFalse(destination.exists())

    def test_stream_ceiling_refuses_before_output(self):
        plan=replace(self.plan,result_json_bytes=1,client_bundle_bytes=1,native_bundle_bytes=1,
                     http_stream_bytes=16384)
        destination=self.root/'overflow'
        with self.assertRaisesRegex(ValueError,'stdio capture overflow'):
            self.module.receive_bundle(b'x'*16385,b'',destination,self.binding,limits=plan)
        self.assertFalse(destination.exists())

    def test_actual_pipe_capture_enforces_the_configured_stream_budget(self):
        plan=replace(self.plan,result_json_bytes=1,client_bundle_bytes=1,native_bundle_bytes=1,
                     http_stream_bytes=16384)
        cap=self.module.CapturePipes(self.root/'capture',mode=self.module.HTTP_MODE,limits=plan)
        try:
            cap.stdout.write(b'x'*16385)
        finally:
            result=cap.finish()
        self.assertFalse(result['capture_complete'])
        self.assertTrue(result['reader_thread_absent'])
        self.assertEqual(result['streams']['stdout']['bytes'],16384)
        self.assertLessEqual((self.root/'capture/stdout.raw').stat().st_size,16384)


if __name__=='__main__':
    unittest.main()
