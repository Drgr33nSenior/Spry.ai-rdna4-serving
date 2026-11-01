"""Prove configured worker export and bounded readout with CPU substitutes."""
from dataclasses import asdict,replace
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_entrypoint_cleanup as entry_tests
from test_contract_cleanup import load


class ArtifactBinding(unittest.TestCase):
    def setUp(self):
        self.entry=entry_tests.HttpEntry();self.entry.setUp();self.addCleanup(self.entry.doCleanups)
        self.module=load('cleanup_binding_artifacts',HERE.parent/'proposals/d2-artifact-binding/review/benchmark_artifacts.py')
        self.plan=replace(self.module.HTTPArtifactLimits(),result_json_bytes=4096)
        self.spec=entry_tests.http_spec()
        self.spec['http_serving']['artifact_limits']=asdict(self.plan)
        self.reader=load('cleanup_binding_integration',HERE.parent/'legacy/review/serving_integration.py')

    def test_worker_passes_one_frozen_plan_to_all_three_export_operations(self):
        for arm in ('bf16','kv4'):
            result=self.entry.run_entry(spec=copy.deepcopy(self.spec),arm=arm,
                                       artifact_policy_class=self.module.HTTPArtifactLimits)
            self.assertEqual(result['termination'],('SystemExit',0))
            policies=result['artifact_policies']
            self.assertEqual([name for name,_ in policies],['pack','stage','export'])
            self.assertTrue(all(policy is policies[0][1] for _,policy in policies))
            self.assertEqual(policies[0][1],self.plan)

    def test_runtime_preparation_cannot_extend_the_snapshotted_export_plan(self):
        def changed(spec):spec['http_serving']['artifact_limits']['result_json_bytes']=8<<20
        result=self.entry.run_entry(spec=self.spec,artifact_policy_class=self.module.HTTPArtifactLimits,
                                    mutate_runtime_spec=changed)
        self.assertEqual(result['termination'],('SystemExit',0))
        self.assertEqual(result['artifact_policies'][0][1].result_json_bytes,4096)

    def test_invalid_spec_or_owner_ceiling_refuses_before_scratch(self):
        for value in (True,0,1.0,9<<20):
            spec=copy.deepcopy(self.spec);spec['http_serving']['artifact_limits']['result_json_bytes']=value
            result=self.entry.run_entry(spec=spec,artifact_policy_class=self.module.HTTPArtifactLimits)
            self.assertEqual(result['termination'][0],'ValueError')
            self.assertFalse(any(row[0] in ('scratch','runtime') for row in result['initialized']))
            self.assertIsNone(result['files'])
        result=self.entry.run_entry(spec=self.spec)
        self.assertIn('configured artifact helper required',result['termination'][1])
        self.assertIsNone(result['files'])

    def test_client_tree_limits_refuse_before_extraction(self):
        def member(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                                   base64=base64.b64encode(raw).decode())
        raw=(json.dumps(dict(schema=1,mode=self.reader.MODE,files={'a':member(b'12'),'b':member(b'34')}),
                        sort_keys=True,separators=(',',':'))+'\n').encode()
        for changes in ({'client_tree_files':1},{'client_tree_bytes':3},{'http_member_read_bytes':1}):
            destination=self.entry.root/next(iter(changes))
            limits=dict(asdict(self.plan),**changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.reader._unpack_client(raw,destination,limits=limits)
            self.assertFalse(destination.exists())
        destination=self.entry.root/'valid'
        self.reader._unpack_client(raw,destination,limits=asdict(self.plan))
        self.assertEqual((destination/'a').read_bytes(),b'12')

    def test_readout_rejects_invalid_limits_before_private_imports(self):
        for limits in ({},dict(asdict(self.plan),result_json_bytes=True)):
            with patch.object(self.reader,'_mtp_contract') as contract,self.assertRaises(ValueError):
                self.reader.read_http_artifacts(self.entry.root,extraction_root=self.entry.root/'extract',
                    expected={},owner_arm='bf16',joins=[],supervision={},limits=limits)
            contract.assert_not_called()

    def test_readout_enforces_the_configured_outer_member_before_json_or_extraction(self):
        root=self.entry.root
        (root/'http-client-bundle.json').write_bytes(b'x'*1025)
        limits=dict(asdict(self.plan),client_bundle_bytes=1024)
        expected=dict(arm='control',source_manifest_sha256='a'*64,session_sha256='b'*64,spec_sha256='c'*64)
        contract=SimpleNamespace(instrumentation_enabled=lambda:False,owner_arm_map=lambda:{'bf16':'control'})
        with patch.object(self.reader,'_mtp_contract',return_value=contract), \
                patch.object(self.reader,'_private_reader') as private,self.assertRaises(ValueError):
            self.reader.read_http_artifacts(root,extraction_root=root/'extract',expected=expected,
                owner_arm='bf16',joins=[],supervision={},limits=limits)
        private.assert_not_called()
        self.assertFalse((root/'extract').exists())


if __name__=='__main__':unittest.main()
