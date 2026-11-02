"""Verify deterministic session rendering and unchanged sandbox options."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_source_builder as source_tests


class FreshIdentity(unittest.TestCase):
    def setUp(self):
        self.fixture=source_tests.SourceBuild()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.build()
        self.files={'RESOURCE-MANIFEST.json':(self.fixture.root/'output/review/RESOURCE-MANIFEST.json').read_bytes()}
        self.original=json.loads(self.files['RESOURCE-MANIFEST.json'])
        self.configuration={'pins':{'run':'a'*64}}

    def render(self,files=None,packet='fedcba0987654321',configuration=None):
        return source_tests.builder.render_session_identity(self.files if files is None else files,
            self.configuration if configuration is None else configuration,packet)

    def test_same_inputs_are_deterministic_and_distinct_ids_change_the_session(self):
        left,right=copy.deepcopy(self.files),copy.deepcopy(self.files)
        one=self.render(left);two=self.render(right)
        self.assertEqual(left,right);self.assertEqual(one,two)
        other=copy.deepcopy(right)
        three=self.render(other,packet='0000000000000001')
        self.assertNotEqual(three['session_sha256'],one['session_sha256'])
        self.assertFalse(one['launch_authorized'])

    def test_all_root_paths_session_arguments_and_unit_hashes_change_together(self):
        identity=self.render();value=json.loads(self.files['RESOURCE-MANIFEST.json'])
        self.assertEqual(value['root'],identity['target_root'])
        self.assertEqual(value['session_sha256'],identity['session_sha256'])
        self.assertNotIn(self.original['root'],self.files['RESOURCE-MANIFEST.json'].decode())
        self.assertNotIn(self.original['session_sha256'],self.files['RESOURCE-MANIFEST.json'].decode())
        for arm,argv in value['workload_argv'].items():
            self.assertEqual(argv[argv.index('--session-sha256')+1],identity['session_sha256'])
            self.assertEqual(value['units'][arm],'spry-fp8-benchmark-'+identity['session_sha256'][:16]+'-'+arm+'.scope')
            self.assertEqual(value['workload_argv_sha256'][arm],source_tests.builder.digest(source_tests.builder.canonical_bare(argv)))
            for before,after in zip(self.original['workload_argv'][arm],argv,strict=True):
                if before.startswith(self.original['root']+'/'):
                    self.assertEqual(after,identity['target_root']+before[len(self.original['root']):])
                elif before==self.original['session_sha256']:
                    self.assertEqual(after,identity['session_sha256'])
                else:self.assertEqual(after,before)

    def test_run_config_identity_contributes_to_the_session_hash(self):
        left,right=copy.deepcopy(self.files),copy.deepcopy(self.files)
        one=self.render(left)
        two=self.render(right,configuration={'pins':{'run':'b'*64}})
        self.assertNotEqual(one['session_sha256'],two['session_sha256'])

    def test_invalid_or_historical_ids_refuse_without_changing_input_bytes(self):
        for packet in (None,True,'','../root','ABCDEF0123456789','f'*15,'f'*17,'1234567890abcdef'):
            files=copy.deepcopy(self.files)
            with self.subTest(packet=packet),self.assertRaises(ValueError):self.render(files,packet=packet)
            self.assertEqual(files,self.files)

    def test_missing_authentication_and_embedded_root_paths_refuse(self):
        with self.assertRaisesRegex(ValueError,'authenticated run configuration'):
            self.render(configuration={})
        value=copy.deepcopy(self.original)
        value['workload_argv']['bf16'].insert(1,'unbound='+value['root'])
        files={'RESOURCE-MANIFEST.json':source_tests.builder.canonical(value)}
        with self.assertRaisesRegex(ValueError,'complete root boundary'):self.render(files)

    def test_builder_requires_config_before_creating_fresh_output(self):
        destination=self.fixture.root/'refused'
        with self.assertRaisesRegex(ValueError,'authenticated run configuration'):
            source_tests.builder.build(self.fixture.packet,destination,source_root=self.fixture.source,
                                       packet_id='fedcba0987654321')
        self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
