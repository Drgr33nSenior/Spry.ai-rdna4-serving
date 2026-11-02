"""Final campaign membership and protected bootstrap joins; no external effects."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import prepare_gpu_run as prepare
import test_configured_caller as caller_tests
import test_measurement_preparation as measurement_tests
caller=caller_tests.caller
from spry_common.canon import canonical_line as canonical,digest


class FinalArchive(unittest.TestCase):
    def test_verified_package_counts_keep_the_existing_ceiling(self):
        for count in (78,79,80):
            files={'review/f'+str(index)+'.py':b'# inert\n' for index in range(count)}
            self.assertTrue(prepare.archive_bytes(files,compressed=True,expected_members=count))
        for count in (0,77,81,True):
            with self.assertRaises(ValueError):prepare.archive_bytes({},compressed=True,expected_members=count)


class InstrumentedQualification(unittest.TestCase):
    def test_explicit_qualification_preserves_work_checks_and_cannot_be_a_production_comparison(self):
        fixture=measurement_tests.MeasurementPreparation();fixture.setUp()
        fixture.run['instrumentation']='on'
        for files in (fixture.before,fixture.after):
            files['FINAL-SETTINGS.json']=canonical(dict(instrumentation='on'))
        compare=lambda **kwargs: prepare.prepare_measurement.compare_inputs(fixture.before,fixture.after,
            run=fixture.run,canonical_sources={'worker_entry.py'},protected_sources={},**kwargs)
        with self.assertRaisesRegex(ValueError,'production mode'):compare()
        result=compare(qualification=True)
        self.assertTrue(result['qualification_only']);self.assertEqual(result['instrumentation'],'on')
        fixture.after['FINAL-REQUESTS.zlib']+=b'changed'
        with self.assertRaisesRegex(ValueError,'measurement input differs'):compare(qualification=True)
        for value in (1,None,'on'):
            with self.assertRaisesRegex(ValueError,'boolean qualification'):compare(qualification=value)


class ProtectedBootstrapCaller(unittest.TestCase):
    def setUp(self):
        self.fixture=caller_tests.ConfiguredCaller();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        fixture=self.fixture;name='review/spry_common_canon.py';raw=b'# exact CPU fixture common\n';pin=digest(raw)
        (fixture.source/'owner-upload'/name).write_bytes(raw)
        manifest=dict(schema=3,package_files={name:pin});protected={name:pin}
        (fixture.source/'package-v2/MANIFEST.json').write_bytes(canonical(manifest))
        (fixture.source/'package-v2/PROTECTED-SHA256.json').write_bytes(canonical(protected))
        fixture.template.update(bootstrap_common_sha256=pin,manifest_sha256=digest(canonical(manifest)),
            protected_set_sha256=digest(canonical(protected)))
        fixture.template['upload_inventory'][name]=pin;self.write_templates()
    def write_templates(self):
        f=self.fixture
        (f.source/'LAUNCHER-BINDINGS-TEMPLATE.json').write_bytes(canonical(f.template))
        (f.work/'LAUNCHER-BINDINGS.json').write_bytes(canonical(dict(f.template,boot_id=f.auth['boot_id'])))
        f.auth.update(binding_template_sha256=digest((f.source/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes()),
            binding_sha256=digest((f.work/'LAUNCHER-BINDINGS.json').read_bytes()),
            manifest_sha256=f.template['manifest_sha256'],protected_set_sha256=f.template['protected_set_sha256'])
        f.write_auth()
    def test_exact_common_upload_joins_template_manifest_and_protected_inventory(self):
        _,binding,_=self.fixture.validate()
        self.assertEqual(binding['bootstrap_common_sha256'],binding['upload_inventory']['review/spry_common_canon.py'])
    def test_missing_changed_or_unbound_common_refuses(self):
        f=self.fixture;original=copy.deepcopy(f.template)
        f.template['upload_inventory'].pop('review/spry_common_canon.py');self.write_templates()
        with self.assertRaisesRegex(ValueError,'bootstrap common'):f.validate()
        f.template=copy.deepcopy(original);f.template['bootstrap_common_sha256']='f'*64;self.write_templates()
        with self.assertRaisesRegex(ValueError,'bootstrap common'):f.validate()
        f.template=copy.deepcopy(original);self.write_templates()
        (f.source/'owner-upload/review/spry_common_canon.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):f.validate()
    def test_additional_upload_cannot_hide_behind_common_support(self):
        f=self.fixture;(f.source/'owner-upload/review/extra.py').write_bytes(b'# extra')
        with self.assertRaisesRegex(ValueError,'membership'):f.validate()


if __name__=='__main__':unittest.main()
