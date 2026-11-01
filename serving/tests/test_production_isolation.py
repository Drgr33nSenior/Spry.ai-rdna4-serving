"""Actual production file hashes, import refusal, caches and exclusive mask creation."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_source_builder import builder
from test_import_cleanup import stale_bytecode
import test_source_cleanup as runtime_tests

PROPOSAL=HERE.parent/'proposals/production-isolation'
SOURCE=HERE.parent/'legacy/review'


class ProductionIsolation(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-production-view-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()
        self.files={'FINAL-SETTINGS.json':builder.canonical(dict(instrumentation='production')),
                    'mtp_contract.py':(SOURCE/'mtp_contract.py').read_bytes()}
        self.policy=builder.production_evidence_policy(self.files)

    def test_policy_has_six_explicit_files_and_instrumented_mode_has_no_masks(self):
        self.assertEqual(len(self.policy['files']),6)
        self.assertEqual(self.policy['sha256'],builder.digest(self.policy['raw']))
        files=dict(self.files,**{'FINAL-SETTINGS.json':builder.canonical(dict(instrumentation='on'))})
        self.assertIsNone(builder.production_evidence_policy(files))

    def test_policy_is_literal_and_cannot_execute_source_during_authoring(self):
        marker=self.root/'executed'
        replacement="PRODUCTION_EVIDENCE_FILES = __import__('pathlib').Path("+repr(str(marker))+").write_text('bad')"
        tree=ast.parse(self.files['mtp_contract.py'])
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='PRODUCTION_EVIDENCE_FILES' for t in n.targets))
        lines=self.files['mtp_contract.py'].decode().splitlines(True)
        lines[node.lineno-1:node.end_lineno]=[replacement+'\n']
        with self.assertRaises(ValueError):
            builder.production_evidence_policy(dict(self.files,**{'mtp_contract.py':''.join(lines).encode()}))
        self.assertFalse(marker.exists())

    def test_all_six_effective_files_refuse_import_even_with_stale_original_bytecode(self):
        prefix=self.root/'fresh-bytecode';prefix.mkdir()
        for index,name in enumerate(self.policy['files']):
            path=self.root/name
            path.write_bytes(b'raise AssertionError("original evidence code executed")\n')
            stale_bytecode(path)
            path.write_bytes(self.policy['raw'])
            with self.subTest(name=name),patch.object(sys,'pycache_prefix',str(prefix)), \
                    patch.object(sys,'dont_write_bytecode',True):
                spec=importlib.util.spec_from_file_location('blocked_evidence_'+str(index),path)
                module=importlib.util.module_from_spec(spec)
                with self.assertRaisesRegex(ModuleNotFoundError,'unavailable in production'):
                    spec.loader.exec_module(module)
        self.assertFalse(list(prefix.rglob('*.pyc')))

    def controller(self,instrumented=False):
        tree=ast.parse((PROPOSAL/'review/benchmark_physical_controller.py').read_bytes())
        wanted={'need','_canonical','prepare_production_evidence_mask'}
        body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
        module=ModuleType('production_mask_cpu_fixture')
        module.__dict__.update(Path=Path,os=os,mtp_contract=SimpleNamespace(
            instrumentation_enabled=lambda:instrumented,PRODUCTION_EVIDENCE_MASK_NAME=self.policy['name'],
            PRODUCTION_EVIDENCE_MASK_BYTES=self.policy['raw']))
        exec(compile(ast.Module(body=body,type_ignores=[]),'production_mask_cpu_fixture','exec'),module.__dict__)
        return module

    def test_controller_creates_one_exclusive_regular_mask_and_never_overwrites(self):
        controller=self.controller()
        path=controller.prepare_production_evidence_mask(self.root)
        self.assertEqual(path.read_bytes(),self.policy['raw']);self.assertEqual(path.stat().st_nlink,1)
        self.assertFalse(path.stat().st_mode&0o222)
        with self.assertRaises(FileExistsError):controller.prepare_production_evidence_mask(self.root)

    def test_instrumented_controller_creates_no_mask(self):
        self.assertIsNone(self.controller(True).prepare_production_evidence_mask(self.root))
        self.assertFalse((self.root/self.policy['name']).exists())

    def test_controller_refuses_a_preexisting_link_without_touching_its_target(self):
        target=self.root/'target';target.write_bytes(b'keep')
        (self.root/self.policy['name']).symlink_to(target)
        with self.assertRaises(FileExistsError):self.controller().prepare_production_evidence_mask(self.root)
        self.assertEqual(target.read_bytes(),b'keep')

    def test_unchanged_runtime_source_validator_checks_the_actual_masked_bytes(self):
        fixture=runtime_tests.OrdinaryRuntime();fixture.setUp();self.addCleanup(fixture.doCleanups)
        sources={}
        for name in self.policy['files']:
            path=self.root/name;path.write_bytes(self.policy['raw'])
            sources[name[:-3]]=dict(path=str(path),sha256=self.policy['sha256'])
        template=dict(binding_template=dict(sources=sources),worker_template={},
                      source_manifest_sha256=fixture.runtime._digest(sources))
        output=self.root/'output';output.mkdir()
        # All six actual file hashes pass. Stop at the following device-shape
        # check; no native model/config or validator substitute is used here.
        with self.assertRaisesRegex(ValueError,'expected device binding differs'):
            fixture.runtime._materialize(template,template_sha256='a'*64,session_sha256='b'*64,
                spec_sha256='c'*64,output=output,deadline=100,expected_devices={})
        template=copy.deepcopy(template)
        next(iter(template['binding_template']['sources'].values()))['sha256']='f'*64
        template['source_manifest_sha256']=fixture.runtime._digest(template['binding_template']['sources'])
        output=self.root/'wrong-hash';output.mkdir()
        with self.assertRaisesRegex(ValueError,'template source changed'):
            fixture.runtime._materialize(template,template_sha256='a'*64,session_sha256='b'*64,
                spec_sha256='c'*64,output=output,deadline=100,expected_devices={})


if __name__=='__main__':unittest.main()
