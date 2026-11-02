"""Exercise the real root entry's exact source-set gate before workload creation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import build as builder


def digest(raw):return hashlib.sha256(raw).hexdigest()


class WorkloadBoundaryReached(RuntimeError):pass


class BootstrapCommonStage(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='bootstrap-common-stage-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()
        self.stage=self.root/'stage';self.review=self.stage/'review';self.review.mkdir(parents=True)
        parent=HERE/'proposals/common-closure/review'
        for name in ('host_process_owner.py','benchmark_release_observer.py','collect_benchmark_platform.py','benchmark_artifacts.py'):
            (self.review/name).write_bytes((parent/name).read_bytes())
        (self.review/'spry_common_canon.py').write_bytes((HERE/'spry_common/selected.py').read_bytes())
        self.path=self.review/'benchmark_host_transport.py'
        self.path.write_bytes((HERE/'proposals/bootstrap-common-stage/review/benchmark_host_transport.py').read_bytes())
        self.module=self.load('bootstrap_common_stage_actual',self.path)
        self.manifest=dict(schema=3,run_config='config/runs/w5-v2/run.json',
            package_files={'review/'+p.name:digest(p.read_bytes()) for p in self.review.iterdir()},
            target_root=str(self.root/'workload'))
        self.members={self.manifest['run_config']:(HERE/'config/runs/w5-v2/run.json').read_bytes()}
        self.archive=self.stage/'payload.tar';self.archive.write_bytes(b'inert archive identity fixture')
        self.args=SimpleNamespace(original_receipt=(100,'2026-11-02T00:00:00Z'),
            archive=str(self.archive),archive_sha256=digest(self.archive.read_bytes()),
            manifest_sha256='b'*64,expected_self_sha256=digest(self.path.read_bytes()),
            run_config_sha256='c'*64,host_sha256='d'*64,protected_set_sha256='e'*64,approved_boot_id='f'*36)
    def load(self,name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{name:module}):spec.loader.exec_module(module)
        return module
    def entry(self,module=None,*,inspector=None):
        module=self.module if module is None else module
        self.checked_sources=[]
        original_read=Path.read_text
        def read(path,*args,**kwargs):
            return self.args.approved_boot_id if str(path)=='/proc/sys/kernel/random/boot_id' else original_read(path,*args,**kwargs)
        def checked(path,pin):
            raw=builder.read_regular(path,1<<20)
            if digest(raw)!=pin:raise ValueError('fixture root source hash differs')
            self.checked_sources.append(path.name)
            return raw
        # Named substitutes: root credential/parent-directory admission, archive
        # authentication, boot binding and monotonic clock. Real source hashing,
        # artifact module import, limit derivation and exact stage inventory run.
        with patch.multiple(module.os,getuid=lambda:0,geteuid=lambda:0,getgid=lambda:0,getegid=lambda:0), \
             patch.object(module,'protected_stage',return_value=self.stage), \
             patch.object(module,'root_source',side_effect=checked), \
             patch.object(module,'inspect_archive',side_effect=inspector,return_value=(self.manifest,self.members)), \
             patch.object(module,'bind_manifest_boot',side_effect=lambda manifest,boot:dict(manifest,boot_id=boot)), \
             patch.object(module.time,'monotonic',return_value=100), \
             patch.object(Path,'read_text',new=read), \
             patch.object(Path,'mkdir',side_effect=WorkloadBoundaryReached('before model workload')):
            return module.human_main(self.args)
    def test_authenticated_common_reaches_workload_boundary_after_all_staged_source_checks(self):
        with self.assertRaisesRegex(WorkloadBoundaryReached,'before model workload'):self.entry()
        self.assertEqual(set(self.checked_sources),{p.name for p in self.review.iterdir()})
        self.assertIn('artifact_audit',vars(self.args))
        self.assertFalse(Path(self.manifest['target_root']).exists())
    def test_retained_native_failure_reproduces_with_the_previous_transport(self):
        old=HERE/'proposals/common-closure/review/benchmark_host_transport.py'
        self.path.write_bytes(old.read_bytes());self.args.expected_self_sha256=digest(self.path.read_bytes())
        self.manifest['package_files']['review/benchmark_host_transport.py']=self.args.expected_self_sha256
        module=self.load('bootstrap_common_stage_old',self.path)
        with self.assertRaisesRegex(ValueError,'protected stage must contain only'):self.entry(module)
        self.assertFalse(Path(self.manifest['target_root']).exists())
    def test_unexpected_source_still_refuses_before_workload_creation(self):
        (self.review/'unexpected.py').write_bytes(b'raise AssertionError("never executed")\n')
        with self.assertRaisesRegex(ValueError,'protected stage must contain only'):self.entry()
    def test_missing_or_changed_common_refuses_even_after_import_authentication(self):
        common=self.review/'spry_common_canon.py';raw=common.read_bytes()
        common.unlink()
        with self.assertRaises(FileNotFoundError):self.entry()
        common.write_bytes(raw+b'# changed\n')
        with self.assertRaisesRegex(ValueError,'root source hash differs'):self.entry()
    def test_common_link_is_not_an_admitted_source(self):
        common=self.review/'spry_common_canon.py';common.unlink();common.symlink_to(HERE/'spry_common/selected.py')
        with self.assertRaises((ValueError,OSError)):self.entry()


if __name__=='__main__':unittest.main()
