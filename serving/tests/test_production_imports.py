"""Ordinary startup imports no parity code; instrumented callbacks retain it."""
import builtins
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load


class LauncherImports(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-parity-import-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()

    def exercise(self,instrumented):
        contract=SimpleNamespace(instrumentation_enabled=lambda:instrumented)
        worker=SimpleNamespace(_source_module=Mock(return_value=contract),run_scheduler_process=Mock())
        manifest=dict(evidence_dir=str(self.root),sources=dict(mtp_contract=dict(path='pinned',sha256='a'*64)),expected_ranks=[0,1])
        managers=(object(),object());manager=Mock(return_value=managers)
        names=('sglang','sglang.srt','sglang.srt.entrypoints','sglang.srt.entrypoints.engine')
        modules={name:ModuleType(name) for name in names}
        for name,module in modules.items():module.__path__=[]
        modules[names[-1]].init_tokenizer_manager=manager
        parity=SimpleNamespace(collect=Mock())
        imported=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name=='serving_parity':
                imported.append(name)
                if not instrumented:raise AssertionError('production parity import')
            return original(name,*args,**kwargs)
        def launch(server_args,**kwargs):return kwargs['init_tokenizer_manager_func']('native-manager-args')
        with patch.dict(sys.modules,dict(modules,worker_entry=worker,serving_parity=parity)), \
                patch.object(builtins,'__import__',side_effect=imports):
            launcher=load('cleanup_conditional_launcher',HERE.parent/'legacy/review/serving_launcher.py')
            with patch.object(launcher,'_read_checked_manifest',return_value=manifest):
                result=launcher.launch_native_http(object(),native_launch_server=launch,
                    worker_manifest=self.root/'worker.json',worker_manifest_sha256='b'*64)
        self.assertIs(result,managers);manager.assert_called_once_with('native-manager-args')
        if instrumented:
            parity.collect.assert_called_once_with(*managers,manifest);self.assertEqual(imported,['serving_parity'])
        else:
            parity.collect.assert_not_called();self.assertEqual(imported,[])

    def test_production_import_and_tokenizer_callback_skip_parity(self):self.exercise(False)
    def test_instrumented_callback_retains_one_parity_collection(self):self.exercise(True)


class RuntimeImports(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-runtime-import-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()
        self.config=SimpleNamespace(settings=lambda:dict(phase='paired'),LIFECYCLE_PHASE='lifecycle')
        self.contract=SimpleNamespace(HIP_PHASE_ARMS={},hip_sampling_arm=lambda phase,arm:True)
        self.parity=SimpleNamespace(validate_hip_sampling_components=Mock(return_value='validated'))
        self.binding=dict(arm='control');self.material=dict(entry_binding=self.binding,native_dir=self.root)

    def exercise(self,required):
        if required:
            self.contract.HIP_PHASE_ARMS={'paired':'control'}
            for rank in (0,1):(self.root/f'HIP-SAMPLING-COMPONENT-rank-{rank}.json').write_text(json.dumps(dict(rank=rank)))
        imports=[];original=builtins.__import__
        def checked(name,*args,**kwargs):
            if name=='serving_parity':
                imports.append(name)
                if not required:raise AssertionError('ordinary parity import')
            return original(name,*args,**kwargs)
        seams=dict(mtp_contract=self.config,serving_contract=self.contract,serving_client=SimpleNamespace(ServingClient=Mock()),
            serving_entry=SimpleNamespace(run_bound=Mock()),serving_parity=self.parity)
        with patch.dict(sys.modules,seams),patch.object(builtins,'__import__',side_effect=checked):
            runtime=load('cleanup_conditional_runtime',HERE.parent/'legacy/review/serving_runtime.py')
            self.assertNotIn('serving_parity',runtime.__dict__)
            value=runtime._hip_sampling_ready(self.material,dict(workers=['cpu-worker-0','cpu-worker-1']))
        if required:
            self.assertEqual(value,'validated');self.assertEqual(imports,['serving_parity'])
            self.parity.validate_hip_sampling_components.assert_called_once()
        else:
            self.assertIsNone(value);self.assertEqual(imports,[])
            self.parity.validate_hip_sampling_components.assert_not_called()

    def test_ordinary_readiness_does_not_import_parity(self):self.exercise(False)
    def test_retained_component_reader_still_loads_its_parity_validator(self):self.exercise(True)


if __name__=='__main__':unittest.main()
