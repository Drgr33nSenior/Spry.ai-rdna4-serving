"""Check observed rank identity and production startup without native imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load


class HostBinding(unittest.TestCase):
    def setUp(self):
        source=HERE.parent/'legacy/review'
        self.contract=load('cleanup_host_contract',source/'mtp_contract.py')
        self.worker=load('cleanup_host_worker',source/'worker_entry.py')
        self.host=json.loads((HERE.parent/'config/host/r9700-pair.json').read_bytes())
        # Index 0 is c3; canonical JSON orders 23 first. This catches key-order inference.
        self.devices=[dict(index=index,bdf=bdf,uuid=self.host['devices'][bdf]['uuid'],
            architecture='gfx1201',vram_bytes=24<<30) for index,bdf in enumerate(('0000:c3:00.0','0000:23:00.0'))]
        self.renders=[dict(bdf=bdf,sysfs_bdf=bdf,**item) for bdf,item in self.host['devices'].items()]
        temporary=tempfile.TemporaryDirectory(prefix='host-binding-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.manifest=dict(mode='native-http-fp8-sprint01-v1',arm='control',expected_devices=self.host['devices'],
            expected_ranks=[0,1],session_sha256='a'*64,spec_sha256='b'*64,evidence_dir=str(self.root),
            preweight=dict(deadline_monotonic=200,gpu_identity={'kind':'gpu'},graph_observer={'kind':'graph'}),
            arm_config=dict(backend_source_sha256='c'*64),sources={'mtp_contract':{},'gemm':{},'scheduler':{}})

    def bind(self,devices=None,renders=None,indices=None,rank=0,gpu_id=0):
        return self.contract.bind_observed_rank(self.devices if devices is None else devices,
            self.renders if renders is None else renders,self.host['devices'],
            [0,1] if indices is None else indices,rank,gpu_id)

    def test_observed_index_controls_binding_independently_of_json_order(self):
        self.assertEqual(self.bind()['bdf'],'0000:c3:00.0')
        self.assertEqual(self.bind(rank=1,gpu_id=1)['bdf'],'0000:23:00.0')
        self.assertEqual(self.bind(indices=[1,0],gpu_id=1)['bdf'],'0000:23:00.0')
        self.assertEqual(self.bind(devices=list(reversed(self.devices)))['bdf'],'0000:c3:00.0')

    def test_invalid_rank_index_and_duplicates_refuse(self):
        for indices,rank,gpu in (([0,0],0,0),([True,1],0,0),([0,1],True,0),
                                ([0,1],0,True),([0,1],0,1),([0,1],2,0)):
            with self.subTest(indices=indices,rank=rank,gpu=gpu),self.assertRaises(ValueError):
                self.bind(indices=indices,rank=rank,gpu_id=gpu)
        devices=copy.deepcopy(self.devices);devices[1]['index']=0
        with self.assertRaises(ValueError):self.bind(devices=devices)

    def test_wrong_bdf_uuid_or_render_observation_refuses(self):
        for field,value in (('bdf','0000:01:00.0'),('uuid','wrong')):
            devices=copy.deepcopy(self.devices);devices[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.bind(devices=devices)
        for field,value in (('render','/dev/dri/renderD999'),('sysfs_bdf','0000:c3:00.0'),('uuid','wrong')):
            renders=copy.deepcopy(self.renders);renders[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.bind(renders=renders)

    def preweight(self,instrumented):
        helper=SimpleNamespace(physical_devices=Mock(return_value=(self.devices,{'no_load_handle':True})))
        graph=SimpleNamespace(install=Mock())
        source=Mock(side_effect=lambda value:helper if value['kind']=='gpu' else graph)
        with patch.dict(sys.modules,torch=SimpleNamespace()), \
                patch.object(self.worker.time,'monotonic',return_value=100), \
                patch.object(self.worker,'_source_module',source), \
                patch.object(self.worker,'_render_device_bindings',return_value=self.renders), \
                patch.object(self.worker,'_validate_serving_startup') as startup, \
                patch.object(self.worker,'_atomic_json') as write, \
                patch.object(self.worker,'_mtp_preflight') as mtp:
            result=self.worker._preweight_before_scheduler(self.manifest,0,object(),gpu_id=0,
                contract=self.contract,instrumented=instrumented)
            startup.assert_called_once()
        return result,helper,graph,source,write,mtp

    def test_production_checks_identity_without_importing_observers_or_writing_evidence(self):
        result,helper,graph,source,write,mtp=self.preweight(False)
        self.assertEqual(result['rank_binding'],self.bind())
        helper.physical_devices.assert_called_once()
        self.assertEqual(source.call_count,1)
        graph.install.assert_not_called();write.assert_not_called();mtp.assert_not_called()

    def test_instrumented_startup_retains_evidence_and_observer_dispatch(self):
        result,helper,graph,source,write,mtp=self.preweight(True)
        self.assertEqual(result['rank_binding'],self.bind())
        helper.physical_devices.assert_called_once()
        graph.install.assert_called_once();write.assert_called_once();mtp.assert_called_once()

    def test_actual_readout_refuses_stale_rank_bdf_and_type_aliases(self):
        integration=load('cleanup_host_integration',HERE.parent/'legacy/review/serving_integration.py')
        record,_,_,_,_,_=self.preweight(True)
        with patch.object(integration,'_mtp_contract',return_value=self.contract):
            integration._validate_observed_rank(record,self.manifest,0)
            for key,value in (('rank',True),('hip_index',True),('bdf','0000:23:00.0'),('uuid','wrong')):
                changed=copy.deepcopy(record);changed['rank_binding'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    integration._validate_observed_rank(changed,self.manifest,0)

    def test_expired_or_wrong_rank_refuses_before_identity_queries(self):
        for now,gpu in ((200,0),(100,1)):
            with self.subTest(now=now,gpu=gpu),patch.object(self.worker.time,'monotonic',return_value=now), \
                    patch.object(self.worker,'_source_module') as source,self.assertRaises(ValueError):
                self.worker._preweight_before_scheduler(self.manifest,0,object(),gpu_id=gpu,
                    contract=self.contract,instrumented=False)
            source.assert_not_called()

    def test_actual_scheduler_checks_identity_before_delegating_once_in_production(self):
        events=[]
        helper=SimpleNamespace(physical_devices=Mock(side_effect=lambda *args:(events.append('identity') or self.devices,{})))
        target=Mock(side_effect=lambda *args,**kwargs:events.append('scheduler') or 'served')
        contract=SimpleNamespace(instrumentation_enabled=lambda:False,dense_mxfp4_enabled=lambda:False,
            verify_attention_observed=lambda arm:False,bind_observed_rank=self.contract.bind_observed_rank)
        def source(value):return helper if value.get('kind')=='gpu' else contract
        with patch.dict(sys.modules,torch=SimpleNamespace()), \
                patch.object(self.worker.time,'monotonic',return_value=100), \
                patch.object(self.worker,'_read_manifest',return_value=self.manifest), \
                patch.object(self.worker,'_validate_manifest',side_effect=lambda value:value), \
                patch.object(self.worker,'_worker_record',return_value={}), \
                patch.object(self.worker,'_atomic_json'),patch.object(self.worker,'_native_module'), \
                patch.object(self.worker,'_validate_serving_startup',side_effect=lambda *args:events.append('configuration')), \
                patch.object(self.worker,'_source_module',side_effect=source), \
                patch.object(self.worker,'_render_device_bindings',return_value=self.renders), \
                patch.object(self.worker,'_source_selection',return_value=target):
            result=self.worker.run_scheduler_process(object(),object(),0,0,0,None,object())
        self.assertEqual(result,'served')
        self.assertEqual(events,['configuration','identity','scheduler'])
        target.assert_called_once()


if __name__=='__main__':unittest.main()
