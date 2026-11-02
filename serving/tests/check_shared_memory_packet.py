"""Exercise shared-head and packed-view source through the consumed CPU packet."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_pre_replay_packet as replay
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest
configured=replay.configured
PROPOSAL=HERE.parent/'proposals/shared-memory'
configured.PROPOSAL=PROPOSAL


class SharedMemoryPacket(replay.PreReplayPacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/pre-replay').resolve())
        parent=output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('memory-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate)
        self.assertEqual(set(before)-set(after),set())
        self.assertEqual(set(after)-set(before),{'review/spry_fp8_head.py'})
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        code={'review/'+name for name in record['runtime_changes']}
        protected=set(record['changes'])
        derived={'MANIFEST.json','MANIFEST-PROPOSAL.json','PROTECTED-SHA256.json','review/RESOURCE-MANIFEST.json',
                 'review/BENCHMARK-SPEC2.json','review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'}
        changed={name for name in set(before)&set(after) if before[name]!=after[name]}
        self.assertEqual(changed,code|protected|derived)
        self.assertEqual(len(after)-1,79)
        self.evidence['memory_payload_delta']=dict(changed=sorted(changed),added=['review/spry_fp8_head.py'],
            archive_members=79,owner_archive_ceiling=80,kernels_and_other_active_overlays_requests_settings_unchanged=True,
            native_execution=False)

    def test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins(self):
        super().test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins()
        package,pins,_=self.package('memory-archive')
        files={p.relative_to(package).as_posix():p.read_bytes() for p in package.rglob('*')
               if p.is_file() and p.name!='MANIFEST-PROPOSAL.json'}
        transport=load('memory_transport',package/'review/benchmark_host_transport.py')
        for name,label in [('isolated_scheme.py','MXFP4 archive'),('p1_prefill.py','P1 prefill'),
                           ('d3b_target_head_adapter.py','D3b target'),('d3b_head_adapter.py','drafter-head'),
                           ('spry_post_load.py','post-load'),('spry_fp8_head.py','post-load')]:
            raw=self.authenticated.get('author/package-v2/review/'+name,b'# unbound helper\n')
            bad=dict(files);key='review/'+name;bad[key]=raw
            manifest=json.loads(files['MANIFEST.json']);manifest['package_files'][key]=digest(raw)
            bad['MANIFEST.json']=canonical(manifest);archive=self.archive('memory-rollback-'+name,bad)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,label):
                transport.inspect_archive(archive,digest(archive.read_bytes()),digest(bad['MANIFEST.json']),configuration_pins=pins)
        self.evidence['memory_archive_refusals']=dict(rollback_sources=6,self_consistent_manifest_pins=True,
            native_execution=False)

    def test_packaged_hook_is_source_bound_in_both_workers_and_refuses_changed_bytes(self):
        package,_,_=self.package('memory-source');review=package/'review'
        contract=load('memory_packet_contract',review/'mtp_contract.py')
        worker=load('memory_packet_worker',review/'worker_entry.py')
        sources={}
        for role,expected in [('post_load',contract.POST_LOAD_SOURCE),('shared_fp8_head',contract.SHARED_FP8_HEAD_SOURCE)]:
            sources[role]=dict(expected,path=str(review/Path(expected['path']).name))
            for arm in ('control','mtp'):
                binding=json.loads((review/('HTTP-BINDING-'+arm+'.json')).read_bytes())
                self.assertEqual(binding['binding_template']['sources'][role],expected)
        expected=SimpleNamespace(POST_LOAD_SOURCE=sources['post_load'],SHARED_FP8_HEAD_SOURCE=sources['shared_fp8_head'])
        with patch.dict(sys.modules):
            for name in ('spry_post_load','spry_fp8_head'):sys.modules.pop(name,None)
            worker._load_post_load_hook(dict(sources=sources),expected)
            helper=sys.modules['spry_fp8_head'];helper.shared_fp8_head=lambda *a,**k:None
            with self.assertRaisesRegex(worker.WorkerEntryError,'callable/code/globals changed'):
                worker._load_post_load_hook(dict(sources=sources),expected)
            sys.modules.pop('spry_fp8_head')
            (review/'spry_fp8_head.py').write_bytes(b'raise AssertionError("changed helper executed")\n')
            with self.assertRaisesRegex(worker.WorkerEntryError,'source changed'):
                worker._load_post_load_hook(dict(sources=sources),expected)
        self.evidence['memory_source_binding']=dict(arms=2,actual_packaged_worker=True,
            helper_code_and_globals_authenticated=True,changed_bytes_refused=True,native_execution=False)

    def test_packaged_memory_reader_refuses_incomplete_and_old_witnesses(self):
        package,_,_=self.package('memory-reader');review=package/'review'
        contract=load('memory_reader_contract',review/'mtp_contract.py')
        sources=dict(shared_fp8_head=contract.SHARED_FP8_HEAD_SOURCE)
        helper=contract.memory_recovery_helper(sources)
        cases=load('memory_reader_cases',HERE/'test_shared_memory.py')
        proof=cases.memory_record(0);self.assertIs(helper.validate_memory_recovery(proof,rank=0),proof)
        proof['sharing']['draft']['fp8_pointer']+=256
        with self.assertRaisesRegex(ValueError,'same head objects'):helper.validate_memory_recovery(proof,rank=0)
        observer=load('memory_observer_packet',review/'serving_graph_observer.py')
        worker_module=load('memory_producer_packet',review/'worker_entry.py')
        binding=json.loads((review/'HTTP-BINDING-control.json').read_bytes())['binding_template']
        binding['evidence_dir']=str(self.root/'memory-witness-no-write')
        worker=dict(rank=0,pid=123,start_ticks=456,boot_id='source-only-fixture',identity_sha256='e'*64)
        sources=binding['sources']
        policy=contract.precision_loading_policy()
        encoded=dict(policy,model_loader_extra_config=json.dumps(policy['model_loader_extra_config'],sort_keys=True,separators=(',',':')))
        loading=dict(sources=contract.precision_loading_sources(sources),resolved=encoded,runtime=dict(encoded),
            loader_class='DefaultModelLoader',loader_module=sources['model_loader']['module'],
            load_config_class='LoadConfig',load_config_module=sources['loader_load_config']['module'],
            load_config=dict(load_format='safetensors',model_loader_extra_config=policy['model_loader_extra_config']),
            loader_config_is_runner_config=True,
            scope='effective stock configuration and loaded classes; cache advice syscall count is not observed')
        rows=[]
        for plan in contract.mxfp4_linear_layout(0):
            row=dict(path=plan['path'],class_name=plan['class_name'],class_module=sources['head_linear']['module'],
                method_class='QuarkLinearMethod',method_module=sources['quark_config']['module'],
                scheme_class='IsolatedPackedMXFP4',scheme_module=sources['mxfp4_scheme']['module'],loaded=True)
            for name,shape,cls in [('weight',plan['weight_shape'],'PackedvLLMParameter'),
                                   ('weight_scale',plan['scale_shape'],'GroupQuantScaleParameter')]:
                row[name]=dict(class_name=cls,class_module=sources['quark_parameter']['module'],dtype='torch.uint8',
                    shape=shape,stride=[shape[1],1],device='cuda:0',is_cuda=True,contiguous=True,
                    element_bytes=1,elements=shape[0]*shape[1])
            rows.append(row)
        inventory=dict(sources=contract.mxfp4_sources(sources),modules=rows,selection_calls=dict(target=1,delegated=1),
            selection_active=True,target_root=contract.MODEL_PATH,revision=contract.target_model_revision(),
            scope='loaded parameter metadata; no numerical equivalence claim')
        representative=next(row for row in rows if row['path']=='model.layers.0.mlp.down_proj')
        linear=type('RowParallelLinear',(),{'__module__':sources['head_linear']['module']})()
        linear.quant_method=type('QuarkLinearMethod',(),{'__module__':sources['quark_config']['module']})()
        linear.weight=SimpleNamespace(dtype='torch.uint8',shape=representative['weight']['shape'],
            device='cuda:0',is_cuda=True,is_contiguous=lambda:True)
        modules={sources['head_linear']['module']:SimpleNamespace(RowParallelLinear=type(linear)),
            sources['quark_config']['module']:SimpleNamespace(QuarkLinearMethod=type(linear.quant_method))}
        b8=SimpleNamespace(data_ptr=lambda:256);scales=SimpleNamespace(data_ptr=lambda:512)
        sharing=dict(fp8_id=id(b8),scales_id=id(scales),fp8_pointer=256,scales_pointer=512)
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=linear))]),
            lm_head=SimpleNamespace(_spry_shared_draft_binding=dict(sharing)),
            logits_processor=SimpleNamespace(_compute_lm_head=SimpleNamespace(b8=b8,scales=scales)))
        runner=SimpleNamespace(model=model,model_config=SimpleNamespace(quantization='quark'),
            _spry_memory_recovery=cases.memory_record(0))
        del runner._spry_memory_recovery['sharing']
        captured={}
        # Only native class resolution and unchanged loading/weight metadata
        # extraction are substituted. Producer, 304-row reader and memory
        # reader are the actual packaged callables.
        with patch.dict(sys.modules,mtp_contract=contract), \
             patch.object(worker_module,'_native_module',side_effect=lambda source:modules[source['module']]), \
             patch.object(worker_module,'_precision_loading_policy',return_value=loading), \
             patch.object(worker_module,'_mxfp4_loaded_weights',return_value=inventory), \
             patch.object(worker_module,'_atomic_json',side_effect=lambda path,value:captured.update(record=value)), \
             patch.object(contract,'full_enabled_benchmark',return_value=False):
            worker_module._precision_weights(SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner)),
                binding,worker,observer,contract,{})
            record=captured['record']
            self.assertEqual(record['extra_tensor_operations'],0)
            self.assertEqual(record['memory_recovery']['head']['transient_quantizations'],1)
            self.assertIs(observer.validate_precision_weights(record,binding=binding,worker=worker),record)
            record['memory_recovery']['packed'].pop()
            with self.assertRaisesRegex(ValueError,'all GDN layers'):
                observer.validate_precision_weights(record,binding=binding,worker=worker)
            del record['memory_recovery']
            with self.assertRaisesRegex(ValueError,'schema/profile'):
                observer.validate_precision_weights(record,binding=binding,worker=worker)
        (review/'spry_fp8_head.py').write_bytes(b'# changed helper\n')
        with self.assertRaises(ValueError):contract.memory_recovery_helper(sources)
        self.evidence['memory_reader']=dict(actual_packaged_producer_observer_and_helper=True,
            loaded_weight_rows=304,incomplete_sharing_and_missing_layer_refused=True,old_witness_refused=True,
            changed_helper_refused=True,source_only_descriptors=True,native_execution=False)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args();configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=SharedMemoryPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
