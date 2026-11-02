"""Final private source pins, worker aliases and production masks in real JSON."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import build as builder


class FinalTemplatePins(unittest.TestCase):
    def setUp(self):
        self.files={
            'FINAL-SETTINGS.json':builder.canonical(dict(instrumentation='on')),
            'benchmark_gpu_identity.py':(HERE/'proposals/common-closure/review/benchmark_gpu_identity.py').read_bytes(),
            'serving_graph_observer.py':b'# exact observer source\n',
            'BENCHMARK-SPEC2.json':builder.canonical(dict(experiment_mode=builder.MODE,
                http_serving=dict(bindings={arm:dict(sha256='a'*64) for arm in ('control','mtp')})))}
        for arm in ('control','mtp'):
            sources=dict(gpu_identity=dict(path='/mnt/review/benchmark_gpu_identity.py',module='benchmark_gpu_identity',
                         sha256='7324842c4f779749d561dc586168bd923e84e6900d17e3d22c140647bcd133df'),
                         graph_observer=dict(path='/mnt/review/serving_graph_observer.py',module='serving_graph_observer',sha256='b'*64),
                         native=dict(path='/mnt/composition/model_runner.py',module='stock_runner',sha256='c'*64))
            value=dict(mode=builder.MODE,arm=arm,source_manifest_sha256='d'*64,
                binding_template=dict(sources=sources,source_manifest_sha256='d'*64),
                worker_template=dict(sources=dict(same_as='binding_template.sources',source_manifest_sha256='d'*64),
                    source_manifest_sha256='d'*64,preweight=dict(graph_observer=copy.deepcopy(sources['graph_observer']))))
            self.files['HTTP-BINDING-'+arm+'.json']=builder.canonical(value)
    def templates(self):return {arm:json.loads(self.files['HTTP-BINDING-'+arm+'.json']) for arm in ('control','mtp')}
    def test_final_identity_bytes_refresh_both_templates_aliases_and_spec_pins(self):
        builder.refresh_http_source_bindings(self.files)
        spec=json.loads(self.files['BENCHMARK-SPEC2.json'])
        for arm,value in self.templates().items():
            sources=value['binding_template']['sources'];pin=builder.digest(builder.canonical_bare(sources))
            self.assertEqual(sources['gpu_identity']['sha256'],builder.digest(self.files['benchmark_gpu_identity.py']))
            self.assertEqual(sources['native']['sha256'],'c'*64)
            self.assertEqual(value['source_manifest_sha256'],pin)
            for name in ('binding_template','worker_template'):self.assertEqual(value[name]['source_manifest_sha256'],pin)
            self.assertEqual(value['worker_template']['sources']['source_manifest_sha256'],pin)
            self.assertEqual(value['worker_template']['preweight']['graph_observer'],sources['graph_observer'])
            self.assertEqual(spec['http_serving']['bindings'][arm]['sha256'],builder.digest(self.files['HTTP-BINDING-'+arm+'.json']))
    def test_refresh_is_idempotent_without_registering_extracted_helpers_again(self):
        builder.refresh_http_source_bindings(self.files);before=copy.deepcopy(self.files)
        builder.refresh_http_source_bindings(self.files);self.assertEqual(self.files,before)
    def test_production_mask_remains_bound_while_identity_source_uses_its_real_bytes(self):
        self.files['FINAL-SETTINGS.json']=builder.canonical(dict(instrumentation='production'))
        self.files['mtp_contract.py']=(HERE/'spry_serving/common_mtp_contract.py').read_bytes()
        policy=builder.production_evidence_policy(self.files)
        self.assertIn('serving_graph_observer.py',policy['files'])
        builder.refresh_http_source_bindings(self.files)
        for value in self.templates().values():
            sources=value['binding_template']['sources']
            self.assertEqual(sources['graph_observer']['sha256'],policy['sha256'])
            self.assertEqual(sources['gpu_identity']['sha256'],builder.digest(self.files['benchmark_gpu_identity.py']))
    def test_missing_declared_private_source_refuses(self):
        self.files.pop('benchmark_gpu_identity.py')
        with self.assertRaisesRegex(ValueError,'private source missing'):builder.refresh_http_source_bindings(self.files)
    def test_unreviewed_worker_alias_shape_refuses(self):
        name='HTTP-BINDING-control.json';value=json.loads(self.files[name]);value['worker_template']['sources']['extra']='unreviewed'
        self.files[name]=builder.canonical(value)
        with self.assertRaisesRegex(ValueError,'worker source alias differs'):builder.refresh_http_source_bindings(self.files)


if __name__=='__main__':unittest.main()
