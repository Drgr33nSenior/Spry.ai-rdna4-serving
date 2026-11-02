"""Check parity import isolation through both actual authenticated HTTP chains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_production_startup_packet as startup_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha
from test_source_builder import builder

PROPOSAL=HERE.parent/'proposals/production-imports'


class ProductionImportsPacket(startup_checks.ProductionStartupPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(startup_checks,'PROPOSAL',PROPOSAL):super().setUpClass()

    def test_exact_diff_and_unmodified_release_validators(self):
        before=json.loads((HERE.parent/'proposals/production-startup/PROPOSAL.json').read_bytes())
        self.assertEqual(self.record['protected_files'],before['protected_files'])
        self.assertEqual(self.record['protected_set_sha256'],before['protected_set_sha256'])
        for name,pin in self.record['protected_files'].items():self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),self.record['source_patch_sha256'])
        self.evidence.update(protected_helpers_unchanged_from_startup_stage=True,
            exact_source_pins_verified=True,release_validators_unchanged=True,owner_ceilings_changed=False)

    def test_both_production_http_chains_exclude_parity_and_instrumented_chains_retain_it(self):
        cases=[]
        with self.controller() as (controller,_,_,_,root):
            model=controller.adapter.readout.bench
            original_settings=(root/'review/FINAL-SETTINGS.json').read_bytes()
            settings=json.loads(original_settings)
            for instrumented in (False,True):
                settings['instrumentation']='on' if instrumented else 'production'
                raw_settings=canonical(settings)
                (root/'review/FINAL-SETTINGS.json').write_bytes(raw_settings)
                spec=json.loads(self.members['review/BENCHMARK-SPEC2.json'])
                templates={}
                for logical in ('control','mtp'):
                    name=f'HTTP-BINDING-{logical}.json'
                    value=json.loads(self.members['review/'+name])
                    sources=value['binding_template']['sources']
                    for row in sources.values():
                        if row['path'].startswith('/mnt/review/'):
                            row['path']=str(root/'review'/Path(row['path']).name)
                            if row['path']==str(root/'review/FINAL-SETTINGS.json'):
                                row['sha256']=sha(raw_settings)
                            if instrumented and Path(row['path']).name in getattr(controller.mtp_contract,'PRODUCTION_EVIDENCE_FILES',()):
                                row['sha256']=sha(Path(row['path']).read_bytes())
                    pin=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    value['source_manifest_sha256']=pin
                    for key in ('binding_template','worker_template'):value[key]['source_manifest_sha256']=pin
                    value['worker_template']['sources']['source_manifest_sha256']=pin
                    value['worker_template']['preweight']['graph_observer']=dict(sources['graph_observer'])
                    templates['/mnt/review/'+name]=canonical(value)
                    spec['http_serving']['bindings'][logical]['sha256']=sha(templates['/mnt/review/'+name])
                checked=model.checked_file
                def local_template(path,expected,maximum=8<<20):
                    raw=templates.get(str(path))
                    if raw is None:return checked(path,expected,maximum)
                    self.assertEqual(sha(raw),expected);self.assertLessEqual(len(raw),maximum)
                    return raw
                for arm in ('bf16','kv4'):
                    with isolated_modules(),patch.object(model,'checked_file',side_effect=local_template):
                        for name in builder.HTTP_IMPORTS:sys.modules.pop(name,None)
                        runtime=model.http_runtime_module(spec,arm)
                        self.assertEqual(Path(runtime.__file__),root/'review/serving_runtime.py')
                        self.assertEqual('serving_parity' in sys.modules,instrumented)
                        for name in builder.HTTP_IMPORTS:
                            if name=='serving_parity' and not instrumented:continue
                            self.assertEqual(Path(sys.modules[name].__file__),root/'review'/(name+'.py'))
                        cases.append(dict(arm=arm,instrumented=instrumented,parity_imported=instrumented))
            (root/'review/FINAL-SETTINGS.json').write_bytes(original_settings)
        self.evidence['actual_http_parity_imports']=dict(cases=cases,
            actual_source_modules=True,native_imports_executed=False,
            instrumented_fixture='Explicit mode and matching source JSON hashes at the import boundary only; whole-run mutation verifier unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args();inventory_checks.PACKET=options.packet
    with isolated_modules():result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(ProductionImportsPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
