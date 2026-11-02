"""Verify the shared common source across the protected and serving package."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_phase_closure_packet as phase
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest
configured=phase.configured
PROPOSAL=HERE.parent/'proposals/common-closure'
configured.PROPOSAL=PROPOSAL


class CommonClosurePacket(phase.PhaseClosurePacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/phase-closure').resolve())
        parent=output/'package';(parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('common-closure-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate);self.assertEqual(set(before),set(after))
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        code={'review/'+name for name in record['runtime_changes']};protected=set(record['changes'])
        derived={'MANIFEST.json','MANIFEST-PROPOSAL.json','PROTECTED-SHA256.json','review/RESOURCE-MANIFEST.json',
                 'review/BENCHMARK-SPEC2.json','review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'}
        changed={name for name in before if before[name]!=after[name]}
        self.assertEqual(changed,code|protected|derived);self.assertEqual(len(after)-1,79)
        protection=json.loads(after['PROTECTED-SHA256.json']);self.assertEqual(len(protection),10)
        self.assertEqual(protection['review/spry_common_canon.py'],digest(after['review/spry_common_canon.py']))
        self.evidence['common_closure_delta']=dict(changed=sorted(changed),archive_members=79,
            protected_helper_roles=9,protected_common_sources=1,kernel_and_tensor_request_settings_and_pool_bytes_unchanged=True,
            native_execution=False)

    def test_packaged_common_source_change_or_removal_refuses_before_code_exec(self):
        package,_,_=self.package('common-closure-refusals');review=package/'review'
        helper=review/'spry_common_canon.py';raw=helper.read_bytes()
        helper.write_bytes(b'raise AssertionError("changed source executed")\n')
        with self.assertRaisesRegex(ValueError,'shared common source'):
            load('common_closure_changed_contract',review/'mtp_contract.py')
        helper.unlink()
        with self.assertRaises(FileNotFoundError):load('common_closure_missing_contract',review/'mtp_contract.py')
        helper.write_bytes(raw)
        self.evidence['shared_source_refusals']=dict(changed=True,missing=True,before_execution=True)

    def test_packaged_guard_semantics_and_frozen_json_pins_match_the_parent(self):
        parent,candidate=self.pair('common-closure-semantics');review=candidate/'review'
        modules={};record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        guards=0;serializers=0
        with patch.dict(sys.modules):
            old_path=list(sys.path);sys.path.insert(0,str(review))
            try:
                for name in sorted(set(record['runtime_changes'])|set(record['protected_files'])):
                    spec=importlib.util.spec_from_file_location(name[:-3],review/name)
                    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
                    spec.loader.exec_module(module);modules[name]=module
                for name,module in modules.items():
                    # Execute every actual public guard at its retained API name.
                    for guard_name in ('need','_need','_require','require'):
                        function=getattr(module,guard_name,None)
                        if not callable(function):continue
                        self.assertIsNone(function(True,object()))
                        try:function(False,'common guard fixture')
                        except Exception as error:
                            self.assertIsInstance(error,(ValueError,RuntimeError));self.assertTrue(str(error).endswith('common guard fixture'))
                        else:self.fail(name+' accepted a false guard')
                        guards+=1
                for path in review.glob('*.json'):
                    value=json.loads(path.read_bytes())
                    self.assertEqual(modules['mtp_contract.py'].json_line(value),canonical(value));serializers+=1
            finally:sys.path[:]=old_path
        self.assertEqual(guards,26)
        self.evidence['common_closure_semantics']=dict(actual_packaged_modules=len(modules),guards=guards,
            canonical_json_assets=serializers,source_only=True,native_execution=False)

    def test_self_consistent_substituted_common_inventory_refuses(self):
        package,pins,_=self.package('common-closure-archive')
        files={path.relative_to(package).as_posix():path.read_bytes() for path in package.rglob('*')
            if path.is_file() and path.name!='MANIFEST-PROPOSAL.json'}
        transport=load('common_closure_transport',package/'review/benchmark_host_transport.py')
        raw=b'# substituted common source\n';key='review/spry_common_canon.py'
        bad=dict(files);bad[key]=raw;manifest=json.loads(bad['MANIFEST.json']);manifest['package_files'][key]=digest(raw)
        bad['MANIFEST.json']=canonical(manifest);protected=json.loads(bad['PROTECTED-SHA256.json']);protected[key]=digest(raw)
        bad['PROTECTED-SHA256.json']=canonical(protected);changed_pins=dict(pins,protected_set_sha256=digest(bad['PROTECTED-SHA256.json']))
        archive=self.archive('substituted-common',bad)
        with self.assertRaisesRegex(ValueError,'frozen common source pin'):
            transport.inspect_archive(archive,digest(archive.read_bytes()),digest(bad['MANIFEST.json']),configuration_pins=changed_pins)
        self.evidence['protected_common_refusal']=dict(self_consistent_manifest_and_inventory=True,
            data_pins_cannot_override_frozen_common_code=True,native_execution=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args();configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=CommonClosurePacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
