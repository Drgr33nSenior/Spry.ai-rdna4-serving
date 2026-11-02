"""Verify the post-load package against the authenticated closed HTTP control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_closed_http_packet as closed
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest
configured=closed.configured
PROPOSAL=HERE.parent/'proposals/post-load'
configured.PROPOSAL=PROPOSAL

class PostLoadPacket(closed.ClosedHttpPacket):
    evidence={}

    def pair(self,name):
        candidate,_,_=self.package(name)
        output=self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(),output,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/closed-http').resolve())
        parent=output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent,candidate

    def test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins(self):
        super().test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins()
        package,pins,_=self.package('post-load-archive')
        files={p.relative_to(package).as_posix():p.read_bytes() for p in package.rglob('*')
               if p.is_file() and p.name!='MANIFEST-PROPOSAL.json'}
        transport=load('post_load_transport',package/'review/benchmark_host_transport.py')
        for label,name,raw in (
            ('runner-rollback','review/d3b_combined_model_runner.py',self.authenticated['author/package-v2/review/d3b_combined_model_runner.py']),
            ('hook-change','review/spry_post_load.py',b'# substituted startup hook\n')):
            bad=dict(files);bad[name]=raw
            manifest=json.loads(files['MANIFEST.json']);manifest['package_files'][name]=digest(raw)
            bad['MANIFEST.json']=canonical(manifest);archive=self.archive(label,bad)
            with self.subTest(variant=label),self.assertRaisesRegex(ValueError,'exact post-load runner/helper'):
                transport.inspect_archive(archive,digest(archive.read_bytes()),digest(bad['MANIFEST.json']),configuration_pins=pins)
        self.evidence['post_load_archive_refusals']=dict(runner_rollback=True,changed_hook=True,
            self_consistent_manifest_pins=True,native_execution=False)

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent,candidate=self.pair('post-load-delta')
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(parent),data(candidate)
        self.assertEqual(set(before)-set(after),set())
        self.assertEqual(set(after)-set(before),{'review/spry_post_load.py'})
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        code={'review/'+name for name,row in record['runtime_changes'].items() if row['baseline_sha256']!=row['proposed_sha256']}
        protected=set(record['changes'])
        derived={'MANIFEST.json','MANIFEST-PROPOSAL.json','PROTECTED-SHA256.json','review/RESOURCE-MANIFEST.json',
                 'review/BENCHMARK-SPEC2.json','review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json'}
        changed={name for name in set(before)&set(after) if before[name]!=after[name]}
        self.assertEqual(changed,code|protected|derived)
        self.assertEqual(len(after)-1,78)
        self.evidence['post_load_payload_delta']=dict(changed=sorted(changed),added=['review/spry_post_load.py'],
            archive_members=78,owner_archive_ceiling=80,kernels_and_all_other_active_overlays_byte_identical=True,
            native_execution=False)

    def test_packaged_hook_is_source_bound_in_both_workers_and_refuses_changed_bytes(self):
        package,_,_=self.package('post-load-source');review=package/'review'
        contract=load('post_load_packet_contract',review/'mtp_contract.py')
        for arm in ('control','mtp'):
            value=json.loads((review/('HTTP-BINDING-'+arm+'.json')).read_bytes())
            self.assertEqual(value['binding_template']['sources']['post_load'],contract.POST_LOAD_SOURCE)
            self.assertEqual(value['binding_template']['sources']['model_runner']['sha256'],digest((review/'d3b_combined_model_runner.py').read_bytes()))
        worker=load('post_load_packet_worker',review/'worker_entry.py')
        source=dict(contract.POST_LOAD_SOURCE,path=str(review/'spry_post_load.py'))
        expected=SimpleNamespace(POST_LOAD_SOURCE=source)
        manifest=dict(sources=dict(post_load=source))
        with patch.dict(sys.modules):
            sys.modules.pop('spry_post_load',None)
            module=worker._load_post_load_hook(manifest,expected)
            self.assertEqual(module.__file__,source['path'])
            self.assertTrue(callable(module.run))
            (review/'spry_post_load.py').write_bytes(b'raise AssertionError("changed hook executed")\n')
            with self.assertRaisesRegex(worker.WorkerEntryError,'source changed'):
                worker._load_post_load_hook(manifest,expected)
        self.evidence['post_load_source_binding']=dict(arms=2,actual_packaged_worker=True,
            authenticated_source_sha256=source['sha256'],changed_bytes_refused=True,
            local_path_projection_only=True,native_execution=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        test_source_sha256=digest(Path(__file__).read_bytes()),cases=PostLoadPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
