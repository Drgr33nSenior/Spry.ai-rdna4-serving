"""Exercise spec-bound limits through actual CPU export and readout."""
import argparse
import difflib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_host_binding_packet as host_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_config_binding_proposal import canonical,sha
from test_source_builder import builder

PROPOSAL=HERE.parent/'proposals/d2-artifact-binding'


class ArtifactBindingPacket(host_checks.HostBindingPacket):
    evidence={}

    def run_owner(self, controller, *args, proposed=True, **kwargs):
        if proposed:
            return super().run_owner(controller,*args,proposed=True,**kwargs)
        # The retained owner accepts its original spec schema. Compare its
        # actual decisions using that schema; the proposed path keeps the new
        # generated field and every validator intact.
        name='review/BENCHMARK-SPEC2.json'
        legacy=json.loads(self.members[name])
        legacy['http_serving'].pop('artifact_limits')
        with patch.dict(self.members,{name:canonical(legacy)}):
            result=super().run_owner(controller,*args,proposed=False,**kwargs)
        self.evidence['baseline_owner_spec_projection']='Remove only the new generated artifact_limits field for the retained owner API; no validator substitution.'
        return result

    @classmethod
    def setUpClass(cls):
        original_build=builder.build
        def configured(*args,**kwargs):
            return original_build(*args,**dict(kwargs,configured_artifacts=True))
        with patch.object(host_checks,'PROPOSAL',PROPOSAL),patch.object(builder,'build',side_effect=configured):
            super().setUpClass()
        original=Path(cls.postrun.__file__).read_text()
        adapted=original
        replacements={
            'from pathlib import Path\n':'from pathlib import Path\nfrom dataclasses import asdict\n',
            "        spec = json.loads((root / 'review/BENCHMARK-SPEC2.json').read_bytes())\n":
                "        spec = json.loads((root / 'review/BENCHMARK-SPEC2.json').read_bytes())\n        artifact_plan = artifacts.HTTPArtifactLimits.from_limits(spec['http_serving']['artifact_limits'])\n",
            'artifacts.pack_http_artifacts(output, art_binding)':'artifacts.pack_http_artifacts(output, art_binding, limits=artifact_plan)',
            'artifacts.stage_artifacts(output, art_binding)':'artifacts.stage_artifacts(output, art_binding, limits=artifact_plan)',
            'artifacts.export_bundle(export, art_binding, sink)':'artifacts.export_bundle(export, art_binding, sink, limits=artifact_plan)',
            'artifacts.receive_bundle(stdout, stderr, destination, art_binding)':'artifacts.receive_bundle(stdout, stderr, destination, art_binding, limits=artifact_plan)',
            'joins=joins, supervision=supervision)':'joins=joins, supervision=supervision, limits=asdict(artifact_plan))',
        }
        for before,after in replacements.items():
            if adapted.count(before)!=1:raise AssertionError('artifact fixture call site changed: '+before)
            adapted=adapted.replace(before,after)
        raw_patch=''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),
            fromfile='a/w5_configured_evidence.py',tofile='b/w5_artifact_evidence.py'))
        expected=(PROPOSAL/'FIXTURE-ARTIFACT-BINDING.patch').read_text()
        if raw_patch!=expected:raise AssertionError('artifact fixture patch differs')
        path=Path(cls.postrun.__file__).with_name('w5_artifact_evidence.py')
        path.write_text(adapted);cls.postrun=load('cleanup_artifact_bound_evidence',path)
        cls.evidence.update(artifact_fixture_sha256=sha(adapted.encode()),
            artifact_fixture_patch_sha256=sha(raw_patch.encode()),
            inherited_host_checker_sha256=sha((HERE/'check_host_binding_packet.py').read_bytes()),
            configured_artifacts_spec_projection=True)

    def test_exact_diff_and_unmodified_release_validators(self):
        name='review/benchmark_physical_controller.py'
        before=HERE.parent/'proposals/d2-host-binding'/name
        expected=''.join(difflib.unified_diff(before.read_text().splitlines(True),
            (PROPOSAL/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        for name,pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in self.record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.evidence.update(exact_patch_verified=True,release_validators_unchanged=True,
            owner_adapter_unchanged=True,transport_unchanged=True)

    def test_spec_projection_matches_the_verified_root_plan(self):
        with self.controller() as (controller,path,_,_,_):
            verified=controller.verify_configured_spec(self.pins,json.loads(path.read_bytes()),
                deadline=time.monotonic()+105)
            self.assertEqual(verified['spec']['http_serving']['artifact_limits'],verified['artifact_limits'])
            self.assertEqual(len(verified['artifact_limits']),16)
        self.evidence['root_spec_worker_artifact_plan_agreement']=True

    def test_reapproved_spec_cannot_replace_one_configured_artifact_limit(self):
        for mutation in ('changed','missing','extra'):
            with self.subTest(mutation=mutation),self.controller() as (controller,path,_,_,root):
                manifest=json.loads(path.read_bytes());spec_path=root/'review/BENCHMARK-SPEC2.json'
                spec=json.loads(spec_path.read_bytes())
                values=spec['http_serving']['artifact_limits']
                if mutation=='changed':values['client_tree_files']-=1
                elif mutation=='missing':del spec['http_serving']['artifact_limits']
                else:values['unknown']=1
                spec_path.write_bytes(canonical(spec));pin=sha(spec_path.read_bytes())
                manifest['spec_sha256']=pin;manifest['package_files']['review/BENCHMARK-SPEC2.json']=pin
                path.write_bytes(canonical(manifest))
                with self.assertRaisesRegex(ValueError,'configuration-generated spec differs'):
                    controller.verify_configured_spec(self.pins,manifest,deadline=time.monotonic()+105)
        self.evidence['artifact_spec_tampering_refusals']=3


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    host_checks.stop_checks.artifact_checks.worker_checks.session_checks.inventory_checks.PACKET=options.packet
    with isolated_modules():
        result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(ArtifactBindingPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
