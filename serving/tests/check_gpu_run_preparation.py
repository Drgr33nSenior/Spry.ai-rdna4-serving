"""Check actual GPU source packets and materialize inert grading fixtures on CPU."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
import prepare_gpu_run as prepare
from launcher import run_native as caller
from test_quality_policy import arm_results

ROOT=None
BOOT=None


class GPURunPreparation(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.root=ROOT.resolve(strict=True)
        temporary=tempfile.TemporaryDirectory(prefix='gpu-preparation-check-');cls.addClassCleanup(temporary.cleanup)
        cls.scratch=Path(temporary.name).resolve()

    def test_both_source_archives_authenticate_and_keep_eighty_members(self):
        cases=[]
        for variant in ('baseline','cleanup'):
            author=self.root/variant/'author'
            binding=prepare.parse_json((author/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
            path=author/'package-v2/payload.tar'
            name='benchmark_host_transport'
            transport=caller.checked_module(author/'owner-upload/review',name,binding['upload_inventory']['review/'+name+'.py'])
            kwargs=({'configuration_pins':{key:binding[key] for key in ('run_config_sha256','host_sha256','protected_set_sha256')}}
                    if variant=='cleanup' else {})
            manifest,members=transport.inspect_archive(path,binding['archive_sha256'],binding['manifest_sha256'],**kwargs)
            self.assertEqual(len(members),80)
            self.assertEqual(prepare.canonical(manifest),members['MANIFEST.json'])
            for relative,raw in members.items():self.assertEqual((author/'package-v2'/relative).read_bytes(),raw)
            cases.append(dict(variant=variant,members=80,archive_bytes=path.stat().st_size,schema=manifest['schema']))
        self.evidence['source_archives']=cases

    def test_generated_shell_arguments_match_each_real_transport_parser(self):
        for variant in ('baseline','cleanup'):
            author=self.root/variant/'author';upload=author/'owner-upload'
            binding=prepare.parse_json((author/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
            script=(upload/'RUN-ON-WORKSTATION.sh').read_text()
            command=script.split('set +e\n',1)[1].split('\nbenchmark_rc=',1)[0]
            argv=shlex.split(command.replace('\\\n',''))
            self.assertNotIn('+',argv)
            transport=caller.checked_module(upload/'review','benchmark_host_transport',binding['upload_inventory']['review/benchmark_host_transport.py'])
            with patch.object(transport,'human_main',return_value=0) as human:
                self.assertEqual(transport.main(argv[7:]),0)
            args=human.call_args.args[0]
            self.assertEqual(args.archive_sha256,binding['archive_sha256'])
            if variant=='cleanup':
                self.assertEqual(args.run_config_sha256,binding['run_config_sha256'])
                self.assertEqual(args.protected_set_sha256,binding['protected_set_sha256'])
                self.assertEqual(args.approved_boot_id,'$approved_boot')
        self.evidence['real_transport_cli_parsers_passed']=True

    def test_actual_caller_source_context_accepts_only_a_separate_boot_binding(self):
        for variant in ('baseline','cleanup'):
            campaign=self.root/variant;author=campaign/'author'
            work=self.scratch/('context-'+variant);shutil.copytree(campaign/'caller',work)
            request=prepare.parse_json((campaign/'APPROVAL-REQUEST.json').read_bytes())
            template_raw=(author/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes();template=prepare.parse_json(template_raw)
            runtime=prepare.canonical(dict(template,boot_id=BOOT));(work/'LAUNCHER-BINDINGS.json').write_bytes(runtime)
            auth=dict(request,status='REGISTERED_CPU_FIXTURE',source_root=str(author),
                owner_invocations_allowed=1,model_starts_allowed=2,no_automatic_retry=True,
                owner_seconds=template['total_seconds'],caller_seconds=template['caller_seconds'],boot_id=BOOT,
                binding_template_sha256=prepare.digest(template_raw),binding_sha256=prepare.digest(runtime))
            raw=prepare.canonical(auth);(work/'RUN-AUTHORIZATION.json').write_bytes(raw)
            actual,_,source=caller.validate_context(work,prepare.digest(raw))
            self.assertEqual(actual,auth);self.assertEqual(source,author)
            self.assertFalse((campaign/'caller/RUN-AUTHORIZATION.json').exists())
        self.evidence['caller_contexts_verified']=True

    def test_both_actual_quality_templates_materialize_without_execution_authority(self):
        cases=[]
        for variant in ('baseline','cleanup'):
            template=self.root/variant/'author/quality-template'
            inventory=prepare.parse_json((template/'PROJECTION-INVENTORY.json').read_bytes())
            pin=prepare.digest((template/'PROJECTION-INVENTORY.json').read_bytes())
            binder=caller.checked_module(template,'bind_future',inventory['files']['bind_future.py']['sha256'])
            binder.authenticate_projection(template,pin)
            q=caller.checked_module(template/'source','quality_contract',inventory['files']['source/quality_contract.py']['sha256'])
            producer=caller.checked_module(template/'producer','final_quality_inputs',inventory['files']['producer/final_quality_inputs.py']['sha256'])
            menu=q.strict((template/'source/EXPECTED-QUALITY.json').read_bytes())
            value=producer.build_input(arm_results(menu),menu=menu);q.validate_input(value,menu)
            materializer=caller.checked_module(template,'prepare_packet',inventory['files']['prepare_packet.py']['sha256'])
            destination=self.scratch/('quality-'+variant)
            proof=materializer.materialize(q.encoded(value),destination,projection_inventory_sha256=pin,fixture=True)
            self.assertTrue(proof['fixture']);self.assertFalse(proof['execution_authorized'])
            self.assertEqual(proof['rows'],352)
            with self.assertRaisesRegex(ValueError,'full recomputed serving input binding'):
                binder.require_replay_binding(destination)
            cases.append(dict(variant=variant,rows=352,fixture=True,execution_authorized=False,
                              cpu_recipe_bytes=(destination/'REMOTE-RECIPE.py').stat().st_size))
        self.evidence['quality_template_materialization']=cases

    def test_review_contains_no_registered_authorization_and_both_requests_are_unfunded(self):
        for variant in ('baseline','cleanup'):
            root=self.root/variant
            self.assertFalse((root/'caller/RUN-AUTHORIZATION.json').exists())
            self.assertFalse((root/'caller/LAUNCHER-BINDINGS.json').exists())
            request=prepare.parse_json((root/'APPROVAL-REQUEST.json').read_bytes())
            self.assertEqual(request['maximum_starts'],0)
            self.assertFalse(request['model_start_authorized'])
            self.assertEqual(request['requested_model_starts'],2)
            self.assertEqual(request['requested_generations'],352)
        self.evidence.update(model_starts_executed=0,launch_authorized=False,registration_created=False)

    def test_actual_one_use_entrypoint_joins_sealed_sources_before_external_effects(self):
        for variant in ('baseline','cleanup'):
            campaign=self.root/variant;author=campaign/'author'
            work=self.scratch/('driver-'+variant);shutil.copytree(campaign/'caller',work)
            request=prepare.parse_json((campaign/'APPROVAL-REQUEST.json').read_bytes())
            pins=request['caller_inventory']
            sealer=caller.checked_module(work,'seal_evidence',pins['seal_evidence.py'])
            source_seal=sealer.verify(author,prepare.digest((author/'FINAL-ARTIFACTS.sha256').read_bytes()))
            record=dict(owner_answer='CPU fixture only; every external effect substituted',
                        approval_request_sha256=prepare.digest((work/'APPROVAL-REQUEST.json').read_bytes()),source_seal=source_seal)
            record_path=self.scratch/('CPU-FIXTURE-APPROVAL-'+variant+'.json');record_path.write_bytes(prepare.canonical(record))
            record_pin=dict(path=str(record_path),sha256=prepare.digest(record_path.read_bytes()))
            template_raw=(author/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes();template=prepare.parse_json(template_raw)
            runtime=prepare.canonical(dict(template,boot_id=BOOT));(work/'LAUNCHER-BINDINGS.json').write_bytes(runtime)
            grant=dict(request,status='REGISTERED_CPU_FIXTURE',source_root=str(author),source_seal=source_seal,
                registered=True,proposal=False,gpu_allowance=True,global_gpu_attempt_index=1,
                owner_answer=record['owner_answer'],approval_record=record_pin,registration_authority=record_pin,
                approval_request_sha256=record['approval_request_sha256'],
                owner_invocations_allowed=1,model_starts_allowed=2,no_automatic_retry=True,
                owner_seconds=template['total_seconds'],caller_seconds=template['caller_seconds'],boot_id=BOOT,
                generations=request['requested_generations'],cpu_grading_seconds=request['requested_cpu_grading_seconds'],
                binding_template_sha256=prepare.digest(template_raw),binding_sha256=prepare.digest(runtime))
            registration=prepare.canonical(grant);(work/'REGISTRATION.json').write_bytes(registration)
            authorization=prepare.canonical(dict(grant,registration_sha256=prepare.digest(registration)))
            (work/'RUN-AUTHORIZATION.json').write_bytes(authorization)
            driver=caller.checked_module(work,'execute_once',pins['execute_once.py'])
            with patch.object(driver,'run',return_value=dict(status='CPU_FIXTURE_NO_EXECUTION')) as dispatch:
                self.assertEqual(driver.main(['--authorization-sha256',prepare.digest(authorization)]),1)
                with self.assertRaises(FileExistsError):driver.main(['--authorization-sha256',prepare.digest(authorization)])
            dispatch.assert_called_once();self.assertTrue((work/'CALLER-INVOKED.json').is_file())
        self.evidence['one_use_entrypoint_external_dispatch_substituted']=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True);parser.add_argument('--boot-id',required=True)
    parser.add_argument('--report',type=Path)
    args,remaining=parser.parse_known_args();ROOT=args.preparation;BOOT=args.boot_id
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if args.report:
        report=dict(GPURunPreparation.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=prepare.digest(Path(__file__).read_bytes()))
        with args.report.open('xb') as stream:stream.write(prepare.canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
