"""Check session timing through the complete authenticated W5 CPU composition."""
import argparse
import ast
from datetime import datetime, timezone, timedelta
import difflib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_owner_timing_packet as owner_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_session_timing_proposal import PROPOSAL, CONTROLLER, TRANSPORT, canonical, sha
from test_source_builder import tree_bytes

inventory_checks = owner_checks.spec_checks.inventory_checks


class SessionTimingPacket(owner_checks.OwnerTimingPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        # Reuse the preceding authenticated packet assembler. Its old transport
        # location and required owner source are supplied by a temporary proposal
        # view; the actual files and pins are unchanged by this test adapter.
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-session-proposal-')
        cls.addClassCleanup(temporary.cleanup)
        view = Path(temporary.name)
        shutil.copytree(PROPOSAL, view, dirs_exist_ok=True)
        if not (view/'review/benchmark_owner_adapter.py').is_file():
            shutil.copyfile(HERE.parent / 'proposals/d2-owner-timing/review/benchmark_owner_adapter.py',
                            view / 'review/benchmark_owner_adapter.py')
        (view / 'integration/review').mkdir(parents=True)
        shutil.copyfile(PROPOSAL / TRANSPORT, view / 'integration' / TRANSPORT)
        with patch.object(owner_checks, 'PROPOSAL', view), patch.object(inventory_checks, 'BASELINE', view):
            super().setUpClass()
        cls.evidence.update(inherited_owner_checker_sha256=sha((HERE/'check_owner_timing_packet.py').read_bytes()),
            transport_sha256=sha(cls.members[TRANSPORT]), shared_runtime_timing_source=TRANSPORT)

    def test_exact_diff_and_unmodified_release_validators(self):
        baselines = {CONTROLLER: HERE.parent/'proposals/d2-owner-timing'/CONTROLLER,
                     TRANSPORT: HERE.parent/'proposals/d1-package-pruning/integration'/TRANSPORT}
        expected = ''.join(''.join(difflib.unified_diff(path.read_text().splitlines(True),
            (PROPOSAL/name).read_text().splitlines(True), fromfile='a/'+name,tofile='b/'+name))
            for name,path in baselines.items())
        self.assertEqual((PROPOSAL/'D2.patch').read_text(),expected)
        self.assertEqual(self.members['review/benchmark_owner_adapter.py'],
                         (HERE.parent/'proposals/d2-owner-timing/review/benchmark_owner_adapter.py').read_bytes())
        changed = {'validate_manifest','_run_bound_controller','verify_configured_spec',
                   'validate_physical_grant','seal_result','_run_physical_after_retained_closure','main'}
        stable = lambda raw: [ast.dump(n) for n in ast.parse(raw).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in changed]
        self.assertEqual(stable(baselines[CONTROLLER].read_bytes()),stable((PROPOSAL/CONTROLLER).read_bytes()))
        self.evidence.update(exact_patch_verified=True, owner_adapter_unchanged=True,
                             release_validators_unchanged=True)

    def test_archive_stages_and_grant_authenticates_at_the_unchanged_80_member_ceiling(self):
        transport = load('cleanup_session_archive_transport', self.package/TRANSPORT)
        archive = self.archive('configured.tar', self.members)
        manifest, payload = transport.inspect_archive(archive, sha(archive.read_bytes()),
            sha(self.members['MANIFEST.json']), configuration_pins=self.pins)
        self.assertEqual(payload, self.members)
        self.assertEqual(transport._archive_member_limit(manifest,payload),80)
        timing = transport.SessionTiming.from_limits(json.loads(payload[manifest['run_config']])['limits'])
        staged = self.root/'staged'
        staged.mkdir(); (staged/'review').mkdir()
        fd = os.open(staged,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            transport.stage_members(fd,payload)
        finally:
            os.close(fd)
        self.addCleanup(lambda:[p.chmod(0o755) for p in (staged/'config').rglob('*') if p.is_dir()])
        self.addCleanup((staged/'config').chmod,0o755)
        self.assertEqual(tree_bytes(staged),payload)
        with self.controller(staged) as (controller,path,digest,_,_):
            value = json.loads(path.read_bytes())
            parent = dict(pid=100,uid=0,start_ticks=200,boot_id=value['boot_id'])
            grant = transport.new_grant(value,digest,received_utc='2026-01-01T00:00:00Z',
                received_monotonic=100,parent=parent,configuration_pins=self.pins,timing=timing)
            utc = lambda:datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(seconds=1)
            self.assertEqual(controller.validate_physical_grant(grant,value,digest,parent,
                clock=lambda:101,utc_now=utc),grant)
            for key,bad in (('cumulative_ceiling_seconds',6359),('cumulative_ceiling_seconds',6361),
                            ('cumulative_ceiling_seconds',6360.0),('fixed_reserve_seconds',299),
                            ('fixed_reserve_seconds',300.0),('received_monotonic',-5)):
                with self.subTest(key=key,value=bad),self.assertRaises(ValueError):
                    controller.validate_physical_grant(dict(grant,**{key:bad}),value,digest,parent,
                        clock=lambda:101,utc_now=utc)
            with self.assertRaisesRegex(ValueError,'receipt window expired'):
                controller.validate_physical_grant(grant,value,digest,parent,clock=lambda:205,utc_now=utc)
        archive = self.archive('overflow.tar',dict(self.members,**{'review/extra.py':b'# inert\n'}))
        with self.assertRaisesRegex(ValueError,'archive expanded size/member limit'):
            transport.inspect_archive(archive,sha(archive.read_bytes()),sha(self.members['MANIFEST.json']),
                                      configuration_pins=self.pins)
        self.evidence['archive'] = dict(members=80,ceiling=80,inspected=True,staged=True,
            synthetic_grant_accepted=True,member_81_refused=True,owner_ceilings_changed=False,
            configured_total_seconds=grant['cumulative_ceiling_seconds'],
            configured_seal_reserve_seconds=grant['fixed_reserve_seconds'],timing_grant_refusals=7)

    def test_controller_passes_verified_session_deadline_to_campaign(self):
        with self.controller() as (controller,path,digest,_,root):
            configured = controller.verify_configured_spec(self.pins,json.loads(path.read_bytes()),
                deadline=time.monotonic()+105)
            with self.owner_case(controller.adapter) as case:
                value = dict(json.loads(path.read_bytes()),target_root=str(case.root),
                             session_sha256=case.lease.session,boot_id=case.lease.boot)
                prepared = dict(configured,manifest=value,spec_path=str(root/'review/BENCHMARK-SPEC2.json'),
                    resource_path='inert-resource-fixture',resource_manifest=dict(
                        gate_path=str(case.script),gate_sha256=case.script_sha))
                grant = dict(schema=1,scope='inert-controller-cpu-composition',launch_authorized=False,
                    maximum_starts=0,proposed_complete_seconds=6360,manifest_sha256=digest)
                boundary = Mock(side_effect=RuntimeError('synthetic stop before lease acquisition'))
                effect = Mock()
                # Native pre-launch remains blocked at the external build proof.
                # Substitute that prepared result and scope argv, then observe
                # actual controller arguments without entering the campaign.
                with patch.object(controller,'validate_manifest',return_value=prepared) as validate, \
                        patch.object(controller.resources,'scope_argv',side_effect=lambda _,__,arm:
                            next(row.argv for row in case.arms if row.arm==arm)), \
                        patch.object(controller.adapter,'run_campaign',boundary):
                    report = controller._run_bound_controller(manifest_path=str(path),manifest_sha256=digest,
                        grant=grant,lease=case.lease,observer=effect,read_inner_result=effect,
                        read_resource_receipt=effect,resource_check=effect,streams={},supervisor=effect,
                        clock=lambda:1000)
                self.assertIn('synthetic stop before lease acquisition',report['failure'])
                self.assertEqual(validate.call_args.kwargs['deadline'],1105)
                self.assertEqual(boundary.call_args.kwargs['active_deadline'],7060)
                self.assertEqual(boundary.call_args.kwargs['timing'],configured['owner_timing'])
                self.assertEqual(report['proposed_complete_seconds'],6360)
                self.assertEqual(report['seal_reserve_seconds'],300)
                self.assertIsNone(case.lease.fd)
                effect.assert_not_called()
        self.evidence['controller_timing_arguments'] = dict(preparation_deadline=1105,active_deadline=7060,
            original_receipt=1000,verified_owner_plan=True,stopped_before_lease=True,
            substitutes=['prepared pre-launch result','scope argv','campaign endpoint'])

    def test_complete_controller_seals_with_verified_session_plan(self):
        with self.controller() as (controller,path,_,_,_):
            configured = controller.verify_configured_spec(self.pins,json.loads(path.read_bytes()),
                deadline=time.monotonic()+105)
            root = self.root/'configured-seal'
            root.mkdir(); (root/'evidence.json').write_bytes(b'{}\n')
            result = controller.seal_result(root,timing=configured['session_timing'],
                received_monotonic=100,received_utc='2026-01-01T00:00:00Z',clock=lambda:6459.5)
            self.assertEqual(result['charged_seconds'],6360)
            self.assertFalse(result['qualification'])
            self.assertTrue((root/'FINAL-ARTIFACTS.sha256').is_file())
            self.assertTrue(root.with_name(root.name+'.closeout.json').is_file())
        self.evidence['configured_controller_seal'] = dict(verified_plan=True,original_receipt=100,
            seal_deadline=6460,charged_seconds=6360,exclusive_seal_and_closeout_written=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining = parser.parse_known_args()
    inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report is not None:
        report = dict(SessionTimingPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
