"""Verify configured artifact consumers in the authenticated CPU composition."""
import argparse
import difflib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import shutil
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_worker_timing_packet as worker_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha
from test_source_builder import builder
from test_artifact_limits import PROPOSAL


class ArtifactLimitsPacket(worker_checks.WorkerTimingPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        temporary=tempfile.TemporaryDirectory(prefix='artifact-proposal-')
        cls.addClassCleanup(temporary.cleanup)
        view=Path(temporary.name)
        shutil.copytree(HERE.parent/'proposals/d2-worker-timing',view,dirs_exist_ok=True)
        for name in ('PROPOSAL.json','review/benchmark_physical_controller.py'):
            shutil.copyfile(PROPOSAL/name,view/name)
        transport = 'review/benchmark_host_transport.py'
        if (PROPOSAL/transport).is_file():
            shutil.copyfile(PROPOSAL/transport,view/transport)
        owner='review/benchmark_owner_adapter.py'
        if (PROPOSAL/owner).is_file():
            shutil.copyfile(PROPOSAL/owner,view/owner)
        with patch.object(worker_checks,'PROPOSAL',view):
            super().setUpClass()
        review=cls.package/'review'
        original={p.name:p.read_bytes() for p in review.iterdir()}
        files=dict(original)
        files['benchmark_artifacts.py']=(PROPOSAL/'review/benchmark_artifacts.py').read_bytes()
        manifest=json.loads((cls.package/'MANIFEST.json').read_bytes())
        builder.render_resource_manifest(files,original,manifest)
        for name in ('benchmark_artifacts.py','RESOURCE-MANIFEST.json'):
            (review/name).write_bytes(files[name])
            manifest['package_files']['review/'+name]=sha(files[name])
        manifest['resource_manifest_sha256']=sha(files['RESOURCE-MANIFEST.json'])
        (cls.package/'MANIFEST.json').write_bytes(canonical(manifest))
        cls.members={name:(cls.package/name).read_bytes() for name in cls.members}
        cls.evidence.update(package_inventory=manifest['package_files'],
            inherited_worker_checker_sha256=sha((HERE/'check_worker_timing_packet.py').read_bytes()),
            artifact_helper_sha256=sha(files['benchmark_artifacts.py']),
            worker_export_config_binding_migrated=False)

    def test_exact_diff_and_unmodified_release_validators(self):
        baselines={'review/benchmark_artifacts.py':self.files['author/package-v2/review/benchmark_artifacts.py'],
            'review/benchmark_physical_controller.py':(HERE.parent/'proposals/d2-worker-timing/review/benchmark_physical_controller.py').read_bytes()}
        expected=''.join(''.join(difflib.unified_diff(before.decode().splitlines(True),
            (PROPOSAL/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
            for name,before in baselines.items())
        self.assertEqual(expected,(PROPOSAL/'D2.patch').read_text())
        for name,pin in self.record['protected_files'].items():
            self.assertEqual(sha(self.members['review/'+name]),pin)
        self.evidence.update(exact_patch_verified=True,release_validators_unchanged=True,
                             owner_adapter_unchanged=True,transport_unchanged=True)

    def test_verified_configuration_exposes_exact_artifact_primaries(self):
        import time
        with self.controller() as (controller,path,_,_,_):
            configured=controller.verify_configured_spec(self.pins,json.loads(path.read_bytes()),
                deadline=time.monotonic()+105)
            run=json.loads(self.members['config/runs/w5-v2/run.json'])
            self.assertEqual(configured['artifact_limits'],
                {name:run['limits'][name] for name in configured['artifact_limits']})
            self.assertEqual(len(configured['artifact_limits']),16)
        self.evidence['configured_artifact_primaries_verified']=16

    def check_postrun(self, shape):
        original=self.postrun.connection
        def synchronized_connection(respond):
            # The in-memory HTTP stand-in can otherwise finish one C2 request
            # before the OS schedules its peer. Hold each fake response until
            # both real client threads have entered getresponse. Keep the
            # production overlap validator and recorded clocks unchanged.
            contract=importlib.import_module('serving_contract')
            cases=contract.requests()
            groups=contract.mtp_contract.paired_group_specs()
            by_ordinal={ordinal:group['id'] for group in groups if group['concurrency']==2
                        for ordinal in group['ordinals']}
            rid_group={case['body']['rid']:by_ordinal[case['_kv_ordinal']]
                       for case in cases if case['_kv_ordinal'] in by_ordinal}
            barriers={name:threading.Barrier(2) for name in set(rid_group.values())}
            base=original(respond)
            class Connection(base):
                def getresponse(self):
                    group=rid_group.get(self.body['rid'])
                    if group is not None:
                        barriers[group].wait(timeout=30)
                    return super().getresponse()
            return Connection
        with patch.object(self.postrun,'connection',synchronized_connection):
            super().check_postrun(shape)
        self.evidence['postrun_'+shape]['http_fixture_synchronization']='C2 response barrier; real client clocks and overlap check unchanged'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    worker_checks.session_checks.inventory_checks.PACKET=options.packet
    with isolated_modules():
        result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(ArtifactLimitsPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
