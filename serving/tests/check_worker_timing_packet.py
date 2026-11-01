"""Check configured worker/client timing in the complete authenticated CPU package."""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_session_timing_packet as session_checks
from check_protected_packet import isolated_modules
from test_worker_timing import PROPOSAL, CONTROLLER
from test_config_binding_proposal import canonical, sha
from test_source_builder import builder


class WorkerTimingPacket(session_checks.SessionTimingPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-worker-proposal-')
        cls.addClassCleanup(temporary.cleanup)
        view = Path(temporary.name)
        shutil.copytree(HERE.parent/'proposals/d2-session-timing',view,dirs_exist_ok=True)
        shutil.copyfile(PROPOSAL/CONTROLLER,view/CONTROLLER)
        shutil.copyfile(PROPOSAL/'PROPOSAL.json',view/'PROPOSAL.json')
        transport = 'review/benchmark_host_transport.py'
        if (PROPOSAL/transport).is_file():
            shutil.copyfile(PROPOSAL/transport,view/transport)
        owner='review/benchmark_owner_adapter.py'
        if (PROPOSAL/owner).is_file():
            shutil.copyfile(PROPOSAL/owner,view/owner)
        with patch.object(session_checks,'PROPOSAL',view):
            super().setUpClass()
        cls.evidence.update(inherited_session_checker_sha256=sha((HERE/'check_session_timing_packet.py').read_bytes()),
            worker_source_pins={name:row['proposed_sha256'] for name,row in cls.record['source_changes'].items()},
            performance_improvement_measured=False)

    def test_exact_diff_and_unmodified_release_validators(self):
        baseline=HERE.parent/'proposals/d2-session-timing'/CONTROLLER
        expected=''.join(difflib.unified_diff(baseline.read_text().splitlines(True),
            (PROPOSAL/CONTROLLER).read_text().splitlines(True),fromfile='a/'+CONTROLLER,tofile='b/'+CONTROLLER))
        self.assertEqual((PROPOSAL/'D2.patch').read_text(),expected)
        for name,row in self.record['source_changes'].items():
            source=HERE.parent/name
            self.assertEqual(sha(source.read_bytes()),row['proposed_sha256'])
            self.assertEqual(self.members['review/'+source.name],source.read_bytes())
        for name in ('benchmark_owner_adapter.py','benchmark_host_transport.py'):
            self.assertEqual(self.record['protected_files'][name],
                json.loads((HERE.parent/'proposals/d2-session-timing/PROPOSAL.json').read_bytes())['protected_files'][name])
        self.evidence.update(exact_patch_verified=True,owner_adapter_unchanged=True,
                             transport_unchanged=True,release_validators_unchanged=True)

    def test_actual_import_chain_consumes_shorter_timing_and_preserves_socket_deadline(self):
        decisions=[]
        with self.controller() as (controller,_,_,_,root):
            model=controller.adapter.readout.bench
            spec=json.loads((root/'review/BENCHMARK-SPEC2.json').read_bytes())
            projection=self.root/'worker-template-projection'
            projection.mkdir()
            # Project virtual paths in temporary JSON only. The actual edited
            # modules and their authenticated source hashes stay unchanged.
            for arm in ('control','mtp'):
                name=f'HTTP-BINDING-{arm}.json'
                value=json.loads((root/'review'/name).read_bytes())
                sources=value['binding_template']['sources']
                for row in sources.values():
                    if row['path'].startswith('/mnt/review/'):
                        row['path']=str(root/'review'/Path(row['path']).name)
                pin=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                value['source_manifest_sha256']=pin
                for key in ('binding_template','worker_template'):
                    value[key]['source_manifest_sha256']=pin
                value['worker_template']['sources']['source_manifest_sha256']=pin
                value['worker_template']['preweight']['graph_observer']=dict(sources['graph_observer'])
                raw=canonical(value)
                (projection/name).write_bytes(raw)
                spec['http_serving']['bindings'][arm]['sha256']=sha(raw)
            checked=model.checked_file
            def template(path,expected,maximum=8<<20):
                if str(path) in ('/mnt/review/HTTP-BINDING-control.json','/mnt/review/HTTP-BINDING-mtp.json'):
                    path=projection/Path(path).name
                return checked(path,expected,maximum)
            for arm in ('bf16','kv4'):
                with isolated_modules(),patch.object(model,'checked_file',side_effect=template):
                    for name in builder.HTTP_IMPORTS:
                        sys.modules.pop(name,None)
                    runtime=model.http_runtime_module(spec,arm)
                    self.assertEqual(Path(runtime.__file__),root/'review/serving_runtime.py')
                    self.assertEqual(runtime.mtp_contract.instrumentation_enabled(),False)
                    logical=runtime.mtp_contract.owner_arm_map()[arm]
                    value=runtime.mtp_contract.expand_http_template(json.loads((projection/f'HTTP-BINDING-{logical}.json').read_bytes()))
                    for client_seconds,worker_seconds in ((2040,2280),(1940,2180)):
                        case=copy.deepcopy(spec)
                        case['worker_seconds']=worker_seconds
                        case['http_serving']['client_seconds']=client_seconds
                        self.assertIs(model.validate_spec(case),case)
                        runtime.validate_http_specification(case['http_serving'])
                        for ready in (100,500):
                            destination=self.root/f'callback-{arm}-{client_seconds}-{ready}'
                            destination.mkdir()
                            binding=dict(value['binding_template'],session_sha256='b'*64,spec_sha256=sha(canonical(case)))
                            material=dict(entry_binding=binding,worker_template=value['worker_template'],
                                          native_dir=destination,client_dir=destination/'client')
                            now=[ready]
                            recorded={}
                            def factory(**kwargs):
                                recorded.update(kwargs)
                                client=runtime.serving_client.ServingClient(**kwargs,clock=lambda:now[0])
                                connection=SimpleNamespace(sock=Mock())
                                now[0]=kwargs['absolute_deadline']-.5
                                client._set_socket_timeout(connection)
                                connection.sock.settimeout.assert_called_once_with(.5)
                                now[0]=kwargs['absolute_deadline']
                                with self.assertRaisesRegex(ValueError,'absolute client deadline elapsed'):
                                    client._set_socket_timeout(connection)
                                self.assertEqual(connection.sock.settimeout.call_count,1)
                                return SimpleNamespace(run=lambda *args:{'cpu_fixture':True})
                            with patch.object(runtime.time,'monotonic',side_effect=lambda:now[0]), \
                                    patch.object(runtime,'_hip_sampling_ready'):
                                runtime._client_callback(material,deadline=100+worker_seconds,
                                    client_seconds=client_seconds,client_factory=factory,holder={})({})
                            expected=min(100+worker_seconds,ready+client_seconds)
                            self.assertEqual(recorded['absolute_deadline'],expected)
                            decisions.append(dict(arm=arm,client_seconds=client_seconds,worker_seconds=worker_seconds,
                                ready=ready,client_deadline=expected,socket_expiry_refused=True))
        self.evidence['actual_worker_client_timing']=dict(cases=decisions,
            modules_imported_through_authenticated_chain=True,source_bytes_unchanged=True,
            substitutes=['virtual template paths','native HIP readiness','clock','HTTP request execution'],
            whole_run_config_mutations_supported=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args()
    session_checks.inventory_checks.PACKET=options.packet
    with isolated_modules():
        result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report is not None:
        report=dict(WorkerTimingPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
                    test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream: stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
