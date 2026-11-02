"""Configured worker and client budgets through complete CPU runtime modules."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_entrypoint_cleanup as entry_tests
import test_source_cleanup as runtime_tests
from test_contract_cleanup import SOURCE, load
from test_config_binding_proposal import canonical, sha

PROPOSAL = HERE.parent / 'proposals/d2-worker-timing'
CONTROLLER = 'review/benchmark_physical_controller.py'


def configured_spec(client=1940, worker=2180):
    spec = entry_tests.http_spec()
    spec['http_serving']['client_seconds'] = client
    spec['worker_seconds'] = worker
    return spec


class WorkerEntryTiming(unittest.TestCase):
    def setUp(self):
        self.entry = entry_tests.HttpEntry()
        self.entry.setUp()
        self.addCleanup(self.entry.doCleanups)

    def test_entry_uses_validated_worker_budget_for_both_arms(self):
        for arm in ('bf16', 'kv4'):
            for client, worker in ((2040, 2280), (1940, 2180), (1, 241)):
                with self.subTest(arm=arm, worker=worker):
                    result = self.entry.run_entry(spec=configured_spec(client, worker), arm=arm)
                    self.assertEqual(result['termination'], ('SystemExit', 0))
                    self.assertEqual(result['events'][0][3], 100 + worker)
                    self.assertEqual([row[0] for row in result['events']], ['execute', 'pack', 'stage', 'export'])

    def test_invalid_timing_refuses_before_helpers_scratch_or_output(self):
        cases = [(True, 2280), (2040.0, 2280), (0, 240), (2041, 2280),
                 (2040, 2279), (2040, 2281), (2040, True), (2040, 2280.0), (1, 240)]
        for client, worker in cases:
            with self.subTest(client=client, worker=worker):
                result = self.entry.run_entry(spec=configured_spec(client, worker))
                self.assertEqual(result['termination'][0], 'ValueError')
                self.assertEqual(result['initialized'], [])
                self.assertIsNone(result['files'])
        spec = configured_spec()
        del spec['worker_seconds']
        result = self.entry.run_entry(spec=spec)
        self.assertEqual(result['termination'][0], 'ValueError')
        self.assertEqual(result['initialized'], [])

    def test_worker_budget_is_snapshotted_before_runtime_import(self):
        module = self.entry.model
        original = module.validate_spec
        def validate(spec):
            result = original(spec)
            adjacent = module.authenticated_adjacent.side_effect
            def changed(*args):
                spec['worker_seconds'] = 4000
                return adjacent(*args)
            module.authenticated_adjacent.side_effect = changed
            return result
        # Change the mapping at the later artifact-helper boundary.
        with patch.object(module,'validate_spec',side_effect=validate):
            result = self.entry.run_entry(spec=configured_spec())
        self.assertEqual(result['events'][0][3], 2280)
        self.assertEqual(result['events'][0][2]['worker_seconds'],4000)


class RuntimeTiming(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime_tests.OrdinaryRuntime()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime = self.fixture.runtime
        self.root = self.fixture.root
        self.config = self.fixture.config
        self.config.settings = lambda: {'phase': 'sprint01-paired-serving'}
        self.config.FAIL_FAST_PHASES = ()
        self.config.precision_reference_enabled = lambda: False
        self.config.owner_arm_map = lambda: {'bf16': 'control', 'kv4': 'mtp'}
        self.config.phase_config = Mock(side_effect=AssertionError('phase duration used'))
        self.args = SimpleNamespace(arm='kv4', session_sha256='2'*64, spec_sha256='3'*64)

    def test_callback_uses_configured_client_budget_and_clamps_to_shared_deadline(self):
        for ready, deadline, expected in ((100, 2380, 2040), (500, 2280, 2280)):
            with self.subTest(ready=ready):
                # Each callback writes one exclusive readiness record.
                path = self.root / str(ready)
                path.mkdir()
                material = dict(self.fixture.material, native_dir=path)
                client = SimpleNamespace(run=Mock(return_value={'status':'complete'}))
                factory = Mock(return_value=client)
                with patch.object(self.runtime.time, 'monotonic', return_value=ready), \
                        patch.object(self.runtime, '_hip_sampling_ready'):
                    callback = self.runtime._client_callback(material, deadline=deadline, client_seconds=1940,
                                                            client_factory=factory, holder={})
                    callback({})
                self.assertEqual(factory.call_args.kwargs['absolute_deadline'], expected)
                factory.assert_called_once()
                client.run.assert_called_once()

    def test_readiness_cannot_extend_an_expired_worker_deadline(self):
        now = [100]
        factory = Mock()
        def delayed(*args):
            now[0] = 200
        with patch.object(self.runtime.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(self.runtime, '_hip_sampling_ready', side_effect=delayed):
            callback = self.runtime._client_callback(self.fixture.material, deadline=200, client_seconds=1940,
                                                    client_factory=factory, holder={})
            with self.assertRaisesRegex(ValueError, 'elapsed during native readiness'):
                callback({})
        factory.assert_not_called()
        self.assertFalse((self.root/'NATIVE-READINESS-CALLBACK.json').exists())

    def test_invalid_callback_duration_refuses_before_client_creation(self):
        for seconds in (None, True, 0, 2041, 1940.0):
            factory = Mock()
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.runtime._client_callback(self.fixture.material, deadline=2280, client_seconds=seconds,
                                              client_factory=factory, holder={})
            factory.assert_not_called()

    def test_execute_checks_worker_reserve_and_absolute_deadline_before_materialization(self):
        for client, worker, deadline in ((2040, 2279, 2000), (1940, 2180, 2281),
                                         (1940, True, 2000), (1940, 2180, 100)):
            with self.subTest(client=client, worker=worker, deadline=deadline), \
                    patch.object(self.runtime.time, 'monotonic', return_value=100), \
                    patch.object(self.runtime, '_read_template') as template, \
                    patch.object(self.runtime, '_materialize') as material, self.assertRaises(ValueError):
                self.runtime.execute(self.args, configured_spec(client,worker), self.root,
                                     deadline=deadline, artifacts=object())
            template.assert_not_called()
            material.assert_not_called()

    def test_complete_execute_passes_shorter_budget_and_preserves_result_schema(self):
        spec = configured_spec()
        template = {'source_manifest_sha256':'1'*64}
        summary = {'status':'complete'}
        client = SimpleNamespace(run=Mock(return_value=summary))
        factory = Mock(spec=[], return_value=client)
        (self.root/'paired').mkdir()
        (self.root/'paired/RUN-SUMMARY.json').write_bytes(b'{"paired":true}\n')
        def entry(binding, *, client):
            return client({'gate':'ready'})
        def materialize(*args, **kwargs):
            # A downstream preparation callback must not replace the already
            # validated scalar passed to the client callback.
            spec['http_serving']['client_seconds'] = 2040
            return self.fixture.material
        with patch.object(self.runtime.time, 'monotonic', return_value=100), \
                patch.object(self.runtime, '_read_template', return_value=(template,'4'*64)), \
                patch.object(self.runtime, '_materialize', side_effect=materialize) as material, \
                patch.object(self.runtime, '_hip_sampling_ready'):
            result = self.runtime.execute(self.args, spec, self.root, deadline=2280, artifacts=object(),
                                          client_factory=factory, entry_runner=entry)
        self.assertEqual(material.call_args.kwargs['deadline'],2280)
        self.assertEqual(factory.call_args.kwargs['absolute_deadline'],2040)
        self.assertEqual(result['status'],'HTTP_ARM_COMPLETE')
        self.assertEqual(set(result), {'schema','mode','status','arm','logical_arm','session_sha256',
            'spec_sha256','source_manifest_sha256','client_summary','native_gate','parent_error','qualification'})
        self.assertEqual(json.loads((self.root/'result.json').read_bytes()),result)
        self.assertFalse(result['qualification'])
        self.config.phase_config.assert_not_called()

    def test_actual_client_socket_timeout_uses_callback_absolute_deadline(self):
        with patch.dict(sys.modules,serving_contract=SimpleNamespace()):
            client_module = load('worker_timing_client',SOURCE/'serving_client.py')
        now = [2039.5]
        client = client_module.ServingClient(port=18088,output_root=self.root,absolute_deadline=2040,
            expected_gate={},live_worker_checker=Mock(),clock=lambda:now[0])
        connection = SimpleNamespace(sock=Mock())
        client._set_socket_timeout(connection)
        connection.sock.settimeout.assert_called_once_with(.5)
        now[0] = 2040
        with self.assertRaisesRegex(ValueError,'absolute client deadline elapsed'):
            client._set_socket_timeout(connection)
        self.assertEqual(connection.sock.settimeout.call_count,1)


class ProposalIntegrity(unittest.TestCase):
    def test_controller_only_replaces_the_active_phase_duration_binding(self):
        before = (HERE.parent/'proposals/d2-session-timing'/CONTROLLER).read_text()
        after = (PROPOSAL/CONTROLLER).read_text()
        expected = before.replace('context, seconds = mtp_contract.phase_config()[:2]',
                                  'context = mtp_contract.phase_config()[0]').replace(
            "http.get('client_seconds') == seconds", "http.get('client_seconds') == session_timing.client_seconds")
        self.assertEqual(after,expected)
        self.assertIn('workers["worker_seconds"] == 2280',after)  # Retained legacy-only diagnostic path.

    def test_pins_and_separate_source_patch_are_exact(self):
        record = json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        self.assertEqual(sha((PROPOSAL/'D2.patch').read_bytes()),record['patch_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),record['source_patch_sha256'])
        self.assertEqual(sha((PROPOSAL/CONTROLLER).read_bytes()),record['proposed_sha256'])
        historical={}
        if any(sha((HERE.parent/name).read_bytes())!=row['proposed_sha256']
               for name,row in record['source_changes'].items()):
            successor=HERE.parent/'proposals/d2-artifact-binding'
            next_record=json.loads((successor/'PROPOSAL.json').read_bytes())
            stages=[successor]
            if any(sha((HERE.parent/name).read_bytes())!=row['proposed_sha256']
                   for name,row in next_record['source_changes'].items()):
                stages.insert(0,HERE.parent/'proposals/production-imports')
                imports_record=json.loads((stages[0]/'PROPOSAL.json').read_bytes())
                if sha((HERE.parent/'build.py').read_bytes())!=imports_record['source_changes']['build.py']['proposed_sha256']:
                    stages.insert(0,HERE.parent/'proposals/production-isolation')
                    # Reverse the current, independently pinned source change
                    # before checking the retained production-isolation stage.
                    stages.insert(0,HERE.parent/'proposals/configuration-mutations')
                    stages.insert(0,HERE.parent/'proposals/runtime-binding')
                    stages.insert(0,HERE.parent/'proposals/common-helpers')
                    stages.insert(0,HERE.parent/'proposals/closed-http')
                    stages.insert(0,HERE.parent/'proposals/post-load')
                    stages.insert(0,HERE.parent/'proposals/source-references')
                    stages.insert(0,HERE.parent/'proposals/pre-replay')
                    stages.insert(0,HERE.parent/'proposals/shared-memory')
                    stages.insert(0,HERE.parent/'proposals/phase-closure')
                    stages.insert(0,HERE.parent/'proposals/common-closure')
            records=[json.loads((stage/'PROPOSAL.json').read_bytes()) for stage in stages]
            with tempfile.TemporaryDirectory(prefix='worker-source-history-') as temporary:
                root=Path(temporary).resolve()
                for name in {name for value in records for name in value['source_changes']}:
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes((HERE.parent/name).read_bytes())
                for stage,value in zip(stages,records,strict=True):
                    raw=(stage/'SOURCE.patch').read_bytes()
                    self.assertEqual(sha(raw),value['source_patch_sha256'])
                    changed=value.get('direct_source_changes',value['source_changes'])
                    for name,row in changed.items():
                        self.assertEqual(sha((root/name).read_bytes()),row['proposed_sha256'])
                    result=subprocess.run(['/usr/bin/patch','--batch','--fuzz=0','-R','-p1'],
                        input=raw,cwd=root,capture_output=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stdout.decode())
                    for name,row in changed.items():
                        self.assertEqual(sha((root/name).read_bytes()),row['baseline_sha256'])
                historical={name:(root/name).read_bytes() for name in record['source_changes']}
            for name,row in record['source_changes'].items():
                self.assertEqual(next_record['source_changes'][name]['baseline_sha256'],row['proposed_sha256'])
        for name,row in record['source_changes'].items():
            self.assertEqual(sha(historical.get(name,(HERE.parent/name).read_bytes())),row['proposed_sha256'])
        self.assertEqual(sha(canonical({'review/'+name:pin for name,pin in record['protected_files'].items()})),
                         record['protected_set_sha256'])
        for key in ('owner_ceilings_changed','whole_run_config_mutations_supported','consumed_by_builder',
                    'owner_approved','launch_authorized','entire_D2_complete'):
            self.assertFalse(record[key])


if __name__ == '__main__':
    unittest.main()
