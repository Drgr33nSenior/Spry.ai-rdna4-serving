"""Source extraction and ordinary-client regressions with CPU-only seams."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_lifecycle_safety import SOURCE, integration, probe


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composed_fixture():
    """The retained 176-row wire shape, including its cancellation ordinal."""
    join = integration._kv_graph_join
    rows = [dict(_kv_ordinal=i, case=f'case-{i}', body={'input': i}) for i in range(176)]
    groups = [dict(id=f'group-{i}', ordinals=[i], concurrency=1) for i in range(176)]
    workers = [dict(rank=rank, pid=100 + rank) for rank in (0, 1)]
    wire = dict(arm='mtp', wire_complete=True, rows=[dict(
        status='CANCELLED_COMPLETE' if i == 48 else 'COMPLETE',
        spec_tokens_details=dict(spec_verify_ct=1)) for i in range(176)])
    observations = {}
    for i, row in enumerate(rows):
        for phase, offset in (('before', 0), ('after', 1)):
            ranks = [dict(worker, captures=2, backend_identity='backend', gdn_sha256='gdn',
                replays=i + offset, completed_commits=i + offset,
                bs1_replays=i + offset, bs2_replays=0) for worker in workers]
            observations[f'request-{i:02d}-{phase}.json'] = dict(
                phase=phase, ordinal=i, case=row['case'], arm='mtp',
                body_sha256=hashlib.sha256(join.canonical(row['body'])).hexdigest(),
                group_id=f'group-{i}', group_ordinals=[i], snapshot=dict(ranks=ranks),
                observed_monotonic=2 * i + offset)
    return observations, rows, wire, groups, workers


class ExtractedValidation(unittest.TestCase):
    def test_complete_composed_evidence_keeps_request_and_cancel_joins(self):
        observed, rows, wire, groups, workers = composed_fixture()
        join = integration._kv_graph_join
        def counts(delta, responses, concurrency):
            self.assertEqual(responses, [1])
            self.assertEqual(concurrency, 1)
            self.assertEqual(delta, dict(replays=1, completed_commits=1, bs1_replays=1, bs2_replays=0))
            return dict(delta, response_verifier_count=sum(responses))
        planned = join.planned_groups(rows, groups)
        self.assertEqual(planned, groups)
        result = join.validate(observed, rows, wire, planned, workers, paired_group_counts=counts)
        self.assertEqual(len(result), 176)
        self.assertIsNone(result[48]['ranks'][0]['response_verifier_count'])
        self.assertEqual(result[175]['ranks'][1]['response_verifier_count'], 1)
        self.assertFalse(join.canonical(result).endswith(b'\n'))

    def test_composed_evidence_rejects_body_identity_counter_and_membership_drift(self):
        join = integration._kv_graph_join
        def mutate_body(observed):
            observed['request-00-before.json']['body_sha256'] = '0' * 64
        def mutate_counter(observed):
            observed['request-01-before.json']['snapshot']['ranks'][0]['replays'] += 1
        def mutate_identity(observed):
            observed['request-00-after.json']['snapshot']['ranks'][0]['pid'] += 1
        def mutate_member(observed):
            del observed['request-175-after.json']
        for mutate in (mutate_body, mutate_counter, mutate_identity, mutate_member):
            with self.subTest(mutation=mutate.__name__):
                observed, rows, wire, groups, workers = composed_fixture()
                mutate(observed)
                with self.assertRaises(ValueError):
                    join.validate(observed, rows, wire, groups, workers,
                                  paired_group_counts=lambda delta, *args: delta)

    def test_metadata_preserves_dtype_payload_and_storage_alias_validation(self):
        metadata = probe._target_kv_metadata
        def tensor(dtype, pointer, shape, width):
            return dict(shape=shape, dtype=dtype, device='cuda:0', device_type='cuda',
                        pointer=pointer, elements=shape[0] * shape[1] * shape[2], element_bytes=width)
        row = dict(route='verify_decode', capture=True, layer=0,
            q=tensor('torch.bfloat16', 1, [8, 12, 256], 2),
            k_view=tensor('torch.float8_e4m3fn', 2, [16, 2, 256], 1),
            v_view=tensor('torch.float8_e4m3fn', 3, [16, 2, 256], 1),
            k_backing=tensor('torch.uint8', 2, [16, 2, 256], 1),
            v_backing=tensor('torch.uint8', 3, [16, 2, 256], 1),
            k_scale=1.0, v_scale=1.0, extra_tensor_operations=0, cpu_tensor_copies=0)
        self.assertIs(metadata.validate_route(row), row)
        for field, key, value in [('k_view', 'pointer', 99), ('q', 'dtype', 'torch.float32'),
                                  ('v_backing', 'elements', 1), ('q', 'device_type', 'cpu')]:
            bad = copy.deepcopy(row)
            bad[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                metadata.validate_route(bad)

    def test_imported_helpers_have_real_source_files(self):
        self.assertEqual(Path(probe._target_kv_metadata.__file__), SOURCE / 'target_kv_metadata.py')
        self.assertEqual(Path(integration._kv_graph_join.__file__), SOURCE / 'kv_graph_join.py')
        self.assertFalse(hasattr(probe, 'betterbench_owner'))
        self.assertFalse(hasattr(integration, 'betterbench_owner'))

    def test_probe_and_integration_have_no_embedded_source_literals(self):
        for name in ('j01_probe.py', 'serving_integration.py'):
            tree = ast.parse((SOURCE / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'compile':
                    self.assertFalse(isinstance(node.args[0], ast.Constant), name)


class SourceBinding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-module-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ('j01_probe.py', 'observer_health.py', 'target_kv_metadata.py',
                     'serving_integration.py', 'kv_graph_join.py'):
            shutil.copyfile(SOURCE / name, self.root / name)

    def test_changed_source_is_rejected_before_execution(self):
        for parent, helper in [('j01_probe.py', 'target_kv_metadata.py'),
                               ('serving_integration.py', 'kv_graph_join.py')]:
            with self.subTest(helper=helper):
                path = self.root / helper
                path.write_text('raise RuntimeError("untrusted source executed")\n')
                with self.assertRaisesRegex(ValueError, 'source binding'):
                    load('cleanup_changed_source', self.root / parent)
                shutil.copyfile(SOURCE / helper, path)

    def test_missing_or_linked_helper_is_rejected(self):
        for parent, helper in [('j01_probe.py', 'target_kv_metadata.py'),
                               ('serving_integration.py', 'kv_graph_join.py')]:
            with self.subTest(helper=helper):
                path = self.root / helper
                path.unlink()
                with self.assertRaises((OSError, ValueError)):
                    load('cleanup_missing_source', self.root / parent)
                path.symlink_to(SOURCE / helper)
                with self.assertRaises(ValueError):
                    load('cleanup_symlinked_source', self.root / parent)
                path.unlink()
                shutil.copyfile(SOURCE / helper, path)
                alias = path.with_suffix('.alias')
                os.link(path, alias)
                with self.assertRaises(ValueError):
                    load('cleanup_linked_source', self.root / parent)
                alias.unlink()

    def test_unchecked_cached_bytecode_cannot_replace_authenticated_source(self):
        for parent, helper in [('j01_probe.py', 'target_kv_metadata.py'),
                               ('serving_integration.py', 'kv_graph_join.py')]:
            with self.subTest(helper=helper):
                path = self.root / helper
                original = path.read_bytes()
                path.write_text('raise RuntimeError("cached code executed")\n')
                py_compile.compile(str(path), doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
                path.write_bytes(original)
                load('cleanup_cached_source', self.root / parent)

    def test_integration_import_does_not_load_probe_or_betterbench(self):
        (self.root / 'j01_probe.py').write_text('raise RuntimeError("probe imported eagerly")\n')
        module = load('cleanup_reader_only', self.root / 'serving_integration.py')
        self.assertTrue(callable(module.read_http_artifacts))


class OrdinaryRuntime(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(instrumentation_enabled=lambda: False,
            phase_config=lambda: (131040, 2040), settings=lambda: {})
        self.contract = SimpleNamespace(validate_kv_summary=Mock())
        seams = dict(mtp_contract=self.config, serving_contract=self.contract,
            serving_client=SimpleNamespace(ServingClient=Mock()),
            serving_entry=SimpleNamespace(run_bound=Mock()), serving_parity=SimpleNamespace())
        with patch.dict(sys.modules, seams):
            self.runtime = load('cleanup_runtime', SOURCE / 'serving_runtime.py')
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-client-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.binding = dict(arm='mtp', source_manifest_sha256='1' * 64,
            session_sha256='2' * 64, spec_sha256='3' * 64,
            canonical_server_cli=['--port', '18088'], sources={})
        self.material = dict(entry_binding=self.binding, native_dir=self.root, client_dir=self.root,
                             worker_template=dict(preweight=dict(graph_observer={})))

    def callback(self, factory, holder):
        return self.runtime._client_callback(self.material, deadline=time.monotonic() + 60,
                                             client_seconds=2040, client_factory=factory, holder=holder)

    def test_ordinary_client_receives_existing_gate_and_no_production_observer(self):
        summary = {'status': 'complete'}
        client = SimpleNamespace(run=Mock(return_value=summary))
        factory = Mock(return_value=client)
        holder = {}
        callback = self.callback(factory, holder)
        gate = {'gate': 'ready'}
        with patch.object(self.runtime, '_hip_sampling_ready') as ready:
            self.assertIs(callback(gate), summary)
        ready.assert_called_once_with(self.material, gate)
        self.assertIs(holder['summary'], summary)
        self.assertIsNone(factory.call_args.kwargs['request_observer'])
        self.assertEqual(factory.call_args.kwargs['port'], 18088)
        client.run.assert_called_once_with(gate, hashlib.sha256(self.runtime._canonical(gate)).hexdigest())
        self.assertEqual(json.loads((self.root / 'NATIVE-READINESS-CALLBACK.json').read_bytes()), gate)

    def test_instrumented_client_retains_request_observer_callback(self):
        self.config.instrumentation_enabled = lambda: True
        factory = Mock(return_value=SimpleNamespace(run=lambda *args: {}))
        with patch.object(self.runtime, '_hip_sampling_ready'):
            self.callback(factory, {})({})
        self.assertTrue(callable(factory.call_args.kwargs['request_observer']))

    def test_client_failure_is_preserved_without_retry(self):
        error = RuntimeError('client failed')
        client = SimpleNamespace(run=Mock(side_effect=error))
        factory = Mock(return_value=client)
        holder = {}
        with patch.object(self.runtime, '_hip_sampling_ready'), self.assertRaises(RuntimeError) as raised:
            self.callback(factory, holder)({})
        self.assertIs(raised.exception, error)
        factory.assert_called_once()
        client.run.assert_called_once()
        self.assertNotIn('summary', holder)

    def test_retired_client_is_rejected_before_template_or_native_work(self):
        def retired(**kwargs):
            self.fail('retired client executed')
        retired.betterbench_owner_mode = 'betterbench-selected-owner-checkpoint-v2'
        with patch.object(self.runtime, '_read_template') as template, \
             self.assertRaisesRegex(self.runtime.ServingRuntimeError, 'retired BetterBench'):
            self.runtime.execute(SimpleNamespace(arm='kv4'), {}, self.root,
                deadline=time.monotonic() + 60, artifacts=object(), client_factory=retired)
        template.assert_not_called()

    def test_summary_validation_keeps_raw_paired_evidence(self):
        summary, raw = {}, b'{"paired":true}\n'
        self.runtime._complete_normal_stop_summary(summary, paired_raw=raw)
        self.contract.validate_kv_summary.assert_called_once_with(summary, raw)

    def execute_ordinary(self, client):
        self.config.owner_arm_map = lambda: dict(kv4='mtp', bf16='control')
        args = SimpleNamespace(arm='kv4', session_sha256='2' * 64, spec_sha256='3' * 64)
        http = dict(bindings=dict(mtp={}), client_seconds=2040, expected_devices=[])
        template = dict(source_manifest_sha256='1' * 64)
        gate = {'gate': 'ready'}
        def factory(**kwargs):
            return client
        def entry(binding, *, client):
            self.assertIs(binding, self.binding)
            return client(gate)
        with patch.object(self.runtime, 'validate_http_specification'), \
             patch.object(self.runtime, '_read_template', return_value=(template, '4' * 64)), \
             patch.object(self.runtime, '_materialize', return_value=self.material), \
             patch.object(self.runtime, '_hip_sampling_ready'):
            return self.runtime.execute(args, dict(http_serving=http, worker_seconds=2280), self.root,
                deadline=time.monotonic() + 60, artifacts=object(),
                client_factory=factory, entry_runner=entry)

    def test_actual_execute_preserves_complete_result_and_paired_summary(self):
        raw = b'{"paired":true}\n'
        (self.root / 'paired').mkdir()
        (self.root / 'paired' / 'RUN-SUMMARY.json').write_bytes(raw)
        summary = {'status': 'complete'}
        client = SimpleNamespace(run=Mock(return_value=summary))
        result = self.execute_ordinary(client)
        self.assertEqual(result['status'], 'HTTP_ARM_COMPLETE')
        self.assertEqual(result['client_summary'], summary)
        self.assertEqual(result['native_gate'], {'gate': 'ready'})
        self.assertIsNone(result['parent_error'])
        self.assertEqual(json.loads((self.root / 'result.json').read_bytes()), result)
        self.contract.validate_kv_summary.assert_called_once_with(summary, raw)
        client.run.assert_called_once()

    def test_actual_execute_records_failure_and_preserves_exception(self):
        error = RuntimeError('client failed')
        client = SimpleNamespace(run=Mock(side_effect=error))
        with self.assertRaises(RuntimeError) as raised:
            self.execute_ordinary(client)
        self.assertIs(raised.exception, error)
        result = json.loads((self.root / 'result.json').read_bytes())
        self.assertEqual(result['status'], 'HTTP_ARM_FAILED')
        self.assertEqual(result['native_gate'], {'gate': 'ready'})
        self.assertIsNone(result['client_summary'])
        self.assertEqual(result['parent_error'], 'RuntimeError: client failed')
        client.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
