"""Exercise the real entry point with CPU substitutes for runtime and export."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import SOURCE, canonical, load


MODE = 'native-http-fp8-sprint01-v1'


def http_spec():
    return dict(experiment_mode=MODE, worker_seconds=2280, http_serving=dict(schema=1, mode=MODE,
        client_seconds=2040,
        bindings={arm: dict(path=f'review/HTTP-BINDING-{arm}.json', sha256='a' * 64)
                  for arm in ('control', 'mtp')},
        expected_devices={str(index): dict(uuid=str(index), render=str(index))
                          for index in range(2)}))


def exercise_main(module, root, *, spec=None, arm='bf16', outcome='complete',
                  raw_transport=False, edit_args=None, export_failure=False,
                  existing_output=False, artifact_policy_class=None, mutate_runtime_spec=None):
    """Return entry-point behavior; neither native work nor real mounts are used."""
    spec = http_spec() if spec is None else spec
    root.mkdir()
    spec_path = root / 'spec.json'
    spec_path.write_bytes(canonical(spec))
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    output = root / 'result'
    if existing_output:
        output.mkdir()
        (output / 'sentinel').write_bytes(b'existing')
    argv = ['model_benchmark.py', '--execute-reviewed-benchmark', '--spec', str(spec_path),
            '--spec-sha256', digest, '--output', str(output), '--arm', arm,
            '--physical-artifacts', '--session-sha256', 'b' * 64,
            '--artifacts-sha256', 'c' * 64]
    if edit_args:
        edit_args(argv)
    events, initialized, policies = [], [], []
    sink = io.BytesIO()
    stdout = SimpleNamespace(buffer=sink, flush=lambda: events.append(('flush',)))

    def execute(args, selected, path, *, deadline, artifacts):
        events.append(('execute', args.arm, selected, deadline, path.stat().st_mode & 0o777))
        result = dict(status='HTTP_ARM_COMPLETE' if outcome == 'complete' else 'HTTP_ARM_FAILED')
        if outcome != 'error_without_result':
            (path / 'result.json').write_bytes(canonical(result))
        if outcome.startswith('error'):
            raise RuntimeError('injected runtime failure')
        if outcome == 'interrupt':
            raise KeyboardInterrupt('injected interrupt')
        return result

    def pack(path, binding, **kwargs):
        policies.append(('pack',kwargs.get('limits')))
        events.append(('pack', copy.deepcopy(binding)))

    def stage(path, binding, **kwargs):
        policies.append(('stage',kwargs.get('limits')))
        events.append(('stage', copy.deepcopy(binding)))
        return (path / 'result.json').read_bytes()

    def export(value, *, binding, sink, **kwargs):
        policies.append(('export',kwargs.get('limits')))
        events.append(('export', copy.deepcopy(binding)))
        if export_failure:
            raise OSError('injected export failure')
        sink.write(value)

    def raw_export(path, *, binding, sink):
        export((path / 'result.json').read_bytes(), binding=binding, sink=sink)

    artifacts = SimpleNamespace(pack_http_artifacts=pack, stage_artifacts=stage,
        export_bundle=export, export_raw_frames=raw_export, RAW_TRANSPORT='test-raw')
    if artifact_policy_class is not None:
        artifacts.HTTPArtifactLimits = artifact_policy_class
    runtime = SimpleNamespace(execute=execute, mtp_contract=SimpleNamespace(
        raw_attention_capture_enabled=lambda: raw_transport))

    def adjacent(filename, expected):
        initialized.append(('artifacts', filename, expected))
        return artifacts

    def runtime_module(selected, selected_arm):
        if mutate_runtime_spec is not None:
            mutate_runtime_spec(selected)
        initialized.append(('runtime', selected_arm))
        return runtime

    with patch.object(module, 'authenticated_adjacent', side_effect=adjacent), \
         patch.object(module, 'http_runtime_module', side_effect=runtime_module), \
         patch.object(module, 'prepare_serving_scratch_directories',
                      side_effect=lambda: initialized.append(('scratch',))), \
         patch.object(module.time, 'monotonic', return_value=100.0), \
         patch.object(sys, 'argv', argv), patch.object(sys, 'stdout', stdout):
        try:
            module.main()
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            termination = (type(error).__name__, error.code if isinstance(error, SystemExit) else str(error))
        else:
            termination = ('returned', None)
    files = {p.name: p.read_bytes() for p in output.iterdir()} if output.exists() else None
    return dict(termination=termination, events=events, initialized=initialized,
                exported=sink.getvalue(), files=files, artifact_policies=policies)


class HttpEntry(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-entry-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = load('cleanup_http_entry', SOURCE / 'model_benchmark.py')
        self.sequence = 0

    def run_entry(self, **kwargs):
        self.sequence += 1
        return exercise_main(self.model, self.root / str(self.sequence), **kwargs)

    def test_both_arms_preserve_deadline_binding_permissions_and_export_order(self):
        for arm in ('bf16', 'kv4'):
            result = self.run_entry(arm=arm)
            self.assertEqual(result['termination'], ('SystemExit', 0))
            self.assertEqual([row[0] for row in result['events']], ['execute', 'pack', 'stage', 'export'])
            self.assertEqual(result['events'][0][1:], (arm, http_spec(), 2380.0, 0o700))
            self.assertEqual(result['events'][-1][1], dict(arm=arm, experiment_mode=MODE,
                session_sha256='b' * 64, spec_sha256=hashlib.sha256(canonical(http_spec())).hexdigest()))
            self.assertEqual(result['exported'], result['files']['result.json'])

    def test_failed_and_interrupted_runs_export_existing_result_and_preserve_failure(self):
        for outcome, termination in (
            ('failed', ('SystemExit', 1)),
            ('error_with_result', ('RuntimeError', 'injected runtime failure')),
            ('interrupt', ('KeyboardInterrupt', 'injected interrupt'))):
            result = self.run_entry(outcome=outcome)
            self.assertEqual(result['termination'], termination)
            self.assertEqual(result['exported'], result['files']['result.json'])
            self.assertEqual(sum(event[0] == 'execute' for event in result['events']), 1)

    def test_missing_result_is_not_exported(self):
        result = self.run_entry(outcome='error_without_result')
        self.assertEqual(result['termination'][0], 'RuntimeError')
        self.assertEqual([row[0] for row in result['events']], ['execute'])
        self.assertEqual(result['exported'], b'')

    def test_export_failure_cannot_report_success(self):
        result = self.run_entry(export_failure=True)
        self.assertEqual(result['termination'], ('OSError', 'injected export failure'))

    def test_compatibility_raw_transport_preserves_flush_and_binding(self):
        result = self.run_entry(raw_transport=True)
        self.assertEqual(result['termination'], ('SystemExit', 0))
        self.assertEqual([row[0] for row in result['events']], ['execute', 'pack', 'flush', 'export'])
        self.assertEqual(result['events'][-1][1]['artifact_transport'], 'test-raw')

    def test_retired_and_unknown_modes_refuse_before_helpers_and_output(self):
        for mode in ('independent-bf16-reference-v1', 'native-model-context-retrieval-32k-v1',
                     'v4-repeatability-control-v1', 'native-decode-layer-capture-v1', 'unknown', None):
            spec = http_spec()
            spec['experiment_mode'] = mode
            result = self.run_entry(spec=spec)
            self.assertIn('only the W5', result['termination'][1])
            self.assertEqual(result['initialized'], [])
            self.assertIsNone(result['files'])

    def test_malformed_http_spec_refuses_before_helpers_and_output(self):
        for field in ('schema', 'mode', 'bindings', 'client_seconds', 'expected_devices'):
            spec = http_spec()
            spec['http_serving'].pop(field)
            result = self.run_entry(spec=spec)
            self.assertEqual(result['termination'][0], 'ValueError')
            self.assertEqual(result['initialized'], [])
            self.assertIsNone(result['files'])

    def test_execution_and_physical_flags_remain_mandatory(self):
        def remove_pair(argv, key):
            index = argv.index(key)
            del argv[index:index + 2]
        variants = (
            lambda args: args.remove('--execute-reviewed-benchmark'),
            lambda args: remove_pair(args, '--arm'),
            lambda args: args.remove('--physical-artifacts'),
            lambda args: remove_pair(args, '--session-sha256'),
            lambda args: args.append('--execute-serving-screen'),
        )
        for edit in variants:
            result = self.run_entry(edit_args=edit)
            self.assertEqual(result['initialized'], [])
            self.assertIsNone(result['files'])
            self.assertNotEqual(result['termination'], ('SystemExit', 0))

    def test_changed_spec_pin_refuses_before_helpers(self):
        def corrupt(args):
            args[args.index('--spec-sha256') + 1] = '0' * 64
        result = self.run_entry(edit_args=corrupt)
        self.assertEqual(result['termination'], ('ValueError', 'input digest'))
        self.assertEqual(result['initialized'], [])

    def test_existing_output_is_preserved_without_runtime_dispatch(self):
        result = self.run_entry(existing_output=True)
        self.assertEqual(result['termination'][0], 'FileExistsError')
        self.assertEqual(result['files'], {'sentinel': b'existing'})
        self.assertEqual(result['events'], [])


if __name__ == '__main__':
    unittest.main()
