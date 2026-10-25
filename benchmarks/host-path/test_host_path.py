"""Host-only source tests; no native runtime, GPU, network, or dependencies."""
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'author' / 'review'
# Exact method-source hashes from the authenticated selected-baseline observer.
# Source segments avoid Python-version-dependent AST representation hashes.
BASELINE_METHOD_SHA256 = {
    'write_counter': '2c3f7c14128cfbbf0032681f5c3215134cab3be6eab5971ac5bf5aa02c024b1c',
    'commit_complete': '03b78f220add9cbe0219a4c450ea86a856659f0faf087a10718a2f660d3082e2',
    'publish': 'acbefff303c04e0e3c37c59167ede3c917ae152d3e0d6bcb0195011bb8d8337a',
    'checked_backend': '10521c4397cfcd94b65db7ba6ffb1582c774a5ea256553dbc15ceb7b3719a877',
    'capture_forward': '6ee77f84eb424e251994519730465cdad7625c391ab143f3e9363f19e42b70c1',
    'backend_replayed': 'bfeab1f22042d68dd6fd16b52f19c25654a76b35b07b14ff5676109e0f75fb85',
}


def load_source(name, path):
    """Compile and execute the entire source, retaining a real source loader."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


contract = load_source('mtp_contract', SOURCE / 'mtp_contract.py')
graph = load_source('host_path_test_graph', SOURCE / 'serving_graph_observer.py')
entry = load_source('host_path_test_entry', SOURCE / 'serving_entry.py')


class HostPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='host-path-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        helper_path = self.root / 'completed_commit.py'
        helper_path.write_text('def commit_gdn_replayssm_fold_after_verify():\n    return None\n')
        name = 'host_path_completed_commit_' + self.root.name.replace('-', '_')
        self.helper_module = load_source(name, helper_path)
        self.addCleanup(sys.modules.pop, name, None)
        self.helper = self.helper_module.commit_gdn_replayssm_fold_after_verify
        self.sources = {role: {} for role in graph.GDN_ROLES}
        self.sources['replay_fold'] = dict(module=name, path=str(helper_path), sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest())
        self.hybrid = object()
        self.backend = SimpleNamespace(_cuda_graph_runner=SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=self.hybrid, is_draft_worker=False), raw_bs=1))

    def observer(self, arm='mtp', directory=None):
        directory = directory or self.root
        value = graph.Observer.__new__(graph.Observer)
        def close_maps():
            for name in ('counter', 'commit_counter'):
                mapping = getattr(value, name, None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
        self.addCleanup(close_maps)
        value.__init__(0, directory, '0' * 64, arm, sources=self.sources)
        value.target_backend = self.backend
        return value

    def counter_path(self):
        return self.root / 'replay-commit-counter-rank-0.bin'

    def commits(self):
        raw = self.counter_path().read_bytes()
        self.assertEqual(len(raw), 8)
        return struct.unpack('<Q', raw)[0]

    def test_exact_arm_and_checkpoint_selection(self):
        self.assertIs(contract.betterbench_checkpoint_enabled(), True)
        for enabled in (False, True):
            with patch.object(contract, 'betterbench_checkpoint_enabled', return_value=enabled):
                for arm in ('control', 'mtp'):
                    with self.subTest(checkpoint=enabled, arm=arm):
                        candidate = enabled and arm == 'mtp'
                        self.assertIs(contract.host_path_cleanup_enabled(arm), candidate)
                        self.assertEqual(contract.expected_decode_log_interval(arm), 100 if candidate else 1)
                for invalid in ('candidate', '', None, 1):
                    with self.assertRaises(ValueError):
                        contract.expected_decode_log_interval(invalid)

    def test_candidate_initialization_and_live_eight_byte_mapping(self):
        with patch.object(os, 'open', wraps=os.open) as opened, patch.object(os, 'ftruncate', wraps=os.ftruncate) as truncated:
            value = self.observer()
        commit_opens = [call for call in opened.call_args_list if call.args[0] == self.counter_path()]
        self.assertEqual(len(commit_opens), 1)
        self.assertEqual(commit_opens[0].args[1:], (os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
        self.assertEqual([call.args[1] for call in truncated.call_args_list], [32, 8])
        self.assertEqual(self.counter_path().stat().st_mode & 0o777, 0o600)
        self.assertIsInstance(value.commit_counter, mmap.mmap)
        self.assertEqual(len(value.commit_counter), 8)
        self.assertEqual(self.commits(), 0)
        with self.counter_path().open('rb') as source, mmap.mmap(source.fileno(), 8, access=mmap.ACCESS_READ) as reader:
            for expected in range(1, 5):
                value.commit_complete(self.hybrid, self.helper)
                self.assertEqual(struct.unpack_from('<Q', reader)[0], expected)
                self.assertEqual(self.commits(), expected)

    def test_first_completed_commit_authenticates_and_publishes_before_increment(self):
        for arm in ('control', 'mtp'):
            with self.subTest(arm=arm):
                directory = self.root / arm
                directory.mkdir()
                value = self.observer(arm, directory)
                observed = []
                original = value.publish
                def publish():
                    observed.append((value.commits, struct.unpack('<Q', (directory / 'replay-commit-counter-rank-0.bin').read_bytes())[0]))
                    original()
                with patch.object(value, 'publish', side_effect=publish) as published:
                    value.commit_complete(self.hybrid, self.helper)
                    value.commit_complete(self.hybrid, self.helper)
                self.assertEqual(observed, [(0, 0)])
                published.assert_called_once_with()
                self.assertEqual(value.commits, 2)
                record = json.loads((directory / 'graph-captures-rank-0.json').read_bytes())
                self.assertEqual(record['commit_source'], dict(self.sources['replay_fold'], name=self.helper.__name__))

    def test_candidate_steady_commits_have_no_filesystem_calls(self):
        value = self.observer()
        value.commit_complete(self.hybrid, self.helper)
        with ExitStack() as stack:
            for name in ('open', 'close', 'write', 'ftruncate', 'fsync', 'fdatasync'):
                if hasattr(os, name):
                    stack.enter_context(patch.object(os, name, side_effect=AssertionError('per-commit os.' + name)))
            for name in ('open', 'read_bytes', 'write_bytes'):
                stack.enter_context(patch.object(Path, name, side_effect=AssertionError('per-commit Path.' + name)))
            stack.enter_context(patch('builtins.open', side_effect=AssertionError('per-commit open')))
            for _ in range(4):
                value.commit_complete(self.hybrid, self.helper)
        self.assertEqual(self.commits(), 5)

    def test_control_and_non_checkpoint_keep_legacy_writes(self):
        for checkpoint, arm in ((True, 'control'), (False, 'control'), (False, 'mtp')):
            with self.subTest(checkpoint=checkpoint, arm=arm):
                directory = self.root / (str(checkpoint) + arm)
                directory.mkdir()
                path = directory / 'replay-commit-counter-rank-0.bin'
                path.write_bytes(b'previous counter content')
                with patch.object(contract, 'betterbench_checkpoint_enabled', return_value=checkpoint):
                    value = self.observer(arm, directory)
                self.assertIsNone(value.commit_counter)
                self.assertEqual(path.read_bytes(), struct.pack('<Q', 0))
                value.commit_complete(self.hybrid, self.helper)
                with patch.object(os, 'open', wraps=os.open) as opened, patch.object(os, 'write', wraps=os.write) as written, patch.object(os, 'close', wraps=os.close) as closed:
                    value.commit_complete(self.hybrid, self.helper)
                opened.assert_called_once_with(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                self.assertEqual(written.call_count, 1)
                self.assertEqual(written.call_args.args[1], struct.pack('<Q', 2))
                self.assertEqual(closed.call_count, 1)
                self.assertEqual(path.read_bytes(), struct.pack('<Q', 2))

    def test_candidate_refuses_existing_regular_counter(self):
        sentinel = b'preserve existing data'
        self.counter_path().write_bytes(sentinel)
        with self.assertRaises(FileExistsError):
            self.observer()
        self.assertEqual(self.counter_path().read_bytes(), sentinel)

    def test_both_arms_refuse_counter_symlink(self):
        target = self.root / 'untouched'
        target.write_bytes(b'preserve target')
        for arm in ('control', 'mtp'):
            with self.subTest(arm=arm):
                directory = self.root / arm
                directory.mkdir()
                link = directory / 'replay-commit-counter-rank-0.bin'
                link.symlink_to(target)
                with self.assertRaises(OSError):
                    self.observer(arm, directory)
                self.assertTrue(link.is_symlink())
                self.assertEqual(target.read_bytes(), b'preserve target')

    def test_finite_bound_fails_without_increment_for_both_arms(self):
        self.assertEqual(contract.commit_observation_limit(), 44 * 4096 + 9 * 16 + 4096)
        for arm in ('control', 'mtp'):
            with self.subTest(arm=arm):
                directory = self.root / arm
                directory.mkdir()
                value = self.observer(arm, directory)
                value.commit_complete(self.hybrid, self.helper)
                value.commits = value.commit_limit - 1
                value.write_commit_counter()
                value.commit_complete(self.hybrid, self.helper)
                with self.assertRaisesRegex(ValueError, 'finite commit observation count'):
                    value.commit_complete(self.hybrid, self.helper)
                self.assertEqual(value.commits, value.commit_limit)
                self.assertEqual((directory / 'replay-commit-counter-rank-0.bin').read_bytes(), struct.pack('<Q', value.commit_limit))

    def test_backend_helper_and_code_substitution_rejected_before_increment(self):
        for arm in ('control', 'mtp'):
            with self.subTest(arm=arm):
                directory = self.root / arm
                directory.mkdir()
                value = self.observer(arm, directory)
                with self.assertRaisesRegex(ValueError, 'backend identity'):
                    value.commit_complete(object(), self.helper)
                value.commit_complete(self.hybrid, self.helper)
                with self.assertRaisesRegex(ValueError, 'helper substituted'):
                    value.commit_complete(self.hybrid, lambda: None)
                original_code = self.helper.__code__
                try:
                    self.helper.__code__ = (lambda: None).__code__
                    with self.assertRaisesRegex(ValueError, 'helper substituted'):
                        value.commit_complete(self.hybrid, self.helper)
                finally:
                    self.helper.__code__ = original_code
                self.assertEqual(value.commits, 1)

    def test_first_commit_source_hash_failure_does_not_increment(self):
        value = self.observer()
        Path(self.sources['replay_fold']['path']).write_text('# modified helper source\n')
        with self.assertRaisesRegex(ValueError, 'helper source identity'):
            value.commit_complete(self.hybrid, self.helper)
        self.assertEqual(value.commits, 0)
        self.assertEqual(self.commits(), 0)
        self.assertIsNone(value.commit_identity)

    def test_replay_counter_layout_and_updates_preserved(self):
        for checkpoint, arm, closure in ((True, 'control', True), (True, 'mtp', True), (False, 'mtp', False)):
            with self.subTest(checkpoint=checkpoint, arm=arm, closure=closure):
                directory = self.root / (str(checkpoint) + arm)
                directory.mkdir()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(contract, 'betterbench_checkpoint_enabled', return_value=checkpoint))
                    if not closure:
                        stack.enter_context(patch.object(contract, 'settings', return_value=dict(phase='main-forward', performance_benchmark={'mode': 'm2047-fixed1024-c1-v1'}, requests=[{'maximum_output_tokens': 1024}] * 4)))
                    value = self.observer(arm, directory)
                value.identity = {'fixture': True}
                self.backend._cuda_graph_runner.raw_bs = 1
                value.backend_replayed(self.backend, SimpleNamespace(size=1), SimpleNamespace(batch_size=1))
                if closure:
                    self.backend._cuda_graph_runner.raw_bs = 2
                    value.backend_replayed(self.backend, SimpleNamespace(size=2), SimpleNamespace(batch_size=2))
                raw = (directory / 'graph-counter-rank-0.bin').read_bytes()
                self.assertEqual(raw, struct.pack('<QQQQ', 0, 2, 1, 1) if closure else struct.pack('<QQ', 0, 1))

    def verify_resolved(self, arm, interval, *, metrics=True, checkpoint=True, batch=True):
        # Isolate unrelated source/runtime contracts while executing the complete
        # actual entry function and the actual new arm/checkpoint helpers.
        resolved = dict(tp_size=2, context_length=32768, chunked_prefill_size=2048,
            mem_fraction_static=.8, max_running_requests=2, kv_cache_dtype='auto',
            speculative_draft_kv_cache_dtype='auto', enable_linear_replayssm=True,
            enable_linear_replayssm_spec=True)
        args = SimpleNamespace(resolve_once=lambda: None, enable_metrics=metrics,
            decode_log_interval=interval, tokenizer_worker_num=1)
        cfg = SimpleNamespace(**resolved, fp8_gemm_runner_backend='triton',
            speculative_algorithm=None, enable_multi_layer_eagle=False)
        runtime = SimpleNamespace(resolving_view=lambda value: cfg)
        binding = dict(arm=arm, resolved_contract=resolved, speculation={})
        with ExitStack() as stack:
            replacements = dict(memory_pool_config=None, cache_config=None,
                precision_reference_enabled=False, serving_seed_required=False,
                target_quantization='fp8', expected_environment={contract.DRAFT_GRAPH_ENV: '1'},
                validate_contract=dict(enabled=False, speculative_algorithm=None, enable_multi_layer_eagle=False),
                settings={'phase': contract.PAIRED_PHASE if batch else 'main-forward'},
                betterbench_checkpoint_enabled=checkpoint)
            for name, value in replacements.items():
                stack.enter_context(patch.object(contract, name, return_value=value))
            stack.enter_context(patch.dict(os.environ, {contract.DRAFT_GRAPH_ENV: '1'}, clear=True))
            entry._verify_resolved(runtime, args, binding)

    def test_resolved_exact_intervals_and_metrics_for_each_arm(self):
        for checkpoint in (False, True):
            for arm in ('control', 'mtp'):
                expected = 100 if checkpoint and arm == 'mtp' else 1
                with self.subTest(checkpoint=checkpoint, arm=arm):
                    self.verify_resolved(arm, expected, checkpoint=checkpoint)
                    for invalid in (0, -1, 2, 99, 101, True, float(expected), str(expected), None, 1 if expected == 100 else 100):
                        with self.assertRaisesRegex(entry.EntryError, 'metric/log observations'):
                            self.verify_resolved(arm, invalid, checkpoint=checkpoint)
                    for metrics in (False, 1, None):
                        with self.assertRaisesRegex(entry.EntryError, 'metric/log observations'):
                            self.verify_resolved(arm, expected, metrics=metrics, checkpoint=checkpoint)

    def test_non_batch_resolved_behavior_preserved(self):
        for arm in ('control', 'mtp'):
            self.verify_resolved(arm, 17, metrics=False, checkpoint=False, batch=False)

    def test_commit_validation_and_replay_methods_are_unchanged(self):
        source = (SOURCE / 'serving_graph_observer.py').read_text()
        tree = ast.parse(source)
        observer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Observer')
        current = {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() for node in observer.body if isinstance(node, ast.FunctionDef)}
        for name, expected in BASELINE_METHOD_SHA256.items():
            with self.subTest(method=name):
                self.assertEqual(expected, current[name])


if __name__ == '__main__':
    unittest.main(verbosity=2)
