"""Actual worker digest checks with local files and inert mount observations."""
import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load

WORKER = HERE.parent / 'spry_serving/startup_worker.py'


class StartupDigests(unittest.TestCase):
    def setUp(self):
        self.worker = load('cleanup_startup_digests_worker', WORKER)
        temporary = tempfile.TemporaryDirectory(prefix='startup-digests-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / 'helper.py'
        self.path.write_bytes(b'VALUE = 1\n')
        self.manifest = {'sources': {'helper': {'path': str(self.path)}}}
        self.readonly = patch.object(self.worker, '_readonly_source', return_value=True)
        self.readonly.start()
        self.addCleanup(self.readonly.stop)
        self.hashed = patch.object(self.worker, '_uncached_file_digest', wraps=self.worker._uncached_file_digest)
        self.hashes = self.hashed.start()
        self.addCleanup(self.hashed.stop)

    def digest(self):
        return self.worker._file_digest(self.path)

    def test_readonly_declared_source_is_hashed_once_for_five_checks(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.worker._source_digest_scope(self.manifest):
            self.assertEqual([self.digest() for _ in range(5)], [expected] * 5)
        self.hashes.assert_called_once_with(self.path)
        self.assertIsNone(self.worker._SOURCE_DIGEST_SCOPE)

    def test_outside_scope_writable_and_undeclared_files_are_always_rehashed(self):
        for _ in range(3): self.digest()
        with self.worker._source_digest_scope(self.manifest), patch.object(self.worker, '_readonly_source', return_value=False):
            for _ in range(3): self.digest()
        with self.worker._source_digest_scope({'sources': {}}):
            for _ in range(3): self.digest()
        self.assertEqual(self.hashes.call_count, 9)

    def test_changed_bytes_with_restored_size_and_mtime_refuse_at_real_helper_import(self):
        source = dict(path=str(self.path), module='startup_digest_fixture',
                      sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.addCleanup(sys.modules.pop, source['module'], None)
        old = self.path.stat()
        with self.worker._source_digest_scope(self.manifest):
            module = self.worker._source_module(source)
            self.assertEqual(module.VALUE, 1)
            self.path.write_bytes(b'VALUE = 2\n')
            os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns))
            self.assertEqual(self.path.stat().st_size, old.st_size)
            with self.assertRaisesRegex(ValueError, 'private helper source changed'):
                self.worker._source_module(source)
        self.assertEqual(self.hashes.call_count, 2)

    def test_inode_and_link_count_changes_force_fresh_hashes(self):
        with self.worker._source_digest_scope(self.manifest):
            expected = self.digest()
            replacement = self.root / 'replacement.py'
            replacement.write_bytes(self.path.read_bytes())
            old = self.path.stat()
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
            replacement.replace(self.path)
            self.assertEqual(self.digest(), expected)
            os.link(self.path, self.root / 'second-link.py')
            self.assertEqual(self.digest(), expected)
        self.assertEqual(self.hashes.call_count, 3)

    def test_a_cached_source_cannot_be_replaced_by_a_symlink(self):
        with self.worker._source_digest_scope(self.manifest):
            self.digest()
            target = self.root / 'moved.py'
            self.path.rename(target)
            self.path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'canonical'):
                self.digest()
        self.assertEqual(self.hashes.call_count, 1)

    def test_writable_view_discards_the_previous_readonly_cache(self):
        with self.worker._source_digest_scope(self.manifest):
            self.digest()
            with patch.object(self.worker, '_readonly_source', return_value=False):
                self.digest(); self.digest()
            self.digest(); self.digest()
        self.assertEqual(self.hashes.call_count, 4)

    def test_unknown_mount_flags_never_enable_reuse(self):
        self.readonly.stop()
        for flags, expected in ((0, False), (getattr(os, 'ST_RDONLY', 0), bool(getattr(os, 'ST_RDONLY', 0)))):
            with patch.object(self.worker.os, 'statvfs', return_value=SimpleNamespace(f_flag=flags)):
                self.assertIs(self.worker._readonly_source(self.path), expected)
        with self.worker._source_digest_scope(self.manifest), patch.object(self.worker.os, 'statvfs', side_effect=OSError('unavailable')):
            self.digest(); self.digest()
        self.assertEqual(self.hashes.call_count, 2)

    def test_inherited_process_scope_and_later_scope_cannot_reuse_digests(self):
        with self.worker._source_digest_scope(self.manifest):
            self.digest()
            with patch.object(self.worker.os, 'getpid', return_value=os.getpid() + 1):
                self.digest(); self.digest()
        with self.worker._source_digest_scope(self.manifest): self.digest()
        self.assertEqual(self.hashes.call_count, 4)

    def test_file_change_during_hashing_cannot_publish_a_cache_entry(self):
        original = self.worker._uncached_file_digest
        def changed(path):
            value = original(path)
            path.write_bytes(b'VALUE = 2\n')
            return value
        with self.worker._source_digest_scope(self.manifest), patch.object(self.worker, '_uncached_file_digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed while hashing'): self.digest()
            self.assertEqual(self.worker._SOURCE_DIGEST_SCOPE['digests'], {})

    def test_entry_scope_preserves_arguments_and_clears_on_success_or_failure(self):
        arguments = [object(), object(), 1, 1, 0, None, object()]
        keywords = dict(display_tp_rank=1, display_dp_rank=None, display_moe_ep_rank=None)
        def delegated(*args, **kwargs):
            self.assertEqual(args, (self.manifest, *arguments)); self.assertEqual(kwargs, keywords)
            self.assertIsNotNone(self.worker._SOURCE_DIGEST_SCOPE)
            self.digest(); self.digest()
            return 'served'
        with patch.object(self.worker, '_read_manifest', return_value=self.manifest), \
                patch.object(self.worker, '_run_scheduler_process', side_effect=delegated):
            self.assertEqual(self.worker.run_scheduler_process(*arguments, **keywords), 'served')
        self.assertIsNone(self.worker._SOURCE_DIGEST_SCOPE)
        with patch.object(self.worker, '_read_manifest', return_value=self.manifest), \
                patch.object(self.worker, '_run_scheduler_process', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.worker.run_scheduler_process(*arguments)
        self.assertIsNone(self.worker._SOURCE_DIGEST_SCOPE)
        self.hashes.assert_called_once()

    def test_actual_manifest_validator_reuses_bytes_and_still_rejects_a_changed_pin(self):
        pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        sources = {role: dict(module='sglang.fixture.' + role, path=str(self.path), sha256=pin)
                   for role in self.worker._MTP_SOURCE_ROLES | self.worker._DFLASH_SOURCE_ROLES}
        sources.update(gpu_identity=dict(module='benchmark_gpu_identity', path=str(self.path), sha256=pin),
            graph_observer=dict(module='serving_graph_observer', path=str(self.path), sha256=pin),
            scheduler=dict(module='sglang.fixture.scheduler', target='run', path=str(self.path), sha256=pin),
            worker_entry=dict(path=str(WORKER.resolve()), sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest()),
            serving_launcher=dict(path=str(self.path), sha256=pin), mtp_contract=dict(path=str(self.path), sha256=pin))
        sources['model_config']['module'] = 'sglang.srt.configs.model_config'
        value = dict(schema=1, mode='native-http-fp8-sprint01-v1', arm='mtp', session_sha256='a'*64,
            spec_sha256='b'*64, sources=sources, source_manifest_sha256=self.worker._digest(sources),
            expected_ranks=[0, 1], expected_devices={str(rank):dict(uuid=str(rank), render=str(rank)) for rank in (0, 1)},
            evidence_dir=str(self.root), arm_config=dict(disable_cuda_graph=False,
                backend_source_sha256='c'*64, shared_config_sha256='d'*64),
            preweight=dict(deadline_monotonic=200, gpu_identity=sources['gpu_identity'], graph_observer=sources['graph_observer']))
        with self.worker._source_digest_scope(value):
            for _ in range(5): self.assertIs(self.worker._validate_manifest(value), value)
            sources['model_config']['sha256'] = '0'*64
            value['source_manifest_sha256'] = self.worker._digest(sources)
            with self.assertRaisesRegex(ValueError, 'source hash mismatch: model_config'):
                self.worker._validate_manifest(value)
        self.assertEqual(Counter(call.args[0] for call in self.hashes.call_args_list),
                         Counter({self.path: 1, WORKER.resolve(): 1}))

    def test_all_other_worker_bodies_and_native_cleanup_are_unchanged(self):
        before = ast.parse((HERE.parent/'spry_serving/worker_entry.py').read_bytes())
        after = ast.parse(WORKER.read_bytes())
        functions = {node.name: node for node in after.body if isinstance(node, ast.FunctionDef)}
        for node in before.body:
            if isinstance(node, ast.FunctionDef) and node.name not in ('_file_digest', 'run_scheduler_process'):
                with self.subTest(function=node.name): self.assertEqual(ast.dump(node), ast.dump(functions[node.name]))
        old = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == 'run_scheduler_process')
        new = functions['_run_scheduler_process']
        self.assertEqual(ast.dump(ast.Module(body=old.body[2:], type_ignores=[])),
                         ast.dump(ast.Module(body=new.body[1:], type_ignores=[])))


if __name__ == '__main__': unittest.main()
