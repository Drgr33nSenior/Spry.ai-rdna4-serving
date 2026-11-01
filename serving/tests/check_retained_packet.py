"""Explicit private-packet boundary checks; imports run on fresh copies only."""
import argparse
import ast
import base64
import hashlib
import importlib.util
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
from test_lifecycle_safety import installed
from test_observer_safety import observed_graph, capture, bind_commit, target_kv, probe
from test_source_cleanup import SOURCE, composed_fixture, integration


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RetainedPacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        root = PACKET.resolve(strict=True)
        manifest = root / 'FINAL-ARTIFACTS.sha256'
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata['source_seal_sha256']:
            raise AssertionError('retained packet seal pin differs')
        members = dict((name, digest) for digest, name in
                       (line.split('  ', 1) for line in manifest.read_text().splitlines()))
        cls.temporary = tempfile.TemporaryDirectory(prefix='cleanup-retained-')
        cls.addClassCleanup(cls.temporary.cleanup)
        copied = Path(cls.temporary.name).resolve()
        names = ('quality-template-w5/seal_evidence.py',
                 'author/package-v2/review/j01_probe.py',
                 'author/package-v2/review/serving_integration.py',
                 'author/package-v2/review/serving_client.py',
                 'author/package-v2/review/benchmark_artifacts.py')
        for name in names:
            original = root / name
            if hashlib.sha256(original.read_bytes()).hexdigest() != members[name]:
                raise AssertionError('retained source pin differs: ' + name)
            shutil.copyfile(original, copied / original.name)
        verifier = load('cleanup_retained_seal', copied / 'seal_evidence.py')
        verified = verifier.verify(root, metadata['source_seal_sha256'])
        if verified['members'] != metadata['source_seal_members']:
            raise AssertionError('retained member count differs')
        cls.baseline = load('cleanup_retained_probe', copied / 'j01_probe.py')
        cls.baseline_reader = load('cleanup_retained_reader', copied / 'serving_integration.py')
        cls.copied = copied
        cls.exporter = load('cleanup_retained_exporter', copied / 'benchmark_artifacts.py')
        contract = SimpleNamespace(canonical=lambda value:
            (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
        with patch.dict(sys.modules, serving_contract=contract):
            cls.baseline_client = load('cleanup_retained_client', copied / 'serving_client.py')
            cls.current_client = load('cleanup_current_client', SOURCE / 'serving_client.py')

    def test_extracted_validators_are_byte_identical_to_sealed_embedded_sources(self):
        for parent, helper in [('j01_probe.py', 'target_kv_metadata.py'),
                               ('serving_integration.py', 'kv_graph_join.py'),
                               ('serving_client.py', 'request_receipts.py')]:
            with self.subTest(helper=helper):
                tree = ast.parse((self.copied / parent).read_bytes())
                literals = [node.value for node in ast.walk(tree)
                            if isinstance(node, ast.Constant) and isinstance(node.value, bytes)]
                self.assertEqual(literals.count((SOURCE / helper).read_bytes()), 1)

    def test_actual_old_and_new_client_receipts_and_export_bytes_match(self):
        with tempfile.TemporaryDirectory(prefix='cleanup-receipt-equality-') as temporary:
            roots = [Path(temporary).resolve() / name / 'paired' for name in ('before', 'after')]
            for client, root in zip((self.baseline_client, self.current_client), roots):
                for ordinal in (0, 48, 171, 175):
                    prefix = f'{ordinal:02d}-retained-shape'
                    for side, suffix in [('requests', 'raw'), ('requests', 'intent.json'),
                                         ('requests', 'send-entered.json'), ('requests', 'send-returned.json'),
                                         ('responses', 'headers.json'), ('responses', 'partial.raw'),
                                         ('responses', 'record.json')]:
                        path = root / side / (prefix + '.' + suffix)
                        if suffix.endswith('.json'):
                            client._write_json(path, dict(ordinal=ordinal, side=side, suffix=suffix))
                        else:
                            client._write_exclusive(path, b'\x00\xff\n' if suffix == 'raw' else b'')
            physical = [{p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                        for root in roots]
            self.assertEqual(physical[0], physical[1])
            self.assertEqual(self.baseline_client._kv_receipts.logical_names(roots[0]),
                             self.current_client._kv_receipts.logical_names(roots[1]))
            self.assertEqual(self.exporter._http_tree(roots[0], 4 << 20, files=256),
                             self.exporter._http_tree(roots[1], 4 << 20, files=256))

    def test_retained_and_extracted_receipts_reject_tampered_field_hash(self):
        for receipts in (self.baseline_client._kv_receipts, self.current_client._kv_receipts):
            value = dict(schema=1, kind='native-http-request-receipts-v1', fields={
                'requests/00-sample.raw': dict(bytes=1, sha256='0' * 64, base64='eA==')})
            with self.assertRaisesRegex(ValueError, 'bytes/hash'):
                receipts.decode(receipts.canonical(value))

    def test_actual_old_and_new_composed_join_outputs_are_byte_identical(self):
        args = composed_fixture()
        before, after = self.baseline_reader._kv_graph_join, integration._kv_graph_join
        counts = lambda delta, responses, concurrency: dict(delta, response_verifier_count=sum(responses))
        original = before.validate(*args, paired_group_counts=counts)
        current = after.validate(*args, paired_group_counts=counts)
        self.assertEqual(before.canonical(original), after.canonical(current))
        self.assertEqual(hashlib.sha256(before.canonical(original)).hexdigest(),
                         hashlib.sha256(after.canonical(current)).hexdigest())

    def test_original_cap_failure_reproduced_after_successful_allocation(self):
        observer = self.baseline.LifecycleSlots.__new__(self.baseline.LifecycleSlots)
        observer.seq = 8192
        observer.indices = lambda value: []
        original = Mock(return_value=object())
        with self.assertRaises(AssertionError):
            observer.on_index_alloc(original)
        original.assert_called_once_with()

    def test_actual_protected_exporter_retains_both_health_receipts(self):
        with tempfile.TemporaryDirectory(prefix='cleanup-export-') as temporary:
            output = Path(temporary).resolve()
            native = output / 'native'
            native.mkdir(mode=0o700)
            expected = {}
            for rank in (0, 1):
                with installed(rank) as (observer, allocator, pool, cache, root, log):
                    observer.restore()
                    name = f'LIFECYCLE-HEALTH-rank-{rank}.json'
                    expected[name] = (root / name).read_bytes()
                    (native / name).write_bytes(expected[name])
            self.exporter.pack_http_artifacts(output, dict(
                experiment_mode=self.exporter.HTTP_MODE, arm='kv4',
                session_sha256='1' * 64, spec_sha256='2' * 64))
            bundled = json.loads((output / 'native-http-bundle.json').read_bytes())['observations']
            self.assertEqual(set(bundled), set(expected))
            for name, raw in expected.items():
                self.assertEqual(base64.b64decode(bundled[name]['base64'], validate=True), raw)
                self.assertEqual(bundled[name]['sha256'], hashlib.sha256(raw).hexdigest())

    def test_actual_exporter_retains_failed_publication_marker(self):
        with installed() as (observer, allocator, pool, cache, root, log):
            with patch.object(self.baseline.os, 'fsync', side_effect=OSError('injected')):
                observer.restore()
            bundled = self.exporter._http_tree(root, 4 << 20, files=256)
            self.assertEqual(set(bundled), {'LIFECYCLE-HEALTH-rank-0.pending'})

    def test_actual_exporter_rejects_linked_incomplete_publication(self):
        with installed() as (observer, allocator, pool, cache, root, log):
            with patch.object(Path, 'unlink', side_effect=OSError('injected')):
                observer.restore()
            with self.assertRaisesRegex(ValueError, 'singly linked'):
                self.exporter._http_tree(root, 4 << 20, files=256)

    def test_actual_exporter_preserves_final_graph_health_and_counters(self):
        with observed_graph() as (observer, backend, helper, manifest, worker, root, log):
            capture(observer, backend)
            bind_commit(observer, helper)
            observer.commit_complete(backend._cuda_graph_runner.model_runner.attn_backend, helper)
            observer.finish(manifest, worker)
            bundled = self.exporter._http_tree(root, 4 << 20, files=256)
            for name in ('OBSERVER-HEALTH-graph-rank-0.json', 'graph-counter-rank-0.bin',
                         'replay-commit-counter-rank-0.bin'):
                self.assertEqual(base64.b64decode(bundled[name]['base64'], validate=True), (root / name).read_bytes())

    def test_actual_exporter_preserves_buffered_target_kv_journal_and_health(self):
        with target_kv() as (call, restore, manifest, worker, module, root, log):
            with patch.object(probe._target_kv_metadata, 'route', return_value={}):
                call()
            restore()
            bundled = self.exporter._http_tree(root, 4 << 20, files=256)
            for name in ('OBSERVER-HEALTH-target_kv-rank-0.json', 'TARGET-KV-ROUTES-rank-0.jsonl'):
                self.assertEqual(base64.b64decode(bundled[name]['base64'], validate=True), (root / name).read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    options, remaining = parser.parse_known_args()
    PACKET = options.packet
    unittest.main(argv=[sys.argv[0], *remaining])
