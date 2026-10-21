"""CPU OCI fixtures: no extraction, runtime imports, host mutation or GPU."""
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runtime_verifier_tests', ROOT / 'tools/verify_benchmark_runtime.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def file(raw=b'fixture', executable=0):
    return {'kind': 'file', 'bytes': len(raw), 'sha256': verify.sha(raw), 'executable': executable}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def image(self, layers):
        folder = self.root / 'image'
        (folder / 'base/blobs/sha256').mkdir(parents=True)
        records, diffs = [], []
        for layer in layers:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                for name, kind, data in layer:
                    info = tarfile.TarInfo(name)
                    info.mode = 0o755 if kind == 'directory' else 0o644
                    if kind == 'directory':
                        info.type = tarfile.DIRTYPE
                    elif kind == 'symlink':
                        info.type, info.linkname = tarfile.SYMTYPE, data
                    elif kind == 'hardlink':
                        info.type, info.linkname = tarfile.LNKTYPE, data
                    elif kind == 'fifo':
                        info.type = tarfile.FIFOTYPE
                    else:
                        info.size = len(data)
                    archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
            raw = buffer.getvalue()
            compressed = gzip.compress(raw, mtime=0)
            pin = verify.sha(compressed)
            (folder / 'base/blobs/sha256' / pin).write_bytes(compressed)
            records.append({'digest': 'sha256:' + pin, 'size': len(compressed)})
            diffs.append('sha256:' + verify.sha(raw))
        config = verify.canonical({'rootfs': {'diff_ids': diffs}})
        manifest = verify.canonical({'config': {'digest': 'sha256:' + verify.sha(config)}, 'layers': records})
        (folder / 'base-config.json').write_bytes(config)
        (folder / 'base-manifest.json').write_bytes(manifest)
        return folder, verify.sha(manifest), verify.sha(config)

    def derive(self, folder, manifest, config):
        return verify.derive_image(folder, time.monotonic() + 10, manifest_pin=manifest, config_pin=config)

    def test_actual_oci_stream_and_whiteouts_derive_without_extracting(self):
        folder, manifest, config = self.image([
            [('usr/', 'directory', None), ('usr/old', 'file', b'old'), ('usr/retained', 'file', b'keep')],
            [('usr/.wh.old', 'file', b''), ('usr/new', 'file', b'new')]])
        output = self.derive(folder, manifest, config)
        self.assertEqual(output['content_inventory'], {'usr': {'kind': 'directory'},
            'usr/retained': file(b'keep'), 'usr/new': file(b'new')})
        self.assertEqual(len(output['layers']), 2)
        self.assertFalse((folder / 'usr').exists())

    def test_opaque_removes_only_lower_entries_even_when_whiteout_is_last(self):
        tree = {'a': {'kind': 'directory'}, 'a/lower': file()}
        verify.apply_layer(tree, [('a/upper', file(b'new'))], ['a/.wh..wh..opq'])
        self.assertEqual(tree, {'a': {'kind': 'directory'}, 'a/upper': file(b'new')})

    def test_hardlink_copies_content_at_link_creation_not_later_replacement(self):
        tree = {}
        verify.apply_layer(tree, [('a', file(b'old')), ('b', {'kind': 'hardlink', 'target': 'a'}),
            ('a', file(b'new'))], [])
        self.assertEqual(tree['b'], file(b'old'))
        self.assertEqual(tree['a'], file(b'new'))
        verify.apply_layer(tree, [('forward', {'kind': 'hardlink', 'target': 'future'}),
            ('future', file(b'future'))], [])
        self.assertEqual(tree['forward'], file(b'future'))

    def test_unresolved_hardlinks_and_symlink_parent_refuse(self):
        with self.assertRaises(ValueError):
            verify.apply_layer({}, [('a', {'kind': 'hardlink', 'target': 'missing'})], [])
        with self.assertRaises(ValueError):
            verify.apply_layer({'a': {'kind': 'symlink', 'target': '/outside'}}, [('a/child', file())], [])

    def test_file_directory_replacement_removes_stale_children(self):
        tree = {'a': {'kind': 'directory'}, 'a/old': file()}
        verify.apply_layer(tree, [('a', file(b'new'))], [])
        self.assertEqual(tree, {'a': file(b'new')})
        verify.apply_layer(tree, [('a', {'kind': 'directory'}), ('a/new', file())], [])
        self.assertEqual(tree['a/new'], file())

    def test_current_tree_never_follows_symlink_and_retains_exec_bits(self):
        base = self.root / 'base'
        base.mkdir()
        (base / 'file').write_bytes(b'owned')
        (base / 'file').chmod(0o755)
        (base / 'absolute').symlink_to('/not/read/by/this/verifier')
        (base / 'relative').symlink_to('file')
        actual, stamps = verify.current_tree(base, time.monotonic() + 5)
        self.assertEqual(actual['file'], file(b'owned', 0o111))
        self.assertEqual(actual['absolute'], {'kind': 'symlink', 'target': '/not/read/by/this/verifier'})
        self.assertEqual(set(stamps), set(actual))

    def test_current_hardlinked_regular_files_keep_same_content(self):
        base = self.root / 'base'
        base.mkdir()
        (base / 'a').write_bytes(b'owned')
        os.link(base / 'a', base / 'b')
        actual, _ = verify.current_tree(base, time.monotonic() + 5)
        self.assertEqual(actual['a'], actual['b'])

    def test_comparison_keeps_all_counts_and_bounded_details(self):
        result = verify.compare({'same': file(), 'changed': file(b'a'), 'missing': file()},
            {'same': file(), 'changed': file(b'b'), 'extra': file()})
        self.assertFalse(result['matches'])
        self.assertEqual([result[key] for key in ('missing_count', 'extra_count', 'changed_count')], [1, 1, 1])
        self.assertEqual(result['changed'][0]['path'], 'changed')
        self.assertTrue(verify.compare({'same': file()}, {'same': file()})['matches'])

    def test_bad_manifest_layer_size_and_corrupt_blob_refuse(self):
        folder, manifest, config = self.image([[('a', 'file', b'owned')]])
        with self.assertRaises(ValueError):
            self.derive(folder, '0' * 64, config)
        blob = next((folder / 'base/blobs/sha256').iterdir())
        blob.write_bytes(blob.read_bytes() + b'extra')
        with self.assertRaises(ValueError):
            self.derive(folder, manifest, config)

    def test_path_traversal_special_file_and_deadline_refuse(self):
        for value in ('/absolute', '../escape', 'a/../../escape', 'a\\b', ''):
            with self.assertRaises(ValueError):
                verify.relative(value)
        folder, manifest, config = self.image([[('fifo', 'fifo', None)]])
        with self.assertRaises(ValueError):
            self.derive(folder, manifest, config)
        with self.assertRaises(ValueError):
            verify.current_tree(self.root, time.monotonic() - 1)

    def test_retained_composition_checks_actual_base_prefix_and_links(self):
        base = self.root / 'base-rootfs'
        prefix = self.root / 'build-output3/build-venv'
        base.mkdir()
        prefix.mkdir(parents=True)
        (base / 'interpreter').write_bytes(b'inert ELF fixture; never execute')
        (prefix / 'module.py').write_bytes(b'# inert fixture\n')
        (prefix / 'link').symlink_to('module.py')
        mapping, _ = verify.current_tree(base, time.monotonic() + 2)
        expected = {'module.py': verify.sha(b'# inert fixture\n'), 'link': {'symlink': 'module.py'}}
        pin = verify.sha(verify.canonical(expected).rstrip(b'\n'))
        facts = {'installed_prefix_files': expected, 'installed_inventory_sha256': pin}
        with patch.object(verify, 'BASE_CONTENT_SHA256', verify.sha(verify.canonical(mapping))), \
             patch.object(verify, 'BASE_MEMBERS', len(mapping)), patch.object(verify, 'PREFIX_SHA256', pin), \
             patch.object(verify, 'PREFIX_MEMBERS', 2):
            result = verify.verify_retained(self.root, facts, time.monotonic() + 3)
            self.assertEqual(result['prefix_members'], 2)
            self.assertIn(str(prefix / 'link'), result['stamps'])
            (prefix / 'module.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'prefix'):
                verify.verify_retained(self.root, facts, time.monotonic() + 3)
            (base / 'interpreter').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'base-rootfs'):
                verify.verify_retained(self.root, facts, time.monotonic() + 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
