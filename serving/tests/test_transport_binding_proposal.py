"""CPU checks for configuration transport; no root, RPC, observer or model starts."""
import ast
import copy
from contextlib import redirect_stderr
from datetime import datetime, timezone, timedelta
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_config_binding_proposal import canonical, sha, isolated_controller, PROPOSAL as CONTROLLER_PROPOSAL

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent / 'proposals/d2-transport-binding'
TRANSPORT = 'review/benchmark_host_transport.py'
spec = importlib.util.spec_from_file_location('cleanup_transport_proposal', PROPOSAL / TRANSPORT)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class TransportBinding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-transport-binding-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.members = {'review/' + name: b'# inert CPU helper\n' for name in transport.PROTECTED_SOURCE_NAMES}
        self.members[TRANSPORT] = (PROPOSAL / TRANSPORT).read_bytes()
        self.members['review/benchmark_physical_controller.py'] = (
            CONTROLLER_PROPOSAL / 'review/benchmark_physical_controller.py').read_bytes()
        self.run_name = 'config/runs/w5-v2/run.json'
        self.host_name = 'config/host/r9700-pair.json'
        self.menu_name = 'config/runs/w5-v2/menu.json'
        for name in (self.run_name, self.host_name, self.menu_name):
            self.members[name] = canonical(json.loads((HERE.parent / name).read_bytes()))
        files = {name: sha(raw) for name, raw in self.members.items() if name.startswith('review/')}
        self.members['PROTECTED-SHA256.json'] = canonical(files)
        self.pins = dict(run_config_sha256=sha(self.members[self.run_name]),
                        host_sha256=sha(self.members[self.host_name]),
                        protected_set_sha256=sha(self.members['PROTECTED-SHA256.json']))
        self.manifest = dict(schema=2, scope='matched-benchmark-physical-package-v2',
            launch_authorized=False, proposed_complete_seconds=6360,
            package_root='/home/developer/fp8-kv-benchmark.fixture123',
            target_root='/home/developer/fp8-kv-benchmark.fixture123',
            boot_id='12345678-1234-1234-1234-123456789abc', session_sha256='c' * 64,
            experiment_mode='native-http-fp8-sprint01-v1', package_files=files, run_config=self.run_name)
        self.parent = dict(pid=100, start_ticks=200, uid=0, boot_id=self.manifest['boot_id'])
        self.refresh_manifest()

    def refresh_manifest(self):
        self.members['MANIFEST.json'] = canonical(self.manifest)
        self.manifest_pin = sha(self.members['MANIFEST.json'])

    def archive(self, *, duplicate=None, special=None):
        path = self.root / 'payload.tar'
        with tarfile.open(path, 'w') as archive:
            for name, raw in self.members.items():
                item = tarfile.TarInfo(name)
                item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
            if duplicate:
                item = tarfile.TarInfo(duplicate)
                archive.addfile(item, io.BytesIO())
            if special:
                name, kind = special
                item = tarfile.TarInfo(name)
                item.type, item.linkname = kind, '/outside'
                archive.addfile(item)
        return path, sha(path.read_bytes())

    def inspect(self, **kwargs):
        path, pin = self.archive(**kwargs)
        return transport.inspect_archive(path, pin, self.manifest_pin, configuration_pins=self.pins)

    def grant(self, pins=None, parent=None):
        return transport.new_grant(self.manifest, self.manifest_pin, received_utc='2026-01-01T00:00:00Z',
            received_monotonic=100, parent=self.parent if parent is None else parent,
            configuration_pins=self.pins if pins is None else pins)

    def stage(self):
        root = self.root / 'staged'
        root.mkdir()
        (root / 'review').mkdir()
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            transport.stage_members(fd, self.members)
        finally:
            os.close(fd)
        # TemporaryDirectory must be able to remove read-only config directories.
        self.addCleanup(self.unlock, root)
        return root

    @staticmethod
    def unlock(root):
        for path in (root / 'config', root / 'config/host', root / 'config/runs', root / 'config/runs/w5-v2'):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)

    def test_archive_staging_and_grant_join_the_controller_approval(self):
        manifest, members = self.inspect()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(members, self.members)
        self.assertFalse((self.root / 'review').exists())
        root = self.stage()
        self.assertEqual({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}, members)
        for name in ('config', 'config/host', 'config/runs', 'config/runs/w5-v2'):
            self.assertEqual(stat.S_IMODE((root / name).stat().st_mode), 0o555)
        for name in members:
            self.assertEqual(stat.S_IMODE((root / name).stat().st_mode), 0o444)
        grant = self.grant()
        controller = isolated_controller(root / 'review/benchmark_physical_controller.py')
        actual = controller.validate_physical_grant(grant, dict(manifest, package_root=str(root)),
            self.manifest_pin, self.parent, clock=lambda: 101,
            utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1))
        self.assertEqual(actual, grant)
        self.assertEqual(set(grant), controller.GRANT_FIELDS)
        self.assertEqual(grant['cumulative_ceiling_seconds'], 6360)
        self.assertEqual(grant['fixed_reserve_seconds'], 300)
        self.assertEqual(grant['maximum_starts'], 2)
        self.assertEqual(grant['arm_order'], ['bf16', 'kv4'])

    def test_independent_pins_are_required_and_cannot_be_inferred_or_override_grant(self):
        path, pin = self.archive()
        for name in self.pins:
            for kind in ('missing', 'wrong', 'invalid'):
                pins = dict(self.pins)
                if kind == 'missing':
                    del pins[name]
                else:
                    pins[name] = '0' * 64 if kind == 'wrong' else True
                with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                    transport.inspect_archive(path, pin, self.manifest_pin, configuration_pins=pins)
        for pins in ({}, dict(self.pins, maximum_starts=20), dict(self.pins, host_sha256=True)):
            with self.assertRaises(ValueError):
                self.grant(pins)
        with self.assertRaisesRegex(ValueError, 'native root parent'):
            self.grant(parent=dict(self.parent, uid=1000))

    def test_changed_configuration_and_each_protected_helper_refuse(self):
        for name in (self.run_name, self.host_name, self.menu_name, 'PROTECTED-SHA256.json',
                     *self.manifest['package_files']):
            raw = self.members[name]
            self.members[name] = raw + b' '
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.inspect()
            finally:
                self.members[name] = raw

    def test_rehashed_but_noncanonical_or_invalid_json_refuses(self):
        raw = self.members[self.host_name]
        value = json.loads(raw)
        for bad in (raw + b' ', b'{"schema":1,"schema":1}\n', b'{"x":NaN}\n', b'[]\n',
                    canonical(dict(value, boot_id=self.manifest['boot_id'])),
                    canonical(dict(value, schema=True))):
            self.members[self.host_name] = bad
            self.pins['host_sha256'] = sha(bad)
            run = json.loads(self.members[self.run_name])
            run['host']['sha256'] = sha(bad)
            self.members[self.run_name] = canonical(run)
            self.pins['run_config_sha256'] = sha(self.members[self.run_name])
            with self.subTest(bad=bad[:40]), self.assertRaises(ValueError):
                self.inspect()

    def test_reference_paths_and_exact_config_inventory_refuse(self):
        original = self.members[self.run_name]
        for key, bad in (('run_id', 'different'), ('host', '../outside.json'), ('host', '/outside'),
                         ('menu', '../menu.json')):
            run = json.loads(original)
            if key == 'run_id':
                run[key] = bad
            else:
                run[key]['file'] = bad
            self.members[self.run_name] = canonical(run)
            self.pins['run_config_sha256'] = sha(self.members[self.run_name])
            with self.assertRaises(ValueError):
                self.inspect()
        self.members[self.run_name] = original
        self.pins['run_config_sha256'] = sha(original)
        for name in ('config/host/extra.json', 'config/runs/extra/run.json', 'review/extra.py', '../escape'):
            self.members[name] = b'{}\n'
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.inspect()
            finally:
                del self.members[name]
        self.manifest['package_files'][self.run_name] = sha(original)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'independently approved'):
            self.inspect()

    def test_protected_set_must_be_exact_and_join_package(self):
        path = 'PROTECTED-SHA256.json'
        original = json.loads(self.members[path])
        for kind in ('missing', 'extra', 'package'):
            value = dict(original)
            if kind == 'missing':
                value.pop(TRANSPORT)
            elif kind == 'extra':
                value['review/unexpected.py'] = '0' * 64
            else:
                value[TRANSPORT] = '0' * 64
            self.members[path] = canonical(value)
            self.pins['protected_set_sha256'] = sha(self.members[path])
            with self.assertRaises(ValueError):
                self.inspect()

    def test_archive_links_duplicates_and_physical_contract_still_refuse(self):
        for name in (self.run_name, 'PROTECTED-SHA256.json'):
            with self.assertRaises(ValueError):
                self.inspect(duplicate=name)
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE):
            with self.assertRaises(ValueError):
                self.inspect(special=('config/host/extra.json', kind))
        original = copy.deepcopy(self.manifest)
        for key, bad in (('launch_authorized', True), ('proposed_complete_seconds', 6361),
                         ('package_root', '/other'), ('target_root', '/tmp/other')):
            self.manifest = dict(original, **{key: bad})
            self.refresh_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inspect()

    def test_member_count_and_configuration_byte_caps_remain_enforced(self):
        # The ordinary branch remains capped at 64, including the four config files.
        while len(self.members) < 64:
            name = f'review/extra-{len(self.members)}.py'
            self.members[name] = b'# inert\n'
            self.manifest['package_files'][name] = sha(self.members[name])
        self.refresh_manifest()
        self.inspect()
        self.members['review/one-too-many.py'] = b'# inert\n'
        self.manifest['package_files']['review/one-too-many.py'] = sha(self.members['review/one-too-many.py'])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'mode-specific member limit'):
            self.inspect()
        for name, limit, pin_key in ((self.run_name, 256 << 10, 'run_config_sha256'),
                                    (self.host_name, 64 << 10, 'host_sha256'),
                                    ('PROTECTED-SHA256.json', 64 << 10, 'protected_set_sha256'),
                                    (self.menu_name, 1 << 20, None)):
            old = self.members[name]
            old_run = self.members[self.run_name]
            self.members[name] = b' ' * (limit + 1)
            pins = dict(self.pins)
            if pin_key:
                pins[pin_key] = sha(self.members[name])
            if name in (self.host_name, self.menu_name):
                run = json.loads(old_run)
                run['host' if name == self.host_name else 'menu']['sha256'] = sha(self.members[name])
                self.members[self.run_name] = canonical(run)
                pins['run_config_sha256'] = sha(self.members[self.run_name])
            with self.assertRaisesRegex(ValueError, 'configuration member digest/size'):
                transport.configuration_members(self.manifest, self.members, pins)
            self.members[name] = old
            self.members[self.run_name] = old_run

    def test_staging_refuses_existing_or_linked_config_directory(self):
        for linked in (False, True):
            root = self.root / ('linked' if linked else 'existing')
            root.mkdir()
            (root / 'review').mkdir()
            if linked:
                (root / 'config').symlink_to(self.root, target_is_directory=True)
            else:
                (root / 'config').mkdir()
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with self.assertRaises(FileExistsError):
                    transport.stage_members(fd, self.members)
            finally:
                os.close(fd)
            self.assertEqual(list((root / 'review').iterdir()), [])

    def test_bad_approval_in_human_entry_refuses_before_any_staging_or_import(self):
        path, pin = self.archive()
        args = SimpleNamespace(original_receipt=(100, '2026-01-01T00:00:00Z'), archive=str(path),
            archive_sha256=pin, manifest_sha256=self.manifest_pin,
            expected_self_sha256=sha(self.members[TRANSPORT]), **dict(self.pins, host_sha256='0' * 64))
        with patch.multiple(transport.os, getuid=lambda: 0, geteuid=lambda: 0, getgid=lambda: 0, getegid=lambda: 0), \
                patch.object(transport, 'protected_stage'), patch.object(transport, 'root_source'), \
                patch.object(Path, 'mkdir') as mkdir, patch.object(transport, 'stage_members') as stage, \
                patch.object(transport, 'new_grant') as grant, \
                patch.object(transport.importlib.util, 'spec_from_file_location') as load:
            with self.assertRaisesRegex(ValueError, 'run/host approval binding'):
                transport.human_main(args)
            for effect in (mkdir, stage, grant, load):
                effect.assert_not_called()

    def test_cli_requires_and_passes_all_three_owner_supplied_pins(self):
        base = ['--human-root-benchmark', '--archive', '/fixture/payload.tar', '--archive-sha256', 'a' * 64,
                '--manifest-sha256', 'b' * 64, '--expected-self-sha256', 'c' * 64]
        options = [part for key, pin in self.pins.items() for part in ('--' + key.replace('_', '-'), pin)]
        with patch.object(transport, 'human_main', return_value=0) as human:
            self.assertEqual(transport.main(base + options), 0)
            args = human.call_args.args[0]
            self.assertEqual({key: getattr(args, key) for key in self.pins}, self.pins)
            for i in range(0, len(options), 2):
                human.reset_mock()
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    transport.main(base + options[:i] + options[i + 2:])
                self.assertEqual(error.exception.code, 2)
                human.assert_not_called()


class ProposalIntegrity(unittest.TestCase):
    def test_hashes_unchanged_boundaries_and_approval_scope(self):
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        raw = (PROPOSAL / TRANSPORT).read_bytes()
        self.assertEqual(sha(raw), record['proposed_sha256'])
        self.assertEqual(sha((PROPOSAL / 'D2.patch').read_bytes()), record['patch_sha256'])
        tree = ast.parse(raw)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in record['unchanged_definition_sha256']:
                self.assertEqual(sha(ast.get_source_segment(raw.decode(), node).encode()),
                                 record['unchanged_definition_sha256'][node.name], node.name)
        self.assertEqual((transport.PACKET_LIMIT, transport.ARCHIVE_LIMIT, transport.TOTAL_SECONDS, transport.SEAL_RESERVE),
                         (2 << 20, 16 << 20, 6360, 300))
        source = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        self.assertEqual(record['baseline_sha256'], source['protected_source_files']['benchmark_host_transport.py'])
        pins = dict(json.loads((CONTROLLER_PROPOSAL / 'PROPOSAL.json').read_bytes())['proposed_protected_files'])
        pins['benchmark_host_transport.py'] = sha(raw)
        self.assertEqual(pins, record['proposed_protected_files'])
        self.assertEqual(sha(canonical({'review/' + name: pin for name, pin in pins.items()})),
                         record['proposed_protected_set_sha256'])
        self.assertFalse(record['owner_approved'])
        self.assertFalse(record['consumed_by_builder'])
        self.assertFalse(record['entire_D2_complete'])
        approval = json.loads((HERE.parent / 'proposals/OWNER-APPROVAL-20261101.json').read_bytes())
        self.assertEqual(approval['owner_words'], 'OK, approved')
        self.assertEqual(approval['presented_configuration_binding_patch_sha256'],
                         sha((CONTROLLER_PROPOSAL / 'D2.patch').read_bytes()))
        self.assertFalse(approval['gpu_launch_authorized'])


if __name__ == '__main__':
    unittest.main()
