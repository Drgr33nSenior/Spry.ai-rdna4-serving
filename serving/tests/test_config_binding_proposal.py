"""CPU boundary tests for the isolated D2 configuration-approval proposal."""
import ast
import copy
from datetime import datetime, timezone, timedelta
import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent / 'proposals/d2-config-binding'
D1 = HERE.parent / 'proposals/d1-controller-inventory'
CONTROLLER = 'review/benchmark_physical_controller.py'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON member')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite JSON constant')


def isolated_controller(path):
    """Execute exact selected definitions; private tests load the entire module."""
    tree = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
    names = {'need', '_digest', '_canonical', '_stamp', 'mapping_sha256', 'verify_unchanged',
             '_configuration_file', 'authenticate_configuration_files', 'validate_physical_grant'}
    constants = {'GRANT_FIELDS', 'PROTECTED_SOURCE_NAMES'}
    body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants
                                                   for target in node.targets)]
    module = ModuleType('configuration_binding_cpu_fixture')
    module.__file__ = str(path)
    module.adapter = SimpleNamespace(readout=SimpleNamespace(_object=strict_pairs, _constant=reject_constant))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROPOSAL / CONTROLLER), 'exec'), module.__dict__)
    return module


class ConfigurationBinding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-config-binding-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.controller = isolated_controller(self.root / CONTROLLER)
        (self.root / 'review').mkdir()
        for name in self.controller.PROTECTED_SOURCE_NAMES:
            raw = (PROPOSAL / CONTROLLER).read_bytes() if name == Path(CONTROLLER).name else b'# inert CPU fixture\n'
            (self.root / 'review' / name).write_bytes(raw)
        self.run_path = self.root / 'config/runs/w5-v2/run.json'
        self.host_path = self.root / 'config/host/r9700-pair.json'
        self.menu_path = self.run_path.with_name('menu.json')
        for path, source in ((self.run_path, HERE.parent / 'config/runs/w5-v2/run.json'),
                             (self.menu_path, HERE.parent / 'config/runs/w5-v2/menu.json'),
                             (self.host_path, HERE.parent / 'config/host/r9700-pair.json')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical(json.loads(source.read_bytes())))
        pins = {'review/' + name: sha((self.root / 'review' / name).read_bytes())
                for name in self.controller.PROTECTED_SOURCE_NAMES}
        (self.root / 'PROTECTED-SHA256.json').write_bytes(canonical(pins))
        self.boot = '12345678-1234-1234-1234-123456789abc'
        self.parent = dict(pid=100, start_ticks=200, uid=0, boot_id=self.boot)
        self.manifest = dict(schema=2, package_root=str(self.root), package_files=pins,
                             boot_id=self.boot, run_config='config/runs/w5-v2/run.json')
        self.grant = dict(schema=2, scope='one-matched-fp8-kv-benchmark', model_start_authorized=True,
            maximum_starts=2, arm_order=['bf16', 'kv4'], cumulative_ceiling_seconds=6360,
            fixed_reserve_seconds=300, received_utc='2026-01-01T00:00:00Z', received_monotonic=100,
            boot_id=self.boot, manifest_sha256='a' * 64, package_sha256=self.controller.mapping_sha256(pins),
            source_manifest_sha256=self.controller.mapping_sha256(pins), transport_parent=self.parent,
            run_config_sha256=sha(self.run_path.read_bytes()), host_sha256=sha(self.host_path.read_bytes()),
            protected_set_sha256=sha((self.root / 'PROTECTED-SHA256.json').read_bytes()))

    def authenticate(self):
        return self.controller.authenticate_configuration_files(self.grant, self.manifest,
                                                                 deadline=205, clock=lambda: 101)

    def validate_grant(self, grant=None, parent=None, clock=lambda: 101):
        return self.controller.validate_physical_grant(grant or self.grant, self.manifest, 'a' * 64,
            self.parent if parent is None else parent, clock=clock,
            utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1))

    def test_valid_approval_returns_grant_and_stamps_all_bound_inputs(self):
        self.assertEqual(self.validate_grant(), self.grant)
        stamps = self.authenticate()
        expected = {str(self.root / name) for name in self.manifest['package_files']} | {
            str(self.run_path), str(self.host_path), str(self.menu_path), str(self.root / 'PROTECTED-SHA256.json')}
        self.assertTrue(expected <= set(stamps))
        self.assertIn(str(self.root / 'config'), stamps)
        self.assertIn(str(self.root / 'review'), stamps)
        self.controller.verify_unchanged(dict(stamps=stamps))

    def test_wrong_or_missing_approval_pins_refuse(self):
        for key in ('run_config_sha256', 'host_sha256', 'protected_set_sha256'):
            for change in ('wrong', 'missing'):
                grant = copy.deepcopy(self.grant)
                if change == 'wrong':
                    grant[key] = '0' * 64
                else:
                    del grant[key]
                with self.subTest(key=key, change=change), self.assertRaises(ValueError):
                    self.validate_grant(grant)

    def test_run_host_menu_and_protected_bytes_cannot_change(self):
        for path in (self.run_path, self.host_path, self.menu_path,
                     self.root / 'PROTECTED-SHA256.json', self.root / 'review/benchmark_artifacts.py'):
            before = path.read_bytes()
            path.write_bytes(before + b' ')
            try:
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    self.authenticate()
            finally:
                path.write_bytes(before)

    def test_protected_set_is_exact_and_must_match_package(self):
        path = self.root / 'PROTECTED-SHA256.json'
        before = json.loads(path.read_bytes())
        for change in ('missing', 'extra', 'package_disagreement'):
            value = dict(before)
            if change == 'missing':
                value.pop('review/benchmark_artifacts.py')
            elif change == 'extra':
                value['review/unknown.py'] = '0' * 64
            else:
                value['review/benchmark_artifacts.py'] = '0' * 64
            path.write_bytes(canonical(value))
            self.grant['protected_set_sha256'] = sha(path.read_bytes())
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.authenticate()

    def test_changed_reference_paths_and_run_identity_refuse_even_when_repinned(self):
        original = json.loads(self.run_path.read_bytes())
        for field, value in (('host', '../outside.json'), ('host', '/absolute.json'),
                             ('menu', '../menu.json'), ('run_id', 'different')):
            run = copy.deepcopy(original)
            if field == 'run_id':
                run[field] = value
            else:
                run[field]['file'] = value
            self.run_path.write_bytes(canonical(run))
            self.grant['run_config_sha256'] = sha(self.run_path.read_bytes())
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.authenticate()
        for name in ('../run.json', '/absolute/run.json', 'config/runs/../w5-v2/run.json'):
            self.manifest['run_config'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.authenticate()

    def test_run_host_pin_must_match_approval_and_host_cannot_hold_boot_id(self):
        run = json.loads(self.run_path.read_bytes())
        run['host']['sha256'] = '0' * 64
        self.run_path.write_bytes(canonical(run))
        self.grant['run_config_sha256'] = sha(self.run_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'run/host'):
            self.authenticate()
        host = json.loads(self.host_path.read_bytes())
        host['boot_id'] = self.boot
        self.host_path.write_bytes(canonical(host))
        self.grant['host_sha256'] = run['host']['sha256'] = sha(self.host_path.read_bytes())
        self.run_path.write_bytes(canonical(run))
        self.grant['run_config_sha256'] = sha(self.run_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'boot ID belongs in the approval'):
            self.authenticate()

    def test_noncanonical_duplicate_and_nonfinite_json_refuse_with_matching_pins(self):
        for raw in (b'{ "schema": 1 }\n', b'{"schema":1,"schema":1}\n', b'{"schema":NaN}\n', b'[]\n'):
            self.run_path.write_bytes(raw)
            self.grant['run_config_sha256'] = sha(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.authenticate()

    def test_linked_and_oversized_inputs_refuse(self):
        original = self.menu_path.read_bytes()
        self.menu_path.unlink()
        self.menu_path.symlink_to(self.run_path)
        with self.assertRaises((ValueError, OSError)):
            self.authenticate()
        self.menu_path.unlink()
        self.menu_path.write_bytes(original)
        alias = self.root / 'menu-alias'
        os.link(self.menu_path, alias)
        with self.assertRaisesRegex(ValueError, 'singly linked'):
            self.authenticate()
        alias.unlink()
        self.menu_path.write_bytes(b' ' * ((1 << 20) + 1))
        with self.assertRaisesRegex(ValueError, 'bounded'):
            self.authenticate()

    def test_fifo_is_opened_nonblocking_and_refused(self):
        self.menu_path.unlink()
        os.mkfifo(self.menu_path)
        original_open = os.open
        def checked_open(path, flags, *args, **kwargs):
            self.assertTrue(flags & os.O_NONBLOCK)
            return original_open(path, flags, *args, **kwargs)
        with patch.object(self.controller.os, 'open', side_effect=checked_open):
            with self.assertRaisesRegex(ValueError, 'bounded'):
                self.authenticate()

    def test_parent_receipt_and_owner_ceilings_still_refuse(self):
        for key, value in (('maximum_starts', True), ('arm_order', ['kv4', 'bf16']),
                           ('cumulative_ceiling_seconds', 6361), ('fixed_reserve_seconds', 299),
                           ('package_sha256', '0' * 64), ('source_manifest_sha256', '0' * 64),
                           ('boot_id', 'other'), ('received_monotonic', -1000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_grant(dict(self.grant, **{key: value}))
        with self.assertRaises(ValueError):
            self.validate_grant(parent=dict(self.parent, uid=1000))

    def test_deadline_and_post_authentication_changes_refuse(self):
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.controller.authenticate_configuration_files(self.grant, self.manifest,
                                                              deadline=101, clock=lambda: 101)
        stamps = self.authenticate()
        self.menu_path.write_bytes(self.menu_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'authenticated input changed'):
            self.controller.verify_unchanged(dict(stamps=stamps))

    def test_directory_substitution_is_caught_by_existing_recheck(self):
        stamps = self.authenticate()
        directory = self.root / 'config'
        directory.rename(self.root / 'moved-config')
        directory.symlink_to(self.root / 'moved-config', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'authenticated input changed'):
            self.controller.verify_unchanged(dict(stamps=stamps))

    def test_input_changed_while_other_files_are_read_is_refused(self):
        original_read = os.read
        menu_inode = self.menu_path.stat().st_ino
        changed = []
        def change_earlier_input(fd, size):
            raw = original_read(fd, size)
            if os.fstat(fd).st_ino == menu_inode and not changed:
                self.host_path.write_bytes(self.host_path.read_bytes() + b' ')
                changed.append(True)
            return raw
        with patch.object(self.controller.os, 'read', side_effect=change_earlier_input):
            with self.assertRaisesRegex(ValueError, 'authenticated input changed'):
                self.authenticate()
        self.assertEqual(changed, [True])

    def test_configuration_cannot_be_read_for_a_different_executed_root(self):
        self.controller.__file__ = str(self.root / 'other-controller.py')
        Path(self.controller.__file__).write_bytes(b'# other\n')
        with self.assertRaisesRegex(ValueError, 'executed controller root'):
            self.authenticate()


class ProposalIntegrity(unittest.TestCase):
    def test_patch_pins_scope_and_preserved_checks(self):
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        before, after = (D1 / CONTROLLER).read_bytes(), (PROPOSAL / CONTROLLER).read_bytes()
        diff = ''.join(difflib.unified_diff(before.decode().splitlines(keepends=True),
            after.decode().splitlines(keepends=True), fromfile='a/' + CONTROLLER, tofile='b/' + CONTROLLER)).encode()
        self.assertEqual(diff, (PROPOSAL / 'D2.patch').read_bytes())
        for key, raw in (('baseline_sha256', before), ('proposed_sha256', after), ('patch_sha256', diff)):
            self.assertEqual(record[key], sha(raw))
        old, new = ast.parse(before), ast.parse(after)
        functions = [{n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in (old, new)]
        changed = {name for name in functions[0] if functions[0][name] != functions[1][name]}
        self.assertEqual(changed, {'validate_manifest', '_run_bound_controller', 'validate_physical_grant'})
        old_grant = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_physical_grant')
        new_grant = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == old_grant.name)
        self.assertEqual([ast.dump(n) for n in old_grant.body],
                         [ast.dump(n) for n in new_grant.body[:-2] + new_grant.body[-1:]])
        # Only the declared manifest field and pre-lease authentication call may
        # differ; retain the rest of pre-launch and post-run control flow exactly.
        for function_name in ('validate_manifest', '_run_bound_controller'):
            baseline = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
            restored = copy.deepcopy(next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == function_name))
            removed = 0
            if function_name == 'validate_manifest':
                for node in ast.walk(restored):
                    if isinstance(node, ast.Set):
                        kept = [item for item in node.elts if not isinstance(item, ast.Constant) or item.value != 'run_config']
                        removed += len(node.elts) - len(kept)
                        node.elts = kept
            else:
                block = next(node for node in restored.body if isinstance(node, ast.Try))
                kept = []
                for node in block.body:
                    if isinstance(node, ast.If) and any(isinstance(item, ast.Call) and
                            isinstance(item.func, ast.Name) and item.func.id == 'authenticate_configuration_files'
                            for item in ast.walk(node)):
                        removed += 1
                    else:
                        kept.append(node)
                block.body = kept
            self.assertEqual(removed, 1)
            self.assertEqual(ast.dump(baseline), ast.dump(restored))
        unchanged_assignments = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, ast.Assign)
            and not any(isinstance(t, ast.Name) and t.id in {'GRANT_FIELDS', 'PROTECTED_SOURCE_NAMES'} for t in n.targets)]
        self.assertEqual(unchanged_assignments(old), unchanged_assignments(new))
        pins = {'review/' + name: pin for name, pin in record['proposed_protected_files'].items()}
        self.assertEqual(record['proposed_protected_set_sha256'], sha(canonical(pins)))
        expected = json.loads((D1 / 'PROPOSAL.json').read_bytes())['proposed_protected_files']
        expected[Path(CONTROLLER).name] = sha(after)
        self.assertEqual(record['proposed_protected_files'], expected)
        self.assertFalse(record['owner_approved'])
        self.assertFalse(record['consumed_by_builder'])
        self.assertFalse(record['entire_D2_complete'])


if __name__ == '__main__':
    unittest.main()
