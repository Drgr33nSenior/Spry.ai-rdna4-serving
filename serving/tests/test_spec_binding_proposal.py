"""Canonical spec reproduction and refusal tests without the private runtime."""
import ast
import copy
import difflib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_config_binding_proposal import canonical, sha, strict_pairs, reject_constant

PROPOSAL = HERE.parent / 'proposals/d2-spec-binding'
BASELINE = HERE.parent / 'proposals/d2-cleaned-inventory'
CONTROLLER = 'review/benchmark_physical_controller.py'


def isolated_controller(path):
    """Use exact source definitions; private checks import the complete controller."""
    tree = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
    names = {'need', '_digest', '_canonical', '_stamp', 'mapping_sha256', 'verify_unchanged',
             '_configuration_file', 'authenticate_configuration_files', 'render_configured_spec',
             'verify_configured_spec', '_run_physical_after_retained_closure', 'http_expected_binding'}
    constants = {'PROTECTED_SOURCE_NAMES', 'SUPPORTED_RUN_SHA256', 'SUPPORTED_SETTINGS_SHA256',
                 'SPEC_TEMPLATE_SHA256'}
    body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants
                                                   for target in node.targets)]
    module = ModuleType('spec_binding_cpu_fixture')
    module.__file__ = str(path)
    module.adapter = SimpleNamespace(readout=SimpleNamespace(_object=strict_pairs, _constant=reject_constant))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROPOSAL / CONTROLLER), 'exec'), module.__dict__)
    return module


class SpecBinding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-spec-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'review').mkdir()
        self.controller = isolated_controller(self.root / CONTROLLER)
        for name in self.controller.PROTECTED_SOURCE_NAMES:
            (self.root / 'review' / name).write_bytes((PROPOSAL / CONTROLLER).read_bytes()
                if name == Path(CONTROLLER).name else b'# inert protected fixture\n')
        self.run = json.loads((HERE.parent / 'config/runs/w5-v2/run.json').read_bytes())
        self.host = json.loads((HERE.parent / self.run['host']['file']).read_bytes())
        for name in ('config/runs/w5-v2/run.json', 'config/runs/w5-v2/menu.json', self.run['host']['file']):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical(json.loads((HERE.parent / name).read_bytes())))
        self.template = json.loads((HERE / 'fixtures/w5/BENCHMARK-SPEC2.json').read_bytes())
        for name in self.template['graph_projection']['helpers']:
            (self.root / 'review' / name).write_bytes(b'# inert source fixture\n')
        for arm in ('control', 'mtp'):
            (self.root / 'review' / ('HTTP-BINDING-' + arm + '.json')).write_bytes(canonical({'arm': arm}))
        (self.root / 'review/FINAL-SETTINGS.json').write_bytes((HERE / 'fixtures/w5/FINAL-SETTINGS.json').read_bytes())
        files = {'review/' + p.name: sha(p.read_bytes()) for p in (self.root / 'review').iterdir()}
        protected = {'review/' + name: files['review/' + name] for name in self.controller.PROTECTED_SOURCE_NAMES}
        (self.root / 'PROTECTED-SHA256.json').write_bytes(canonical(protected))
        self.manifest = dict(package_root=str(self.root), package_files=files,
                             run_config='config/runs/w5-v2/run.json')
        self.pins = dict(run_config_sha256=sha((self.root / self.manifest['run_config']).read_bytes()),
            host_sha256=sha((self.root / self.run['host']['file']).read_bytes()),
            protected_set_sha256=sha((self.root / 'PROTECTED-SHA256.json').read_bytes()))
        spec = self.controller.render_configured_spec(self.template, self.run, self.host, files)
        self.write_spec(canonical(spec))

    def write_spec(self, raw):
        (self.root / 'review/BENCHMARK-SPEC2.json').write_bytes(raw)
        self.manifest['spec_sha256'] = sha(raw)
        self.manifest['package_files']['review/BENCHMARK-SPEC2.json'] = sha(raw)

    def verify(self):
        return self.controller.verify_configured_spec(self.pins, self.manifest, deadline=205, clock=lambda: 101)

    def test_reproduces_exact_retained_spec_and_generated_source_bindings(self):
        files = {'review/' + name: pin for name, pin in self.template['graph_projection']['helpers'].items()}
        files.update({row['path']: row['sha256'] for row in self.template['http_serving']['bindings'].values()})
        value = self.controller.render_configured_spec(self.template, self.run, self.host, files)
        self.assertEqual(canonical(value), (HERE / 'fixtures/w5/BENCHMARK-SPEC2.json').read_bytes())
        verified = self.verify()
        self.assertEqual(verified['spec_sha256'], self.manifest['spec_sha256'])
        self.assertEqual(canonical(verified['spec']), (self.root / 'review/BENCHMARK-SPEC2.json').read_bytes())
        self.assertIn(str(self.root / 'review/graph_screen.py'), verified['stamps'])

    def test_repinned_fixed_and_configured_spec_mutations_refuse(self):
        raw = (self.root / 'review/BENCHMARK-SPEC2.json').read_bytes()
        changes = [lambda v: v.update(qualification=True), lambda v: v.update(unexpected=1),
                   lambda v: v.pop('scope'), lambda v: v.update(worker_seconds=2279),
                   lambda v: v['server_args'].update(disable_overlap_schedule=1),
                   lambda v: v['http_serving'].update(client_seconds=2039),
                   lambda v: v['http_serving']['expected_devices'].clear(),
                   lambda v: v['http_serving']['bindings']['control'].update(sha256='0' * 64),
                   lambda v: v['graph_projection']['helpers'].update(**{'graph_screen.py': '0' * 64}),
                   lambda v: v.update(physical_slots=True), lambda v: v.pop('gpu_ids')]
        for mutate in changes:
            value = json.loads(raw)
            mutate(value)
            self.write_spec(canonical(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.verify()

    def test_noncanonical_duplicate_and_nonfinite_spec_refuse_even_when_repinned(self):
        raw = (self.root / 'review/BENCHMARK-SPEC2.json').read_bytes()
        for changed in (raw + b' ', b'{"qualification":false,' + raw[1:],
                        raw.replace(b'"worker_seconds":2280', b'"worker_seconds":NaN')):
            self.write_spec(changed)
            with self.assertRaises(ValueError):
                self.verify()

    def test_unsupported_run_and_settings_changes_refuse_when_reapproved(self):
        path = self.root / self.manifest['run_config']
        for key, value in (('instrumentation', 'on'), ('unrecognized', True)):
            run = copy.deepcopy(self.run)
            run[key] = value
            path.write_bytes(canonical(run))
            self.pins['run_config_sha256'] = sha(path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'unsupported run configuration'):
                self.verify()
        path.write_bytes(canonical(self.run))
        self.pins['run_config_sha256'] = sha(path.read_bytes())
        settings = self.root / 'review/FINAL-SETTINGS.json'
        settings.write_bytes(canonical({'instrumentation': 'on'}))
        self.manifest['package_files']['review/FINAL-SETTINGS.json'] = sha(settings.read_bytes())
        with self.assertRaisesRegex(ValueError, 'supported W5 settings'):
            self.verify()

    def test_missing_or_wrong_independent_approval_pins_refuse(self):
        for key in self.pins:
            pins = dict(self.pins, **{key: '0' * 64})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.controller.verify_configured_spec(pins, self.manifest, deadline=205, clock=lambda: 101)
        with self.assertRaisesRegex(ValueError, 'three configuration approval pins'):
            self.controller.verify_configured_spec(None, self.manifest, deadline=205, clock=lambda: 101)

    def test_source_substitution_and_expired_work_refuse(self):
        for name in ('HTTP-BINDING-control.json', 'HTTP-BINDING-mtp.json', 'graph_screen.py',
                     'graph_integration.py', 'graph_outer.py', 'context_tokenizer.py'):
            path = self.root / 'review' / name
            raw = path.read_bytes()
            path.write_bytes(raw + b' ')
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.verify()
            finally:
                path.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.controller.verify_configured_spec(self.pins, self.manifest, deadline=100, clock=lambda: 101)

    def test_changes_during_or_after_reproduction_are_detected(self):
        render = self.controller.render_configured_spec
        def changed(*args):
            result = render(*args)
            (self.root / self.manifest['run_config']).touch()
            return result
        with patch.object(self.controller, 'render_configured_spec', changed), self.assertRaisesRegex(ValueError, 'input changed'):
            self.verify()
        verified = self.verify()
        (self.root / 'review/graph_outer.py').touch()
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.controller.verify_unchanged(verified)

    def test_bad_spec_refuses_before_physical_output_or_lease_construction(self):
        self.manifest['target_root'] = str(self.root)
        self.pins['received_monotonic'] = 0
        with patch.object(self.controller, 'verify_configured_spec', side_effect=ValueError('bad spec')) as verify:
            with self.assertRaisesRegex(ValueError, 'bad spec'):
                self.controller._run_physical_after_retained_closure(manifest_path=str(self.root / 'MANIFEST.json'),
                    manifest_sha256='a' * 64, grant_path=str(self.root / 'GRANT.json'), grant=self.pins,
                    manifest=self.manifest, rpc=None)
        verify.assert_called_once()
        self.assertFalse((self.root / 'results').exists())

    def test_http_binding_requires_the_verified_pin_before_template_access(self):
        spec = json.loads((self.root / 'review/BENCHMARK-SPEC2.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'verified HTTP spec identity'):
            self.controller.http_expected_binding(self.root, spec, owner_arm='bf16',
                                                  session_sha256='a' * 64, spec_sha256='0' * 64)


class ExactSpecProposal(unittest.TestCase):
    def test_exact_patch_pins_and_no_global_spec_override(self):
        before = (BASELINE / CONTROLLER).read_bytes()
        after = (PROPOSAL / CONTROLLER).read_bytes()
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        diff = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
            fromfile='a/' + CONTROLLER, tofile='b/' + CONTROLLER)).encode()
        self.assertEqual(diff, (PROPOSAL / 'D2.patch').read_bytes())
        self.assertEqual((sha(before), sha(after), sha(diff)),
                         tuple(record[key] for key in ('baseline_sha256', 'proposed_sha256', 'patch_sha256')))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'SPEC_SHA256' for n in ast.walk(ast.parse(after))))
        self.assertEqual(sha((HERE.parent / record['spec_fixture']).read_bytes()), record['spec_fixture_sha256'])
        for key in ('owner_approved', 'consumed_by_builder', 'launch_authorized', 'owner_ceilings_changed', 'entire_D2_complete'):
            self.assertFalse(record[key])

    def test_unrelated_functions_imports_and_owner_constants_are_unchanged(self):
        allowed = {'validate_manifest', '_run_bound_controller', 'authenticate_configuration_files',
                   'render_configured_spec', 'verify_configured_spec', 'validate_physical_grant',
                   'checked_physical_resources', 'checked_native_records', 'checked_independent_records',
                   'http_expected_binding', '_run_physical_after_retained_closure'}
        constants = {'SPEC_SHA256', 'SPEC_TEMPLATE_SHA256', 'SUPPORTED_RUN_SHA256', 'SUPPORTED_SETTINGS_SHA256'}
        trees = []
        for path in (BASELINE / CONTROLLER, PROPOSAL / CONTROLLER):
            tree = ast.parse(path.read_bytes())
            tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in allowed)
                and not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
            trees.append(ast.dump(tree))
        self.assertEqual(*trees)


if __name__ == '__main__':
    unittest.main()
