"""Exact deletion-only package pruning and separate D2 composition checks."""
import ast
import difflib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_protected_proposal import restore_baseline, sha
from test_config_binding_proposal import canonical

PROPOSAL = HERE.parent / 'proposals/d1-package-pruning'
CONTROLLER = 'review/benchmark_physical_controller.py'
TRANSPORT = 'review/benchmark_host_transport.py'


def patch_parts(raw):
    parts = raw.split(b'--- a/')
    if parts[0] or len(parts) != 3:
        raise ValueError('exact two-file patch required')
    return {part.split(b'\n', 1)[0].decode(): b'--- a/' + part for part in parts[1:]}


class PackagePruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.patches = patch_parts((PROPOSAL / 'D1.patch').read_bytes())
        cls.before = {name: restore_baseline((PROPOSAL / name).read_bytes(), raw, name)
                      for name, raw in cls.patches.items()}

    def test_exact_five_line_removal_and_pins(self):
        self.assertEqual(set(self.patches), {CONTROLLER, TRANSPORT})
        deleted = 0
        for name, before in self.before.items():
            after = (PROPOSAL / name).read_bytes()
            change = self.record['changes'][name]
            self.assertEqual(sha(before), change['baseline_sha256'])
            self.assertEqual(sha(after), change['proposed_sha256'])
            exact = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                fromfile='a/' + name, tofile='b/' + name)).encode()
            self.assertEqual(exact, self.patches[name])
            deleted += len(before.splitlines()) - len(after.splitlines())
        self.assertEqual(deleted, self.record['D1_removed_lines'])
        self.assertEqual(deleted, 5)
        self.assertEqual(self.record['D1_added_lines'], 0)
        self.assertEqual(self.before[CONTROLLER],
                         (HERE.parent / 'proposals/d1-controller-inventory' / CONTROLLER).read_bytes())
        self.assertEqual(sha((PROPOSAL / 'D1.patch').read_bytes()), self.record['D1_patch_sha256'])

    def test_function_bodies_imports_and_remaining_inventory_are_unchanged(self):
        for name, before in self.before.items():
            old, new = ast.parse(before), ast.parse((PROPOSAL / name).read_bytes())
            behavior = lambda tree: [ast.dump(node) for node in tree.body if isinstance(node,
                (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
            self.assertEqual(behavior(old), behavior(new), name)
            if name == CONTROLLER:
                old.body = [node for node in old.body if not (
                    isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == 'LIVE_SOURCE_NAMES' and isinstance(node.value, ast.Tuple)
                    and any(isinstance(item, ast.Constant) and item.value == 'divergence_capture.py'
                            for item in node.value.elts)) and not (
                    isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and any(isinstance(item, ast.Constant) and item.value == 'review/CAPTURE-BUILD-PROOF.json'
                            for item in ast.walk(node)))]
            else:
                assignment = next(node for node in old.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'D3B_TARGET_FILES' for t in node.targets))
                index = next(i for i, key in enumerate(assignment.value.keys)
                             if key.value == 'review/capture_model_runner.py')
                del assignment.value.keys[index]
                del assignment.value.values[index]
            self.assertEqual(ast.dump(old), ast.dump(new), name)

    def test_composed_D2_is_exactly_the_previous_changes_with_same_deletions(self):
        parts = patch_parts((PROPOSAL / 'D2.patch').read_bytes())
        for name, previous in ((CONTROLLER, 'd2-config-binding'), (TRANSPORT, 'd2-transport-binding')):
            before = (PROPOSAL / name).read_bytes()
            integrated = (PROPOSAL / 'integration' / name).read_bytes()
            prior = (HERE.parent / 'proposals' / previous / name).read_bytes()
            # Re-create the deletion patch with the composed source's shifted hunk offsets.
            diff = ''.join(difflib.unified_diff(prior.decode().splitlines(True), integrated.decode().splitlines(True),
                fromfile='a/' + name, tofile='b/' + name)).encode()
            self.assertEqual(restore_baseline(integrated, diff, name), prior)
            removed = lambda raw: [line for line in raw.splitlines()
                                   if line.startswith(b'-') and not line.startswith(b'---')]
            self.assertEqual(removed(diff), removed(self.patches[name]))
            exact = ''.join(difflib.unified_diff(before.decode().splitlines(True), integrated.decode().splitlines(True),
                fromfile='a/' + name, tofile='b/' + name)).encode()
            self.assertEqual(parts[name], exact)
            self.assertEqual(sha(integrated), self.record['changes'][name]['integrated_sha256'])
        self.assertEqual(sha((PROPOSAL / 'D2.patch').read_bytes()), self.record['D2_patch_sha256'])

    def test_inventory_and_both_protected_sets_are_exact_and_isolated(self):
        source = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        inventory_raw = (PROPOSAL / 'PACKAGE-INVENTORY.json').read_bytes()
        inventory = json.loads(inventory_raw)
        self.assertEqual(canonical(inventory), inventory_raw)
        self.assertEqual(sha(inventory_raw), self.record['package_inventory_sha256'])
        self.assertEqual(len(inventory), self.record['pruned_package_members'])
        self.assertEqual(len(inventory), 74)
        self.assertFalse(set(inventory) & set(self.record['removed_package_files']))
        for label, key in (('proposed', 'proposed_sha256'), ('integrated', 'integrated_sha256')):
            pins = dict(source['protected_source_files'])
            pins.update({Path(name).name: row[key] for name, row in self.record['changes'].items()})
            self.assertEqual(self.record[label + '_protected_files'], pins)
            self.assertEqual(sha(canonical({'review/' + name: pin for name, pin in pins.items()})),
                             self.record[label + '_protected_set_sha256'])
        for key in ('consumed_by_builder', 'owner_approved', 'launch_authorized', 'entire_D1_complete',
                    'entire_D2_complete', 'owner_ceilings_changed', 'cleaned_source_archive_fits'):
            self.assertFalse(self.record[key], key)

    def test_added_code_or_changed_patch_context_is_rejected(self):
        for name, patch in self.patches.items():
            with self.assertRaises(ValueError):
                restore_baseline((PROPOSAL / name).read_bytes(), patch + b'+raise RuntimeError()\n', name)
            with self.assertRaises(ValueError):
                restore_baseline((PROPOSAL / name).read_bytes(), patch, 'review/other.py')


if __name__ == '__main__':
    unittest.main()
