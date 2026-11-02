"""Keep the cleaned inventory change separate from D1 and runtime policy."""
import ast
import difflib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_config_binding_proposal import canonical, sha

PROPOSAL = HERE.parent / 'proposals/d2-cleaned-inventory'
BASELINE = HERE.parent / 'proposals/d1-package-pruning'
CONTROLLER = 'review/benchmark_physical_controller.py'
TRANSPORT = 'review/benchmark_host_transport.py'


class CleanedInventory(unittest.TestCase):
    def test_exact_patch_and_composed_protected_set(self):
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        before = (BASELINE / 'integration' / CONTROLLER).read_bytes()
        after = (PROPOSAL / CONTROLLER).read_bytes()
        patch = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
            fromfile='a/' + CONTROLLER, tofile='b/' + CONTROLLER)).encode()
        self.assertEqual(patch, (PROPOSAL / 'D2.patch').read_bytes())
        for key, raw in (('baseline_sha256', before), ('proposed_sha256', after), ('patch_sha256', patch)):
            self.assertEqual(record[key], sha(raw))
        prior = json.loads((BASELINE / 'PROPOSAL.json').read_bytes())
        pins = dict(prior['integrated_protected_files'])
        pins[Path(CONTROLLER).name] = sha(after)
        self.assertEqual(pins, record['protected_files'])
        self.assertEqual(sha((BASELINE / 'integration' / TRANSPORT).read_bytes()), pins[Path(TRANSPORT).name])
        self.assertEqual(sha(canonical({'review/' + name: pin for name, pin in pins.items()})),
                         record['protected_set_sha256'])
        for key in ('owner_approved', 'consumed_by_builder', 'launch_authorized',
                    'owner_ceilings_changed', 'entire_D2_complete'):
            self.assertFalse(record[key], key)

    def test_only_the_seven_named_source_inventory_entries_change(self):
        before = ast.parse((BASELINE / 'integration' / CONTROLLER).read_bytes())
        after = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
        removed = {'batch_telemetry.py', 'cache_profile.py', 'prepare_benchmark_package.py'}
        added = {'observer_health.py', 'target_kv_metadata.py', 'kv_graph_join.py', 'request_receipts.py'}
        for node in before.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id in ('SOURCE_NAMES', 'LIVE_SOURCE_NAMES') for t in node.targets):
                for value in ast.walk(node.value):
                    if isinstance(value, ast.Tuple):
                        value.elts = [item for item in value.elts if not
                            (isinstance(item, ast.Constant) and item.value in removed)]
        additions = [node for node in after.body if isinstance(node, ast.AugAssign)
                     and isinstance(node.target, ast.Name) and node.target.id == 'LIVE_SOURCE_NAMES'
                     and isinstance(node.value, ast.Tuple)
                     and {item.value for item in node.value.elts} == added]
        self.assertEqual(len(additions), 1)
        after.body.remove(additions[0])
        # Includes all function bodies, imports, constants, ceilings and refusal checks.
        self.assertEqual(ast.dump(before), ast.dump(after))


if __name__ == '__main__':
    unittest.main()
