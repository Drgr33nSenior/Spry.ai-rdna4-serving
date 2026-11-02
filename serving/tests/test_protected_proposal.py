"""Verify the exact, inactive D1 proposal without access to private evidence."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent / 'proposals/d1-controller-inventory'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def restore_baseline(proposed, patch, filename='review/benchmark_physical_controller.py'):
    """Reverse this deletion-only unified diff, checking every context line."""
    lines = proposed.splitlines(keepends=True)
    restored, cursor, old_count, new_count = [], 0, 0, 0
    expected = None
    rows = patch.splitlines(keepends=True)
    if rows[:2] != [('--- a/' + filename + '\n').encode(),
                    ('+++ b/' + filename + '\n').encode()]:
        raise ValueError('exact source patch required')
    for row in rows[2:]:
        if row.startswith(b'@@ '):
            if expected is not None and (old_count, new_count) != expected:
                raise ValueError('hunk length differs')
            match = re.fullmatch(rb'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', row)
            if match is None:
                raise ValueError('explicit hunk counts required')
            old_start, old_size, new_start, new_size = map(int, match.groups())
            if new_start - 1 < cursor:
                raise ValueError('overlapping hunks')
            restored.extend(lines[cursor:new_start - 1])
            cursor = new_start - 1
            if len(restored) != old_start - 1:
                raise ValueError('original hunk offset differs')
            expected, old_count, new_count = (old_size, new_size), 0, 0
        elif expected is not None and row.startswith(b'-'):
            restored.append(row[1:])
            old_count += 1
        elif expected is not None and row.startswith(b' '):
            if cursor >= len(lines) or lines[cursor] != row[1:]:
                raise ValueError('patch context differs')
            restored.append(lines[cursor])
            cursor += 1
            old_count += 1
            new_count += 1
        else:
            raise ValueError('D1 must contain only deletions and context')
    if expected is None or (old_count, new_count) != expected:
        raise ValueError('final hunk length differs')
    restored.extend(lines[cursor:])
    return b''.join(restored)


class ProtectedProposal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.proposed = (PROPOSAL / cls.record['changed_file']).read_bytes()
        cls.patch = (PROPOSAL / 'D1.patch').read_bytes()
        cls.baseline = restore_baseline(cls.proposed, cls.patch)

    def test_exact_deletion_patch_and_source_pins(self):
        for key, raw in [('baseline_sha256', self.baseline),
                         ('proposed_sha256', self.proposed), ('patch_sha256', self.patch)]:
            self.assertEqual(sha(raw), self.record[key], key)
        exact = ''.join(difflib.unified_diff(
            self.baseline.decode().splitlines(keepends=True),
            self.proposed.decode().splitlines(keepends=True),
            fromfile='a/' + self.record['changed_file'],
            tofile='b/' + self.record['changed_file'])).encode()
        self.assertEqual(self.patch, exact)
        self.assertEqual(self.record['added_lines'], 0)
        self.assertEqual(len(self.baseline.splitlines()) - len(self.proposed.splitlines()),
                         self.record['removed_lines'])

    def test_remaining_function_sources_are_byte_identical(self):
        functions = []
        for raw in (self.baseline, self.proposed):
            text = raw.decode()
            functions.append({node.name: ast.get_source_segment(text, node)
                for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)})
        self.assertEqual(set(functions[0]) - set(functions[1]), {'run_controller'})
        self.assertEqual(functions[1], {name: text for name, text in functions[0].items()
                                      if name != 'run_controller'})
        compile(self.proposed, 'proposal-controller.py', 'exec')

    def test_constants_imports_and_ceilings_are_unchanged(self):
        definitions = []
        for raw in (self.baseline, self.proposed):
            definitions.append([ast.dump(node) for node in ast.parse(raw).body
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom))])
        self.assertEqual(*definitions)

    def test_proposal_is_isolated_from_the_consumed_protected_set(self):
        source = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        baseline = self.record['baseline_protected_files']
        proposed = self.record['proposed_protected_files']
        self.assertEqual(baseline, source['protected_source_files'])
        self.assertEqual(len(baseline), 9)
        self.assertEqual([name for name in baseline if baseline[name] != proposed[name]],
                         ['benchmark_physical_controller.py'])
        self.assertEqual(self.record['source_seal_sha256'], source['source_seal_sha256'])
        for label, pins in [('baseline', baseline), ('proposed', proposed)]:
            raw = (json.dumps({'review/' + name: pin for name, pin in pins.items()},
                sort_keys=True, separators=(',', ':')) + '\n').encode()
            self.assertEqual(sha(raw), self.record[label + '_protected_set_sha256'])
        self.assertFalse(self.record['owner_approved'])
        self.assertFalse(self.record['launch_authorized'])
        self.assertFalse(self.record['entire_D1_complete'])
        self.assertFalse((HERE.parent / 'legacy' / self.record['changed_file']).exists())

    def test_added_code_and_changed_context_are_refused(self):
        for bad in (self.patch + b'+print("added")\n',
                    self.patch.replace(b'     return report\n', b'     return changed\n')):
            with self.subTest(patch=sha(bad)), self.assertRaises(ValueError):
                restore_baseline(self.proposed, bad)


if __name__ == '__main__':
    unittest.main()
