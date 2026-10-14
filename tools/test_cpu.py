"""Run the retained112-case source contract selection; never import a GPU runtime."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.test-build/tmp'
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'tests/reference')]

def inventory():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ('prototype', 'tests') for path in sorted((ROOT / folder).rglob('*'))
            if path.is_file()}

before = inventory()
name = 'sglang.srt.mem_cache.unified_cache.component_type'
path = ROOT / 'prototype/image-source/python/sglang/srt/mem_cache/unified_cache/component_type.py'
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules[name] = module

def original_native_snapshot():
    import test_modern_stack
    with patch.object(test_modern_stack, 'OWNED', ROOT / 'tests/fixtures/original-native'):
        test_modern_stack.ModernTests('test_native_inputs_exact_accepted_hashes').test_native_inputs_exact_accepted_hashes()

suite = unittest.TestSuite()
for name in ('test_lifecycle0521', 'test_allocation0521', 'test_unified_tree0521', 'test_cache0521',
             'test_modern_stack', 'test_strided_caller', 'test_bf16_integration', 'test_bf16_closure'):
    loaded = unittest.defaultTestLoader.loadTestsFromName(name)
    if name == 'test_modern_stack':
        for group in loaded:
            for test in group:
                if test._testMethodName == 'test_native_inputs_exact_accepted_hashes':
                    suite.addTest(unittest.FunctionTestCase(original_native_snapshot))
                else:
                    suite.addTest(test)
    else:
        suite.addTests(loaded)
for name in ('test_positive_exact_modern_inventory_and_patch', 'test_inventory_exact_missing_extra_changed_and_native',
             'test_baseline_image_schema_and_added_accounting_refuse', 'test_installed_interpreter_library_and_roles_binding',
             'test_receipt_positive_and_independent_mutations', 'test_receipt_duplicate_paths_and_missing_argument_refuse',
             'test_actual_runtime_gate_modern_positive_and_partial_refusal', 'test_actual_v0521_gated_radix_hybrid_and_parent_writer'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_modern_identity.ModernIdentityTests.' + name))
assert suite.countTestCases() == 90
for name in ('test_kiss_repairs', 'test_runner_activation'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
assert suite.countTestCases() == 112
result = unittest.TextTestRunner(verbosity=2).run(suite)
for folder in ('prototype', 'tests', 'tools'):
    for path in (ROOT / folder).rglob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
after = inventory()
summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'skips': len(result.skipped), 'source_continuity': before == after,
    'torch_imported': 'torch' in sys.modules, 'sglang_package_imported': 'sglang' in sys.modules,
    'native_execution': False}
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped and before == after
    and not summary['torch_imported'] and not summary['sglang_package_imported'] else 1)
