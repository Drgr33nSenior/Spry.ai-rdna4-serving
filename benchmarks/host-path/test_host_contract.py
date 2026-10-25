"""Source-bound tests of changed worker and retained reader predicates."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import test_host_path as host

ROOT = Path(__file__).resolve().parent


def predicate(filename, function, message, args):
    source = filename.read_text()
    # Check the complete module under the executing interpreter first.
    compile(source, str(filename), 'exec', dont_inherit=True)
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    matches = []
    for node in ast.walk(definition):
        if isinstance(node, ast.If) and any(isinstance(n, ast.Constant) and n.value == message for n in ast.walk(node)):
            matches.append(node)
    # The smallest containing if is the unchanged producer/consumer block.
    selected = min(matches, key=lambda node: node.end_lineno - node.lineno)
    wrapper = ast.parse('def check(' + ','.join(args) + '):\n    pass\n')
    wrapper.body[0].body = [copy.deepcopy(selected)]
    namespace = {'_require': host.entry._require, '_need': host.graph.need}
    exec(compile(ast.fix_missing_locations(wrapper), str(filename), 'exec', dont_inherit=True), namespace)
    return namespace, namespace['check']


class HostContractTests(unittest.TestCase):
    def test_actual_worker_and_reader_log_predicates(self):
        worker_ns, worker = predicate(ROOT / 'author/review/worker_entry.py', '_mtp_preflight',
            'closure native observability configuration differs', ['cfg', 'contract_module', 'manifest', 'value'])
        reader_ns, reader = predicate(ROOT / 'LOGICAL-serving_integration.py', '_validate_native_observations',
            'closure native rank observability differs', ['mtp', 'expected'])
        reader_ns['mtp_contract'] = host.contract
        for arm, wanted in [('control', 1), ('mtp', 100)]:
            flags = {'enable_metrics': True, 'decode_log_interval': wanted}
            result = {}
            worker(SimpleNamespace(**flags), host.contract, {'arm': arm}, result)
            self.assertEqual(result, {'resolved_observability': flags})
            reader(result, {'arm': arm})
            for interval in [None, False, True, str(wanted), float(wanted), 101, 100 if arm == 'control' else 1]:
                bad = dict(flags, decode_log_interval=interval)
                with self.assertRaises(ValueError):
                    worker(SimpleNamespace(**bad), host.contract, {'arm': arm}, {})
                with self.assertRaises(ValueError):
                    reader({'resolved_observability': bad}, {'arm': arm})
            for metrics in [False, 1, None]:
                bad = dict(flags, enable_metrics=metrics)
                with self.assertRaises(ValueError):
                    worker(SimpleNamespace(**bad), host.contract, {'arm': arm}, {})
                with self.assertRaises(ValueError):
                    reader({'resolved_observability': bad}, {'arm': arm})


if __name__ == '__main__':
    unittest.main(verbosity=2)
