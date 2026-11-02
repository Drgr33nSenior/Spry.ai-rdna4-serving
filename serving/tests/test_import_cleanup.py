"""Receipt I/O and authenticated imports through complete CPU-only modules."""
import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import py_compile
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import SOURCE, canonical, load


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale_bytecode(path):
    raw = path.read_bytes()
    path.write_text('raise RuntimeError("untrusted cached code")\n')
    py_compile.compile(str(path), doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    path.write_bytes(raw)


class Copies(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-imports-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def copy(self, *names):
        for name in names:
            shutil.copyfile(SOURCE / name, self.root / name)

    def client(self):
        with patch.dict(sys.modules, serving_contract=SimpleNamespace(canonical=lambda value: canonical(value) + b'\n')):
            return load('cleanup_client', self.root / 'serving_client.py')


class ReceiptModules(Copies):
    def setUp(self):
        super().setUp()
        self.copy('serving_client.py', 'request_receipts.py')
        self.module = self.client()
        self.receipts = self.module._kv_receipts
        self.block = self.root / 'output' / 'paired'

    def test_actual_client_write_read_and_inventory_preserve_bytes(self):
        fields = {'requests/00-sample.raw': b'{"prompt":"test"}\n',
                  'requests/00-sample.intent.json': b'{"prepared":1}\n',
                  'responses/00-sample.partial.raw': b'',
                  'responses/00-sample.record.json': b'{"status":200}\n'}
        for name, raw in fields.items():
            self.module._write_exclusive(self.block / name, raw)
            self.assertFalse((self.block / name).exists())
        for name, raw in fields.items():
            self.assertEqual(self.receipts.read(self.block / name), raw)
        self.assertEqual(self.receipts.logical_names(self.block), set(fields))
        envelope = self.block / 'receipts' / '00-sample.json'
        self.assertEqual(set(self.receipts.decode(envelope.read_bytes())[1]), set(fields))
        self.assertEqual(Path(self.receipts.__file__), self.root / 'request_receipts.py')

    def test_concurrent_fields_keep_both_write_once_records(self):
        names = ['requests/00-sample.raw', 'responses/00-sample.record.json']
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda name: self.module._write_exclusive(self.block / name, name.encode()), names))
        self.assertEqual(self.receipts.logical_names(self.block), set(names))
        with self.assertRaisesRegex(ValueError, 'write-once'):
            self.module._write_exclusive(self.block / names[0], b'replace')

    def test_receipt_digest_and_canonical_encoding_remain_required(self):
        path = self.block / 'requests' / '00-sample.raw'
        envelope = self.module._write_exclusive(path, b'original')
        value = json.loads(envelope.read_bytes())
        with self.assertRaisesRegex(ValueError, 'canonical receipt'):
            self.receipts.decode(json.dumps(value).encode())
        value['fields']['requests/00-sample.raw']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bytes/hash'):
            self.receipts.decode(self.receipts.canonical(value))

    def test_member_bound_and_failed_publication_preserve_existing_record(self):
        path = self.block / 'requests' / '00-sample.raw'
        with self.assertRaisesRegex(ValueError, 'bounded raw'):
            self.module._write_exclusive(path, b'x' * (self.receipts.MEMBER_LIMIT + 1))
        envelope = self.module._write_exclusive(path, b'original')
        original = envelope.read_bytes()
        with patch.object(self.receipts.os, 'replace', side_effect=OSError('injected')), self.assertRaises(OSError):
            self.module._write_exclusive(self.block / 'responses' / '00-sample.record.json', b'new')
        self.assertEqual(envelope.read_bytes(), original)
        self.assertTrue(envelope.with_suffix('.next').is_file())
        with self.assertRaises(ValueError):
            self.receipts.logical_names(self.block)

    def test_receipt_source_missing_changed_and_linked_refuse(self):
        path = self.root / 'request_receipts.py'
        original = path.read_bytes()
        path.write_text('raise RuntimeError("untrusted source")\n')
        with self.assertRaisesRegex(ValueError, 'source binding'):
            self.client()
        path.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.client()
        path.symlink_to(SOURCE / path.name)
        with self.assertRaises(ValueError):
            self.client()
        path.unlink()
        path.write_bytes(original)
        os.link(path, path.with_suffix('.alias'))
        with self.assertRaises(ValueError):
            self.client()

    def test_receipt_source_ignores_unchecked_cached_bytecode(self):
        stale_bytecode(self.root / 'request_receipts.py')
        self.assertTrue(callable(self.client()._kv_receipts.write))


class ReaderImports(Copies):
    def setUp(self):
        super().setUp()
        self.copy('serving_integration.py', 'kv_graph_join.py')
        self.reader = load('cleanup_reader', self.root / 'serving_integration.py')

    def source(self, name, raw):
        path = self.root / (name + '.py')
        path.write_text(raw)
        return dict(path=str(path), sha256=pin(path))

    def test_contract_scope_registers_self_and_restores_previous_module(self):
        source = self.source('mtp_contract', 'import mtp_contract\nSELF = mtp_contract\n')
        previous = SimpleNamespace(previous=True)
        with patch.dict(sys.modules, mtp_contract=previous):
            with self.reader._gdn_contract_scope({'mtp_contract': source}) as module:
                self.assertIs(module.SELF, module)
                self.assertIs(sys.modules['mtp_contract'], module)
            self.assertIs(sys.modules['mtp_contract'], previous)

    def test_contract_scope_restores_registry_after_process_control_exception(self):
        source = self.source('mtp_contract', 'raise KeyboardInterrupt("injected")\n')
        with patch.dict(sys.modules):
            sys.modules.pop('mtp_contract', None)
            with self.assertRaises(KeyboardInterrupt), self.reader._gdn_contract_scope({'mtp_contract': source}):
                self.fail('failed contract yielded')
            self.assertNotIn('mtp_contract', sys.modules)

    def test_graph_reader_loads_only_pinned_source_and_ignores_pyc(self):
        source = self.source('serving_graph_observer',
            'def validate_gdn_record(*args, **kwargs): return "checked"\ndef canonical(value): return b"record"\n')
        source['module'] = 'serving_graph_observer'
        stale_bytecode(self.root / 'serving_graph_observer.py')
        module = self.reader._gdn_observer({'graph_observer': source}, require_gdn=False)
        self.assertEqual(module.validate_gdn_record(), 'checked')
        source['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.reader._gdn_observer({'graph_observer': source}, require_gdn=False)

    def test_graph_frontend_preserves_real_screen_and_rejects_substitution(self):
        self.copy('graph_integration.py', 'graph_screen.py')
        stale_bytecode(self.root / 'graph_screen.py')
        graph = load('cleanup_graph_frontend', self.root / 'graph_integration.py')
        self.assertEqual(graph.screen.canonical({'b': 2, 'a': 1}), b'{"a":1,"b":2}\n')
        (self.root / 'graph_screen.py').write_text('raise RuntimeError("untrusted screen")\n')
        with self.assertRaisesRegex(ValueError, 'exact adjacent'):
            load('cleanup_bad_graph_frontend', self.root / 'graph_integration.py')


class ModelImports(Copies):
    NAMES = ('mtp_contract', 'serving_contract', 'serving_client', 'serving_parity',
             'worker_entry', 'serving_launcher', 'serving_entry', 'serving_runtime')

    def setUp(self):
        super().setUp()
        self.copy('model_benchmark.py')
        self.model = load('cleanup_model_frontend', self.root / 'model_benchmark.py')

    @contextlib.contextmanager
    def http_tree(self, failure=None, *, instrumented=True):
        sources = {}
        for name in self.NAMES:
            raw = 'import sys\nSELF = sys.modules[__name__]\nVALUE = ' + repr(name) + '\n'
            if name == 'mtp_contract':
                # The bootstrap module is not registered; only ordinary imports use SELF.
                raw = 'import sys\nVALUE = "mtp_contract"\n'
                raw += ('def owner_arm_map(): return {"bf16":"control", "kv4":"mtp"}\n'
                        'def expand_http_template(value): return value\n'
                        'def validate_contract(value): return value\n'
                        'def contract_for_arm(arm): return {"selected":"speculation"}\n')
                raw += 'def instrumentation_enabled(): return ' + repr(instrumented) + '\n'
            if name == 'serving_runtime':
                raw += 'import mtp_contract\nDEPENDENCY = mtp_contract.VALUE\n'
            if name == failure:
                raw += 'raise KeyboardInterrupt("injected import failure")\n'
            path = self.root / (name + '.py')
            path.write_text(raw)
            sources[name] = dict(path=str(path), sha256=pin(path))
        speculation = {'selected': 'speculation'}
        outer = dict(schema=1, mode='native-http-fp8-sprint01-v1', arm='control', port=18087,
            source_manifest_sha256='a' * 64, speculation=speculation,
            binding_template=dict(sources=sources, speculation=speculation), worker_template=dict(sources=sources))
        path = self.root / 'HTTP-BINDING-control.json'
        path.write_bytes(canonical(outer))
        spec = dict(http_serving=dict(bindings=dict(control=dict(
            path='review/HTTP-BINDING-control.json', sha256=pin(path)))))
        checked = self.model.checked_file
        def local_template(path, expected, maximum=8 << 20):
            if str(path) == '/mnt/review/HTTP-BINDING-control.json':
                path = self.root / Path(path).name
            return checked(path, expected, maximum)
        with patch.dict(sys.modules), patch.object(self.model, 'checked_file', side_effect=local_template):
            for name in self.NAMES:
                sys.modules.pop(name, None)
            yield spec

    def test_http_chain_registers_dependencies_and_binds_each_loader_to_its_own_bytes(self):
        with self.http_tree() as spec:
            for name in self.NAMES:
                stale_bytecode(self.root / (name + '.py'))
            runtime = self.model.http_runtime_module(spec, 'bf16')
            self.assertEqual(runtime.DEPENDENCY, 'mtp_contract')
            self.assertIs(runtime.SELF, runtime)
            for name in self.NAMES:
                path = self.root / (name + '.py')
                code = sys.modules[name].__spec__.loader.get_code(name)
                self.assertEqual(code, compile(path.read_bytes(), str(path), 'exec', dont_inherit=True))

    def test_http_chain_rolls_back_every_registered_helper_on_failure(self):
        with self.http_tree(failure='serving_runtime') as spec:
            with self.assertRaises(KeyboardInterrupt):
                self.model.http_runtime_module(spec, 'bf16')
            self.assertFalse(set(self.NAMES) & sys.modules.keys())

    def test_production_does_not_read_or_import_the_parity_source(self):
        with self.http_tree(instrumented=False) as spec:
            (self.root/'serving_parity.py').write_text('raise AssertionError("production parity import")\n')
            runtime=self.model.http_runtime_module(spec,'bf16')
            self.assertEqual(runtime.DEPENDENCY,'mtp_contract')
            self.assertNotIn('serving_parity',sys.modules)
            self.assertEqual(set(self.NAMES)-set(sys.modules),{'serving_parity'})

    def test_production_rejects_an_ambient_parity_module(self):
        with self.http_tree(instrumented=False) as spec:
            previous=SimpleNamespace(ambient=True);sys.modules['serving_parity']=previous
            with self.assertRaisesRegex(ValueError,'already imported'):
                self.model.http_runtime_module(spec,'bf16')
            self.assertIs(sys.modules['serving_parity'],previous)
            self.assertNotIn('mtp_contract',sys.modules)

    def test_observation_mode_must_be_a_boolean(self):
        with self.http_tree(instrumented=0) as spec:
            with self.assertRaisesRegex(ValueError,'mode must be boolean'):
                self.model.http_runtime_module(spec,'bf16')
            self.assertFalse(set(self.NAMES)&sys.modules.keys())

    def test_changed_observation_mode_rolls_back_the_production_import_chain(self):
        with self.http_tree(instrumented=False) as spec:
            execute=SourceFileLoader.exec_module
            def changed(loader,module):
                execute(loader,module)
                if module.__name__=='mtp_contract':module.instrumentation_enabled=lambda:True
            with patch.object(SourceFileLoader,'exec_module',new=changed):
                with self.assertRaisesRegex(ValueError,'observation mode changed'):
                    self.model.http_runtime_module(spec,'bf16')
            self.assertFalse(set(self.NAMES)&sys.modules.keys())

    def test_http_chain_refuses_ambient_module_without_replacing_it(self):
        with self.http_tree() as spec:
            previous = SimpleNamespace(ambient=True)
            sys.modules['serving_client'] = previous
            with self.assertRaisesRegex(ValueError, 'already imported'):
                self.model.http_runtime_module(spec, 'bf16')
            self.assertIs(sys.modules['serving_client'], previous)
            self.assertNotIn('mtp_contract', sys.modules)

    def test_http_chain_rejects_source_substitution_before_execution(self):
        with self.http_tree() as spec:
            (self.root / 'worker_entry.py').write_text('raise RuntimeError("untrusted worker")\n')
            with self.assertRaisesRegex(ValueError, 'input digest'):
                self.model.http_runtime_module(spec, 'bf16')
            self.assertFalse(set(self.NAMES) & sys.modules.keys())

    def test_adjacent_loader_executes_checked_bytes_when_named_file_changes(self):
        path = self.root / 'sample.py'
        path.write_text('VALUE = "verified"\n')
        expected = pin(path)
        execute = SourceFileLoader.exec_module
        def replace_file(loader, module):
            path.write_text('raise RuntimeError("changed after verification")\n')
            return execute(loader, module)
        with patch.object(SourceFileLoader, 'exec_module', new=replace_file):
            self.assertEqual(self.model.authenticated_adjacent('sample.py', expected).VALUE, 'verified')

    def test_inactive_graph_route_retains_its_existing_source_pin_refusal(self):
        self.copy('graph_integration.py', 'graph_screen.py')
        with self.assertRaisesRegex(ValueError, 'input digest'):
            self.model._GRAPH.MODE
        self.assertIsNone(self.model._GRAPH.module)

    def test_refactored_frontends_have_no_direct_exec_or_embedded_source(self):
        for name in ('serving_client.py', 'model_benchmark.py', 'graph_integration.py', 'serving_integration.py'):
            for node in ast.walk(ast.parse((SOURCE / name).read_bytes())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ('exec', 'compile'), name)


if __name__ == '__main__':
    unittest.main()
