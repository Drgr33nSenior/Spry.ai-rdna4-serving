"""Actual fingerprint expressions preserve code identity and ignore stale bytecode."""
import ast
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_import_cleanup import stale_bytecode
from test_contract_cleanup import load

SOURCE=HERE.parent/'legacy/review'


class SourceFingerprints(unittest.TestCase):
    def test_all_six_actual_expressions_keep_the_previous_code_objects_without_execution(self):
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-fingerprint-')
        self.addCleanup(temporary.cleanup);root=Path(temporary.name).resolve()
        path=root/'native_source.py'
        raw=(b'from __future__ import annotations\n'
             b'class Runner:\n    def method(self, item: MissingType):\n        return item\n'
             b'def factory(value):\n    def closure():\n        return value\n    return closure\n'
             b'raise AssertionError("source must never execute")\n')
        path.write_bytes(raw);stale_bytecode(path)
        source=dict(module='pinned.native_source',path=str(path))
        scope=dict(Path=Path,SourceFileLoader=SourceFileLoader,source=source,path=path,raw=raw,
            self=SimpleNamespace(source=source,sources={'head_logits':source}))
        expected=compile(raw,str(path),'exec',dont_inherit=True)
        self.assertNotEqual(SourceFileLoader(source['module'],str(path)).get_code(source['module']),expected)
        count=0
        for filename in ('worker_entry.py','j01_probe.py','verify_decode_adapter.py'):
            for node in ast.walk(ast.parse((SOURCE/filename).read_bytes())):
                if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                        and node.func.attr=='source_to_code' and isinstance(node.func.value,ast.Call)
                        and isinstance(node.func.value.func,ast.Name) and node.func.value.func.id=='SourceFileLoader'):
                    continue
                with self.subTest(file=filename,line=node.lineno):
                    value=eval(compile(ast.Expression(node),filename,'eval',dont_inherit=True),scope)
                    self.assertEqual(value,expected);self.assertEqual(value.co_filename,str(path))
                count+=1
        self.assertEqual(count,6)

    def test_actual_adapter_fingerprint_ignores_stale_bytecode_and_rejects_substitution(self):
        temporary=tempfile.TemporaryDirectory(prefix='cleanup-method-fingerprint-')
        self.addCleanup(temporary.cleanup);path=Path(temporary.name).resolve()/'native_method.py'
        raw=b'class Runner:\n    def method(self, item):\n        return item\n'
        path.write_bytes(raw);stale_bytecode(path)
        module=ModuleType('pinned.native_method');exec(compile(raw,str(path),'exec',dont_inherit=True),module.__dict__)
        adapter=load('cleanup_fingerprint_adapter',SOURCE/'verify_decode_adapter.py')
        source=dict(module=module.__name__,path=str(path))
        original=module.Runner.__dict__['method']
        self.assertIs(adapter.checked_method(module.Runner,'method',source),original)
        changed=ModuleType(module.__name__)
        exec(compile(b'class Runner:\n    def method(self,item):\n        return None\n',str(path),'exec',dont_inherit=True),changed.__dict__)
        module.Runner.method=changed.Runner.__dict__['method']
        with self.assertRaisesRegex(ValueError,'adapter native method changed'):
            adapter.checked_method(module.Runner,'method',source)

    def test_all_canonical_review_sources_have_no_direct_exec_or_compile_calls(self):
        for path in SOURCE.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,('exec','compile'),str(path))


if __name__=='__main__':unittest.main()
