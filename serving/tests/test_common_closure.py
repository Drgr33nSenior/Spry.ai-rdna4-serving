"""Common semantics and authenticated bootstrap without native imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from spry_common import canon,selected
import propose_packet
from test_import_cleanup import stale_bytecode


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    finally:sys.modules.pop(name,None)
    return module


class CommonSemantics(unittest.TestCase):
    def test_encodings_and_raw_pins_match_for_all_supported_value_kinds(self):
        for value in (None,True,False,0,-1,1.25,'é\n雪\u2028',[],{},[1,'two',None],{'z':[True,1.0],'a':{'line':'\r\n'}},('tuple',3)):
            self.assertEqual(selected.canonical_bare(value),canon.canonical_bare(value))
            self.assertEqual(selected.canonical_line(value),canon.canonical_line(value))
            raw=canon.canonical_line(value);self.assertEqual(selected.digest(raw),canon.digest(raw))
        for value in (float('nan'),float('inf'),{'n':float('-inf')},object()):
            with self.assertRaises((ValueError,TypeError)):selected.canonical_line(value)
    def test_error_class_args_and_prefix_laziness(self):
        class CustomError(RuntimeError):pass
        marker=object()
        self.assertIsNone(selected.need(True,marker,error=CustomError,prefix='prefix: '))
        for error in (ValueError,RuntimeError,CustomError):
            for prefix in ('','prefix: '):
                with self.assertRaises(error) as caught:selected.need(False,'message',error=error,prefix=prefix)
                self.assertIs(type(caught.exception),error);self.assertEqual(caught.exception.args,(prefix+'message',))
        with self.assertRaises(ValueError) as caught:selected.need(False,marker)
        self.assertEqual(caught.exception.args,(marker,))
    def test_json_refusals_remain_strict(self):
        for raw in (b'{"a":1,"a":2}',b'NaN',b'Infinity',b'-Infinity'):
            with self.assertRaises(ValueError):selected.parse_json(raw)
    def test_protected_inventory_explicitly_includes_the_common_source(self):
        metadata=json.loads((ROOT/'SOURCE.json').read_bytes());helpers={name:b'fixture' for name in metadata['protected_source_files']}
        pin=hashlib.sha256((ROOT/'spry_common/selected.py').read_bytes()).hexdigest()
        record=dict(shared_common_source=dict(file='spry_common/selected.py',sha256=pin,protected=True))
        result=propose_packet.protected_inventory(helpers,record)
        self.assertEqual(len(result),10);self.assertEqual(result['review/spry_common_canon.py'],pin)
        self.assertEqual(len(propose_packet.protected_inventory(helpers,{})),9)
        for common in ([],dict(protected=False),dict(protected=True,file='spry_common/canon.py')):
            with self.assertRaises(ValueError):propose_packet.protected_inventory(helpers,dict(shared_common_source=common))


class CommonBootstrap(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='protected-common-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.helper=self.root/'spry_common_canon.py'
        self.raw=(ROOT/'spry_common/selected.py').read_bytes();self.helper.write_bytes(self.raw)
        self.consumer=self.root/'benchmark_gpu_identity.py'
        self.consumer.write_bytes((ROOT/'proposals/common-closure/review/benchmark_gpu_identity.py').read_bytes())
    def test_root_uses_checked_common_without_importing_unprotected_contract_or_ambient_module(self):
        trap=ModuleType('mtp_contract')
        def access(name):raise AssertionError('unprotected contract access: '+name)
        trap.__getattr__=access
        fake=ModuleType('_spry_common');fake.need=lambda *a,**k:None
        with patch.dict(sys.modules,mtp_contract=trap,_spry_common=fake,spry_common_canon=fake):
            consumer=load('root_common_positive',self.consumer)
        self.assertIsNot(consumer._SPRY_COMMON,fake)
        with self.assertRaisesRegex(RuntimeError,'expected refusal'):consumer.need(False,'expected refusal')
        self.assertEqual(consumer.canonical({'a':1}),b'{"a":1}\n')
    def test_changed_missing_linked_and_oversized_common_refuse_before_execution(self):
        import os
        for variant in ('changed','missing','symlink','hardlink','oversized'):
            directory=self.root/variant;directory.mkdir();consumer=directory/self.consumer.name
            consumer.write_bytes(self.consumer.read_bytes());helper=directory/self.helper.name
            if variant=='changed':helper.write_bytes(b'raise AssertionError("changed code executed")\n')
            elif variant=='symlink':helper.symlink_to(self.helper)
            elif variant=='hardlink':os.link(self.helper,helper)
            elif variant=='oversized':helper.write_bytes(b'#'*4097)
            try:
                with self.subTest(variant=variant),self.assertRaises((RuntimeError,OSError)):
                    load('root_common_bad_'+variant,consumer)
            finally:
                if variant=='hardlink':helper.unlink()
    def test_stale_bytecode_cannot_replace_checked_source(self):
        stale_bytecode(self.helper)
        consumer=load('root_common_bytecode',self.consumer)
        self.assertEqual(consumer.canonical({'a':1}),b'{"a":1}\n')
        self.assertEqual(consumer._SPRY_COMMON.need.__code__.co_filename,str(self.helper))


if __name__=='__main__':unittest.main()
