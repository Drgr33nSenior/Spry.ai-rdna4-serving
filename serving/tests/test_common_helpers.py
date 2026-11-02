"""Shared canonical bytes and the retained graph projection API; CPU only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from test_contract_cleanup import load
from test_import_cleanup import stale_bytecode
import propose_packet


class CommonHelpers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='shared-common-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.helper = self.root/'spry_common_canon.py'
        self.helper.write_bytes((ROOT/'spry_common/canon.py').read_bytes())
        self.contract_path = self.root/'mtp_contract.py'
        self.contract_path.write_bytes((ROOT/'spry_serving/shared_contract.py').read_bytes())

    def contract(self): return load('shared_common_test_contract', self.contract_path)

    def test_both_json_encodings_and_raw_hash_keep_exact_previous_bytes(self):
        contract = self.contract()
        values = [None, True, False, 0, -1, 1.25, 'é\n雪\u2028', [], {}, [1, 'two', None],
                  {'z': [True, 1.0], 'a': {'line': '\r\n'}}, ('tuple', 3)]
        for value in values:
            with self.subTest(value=value):
                old = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                self.assertEqual(contract.canonical_bare(value), old)
                self.assertEqual(contract.json_line(value), old+b'\n')
                self.assertEqual(contract.digest_bytes(old), hashlib.sha256(old).hexdigest())
        for value in [float('nan'), float('inf'), {'n': float('-inf')}, object()]:
            with self.subTest(value=repr(value)), self.assertRaises((ValueError, TypeError)):
                contract.json_line(value)

    def test_receipts_share_the_authenticated_contract_functions(self):
        contract = self.contract()
        with patch.dict(sys.modules, mtp_contract=contract):
            receipts = load('shared_common_test_receipts', ROOT/'spry_evidence/shared_receipts.py')
        self.assertIs(receipts.canonical, contract.json_line)
        self.assertIs(receipts.sha, contract.digest_bytes)
        self.assertIs(contract.json_line, contract._SHARED_CANON.canonical_line)
        self.assertEqual(Path(contract._SHARED_CANON.__file__), self.helper)

    def test_missing_changed_linked_and_oversized_helpers_refuse_before_execution(self):
        raw = self.helper.read_bytes()
        for kind in ('missing', 'changed', 'symlink', 'hardlink', 'oversized'):
            with self.subTest(kind=kind):
                directory = self.root/kind; directory.mkdir()
                path = directory/'spry_common_canon.py'
                (directory/'mtp_contract.py').write_bytes(self.contract_path.read_bytes())
                if kind == 'changed': path.write_bytes(b'raise AssertionError("must not execute")\n')
                elif kind == 'symlink': path.symlink_to(self.helper)
                elif kind == 'hardlink':
                    import os
                    os.link(self.helper, path)
                elif kind == 'oversized': path.write_bytes(b'#' * 4097)
                with self.assertRaises((ValueError, OSError)):
                    load('invalid_shared_common_'+kind, directory/'mtp_contract.py')
                if kind == 'hardlink': path.unlink()
        self.assertEqual(self.helper.read_bytes(), raw)

    def test_shared_loader_ignores_stale_bytecode(self):
        stale_bytecode(self.helper)
        contract = self.contract()
        self.assertEqual(contract.json_line({'a': 1}), b'{"a":1}\n')
        self.assertEqual(contract.json_line.__code__.co_filename, str(self.helper))

    def projection_fixture(self):
        old = load('shared_projection_control', ROOT/'legacy/review/graph_outer.py')
        new = load('shared_projection_candidate', ROOT/'spry_serving/graph_projection.py')
        value = dict(schema=1, mode=old.MODE, overlay=dict(file='graph_full_cuda_graph_backend.py',
            path=old.BACKEND_PATH, sha256='a'*64,
            original_sha256='fcfca3c5d286a77594742fd05791fe5ad4c7fd46327ab7f126b2cdcf697b685d'),
            helpers={name:'b'*64 for name in ('graph_screen.py','graph_integration.py','graph_outer.py','context_tokenizer.py')},
            masks=5, mask_bytes_each=1<<20, memory=copy.deepcopy(old.MEMORY_POLICY))
        candidate = copy.deepcopy(value)
        candidate['helpers']['spry_common_canon.py'] = candidate['helpers'].pop('graph_integration.py')
        return old, new, value, candidate

    def test_projection_mounts_and_resource_decisions_match_the_control(self):
        old, new, before, after = self.projection_fixture()
        argv = ['--size', str(2816*1024**2), '--tmpfs', '/mnt/output',
                '/mnt/work/build-venv/bin/python3', '-I', '-B', '/mnt/review/model_benchmark.py']
        self.assertEqual(old.project_argv(argv, '/mnt/review', before), new.project_argv(argv, '/mnt/review', after))
        for field, value in [('masks', True), ('mask_bytes_each', 0), ('memory', {}), ('schema', True)]:
            for module, base in [(old,before),(new,after)]:
                changed=copy.deepcopy(base); changed[field]=value
                with self.subTest(field=field,module=module.__name__), self.assertRaises(ValueError):
                    module.project_argv(argv, '/mnt/review', changed)
        nodes = lambda p: {n.name:n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        left, right = nodes(ROOT/'legacy/review/graph_outer.py'), nodes(ROOT/'spry_serving/graph_projection.py')
        self.assertEqual(ast.dump(left['project_argv']), ast.dump(right['project_argv']))
        self.assertEqual(set(right), {'need','validate_projection','project_argv'})

    def test_retired_entry_hooks_are_removed_and_unsupported_modes_still_refuse(self):
        candidate = load('shared_common_entry', ROOT/'spry_serving/benchmark.py')
        for name in ('cache_arm','read_graph_evidence','_GRAPH'): self.assertFalse(hasattr(candidate,name))
        for mode in ('native-model-context-retrieval-32k-v1','independent-bf16-reference-v1','',None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'only the W5'):
                candidate.validate_spec(dict(experiment_mode=mode))

    def test_source_exchange_keeps_membership_count_and_authenticates_both_sides(self):
        proposal = json.loads((ROOT/'proposals/common-helpers/PROPOSAL.json').read_bytes())
        before = {'graph_integration.py':(ROOT/'legacy/review/graph_integration.py').read_bytes(),
                  'BENCHMARK-SPEC2.json':json.dumps(dict(graph_projection=dict(helpers={name:'a'*64 for name in
                    ('graph_screen.py','graph_integration.py','graph_outer.py','context_tokenizer.py')}),
                    http_serving=dict(bindings={arm:{} for arm in ('control','mtp')}))).encode()}
        for arm in ('control','mtp'):
            before['HTTP-BINDING-'+arm+'.json'] = json.dumps(dict(binding_template=dict(sources={}),
                worker_template=dict(sources=dict(same_as='binding_template.sources')))).encode()
        files=copy.deepcopy(before); propose_packet.install_shared_common(files,proposal)
        self.assertEqual(len(files),len(before)); self.assertNotIn('graph_integration.py',files)
        self.assertEqual(files['spry_common_canon.py'],self.helper.read_bytes())
        for arm in ('control','mtp'):
            binding=json.loads(files['HTTP-BINDING-'+arm+'.json'])
            source=binding['binding_template']['sources']['spry_common_canon']
            self.assertEqual(source['sha256'],hashlib.sha256(self.helper.read_bytes()).hexdigest())
        bad=copy.deepcopy(before); bad['graph_integration.py']+=b'# changed\n'
        with self.assertRaisesRegex(ValueError,'replaced graph integration differs'):
            propose_packet.install_shared_common(bad,proposal)
        self.assertNotIn('spry_common_canon.py',bad)


if __name__ == '__main__': unittest.main()
