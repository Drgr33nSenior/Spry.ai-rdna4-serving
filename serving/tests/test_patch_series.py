"""Validate exact overlay reproduction and refusal before source mutation."""
import copy
import difflib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_source_builder import builder,tree_bytes


class PatchSeries(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='patch-series-test-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.source=self.root/'source';self.upstream=self.root/'upstream'
        (self.source/'upstream').mkdir(parents=True)
        (self.source/'patches/sglang-0.5.21').mkdir(parents=True)
        self.upstream.mkdir()
        self.order=json.loads((HERE.parent/'config/runs/w5-v2/run.json').read_bytes())['overlays']
        self.rows=[];self.files={}
        for index,name in enumerate(self.order):
            source=f'python/sglang/module{index}.py';target='image-source/'+source
            before=f'VALUE = {index}\n'.encode();after=f'VALUE = {index+100}\n'.encode()
            path=self.upstream/source;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(before)
            raw=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
                fromfile='a/'+target,tofile='b/'+target)).encode()
            patch_name=f'patches/sglang-0.5.21/{index:04d}.patch'
            (self.source/patch_name).write_bytes(raw)
            self.files[name]=after
            self.rows.append(dict(source=source,target=target,overlay=name,original_sha256=builder.digest(before),
                overlay_sha256=builder.digest(after),patch=patch_name,patch_sha256=builder.digest(raw)))
        self.lock=dict(schema=1,version='cpu-fixture',scope='active W5 runtime and graph/GEMM overlays',
                       complete=True,rows=self.rows)
        self.write_lock()

    def write_lock(self):
        (self.source/'upstream/sglang-0.5.21.lock').write_bytes(builder.canonical(self.lock))

    def reproduce(self):
        return builder.reproduce_overlays(self.files,self.upstream,self.source,self.order)

    def test_real_patch_tool_reproduces_all_outputs_without_changing_upstream(self):
        original=tree_bytes(self.upstream);expected=copy.deepcopy(self.files)
        report=self.reproduce()
        self.assertEqual(self.files,expected)
        self.assertEqual(tree_bytes(self.upstream),original)
        self.assertEqual(len(report['overlays']),11)
        self.assertTrue(report['exact_output_bytes'])

    def test_wrong_upstream_or_patch_hash_refuses_before_tool_execution(self):
        for key in ('original_sha256','patch_sha256','overlay_sha256'):
            before=self.rows[0][key];self.rows[0][key]='0'*64;self.write_lock()
            with self.subTest(key=key),patch.object(builder.subprocess,'run') as execute,self.assertRaises(ValueError):
                self.reproduce()
            execute.assert_not_called()
            self.rows[0][key]=before

    def test_unknown_incomplete_duplicate_or_reordered_lock_refuses(self):
        original=copy.deepcopy(self.lock)
        for mutation in ('extra','incomplete','duplicate','reorder'):
            self.lock=copy.deepcopy(original)
            if mutation=='extra':self.lock['unknown']=True
            elif mutation=='incomplete':self.lock['complete']=False
            elif mutation=='duplicate':self.lock['rows'][1]=self.lock['rows'][0]
            else:self.lock['rows'].reverse()
            self.write_lock()
            with self.subTest(mutation=mutation),patch.object(builder.subprocess,'run') as execute,self.assertRaises(ValueError):
                self.reproduce()
            execute.assert_not_called()

    def test_unsafe_paths_and_extra_patch_targets_refuse_even_with_updated_hashes(self):
        for key,value in (('source','../escape'),('target','image-source/../escape'),('patch','/tmp/escape')):
            original=self.rows[0][key];self.rows[0][key]=value;self.write_lock()
            with self.subTest(key=key),self.assertRaises(ValueError):self.reproduce()
            self.rows[0][key]=original
        path=self.source/self.rows[0]['patch']
        raw=path.read_bytes()+b'--- a/escape\n+++ b/escape\n@@ -1 +1 @@\n-x\n+y\n'
        path.write_bytes(raw);self.rows[0]['patch_sha256']=builder.digest(raw);self.write_lock()
        with patch.object(builder.subprocess,'run') as execute,self.assertRaises(ValueError):self.reproduce()
        execute.assert_not_called()

    def test_failed_context_does_not_publish_partial_rendered_sources(self):
        original=copy.deepcopy(self.files)
        raw=b'BROKEN = 0\n'
        (self.upstream/self.rows[0]['source']).write_bytes(raw)
        self.rows[0]['original_sha256']=builder.digest(raw);self.write_lock()
        with self.assertRaisesRegex(ValueError,'did not apply exactly'):self.reproduce()
        self.assertEqual(self.files,original)


if __name__=='__main__':unittest.main()
