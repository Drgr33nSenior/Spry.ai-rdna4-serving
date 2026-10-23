"""Check the optional 64K adapter patch without importing GPU libraries."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(os.environ.get('S33_PUBLIC_REPO', Path(__file__).resolve().parents[1])).resolve()
PATCH = Path(os.environ.get('S33_PATCH', REPO / 'patches/sglang-dflash-splitkv-context64.patch')).resolve()
BASE_SHA = 'f62a549fa6b1db802d965f1b821866d590505045d916b180aded998544471128'
PATCHED_SHA = 'ca2d698bb3a8ee1488ead6f37fe373d5e6da9af9c70a9e7eae29e8a3bf2bc7fd'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def geometry(context):
    tensor = lambda shape: SimpleNamespace(shape=shape, dtype='torch.bfloat16')
    backend = SimpleNamespace(page_size=1, dcp_size=1, use_mla=False,
        enable_deterministic=False, static_kv_splits=False, split_tile_size=None,
        max_kv_splits=16, device_core_count=64, num_head=12, num_kv_head=2,
        kv_index_translator=SimpleNamespace(reads_are_translated=False), max_context_len=context)
    batch = SimpleNamespace(batch_size=1, forward_mode=SimpleNamespace(is_target_verify=lambda: True))
    args = [tensor((8,12,256)), tensor((8,2,256)), tensor((8,2,256)), tensor((8,12,256)),
        tensor((context+1,2,256)), tensor((context+1,2,256)),
        SimpleNamespace(numel=lambda: 2), SimpleNamespace(numel=lambda: 2), None, None,
        True, None, 8, 1.0, 1.0, 0.0625]
    return backend, batch, args


class Context64PublicPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.original = (REPO/'prototype/serving/verify_decode_adapter.py').read_bytes()
        if sha(cls.original) != BASE_SHA:
            raise AssertionError('public prototype source pin differs')
        target = root/'prototype/serving/verify_decode_adapter.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(cls.original)
        subprocess.run(['patch','--batch','--forward','-p1','-i',str(PATCH)], cwd=root,
                       check=True, capture_output=True, timeout=10)
        cls.patched = target.read_bytes()
        spec=importlib.util.spec_from_file_location('context64_public_candidate', target)
        cls.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_patch_exactly_changes_guard_and_preserves_public_arm_policy(self):
        self.assertEqual(sha(self.patched),PATCHED_SHA)
        self.assertEqual(self.patched,self.original.replace(b'backend.max_context_len <= 32768',b'backend.max_context_len <= 65536'))
        self.assertIn(b"self.enabled = manifest['arm'] == 'mtp'",self.patched)

    def test_context_boundaries(self):
        for context in (8,32768,32769,65536):
            self.assertTrue(self.module.eligible(*geometry(context),{}))
        for context in (7,65537,131072):
            self.assertFalse(self.module.eligible(*geometry(context),{}))

    def test_batch_and_geometry_remain_guarded(self):
        backend,batch,args=geometry(65536)
        batch.batch_size=2
        self.assertFalse(self.module.eligible(backend,batch,args,{}))
        batch.batch_size=1
        args[0].shape=(7,12,256)
        self.assertFalse(self.module.eligible(backend,batch,args,{}))

    def test_bf16_and_scale_requirements_remain(self):
        for index in (0,1,2,3,4,5):
            backend,batch,args=geometry(65536);args[index].dtype='torch.float8_e4m3fn'
            self.assertFalse(self.module.eligible(backend,batch,args,{}))
        for index in (13,14,15):
            backend,batch,args=geometry(65536);args[index]=2.0
            self.assertFalse(self.module.eligible(backend,batch,args,{}))

    def test_unsupported_attention_options_remain_guarded(self):
        for options in ({'sliding_window_size':4096},{'logit_cap':1},{'sinks':object()}):
            self.assertFalse(self.module.eligible(*geometry(65536),options))

    def test_packed_index_integer_capacity(self):
        context=65536
        max_prefix=context-8
        packed=sum(max_prefix+i for i in range(1,9))
        self.assertLessEqual(packed,8*context)
        self.assertLess(packed,2**31)
        self.assertEqual(8*context*8,4194304)
        self.assertEqual(context+2*2*8,65568)

if __name__=='__main__':
    unittest.main()
