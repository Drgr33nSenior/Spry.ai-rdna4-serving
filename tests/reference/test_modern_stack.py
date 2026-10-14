"""Inert local metadata and source-bound 0.5.21 call fixtures, no runtime import."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
import sys

ROOT = Path(__file__).resolve().parents[2] / "prototype"
OWNED = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import bf16_source_identity as source
import collect_bf16_toolchain as collector


class ModernTests(unittest.TestCase):
    def test_missing_shared_receipt_cannot_fall_back(self):
        with self.assertRaisesRegex(ValueError, "modern target"):
            source.inventory(OWNED / "prototype", OWNED / "tests/fixtures/build-stock", OWNED / "tests/fixtures/STOCK-INPUTS.json", source.digest(OWNED / "tests/fixtures/STOCK-INPUTS.json"))

    def test_actual_new_parent_writer_maps_physical_layer(self):
        import test_runtime_contract as contract
        path = OWNED / "tests/fixtures/stock-v0521/python/sglang/srt/mem_cache/memory_pool.py"
        tree = ast.parse(path.read_bytes())
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HybridLinearKVPool")
        function = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "set_kv_buffer")
        function.decorator_list = []
        calls = []
        def writer(layer, loc, k, v, k_scale, v_scale, **kwargs):
            calls.append(contract.uq.qwen_writer_metadata(layer, loc, k, v, k_scale, v_scale, **kwargs))
        namespace = {"RadixAttention": object, "torch": types.SimpleNamespace(Tensor=object),
            "Optional": __import__("typing").Optional, "unwrap_write_loc": lambda value: (value, None, None)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        for physical, compact in ((3, 0), (7, 1), (11, 2), (59, 14), (63, 15)):
            pool = types.SimpleNamespace(use_mla=False, _transfer_full_attention_id=lambda n: contract.uq.QWEN_FULL_LAYERS.index(n),
                full_kv_pool=types.SimpleNamespace(set_kv_buffer=writer))
            layer = types.SimpleNamespace(layer_id=physical)
            namespace["set_kv_buffer"](pool, layer, object(), object(), object())
            self.assertEqual(calls[-1]["compact_id"], compact)
        self.assertTrue(calls[0]["protected"])
        self.assertFalse(calls[2]["protected"])

    def test_legacy_environment_and_numerical_flags_preserved(self):
        env = source.compiler_environment({}, {"clang": "/inert/sdk/bin/clang++"}, Path("/inert/tmp"))
        self.assertEqual(env["ROCM_PATH"], "/opt/rocm")
        self.assertIn("-ffp-contract=off", source.FLAGS)
        self.assertIn("-fno-fast-math", source.FLAGS)
        self.assertEqual(env["MAX_JOBS"], "2")

    def test_modern_environment_refuses_stale_selection(self):
        env = source.compiler_environment({}, {"clang": "/inert/sdk/bin/clang++"}, Path("/inert/tmp"))
        for key in ("TMPDIR", "XDG_CACHE_HOME", "PYTHONDONTWRITEBYTECODE", "LC_ALL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS", "HIPCC_JOBS"):
            del env[key]
        env["ROCM_PATH"] = "/inert/10.1-sdk"
        accepted = source.compiler_environment({"stack_receipt_sha256": "a"*64, "compiler_environment": env}, {"clang": "/inert/sdk/bin/clang++"}, Path("/inert/tmp"))
        self.assertEqual(accepted["ROCM_PATH"], "/inert/10.1-sdk")
        for field, bad in (("HIP_CLANG_PATH", "/stale"), ("HIP_COMPILER", "gcc"), ("OFFLOAD_BUNDLER_COMPRESS", "1")):
            value = copy.deepcopy(env);value[field] = bad
            with self.assertRaises(ValueError):
                source.compiler_environment({"stack_receipt_sha256": "a"*64, "compiler_environment": value}, {"clang": "/inert/sdk/bin/clang++"}, Path("/inert/tmp"))

    def test_static_elf_metadata_independent_fixture(self):
        with tempfile.TemporaryDirectory(dir=OWNED / ".test-build/tmp") as folder:
            path = Path(folder) / "INERT-NOT-EXECUTABLE"
            interpreter = b"/inert/loader\0"
            header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01"+b"\0"*10, 2, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
            segment = struct.pack("<IIQQQQQQ", 3, 4, 120, 0, 0, len(interpreter), len(interpreter), 1)
            path.write_bytes(header + segment + interpreter)
            self.assertEqual(collector.elf_metadata(path)["interpreter"], "/inert/loader")
            self.assertEqual(collector.elf_metadata(path)["needed"], [])
            path.write_bytes(b"INERT")
            with self.assertRaises(ValueError): collector.elf_metadata(path)

    def test_shared_receipt_rejects_wrong_target_and_unreviewed(self):
        contract = source.stack_contract(ROOT)
        with tempfile.TemporaryDirectory(dir=OWNED / ".test-build/tmp") as folder:
            path = Path(folder) / "INERT-receipt.json"
            for target in ({"sglang": "0.5.18", "rocm": "10.1.0"}, contract.TARGET):
                path.write_bytes(contract.canonical({"schema": 1, "decision": "unreviewed", "target": target}))
                with self.assertRaises(ValueError): contract.read_receipt(path, source.digest(path))
            with self.assertRaises(ValueError): contract.read_receipt(path, "0"*64)

    def test_native_inputs_exact_accepted_hashes(self):
        expected = json.loads((OWNED / "baseline/SOURCE-IDENTITY.json").read_bytes())["native_sources"]
        self.assertEqual({p: source.digest(OWNED / "prototype" / p) for p in source.NATIVE}, expected)
