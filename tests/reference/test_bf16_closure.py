"""Actual BF16 functions with inert files and declared CPU allocation/library spies."""
import ast
from abc import ABC
import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_runtime_contract import uq, UPSTREAM, ROOT as CANDIDATE
from test_runtime_evidence import evidence


class SyntheticBuild:
    """Labelled inert fixture; never a compiler output or an authority receipt."""
    def __init__(self, root):
        self.root = Path(root)
        self.package = self.root / "image-source/python/sglang"
        shutil.copytree(UPSTREAM / "python/sglang", self.package)
        self.native = {}
        for name in ("runtime/uq.h", "runtime/attention.hip", "runtime/bf16_math.h", "codec/codec.h"):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CANDIDATE / name, dest)
            self.native[name] = evidence.digest(dest)
        self.binary = self.package / "srt/layers/attention/libuq_bf16_v2.so"
        self.binary.write_bytes(b"INERT CPU fixture. Never load this as a library.\n")
        self.proof = self.binary.with_name("ultraquant-bf16-build.json")
        self.value = {"schema": 2, "image_manifest": evidence.BF16_IMAGE,
            "image_config": evidence.BF16_IMAGE_CONFIG,
            "qualification": "build-reviewed-not-serving-qualified",
            "fixture_label": "synthetic CPU acceptance fixture; no build proof or authority",
            "repair_files": dict(evidence.BF16_REPAIRS),
            "kv_patch_sha256": "1" * 64, "review_receipt_sha256": "2" * 64,
            "binary_sha256": evidence.digest(self.binary), "compiler_revision": "INERT compiler fixture",
            "compiler_flags": ["INERT flag fixture"], "native_sources": self.native}
        self.refresh_inventory()

    def refresh_inventory(self):
        excluded = {self.binary, self.proof}
        files = {p.relative_to(self.package).as_posix(): evidence.digest(p)
                 for p in self.package.rglob("*") if p.is_file() and p not in excluded}
        self.value["runtime_files"] = files
        self.value["source_sha256"] = hashlib.sha256(evidence.canonical(files)).hexdigest()
        self.write()

    def write(self):
        self.proof.write_bytes(evidence.canonical(self.value))


class Device:
    type = "cuda"
    def __eq__(self, other):
        return isinstance(other, Device)


class Tensor:
    _next = 4096
    def __init__(self, shape, dtype, device, values=None, contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.values, self.contiguous = values, contiguous
        Tensor._next += 4096
        self.pointer = Tensor._next
    def numel(self):
        return math.prod(self.shape)
    def element_size(self):
        return {"bfloat16": 2, "uint8": 1, "int32": 4, "int64": 8}[self.dtype]
    def data_ptr(self):
        return self.pointer
    def is_contiguous(self):
        return self.contiguous
    def cpu(self):
        return self
    def tolist(self):
        return list(self.values)
    def view(self, *shape):
        if math.prod(shape) != self.numel():
            raise ValueError("CPU view shape mismatch")
        value = Tensor(shape, self.dtype, self.device, self.values, self.contiguous)
        value.pointer = self.pointer
        return value


class CPUStandins:
    """No tensor computation. Shape/byte/allocation spies and inert callables."""
    def __init__(self):
        self.allocations, self.library_loads, self.calls = [], [], []
        self.device = Device()
        self.torch = SimpleNamespace(bfloat16="bfloat16", uint8="uint8", int32="int32", int64="int64",
            version=SimpleNamespace(hip="CPU substitute"), device=lambda value: self.device,
            cuda=SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(gcnArchName="gfx1201"),
                is_current_stream_capturing=lambda: False,
                device=lambda device: contextlib.nullcontext(),
                current_stream=lambda device: SimpleNamespace(cuda_stream=0)),
            full=self.full, empty=self.empty, tensor=self.tensor)
        parent = self
        class KVCacheStandin:
            def __init__(self, size, page_size, dtype, layer_num, device, *args):
                self.size, self.layer_num, self.start_layer = size, layer_num, 0
                parent.calls.append(("base-pool-constructor", size))
        self.base = KVCacheStandin

    def full(self, shape, value, dtype, device):
        tensor = Tensor(shape, dtype, self.device)
        self.allocations.append(("full", shape, value, dtype))
        return tensor
    def empty(self, shape, dtype, device):
        shape = (shape,) if type(shape) is int else shape
        tensor = Tensor(shape, dtype, self.device)
        self.allocations.append(("empty", shape, None, dtype))
        return tensor
    def tensor(self, values, dtype, device):
        tensor = Tensor((len(values),), dtype, self.device, values)
        self.allocations.append(("tensor", tensor.shape, list(values), dtype))
        return tensor
    def library(self, path):
        self.library_loads.append(path)
        parent = self
        class InertFunction:
            def __init__(self, name):
                self.name = name
            def __call__(self, *args):
                parent.calls.append((self.name, args))
                return 0
        return SimpleNamespace(uq_bf16_append_v2=InertFunction("uq_bf16_append_v2"),
                               uq_bf16_attention_v2=InertFunction("uq_bf16_attention_v2"))


def actual_classes(fixture, standins):
    path = fixture.package / "srt/layers/attention/ultraquant_backend.py"
    tree = ast.parse(path.read_bytes())
    names = {"Native", "BF16CacheDescriptor", "BF16Native", "QwenBF16MixedPool", "QwenBF16SourceAdapter"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    ns = {"ctypes": ctypes, "Path": Path, "__file__": str(path), "torch": standins.torch,
          "KVCache": standins.base, "causal_rows": uq.causal_rows, "ABC": ABC,
          "debug_kernel_api": lambda function: function, "is_npu": lambda: False}
    base_path = CANDIDATE.parent / "stock/srt/layers/attention/base_attn_backend.py"
    base = next(n for n in ast.parse(base_path.read_bytes()).body
                if isinstance(n, ast.ClassDef) and n.name == "AttentionBackend")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, base]+nodes, type_ignores=[])), str(path), "exec"), ns)
    return SimpleNamespace(**{name: ns[name] for name in names}, AttentionBackend=ns["AttentionBackend"])


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bf16-inert-", dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticBuild(self.temp.name)
        self.policy = uq.bf16_pool_policy(rank=0, physical_slots=31)
        self.substitutes = CPUStandins()
        self.classes = actual_classes(self.fixture, self.substitutes)
        self.module_patch = patch.dict(sys.modules, {
            "sglang.srt.layers.attention.ultraquant_contract": uq,
            "sglang.srt.layers.attention.ultraquant_evidence": evidence,
            "sglang.srt.layers.attention.ultraquant_backend": self.classes})
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        self.library_patch = patch.object(ctypes, "CDLL", side_effect=self.substitutes.library)
        self.library_patch.start(); self.addCleanup(self.library_patch.stop)

    def identity(self, policy=None):
        return evidence.bf16_build_identity(self.fixture.package, policy or self.policy)

    def construct(self, rank=0, **kwargs):
        connect_fixture = kwargs.pop("connect_fixture", True)
        args = dict(size=31, page_size=1, dtype="bfloat16", head_num=2, head_dim=256, layer_num=16,
                    device="CPU declared cuda substitute", enable_memory_saver=False,
                    policy=uq.bf16_pool_policy(rank=rank, physical_slots=31))
        args.update(kwargs)
        pool = self.classes.QwenBF16MixedPool(**args)
        if connect_fixture:
            # Declared preallocated lifecycle wiring for the retained suites.
            # New lifecycle tests execute the actual allocator instead.
            pool.owner.bind_allocator(SimpleNamespace(page_size=1), pool.size)
            pool.owner.allocate(range(1, pool.size+1))
        return pool

    def expect_gate(self, expression):
        with self.assertRaisesRegex(ValueError, "^" + expression + "$"):
            self.identity()

    def test_identity_positive_immutable_both_ranks(self):
        identities = []
        for rank in (0, 1):
            result = self.identity(uq.bf16_pool_policy(rank=rank, physical_slots=31))
            self.assertIsInstance(result["canonical"], bytes)
            self.assertEqual(result["sha256"], hashlib.sha256(result["canonical"]).hexdigest())
            with self.assertRaises(TypeError):
                result["sha256"] = "changed"
            identities.append(result["sha256"])
        self.assertNotEqual(*identities)
        self.assertFalse(self.substitutes.library_loads)

    def test_missing_proof(self):
        self.fixture.proof.unlink()
        self.expect_gate("missing reviewed BF16 build proof")

    def test_wrong_image_config_repair_reaches_identity_gate(self):
        original = json.loads(json.dumps(self.fixture.value))
        for field, value in (("image_manifest", "0"*64), ("image_config", "0"*64),
                             ("repair_files", {})):
            self.fixture.value = json.loads(json.dumps(original))
            self.fixture.value[field] = value; self.fixture.write()
            self.expect_gate("stale image/repair/build identity")

    def test_missing_and_invalid_compiler_flags(self):
        original = json.loads(json.dumps(self.fixture.value))
        for field, value in (("compiler_revision", None), ("compiler_flags", []),
                             ("compiler_flags", None), ("compiler_flags", [None])):
            self.fixture.value = json.loads(json.dumps(original))
            self.fixture.value[field] = value; self.fixture.write()
            self.expect_gate("missing compiler/flags proof")

    def test_malformed_digests(self):
        for field in ("kv_patch_sha256", "source_sha256", "binary_sha256", "review_receipt_sha256"):
            original = self.fixture.value[field]
            for value in (None, "short", "G"*64):
                self.fixture.value[field] = value; self.fixture.write()
                self.expect_gate("missing build digest")
            self.fixture.value[field] = original

    def test_changed_runtime_file_reaches_hash_gate(self):
        target = self.fixture.package / "srt/layers/attention/ultraquant_contract.py"
        target.write_bytes(target.read_bytes()+b"\n# mutation\n")
        self.expect_gate("effective source hash changed")

    def test_added_and_missing_runtime_files_reach_membership(self):
        target = self.fixture.package / "extra.py"
        target.write_bytes(b"# inert extra\n")
        self.expect_gate("effective source membership changed")
        target.unlink()
        (self.fixture.package / "srt/layers/attention/ultraquant_contract.py").unlink()
        self.expect_gate("effective source membership changed")

    def test_changed_native_source_reaches_native_gate(self):
        target = self.fixture.root / "runtime/bf16_math.h"
        target.write_bytes(target.read_bytes()+b"\n// inert change\n")
        self.expect_gate("native source changed since build")

    def test_repair_bytes_gate_after_consistent_hash_rewrite(self):
        target = self.fixture.package / next(iter(evidence.BF16_REPAIRS))
        target.write_bytes(target.read_bytes()+b"\n# inert repair change\n")
        self.fixture.refresh_inventory()
        self.expect_gate("repair bytes changed")

    def test_missing_stale_and_replaced_binary_reach_binary_gate(self):
        self.fixture.binary.unlink()
        self.expect_gate("experimental runtime source/binary must be a regular file")
        for data in (b"STALE inert binary\n", b"REPLACED inert binary\n"):
            self.fixture.binary.write_bytes(data)
            self.expect_gate("missing or stale BF16 binary")

    def test_prefix_tags_process_and_foreign_owner(self):
        owner = evidence.BF16CacheOwner(self.identity())
        tags = ("user", "lora")
        key = owner.prefix_key(tags)
        self.assertEqual(key[-1], tags)
        owner.validate_prefix(key, tags)
        with self.assertRaisesRegex(ValueError, "stale, mixed, cross-rank or foreign cache prefix"):
            owner.validate_prefix(evidence.BF16CacheOwner(self.identity()).prefix_key(tags), tags)
        with patch.object(evidence.os, "getpid", return_value=os.getpid()+1):
            with self.assertRaisesRegex(ValueError, "cross-process BF16 cache reuse is forbidden"):
                owner.check_process()

    def test_constructor_positive_shapes_dtypes_and_bytes_both_ranks(self):
        for rank in (0, 1):
            pool = self.construct(rank=rank)
            self.assertEqual((len(pool.k_buffer), len(pool.v_buffer)), (16, 16))
            for i, (k, v) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
                protected = i in (0, 1, 14, 15)
                self.assertEqual(k.shape, (32, 2, 256 if protected else 136))
                self.assertEqual(k.dtype, "bfloat16" if protected else "uint8")
                self.assertEqual((k.shape, k.dtype), (v.shape, v.dtype))
            self.assertEqual(sum(pool.get_kv_size_bytes()), 32*14720)
            self.assertEqual(pool.policy["rank"], rank)

    def test_constructor_proof_refusal_before_allocation_or_native(self):
        original = self.fixture.proof.read_bytes()
        self.fixture.proof.unlink()
        with self.assertRaisesRegex(ValueError, "missing reviewed BF16 build proof"):
            self.construct()
        self.assertEqual((self.substitutes.library_loads, self.substitutes.allocations, self.substitutes.calls), ([], [], []))
        self.fixture.proof.write_bytes(original)
        self.fixture.value["image_manifest"] = "0"*64; self.fixture.write()
        with self.assertRaisesRegex(ValueError, "stale image/repair/build identity"):
            self.construct()
        self.assertEqual((self.substitutes.library_loads, self.substitutes.allocations, self.substitutes.calls), ([], [], []))

    def test_native_declared_ctypes_layout_and_signatures(self):
        pool = self.construct()
        descriptor = pool.native.descriptor(pool.k_buffer[2], pool.v_buffer[2], 32, False)
        cls = self.classes.BF16CacheDescriptor
        self.assertEqual(ctypes.sizeof(cls), 64)
        self.assertEqual([getattr(cls, name).offset for name, _ in cls._fields_], [0, 8, 16, 24, 56, 60])
        self.assertEqual(descriptor.bytes_per_buffer, 32*2*136)
        self.assertEqual(pool.native.lib.uq_bf16_append_v2.argtypes,
            [ctypes.c_void_p]*4+[ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p])
        self.assertEqual(pool.native.lib.uq_bf16_attention_v2.argtypes,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
             ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64,
             ctypes.c_void_p, ctypes.c_void_p])
        self.assertEqual(pool.native.lib.uq_bf16_attention_v2.restype, ctypes.c_int)

    def adapter(self, rank=0):
        pool = self.construct(rank=rank)
        adapter = self.classes.QwenBF16SourceAdapter(pool)
        tags = ("user", "lora")
        adapter.prepare([[9, 4, 12, 3]], [2], [2], [12, 3], pool.owner.prefix_key(tags), tags)
        return pool, adapter

    def test_prepare_and_attention_routes_both_ranks_noncontiguous(self):
        for rank in (0, 1):
            pool, adapter = self.adapter(rank)
            self.assertEqual(pool.immutable, frozenset({9, 4}))
            q = Tensor((2, 12, 256), "bfloat16", self.substitutes.device)
            for physical in (3, 11, 55, 63):
                output = adapter.attention(q, physical, 0.0625)
                self.assertEqual(output.shape, (2, 12, 256))
                name, args = self.substitutes.calls[-1]
                self.assertEqual(name, "uq_bf16_attention_v2")
                descriptor = ctypes.cast(args[0], ctypes.POINTER(self.classes.BF16CacheDescriptor)).contents
                index = uq.QWEN_FULL_LAYERS.index(physical)
                self.assertEqual(descriptor.k, pool.k_buffer[index].data_ptr())
                self.assertEqual(descriptor.protected_layer, physical in (3, 7, 59, 63))
                self.assertEqual(args[5], 4)
            # Q is supplied after native MRoPE by the caller. The spy cannot prove rotation.

    def test_invalid_attention_scale_refuses_before_allocation_and_call(self):
        pool, adapter = self.adapter()
        q = Tensor((2, 12, 256), "bfloat16", self.substitutes.device)
        for scale in (0, -1, 2, True, "0.1", float("nan"), float("inf")):
            allocations, calls = len(self.substitutes.allocations), len(self.substitutes.calls)
            with self.assertRaisesRegex(ValueError, "unsupported BF16 attention scale"):
                adapter.attention(q, 11, scale)
            self.assertEqual((len(self.substitutes.allocations), len(self.substitutes.calls)), (allocations, calls))

    def test_invalid_layer_dtype_and_prefix_do_not_call_native(self):
        pool, adapter = self.adapter()
        q = Tensor((2, 12, 256), "bfloat16", self.substitutes.device)
        count = len(self.substitutes.calls)
        for physical in (0, 64, True, "11"):
            with self.assertRaises(ValueError):
                adapter.attention(q, physical, 0.0625)
        with self.assertRaisesRegex(ValueError, "unsupported UltraQuant tensor device, dtype or shape"):
            adapter.attention(Tensor((2, 12, 256), "float16", self.substitutes.device), 11, 0.0625)
        with self.assertRaisesRegex(ValueError, "stale, mixed, cross-rank or foreign cache prefix"):
            adapter.prepare([[9, 4, 12, 3]], [2], [2], [12, 3], ("stale",), ("user", "lora"))
        self.assertEqual(len(self.substitutes.calls), count)

    def test_failed_prepare_invalidates_stale_metadata(self):
        pool, adapter = self.adapter()
        with self.assertRaises(ValueError):
            adapter.prepare([[9, 4, 12, 3]], [2], [2], [9, 3], pool.owner.prefix_key(None), None)
        self.assertIsNone(adapter.metadata)
        before = len(self.substitutes.calls)
        with self.assertRaisesRegex(ValueError, "causal metadata required"):
            adapter.attention(Tensor((2, 12, 256), "bfloat16", self.substitutes.device), 11, 0.0625)
        self.assertEqual(len(self.substitutes.calls), before)

    def test_prepare_return_cannot_mutate_committed_metadata(self):
        pool = self.construct()
        adapter = self.classes.QwenBF16SourceAdapter(pool)
        rows = adapter.prepare([[2]], [0], [1], [2], pool.owner.prefix_key(None), None)
        self.assertEqual(rows, ((2,), (0,), (1,)))
        with self.assertRaises(TypeError):
            rows[0][0] = 0

    def test_actual_factory_and_fixed_payload_helper(self):
        def function(file, name):
            tree = ast.parse(file.read_bytes())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), ns)
            return ns[name]
        folder = UPSTREAM / "python/sglang/srt/model_executor"
        factory = function(folder / "model_runner_kv_cache_mixin.py", "qwen_bf16_full_pool_factory")
        payload = function(folder / "pool_configurator.py", "qwen_bf16_fixed_payload")
        config = (CANDIDATE / "tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes()
        for rank in (0, 1):
            policy = uq.bf16_pool_policy(rank=rank, physical_slots=31)
            result = factory(config, policy)
            self.assertIs(result.func, self.classes.QwenBF16MixedPool)
            self.assertEqual(result.keywords["policy"], policy)
            self.assertEqual(payload(policy)["kv_payload_with_slot_zero"], 32*14720)
            self.assertEqual(payload(policy)["gdn_bytes_per_state_slot"], 73.40625*1024**2)
        with self.assertRaisesRegex(ValueError, "fixed official FP8 Qwen config required"):
            factory(b"wrong config", self.policy)

    def test_writer_mask_scale_and_slots_refuse_before_call(self):
        pool = self.construct()
        loc = Tensor((1,), "int64", self.substitutes.device, [2])
        k = Tensor((1, 2, 256), "bfloat16", self.substitutes.device)
        before = len(self.substitutes.calls)
        for kwargs in ({"k_scale": 2}, {"v_scale": False}, {"dcp_kv_mask": []}):
            with self.assertRaises(ValueError):
                pool.set_kv_buffer(None, loc, k, k, layer_id_override=2, **kwargs)
        pool.immutable = frozenset({2})
        with self.assertRaisesRegex(ValueError, "duplicate write or immutable shared prefix"):
            pool.set_kv_buffer(None, loc, k, k, layer_id_override=2)
        self.assertEqual(len(self.substitutes.calls), before)
