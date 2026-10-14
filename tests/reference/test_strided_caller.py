"""CPU-only storage/stride evidence; no torch, device or native execution."""
import ast
from array import array
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import test_bf16_integration as integration
from test_bf16_closure import Tensor
from test_runtime_contract import ROOT

STOCK = ROOT.parent / "stock/srt"
DONOR = ROOT.parent / "reference-u"


class LayoutTensor:
    """Small CPU stand-in with real uint16 storage, offsets and element strides.

    Values are BF16 bit tags, never converted or used for numerical arithmetic.
    Only the split and trailing-dimension views needed by the pinned caller are
    supported. Logical order uses an independent coordinate walk.
    """
    pointers = {}
    copies = []

    def __init__(self, storage, shape, device, strides=None, offset=0, dtype="bfloat16"):
        self.storage, self.shape, self.device = storage, tuple(shape), device
        self.offset, self.dtype = offset, dtype
        self.strides = tuple(strides) if strides is not None else tuple(
            math.prod(self.shape[i+1:]) for i in range(len(self.shape)))
        assert len(self.strides) == len(self.shape) and all(s >= 0 for s in self.strides)
        assert self.offset >= 0 and self.offset + sum((n-1)*s for n, s in zip(self.shape, self.strides)) < len(storage)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2

    def storage_offset(self):
        return self.offset

    def stride(self):
        return self.strides

    def data_ptr(self):
        pointer = self.storage.buffer_info()[0] + 2*self.offset
        self.pointers[pointer] = self
        return pointer

    def is_contiguous(self):
        stride = 1
        for size, actual in reversed(list(zip(self.shape, self.strides))):
            if size != 1 and stride != actual:
                return False
            stride *= size
        return True

    def logical(self):
        return [self.storage[self.offset + sum(i*s for i, s in zip(index, self.strides))]
                for index in itertools.product(*(range(n) for n in self.shape))]

    def contiguous(self):
        if self.is_contiguous():
            return self
        result = LayoutTensor(array("H", self.logical()), self.shape, self.device, dtype=self.dtype)
        self.copies.append((self, result))
        return result

    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // -math.prod(shape)
        assert math.prod(shape) == self.numel()
        if shape[0] == self.shape[0] and len(shape) > 1:
            # Split row views can reshape only their dense trailing dimensions.
            assert shape[0] == self.shape[0] and self.strides[-1] == 1
            assert self.strides[1:] == tuple(math.prod(self.shape[i+1:]) for i in range(1, len(self.shape)))
            strides = (self.strides[0],) + tuple(math.prod(shape[i+1:]) for i in range(1, len(shape)))
        else:
            assert self.is_contiguous()
            strides = tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
        return LayoutTensor(self.storage, shape, self.device, strides, self.offset, self.dtype)

    def split(self, sizes, dim=-1):
        assert dim == -1 and len(self.shape) == 2 and sum(sizes) == self.shape[-1]
        result, offset = [], self.offset
        for size in sizes:
            result.append(LayoutTensor(self.storage, (self.shape[0], size), self.device,
                                       self.strides, offset, self.dtype))
            offset += size*self.strides[-1]
        return result

    def cpu(self):
        raise AssertionError("Q/K/V must not be copied to host")


def actual_method(path, name, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for cls in tree.body if isinstance(cls, ast.ClassDef)
             for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name)
    integration.execute(nodes, namespace, path)
    return namespace[name]


class StridedCallerTests(integration.IntegrationTests):
    def setup_batch(self, rank, scenario):
        pool, parent, adapter, batch = self.batch_adapter(rank=rank)
        if scenario == "initial":
            rows, prefixes, extends = {0: [9, 4, 12]}, [0], [3]
        elif scenario == "continuation":
            rows, prefixes, extends = {0: [9, 4, 12, 3]}, [2], [2]
        elif scenario == "decode":
            rows, prefixes, extends = {0: [9, 4], 1: [12, 3]}, [1, 1], [1, 1]
            batch.forward_mode = integration.Mode("DECODE")
        else:
            rows, prefixes, extends = {0: [9]}, [0], [1]
        adapter.req_to_token_pool.req_to_token = integration.RequestMap(rows, self.substitutes.device)
        reqs = list(rows)
        loc = [slot for seq, prefix in zip(rows.values(), prefixes) for slot in seq[prefix:]]
        batch.req_pool_indices = Tensor((len(reqs),), "int64", self.substitutes.device, reqs)
        batch.seq_lens = Tensor((len(reqs),), "int32", self.substitutes.device, [len(x) for x in rows.values()])
        batch.out_cache_loc = Tensor((len(loc),), "int64", self.substitutes.device, loc)
        batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu = prefixes, extends
        dispatch = integration.hybrid(adapter)
        dispatch.init_forward_metadata_out_graph(batch)
        dispatch.init_forward_metadata_in_graph(batch)
        return pool, adapter, batch, dispatch

    def caller(self, batch, dispatch, physical, rank=0):
        n = batch.out_cache_loc.numel()
        # Every row/head/channel has a distinct tag; Q/gate/K are canaries for V.
        base = 4096 + rank*8000
        storage = array("H", (base+i for i in range(n*7168)))
        projection = LayoutTensor(storage, (n, 7168), self.substitutes.device)
        seen = {}
        def fused(qgate, key, *unused):
            seen["qgate"], seen["split_k"] = qgate, key
            qbits = [qgate.storage[qgate.offset+r*7168+h*512+c]
                     for r in range(n) for h in range(12) for c in range(256)]
            gatebits = [qgate.storage[qgate.offset+r*7168+h*512+256+c]
                        for r in range(n) for h in range(12) for c in range(256)]
            # Declared fused normalization substitute: select fields only.
            return (LayoutTensor(array("H", qbits), (n, 3072), projection.device),
                    LayoutTensor(array("H", key.logical()), (n, 512), projection.device),
                    LayoutTensor(array("H", gatebits), (n, 3072), projection.device))
        def rotary(positions, q, k):
            seen["post_mrope_q"], seen["post_mrope_k"] = q, k
            return q, k
        ns = {"fused_qk_gemma_rmsnorm_with_gate": fused, "_is_cuda": False, "_is_hip": True,
              "_is_cpu": False, "_is_xpu": False, "_is_npu": False,
              "fused_sigmoid_mul": lambda output, gate, inplace: output}
        prep = actual_method(STOCK / "models/qwen3_5.py", "forward_prepare_fused_gate", ns)
        self_attn = actual_method(STOCK / "models/qwen3_5.py", "self_attention", ns)
        radix_ns = {"get_tc_piecewise_forward_context": lambda: None,
                    "get_attn_backend": lambda: dispatch}
        radix = actual_method(STOCK / "layers/radix_attention.py", "forward", radix_ns)
        layer = self.layer(physical)
        model = SimpleNamespace(qkv_proj=lambda hidden: (projection, None),
            q_size=3072, kv_size=512, num_heads=12, head_dim=256, attn_output_gate=True,
            q_norm=SimpleNamespace(weight=SimpleNamespace(data=None), variance_epsilon=1e-6),
            k_norm=SimpleNamespace(weight=SimpleNamespace(data=None)), rotary_emb=rotary,
            o_proj=lambda output: (output, None))
        model.forward_prepare_fused_gate = MethodType(prep, model)
        model.attn = lambda q, k, v, b: radix(layer, q, k, v, b)
        q, k, v, gate = model.forward_prepare_fused_gate(None, projection)
        seen.update(q=q, k=k, v=v, storage=storage, projection=projection, base=base,
                    radix=radix, layer=layer, model=model, self_attention=self_attn)
        return seen

    def assert_order(self, seen, n, protected):
        appended = [args for name, args in self.substitutes.calls if name == "uq_bf16_append_v2"][-1]
        read = [args for name, args in self.substitutes.calls if name == "uq_bf16_attention_v2"][-1]
        base = seen["base"]
        # Independent fixed source-derived indices, not a round trip.
        expected = {
            "q": [base+r*7168+h*512+c for r in range(n) for h in range(12) for c in range(256)],
            "k": [base+r*7168+6144+h*256+c for r in range(n) for h in range(2) for c in range(256)],
            "v": [base+r*7168+6656+h*256+c for r in range(n) for h in range(2) for c in range(256)],
        }
        for name, pointer in (("q", read[2]), ("k", appended[2]), ("v", appended[3])):
            value = LayoutTensor.pointers[pointer]
            self.assertTrue(value.is_contiguous())
            self.assertEqual(value.dtype, "bfloat16")
            self.assertEqual(value.logical(), expected[name])
        self.assertEqual(appended[0]._obj.protected_layer, int(protected))
        self.assertEqual(read[0]._obj.protected_layer, int(protected))

    def test_actual_gated_model_radix_hybrid_chain_both_ranks_and_routes(self):
        for rank in (0, 1):
            for scenario in ("initial", "continuation", "decode", "single"):
                for physical in (3, 11, 59, 63):
                    with self.subTest(rank=rank, scenario=scenario, layer=physical):
                        pool, adapter, batch, dispatch = self.setup_batch(rank, scenario)
                        seen = self.caller(batch, dispatch, physical, rank)
                        n = batch.out_cache_loc.numel()
                        view = seen["v"].view(n, 2, 256)
                        self.assertEqual(view.stride(), (7168, 256, 1))
                        self.assertEqual(view.storage_offset(), 6656)
                        self.assertEqual(view.is_contiguous(), n == 1)
                        original = seen["storage"].tobytes()
                        before = self.native_count("uq_bf16_append_v2")
                        LayoutTensor.copies.clear()
                        output = seen["self_attention"](seen["model"], None, seen["projection"], batch)
                        self.assertEqual(output.shape, (n, 3072))
                        self.assertEqual(self.native_count("uq_bf16_append_v2"), before+1)
                        self.assert_order(seen, n, physical in (3, 7, 59, 63))
                        self.assertEqual(len(LayoutTensor.copies), int(n > 1))
                        self.assertEqual(seen["storage"].tobytes(), original)
                        with self.assertRaisesRegex(ValueError, "already appended"):
                            seen["self_attention"](seen["model"], None, seen["projection"], batch)
                        self.assertEqual(self.native_count("uq_bf16_append_v2"), before+1)

    def test_same_caller_layout_old_refuses_repaired_accepts(self):
        _, adapter, batch, dispatch = self.setup_batch(0, "initial")
        seen = self.caller(batch, dispatch, 11)
        old_path = DONOR / "ultraquant_backend.py"
        tree = ast.parse(old_path.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenBF16SourceAdapter")
        old = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward")
        ns = {"torch": self.substitutes.torch}
        integration.execute([old], ns, old_path)
        with patch.object(adapter, "_forward", MethodType(ns["_forward"], adapter)):
            with self.assertRaisesRegex(ValueError, "native tensors must be contiguous"):
                seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)
        output = seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)
        self.assertEqual(output.shape, (3, 3072))
        self.assert_order(seen, 3, False)

    def test_native_guard_still_refuses_unnormalized_pointer(self):
        pool, _, batch, dispatch = self.setup_batch(0, "initial")
        seen = self.caller(batch, dispatch, 11)
        with self.assertRaisesRegex(ValueError, "native tensors must be contiguous"):
            pool.native.tensor(seen["v"].view(3, 2, 256), "bfloat16", (3, 2, 256))
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def test_read_only_optional_pair_and_no_copy_for_dense_input(self):
        _, adapter, batch, dispatch = self.setup_batch(1, "initial")
        seen = self.caller(batch, dispatch, 3, 1)
        radix, layer = seen["radix"], seen["layer"]
        LayoutTensor.copies.clear()
        radix(layer, seen["q"], None, None, batch, save_kv_cache=False)
        self.assertEqual(LayoutTensor.copies, [])
        radix(layer, seen["q"], seen["k"], seen["v"], batch, save_kv_cache=False)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)
        with self.assertRaisesRegex(ValueError, "paired K/V"):
            adapter.forward(seen["q"], None, seen["v"], layer, batch, save_kv_cache=False)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def test_strided_q_k_v_all_preserve_order(self):
        _, adapter, batch, dispatch = self.setup_batch(0, "initial")
        seen = self.caller(batch, dispatch, 11)
        n = 3
        # Keep Q logical channels dense inside a deliberately padded row.
        qstore = array("H", [65500] * (n*4000))
        for row in range(n):
            for head in range(12):
                for channel in range(256):
                    qstore[row*4000+head*256+channel] = seen["base"]+row*7168+head*512+channel
        q = LayoutTensor(qstore, (n, 3072), self.substitutes.device, (4000, 1))
        k = seen["projection"].split([6144, 512, 512])[-2]
        LayoutTensor.copies.clear()
        seen["radix"](seen["layer"], q, k, seen["v"], batch)
        self.assert_order(seen, n, False)
        self.assertEqual(len(LayoutTensor.copies), 3)
        self.assertTrue(all(value == 65500 for row in range(n) for value in qstore[row*4000+3072:(row+1)*4000]))

    def test_invalid_inputs_refuse_before_native_calls_or_copies(self):
        _, adapter, batch, dispatch = self.setup_batch(0, "initial")
        seen = self.caller(batch, dispatch, 11)
        q, k, v, layer = seen["q"], seen["k"], seen["v"], seen["layer"]
        def refused(q=q, k=k, v=v, layer=layer, **kwargs):
            before = len(self.substitutes.calls)
            LayoutTensor.copies.clear()
            with self.assertRaises(ValueError):
                adapter.forward(q, k, v, layer, batch, **kwargs)
            self.assertEqual(len(self.substitutes.calls), before)
            self.assertEqual(LayoutTensor.copies, [])
        for value in ("float16", "float32"):
            with patch.object(v, "dtype", value):
                refused()
        with patch.object(v, "device", object()):
            refused()
        with patch.object(v, "shape", (3, 513)):
            refused()
        refused(layer=self.layer(0))
        refused(k_scale=2)
        refused(dcp_kv_mask=[])
        with patch.object(layer, "scaling", float("nan")):
            refused()
        with patch.object(batch.out_cache_loc, "values", [9, 4, 31]):
            refused()
        with patch.object(adapter, "metadata", ((), tuple(range(4097)), ())):
            refused()
        adapter.invalidate()
        refused()

    def test_normalization_workspace_bound_is_exact(self):
        q, k, v = (4096*heads*256*2 for heads in (12, 2, 2))
        self.assertEqual((q, k, v), (25165824, 4194304, 4194304))
        self.assertEqual(q+k+v, 32*1024**2)
        self.assertLess(q+k+v + 4096*12*256*2 + 1048576*8+4096*8, 2*1024**3)


# The retained module runs all original tests exactly once.
for _name in dir(integration.IntegrationTests):
    if _name.startswith("test_") and _name not in vars(StridedCallerTests):
        setattr(StridedCallerTests, _name, None)
