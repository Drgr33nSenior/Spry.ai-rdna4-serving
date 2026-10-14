"""Modern production methods with retained inert tensor/library/caller fixtures."""
import ast
from abc import ABC
import ctypes
from enum import Enum
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_bf16_closure as closure
import test_bf16_integration as integration
import test_strided_caller as strided
import test_unified_tree0521 as trees
from test_cache0521 import SRT, STOCK, cache, evidence, Ints, setup, uq


class RepairTests(unittest.TestCase):
    def test_actual_implicit_unit_scale_tensors_and_strict_refusals(self):
        case = self.fixture()
        class Scalar(closure.Tensor):
            def __init__(self, value=1.0, shape=(), dtype="float32", device=None):
                super().__init__(shape, dtype, device or case.substitutes.device)
                self.scalar = value
            def item(self):
                return self.scalar
        case.substitutes.torch.Tensor = closure.Tensor
        case.substitutes.torch.float32 = "float32"
        for rank in (0, 1):
            pool, _, adapter, batch = case.batch_adapter(rank)
            adapter.init_forward_metadata(batch)
            layer = case.layer()
            layer.k_scale, layer.v_scale = Scalar(), Scalar()
            layer.k_scale_float = layer.v_scale_float = 1.0
            adapter.forward_extend(*case.inputs(batch), layer, batch)
            self.assertEqual(case.native_count("uq_bf16_append_v2"), rank + 1)
        variants = [Scalar(2.0), Scalar(float("nan")), Scalar(shape=(1,)),
                    Scalar(dtype="bfloat16"), Scalar(device="wrong-device")]
        for name in ("k_scale", "v_scale"):
            for value, unit in [(v, 1.0) for v in variants] + [(Scalar(), 2.0)]:
                pool, _, adapter, batch = case.batch_adapter()
                adapter.init_forward_metadata(batch)
                layer = case.layer()
                layer.k_scale, layer.v_scale = Scalar(), Scalar()
                layer.k_scale_float = layer.v_scale_float = 1.0
                setattr(layer, name, value)
                setattr(layer, name + "_float", unit)
                before = (len(case.substitutes.calls), len(case.substitutes.allocations))
                with self.assertRaisesRegex(ValueError, "scale"):
                    adapter.forward_extend(*case.inputs(batch), layer, batch)
                self.assertEqual((len(case.substitutes.calls), len(case.substitutes.allocations)), before)
                self.assertEqual(adapter._written_layers, set())
        pool, _, adapter, batch = case.batch_adapter()
        adapter.init_forward_metadata(batch)
        layer = case.layer()
        for explicit in (Scalar(), 2.0, float("nan")):
            before = len(case.substitutes.calls)
            with self.assertRaisesRegex(ValueError, "scale"):
                adapter.forward_extend(*case.inputs(batch), layer, batch, k_scale=explicit)
            self.assertEqual(len(case.substitutes.calls), before)

    def fixture(self, kind=integration.IntegrationTests):
        case = kind()
        case.setUp()
        self.addCleanup(case.doCleanups)
        # Preserve every retained test. This fixture replaces only its source
        # bindings with the effective candidate and refreshes inert proof hashes.
        for name in ("ultraquant_backend.py", "ultraquant_evidence.py"):
            shutil.copyfile(SRT / "layers/attention" / name,
                            case.fixture.package / "srt/layers/attention" / name)
        case.fixture.refresh_inventory()
        path = case.fixture.package / "srt/layers/attention/ultraquant_backend.py"
        names = {"Native", "BF16CacheDescriptor", "BF16Native", "QwenBF16MixedPool", "QwenBF16SourceAdapter"}
        nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name in names]
        base_path = SRT / "layers/attention/base_attn_backend.py"
        bases = [n for n in ast.parse(base_path.read_bytes()).body if isinstance(n, ast.ClassDef)]
        namespace = {"ctypes": ctypes, "Path": Path, "__file__": str(path), "torch": case.substitutes.torch,
                     "KVCache": case.substitutes.base, "causal_rows": uq.causal_rows, "ABC": ABC, "Enum": Enum,
                     "debug_kernel_api": lambda function: function, "is_npu": lambda: False}
        integration.execute(bases + nodes, namespace, path)
        case.classes = NS(**{name: namespace[name] for name in names})
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.layers.attention.ultraquant_evidence": evidence,
            "sglang.srt.layers.attention.ultraquant_backend": case.classes}))
        self.enterContext(patch.object(integration, "STOCK", STOCK))
        return case

    def test_failed_in_callback_revokes_old_batch_for_both_ranks(self):
        case = self.fixture()
        for rank in (0, 1):
            for reason in ("other-batch", "mode", "speculation", "capture"):
                pool, _, adapter, batch = case.batch_adapter(rank)
                dispatch = integration.hybrid(adapter)
                dispatch.init_forward_metadata(batch)
                other = NS(**vars(batch)) if reason == "other-batch" else batch
                if reason == "mode":
                    other.forward_mode = integration.Mode("MIXED")
                if reason == "speculation":
                    other.spec_info = object()
                with patch.object(case.substitutes.torch.cuda, "is_current_stream_capturing",
                                  return_value=reason == "capture"):
                    with self.assertRaises(ValueError):
                        dispatch.init_forward_metadata_in_graph(other)
                self.assertIsNone(adapter.metadata)
                self.assertIsNone(adapter._device_metadata)
                self.assertIsNone(adapter._slot_snapshot)
                self.assertIsNone(adapter._batch)
                self.assertEqual(pool.immutable, frozenset())
                batch.forward_mode, batch.spec_info = integration.Mode("EXTEND"), None
                with self.assertRaisesRegex(ValueError, "missing or stale"):
                    dispatch.forward(*case.inputs(batch), case.layer(), batch)
        self.assertEqual(case.native_count("uq_bf16_append_v2"), 0)
        self.assertEqual(case.native_count("uq_bf16_attention_v2"), 0)

    def test_reader_tensors_created_once_and_replaced_atomically(self):
        case = self.fixture()
        for rank in (0, 1):
            pool, _, adapter, batch = case.batch_adapter(rank, prefix=2)
            dispatch = integration.hybrid(adapter)
            before = len(case.substitutes.allocations)
            dispatch.init_forward_metadata(batch)
            tensors = adapter._device_metadata
            self.assertEqual([x.values for x in tensors], [[9, 4, 12], [0], [3]])
            self.assertEqual([x.dtype for x in tensors], ["int64", "int32", "int32"])
            q, _, _ = case.inputs(batch)
            for layer in uq.QWEN_FULL_LAYERS:
                dispatch.forward(q, None, None, case.layer(layer), batch, save_kv_cache=False)
                args = case.substitutes.calls[-1][1]
                self.assertEqual((args[4], args[6], args[7]), tuple(t.data_ptr() for t in tensors))
                self.assertIs(adapter._device_metadata, tensors)
            allocations = case.substitutes.allocations[before:]
            self.assertEqual(sum(x[0] == "tensor" for x in allocations), 3)
            self.assertEqual(sum(x[0] == "empty" for x in allocations), 16)
            self.assertEqual(case.native_count("uq_bf16_append_v2"), 0)
            replacement = NS(**vars(batch))
            dispatch.init_forward_metadata(replacement)
            self.assertNotEqual(tuple(t.data_ptr() for t in tensors),
                                tuple(t.data_ptr() for t in adapter._device_metadata))
            pool.owner.invalidate()
            self.assertIsNone(adapter._device_metadata)
            with self.assertRaisesRegex(ValueError, "missing or stale"):
                dispatch.forward(q, None, None, case.layer(), replacement, save_kv_cache=False)

    def test_prepare_allocation_failure_preserves_prefix_but_revokes_metadata(self):
        case = self.fixture()
        pool, _, adapter, batch = case.batch_adapter(prefix=2)
        adapter.init_forward_metadata(batch)
        real_tensor = case.substitutes.torch.tensor
        calls = []
        def failing_tensor(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise MemoryError("inert metadata allocation failure")
            return real_tensor(*args, **kwargs)
        with patch.object(case.substitutes.torch, "tensor", side_effect=failing_tensor):
            with self.assertRaises(MemoryError):
                adapter.init_forward_metadata(batch)
        self.assertIsNone(adapter.metadata)
        self.assertIsNone(adapter._device_metadata)
        self.assertEqual(pool.immutable, frozenset((9, 4)))
        self.assertEqual(case.native_count("uq_bf16_append_v2"), 0)
        self.assertEqual(case.native_count("uq_bf16_attention_v2"), 0)

    def test_changed_request_map_clears_host_and_device_metadata(self):
        case = self.fixture()
        _, _, adapter, batch = case.batch_adapter()
        adapter.init_forward_metadata(batch)
        adapter.req_to_token_pool.req_to_token.rows[0] = [9, 4, 13]
        with self.assertRaisesRegex(ValueError, "request map changed"):
            adapter.forward(*case.inputs(batch), case.layer(), batch)
        self.assertIsNone(adapter.metadata)
        self.assertIsNone(adapter._device_metadata)
        self.assertEqual(case.native_count("uq_bf16_append_v2"), 0)

    def test_modern_actual_strided_caller_chain_retains_logical_bits(self):
        case = self.fixture(strided.StridedCallerTests)
        original = strided.actual_method
        def actual_method(path, name, namespace):
            namespace["_use_aiter"] = False
            return original(path, name, namespace)
        with patch.object(strided, "STOCK", STOCK), patch.object(strided, "actual_method", actual_method):
            for rank in (0, 1):
                for scenario in ("initial", "continuation", "decode", "single"):
                    for physical in (3, 11, 59, 63):
                        _, adapter, batch, dispatch = case.setup_batch(rank, scenario)
                        tensors = adapter._device_metadata
                        seen = case.caller(batch, dispatch, physical, rank)
                        seen["layer"].use_prefill_attention_wrapper = True
                        output = seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)
                        self.assertEqual(output.shape, (batch.out_cache_loc.numel(), 3072))
                        case.assert_order(seen, batch.out_cache_loc.numel(), physical in (3, 59, 63))
                        self.assertIs(adapter._device_metadata, tensors)
                        with self.assertRaisesRegex(ValueError, "already appended"):
                            seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)

    def test_deep_actual_unified_nodes_do_not_recurse_or_concatenate(self):
        ns = trees.load_tree()
        depth = sys.getrecursionlimit() + 100
        self.assertLess(depth, 4096)
        owner = evidence.BF16CacheOwner({"canonical": b"INERT deep tree"})
        owner.bind_allocator(NS(page_size=1), depth + 1)
        owner.allocate(range(1, depth + 2))
        tree = cache(owner)
        parent = tree.root_node
        for slot in range(1, depth + 1):
            node = ns["UnifiedTreeNode"]((ns["ComponentType"].FULL,))
            node.parent = parent
            node.component_data[0].value = Ints([slot])
            node.component_data[0].lock_ref = 0
            parent.children[slot] = node
            parent = node
        # Source-bound old recursive operation fails on these same real nodes.
        with self.assertRaises(RecursionError):
            tree.all_values_flatten()
        with patch.object(tree, "all_values_flatten", side_effect=AssertionError("recursive flatten used")):
            self.assertEqual(owner.cached_slots(), frozenset(range(1, depth + 1)))
            owner.check_writable([depth + 1])
            with self.assertRaisesRegex(ValueError, "immutable"):
                owner.check_writable([1])
        parent.children["cycle"] = tree.root_node.children[1]
        with self.assertRaisesRegex(ValueError, "invalid BF16 prefix ownership tree"):
            owner.cached_slots()

    def test_ownership_stays_live_until_actual_tree_release(self):
        owner, allocator = setup()
        values = allocator.alloc(3)
        tree = cache(owner)
        child = NS(children={}, component_data=[NS(value=values[:2], lock_ref=0)])
        tree.root_node.children["out-of-batch"] = child
        old = owner.snapshot([1])
        owner.invalidate()
        with self.assertRaisesRegex(ValueError, "actual cache release"):
            allocator.free(values[:1])
        child.component_data[0].value = None
        allocator.free(values[:2])
        self.assertFalse(owner.cached_slots())
        with self.assertRaisesRegex(ValueError, "stale"):
            owner.validate_snapshot(old)
        self.assertFalse(hasattr(owner, "release_cached"))
        self.assertFalse(hasattr(owner, "_release_permission"))

    def test_classic_cache_refuses_binding_without_hidden_compatibility(self):
        owner, _ = setup()
        classic = NS(page_size=1, root_node=NS(children={}))
        with self.assertRaisesRegex(ValueError, "Python unified"):
            owner.bind_cache(classic)
        self.assertIsNone(owner._cache)

    def test_oversized_or_unallocated_cached_values_refuse(self):
        owner, allocator = setup()
        allocator.alloc(1)
        tree = cache(owner)
        child = NS(children={}, component_data=[NS(value=Ints(range(1, 9)))])
        tree.root_node.children[1] = child
        with self.assertRaisesRegex(ValueError, "ownership size"):
            owner.cached_slots()
        child.component_data[0].value = Ints([2])
        with self.assertRaisesRegex(ValueError, "not live"):
            owner.cached_slots()
