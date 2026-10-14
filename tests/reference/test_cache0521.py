"""Pinned caller bodies; CPU integer transport only, no runtime imports."""
import abc
import ast
from array import array
from collections import defaultdict
import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch

from test_cache_lifecycle import CPU, Ints, classes
from test_bf16_integration import execute
from test_runtime_contract import uq

OWNED = Path(__file__).resolve().parents[2]
SRT = OWNED / "prototype/image-source/python/sglang/srt"
STOCK = OWNED / "tests/fixtures/stock-v0521/python/sglang/srt"
spec = importlib.util.spec_from_file_location("cache0521_evidence", SRT / "layers/attention/ultraquant_evidence.py")
evidence = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_contract": uq}):
    spec.loader.exec_module(evidence)


def method(relative, cls, name, ns=None, stock=False):
    path = (STOCK if stock else SRT) / relative
    tree = ast.parse(path.read_bytes())
    parent = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {"torch": CPU, "array": array, "defaultdict": defaultdict}
    namespace.update(ns or {})
    execute([node], namespace, path)
    return namespace[name]


def setup(rank=0, sorted_free=False, baseline=False):
    ns = {"abc": abc, "torch": CPU}
    classes(SRT / "mem_cache/allocator/base.py", {"BaseTokenToKVPoolAllocator"}, ns)
    classes(SRT / "mem_cache/allocator/token.py", {"TokenToKVPoolAllocator"}, ns)
    owner = None if baseline else evidence.BF16CacheOwner({"canonical": f"INERT-rank-{rank}".encode()})
    allocator = ns["TokenToKVPoolAllocator"](7, "bf16", "CPU", NS(full_kv_pool=NS(owner=owner)), sorted_free)
    return owner, allocator


class Cache:
    pass


def cache(owner):
    result = Cache()
    result._uq_owner, result.page_size = owner, 1
    result._tree_core_backend = "python"
    result.root_node = NS(children={})
    result.tree_core = NS(root_node=result.root_node, device="CPU")
    flatten = method("mem_cache/unified_cache/unified_tree_core.py", "UnifiedTreeCore", "all_values_flatten", {"BASE_COMPONENT_TYPE": 0})
    result.all_values_flatten = lambda: flatten(result.tree_core)
    if owner:
        owner.bind_cache(result)
    return result


def key_classes():
    ns = {"array": array, "torch": CPU, "dataclasses": dataclasses}
    classes(SRT / "mem_cache/radix_cache.py", {"RadixKey"}, ns)
    classes(SRT / "mem_cache/base_prefix_cache.py", {"MatchPrefixParams", "InsertParams"}, ns)
    return ns


class Cache0521Tests(unittest.TestCase):
    def test_page_one_alloc_failed_empty_grouped_free_and_reuse(self):
        for rank in (0, 1):
            owner, alloc = setup(rank, True)
            slots = alloc.alloc(7)
            before = owner.snapshot(slots)
            self.assertIsNone(alloc.alloc(1))
            alloc.free(Ints([]))
            alloc.free_group_begin()
            alloc.free(slots[:2])
            with self.assertRaisesRegex(ValueError, "stale"):
                owner.validate_snapshot(before)
            self.assertEqual(owner._pending, {1, 2})
            alloc.free_group_end()
            self.assertFalse(owner._pending)
            fresh = alloc.alloc(2)
            self.assertEqual(fresh.values, [1, 2])
            self.assertNotEqual(owner.snapshot(fresh), before[:2])
            with self.assertRaises(ValueError):
                alloc.free(Ints([1, 1]))
            with self.assertRaises(ValueError):
                alloc.clear()
            with self.assertRaises(ValueError):
                alloc.resize(NS(max_total_num_tokens=9))
            self.assertEqual(alloc.size, 7)

    def test_owner_absent_keeps_allocator_trace(self):
        for sorted_free in (False, True):
            owner, alloc = setup(baseline=True, sorted_free=sorted_free)
            self.assertIsNone(owner)
            slots = alloc.alloc(7)
            alloc.free_group_begin(); alloc.free(slots[1:3]); alloc.free_group_end()
            self.assertEqual(alloc.alloc(2).values, [2, 3])
            alloc.clear()
            self.assertEqual(alloc.alloc(7).values, list(range(1, 8)))

    def test_actual_tree_values_protect_zero_reference_out_of_batch_prefix(self):
        owner, alloc = setup()
        slots = alloc.alloc(4)
        tree = cache(owner)
        node = NS(children={}, component_data=[NS(value=slots[:2], lock_ref=0)])
        tree.root_node.children["outside-current-batch"] = node
        self.assertEqual(owner.cached_slots(), frozenset((1, 2)))
        with self.assertRaisesRegex(ValueError, "immutable"):
            owner.check_writable([1])
        with self.assertRaisesRegex(ValueError, "actual cache release"):
            alloc.free(Ints([1]))
        self.assertEqual(owner.snapshot([1]), ((1, 1),))
        node.component_data[0].value = None  # declared tree-core tombstone boundary
        alloc.free(Ints([1, 2]))
        self.assertEqual(owner.cached_slots(), frozenset())

    def test_actual_full_component_split_preserves_ownership(self):
        owner, alloc = setup()
        tree = cache(owner)
        slots = alloc.alloc(4)
        cd = lambda value: NS(value=value, host_value=None, lock_ref=0, session_ref=0, session_ids=None)
        child = NS(children={}, component_data=[cd(slots)])
        parent = NS(key=[10, 11], children={"tail": child}, component_data=[cd(None)])
        split = method("mem_cache/unified_cache/components/full.py", "FullComponent", "redistribute_on_node_split")
        split(NS(component_type=0), parent, child)
        tree.root_node.children["prefix"] = parent
        self.assertEqual(owner.cached_slots(), frozenset((1, 2, 3, 4)))
        self.assertEqual(parent.component_data[0].value.values, [1, 2])
        self.assertEqual(child.component_data[0].value.values, [3, 4])

    def test_actual_unified_match_insert_namespace_no_tag_rewrite(self):
        ns = key_classes()
        Key = ns["RadixKey"]
        for rank in (0, 1):
            owner, alloc = setup(rank)
            slots = alloc.alloc(2)
            tree = cache(owner)
            tree._uq_params = MethodType(method("mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "_uq_params", ns), tree)
            tree.disable = False; tree.linker = None
            tree.session = NS(try_match_prefix=lambda p: None)
            tree._components_tuple = (); tree._apply_cache_actions = lambda actions: None
            seen = []
            tree.tree_core.match_prefix = lambda p: (seen.append(p) or NS(cache_actions=[]))
            tree.tree_core.has_ongoing_insert = lambda: False
            tree.tree_core.begin_insert = lambda p: (seen.append(p) or NS(actions=[], result=NS(cache_actions=[])))
            tree.tree_core.end_insert = lambda: []
            match = method("mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "match_prefix")
            insert = method("mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "insert")
            raw = Key(array("q", [10, 11]), "user/LoRA", cache_salt="salt")
            req = NS()
            match(tree, ns["MatchPrefixParams"](raw, req=req))
            insert(tree, ns["InsertParams"](key=raw, value=slots))
            match(tree, ns["MatchPrefixParams"](seen[0].key, req=req))
            self.assertIsNone(raw.kv4_namespace)
            self.assertEqual((raw.extra_key, raw.cache_salt), ("user/LoRA", "salt"))
            self.assertEqual(seen[0].key.child_key(), seen[1].key.child_key())
            self.assertEqual(seen[0].key[:1].kv4_namespace, seen[0].key.kv4_namespace)
            foreign, _ = setup(1-rank)
            raw.kv4_namespace = foreign.namespace((raw.extra_key, raw.cache_salt))
            with self.assertRaisesRegex(ValueError, "foreign"):
                match(tree, ns["MatchPrefixParams"](raw))
            self.assertEqual(len(seen), 3)

    def test_rust_refuses_before_tree_core_selection(self):
        owner, alloc = setup()
        params = NS(token_to_kv_pool_allocator=alloc, tree_core_backend="rust")
        calls = []
        init = method("mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "__init__", {"select_tree_core_backend": lambda p: calls.append(p)})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            init(NS(), params)
        self.assertEqual(calls, [])

    def test_release_cancellation_recurrent_trace_same_as_stock(self):
        for holds_kv in (False, True):
            traces = []
            for stock in (True, False):
                trace = []
                owner, alloc = setup()
                owner.invalidate = lambda: trace.append("invalidate")
                kv = NS(holds_kv=holds_kv, is_kv_released=not holds_kv,
                        holds_mamba=True, mamba_pool_idx=Ints([4]), cache_protected_len=1,
                        kv_allocated_len=3, mark_kv_released=lambda: trace.append("released"))
                req = NS(kv=kv, owned_kv_len=lambda: 3, skip_radix_cache_insert=False, _uq_owner=owner)
                pool = NS(mamba_allocator=NS(free=lambda x: trace.append(("mamba-free", x.values))),
                          free=lambda r: trace.append("req-free"))
                tree = NS(_uq_owner=owner, req_to_token_pool=pool, supports_mamba=lambda: True,
                          claim_kv_row=lambda r: False, insert_req=lambda *a, **k: trace.append("insert"),
                          free_kv_row=lambda *a: trace.append("free-row"), unpin=lambda r: trace.append("unpin"),
                          on_release=lambda *a, **k: trace.append("on-release"))
                fn = method("mem_cache/common.py", None, "release_kv_cache", {
                    "HybridReqToTokenPool": type("UnusedHybrid", (), {}),
                    "_release_overallocated_kv_indices": lambda *a: trace.append("overallocated")}, stock=stock)
                fn(req, tree, is_insert=False)
                traces.append([event for event in trace if event != "invalidate"])
                if not stock:
                    self.assertEqual(trace[0], "invalidate")
            self.assertEqual(*traces)

    def test_retract_actual_body_preserves_state_reset_trace(self):
        results = []
        for stock in (True, False):
            invalidated = []
            req = NS(_uq_owner=NS(invalidate=lambda: invalidated.append(True)),
                     retraction_count=0, kv=NS(holds_kv=False), input_embeds=None)
            fn = method("managers/schedule_batch.py", "Req", "reset_for_retract",
                        {"DecLockRefParams": lambda: "INERT-lock"}, stock=stock)
            fn(req)
            results.append(vars(req.kv))
            self.assertEqual(req.retraction_count, 1)
            self.assertEqual(invalidated, [] if stock else [True])
        self.assertEqual(*results)

    def test_batch_replacement_invalidates_before_empty_filter(self):
        invalidated = []
        batch = NS(tree_cache=NS(_uq_owner=NS(invalidate=lambda: invalidated.append(True))), reqs=[object()])
        fn = method("managers/schedule_batch.py", "ScheduleBatch", "filter_batch", {
            "strip_beam_tail": lambda b: None, "CaptureHiddenMode": NS(NULL="null")})
        fn(batch, keep_indices=[])
        self.assertEqual(batch.reqs, [])
        self.assertEqual(invalidated, [True])

    def test_unsupported_restore_offload_refuse(self):
        owner, alloc = setup()
        with self.assertRaisesRegex(ValueError, "offload"):
            alloc.get_cpu_copy(Ints([]))
        with self.assertRaisesRegex(ValueError, "restore"):
            alloc.load_cpu_copy(None, Ints([]))

    def test_classic_and_post_capture_refuse_before_mutation_or_device_query(self):
        owner, alloc = setup()
        fn = method("mem_cache/radix_cache.py", "RadixCache", "__init__")
        target = NS()
        with self.assertRaisesRegex(ValueError, "Python unified"):
            fn(target, NS(token_to_kv_pool_allocator=alloc))
        self.assertEqual(vars(target), {})
        resize = method("model_executor/model_runner_components/kv_pool_runtime.py", None,
                        "compute_post_capture_kv_resize")
        with self.assertRaisesRegex(ValueError, "post-capture"):
            resize(NS(token_to_kv_pool=NS(full_kv_pool=NS(owner=owner))))

    def test_mamba_actual_action_drain_trace_matches_stock(self):
        class Evict:
            tail_node_id = 13
        class Free:
            indices = [Ints([4]), Ints([9])]
        traces = []
        for stock in (True, False):
            trace = []
            fn = method("mem_cache/unified_cache/components/mamba.py", "MambaComponent", "apply_component_action", {
                "MambaEvictExcessPathStates": Evict, "FreeComponentDeviceSlot": Free,
                "FreeComponentHostSlot": type("Host", (), {})}, stock=stock)
            component = NS(tree_core=NS(evict_excess_path_states=lambda *a: trace.append(("evict", a[0]))),
                cache=NS(_free_values=lambda *a: trace.append("drain")),
                _free_mamba_value=lambda values: trace.append(("free-state", values.values)))
            fn(component, Evict()); fn(component, Free())
            traces.append(trace)
        self.assertEqual(*traces)
        self.assertEqual(traces[0], [("evict", 13), "drain", ("free-state", [4]), ("free-state", [9])])

    def test_failed_prepare_and_foreign_process_namespace(self):
        owner, alloc = setup()
        invalidations = []
        owner.invalidate = lambda: invalidations.append(True)
        tree = cache(owner)
        ns = key_classes()
        fn = method("mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "_uq_params", ns)
        bad = ns["InsertParams"](key=ns["RadixKey"](array("q", [2])), value=Ints([1]))
        with self.assertRaisesRegex(ValueError, "not live"):
            fn(tree, bad)
        self.assertEqual(invalidations, [True])
        with patch.object(evidence.os, "getpid", return_value=owner._pid + 1):
            with self.assertRaisesRegex(ValueError, "cross-process"):
                owner.namespace(("user", "salt"))
