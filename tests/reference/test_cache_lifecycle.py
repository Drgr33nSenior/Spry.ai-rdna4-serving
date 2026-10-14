"""Actual cache/allocator/release bodies with declared CPU integer transport."""
import abc
import ast
from array import array
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
import heapq
import gc
import logging
from functools import lru_cache
import sys
import time
from types import MethodType, SimpleNamespace
from typing import NamedTuple, Any
from unittest.mock import patch
import test_bf16_integration as integration
from test_bf16_closure import Tensor
from test_runtime_contract import ROOT
from test_runtime_evidence import evidence

SRT = ROOT / "image-source/python/sglang/srt"
STOCK = ROOT.parent / "stock/srt"


class Ints:
    """Integer transport only; no recurrent or attention computation."""
    def __init__(self, values, dtype="int64"):
        self.values, self.dtype = list(values), dtype
        self.device = "CPU"
        self.shape = (len(self.values),)
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        return iter(self.values)
    def __getitem__(self, span):
        return Ints(self.values[span], self.dtype) if isinstance(span, slice) else self.values[span]
    def __setitem__(self, span, value):
        self.values[span] = list(value)
    def numel(self):
        return len(self.values)
    def cpu(self):
        return self
    def tolist(self):
        return list(self.values)
    def to(self, dtype=None, copy=False, **kwargs):
        return Ints(self.values, dtype or self.dtype)
    def clone(self):
        return Ints(self.values, self.dtype)
    def unsqueeze(self, unused):
        return self
    def item(self):
        assert len(self.values) == 1
        return self.values[0]


class Map:
    def __init__(self, rows):
        self.rows = rows
    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, span = key
            return self.rows[row][span]
        return self.rows[key]


CPU = SimpleNamespace(int64="int64", int32="int32", device=str,
    arange=lambda begin, end, **kw: Ints(range(begin, end)),
    empty=lambda shape, **kw: Ints([]),
    tensor=lambda values, **kw: Ints(values),
    cat=lambda groups: Ints([x for group in groups for x in group]),
    sort=lambda values: (Ints(sorted(values.values)), None),
    equal=lambda a, b: a.values == b.values)


def classes(path, names, ns):
    tree = ast.parse(path.read_bytes())
    return integration.execute([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], ns, path)


@lru_cache(maxsize=1)
def allocator_classes():
    ns = {"abc": abc, "torch": CPU}
    classes(SRT / "mem_cache/allocator/base.py", {"BaseTokenToKVPoolAllocator"}, ns)
    classes(SRT / "mem_cache/allocator/token.py", {"TokenToKVPoolAllocator"}, ns)
    return ns["TokenToKVPoolAllocator"]


class Events:
    def _record_all_cleared_event(self):
        pass
    def _record_store_event(self, node):
        pass
    def _record_remove_event(self, node):
        pass


class Sessions:
    def _reset_session_radix_state(self):
        pass
    def _tag_session_leaf(self, *args, **kwargs):
        pass
    def _discard_session_leaf(self, node):
        pass


def radix_classes(stock=False):
    base = SRT / "mem_cache/base_prefix_cache.py"
    tree = ast.parse(base.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BasePrefixCache")
    hooks = [n for n in cls.body if isinstance(n, ast.FunctionDef) and (
        n.name.startswith("_uq_") or n.name == "update_eviction_metrics")]
    hooks += [n for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
              and n.target.id == "metrics_collector"]
    node = ast.ClassDef(name="BasePrefixCache", bases=[ast.Name(id="object", ctx=ast.Load())],
                       keywords=[], body=hooks, decorator_list=[])
    ns = {"torch": CPU, "array": array, "defaultdict": defaultdict, "sys": sys, "time": time,
          "heapq": heapq, "logger": logging.getLogger(__name__),
          "KVCacheEventMixin": Events, "SessionRadixCacheMixin": Sessions,
          "get_eviction_strategy": lambda name: SimpleNamespace(get_priority=lambda node: node.last_access_time),
          "split_node_hash_value": lambda values, split, page: (None, None),
          "get_hash_str": lambda *args: "CPU inert hash",
          "get_last_access_time": time.monotonic, "get_tensor_model_parallel_rank": lambda: 0}
    data = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in
            {"MatchPrefixParams", "InsertParams", "InsertResult", "EvictParams", "EvictResult",
             "IncLockRefResult", "DecLockRefParams", "DecLockRefResult", "MatchResult"}]
    ns.update(dataclasses=dataclasses, NamedTuple=NamedTuple, Any=Any)
    integration.execute(data+[node], ns, base)
    path = (STOCK if stock else SRT) / "mem_cache/radix_cache.py"
    classes(path, {"RadixKey", "TreeNode", "RadixCache"}, ns)
    return ns


def mamba_classes(stock=False):
    ns = radix_classes(stock)
    path = (STOCK if stock else SRT) / "mem_cache/mamba_radix_cache.py"
    ns.update(float64=float, TokenToKVPoolAllocator=allocator_classes(), PagedTokenToKVPoolAllocator=type("Paged", (), {}),
              UnifiedMambaTokenToKVPoolAllocator=type("Unified", (), {}),
              get_global_server_args=lambda: SimpleNamespace(mamba_cache_chunk_size=1))
    classes(path, {"TreeNode", "LRUList", "MambaRadixCache"}, ns)
    return ns


class CacheLifecycleTests(integration.IntegrationTests):
    def setup_cache(self, rank=0, need_sort=False, mamba=False, baseline=False):
        pool = self.construct(rank=rank, connect_fixture=False)
        cls = allocator_classes()
        full = SimpleNamespace(full_kv_pool=pool) if not baseline else object()
        allocator = cls(31, "bfloat16", "CPU", full, need_sort)
        ns = mamba_classes(baseline) if mamba else radix_classes(baseline)
        req_pool = SimpleNamespace(req_to_token=Map({}), free_mamba_cache=lambda *a, **k: None)
        params = SimpleNamespace(disable=False, req_to_token_pool=req_pool, token_to_kv_pool_allocator=allocator,
            page_size=1, enable_kv_cache_events=False, enable_session_radix_cache=False, is_eagle=False,
            disable_finished_insert=False, eviction_policy="lru", enable_metrics=False,
            enable_mamba_extra_buffer=False, enable_mamba_extra_buffer_lazy=False)
        cache = ns["MambaRadixCache" if mamba else "RadixCache"](params)
        return pool, allocator, cache, ns

    def key(self, ns, tokens, tag=None):
        return ns["RadixKey"](array("q", tokens), tag)

    def insert(self, cache, ns, values, tokens=(1, 2, 3), tag=None, **kwargs):
        return cache.insert(ns["InsertParams"](key=self.key(ns, tokens, tag), value=values, **kwargs))

    def match(self, cache, ns, tokens=(1, 2, 3), tag=None, **kwargs):
        return cache.match_prefix(ns["MatchPrefixParams"](key=self.key(ns, tokens, tag), **kwargs))

    def test_actual_namespace_match_insert_tags_both_ranks(self):
        for rank in (0, 1):
            pool, allocator, cache, ns = self.setup_cache(rank)
            values = allocator.alloc(3)
            tag = ("user", "lora")
            original = self.key(ns, (1, 2, 3), tag)
            cache.insert(ns["InsertParams"](key=original, value=values))
            self.assertIs(original.extra_key, tag)
            self.assertEqual(self.match(cache, ns, tag=tag).device_indices.tolist(), values.tolist())
            for other in (("other-user", "lora"), ("user", "other-lora"), None):
                self.assertEqual(len(self.match(cache, ns, tag=other).device_indices), 0)
            wrapped = cache._uq_key(original)
            self.assertEqual(cache._uq_key(wrapped).extra_key, wrapped.extra_key)
            self.assertIs(wrapped.extra_key.original_tag, tag)
            foreign = evidence.BF16CacheOwner({"canonical": b"foreign build/rank"}).namespace(tag)
            with self.assertRaisesRegex(ValueError, "foreign"):
                self.match(cache, ns, tag=foreign)
            with patch.object(evidence.os, "getpid", return_value=pool.owner._pid+1):
                with self.assertRaisesRegex(ValueError, "cross-process"):
                    self.match(cache, ns, tag=tag)
            self.assertNotEqual(pool.owner.namespace(True), pool.owner.namespace(1))

    def test_cache_owned_zero_refs_outside_batch_split_duplicate_and_evict(self):
        pool, allocator, cache, ns = self.setup_cache()
        values = allocator.alloc(3)
        result = self.insert(cache, ns, values)
        node = result.last_device_node
        self.assertEqual(node.lock_ref, 0)
        with self.assertRaisesRegex(ValueError, "immutable"):
            pool.owner.check_writable(values.tolist())
        cache.inc_lock_ref(node); cache.inc_lock_ref(node)
        cache.dec_lock_ref(node); cache.dec_lock_ref(node)
        self.assertEqual(pool.owner.cached_slots(), frozenset(values.tolist()))
        self.match(cache, ns, tokens=(1, 2))
        self.assertEqual(pool.owner.cached_slots(), frozenset(values.tolist()))
        duplicate = allocator.alloc(3)
        self.assertEqual(self.insert(cache, ns, duplicate).prefix_len, 3)
        allocator.free(duplicate)
        with self.assertRaisesRegex(ValueError, "actual cache release"):
            allocator.free(values)
        snapshot = pool.owner.snapshot(values.tolist())
        evicted = cache.evict(ns["EvictParams"](num_tokens=3))
        self.assertEqual(evicted.num_tokens_evicted, 3)
        self.assertEqual(pool.owner.cached_slots(), frozenset())
        with self.assertRaisesRegex(ValueError, "stale"):
            pool.owner.validate_snapshot(snapshot)

    def test_actual_allocator_grouped_empty_failure_merge_and_numeric_reuse(self):
        pool, allocator, cache, ns = self.setup_cache(need_sort=True)
        all_slots = allocator.alloc(31)
        before = dict(pool.owner._generations)
        self.assertIsNone(allocator.alloc(1))
        self.assertEqual(before, pool.owner._generations)
        allocator.free(Ints([]))
        snapshot = pool.owner.snapshot([1, 2])
        allocator.free_group_begin()
        allocator.free(Ints([1])); allocator.free(Ints([2]))
        with self.assertRaisesRegex(ValueError, "stale"):
            pool.owner.validate_snapshot(snapshot)
        self.assertIsNone(allocator.alloc(1))
        allocator.free_group_end()
        self.assertEqual(pool.owner._pending, set())
        reused = allocator.alloc(2)
        self.assertEqual(reused.tolist(), [1, 2])
        fresh = pool.owner.snapshot(reused.tolist())
        self.assertNotEqual(snapshot, fresh)
        pool.owner.validate_snapshot(fresh)
        with self.assertRaises(ValueError):
            allocator.free(Ints([1, 1]))
        with self.assertRaises(ValueError):
            allocator.alloc(-1)
        with self.assertRaisesRegex(ValueError, "restore"):
            allocator.restore_state(allocator.backup_state())
        with self.assertRaisesRegex(ValueError, "reset"):
            allocator.clear()
        self.assertLessEqual(len(pool.owner._generations)+len(pool.owner._pending), 31)

    def test_actual_cached_group_free_retires_before_numeric_reuse(self):
        pool, allocator, cache, ns = self.setup_cache(need_sort=True)
        values = allocator.alloc(31)
        self.insert(cache, ns, values[:3])
        snapshot = pool.owner.snapshot(values[:3].tolist())
        allocator.free_group_begin()
        cache.evict(ns["EvictParams"](num_tokens=3))
        self.assertEqual(pool.owner._pending, {1, 2, 3})
        with self.assertRaises(ValueError):
            pool.owner.validate_snapshot(snapshot)
        allocator.free_group_end()
        self.assertEqual(allocator.alloc(3).tolist(), [1, 2, 3])

    def test_actual_request_completion_and_continuation_keep_protection(self):
        pool, allocator, cache, ns = self.setup_cache()
        slots = allocator.alloc(3)
        cache.req_to_token_pool.req_to_token.rows[0] = slots
        writes = []
        cache.req_to_token_pool.write = lambda where, values: writes.append((where, values.tolist()))
        req = SimpleNamespace(req_pool_idx=0, extra_key=("user", "lora"), origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"), cache_protected_len=0, last_node=cache.root_node, priority=0,
            get_fill_ids=lambda: array("q", [1, 2, 3]), pop_committed_kv_cache=lambda: 3)
        cache.cache_unfinished_req(req)
        self.assertEqual(req.cache_protected_len, 3)
        self.assertEqual(pool.owner.cached_slots(), frozenset(slots.tolist()))
        cache.cache_finished_req(req)
        self.assertEqual(pool.owner.cached_slots(), frozenset(slots.tolist()))
        self.assertEqual(req.last_node.lock_ref, 0)
        self.assertEqual(len(writes), 1)

    def test_adapter_prefix_generation_and_request_map_guard_before_calls(self):
        for rank in (0, 1):
            pool, parent, adapter, batch = self.batch_adapter(rank=rank)
            adapter.init_forward_metadata(batch)
            q, k, v = self.inputs(batch)
            before = self.native_count("uq_bf16_attention_v2")
            snapshot = adapter._slot_snapshot
            # Prefix reads must become stale too, not only fresh output slots.
            pool.owner.retire([9])
            self.assertIsNone(adapter.metadata)
            pool.owner.allocate([9])
            with self.assertRaisesRegex(ValueError, "stale"):
                pool.owner.validate_snapshot(snapshot)
            with self.assertRaisesRegex(ValueError, "missing or stale"):
                adapter.forward(q, k, v, self.layer(), batch)
            adapter.init_forward_metadata(batch)
            adapter.forward(q, None, None, self.layer(), batch, save_kv_cache=False)
            self.assertEqual(self.native_count("uq_bf16_attention_v2"), before+1)
            adapter.req_to_token_pool.req_to_token.rows[0] = [9, 4, 13]
            with self.assertRaisesRegex(ValueError, "request map"):
                adapter.forward(q, k, v, self.layer(), batch)
            self.assertIsNone(adapter.metadata)
            self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def test_cache_owned_prefix_append_refuses_outside_current_batch(self):
        pool, allocator, cache, ns = self.setup_cache()
        slots = allocator.alloc(3)
        self.insert(cache, ns, slots)
        parent, _ = integration.actual_parent(pool, self.substitutes)
        loc = Tensor((1,), "int64", self.substitutes.device, [2])
        k = Tensor((1, 2, 256), "bfloat16", self.substitutes.device)
        before = len(self.substitutes.calls)
        with self.assertRaisesRegex(ValueError, "cache-owned"):
            parent.set_kv_buffer(self.layer(11), loc, k, k)
        self.assertEqual(len(self.substitutes.calls), before)

    def test_baseline_without_owner_algorithms_trace_unchanged(self):
        _, allocator, cache, ns = self.setup_cache(baseline=True)
        values = allocator.alloc(3)
        self.insert(cache, ns, values, tag="user")
        self.assertEqual(self.match(cache, ns, tag="user").device_indices.tolist(), [1, 2, 3])
        allocator.free_group_begin()
        cache.evict(ns["EvictParams"](num_tokens=3))
        allocator.free_group_end()
        self.assertEqual(allocator.available_size(), 31)
        allocator.restore_state(allocator.backup_state())
        allocator.clear()
        self.assertIsNone(allocator._uq_owner)

    def test_actual_mamba_match_cow_and_checkpoint_free_trace_matches_baseline(self):
        traces = []
        for baseline in (True, False):
            pool, allocator, cache, ns = self.setup_cache(mamba=True, baseline=baseline)
            trace = []
            state_alloc = SimpleNamespace(alloc=lambda n: trace.append(("state_alloc", n)) or Ints([101]),
                                          free=lambda values: trace.append(("state_free", values.tolist())))
            cache.req_to_token_pool.mamba_allocator = state_alloc
            cache.req_to_token_pool.mamba_pool = SimpleNamespace(replayssm_write_pos=None)
            cache.req_to_token_pool.mamba_ckpt_pool = None
            values = allocator.alloc(3)
            self.insert(cache, ns, values, mamba_value=Ints([100]))
            req = SimpleNamespace(mamba_pool_idx=None)
            matched = self.match(cache, ns, cow_mamba=True, req=req)
            self.assertEqual(matched.device_indices.tolist(), [1, 2, 3])
            self.assertEqual(req.mamba_pool_idx, 101)
            self.assertEqual(req.mamba_cow_src_index.tolist(), [100])
            cache.evict(ns["EvictParams"](num_tokens=3))
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], [("state_alloc", 1), ("state_free", [100])])

    def test_actual_scheduler_abort_batch_filter_and_common_release_invalidate(self):
        pool, parent, adapter, batch = self.batch_adapter()
        adapter.init_forward_metadata(batch)
        cache = SimpleNamespace(_uq_invalidate=pool.owner.invalidate)
        path = SRT / "managers/schedule_batch.py"
        ns = {}
        integration.methods(path, "ScheduleBatch", {"filter_batch"}, ns)
        target = SimpleNamespace(tree_cache=cache, reqs=[object()])
        ns["filter_batch"](target, keep_indices=[])
        self.assertIsNone(adapter.metadata)
        self.assertEqual(target.reqs, [])
        adapter.init_forward_metadata(batch)
        path = SRT / "managers/scheduler.py"
        tree = ast.parse(path.read_bytes())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "abort_request")
        ns = {"DisaggregationMode": SimpleNamespace(DECODE="decode", PREFILL="prefill")}
        integration.execute([method], ns, path)
        target = SimpleNamespace(tree_cache=cache, chunked_req=None, waiting_queue=[], enable_hicache_storage=False,
            disaggregation_mode="null", grammar_manager=SimpleNamespace(abort_requests=lambda req: None),
            running_batch=SimpleNamespace(reqs=[]), cur_batch=None, last_batch=None)
        ns["abort_request"](target, SimpleNamespace(abort_all=True, rid=""))
        self.assertIsNone(adapter.metadata)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def test_actual_common_release_drops_request_not_cached_prefix(self):
        pool, allocator, cache, ns = self.setup_cache()
        values = allocator.alloc(3)
        cache.req_to_token_pool.req_to_token.rows[0] = values
        trace = []
        cache.req_to_token_pool.free = lambda req: trace.append(("request_free", req.req_pool_idx))
        cache.supports_mamba = lambda: False
        req = SimpleNamespace(req_pool_idx=0, origin_input_ids=array("q", [1, 2, 3]), output_ids=array("q"),
            extra_key="user", cache_protected_len=0, last_node=cache.root_node, priority=0,
            pop_committed_kv_cache=lambda: 3, pop_overallocated_kv_cache=lambda: (3, 3),
            kv_committed_len=3, kv_allocated_len=3)
        path = SRT / "mem_cache/common.py"
        tree = ast.parse(path.read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="release_kv_cache")
        env = {"get_global_server_args": lambda: SimpleNamespace(page_size=1, speculative_algorithm=None, strip_thinking_cache=False),
               "HybridReqToTokenPool": type("Hybrid", (), {})}
        integration.execute([function], env, path)
        env["release_kv_cache"](req, cache)
        self.assertEqual(trace, [("request_free", 0)])
        self.assertEqual(pool.owner.cached_slots(), frozenset([1, 2, 3]))
        self.assertEqual(len(pool.owner._generations), 3)

    def test_expired_cache_does_not_release_cached_protection(self):
        pool, allocator, cache, ns = self.setup_cache()
        slots = allocator.alloc(3)
        self.insert(cache, ns, slots)
        snapshot = pool.owner.snapshot(slots.tolist())
        del cache
        gc.collect()
        with self.assertRaisesRegex(ValueError, "owner expired"):
            pool.owner.check_writable([1])
        with self.assertRaisesRegex(ValueError, "owner expired"):
            pool.owner.validate_snapshot(snapshot)


for _name in dir(integration.IntegrationTests):
    if _name.startswith("test_") and _name not in vars(CacheLifecycleTests):
        setattr(CacheLifecycleTests, _name, None)
