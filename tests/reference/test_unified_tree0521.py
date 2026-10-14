"""Actual pinned Python tree algorithms. CPU tensors and message containers only."""
import abc
import ast
from array import array
from collections import defaultdict
import copy
import dataclasses
from enum import Enum, IntFlag, auto
import heapq
import logging
import sys
import time
import typing
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from test_cache0521 import SRT, STOCK, setup, CPU, Ints, method, uq
from test_bf16_integration import execute


class Struct:
    """Inert msgspec record substitute; no tree decisions are implemented here."""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
    def __init__(self, *args, **kwargs):
        fields = {}
        for cls in reversed(type(self).__mro__):
            fields.update(getattr(cls, "__annotations__", {}))
        for name, value in zip(fields, args):
            kwargs[name] = value
        for name in fields:
            if name in kwargs:
                value = kwargs.pop(name)
            else:
                value = getattr(type(self), name)
                if isinstance(value, dataclasses.Field):
                    value = value.default_factory() if value.default_factory is not dataclasses.MISSING else value.default
                else:
                    value = copy.deepcopy(value)
            setattr(self, name, value)
        if kwargs:
            raise TypeError(kwargs)


class Events:
    def __init__(self, **kw): self.enabled = False
    def __getattr__(self, name): return lambda *a, **kw: None


class Session:
    def __init__(self, **kw): self.slots = {}
    def __getattr__(self, name): return lambda *a, **kw: None


def load_tree(stock=False):
    transport = copy.copy(CPU)
    transport.Tensor = Ints
    transport.distributed = NS(get_world_size=lambda **kw: 1)
    ns = dict(vars(typing))
    ns.update(__name__=__name__, torch=transport, dataclasses=dataclasses,
        msgspec=NS(Struct=Struct, field=dataclasses.field, structs=NS(replace=lambda x, **kw: type(x)(**(vars(x)|kw)))),
        ABC=abc.ABC, abstractmethod=abc.abstractmethod, array=array, defaultdict=defaultdict,
        Enum=Enum, IntFlag=IntFlag, auto=auto, sys=sys, time=time, heapq=heapq,
        logging=logging, float64=float, logger=logging.getLogger(__name__),
        StorageMedium=NS(GPU="inert-device-label", CPU="inert-host-label"),
        KVCacheEventRecorder=Events, UnifiedSessionRefTracker=Session, StreamingSession=Session,
        StorageExistenceCache=Session, StoragePrefetchRetries=Session,
        rank_consensus=lambda f=None, **kw: f if f is not None else lambda fn: fn,
        page_interleave_shard_size=lambda allocator: 1,
        get_eviction_strategy=lambda *a: NS(get_priority=lambda node: node.last_access_time),
        split_node_hash_value=lambda *a: (None, None),
        compute_node_hash_values=lambda *a: [],
        mamba_cache_chunk_size=lambda: 1, mamba_checkpoint_grid=lambda page: 1,
        get_exec=lambda: NS(mamba=NS(mamba_max_states_per_path=-1)),
        envs=NS(SGLANG_ENABLE_HIERARCHICAL_CACHE=NS(get=lambda: False)))
    def load(relative, names=None):
        path = (STOCK if stock else SRT) / relative
        tree = ast.parse(path.read_bytes())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and (names is None or n.name in names)]
        execute(nodes, ns, path)
    load("mem_cache/unified_cache/component_type.py")
    ns["BASE_COMPONENT_TYPE"] = ns["ComponentType"].FULL
    load("mem_cache/base_prefix_cache.py")
    load("mem_cache/radix_cache.py", {"RadixKey"})
    load("mem_cache/cache_init_params.py")
    ns.update(_NUM_COMPONENT_TYPES=len(ns["ComponentType"]), _LAST_ACCESS_TIME_COUNTER_FLOAT=1.0, _COMPONENT_UUID_COUNTER=1)
    load("mem_cache/unified_cache/components/base.py")
    load("mem_cache/unified_cache/cache_action.py")
    load("mem_cache/unified_cache/unified_tree_core_interface.py")
    ns["NodeId"] = int
    ns["_RECLAIM_DIGEST_MASK"] = (1 << 42)-1
    load("mem_cache/unified_cache/unified_tree_core.py")
    load("mem_cache/unified_cache/components/full.py")
    load("mem_cache/unified_cache/components/mamba.py")
    ns["COMPONENT_REGISTRY"] = {ns["ComponentType"].FULL: ns["FullComponent"], ns["ComponentType"].MAMBA: ns["MambaComponent"]}
    ns["select_tree_core_backend"] = lambda params: params.tree_core_backend
    ns["create_tree_core"] = lambda name, params, components: ns["UnifiedTreeCore"](params, components)
    load("mem_cache/unified_radix_cache.py")
    return ns


class UnifiedTree0521Tests(unittest.TestCase):
    def make(self, rank=0, stock=False, mamba=False):
        ns = load_tree(stock)
        owner, allocator = setup(rank, baseline=stock)
        class HybridReqCPU:
            pass
        req_pool = HybridReqCPU()
        _, req_pool.mamba_allocator = setup(baseline=True)
        components = (ns["ComponentType"].FULL, ns["ComponentType"].MAMBA) if mamba else (ns["ComponentType"].FULL,)
        params = ns["CacheInitParams"](disable=False, req_to_token_pool=req_pool, token_to_kv_pool_allocator=allocator,
            page_size=1, tree_components=components, tree_core_backend="python")
        with patch.dict(sys.modules, {"sglang.srt.mem_cache.memory_pool": NS(HybridReqToTokenPool=HybridReqCPU)}):
            cache = ns["UnifiedRadixCache"](params)
        return ns, owner, allocator, cache

    def test_real_constructor_insert_duplicate_split_rematch_evict_reset(self):
        for rank in (0, 1):
            ns, owner, alloc, cache = self.make(rank)
            key = lambda ids: ns["RadixKey"](array("q", ids), "user", cache_salt="salt")
            slots = alloc.alloc(4)
            old = owner.snapshot(slots)
            first = cache.insert(ns["InsertParams"](key=key([1,2,3,4]), value=slots))
            self.assertEqual(owner.cached_slots(), frozenset((1,2,3,4)))
            with self.assertRaisesRegex(ValueError, "owned slots"):
                cache.reset()
            duplicate = alloc.alloc(2)
            result = cache.insert(ns["InsertParams"](key=key([1,2]), value=duplicate))
            self.assertEqual(result.prefix_len, 2)
            # The actual full-component insertion action releases duplicates.
            with self.assertRaisesRegex(ValueError, "not live"):
                owner.snapshot(duplicate.tolist())
            match = cache.match_prefix(ns["MatchPrefixParams"](key=key([1,2])))
            self.assertEqual(match.device_indices.tolist(), [1,2])
            observations = []
            original_free = alloc.free
            def observed_free(values):
                observations.append((values.tolist(), owner.cached_slots()))
                original_free(values)
            alloc.free = observed_free
            result = cache.evict(ns["EvictParams"](num_tokens=4))
            self.assertEqual(result.num_tokens_evicted, 4)
            self.assertTrue(observations)
            for released, cached in observations:
                self.assertFalse(set(released) & cached)
            self.assertFalse(owner.cached_slots())
            with self.assertRaisesRegex(ValueError, "stale"):
                owner.validate_snapshot(old)
            cache.reset(); alloc.clear()
            self.assertEqual(alloc.alloc(4).tolist(), [1,2,3,4])
            self.assertNotEqual(owner.snapshot([1,2,3,4]), old)

    def test_stock_candidate_full_tree_trace(self):
        traces = []
        for stock in (True, False):
            ns, owner, alloc, cache = self.make(stock=stock)
            slots = alloc.alloc(3)
            key = ns["RadixKey"](array("q", [5,6,7]), "same", cache_salt="same-salt")
            insert = cache.insert(ns["InsertParams"](key=key, value=slots))
            match = cache.match_prefix(ns["MatchPrefixParams"](key=key))
            result = cache.evict(ns["EvictParams"](num_tokens=3))
            traces.append((insert.prefix_len, match.device_indices.tolist(), result.num_tokens_evicted, alloc.available_size()))
        self.assertEqual(*traces)

    def test_actual_mamba_tree_cow_and_state_release_trace(self):
        traces = []
        for stock in (True, False):
            ns, owner, alloc, cache = self.make(stock=stock, mamba=True)
            states = cache.req_to_token_pool.mamba_allocator
            state_trace = []
            state_alloc, state_free = states.alloc, states.free
            def allocate(count):
                result = state_alloc(count)
                state_trace.append(("alloc", result.tolist() if result is not None else None))
                return result
            def release(values):
                state_trace.append(("free", values.tolist()))
                state_free(values)
            states.alloc, states.free = allocate, release
            slots = alloc.alloc(3)
            state = states.alloc(1)
            key = ns["RadixKey"](array("q", [8,9,10]), "user", cache_salt="salt")
            cache.insert(ns["InsertParams"](key=key, value=slots, mamba_value=state))
            req = NS(kv=NS(holds_mamba=False))
            matched = cache.match_prefix(ns["MatchPrefixParams"](key=key, cow_mamba=True, req=req))
            self.assertEqual(req.kv.mamba_cow_src_index.tolist(), state.tolist())
            self.assertFalse(req.kv.mamba_needs_clear)
            states.free(Ints([req.kv.mamba_pool_idx]))
            cache.evict(ns["EvictParams"](num_tokens=3, mamba_num=1))
            cache.reset(); alloc.clear()
            traces.append((matched.device_indices.tolist(), state_trace, states.available_size()))
        self.assertEqual(*traces)

    def test_stale_generation_reaches_actual_adapter_read_and_append_gates(self):
        owner, alloc = setup()
        first = alloc.alloc(7)
        snapshot = owner.snapshot([1])
        alloc.free(Ints([1])); self.assertEqual(alloc.alloc(1).tolist(), [1])
        calls = []
        batch = object()
        adapter = NS(pool=NS(owner=owner, native=NS(call=lambda *a: calls.append(a))),
                     metadata=((1,), (0,), (1,)), _slot_snapshot=snapshot, _batch=batch)
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_contract": uq}):
            forward = method("layers/attention/ultraquant_backend.py", "QwenBF16SourceAdapter", "_forward")
            attention = method("layers/attention/ultraquant_backend.py", "QwenBF16SourceAdapter", "attention")
            with self.assertRaisesRegex(ValueError, "stale BF16 slot generation"):
                forward(adapter, None, None, None, None, batch, True, {})
            with self.assertRaisesRegex(ValueError, "stale BF16 slot generation"):
                attention(adapter, None, 11, 0.0625)
        self.assertEqual(calls, [])
