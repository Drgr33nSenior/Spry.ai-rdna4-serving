"""Actual pinned lifecycle bodies; integer tensors and opaque state tags only."""
import ast
from array import array
import copy
import dataclasses
import logging
import sys
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch
import test_unified_tree0521 as trees
import test_allocation0521 as allocation
from test_cache0521 import SRT, STOCK, CPU, Ints, method, classes, uq
from test_bf16_integration import execute
from test_cache_lifecycle import Map


class Zeroes:
    def __init__(self, trace, label): self.trace, self.label = trace, label
    def zero_(self): self.trace.append(self.label)


def request_pool(stock, states, trace):
    path = (STOCK if stock else SRT) / "mem_cache/memory_pool.py"
    tree = ast.parse(path.read_bytes())
    ns = {"torch": CPU, "logger": logging.getLogger(__name__)}
    for name, base, selected in (
        ("ReqToTokenPool", "object", {"clear", "reset_aux_cache_allocator", "free", "free_rows"}),
        ("HybridReqToTokenPool", "ReqToTokenPool", {"clear", "free_mamba_cache", "translate_mamba_indices"}),
    ):
        source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        node = ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())], keywords=[],
            body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in selected], decorator_list=[])
        execute([node], ns, path)
    pool = ns["HybridReqToTokenPool"]()
    pool._alloc_size = 4; pool.free_slots = [1, 2, 3]; pool._aux_cache = None
    pool.req_generation = Zeroes(trace, "request-generations-zero")
    pool.mamba_allocator = states; pool.mamba_ckpt_pool = None
    pool.enable_mamba_extra_buffer = False
    pool.short_conv_pool = NS(clear=lambda: trace.append("disabled-short-conv-clear"))
    pool.ngram_pool = NS(clear=lambda: trace.append("disabled-ngram-clear"))
    pool.req_index_to_mamba_index_mapping = Zeroes(trace, "mamba-map-zero")
    return pool


def scheduler(cache, trace, stock=False):
    running = NS(reqs=[], is_empty=lambda: not running.reqs)
    obj = NS(tree_cache=cache, req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        _engine_paused=False, waiting_queue=[], running_batch=running,
        chunked_req=None, dllm_manager=NS(any_staging_reqs=lambda: False),
        last_batch=None, cur_batch_for_debug="old-debug", enable_overlap=False,
        _pp_microbatches_drained=lambda: True, disaggregation_mode="NULL",
        grammar_manager=NS(grammar_queue=[], clear=lambda: trace.append("grammar-clear")),
        metrics_reporter=NS(reset_metrics=lambda: trace.append("metrics-reset"), is_stats_logging_rank=False),
        draft_worker=None, enable_hisparse=False, enable_hierarchical_cache=False,
        enable_lmcache=False)
    ns = {"logger": logging.getLogger(__name__), "logging": logging,
        "DisaggregationMode": NS(PREFILL="PREFILL", DECODE="DECODE"),
        "current_platform": NS(empty_cache=lambda: trace.append("inert-device-empty"))}
    for name in ("is_fully_idle", "flush_cache"):
        setattr(obj, name, MethodType(method("managers/scheduler.py", "Scheduler", name, ns, stock=stock), obj))
    return obj


class Lifecycle0521Tests(unittest.TestCase):
    def make(self, rank=0, stock=False):
        helper = trees.UnifiedTree0521Tests()
        ns, owner, alloc, cache = helper.make(rank, stock, mamba=True)
        trace = []
        allocator_ns = {"torch": CPU}
        classes((STOCK if stock else SRT) / "mem_cache/allocator/mamba.py", {"MambaSlotAllocator"}, allocator_ns)
        states = allocator_ns["MambaSlotAllocator"](7, "CPU")
        pool = request_pool(stock, states, trace)
        cache.req_to_token_pool = pool
        store = {}
        def copy_state(src, dst):
            trace.append(("opaque-state-copy", src.tolist(), dst.tolist()))
            for s, d in zip(src.tolist(), dst.tolist()): store[d] = store[s]
        pool.mamba_pool = NS(replayssm_write_pos=None, copy_from=copy_state)
        original_alloc, original_free = states.alloc, states.free
        def state_alloc(count):
            result = original_alloc(count)
            trace.append(("state-alloc", result.tolist() if result is not None else None))
            return result
        def state_free(values):
            trace.append(("state-free", values.tolist()))
            original_free(values)
        states.alloc, states.free = state_alloc, state_free
        return ns, owner, alloc, cache, trace, store

    def fill(self, rank=0, stock=False):
        ns, owner, alloc, cache, trace, store = self.make(rank, stock)
        slots = alloc.alloc(3); state = cache.req_to_token_pool.mamba_allocator.alloc(1)
        store[state.item()] = ("opaque-checkpoint", rank, 17)
        key = ns["RadixKey"](array("q", [1, 2, 3]), "user", cache_salt="salt")
        cache.insert(ns["InsertParams"](key=key, value=slots, mamba_value=state))
        return ns, owner, alloc, cache, trace, store, key

    def test_actual_scheduler_flush_release_reset_and_reuse_both_ranks(self):
        for rank in (0, 1):
            ns, owner, alloc, cache, trace, store, key = self.fill(rank)
            before = owner.snapshot([1, 2, 3])
            adapter_type = type("ActualInvalidation", (), {"invalidate": method("layers/attention/ultraquant_backend.py", "QwenBF16SourceAdapter", "invalidate")})
            adapter = adapter_type(); adapter._written_layers = {3}; adapter.metadata = "prepared"
            owner.register_adapter(adapter)
            observed = []
            original_free = alloc.free
            def free(values):
                observed.append((values.tolist(), owner.cached_slots()))
                original_free(values)
            alloc.free = free
            controller = scheduler(cache, trace)
            self.assertTrue(controller.flush_cache())
            self.assertIsNone(adapter.metadata)
            self.assertIsNone(controller.cur_batch_for_debug)
            self.assertFalse(owner.cached_slots())
            self.assertEqual(cache.req_to_token_pool.mamba_allocator.available_size(), 7)
            self.assertTrue(observed)
            for values, protected in observed: self.assertFalse(set(values) & protected)
            self.assertEqual(trace[-2:], ["metrics-reset", "inert-device-empty"])
            with self.assertRaisesRegex(ValueError, "stale"): owner.validate_snapshot(before)
            self.assertEqual(alloc.alloc(3).tolist(), [1, 2, 3])
            self.assertNotEqual(owner.snapshot([1, 2, 3]), before)

    def test_busy_paused_pending_live_rows_and_live_slots_refuse(self):
        for reason in ("busy", "paused-waiting", "row", "kv", "state", "group"):
            ns, owner, alloc, cache, trace, store, key = self.fill()
            controller = scheduler(cache, trace)
            if reason == "busy": controller.running_batch.reqs.append(object())
            if reason == "paused-waiting": controller._engine_paused = True; controller.waiting_queue.append(object())
            if reason == "row": cache.req_to_token_pool.free_slots.pop()
            if reason == "kv": alloc.alloc(1)
            if reason == "state": cache.req_to_token_pool.mamba_allocator.alloc(1)
            if reason == "group": alloc.free_group_begin()
            before = owner.snapshot([1, 2, 3]); stored = dict(store)
            self.assertFalse(controller.flush_cache())
            owner.validate_snapshot(before)
            self.assertEqual(owner.cached_slots(), frozenset((1, 2, 3)))
            self.assertEqual(store, stored)
            self.assertNotIn("metrics-reset", trace)
            self.assertEqual(controller.cur_batch_for_debug, "old-debug")

    def test_locks_and_failed_release_never_report_flush_success(self):
        ns, owner, alloc, cache, trace, store, key = self.fill()
        matched = cache.match_prefix(ns["MatchPrefixParams"](key=key))
        lock = cache.inc_lock_ref(matched.best_match_node)
        controller = scheduler(cache, trace)
        self.assertFalse(controller.flush_cache())
        self.assertEqual(owner.cached_slots(), frozenset((1, 2, 3)))
        cache.dec_lock_ref(matched.best_match_node, lock.to_dec_params())
        original = cache.evict
        cache.evict = lambda params: ns["EvictResult"]()  # Declared fault injection, not a release implementation.
        self.assertFalse(controller.flush_cache())
        self.assertNotIn("metrics-reset", trace)
        self.assertEqual(owner.cached_slots(), frozenset((1, 2, 3)))
        cache.evict = original
        self.assertTrue(controller.flush_cache(empty_cache=False))
        self.assertNotIn("inert-device-empty", trace)

    def test_stock_and_candidate_flush_pool_reset_trace(self):
        traces = []
        for stock in (True, False):
            ns, owner, alloc, cache, trace, store, key = self.fill(stock=stock)
            # Baseline must explicitly release owned states before comparing
            # reset effects; its destructive reset itself does not emit frees.
            if stock: cache.evict(ns["EvictParams"](num_tokens=3, mamba_num=1))
            self.assertTrue(scheduler(cache, trace, stock).flush_cache())
            traces.append((trace, alloc.available_size(), cache.req_to_token_pool.mamba_allocator.available_size()))
        self.assertEqual(*traces)

    def test_actual_snapshot_cow_cancellation_retraction_stored_state_trace(self):
        traces = []
        for stock in (True, False):
            ns, owner, alloc, cache, trace, store = self.make(stock=stock)
            pool = cache.req_to_token_pool; states = pool.mamba_allocator
            active = states.alloc(1); store[active.item()] = ("opaque", 91, 23)
            req = NS(kv=NS(mamba_pool_idx=active, mamba_last_track_seqlen=None), _uq_owner=owner)
            component = cache.components[ns["ComponentType"].MAMBA]
            params = ns["InsertParams"]()
            size = component.prepare_for_caching_req(req, params, 3, is_finished=False)
            self.assertEqual(size, 3)
            self.assertEqual(store[params.mamba_value.item()], store[active.item()])
            slots = alloc.alloc(3)
            key = ns["RadixKey"](array("q", [4, 5, 6]), ("user", "lora"), cache_salt="separate")
            params.key, params.value = key, slots
            inserted = cache.insert(params)
            component.cleanup_after_caching_req(req, False, inserted, params)
            pool.free_mamba_cache(req)
            matched_req = NS(kv=NS(holds_mamba=False), _uq_owner=owner)
            cache.match_prefix(ns["MatchPrefixParams"](key=key, cow_mamba=True, req=matched_req))
            # Convert the integer scalar returned by the inert tensor fixture
            # into its tensor wrapper; no routing or allocation decision changes.
            matched_req.kv.mamba_pool_idx = Ints([matched_req.kv.mamba_pool_idx])
            batch = NS()
            collect = method("managers/schedule_batch.py", "ScheduleBatch", "_collect_deferred_mamba_cow_and_clear", stock=stock)
            collect(batch, [matched_req])
            pool.mamba_pool.copy_from(batch.mamba_cow_src_indices, batch.mamba_cow_dst_indices)
            self.assertEqual(store[batch.mamba_cow_dst_indices.item()], ("opaque", 91, 23))
            matched_req.kv.holds_kv = False; matched_req.kv.is_kv_released = True
            matched_req.kv.holds_mamba = True
            release = method("mem_cache/common.py", None, "release_kv_cache", stock=stock)
            release(matched_req, cache, is_insert=False)
            self.assertEqual(owner.cached_slots() if owner else frozenset((1, 2, 3)), frozenset((1, 2, 3)))
            matched_req.retraction_count = 0; matched_req.input_embeds = None
            retract = method("managers/schedule_batch.py", "Req", "reset_for_retract", {"DecLockRefParams": ns["DecLockRefParams"]}, stock=stock)
            retract(matched_req)
            self.assertIsNone(matched_req.kv.mamba_cow_src_index)
            self.assertIsNone(matched_req.kv.mamba_last_track_seqlen)
            if stock: cache.evict(ns["EvictParams"](num_tokens=3, mamba_num=1))
            self.assertTrue(scheduler(cache, trace, stock).flush_cache())
            traces.append((trace, store, states.available_size(), matched_req.retraction_count))
        self.assertEqual(*traces)

    def test_unsupported_speculative_rollback_refuses_and_invalidates(self):
        invalidated = []
        adapter = NS(invalidate=lambda: invalidated.append(True))
        adapter._eager_mode = MethodType(method("layers/attention/ultraquant_backend.py", "QwenBF16SourceAdapter", "_eager_mode"), adapter)
        prepare = method("layers/attention/ultraquant_backend.py", "QwenBF16SourceAdapter", "init_forward_metadata_out_graph")
        for mode, info in (("TARGET_VERIFY", None), ("EXTEND", object())):
            with self.assertRaisesRegex(ValueError, "ordinary eager"):
                prepare(adapter, NS(forward_mode=NS(name=mode), spec_info=info))
        self.assertEqual(invalidated, [True, True])

    def test_actual_result_abort_releases_only_request_tail_before_flush(self):
        traces = []
        for stock in (True, False):
            ns, owner, alloc, cache, trace, store, key = self.fill(stock=stock)
            pool = cache.req_to_token_pool
            matched = cache.match_prefix(ns["MatchPrefixParams"](key=key))
            lock = cache.inc_lock_ref(matched.best_match_node)
            tail = alloc.alloc(2)
            pool.free_slots.remove(1)
            pool.req_to_token = Map({1: Ints([1, 2, 3] + tail.tolist())})
            kv_ns = {"dataclasses": dataclasses}
            classes((STOCK if stock else SRT) / "managers/schedule_batch.py", {"ReqKvInfo"}, kv_ns)
            kv = kv_ns["ReqKvInfo"](req_pool_idx=1, cache_protected_len=3, kv_allocated_len=5,
                kv_committed_len=5, mamba_pool_idx=pool.mamba_allocator.alloc(1))
            req = NS(kv=kv, _uq_owner=owner, owned_kv_len=lambda: 5, skip_radix_cache_insert=False,
                last_node=matched.best_match_node, lock_receipt=lock.to_dec_params(), swa_prefix_lock_released=False,
                multimodal_inputs=None, session=None,
                time_stats=NS(set_completion_time=lambda: trace.append("completion-time")))
            common_ns = {"get_spec": lambda: NS(speculative_algorithm=None),
                "get_serving": lambda: NS(strip_thinking_cache=False), "HybridReqToTokenPool": type(pool)}
            over = method("mem_cache/common.py", None, "_release_overallocated_kv_indices", common_ns, stock=stock)
            release = method("mem_cache/common.py", None, "release_kv_cache", dict(common_ns, _release_overallocated_kv_indices=over), stock=stock)
            transport = NS(coalesce_ranges=method("mem_cache/common.py", None, "coalesce_ranges", stock=stock),
                free_kv_row_segments=method("mem_cache/common.py", None, "free_kv_row_segments", stock=stock))
            abort = method("managers/scheduler_components/batch_result_processor.py", "SchedulerBatchResultProcessor", "_handle_sampling_mask_abort",
                {"release_kv_cache": release, "get_disagg": lambda: NS(disaggregation_decode_enable_offload_kvcache=False),
                 "get_memory": lambda: NS(enable_hisparse=False)}, stock=stock)
            processor = NS(tree_cache=cache, model_worker=NS())
            old = owner.snapshot([1, 2, 3, 4, 5]) if owner else None
            with patch.dict(sys.modules, {"sglang.srt.mem_cache.common": transport}):
                abort(processor, req)
            self.assertFalse(kv.holds_kv); self.assertFalse(kv.holds_mamba)
            self.assertTrue(kv.is_kv_released)
            self.assertEqual(cache.all_values_flatten().tolist(), [1, 2, 3])
            if owner:
                owner.validate_snapshot(old[:3])
                with self.assertRaisesRegex(ValueError, "stale"): owner.validate_snapshot(old)
            self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
            if stock: cache.evict(ns["EvictParams"](num_tokens=3, mamba_num=1))
            self.assertTrue(scheduler(cache, trace, stock).flush_cache())
            traces.append((trace, alloc.available_size(), pool.mamba_allocator.available_size()))
        self.assertEqual(*traces)


class AllocationBoundary0521Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = allocation.Allocation0521Tests("test_old_mode_gate_is_noop_without_runtime_imports")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def test_actual_private_dataclass_handoff_and_selector(self):
        f = self.fixture
        for rank in (0, 1):
            template = f.kvc(rank)
            ns = dict(f.ns, dataclass=dataclasses.dataclass, field=dataclasses.field,
                mambaish_config=lambda model: template.mambaish_config,
                hybrid_gdn_config=lambda model: None, hybrid_kda_config=lambda model: None)
            classes(SRT / "mem_cache/kv_cache_configurator.py", {"KVCacheConfigurator"}, ns)
            cls = ns["KVCacheConfigurator"]
            values = {field.name: getattr(template, field.name, None) for field in dataclasses.fields(cls)
                if field.init and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING}
            template.layer_info.is_hybrid_swa_mtp_draft = False
            obj = cls(**values, _uq_bf16_request=template._uq_bf16_request)
            self.assertIs(obj._uq_bf16_request, template._uq_bf16_request)
            obj._uq_bf16_admission = obj._validate_bf16_allocation()
            select_ns = dict(vars(f.planner), is_deepseek_v4=lambda c: False,
                DefaultPoolConfigurator=lambda kvc: ("tests/fixtures/baseline", kvc))
            selector = method("model_executor/pool_configurator.py", None, "create_memory_pool_configurator", select_ns)
            selected = selector(obj)
            self.assertIsInstance(selected, f.planner.QwenBF16FixedPoolConfigurator)
            self.assertEqual(selected.calculate_pool_sizes(selected.required_bytes, 1).max_total_num_tokens, 31)
            obj._uq_bf16_request = None
            self.assertEqual(selector(obj)[0], "tests/fixtures/baseline")

    def test_changed_proof_config_and_request_clear_budget_before_init(self):
        f = self.fixture
        for rank in (0, 1):
            for change in ("binary", "config", "removed"):
                kvc = f.kvc(rank); f.admit(kvc)
                calls = []
                kvc._build_req_to_token_pool = lambda **kw: calls.append("allocation")
                original_binary = f.fixture.binary.read_bytes()
                original_dim = f.model.hf_text_config.head_dim
                if change == "binary": f.fixture.binary.write_bytes(b"INERT changed")
                if change == "config": f.model.hf_text_config.head_dim = 128
                if change == "removed": kvc._uq_bf16_request = None
                try:
                    with self.assertRaises(ValueError):
                        kvc._init_pools(sizes=NS(max_total_num_tokens=31), req_to_token_pool=None, token_to_kv_pool_allocator=None)
                    self.assertEqual(calls, [])
                    self.assertIsNone(kvc._uq_bf16_budget)
                finally:
                    f.fixture.binary.write_bytes(original_binary)
                    f.model.hf_text_config.head_dim = original_dim
