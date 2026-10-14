"""Actual allocation/planner bodies; installed proof fixtures are inert CPU data."""
import ast
import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch
from test_cache0521 import SRT, OWNED, method, classes, evidence, uq
import test_modern_identity as modern_fixtures
from test_bf16_closure import SyntheticBuild, CPUStandins, actual_classes
from test_bf16_integration import execute


class Allocation0521Tests(unittest.TestCase):
    def setUp(self):
        self.host = modern_fixtures.ModernIdentityTests("test_positive_exact_modern_inventory_and_patch")
        self.host.setUp()
        self.addCleanup(self.host.doCleanups)
        self.fixture = SyntheticBuild(self.host.folder / "allocation-INERT")
        self.fixture.value.update(image_manifest="a"*64, image_config="b"*64, repair_files={},
            stack_receipt_sha256=self.host.receipt_hash,
            source_revision=self.host.contract.SOURCE_REVISION, serving_composition_complete=True)
        self.fixture.write()
        self.standins = CPUStandins()
        self.torch = self.standins.torch
        self.torch.float8_e4m3fn = "float8"
        self.parallel = NS(tp_rank=0, attn_tp_size=2, attn_dp_size=1, attn_dcp_size=1)
        self.schedule = NS(page_size=1, disable_overlap_schedule=True)
        self.memory = NS(enable_unified_memory=False, enable_hierarchical_cache=False)
        self.execution = NS(features=NS(enable_memory_saver=False), mamba=NS(enable_mamba_extra_buffer=False),
            graph=NS(cuda_graph_config=NS(decode=NS(backend="disabled"), prefill=NS(backend="disabled"))))
        self.config_bytes = (OWNED / "prototype/tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes()
        data = json.loads(self.config_bytes)
        self.model = NS(model_path=uq.QWEN_MODEL, revision=uq.QWEN_REVISION, context_len=4096,
            hf_text_config=NS(**data["text_config"]), hf_config=NS(quantization_config=data["quantization_config"]),
            head_dim=256, get_num_kv_heads=lambda tp, dcp: 2)
        self.ns = {"__file__": str(self.fixture.package / "srt/mem_cache/kv_cache_configurator.py"),
            "torch": self.torch, "get_parallel": lambda: self.parallel, "get_schedule": lambda: self.schedule,
            "get_memory": lambda: self.memory, "get_exec": lambda: self.execution,
            "get_disagg": lambda: NS(disaggregation_mode="null"), "get_spec": lambda: NS(speculative_algorithm=None),
            "current_platform": NS(is_cpu=lambda: False, is_out_of_tree=lambda: False), "_is_npu": False,
            "is_swa_req_ring": lambda a: False,
            "get_available_gpu_memory": lambda *a: 0, "logger": NS(info=lambda *a: None),
            "KVCacheConfigResult": lambda **kw: NS(**kw)}
        planner_ns = {"dataclass": dataclasses.dataclass, "__name__": __name__}
        classes(SRT / "model_executor/pool_configurator.py",
            {"MemoryPoolConfig", "MemoryPoolConfigurator", "QwenBF16FixedPoolConfigurator"}, planner_ns)
        self.planner = NS(**planner_ns)
        self.planner.create_memory_pool_configurator = lambda kvc: self.planner.QwenBF16FixedPoolConfigurator(kvc)
        self.patches = patch.dict(sys.modules, {
            "sglang.srt.layers.attention.ultraquant_contract": uq,
            "sglang.srt.layers.attention.ultraquant_evidence": evidence,
            "sglang.srt.layers.attention.ultraquant_stack_contract": self.host.contract,
            "sglang.srt.model_executor.pool_configurator": self.planner})
        self.patches.start(); self.addCleanup(self.patches.stop)

    def kvc(self, rank=0):
        self.parallel.tp_rank = rank
        request = dict(uq.bf16_source_request(self.config_bytes, rank=rank, physical_slots=31, context_length=4096))
        request.update(stack_receipt=self.host.path, stack_receipt_sha256=self.host.receipt_hash)
        kvc = NS(_uq_bf16_request=request, _uq_bf16_admission=None, _uq_bf16_budget=None,
            model_config=self.model, pp_size=1, page_size=1, model_dtype="bfloat16", kv_cache_dtype="bfloat16",
            kv_cache_dtype_str="auto", is_draft_worker=False, spec_algorithm=NS(is_none=lambda: True),
            is_hybrid_swa=False, use_mla_backend=False, post_capture_kv_active=False,
            req_to_token_pool=None, token_to_kv_pool_allocator=None,
            mambaish_config=NS(full_attention_layer_ids=uq.QWEN_FULL_LAYERS), layer_info=NS(start_layer=0, end_layer=64),
            pool_page_size=1, device="CPU-INERT", gpu_id=0,
            _apply_token_constraints=lambda n: n, _build_mha_quant_method=lambda **kw: None)
        for name in ("_validate_bf16_allocation", "_check_bf16_allocation", "configure", "config_from_budget", "_init_pools", "_build_hybrid_linear_kv_pool"):
            setattr(kvc, name, MethodType(method("mem_cache/kv_cache_configurator.py", "KVCacheConfigurator", name, self.ns), kvc))
        return kvc

    def admit(self, kvc):
        kvc._uq_bf16_admission = kvc._validate_bf16_allocation()
        planner = self.planner.QwenBF16FixedPoolConfigurator(kvc)
        config = kvc.config_from_budget(planner.required_bytes)
        return config, planner

    def test_actual_configure_orders_proof_budget_then_allocation_both_ranks(self):
        for rank in (0, 1):
            kvc = self.kvc(rank)
            events = []
            def resolve(unused):
                events.append("budget")
                self.assertIsNotNone(kvc._uq_bf16_admission)
                config = kvc.config_from_budget((32*14720)+(2<<30)+4)
                config.max_running_requests = 2
                return config
            kvc._resolve_memory_pool_config = resolve
            kvc._derive_pool_sizes = lambda config: NS(max_total_num_tokens=config.max_total_num_tokens,
                max_running_requests=2, full_max_total_num_tokens=None, swa_max_total_num_tokens=None)
            def allocate(**kw):
                kvc._check_bf16_allocation(kw["sizes"].max_total_num_tokens)
                events.append("allocate-INERT")
                return NS(req_to_token_pool="GDN unchanged", token_to_kv_pool="INERT", token_to_kv_pool_allocator="INERT", unified_memory_pool=None)
            kvc._init_pools = allocate
            result = kvc.configure(pre_model_load_memory=0)
            self.assertEqual(events, ["budget", "allocate-INERT"])
            self.assertEqual(result.max_total_num_tokens, 31)
            self.assertEqual(kvc._uq_bf16_admission["policy"]["rank"], rank)
            self.assertEqual(self.standins.library_loads, [])

    def test_missing_stale_partial_proof_before_profile_or_allocate(self):
        mutations = [("serving_composition_complete", False), ("image_config", "0"*64),
                     ("stack_receipt_sha256", "0"*64), ("source_sha256", "0"*64)]
        original = copy.deepcopy(self.fixture.value)
        for key, value in mutations:
            self.fixture.value = copy.deepcopy(original); self.fixture.value[key] = value; self.fixture.write()
            kvc = self.kvc()
            calls = []
            kvc._resolve_memory_pool_config = lambda *a: calls.append("profile")
            kvc._init_pools = lambda **kw: calls.append("allocation")
            with self.assertRaises(ValueError): kvc.configure(pre_model_load_memory=0)
            self.assertEqual(calls, [])
            self.assertIsNone(kvc._uq_bf16_admission)
        self.fixture.value = original; self.fixture.write()
        self.fixture.proof.unlink()
        with self.assertRaisesRegex(ValueError, "missing reviewed"):
            self.kvc().configure(pre_model_load_memory=0)

    def test_config_rank_and_runtime_configuration_refuse(self):
        kvc = self.kvc()
        for attr, bad in (("model_dtype", "float16"), ("page_size", 2), ("post_capture_kv_active", True)):
            before = getattr(kvc, attr); setattr(kvc, attr, bad)
            with self.assertRaises(ValueError): kvc._validate_bf16_allocation()
            setattr(kvc, attr, before)
        self.parallel.tp_rank = 1
        with self.assertRaisesRegex(ValueError, "foreign"):
            kvc._validate_bf16_allocation()
        self.parallel.tp_rank = 0
        self.model.revision = "stale"
        with self.assertRaisesRegex(ValueError, "unknown Qwen"):
            kvc._validate_bf16_allocation()

    def test_payload_slot_zero_status_workspace_constraints(self):
        kvc = self.kvc()
        config, plan = self.admit(kvc)
        self.assertEqual(plan.required_bytes, 32*14720 + 2*1024**3 + 4)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            kvc.config_from_budget(plan.required_bytes-1)
        self.assertIsNone(kvc._uq_bf16_budget)
        for cap in (30, 0):
            with self.assertRaisesRegex(ValueError, "new policy"):
                kvc.config_from_budget(plan.required_bytes, cap_tokens=cap)
            self.assertIsNone(kvc._uq_bf16_budget)
        kvc._apply_token_constraints = lambda n: n-1
        with self.assertRaisesRegex(ValueError, "new policy"):
            kvc.config_from_budget(plan.required_bytes)

    def test_recheck_before_actual_init_pools_allocation(self):
        kvc = self.kvc(); self.admit(kvc)
        self.fixture.binary.write_bytes(b"INERT stale binary")
        calls = []
        kvc._build_req_to_token_pool = lambda **kw: calls.append("GDN allocation")
        with self.assertRaisesRegex(ValueError, "stale BF16 binary"):
            kvc._init_pools(sizes=NS(max_total_num_tokens=31), req_to_token_pool=None, token_to_kv_pool_allocator=None)
        self.assertEqual(calls, [])

    def test_actual_full_pool_factory_and_parent_constructor(self):
        kvc = self.kvc(); self.admit(kvc)
        runtime = actual_classes(self.fixture, self.standins)
        parent_ns = {"KVCache": object, "current_platform": NS(is_out_of_tree=lambda: False),
                     "_is_npu": False, "MHATokenToKVPool": object, "GB": 1024**3}
        parent_init = method("mem_cache/memory_pool.py", "HybridLinearKVPool", "__init__", parent_ns)
        size_method = method("mem_cache/memory_pool.py", "HybridLinearKVPool", "get_kv_size_bytes")
        Parent = type("ActualHybridConstructor", (), {"__init__": parent_init, "get_kv_size_bytes": size_method})
        namespace = dict(self.ns, HybridLinearKVPool=Parent, MHATokenToKVPoolMXFP8=object)
        build = method("mem_cache/kv_cache_configurator.py", "KVCacheConfigurator", "_build_hybrid_linear_kv_pool", namespace)
        with patch.dict(sys.modules, {
            "sglang.srt.layers.attention.ultraquant_backend": runtime,
            "sglang.srt.layers.attention.qsa.config": NS(parse_qsa_profile=lambda c: None),
            "sglang.srt.mem_cache.qsa_kv_pool": NS(QSATokenToKVPool=object)}), patch("ctypes.CDLL", self.standins.library):
            pool = build(kvc, max_total_num_tokens=31, req_to_token_pool=NS(mamba_pool="INERT-state-owner"), mha_pool_class=object)
        self.assertEqual(pool.full_kv_pool.get_kv_size_bytes(), (32*7360, 32*7360))
        self.assertEqual(pool.mamba_pool, "INERT-state-owner")
        self.assertEqual(pool.full_kv_pool.owner._capacity, None)
        self.assertEqual(len(pool.full_kv_pool.k_buffer), 16)

    def test_old_mode_gate_is_noop_without_runtime_imports(self):
        kvc = self.kvc(); kvc._uq_bf16_request = None
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_evidence": None}):
            self.assertIsNone(kvc._validate_bf16_allocation())
            self.assertIsNone(kvc._check_bf16_allocation(10))
