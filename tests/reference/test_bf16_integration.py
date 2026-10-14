"""Source-bound actual callers with declared CPU maps, allocation and native spies."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
from test_runtime_contract import uq, UPSTREAM, ROOT
from test_runtime_evidence import evidence
from test_bf16_closure import ClosureTests, Tensor

STOCK = ROOT.parent / "stock/srt"

def execute(nodes, namespace, path):
    namespace.setdefault("__name__", __name__)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace

def methods(path, cls_name, names, namespace):
    cls = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return execute([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], namespace, path)

def actual_parent(pool, standins):
    path = STOCK / "mem_cache/memory_pool.py"
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HybridLinearKVPool")
    # Preserve exact constructor/writer/mapping/byte-accounting bodies. Base
    # inheritance is a CPU object; state pool is an opaque sentinel.
    selected = {"__init__", "get_kv_size_bytes", "_transfer_full_attention_id", "set_kv_buffer"}
    node = ast.ClassDef(name=cls.name, bases=[ast.Name(id="object", ctx=ast.Load())], keywords=[],
        body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected], decorator_list=[])
    ns = {"MHATokenToKVPool": object, "MLATokenToKVPool": object,
          "current_platform": SimpleNamespace(is_out_of_tree=lambda: False), "_is_npu": False,
          "GB": 1024**3, "unwrap_write_loc": lambda loc: (loc, None, None)}
    execute([node], ns, path)
    parent = ns[cls.name](size=pool.size, dtype="bfloat16", page_size=1, head_num=2, head_dim=256,
        full_attention_layer_ids=list(uq.QWEN_FULL_LAYERS), device="CPU", mamba_pool=object(), full_kv_pool=pool)
    return parent, ns[cls.name]

class Mode:
    def __init__(self, name):
        self.name = name
    def is_decode(self):
        return self.name == "DECODE"
    def is_extend(self):
        return self.name == "EXTEND"
    def is_idle(self):
        return self.name == "IDLE"
    def is_mixed(self):
        return self.name == "MIXED"
    def is_draft_extend_v2(self):
        return self.name == "DRAFT_EXTEND_V2"

class RequestMap:
    shape = (4, 32)
    def __init__(self, rows, device):
        self.rows, self.device = rows, device
    def __getitem__(self, key):
        row, span = key
        values = self.rows[row][span]
        return Tensor((len(values),), "int64", self.device, values)

class LinearSentinel:
    needs_cpu_seq_lens = True
    def init_forward_metadata_out_graph(self, *args, **kwargs):
        pass
    def init_forward_metadata_in_graph(self, *args, **kwargs):
        pass
    def init_forward_metadata(self, *args, **kwargs):
        pass
    def forward_extend(self, *args, **kwargs):
        raise AssertionError("unexpected GDN forward")
    def forward_decode(self, *args, **kwargs):
        raise AssertionError("unexpected GDN forward")

def hybrid(adapter):
    path = STOCK / "layers/attention/hybrid_linear_attn_backend.py"
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HybridLinearAttnBackend")
    selected = {"__init__", "_is_full_attn", "init_forward_metadata", "init_forward_metadata_out_graph",
                "init_forward_metadata_in_graph", "forward", "forward_decode", "forward_extend"}
    node = ast.ClassDef(name=cls.name, bases=[ast.Name(id="AttentionBackend", ctx=ast.Load())], keywords=[],
        body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected], decorator_list=[])
    ns = {"AttentionBackend": adapter.__class__.__bases__[0]}
    execute([node], ns, path)
    return ns[cls.name](adapter, LinearSentinel(), list(uq.QWEN_FULL_LAYERS))

class IntegrationTests(ClosureTests):
    # Inherit fixture setup, not the old test methods (removed below after class
    # definition); the original closure module still runs all its assertions.
    def batch_adapter(self, rank=0, mode="EXTEND", prefix=0, sequence=(9, 4, 12), locations=None):
        pool = self.construct(rank=rank)
        parent, _ = actual_parent(pool, self.substitutes)
        table = RequestMap({0: list(sequence)}, self.substitutes.device)
        req_pool = SimpleNamespace(req_to_token=table)
        adapter = self.classes.QwenBF16SourceAdapter(pool, req_pool, parent, 32)
        extend = 1 if mode == "DECODE" else len(sequence)-prefix
        locations = list(sequence[-extend:]) if locations is None else locations
        b = SimpleNamespace(forward_mode=Mode(mode), spec_info=None,
            req_pool_indices=Tensor((1,), "int64", self.substitutes.device, [0]),
            seq_lens=Tensor((1,), "int32", self.substitutes.device, [len(sequence)]),
            out_cache_loc=Tensor((extend,), "int64", self.substitutes.device, locations),
            extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[extend])
        return pool, parent, adapter, b

    def layer(self, physical=11):
        return SimpleNamespace(layer_id=physical, tp_q_head_num=12, tp_k_head_num=2, tp_v_head_num=2,
                               qk_head_dim=256, v_head_dim=256, scaling=0.0625, k_scale=None, v_scale=None)

    def inputs(self, batch):
        n = batch.out_cache_loc.numel()
        return (Tensor((n, 3072), "bfloat16", self.substitutes.device),
                Tensor((n, 2, 256), "bfloat16", self.substitutes.device),
                Tensor((n, 2, 256), "bfloat16", self.substitutes.device))

    def native_count(self, name):
        return sum(call[0] == name for call in self.substitutes.calls)

    def test_eager_actual_base_and_hybrid_initial_continuation_decode_both_ranks(self):
        for rank in (0, 1):
            for mode, prefix in (("EXTEND", 0), ("EXTEND", 2), ("DECODE", 0)):
                for physical in (3, 11, 59, 63):
                    pool, parent, adapter, batch = self.batch_adapter(rank, mode, prefix)
                    dispatch = hybrid(adapter)
                    dispatch.init_forward_metadata_out_graph(batch)
                    dispatch.init_forward_metadata_in_graph(batch)
                    q, k, v = self.inputs(batch)
                    before = self.native_count("uq_bf16_append_v2")
                    output = dispatch.forward(q, k, v, self.layer(physical), batch)
                    self.assertEqual(output.shape, (batch.out_cache_loc.numel(), 3072))
                    self.assertEqual(self.native_count("uq_bf16_append_v2"), before+1)
                    self.assertIs(dispatch.token_to_kv_pool, parent)
                    self.assertIs(dispatch.req_to_token_pool, adapter.req_to_token_pool)
                    self.assertEqual(pool.immutable, frozenset((9, 4)) if mode=="DECODE" or prefix==2 else frozenset())

    def test_read_only_does_not_append_and_repeated_append_refuses(self):
        pool, parent, adapter, b = self.batch_adapter()
        h = hybrid(adapter); h.init_forward_metadata(b)
        q, k, v = self.inputs(b)
        before = self.native_count("uq_bf16_append_v2")
        h.forward(q, None, None, self.layer(), b, save_kv_cache=False)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), before)
        h.forward(q, k, v, self.layer(), b)
        with self.assertRaisesRegex(ValueError, "already appended"):
            h.forward(q, k, v, self.layer(), b)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), before+1)

    def test_invalidation_and_replaced_batch(self):
        _, _, adapter, b = self.batch_adapter()
        adapter.init_forward_metadata(b)
        q, k, v = self.inputs(b)
        adapter.invalidate()
        with self.assertRaisesRegex(ValueError, "missing or stale"):
            adapter.forward(q, k, v, self.layer(), b)
        adapter.init_forward_metadata(b)
        other = SimpleNamespace(**vars(b))
        with self.assertRaisesRegex(ValueError, "missing or stale"):
            adapter.forward(q, k, v, self.layer(), other)

    def test_unsupported_modes_capture_and_map_bounds_before_calls(self):
        for mode in ("MIXED", "TARGET_VERIFY", "DRAFT_EXTEND_V2", "IDLE"):
            _, _, adapter, b = self.batch_adapter()
            b.forward_mode = Mode(mode)
            with self.assertRaises(ValueError):
                adapter.init_forward_metadata(b)
            self.assertIsNone(adapter.metadata)
        _, _, adapter, b = self.batch_adapter()
        with self.assertRaisesRegex(ValueError, "capture/replay"):
            adapter.init_forward_metadata_out_graph(b, in_capture=True)
        for req_id, total in ((4, 3), (-1, 3), (0, 33)):
            b.req_pool_indices.values, b.seq_lens.values = [req_id], [total]
            with self.assertRaisesRegex(ValueError, "bounds"):
                adapter.init_forward_metadata(b)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)
        self.assertEqual(self.native_count("uq_bf16_attention_v2"), 0)

    def test_bad_forward_arguments_do_not_append(self):
        _, _, adapter, b = self.batch_adapter()
        adapter.init_forward_metadata(b)
        q, k, v = self.inputs(b)
        for kwargs in ({"unknown": None}, {"k_scale": 2}, {"dcp_kv_mask": []}):
            with self.assertRaises(ValueError):
                adapter.forward(q, k, v, self.layer(), b, **kwargs)
        for physical in (0, 64):
            with self.assertRaises(ValueError):
                adapter.forward(q, k, v, self.layer(physical), b)
        bad = self.layer(); bad.scaling = float("nan")
        with self.assertRaises(ValueError):
            adapter.forward(q, k, v, bad, b)
        bad = self.layer(); bad.tp_q_head_num = 24
        with self.assertRaises(ValueError):
            adapter.forward(q, k, v, bad, b)
        q.dtype = "float16"
        with self.assertRaises(ValueError):
            adapter.forward(q, k, v, self.layer(), b)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def planner_namespace(self):
        path = UPSTREAM / "python/sglang/srt/model_executor/pool_configurator.py"
        tree = ast.parse(path.read_bytes())
        names = {"MemoryPoolConfig", "MemoryPoolConfigurator", "DefaultPoolConfigurator", "qwen_bf16_fixed_payload"}
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        return execute(nodes, {"dataclass": dataclass, "torch": self.substitutes.torch}, path)

    def runner_methods(self):
        folder = UPSTREAM / "python/sglang/srt/model_executor"
        ns = {"torch": self.substitutes.torch, "__file__": str(self.fixture.package / "srt/model_executor/model_runner_kv_cache_mixin.py")}
        methods(folder / "model_runner.py", "ModelRunner", {"prepare_qwen_bf16_source", "_get_attention_backend_from_str"}, ns)
        mixin = next(n for n in ast.parse((folder / "model_runner_kv_cache_mixin.py").read_bytes()).body if isinstance(n, ast.ClassDef) and any(isinstance(m, ast.FunctionDef) and m.name=="_resolve_memory_pool_config" for m in n.body))
        names = {"_resolve_memory_pool_config", "_apply_memory_pool_config", "_validate_qwen_bf16_source", "_qwen_bf16_pool_class", "_profile_available_bytes", "handle_max_mamba_cache"}
        execute([n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name in names], ns, folder / "model_runner_kv_cache_mixin.py")
        function = next(n for n in ast.parse((folder / "model_runner_kv_cache_mixin.py").read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name=="qwen_bf16_full_pool_factory")
        execute([function], ns, folder / "model_runner_kv_cache_mixin.py")
        return ns

    def source_runner(self, rank=0):
        args = SimpleNamespace(attention_backend="triton", tp_size=2, pp_size=1, dp_size=1, nnodes=1,
            dcp_size=1, attn_cp_size=1, page_size=1, dtype="bfloat16", kv_cache_dtype="auto",
            disable_overlap_schedule=True, cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
            cuda_graph_config=None, disaggregation_mode="null", cpu_offload_gb=0, mem_fraction_static=0.8)
        model = SimpleNamespace(model_path=uq.QWEN_MODEL, revision=uq.QWEN_REVISION, num_attention_heads=24,
            get_num_kv_heads=lambda tp: 2, head_dim=256, v_head_dim=256, context_len=32)
        runner = SimpleNamespace(server_args=args, model_config=model, dtype="bfloat16", kv_cache_dtype="bfloat16",
            tp_size=2, tp_rank=rank, is_draft_worker=False, use_mla_backend=False, is_hybrid_swa=False,
            start_layer=0, end_layer=64, token_to_kv_pool=None,
            mambaish_config=SimpleNamespace(full_attention_layer_ids=list(uq.QWEN_FULL_LAYERS)))
        ns = self.runner_methods()
        for name in ("prepare_qwen_bf16_source", "_resolve_memory_pool_config", "_apply_memory_pool_config", "_validate_qwen_bf16_source", "_qwen_bf16_pool_class"):
            setattr(runner, name, MethodType(ns[name], runner))
        runner.prepare_qwen_bf16_source((ROOT / "tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes(), rank=rank, physical_slots=31)
        return runner, ns

    def test_actual_fixed_resolution_and_application_preserve_constraints(self):
        for rank in (0, 1):
            runner, ns = self.source_runner(rank)
            plan = self.planner_namespace()
            configurator = object.__new__(plan["DefaultPoolConfigurator"])
            configurator._uq_bf16_policy = runner._uq_bf16_request["policy"]
            configurator._ultraquant = False
            budget = plan["qwen_bf16_fixed_payload"](configurator._uq_bf16_policy)
            self.assertEqual(budget["native_status_bytes"], 4)
            available = budget["kv_payload_with_slot_zero"]+budget["native_status_bytes"]+budget["workspace_reserve"]
            runner._profile_available_bytes = lambda before: available
            runner._apply_token_constraints = lambda tokens: 17
            runner._resolve_max_num_reqs = lambda tokens: 2
            ns["is_deepseek_v4"] = lambda config: False
            runner.model_config.hf_config = object()
            runner.is_draft_worker = False
            events = []
            runner._init_pools = lambda: events.append((runner.max_total_num_tokens, runner.max_running_requests))
            with patch.dict("sys.modules", {"sglang.srt.model_executor.pool_configurator": SimpleNamespace(create_memory_pool_configurator=lambda mr: configurator)}):
                config = runner._resolve_memory_pool_config(0)
                self.assertEqual(config.max_total_num_tokens, 17)
                self.assertEqual(runner._uq_bf16_policy["slots"], 18)
                runner._apply_memory_pool_config(config)
            self.assertEqual(events, [(17, 2)])
            with self.assertRaisesRegex(ValueError, "insufficient"):
                configurator.calculate_pool_sizes(available-1, 1)

    def test_missing_or_stale_proof_precedes_runner_profile_and_allocation(self):
        runner, _ = self.source_runner()
        runner._profile_available_bytes = lambda _: self.fail("profile before proof refusal")
        self.fixture.value["image_config"] = "0"*64
        self.fixture.write()
        with patch.dict("sys.modules", {"sglang.srt.model_executor.pool_configurator": SimpleNamespace(create_memory_pool_configurator=lambda mr: self.fail("planner before proof"))}):
            with self.assertRaisesRegex(ValueError, "stale image"):
                runner._resolve_memory_pool_config(0)
        self.fixture.proof.unlink()
        with patch.dict("sys.modules", {"sglang.srt.model_executor.pool_configurator": SimpleNamespace(create_memory_pool_configurator=lambda mr: self.fail("planner before proof"))}):
            with self.assertRaisesRegex(ValueError, "missing reviewed"):
                runner._resolve_memory_pool_config(0)
        self.assertFalse(self.substitutes.library_loads)
        self.assertFalse(self.substitutes.allocations)

    def test_factory_reaches_actual_hybrid_constructor(self):
        runner, _ = self.source_runner()
        runner._uq_bf16_policy = runner._uq_bf16_request["policy"]
        factory = runner._qwen_bf16_pool_class(object)
        pool, parent_cls = actual_parent(self.construct(), self.substitutes)
        gdn = object()
        parent = parent_cls(size=31, dtype="bfloat16", page_size=1, head_num=2, head_dim=256,
            full_attention_layer_ids=list(uq.QWEN_FULL_LAYERS), device="CPU", mamba_pool=gdn, full_kv_pool_class=factory)
        self.assertIs(parent.mamba_pool, gdn)
        self.assertEqual(sum(parent.get_kv_size_bytes()), 32*14720)

    def test_baseline_factory_and_public_refusal(self):
        runner, ns = self.source_runner()
        del runner._uq_bf16_request
        fallback = object()
        self.assertIs(runner._qwen_bf16_pool_class(fallback), fallback)
        with self.assertRaisesRegex(ValueError, "not an admitted"):
            uq.validate_args(SimpleNamespace(attention_backend=uq.BF16_SOURCE_MODE))

    def test_actual_runner_hybrid_allocation_branch_uses_injected_factory(self):
        runner, ns = self.source_runner()
        runner._uq_bf16_policy = runner._uq_bf16_request["policy"]
        initial, parent_cls = actual_parent(self.construct(), self.substitutes)
        gdn = object()
        runner.max_total_num_tokens, runner.page_size = 31, 1
        runner.device = "CPU"
        runner.req_to_token_pool = SimpleNamespace(mamba_pool=gdn)
        runner.server_args.enable_memory_saver = False
        runner.server_args.speculative_algorithm = None
        path = UPSTREAM / "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
        tree = ast.parse(path.read_bytes())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=="_init_pools")
        branch = next(n for n in ast.walk(function) if isinstance(n, ast.If) and isinstance(n.test, ast.NamedExpr)
            and isinstance(n.test.value, ast.Attribute) and n.test.value.attr=="mambaish_config"
            and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id=="HybridLinearKVPool"
                    for statement in n.body for call in ast.walk(statement)))
        # Select the exact existing branch, retaining its condition and body;
        # do not emulate the constructor arguments in a second implementation.
        harness = ast.FunctionDef(name="actual_runner_branch", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
                                  body=[branch], decorator_list=[])
        ns.update(HybridLinearKVPool=parent_cls, mha_pool_class=object, get_attention_tp_size=lambda: 2)
        execute([harness], ns, path)
        ns["actual_runner_branch"](runner)
        self.assertIs(runner.token_to_kv_pool.mamba_pool, gdn)
        self.assertEqual(sum(runner.token_to_kv_pool.get_kv_size_bytes()), 32*14720)

    def test_actual_gdn_budget_subtraction_is_retained(self):
        runner, ns = self.source_runner()
        ns.update(get_available_gpu_memory=lambda *args, **kwargs: 8.0,
                  get_world_group=lambda: SimpleNamespace(world_size=1, cpu_group=None))
        runner.handle_max_mamba_cache = MethodType(ns["handle_max_mamba_cache"], runner)
        runner._profile_available_bytes = MethodType(ns["_profile_available_bytes"], runner)
        runner.device, runner.gpu_id, runner.mem_fraction_static = "CPU", 0, 0.8
        runner.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        runner.server_args.max_mamba_cache_size = 2
        runner.server_args.enable_dp_attention = False
        runner.mambaish_config.mamba2_cache_params = SimpleNamespace(mamba_cache_per_req=76972032)
        available = runner._profile_available_bytes(0)
        self.assertEqual(available, 8*1024**3 - 2*76972032)
        self.assertEqual(runner.server_args.max_mamba_cache_size, 2)

    def test_actual_runner_backend_hook_and_baseline(self):
        runner, ns = self.source_runner()
        pool, parent, adapter, batch = self.batch_adapter()
        runner.token_to_kv_pool, runner.req_to_token_pool = parent, adapter.req_to_token_pool
        wrapped = []
        ns["attn_backend_wrapper"] = lambda mr, full: wrapped.append(full) or hybrid(full)
        method = MethodType(ns["_get_attention_backend_from_str"], runner)
        component = method("triton")
        self.assertIs(wrapped[-1].pool, pool)
        self.assertEqual(runner.decode_attention_backend_str, uq.BF16_SOURCE_MODE)
        component.init_forward_metadata(batch)
        component.forward(*self.inputs(batch), self.layer(), batch)
        del runner._uq_bf16_request
        marker = object()
        ns["ATTENTION_BACKENDS"] = {"triton": lambda mr: marker}
        ns["attn_backend_wrapper"] = lambda mr, full: full
        self.assertIs(method("triton"), marker)

    def test_graph_replay_invalidates_before_native_calls(self):
        _, _, adapter, batch = self.batch_adapter()
        adapter.init_forward_metadata(batch)
        for name in ("init_cuda_graph_state", "init_forward_metadata_replay_cuda_graph",
                     "init_forward_metadata_for_breakable_cuda_graph_capture",
                     "prepare_forward_metadata_for_breakable_cuda_graph_replay"):
            with self.assertRaisesRegex(ValueError, "capture/replay"):
                getattr(adapter, name)()
            self.assertIsNone(adapter.metadata)
        self.assertEqual(self.native_count("uq_bf16_append_v2"), 0)

    def test_explicit_source_settings_refuse_unsupported_paths(self):
        runner, _ = self.source_runner()
        values = vars(runner.server_args).copy()
        for name, value in (("enable_unified_memory", True), ("enable_two_batch_overlap", True),
                            ("speculative_algorithm", "NEXTN"), ("page_size", 16),
                            ("disable_overlap_schedule", False), ("dcp_size", 2)):
            args = SimpleNamespace(**values)
            setattr(args, name, value)
            with self.assertRaises(ValueError):
                uq.validate_bf16_source_args(args)
        with self.assertRaisesRegex(ValueError, "fixed official FP8"):
            uq.bf16_source_request(b"wrong", rank=0, physical_slots=31, context_length=32)
        with self.assertRaisesRegex(ValueError, "context bound"):
            uq.bf16_source_request((ROOT / "tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes(),
                                   rank=0, physical_slots=31, context_length=131073)


# Avoid repeating inherited tests while retaining the original 50-test module.
for _name in list(vars(ClosureTests)):
    if _name.startswith("test_") and _name not in vars(IntegrationTests):
        setattr(IntegrationTests, _name, None)
del ClosureTests
