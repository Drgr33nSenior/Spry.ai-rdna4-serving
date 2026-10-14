"""CPU tests of the actual runtime metadata and admission algorithms."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2] / "tests/fixtures/legacy"
UPSTREAM = ROOT / "image-source"
CONTRACT = UPSTREAM / "python/sglang/srt/layers/attention/ultraquant_contract.py"
spec = importlib.util.spec_from_file_location("uq_contract", CONTRACT)
uq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uq)


def valid_args():
    return SimpleNamespace(
        attention_backend=uq.MODE, prefill_attention_backend=None, decode_attention_backend=None,
        tp_size=1, pp_size=1, dp_size=1, nnodes=1, attn_cp_size=1, dcp_size=1, page_size=1,
        dtype="float16", kv_cache_dtype="auto", disable_overlap_schedule=True,
        cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
        cuda_graph_config=None, disaggregation_mode="null", cpu_offload_gb=0,
        max_total_tokens=10000, context_length=4096, chunked_prefill_size=512,
        max_running_requests=8,
    )


class MetadataTests(unittest.TestCase):
    def test_prefill_decode_continuation_and_shared_prefix(self):
        self.assertEqual(uq.causal_rows([[9, 4, 2]], [0], [3], [9, 4, 2], 32),
                         ([9, 4, 2], [0, 0, 0], [1, 2, 3]))
        self.assertEqual(uq.causal_rows([[9, 4, 2, 7]], [3], [1], [7], 32),
                         ([9, 4, 2, 7], [0], [4]))
        # Two independent requests reuse immutable physical prefix 9,4. Their
        # fresh suffixes are not contiguous; no identity map can pass this case.
        self.assertEqual(uq.causal_rows([[9, 4, 12, 3], [9, 4, 22]], [2, 2], [2, 1],
                                      [12, 3, 22], 32),
                         ([9, 4, 12, 3, 9, 4, 22], [0, 0, 4], [3, 4, 3]))

    def test_append_matches_reused_and_evicted_slot_map(self):
        # Slot 2 is newly assigned after eviction. Current map, not physical
        # adjacency or historical length, determines what the reader sees.
        self.assertEqual(uq.causal_rows([[9, 2]], [1], [1], [2], 32)[2], [2])
        cases = [([[9, 4]], [1], [1], [9]), ([[9, 9]], [1], [1], [9]),
                 ([[0]], [0], [1], [0]), ([[32]], [0], [1], [32]),
                 ([[9, 4]], [0], [1], [4]), ([[9], [4, 9]], [0, 1], [1, 1], [9, 9]),
                 ([[9, 4], [2, 9]], [1, 1], [1, 1], [4, 9])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                uq.causal_rows(*case, 32)

    def test_bounds(self):
        for prefix, extend in ((True, 1), (0, False), (-1, 2), (1, 0)):
            with self.assertRaises(ValueError):
                uq.causal_rows([[1]], [prefix], [extend], [1], 32)
        with self.assertRaises(ValueError):
            uq.causal_rows([list(range(1, 4098))], [0], [4097], list(range(1, 4098)), 8192)


class AdmissionTests(unittest.TestCase):
    def test_gpu_test_refuses_before_import_without_explicit_argument(self):
        result = subprocess.run([sys.executable, str(ROOT / "tests/gpu_runtime.py")],
                                env={"PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU execution NOT RUN", result.stderr)
        self.assertNotIn("No module named", result.stderr)

    def test_baseline_absence_is_noop(self):
        uq.validate_args(SimpleNamespace(attention_backend="triton"))

    def test_supported_request_and_refusals(self):
        uq.validate_args(valid_args())
        for field, value in (("tp_size", 2), ("dtype", "bfloat16"), ("page_size", 16),
                             ("kv_cache_dtype", "fp4_e2m1"), ("disable_overlap_schedule", False),
                             ("cuda_graph_backend_decode", None), ("enable_torch_compile", True),
                             ("max_total_tokens", None), ("context_length", 131073),
                             ("max_running_requests", 129),
                             ("prefill_attention_backend", "triton"), ("quantization", "fp8"),
                             ("torchao_config", "int8wo"), ("modelopt_quant", "fp8"),
                             ("quantize_and_serve", True),
                             ("dcp_size", 2), ("enable_page_major_kv_layout", True),
                             ("enable_unified_memory", True)):
            args = valid_args()
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                uq.validate_args(args)

    def test_no_auto_fallback(self):
        args = valid_args()
        args._ultraquant_requested = True
        args.attention_backend = "triton"
        with self.assertRaisesRegex(ValueError, "changed during"):
            uq.validate_args(args)

    def test_runner_rechecks_after_model_specific_backend_rewrite(self):
        file = UPSTREAM / "python/sglang/srt/model_executor/model_runner.py"
        tree = ast.parse(file.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelRunner")
        fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "configure_kv_cache_dtype")
        import builtins
        real_import = builtins.__import__

        def test_import(name, *args, **kwargs):
            if name.endswith("ultraquant_contract"):
                return uq
            return real_import(name, *args, **kwargs)

        namespace = {"__builtins__": dict(vars(builtins), __import__=test_import)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), "exec"), namespace)
        args = valid_args()
        args._ultraquant_requested = True
        args.attention_backend = "triton"
        with self.assertRaisesRegex(ValueError, "changed during"):
            namespace["configure_kv_cache_dtype"](SimpleNamespace(server_args=args))

    def test_model_architecture_and_gqa(self):
        model = SimpleNamespace(hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                                head_dim=128, v_head_dim=128, num_attention_heads=32,
                                get_num_kv_heads=lambda tp: 8)
        uq.validate_model(model)
        model.hf_config.architectures = ["Qwen3_5ForConditionalGeneration"]
        with self.assertRaisesRegex(ValueError, "Llama"):
            uq.validate_model(model)
        model.hf_config.architectures = ["LlamaForCausalLM"]
        model.num_attention_heads = 31
        with self.assertRaises(ValueError):
            uq.validate_model(model)

    def test_exact_cache_allocation_and_workspace(self):
        # Two layers, two KV heads, D64: 2*2*2*2*17 = 272 B/token.
        self.assertEqual(uq.cell_bytes(2, 2, 64), 272)
        self.assertEqual(uq.capacity(uq.WORKSPACE_RESERVE + 272 * 101, 272), 100)
        self.assertEqual(uq.capacity(uq.WORKSPACE_RESERVE - 1, 272), 0)

    def test_actual_pool_configurator_methods(self):
        # Execute the changed production methods, not a copy of their algorithm;
        # unrelated Torch imports are excluded because this is a CPU source test.
        file = UPSTREAM / "python/sglang/srt/model_executor/pool_configurator.py"
        tree = ast.parse(file.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DefaultPoolConfigurator")
        fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_pool_sizes")
        for arg in fn.args.args:
            arg.annotation = None
        fn.returns = None
        namespace = {"MemoryPoolConfig": lambda **kwargs: kwargs}
        import builtins
        real_import = builtins.__import__

        def test_import(name, *args, **kwargs):
            if name.endswith("ultraquant_contract"):
                return uq
            return real_import(name, *args, **kwargs)

        namespace["__builtins__"] = dict(vars(builtins), __import__=test_import)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), "exec"), namespace)
        calculate = namespace["calculate_pool_sizes"]
        self.assertEqual(calculate(SimpleNamespace(_ultraquant=True, _cell_size=272),
                                   uq.WORKSPACE_RESERVE + 272 * 101, 1), {"max_total_num_tokens": 100})
        self.assertEqual(calculate(SimpleNamespace(_ultraquant=False, _cell_size=272),
                                   272 * 101, 1), {"max_total_num_tokens": 101})


class QwenPlanningTests(unittest.TestCase):
    """Actual pure contract functions; no Torch, GPU pool or GDN execution."""

    def plan(self, rank=0, **changes):
        config = (ROOT / "tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes()
        identity = json.loads((ROOT / "patches/source.json").read_bytes())
        options = dict(model=uq.QWEN_MODEL, revision=uq.QWEN_REVISION, rank=rank,
                       source_tree_sha256=identity["source_tree_sha256"])
        options.update(changes)
        return uq.qwen_planning_metadata(config, **options)

    def test_future_mode_refuses_before_runtime_import_or_fallback(self):
        import builtins
        real_import = builtins.__import__

        def cpu_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "triton", "sglang"):
                self.fail(f"unexpected runtime import: {name}")
            return real_import(name, *args, **kwargs)

        # Load the actual source under the import guard, not a copied validator.
        namespace = {"__builtins__": dict(vars(builtins), __import__=cpu_import)}
        exec(compile(CONTRACT.read_text(), str(CONTRACT), "exec"), namespace)
        for field in ("attention_backend", "prefill_attention_backend", "decode_attention_backend"):
            args = SimpleNamespace(attention_backend="triton")
            setattr(args, field, uq.QWEN_PLANNING_MODE)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cannot authorize"):
                namespace["validate_args"](args)
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            namespace["validate_args"](SimpleNamespace(
                attention_backend="triton", _ultraquant_qwen_planning_requested=True))
        registry = UPSTREAM / "python/sglang/srt/layers/attention/attention_registry.py"
        self.assertNotIn(uq.QWEN_PLANNING_MODE, registry.read_text())

    def test_exact_config_rank_geometry_precision_and_memory(self):
        for rank in (0, 1):
            plan = self.plan(rank)
            self.assertFalse(plan["native_launch_allowed"])
            self.assertFalse(plan["qualified"])
            with self.assertRaises(TypeError):
                plan["rank"] = 1 - rank
            self.assertEqual((plan["global_query_heads"], plan["global_kv_heads"],
                              plan["query_heads"], plan["kv_heads"], plan["head_dim"]),
                             (24, 4, 12, 2, 256))
            self.assertEqual(plan["full_layer_ids"], tuple(range(3, 64, 4)))
            self.assertEqual((len(plan["full_layer_ids"]), plan["gdn_layers"]), (16, 48))
            self.assertEqual(plan["protected_layer_ids"], (3, 7, 59, 63))
            self.assertEqual(plan["protected_compact_ids"], (0, 1, 14, 15))
            self.assertEqual((plan["input_dtype"], plan["output_dtype"], plan["v_rotation"]),
                             ("bfloat16", "bfloat16", "none"))
            # Independent count from per-layer K and V costs, not helper constants.
            packed_head = (256 // 32) * (32 // 2 + 1)
            protected_pair = 2 * 256 * 2
            expected = sum(2 * (protected_pair if layer in (3, 7, 59, 63)
                                else 2 * packed_head) for layer in range(3, 64, 4))
            self.assertEqual(expected, 14720)
            self.assertEqual(plan["payload_bytes_per_token_per_rank"], expected)
            payload = uq.qwen_payload_bytes(32768)
            self.assertEqual(payload["token_payload"], 460 * 1024 * 1024)
            self.assertEqual(payload["reserved_slot_zero"], 14720)
            self.assertEqual(payload["page_padding"], 0)
            self.assertEqual(payload["total_payload_with_slot_zero"], 460 * 1024 * 1024 + 14720)
            gdn = 48 * (24 * 128 * 128 * 4 + (2 * 8 * 128 + 24 * 128) * 3 * 2)
            self.assertEqual(plan["gdn_bytes_per_state_slot_per_rank_at_retained_defaults"], gdn)
            self.assertEqual(gdn / (1024 * 1024), 73.40625)

    def test_config_identity_and_planning_refusals(self):
        fixture = (ROOT / "tests/fixtures/qwen3.8-27b-fp8.config.json").read_bytes()
        self.assertEqual(hashlib.sha256(fixture).hexdigest(), uq.QWEN_CONFIG_SHA256)
        for field, value in (("model", "Qwen/other"), ("revision", "unknown"),
                             ("source_revision", "unknown"), ("source_tree_sha256", "bad"),
                             ("engine", "vllm"), ("rank", -1), ("rank", 2), ("rank", True),
                             ("rank", False), ("tp_size", 1), ("tp_size", True),
                             ("input_dtype", "float16"), ("output_dtype", "float16"),
                             ("storage_format", "vllm-ultraquant"),
                             ("rotation_policy", "pre-rope"), ("page_size", 16),
                             ("page_size", True), ("graphs", True), ("overlap", True),
                             ("mtp", True), ("graphs", None), ("mtp", 0)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.plan(**{field: value})
        for data in (fixture + b" ", fixture[:-1], b"{}", bytearray(fixture), b"x" * (1024 * 1024 + 1)):
            with self.subTest(config_type=type(data)), self.assertRaises(ValueError):
                uq.qwen_planning_metadata(data, model=uq.QWEN_MODEL, revision=uq.QWEN_REVISION,
                                         rank=0, source_tree_sha256=self.plan()["source_tree_sha256"])
        for slots in (-1, True, 1048576, "32768"):
            with self.assertRaises(ValueError):
                uq.qwen_payload_bytes(slots)

    def test_prefix_identity_rejects_stale_mixed_and_other_rank(self):
        plan = self.plan()
        identity = plan["prefix_identity"]
        uq.validate_qwen_prefix_identity(plan, identity)
        for index in range(len(identity)):
            changed = identity[:index] + (("stale",),) + identity[index + 1:]
            with self.subTest(identity_field=index), self.assertRaises(ValueError):
                uq.validate_qwen_prefix_identity(plan, changed)
        for other in (self.plan(1)["prefix_identity"], list(identity), None):
            with self.assertRaises(ValueError):
                uq.validate_qwen_prefix_identity(plan, other)
        changed_source = self.plan(source_tree_sha256="a" * 64)["prefix_identity"]
        with self.assertRaises(ValueError):
            uq.validate_qwen_prefix_identity(plan, changed_source)

    def test_physical_compact_routes_and_refusals(self):
        for compact, physical in enumerate(range(3, 64, 4)):
            expected = uq.qwen_layer_route(physical_id=physical, compact_id=compact)
            self.assertEqual(expected, uq.qwen_layer_route(compact_id=compact))
            self.assertEqual(expected, uq.qwen_layer_route(physical_id=physical))
            protected = physical in (3, 7, 59, 63)
            self.assertEqual(expected["protected"], protected)
            self.assertEqual(expected["storage"], "bfloat16" if protected else uq.QWEN_STORAGE)
            self.assertEqual(expected["rotation"], "none" if protected else uq.QWEN_ROTATION)
        for physical in set(range(64)) - set(range(3, 64, 4)):
            with self.assertRaises(ValueError):
                uq.qwen_layer_route(physical_id=physical)
        for kwargs in ({}, {"physical_id": 64}, {"physical_id": True}, {"compact_id": True},
                       {"compact_id": -1}, {"compact_id": 16}, {"physical_id": 3, "compact_id": 15}):
            with self.assertRaises(ValueError):
                uq.qwen_layer_route(**kwargs)

    def hybrid_writer(self):
        # Execute the retained production forwarding and ID-transfer methods.
        # unwrap_write_loc and the destination pool are CPU boundary stubs.
        file = UPSTREAM / "python/sglang/srt/mem_cache/memory_pool.py"
        tree = ast.parse(file.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "HybridLinearKVPool")
        functions = [next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == name)
                     for name in ("_transfer_full_attention_id", "set_kv_buffer")]
        for fn in functions:
            for arg in fn.args.args + fn.args.kwonlyargs:
                arg.annotation = None
            fn.returns = None
        namespace = {"unwrap_write_loc": lambda loc: (loc, None, None)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(file), "exec"), namespace)

        class FullPoolStub:
            def __init__(self):
                self.writes = []

            def set_kv_buffer(self, *args, **kwargs):
                route = uq.qwen_writer_metadata(*args, **kwargs)
                self.writes.append((route, args, kwargs))

        pool = SimpleNamespace(use_mla=False, full_kv_pool=FullPoolStub(),
                               full_attention_layer_id_mapping=dict(zip(range(3, 64, 4), range(16))))
        pool._transfer_full_attention_id = MethodType(namespace["_transfer_full_attention_id"], pool)
        return pool, MethodType(namespace["set_kv_buffer"], pool)

    def test_actual_hybrid_writer_boundary_and_pre_mutation_refusals(self):
        pool, write = self.hybrid_writer()
        loc, k, v = [9], b"opaque CPU K fixture", b"opaque CPU V fixture"
        for physical in (3, 7, 11, 55, 59, 63):
            write(SimpleNamespace(layer_id=physical), loc, k, v)
            route, args, kwargs = pool.full_kv_pool.writes[-1]
            self.assertIsNone(args[0])
            self.assertIs(args[1], loc)
            self.assertEqual(args[2:], (k, v, 1.0, 1.0))
            self.assertEqual(kwargs, {"layer_id_override": (physical - 3) // 4, "dcp_kv_mask": None})
            self.assertEqual(route, uq.qwen_layer_route(physical_id=physical))
        write(SimpleNamespace(layer_id=11), loc, k, v, None, 1)
        for field, value in (("k_scale", 2), ("v_scale", 0), ("k_scale", float("nan")),
                             ("v_scale", float("inf")), ("k_scale", True),
                             ("v_scale", "1"), ("dcp_kv_mask", []), ("dcp_kv_mask", False)):
            before = list(pool.full_kv_pool.writes)
            with self.subTest(field=field), self.assertRaises(ValueError):
                write(SimpleNamespace(layer_id=11), loc, k, v, **{field: value})
            self.assertEqual(pool.full_kv_pool.writes, before)
        before = list(pool.full_kv_pool.writes)
        for physical in (0, 2, 64):
            with self.assertRaises(ValueError):
                write(SimpleNamespace(layer_id=physical), loc, k, v)
        for kwargs in ({}, {"layer_id_override": 63}, {"layer_id_override": True}):
            with self.assertRaises(ValueError):
                uq.qwen_writer_metadata(None, loc, k, v, **kwargs)
        with self.assertRaises(ValueError):
            uq.qwen_writer_metadata(SimpleNamespace(layer_id=3), loc, k, v, layer_id_override=15)
        for layer in (SimpleNamespace(), SimpleNamespace(layer_id=None),
                      SimpleNamespace(layer_id=True), SimpleNamespace(layer_id="3")):
            with self.assertRaises(ValueError):
                pool.full_kv_pool.set_kv_buffer(layer, loc, k, v, layer_id_override=0)
        self.assertEqual(pool.full_kv_pool.writes, before)

    def test_actual_causal_rows_cancellation_metadata_and_slot_reuse(self):
        # Cancellation is represented only by removing a request's metadata.
        # This test runs causal_rows, not a scheduler or a recurrent-state model.
        sequences = [[9, 4, 12, 3], [9, 4, 22]]
        original = [row[:] for row in sequences]
        uq.causal_rows(sequences, [2, 2], [2, 1], [12, 3, 22], 32)
        self.assertEqual(sequences, original)
        next_sequences = [[9, 4, 22, 6], [9, 4, 12]]
        self.assertEqual(uq.causal_rows(next_sequences, [3, 2], [1, 1], [6, 12], 32),
                         ([9, 4, 22, 6, 9, 4, 12], [0, 4], [4, 3]))
        pool, write = self.hybrid_writer()
        # Invalid reuse of another live request's prefix rejects before the stub.
        with self.assertRaises(ValueError):
            uq.causal_rows([[9, 4, 22, 6], [9, 4, 22]], [3, 2], [1, 1], [6, 22], 32)
            write(SimpleNamespace(layer_id=11), [6, 22], b"K", b"V")
        self.assertEqual(pool.full_kv_pool.writes, [])


if __name__ == "__main__":
    unittest.main()
