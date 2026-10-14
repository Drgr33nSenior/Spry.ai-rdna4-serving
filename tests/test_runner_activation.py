"""Actual caller bodies with inert CPU files/devices; not serving evidence."""
import ast
import ctypes
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_allocation0521 as allocation
from test_cache0521 import SRT, method, evidence, uq

RUNNER = "model_executor/model_runner.py"
SETUP = "model_executor/model_runner_components/attention_backend_setup.py"


class RunnerActivationTests(unittest.TestCase):
    def test_actual_pool_constructor_identity_reaches_actual_setup(self):
        # The retained synthetic package starts from the historical fixture.
        # Bind this new boundary regression to the effective production bytes.
        backend = "layers/attention/ultraquant_backend.py"
        selected = (SRT / backend).read_bytes()
        (self.fixture.fixture.package / "srt" / backend).write_bytes(selected)
        self.fixture.fixture.refresh_inventory()
        request, identity = self.admit(self.fixture.kvc()._uq_bf16_request)
        runtime = allocation.actual_classes(self.fixture.fixture, self.fixture.standins)
        with patch.object(ctypes, "CDLL", side_effect=self.fixture.standins.library):
            pool = runtime.QwenBF16MixedPool(size=31, page_size=1, dtype="bfloat16",
                head_num=2, head_dim=256, layer_num=16, device="CPU declared cuda substitute",
                enable_memory_saver=False, policy=request["policy"],
                stack_receipt=request["stack_receipt"],
                stack_receipt_sha256=request["stack_receipt_sha256"])
        self.assertEqual(pool.identity, identity)
        with self.assertRaises(TypeError):
            pool.identity["canonical"] = b"changed"
        runner = NS(_uq_bf16_request=request, _uq_bf16_identity=identity,
            token_to_kv_pool=NS(full_kv_pool=pool), req_to_token_pool=object())
        wrapped = []
        build = method(SETUP, None, "_build_resolved_backend", {
            "__file__": str(self.fixture.fixture.package / "srt/model_executor/model_runner_components/attention_backend_setup.py"),
            "attn_backend_wrapper": lambda r, backend: wrapped.append(backend) or backend})
        resolved = NS(prefill=uq.BF16_SOURCE_MODE, decode=uq.BF16_SOURCE_MODE, is_draft_override=False)
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_backend": runtime}):
            adapter = build(model_runner=runner, resolved=resolved, init_new_workspace=False)
            self.assertIsInstance(adapter, runtime.QwenBF16SourceAdapter)
            self.assertIs(adapter.pool, pool)
            self.assertEqual(len(wrapped), 1)
            runner._uq_bf16_identity = {"canonical": b"stale"}
            with self.assertRaisesRegex(ValueError, "identity"):
                build(model_runner=runner, resolved=resolved, init_new_workspace=False)
            self.assertEqual(len(wrapped), 1)
        tree = ast.parse((SRT / "layers/attention/ultraquant_backend.py").read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenBF16MixedPool")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        stores = {n.targets[0].attr: i for i, n in enumerate(init.body)
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)}
        self.assertLess(stores["identity"], stores["owner"])
        self.assertLess(stores["identity"], stores["native"])

    def test_actual_local_model_config_exact_and_revision_gate(self):
        kvc = self.fixture.kvc()
        folder = self.fixture.host.folder / "local-model"
        folder.mkdir()
        (folder / "config.json").write_bytes(self.fixture.config_bytes)
        self.fixture.model.model_path = str(folder)
        self.assertIsNotNone(kvc._validate_bf16_allocation())
        self.fixture.model.revision = "wrong"
        with self.assertRaisesRegex(ValueError, "unknown Qwen"):
            kvc._validate_bf16_allocation()
        self.assertEqual(self.fixture.standins.library_loads, [])

    def test_actual_local_model_config_changed_missing_symlink_nonlocal_refuse(self):
        kvc = self.fixture.kvc()
        folder = self.fixture.host.folder / "local-model-refusal"
        folder.mkdir()
        config = folder / "config.json"
        self.fixture.model.model_path = str(folder)
        with self.assertRaises(FileNotFoundError):
            kvc._validate_bf16_allocation()
        config.write_bytes(b" " + self.fixture.config_bytes[1:])
        with self.assertRaisesRegex(ValueError, "pinned bytes"):
            kvc._validate_bf16_allocation()
        config.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "size"):
            kvc._validate_bf16_allocation()
        config.unlink()
        target = folder / "real.json"
        target.write_bytes(self.fixture.config_bytes)
        config.symlink_to(target)
        with self.assertRaises(OSError):
            kvc._validate_bf16_allocation()
        for path in ("unknown/model", "https://example.invalid/model"):
            self.fixture.model.model_path = path
            with self.assertRaisesRegex(ValueError, "unknown local"):
                kvc._validate_bf16_allocation()
        self.assertEqual(self.fixture.standins.library_loads, [])

    def test_actual_normalized_config_and_exact_resolved_layer_map(self):
        kvc = self.fixture.kvc()
        text = self.fixture.model.hf_text_config
        self.assertIsNotNone(kvc._validate_bf16_allocation())
        original_layers = text.layer_types
        text.layer_types = None
        text.rope_parameters = dict(text.rope_parameters, type=text.rope_parameters["rope_type"])
        self.assertIsNotNone(kvc._validate_bf16_allocation())
        del text.layer_types
        self.assertIsNotNone(kvc._validate_bf16_allocation())
        kvc.mambaish_config.full_attention_layer_ids = uq.QWEN_FULL_LAYERS[:-1]
        with self.assertRaisesRegex(ValueError, "unsupported BF16 allocation"):
            kvc._validate_bf16_allocation()
        kvc.mambaish_config.full_attention_layer_ids = uq.QWEN_FULL_LAYERS
        text.layer_types = original_layers[:-1]
        with self.assertRaisesRegex(ValueError, "layer types"):
            kvc._validate_bf16_allocation()
        self.assertEqual(self.fixture.standins.library_loads, [])

    def test_actual_normalized_rope_rejects_unknown_missing_changed_values(self):
        kvc = self.fixture.kvc()
        text = self.fixture.model.hf_text_config
        original = dict(text.rope_parameters)
        bad = [dict(original, type="wrong"), dict(original, extra_default=False)]
        bad.extend({k: v for k, v in original.items() if k != key} for key in original)
        bad.extend(dict(original, **{key: None}) for key in original)
        for rope in bad:
            with self.subTest(rope=rope):
                text.rope_parameters = rope
                with self.assertRaisesRegex(ValueError, "rope"):
                    kvc._validate_bf16_allocation()
        text.rope_parameters = original
        self.assertIsNotNone(kvc._validate_bf16_allocation())
        self.assertEqual(self.fixture.standins.library_loads, [])

    def test_selected_native_exact_four_pins(self):
        root = SRT.parents[3]
        expected = {
            "runtime/attention.hip": "ced35705d23a1879322c7c8e670c527e28059e841a544283e6beb41668421a3c",
            "runtime/bf16_math.h": "0290613cdaf8ddf057e010b9b37f653e96401ef44648e2ce1018b9c6958a0d11",
            "runtime/uq.h": "e4b278486a8a3434fbd1e90f799cdebc9e240bfa0f2e018524a911aea0ef75cb",
            "codec/codec.h": "852ce3209f934938e87818f77427258e16fd867880e45036a785db9f4e318446",
        }
        self.assertEqual({p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                          for p in expected}, expected)

    def setUp(self):
        self.fixture = allocation.Allocation0521Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = NS(tp_size=2, pp_size=1, dp_size=1, nnodes=1, attn_cp_size=1,
            dcp_size=1, page_size=1, dtype="bfloat16", kv_cache_dtype="auto",
            disable_overlap_schedule=True, cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled", disaggregation_mode="null", cpu_offload_gb=0)
        self.prepare = method(RUNNER, "ModelRunner", "_prepare_uq_bf16_request", {
            "__file__": str(self.fixture.fixture.package / "srt/model_executor/model_runner.py"),
            "get_parallel": lambda: self.fixture.parallel})

    def admit(self, request):
        return self.prepare(request, server_args=self.args, is_draft_worker=False,
            req_to_token_pool=None, token_to_kv_pool_allocator=None)

    def test_actual_prepare_both_ranks_immutable_and_proof_refusal(self):
        for rank in (0, 1):
            request = self.fixture.kvc(rank)._uq_bf16_request
            frozen, identity = self.admit(request)
            self.assertEqual(frozen["policy"]["rank"], rank)
            self.assertIsInstance(identity["canonical"], bytes)
            request["context_length"] = 1
            self.assertEqual(frozen["context_length"], 4096)
            with self.assertRaises(TypeError):
                frozen["policy"]["rank"] = 9
            with self.assertRaises(TypeError):
                frozen["context_length"] = 1
        request = self.fixture.kvc()._uq_bf16_request
        self.fixture.fixture.proof.unlink()
        with self.assertRaisesRegex(ValueError, "proof|build"):
            self.admit(request)
        self.assertEqual(self.fixture.standins.calls, [])

    def test_actual_prepare_rejects_public_mode_rank_and_missing_stack(self):
        request = self.fixture.kvc()._uq_bf16_request
        for key in ("stack_receipt", "stack_receipt_sha256"):
            with self.assertRaises(ValueError):
                self.admit(dict(request, **{key: None}))
        self.fixture.parallel.tp_rank = 1
        with self.assertRaisesRegex(ValueError, "policy"):
            self.admit(request)
        self.fixture.parallel.tp_rank = 0
        self.args.attention_backend = uq.BF16_SOURCE_MODE
        with self.assertRaisesRegex(ValueError, "not an admitted"):
            self.admit(request)
        self.assertEqual(self.admit(None), (None, None))

    def test_actual_configurator_caller_passes_snapshot_and_refuses_changed_identity(self):
        request, identity = self.admit(self.fixture.kvc()._uq_bf16_request)
        class Fields(NS):
            def __getattr__(self, key):
                return None
        runner = Fields(_uq_bf16_request=request, _uq_bf16_identity=identity,
            _prepare_uq_bf16_request=self.prepare, server_args=self.args, is_draft_worker=False)
        calls = []
        configure = method(RUNNER, "ModelRunner", "init_kv_cache_configurator", {
            "KVCacheConfigurator": lambda **kw: calls.append(kw) or NS(**kw),
            "get_parallel": lambda: NS(attn_dp_size=1), "is_post_capture_kv_active": lambda **kw: False})
        configure(runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["_uq_bf16_request"], request)
        self.fixture.fixture.value["compiler_revision"] = "different INERT revision"
        self.fixture.fixture.write()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            configure(runner)
        self.assertEqual(len(calls), 1)

    def test_actual_setup_identity_before_adapter_and_stock_fallback(self):
        request, identity = self.admit(self.fixture.kvc()._uq_bf16_request)
        class Pool:
            pass
        pool = Pool()
        pool.identity = identity
        runner = NS(_uq_bf16_request=request, _uq_bf16_identity=identity,
            token_to_kv_pool=NS(full_kv_pool=pool), req_to_token_pool=object())
        events = []
        def adapter(*args, **kw):
            events.append(("adapter", args, kw))
            return NS(marker="adapter")
        build = method(SETUP, None, "_build_resolved_backend", {
            "__file__": str(self.fixture.fixture.package / "srt/model_executor/model_runner_components/attention_backend_setup.py"),
            "attn_backend_wrapper": lambda r, a: events.append(("stock-wrapper", r, a)) or a,
            "_build_backend_from_str": lambda **kw: events.append(("stock", kw)) or "stock"})
        resolved = NS(prefill=uq.BF16_SOURCE_MODE, decode=uq.BF16_SOURCE_MODE, is_draft_override=False)
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_backend": NS(
                QwenBF16MixedPool=Pool, QwenBF16SourceAdapter=adapter)}):
            result = build(model_runner=runner, resolved=resolved, init_new_workspace=False)
            self.assertEqual(result.marker, "adapter")
            self.assertEqual([e[0] for e in events], ["adapter", "stock-wrapper"])
            self.assertIs(events[0][2]["token_to_kv_pool"], runner.token_to_kv_pool)
            pool.identity = {"canonical": b"stale"}
            with self.assertRaisesRegex(ValueError, "identity"):
                build(model_runner=runner, resolved=resolved, init_new_workspace=False)
            self.assertEqual(len(events), 2)
        runner._uq_bf16_request = None
        resolved.prefill = resolved.decode = "triton"
        self.assertEqual(build(model_runner=runner, resolved=resolved, init_new_workspace=False), "stock")

    def test_constructor_and_unchanged_stock_hybrid_source(self):
        root = SRT.parents[4]
        tree = ast.parse((SRT / RUNNER).read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertEqual(init.args.kwonlyargs[0].arg, "_uq_bf16_request")
        self.assertEqual(init.body[0].value.func.attr, "_prepare_uq_bf16_request")
        # Wrapper and hybrid dispatch are byte-identical to the authenticated fixture source.
        for name in ("attention_registry.py", "hybrid_linear_attn_backend.py"):
            self.assertEqual((SRT / "layers/attention" / name).read_bytes(),
                (root / "tests/fixtures/stock-v0521/python/sglang/srt/layers/attention" / name).read_bytes())

    def test_actual_stock_wrapper_constructs_gdn_and_hybrid(self):
        class Other:
            pass
        class GDN:
            needs_cpu_seq_lens = True
            def __init__(self, runner):
                self.runner = runner
        class Hybrid:
            __init__ = method("layers/attention/hybrid_linear_attn_backend.py",
                "HybridLinearAttnBackend", "__init__")
        modules = {
            "sglang.srt.configs.dots3": NS(Dots3Config=Other),
            "sglang.srt.configs.model_config": NS(is_minimax_sparse=lambda c: False),
            "sglang.srt.configs.inkling": NS(InklingMMConfig=Other, InklingModelConfig=Other),
            "sglang.kernels.ops.attention.fla.utils": NS(check_environments=lambda: None),
            "sglang.srt.layers.attention.linear.kda_backend": NS(KDAAttnBackend=Other),
            "sglang.srt.layers.attention.linear.lightning_backend": NS(LightningAttentionBackend=Other),
            "sglang.srt.layers.attention.linear.utils": NS(resolve_linear_attn_backends=lambda **kw: "INERT-GDN-policy"),
            "sglang.srt.utils": NS(is_blackwell=lambda: False, is_npu=lambda: False, is_xpu=lambda: False),
            "sglang.srt.layers.attention.hybrid_linear_attn_backend": NS(HybridLinearAttnBackend=Hybrid, Mamba2AttnBackend=Other),
            "sglang.srt.layers.attention.linear.gdn_backend": NS(GDNAttnBackend=GDN, flashinfer_gdn_prefill_default=lambda r: None),
            "sglang.srt.layers.attention.qsa.config": NS(is_qwen_qsa=lambda c: False),
        }
        wrapper = method("layers/attention/attention_registry.py", None, "attn_backend_wrapper", {
            "hybrid_gdn_config": lambda c: object(),
            "mambaish_config": lambda c: NS(full_attention_layer_ids=uq.QWEN_FULL_LAYERS),
            "logger": NS(info=lambda *a: None)})
        runner = NS(model_config=NS(hf_text_config=NS(), hf_config=NS()),
            use_mla_backend=False, is_draft_worker=False)
        full = NS(token_to_kv_pool=object(), req_to_token_pool=object(),
            kv_index_translator=None, max_context_len=4096, needs_cpu_seq_lens=True)
        with patch.dict(sys.modules, modules):
            result = wrapper(runner, full)
        self.assertIsInstance(result, Hybrid)
        self.assertIs(result.full_attn_backend, full)
        self.assertIsInstance(result.linear_attn_backend, GDN)
        self.assertIs(result.linear_attn_backend.runner, runner)
        self.assertEqual(result.full_attn_layers, uq.QWEN_FULL_LAYERS)
        self.assertEqual(runner.linear_attn_backends, "INERT-GDN-policy")
