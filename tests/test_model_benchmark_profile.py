"""Bounded attribution composition with explicit CPU substitutes for Torch/HIP.

No runtime or GPU library is imported. Source identities in synthetic records
are declared fixtures; the production validator and entry functions are real.
"""
import copy
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("profile_benchmark_tests", ROOT / "tests/test_model_benchmark.py")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
bench = base.bench


def profile_fixture_records(arm="bf16", *, session_sha256="a" * 64, spec_sha256="b" * 64):
    """Declared synthetic complete records, suitable for independent-reader tests."""
    records = base.completed_records()
    for rank, record in enumerate(records):
        record["arm"] = arm
        identity = base.native_identity(rank, spec_sha256)
        identity.update(arm=arm, session_sha256=session_sha256)
        record["native_identity"] = identity
        record["instrumented_timings"] = True
        record["profile_method"] = bench.PROFILE_METHOD
        sources = [{"path": bench.PROFILE_PACKAGE_ROOT + "/" + relative,
                    "sha256": pin, "bytes": (ROOT / "prototype/image-source/python/sglang" / relative).stat().st_size,
                    "qualnames": list(names)} for relative, (pin, names) in bench.PROFILE_SOURCES.items()]
        windows = []
        for ordinal, item in enumerate(bench.request_plan({})):
            for step in range(32):
                if not bench.profile_selected(item, step):
                    continue
                binding = bench.profile_binding(item, ordinal, step, rank, arm, session_sha256, spec_sha256)
                binding["pid"] = identity["process"]["pid"]
                names = ["native_gemm", "nccl_allreduce"] if arm == "bf16" else [
                    "void bf16_v2::validate_read(...)", "void bf16_v2::read(...)",
                    "void bf16_v2::write(...)", "Memcpy DtoH"]
                events = [NS(name=name, device_type="DeviceType.CUDA", device_index=rank,
                             time_range=NS(start=i * 100.0, end=i * 100.0 + 50.0))
                          for i, name in enumerate(names)]
                cpu = [{"path": source["path"], "qualname": name, "sha256": source["sha256"],
                        "calls": 1, "recursive_calls": 0, "inclusive_seconds": .01,
                        "exclusive_seconds": .001} for source in sources for name in source["qualnames"]]
                windows.append({**binding, "method": bench.PROFILE_METHOD,
                    "instrumented_timings": True, "complete": True, "errors": [],
                    "cpu": cpu, "gpu": bench.profile_gpu_rows(events, rank),
                    "observer_restored": True, "monotonic_start": ordinal * 100.0 + step,
                    "monotonic_end": ordinal * 100.0 + step + .5})
        record["diagnostic_profile"] = {**bench.PROFILE_CONFIG, "method": bench.PROFILE_METHOD,
            "complete": True, "qualification": False, "sources": sources, "windows": windows,
            "entry_sha256": bench.hashlib.sha256((ROOT / "tools/model_benchmark.py").read_bytes()).hexdigest(),
            "native_binary_sha256": bench.BINARY_SHA256}
    return records


class Context:
    def __init__(self, log, name):
        self.log, self.name = log, name
    def __enter__(self):
        self.log.append("enter:" + self.name)
        return self
    def __exit__(self, *error):
        self.log.append("exit:" + self.name)
        return False


def forward_boundaries(*, failure=None, start_failure=None, stop_failure=None, supported=True, gpu_events=None):
    """Actual cProfile observes only these explicitly synthetic code boundaries."""
    log = []
    source_paths = {Path(relative).name: bench.PROFILE_PACKAGE_ROOT + "/" + relative
                    for relative in bench.PROFILE_SOURCES}
    scope = {"NS": NS, "log": log, "failure": failure}
    exec(compile("class ForwardBatch:\n"
                 " @staticmethod\n"
                 " def init_new(*args, **kwargs):\n"
                 "  log.append('init'); return 'fb'\n", source_paths["forward_batch_info.py"], "exec"), scope)
    exec(compile("class ModelRunner:\n"
                 " def forward(self, fb):\n"
                 "  log.append('forward')\n"
                 "  if failure is not None: raise failure\n"
                 "  return NS(logits_output='logits')\n"
                 " def sample(self, logits, fb):\n"
                 "  log.append('sample'); return 'sampled'\n", source_paths["model_runner.py"], "exec"), scope)
    events = gpu_events if gpu_events is not None else [NS(name="native_gemm", device_type="DeviceType.CUDA",
        device_index=0, time_range=NS(start=0.0, end=20.0))]
    class GPUProfile(Context):
        def __init__(self, **kwargs):
            super().__init__(log, "gpu")
            self.options = kwargs
        def __enter__(self):
            super().__enter__()
            if start_failure:
                raise start_failure
            return self
        def __exit__(self, *args):
            super().__exit__(*args)
            if stop_failure:
                raise stop_failure
            return False
        def events(self):
            return events() if callable(events) else events
    torch = NS(cuda=NS(synchronize=lambda: log.append("sync")), no_grad=lambda: Context(log, "no_grad"),
        profiler=NS(supported_activities=lambda: {"CPU", "CUDA"} if supported else {"CPU"},
            ProfilerActivity=NS(CPU="CPU", CUDA="CUDA"), profile=GPUProfile,
            record_function=lambda name: Context(log, name)))
    return torch, scope["ModelRunner"](), scope["ForwardBatch"], log


def rank_boundaries(arm, *, unsupported=False, rank=0, profile_event_factory=None):
    """The native/process boundaries for the actual run_rank cleanup composition."""
    calls = {"forward": 0, "sample": 0, "release": 0, "destroy": 0, "profile": []}
    class Values:
        def __init__(self, count):
            self.shape = (count, 8192)
        def float(self):
            return self
        def log_softmax(self, **kwargs):
            return self
        def __getitem__(self, item):
            return NS(item=lambda: -1.0)
    class Tokens(list):
        def tolist(self):
            return list(self)
        def to(self, *args, **kwargs):
            return self
    owner = NS(refuse_reset=lambda: None)
    class Runner:
        def __init__(self, **kwargs):
            self._uq_bf16_request = kwargs["_uq_bf16_request"]
            pool = type("QwenBF16MixedPool" if arm == "kv4" else "MHATokenToKVPool", (),
                        {"owner": owner} if arm == "kv4" else {"dtype": "bf16"})()
            self.token_to_kv_pool = NS(full_kv_pool=pool)
            self.token_to_kv_pool_allocator = NS(available_size=lambda: 4096)
            self.req_to_token_pool = NS(free_slots=[0, 1], mamba_allocator=NS(available_size=lambda: 68))
            self.attn_backend = NS(full_attn_backend=type(
                "QwenBF16SourceAdapter" if arm == "kv4" else "TritonAttnBackend", (), {})())
        def alloc_memory_pool(self):
            pass
        init_attention_backends = init_cuda_graphs = alloc_memory_pool
        def forward(self, fb):
            calls["forward"] += 1
            return NS(logits_output=NS(next_token_logits=Values(fb.batch_size)))
        def sample(self, logits, fb):
            calls["sample"] += 1
            return Tokens([7] * fb.batch_size)
    class Req:
        def __init__(self, **kwargs):
            self.full_untruncated_fill_ids = kwargs["origin_input_ids"]
            self.output_ids, self.prefix_indices, self.last_node = [], [], None
            self.kv = NS(holds_kv=True, holds_mamba=True)
        def init_next_round_input(self, cache):
            pass
        def set_extend_range(self, *args):
            pass
    class Batch:
        @staticmethod
        def init_new(**kwargs):
            return Batch(kwargs["reqs"])
        def __init__(self, reqs):
            self.reqs, self.input_ids, self.prefill_input_ids_cpu = reqs, Tokens([0] * len(reqs)), None
        def prepare_for_extend(self):
            pass
        prepare_for_decode = prepare_for_extend
    class FB:
        @staticmethod
        def init_new(batch, runner, **kwargs):
            return NS(batch_size=len(batch.reqs))
    class Cache:
        def __init__(self, params):
            pass
        def inc_lock_ref(self, node):
            return NS(to_dec_params=lambda: None)
        def reset(self):
            pass
        release_for_idle_bf16_flush = reset
    def release(req, cache, is_insert):
        calls["release"] += 1
        req.kv.holds_kv = req.kv.holds_mamba = False
    def destroy():
        calls["destroy"] += 1
    # These code identities declare CPU replacement functions, not native source.
    for method, relative, qualname in (
            (Runner.forward, "srt/model_executor/model_runner.py", "ModelRunner.forward"),
            (Runner.sample, "srt/model_executor/model_runner.py", "ModelRunner.sample"),
            (FB.init_new, "srt/model_executor/forward_batch_info.py", "ForwardBatch.init_new")):
        method.__code__ = method.__code__.replace(co_filename=bench.PROFILE_PACKAGE_ROOT + "/" + relative,
                                                 co_qualname=qualname)
    torch = ModuleType("torch")
    gpu_names = ["bf16_v2::read(...)", "bf16_v2::validate_read(...)", "bf16_v2::write(...)"]
    gpu_events = [NS(name=name, device_type="CUDA", device_index=rank,
                     time_range=NS(start=i * 10., end=i * 10. + 5.)) for i, name in enumerate(gpu_names)]
    if profile_event_factory is not None:
        gpu_events = lambda: profile_event_factory(len(calls["profile"]))
    observed_torch, _, _, _ = forward_boundaries(supported=not unsupported, gpu_events=gpu_events)
    torch.profiler = observed_torch.profiler
    original_profile = torch.profiler.profile
    def make_profile(**kwargs):
        calls["profile"].append(kwargs)
        return original_profile(**kwargs)
    torch.profiler.profile = make_profile
    torch.cuda, torch.no_grad = observed_torch.cuda, observed_torch.no_grad
    torch.bfloat16, torch.int64 = "bf16", "int64"
    torch.set_num_threads = torch.set_num_interop_threads = lambda value: None
    torch.isfinite = lambda values: NS(all=lambda: NS(item=lambda: True))
    dist = ModuleType("torch.distributed")
    dist.all_gather_object = lambda peers, value, **kwargs: peers.__setitem__(slice(None), [value, value])
    torch.distributed = dist
    entries = {
        "sglang.srt.configs.model_config": {"ModelConfig": NS(from_server_args=lambda args: NS(vocab_size=8192))},
        "sglang.srt.distributed": {"bootstrap": NS(init_parallel_runtime=lambda **kwargs: None,
                                                   init_layer_runtime=lambda **kwargs: None)},
        "sglang.srt.distributed.parallel_state": {"destroy_distributed_environment": destroy,
            "destroy_model_parallel": destroy, "get_tp_group": lambda: NS(cpu_group="declared CPU")},
        "sglang.srt.entrypoints.engine": {"_set_envs_and_config": lambda args: None},
        "sglang.srt.managers.schedule_batch": {"Req": Req, "ScheduleBatch": Batch},
        "sglang.srt.mem_cache.cache_init_params": {"CacheInitParams": lambda **kwargs: NS(**kwargs)},
        "sglang.srt.mem_cache.common": {"release_kv_cache": release},
        "sglang.srt.mem_cache.unified_radix_cache": {"UnifiedRadixCache": Cache},
        "sglang.srt.mem_cache.unified_cache.component_type": {"ComponentType": NS(FULL="full", MAMBA="mamba")},
        "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": FB},
        "sglang.srt.model_executor.model_runner": {"ModelRunner": Runner},
        "sglang.srt.runtime_context": {"SpawnRanks": lambda **kwargs: NS(**kwargs),
            "get_device": lambda: NS(device="CPU boundary"),
            "get_model": lambda: NS(is_startup_weight_load_overlap=False),
            "publish": lambda *args, **kwargs: None, "spawn_world_rank": lambda *args, **kwargs: kwargs["tp_rank"]},
        "sglang.srt.sampling.sampling_params": {"SamplingParams": lambda **kwargs: NS(**kwargs)},
        "sglang.srt.server_args": {"ServerArgs": lambda **kwargs: NS(**kwargs, resolve_once=lambda: None)},
        "sglang.srt.speculative.spec_info": {"SpeculativeAlgorithm": NS(NONE="none")},
        "sglang.srt.layers.attention.ultraquant_contract": {"bf16_source_request": lambda *args, **kwargs: {}},
        "sglang.srt.layers.moe": {"initialize_moe_config": lambda: None},
        "sglang.srt.layers.quantization.fp4_utils": {"initialize_fp4_gemm_config": lambda: None},
        "sglang.srt.layers.quantization.fp8_utils": {"initialize_fp8_gemm_config": lambda: None},
        "sglang.kernels.ops.mamba.triton_ops": {"initialize_mamba_selective_state_update_backend": lambda args: None}}
    modules = {"torch": torch, "torch.distributed": dist}
    for name, fields in entries.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        modules[name] = module
    return modules, calls


class ModelBenchmarkProfileTests(unittest.TestCase):
    def binding(self):
        item = list(bench.request_plan({}))[1]
        return bench.profile_binding(item, 1, 0, 0, "bf16", "a" * 64, "b" * 64)

    def test_selection_exactly_four_existing_forwards_no_new_work(self):
        all_steps = [(ordinal, item, step) for ordinal, item in enumerate(bench.request_plan({}))
                     for step in range(32)]
        selected = [(ordinal, item["cell"], item["sample"], step) for ordinal, item, step in all_steps
                    if bench.profile_selected(item, step)]
        self.assertEqual(len(all_steps), 960)
        self.assertEqual(selected, [(1, "C1-128", 1, 0), (1, "C1-128", 1, 1),
                                    (13, "C1-3966", 1, 0), (13, "C1-3966", 1, 1)])
        self.assertEqual(sum(item["concurrency"] for item in bench.request_plan({})), 42)

    def test_optional_spec_exact_type_and_value_baseline_unchanged(self):
        self.assertEqual(bench.validate_spec(base.fixture()), base.fixture())
        for value in (None, False, {}, {"schema": True, "mode": "bounded-attribution-v1"},
                      {**bench.PROFILE_CONFIG, "extra": True}, {"schema": 1, "mode": "cpu-only"}):
            spec = base.fixture()
            spec["diagnostic_profile"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.validate_spec(spec)
        spec = base.fixture()
        spec["diagnostic_profile"] = dict(bench.PROFILE_CONFIG)
        self.assertEqual(bench.validate_spec(spec), spec)

    def test_actual_timed_forward_labels_each_section_once(self):
        torch, runner, fb, log = forward_boundaries()
        ticks = iter([1.0, 1.5])
        answer = bench.timed_forward(runner, "batch", fb, torch, lambda: next(ticks), profile_labels=True)
        self.assertEqual(answer, ("logits", "sampled", .5, "fb"))
        self.assertEqual(log.count("forward"), 1)
        self.assertEqual(log.count("init"), 1)
        self.assertEqual(log.count("sample"), 1)
        self.assertEqual([event for event in log if event.startswith("enter:spry.")],
                         ["enter:" + label for label in bench.PROFILE_LABELS])
        torch, runner, fb, log = forward_boundaries()
        bench.timed_forward(runner, "batch", fb, torch)
        self.assertFalse(any("spry." in entry for entry in log))

    def test_real_cprofile_wall_observes_only_exact_source_and_qualname(self):
        torch, runner, fb, log = forward_boundaries()
        records = profile_fixture_records()
        sources = records[0]["diagnostic_profile"]["sources"]
        windows = []
        self.assertIsNone(sys.getprofile())
        answer = bench.profiled_forward(runner, "batch", fb, torch, self.binding(), sources, windows)
        self.assertEqual(answer[:2], ("logits", "sampled"))
        self.assertIsNone(sys.getprofile())
        self.assertEqual(len(windows), 1)
        row = windows[0]
        self.assertTrue(row["complete"])
        self.assertNotIn("gpu_event_rejection", row)
        self.assertNotIn("gpu_collection_failure", row)
        self.assertTrue(row["observer_restored"])
        self.assertEqual({row["qualname"] for row in row["cpu"]},
                         {"ForwardBatch.init_new", "ModelRunner.forward", "ModelRunner.sample"})
        self.assertTrue(all(item["calls"] == 1 and item["inclusive_seconds"] >= item["exclusive_seconds"] >= 0
                            for item in row["cpu"]))
        self.assertEqual(log.count("forward"), 1)
        self.assertEqual(log.count("exit:gpu"), 1)
        wrong = copy.deepcopy(sources)
        for source in wrong:
            source["path"] = "/unbound/" + Path(source["path"]).name
        windows = []
        bench.profiled_forward(runner, "batch", fb, torch, self.binding(), wrong, windows)
        self.assertEqual(windows[0]["cpu"], [])

    def test_original_forward_error_survives_observer_error_and_still_restores(self):
        original = RuntimeError("declared forward failure")
        torch, runner, fb, log = forward_boundaries(failure=original,
            stop_failure=RuntimeError("declared profiler stop failure"))
        windows = []
        with self.assertRaises(RuntimeError) as error:
            bench.profiled_forward(runner, "batch", fb, torch, self.binding(),
                profile_fixture_records()[0]["diagnostic_profile"]["sources"], windows)
        self.assertIs(error.exception, original)
        self.assertIsNone(sys.getprofile())
        self.assertFalse(windows[0]["complete"])
        self.assertIn("declared forward failure", windows[0]["errors"][0])
        self.assertTrue(any("GPU stop" in value for value in windows[0]["errors"]))
        self.assertEqual(log.count("exit:gpu"), 1)
        self.assertEqual(log.count("sample"), 0)

    def test_partially_started_gpu_observer_still_exits_in_finally(self):
        original = RuntimeError("declared partial profiler initialization")
        torch, runner, fb, log = forward_boundaries(start_failure=original)
        windows = []
        with self.assertRaises(RuntimeError) as error:
            bench.profiled_forward(runner, "batch", fb, torch, self.binding(), [], windows)
        self.assertIs(error.exception, original)
        self.assertEqual(log.count("enter:gpu"), 1)
        self.assertEqual(log.count("exit:gpu"), 1)
        self.assertNotIn("forward", log)
        self.assertFalse(windows[0]["complete"])
        self.assertIsNone(sys.getprofile())

    def test_gpu_unsupported_empty_and_collection_error_refuse_without_fallback(self):
        for options in ({"supported": False}, {"gpu_events": []},
                        {"gpu_events": [NS(device_type="CUDA")]},
                        {"stop_failure": RuntimeError("observer-stop")}):
            torch, runner, fb, log = forward_boundaries(**options)
            windows = []
            with self.subTest(options=options), self.assertRaises(ValueError):
                bench.profiled_forward(runner, "batch", fb, torch, self.binding(),
                    profile_fixture_records()[0]["diagnostic_profile"]["sources"], windows)
            self.assertFalse(windows[0]["complete"])
            self.assertTrue(windows[0]["errors"])
            self.assertIsNone(sys.getprofile())
            if not options.get("supported", True):
                self.assertNotIn("forward", log)

    def test_existing_cpu_observer_is_preserved_and_refused(self):
        torch, runner, fb, log = forward_boundaries()
        observer = lambda *args: None
        sys.setprofile(observer)
        try:
            with self.assertRaisesRegex(ValueError, "pre-existing"):
                bench.profiled_forward(runner, "batch", fb, torch, self.binding(), [], [])
            self.assertIs(sys.getprofile(), observer)
            self.assertNotIn("forward", log)
        finally:
            sys.setprofile(None)

    def test_sources_exact_canonical_paths_pins_and_no_symlinks(self):
        fixture = {"package_root": bench.PROFILE_PACKAGE_ROOT}
        with patch.object(bench.Path, "resolve", autospec=True, side_effect=lambda path: path), \
                patch.object(bench, "checked_file", return_value=b"declared source") as check:
            rows = bench.profile_sources(fixture)
        self.assertEqual(len(rows), 4)
        self.assertEqual(check.call_count, 4)
        self.assertEqual({call.args[1] for call in check.call_args_list},
                         {value[0] for value in bench.PROFILE_SOURCES.values()})
        with self.assertRaises(ValueError):
            bench.profile_sources({"package_root": "/unbound"})
        with patch.object(bench.Path, "resolve", return_value=Path("/unbound")), self.assertRaises(ValueError):
            bench.profile_sources(fixture)

    def test_both_ranks_four_windows_complete_only_with_explicit_profile_mode(self):
        for arm in ("bf16", "kv4"):
            rows = profile_fixture_records(arm)
            self.assertIs(bench.validate_profile_records(rows, arm,
                session_sha256="a" * 64, spec_sha256="b" * 64), rows)
            self.assertEqual(len(bench.summarize(rows, arm, expected_profile=True)), 30)
            with self.assertRaisesRegex(ValueError, "unexpected attribution"):
                bench.summarize(rows, arm)
            self.assertTrue(all(row["diagnostic_profile"]["qualification"] is False for row in rows))
        with self.assertRaises(ValueError):
            bench.summarize(base.completed_records(), "bf16", expected_profile=True)

    def test_actual_rank_entry_runs_960_forwards_profiles_four_and_cleans_on_observer_failure(self):
        for arm, unsupported in (("bf16", False), ("kv4", False), ("kv4", True)):
            modules, calls = rank_boundaries(arm, unsupported=unsupported)
            spec = base.fixture()
            spec["diagnostic_profile"] = dict(bench.PROFILE_CONFIG)
            identity = base.native_identity(0)
            identity["arm"] = arm
            sources = profile_fixture_records(arm)[0]["diagnostic_profile"]["sources"]
            with self.subTest(arm=arm, unsupported=unsupported), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.dict(sys.modules, modules), patch.object(bench, "preflight", return_value=b"{}"), \
                        patch.object(bench, "preweight_identity", return_value=identity), \
                        patch.object(bench, "profile_sources", return_value=sources), \
                        patch.object(bench.os, "setsid"), patch.object(bench.os, "dup2"), \
                        patch.object(bench.resource, "setrlimit"), self.assertRaises(SystemExit) as exited:
                    bench.run_rank(spec, arm, 0, output, True, "a" * 64, "b" * 64)
                result = bench.read_rank_result(output / "rank-0.json")
                self.assertEqual(exited.exception.code, 1 if unsupported else 0)
                self.assertEqual(calls["destroy"], 2)
                self.assertTrue(calls["release"])
                self.assertEqual(result["cleanup_errors"], [])
                self.assertFalse(result["qualification"])
                self.assertIsNone(sys.getprofile())
                if unsupported:
                    self.assertEqual(calls["forward"], 32)
                    self.assertIsNotNone(result["failure"])
                    self.assertEqual(result["completed_batches"], 1)
                    self.assertFalse(result["diagnostic_profile"]["complete"])
                    self.assertEqual(len(result["diagnostic_profile"]["windows"]), 1)
                else:
                    self.assertEqual(calls["forward"], 960)
                    self.assertEqual(calls["sample"], 960)
                    self.assertEqual(len(calls["profile"]), 4)
                    self.assertTrue(result["diagnostic_profile"]["complete"])
                    self.assertEqual(result["completed_requests"], 42)
                    for options in calls["profile"]:
                        self.assertEqual(options["activities"], ["CPU", "CUDA"])
                        self.assertFalse(options["record_shapes"])
                        self.assertFalse(options["profile_memory"])
                        self.assertFalse(options["with_stack"])
                        self.assertFalse(options["with_flops"])

    def test_complete_validator_rejects_substitution_missing_nonfinite_size_counts_and_bindings(self):
        original = profile_fixture_records("kv4")
        mutations = [
            lambda p: p.__setitem__("complete", False),
            lambda p: p.__setitem__("qualification", True),
            lambda p: p.__setitem__("entry_sha256", "0" * 64),
            lambda p: p.__setitem__("native_binary_sha256", "0" * 64),
            lambda p: p["sources"][0].__setitem__("path", "/unbound/ultraquant_backend.py"),
            lambda p: p["sources"][0].__setitem__("sha256", "0" * 64),
            lambda p: p["windows"].pop(),
            lambda p: p["windows"].append(copy.deepcopy(p["windows"][0])),
            lambda p: p["windows"].reverse(),
            lambda p: p["windows"][0].__setitem__("pid", 999),
            lambda p: p["windows"][0].__setitem__("session_sha256", "0" * 64),
            lambda p: p["windows"][0].__setitem__("spec_sha256", "0" * 64),
            lambda p: p["windows"][0].__setitem__("ordinal", 0),
            lambda p: p["windows"][0].__setitem__("phase", "decode"),
            lambda p: p["windows"][0].__setitem__("monotonic_end", -1),
            lambda p: p["windows"][0].__setitem__("observer_restored", False),
            lambda p: p["windows"][0]["cpu"][0].__setitem__("inclusive_seconds", float("nan")),
            lambda p: p["windows"][0]["cpu"][0].__setitem__("calls", 0),
            lambda p: p["windows"][0]["cpu"][0].__setitem__("sha256", "0" * 64),
            lambda p: p["windows"][0]["cpu"].clear(),
            lambda p: p["windows"][0]["gpu"].__setitem__("intervals", []),
            lambda p: p["windows"][0]["gpu"].__setitem__("all_gpu_events_retained", False),
            lambda p: p["windows"][0]["gpu"]["intervals"][0].__setitem__(1, float("inf")),
            lambda p: p["windows"][0]["gpu"]["intervals"][0].__setitem__(3, 1),
            lambda p: p["windows"][0]["gpu"]["categories"]["kv_validation"].__setitem__("events", 999),
            lambda p: p["windows"][0].__setitem__("padding", "x" * (256 << 10)),
        ]
        for index, mutate in enumerate(mutations):
            rows = copy.deepcopy(original)
            mutate(rows[0]["diagnostic_profile"])
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                bench.validate_profile_records(rows, "kv4", session_sha256="a" * 64, spec_sha256="b" * 64)
        for keyword in ({"session_sha256": "0" * 64}, {"spec_sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                bench.validate_profile_records(original, "kv4", **keyword)

    def test_gpu_kernel_fallback_event_and_output_bounds_refuse(self):
        rows = profile_fixture_records("kv4")
        gpu = rows[0]["diagnostic_profile"]["windows"][0]["gpu"]
        event = NS(name="native_gemm", device_type="CUDA", device_index=0,
                   time_range=NS(start=0, end=1))
        rows[0]["diagnostic_profile"]["windows"][0]["gpu"] = bench.profile_gpu_rows([event], 0)
        with self.assertRaisesRegex(ValueError, "active v2 GPU"):
            bench.validate_profile_records(rows, "kv4")
        with patch.object(bench, "PROFILE_EVENT_LIMIT", 1), self.assertRaises(ValueError):
            bench.profile_gpu_rows([event, event], 0)
        torch, runner, fb, log = forward_boundaries()
        windows = []
        with patch.object(bench, "PROFILE_WINDOW_LIMIT", 1), self.assertRaises(ValueError):
            bench.profiled_forward(runner, "batch", fb, torch, self.binding(),
                profile_fixture_records()[0]["diagnostic_profile"]["sources"], windows)
        self.assertFalse(windows[0]["complete"])
        self.assertIsNone(windows[0]["gpu"])
        self.assertIsNone(sys.getprofile())

    def rejected_events(self):
        """Each fixture follows one CPU event and one accepted rank-1 CUDA event."""
        valid = NS(name="native_gemm", device_type="DeviceType.CUDA", device_index=1,
                   time_range=NS(start=0., end=1.))
        cases = [("name_type", "name", False), ("name_empty", "name", ""),
                 ("name_utf8_bytes_exceeds_16384", "name", "界" * 6000),
                 ("name_utf8_encoding", "name", "bad\ud800name"),
                 ("device_index_type", "device_index", True),
                 ("device_index_type", "device_index", "1"),
                 ("device_index_mismatch", "device_index", 0),
                 ("start_type", "start", True), ("start_type", "start", "0"),
                 ("start_nonfinite", "start", float("nan")),
                 ("start_nonfinite", "start", float("inf")),
                 ("start_nonfinite", "start", -float("inf")),
                 ("start_negative", "start", -1.),
                 ("end_type", "end", False),
                 ("end_nonfinite", "end", float("nan")),
                 ("end_nonfinite", "end", float("inf")),
                 ("end_nonfinite", "end", -float("inf")),
                 ("end_negative", "end", -1.),
                 ("end_before_start", "end", .5)]
        for reason, field, value in cases:
            rejected = NS(name="bf16_v2::read(...)", device_type="DeviceType.CUDA", device_index=1,
                          time_range=NS(start=1., end=2.))
            setattr(rejected.time_range if field in ("start", "end") else rejected, field, value)
            yield reason, [NS(device_type="DeviceType.CPU"), valid, rejected]

    def assert_rejected_event(self, rejection, reason):
        self.assertEqual((rejection["event_index"], rejection["cuda_event_index"],
                          rejection["index_base"], rejection["expected_device_index"]), (2, 1, 0, 1))
        self.assertIn(reason, rejection["reasons"])
        encoded = json.dumps(rejection, sort_keys=True, allow_nan=False).encode("utf-8")
        self.assertLess(len(encoded), 8192)
        self.assertEqual(json.loads(encoded), rejection)
        self.assertEqual(rejection["observed"]["device_type"]["prefix"], "DeviceType.CUDA")

    def test_actual_gpu_parser_retains_every_rejection_after_valid_cuda_event(self):
        for reason, events in self.rejected_events():
            with self.subTest(reason=reason), self.assertRaises(bench.ProfileGPUEventError) as raised:
                bench.profile_gpu_rows(events, 1)
            rejection = raised.exception.rejection
            self.assert_rejected_event(rejection, reason)
            if reason == "name_utf8_bytes_exceeds_16384":
                name = rejection["observed"]["name"]
                self.assertTrue(name["prefix_truncated"])
                self.assertEqual(name["characters"], 6000)
                self.assertLessEqual(name["prefix_utf8_bytes"], 512)
                self.assertEqual(name["prefix_utf8_bytes"], len(name["prefix"].encode("utf-8")))
            if reason.endswith("_nonfinite"):
                field = reason.split("_")[0]
                self.assertIs(rejection["observed"][field]["finite"], False)
                self.assertIsInstance(rejection["observed"][field]["value"], str)
            if reason == "end_before_start":
                self.assertIs(rejection["end_at_or_after_start"], False)
                self.assertEqual(rejection["observed"]["start"]["value"], 1.)
                self.assertEqual(rejection["observed"]["end"]["value"], .5)
        valid = NS(name="native_gemm", device_type="CUDA", device_index=1,
                   time_range=NS(start=2., end=3.))
        gpu = bench.profile_gpu_rows([NS(device_type="CPU"), valid], 1)
        self.assertEqual(gpu, {"names": ["native_gemm"], "intervals": [[0, 2., 1., 1]],
            "categories": {"gemm": {"events": 1, "duration_us": 1.}},
            "all_gpu_events_retained": True, "interval_unit": "microseconds",
            "interval_clock": "torch-profiler-relative", "duration_sum_is_wall_time": False})

    def test_actual_profiled_forward_keeps_rejections_in_incomplete_rank1_window(self):
        binding = dict(self.binding(), rank=1, arm="kv4", ordinal=13, cell="C1-3966")
        for reason, events in self.rejected_events():
            torch, runner, fb, log = forward_boundaries(gpu_events=events)
            windows = []
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "incomplete attribution"):
                bench.profiled_forward(runner, "batch", fb, torch, binding, [], windows)
            row = windows[0]
            self.assertFalse(row["complete"])
            self.assertIsNone(row["gpu"])
            self.assertTrue(row["observer_restored"])
            self.assertIsNone(sys.getprofile())
            self.assert_rejected_event(row["gpu_event_rejection"], reason)
            self.assertEqual(log.count("exit:gpu"), 1)
            self.assertEqual(log.count("forward"), 1)
            self.assertLess(len(json.dumps(row, allow_nan=False).encode()), bench.PROFILE_WINDOW_LIMIT)

    def test_unavailable_event_attributes_and_callback_errors_keep_bounded_metadata(self):
        class BrokenEvent:
            device_type, device_index = "CUDA", 1
            time_range = NS(start=0., end=1.)
            @property
            def name(self):
                raise RuntimeError("界" * 5000)
        cases = [NS(device_type="CUDA"), BrokenEvent(), NS(name="a", device_index=1)]
        for event in cases:
            with self.subTest(event=type(event).__name__), self.assertRaises(bench.ProfileGPUEventError) as raised:
                bench.profile_gpu_rows([event], 1)
            rejection = raised.exception.rejection
            self.assertTrue(any(reason.endswith("_unavailable") for reason in rejection["reasons"]))
            for value in rejection["observed"].values():
                if not value["available"]:
                    self.assertLessEqual(len(value["error"]["message_prefix"].encode("utf-8")), 256)
            self.assertLess(len(json.dumps(rejection, allow_nan=False).encode()), 8192)
        def unavailable_events():
            raise RuntimeError("events callback unavailable " + "界" * 5000)
        torch, runner, fb, log = forward_boundaries(gpu_events=unavailable_events)
        windows = []
        with self.assertRaises(ValueError):
            bench.profiled_forward(runner, "batch", fb, torch, dict(self.binding(), rank=1), [], windows)
        failure = windows[0]["gpu_collection_failure"]
        self.assertEqual((failure["stage"], failure["events_available"], failure["expected_device_index"]),
                         ("events", False, 1))
        self.assertEqual(failure["error"]["type"], "RuntimeError")
        self.assertLessEqual(len(failure["error"]["message_prefix"].encode("utf-8")), 256)
        self.assertFalse(windows[0]["complete"])
        self.assertTrue(windows[0]["observer_restored"])
        self.assertIsNone(sys.getprofile())
        json.dumps(windows, allow_nan=False)

    def test_rejection_metadata_does_not_retain_native_events_or_event_list(self):
        class NativeEvent:
            device_type, device_index, name = "CUDA", 1, "native_gemm"
            time_range = NS(start=float("nan"), end=1.)
        class NativeEvents(list):
            pass
        event = NativeEvent()
        events = NativeEvents([event])
        event_ref, events_ref = weakref.ref(event), weakref.ref(events)
        try:
            bench.profile_gpu_rows(events, 1)
        except bench.ProfileGPUEventError as error:
            rejection = error.rejection
        del event, events
        gc.collect()
        self.assertIsNone(event_ref())
        self.assertIsNone(events_ref())
        json.dumps(rejection, allow_nan=False)

    def test_original_forward_exception_survives_rejected_event_collection(self):
        original = RuntimeError("declared forward failure")
        reason, events = next(self.rejected_events())
        torch, runner, fb, log = forward_boundaries(failure=original, gpu_events=events)
        windows = []
        with self.assertRaises(RuntimeError) as raised:
            bench.profiled_forward(runner, "batch", fb, torch, dict(self.binding(), rank=1), [], windows)
        self.assertIs(raised.exception, original)
        self.assert_rejected_event(windows[0]["gpu_event_rejection"], reason)
        self.assertFalse(windows[0]["complete"])
        self.assertTrue(windows[0]["observer_restored"])
        self.assertIsNone(sys.getprofile())
        self.assertEqual(log.count("exit:gpu"), 1)

    def test_actual_rank1_kv4_cleanup_preserves_third_window_rejected_event(self):
        valid = NS(name="bf16_v2::read(...)", device_type="CUDA", device_index=1,
                   time_range=NS(start=0., end=1.))
        rejected = NS(name="bf16_v2::write(...)", device_type="CUDA", device_index=1,
                      time_range=NS(start=float("nan"), end=2.))
        modules, calls = rank_boundaries("kv4", rank=1,
            profile_event_factory=lambda window: [valid, rejected] if window == 3 else [valid])
        spec = base.fixture()
        spec["diagnostic_profile"] = dict(bench.PROFILE_CONFIG)
        identity = base.native_identity(1)
        identity["arm"] = "kv4"
        sources = profile_fixture_records("kv4")[1]["diagnostic_profile"]["sources"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(sys.modules, modules), patch.object(bench, "preflight", return_value=b"{}"), \
                    patch.object(bench, "preweight_identity", return_value=identity), \
                    patch.object(bench, "profile_sources", return_value=sources), \
                    patch.object(bench.os, "setsid"), patch.object(bench.os, "dup2"), \
                    patch.object(bench.resource, "setrlimit"), self.assertRaises(SystemExit) as exited:
                bench.run_rank(spec, "kv4", 1, output, True, "a" * 64, "b" * 64)
            result = bench.read_rank_result(output / "rank-1.json")
        self.assertEqual(exited.exception.code, 1)
        self.assertEqual(calls["destroy"], 2)
        self.assertEqual(calls["forward"], 417)
        self.assertEqual(calls["release"], 14)
        self.assertEqual(result["cleanup_errors"], [])
        self.assertEqual(result["completed_batches"], 13)
        self.assertEqual(result["intent_batches"], 14)
        profile = result["diagnostic_profile"]
        self.assertFalse(profile["complete"])
        self.assertEqual(len(profile["windows"]), 3)
        row = profile["windows"][-1]
        self.assertEqual((row["rank"], row["arm"], row["ordinal"], row["cell"], row["phase"]),
                         (1, "kv4", 13, "C1-3966", "prefill"))
        self.assertFalse(row["complete"])
        self.assertIsNone(row["gpu"])
        self.assertTrue(row["observer_restored"])
        self.assertEqual(row["gpu_event_rejection"]["cuda_event_index"], 1)
        self.assertIn("start_nonfinite", row["gpu_event_rejection"]["reasons"])
        self.assertIsNone(sys.getprofile())


if __name__ == "__main__":
    unittest.main()
