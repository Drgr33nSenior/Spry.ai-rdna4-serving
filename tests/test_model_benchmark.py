"""CPU contract tests; all hardware, imports and process effects are declared stubs."""
import copy
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("model_benchmark", ROOT / "tools/model_benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def fixture():
    args = dict(attention_backend="triton", prefill_attention_backend="triton",
        decode_attention_backend="triton", tp_size=2, pp_size=1, dp_size=1, nnodes=1,
        attn_cp_size=1, dcp_size=1, disaggregation_mode="null", context_length=4096,
        max_total_tokens=4096, max_running_requests=2, chunked_prefill_size=-1,
        mem_fraction_static=0.8, dtype="bfloat16", kv_cache_dtype="auto", page_size=1,
        disable_overlap_schedule=True, disable_custom_all_reduce=True,
        mamba_radix_cache_strategy="no_buffer", cpu_offload_gb=0,
        cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
        quantization="fp8", fp8_gemm_runner_backend="triton", load_format="safetensors",
        startup_weight_load_mode="serial", trust_remote_code=False, speculative_algorithm=None,
        random_seed=592137615,
        revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", model_path="/inert/model")
    prompts = {name: [list(range(10, 10 + length - 1)) + [50 + i]
                     for i in range(count)] for name, count, length in bench.CELLS}
    return {"schema": 1, "scope": bench.SCOPE, "experiment_mode": "model-benchmark-v1",
        "worker_seconds": 1200, "gpu_ids": [0, 1], "context_length": 4096,
        "physical_slots": 4096, "outputs": 32, "warmup_batches": 1, "measured_batches": 5,
        "cells": [{"name": name, "concurrency": count, "input_tokens": length}
                  for name, count, length in bench.CELLS], "frozen_prompt_ids": prompts,
        "frozen_prompt_sha256": {key: [bench.ids_sha256(ids) for ids in rows]
                                 for key, rows in prompts.items()}, "server_args": args,
        "config_path": "/inert/config.json", "package_root": "/inert/sglang",
        "stack_receipt": "/inert/stack.json", "stack_receipt_sha256": "a" * 64,
        "build_proof_sha256": "b" * 64, "nccl_port": 29677}


def completed_records():
    rows = []
    for rank in (0, 1):
        batches = []
        for i, plan in enumerate(bench.request_plan({})):
            batches.append({**plan, "ordinal": i, "complete": True,
                "generated_ids": [[7] * 32 for _ in plan["request_ids"]],
                "generated_logprobs": [[-1.0] * 32 for _ in plan["request_ids"]],
                "step_seconds": [0.01 + rank * 0.01] * 32, "batch_wall_seconds": 2.0})
        rows.append({"rank": rank, "arm": "bf16", "batches": batches, "failure": None,
            "cleanup_errors": [], "owner_cleanup": True, "intent_batches": 30,
            "completed_batches": 30, "intent_requests": 42, "completed_requests": 42})
    return rows


def native_identity(rank, spec_sha256="b" * 64):
    """Declared CPU hardware records; validators remain production bodies."""
    helper = bench.authenticated_adjacent("benchmark_gpu_identity.py", bench.GPU_IDENTITY_SHA256)
    devices = [{"index": index, "bdf": bdf, "uuid": uuid, "architecture": "gfx1201",
                "vram_bytes": 32 << 30} for index, (bdf, uuid) in enumerate(bench.PHYSICAL_DEVICES.items())]
    proof = {"library": str(helper.LIBRARY), "sha256": helper.LIBRARY_SHA256,
        "bytes": helper.LIBRARY_BYTES, "loaded_inode": 123, "loaded_device": 456,
        "no_load_handle": True, "symbol": "hipDeviceGetPCIBusId", "header_sha256": helper.HEADER_SHA256}
    return {"schema": 1, "scope": "benchmark-native-preweight-identity-v1", "rank": rank,
        "gpu_id": rank, "arm": "bf16", "session_sha256": "a" * 64, "spec_sha256": spec_sha256,
        "helper_sha256": bench.GPU_IDENTITY_SHA256, "devices": devices, "hip": proof,
        "process": {"pid": 42 + rank, "start_ticks": 1234 + rank, "uid": 1000, "gid": 1000,
            "boot_id": "12345678-1234-1234-1234-123456789abc", "pid_namespace": "pid:[123456]"},
        "before_model_config": True, "before_model_runner": True, "qualification": False}


def write_progress(output, rank, *, arm="bf16", spec_hash="b" * 64, pid=42,
                   batches=30, finish=True):
    with patch.object(bench.os, "getpid", return_value=pid):
        progress = bench.RankProgress(output, rank, arm, "a" * 64, spec_hash)
    try:
        for phase in bench.PROGRESS_PHASES[:bench.PROGRESS_PHASES.index("ready") + 1]:
            progress.record(phase)
        for ordinal, item in enumerate(list(bench.request_plan({}))[:batches]):
            progress.record("batch_intent", **item, ordinal=ordinal)
            progress.record("batch_complete", **item, ordinal=ordinal)
        if finish:
            progress.record("cleanup")
            progress.record("rank_finished", success=batches == 30)
    finally:
        progress.close()


class ModelBenchmarkTests(unittest.TestCase):
    def test_progress_counts_keep_partial_unknown_and_reject_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            kw = dict(rank=0, arm="bf16", session_sha256="a" * 64, spec_sha256="b" * 64)
            self.assertIsNone(bench.read_progress(output, **kw)["intent_requests"])
            write_progress(output, 0, batches=2, finish=False)
            raw = (output / "rank-0-progress.jsonl").read_bytes()
            good = bench.progress_readout(raw, **kw)
            self.assertTrue(good["valid"])
            self.assertFalse(good["counts_complete"])
            self.assertEqual((good["intent_batches"], good["completed_batches"]), (2, 2))
            torn = bench.progress_readout(raw + b'{"unfinished":', **kw)
            self.assertFalse(torn["valid"])
            self.assertEqual(torn["completed_batches"], 2)
            self.assertFalse(torn["counts_complete"])
            rows = [json.loads(line) for line in raw.splitlines()]
            for changed in (rows[:1] + rows[2:], rows[:1] + [rows[2], rows[1]] + rows[3:]):
                changed = copy.deepcopy(changed)
                for sequence, row in enumerate(changed):
                    row["sequence"] = sequence
                observed = bench.progress_readout(b"".join(json.dumps(row).encode() + b"\n"
                    for row in changed), **kw)
                self.assertFalse(observed["valid"])
                self.assertIn("startup order", observed["error"])
            for key, value in (("pid", 999), ("rank", 1), ("sequence", 19),
                               ("session_sha256", "e" * 64), ("monotonic", -1)):
                changed = copy.deepcopy(rows)
                changed[2][key] = value
                observed = bench.progress_readout(b"".join(json.dumps(row).encode() + b"\n"
                    for row in changed), **kw)
                self.assertFalse(observed["valid"], key)
                self.assertEqual(observed["intent_batches"], 0)
            with patch.object(bench, "PROGRESS_LIMIT", 1):
                self.assertFalse(bench.progress_readout(raw, **kw)["valid"])

    def test_killed_cpu_worker_preserves_log_and_durable_intent_through_real_transport(self):
        artifacts = bench.authenticated_adjacent("benchmark_artifacts.py",
            bench.hashlib.sha256((ROOT / "tools/benchmark_artifacts.py").read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            script = ("import os,runpy,signal,sys;from pathlib import Path;"
                "b=runpy.run_path(sys.argv[1]);p=Path(sys.argv[2]);"
                "w=b['RankProgress'](p,0,'bf16','a'*64,'b'*64);"
                "[w.record(p) for p in b['PROGRESS_PHASES'][:b['PROGRESS_PHASES'].index('ready')+1]];"
                "w.record('batch_intent',ordinal=0,**next(b['request_plan']({})));"
                "f=os.open(p/'rank-0.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);"
                "os.write(f,b'native failure marker\\x00\\xff');os.fsync(f);os.close(f);"
                "os.kill(os.getpid(),signal.SIGKILL)")
            child = subprocess.Popen([sys.executable, "-I", "-B", "-S", "-c", script,
                str(ROOT / "tools/model_benchmark.py"), str(output)], start_new_session=True)
            try:
                self.assertEqual(child.wait(timeout=10), -9)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
            with self.assertRaises(ProcessLookupError):
                os.killpg(child.pid, 0)
            binding = dict(arm="bf16", session_sha256="a" * 64, spec_sha256="b" * 64)
            stage = artifacts.stage_artifacts(output, binding)
            sink = io.BytesIO()
            artifacts.export_bundle(stage, binding, sink)
            receipt = artifacts.receive_bundle(sink.getvalue(), b"", output / "received", binding)
            self.assertFalse(receipt["all_members_present"])
            self.assertFalse(receipt["qualification"])
            self.assertEqual((output / "received/rank-0.log.tail").read_bytes(), b"native failure marker\x00\xff")
            progress = bench.read_progress(output / "received", rank=0, **binding)
            self.assertEqual((progress["intent_requests"], progress["completed_requests"]), (1, 0))
            self.assertFalse(progress["counts_complete"])
            self.assertIsNone(bench.read_progress(output / "received", rank=1, **binding)["intent_requests"])

    def test_physical_identity_source_is_byte_identical_inert_and_source_only(self):
        path = ROOT / "tools/benchmark_gpu_identity.py"
        self.assertEqual(bench.hashlib.sha256(path.read_bytes()).hexdigest(), bench.GPU_IDENTITY_SHA256)
        with patch.object(bench.importlib.util, "spec_from_file_location",
                          wraps=bench.importlib.util.spec_from_file_location) as selection:
            module = bench.authenticated_adjacent(path.name, bench.GPU_IDENTITY_SHA256)
        self.assertTrue(callable(module.physical_devices))
        self.assertEqual(selection.call_count, 1)
        with self.assertRaisesRegex(ValueError, "digest"):
            bench.authenticated_adjacent(path.name, "0" * 64)
        self.assertNotIn("torch", sys.modules)

    def test_preweight_identity_validates_devices_proof_and_exact_rank_binding(self):
        helper = bench.authenticated_adjacent("benchmark_gpu_identity.py", bench.GPU_IDENTITY_SHA256)
        expected = native_identity(0)
        for rank in (0, 1):
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(bench, "authenticated_adjacent", return_value=helper), \
                    patch.object(bench, "native_process_identity", return_value=native_identity(rank)["process"]), \
                    patch.object(helper, "physical_devices", return_value=(expected["devices"], expected["hip"])) as query:
                output = Path(directory)
                record = bench.preweight_identity(NS(), fixture(), "bf16", rank, output,
                    "a" * 64, "b" * 64, bench.time.monotonic() + 600)
                self.assertEqual(record, native_identity(rank))
                self.assertEqual(json.loads((output / f"rank-{rank}-identity.json").read_bytes()), record)
                bench.validate_rank_identity(record, rank=rank, arm="bf16", session_sha256="a" * 64,
                    spec_sha256="b" * 64)
                self.assertLessEqual(query.call_args.args[2], bench.time.monotonic() + 60)
                with self.assertRaises(FileExistsError):
                    bench.preweight_identity(NS(), fixture(), "bf16", rank, output, "a" * 64,
                        "b" * 64, bench.time.monotonic() + 600)
        for mutate in (lambda r: r["devices"][0].__setitem__("uuid", "wrong"),
                       lambda r: r["devices"][1].__setitem__("bdf", r["devices"][0]["bdf"]),
                       lambda r: r["hip"].__setitem__("sha256", "0" * 64),
                       lambda r: r.__setitem__("rank", 1),
                       lambda r: r.__setitem__("session_sha256", "0" * 64),
                       lambda r: r.__setitem__("helper_sha256", "0" * 64)):
            changed = copy.deepcopy(expected)
            mutate(changed)
            with self.assertRaises((RuntimeError, ValueError)):
                bench.validate_rank_identity(changed, rank=0, arm="bf16", session_sha256="a" * 64,
                    spec_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            bench.write_identity(Path(directory) / "too-large", {"value": "x" * (64 << 10)})

    def test_actual_rank_identity_gate_precedes_native_model_import_for_both_ranks(self):
        """Real entry/helper/validators; only Torch+HIP and later imports are CPU boundaries."""
        helper = bench.authenticated_adjacent("benchmark_gpu_identity.py", bench.GPU_IDENTITY_SHA256)
        real_import = builtins.__import__
        torch = ModuleType("torch")
        dist = ModuleType("torch.distributed")
        torch.distributed = dist
        for rank in (0, 1):
            for wrong in (False, True):
                with self.subTest(rank=rank, wrong=wrong), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    expected = native_identity(rank)
                    if wrong:
                        expected["devices"][0]["uuid"] = "wrong"
                    events = []
                    def imports(name, *args, **kwargs):
                        if name.startswith("sglang"):
                            events.append(name)
                            self.assertTrue((output / f"rank-{rank}-identity.json").is_file())
                            raise RuntimeError("declared CPU native import boundary")
                        return real_import(name, *args, **kwargs)
                    with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                            patch.object(builtins, "__import__", side_effect=imports), \
                            patch.object(bench, "authenticated_adjacent", return_value=helper), \
                            patch.object(bench, "native_process_identity", return_value=native_identity(rank)["process"]), \
                            patch.object(helper, "physical_devices", return_value=(expected["devices"], expected["hip"])), \
                            patch.object(bench, "preflight", return_value=b"{}"), \
                            patch.object(bench.os, "setsid"), patch.object(bench.os, "dup2"), \
                            patch.object(bench.resource, "setrlimit"), self.assertRaises(SystemExit) as exited:
                        bench.run_rank(fixture(), "bf16", rank, output, True, "a" * 64, "b" * 64)
                    self.assertEqual(exited.exception.code, 1)
                    record = json.loads((output / f"rank-{rank}.json").read_bytes())
                    self.assertFalse(record["qualification"])
                    self.assertEqual(record["batches"], [])
                    self.assertEqual(bool(events), not wrong)
                    self.assertEqual("native_identity" in record, not wrong)

    def test_native_process_binding_reads_actual_composition_with_declared_proc_fixture(self):
        expected = native_identity(0)["process"]
        fields = ["S"] + ["0"] * 18 + [str(expected["start_ticks"])]
        raw_stat = b"42 (worker ) with spaces) " + " ".join(fields).encode()
        def read(path, maximum):
            return raw_stat if str(path).endswith("/stat") else (expected["boot_id"] + "\n").encode()
        with patch.object(bench, "read_bounded_file", side_effect=read), \
                patch.object(bench.os, "getpid", return_value=42), \
                patch.object(bench.os, "getuid", return_value=1000), \
                patch.object(bench.os, "getgid", return_value=1000), \
                patch.object(bench.os, "readlink", return_value=expected["pid_namespace"]):
            self.assertEqual(bench.native_process_identity(), expected)
        for key, value in (("uid", 0), ("gid", 0), ("pid", True), ("start_ticks", 0),
                           ("boot_id", "unknown"), ("pid_namespace", "pid:[0]")):
            changed = {**expected, key: value}
            with self.assertRaises(ValueError):
                bench.validate_native_process(changed)

    def test_export_staging_preserves_raw_partial_files_and_refuses_links_or_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            (output / "rank-0.json").write_bytes(b"{malformed original")
            (output / "rank-0.log").write_bytes(b"retained not exported")
            artifacts = bench.authenticated_adjacent("benchmark_artifacts.py",
                bench.hashlib.sha256((ROOT / "tools/benchmark_artifacts.py").read_bytes()).hexdigest())
            binding = dict(arm="bf16", session_sha256="a" * 64, spec_sha256="b" * 64)
            staged = artifacts.stage_artifacts(output, binding)
            self.assertEqual({p.name for p in staged.iterdir()},
                             {"rank-0.json", "rank-0.log.tail", "diagnostics.json"})
            self.assertEqual((staged / "rank-0.json").read_bytes(), b"{malformed original")
            self.assertEqual((staged / "rank-0.log.tail").read_bytes(), b"retained not exported")
            self.assertTrue((output / "rank-0.log").is_file())
            with self.assertRaises(FileExistsError):
                artifacts.stage_artifacts(output, binding)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            (output / "rank-1.json").symlink_to(output / "missing")
            with self.assertRaises(OSError):
                artifacts.stage_artifacts(output, binding)

    def test_physical_cli_requires_all_independent_bindings_before_any_child(self):
        base = ["model_benchmark.py", "--execute-reviewed-benchmark", "--arm", "bf16"]
        for tail in (["--physical-artifacts"], ["--session-sha256", "a" * 64],
                     ["--artifacts-sha256", "a" * 64],
                     ["--physical-artifacts", "--session-sha256", "a" * 64,
                      "--artifacts-sha256", "0" * 64]):
            with self.subTest(tail=tail), patch.object(sys, "argv", base + tail), \
                    patch.object(bench, "execute_arm") as child, self.assertRaises(ValueError):
                bench.main()
            child.assert_not_called()

    def test_actual_physical_main_exports_success_and_malformed_failure_only_after_reap(self):
        bench_import = bench.authenticated_adjacent
        artifacts = bench.authenticated_adjacent("benchmark_artifacts.py",
            bench.hashlib.sha256((ROOT / "tools/benchmark_artifacts.py").read_bytes()).hexdigest())
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                raw = json.dumps(fixture()).encode()
                spec_hash = bench.hashlib.sha256(raw).hexdigest()
                (root / "spec.json").write_bytes(raw)
                output = root / "output"
                children = []
                class Child:
                    def __init__(self, target, args):
                        self.rank, self.output = args[2], args[3]
                        self.pid, self.exitcode, self.joins = 900001 + self.rank, 0, []
                        children.append(self)
                    def start(self):
                        record = completed_records()[self.rank]
                        identity = native_identity(self.rank, spec_hash)
                        record["native_identity"] = identity
                        (self.output / f"rank-{self.rank}.json").write_bytes(b"{" if malformed and self.rank == 0
                            else json.dumps(record).encode())
                        (self.output / f"rank-{self.rank}-identity.json").write_bytes(json.dumps(identity).encode())
                        (self.output / f"rank-{self.rank}.log").write_bytes(b"private retained log")
                        write_progress(self.output, self.rank, spec_hash=spec_hash, pid=self.pid)
                    def join(self, seconds):
                        self.joins.append(seconds)
                    def is_alive(self):
                        return False
                sink = io.BytesIO()
                real_export = artifacts.export_bundle
                def export(path, binding, sink):
                    self.assertTrue(all(child.joins[-1] == 0 for child in children))
                    self.assertEqual(path, output / "export")
                    return real_export(path, binding, sink)
                digest = bench.hashlib.sha256((ROOT / "tools/benchmark_artifacts.py").read_bytes()).hexdigest()
                argv = ["model_benchmark.py", "--execute-reviewed-benchmark", "--spec", str(root / "spec.json"),
                    "--spec-sha256", spec_hash, "--output", str(output), "--arm", "bf16", "--physical-artifacts",
                    "--session-sha256", "a" * 64, "--artifacts-sha256", digest]
                with patch.object(sys, "argv", argv), patch.object(sys, "stdout", NS(buffer=sink)), \
                        patch.object(bench, "preflight", return_value=b"{}"), \
                        patch.object(bench.multiprocessing, "get_context", return_value=NS(Process=Child)), \
                        patch.object(bench.os, "killpg", side_effect=ProcessLookupError), \
                        patch.object(artifacts, "export_bundle", side_effect=export), \
                        patch.object(bench, "authenticated_adjacent", side_effect=lambda name, pin:
                            artifacts if name == "benchmark_artifacts.py" else
                            bench_import(name, pin)), self.assertRaises(SystemExit) as exited:
                    bench.main()
                self.assertEqual(exited.exception.code, int(malformed))
                bundle = json.loads(sink.getvalue().split(b" ", 1)[1])
                self.assertEqual(set(bundle["members"]), set(artifacts.MEMBER_LIMITS))
                self.assertEqual(bundle["binding"]["spec_sha256"], spec_hash)
                result = json.loads((output / "result.json").read_bytes())
                self.assertFalse(result["qualification"])
                self.assertEqual(result["native_identity_verified"], not malformed)
                if malformed:
                    self.assertEqual((output / "export/rank-0.json").read_bytes(), b"{")

    def test_repository_protocol_matches_actual_plan_counts_and_unfunded_envelope(self):
        profile = json.loads((ROOT / "profiles/benchmark-fp8-kv4-modelrunner.json").read_bytes())
        self.assertEqual(profile["status"], "PREPARATION_ONLY_NATIVE_UNRUN")
        self.assertIs(profile["launch_authorized"], False)
        self.assertEqual(profile["settings"]["random_seed"], 592137615)
        self.assertEqual(profile["cells"], fixture()["cells"])
        plans = list(bench.request_plan({}))
        observed = {"batches": len(plans), "measured_batches": sum(row["measured"] for row in plans),
            "requests": sum(row["concurrency"] for row in plans),
            "measured_requests": sum(row["concurrency"] for row in plans if row["measured"]),
            "input_tokens": sum(row["concurrency"] * row["input_tokens"] for row in plans),
            "output_tokens": sum(row["concurrency"] * 32 for row in plans),
            "prefill_forwards_per_rank": len(plans), "decode_forwards_per_rank": len(plans) * 31}
        self.assertEqual(observed, profile["per_arm_counts"])
        self.assertEqual(observed, {"batches": 30, "measured_batches": 25, "requests": 42,
            "measured_requests": 35, "input_tokens": 44532, "output_tokens": 1344,
            "prefill_forwards_per_rank": 30, "decode_forwards_per_rank": 930})
        envelope = profile["proposed_envelope"]
        total = envelope["admission"] + 2 * (envelope["load_and_warmup_per_arm"] +
            envelope["measurements_per_arm"] + envelope["owned_cleanup_per_arm"]) + \
            envelope["collection_and_independent_release"] + envelope["sealing"]
        self.assertEqual(total, 120 + 2 * (240 + 450 + 60) + 180 + 300)
        self.assertEqual(total, profile["proposed_complete_envelope_seconds"])
        self.assertEqual(total, 2100)

    def main_case(self, payloads):
        """Execute real main with only process/transport effects substituted."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            raw = json.dumps(fixture()).encode()
            spec_path.write_bytes(raw)
            output = root / "output"
            class Child:
                def __init__(self, target, args):
                    self.rank, self.output = args[2], args[3]
                    self.pid, self.exitcode = 900001 + self.rank, 0
                def start(self):
                    payload = payloads[self.rank]
                    if payload is not None:
                        (self.output / f"rank-{self.rank}.json").write_bytes(
                            payload if isinstance(payload, bytes) else json.dumps(payload).encode())
                def join(self, seconds):
                    pass
                def is_alive(self):
                    return False
            argv = ["model_benchmark.py", "--execute-reviewed-benchmark", "--spec", str(spec_path),
                    "--spec-sha256", bench.hashlib.sha256(raw).hexdigest(), "--output", str(output),
                    "--arm", "bf16"]
            with patch.object(sys, "argv", argv), patch.object(bench, "preflight", return_value=b"{}"), \
                    patch.object(bench.multiprocessing, "get_context", return_value=NS(Process=Child)), \
                    patch.object(bench.os, "killpg", side_effect=ProcessLookupError) as absence, \
                    self.assertRaises(SystemExit) as exited:
                bench.main()
            self.assertEqual(absence.call_count, 2)
            return exited.exception.code, json.loads((output / "result.json").read_bytes())

    def test_actual_main_refuses_malformed_truncated_rank_and_still_publishes_release(self):
        originals = completed_records()
        code, result = self.main_case(originals)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "MODEL_BENCHMARK_COMPLETE")
        cases = [b"{", b"x" * ((2 << 20) + 1), b'{"bad":NaN}', None]
        for mutation in (lambda row: row["batches"].pop(),
                         lambda row: row.__setitem__("rank", 1),
                         lambda row: row.__setitem__("arm", "kv4")):
            changed = copy.deepcopy(originals[0])
            mutation(changed)
            cases.append(changed)
        for payload in cases:
            with self.subTest(kind=type(payload).__name__):
                code, result = self.main_case([payload, originals[1]])
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "MODEL_BENCHMARK_FAILED")
                self.assertEqual(result["groups_observed_absent"], [True, True])
                self.assertTrue(result["parent_error"])
                self.assertFalse(result["qualification"])
                self.assertLess(len(json.dumps(result)), 8 << 20)
                if not isinstance(payload, dict):
                    self.assertEqual(len(result["rank_parse_errors"]), 1)
                    self.assertLessEqual(len(result["rank_parse_errors"][0]["message"]), 1024)

    def test_actual_rank_composition_32_linked_outputs_both_arms_with_cpu_stubs(self):
        for arm, write_failure in (("bf16", False), ("kv4", False), ("bf16", True)):
            calls = {"forward": 0, "decode": 0, "destroy": 0, "release": 0}
            class Scalar:
                def item(self):
                    return -1.0
            class Values:
                def __init__(self, count):
                    self.shape = (count, 8192)
                def float(self):
                    return self
                def log_softmax(self, **kwargs):
                    return self
                def __getitem__(self, index):
                    return Scalar()
            class Tokens(list):
                def tolist(self):
                    return list(self)
                def to(self, *args, **kwargs):
                    return self
            class Guard:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
            owner = NS(refuse_reset=lambda: None)
            class Runner:
                def __init__(self, **kwargs):
                    self._uq_bf16_request = kwargs["_uq_bf16_request"]
                    pool_name = "QwenBF16MixedPool" if arm == "kv4" else "MHATokenToKVPool"
                    pool_fields = {"owner": owner} if arm == "kv4" else {"dtype": "bf16"}
                    self.token_to_kv_pool = NS(full_kv_pool=type(pool_name, (), pool_fields)())
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
                    return Tokens([7] * fb.batch_size)
            class Req:
                def __init__(self, **kwargs):
                    self.origin_input_ids = kwargs["origin_input_ids"]
                    self.full_untruncated_fill_ids = self.origin_input_ids
                    self.output_ids = []
                    self.kv = NS(holds_kv=True, holds_mamba=True)
                    self.prefix_indices = []
                    self.last_node = None
                def init_next_round_input(self, cache):
                    pass
                def set_extend_range(self, *args):
                    pass
            class Batch:
                @staticmethod
                def init_new(**kwargs):
                    return Batch(kwargs["reqs"])
                def __init__(self, reqs):
                    self.reqs = reqs
                    self.input_ids = Tokens([0] * len(reqs))
                    self.prefill_input_ids_cpu = None
                def prepare_for_extend(self):
                    pass
                def prepare_for_decode(self):
                    calls["decode"] += 1
                    self_test.assertEqual(list(self.input_ids), [req.output_ids[-1] for req in self.reqs])
            class FB:
                @staticmethod
                def init_new(batch, runner, **kwargs):
                    return NS(batch_size=len(batch.reqs))
            class Cache:
                def __init__(self, args):
                    pass
                def inc_lock_ref(self, node):
                    return NS(to_dec_params=lambda: None)
                def reset(self):
                    pass
                release_for_idle_bf16_flush = reset
            def release(req, cache, is_insert):
                self.assertFalse(is_insert)
                calls["release"] += 1
                req.kv.holds_kv = req.kv.holds_mamba = False
            def destroy():
                calls["destroy"] += 1
            torch = ModuleType("torch")
            torch.bfloat16, torch.int64 = "bf16", "int64"
            torch.set_num_threads = torch.set_num_interop_threads = lambda x: None
            torch.cuda = NS(synchronize=lambda: None)
            torch.no_grad = Guard
            torch.isfinite = lambda x: NS(all=lambda: NS(item=lambda: True))
            dist = ModuleType("torch.distributed")
            dist.all_gather_object = lambda peers, value, **kwargs: peers.__setitem__(slice(None), [value, value])
            torch.distributed = dist
            modules = {"torch": torch, "torch.distributed": dist}
            entries = {
                "sglang.srt.configs.model_config": {"ModelConfig": NS(from_server_args=lambda args: NS(vocab_size=8192))},
                "sglang.srt.distributed": {"bootstrap": NS(init_parallel_runtime=lambda **kw: None,
                                                           init_layer_runtime=lambda **kw: None)},
                "sglang.srt.distributed.parallel_state": {"destroy_distributed_environment": destroy,
                    "destroy_model_parallel": destroy, "get_tp_group": lambda: NS(cpu_group="inert")},
                "sglang.srt.entrypoints.engine": {"_set_envs_and_config": lambda x: None},
                "sglang.srt.managers.schedule_batch": {"Req": Req, "ScheduleBatch": Batch},
                "sglang.srt.mem_cache.cache_init_params": {"CacheInitParams": lambda **kw: NS(**kw)},
                "sglang.srt.mem_cache.common": {"release_kv_cache": release},
                "sglang.srt.mem_cache.unified_radix_cache": {"UnifiedRadixCache": Cache},
                "sglang.srt.mem_cache.unified_cache.component_type": {"ComponentType": NS(FULL="full", MAMBA="mamba")},
                "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": FB},
                "sglang.srt.model_executor.model_runner": {"ModelRunner": Runner},
                "sglang.srt.runtime_context": {"SpawnRanks": lambda **kw: NS(**kw),
                    "get_device": lambda: NS(device="inert"), "get_model": lambda: NS(is_startup_weight_load_overlap=False),
                    "publish": lambda *a, **kw: None, "spawn_world_rank": lambda *a, **kw: kw["tp_rank"]},
                "sglang.srt.sampling.sampling_params": {"SamplingParams": lambda **kw: NS(**kw)},
                "sglang.srt.server_args": {"ServerArgs": lambda **kw: NS(**kw, resolve_once=lambda: None)},
                "sglang.srt.speculative.spec_info": {"SpeculativeAlgorithm": NS(NONE="none")},
                "sglang.srt.layers.attention.ultraquant_contract": {"bf16_source_request": lambda *a, **kw: {"proof": True}},
                "sglang.srt.layers.moe": {"initialize_moe_config": lambda: None},
                "sglang.srt.layers.quantization.fp4_utils": {"initialize_fp4_gemm_config": lambda: None},
                "sglang.srt.layers.quantization.fp8_utils": {"initialize_fp8_gemm_config": lambda: None},
                "sglang.kernels.ops.mamba.triton_ops": {"initialize_mamba_selective_state_update_backend": lambda args: None},
            }
            for name, fields in entries.items():
                module = ModuleType(name)
                module.__dict__.update(fields)
                modules[name] = module
            self_test = self
            original_record = bench.RankProgress.record
            def record(progress, phase, **details):
                if write_failure and phase == "batch_complete":
                    raise OSError("declared diagnostic write failure after release")
                return original_record(progress, phase, **details)
            with tempfile.TemporaryDirectory() as directory:
                with patch.dict(sys.modules, modules), patch.object(bench, "preflight", return_value=b"{}"), \
                        patch.object(bench.os, "setsid"), patch.object(bench.os, "dup2"), \
                        patch.object(bench.resource, "setrlimit"), \
                        patch.object(bench.RankProgress, "record", record), self.assertRaises(SystemExit) as exited:
                    bench.run_rank(fixture(), arm, 0, Path(directory))
                observed = bench.read_progress(Path(directory), rank=0, arm=arm,
                    session_sha256=None, spec_sha256=None)
                self.assertTrue(observed["valid"], observed)
                self.assertEqual(observed["finished"], not write_failure)
                self.assertEqual(observed["intent_requests"], 1 if write_failure else 42)
                self.assertEqual(observed["completed_batches"], 0 if write_failure else 30)
                if write_failure:
                    result = json.loads((Path(directory) / "rank-0.json").read_bytes())
                    self.assertEqual(result["cleanup_errors"], [])
                    self.assertIn("diagnostic write failure", result["failure"]["message"])
            self.assertEqual(exited.exception.code, int(write_failure))
            expected = {"forward": 32, "decode": 31, "destroy": 2, "release": 1} if write_failure else \
                {"forward": 960, "decode": 930, "destroy": 2, "release": 42}
            self.assertEqual(calls, expected)

    def test_isolated_cli_help_and_refusal_do_not_import_runtime(self):
        for extra, code in ((["--help"], 0), ([], 1)):
            child = subprocess.run([sys.executable, "-I", "-B", "-S",
                                    str(ROOT / "tools/model_benchmark.py"), *extra],
                                   capture_output=True, text=True, timeout=10,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(child.returncode, code, child.stderr)
            if not extra:
                self.assertIn("NOT_RUN", child.stderr)

    def test_exact_schedule_counts_and_prompt_binding(self):
        spec = fixture()
        self.assertIs(bench.validate_spec(spec), spec)
        plans = list(bench.request_plan(spec))
        self.assertEqual(len(plans), 30)
        self.assertEqual(sum(row["concurrency"] for row in plans), 42)
        self.assertEqual(sum(row["measured"] for row in plans), 25)
        self.assertEqual(sum(row["concurrency"] * 32 for row in plans), 1344)
        self.assertEqual([row["cell"] for row in plans[::6]], [row[0] for row in bench.CELLS])
        for key, bad in (("outputs", 31), ("warmup_batches", 2), ("measured_batches", 6),
                         ("worker_seconds", 1201), ("gpu_ids", [0]), ("scope", "public")):
            changed = copy.deepcopy(spec)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_spec(changed)

    def test_frozen_prompt_and_native_settings_refusals(self):
        spec = fixture()
        for edit in (lambda s: s["frozen_prompt_ids"]["C1-128"][0].__setitem__(0, 9),
                     lambda s: s["frozen_prompt_ids"]["C2-128"].pop(),
                     lambda s: s["server_args"].__setitem__("quantization", "mxfp4"),
                     lambda s: s["server_args"].__setitem__("page_size", 16),
                     lambda s: s["server_args"].__setitem__("chunked_prefill_size", 2048),
                     lambda s: s["server_args"].__setitem__("random_seed", 592137616)):
            changed = copy.deepcopy(spec)
            edit(changed)
            with self.assertRaises(ValueError):
                bench.validate_spec(changed)

    def test_both_arms_require_original_build_composition_gate(self):
        proof = {"schema": 2, "serving_composition_complete": True,
            "qualification": "build-reviewed-not-serving-qualified",
            "stack_receipt_sha256": "a" * 64, "binary_sha256": bench.BINARY_SHA256}
        def read(path, digest, *args):
            return json.dumps(proof).encode() if str(path).endswith("build.json") else b"{}"
        with patch.object(bench, "checked_file", side_effect=read) as reads:
            self.assertEqual(bench.preflight(fixture()), b"{}")
            self.assertEqual(reads.call_count, 3)
            proof["serving_composition_complete"] = False
            with self.assertRaises(ValueError):
                bench.preflight(fixture())

    def test_arm_pool_and_backend_identity(self):
        pool = type("MHATokenToKVPool", (), {"dtype": "bf16"})()
        backend = type("TritonAttnBackend", (), {})()
        runner = NS(_uq_bf16_request=None, token_to_kv_pool=NS(full_kv_pool=pool),
                    attn_backend=NS(full_attn_backend=backend))
        self.assertIsNone(bench.assert_pool_arm(runner, "bf16", NS(bfloat16="bf16")))
        pool.owner = object()
        with self.assertRaises(ValueError):
            bench.assert_pool_arm(runner, "bf16", NS(bfloat16="bf16"))
        runner.token_to_kv_pool.full_kv_pool = type("QwenBF16MixedPool", (), {"owner": object()})()
        runner.attn_backend.full_attn_backend = type("QwenBF16SourceAdapter", (), {})()
        runner._uq_bf16_request = {"authenticated": True}
        self.assertIsNotNone(bench.assert_pool_arm(runner, "kv4", NS()))
        runner._uq_bf16_request = None
        with self.assertRaises(ValueError):
            bench.assert_pool_arm(runner, "kv4", NS())

    def test_batch_release_uses_actual_arm_branch_and_refuses_capacity_drift(self):
        events = []
        cache = NS(reset=lambda: events.append("reset"),
            release_for_idle_bf16_flush=lambda: events.append("owner-release"))
        owner = NS(refuse_reset=lambda: events.append("owner-check"))
        request = NS(kv=NS(holds_kv=True, holds_mamba=True))
        release = lambda req, tree, is_insert: events.append(("release", is_insert))
        bench.finish_batch(cache, None, "bf16", [request], release, lambda: [4096, 2, 68], [4096, 2, 68])
        self.assertEqual(events, [("release", False), "reset"])
        events.clear()
        bench.finish_batch(cache, owner, "kv4", [request], release, lambda: [4096, 2, 68], [4096, 2, 68])
        self.assertEqual(events, [("release", False), "owner-release", "reset", "owner-check"])
        with self.assertRaisesRegex(ValueError, "capacities"):
            bench.finish_batch(cache, None, "bf16", [], release, lambda: [4095, 2, 68], [4096, 2, 68])

    def test_timed_region_contains_only_declared_model_work_and_syncs(self):
        events = []
        class Guard:
            def __enter__(self):
                events.append("no-grad-enter")
            def __exit__(self, *args):
                events.append("no-grad-exit")
        torch = NS(cuda=NS(synchronize=lambda: events.append("sync")), no_grad=Guard)
        forward_batch = NS(init_new=lambda *a, **kw: events.append("ForwardBatch") or "fb")
        runner = NS(forward=lambda fb: events.append("forward") or NS(logits_output="logits"),
                    sample=lambda *args: events.append("sample") or "sampled")
        ticks = iter([10.0, 10.25])
        got = bench.timed_forward(runner, "batch", forward_batch, torch, lambda: next(ticks))
        self.assertEqual(got, ("logits", "sampled", 0.25, "fb"))
        self.assertEqual(events, ["sync", "no-grad-enter", "ForwardBatch", "forward", "sample", "no-grad-exit", "sync"])

    def test_rank_max_summary_and_partial_refusals(self):
        records = completed_records()
        samples = bench.summarize(records, "bf16")
        self.assertEqual(len(samples), 30)
        self.assertAlmostEqual(samples[0]["prefill_rank_max_seconds"], .02)
        self.assertAlmostEqual(samples[0]["decode_31_seconds"], .62)
        for mutate in (lambda r: r[0].__setitem__("completed_requests", 41),
                       lambda r: r[0]["batches"][0].__setitem__("complete", False),
                       lambda r: r[0]["batches"][0]["generated_ids"][0].__setitem__(0, 8),
                       lambda r: r[1]["batches"][0]["step_seconds"].__setitem__(0, float("nan")),
                       lambda r: r[0].__setitem__("cleanup_errors", ["survivor"]),
                       lambda r: r[0]["batches"].pop(),
                       lambda r: r[0]["batches"].append(copy.deepcopy(r[0]["batches"][0])),
                       lambda r: r[1].__setitem__("rank", 0),
                       lambda r: r[0].__setitem__("arm", "kv4")):
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                bench.summarize(changed, "bf16")

    def test_output_exclusive_bound_and_default_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            bench.write_result(path, {"qualification": False})
            with self.assertRaises(FileExistsError):
                bench.write_result(path, {})
            with self.assertRaises(ValueError):
                bench.write_result(Path(directory) / "too-large", "x" * (8 << 20))
        with patch.object(sys, "argv", ["model_benchmark.py"]), self.assertRaisesRegex(SystemExit, "NOT_RUN"):
            bench.main()
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
