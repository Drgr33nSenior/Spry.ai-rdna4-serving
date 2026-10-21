"""Validate the historical FP8 artifacts without a native runtime or network."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "profiles/fp8-bf16-kv-4k-qualified.json"
SUMMARY = "provenance/FP8-BASELINE-SUMMARY.json"
INVENTORY = "provenance/FP8-MODEL-INVENTORY.json"
INVENTORY_SHA256 = "892e95cdabf87bbf96298e258e5889e2572acf21659121d811ca9a09eac09986"
IMAGE = "localhost/workstation/sglang@sha256:f2f0d8b134b7650b4ead961fac6c23751e11d98757fe3f7233be11c817948945"
IMAGE_CONFIG = "f10ff719d78e659dff0e5c02ecae61b3aad45a8d06d296210614417d344bfa06"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MODEL_CONFIG = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
VERSIONS = {
    "python": "3.14.4", "sglang": "0.5.21", "sglang-kernel": "0.4.7",
    "torch": "2.13.0+rocm10.1.0", "rocm_sdk": "10.1.0", "hip": "7.16.26385",
    "triton": "3.8.0+git669b31ac.rocm10.1.0", "rccl": "2.30.7", "psutil": "7.2.2",
}
NATIVE_PINS = {
    "srt/server_args.py": "1dc6d150bf7c4ef5c2316c2ce21ca28ea557ac26fcdb150d21ac48fff1a784e6",
    "srt/configs/model_config.py": "be3a0168bdd3b4a6b1f7778420b83f95c315535812b44e9422e1b48dd856a11e",
    "srt/layers/quantization/__init__.py": "13a9f194ce7693840529647e818fd784cbd221b7120ceafa98a986762a85512c",
    "srt/layers/quantization/fp8.py": "89d769cd5c3702c65f9d7839115e032833ec6c0d26b024e76ae5075219bc8a15",
    "srt/layers/attention/linear/gdn_backend.py": "c12e054cbe9f85cf15b0590a8ea7e3e175f62121160c3f548e0aa7ce71d1af0d",
    "srt/model_executor/model_runner.py": "36e7a69f345cf09f24699eea36a4ab0a2a305acb8b8a7e9b4e975f3fd09527f3",
    "srt/managers/scheduler.py": "b61df5b750adc748a99ceb381b8494c199fb3f704d793da5a0217bd6bbb606c4",
}
SETTINGS = {
    "tp": 2, "context": 4096, "dtype": "bfloat16", "kv_storage": "BF16",
    "kv_cache_dtype_argument": "auto", "page_size": 1, "mem_fraction_static": 0.8,
    "max_running_requests": 2, "attention_backend": "triton", "linear_attn_backend": "triton",
    "mamba_cache_policy": "no_buffer", "disable_overlap_schedule": True,
    "disable_custom_all_reduce": True, "disable_cuda_graph": False,
    "chunked_prefill_size_resolved": 4096, "parsed_graph_max_batch_size": 48,
    "parsed_graph_batches": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48],
    "observed_capture_batches_per_rank": [[1, 2], [1, 2]], "random_seed": 592137615,
    "cpu_offload_gb": 0, "speculative_decoding": False, "aiter_serving": False,
    "arithmetic_overlay": False, "kv4_overlay": False, "capture_hooks": False,
}
ENVIRONMENT = {
    "HOME": "/tmp", "LC_ALL": "C", "SGLANG_USE_AITER": "0", "SGLANG_USE_AITER_AR": "0",
    "SGLANG_ROCM_FUSED_DECODE_MLA": "0", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MAX_JOBS": "4", "NCCL_DEBUG": "INFO",
    "ROCM_HOME": "/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel",
    "ROCM_PATH": "/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel",
    "HIP_PATH": "/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel",
    "PATH": "/opt/workstation/sglang/bin:/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel/bin:/usr/local/bin:/usr/bin:/bin",
    "SGLANG_JIT_CACHE_DIR": "/cache/sglang-jit", "TRITON_CACHE_DIR": "/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor", "XDG_CACHE_HOME": "/cache/xdg",
    "HF_HOME": "/cache/huggingface",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_profile(profile):
    require(profile["schema"] == 1 and profile["name"] == "fp8-bf16-kv-4k-qualified", "profile mode")
    runtime, model = profile["runtime"], profile["model"]
    require(runtime["image"] == IMAGE and runtime["image_config_sha256"] == IMAGE_CONFIG, "image identity")
    require(runtime["versions"] == VERSIONS, "exact version tuple")
    require(runtime["native_source_sha256"] == NATIVE_PINS, "native source identities")
    require(runtime["sglang_revision"] == "e00930c5489053f26d86b179cee0d087f846acbb", "source revision")
    require(runtime["prefix"] == "/opt/workstation/sglang" and
            runtime["interpreter"] == "/opt/workstation/sglang/bin/python3", "private runtime selection")
    require(runtime["image_published"] is False and runtime["image_pull_policy"] == "Never", "image availability")
    require(model["repository"] == "Qwen/Qwen3.8-27B-FP8" and
            model["revision"] == MODEL_REVISION and model["config_sha256"] == MODEL_CONFIG, "model identity")
    require(model["files"] == 81 and model["payload_bytes"] == 30890049597 and
            model["inventory_sha256"] == INVENTORY_SHA256 and model["inventory_file"] == INVENTORY, "inventory binding")
    require(model["weight_conversion"] is False, "no format conversion")
    require(profile["settings"] == SETTINGS, "historical native settings, not eager or KV4")
    require(profile["historical_environment"] == ENVIRONMENT, "historical runtime environment")
    require(profile["cache_policy"] == {
        "private_tmp": "/tmp", "private_cache": "/cache", "fresh_private_caches": True,
        "accepted_cache_reused": False, "accepted_cache_deleted": False,
        "lazy_compilation_inside_request_possible": True,
        "timing_scope": "Complete request processing; not pure warm-kernel time",
    }, "private cache and compilation scope")
    require(profile["authorization"] == {
        "launch_authorized": False, "public_deployment_authorized": False,
        "source_promotion": False, "default_changed": False,
    }, "no execution or promotion authority")


def validate_summary(summary):
    require(summary["schema"] == 1 and summary["status"] == "QUALIFIED C1 AND C2" and
            summary["scope"] == "Historical bounded 4K TP2 FP8 implementation compatibility", "bounded historical scope")
    require(summary["profile"] == PROFILE and summary["model_inventory"] == INVENTORY and
            summary["model_inventory_sha256"] == INVENTORY_SHA256, "artifact references")
    evidence = summary["evidence"]
    require(evidence["manifest_sha256"] == "98274e5d1cb0948b01490c9255485936f83f3c5ae4de6742e3a9d60e7a7dc328" and
            evidence["members"] == 375 and evidence["raw_evidence_distributed"] is False, "donor seal")
    require(summary["claims"] == {
        "qualification": False, "production_qualification": False, "source_promotion": False,
        "launch_authorized": False, "public_deployment_authorized": False, "default_changed": False,
        "context_32768_qualified": False, "bf16_fidelity_established": False,
        "mxfp4_ranking_established": False, "performance_gain_established": False,
        "kv4_model_quality_qualified": False,
    }, "no unsupported qualification, performance or deployment claims")
    observed = summary["observations"]
    require(observed["strict_repeats_exact"] is True and observed["named_512_repeat_exact"] is True and
            observed["teacher_total_scored_positions"] == 512 and observed["teacher_c1_c2_exact"] is True and
            observed["teacher_c2_per_cell_ratios"] == [1.0] * 4 and
            observed["teacher_c2_overall_ratio"] == 1.0, "retained numerical results")
    require(observed["external_generations"] == 56 and observed["normal_terminal_generations"] == 55 and
            observed["expected_cancelled_generations"] == 1 and observed["flush_controls"] == 12 and
            observed["abort_controls"] == 1 and observed["health_controls"] == 1 and
            observed["durable_intents"] == 70, "durable request counts")
    require(observed["code_tasks_status"] == "UNGRADED" and observed["code_tasks_per_phase"] == 2 and
            observed["gated_functional_tasks_per_phase"] == 8 and
            observed["actual_decode_batch2_observations"] == 3 and
            observed["teacher_forward_batching"] == "singleton under offered C2; not batched-scoring invariance", "task and batching limits")
    require(observed["cleanup_verified"] is True and observed["exact_uid_disposal_verified"] is True and
            observed["independent_process_pod_device_allocation_lock_release_verified"] is True and
            observed["graceful_distributed_rank_destruction_independently_qualified"] is False, "cleanup scope")


def validate_inventory(raw):
    require(hashlib.sha256(raw).hexdigest() == INVENTORY_SHA256, "exact retained inventory bytes")
    records = json.loads(raw)
    require(isinstance(records, list) and len(records) == 81, "81 payload files")
    names = []
    for record in records:
        require(set(record) == {"name", "size", "sha256"}, "inventory schema")
        name = record["name"]
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name, "safe relative path")
        require(type(record["size"]) is int and record["size"] >= 0, "payload size")
        require(len(record["sha256"]) == 64 and all(c in "0123456789abcdef" for c in record["sha256"]), "payload hash")
        names.append(name)
    require(names == sorted(set(names)) and sum(row["size"] for row in records) == 30890049597, "exact inventory extent")
    require(next(row["sha256"] for row in records if row["name"] == "config.json") == MODEL_CONFIG, "config identity")


class BaselineArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile_raw = (ROOT / PROFILE).read_bytes()
        cls.profile = json.loads(cls.profile_raw)
        cls.summary = json.loads((ROOT / SUMMARY).read_bytes())
        cls.inventory_raw = (ROOT / INVENTORY).read_bytes()

    def test_retained_artifacts(self):
        validate_profile(self.profile)
        validate_summary(self.summary)
        validate_inventory(self.inventory_raw)
        self.assertEqual(self.summary["profile_sha256"], hashlib.sha256(self.profile_raw).hexdigest())
        pins = json.loads((ROOT / "provenance/STACK-PINS.json").read_bytes())
        self.assertEqual(pins["sglang"]["revision"], self.profile["runtime"]["sglang_revision"])
        self.assertEqual(pins["model_revision"], self.profile["model"]["revision"])
        for name in ("python", "torch", "hip", "triton", "rccl", "sglang_kernel", "rocm_sdk"):
            self.assertEqual(pins[name], VERSIONS[name.replace("sglang_kernel", "sglang-kernel")])

    def test_wrong_image_repository_digest_or_config(self):
        for field, value in (("image", IMAGE.replace("localhost/workstation", "untrusted")),
                             ("image", IMAGE[:-1] + "0"), ("image_config_sha256", "0" * 64)):
            candidate = deepcopy(self.profile)
            candidate["runtime"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_profile(candidate)

    def test_wrong_version_or_source(self):
        for category, name in (("versions", "sglang"), ("native_source_sha256", "srt/layers/quantization/fp8.py")):
            candidate = deepcopy(self.profile)
            candidate["runtime"][category][name] += "+changed"
            with self.subTest(category=category), self.assertRaises(ValueError):
                validate_profile(candidate)
        candidate = deepcopy(self.profile)
        candidate["historical_environment"]["ROCM_HOME"] = "/opt/rocm"
        with self.assertRaises(ValueError):
            validate_profile(candidate)
        candidate = deepcopy(self.profile)
        candidate["cache_policy"]["accepted_cache_reused"] = True
        with self.assertRaises(ValueError):
            validate_profile(candidate)

    def test_context_eager_graph_and_kv4_changes_refused(self):
        for field, value in (("context", 32768), ("disable_cuda_graph", True),
                             ("observed_capture_batches_per_rank", [[1], [1]]),
                             ("kv_storage", "KV4"), ("kv4_overlay", True), ("chunked_prefill_size_resolved", 2048)):
            candidate = deepcopy(self.profile)
            candidate["settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_profile(candidate)

    def test_missing_source_and_false_authority_refused(self):
        candidate = deepcopy(self.profile)
        del candidate["runtime"]["native_source_sha256"]["srt/server_args.py"]
        with self.assertRaises(ValueError):
            validate_profile(candidate)
        candidate = deepcopy(self.profile)
        candidate["authorization"]["launch_authorized"] = True
        with self.assertRaises(ValueError):
            validate_profile(candidate)

    def test_changed_summary_scope_or_promotion_refused(self):
        candidate = deepcopy(self.summary)
        candidate["scope"] = "Production serving qualified"
        with self.assertRaises(ValueError):
            validate_summary(candidate)
        for field in self.summary["claims"]:
            candidate = deepcopy(self.summary)
            candidate["claims"][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_summary(candidate)

    def test_task_batching_and_cleanup_overclaims_refused(self):
        for field, value in (("code_tasks_status", "PASSED"), ("teacher_forward_batching", "batched"),
                             ("cleanup_verified", False), ("graceful_distributed_rank_destruction_independently_qualified", True)):
            candidate = deepcopy(self.summary)
            candidate["observations"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_summary(candidate)

    def test_corrupt_or_incomplete_inventory_refused(self):
        records = json.loads(self.inventory_raw)
        for candidate in (records[:-1], records + [records[0]], [{**records[0], "name": "../escape"}] + records[1:]):
            with self.assertRaises(ValueError):
                validate_inventory(json.dumps(candidate).encode())

    def test_no_runtime_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("sglang", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
