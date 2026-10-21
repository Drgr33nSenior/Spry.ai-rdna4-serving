"""Synthetic CPU records only: these tests contain no measured GPU results."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


readout = load("readout_under_test", ROOT / "tools/analyze_model_benchmark.py")
fixtures = load("synthetic_benchmark_fixtures", ROOT / "tests/test_model_benchmark.py")
DIGEST = "a" * 64


def arm_result(arm, seconds=0.02):
    ranks = fixtures.completed_records()
    for rank in ranks:
        rank.update(arm=arm, qualification=False, serving_quality_qualified=False)
        for batch in rank["batches"]:
            batch["step_seconds"] = [seconds] * 32
    return {"schema": 1, "scope": readout.bench.SCOPE, "arm": arm,
        "status": "MODEL_BENCHMARK_COMPLETE", "spec_sha256": DIGEST,
        "timed_out": False, "exits": [0, 0], "ranks": ranks,
        "samples": readout.bench.summarize(ranks, arm), "parent_error": None,
        "cleanup_errors": [], "rank_parse_errors": [], "groups_observed_absent": [True, True],
        "qualification": False, "serving_quality_qualified": False,
        "http_throughput_measured": False, "independent_release_required": True}


class BenchmarkReadoutTests(unittest.TestCase):
    def test_positive_math_keeps_warmups_and_scope_limits(self):
        left, right = arm_result("bf16", 0.02), arm_result("kv4", 0.01)
        # A deliberately large retained warmup cannot affect five-sample statistics.
        for rank in left["ranks"]:
            rank["batches"][0]["step_seconds"] = [1.0] * 32
        left["samples"] = readout.bench.summarize(left["ranks"], "bf16")
        report = readout.analyze(fixtures.fixture(), DIGEST, left, right)
        self.assertEqual(len(report["cells"]), 5)
        for cell in report["cells"]:
            self.assertEqual(len(cell["arms"]["bf16"]["measured_samples"]), 5)
            self.assertFalse(cell["arms"]["bf16"]["warmup"]["measured"])
            self.assertAlmostEqual(cell["kv4_faster_percent_by_latency_median"]["decode_31_seconds"], 100)
            self.assertAlmostEqual(cell["arms"]["kv4"]["statistics"]["decode_tokens_per_second"]["median"],
                                   100 * cell["concurrency"])
        self.assertEqual(len(report["output_comparisons"]), 42)
        for key in ("qualification", "physical_campaign_complete", "independent_release_verified",
                    "accepted_live_cleanup_authority", "http_throughput_measured", "ttft_measured"):
            self.assertIs(report[key], False)
        self.assertIsNone(report["overall_winner"])
        self.assertEqual(report["cells"][0]["arms"]["bf16"]["warmup"]["prefill_rank_max_seconds"], 1.0)

    def test_generated_common_prefix_excludes_divergence_and_later_scores(self):
        left, right = arm_result("bf16"), arm_result("kv4")
        for rank in right["ranks"]:
            batch = rank["batches"][0]
            batch["generated_ids"][0][3] = 8
            batch["generated_logprobs"][0] = [-1.25] * 3 + [-1000.0] * 29
            batch = rank["batches"][1]
            batch["generated_ids"][0][0] = 8
        right["samples"] = readout.bench.summarize(right["ranks"], "kv4")
        rows = readout.analyze(fixtures.fixture(), DIGEST, left, right)["output_comparisons"]
        self.assertEqual(rows[0]["first_different_generated_token"], 3)
        self.assertEqual(rows[0]["matching_generated_common_prefix_tokens"], 3)
        self.assertEqual(rows[0]["largest_common_prefix_logprob_gap"]["absolute_gap"], 0.25)
        self.assertEqual(rows[1]["matching_generated_common_prefix_tokens"], 0)
        self.assertIsNone(rows[1]["largest_common_prefix_logprob_gap"])

    def test_partial_identity_mode_exit_cleanup_and_stored_summary_refuse(self):
        edits = (
            lambda r: r.update(status="MODEL_BENCHMARK_FAILED"),
            lambda r: r.update(spec_sha256="b" * 64),
            lambda r: r.update(scope="other"),
            lambda r: r.update(arm="bf16"),
            lambda r: r.update(exits=[0, 1]),
            lambda r: r.update(groups_observed_absent=[True, False]),
            lambda r: r.update(qualification=True),
            lambda r: r.pop("parent_error"),
            lambda r: r["ranks"][1].update(rank=0),
            lambda r: r["ranks"][0]["batches"].pop(),
            lambda r: r["samples"][0].update(prefill_rank_max_seconds=99),
        )
        for index, edit in enumerate(edits):
            right = arm_result("kv4")
            edit(right)
            with self.subTest(index=index), self.assertRaises((ValueError, KeyError, TypeError)):
                readout.analyze(fixtures.fixture(), DIGEST, arm_result("bf16"), right)

    def test_zero_nonfinite_malformed_and_wrong_rank_bindings_refuse(self):
        for value in (0, -1, float("inf"), float("nan"), True, "0.1"):
            right = arm_result("kv4")
            right["ranks"][0]["batches"][0]["step_seconds"][0] = value
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                readout.analyze(fixtures.fixture(), DIGEST, arm_result("bf16"), right)
        right = arm_result("kv4")
        right["ranks"][1]["batches"][0]["generated_ids"][0][0] = 8
        with self.assertRaises(ValueError):
            readout.analyze(fixtures.fixture(), DIGEST, arm_result("bf16"), right)
        wrong = fixtures.fixture()
        wrong["server_args"]["quantization"] = "mxfp4"
        with self.assertRaises(ValueError):
            readout.analyze(wrong, DIGEST, arm_result("bf16"), arm_result("kv4"))

    def test_bounded_json_hash_duplicate_nonfinite_link_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    readout.read_json(path)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                readout.read_json(path, "f" * 64)
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                readout.read_json(link)
            with self.assertRaises(ValueError):
                readout.read_json(Path("relative.json"))

    def test_isolated_cli_analyzes_synthetic_inputs_and_refuses_overwrite(self):
        command = [sys.executable, "-I", "-B", "-S", str(ROOT / "tools/analyze_model_benchmark.py")]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        help_result = subprocess.run([*command, "--help"], capture_output=True, timeout=10, env=env)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps(fixtures.fixture()))
            digest = hashlib.sha256(spec.read_bytes()).hexdigest()
            for arm in ("bf16", "kv4"):
                record = arm_result(arm)
                record["spec_sha256"] = digest
                (root / (arm + ".json")).write_text(json.dumps(record))
            args = [*command, "--spec", str(spec), "--expected-spec-sha256", digest,
                    "--bf16", str(root / "bf16.json"), "--kv4", str(root / "kv4.json"),
                    "--output", str(root / "readout.json")]
            result = subprocess.run(args, capture_output=True, timeout=10, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads((root / "readout.json").read_bytes())["qualification"])
            repeated = subprocess.run(args, capture_output=True, timeout=10, env=env)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn(b"FileExistsError", repeated.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
