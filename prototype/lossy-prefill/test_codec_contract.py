"""CPU-only contract tests; these do not import Torch, Triton or the launcher."""

import ast
import copy
import math
from pathlib import Path
import struct
import time
import unittest

import codec


def completed_record():
    spec = codec.layout(2048 * codec.HIDDEN)
    result = {
        "schema_version": 1,
        "scope": "synthetic_eager_rccl_component_only",
        "status": "accepted",
        "stage": "complete",
        "devices": [0, 1],
        "schedule": codec.schedule_contract(),
        "precision_acceptance_implied": False,
        "serving_benefit_claimed": False,
        "graph": {"status": "not_run", "captures": 0, "replays": 0},
        "memory": {
            "largest_shape_layout": spec,
            "device_tensor_limit_bytes_each": codec.MAX_DEVICE_TENSOR_BYTES,
            "device_tensor_limit_bytes_aggregate": codec.MAX_TOTAL_TENSOR_BYTES,
            "host_tensor_budget_bytes": codec.MAX_HOST_TENSOR_BYTES,
            "observed_torch_allocated_bytes_each": [spec["live_tensor_bytes_per_device"]] * 2,
        },
        "feasibility": {"status": "passed", "bf16_sum_exact": True, "uint8_gather_exact": True},
        "numerical": [],
        "timings": [],
    }
    for rows in codec.ROWS:
        for pattern in codec.PATTERNS:
            result["numerical"].append({
                "shape": [rows, codec.HIDDEN], "pattern": pattern,
                "rank_outputs_bitwise_equal": True,
                "baseline_matches_cpu_bf16_sum": True,
                "quantization_codes_and_scales_exact": True,
                "codec_output_matches_cpu_bitwise": True,
                "analytical_error_bound_passed": True,
                "input_normalized_rms_error": 0.005,
                "max_absolute_error": 0.01, "absolute_rms_error": 0.005,
                "max_absolute_input": 1.0, "max_analytical_bound_fraction": 0.5,
                "relative_output_l2": None if pattern == "cancellation" else 0.005,
                "screen_limit": codec.SCREEN_LIMIT, "screen_passed": True,
            })
        pairs = []
        for index in range(codec.OBSERVATIONS):
            pairs.append({
                "order": ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"],
                "baseline": {"device_ms": [1.0, 2.0], "host_completion_ms": 3.0},
                "candidate": {"device_ms": [0.8, 1.5], "host_completion_ms": 2.5},
            })
        result["timings"].append({"shape": [rows, codec.HIDDEN], **codec._summarize_pairs(pairs)})
    result["decision"] = {
        "numerical_screen_passed": True,
        "input_normalized_rms_limit": codec.SCREEN_LIMIT,
        "minimum_device_speedup_fraction_at_2048": codec.BENEFIT_THRESHOLD,
        "device_speedup_fraction_at_2048": result["timings"][-1]["device_speedup_fraction"],
        "useful_component_threshold_met": True,
    }
    return result


class CodecArithmeticTests(unittest.TestCase):
    def test_ties_are_away_from_zero(self):
        self.assertEqual([codec.round_half_away(x) for x in (-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5)], [-3, -2, -1, 0, 1, 2, 3])
        packed = codec.pack_reference_block([127.0, -127.0, 0.5, -0.5, 1.5, -1.5])
        self.assertEqual(struct.unpack_from("<6b", packed), (127, -127, 1, -1, 2, -2))
        self.assertEqual(struct.unpack_from("<f", packed, 256)[0], 1.0)

    def test_packed_layout_and_zero_floor(self):
        packed = codec.pack_reference_block([0.0] * 256)
        self.assertEqual(len(packed), 260)
        self.assertEqual(packed[:256], bytes(256))
        self.assertEqual(struct.unpack_from("<f", packed, 256)[0], codec.SCALE_FLOOR)
        self.assertEqual(codec.unpack_reference_block(packed), [0.0] * 256)
        partial = codec.pack_reference_block([127.0])
        self.assertEqual(partial[1:256], bytes(255))

    def test_small_and_large_inputs_have_bounded_quantization(self):
        for multiplier in (1e-5, 1.0, 1e5):
            values = [codec._f32(math.sin(i * 0.25) * multiplier) for i in range(256)]
            packed = codec.pack_reference_block(values)
            scale = struct.unpack_from("<f", packed, 256)[0]
            self.assertTrue(math.isfinite(scale) and scale > 0)
            self.assertNotIn(-128, struct.unpack_from("<256b", packed))
            for actual, expected in zip(codec.unpack_reference_block(packed), values):
                self.assertLessEqual(abs(actual - expected), scale * 0.50001 + abs(expected) * 2**-22)

    def test_opposite_blocks_cancel_exactly(self):
        values = [math.cos(i) * 3 for i in range(256)]
        left = codec.unpack_reference_block(codec.pack_reference_block(values))
        right = codec.unpack_reference_block(codec.pack_reference_block([-x for x in values]))
        self.assertTrue(all(a + b == 0 for a, b in zip(left, right)))

    def test_nonfinite_and_invalid_blocks_rejected(self):
        for values in ([], [0.0] * 257, [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                codec.pack_reference_block(values)
        with self.assertRaises(ValueError):
            codec.unpack_reference_block(bytes(256))

    def test_cancellation_safe_metrics(self):
        result = codec.normalized_errors(1.0, 10000.0, 0.0, 100)
        self.assertEqual(result["input_normalized_rms_error"], 0.01)
        self.assertEqual(result["absolute_rms_error"], 0.1)
        self.assertIsNone(result["relative_output_l2"])
        self.assertEqual(codec.normalized_errors(0, 0, 0, 256)["input_normalized_rms_error"], 0)
        with self.assertRaises(codec.ComponentFault):
            codec.normalized_errors(1, 0, 0, 256)


class ScheduleAndResultTests(unittest.TestCase):
    def test_memory_and_exact_finite_schedule(self):
        spec = codec.layout(2048 * 5120)
        self.assertEqual(spec["blocks"], 40960)
        self.assertEqual(spec["packed_bytes_per_rank"], 10649600)
        self.assertEqual(spec["live_tensor_bytes_per_device"], 94863360)
        self.assertEqual(spec["live_tensor_bytes_aggregate"], 189726720)
        self.assertLess(spec["live_tensor_bytes_aggregate"], 256 * 1024**2)
        schedule = codec.schedule_contract()
        self.assertEqual(schedule["all_reduce_calls"], 64)
        self.assertEqual(schedule["all_gather_calls"], 64)
        self.assertEqual(schedule["quantize_kernel_launches"], 126)
        self.assertEqual(schedule["unpack_sum_kernel_launches"], 126)
        self.assertEqual((schedule["graph_captures"], schedule["graph_replays"], schedule["retries"]), (0, 0, 0))
        self.assertEqual(codec.layout(257)["packed_bytes_per_rank"], 520)

    def test_event_wait_has_deadline_and_no_synchronize(self):
        class Pending:
            def query(self):
                return False
        with self.assertRaises(codec.ComponentFault):
            codec.wait_events([Pending()], time.monotonic() - 1)
        class Done:
            def query(self):
                return True
        codec.wait_events([Done()], time.monotonic() + 1)

    def test_known_blockers_do_not_swallow_hardware_or_unknown_errors(self):
        self.assertTrue(codec._known_gate_blocker(RuntimeError("Bootstrap : no socket interface found")))
        for text in ("illegal memory access", "GPU reset during Bootstrap :", "Bootstrap : unhandled cuda error", "unknown runtime failure"):
            self.assertFalse(codec._known_gate_blocker(RuntimeError(text)))

    def test_valid_final_evidence_and_nonfatal_rejection(self):
        result = completed_record()
        self.assertEqual(codec.validate_result(result), [])
        rejected = copy.deepcopy(result)
        rejected["status"] = "rejected"
        rejected["numerical"][0].update(input_normalized_rms_error=0.03, screen_passed=False)
        rejected["decision"]["numerical_screen_passed"] = False
        self.assertEqual(codec.validate_result(rejected), [])

    def test_inconsistent_or_incomplete_evidence_rejected(self):
        changes = (
            lambda r: r.update(status="running"),
            lambda r: r["schedule"].update(graph_captures=1),
            lambda r: r["numerical"].pop(),
            lambda r: r["timings"][0]["pairs"].pop(),
            lambda r: r["timings"][0]["pairs"][0]["baseline"].update(host_completion_ms=float("nan")),
            lambda r: r["decision"].update(useful_component_threshold_met=False),
            lambda r: r["memory"].update(observed_torch_allocated_bytes_each=[256 * 1024**2] * 2),
            lambda r: r.update(serving_benefit_claimed=True),
            lambda r: r.update(graph=None),
            lambda r: r["numerical"][0].pop("relative_output_l2"),
            lambda r: r["feasibility"].pop("bf16_sum_exact"),
            lambda r: r["timings"][0]["pairs"][0]["baseline"].update(device_ms=None),
        )
        for change in changes:
            record = completed_record()
            change(record)
            self.assertTrue(codec.validate_result(record))

    def test_local_reader_imports_no_framework_and_no_sync_calls(self):
        for filename in ("codec.py", "kernels.py"):
            tree = ast.parse(Path(__file__).with_name(filename).read_text())
            for statement in tree.body:
                if isinstance(statement, ast.Import):
                    self.assertFalse(any(name.name in ("torch", "triton") for name in statement.names))
                if isinstance(statement, ast.ImportFrom):
                    self.assertNotIn(statement.module, ("torch", "triton"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "synchronize")


if __name__ == "__main__":
    unittest.main()
