"""Offline descriptive comparison of two complete inner-model timing records.

This tool neither starts work nor authenticates protected-host cleanup. Input
measurements must come from the separately reviewed physical owner workflow.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


_ENTRY = Path(__file__).resolve().with_name("model_benchmark.py")
_MODULE_SPEC = importlib.util.spec_from_file_location("spry_benchmark_contract", _ENTRY)
bench = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(bench)
MAXIMUM_BYTES = 8 << 20


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON member")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def read_json(path, expected_sha256=None):
    raw = bench.read_bounded_file(Path(path), MAXIMUM_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("spec digest mismatch")
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value, digest


def checked_arm(result, arm, spec_sha256, *, expected_profile=False, session_sha256=None):
    """Recompute the complete native-record contract, without native imports."""
    required = {"schema", "scope", "arm", "status", "spec_sha256", "timed_out", "exits",
                "groups_observed_absent", "parent_error", "cleanup_errors", "rank_parse_errors",
                "qualification", "serving_quality_qualified", "http_throughput_measured",
                "independent_release_required", "ranks", "samples"}
    if (not required.issubset(result) or result.get("schema") != 1 or result.get("scope") != bench.SCOPE
            or result.get("arm") != arm
            or result.get("status") != "MODEL_BENCHMARK_COMPLETE"
            or result.get("spec_sha256") != spec_sha256
            or result.get("timed_out") is not False
            or result.get("exits") != [0, 0]
            or any(type(value) is not int for value in result["exits"])
            or result.get("groups_observed_absent") != [True, True]
            or any(value is not True for value in result["groups_observed_absent"])
            or result.get("parent_error") is not None
            or result.get("cleanup_errors") != []
            or result.get("rank_parse_errors") != []
            or result.get("qualification") is not False
            or result.get("serving_quality_qualified") is not False
            or result.get("http_throughput_measured") is not False
            or result.get("independent_release_required") is not True):
        raise ValueError("incomplete or unbound inner-model arm")
    ranks = result.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError("exactly two rank records required")
    for rank in ranks:
        if (not isinstance(rank, dict) or rank.get("qualification") is not False
                or rank.get("serving_quality_qualified") is not False
                or type(rank.get("rank")) is not int):
            raise ValueError("rank qualification contract")
        for batch in rank.get("batches", []):
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in batch.get("step_seconds", [])):
                raise ValueError("positive finite per-rank step timing required")
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   for row in batch.get("generated_logprobs", []) for value in row):
                raise ValueError("finite numeric logprobs required")
    if expected_profile:
        if (result.get("instrumented_timings") is not True
                or result.get("profile_method") != bench.PROFILE_METHOD
                or result.get("diagnostic_profile_complete") is not True
                or result.get("diagnostic_profile") != bench.PROFILE_CONFIG):
            raise ValueError("profiled arm must disclose instrumented timings")
        bench.validate_profile_records(ranks, arm,
            session_sha256=session_sha256, spec_sha256=spec_sha256)
    elif (result.get("instrumented_timings", False) is not False
            or any(rank.get("diagnostic_profile") is not None for rank in ranks)):
        raise ValueError("unexpected profiling in an unprofiled arm")
    samples = bench.summarize(ranks, arm, expected_profile=expected_profile)
    if result.get("samples") != samples:
        raise ValueError("stored summary differs from rank measurements")
    for sample in samples:
        for key in ("prefill_rank_max_seconds", "decode_31_seconds",
                    "forward_sample_total_seconds", "batch_wall_rank_max_seconds"):
            value = sample[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite timing required")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in sample["decode_rank_max_step_seconds"]):
            raise ValueError("positive finite decode steps required")
    return samples


def _statistics(values):
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def _interval_union(intervals):
    end, total = None, 0.0
    for start, duration in sorted(intervals):
        stop = start + duration
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end) if end is not None else stop
    return total


def attribution_rows(result):
    """Summarize retained intervals; overlapping durations are not a wall share."""
    rows = []
    for rank in result["ranks"]:
        for window in rank["diagnostic_profile"]["windows"]:
            gpu = window["gpu"]
            intervals = [(row[1], row[2]) for row in gpu["intervals"]]
            first = min(start for start, duration in intervals)
            last = max(start + duration for start, duration in intervals)
            cpu = {row["qualname"]: row for row in window["cpu"]}
            rows.append({key: window[key] for key in ("rank", "arm", "pid", "cell", "sample", "step", "phase")}
                | {"gpu_event_count": len(intervals), "gpu_interval_union_us": _interval_union(intervals),
                   "first_to_last_gpu_event_us": last - first,
                   "gpu_categories_by_duration_sum": sorted(
                       ({"category": name, **values} for name, values in gpu["categories"].items()),
                       key=lambda row: row["duration_us"], reverse=True),
                   "native_call_inclusive_seconds": cpu.get("Native.call", {}).get("inclusive_seconds"),
                   "cpu_functions_by_exclusive_wall": sorted(window["cpu"],
                       key=lambda row: row["exclusive_seconds"], reverse=True),
                   "time_scope": "Instrumented selected forward; CPU native-call wall includes GPU waiting.",
                   "gpu_occupancy_or_bandwidth_measured": False})
    return rows


def compare_outputs(left, right):
    """Only equal generated tokens with equal preceding histories are comparable."""
    observations = []
    for ordinal, (a, b) in enumerate(zip(left["ranks"][0]["batches"],
                                           right["ranks"][0]["batches"])):
        for request, request_id in enumerate(a["request_ids"]):
            x, y = a["generated_ids"][request], b["generated_ids"][request]
            first = next((index for index, pair in enumerate(zip(x, y))
                          if pair[0] != pair[1]), None)
            common = len(x) if first is None else first
            gaps = []
            for rank in (0, 1):
                p = left["ranks"][rank]["batches"][ordinal]["generated_logprobs"][request]
                q = right["ranks"][rank]["batches"][ordinal]["generated_logprobs"][request]
                gaps.extend({"rank": rank, "generated_position": index,
                             "absolute_gap": abs(p[index] - q[index])}
                            for index in range(common))
            observations.append({"cell": a["cell"], "sample": a["sample"],
                "measured": a["measured"], "request_id": request_id,
                "bf16_generated_ids": x, "kv4_generated_ids": y,
                "generated_ids_equal": first is None,
                "first_different_generated_token": first,
                "matching_generated_common_prefix_tokens": common,
                "largest_common_prefix_logprob_gap": (
                    max(gaps, key=lambda item: item["absolute_gap"]) if gaps else None)})
    return observations


def analyze(spec, spec_sha256, left, right):
    bench.validate_spec(spec)
    profiled = spec.get("diagnostic_profile") is not None
    samples = {"bf16": checked_arm(left, "bf16", spec_sha256, expected_profile=profiled),
               "kv4": checked_arm(right, "kv4", spec_sha256, expected_profile=profiled)}
    cells = []
    metrics = ("prefill_rank_max_seconds", "decode_31_seconds", "forward_sample_total_seconds")
    for name, concurrency, length in bench.CELLS:
        arms = {}
        for arm in ("bf16", "kv4"):
            rows = [dict(row, decode_tokens_per_second=31 * concurrency / row["decode_31_seconds"])
                    for row in samples[arm] if row["cell"] == name]
            arms[arm] = {"warmup": rows[0], "measured_samples": rows[1:],
                "statistics": {key: _statistics([row[key] for row in rows[1:]])
                               for key in (*metrics, "decode_tokens_per_second")}}
        faster = {key: 100 * (arms["bf16"]["statistics"][key]["median"]
                              / arms["kv4"]["statistics"][key]["median"] - 1)
                  for key in metrics}
        cells.append({"cell": name, "concurrency": concurrency, "input_tokens": length,
                      "arms": arms, "kv4_faster_percent_by_latency_median": faster})
    report = {"schema": 1, "status": "OFFLINE_COMPARISON_COMPLETE",
        "scope": "inner-model-timing-only", "spec_sha256": spec_sha256,
        "qualification": False, "serving_quality_qualified": False,
        "physical_campaign_complete": False, "independent_release_verified": False,
        "accepted_live_cleanup_authority": False, "http_throughput_measured": False,
        "ttft_measured": False, "end_to_end_throughput_measured": False,
        "overall_winner": None, "cells": cells, "output_comparisons": compare_outputs(left, right),
        "interpretation": ["One fixed-order two-arm inner-model screen, not independent-start evidence.",
            "Warmups are retained but excluded from the five-sample statistics.",
            "Positive latency-ratio percent means KV4 faster in that cell and metric only.",
            "Logprob comparisons stop before the first different generated token.",
            "Worker/process-group records do not establish protected-host or device release.",
            "No quality, HTTP, native performance or general serving winner is established."]}
    if profiled:
        report.update(status="OFFLINE_ATTRIBUTION_COMPLETE", instrumented_timings=True,
            profile_method=bench.PROFILE_METHOD,
            attribution={arm: attribution_rows(result)
                         for arm, result in (("bf16", left), ("kv4", right))},
            diagnostic_profiles={arm: [rank["diagnostic_profile"] for rank in result["ranks"]]
                                 for arm, result in (("bf16", left), ("kv4", right))})
        report["interpretation"].insert(0,
            "Selected existing forwards include CPU/GPU profiling overhead; these timings are diagnostic observations.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--kv4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec, digest = read_json(args.spec, args.expected_spec_sha256)
    left, left_digest = read_json(args.bf16)
    right, right_digest = read_json(args.kv4)
    report = analyze(spec, digest, left, right)
    report["input_result_sha256"] = {"bf16": left_digest, "kv4": right_digest}
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # This identifies this offline parser contract, not the physically executed file.
    report["analysis_contract_source_sha256"] = hashlib.sha256(_ENTRY.read_bytes()).hexdigest()
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("absolute output under existing owned parent required")
    bench.write_result(output, report)


if __name__ == "__main__":
    main()
