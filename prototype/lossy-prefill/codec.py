"""Bounded synthetic BF16/RCCL versus block-256 INT8 component experiment.

The caller authenticates sources, imports frameworks, checks physical GPU UUIDs,
preloads ``kernels``, and owns a process watchdog. ``end`` is an absolute
time.monotonic() deadline. Framework calls may block inside native code; event
polling alone cannot interrupt those calls, so the external watchdog is required.
This module has no framework imports or GPU work at import time.
"""

import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time


BLOCK = 256
BLOCK_BYTES = 260
SCALE_FLOOR = 2.0**-126
ROWS = (8, 16, 2048)
HIDDEN = 5120
PATTERNS = ("normal", "small", "high_dynamic_range", "cancellation", "zeros")
WARMUPS = 4
OBSERVATIONS = 12
SCREEN_LIMIT = 0.02
BENEFIT_THRESHOLD = 0.05
CHUNK_ELEMENTS = 256 * 1024
MAX_DEVICE_TENSOR_BYTES = 128 * 1024**2
MAX_TOTAL_TENSOR_BYTES = 256 * 1024**2
MAX_HOST_TENSOR_BYTES = 512 * 1024**2
STATUSES = ("accepted", "rejected", "inconclusive", "hard_fault")


class ComponentFault(RuntimeError):
    pass


def layout(elements):
    if not isinstance(elements, int) or isinstance(elements, bool) or elements <= 0:
        raise ValueError("elements must be a positive integer")
    blocks = (elements + BLOCK - 1) // BLOCK
    packed = blocks * BLOCK_BYTES
    # Per card: input, baseline output, candidate output; local and two-rank packs.
    per_device = 3 * elements * 2 + 3 * packed
    return {
        "elements": elements,
        "blocks": blocks,
        "packed_bytes_per_rank": packed,
        "storage_float32_elements": blocks * 65,
        "gathered_bytes_per_device": 2 * packed,
        "live_tensor_bytes_per_device": per_device,
        "live_tensor_bytes_aggregate": 2 * per_device,
    }


def schedule_contract():
    return {
        "shapes": [[r, HIDDEN] for r in ROWS],
        "patterns": list(PATTERNS),
        "feasibility_attempts": 1,
        "numerical_cases": len(ROWS) * len(PATTERNS),
        "warmup_pairs_per_shape": WARMUPS,
        "measured_pairs_per_shape": OBSERVATIONS,
        "order": "baseline-first on even pairs; candidate-first on odd pairs",
        "all_reduce_calls": 1 + len(ROWS) * (len(PATTERNS) + WARMUPS + OBSERVATIONS),
        "all_gather_calls": 1 + len(ROWS) * (len(PATTERNS) + WARMUPS + OBSERVATIONS),
        "quantize_kernel_launches": 2 * len(ROWS) * (len(PATTERNS) + WARMUPS + OBSERVATIONS),
        "unpack_sum_kernel_launches": 2 * len(ROWS) * (len(PATTERNS) + WARMUPS + OBSERVATIONS),
        "graph_captures": 0,
        "graph_replays": 0,
        "retries": 0,
    }


def round_half_away(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite quantization operand")
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack_reference_block(values):
    """Small stdlib layout oracle; runtime uses chunked CPU Torch references."""
    if not 1 <= len(values) <= BLOCK:
        raise ValueError("one block must contain 1 to 256 values")
    values = [_f32(x) for x in values]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite quantization operand")
    scale = max(_f32(max(abs(x) for x in values) / 127.0), SCALE_FLOOR)
    codes = []
    for value in values:
        quotient = _f32(value / scale)
        magnitude = math.floor(_f32(abs(quotient) + 0.5))
        code = magnitude if quotient >= 0 else -magnitude
        codes.append(max(-127, min(127, code)))
    codes.extend([0] * (BLOCK - len(codes)))
    return struct.pack("<256bf", *codes, scale)


def unpack_reference_block(packed):
    if len(packed) != BLOCK_BYTES:
        raise ValueError("packed block must contain 260 bytes")
    values = struct.unpack("<256bf", packed)
    return [_f32(code * values[-1]) for code in values[:-1]]


def normalized_errors(error_squared, input_squared, reference_squared, elements):
    if elements <= 0 or not all(math.isfinite(v) and v >= 0 for v in (error_squared, input_squared, reference_squared)):
        raise ValueError("invalid error accumulator")
    if input_squared == 0 and error_squared != 0:
        raise ComponentFault("nonzero output error for zero input norm")
    return {
        "absolute_rms_error": math.sqrt(error_squared / elements),
        "input_normalized_rms_error": math.sqrt(error_squared / input_squared) if input_squared else 0.0,
        "relative_output_l2": math.sqrt(error_squared / reference_squared) if reference_squared else None,
    }


def _deadline(end):
    if time.monotonic() >= end:
        raise ComponentFault("finite component deadline exceeded")


def wait_events(events, end):
    """Poll only; no event/device synchronize or unbounded wait."""
    pending = list(events)
    while pending:
        _deadline(end)
        pending = [event for event in pending if not event.query()]


def _write(path, record):
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _known_gate_blocker(exc):
    """Only specific unavailable/backend/bootstrap failures are non-faults."""
    message = str(exc).lower()
    hardware = (
        "illegal memory", "memory access fault", "page fault", "gpu reset",
        "device-side assert", "launch failure", "hiperrorillegaladdress",
        "hsa_status_error_exception", "nonfinite", "unhandled cuda error",
        "unhandled hip error", "hip error", "cuda error", "device lost",
    )
    if any(text in message for text in hardware):
        return False
    return any(text in message for text in (
        "not compiled with nccl", "no socket interface found", "bootstrap :",
        "bootstrap:", "network is unreachable", "no route to host",
        "connection refused", "ncclinvalidtype", "unsupported data type",
        "unsupported dtype", "invalid scalar type",
    ))


def _record_events(torch, streams):
    events = []
    for stream in streams:
        with torch.cuda.device(stream.device):
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)
    return events


def _copy_to_host(torch, pairs, streams, end):
    for rank, copies in enumerate(pairs):
        with torch.cuda.device(streams[rank].device), torch.cuda.stream(streams[rank]):
            for host, device in copies:
                host.copy_(device, non_blocking=True)
    wait_events(_record_events(torch, streams), end)


def _feasibility(torch, nccl, devices, streams, end):
    """Exactly one tiny call per collective, including BF16 and uint8 contracts."""
    inputs, outputs, byte_inputs, byte_outputs = [], [], [], []
    host_inputs, host_outputs, host_bytes, host_gathered = [], [], [], []
    for rank, device in enumerate(devices):
        h = torch.arange(16, dtype=torch.float32).to(torch.bfloat16).pin_memory()
        h.add_(rank * 16)
        hb = (torch.arange(32, dtype=torch.int16) + rank * 64).to(torch.uint8).pin_memory()
        host_inputs.append(h)
        host_bytes.append(hb)
        host_outputs.append(torch.empty_like(h, pin_memory=True))
        host_gathered.append(torch.empty(64, dtype=torch.uint8, pin_memory=True))
        with torch.cuda.device(device), torch.cuda.stream(streams[rank]):
            x = torch.empty(16, device=device, dtype=torch.bfloat16)
            x.copy_(h, non_blocking=True)
            xb = torch.empty(32, device=device, dtype=torch.uint8)
            xb.copy_(hb, non_blocking=True)
            inputs.append(x)
            outputs.append(torch.empty_like(x))
            byte_inputs.append(xb)
            byte_outputs.append(torch.empty(64, device=device, dtype=torch.uint8))
    wait_events(_record_events(torch, streams), end)
    _deadline(end)
    nccl.all_reduce(inputs, outputs, streams=streams)
    wait_events(_record_events(torch, streams), end)
    _deadline(end)
    nccl.all_gather(byte_inputs, byte_outputs, streams=streams)
    wait_events(_record_events(torch, streams), end)
    _copy_to_host(torch, [
        [(host_outputs[r], outputs[r]), (host_gathered[r], byte_outputs[r])]
        for r in range(2)
    ], streams, end)
    expected_sum = (host_inputs[0].float() + host_inputs[1].float()).to(torch.bfloat16)
    expected_bytes = torch.cat(host_bytes)
    if not all(torch.equal(value, expected_sum) for value in host_outputs):
        raise ComponentFault("tiny BF16 all_reduce failed exact CPU sum check")
    if not all(torch.equal(value, expected_bytes) for value in host_gathered):
        raise ComponentFault("tiny uint8 all_gather failed exact rank-order check")
    return {"status": "passed", "bf16_sum_exact": True, "uint8_gather_exact": True}


def _allocate(torch, devices, streams, elements):
    spec = layout(elements)
    result = {"layout": spec}
    for name in ("input", "baseline", "candidate", "packed", "gathered",
                 "host_input", "host_baseline", "host_candidate", "host_packed"):
        result[name] = []
    for rank, device in enumerate(devices):
        with torch.cuda.device(device), torch.cuda.stream(streams[rank]):
            for name in ("input", "baseline", "candidate"):
                result[name].append(torch.empty(elements, device=device, dtype=torch.bfloat16))
                result["host_" + name].append(torch.empty(elements, dtype=torch.bfloat16, pin_memory=True))
            for name, multiplier in (("packed", 1), ("gathered", 2)):
                result[name].append(torch.empty(
                    multiplier * spec["storage_float32_elements"], device=device, dtype=torch.float32
                ))
            result["host_packed"].append(torch.empty(
                spec["storage_float32_elements"], dtype=torch.float32, pin_memory=True
            ))
    result["packed_i8"] = [value.view(torch.int8) for value in result["packed"]]
    result["packed_u8"] = [value.view(torch.uint8) for value in result["packed"]]
    result["gathered_i8"] = [value.view(torch.int8) for value in result["gathered"]]
    result["gathered_u8"] = [value.view(torch.uint8) for value in result["gathered"]]
    result["start_events"] = [torch.cuda.Event(enable_timing=True) for _ in devices]
    result["stop_events"] = [torch.cuda.Event(enable_timing=True) for _ in devices]
    return result


def _fill_inputs(torch, buffers, pattern, streams, end):
    hosts = buffers["host_input"]
    n = buffers["layout"]["elements"]
    generators = [torch.Generator(device="cpu").manual_seed(20261025 + rank) for rank in range(2)]
    for start in range(0, n, CHUNK_ELEMENTS):
        _deadline(end)
        count = min(CHUNK_ELEMENTS, n - start)
        for rank in range(2):
            if pattern == "zeros":
                hosts[rank][start:start + count].zero_()
                continue
            if pattern == "cancellation" and rank == 1:
                hosts[rank][start:start + count].copy_(-hosts[0][start:start + count])
                continue
            values = torch.randn(count, generator=generators[rank], dtype=torch.float32)
            if pattern == "small":
                values.mul_(1e-5)
            elif pattern == "high_dynamic_range":
                values[::BLOCK] = 1e5 if rank == 0 else -1e5
                values[BLOCK // 2::BLOCK] = -1e3 if rank == 0 else 1e3
            hosts[rank][start:start + count].copy_(values)
    for rank, stream in enumerate(streams):
        with torch.cuda.device(stream.device), torch.cuda.stream(stream):
            buffers["input"][rank].copy_(hosts[rank], non_blocking=True)
    wait_events(_record_events(torch, streams), end)


def _path(torch, nccl, kernels, buffers, streams, kind, end):
    _deadline(end)
    before = time.perf_counter()
    for event, stream in zip(buffers["start_events"], streams):
        event.record(stream)
    if kind == "baseline":
        nccl.all_reduce(buffers["input"], buffers["baseline"], streams=streams)
    else:
        pack, unpack_sum = kernels
        spec = buffers["layout"]
        for rank, stream in enumerate(streams):
            with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                pack[(spec["blocks"],)](
                    buffers["input"][rank], buffers["packed_i8"][rank], buffers["packed"][rank],
                    N=spec["elements"], FLOOR=SCALE_FLOOR, num_warps=4, enable_fp_fusion=False,
                )
        nccl.all_gather(buffers["packed_u8"], buffers["gathered_u8"], streams=streams)
        for rank, stream in enumerate(streams):
            with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                unpack_sum[(spec["blocks"],)](
                    buffers["gathered_i8"][rank], buffers["gathered"][rank], buffers["candidate"][rank],
                    N=spec["elements"], BLOCKS=spec["blocks"], num_warps=4, enable_fp_fusion=False,
                )
    for event, stream in zip(buffers["stop_events"], streams):
        event.record(stream)
    wait_events(buffers["stop_events"], end)
    completion_ms = (time.perf_counter() - before) * 1000
    device_ms = [
        start.elapsed_time(stop) for start, stop in zip(buffers["start_events"], buffers["stop_events"])
    ]
    if not all(math.isfinite(value) and value > 0 for value in [*device_ms, completion_ms]):
        raise ComponentFault("nonfinite or nonpositive complete-path timing")
    return {"device_ms": device_ms, "host_completion_ms": completion_ms}


def _numerics(torch, buffers, streams, end):
    _copy_to_host(torch, [
        [(buffers["host_" + name][r], buffers[name][r]) for name in ("baseline", "candidate", "packed")]
        for r in range(2)
    ], streams, end)
    if not torch.equal(buffers["host_candidate"][0].view(torch.int16), buffers["host_candidate"][1].view(torch.int16)):
        raise ComponentFault("rank candidate outputs differ bitwise")
    if not torch.equal(buffers["host_baseline"][0].view(torch.int16), buffers["host_baseline"][1].view(torch.int16)):
        raise ComponentFault("rank baseline outputs differ bitwise")
    n = buffers["layout"]["elements"]
    error2 = input2 = reference2 = 0.0
    max_abs = max_input = max_bound_ratio = 0.0
    for start in range(0, n, CHUNK_ELEMENTS):
        _deadline(end)
        stop = min(start + CHUNK_ELEMENTS, n)
        # Every requested shape and chunk contains an integral number of blocks.
        first, last = start // BLOCK, stop // BLOCK
        xs, scales, decoded = [], [], []
        for rank in range(2):
            x = buffers["host_input"][rank][start:stop].float().reshape(-1, BLOCK)
            maximum = x.abs().amax(dim=1)
            # A tensor divisor prevents the CPU scalar-divide reciprocal shortcut.
            scale = (maximum / torch.full_like(maximum, 127.0)).clamp_min(SCALE_FLOOR)
            scaled = x / scale[:, None]
            codes = (torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)).clamp(-127, 127).to(torch.int8)
            packed = buffers["host_packed"][rank].reshape(-1, 65)[first:last]
            actual_scales = packed[:, 64]
            actual_codes = packed.view(torch.int8).reshape(-1, BLOCK_BYTES)[:, :BLOCK]
            if not torch.isfinite(x).all() or not torch.isfinite(actual_scales).all():
                raise ComponentFault("nonfinite input or packed scale")
            if not torch.equal(scale, actual_scales) or not torch.equal(codes, actual_codes):
                raise ComponentFault("GPU codes or FP32 scales disagree with exact CPU codec reference")
            xs.append(x)
            scales.append(scale)
            decoded.append(codes.float() * scale[:, None])
        reference = xs[0].double() + xs[1].double()
        expected_baseline = (xs[0] + xs[1]).to(torch.bfloat16).flatten()
        baseline = buffers["host_baseline"][0][start:stop]
        if not torch.equal(baseline.view(torch.int16), expected_baseline.view(torch.int16)):
            raise ComponentFault("BF16 baseline disagrees with correctly rounded CPU sum")
        decoded_sum = decoded[0] + decoded[1]
        expected_codec = decoded_sum.to(torch.bfloat16).flatten()
        actual = buffers["host_candidate"][0][start:stop]
        if not torch.isfinite(actual).all():
            raise ComponentFault("nonfinite candidate output")
        if not torch.equal(actual.view(torch.int16), expected_codec.view(torch.int16)):
            raise ComponentFault("candidate output disagrees bitwise with CPU FP32-decode/BF16 reference")
        error = actual.double().reshape(-1, BLOCK) - reference
        # Quantization half-step + BF16 round-to-nearest + conservative FP32 arithmetic allowance.
        bound = (0.50001 * (scales[0].double() + scales[1].double())[:, None]
                 + decoded_sum.double().abs() * (2.0**-8)
                 + 4 * (2.0**-23) * (decoded[0].double().abs() + decoded[1].double().abs())
                 + 2.0**-133)
        if (error.abs() > bound).any():
            raise ComponentFault("candidate exceeds analytical quantization and rounding bound")
        error2 += error.square().sum().item()
        input2 += (xs[0].double().square() + xs[1].double().square()).sum().item()
        reference2 += reference.square().sum().item()
        max_abs = max(max_abs, error.abs().max().item())
        max_input = max(max_input, xs[0].abs().max().item(), xs[1].abs().max().item())
        max_bound_ratio = max(max_bound_ratio, (error.abs() / bound).max().item())
    metrics = normalized_errors(error2, input2, reference2, n)
    result = {
        "rank_outputs_bitwise_equal": True,
        "baseline_matches_cpu_bf16_sum": True,
        "quantization_codes_and_scales_exact": True,
        "codec_output_matches_cpu_bitwise": True,
        "analytical_error_bound_passed": True,
        "max_absolute_error": max_abs,
        **metrics,
        "normalization": "norm(error) / sqrt(norm(input0)^2 + norm(input1)^2)",
        "max_absolute_input": max_input,
        "max_analytical_bound_fraction": max_bound_ratio,
        "screen_limit": SCREEN_LIMIT,
        "screen_passed": metrics["input_normalized_rms_error"] <= SCREEN_LIMIT,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ComponentFault("nonfinite error metric")
    return result


def _summarize_pairs(pairs):
    medians = {}
    for kind in ("baseline", "candidate"):
        medians[kind] = {
            "device_ms": [statistics.median(pair[kind]["device_ms"][r] for pair in pairs) for r in range(2)],
            "host_completion_ms": statistics.median(pair[kind]["host_completion_ms"] for pair in pairs),
        }
    return {
        "pairs": pairs,
        "medians": medians,
        "device_speedup_fraction": [
            1 - medians["candidate"]["device_ms"][r] / medians["baseline"]["device_ms"][r] for r in range(2)
        ],
        "host_speedup_fraction": 1 - medians["candidate"]["host_completion_ms"] / medians["baseline"]["host_completion_ms"],
    }


def run(torch, triton, devices, end, output_path):
    """Execute one finite component gate on two caller-verified physical devices."""
    path = Path(output_path)
    record = {
        "schema_version": 1,
        "scope": "synthetic_eager_rccl_component_only",
        "status": "running",
        "stage": "preflight",
        "devices": list(devices),
        "runtime": {"torch": str(torch.__version__), "hip": str(torch.version.hip),
                    "triton": str(triton.__version__), "collective": "torch.cuda.nccl backed by RCCL on ROCm"},
        "schedule": schedule_contract(),
        "precision_acceptance_implied": False,
        "serving_benefit_claimed": False,
        "graph": {"status": "not_run", "captures": 0, "replays": 0,
                  "blocker": "pinned multi-device complete-path capture contract not yet established"},
        "memory": {
            "largest_shape_layout": layout(ROWS[-1] * HIDDEN),
            "largest_shape_pinned_host_bytes": 2 * (3 * ROWS[-1] * HIDDEN * 2 + layout(ROWS[-1] * HIDDEN)["packed_bytes_per_rank"]),
            "device_tensor_limit_bytes_each": MAX_DEVICE_TENSOR_BYTES,
            "device_tensor_limit_bytes_aggregate": MAX_TOTAL_TENSOR_BYTES,
            "host_tensor_budget_bytes": MAX_HOST_TENSOR_BYTES,
            "excludes": "framework/driver contexts, JIT compiler, RCCL internal workspace and process RSS",
            "observed_torch_allocated_bytes_each": [0, 0],
        },
        "timing_contract": "full eager quantize/all_gather/dequantize versus BF16 all_reduce; event time per device and host completion including event polling; CPU validation excluded",
        "numerical": [],
        "timings": [],
    }
    _write(path, record)
    try:
        if len(devices) != 2 or any(type(device) is not int or device < 0 for device in devices) or len(set(devices)) != 2:
            raise ValueError("exactly two distinct caller-verified integer device ordinals are required")
        if not isinstance(end, (float, int)) or not math.isfinite(end):
            raise ValueError("end must be a finite monotonic deadline")
        _deadline(end)
        missing = [name for name in ("_nccl_all_reduce", "_nccl_all_gather") if not hasattr(torch._C, name)]
        if missing:
            record.update(status="inconclusive", reason="installed native backend lacks " + ", ".join(missing))
            _write(path, record)
            return record
        nccl = torch.cuda.nccl
        streams = [torch.cuda.Stream(device=device) for device in devices]
        record["stage"] = "native_feasibility"
        _write(path, record)
        try:
            record["feasibility"] = _feasibility(torch, nccl, devices, streams, end)
        except RuntimeError as exc:
            if not isinstance(exc, ComponentFault) and _known_gate_blocker(exc):
                record.update(status="inconclusive", reason="native collective feasibility blocked", native_reason=str(exc)[:2048])
                _write(path, record)
                return record
            raise
        module = sys.modules.get("kernels")
        if module is None:
            raise ValueError("caller must preload authenticated kernels module")
        kernels = module.build(triton)
        for rows in ROWS:
            _deadline(end)
            buffers = _allocate(torch, devices, streams, rows * HIDDEN)
            allocated = [torch.cuda.memory_allocated(device) for device in devices]
            record["memory"]["observed_torch_allocated_bytes_each"] = [
                max(old, value) for old, value in zip(record["memory"]["observed_torch_allocated_bytes_each"], allocated)
            ]
            if any(value > MAX_DEVICE_TENSOR_BYTES for value in allocated) or sum(allocated) > MAX_TOTAL_TENSOR_BYTES:
                raise ComponentFault("observed live Torch allocation exceeds declared tensor budget")
            for pattern in PATTERNS:
                record["stage"] = f"numerical/{rows}/{pattern}"
                _fill_inputs(torch, buffers, pattern, streams, end)
                _path(torch, nccl, kernels, buffers, streams, "baseline", end)
                _path(torch, nccl, kernels, buffers, streams, "candidate", end)
                numerical = _numerics(torch, buffers, streams, end)
                record["numerical"].append({"shape": [rows, HIDDEN], "pattern": pattern, **numerical})
                _write(path, record)
            record["stage"] = f"timing/{rows}"
            _fill_inputs(torch, buffers, "normal", streams, end)
            pairs = []
            for pair_index in range(WARMUPS + OBSERVATIONS):
                order = ("baseline", "candidate") if pair_index % 2 == 0 else ("candidate", "baseline")
                pair = {"order": list(order)}
                for kind in order:
                    pair[kind] = _path(torch, nccl, kernels, buffers, streams, kind, end)
                if pair_index >= WARMUPS:
                    pairs.append(pair)
            record["timings"].append({"shape": [rows, HIDDEN], **_summarize_pairs(pairs)})
            _write(path, record)
            del buffers
        passed = all(case["screen_passed"] for case in record["numerical"])
        speedups = record["timings"][-1]["device_speedup_fraction"]
        useful = all(value >= BENEFIT_THRESHOLD for value in speedups)
        record["decision"] = {
            "numerical_screen_passed": passed,
            "input_normalized_rms_limit": SCREEN_LIMIT,
            "minimum_device_speedup_fraction_at_2048": BENEFIT_THRESHOLD,
            "device_speedup_fraction_at_2048": speedups,
            "useful_component_threshold_met": useful,
            "host_speedup_fraction_at_2048": record["timings"][-1]["host_speedup_fraction"],
            "meaning": "component evidence only; model activation precision and serving path remain untested",
        }
        record.update(status="accepted" if passed and useful else "rejected", stage="complete")
        problems = validate_result(record)
        if problems:
            raise ComponentFault("internal result validation failed: " + "; ".join(problems))
        _write(path, record)
        return record
    except Exception as exc:
        record.update(status="hard_fault", error_type=type(exc).__name__, error=str(exc)[:2048])
        _write(path, record)
        raise


def validate_result(record):
    """Return schema/decision problems without importing Torch or Triton."""
    try:
        return _validate_result(record)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return ["malformed result structure: " + type(exc).__name__]


def _validate_result(record):
    problems = []
    if not isinstance(record, dict):
        return ["result is not an object"]
    if record.get("schema_version") != 1:
        problems.append("unknown schema version")
    if record.get("scope") != "synthetic_eager_rccl_component_only":
        problems.append("incorrect experiment scope")
    devices = record.get("devices", [])
    if len(devices) != 2 or not all(type(device) is int and device >= 0 for device in devices) or len(set(devices)) != 2:
        problems.append("invalid physical device ordinals")
    status = record.get("status")
    if status not in STATUSES:
        problems.append("unknown or unfinished outcome status")
    if record.get("schedule") != schedule_contract():
        problems.append("finite schedule differs from declared contract")
    if record.get("precision_acceptance_implied") is not False or record.get("serving_benefit_claimed") is not False:
        problems.append("unsupported precision or serving claim")
    graph = record.get("graph", {})
    if graph.get("status") != "not_run" or graph.get("captures") != 0 or graph.get("replays") != 0:
        problems.append("graph execution differs from eager-only contract")
    memory = record.get("memory", {})
    if memory.get("largest_shape_layout") != layout(ROWS[-1] * HIDDEN):
        problems.append("incorrect memory layout bound")
    if memory.get("device_tensor_limit_bytes_each") != MAX_DEVICE_TENSOR_BYTES or memory.get("device_tensor_limit_bytes_aggregate") != MAX_TOTAL_TENSOR_BYTES:
        problems.append("incorrect device tensor budget")
    if memory.get("host_tensor_budget_bytes") != MAX_HOST_TENSOR_BYTES:
        problems.append("incorrect host tensor budget")
    observed = memory.get("observed_torch_allocated_bytes_each", [])
    if len(observed) != 2 or not all(type(value) is int and value >= 0 for value in observed):
        problems.append("invalid observed allocation sizes")
    elif status != "hard_fault" and (max(observed) > MAX_DEVICE_TENSOR_BYTES or sum(observed) > MAX_TOTAL_TENSOR_BYTES):
        problems.append("observed live tensors exceeded memory budget")
    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        return True
    if not finite_tree(record):
        problems.append("nonfinite number in result")
    if status == "inconclusive" and not record.get("reason"):
        problems.append("inconclusive outcome lacks reason")
    if status == "hard_fault" and not record.get("error"):
        problems.append("hard fault lacks error")
    if status in ("accepted", "rejected"):
        if record.get("stage") != "complete" or record.get("feasibility", {}).get("status") != "passed":
            problems.append("completed outcome lacks successful native feasibility")
        if any(record.get("feasibility", {}).get(key) is not True for key in ("bf16_sum_exact", "uint8_gather_exact")):
            problems.append("native feasibility lacks exact BF16 and uint8 checks")
        cases = record.get("numerical", [])
        keys = [(tuple(case.get("shape", [])), case.get("pattern")) for case in cases]
        expected = [((rows, HIDDEN), pattern) for rows in ROWS for pattern in PATTERNS]
        if keys != expected:
            problems.append("incomplete or reordered numerical coverage")
        correctness_fields = ("rank_outputs_bitwise_equal", "baseline_matches_cpu_bf16_sum",
                              "quantization_codes_and_scales_exact", "codec_output_matches_cpu_bitwise",
                              "analytical_error_bound_passed")
        for case in cases:
            if any(case.get(key) is not True for key in correctness_fields):
                problems.append("numerical case lacks exact correctness checks")
                break
            value = case.get("input_normalized_rms_error")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                problems.append("invalid normalized error")
                break
            if case.get("screen_limit") != SCREEN_LIMIT or case.get("screen_passed") is not (value <= SCREEN_LIMIT):
                problems.append("inconsistent numerical screen")
                break
            for name in ("max_absolute_error", "absolute_rms_error", "max_absolute_input", "max_analytical_bound_fraction"):
                metric = case.get(name)
                if not isinstance(metric, (float, int)) or isinstance(metric, bool) or metric < 0:
                    problems.append("missing or invalid numerical diagnostic " + name)
            relative = case.get("relative_output_l2")
            if "relative_output_l2" not in case or (relative is not None and (not isinstance(relative, (float, int)) or isinstance(relative, bool) or relative < 0)):
                problems.append("missing or invalid relative output diagnostic")
            if isinstance(case.get("max_analytical_bound_fraction"), (float, int)) and case["max_analytical_bound_fraction"] > 1:
                problems.append("analytical error bound exceeded")
        timings = record.get("timings", [])
        if [item.get("shape") for item in timings] != [[rows, HIDDEN] for rows in ROWS]:
            problems.append("incomplete timing coverage")
        for item in timings:
            pairs = item.get("pairs", [])
            if len(pairs) != OBSERVATIONS:
                problems.append("incorrect timing observation count")
                continue
            valid = True
            for index, pair in enumerate(pairs):
                expected_order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
                if pair.get("order") != expected_order:
                    problems.append("timing order did not alternate")
                for kind in ("baseline", "candidate"):
                    entry = pair.get(kind, {})
                    times = entry.get("device_ms", [])
                    host = entry.get("host_completion_ms")
                    if len(times) != 2 or not all(isinstance(t, (float, int)) and not isinstance(t, bool) and t > 0 for t in [*times, host]):
                        valid = False
            if not valid:
                problems.append("invalid complete-path timing")
            elif any(item.get(key) != value for key, value in _summarize_pairs(pairs).items()):
                problems.append("timing summary differs from observations")
        decision = record.get("decision", {})
        if cases and len(timings) == len(ROWS):
            passed = all(case.get("screen_passed") is True for case in cases)
            speedups = timings[-1].get("device_speedup_fraction", [])
            useful = len(speedups) == 2 and all(isinstance(t, (float, int)) and t >= BENEFIT_THRESHOLD for t in speedups)
            if decision.get("numerical_screen_passed") is not passed or decision.get("useful_component_threshold_met") is not useful:
                problems.append("decision does not match numerical and timing evidence")
            if decision.get("input_normalized_rms_limit") != SCREEN_LIMIT or decision.get("minimum_device_speedup_fraction_at_2048") != BENEFIT_THRESHOLD:
                problems.append("decision thresholds changed")
            if decision.get("device_speedup_fraction_at_2048") != speedups:
                problems.append("decision speedup differs from measured evidence")
            if (status == "accepted") != (passed and useful):
                problems.append("outcome status disagrees with component decision")
    return problems
