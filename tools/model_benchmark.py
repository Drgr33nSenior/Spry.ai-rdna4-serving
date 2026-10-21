"""Private, explicitly reviewed TP2 timing screen; never a serving benchmark.

No hardware or serving imports occur unless the execution flag is supplied.
The outer owner must authenticate the runtime, model, spec and two devices.
Both arms use the same private composition; only KV allocation/reader changes.
"""
import argparse
from array import array
import cProfile
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import re
import signal
import stat
import sys
import time


SCOPE = "reviewed-private-tp2-model-benchmark-v1"
CELLS = [("C1-128", 1, 128), ("C1-1024", 1, 1024),
         ("C1-3966", 1, 3966), ("C2-128", 2, 128), ("C2-1024", 2, 1024)]
CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
BINARY_SHA256 = "d09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171"
GPU_IDENTITY_SHA256 = "ccd819f15d2df5e13acd1f7575d6a296be6ac5c9f1096658d34894e9cf3989bb"
PHYSICAL_DEVICES = {"0000:c3:00.0": "39373864-6465-3038-6163-303837666533",
                    "0000:23:00.0": "34663466-6561-3362-3832-346261366439"}
PROFILE_CONFIG = {"schema": 1, "mode": "bounded-attribution-v1"}
PROFILE_METHOD = "cProfile-wall-ns+torch-profiler-cpu-cuda-v1"
PROFILE_WINDOW_LIMIT = 256 << 10
PROFILE_LIMIT = 1 << 20
PROFILE_EVENT_LIMIT = 8192
PROFILE_LABELS = ("spry.forward_batch_init", "spry.model_forward", "spry.sample")
# Fixed unchanged sources from the authenticated serving composition. These are
# attribution targets, not a permissive filename-suffix filter.
PROFILE_SOURCES = {
    "srt/layers/attention/ultraquant_backend.py": (
        "db5e094980b84c27e3d5640ba7f1df1f3c83e18cd5b261c7a805858a1feabc24",
        ("Native.call", "QwenBF16MixedPool.set_kv_buffer",
         "QwenBF16SourceAdapter.init_forward_metadata_out_graph",
         "QwenBF16SourceAdapter._forward", "QwenBF16SourceAdapter.prepare",
         "QwenBF16SourceAdapter.attention")),
    "srt/layers/attention/ultraquant_evidence.py": (
        "8d90aedeb10f22e13cf83fe146dad1d2f50a8f0be0a3d831f0d1ce16dcee1fb8",
        ("BF16CacheOwner.validate_snapshot", "BF16CacheOwner.check_writable",
         "BF16CacheOwner.cached_slots", "BF16CacheOwner._slots")),
    "srt/model_executor/model_runner.py": (
        "d8ce6f38279913d11b4a7e910a7fd1432e952a0bdcf33ebe726b6f49af0ba6ba",
        ("ModelRunner.forward", "ModelRunner.sample")),
    "srt/model_executor/forward_batch_info.py": (
        "ef4426b3ece1b515eeb5223d394de6a8d154d912bcf62e7ed4bae20e17064797",
        ("ForwardBatch.init_new",)),
}
PROFILE_PACKAGE_ROOT = "/mnt/composition/image-source/python/sglang"


def profile_selected(item, step):
    """Four existing forwards, never extra requests or warmups."""
    return item["cell"] in ("C1-128", "C1-3966") and item["sample"] == 1 and step in (0, 1)


def profile_sources(spec):
    root = Path(spec["package_root"])
    if str(root) != PROFILE_PACKAGE_ROOT:
        raise ValueError("profile canonical package identity")
    rows = []
    for relative, (pin, names) in PROFILE_SOURCES.items():
        path = root / relative
        if path.resolve() != path:
            raise ValueError("profile source link/canonical identity")
        raw = checked_file(path, pin)
        rows.append({"path": str(path), "sha256": pin, "bytes": len(raw),
                     "qualnames": list(names)})
    return rows


def profile_cpu_rows(stats, sources):
    allowed = {(source["path"], name): source["sha256"]
               for source in sources for name in source["qualnames"]}
    rows = []
    for entry in stats:
        code = entry.code
        if not hasattr(code, "co_filename"):
            continue  # C calls are included in the owning Python inclusive wall.
        key = (code.co_filename, code.co_qualname)
        if key in allowed:
            rows.append({"path": key[0], "qualname": key[1], "sha256": allowed[key],
                "calls": entry.callcount, "recursive_calls": entry.reccallcount,
                "inclusive_seconds": entry.totaltime, "exclusive_seconds": entry.inlinetime})
    return sorted(rows, key=lambda row: (row["path"], row["qualname"]))


def profile_event_category(name):
    if re.search(r"(?:bf16_v2::|_ZN7bf16_v2\d*)validate_(?:read|write)", name):
        return "kv_validation"
    if re.search(r"(?:bf16_v2::|_ZN7bf16_v2\d*)read(?:\(|E)", name):
        return "kv_attention"
    if re.search(r"(?:bf16_v2::|_ZN7bf16_v2\d*)write(?:\(|E)", name):
        return "kv_encode"
    lowered = name.lower()
    if "memcpy" in lowered or "memset" in lowered:
        return "copy_or_memset"
    if "nccl" in lowered or "rccl" in lowered or "allreduce" in lowered:
        return "collective"
    if "gdn" in lowered or "mamba" in lowered or "gated_delta" in lowered:
        return "gdn_or_mamba"
    if any(word in lowered for word in ("gemm", "matmul", "igemm")):
        return "gemm"
    return "other"


def profile_has_kernel(names, symbol):
    return any(re.search(r"(?:bf16_v2::|_ZN7bf16_v2\d*)" + re.escape(symbol) + r"(?:\(|E)", name)
               for name in names)


class ProfileGPUEventError(ValueError):
    """A rejected event carries only bounded JSON data, never native objects."""
    def __init__(self, rejection):
        super().__init__("GPU event identity/time")
        self.rejection = rejection


def profile_rejection_prefix(value, limit=512):
    # Slice before encoding, including for oversized or ill-formed names.
    return value[:limit].encode("utf-8", "backslashreplace")[:limit].decode("utf-8", "ignore")


def profile_rejection_value(value):
    """Describe a scalar without repr callbacks or non-finite JSON numbers."""
    result = {"type": type(value).__name__[:64]}
    if value is None or type(value) is bool:
        result["value"] = value
    elif type(value) is int:
        if value.bit_length() <= 1024:
            result["value"] = value
        else:
            result.update(value="integer outside diagnostic bound", bits=value.bit_length())
    elif type(value) is float:
        result["value"] = value if math.isfinite(value) else (
            "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity")
    elif type(value) is str:
        prefix = profile_rejection_prefix(value)
        result.update(prefix=prefix, prefix_utf8_bytes=len(prefix.encode("utf-8")),
                      prefix_truncated=prefix != value, characters=len(value))
    return result


def profile_gpu_rows(events, device_index):
    """Retain every GPU interval and a deduplicated name table, or refuse."""
    names, rows = [], []
    cuda_index = 0
    for event_index, event in enumerate(events):
        values, unavailable = {}, {}
        def read(field, getter):
            try:
                values[field] = getter()
            except Exception as error:
                unavailable[field] = {"type": type(error).__name__[:64],
                    "message_prefix": profile_rejection_prefix(str(error), 256)}
        read("device_type", lambda: event.device_type)
        if "device_type" in values:
            read("device_type_label", lambda: str(values["device_type"]).split(".")[-1])
        if values.get("device_type_label") not in (None, "CUDA"):
            continue
        observed_cuda_index = cuda_index if values.get("device_type_label") == "CUDA" else None
        read("name", lambda: event.name)
        read("device_index", lambda: event.device_index)
        read("start", lambda: event.time_range.start)
        read("end", lambda: event.time_range.end)
        reasons = [field + "_unavailable" for field in unavailable]
        name, device = values.get("name"), values.get("device_index")
        start, end = values.get("start"), values.get("end")
        if "name" in values:
            if type(name) is not str:
                reasons.append("name_type")
            elif not name:
                reasons.append("name_empty")
            else:
                try:
                    if len(name) > 16384 or len(name.encode("utf-8")) > 16384:
                        reasons.append("name_utf8_bytes_exceeds_16384")
                except UnicodeEncodeError:
                    reasons.append("name_utf8_encoding")
        if "device_index" in values:
            if type(device) is not int:
                reasons.append("device_index_type")
            elif device != device_index:
                reasons.append("device_index_mismatch")
        finite = {}
        for field in ("start", "end"):
            value = values.get(field)
            finite[field] = None
            if field not in values:
                continue
            if type(value) not in (int, float):
                reasons.append(field + "_type")
                continue
            try:
                finite[field] = math.isfinite(value)
            except OverflowError:
                finite[field] = False
            if not finite[field]:
                reasons.append(field + "_nonfinite")
            elif value < 0:
                reasons.append(field + "_negative")
        ordered = end >= start if all(finite.values()) else None
        if ordered is False:
            reasons.append("end_before_start")
        if reasons:
            observed = {field: {"available": True, **profile_rejection_value(value)}
                        for field, value in values.items()}
            observed.update({field: {"available": False, "error": error}
                             for field, error in unavailable.items()})
            for field in ("start", "end"):
                observed[field]["finite"] = finite[field]
            raise ProfileGPUEventError({"event_index": event_index,
                "cuda_event_index": observed_cuda_index, "index_base": 0,
                "expected_device_index": device_index, "reasons": reasons,
                "observed": observed, "end_at_or_after_start": ordered})
        cuda_index += 1
        if name not in names:
            names.append(name)
        duration = end - start
        rows.append([names.index(name), start, duration, device])
        if len(rows) > PROFILE_EVENT_LIMIT:
            raise ValueError("GPU event count bound; no truncation permitted")
    if not rows:
        raise ValueError("GPU activity unavailable or empty; CPU fallback refused")
    return {"names": names, "intervals": rows, "categories": profile_categories(names, rows),
            "all_gpu_events_retained": True, "interval_unit": "microseconds",
            "interval_clock": "torch-profiler-relative", "duration_sum_is_wall_time": False}


def profile_categories(names, intervals):
    summaries = {}
    for name_index, _, duration, _ in intervals:
        category = profile_event_category(names[name_index])
        total = summaries.setdefault(category, {"events": 0, "duration_us": 0.0})
        total["events"] += 1
        total["duration_us"] += duration
        if not math.isfinite(total["duration_us"]):
            raise ValueError("GPU category duration overflow")
    return summaries


def profile_binding(item, ordinal, step, rank, arm, session_sha256, spec_sha256):
    return {"schema": 1, "mode": PROFILE_CONFIG["mode"], "rank": rank, "arm": arm,
        "pid": os.getpid(), "session_sha256": session_sha256, "spec_sha256": spec_sha256,
        "ordinal": ordinal, "cell": item["cell"], "sample": item["sample"],
        "request_ids": item["request_ids"], "step": step,
        "phase": "prefill" if step == 0 else "decode", "qualification": False}


def profiled_forward(runner, batch, forward_batch, torch, binding, sources, destination):
    """Observers surround exactly one existing forward; errors retain partial data."""
    row = dict(binding, method=PROFILE_METHOD, instrumented_timings=True,
        complete=False, errors=[], cpu=[], gpu=None, observer_restored=False,
        monotonic_start=time.monotonic(), monotonic_end=None)
    destination.append(row)
    py_profile = gpu_profile = None
    py_started = gpu_started = False
    original_error = None
    try:
        if sys.getprofile() is not None:
            raise ValueError("pre-existing CPU observer; profile replacement refused")
        activities = torch.profiler.supported_activities()
        if torch.profiler.ProfilerActivity.CUDA not in activities:
            raise ValueError("GPU activity unsupported; CPU fallback refused")
        gpu_profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False, with_flops=False,
            with_modules=False)
        py_profile = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-9)
        gpu_started = True
        gpu_profile.__enter__()
        py_started = True
        py_profile.enable()
        answer = timed_forward(runner, batch, forward_batch, torch, profile_labels=True)
    except BaseException as error:
        original_error = error
        row["errors"].append(type(error).__name__ + ": " + str(error)[:512])
        raise
    finally:
        observer_errors = []
        if py_started:
            try:
                py_profile.disable()
            except BaseException as error:
                observer_errors.append("CPU stop: " + type(error).__name__ + ": " + str(error)[:256])
        if gpu_started:
            try:
                gpu_profile.__exit__(type(original_error) if original_error else None,
                                     original_error, original_error.__traceback__ if original_error else None)
            except BaseException as error:
                observer_errors.append("GPU stop: " + type(error).__name__ + ": " + str(error)[:256])
        row["observer_restored"] = sys.getprofile() is None and not observer_errors
        try:
            if py_profile is not None:
                row["cpu"] = profile_cpu_rows(py_profile.getstats(), sources)
            if gpu_profile is not None and gpu_started:
                try:
                    events = gpu_profile.events()
                except Exception as error:
                    row["gpu_collection_failure"] = {"stage": "events", "events_available": False,
                        "expected_device_index": binding["rank"], "error": {
                            "type": type(error).__name__[:64],
                            "message_prefix": profile_rejection_prefix(str(error), 256)}}
                    raise
                row["gpu"] = profile_gpu_rows(events, binding["rank"])
        except BaseException as error:
            if isinstance(error, ProfileGPUEventError):
                row["gpu_event_rejection"] = error.rejection
            observer_errors.append("collection: " + type(error).__name__ + ": " + str(error)[:256])
        row["monotonic_end"] = time.monotonic()
        row["errors"].extend(observer_errors)
        row["complete"] = not row["errors"] and row["observer_restored"] and row["gpu"] is not None
        try:
            oversized = (len(json.dumps(row, sort_keys=True, allow_nan=False).encode()) > PROFILE_WINDOW_LIMIT
                or len(json.dumps(destination, sort_keys=True, allow_nan=False).encode()) > PROFILE_LIMIT - (16 << 10))
        except (TypeError, ValueError) as error:
            oversized = True
            row["errors"].append("profile serialization: " + type(error).__name__ + ": " + str(error)[:256])
        if oversized:
            # Retain a bounded explicit failure, never a truncated success.
            row.update(complete=False, gpu=None, cpu=[])
            row["errors"].append("profile window exceeded 256 KiB; oversized observations rejected")
        if original_error is None and not row["complete"]:
            raise ValueError("incomplete attribution: " + "; ".join(row["errors"])[:1024])
    return answer


def validate_profile_records(records, expected_arm, *, session_sha256=None, spec_sha256=None):
    """The parent and independent reader share strict completion/identity gates."""
    if expected_arm not in ("bf16", "kv4") or len(records) != 2:
        raise ValueError("profile arm/rank inventory")
    for rank, record in enumerate(records):
        if type(record) is not dict or record.get("rank") != rank or record.get("arm") != expected_arm:
            raise ValueError("profile rank identity")
        identity = record.get("native_identity", {})
        validate_rank_identity(identity, rank=rank, arm=expected_arm,
            session_sha256=session_sha256 if session_sha256 is not None else identity.get("session_sha256"),
            spec_sha256=spec_sha256 if spec_sha256 is not None else identity.get("spec_sha256"))
        expected_pid = identity.get("process", {}).get("pid")
        profile = record.get("diagnostic_profile")
        if (type(profile) is not dict or profile.get("schema") != 1
                or profile.get("mode") != PROFILE_CONFIG["mode"] or profile.get("method") != PROFILE_METHOD
                or profile.get("complete") is not True or profile.get("qualification") is not False
                or record.get("instrumented_timings") is not True or record.get("profile_method") != PROFILE_METHOD
                or len(json.dumps(profile, sort_keys=True, allow_nan=False).encode()) > PROFILE_LIMIT
                or type(profile.get("sources")) is not list or len(profile["sources"]) != len(PROFILE_SOURCES)
                or type(profile.get("windows")) is not list or len(profile["windows"]) != 4
                or not valid_digest(profile.get("entry_sha256"))
                or profile["entry_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or profile.get("native_binary_sha256") != BINARY_SHA256
                or type(expected_pid) is not int or expected_pid <= 0):
            raise ValueError("profile incomplete/source/output binding")
        source_map = {}
        for source, (relative, (pin, qualnames)) in zip(profile["sources"], PROFILE_SOURCES.items()):
            path = PROFILE_PACKAGE_ROOT + "/" + relative
            if (type(source) is not dict or set(source) != {"path", "sha256", "bytes", "qualnames"}
                    or source["path"] != path or source["sha256"] != pin
                    or type(source["bytes"]) is not int or source["bytes"] <= 0
                    or source["qualnames"] != list(qualnames)):
                raise ValueError("profile source identity")
            source_map.update({(path, name): pin for name in qualnames})
        expected_windows = [(ordinal, item, step) for ordinal, item in enumerate(request_plan({}))
                            for step in (0, 1) if profile_selected(item, step)]
        previous_end = -1.0
        for window, (ordinal, item, step) in zip(profile["windows"], expected_windows):
            expected = profile_binding(item, ordinal, step, rank, expected_arm,
                session_sha256 if session_sha256 is not None else identity.get("session_sha256"),
                spec_sha256 if spec_sha256 is not None else identity.get("spec_sha256"))
            expected["pid"] = expected_pid
            if (type(window) is not dict or any(type(window.get(key)) is not type(value)
                    or window[key] != value for key, value in expected.items())
                    or not valid_digest(window["session_sha256"]) or not valid_digest(window["spec_sha256"])
                    or window.get("complete") is not True or window.get("observer_restored") is not True
                    or window.get("errors") != [] or window.get("method") != PROFILE_METHOD
                    or window.get("instrumented_timings") is not True
                    or len(json.dumps(window, sort_keys=True, allow_nan=False).encode()) > PROFILE_WINDOW_LIMIT
                    or any(type(window.get(key)) not in (int, float) or not math.isfinite(window[key])
                           or window[key] < 0 for key in ("monotonic_start", "monotonic_end"))
                    or window["monotonic_end"] < window["monotonic_start"]
                    or window["monotonic_start"] < previous_end
                    or type(window.get("cpu")) is not list or not window["cpu"]):
                raise ValueError("profile window completion/identity/time")
            previous_end = window["monotonic_end"]
            seen = set()
            for cpu in window["cpu"]:
                key = (cpu.get("path"), cpu.get("qualname"))
                if (key in seen or key not in source_map or cpu.get("sha256") != source_map[key]
                        or type(cpu.get("calls")) is not int or cpu["calls"] <= 0
                        or type(cpu.get("recursive_calls")) is not int
                        or not 0 <= cpu["recursive_calls"] <= cpu["calls"]
                        or any(type(cpu.get(field)) not in (int, float) or not math.isfinite(cpu[field])
                               or cpu[field] < 0 for field in ("inclusive_seconds", "exclusive_seconds"))
                        or cpu["exclusive_seconds"] > cpu["inclusive_seconds"]):
                    raise ValueError("profile CPU source/count/time")
                seen.add(key)
            required_cpu = {"ForwardBatch.init_new", "ModelRunner.forward", "ModelRunner.sample"}
            if expected_arm == "kv4":
                required_cpu.update({"Native.call", "QwenBF16MixedPool.set_kv_buffer",
                    "QwenBF16SourceAdapter.init_forward_metadata_out_graph",
                    "QwenBF16SourceAdapter._forward", "QwenBF16SourceAdapter.attention",
                    "BF16CacheOwner.validate_snapshot"})
            if not required_cpu.issubset({name for _, name in seen}):
                raise ValueError("profile mandatory CPU calls absent")
            if any(cpu["calls"] != 1 for cpu in window["cpu"] if cpu["qualname"] in
                   ("ForwardBatch.init_new", "ModelRunner.forward", "ModelRunner.sample")):
                raise ValueError("profile must observe one existing model forward/sample")
            gpu = window.get("gpu")
            if (type(gpu) is not dict or type(gpu.get("names")) is not list or not gpu["names"]
                    or any(type(name) is not str or not name or len(name.encode()) > 16384 for name in gpu["names"])
                    or len(set(gpu["names"])) != len(gpu["names"])
                    or type(gpu.get("intervals")) is not list or not 1 <= len(gpu["intervals"]) <= PROFILE_EVENT_LIMIT
                    or gpu.get("all_gpu_events_retained") is not True
                    or gpu.get("interval_unit") != "microseconds"
                    or gpu.get("interval_clock") != "torch-profiler-relative"
                    or gpu.get("duration_sum_is_wall_time") is not False):
                raise ValueError("profile GPU activity inventory")
            used_names = []
            for interval in gpu["intervals"]:
                if (type(interval) is not list or len(interval) != 4
                        or type(interval[0]) is not int or not 0 <= interval[0] < len(gpu["names"])
                        or type(interval[3]) is not int or interval[3] != rank
                        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                               for value in interval[1:3])):
                    raise ValueError("profile GPU interval binding/time")
                if interval[0] not in used_names:
                    used_names.append(interval[0])
            if (used_names != list(range(len(gpu["names"])))
                    or profile_categories(gpu["names"], gpu["intervals"]) != gpu.get("categories")):
                raise ValueError("profile GPU summary/name binding")
            if expected_arm == "kv4" and not all(profile_has_kernel(gpu["names"], name)
                                                for name in ("read", "validate_read")):
                raise ValueError("profile active v2 GPU kernels absent")
    return records


def read_bounded_file(path, maximum):
    """Read one regular no-follow, singly linked file without size/identity races."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute reviewed input required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError("input type/size")
        data = b""
        while len(data) <= maximum:
            block = os.read(fd, min(1048576, maximum + 1 - len(data)))
            if not block:
                break
            data += block
        after = os.fstat(fd)
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                              value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if stamp(before) != stamp(after) or stamp(after) != stamp(os.stat(path, follow_symlinks=False)):
            raise ValueError("input changed")
        if len(data) > maximum:
            raise ValueError("input size")
        return data
    finally:
        os.close(fd)


def checked_file(path, expected, maximum=8 << 20):
    data = read_bounded_file(path, maximum)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("input digest")
    return data


def authenticated_adjacent(filename, expected):
    path = Path(__file__).resolve().with_name(filename)
    raw = checked_file(path, expected, 256 << 10)
    spec = importlib.util.spec_from_file_location("spry_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Execute exactly the authenticated source, never a previously cached pyc.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def valid_digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write_identity(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
    if len(raw) > 64 << 10:
        raise ValueError("native identity output bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


PROGRESS_LIMIT = 64 << 10
PROGRESS_PHASES = ("rank_started", "preflight_complete", "torch_import", "native_identity_complete",
    "serving_imports", "arguments", "model_config", "parallel_runtime", "layer_runtime",
    "model_runner", "memory_pool", "attention_backends", "graphs", "cache", "ready",
    "batch_intent", "batch_complete", "cleanup", "rank_finished")


class RankProgress:
    """Fsynced, bounded phase records outside timed forward/sample operations."""
    def __init__(self, output, rank, arm, session_sha256, spec_sha256):
        self.binding = dict(schema=1, rank=rank, arm=arm, pid=os.getpid(),
            session_sha256=session_sha256, spec_sha256=spec_sha256)
        self.fd = os.open(output / f"rank-{rank}-progress.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.size = self.sequence = 0

    def record(self, phase, **details):
        if phase not in PROGRESS_PHASES:
            raise ValueError("progress phase")
        row = dict(self.binding, sequence=self.sequence, phase=phase,
                   utc_unix=time.time(), monotonic=time.monotonic(), details=details)
        raw = json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n"
        if self.size + len(raw) > PROGRESS_LIMIT:
            raise ValueError("progress output bound")
        offset = 0
        while offset < len(raw):
            written = os.write(self.fd, raw[offset:])
            if written <= 0:
                raise OSError("progress short write")
            offset += written
        os.fsync(self.fd)
        self.size += len(raw)
        self.sequence += 1

    def close(self):
        os.close(self.fd)


def progress_readout(raw, *, rank, arm, session_sha256, spec_sha256):
    """Retain validated prefix counts; absence or a torn tail never means zero work."""
    summary = dict(rank=rank, valid=False, counts_complete=False, finished=False,
        intent_batches=None, completed_batches=None, intent_requests=None,
        completed_requests=None, last_phase=None, pid=None, error=None,
        counts_semantics="validated-prefix-lower-bounds", qualification=False)
    if raw is None:
        summary["error"] = "progress unavailable"
        return summary
    intents = completed = 0
    plans = list(request_plan({}))
    previous_time = -1.0
    ready = finished = False
    startup = [phase for phase in PROGRESS_PHASES[:PROGRESS_PHASES.index("ready") + 1]
               if phase != "native_identity_complete" or session_sha256 is not None]
    startup_index = 0
    cleaning = False
    try:
        if not raw or len(raw) > PROGRESS_LIMIT:
            raise ValueError("progress empty/oversized")
        for sequence, line in enumerate(raw.splitlines(keepends=True)):
            if not line.endswith(b"\n"):
                raise ValueError("partial progress line")
            def pairs(items):
                value = {}
                for key, item in items:
                    if key in value:
                        raise ValueError("duplicate progress key")
                    value[key] = item
                return value
            row = json.loads(line, object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite progress")))
            expected = dict(schema=1, rank=rank, arm=arm, session_sha256=session_sha256,
                            spec_sha256=spec_sha256, sequence=sequence)
            if (type(row) is not dict or set(row) != set(expected) |
                    {"pid", "phase", "utc_unix", "monotonic", "details"} or
                    any(type(row.get(key)) is not type(value) or row[key] != value
                        for key, value in expected.items()) or
                    type(row["pid"]) is not int or row["pid"] <= 0 or
                    (sequence and row["pid"] != summary["pid"]) or
                    row["phase"] not in PROGRESS_PHASES or type(row["details"]) is not dict or
                    any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0
                        for key in ("utc_unix", "monotonic")) or
                    row["monotonic"] < previous_time or finished):
                raise ValueError("progress identity/order/type")
            phase, details = row["phase"], row["details"]
            if (sequence == 0) != (phase == "rank_started"):
                raise ValueError("progress start order")
            if cleaning and phase != "rank_finished":
                raise ValueError("progress after cleanup")
            if phase in startup:
                if startup_index >= len(startup) or phase != startup[startup_index] or intents:
                    raise ValueError("progress startup order")
                startup_index += 1
            elif phase == "native_identity_complete":
                raise ValueError("unexpected native identity phase")
            if phase == "cleanup":
                cleaning = True
            if phase == "ready":
                if ready or intents:
                    raise ValueError("duplicate/late readiness")
                ready = True
            if phase in ("batch_intent", "batch_complete"):
                ordinal = intents if phase == "batch_intent" else completed
                if (not ready or ordinal >= len(plans) or
                        json.dumps(details, sort_keys=True) != json.dumps(dict(plans[ordinal], ordinal=ordinal), sort_keys=True) or
                        (phase == "batch_intent" and intents != completed) or
                        (phase == "batch_complete" and intents != completed + 1)):
                    raise ValueError("progress batch binding/order")
                if phase == "batch_intent":
                    intents += 1
                else:
                    completed += 1
            elif details and phase != "rank_finished":
                raise ValueError("unexpected phase details")
            if phase == "rank_finished":
                if not cleaning or set(details) != {"success"} or type(details["success"]) is not bool:
                    raise ValueError("progress finish record")
                if details["success"] and (intents != 30 or completed != 30):
                    raise ValueError("premature successful finish")
                finished = True
                summary["finished"] = details["success"]
            previous_time = row["monotonic"]
            summary.update(last_phase=phase, pid=row["pid"], intent_batches=intents,
                completed_batches=completed,
                intent_requests=sum(p["concurrency"] for p in plans[:intents]),
                completed_requests=sum(p["concurrency"] for p in plans[:completed]))
        summary.update(valid=True, counts_complete=finished)
    except (ValueError, TypeError, KeyError) as error:
        summary["error"] = type(error).__name__ + ": " + str(error)[:256]
    return summary


def read_progress(output, *, rank, arm, session_sha256, spec_sha256):
    try:
        raw = read_bounded_file(output / f"rank-{rank}-progress.jsonl", PROGRESS_LIMIT)
    except FileNotFoundError:
        raw = None
    except (OSError, ValueError) as error:
        result = progress_readout(None, rank=rank, arm=arm,
            session_sha256=session_sha256, spec_sha256=spec_sha256)
        result["error"] = type(error).__name__ + ": " + str(error)[:256]
        return result
    return progress_readout(raw, rank=rank, arm=arm,
        session_sha256=session_sha256, spec_sha256=spec_sha256)


def native_process_identity():
    """Namespace-local PID is joined to the host observation by namespace/start."""
    pid = os.getpid()
    def start_ticks():
        raw = read_bounded_file(Path("/proc/self/stat"), 65536)
        right = raw.rfind(b")")
        if right < 0 or int(raw.split(b"(", 1)[0]) != pid:
            raise ValueError("native worker process stat identity")
        return int(raw[right + 2:].split()[19])
    start = start_ticks()
    record = {"pid": pid, "start_ticks": start, "uid": os.getuid(), "gid": os.getgid(),
        "boot_id": read_bounded_file(Path("/proc/sys/kernel/random/boot_id"), 256).decode("ascii").strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid")}
    if start != start_ticks():
        raise ValueError("native worker process changed")
    validate_native_process(record)
    return record


def validate_native_process(record):
    if (type(record) is not dict or set(record) != {"pid", "start_ticks", "uid", "gid", "boot_id", "pid_namespace"}
            or any(type(record[key]) is not int or record[key] <= 0 for key in ("pid", "start_ticks"))
            or type(record["uid"]) is not int or type(record["gid"]) is not int
            or (record["uid"], record["gid"]) != (1000, 1000)
            or type(record["boot_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", record["boot_id"]) is None
            or type(record["pid_namespace"]) is not str
            or re.fullmatch(r"pid:\[[1-9][0-9]*\]", record["pid_namespace"]) is None):
        raise ValueError("native worker process binding")


def preweight_identity(torch, spec, arm, rank, output, session_sha256, spec_sha256, deadline):
    if not valid_digest(session_sha256) or not valid_digest(spec_sha256) or rank not in (0, 1):
        raise ValueError("native identity session/spec/rank binding")
    helper = authenticated_adjacent("benchmark_gpu_identity.py", GPU_IDENTITY_SHA256)
    devices, proof = helper.physical_devices(torch, PHYSICAL_DEVICES, min(deadline, time.monotonic() + 60))
    validate_physical_details(helper, devices, proof)
    if time.monotonic() >= deadline:
        raise TimeoutError("native identity exceeded existing worker deadline")
    record = {"schema": 1, "scope": "benchmark-native-preweight-identity-v1", "rank": rank,
        "gpu_id": spec["gpu_ids"][rank], "arm": arm, "session_sha256": session_sha256,
        "spec_sha256": spec_sha256, "helper_sha256": GPU_IDENTITY_SHA256,
        "devices": devices, "hip": proof, "process": native_process_identity(), "before_model_config": True,
        "before_model_runner": True, "qualification": False}
    write_identity(output / f"rank-{rank}-identity.json", record)
    return record


def validate_physical_details(helper, devices, proof):
    if (type(devices) is not list or len(devices) != 2 or any(type(row) is not dict or
            set(row) != {"index", "bdf", "uuid", "architecture", "vram_bytes"} for row in devices)
            or type(proof) is not dict or set(proof) != {"library", "sha256", "bytes", "loaded_inode",
                "loaded_device", "no_load_handle", "symbol", "header_sha256"}):
        raise ValueError("exact native device/proof observations required")
    helper.validate_devices(devices, PHYSICAL_DEVICES)
    helper.validate_proof(proof)


def validate_rank_identity(record, *, rank, arm, session_sha256, spec_sha256):
    expected = {"schema": 1, "scope": "benchmark-native-preweight-identity-v1", "rank": rank,
        "gpu_id": rank, "arm": arm, "session_sha256": session_sha256, "spec_sha256": spec_sha256,
        "helper_sha256": GPU_IDENTITY_SHA256, "before_model_config": True, "before_model_runner": True,
        "qualification": False}
    if type(record) is not dict or set(record) != set(expected) | {"devices", "hip", "process"} or any(
            type(record.get(key)) is not type(value) or record.get(key) != value for key, value in expected.items()):
        raise ValueError("native identity record binding")
    validate_native_process(record["process"])
    helper = authenticated_adjacent("benchmark_gpu_identity.py", GPU_IDENTITY_SHA256)
    validate_physical_details(helper, record["devices"], record["hip"])
    return record


def read_rank_result(path):
    def refuse_constant(value):
        raise ValueError("nonfinite JSON constant")
    record = json.loads(read_bounded_file(path, 2 << 20), parse_constant=refuse_constant)
    if (not isinstance(record, dict)
            or len(json.dumps(record, sort_keys=True, allow_nan=False).encode()) > 2 << 20):
        raise ValueError("rank result type/serialized bound")
    return record


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_result(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
    if len(raw) > 8 << 20:
        raise ValueError("result bound")
    with open(path, "xb") as stream:
        stream.write(raw)


def validate_spec(spec):
    if (spec.get("schema") != 1 or spec.get("scope") != SCOPE
            or spec.get("experiment_mode") != "model-benchmark-v1"):
        raise ValueError("explicit benchmark scope required")
    if "diagnostic_profile" in spec and (type(spec["diagnostic_profile"]) is not dict
            or set(spec["diagnostic_profile"]) != set(PROFILE_CONFIG)
            or any(type(spec["diagnostic_profile"][key]) is not type(value)
                   or spec["diagnostic_profile"][key] != value for key, value in PROFILE_CONFIG.items())):
        raise ValueError("exact bounded attribution profile configuration required")
    if (spec.get("gpu_ids") != [0, 1] or type(spec.get("worker_seconds")) is not int
            or not 60 <= spec["worker_seconds"] <= 1200):
        raise ValueError("two ranks and bounded worker deadline required")
    if spec.get("context_length") != 4096 or spec.get("physical_slots") != 4096:
        raise ValueError("fixed context/slot bounds")
    if (spec.get("outputs") != 32 or spec.get("warmup_batches") != 1
            or spec.get("measured_batches") != 5):
        raise ValueError("fixed generation/sample counts")
    expected_cells = [{"name": name, "concurrency": count, "input_tokens": length}
                      for name, count, length in CELLS]
    if spec.get("cells") != expected_cells:
        raise ValueError("fixed ordered cells")
    prompts = spec.get("frozen_prompt_ids", {})
    digests = spec.get("frozen_prompt_sha256", {})
    if set(prompts) != {row[0] for row in CELLS} or set(digests) != set(prompts):
        raise ValueError("exact prompt inventory")
    for name, count, length in CELLS:
        rows = prompts[name]
        if (not isinstance(rows, list) or len(rows) != count
                or not isinstance(digests[name], list) or len(digests[name]) != count):
            raise ValueError("prompt batch shape")
        for ids, digest in zip(rows, digests[name]):
            if (not isinstance(ids, list) or len(ids) != length
                    or any(type(token) is not int or token < 0 for token in ids)
                    or ids_sha256(ids) != digest):
                raise ValueError("frozen prompt identity")
        if count == 2 and (rows[0] == rows[1] or rows[0][-1] == rows[1][-1]):
            raise ValueError("paired prompts need distinct frozen tails")
    required = {"attention_backend": "triton", "prefill_attention_backend": "triton",
                "decode_attention_backend": "triton", "tp_size": 2, "pp_size": 1,
                "dp_size": 1, "nnodes": 1, "attn_cp_size": 1, "dcp_size": 1,
                "disaggregation_mode": "null", "context_length": 4096,
                "max_total_tokens": 4096, "max_running_requests": 2,
                "chunked_prefill_size": -1, "mem_fraction_static": 0.8,
                "dtype": "bfloat16", "kv_cache_dtype": "auto", "page_size": 1,
                "disable_overlap_schedule": True, "disable_custom_all_reduce": True,
                "mamba_radix_cache_strategy": "no_buffer", "cpu_offload_gb": 0,
                "cuda_graph_backend_decode": "disabled", "cuda_graph_backend_prefill": "disabled",
                "quantization": "fp8", "fp8_gemm_runner_backend": "triton",
                "load_format": "safetensors", "startup_weight_load_mode": "serial",
                "trust_remote_code": False, "speculative_algorithm": None,
                "random_seed": 592137615,
                "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"}
    if any(spec.get("server_args", {}).get(key) != value for key, value in required.items()):
        raise ValueError("frozen native settings")
    return spec


def preflight(spec):
    validate_spec(spec)
    config = checked_file(spec["config_path"], CONFIG_SHA256)
    checked_file(spec["stack_receipt"], spec["stack_receipt_sha256"])
    proof = json.loads(checked_file(Path(spec["package_root"]) /
        "srt/layers/attention/ultraquant-bf16-build.json", spec["build_proof_sha256"]))
    if (proof.get("schema") != 2 or proof.get("serving_composition_complete") is not True
            or proof.get("qualification") != "build-reviewed-not-serving-qualified"
            or proof.get("stack_receipt_sha256") != spec["stack_receipt_sha256"]
            or proof.get("binary_sha256") != BINARY_SHA256):
        raise ValueError("closed serving composition/build gate")
    return config


def request_plan(spec):
    """Exactly 30 batches / 42 requests per arm, including declared warmups."""
    for cell, concurrency, length in CELLS:
        for sample in range(6):
            yield {"cell": cell, "concurrency": concurrency, "input_tokens": length,
                   "sample": sample, "measured": sample > 0,
                   "request_ids": [f"{cell}-{sample}-{i}" for i in range(concurrency)]}


def assert_pool_arm(runner, arm, torch):
    pool = runner.token_to_kv_pool.full_kv_pool
    owner = getattr(pool, "owner", None)
    backend = getattr(runner.attn_backend, "full_attn_backend", runner.attn_backend)
    if arm == "bf16":
        if (getattr(runner, "_uq_bf16_request", None) is not None or owner is not None
                or type(pool).__name__ != "MHATokenToKVPool" or pool.dtype != torch.bfloat16
                or type(backend).__name__ != "TritonAttnBackend"):
            raise ValueError("ordinary BF16 pool required without KV4 owner")
    elif arm == "kv4":
        if (owner is None or runner._uq_bf16_request is None
                or type(pool).__name__ != "QwenBF16MixedPool"
                or type(backend).__name__ != "QwenBF16SourceAdapter"):
            raise ValueError("authenticated KV4 owner required")
    else:
        raise ValueError("unknown arm")
    return owner


def finish_batch(cache, owner, arm, requests, release, capacity, baseline):
    for req in requests:
        if req.kv.holds_kv or req.kv.holds_mamba:
            release(req, cache, is_insert=False)
    if arm == "kv4":
        cache.release_for_idle_bf16_flush()
    elif owner is not None:
        raise ValueError("BF16 cleanup must not own KV4")
    cache.reset()
    if owner is not None:
        owner.refuse_reset()
    if capacity() != baseline:
        raise ValueError("batch release failed to restore pool capacities")


def timed_forward(runner, batch, forward_batch, torch, clock=time.perf_counter, *, profile_labels=False):
    """Host synchronized ForwardBatch init + forward + sample, on each TP rank."""
    torch.cuda.synchronize()
    start = clock()
    with torch.no_grad():
        if profile_labels:
            with torch.profiler.record_function(PROFILE_LABELS[0]):
                fb = forward_batch.init_new(batch, runner, return_hidden_states_before_norm=False)
            with torch.profiler.record_function(PROFILE_LABELS[1]):
                logits = runner.forward(fb).logits_output
            with torch.profiler.record_function(PROFILE_LABELS[2]):
                sampled = runner.sample(logits, fb)
        else:
            fb = forward_batch.init_new(batch, runner, return_hidden_states_before_norm=False)
            logits = runner.forward(fb).logits_output
            sampled = runner.sample(logits, fb)
    torch.cuda.synchronize()
    return logits, sampled, clock() - start, fb


def checked_observation(logits, sampled, fb, concurrency, vocab_size, torch, dist, group):
    """Finite checks, reductions and TP agreement are deliberately NOT timed."""
    values = logits.next_token_logits
    if (fb.batch_size != concurrency or tuple(values.shape) != (concurrency, vocab_size)
            or not torch.isfinite(values).all().item()):
        raise ValueError("nonfinite/wrong-shaped logits or actual batch")
    tokens = sampled.tolist()
    if not isinstance(tokens, list) or len(tokens) != concurrency or any(
            type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError("sample shape or vocabulary")
    peers = [None, None]
    dist.all_gather_object(peers, tokens, group=group)
    if peers != [tokens, tokens]:
        raise ValueError("TP output disagreement")
    # Diagnostic selected-token log-probabilities, not a cross-arm quality gate.
    selected = values.float().log_softmax(dim=-1)
    scores = [float(selected[i, token].item()) for i, token in enumerate(tokens)]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("nonfinite selected score")
    return tokens, scores


def run_rank(spec, arm, rank, output, physical_artifacts=False, session_sha256=None, spec_sha256=None):
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (spec["worker_seconds"], spec["worker_seconds"]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    log = os.open(output / f"rank-{rank}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    result = {"rank": rank, "arm": arm, "batches": [], "failure": None,
              "cleanup_errors": [], "owner_cleanup": False, "qualification": False,
              "intent_batches": 0, "intent_requests": 0,
              "serving_quality_qualified": False, "benchmark_scope": "host-synchronized model forward+sample"}
    cache = runner = owner = baseline = None
    requests = []
    distributed = False
    started = time.monotonic()
    progress = None
    try:
        progress = RankProgress(output, rank, arm, session_sha256, spec_sha256)
        progress.record("rank_started")
        config = preflight(spec)
        progress.record("preflight_complete")
        import sys
        sys.path.insert(0, str(Path(spec["package_root"]).parent))
        progress.record("torch_import")
        import torch
        if physical_artifacts:
            result["native_identity"] = preweight_identity(torch, spec, arm, rank, output,
                session_sha256, spec_sha256, started + spec["worker_seconds"])
            progress.record("native_identity_complete")
        progress.record("serving_imports")
        import torch.distributed as dist
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import bootstrap
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel, get_tp_group,
        )
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.common import release_kv_cache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import SpawnRanks, get_device, get_model, publish, spawn_world_rank
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.layers.attention.ultraquant_contract import bf16_source_request
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.kernels.ops.mamba.triton_ops import initialize_mamba_selective_state_update_backend

        progress.record("arguments")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        args = ServerArgs(**spec["server_args"])
        args.resolve_once()
        _set_envs_and_config(args)
        publish(args, role="scheduler", ranks=SpawnRanks(
            world_rank=spawn_world_rank(args, tp_rank=rank, pp_rank=0), gpu_id=spec["gpu_ids"][rank]))
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()  # Native configuration constant, not FP4 weights.
        progress.record("model_config")
        model_config = ModelConfig.from_server_args(args)
        progress.record("parallel_runtime")
        bootstrap.init_parallel_runtime(server_args=args, device=get_device().device, dist_port=spec["nccl_port"])
        distributed = True
        progress.record("layer_runtime")
        bootstrap.init_layer_runtime(model_config=model_config)
        if get_model().is_startup_weight_load_overlap:
            raise ValueError("benchmark requires synchronous weight loading")
        request = None
        if arm == "kv4":
            request = dict(bf16_source_request(config, rank=rank, physical_slots=4096, context_length=4096))
            request.update(stack_receipt=spec["stack_receipt"], stack_receipt_sha256=spec["stack_receipt_sha256"])
        progress.record("model_runner")
        runner = ModelRunner(model_config=model_config, mem_fraction_static=args.mem_fraction_static,
            gpu_id=spec["gpu_ids"][rank], nccl_port=spec["nccl_port"], server_args=args,
            _uq_bf16_request=request)
        progress.record("memory_pool")
        runner.alloc_memory_pool()
        progress.record("attention_backends")
        runner.init_attention_backends()
        owner = assert_pool_arm(runner, arm, torch)
        initialize_mamba_selective_state_update_backend(args)
        progress.record("graphs")
        runner.init_cuda_graphs()
        progress.record("cache")
        cache = UnifiedRadixCache(CacheInitParams(disable=False,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, page_size=1,
            tp_cache_group=get_tp_group().cpu_group,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            tree_core_backend="python", enable_mamba_extra_buffer=False))

        def capacity():
            return [runner.token_to_kv_pool_allocator.available_size(),
                    len(runner.req_to_token_pool.free_slots),
                    runner.req_to_token_pool.mamba_allocator.available_size()]

        baseline = capacity()
        result["capacity_before"] = baseline
        result["actual_pool"] = type(runner.token_to_kv_pool.full_kv_pool).__name__
        result["actual_attention_backend"] = type(runner.attn_backend).__name__
        if spec.get("diagnostic_profile") is not None:
            if not physical_artifacts:
                raise ValueError("attribution requires authenticated native worker identity")
            result["diagnostic_profile"] = {
                **PROFILE_CONFIG, "method": PROFILE_METHOD, "complete": False,
                "qualification": False, "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_binary_sha256": BINARY_SHA256, "sources": profile_sources(spec), "windows": [],
                "limitations": ["Instrumented timings include observer overhead.",
                    "CPU inclusive wall includes children and synchronous GPU waits; do not sum inclusive rows.",
                    "GPU intervals may overlap; summed durations are not elapsed wall time.",
                    "Kernel categories are name-derived observations, not hardware-counter attribution."]}
            result["instrumented_timings"] = True
            result["profile_method"] = PROFILE_METHOD
        progress.record("ready")
        for ordinal, item in enumerate(request_plan(spec)):
            if time.monotonic() - started >= spec["worker_seconds"]:
                raise TimeoutError("worker deadline before batch")
            if capacity() != baseline:
                raise ValueError("nonempty pools before batch")
            record = dict(item, ordinal=ordinal, generated_ids=[[] for _ in item["request_ids"]],
                          generated_logprobs=[[] for _ in item["request_ids"]], step_seconds=[],
                          complete=False, capacity_before=capacity())
            result["batches"].append(record)
            # Durable intent precedes allocation and every model operation.
            write_result(output / f"rank-{rank}-intent-{ordinal:02d}.json", {
                "arm": arm, "rank": rank, **item,
                "prompt_sha256": spec["frozen_prompt_sha256"][item["cell"]],
                "outputs_per_request": 32, "intent_utc_unix": time.time()})
            progress.record("batch_intent", **item, ordinal=ordinal)
            result["intent_batches"] += 1
            result["intent_requests"] += item["concurrency"]
            requests = []
            wall_start = time.perf_counter()
            for name, ids in zip(item["request_ids"], spec["frozen_prompt_ids"][item["cell"]]):
                if any(token >= model_config.vocab_size for token in ids):
                    raise ValueError("prompt token outside vocabulary")
                req = Req(rid=name, origin_input_text="reviewed frozen benchmark token IDs",
                    origin_input_ids=array("q", ids),
                    sampling_params=SamplingParams(temperature=0, max_new_tokens=32, ignore_eos=True))
                requests.append(req)
                req.logprob_start_len = -1
                req.init_next_round_input(cache)
                if len(req.prefix_indices):
                    raise ValueError("request prefix must be cold")
                req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
                req.set_extend_range(0, len(req.full_untruncated_fill_ids))
            batch = ScheduleBatch.init_new(reqs=requests, req_to_token_pool=runner.req_to_token_pool,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, tree_cache=cache,
                model_config=model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
            batch.prepare_for_extend()
            if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
                batch.prefill_input_ids_cpu = None
            for step in range(32):
                if time.monotonic() - started >= spec["worker_seconds"]:
                    raise TimeoutError("worker deadline before forward")
                if spec.get("diagnostic_profile") is not None and profile_selected(item, step):
                    profile = result["diagnostic_profile"]
                    binding = profile_binding(item, ordinal, step, rank, arm, session_sha256, spec_sha256)
                    logits, sampled, seconds, fb = profiled_forward(runner, batch, ForwardBatch,
                        torch, binding, profile["sources"], profile["windows"])
                else:
                    logits, sampled, seconds, fb = timed_forward(runner, batch, ForwardBatch, torch)
                tokens, scores = checked_observation(logits, sampled, fb, item["concurrency"],
                    model_config.vocab_size, torch, dist, get_tp_group().cpu_group)
                record["step_seconds"].append(seconds)
                for i, (req, token, score) in enumerate(zip(requests, tokens, scores)):
                    req.output_ids.append(token)
                    record["generated_ids"][i].append(token)
                    record["generated_logprobs"][i].append(score)
                if step < 31:
                    batch.input_ids = sampled.to(torch.int64)
                    batch.prepare_for_decode()
            record["batch_wall_seconds"] = time.perf_counter() - wall_start
            finish_batch(cache, owner, arm, requests, release_kv_cache, capacity, baseline)
            record.update(complete=True, capacity_after=capacity(), actual_batch_size=item["concurrency"])
            requests = []
            progress.record("batch_complete", **item, ordinal=ordinal)
        result["owner_cleanup"] = True
        result["capacity_after"] = capacity()
        if spec.get("diagnostic_profile") is not None:
            profile = result["diagnostic_profile"]
            profile["complete"] = len(profile["windows"]) == 4 and all(row["complete"] for row in profile["windows"])
            if not profile["complete"]:
                raise ValueError("attribution requires all four existing forward windows")
            if len(json.dumps(profile, sort_keys=True, allow_nan=False).encode()) > PROFILE_LIMIT:
                profile["complete"] = False
                raise ValueError("attribution aggregate output bound")
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        # A failed diagnostic write must not skip native cleanup.
        if progress is not None:
            try:
                progress.record("cleanup")
            except (OSError, ValueError) as exc:
                result["cleanup_errors"].append("progress: " + str(exc)[:512])
        if cache is not None and baseline is not None:
            try:
                finish_batch(cache, owner, arm, requests, release_kv_cache, capacity, baseline)
            except BaseException as exc:
                result["cleanup_errors"].append(type(exc).__name__ + ": " + str(exc)[:512])
        if distributed:
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except BaseException as exc:
                result["cleanup_errors"].append(type(exc).__name__ + ": " + str(exc)[:512])
        result["elapsed_seconds"] = time.monotonic() - started
        result["completed_batches"] = sum(row["complete"] for row in result["batches"])
        result["completed_requests"] = sum(row["concurrency"] for row in result["batches"] if row["complete"])
        try:
            write_result(output / f"rank-{rank}.json", result)
            if progress is not None:
                progress.record("rank_finished", success=result["failure"] is None and
                    not result["cleanup_errors"] and result["owner_cleanup"])
        finally:
            if progress is not None:
                progress.close()
    raise SystemExit(0 if result["failure"] is None and not result["cleanup_errors"]
                     and result["owner_cleanup"] else 1)


def summarize(records, expected_arm, *, expected_profile=False):
    if expected_arm not in ("bf16", "kv4"):
        raise ValueError("explicit expected arm required")
    if (len(records) != 2 or any(row is None or row["failure"] is not None
            or row["cleanup_errors"] or not row["owner_cleanup"]
            or row["intent_batches"] != 30 or row["completed_batches"] != 30
            or row["intent_requests"] != 42 or row["completed_requests"] != 42
            or row["arm"] != expected_arm or len(row["batches"]) != 30
            for row in records)):
        raise ValueError("incomplete arm")
    if [record["rank"] for record in records] != [0, 1]:
        raise ValueError("exact ordered rank identities required")
    if expected_profile:
        validate_profile_records(records, expected_arm)
    elif any("diagnostic_profile" in record or record.get("instrumented_timings") is True for record in records):
        raise ValueError("unexpected attribution in uninstrumented benchmark")
    samples = []
    for index, plan in enumerate(request_plan({})):
        a, b = (record["batches"][index] for record in records)
        if (any(row[key] != plan[key] for row in (a, b) for key in plan)
                or not a["complete"] or not b["complete"]
                or a["generated_ids"] != b["generated_ids"]
                or any(len(row["step_seconds"]) != 32 for row in (a, b))):
            raise ValueError("rank schedule/output mismatch")
        for row in (a, b):
            if (not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                        for x in row["step_seconds"])
                    or not math.isfinite(row["batch_wall_seconds"]) or row["batch_wall_seconds"] < 0
                    or len(row["generated_ids"]) != plan["concurrency"]
                    or len(row["generated_logprobs"]) != plan["concurrency"]
                    or any(len(ids) != 32 or any(type(x) is not int or x < 0 for x in ids)
                           for ids in row["generated_ids"])
                    or any(len(scores) != 32 or not all(math.isfinite(x) for x in scores)
                           for scores in row["generated_logprobs"])):
                raise ValueError("malformed rank measurements")
        steps = [max(x, y) for x, y in zip(a["step_seconds"], b["step_seconds"])]
        if not all(math.isfinite(x) and x >= 0 for x in steps):
            raise ValueError("invalid timing")
        samples.append({**plan, "prefill_rank_max_seconds": steps[0],
            "decode_rank_max_step_seconds": steps[1:], "decode_31_seconds": sum(steps[1:]),
            "forward_sample_total_seconds": sum(steps),
            "batch_wall_rank_max_seconds": max(a["batch_wall_seconds"], b["batch_wall_seconds"])})
    return samples


def execute_arm(args, spec, output, children):
    ctx = multiprocessing.get_context("spawn")
    deadline = time.monotonic() + spec["worker_seconds"]
    timed_out = False
    parent_error = None
    cleanup_errors = []
    started = time.monotonic()

    def terminate_parent(signum, frame):
        raise RuntimeError("outer supervisor requested termination")

    prior_term = signal.signal(signal.SIGTERM, terminate_parent)
    try:
        for rank in (0, 1):
            child = ctx.Process(target=run_rank, args=(spec, args.arm, rank, output,
                args.physical_artifacts, args.session_sha256, args.spec_sha256))
            child.start()
            children.append(child)
        write_result(output / "owned-workers.json", {"pids": [p.pid for p in children],
                     "worker_seconds": spec["worker_seconds"]})
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
        timed_out = any(child.is_alive() for child in children)
    except BaseException as exc:
        parent_error = type(exc).__name__ + ": " + str(exc)[:2048]
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in children:
            if child.is_alive():
                for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                    if not child.is_alive():
                        break
                    try:
                        if os.getpgid(child.pid) != child.pid:
                            child.terminate() if sig == signal.SIGTERM else child.kill()
                        else:
                            os.killpg(child.pid, sig)
                        child.join(wait)
                    except ProcessLookupError:
                        child.join(wait)
                    except OSError as exc:
                        cleanup_errors.append(type(exc).__name__ + ": " + str(exc)[:512])
        signal.signal(signal.SIGTERM, prior_term)
    groups_absent = []
    for child in children:
        try:
            os.killpg(child.pid, 0)
            groups_absent.append(False)
        except ProcessLookupError:
            groups_absent.append(True)
        except OSError:
            groups_absent.append(False)
    records = []
    parse_errors = []
    for rank in (0, 1):
        try:
            records.append(read_rank_result(output / f"rank-{rank}.json"))
        except Exception as exc:
            records.append(None)
            parse_errors.append({"rank": rank, "type": type(exc).__name__, "message": str(exc)[:1024]})
    if parse_errors:
        parent_error = parent_error or "rank result parsing failed"
    if args.physical_artifacts:
        for rank, record in enumerate(records):
            try:
                identity = json.loads(read_bounded_file(output / f"rank-{rank}-identity.json", 64 << 10))
                validate_rank_identity(identity, rank=rank, arm=args.arm,
                    session_sha256=args.session_sha256, spec_sha256=args.spec_sha256)
                if record is None or record.get("native_identity") != identity:
                    raise ValueError("final rank does not bind preweight identity")
            except Exception as exc:
                parent_error = parent_error or ("native identity: " + type(exc).__name__ + ": " + str(exc)[:1024])
    samples = []
    try:
        samples = summarize(records, args.arm, expected_profile=spec.get("diagnostic_profile") is not None)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        parent_error = parent_error or (type(exc).__name__ + ": " + str(exc)[:1024])
    progress_records = [read_progress(output, rank=rank, arm=args.arm,
        session_sha256=args.session_sha256, spec_sha256=args.spec_sha256) for rank in (0, 1)]
    if args.physical_artifacts and not all(row["valid"] and row["finished"] for row in progress_records):
        parent_error = parent_error or "rank progress incomplete"
    passed = (parent_error is None and not cleanup_errors and all(groups_absent)
              and not timed_out and len(children) == 2 and all(p.exitcode == 0 for p in children))
    result = {"schema": 1, "scope": SCOPE, "arm": args.arm,
        "status": "MODEL_BENCHMARK_COMPLETE" if passed else "MODEL_BENCHMARK_FAILED",
        "spec_sha256": args.spec_sha256, "timed_out": timed_out,
        "exits": [p.exitcode for p in children], "ranks": records, "samples": samples,
        "parent_error": parent_error, "cleanup_errors": cleanup_errors,
        "rank_parse_errors": parse_errors,
        "groups_observed_absent": groups_absent, "arm_wall_seconds": time.monotonic() - started,
        "qualification": False, "serving_quality_qualified": False,
        "rank_progress": progress_records,
        "http_throughput_measured": False, "independent_release_required": True}
    if args.physical_artifacts:
        result.update(physical_artifacts=True, session_sha256=args.session_sha256,
            native_identity_verified=parent_error is None)
    if spec.get("diagnostic_profile") is not None:
        result.update(diagnostic_profile=PROFILE_CONFIG, profile_method=PROFILE_METHOD,
            instrumented_timings=True, diagnostic_profile_complete=passed)
    write_result(output / "result.json", result)
    raise SystemExit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-reviewed-benchmark", action="store_true")
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--output")
    parser.add_argument("--arm", choices=("bf16", "kv4"))
    parser.add_argument("--physical-artifacts", action="store_true")
    parser.add_argument("--session-sha256")
    parser.add_argument("--artifacts-sha256")
    args = parser.parse_args()
    if not args.execute_reviewed_benchmark:
        raise SystemExit("NOT_RUN: explicit reviewed runtime/spec required")
    if args.arm is None:
        raise ValueError("explicit arm required")
    if args.physical_artifacts:
        if not valid_digest(args.session_sha256) or not valid_digest(args.artifacts_sha256):
            raise ValueError("physical mode requires independently pinned session/artifact identities")
        artifacts = authenticated_adjacent("benchmark_artifacts.py", args.artifacts_sha256)
    elif args.session_sha256 is not None or args.artifacts_sha256 is not None:
        raise ValueError("physical artifact flags must be supplied together")
    spec = json.loads(checked_file(args.spec, args.spec_sha256))
    preflight(spec)
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("fresh absolute owned output required")
    output.mkdir(mode=0o700)
    children = []
    try:
        execute_arm(args, spec, output, children)
    finally:
        if args.physical_artifacts:
            if any(child.is_alive() for child in children):
                raise RuntimeError("artifact export refused while an owned child survives")
            for child in children:
                child.join(0)
                if child.exitcode is None:
                    raise RuntimeError("artifact export refused before owned child reap")
            binding = {"session_sha256": args.session_sha256,
                "arm": args.arm, "spec_sha256": args.spec_sha256}
            artifacts.export_bundle(artifacts.stage_artifacts(output, binding),
                binding=binding, sink=sys.stdout.buffer)


if __name__ == "__main__":
    main()
