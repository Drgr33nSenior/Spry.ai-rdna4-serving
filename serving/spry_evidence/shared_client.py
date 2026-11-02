"""Bounded native-HTTP client for the frozen FP8 normal-stop screen.

This module launches nothing.  Its caller must provide a post-warmup native
readiness record and a live-worker checker before it can send one request.
"""
import hashlib
import http.client
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import math
import re
import socket
import threading
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import serving_contract


def _load_receipts():
    """Load the receipt helper from its pinned source, ignoring cached bytecode."""
    path = Path(__file__).with_name('request_receipts.py')
    if (path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1
            or path.stat().st_size > 64 << 10):
        raise ValueError('request receipt source binding')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '54419f977a5a344973b6f7679a40265848b353194a1c7992b1e2c2660a0a47ee':
        raise ValueError('request receipt source binding')
    loader = SourceFileLoader('_kv_receipts', str(path))
    loader.get_code = lambda fullname: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_kv_receipts = _load_receipts()


MAX_RESPONSE_BYTES = 2 << 20
MAX_GATE_BYTES = 64 << 10
CLOSURE_MODE = 'native-http-fp8-sprint01-v1'
MAX_METRICS_BYTES = 1 << 20
CANCEL_RID = 'fp8-closure-64Q8v0X1-cancel'
ENDPOINT = "/v1/chat/completions"
GATE_ROOT_FIELDS = {
    "schema", "mode", "status", "arm", "source_manifest_sha256",
    "session_sha256", "spec_sha256", "post_warmup_native_ready", "workers",
}
GATE_WORKER_FIELDS = {"rank", "pid", "start_ticks", "boot_id", "identity_sha256"}


class ClientRefusal(ValueError):
    """A bounded screen precondition or protocol shape was not met."""


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ClientRefusal('nonfinite JSON')))


def _digest_text(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition, message):
    if not condition:
        raise ClientRefusal(message)


def closure_mode():
    return (serving_contract.MODE == CLOSURE_MODE and len(serving_contract.CASES)==53 and
            serving_contract.CASES[48]=='cancel-victim')


def inspect_prompt_flush(bundle, ordinal):
    """Require the exact immediate stock flush acknowledgment before another send."""
    _require(serving_contract.mtp_contract.prompt_minimization_enabled() and type(ordinal) is int and ordinal in (1, 2),
             'prompt flush phase/ordinal differs')
    record = bundle['record']
    _require(bundle['request_raw'] == b'' and bundle['response_raw'] == serving_contract.PROMPT_FLUSH_ACK and
             record['name'] == f'prompt-flush-before-{ordinal:02d}' and record['method'] == 'POST' and
             record['endpoint'] == serving_contract.PROMPT_FLUSH_ENDPOINT and record['status'] == 200 and
             record['failure'] is None and record['response_truncated'] is False and
             record['request_sha256'] == _sha256(b'') and
             record['response_sha256'] == _sha256(serving_contract.PROMPT_FLUSH_ACK),
             'native immediate prompt flush acknowledgment differs')
    return serving_contract.prompt_flush_contract()


def _sse_events(raw, *, final):
    """Parse bounded native SSE frames; retain wire bytes separately, unchanged."""
    _require(type(raw) is bytes and len(raw) <= MAX_RESPONSE_BYTES, "bounded SSE bytes required")
    frames = re.split(br"\r?\n\r?\n", raw)
    if final:
        _require(frames[-1] == b"", "incomplete SSE frame")
    events = []
    for frame in frames[:-1]:
        _require(frame, "empty SSE frame")
        lines = frame.decode("utf-8").splitlines()
        event, data = None, None
        for line in lines:
            if line.startswith("event: "):
                _require(event is None, "duplicate SSE event field")
                event = line[7:]
            elif line.startswith("data: "):
                _require(data is None, "duplicate SSE data field")
                data = line[6:]
            else:
                raise ClientRefusal("unexpected SSE field")
        _require(event in (None, "sglext_ids") and data is not None, "SSE event or data differs")
        if data == "[DONE]":
            _require(event is None, "named SSE DONE")
            events.append(None)
        else:
            value = _strict_json(data)
            _require(type(value) is dict and "error" not in value, "native SSE error or non-object")
            _require(type(value.get("choices")) is list and
                     all(type(choice) is dict for choice in value["choices"]), "invalid SSE choices")
            pending = [value]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, float):
                    _require(math.isfinite(item), "nonfinite SSE number")
            events.append(value)
    return events


def _generated_event(event):
    if not isinstance(event, dict):
        return False
    choices = event.get("choices")
    return (type(choices) is list and len(choices) == 1 and
            type(choices[0]) is dict and type(choices[0].get("delta")) is dict and
            type(choices[0]["delta"].get("content")) is str and
            bool(choices[0]["delta"]["content"]))


def evicted_tokens(raw):
    """Sum the exact native Prometheus counter, never a flush or cache estimate."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_METRICS_BYTES, "metrics raw bound")
    values, samples = [], set()
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("sglang:evicted_tokens_total"):
            continue
        match = re.fullmatch(r'(sglang:evicted_tokens_total(?:\{[^{}\r\n]*\})?)\s+(\S+)(?:\s+\d+)?', line)
        _require(match is not None, "malformed native eviction counter")
        _require(match[1] not in samples, "duplicate native eviction counter")
        samples.add(match[1])
        value = float(match[2])
        _require(math.isfinite(value) and value >= 0, "nonfinite or negative eviction counter")
        values.append(value)
    _require(values, "native eviction counter missing")
    total = math.fsum(values)
    _require(math.isfinite(total), "eviction counter sum overflow")
    return total


def _metric_labels(text):
    pairs = list(re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"\\]*)"(?:,|$)', text))
    _require(''.join(item.group(0) for item in pairs) == text, 'malformed native metric labels')
    labels = dict(item.groups() for item in pairs)
    _require(len(labels) == len(pairs), 'duplicate native metric label')
    return labels


def _absent_counter_pool_evidence(text, pool):
    """Authenticate companion gauges, not an inferred zero eviction count."""
    names = {'max_total_num_tokens', 'kv_cache_memory_usage_gb',
             'kv_available_tokens', 'kv_evictable_tokens', 'kv_used_tokens',
             'mamba_available_tokens', 'mamba_evictable_tokens', 'mamba_used_tokens'}
    values = {}
    for line in text.splitlines():
        if not line.startswith('sglang:') or re.split(r'[{\s]', line[7:], maxsplit=1)[0] not in names:
            continue
        match = re.fullmatch(r'sglang:([a-zA-Z_][a-zA-Z0-9_]*)\{([^{}]*)\}\s+(\S+)', line)
        _require(match is not None, 'malformed native companion gauge')
        name, label_text, raw_value = match.groups()
        labels = _metric_labels(label_text)
        expected = dict(engine_type='unified', model_name='fp8-serving-screen',
                        moe_ep_rank='0', pp_rank='0', tp_rank=labels.get('tp_rank'))
        _require(labels == expected and labels['tp_rank'] in ('0', '1'),
                 'native companion gauge scope differs')
        rank = int(labels['tp_rank'])
        _require(name in ('max_total_num_tokens', 'kv_cache_memory_usage_gb') or rank == 0,
                 'native companion occupancy rank differs')
        key = (name, rank)
        _require(key not in values, 'duplicate native companion gauge')
        try:
            value = Decimal(raw_value)
        except InvalidOperation as error:
            raise ClientRefusal('invalid native companion gauge') from error
        _require(value.is_finite() and value >= 0, 'nonfinite or negative native companion gauge')
        _require(name == 'kv_cache_memory_usage_gb' or value == value.to_integral_value(),
                 'noninteger native companion slot count')
        values[key] = value
    expected_keys = {(name, rank) for name in names for rank in
                     ((0, 1) if name in ('max_total_num_tokens', 'kv_cache_memory_usage_gb') else (0,))}
    _require(set(values) == expected_keys, 'missing native companion rank or gauge')
    capacity = pool['max_total_tokens']
    for rank in (0, 1):
        _require(values[('max_total_num_tokens', rank)] == capacity and
                 values[('kv_cache_memory_usage_gb', rank)] * 2**30 == (capacity + 1) * 32768,
                 'native companion capacity or KV payload differs')
    totals = {stem: sum(values[(stem + '_' + kind + '_tokens', 0)]
                        for kind in ('available', 'evictable', 'used')) for stem in ('kv', 'mamba')}
    mamba_slots = serving_contract.mtp_contract.expected_recurrent_slots()
    _require(totals['kv'] == capacity and totals['mamba'] == mamba_slots,
             'native companion occupancy accounting differs')
    return dict(validated=True, max_total_tokens=capacity, mamba_slots=mamba_slots,
                scope='both TP capacity/KV payload; TP0 occupancy',
                eviction_count_inferred=False)


def eviction_observation(raw, arm):
    """Keep candidate non-export separate from a measured counter value."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_METRICS_BYTES, 'metrics raw bound')
    pool = serving_contract.mtp_contract.memory_pool_config(arm)
    candidate = (pool is not None and arm == 'mtp' and
                 serving_contract.mtp_contract.settings()['phase'] == 'sprint3-dflash-pool')
    if not candidate:
        evicted_tokens(raw)  # Preserve the existing strict legacy/control gate.
    series = {}
    text = raw.decode('utf-8')
    for line in text.splitlines():
        if not line.startswith('sglang:evicted_tokens_total'):
            continue
        match = re.fullmatch(r'(sglang:evicted_tokens_total(?:\{([^{}\r\n]*)\})?)\s+(\S+)(?:\s+\d+)?', line)
        _require(match is not None, 'malformed native eviction counter')
        labels = _metric_labels(match[2]) if match[2] is not None else {}
        if candidate:
            expected = dict(cache_type='UnifiedRadixCache', dp_rank='0', pp_rank='0', tp_rank=labels.get('tp_rank'))
            _require(labels == expected and labels['tp_rank'] in ('0', '1'),
                     'native eviction counter labels differ')
        key = json.dumps(labels, sort_keys=True, separators=(',', ':'))
        _require(key not in series, 'duplicate native eviction counter')
        value = float(match[3])
        _require(math.isfinite(value) and value >= 0, 'nonfinite or negative eviction counter')
        series[key] = value
    if series:
        if candidate:
            _require(len(series) == 2, 'missing native eviction counter rank')
        total = math.fsum(series.values())
        _require(math.isfinite(total), 'eviction counter sum overflow')
        return dict(status='OBSERVED', value=total, measured_zero=total == 0,
                    series=series, companion_pool=None)
    _require(candidate, 'native eviction counter missing')
    companion = _absent_counter_pool_evidence(text, pool)
    return dict(status='NOT_EXPORTED', value=None, measured_zero=False,
                series={}, companion_pool=companion)


def pressure_observation(before_raw, after_raw, arm):
    """Compare observed series; never subtract an unavailable counter value."""
    before, after = (eviction_observation(raw, arm) for raw in (before_raw, after_raw))
    for key, value in before['series'].items():
        _require(key in after['series'] and after['series'][key] >= value,
                 'native eviction counter series disappeared or decreased')
    both = before['status'] == after['status'] == 'OBSERVED'
    delta = after['value'] - before['value'] if both else None
    candidate = (serving_contract.mtp_contract.memory_pool_config(arm) is not None and arm == 'mtp' and
                 serving_contract.mtp_contract.settings()['phase'] == 'sprint3-dflash-pool')
    _require(candidate or (delta is not None and delta > 0), 'native eviction counter did not increase')
    observed = delta > 0 if both else None
    return dict(counter='sglang:evicted_tokens_total', before=before['value'], after=after['value'],
                delta=delta, flush_used=False, native_eviction_observed=observed,
                after_observed_total=after['value'],
                observations=dict(before=before, after=after),
                interval_eviction_count_measured=both)


def inspect_cancel_evidence(case, raw_sse, abort_record):
    """Require native graceful abort plus DONE, not dispatch or disconnect alone."""
    _require(closure_mode() and case["body"].get("stream") is True and
             case["body"].get("ignore_eos") is True and case["body"].get("rid") == CANCEL_RID,
             "cancel request contract")
    _require(type(abort_record) is dict and
             set(abort_record) == {"request_raw", "response_raw", "record", "stream"}, "cancel evidence fields")
    action, stream = abort_record["record"], abort_record["stream"]
    expected_body = serving_contract.canonical(dict(rid=CANCEL_RID, abort_all=False))
    _require(abort_record["request_raw"] == expected_body and abort_record["response_raw"] == b"" and
             action["method"] == "POST" and action["endpoint"] == "/abort_request" and
             action["status"] == 200 and action["response_truncated"] is False and
             action["request_sha256"] == _sha256(expected_body) and
             action["response_sha256"] == _sha256(b""), "abort dispatch acknowledgment binding")
    _require(set(stream) == {"schema", "ordinal", "case", "rid", "first_generated_monotonic",
                            "generated_prefix_bytes", "generated_prefix_sha256", "drained_monotonic",
                            "stream_eof", "failure"} and stream["schema"] == 1 and stream["ordinal"] == 22 and
             stream["case"] == case["case"] and stream["rid"] == CANCEL_RID and
             stream["stream_eof"] is True and stream["failure"] is None, "cancel stream evidence binding")
    times = [stream["first_generated_monotonic"], action["started_monotonic"],
             action["finished_monotonic"], stream["drained_monotonic"]]
    _require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in times) and
             times == sorted(times), "cancel chronology differs")
    count = stream["generated_prefix_bytes"]
    _require(type(count) is int and 0 < count <= len(raw_sse) and
             _sha256(raw_sse[:count]) == stream["generated_prefix_sha256"], "generated prefix binding")
    prefix = _sse_events(raw_sse[:count], final=False)
    _require(sum(_generated_event(event) for event in prefix) >= 2 and None not in prefix and
             all(not choice.get("finish_reason") for event in prefix
                 for choice in event.get("choices", [])), "abort was not dispatched during generation")
    events = _sse_events(raw_sse, final=True)
    _require(events and events[-1] is None and events.count(None) == 1, "missing or duplicate SSE DONE")
    aborts, generated, terminal, input_ids, output_ids, usage = 0, 0, False, None, None, None
    for event in events[:-1]:
        _require(event.get("id") == CANCEL_RID and event.get("model") == case["body"].get("model"),
                 "SSE request id or model differs")
        choices = event.get("choices")
        _require(type(choices) is list and len(choices) <= 1, "SSE choices differ")
        _require(not terminal or not choices, "SSE choice after native terminal")
        for choice in choices:
            _require(type(choice) is dict and choice.get("index") == 0, "SSE choice index")
        for choice in choices:
            delta = choice.get("delta", {})
            _require(type(delta) is dict and delta.get("role") in (None, "assistant") and
                     (delta.get("content") is None or type(delta.get("content")) is str) and
                     all(value in (None, "", []) for key, value in delta.items()
                         if key not in ("role", "content")), "tool/reasoning/invalid SSE delta")
            if delta.get("content"):
                _require(not terminal, "generated content after abort")
                generated += 1
            finish = choice.get("finish_reason")
            if finish is not None:
                _require(finish == "abort" and not terminal, "non-abort or duplicate native terminal")
                aborts += 1
                terminal = True
        extension = event.get("sglext")
        if extension is not None:
            _require(type(extension) is dict and terminal, "premature or invalid stream extension")
            if extension.get("input_ids") is not None:
                _require(input_ids is None, "duplicate SSE input IDs")
                input_ids = extension["input_ids"]
            if extension.get("output_ids") is not None:
                _require(output_ids is None, "duplicate SSE output IDs")
                output_ids = extension["output_ids"]
        if event.get("usage") is not None:
            _require(terminal and usage is None and type(event["usage"]) is dict, "duplicate or premature SSE usage")
            usage = event["usage"]
    _require(aborts == 1 and generated >= 2, "native abort or generated content missing")
    expected = serving_contract.expected_case(case["case"])
    _require(type(expected) is dict and serving_contract._ids(expected.get("input_ids")), "frozen cancel prompt IDs missing")
    if input_ids is not None:
        _require(input_ids == expected["input_ids"], "SSE prompt IDs differ from frozen native IDs")
    if output_ids is not None:
        _require(type(output_ids) is list and len(output_ids) == 1 and
                 serving_contract._ids(output_ids[0], maximum=case["body"]["max_completion_tokens"]),
                 "SSE output IDs differ")
    if usage is not None:
        _require(all(type(usage.get(key)) is int and usage[key] >= 0
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")) and
                 usage["prompt_tokens"] == len(expected["input_ids"]) and
                 usage["completion_tokens"] <= case["body"]["max_completion_tokens"] and
                 usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] and
                 (output_ids is None or usage["completion_tokens"] == len(output_ids[0])),
                 "SSE usage/count disagreement")
    return dict(schema=1, mode=serving_contract.MODE, case=case["case"], status="CANCELLED_COMPLETE",
                metric_eligible=False, qualification=False, finish_reason="abort", request_id=CANCEL_RID,
                raw_sse_sha256=_sha256(raw_sse), generated_content_chunks=generated, stream_done=True,
                input_ids_observed=input_ids, output_ids_observed=output_ids, usage_observed=usage,
                prompt_ids_binding="terminal native extension" if input_ids is not None else "frozen request and native preflight",
                code_correctness_graded=False, numerical_equivalence_established=False)


def _write_exclusive(path, raw):
    if _kv_receipts.is_receipt(path):
        return _kv_receipts.write(path, raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(raw)


def _write_json(path, value):
    _write_exclusive(path, serving_contract.canonical(value))


def _safe_name(case_name):
    _require(type(case_name) is str and case_name in serving_contract.CASES, "unknown frozen case")
    return case_name


def read_gate(path):
    """Read a small external readiness record without trusting it yet."""
    raw = Path(path).read_bytes()
    _require(0 < len(raw) <= MAX_GATE_BYTES, "readiness record exceeds bound")
    try:
        decoded = _strict_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClientRefusal("readiness record is not JSON") from error
    _require(type(decoded) is dict, "readiness record must be an object")
    return decoded, _sha256(raw)


def validate_gate(gate, expected):
    """Reject stale, incomplete, substituted, or non-native worker records."""
    _require(type(gate) is dict and set(gate) == GATE_ROOT_FIELDS, "readiness root schema differs")
    _require(type(expected) is dict and set(expected) == {
        "arm", "source_manifest_sha256", "session_sha256", "spec_sha256"}, "expected binding schema differs")
    _require(gate["schema"] == 1, "unsupported readiness schema")
    _require(gate["mode"] == serving_contract.MODE, "readiness mode differs")
    _require(gate["status"] == "NATIVE_POSTWARMUP_READY", "native post-warmup readiness missing")
    _require(gate["post_warmup_native_ready"] is True, "post-warmup readiness is not true")
    _require(expected["arm"] in ("control", "mtp"), "invalid expected arm")
    for key, value in expected.items():
        if key == "arm":
            _require(gate[key] == value, "readiness arm differs")
        else:
            _require(_digest_text(value), "invalid expected " + key)
            _require(gate[key] == value, "readiness " + key + " differs")
    workers = gate["workers"]
    _require(type(workers) is list and len(workers) == 2, "exactly two worker records required")
    normalized = []
    for worker in workers:
        _require(type(worker) is dict and set(worker) == GATE_WORKER_FIELDS, "worker schema differs")
        _require(type(worker["rank"]) is int and worker["rank"] in (0, 1), "worker rank differs")
        for key in ("pid", "start_ticks"):
            _require(type(worker[key]) is int and worker[key] > 0, "invalid worker " + key)
        _require(type(worker["boot_id"]) is str and worker["boot_id"], "invalid worker boot_id")
        _require(_digest_text(worker["identity_sha256"]), "invalid worker identity_sha256")
        normalized.append(dict(worker))
    _require({worker["rank"] for worker in normalized} == {0, 1}, "worker ranks differ")
    _require(len({(worker["pid"], worker["start_ticks"], worker["boot_id"])
                  for worker in normalized}) == 2, "duplicate worker identity")
    return sorted(normalized, key=lambda worker: worker["rank"])


def reobserve_workers(workers, checker):
    """Bind readiness identities to currently live workers immediately before send."""
    _require(callable(checker), "a live-worker checker is required")
    observed = []
    for worker in workers:
        current = checker(dict(worker))
        _require(type(current) is dict and set(current) == {"pid", "start_ticks", "boot_id", "alive"},
                 "live-worker observation schema differs")
        _require(current["alive"] is True, "worker is not live")
        for key in ("pid", "start_ticks", "boot_id"):
            _require(current[key] == worker[key], "live worker " + key + " differs")
        observed.append(dict(current, rank=worker["rank"], identity_sha256=worker["identity_sha256"]))
    return observed


class ServingClient:
    """Send each frozen request once, without retries or redirects."""

    def __init__(self, *, port, output_root, absolute_deadline, expected_gate,
                 live_worker_checker, connection_factory=http.client.HTTPConnection,
                 clock=monotonic, utc_now=_utc_now, request_observer=None):
        _require(type(port) is int and 1 <= port <= 65535, "invalid loopback port")
        _require(type(absolute_deadline) in (int, float) and math.isfinite(absolute_deadline),
                 "finite absolute deadline required")
        _require(isinstance(output_root, (str, Path)), "output root required")
        self.port, self.root = port, Path(output_root)
        self.deadline, self.expected_gate = absolute_deadline, expected_gate
        self.checker, self.factory, self.clock, self.utc_now = (
            live_worker_checker, connection_factory, clock, utc_now)
        self.request_observer = request_observer

    def _remaining(self):
        remaining = self.deadline - self.clock()
        _require(math.isfinite(remaining) and remaining > 0, "absolute client deadline elapsed")
        return remaining

    def _record(self, relative, value):
        _write_json(self.root / relative, value)

    def _set_socket_timeout(self, connection):
        """Keep blocking socket operations inside the one absolute deadline."""
        socket = getattr(connection, "sock", None)
        setter = getattr(socket, "settimeout", None)
        if callable(setter):
            setter(self._remaining())

    def _read_bounded(self, connection, response, maximum=MAX_RESPONSE_BYTES, chunks=None):
        """Return an exact body or an exact bounded prefix marked as truncated."""
        chunks = bytearray() if chunks is None else chunks
        while len(chunks) <= maximum:
            self._set_socket_timeout(connection)
            # HTTPResponse.read1 returns after one underlying read. read(n)
            # can conceal repeated receives from the absolute-deadline check.
            reader = getattr(response, 'read1', None)
            if reader is None:
                reader = response.read
            chunk = reader(min(65536, maximum + 1 - len(chunks)))
            _require(type(chunk) is bytes, "HTTP response returned non-bytes")
            if not chunk:
                return bytes(chunks), False
            chunks.extend(chunk)
            self._remaining()
        return bytes(chunks[:maximum]), True

    def _read_stream(self, connection, response, ordinal, case):
        """Retain bounded SSE bytes and each receive's byte/time identity."""
        raw, truncated = bytearray(), False
        observed = dict(schema=1, ordinal=ordinal, case=case['case'], chunks=[],
            stream_eof=False, failure=None, first_generated_monotonic=None,
            generated_prefix_bytes=None, generated_prefix_sha256=None)
        try:
            while len(raw) <= MAX_RESPONSE_BYTES:
                self._set_socket_timeout(connection)
                reader = getattr(response, 'read1', None) or response.read
                chunk = reader(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
                tick = self.clock()
                _require(type(chunk) is bytes, 'SSE receive returned non-bytes')
                if not chunk:
                    observed['stream_eof'] = True
                    break
                _require(len(observed['chunks']) < serving_contract.MAX_STREAM_CHUNKS, 'SSE receive count bound')
                start = len(raw)
                raw.extend(chunk)
                retained = bytes(raw[:MAX_RESPONSE_BYTES])
                observed['chunks'].append(dict(end=len(retained), sha256=_sha256(retained[start:]),
                                               received_monotonic=tick))
                self._remaining()
                if len(raw) > MAX_RESPONSE_BYTES:
                    truncated = True
                    break
                if observed['first_generated_monotonic'] is None:
                    try:
                        events = serving_contract.stream_events(bytes(raw), final=False)
                    except (ValueError, UnicodeError, RecursionError):
                        # Keep draining malformed streams so their exact bytes survive.
                        continue
                    if any(serving_contract.generated_stream_event(event['value']) for event in events):
                        observed.update(first_generated_monotonic=tick, generated_prefix_bytes=len(raw),
                                        generated_prefix_sha256=_sha256(bytes(raw)))
        except (OSError, ValueError, http.client.HTTPException) as error:
            truncated = True
            observed['failure'] = type(error).__name__ + ': ' + str(error)[:512]
        return bytes(raw[:MAX_RESPONSE_BYTES]), truncated, observed

    def run(self, gate, gate_sha256):
        """Return durable counts; an intent is never reported as a completed request."""
        frozen = serving_contract.requests()
        _require(len(frozen) == len(serving_contract.CASES) and [row["case"] for row in frozen] == list(serving_contract.CASES),
                 "frozen normal-stop request set differs")
        _require(_digest_text(gate_sha256), "invalid readiness digest")
        self._remaining()
        workers = validate_gate(gate, self.expected_gate)
        self._record("readiness/accepted-gate.json", dict(
            schema=1, gate_sha256=gate_sha256, expected=dict(self.expected_gate),
            mode=serving_contract.MODE, accepted_utc=self.utc_now(), workers=workers))
        live = reobserve_workers(workers, self.checker)
        self._record("readiness/live-workers-before-first-generation.json", dict(
            schema=1, gate_sha256=gate_sha256, observed_utc=self.utc_now(), workers=live))

        return self._run_kv_composition(frozen)

    def _one(self, ordinal, case, counts):
        name = _safe_name(case["case"])
        prefix = "%02d-%s" % (ordinal, name)
        body = serving_contract.canonical(case["body"])
        body_sha256 = _sha256(body)
        before_send = self.clock()
        self._remaining()
        _write_exclusive(self.root / ("requests/%s.raw" % prefix), body)
        self._record("requests/%s.intent.json" % prefix, dict(schema=1, ordinal=ordinal,
            case=name, body_sha256=body_sha256, prepared_monotonic=before_send,
            prepared_utc=self.utc_now(), intent_is_not_submission_proof=True))
        counts["intent_count"] += 1
        self._record("requests/%s.send-entered.json" % prefix, dict(schema=1, ordinal=ordinal,
            case=name, body_sha256=body_sha256, entered_monotonic=self.clock(), entered_utc=self.utc_now(),
            send_call_entered=True, submission_unproven_until_return=True))
        counts["send_attempt_entered_count"] += 1
        connection = None
        stream = case['body'].get('stream') is True
        try:
            connection = self.factory("127.0.0.1", self.port, timeout=self._remaining())
            connection.request("POST", ENDPOINT, body=body, headers={
                "Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"})
            sent_at = self.clock()
            self._record("requests/%s.send-returned.json" % prefix, dict(schema=1, ordinal=ordinal,
                case=name, returned_monotonic=sent_at, returned_utc=self.utc_now(), send_call_returned=True,
                request_body_sha256=body_sha256))
            counts["send_call_returned_count"] += 1
            self._set_socket_timeout(connection)
            response = connection.getresponse()
            status, reason = response.status, response.reason
            header_at = self.clock()
            counts["response_observed_count"] += 1
            self._record("responses/%s.headers.json" % prefix, dict(schema=1, ordinal=ordinal,
                case=name, status=status, reason=str(reason), headers_observed_monotonic=header_at,
                headers_observed_utc=self.utc_now(), response_observed=True))
            if stream:
                raw, truncated, observation = self._read_stream(connection, response, ordinal, case)
                self._record('responses/%s.stream.json' % prefix, observation)
            else:
                raw, truncated = self._read_bounded(connection, response)
            received_at = self.clock()
            raw_suffix = ".partial.raw" if truncated else ".raw"
            _write_exclusive(self.root / ("responses/%s%s" % (prefix, raw_suffix)), raw)
            counts["response_body_observed_count"] += 1
            wire = dict(schema=1, ordinal=ordinal, case=name, status=status, reason=str(reason),
                response_sha256=_sha256(raw), response_bytes=len(raw), received_monotonic=received_at,
                received_utc=self.utc_now(), response_observed=True,
                client_elapsed_seconds=received_at - before_send, response_truncated=truncated,
                raw_body_path="responses/%s%s" % (prefix, raw_suffix))
            self._record("responses/%s.record.json" % prefix, wire)
            _require(not truncated, "response body exceeds bound; bounded prefix retained")
            if stream:
                contract = serving_contract.inspect_stream_response(case, raw, wire['client_elapsed_seconds'])
                if contract['status'] == 'NORMAL_STOP_COMPLETE':
                    contract.update(serving_contract.inspect_stream_observation(case, raw, observation,
                        ordinal=ordinal, prepared=before_send, sent=sent_at, headers=header_at, received=received_at))
                    if status != 200:
                        contract.update(status='HTTP_STATUS_REFUSAL', metric_eligible=False)
                    else:
                        counts['normal_complete_count'] += 1
                return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=wire)
            try:
                decoded = _strict_json(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ClientRefusal) as error:
                contract = dict(schema=1, mode=serving_contract.MODE, case=name,
                    status="STRUCTURAL_REFUSAL", metric_eligible=False, qualification=False,
                    error="invalid JSON response: " + type(error).__name__,
                    raw_response_sha256=wire["response_sha256"])
            else:
                contract = serving_contract.inspect_response(case, decoded, wire["client_elapsed_seconds"])
                contract["raw_http_response_sha256"] = wire["response_sha256"]
                if not 200 <= status < 300 and serving_contract.response_completed(contract):
                    contract.update(status="HTTP_STATUS_REFUSAL", metric_eligible=False)
            if contract["status"] == "NORMAL_STOP_COMPLETE":
                counts["normal_complete_count"] += 1
            if 'valid_complete_count' in counts and serving_contract.response_completed(contract):
                counts['valid_complete_count'] += 1
            return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=wire)
        except (OSError, http.client.HTTPException, ClientRefusal, ValueError) as error:
            failure = dict(schema=1, ordinal=ordinal, case=name, failure=type(error).__name__ + ": " + str(error)[:512],
                failure_monotonic=self.clock(), failure_utc=self.utc_now(), response_observed=False)
            self._record("responses/%s.failure.json" % prefix, failure)
            contract = dict(schema=1, mode=serving_contract.MODE, case=name, status="TRANSPORT_OR_STRUCTURAL_REFUSAL",
                metric_eligible=False, qualification=False, error=failure["failure"])
            return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=failure)
        finally:
            if connection is not None:
                connection.close()

    def _control(self, name, method, endpoint, body):
        """Retain dispatch and bounded exact raw acknowledgment, including failures."""
        raw, connection, truncated, failure = bytearray(), None, False, None
        started = self.clock()
        record = dict(schema=1, name=name, method=method, endpoint=endpoint,
            request_sha256=_sha256(body), request_bytes=len(body), started_monotonic=started,
            returned_monotonic=None, headers_monotonic=None, status=None, reason=None)
        _write_exclusive(self.root / ("controls/" + name + ".request.raw"), body)
        try:
            connection = self.factory("127.0.0.1", self.port, timeout=self._remaining())
            connection.request(method, endpoint, body=body, headers={"Content-Type": "application/json"})
            record["returned_monotonic"] = self.clock()
            self._set_socket_timeout(connection)
            response = connection.getresponse()
            record.update(headers_monotonic=self.clock(), status=response.status, reason=str(response.reason))
            _, truncated = self._read_bounded(connection, response, MAX_METRICS_BYTES, raw)
        except (OSError, ValueError, http.client.HTTPException) as error:
            failure, truncated = type(error).__name__ + ": " + str(error)[:512], True
        finally:
            if connection is not None:
                connection.close()
            exact = bytes(raw[:MAX_METRICS_BYTES])
            record.update(finished_monotonic=self.clock(), response_sha256=_sha256(exact),
                response_bytes=len(exact), response_truncated=truncated, failure=failure)
            _write_exclusive(self.root / ("controls/" + name + ".raw"), exact)
            self._record("controls/" + name + ".record.json", record)
        _require(failure is None and not truncated and record["status"] == 200, "control HTTP failure: " + name)
        return dict(request_raw=body, response_raw=exact, record=record)


    def _read_cancel(self, connection, response, case):
        raw, abort, truncated = bytearray(), None, False
        stream = dict(schema=1, ordinal=22, case=case["case"], rid=CANCEL_RID,
            first_generated_monotonic=None, generated_prefix_bytes=None,
            generated_prefix_sha256=None, drained_monotonic=None, stream_eof=False, failure=None)
        try:
            while len(raw) <= MAX_RESPONSE_BYTES:
                self._set_socket_timeout(connection)
                reader = getattr(response, "read1", None) or response.read
                chunk = reader(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
                _require(type(chunk) is bytes, "SSE read returned non-bytes")
                if not chunk:
                    stream["stream_eof"] = True
                    break
                raw.extend(chunk)
                self._remaining()
                _require(len(raw) <= MAX_RESPONSE_BYTES, "SSE body exceeds bound")
                if abort is None:
                    events = _sse_events(bytes(raw), final=False)
                    generated = sum(_generated_event(event) for event in events)
                    if generated and stream["first_generated_monotonic"] is None:
                        stream["first_generated_monotonic"] = self.clock()
                    # The prefix binds the second complete content event that
                    # triggers abort; the first-event observation stays separate.
                    if generated >= 2:
                        _require(None not in events and all(not choice.get("finish_reason")
                            for event in events for choice in event.get("choices", [])), "generation already terminal before abort")
                        stream.update(generated_prefix_bytes=len(raw),
                            generated_prefix_sha256=_sha256(bytes(raw)))
                        abort = self._control("abort", "POST", "/abort_request",
                            serving_contract.canonical(dict(rid=CANCEL_RID, abort_all=False)))
        except (OSError, ValueError, UnicodeError, http.client.HTTPException) as error:
            truncated = True
            stream["failure"] = type(error).__name__ + ": " + str(error)[:512]
        finally:
            stream["drained_monotonic"] = self.clock()
            self._record("controls/cancel-stream.json", stream)
        if abort is not None:
            abort["stream"] = stream
        return bytes(raw[:MAX_RESPONSE_BYTES]), truncated, abort


    def _parallel(self, frozen, ordinals=(24,25), group_id=None):
        """Two owned threads, independent counters, one absolute socket deadline."""
        if group_id is not None:
            groups=serving_contract.mtp_contract.paired_groups(self.expected_gate['arm'])
            _require(any(g['id']==group_id and tuple(g['ordinals'])==tuple(ordinals) and g['concurrency']==2 for g in groups), 'unfrozen parallel group')
        else:
            _require(len(frozen)==27 and all(row.get('_kv_block')=='lifecycle' for row in frozen) and tuple(ordinals)==(24,25), 'legacy parallel group')
        barrier, results, failures, connections = threading.Barrier(2), {}, {}, {}
        local_counts = {i: dict(intent_count=0, send_attempt_entered_count=0, send_call_returned_count=0,
            response_observed_count=0, response_body_observed_count=0, normal_complete_count=0,
            aborted_complete_count=0) for i in ordinals}
        if group_id is not None:
            for row in local_counts.values(): row['valid_complete_count']=0
        def worker(ordinal):
            try:
                barrier.wait(timeout=self._remaining())
                results[ordinal] = self._one_closure(ordinal, frozen[ordinal], local_counts[ordinal], connections=connections)
            except Exception as error:
                failures[ordinal] = type(error).__name__ + ": " + str(error)[:512]
                barrier.abort()
        threads = [threading.Thread(target=worker, args=(i,), name="closure-http-%d" % i) for i in ordinals]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(max(0.0, self.deadline - self.clock()))
        finally:
            if any(thread.is_alive() for thread in threads):
                barrier.abort()
                for connection in list(connections.values()):
                    sock = getattr(connection, "sock", None)
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    connection.close()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(1.0)
        joined = not any(thread.is_alive() for thread in threads)
        self._record("controls/"+("concurrent-"+group_id if group_id is not None else "concurrent")+".json", dict(schema=1, ordinals=list(ordinals),
            joined=joined, failures={str(key): value for key, value in failures.items()},
            native_batch_size_proven=False))
        return [results[i] for i in ordinals if i in results], {
            key: sum(local_counts[i][key] for i in ordinals) for key in local_counts[ordinals[0]]}, (
            joined and not failures and set(results) == set(ordinals))


    def _run_kv_composition(self,frozen):
        expected_count=len(serving_contract.mtp_contract.settings()['requests'])
        _require(len(frozen)==expected_count,'frozen configured request count')
        original_root=self.root;error=None;blocks=[]
        try:
            self.root=original_root/'paired'
            for receipt in ('accepted-gate.json','live-workers-before-first-generation.json'):
                _write_exclusive(self.root/'readiness'/receipt,(original_root/'readiness'/receipt).read_bytes())
            block=self._run_paired(frozen)
            if serving_contract.mtp_contract.instrumentation_enabled():
                blocks.append(dict(name='paired',global_offset=0,summary=block))
            else:
                paired_raw=(self.root/'RUN-SUMMARY.json').read_bytes();paired_executed=block.get('executed_case_count',0)
                _require(paired_raw==serving_contract.canonical(block),'recorded paired summary bytes')
                blocks.append(dict(name='paired',global_offset=0,summary_sha256=hashlib.sha256(paired_raw).hexdigest(),
                                   summary_bytes=len(paired_raw)))
            _require(block.get('frozen_case_count')==block.get('executed_case_count')==expected_count and
                     block.get('paired_error') is None and
                     all(serving_contract.response_completed(row.get('contract',{})) for row in block['rows']),
                     'incomplete natural-stop paired block')
        except (ValueError,OSError,http.client.HTTPException) as failure:
            error=type(failure).__name__+': '+str(failure)[:512]
        finally:
            self.root=original_root
        count=sum(block['summary'].get('executed_case_count',0) if 'summary' in block else paired_executed for block in blocks)
        summary=dict(schema=1,mode=serving_contract.MODE,composition='configured-paired-v1',
            frozen_case_count=expected_count,executed_case_count=count,blocks=blocks,stopped_early=error is not None or count!=expected_count,
            error=error,qualification=False,precision_accepted=False)
        self._record('RUN-SUMMARY.json',summary)
        return summary

    def _run_closure(self, frozen):
        _require(len(frozen) == 27 and frozen[22]["body"].get("rid") == CANCEL_RID,
                 "closure schedule count or cancellation rid")
        rows, counts = [], dict(intent_count=0, send_attempt_entered_count=0, send_call_returned_count=0,
            response_observed_count=0, response_body_observed_count=0, normal_complete_count=0,
            aborted_complete_count=0)
        error, baseline = None, None
        try:
            for ordinal, case in enumerate(frozen):
                if ordinal == 25:
                    continue
                if ordinal == 14:
                    baseline = self._control("metrics-before-pressure", "GET", "/metrics", b"")
                    eviction_observation(baseline["response_raw"], self.expected_gate['arm'])
                group = (24, 25) if ordinal == 24 else (ordinal,)
                if self.request_observer is not None:
                    for index in group:
                        self.request_observer("before", index, frozen[index])
                if ordinal == 24:
                    pair, delta, joined = self._parallel(frozen)
                    rows.extend(pair)
                    for key in counts:
                        counts[key] += delta[key]
                    _require(joined, "concurrent threads failed or did not join; per-case counts retained")
                else:
                    rows.append(self._one_closure(ordinal, case, counts, cancel=(ordinal == 22)))
                if self.request_observer is not None:
                    for index in group:
                        self.request_observer("after", index, frozen[index])
                expected_status = "CANCELLED_COMPLETE" if ordinal == 22 else "NORMAL_STOP_COMPLETE"
                if any(row["contract"]["status"] != expected_status for row in rows[-len(group):]):
                    break
                if ordinal == 19:
                    after = self._control("metrics-after-pressure", "GET", "/metrics", b"")
                    pressure_observation(baseline["response_raw"], after["response_raw"], self.expected_gate['arm'])
        except (ValueError, OSError, http.client.HTTPException) as failure:
            error = type(failure).__name__ + ": " + str(failure)[:512]
            self._record("controls/lifecycle.failure.json", dict(schema=1, failure=error))
        summary = dict(schema=1, mode=serving_contract.MODE, qualification=False,
            code_correctness_graded=False, numerical_equivalence_established=False,
            timing_interpretation="closure HTTP lifecycle only; not native batch2, TTFT or performance",
            endpoint=ENDPOINT, frozen_case_count=len(frozen), executed_case_count=len(rows),
            counts=counts, rows=rows, lifecycle_error=error,
            stopped_early=len(rows) != len(frozen), completed_utc=self.utc_now())
        self._record("RUN-SUMMARY.json", summary)
        return summary


    def _one_closure(self, ordinal, case, counts, *, cancel=False, connections=None):
        name = _safe_name(case["case"])
        prefix = "%02d-%s" % (ordinal, name)
        body = serving_contract.canonical(case["body"])
        body_sha256 = _sha256(body)
        before_send = self.clock()
        self._remaining()
        _write_exclusive(self.root / ("requests/%s.raw" % prefix), body)
        self._record("requests/%s.intent.json" % prefix, dict(schema=1, ordinal=ordinal,
            case=name, body_sha256=body_sha256, prepared_monotonic=before_send,
            prepared_utc=self.utc_now(), intent_is_not_submission_proof=True))
        counts["intent_count"] += 1
        self._record("requests/%s.send-entered.json" % prefix, dict(schema=1, ordinal=ordinal,
            case=name, body_sha256=body_sha256, entered_monotonic=self.clock(), entered_utc=self.utc_now(),
            send_call_entered=True, submission_unproven_until_return=True))
        counts["send_attempt_entered_count"] += 1
        connection = None
        try:
            connection = self.factory("127.0.0.1", self.port, timeout=self._remaining())
            if connections is not None:
                connections[ordinal] = connection
            connection.request("POST", ENDPOINT, body=body, headers={
                "Content-Type": "application/json", "Accept": "text/event-stream" if cancel else "application/json"})
            sent_at = self.clock()
            self._record("requests/%s.send-returned.json" % prefix, dict(schema=1, ordinal=ordinal,
                case=name, returned_monotonic=sent_at, returned_utc=self.utc_now(), send_call_returned=True,
                request_body_sha256=body_sha256))
            counts["send_call_returned_count"] += 1
            self._set_socket_timeout(connection)
            response = connection.getresponse()
            status, reason = response.status, response.reason
            header_at = self.clock()
            counts["response_observed_count"] += 1
            self._record("responses/%s.headers.json" % prefix, dict(schema=1, ordinal=ordinal,
                case=name, status=status, reason=str(reason), headers_observed_monotonic=header_at,
                headers_observed_utc=self.utc_now(), response_observed=True))
            if cancel:
                raw, truncated, abort = self._read_cancel(connection, response, case)
            else:
                raw, truncated = self._read_bounded(connection, response)
            received_at = self.clock()
            raw_suffix = ".partial.raw" if truncated else ".raw"
            _write_exclusive(self.root / ("responses/%s%s" % (prefix, raw_suffix)), raw)
            counts["response_body_observed_count"] += 1
            wire = dict(schema=1, ordinal=ordinal, case=name, status=status, reason=str(reason),
                response_sha256=_sha256(raw), response_bytes=len(raw), received_monotonic=received_at,
                received_utc=self.utc_now(), response_observed=True,
                client_elapsed_seconds=received_at - before_send, response_truncated=truncated,
                raw_body_path="responses/%s%s" % (prefix, raw_suffix))
            self._record("responses/%s.record.json" % prefix, wire)
            _require(not truncated, "cancel stream failed or exceeded bound; bounded prefix retained" if cancel
                     else "response body exceeds bound; bounded prefix retained")
            if cancel:
                try:
                    contract = inspect_cancel_evidence(case, raw, abort)
                    _require(status == 200, "cancel stream HTTP status")
                except (ValueError, TypeError, KeyError, UnicodeError) as error:
                    contract = dict(schema=1, mode=serving_contract.MODE, case=name,
                        status="STRUCTURAL_REFUSAL", metric_eligible=False, qualification=False,
                        error=type(error).__name__ + ": " + str(error)[:512])
                if contract["status"] == "CANCELLED_COMPLETE":
                    counts["aborted_complete_count"] += 1
                return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=wire)
            try:
                decoded = _strict_json(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ClientRefusal) as error:
                contract = dict(schema=1, mode=serving_contract.MODE, case=name,
                    status="STRUCTURAL_REFUSAL", metric_eligible=False, qualification=False,
                    error="invalid JSON response: " + type(error).__name__,
                    raw_response_sha256=wire["response_sha256"])
            else:
                contract = serving_contract.inspect_response(case, decoded, wire["client_elapsed_seconds"])
                contract["raw_http_response_sha256"] = wire["response_sha256"]
                if (case['_kv_block'] in ('paired','context','lifecycle') and
                    serving_contract.mtp_contract.contract_for_arm(self.expected_gate['arm'])['enabled']):
                    contract['spec_tokens_details'] = serving_contract.inspect_speculation(decoded)
                if not 200 <= status < 300 and serving_contract.response_completed(contract):
                    contract.update(status="HTTP_STATUS_REFUSAL", metric_eligible=False)
            if contract["status"] == "NORMAL_STOP_COMPLETE":
                counts["normal_complete_count"] += 1
            if 'valid_complete_count' in counts and serving_contract.response_completed(contract):
                counts['valid_complete_count'] += 1
            return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=wire)
        except (OSError, http.client.HTTPException, ClientRefusal, ValueError) as error:
            failure = dict(schema=1, ordinal=ordinal, case=name, failure=type(error).__name__ + ": " + str(error)[:512],
                failure_monotonic=self.clock(), failure_utc=self.utc_now(), response_observed=False)
            self._record("responses/%s.failure.json" % prefix, failure)
            contract = dict(schema=1, mode=serving_contract.MODE, case=name, status="TRANSPORT_OR_STRUCTURAL_REFUSAL",
                metric_eligible=False, qualification=False, error=failure["failure"])
            return dict(ordinal=ordinal, case=name, body_sha256=body_sha256, contract=contract, wire=failure)
        finally:
            if connection is not None:
                connection.close()


    def _run_paired(self, frozen):
        """Frozen sequential/concurrent groups, one before/after snapshot per group."""
        groups=serving_contract.mtp_contract.paired_groups(self.expected_gate['arm'])
        rows=[];completed_groups=[];error=None
        counts=dict(intent_count=0,send_attempt_entered_count=0,send_call_returned_count=0,
            response_observed_count=0,response_body_observed_count=0,normal_complete_count=0,
            aborted_complete_count=0,valid_complete_count=0)
        try:
            for group in groups:
                ordinals=group['ordinals']
                if self.request_observer is not None:
                    for ordinal in ordinals:self.request_observer('before',ordinal,frozen[ordinal])
                if group['concurrency']==2:
                    pair,delta,joined=self._parallel(frozen,tuple(ordinals),group['id'])
                    rows.extend(pair)
                    for key in counts:counts[key]+=delta[key]
                    _require(joined,'paired concurrent threads failed or did not join; evidence retained')
                else:
                    pair=[]
                    for ordinal in ordinals:
                        row=self._one_closure(ordinal,frozen[ordinal],counts)
                        rows.append(row);pair.append(row)
                        _require(serving_contract.response_completed(row['contract']),'paired sequential response incomplete')
                _require(len(pair)==len(ordinals) and all(serving_contract.response_completed(r['contract']) for r in pair), 'paired response incomplete')
                if self.request_observer is not None:
                    for ordinal in ordinals:self.request_observer('after',ordinal,frozen[ordinal])
                timing=[]
                for ordinal in ordinals:
                    prefix='%02d-%s'%(ordinal,frozen[ordinal]['case'])
                    intent=_strict_json(_kv_receipts.read(self.root/('requests/'+prefix+'.intent.json')).decode())
                    sent=_strict_json(_kv_receipts.read(self.root/('requests/'+prefix+'.send-returned.json')).decode())
                    receipt=_strict_json(_kv_receipts.read(self.root/('responses/'+prefix+'.record.json')).decode())
                    timing.append(dict(prepared=intent['prepared_monotonic'],sent=sent['returned_monotonic'],received=receipt['received_monotonic']))
                record=serving_contract.paired_wire_group(group,timing,[r['contract'] for r in pair])
                self._record('controls/group-'+group['id']+'.json',record)
                completed_groups.append(record)
        except (ValueError,OSError,http.client.HTTPException) as failure:
            error=type(failure).__name__+': '+str(failure)[:512]
            self._record('controls/paired.failure.json',dict(schema=1,failure=error))
        rows.sort(key=lambda r:r['ordinal'])
        summary=dict(schema=1,mode=serving_contract.MODE,qualification=False,
            code_correctness_graded=False,numerical_equivalence_established=False,
            timing_interpretation='instrumented HTTP service throughput; combined TP2',
            screen_policy=serving_contract.screen_policy(),endpoint=ENDPOINT,
            frozen_case_count=len(frozen),executed_case_count=len(rows),counts=counts,
            rows=rows if serving_contract.mtp_contract.instrumentation_enabled() else
                [dict(r,contract=serving_contract.summary_contract(r['contract'])) for r in rows],
            paired_error=error,groups=completed_groups,stopped_early=len(rows)!=len(frozen),completed_utc=self.utc_now())
        self._record('RUN-SUMMARY.json',summary)
        return summary
