"""Bounded raw benchmark export and receipt; no runtime or archive extraction.

The exporter preserves malformed/partial payload bytes. Transport validity is
separate from numerical acceptance. The receiver cannot authorize a workload.
"""
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import stat
import threading


PREFIX = b"SPRY_BENCHMARK_BUNDLE_V2 "
HTTP_MODE = 'native-http-fp8-sprint01-v1'
INDEPENDENT_MODE = 'independent-bf16-reference-v1'
INDEPENDENT_MEMBER_LIMITS = {'result.json': 4 << 20}
HTTP_MEMBER_LIMITS = {'result.json': 8 << 20, 'http-client-bundle.json': 33 << 20,
                      'native-http-bundle.json': 6 << 20}
HTTP_STREAM_LIMIT = 64 << 20
RAW_TRANSPORT = 'j01-raw-attention-frames-v1'
RAW_PREFIX = b'SPRY_RAW_FRAME_V1 '
RAW_LOG_LIMIT_BYTES = 16 << 20
RAW_KV_BYTES = 15365 * 2 * 256 * 2
RAW_FILES = {'q.bf16': 6144, 'k.bf16': RAW_KV_BYTES, 'v.bf16': RAW_KV_BYTES,
             'native_attention.bf16': 6144, 'capture.json': 65536,
             'IDENTITY.json': 65536, 'INCOMPLETE.json': 65536}
RAW_MEMBER_LIMITS = dict(HTTP_MEMBER_LIMITS, **{
    f'raw-rank-{rank}/{name}': limit for rank in (0, 1) for name, limit in RAW_FILES.items()})
RAW_PAYLOAD_LIMIT = sum(RAW_MEMBER_LIMITS.values())
RAW_WIRE_LIMIT = RAW_PAYLOAD_LIMIT + RAW_LOG_LIMIT_BYTES + 65536
# The wire is streamed to its final files, never retained a second time.
# Keep the existing 24 MiB client tree, 4 MiB native tree, 1 MiB per-arm
# metadata, and all 88.5 MiB owner/source/observer/closeout partitions.
RAW_SESSION_LIMIT = 2 * (RAW_PAYLOAD_LIMIT + 2 * RAW_LOG_LIMIT_BYTES + (29 << 20)) + int(88.5 * (1 << 20))
assert RAW_SESSION_LIMIT < 512 << 20
# Count both stdio streams, decoded outer payload, decoded inner trees,
# and the unchanged owner/source/observer/closeout partitions for both arms.
HTTP_SESSION_MAX_BYTES = (2 * (2 * HTTP_STREAM_LIMIT + sum(HTTP_MEMBER_LIMITS.values()) +
                             (24 << 20) + (4 << 20) + (1 << 20)) +
                          (32 << 20) + (20 << 20) + (512 << 10) +
                          (16 << 20) + (16 << 20) + (4 << 20))
assert HTTP_SESSION_MAX_BYTES < 512 << 20
MEMBER_LIMITS = {"rank-0.json": 8 << 20, "rank-1.json": 8 << 20,
                 "result.json": 8 << 20, "rank-0-identity.json": 64 << 10,
                 "rank-1-identity.json": 64 << 10,
                 "rank-0.log.tail": 2 << 20, "rank-1.log.tail": 2 << 20,
                 "rank-0-progress.jsonl": 64 << 10, "rank-1-progress.jsonl": 64 << 10,
                 "owned-workers.json": 64 << 10, "diagnostics.json": 64 << 10}
RAW_LOG_LIMIT = 128 << 20
BUNDLE_LIMIT = 40 << 20
STREAM_LIMIT = 80 << 20
# 3 * ceil(8 MiB / 3) * 4 + 2 * ceil(2 MiB / 3) * 4
# + 6 * ceil(64 KiB / 3) * 4,
# plus 8192 bytes for the fixed JSON keys, names, hashes and binding.
MAX_PAYLOAD_BYTES = sum(MEMBER_LIMITS.values())
MAX_BASE64_BYTES = sum(4 * ((limit + 2) // 3) for limit in MEMBER_LIMITS.values())
MAX_ENCODED_BYTES = len(PREFIX) + MAX_BASE64_BYTES + 8192 + 1
MAX_DURABLE_BYTES = 2 * STREAM_LIMIT + MAX_PAYLOAD_BYTES + (1 << 20)
# Both copies of every observer snapshot/log/control record are included.
# Transport must enforce these independent partitions, not just this arithmetic.
SOURCE_COPIES_LIMIT = 32 << 20
OBSERVER_ALL_COPIES_LIMIT = (20 << 20) + (512 << 10)
OUTER_STDIO_LIMIT = 16 << 20
SUMMARY_AND_CLOSEOUT_LIMIT = 16 << 20
SMALL_METADATA_LIMIT = 4 << 20
MAX_SESSION_DURABLE_BYTES = 2 * MAX_DURABLE_BYTES + SOURCE_COPIES_LIMIT + OBSERVER_ALL_COPIES_LIMIT + \
    OUTER_STDIO_LIMIT + SUMMARY_AND_CLOSEOUT_LIMIT + SMALL_METADATA_LIMIT
assert MAX_ENCODED_BYTES < BUNDLE_LIMIT
assert MAX_DURABLE_BYTES < 512 << 20
# 378.75 MiB for both arms + 32 source + 20.5 observers + 16 outer stdio
# + 16 summaries/closeouts + 4 small metadata = 467.25 MiB. The remaining
# 44.75 MiB is not request time, runtime memory or an output-overflow allowance.
assert MAX_SESSION_DURABLE_BYTES < 512 << 20

CAPTURE_MODE = 'v4-first-divergence-capture-v1'
NATIVE_CAPTURE_MODE = 'native-repeatability-capture-v1'
DECODE_LAYER_CAPTURE_MODE = 'native-decode-layer-capture-v1'
PREFILL_ROW_AUDIT_MODE = 'native-prefill-row-audit-v1'
ASYNC_PREFILL_MODE = 'native-async-prefill-layer-capture-v1'
OPERATION_MEMBER_LIMITS = {'rank-0-operation.json': 4096, 'rank-1-operation.json': 4096}
NATIVE_CACHE_MEMBER_LIMITS = {'rank-0-native-cache.json': 8192, 'rank-1-native-cache.json': 8192}
PREFILL_ROW_MEMBER_LIMITS = {'rank-0-prefill-rows.jsonl': 256 << 10,
                             'rank-1-prefill-rows.jsonl': 256 << 10}
# Each arm retains two 86 MiB streams, two 2 MiB captures and two small
# native identities. These additions fit the original fixed session cap.
NATIVE_MAX_SESSION_DURABLE_BYTES = MAX_SESSION_DURABLE_BYTES + 2 * (
    2 * ((86 << 20) - STREAM_LIMIT) + (4 << 20) + sum(NATIVE_CACHE_MEMBER_LIMITS.values()))
assert NATIVE_MAX_SESSION_DURABLE_BYTES < 512 << 20
# Decode-layer capture exchanges 6 MiB from each of four stdio streams for
# 6 MiB in each of four capture files. All other partitions stay unchanged.
# The mode therefore retains the same 499.28125 MiB worst-case session bound.
DECODE_LAYER_MAX_SESSION_DURABLE_BYTES = MAX_SESSION_DURABLE_BYTES + 2 * (
    (16 << 20) + sum(NATIVE_CACHE_MEMBER_LIMITS.values()) + sum(OPERATION_MEMBER_LIMITS.values()))
assert DECODE_LAYER_MAX_SESSION_DURABLE_BYTES < 512 << 20
# The row-audit sidecars add at most 256 KiB for each rank in each arm. The
# decode binary capture remains unchanged; operation journals are not emitted.
PREFILL_ROW_AUDIT_MAX_SESSION_DURABLE_BYTES = DECODE_LAYER_MAX_SESSION_DURABLE_BYTES + (2 << 20) + 49152
assert PREFILL_ROW_AUDIT_MAX_SESSION_DURABLE_BYTES < 512 << 20
# Exchange 4 MiB of each of four bounded stdio streams for 4 MiB of each
# capture file. The callback and trace metadata retain their existing caps.
ASYNC_PREFILL_MAX_SESSION_DURABLE_BYTES = (
    DECODE_LAYER_MAX_SESSION_DURABLE_BYTES + (1 << 20) + 49152)
assert ASYNC_PREFILL_MAX_SESSION_DURABLE_BYTES < 512 << 20


@dataclass(frozen=True)
class HTTPArtifactLimits:
    """Configured HTTP partitions under the retained protected ceilings."""
    result_json_bytes: int = 8 << 20
    client_bundle_bytes: int = 33 << 20
    native_bundle_bytes: int = 6 << 20
    http_stream_bytes: int = 64 << 20
    client_tree_bytes: int = 24 << 20
    native_tree_bytes: int = 4 << 20
    arm_metadata_bytes: int = 1 << 20
    source_copies_bytes: int = 32 << 20
    observer_copies_bytes: int = (20 << 20) + (512 << 10)
    outer_stdio_bytes: int = 16 << 20
    summary_closeout_bytes: int = 16 << 20
    small_metadata_bytes: int = 4 << 20
    session_bytes: int = 512 << 20
    http_member_read_bytes: int = 8 << 20
    client_tree_files: int = 512
    native_tree_files: int = 256

    def __post_init__(self):
        ceilings = {name: field.default for name, field in self.__dataclass_fields__.items()}
        need(all(type(value) is int and 0 < value <= ceilings[name]
                 for name, value in asdict(self).items()), 'HTTP artifact ceiling or type differs')
        need(self.http_session_max_bytes < self.session_bytes,
             'HTTP durable partitions exceed configured session ceiling')
        need(self.encoded_bytes <= self.http_stream_bytes and self.encoded_bytes <= 63 << 20,
             'HTTP encoded partitions exceed stream/bundle ceiling')

    @classmethod
    def from_limits(cls, values):
        need(type(values) is dict and set(cls.__dataclass_fields__) <= set(values)
             and not {'http_payload_bytes', 'http_session_max_bytes'} & set(values),
             'primary HTTP artifact limits required')
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})

    @property
    def members(self):
        return dict(zip(HTTP_MEMBER_LIMITS, (self.result_json_bytes,
            self.client_bundle_bytes, self.native_bundle_bytes), strict=True))

    @property
    def encoded_bytes(self):
        return len(PREFIX) + sum(4 * ((n + 2) // 3) for n in self.members.values()) + 8193

    @property
    def http_session_max_bytes(self):
        return (2 * (2 * self.http_stream_bytes + sum(self.members.values()) +
                    self.client_tree_bytes + self.native_tree_bytes + self.arm_metadata_bytes)
                + self.source_copies_bytes + self.observer_copies_bytes + self.outer_stdio_bytes
                + self.summary_closeout_bytes + self.small_metadata_bytes)


def _http_limits(binding, limits):
    need(limits is None or (binding.get('experiment_mode') == HTTP_MODE
         and type(limits) is HTTPArtifactLimits), 'HTTP artifact plan required')
    if limits is not None:
        # Revalidate even a frozen instance: object.__setattr__ must not turn
        # a corrupted plan into permission to increase an owner ceiling.
        limits.__post_init__()
    return HTTPArtifactLimits() if limits is None else limits


def audit_durable_trees(workload_root, *, limits, protected_root=None, pending=()):
    """Account finalized evidence and planned seals; never authorize execution."""
    plan = _http_limits({'experiment_mode': HTTP_MODE}, limits)
    caps = {key: getattr(plan, key) for key in ('source_copies_bytes', 'observer_copies_bytes',
        'outer_stdio_bytes', 'summary_closeout_bytes', 'small_metadata_bytes')}
    for arm in ('bf16', 'kv4'):
        caps.update({arm + ':member:' + name: maximum for name, maximum in plan.members.items()})
        caps.update({arm + ':stream:' + name: plan.http_stream_bytes for name in ('stdout.raw', 'stderr.raw')})
        caps[arm + ':client'] = plan.client_tree_bytes
        caps[arm + ':metadata'] = plan.arm_metadata_bytes
    used = dict.fromkeys(caps, 0)
    counts, names, stamps = {}, set(), {}
    total = 0

    def charge(scope, name, size):
        nonlocal total
        need(type(name) is str, 'canonical durable evidence identity required')
        relative = PurePosixPath(name)
        need(scope in ('workload', 'protected', 'workload-closeout', 'protected-closeout')
             and not relative.is_absolute() and str(relative) == name
             and '..' not in relative.parts and name != '.' and type(size) is int and size >= 0,
             'canonical durable evidence identity required')
        key = (scope, name)
        need(key not in names, 'duplicate durable evidence or pending seal')
        names.add(key)
        total += size
        buckets = []
        if scope.endswith('-closeout'):
            buckets = ['summary_closeout_bytes']
        elif scope == 'protected':
            if name.startswith('review/') or name == 'payload.tar':
                buckets = ['source_copies_bytes']
            elif name in ('evidence/controller-capture/stdout.raw', 'evidence/controller-capture/stderr.raw'):
                need(size <= min(8 << 20, plan.outer_stdio_bytes // 2), 'configured outer stream limit')
                buckets = ['outer_stdio_bytes']
            elif name == 'evidence/ROOT-TRANSPORT.json':
                buckets = ['summary_closeout_bytes', 'observer_copies_bytes']
            elif name.startswith('evidence/') and name not in (
                    'evidence/GRANT.json', 'evidence/RECEIPT.json', 'evidence/CONTROLLER-IDENTITY.json'):
                buckets = ['observer_copies_bytes']
            else:
                buckets = ['small_metadata_bytes']
        elif name.startswith(('review/', 'config/')) or name in (
                'MANIFEST.json', 'PROTECTED-SHA256.json', 'DERIVED-LIMITS.json', 'SOURCE-BUILD.json'):
            buckets = ['source_copies_bytes']
        elif name == 'results/CONTROLLER-RESULT.json':
            # Charging complete reports conservatively includes every embedded
            # observer snapshot/control copy, even when fields are repeated.
            buckets = ['summary_closeout_bytes', 'observer_copies_bytes']
        else:
            for arm in ('bf16', 'kv4'):
                capture = 'results/' + arm + '-capture/'
                artifacts = 'results/' + arm + '-artifacts/'
                client = 'results/' + arm + '-http-client-tree/'
                if name.startswith(capture) and name[len(capture):] in ('stdout.raw', 'stderr.raw'):
                    buckets = [arm + ':stream:' + name[len(capture):]]
                elif name.startswith(artifacts):
                    member = name[len(artifacts):]
                    need(member in plan.members, 'unexpected configured HTTP artifact member')
                    buckets = [arm + ':member:' + member]
                elif name.startswith(client):
                    need(size <= plan.http_member_read_bytes, 'configured decoded member bound')
                    bucket = arm + ':client'
                    counts[bucket] = counts.get(bucket, 0) + 1
                    need(counts[bucket] <= plan.client_tree_files, 'configured decoded client file count')
                    buckets = [bucket]
                elif name.startswith('results/' + arm + '-'):
                    buckets = [arm + ':metadata']
                if buckets:
                    break
            if not buckets:
                buckets = ['small_metadata_bytes']
        for bucket in buckets:
            used[bucket] += size
            need(used[bucket] <= caps[bucket], 'configured durable partition exceeded: ' + bucket)
        need(total <= min(plan.session_bytes, plan.http_session_max_bytes), 'configured complete session byte bound')

    roots = [] if workload_root is None else [('workload', Path(workload_root))]
    if protected_root is not None:
        roots.append(('protected', Path(protected_root)))
    need(bool(roots), 'durable evidence root required')
    for _, root in roots:
        need(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(),
             'canonical durable evidence root required')
    if len(roots) == 2:
        left, right = (row[1] for row in roots)
        need(not left.is_relative_to(right) and not right.is_relative_to(left), 'independent durable evidence roots required')
    for scope, root in roots:
        for path in (root, *sorted(root.rglob('*'))):
            before = path.lstat()
            need(not stat.S_ISLNK(before.st_mode), 'durable evidence link refused')
            stamps[path] = _stamp(before)
            if stat.S_ISDIR(before.st_mode):
                continue
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'regular singly linked durable evidence required')
            charge(scope, path.relative_to(root).as_posix(), before.st_size)
        for suffix in ('.closeout.json', '.refusal.json'):
            closeout = root.with_name(root.name + suffix)
            if closeout.exists() or closeout.is_symlink():
                before = closeout.lstat()
                need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and closeout.resolve() == closeout,
                     'canonical durable closeout required')
                stamps[closeout] = _stamp(before)
                charge(scope + '-closeout', suffix[1:], before.st_size)
    for scope, name, size in pending:
        charge(scope, name, size)
    need(all(_stamp(path.lstat()) == stamp for path, stamp in stamps.items()), 'durable evidence changed during audit')
    return dict(total_bytes=total, partition_bytes=used, files=len(names),
        remaining_summary_bytes=plan.summary_closeout_bytes - used['summary_closeout_bytes'],
        remaining_session_bytes=min(plan.session_bytes, plan.http_session_max_bytes) - total,
        observer_reports_charged_conservatively=True, qualification=False)


def member_limits(binding, *, limits=None):
    policy = _http_limits(binding, limits)
    mode = binding.get('experiment_mode')
    if mode == INDEPENDENT_MODE:
        return dict(INDEPENDENT_MEMBER_LIMITS)
    if mode == HTTP_MODE:
        return policy.members
    if mode == "native-model-context-retrieval-32k-v1":
        return dict(MEMBER_LIMITS, **NATIVE_CACHE_MEMBER_LIMITS, **{
            "rank-0-graph-potential.json": 256 << 10, "rank-1-graph-potential.json": 256 << 10})
    capture_limit = (12 if mode == ASYNC_PREFILL_MODE else
                     8 if mode in (DECODE_LAYER_CAPTURE_MODE, PREFILL_ROW_AUDIT_MODE) else 2) << 20
    limits = (dict(MEMBER_LIMITS, **{'rank-0-capture.bin': capture_limit, 'rank-1-capture.bin': capture_limit})
        if mode in (CAPTURE_MODE, NATIVE_CAPTURE_MODE, DECODE_LAYER_CAPTURE_MODE,
                    PREFILL_ROW_AUDIT_MODE, ASYNC_PREFILL_MODE) else MEMBER_LIMITS)
    if mode in (NATIVE_CAPTURE_MODE, DECODE_LAYER_CAPTURE_MODE, PREFILL_ROW_AUDIT_MODE,
                ASYNC_PREFILL_MODE):
        limits.update(NATIVE_CACHE_MEMBER_LIMITS)
    if mode == DECODE_LAYER_CAPTURE_MODE:
        limits.update(OPERATION_MEMBER_LIMITS)
    if mode == PREFILL_ROW_AUDIT_MODE:
        limits.update(PREFILL_ROW_MEMBER_LIMITS)
    if mode in (PREFILL_ROW_AUDIT_MODE, ASYNC_PREFILL_MODE):
        limits.update({'rank-0-callback-profile.json': 256 << 10, 'rank-1-callback-profile.json': 256 << 10})
    if mode in (PREFILL_ROW_AUDIT_MODE, ASYNC_PREFILL_MODE):
        limits.update({'rank-0-trace-ready.json': 4096, 'rank-0-trace-intent.json': 4096, 'rank-0-trace-outcome.json': 4096, 'rank-1-trace-ready.json': 4096, 'rank-1-trace-intent.json': 4096, 'rank-1-trace-outcome.json': 4096})
    return limits


def stream_limit(mode, *, limits=None):
    policy = _http_limits({'experiment_mode': mode}, limits)
    if mode == INDEPENDENT_MODE:
        return 8 << 20
    if mode == HTTP_MODE:
        return policy.http_stream_bytes
    if mode == "native-model-context-retrieval-32k-v1":
        return STREAM_LIMIT
    need(mode in ('model-benchmark-v1', 'v4-repeatability-control-v1', CAPTURE_MODE,
                 NATIVE_CAPTURE_MODE, DECODE_LAYER_CAPTURE_MODE, PREFILL_ROW_AUDIT_MODE,
                 ASYNC_PREFILL_MODE), 'artifact mode')
    return (76 if mode == ASYNC_PREFILL_MODE else
            86 if mode in (CAPTURE_MODE, NATIVE_CAPTURE_MODE) else 80) << 20


def bundle_limit(binding, *, limits=None):
    policy = _http_limits(binding, limits)
    if binding.get('experiment_mode') == HTTP_MODE:
        return 63 << 20 if limits is None else policy.encoded_bytes
    if binding.get('experiment_mode') in (DECODE_LAYER_CAPTURE_MODE, PREFILL_ROW_AUDIT_MODE,
                                          ASYNC_PREFILL_MODE):
        return (72 if binding.get('experiment_mode') == ASYNC_PREFILL_MODE else 61) << 20
    return (46 if binding.get('experiment_mode') in (CAPTURE_MODE, NATIVE_CAPTURE_MODE) else 40) << 20


def encoded_limit(binding, *, limits=None):
    return len(PREFIX) + sum(4 * ((n + 2) // 3) for n in member_limits(binding, limits=limits).values()) + 8193


def need(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _binding(value):
    fields = {"session_sha256", "arm", "spec_sha256"}
    need(type(value) is dict, 'artifact binding object')
    if 'experiment_mode' in value:
        need(value['experiment_mode'] in (CAPTURE_MODE, NATIVE_CAPTURE_MODE,
             DECODE_LAYER_CAPTURE_MODE, PREFILL_ROW_AUDIT_MODE, ASYNC_PREFILL_MODE, 'native-model-context-retrieval-32k-v1', HTTP_MODE,
             INDEPENDENT_MODE),
             'explicit capture artifact mode')
        fields.add('experiment_mode')
    if 'artifact_transport' in value:
        need(value.get('experiment_mode') == HTTP_MODE and
             value['artifact_transport'] == RAW_TRANSPORT, 'raw transport requires HTTP mode')
        fields.add('artifact_transport')
    need(set(value) == fields,
         "exact artifact binding required")
    need(value["arm"] in ("bf16", "kv4") and _digest(value["session_sha256"])
         and _digest(value["spec_sha256"]), "artifact binding identity")
    return value


def _directory(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.normpath(str(path)), "canonical absolute directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _stamp(value):
    return tuple(getattr(value, key) for key in ("st_dev", "st_ino", "st_mode", "st_nlink",
        "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _read_member(directory, name, limit):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
             "bounded regular singly linked artifact required: " + name)
        raw = bytearray()
        while len(raw) <= limit:
            block = os.read(fd, min(65536, limit + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        need(len(raw) == before.st_size and len(raw) <= limit and _stamp(before) == _stamp(os.fstat(fd)) ==
             _stamp(os.stat(name, dir_fd=directory, follow_symlinks=False)), "artifact changed/overflowed")
        return bytes(raw)
    finally:
        os.close(fd)


def _member(directory, name, limit):
    raw = _read_member(directory, name, limit)
    if raw is None:
        return {"exists": False}
    return {"exists": True, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii")}


def _write_member(directory, name, raw):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _stage_log(source, destination, rank):
    name, tail_name = f"rank-{rank}.log", f"rank-{rank}.log.tail"
    missing = {"exists": False, "missing": True, "original_bytes": None, "full_sha256": None,
               "tail_file": tail_name, "tail_bytes": None, "tail_sha256": None,
               "tail_offset": None, "truncated": False}
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source)
    except FileNotFoundError:
        return missing
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= RAW_LOG_LIMIT,
             "bounded regular singly linked rank log required: " + name)
        digest, size = hashlib.sha256(), 0
        while size <= RAW_LOG_LIMIT:
            block = os.read(fd, min(65536, RAW_LOG_LIMIT + 1 - size))
            if not block:
                break
            digest.update(block)
            size += len(block)
        need(size == before.st_size and size <= RAW_LOG_LIMIT, "rank log changed/overflowed")
        offset = max(0, size - MEMBER_LIMITS[tail_name])
        os.lseek(fd, offset, os.SEEK_SET)
        tail = bytearray()
        while len(tail) < size - offset:
            block = os.read(fd, min(65536, size - offset - len(tail)))
            need(block, "rank log shortened during tail capture")
            tail.extend(block)
        need(_stamp(before) == _stamp(os.fstat(fd)) ==
             _stamp(os.stat(name, dir_fd=source, follow_symlinks=False)), "rank log changed during capture")
        _write_member(destination, tail_name, tail)
        return {"exists": True, "missing": False, "original_bytes": size, "full_sha256": digest.hexdigest(),
                "tail_file": tail_name, "tail_bytes": len(tail), "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "tail_offset": offset, "truncated": offset > 0}
    finally:
        os.close(fd)


def _http_record(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                base64=base64.b64encode(raw).decode('ascii'))


def _http_tree(path, maximum, files=128, member_bytes=8 << 20):
    result, total = {}, 0
    if not path.exists():
        return result
    need(path.is_dir() and not path.is_symlink(), 'HTTP evidence tree must be an owned directory')
    entries = list(path.rglob('*'))
    need(len(entries) <= files + 16, 'HTTP evidence member count')
    for entry in entries:
        need(not entry.is_symlink() and (entry.is_dir() or entry.is_file()), 'HTTP linked/special member')
        if entry.is_file():
            name = entry.relative_to(path).as_posix()
            need(all(part not in ('', '.', '..') for part in name.split('/')), 'HTTP relative member')
            directory = _directory(entry.parent)
            try:
                raw = _read_member(directory, entry.name, min(maximum, member_bytes))
            finally:
                os.close(directory)
            need(raw is not None, 'HTTP member vanished')
            total += len(raw)
            need(total <= maximum and len(result) < files, 'HTTP evidence aggregate bound')
            result[name] = _http_record(raw)
    return result


def pack_http_artifacts(output, binding, *, limits=None):
    """Pack owned raw trees after native cleanup; missing evidence stays missing."""
    _binding(binding)
    need(binding.get('experiment_mode') == HTTP_MODE, 'explicit HTTP export mode')
    policy = _http_limits(binding, limits)
    output = Path(output)
    client = dict(schema=1, mode=HTTP_MODE, files=_http_tree(output/'client', policy.client_tree_bytes,
        files=policy.client_tree_files, member_bytes=policy.http_member_read_bytes))
    native = _http_tree(output/'native', policy.native_tree_bytes, files=policy.native_tree_files,
                        member_bytes=policy.http_member_read_bytes)
    gate = native.pop('NATIVE-READINESS.json', None)
    workers = []
    for rank in (0, 1):
        row = native.pop(f'worker-rank-{rank}.json', None)
        if row is not None:
            workers.append(_json(base64.b64decode(row['base64'], validate=True)))
    value = dict(schema=1, mode=HTTP_MODE, gate=gate, workers=workers, observations=native)
    for name, body in (('http-client-bundle.json', client), ('native-http-bundle.json', value)):
        raw = _canonical(body)+b'\n'
        need(len(raw) <= policy.members[name], 'HTTP encoded aggregate bound')
        with (output/name).open('xb') as stream:
            stream.write(raw)


def stage_artifacts(output, binding, *, limits=None):
    """Stage bounded raw evidence after worker reaping; never parse partial results.

    Full rank logs are streamed for their hashes, then only their bounded binary
    tails are retained. Missing evidence stays explicit. Source files are never
    changed, and an existing export directory is never reused.
    """
    _binding(binding)
    selected = member_limits(binding, limits=limits)
    output = Path(output)
    source = _directory(output)
    destination = None
    try:
        os.mkdir("export", mode=0o700, dir_fd=source)
        destination = os.open("export", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=source)
        before, names = os.fstat(source), set(os.listdir(source))
        for name, limit in selected.items():
            if name == "diagnostics.json" or name.endswith(".log.tail"):
                continue
            raw = _read_member(source, name, limit)
            if raw is not None:
                _write_member(destination, name, raw)
        if binding.get('experiment_mode') in (HTTP_MODE, INDEPENDENT_MODE):
            need(set(os.listdir(source)) == names and _stamp(before) == _stamp(os.fstat(source)),
                 'HTTP artifact source changed during staging')
            os.fsync(destination)
            return output / 'export'
        logs = {str(rank): _stage_log(source, destination, rank) for rank in (0, 1)}
        diagnostics = _canonical({"schema": 1, "binding": dict(binding), "logs": logs, "errors": []}) + b"\n"
        need(len(diagnostics) <= MEMBER_LIMITS["diagnostics.json"], "diagnostic metadata overflow")
        need(set(os.listdir(source)) == names and _stamp(before) == _stamp(os.fstat(source)),
             "artifact source directory changed during staging")
        _write_member(destination, "diagnostics.json", diagnostics)
        os.fsync(destination)
        return output / "export"
    finally:
        if destination is not None:
            os.close(destination)
        os.close(source)


def export_bundle(output, binding, sink, *, limits=None):
    """Export the exact mode-bound V2 members; missing files remain explicit."""
    _binding(binding)
    directory = _directory(output)
    try:
        before = os.fstat(directory)
        names = set(os.listdir(directory))
        need(names <= set(member_limits(binding, limits=limits)), "extra export member or directory")
        members = {name: _member(directory, name, limit) for name, limit in member_limits(binding, limits=limits).items()}
        need(set(os.listdir(directory)) == names and _stamp(before) == _stamp(os.fstat(directory)),
             "export directory changed")
        value = {"schema": 2, "binding": dict(binding), "members": members}
        line = PREFIX + _canonical(value) + b"\n"
        need(len(line) <= encoded_limit(binding, limits=limits) <= bundle_limit(binding, limits=limits), "encoded bundle overflow")
        sink.write(line)
        sink.flush()
        return value
    finally:
        os.close(directory)


def receive_bundle(stdout, stderr, destination, binding, *, limits=None):
    """Validate one raw transport record, then exclusively retain original bytes."""
    _binding(binding)
    need(type(stdout) is bytes and type(stderr) is bytes and
         len(stdout) <= stream_limit(binding.get('experiment_mode', 'model-benchmark-v1'), limits=limits)
         and len(stderr) <= stream_limit(binding.get('experiment_mode', 'model-benchmark-v1'), limits=limits), "stdio capture overflow")
    split = stdout.splitlines(keepends=True)
    lines = [line for line in split if line.startswith(b"SPRY_BENCHMARK_BUNDLE_")]
    need(len(lines) == 1 and lines[0].startswith(PREFIX) and len(lines[0]) <= bundle_limit(binding, limits=limits),
         "exactly one bounded V2 artifact bundle")
    need(lines[0].endswith(b"\n"), "incomplete artifact record line")
    value = _json(lines[0][len(PREFIX):-1])
    need(type(value) is dict and set(value) == {"schema", "binding", "members"} and
         type(value["schema"]) is int and value["schema"] == 2 and value["binding"] == binding,
         "artifact schema/binding mismatch")
    _binding(value["binding"])
    need(type(value["members"]) is dict and set(value["members"]) == set(member_limits(binding, limits=limits)),
         "exact artifact members required")
    decoded = {}
    for name, limit in member_limits(binding, limits=limits).items():
        member = value["members"][name]
        need(type(member) is dict and type(member.get("exists")) is bool, "artifact presence type")
        if not member["exists"]:
            need(set(member) == {"exists"}, "missing artifact has extra data")
            continue
        need(set(member) == {"exists", "bytes", "sha256", "base64"} and
             type(member["bytes"]) is int and 0 <= member["bytes"] <= limit and
             _digest(member["sha256"]) and type(member["base64"]) is str and
             len(member["base64"]) <= 4 * ((limit + 2) // 3), "artifact shape/size/hash")
        raw = base64.b64decode(member["base64"], validate=True)
        need(base64.b64encode(raw).decode("ascii") == member["base64"] and
             len(raw) == member["bytes"] and hashlib.sha256(raw).hexdigest() == member["sha256"],
             "artifact payload identity")
        decoded[name] = raw
    destination = Path(destination)
    parent = _directory(destination.parent)
    directory = None
    try:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        directory = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        info = os.fstat(directory)
        need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, "private owned output required")
        for name, raw in decoded.items():
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
        return {"schema": 2, "binding": dict(binding), "transport_valid": True,
                "all_members_present": len(decoded) == len(member_limits(binding, limits=limits)),
                "qualification": False, "members": {name: {key: item for key, item in record.items()
                if key != "base64"} for name, record in value["members"].items()},
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "retained_payload_bytes": sum(len(raw) for raw in decoded.values())}
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


class CapturePipes:
    """One bounded drain thread for an existing authenticated process owner.

    Pass stdout/stderr to the owner's subprocess, call check from its resource
    monitor, then finish after owned children have been reaped. Overflow retains
    at most the cap and fails; it is never a truncated successful collection.
    This class does not signal, own or attest release of any process.
    """
    def __init__(self, output_dir, *, stream_limit=None, mode='model-benchmark-v1',
                 raw_binding=None, raw_destination=None, limits=None):
        ceiling = globals()['stream_limit'](mode, limits=limits)
        self.stream_limit = ceiling if stream_limit is None else stream_limit
        need(type(self.stream_limit) is int and 0 < self.stream_limit <= ceiling,
             "bounded capture stream limit")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(mode=0o700)
        self.directory = _directory(self.output_dir)
        info = os.fstat(self.directory)
        need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
             "private owned capture directory")
        self.sizes = {"stdout": 0, "stderr": 0}
        self.hashes = {name: hashlib.sha256() for name in self.sizes}
        self.failure = None
        self.selector = selectors.DefaultSelector()
        self.writers, self.outputs = {}, {}
        self.stop = threading.Event()
        self.finished = False
        self.raw_receiver = None
        self.wire_sizes = {"stdout": 0, "stderr": 0}
        try:
            for name in self.sizes:
                self.outputs[name] = os.fdopen(os.open(name + ".raw", os.O_WRONLY | os.O_CREAT |
                    os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory), "wb")
            if raw_binding is not None:
                need(mode == HTTP_MODE and raw_destination is not None,
                     'explicit raw capture destination/mode')
                self.raw_receiver = RawFrameReceiver(raw_destination, raw_binding,
                                                    self.outputs['stdout'])
            else:
                need(raw_destination is None, 'unbound raw destination')
            for name in self.sizes:
                read_fd, write_fd = os.pipe()
                os.set_blocking(read_fd, False)
                self.selector.register(read_fd, selectors.EVENT_READ, name)
                self.writers[name] = os.fdopen(write_fd, "wb", buffering=0)
            self.stdout, self.stderr = self.writers["stdout"], self.writers["stderr"]
            self.thread = threading.Thread(target=self._drain, name="spry-bounded-stdio", daemon=True)
            self.thread.start()
        except BaseException:
            self._close()
            raise

    def _drain(self):
        try:
            while self.selector.get_map() and not self.stop.is_set():
                for key, _ in self.selector.select(0.05):
                    block = os.read(key.fd, 65536)
                    if not block:
                        self.selector.unregister(key.fd)
                        os.close(key.fd)
                        continue
                    name = key.data
                    if self.raw_receiver is not None:
                        self.wire_sizes[name] += len(block)
                        if self.failure is not None:
                            continue  # Drain for the owner; never accept after refusal.
                        try:
                            if name == 'stdout':
                                need(self.wire_sizes[name] <= RAW_WIRE_LIMIT, 'raw wire overflow')
                                self.raw_receiver.feed(block)
                            else:
                                need(self.wire_sizes[name] <= RAW_LOG_LIMIT_BYTES, 'raw stderr overflow')
                                self.outputs[name].write(block)
                        except Exception as error:
                            self.failure = type(error).__name__ + ': ' + str(error)
                        continue
                    remaining = self.stream_limit - self.sizes[name]
                    retained = block[:remaining]
                    self.outputs[name].write(retained)
                    self.hashes[name].update(retained)
                    self.sizes[name] += len(retained)
                    if len(block) > remaining:
                        self.failure = self.failure or name + " hard capture overflow"
        except BaseException as error:
            self.failure = self.failure or "capture " + type(error).__name__

    def check(self):
        need(self.failure is None, self.failure or "capture failure")

    def _close(self):
        for writer in self.writers.values():
            if not writer.closed:
                writer.close()
        for key in list(self.selector.get_map().values()):
            self.selector.unregister(key.fd)
            os.close(key.fd)
        self.selector.close()
        for output in self.outputs.values():
            if not output.closed:
                output.flush()
                os.fsync(output.fileno())
                output.close()
        os.close(self.directory)

    def finish(self):
        need(not self.finished, "capture already finalized")
        self.finished = True
        for writer in self.writers.values():
            writer.close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.failure = self.failure or "capture pipe owner survives"
            self.stop.set()
            self.thread.join(timeout=1)
        need(not self.thread.is_alive(), "capture reader failed to terminate")
        raw_receipt = None
        if self.raw_receiver is not None:
            try:
                raw_receipt = self.raw_receiver.finish()
            except Exception as error:
                self.failure = self.failure or type(error).__name__ + ': ' + str(error)
                self.raw_receiver.close()
        self._close()
        if self.raw_receiver is not None:
            for name in self.sizes:
                path = self.output_dir / (name + '.raw')
                self.sizes[name] = path.stat().st_size
                with path.open('rb') as stream:
                    self.hashes[name] = hashlib.file_digest(stream, 'sha256')
        return {"schema": 1, "capture_complete": self.failure is None,
                "failure": self.failure, "reader_thread_absent": True,
                "remote_release_proven": False,
                "streams": {name: {"path": str(self.output_dir / (name + ".raw")),
                    "bytes": self.sizes[name], "sha256": self.hashes[name].hexdigest()}
                    for name in self.sizes},
                **({"raw_transport": raw_receipt, "wire_bytes": self.wire_sizes}
                   if self.raw_receiver is not None else {})}


def raw_transport_binding(binding):
    _binding(binding)
    need(binding.get('artifact_transport') == RAW_TRANSPORT, 'explicit raw transport binding')
    return binding


def _raw_open(root, name):
    """Open a fixed member through canonical, no-follow directory descriptors."""
    parts = name.split('/')
    directory = _directory(Path(root).joinpath(*parts[:-1]))
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(fd)
    if not (stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            info.st_size <= RAW_MEMBER_LIMITS[name]):
        os.close(fd)
        raise ValueError('raw export member type/size')
    return fd, info


def export_raw_frames(output, binding, sink):
    """Stream fixed binary members; missing or partial evidence stays explicit."""
    raw_transport_binding(binding)
    root = Path(output)
    # Never flatten this raw subtree through the legacy base64 native tree.
    for rank in (0, 1):
        directory = root / f'raw-rank-{rank}'
        if directory.exists():
            fd = _directory(directory)
            try:
                need(set(os.listdir(fd)) <= set(RAW_FILES), 'extra raw export member')
            finally:
                os.close(fd)
    for index, (name, limit) in enumerate(RAW_MEMBER_LIMITS.items()):
        try:
            fd, before = _raw_open(root, name)
        except FileNotFoundError:
            header = dict(schema=1, binding=binding, index=index, name=name, exists=False)
            sink.write(RAW_PREFIX + _canonical(header) + b'\n')
            continue
        with os.fdopen(fd, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            header = dict(schema=1, binding=binding, index=index, name=name,
                          exists=True, bytes=before.st_size, sha256=digest)
            sink.write(RAW_PREFIX + _canonical(header) + b'\n')
            emitted = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                block = stream.read(min(65536, remaining))
                need(block, 'short raw source')
                need(sink.write(block) == len(block), 'short raw export write')
                emitted.update(block)
                remaining -= len(block)
            need(emitted.hexdigest() == digest and _stamp(os.fstat(fd)) ==
                 _stamp(before) == _stamp((root / name).lstat()), 'raw source changed')
    sink.write(RAW_PREFIX + _canonical(dict(schema=1, binding=binding,
        index=len(RAW_MEMBER_LIMITS), name='END')) + b'\n')
    sink.flush()


class RawFrameReceiver:
    """Incremental stdout parser; retain binary payload exactly once."""
    def __init__(self, destination, binding, log):
        self.binding = dict(raw_transport_binding(binding))
        self.destination = Path(destination)
        parent = _directory(self.destination.parent)
        try:
            os.mkdir(self.destination.name, mode=0o700, dir_fd=parent)
        finally:
            os.close(parent)
        for rank in (0, 1):
            (self.destination / f'raw-rank-{rank}').mkdir(mode=0o700)
        self.log = log
        self.pending = bytearray()
        self.index = 0
        self.done = False
        self.members = {}
        self.payload = None
        self.remaining = 0
        self.log_bytes = 0
        self.names = list(RAW_MEMBER_LIMITS)

    def _line(self, line):
        if not line.startswith(RAW_PREFIX):
            need(self.index == 0 or self.done, 'log inserted inside binary frames')
            if self.done:
                need(line.startswith(b'SPRY_RESOURCE_V1 '), 'unexpected trailing raw transport data')
            need(len(line) + self.log_bytes <= RAW_LOG_LIMIT_BYTES, 'raw stdout log overflow')
            self.log.write(line)
            self.log_bytes += len(line)
            return
        need(not self.done and len(line) <= 4096, 'duplicate/oversized raw header')
        header = _json(line[len(RAW_PREFIX):])
        need(type(header) is dict and type(header.get('schema')) is int and header['schema'] == 1
             and header.get('binding') == self.binding and type(header.get('index')) is int
             and header['index'] == self.index, 'raw frame identity/order')
        if self.index == len(self.names):
            need(set(header) == {'schema', 'binding', 'index', 'name'} and header['name'] == 'END',
                 'raw terminal frame')
            self.done = True
            return
        name = self.names[self.index]
        need(header.get('name') == name and type(header.get('exists')) is bool,
             'raw frame name/presence')
        fields = {'schema', 'binding', 'index', 'name', 'exists'}
        if not header['exists']:
            need(set(header) == fields, 'missing raw member shape')
            self.members[name] = {'exists': False}
            self.index += 1
            return
        need(set(header) == fields | {'bytes', 'sha256'} and type(header['bytes']) is int
             and 0 <= header['bytes'] <= RAW_MEMBER_LIMITS[name] and _digest(header['sha256']),
             'raw frame size/hash')
        self.header = header
        self.remaining = header['bytes']
        self.payload = (self.destination / name).open('xb')
        self.digest = hashlib.sha256()
        if not self.remaining:
            self._complete()

    def _complete(self):
        self.payload.flush()
        os.fsync(self.payload.fileno())
        self.payload.close()
        self.payload = None
        need(self.digest.hexdigest() == self.header['sha256'], 'raw frame payload hash')
        self.members[self.header['name']] = {
            key: self.header[key] for key in ('exists', 'bytes', 'sha256')}
        self.index += 1

    def feed(self, data):
        need(type(data) is bytes and len(data) <= 65536, 'bounded raw input block')
        self.pending.extend(data)
        while self.pending:
            if self.payload is not None:
                count = min(len(self.pending), self.remaining)
                block = self.pending[:count]
                need(self.payload.write(block) == count, 'short retained raw write')
                self.digest.update(block)
                del self.pending[:count]
                self.remaining -= count
                if not self.remaining:
                    self._complete()
                continue
            end = self.pending.find(b'\n')
            if end < 0:
                need(len(self.pending) <= 65536, 'raw header/log line overflow')
                break
            line = bytes(self.pending[:end + 1])
            del self.pending[:end + 1]
            self._line(line)

    def finish(self):
        need(self.done and not self.pending and self.payload is None, 'incomplete raw frame stream')
        required = [name for name in self.names if not name.endswith('/INCOMPLETE.json')]
        complete = all(self.members[name]['exists'] for name in required)
        complete &= not any(self.members[name]['exists'] for name in self.names
                            if name.endswith('/INCOMPLETE.json'))
        return dict(schema=1, binding=self.binding, transport_valid=True,
                    all_members_present=complete, members=self.members, qualification=False,
                    retained_payload_bytes=sum(row.get('bytes', 0) for row in self.members.values()))

    def close(self):
        if self.payload is not None:
            self.payload.flush()
            self.payload.close()
            self.payload = None
