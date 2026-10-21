"""Bounded raw benchmark export and receipt; no runtime or archive extraction.

The exporter preserves malformed/partial payload bytes. Transport validity is
separate from numerical acceptance. The receiver cannot authorize a workload.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import threading


PREFIX = b"SPRY_BENCHMARK_BUNDLE_V2 "
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
    need(type(value) is dict and set(value) == {"session_sha256", "arm", "spec_sha256"},
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


def stage_artifacts(output, binding):
    """Stage bounded raw evidence after worker reaping; never parse partial results.

    Full rank logs are streamed for their hashes, then only their bounded binary
    tails are retained. Missing evidence stays explicit. Source files are never
    changed, and an existing export directory is never reused.
    """
    _binding(binding)
    output = Path(output)
    source = _directory(output)
    destination = None
    try:
        os.mkdir("export", mode=0o700, dir_fd=source)
        destination = os.open("export", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=source)
        before, names = os.fstat(source), set(os.listdir(source))
        for name, limit in MEMBER_LIMITS.items():
            if name == "diagnostics.json" or name.endswith(".log.tail"):
                continue
            raw = _read_member(source, name, limit)
            if raw is not None:
                _write_member(destination, name, raw)
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


def export_bundle(output, binding, sink):
    """Export the eleven exact V2 members; missing files remain explicit."""
    _binding(binding)
    directory = _directory(output)
    try:
        before = os.fstat(directory)
        names = set(os.listdir(directory))
        need(names <= set(MEMBER_LIMITS), "extra export member or directory")
        members = {name: _member(directory, name, limit) for name, limit in MEMBER_LIMITS.items()}
        need(set(os.listdir(directory)) == names and _stamp(before) == _stamp(os.fstat(directory)),
             "export directory changed")
        value = {"schema": 2, "binding": dict(binding), "members": members}
        line = PREFIX + _canonical(value) + b"\n"
        need(len(line) <= MAX_ENCODED_BYTES <= BUNDLE_LIMIT, "encoded bundle overflow")
        sink.write(line)
        sink.flush()
        return value
    finally:
        os.close(directory)


def receive_bundle(stdout, stderr, destination, binding):
    """Validate one raw transport record, then exclusively retain original bytes."""
    _binding(binding)
    need(type(stdout) is bytes and type(stderr) is bytes and
         len(stdout) <= STREAM_LIMIT and len(stderr) <= STREAM_LIMIT, "stdio capture overflow")
    split = stdout.splitlines(keepends=True)
    lines = [line for line in split if line.startswith(b"SPRY_BENCHMARK_BUNDLE_")]
    need(len(lines) == 1 and lines[0].startswith(PREFIX) and len(lines[0]) <= BUNDLE_LIMIT,
         "exactly one bounded V2 artifact bundle")
    need(lines[0].endswith(b"\n"), "incomplete artifact record line")
    value = _json(lines[0][len(PREFIX):-1])
    need(type(value) is dict and set(value) == {"schema", "binding", "members"} and
         type(value["schema"]) is int and value["schema"] == 2 and value["binding"] == binding,
         "artifact schema/binding mismatch")
    _binding(value["binding"])
    need(type(value["members"]) is dict and set(value["members"]) == set(MEMBER_LIMITS),
         "exact artifact members required")
    decoded = {}
    for name, limit in MEMBER_LIMITS.items():
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
                "all_members_present": len(decoded) == len(MEMBER_LIMITS),
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
    def __init__(self, output_dir, *, stream_limit=None):
        self.stream_limit = STREAM_LIMIT if stream_limit is None else stream_limit
        need(type(self.stream_limit) is int and 0 < self.stream_limit <= STREAM_LIMIT,
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
        try:
            for name in self.sizes:
                self.outputs[name] = os.fdopen(os.open(name + ".raw", os.O_WRONLY | os.O_CREAT |
                    os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory), "wb")
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
        self._close()
        return {"schema": 1, "capture_complete": self.failure is None,
                "failure": self.failure, "reader_thread_absent": True,
                "remote_release_proven": False,
                "streams": {name: {"path": str(self.output_dir / (name + ".raw")),
                    "bytes": self.sizes[name], "sha256": self.hashes[name].hexdigest()}
                    for name in self.sizes}}
