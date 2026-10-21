"""Verify a fixed user-scope resource envelope before the reviewed bwrap exec.

Projection is pure. The execution gate creates no units or cgroups and loads no
GPU libraries. Its real cgroup files, not requested systemd properties, decide
acceptance. Scratch-space quotas are outside this CPU/RAM/PID contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time


MEMORY_BYTES = 38 * 1024 ** 3
TASKS = 512
USER_MANAGER = "/user.slice/user-1000.slice/user@1000.service"
SCOPE = "benchmark-resource-gate-v1"
RESOURCE_PREFIX = "SPRY_RESOURCE_V1 "
FINAL_QUIESCENCE_SECONDS = 2.0
FINAL_SAMPLE_INTERVAL_SECONDS = 0.2
FINAL_SAMPLE_LIMIT = 12
FINAL_SAMPLE_BYTES = 4096
FIELDS = {"schema", "scope", "root", "session_sha256", "benchmark_spec_sha256",
          "gate_path", "gate_sha256", "entry_path", "entry_sha256", "bwrap_sha256",
          "units", "workload_argv", "workload_argv_sha256"}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def argv_sha256(argv):
    return hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def canonical_path(value):
    need(type(value) is str and value.startswith("/") and "\x00" not in value,
         "absolute canonical path required")
    path = Path(value)
    need(str(path) == value and all(part not in (".", "..") for part in value.split("/")),
         "noncanonical path refused")
    return path


def _directory(path):
    """Open every absolute component without following a symlink."""
    path = canonical_path(str(path))
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


def bounded_file(path, maximum=65536):
    path = canonical_path(str(path))
    directory = _directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "regular unlinked control/source file required")
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        need(len(raw) <= maximum, "bounded control/source file required")
        # Source files need the same open identity throughout hashing. Kernel
        # control files generally have size zero; their content is checked below.
        after = os.fstat(fd)
        stamp = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                              item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        need(stamp(info) == stamp(after) == stamp(os.stat(path.name, dir_fd=directory, follow_symlinks=False)),
             "file changed while read")
        return bytes(raw)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate manifest key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_manifest(value):
    cache = type(value) is dict and value.get('experiment_mode') in (
        'native-cache-regression-v1', 'native-cache-counterbalanced-v2', 'native-cache-phase-profile-v1')
    allowed = (FIELDS | {'artifacts_sha256', 'experiment_mode'},) if cache else (FIELDS, FIELDS | {"artifacts_sha256"})
    need(type(value) is dict and set(value) in allowed and type(value["schema"]) is int
         and value["schema"] == 1 and value["scope"] == SCOPE, "resource manifest schema")
    root = canonical_path(value["root"])
    need(root.parent == Path("/home/developer") and
         re.fullmatch(r"fp8-kv-benchmark\.[A-Za-z0-9]{8,64}", root.name),
         "fresh private benchmark root pattern")
    for key in ("session_sha256", "benchmark_spec_sha256", "gate_sha256", "entry_sha256", "bwrap_sha256"):
        need(digest(value[key]), "source/session digest required")
    entry_name = 'cache_gpu_entry.py' if cache else 'model_benchmark.py'
    need(value["gate_path"] == str(root / "review/benchmark_resource_gate.py") and
         value["entry_path"] == str(root / 'review' / entry_name), "exact private source paths")
    for key in ("units", "workload_argv", "workload_argv_sha256"):
        need(type(value[key]) is dict and set(value[key]) == {"bf16", "kv4"}, "exact two-arm mapping")
    for arm in ("bf16", "kv4"):
        expected_unit = "spry-fp8-benchmark-" + value["session_sha256"][:16] + "-" + arm + ".scope"
        need(value["units"][arm] == expected_unit, "session/arm/unit binding")
        argv = value["workload_argv"][arm]
        need(type(argv) is list and 20 <= len(argv) <= 256 and
             all(type(item) is str and item and len(item) <= 8192 and "\x00" not in item for item in argv),
             "bounded exact workload argv")
        need(argv[0] == "/usr/bin/bwrap" and digest(value["workload_argv_sha256"][arm]) and
             argv_sha256(argv) == value["workload_argv_sha256"][arm], "workload argv digest")
        mount = ["--ro-bind", value["entry_path"], "/mnt/review/" + entry_name]
        need(sum(argv[index:index + 3] == mount for index in range(len(argv) - 2)) == 1,
             "exact benchmark entry mount")
        if cache:
            need(digest(value['artifacts_sha256']), 'bounded capture source required')
            tail = ['/mnt/work/build-venv/bin/python3', '-I', '-B', '-S', '/mnt/review/cache_gpu_entry.py',
                    '--execute-native-cache', '--spec-sha256', value['benchmark_spec_sha256'], '--arm', arm]
            need(argv[-len(tail):] == tail and argv.count('--execute-native-cache') == 1
                 and '--share-net' not in argv and '--bind' not in argv,
                 'exact isolated native cache invocation')
            continue
        tail = ["/mnt/work/build-venv/bin/python3", "-I", "-B", "/mnt/review/model_benchmark.py",
                "--execute-reviewed-benchmark", "--spec", "/mnt/review/BENCHMARK-SPEC2.json",
                "--spec-sha256", value["benchmark_spec_sha256"], "--output", argv[-3],
                "--arm", arm]
        physical = ["--physical-artifacts", "--session-sha256", value["session_sha256"],
                    "--artifacts-sha256", value.get("artifacts_sha256")]
        if "artifacts_sha256" in value:
            need(digest(value["artifacts_sha256"]), "artifact source digest required")
            # The physical suffix is a fixed extension of the historical entry.
            body = argv[:-len(physical)]
            need(argv[-len(physical):] == physical, "physical artifact source/session tail")
            tail[-3] = body[-3]
        else:
            body = argv
        need(body[-3] in ("/mnt/output/benchmark", "/mnt/results/benchmark") and
             body[-len(tail):] == tail and argv.count("--execute-reviewed-benchmark") == 1,
             "exact benchmark invocation/arm")
    return value


def read_manifest(path, expected_sha256):
    need(digest(expected_sha256), "manifest digest required")
    raw = bounded_file(path)
    need(hashlib.sha256(raw).hexdigest() == expected_sha256, "resource manifest identity")
    return validate_manifest(_json(raw))


def scope_argv(manifest_path, manifest_sha256, arm, *, python="/usr/bin/python3"):
    """Project one unprivileged scope command; never start it here."""
    manifest = read_manifest(manifest_path, manifest_sha256)
    need(arm in ("bf16", "kv4") and python == "/usr/bin/python3", "fixed host interpreter/arm")
    return ("/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--no-ask-password",
            "--slice=app.slice", "--unit=" + manifest["units"][arm],
            "--property=CPUQuota=2400%", "--property=MemoryMax=" + str(MEMORY_BYTES),
            "--property=MemorySwapMax=0", "--property=TasksMax=512", "--", python, "-I", "-B",
            manifest["gate_path"], "--execute-resource-gated-workload", "--manifest", str(manifest_path),
            "--manifest-sha256", manifest_sha256, "--arm", arm)


def _integer(raw, name):
    text = raw.decode("ascii").strip()
    need(re.fullmatch(r"[0-9]+", text) is not None, "finite integer required: " + name)
    return int(text)


def _events(raw, required):
    result = {}
    for line in raw.decode("ascii").splitlines():
        parts = line.split()
        need(len(parts) == 2 and parts[0] not in result, "malformed resource events")
        result[parts[0]] = _integer(parts[1].encode(), parts[0])
    need(set(required) <= set(result), "missing resource event counter")
    return result


def verify_cgroup(manifest, arm, *, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Read actual own cgroup2 bounds; fixture roots are CPU-only input substitutes."""
    expected = USER_MANAGER + "/app.slice/" + manifest["units"][arm]
    # /proc/self is the one intentional kernel process symlink. Its children are
    # read without following a final symlink; cgroup paths use nofollow throughout.
    proc_self = Path(proc_root) / "self"
    if str(proc_root) == "/proc":
        proc_self = Path("/proc") / str(os.getpid())
    need(bounded_file(proc_self / "cgroup", 4096) == ("0::" + expected + "\n").encode(),
         "own cgroup does not match frozen user scope")
    mounts = bounded_file(proc_self / "mountinfo", 1 << 20).decode("ascii")
    matches = []
    for line in mounts.splitlines():
        left, separator, right = line.partition(" - ")
        fields, filesystem = left.split(), right.split()
        if separator and len(fields) >= 6 and fields[4] == "/sys/fs/cgroup":
            matches.append(filesystem[0] if filesystem else "")
    need(matches == ["cgroup2"], "exact cgroup2 mount required")
    directory = Path(cgroup_root) / expected.lstrip("/")
    directory_fd = _directory(directory)
    try:
        info = os.fstat(directory_fd)
        cgroup_identity = {"device": info.st_dev, "inode": info.st_ino}
    finally:
        os.close(directory_fd)
    for name in ("cgroup.controllers", "cgroup.subtree_control"):
        controllers = bounded_file(directory.parent / name, 4096).decode("ascii").split()
        need({"cpu", "memory", "pids"} <= set(controllers), "required controller unavailable: " + name)
    cpu = bounded_file(directory / "cpu.max", 128).decode("ascii").split()
    need(len(cpu) == 2, "cpu.max shape")
    quota, period = (_integer(item.encode(), "cpu.max") for item in cpu)
    need(1000 <= period <= 1000000 and quota == 24 * period, "actual CPU quota differs from 24 CPUs")
    limits = {name: _integer(bounded_file(directory / name, 128), name)
              for name in ("memory.max", "memory.swap.max", "pids.max")}
    need(limits == {"memory.max": MEMORY_BYTES, "memory.swap.max": 0, "pids.max": TASKS},
         "actual memory/swap/PID limits differ")
    memory = _events(bounded_file(directory / "memory.events", 4096), {"max", "oom", "oom_kill"})
    pids = _events(bounded_file(directory / "pids.events", 4096), {"max"})
    need(all(memory.get(name, 0) == 0 for name in ("max", "oom", "oom_kill", "oom_group_kill"))
         and pids["max"] == 0, "scope already reached a resource/OOM boundary")
    current = {name: _integer(bounded_file(directory / name, 128), name)
               for name in ("memory.current", "pids.current")}
    need(0 < current["memory.current"] < MEMORY_BYTES and 0 < current["pids.current"] <= TASKS,
         "current scoped resource usage invalid")
    return {"cgroup_path": expected, "cpu_max": [quota, period], "limits": limits,
            "memory_events": memory, "pids_events": pids, "initial_usage": current,
            "cgroup_identity": cgroup_identity}


def process_identity(*, proc_root=Path("/proc")):
    path = Path(proc_root) / (str(os.getpid()) if str(proc_root) == "/proc" else "self") / "stat"
    text = bounded_file(path, 8192).decode("ascii")
    before, separator, after = text.rpartition(") ")
    fields = after.split()
    need(separator and len(fields) >= 20 and before.split(" (", 1)[0].isdigit(), "process stat identity")
    identity = {"pid": int(before.split(" (", 1)[0]), "start_ticks": _integer(fields[19].encode(), "start_ticks")}
    need(identity["pid"] == os.getpid() and identity["start_ticks"] > 0, "own process start identity")
    return identity


def _final_cgroup_snapshot(initial, *, child_exit, child_reaped, proc_root=Path("/proc"),
                           cgroup_root=Path("/sys/fs/cgroup"), observation=None):
    """Read one bounded observation; membership may still be settling."""
    observed = {} if observation is None else observation
    need(type(child_exit) is int and child_reaped is True, "workload child must be reaped")
    identity = process_identity(proc_root=proc_root)
    observed["gate_process"] = identity
    need(initial.get("gate_process") == identity, "resource wrapper process changed")
    proc_self = Path(proc_root) / (str(os.getpid()) if str(proc_root) == "/proc" else "self")
    need(bounded_file(proc_self / "cgroup", 4096) == ("0::" + initial["cgroup_path"] + "\n").encode(),
         "resource wrapper cgroup membership changed")
    directory = Path(cgroup_root) / initial["cgroup_path"].lstrip("/")
    descriptor = _directory(directory)
    try:
        info = os.fstat(descriptor)
        observed["cgroup_identity"] = {"device": info.st_dev, "inode": info.st_ino}
        need(initial["cgroup_identity"] == {"device": info.st_dev, "inode": info.st_ino},
             "final cgroup identity changed")
        memory = _events(bounded_file(directory / "memory.events", 4096), {"max", "oom", "oom_kill"})
        observed["memory_events"] = memory
        pids = _events(bounded_file(directory / "pids.events", 4096), {"max"})
        observed["pids_events"] = pids
        events = _events(bounded_file(directory / "cgroup.events", 4096), {"populated", "frozen"})
        observed["cgroup_events"] = events
        usage = observed["usage"] = {}
        for name in ("memory.current", "memory.peak", "pids.current"):
            usage[name] = _integer(bounded_file(directory / name, 128), name)
        processes = bounded_file(directory / "cgroup.procs", 65536).decode("ascii").split()
        observed["processes"] = processes
        observed["threads"] = bounded_file(directory / "cgroup.threads", 65536).decode("ascii").split()
        need(0 < usage["pids.current"] <= TASKS, "final scoped task count invalid")
        for members in (processes, observed["threads"]):
            need(0 < len(members) <= TASKS and len(set(members)) == len(members)
                 and all(re.fullmatch(r"[1-9][0-9]{0,9}", member) and int(member) <= 2147483647
                         for member in members), "final scoped membership malformed")
        need(events["populated"] == 1 and events["frozen"] == 0, "scope wrapper population/freeze differs")
        need(all(memory.get(name, 0) == 0 for name in ("max", "oom", "oom_kill", "oom_group_kill"))
             and pids["max"] == 0, "scope reached a resource/OOM boundary")
        need(0 < usage["memory.current"] <= usage["memory.peak"] <= MEMORY_BYTES,
             "final memory usage/peak invalid")
        observed["limits"] = {name: _integer(bounded_file(directory / name, 128), name)
                              for name in ("memory.max", "memory.swap.max", "pids.max")}
        need(observed["limits"] == initial["limits"],
             "final resource limits changed")
        cpu = bounded_file(directory / "cpu.max", 128).decode("ascii").split()
        observed["cpu_max"] = [_integer(item.encode(), "cpu.max") for item in cpu]
        need(len(cpu) == 2 and observed["cpu_max"] == initial["cpu_max"],
             "final CPU quota changed")
        after = os.stat(directory, follow_symlinks=False)
        need(stat.S_ISDIR(after.st_mode) and initial["cgroup_identity"] ==
             {"device": after.st_dev, "inode": after.st_ino}, "cgroup changed during final observation")
        return {"child_exit": child_exit, "child_reaped": True, "gate_process": identity,
                "cgroup_identity": initial["cgroup_identity"], "memory_events": memory,
                "pids_events": pids, "cgroup_events": events, "final_usage": usage,
                "only_wrapper_remains": (processes == [str(identity["pid"])]
                    and observed["threads"] == [str(identity["pid"])] and usage["pids.current"] == 1),
                "cgroup_release_proven": False,
                "resource_boundaries_clear": True}
    finally:
        os.close(descriptor)


def final_cgroup_observation(initial, *, child_exit, child_reaped, proc_root=Path("/proc"),
                             cgroup_root=Path("/sys/fs/cgroup"), observation=None,
                             clock=time.monotonic, sleeper=time.sleep):
    """Require two consecutive wrapper-only observations within at most 2 s.

    This finite wait runs inside the existing outer owner deadline. It never
    extends that deadline or suppresses an owner signal. Missing/changed control
    files and resource events refuse immediately. Independent release observation
    remains necessary after the wrapper exits. Clock/sleep substitutions are CPU
    fixture inputs and are not exposed by the live CLI.
    """
    observed = {} if observation is None else observation
    started = clock()
    deadline = started + FINAL_QUIESCENCE_SECONDS
    samples = []
    quiescence = {"schema": 1, "started_monotonic": started, "deadline_monotonic": deadline,
                  "maximum_seconds": FINAL_QUIESCENCE_SECONDS,
                  "sample_interval_seconds": FINAL_SAMPLE_INTERVAL_SECONDS,
                  "sample_limit": FINAL_SAMPLE_LIMIT, "sample_bytes_limit": FINAL_SAMPLE_BYTES,
                  "required_consecutive": 2, "consecutive": 0, "complete": False}
    observed.update(quiescence=quiescence, snapshots=samples)
    for index in range(FINAL_SAMPLE_LIMIT):
        need(clock() < deadline, "scoped process/thread survives workload child or quiescence deadline expired")
        sample = {"ordinal": index, "started_monotonic": clock(), "complete": False}
        result, failure = None, None
        try:
            result = _final_cgroup_snapshot(initial, child_exit=child_exit, child_reaped=child_reaped,
                proc_root=proc_root, cgroup_root=cgroup_root, observation=sample)
            sample["only_wrapper_remains"] = result["only_wrapper_remains"]
            sample["complete"] = True
        except BaseException as error:
            failure = error
            sample["failure"] = type(error).__name__ + ": " + str(error)[:512]
        sample["finished_monotonic"] = clock()
        raw = json.dumps(sample, sort_keys=True).encode()
        if len(raw) > FINAL_SAMPLE_BYTES:
            # A malformed or unexpectedly large observation cannot consume the
            # fixed receipt budget. Keep its size/hash and refuse immediately.
            sample = {"ordinal": index, "started_monotonic": sample["started_monotonic"],
                      "finished_monotonic": sample["finished_monotonic"], "complete": False,
                      "observed_bytes": len(raw), "observed_sha256": hashlib.sha256(raw).hexdigest(),
                      "failure": "final scoped observation exceeds retained snapshot bound"}
            failure = ValueError(sample["failure"])
        samples.append(sample)
        # Preserve the prior partial-observation fields for existing consumers,
        # with only one copy of the ordered snapshot array.
        observed.clear()
        observed.update(quiescence=quiescence, snapshots=samples)
        observed.update({key: value for key, value in sample.items() if key not in
                         ("ordinal", "started_monotonic", "finished_monotonic", "complete")})
        quiescence["finished_monotonic"] = sample["finished_monotonic"]
        if failure is not None:
            raise failure
        need(sample["finished_monotonic"] < deadline, "final cgroup observation exceeded quiescence deadline")
        quiescence["consecutive"] = quiescence["consecutive"] + 1 if result["only_wrapper_remains"] else 0
        if quiescence["consecutive"] == 2:
            quiescence["complete"] = True
            return result
        remaining = deadline - clock()
        need(remaining > FINAL_SAMPLE_INTERVAL_SECONDS,
             "scoped process/thread survives workload child or quiescence deadline expired")
        sleeper(FINAL_SAMPLE_INTERVAL_SECONDS)
    raise ValueError("scoped process/thread survives workload child or quiescence sample limit exhausted")


def supervise_workload(argv, *, popen=subprocess.Popen, pidfd_open=None, send_signal=None):
    """Wait/reap one exact workload, forwarding owner signals only through pidfd.

    The existing outer finite process owner retains all deadlines, descendant
    authentication and escalation. This wrapper adds no competing timeout tree.
    """
    opener = os.pidfd_open if pidfd_open is None else pidfd_open
    sender = signal.pidfd_send_signal if send_signal is None else send_signal
    child, descriptor, previous, received = None, None, {}, []
    def forward(number, _frame):
        received.append(number)
        if descriptor is not None:
            sender(descriptor, number, None, 0)
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, forward)
        child = popen(argv, stdin=subprocess.DEVNULL, close_fds=True)
        descriptor = opener(child.pid, 0)
        for number in received:
            sender(descriptor, number, None, 0)
        code = child.wait()
        return {"child_pid": child.pid, "child_exit": code, "child_reaped": True,
                "owner_signals": received}
    finally:
        if child is not None and child.returncode is None:
            # The direct child's pidfd is the only signaling authority here.
            if descriptor is not None:
                try:
                    sender(descriptor, signal.SIGKILL, None, 0)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # Outer authenticated owner must classify surviving work.
        if descriptor is not None:
            os.close(descriptor)
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_gate(manifest, arm, *, gate_path, source_reader=bounded_file, proc_root=Path("/proc"),
             cgroup_root=Path("/sys/fs/cgroup"), credentials=None, execute=os.execv, set_limit=None,
             emit=lambda record: print(json.dumps(record, sort_keys=True), flush=True), supervisor=None,
             observation_clock=time.monotonic, observation_sleeper=time.sleep):
    """Compose all identity/resource refusals before the sole exact workload exec.

    CPU fixtures explicitly substitute source reads, credentials, proc/cgroup
    roots and exec. The live CLI exposes none of those substitutions.
    """
    validate_manifest(manifest)
    need(arm in ("bf16", "kv4"), "fixed arm required")
    identity = credentials if credentials is not None else (os.getuid(), os.geteuid(), os.getgid(), os.getegid())
    need(identity == (1000, 1000, 1000, 1000), "non-root target UID/GID1000 required")
    need(str(gate_path) == manifest["gate_path"], "executed gate source path differs")
    for path, pin in ((manifest["gate_path"], manifest["gate_sha256"]),
                      (manifest["entry_path"], manifest["entry_sha256"]),
                      ("/usr/bin/bwrap", manifest["bwrap_sha256"])):
        need(hashlib.sha256(source_reader(path, 8 << 20)).hexdigest() == pin, "bound source identity differs")
    observed = verify_cgroup(manifest, arm, proc_root=proc_root, cgroup_root=cgroup_root)
    # Preserve the existing rank FSIZE hard limit: RCCL writes a 41,288,068-byte
    # shared-memory file. A lower inherited hard limit would break native setup.
    setter = resource.setrlimit if set_limit is None else set_limit
    setter(resource.RLIMIT_CORE, (0, 0))
    setter(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))
    receipt = {"schema": 1, "scope": SCOPE, "arm": arm, "root": manifest["root"],
               "session_sha256": manifest["session_sha256"], "gate_sha256": manifest["gate_sha256"],
               "entry_sha256": manifest["entry_sha256"], "bwrap_sha256": manifest["bwrap_sha256"],
               "workload_argv_sha256": manifest["workload_argv_sha256"][arm],
               "benchmark_spec_sha256": manifest["benchmark_spec_sha256"],
               "actual_cgroup_verified": True, "scratch_quota_verified": False,
               "rlimits": {"core": [0, 0], "file_size": [128 << 20, 128 << 20]}, **observed}
    argv = manifest["workload_argv"][arm]
    if "artifacts_sha256" in manifest:
        need(supervisor is not None, "physical final-observation supervisor required")
        receipt["gate_process"] = process_identity(proc_root=proc_root)
        receipt["phase"] = "initial"
        emit(receipt)
        final = {"schema": 1, "scope": SCOPE, "phase": "final", "arm": arm,
                 "root": manifest["root"], "session_sha256": manifest["session_sha256"],
                 "benchmark_spec_sha256": manifest["benchmark_spec_sha256"],
                 "gate_sha256": manifest["gate_sha256"], "entry_sha256": manifest["entry_sha256"],
                 "bwrap_sha256": manifest["bwrap_sha256"],
                 "workload_argv_sha256": manifest["workload_argv_sha256"][arm],
                 "complete": False, "qualification": False}
        try:
            child = supervisor(argv)
            final.update(child)
            final["cgroup_observation"] = {}
            final.update(final_cgroup_observation(receipt, child_exit=child["child_exit"],
                child_reaped=child["child_reaped"], proc_root=proc_root, cgroup_root=cgroup_root,
                observation=final["cgroup_observation"], clock=observation_clock, sleeper=observation_sleeper))
            final["complete"] = True
        except BaseException as error:
            final["failure"] = type(error).__name__ + ": " + str(error)[:512]
        finally:
            emit(final)
        need(final["complete"], "final scoped resource observation incomplete")
        return final
    emit(receipt)
    execute(argv[0], argv)
    return receipt  # CPU exec spies only; successful live exec cannot return.


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-resource-gated-workload", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--arm", choices=("bf16", "kv4"))
    args = parser.parse_args(argv)
    need(args.execute_resource_gated_workload and args.manifest and args.manifest_sha256 and args.arm,
         "NOT_RUN: explicit resource-gated execution and manifest required")
    manifest = read_manifest(args.manifest, args.manifest_sha256)
    need(Path(args.manifest).parent == Path(manifest["root"]) / "review", "private manifest location")
    physical = "artifacts_sha256" in manifest
    result = run_gate(manifest, args.arm, gate_path=Path(__file__),
        supervisor=supervise_workload if physical else None,
        emit=lambda record: print((RESOURCE_PREFIX if physical else "") + json.dumps(record, sort_keys=True), flush=True))
    if physical:
        return result["child_exit"] if result["child_exit"] >= 0 else 128 - result["child_exit"]


if __name__ == "__main__":
    sys.exit(main())
