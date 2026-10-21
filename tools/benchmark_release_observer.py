"""Future owner-run read-only host benchmark observer; import performs no queries.

Requires an authenticated native SSH host and isolated root process. It never
acquires a lock, changes an ownership journal, grants a launch or qualifies a model.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time


COLLECTOR_SHA256 = "c33222c324bf841b32d71544450bbf8b418954d373b005f1d032c860f065b4c5"
LOCK = Path("/home/developer/model-serving.izR2m1O9/diagnostic.lock")
PROCESS_FIELDS = {"boot_id", "pid", "ppid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace", "command_sha256", "state"}
REQUEST_FIELDS = {"schema", "mode", "root", "session_sha256", "boot_id", "lock_device", "lock_inode",
    "arm", "root_process", "observed_processes", "journal_sha256", "issued_monotonic", "deadline_monotonic", "native_ssh_host_contract"}


class ObservationError(RuntimeError):
    pass


def digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require(value, reason):
    if not value:
        raise ObservationError(reason)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _pairs(rows):
    value = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def json_bytes(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ObservationError("nonfinite JSON")))


def read_regular(path, maximum, uid, gid=None, mode=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == uid
            and (gid is None or before.st_gid == gid) and (mode is None or stat.S_IMODE(before.st_mode) == mode)
            and before.st_size <= maximum, "unsafe bounded evidence file")
        raw = os.read(fd, maximum + 1)
        after, named = os.fstat(fd), Path(path).lstat()
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == before.st_size and len(raw) <= maximum and all(getattr(before, key) == getattr(after, key) == getattr(named, key) for key in fields), "evidence file changed")
        return raw
    finally:
        os.close(fd)


def source(path, expected):
    require(digest(expected), "independently supplied source digest required")
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "canonical source path required")
    for parent in path.parents:
        info = parent.stat()
        require(info.st_uid == 0 and not info.st_mode & 0o022 or parent == Path("/var/tmp") and info.st_uid == 0 and info.st_mode & stat.S_ISVTX, "source parent is user controlled")
    raw = read_regular(path, 256 * 1024, 0)
    require(not path.stat().st_mode & 0o022 and hashlib.sha256(raw).hexdigest() == expected, "source hash/mode mismatch")
    return raw


def load_collector():
    path = Path(__file__).resolve().with_name("collect_benchmark_platform.py")
    source(path, COLLECTOR_SHA256)
    spec = importlib.util.spec_from_file_location("spry_pinned_platform_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_process(row, boot):
    require(type(row) is dict and set(row) == PROCESS_FIELDS and row["boot_id"] == boot, "process shape/boot mismatch")
    require(all(type(row[key]) is int and row[key] > 0 for key in ("pid", "pgid", "sid", "start_ticks"))
        and type(row["ppid"]) is int and row["ppid"] >= 0 and type(row["uid"]) is type(row["gid"]) is int
        and row["uid"] == row["gid"] == 1000 and digest(row["command_sha256"])
        and type(row["pid_namespace"]) is str and re.fullmatch(r"pid:\[[0-9]+\]", row["pid_namespace"])
        and type(row["state"]) is str and re.fullmatch(r"[A-Za-z]", row["state"]), "invalid process identity")


def validate_request(request, deadline):
    require(type(request) is dict and set(request) == REQUEST_FIELDS and type(request["schema"]) is int
        and request["schema"] == 1 and request["mode"] in ("admission", "locked-admission", "release", "independent")
        and request["native_ssh_host_contract"] is True, "request contract mismatch")
    require(type(request["root"]) is str and re.fullmatch(r"/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}", request["root"])
        and digest(request["session_sha256"]) and type(request["boot_id"]) is str
        and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", request["boot_id"])
        and all(type(request[key]) is int and request[key] > 0 for key in ("lock_device", "lock_inode")), "request root/identity mismatch")
    now, issued, end = time.monotonic(), request["issued_monotonic"], request["deadline_monotonic"]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (issued, end, deadline))
        and issued <= now < end <= deadline and end - issued <= 75 and now < end - 15, "request deadline/freshness mismatch")
    rows = request["observed_processes"]
    require(type(rows) is list and len(rows) <= 512, "process inventory bound")
    for row in rows:
        validate_process(row, request["boot_id"])
    require(len({row["pid"] for row in rows}) == len(rows), "duplicate or substituted process PID")
    if request["mode"] in ("admission", "locked-admission"):
        require(not rows and request["root_process"] is None and request["arm"] is None
            and (request["journal_sha256"] is None if request["mode"] == "admission" else digest(request["journal_sha256"])), "admission cannot assert release")
    else:
        validate_process(request["root_process"], request["boot_id"])
        require(request["root_process"] in rows and len(rows) >= 2 and digest(request["journal_sha256"])
            and (request["arm"] in ("bf16", "kv4") if request["mode"] == "release" else request["arm"] is None), "release identity inventory mismatch")
        require(any(row["pid_namespace"] != request["root_process"]["pid_namespace"] for row in rows), "private process namespace was never observed")
    return end - 15


def process_metadata(pid, proc_root="/proc"):
    """Named process identity only; never inspect its command or environment."""
    folder = Path(proc_root) / str(pid)
    try:
        raw = proc_read(folder / "stat")
        fields = raw[raw.rfind(b")") + 2:].split()
        require(len(raw) <= 65536 and int(raw.split(b"(", 1)[0]) == pid, "malformed process stat")
        status = proc_read(folder / "status")
        require(len(status) <= 65536, "process status bound")
        rows = dict(line.split(b":", 1) for line in status.splitlines() if b":" in line)
        uid, gid = [[int(value) for value in rows[key].split()] for key in (b"Uid", b"Gid")]
        require(len(uid) == len(gid) == 4 and len(set(uid)) == len(set(gid)) == 1, "process credential transition")
        value = {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]), "sid": int(fields[3]),
            "start_ticks": int(fields[19]), "uid": uid[0], "gid": gid[0], "state": fields[0].decode("ascii"),
            "pid_namespace": os.readlink(folder / "ns/pid")}
        again = proc_read(folder / "stat")
        require(int(again[again.rfind(b")") + 2:].split()[19]) == value["start_ticks"], "process reused during observation")
        return value
    except FileNotFoundError:
        return None


def proc_read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, 65537)
        require(len(raw) <= 65536, "bounded proc metadata exceeded")
        return raw
    finally:
        os.close(fd)


def same_process(actual, expected):
    return actual is not None and all(actual[key] == expected[key] for key in ("pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace"))


def check_journal(request, local):
    root = Path(request["root"])
    require(root.resolve() == root, "private root is noncanonical")
    info = root.lstat()
    require(stat.S_ISDIR(info.st_mode) and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (1000, 1000, 0o700), "private root ownership mismatch")
    lock = local["canonical_lock"]
    require(lock["device"] == request["lock_device"] and lock["inode"] == request["lock_inode"]
        and (lock["uid"], lock["gid"], lock["mode"], lock["links"]) == (1000, 1000, 0o600, 1), "canonical lock binding mismatch")
    path = Path(str(LOCK) + ".unresolved.json")
    if request["mode"] == "admission":
        require(not os.path.lexists(path), "admission refuses any unresolved record")
        return {"present": False, "lock_acquired": False}
    raw = read_regular(path, 256 * 1024, 1000, 1000, 0o600)
    require(hashlib.sha256(raw).hexdigest() == request["journal_sha256"], "unresolved source hash mismatch")
    value = json_bytes(raw)
    expected = {"version": 2, "kind": "host-process-benchmark", "root": request["root"], "session_sha256": request["session_sha256"],
        "boot_id": request["boot_id"], "uid": 1000, "gid": 1000, "lock_device": request["lock_device"], "lock_inode": request["lock_inode"]}
    require(all(type(value.get(key)) is type(item) and value.get(key) == item for key, item in expected.items())
        and type(value.get("token")) is str and re.fullmatch(r"[0-9a-f]{32}", value["token"]), "foreign unresolved journal")
    owner = value.get("owner")
    validate_process(owner, request["boot_id"])
    current_owner = process_metadata(owner["pid"])
    require(same_process(current_owner, owner) and current_owner["state"] not in ("Z", "X", "x")
        and value.get("owner_pid") == owner["pid"], "journal owner is stale or unavailable")
    if request["mode"] == "locked-admission":
        require(value.get("state") == "reserved" and value.get("arms") == [], "locked admission requires own untouched reservation")
        return {"present": True, "sha256": request["journal_sha256"], "state": "reserved", "authenticated": True, "lock_acquired": False}
    state = "running" if request["mode"] == "release" else "pending-independent"
    arms = value.get("arms")
    require(value.get("state") == state and type(arms) is list and 1 <= len(arms) <= 2
        and [row.get("arm") for row in arms] == ["bf16", "kv4"][:len(arms)]
        and arms[-1].get("child") == request["root_process"]
        and (request["mode"] != "release" or arms[-1]["arm"] == request["arm"]), "journal state/child/arm mismatch")
    return {"present": True, "sha256": request["journal_sha256"], "state": state, "authenticated": True, "lock_acquired": False}


def process_release(request, deadline, proc_root="/proc"):
    rows = request["observed_processes"]
    namespaces = sorted({row["pid_namespace"] for row in rows if row["pid_namespace"] != request["root_process"]["pid_namespace"]}) if rows else []
    present = []
    for row in rows:
        require(time.monotonic() < deadline, "named process observation deadline")
        actual = process_metadata(row["pid"], proc_root)
        require(actual is None or same_process(actual, row), "observed PID was reused/substituted")
        if actual is not None:
            present.append([row["boot_id"], row["pid"], row["start_ticks"]])
    members = {namespace: [] for namespace in namespaces}
    if not namespaces:
        return {"checked_processes": [], "authenticated_processes_present": [], "pid_namespace_members": {}}
    for count, entry in enumerate(Path(proc_root).iterdir()):
        require(count < 65536 and time.monotonic() < deadline, "namespace inventory bound/deadline")
        if entry.name.isdecimal():
            try:
                namespace = os.readlink(entry / "ns/pid")
            except FileNotFoundError:
                continue
            if namespace in members:
                require(len(members[namespace]) < 512, "namespace member bound")
                members[namespace].append(int(entry.name))
    return {"checked_processes": [list(row) for row in sorted((value["boot_id"], value["pid"], value["start_ticks"]) for value in rows)],
        "authenticated_processes_present": sorted(present), "pid_namespace_members": {key: sorted(value) for key, value in members.items()}}


def collect_snapshot(request, *, expected_observer_sha256, deadline_monotonic):
    work_end = validate_request(request, deadline_monotonic)
    source(Path(__file__).resolve(), expected_observer_sha256)
    require(os.geteuid() == 0 and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "isolated root observer required")
    collector = load_collector()
    snapshot = {"schema": 1, "scope": "host-benchmark-" + {"admission": "admission", "locked-admission": "locked-admission", "release": "release", "independent": "independent-release"}[request["mode"]] + "-v1",
        "root": request["root"], "session_sha256": request["session_sha256"], "boot_id": request["boot_id"],
        "lock_device": request["lock_device"], "lock_inode": request["lock_inode"],
        "collector_source_sha256": expected_observer_sha256, "metadata_source_sha256": COLLECTOR_SHA256,
        "request_canonical_sha256": hashlib.sha256(canonical_json(request)).hexdigest(),
        "qualification": False, "launch_authorized": False, "status": "INCOMPLETE", "failures": [], "commands": [],
        "native_ssh_host_contract_required": True, "fresh_native_uuid_observed": False}
    previous, timer = {}, False
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "existing timer refuses observer")
        def interrupt(number, frame):
            raise ObservationError("observer signal/deadline")
        for number in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupt)
        signal.setitimer(signal.ITIMER_REAL, max(.001, work_end - time.monotonic()))
        timer = True
        snapshot["root_fuser_prerequisites"] = collector.root_fuser_prerequisites()
        require(snapshot["root_fuser_prerequisites"]["passed"] is True, "root visibility prerequisites failed")
        local = collector.local_metadata()
        require(local["hostname"] == collector.TARGET and local["boot_id"] == request["boot_id"] and not local["swap_active"]
            and (local["developer_uid"], local["developer_gid"]) == (1000, 1000)
            and local["available_memory_bytes"] >= 46 * 1024**3 and local["free_disk_bytes"] >= 64 * 1024**3, "target/boot/headroom mismatch")
        snapshot["journal"] = check_journal(request, local)
        snapshot["kubeconfig_metadata"] = collector.kubeconfig_metadata()
        facts = {}
        for name in collector.COMMANDS:
            require(time.monotonic() + 14 <= work_end, "complete observation command envelope does not fit")
            receipt, raw = collector.run_command(name, work_end)
            snapshot["commands"].append(receipt)
            require(receipt["failure"] is None and receipt["reaped"] is True and receipt["group_absent"] is True
                and receipt["exit_code"] == (1 if name == "holders" else 0)
                and not raw["stderr"].strip(), "platform command failed or release uncertain")
            facts[name] = collector.sanitize(name, raw)
        node, policy = facts["node"], facts["policy"]
        require(node["name_matches"] and node["ready"] and node["pressure_free"] and node["kernel_release"] == local["kernel_release"]
            and node["boot_id"] == request["boot_id"] and node["allocatable_gpu"] == "2" and policy["matches_retained_deny_contract"], "node/policy state mismatch")
        require(facts["holders"]["pids"] == [] and facts["pods"]["conservative_gpu_requests"] == 0, "devices or allocations are busy")
        devices = local["devices"]
        require(len(devices) == 2 and {row["bdf"] for row in devices} == set(collector.DEVICES)
            and {row["render"] for row in devices} == {"/dev/dri/renderD128", "/dev/dri/renderD129"}
            and all(row["retained_sysfs_binding_matches"] is True and row["retained_hip_uuid"] == collector.DEVICES[row["bdf"]] for row in devices), "current sysfs/retained HIP binding mismatch")
        snapshot.update(process_release(request, work_end))
        require(not snapshot["authenticated_processes_present"] and not any(snapshot["pid_namespace_members"].values()), "owned processes or namespaces survive")
        check_journal(request, collector.local_metadata())
        require(collector.boot_id() == request["boot_id"] and time.monotonic() < work_end, "end boot/deadline mismatch")
        snapshot.update({"holders": {"visibility": "all-users-platform", "exit_code": 1, "stdout": "", "stderr": "",
                "visibility_contract": "native-host-root-fuser-prerequisites-checked"},
            "allocation": {"scope": "all-namespaces-active-workloads", "gpu_requests": 0},
            "target": {"hostname": collector.TARGET, "uid": 1000, "gid": 1000, "boot_id": request["boot_id"],
                "swap_active": False, "node_ready": True, "pressure_free": True, "expected_policy_unchanged": True},
            "devices": [{"bdf": row["bdf"], "uuid": row["retained_hip_uuid"], "render": row["render"], "sysfs_bdf": row["sysfs_bdf"]} for row in devices],
            "device_identity_provenance": devices, "status": "OBSERVATION_COMPLETE", "observed_monotonic": time.monotonic()})
        if request["mode"] == "release":
            snapshot.update({"arm": request["arm"], "child_pid": request["root_process"]["pid"], "child_start_ticks": request["root_process"]["start_ticks"]})
    except BaseException as error:
        snapshot["status"] = "INCOMPLETE"
        snapshot.pop("observed_monotonic", None)  # Incomplete records cannot pass adapter freshness gates.
        snapshot["failures"].append(type(error).__name__)  # Never retain raw credential/tool errors.
    finally:
        if timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in previous.items():
            signal.signal(number, handler)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-owner-snapshot", action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--request-sha256")
    parser.add_argument("--expected-self-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    require(args.collect_owner_snapshot and digest(args.request_sha256) and digest(args.expected_self_sha256), "explicit observer/source/request binding required")
    path, output = Path(args.request or ""), Path(args.output or "")
    require(path.name == "REQUEST.json" and path.parent == output.parent and output.name == "SNAPSHOT.json"
        and path.parent.parent == Path("/var/tmp") and re.fullmatch(r"fp8-benchmark-observer\.[A-Za-z0-9]{8,64}", path.parent.name)
        and path.parent.resolve() == path.parent, "fixed private observer directory required")
    info = path.parent.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and stat.S_IMODE(info.st_mode) == 0o700, "root-private observer directory required")
    raw = read_regular(path, 256 * 1024, 0, mode=0o444)
    require(hashlib.sha256(raw).hexdigest() == args.request_sha256, "request digest mismatch")
    request = json_bytes(raw)
    snapshot = collect_snapshot(request, expected_observer_sha256=args.expected_self_sha256,
        deadline_monotonic=request["deadline_monotonic"])
    snapshot["request_file_sha256"] = args.request_sha256
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "unexpected output timer")
    def final_deadline(number, frame):
        raise ObservationError("observer output deadline")
    previous = signal.signal(signal.SIGALRM, final_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, request["deadline_monotonic"] - time.monotonic()))
    directory, fd = None, None
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        held = os.fstat(directory)
        require((held.st_dev, held.st_ino) == (info.st_dev, info.st_ino), "private output directory changed")
        fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        raw, position = canonical_json(snapshot), 0
        require(len(raw) <= 2 * 1024 * 1024, "snapshot output bound")
        while position < len(raw):
            position += os.write(fd, raw[position:])
        os.fsync(fd)
        os.fsync(directory)
    finally:
        if fd is not None:
            os.close(fd)
        if directory is not None:
            os.close(directory)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return 0 if snapshot["status"] == "OBSERVATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
