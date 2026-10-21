"""Owner-run, read-only availability snapshot. Never launch or release authority.

Run manually as root with isolated stdlib Python. No sudo, remote shell, GPU
library, lock acquisition, installation or runtime mutation. The collector does
not read credential contents; fixed kubectl internally uses its protected config.
Only a new root-owned private /var/tmp evidence directory is written.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import selectors
import signal
import socket
import stat
import subprocess
import sys
import time


TARGET = "threadripper-ai"
LOCK = Path("/home/developer/model-serving.izR2m1O9/diagnostic.lock")
POLICY = "workstation-serving-baseline-deny"
POLICY_UID = "6fabb1de-4a6c-484b-9d39-3a1cb7217755"
DEVICES = {"0000:c3:00.0": "39373864-6465-3038-6163-303837666533",
           "0000:23:00.0": "34663466-6561-3362-3832-346261366439"}
RETAINED_SYSFS = {"0000:c3:00.0": "978dde08ac087fe3", "0000:23:00.0": "4f4fea3b824ba6d9"}
SYSFS_PROVENANCE = {"runtime_status_sha256": "02de652b9425da39bf34752794e84685af4dc01529d245249f7fa7a586587d6e",
    "native_identity_sha256": "94fbf86375f77e682ea63fb3399d97c0eeda56bdcc8f9408401769100cbbed8d"}
KUBE = ("/usr/bin/k3s", "kubectl", "--kubeconfig=/etc/rancher/k3s/k3s.yaml",
    "--server=https://192.168.2.15:6443", "--context=default", "--namespace=default", "--request-timeout=5s")
PODS_OUTPUT = 'jsonpath={"BEGIN-POD-INVENTORY\\t"}{.kind}{"\\n"}{range .items[*]}{.status.phase}{"\\t"}{range .spec.containers[*]}{.resources.requests.amd\\.com/gpu}{"/"}{.resources.limits.amd\\.com/gpu}{","}{end}{"\\t"}{range .spec.initContainers[*]}{.resources.requests.amd\\.com/gpu}{"/"}{.resources.limits.amd\\.com/gpu}{","}{end}{"\\t"}{range .spec.resourceClaims[*]}{"x"}{end}{range .spec.containers[*]}{range .resources.claims[*]}{"x"}{end}{end}{range .spec.initContainers[*]}{range .resources.claims[*]}{"x"}{end}{end}{"\\t"}{.spec.overhead.amd\\.com/gpu}{"\\n"}{end}{"END-POD-INVENTORY\\n"}'
NODE_OUTPUT = 'jsonpath={.metadata.name}{"\\t"}{.metadata.uid}{"\\t"}{.status.nodeInfo.kernelVersion}{"\\t"}{.status.allocatable.amd\\.com/gpu}{"\\t"}{.status.nodeInfo.bootID}{"\\n"}{range .status.conditions[*]}{.type}{"\\t"}{.status}{"\\n"}{end}'
POLICY_OUTPUT = 'go-template={{.metadata.uid}}{{"\\n"}}{{.metadata.name}}{{"\\n"}}{{len .spec.podSelector}}{{"\\n"}}{{with .spec.podSelector.matchLabels}}{{len .}}{{"\\t"}}{{index . "workstation-serving"}}{{else}}0{{end}}{{"\\n"}}{{with .spec.podSelector.matchExpressions}}{{len .}}{{else}}0{{end}}{{"\\n"}}{{range .spec.policyTypes}}{{.}}{{","}}{{end}}{{"\\n"}}{{with .spec.ingress}}{{len .}}{{else}}0{{end}}{{"\\n"}}{{with .spec.egress}}{{len .}}{{else}}0{{end}}{{"\\n"}}'
COMMANDS = {"pods": KUBE + ("get", "pods", "-A", "-o", PODS_OUTPUT),
    "node": KUBE + ("get", "node", TARGET, "-o", NODE_OUTPUT),
    "policy": KUBE + ("get", "networkpolicy", POLICY, "-o", POLICY_OUTPUT),
    "holders": ("/usr/bin/fuser", "/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129")}
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "HOME": "/nonexistent",
    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
LIMIT = 1024 * 1024


class CollectionError(RuntimeError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def boot_id():
    value = bounded_read("/proc/sys/kernel/random/boot_id", 256).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value):
        raise CollectionError("missing or malformed boot identity")
    return value


def self_stat():
    return Path(__file__).resolve().stat()


def stable_stat(info):
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def bounded_read(path, maximum=65536):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, maximum + 1)
        if len(raw) > maximum:
            raise CollectionError("bounded metadata read exceeded")
        return raw
    finally:
        os.close(fd)


def source_identity(path, deadline):
    canonical = Path(path).resolve(strict=True)
    for directory in canonical.parents:
        info = directory.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise CollectionError("command source has a user-writable ancestor")
    digest = hashlib.sha256()
    fd = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_mode & 0o022 or before.st_size > 512 * 1024**2 or stable_stat(before) != stable_stat(canonical.lstat()):
            raise CollectionError("command source is not root-owned immutable-to-users")
        while chunk := stream.read(1024 * 1024):
            if time.monotonic() >= deadline:
                raise CollectionError("command identity deadline")
            digest.update(chunk)
        if stable_stat(before) != stable_stat(os.fstat(stream.fileno())) or stable_stat(before) != stable_stat(canonical.lstat()):
            raise CollectionError("command source changed")
    return {"path": str(path), "canonical_path": str(canonical), "sha256": digest.hexdigest(), "bytes": before.st_size}


def kubeconfig_metadata():
    path = Path("/etc/rancher/k3s/k3s.yaml")
    for component in reversed((path, *path.parents)):
        info = component.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
            raise CollectionError("protected kubeconfig path metadata is unsafe")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CollectionError("protected kubeconfig is not a regular file")
    # Never read or hash the credential file. kubectl internally uses the
    # existing root-controlled configuration through its fixed argument.
    return {"root_controlled_path": True, "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid, "gid": info.st_gid, "contents_read_by_collector": False}


def create_output(value):
    path = Path(value)
    if path.parent != Path("/var/tmp") or not re.fullmatch(r"fp8-benchmark-metadata\.[A-Za-z0-9]{8,32}", path.name):
        raise CollectionError("output must be fresh /var/tmp/fp8-benchmark-metadata.<8-32 alnum nonce>")
    for component in (Path("/"), Path("/var"), Path("/var/tmp")):
        info = component.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise CollectionError("unsafe output parent")
    parent = os.open("/var/tmp", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)  # Existing entries, including links, refuse.
        fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        info = os.fstat(fd)
        if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o700:
            os.close(fd)
            raise CollectionError("output ownership mismatch")
        os.fsync(parent)
        return path, fd
    finally:
        os.close(parent)


def run_command(name, work_end):
    """Bound all captured bytes; retain only hashes/receipts, never raw logs."""
    argv = COMMANDS[name]
    beginning = time.monotonic()
    command_end, envelope_end = beginning + 8, beginning + 14
    if envelope_end > work_end:
        raise CollectionError("command envelope does not fit")
    before = source_identity(argv[0], command_end)
    if time.monotonic() >= command_end:
        raise CollectionError("command execution envelope exhausted during identity check")
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=ENV, close_fds=True, start_new_session=True)
    selector, chunks, failure, signals = None, {"stdout": bytearray(), "stderr": bytearray()}, None, []
    cleanup_signals, cleanup_handlers = [], {}
    start = time.monotonic()
    try:
        selector = selectors.DefaultSelector()
        for pipe, kind in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, kind)
        while selector.get_map():
            if time.monotonic() >= command_end:
                failure = "COMMAND_TIMEOUT"
                break
            for key, _ in selector.select(min(.05, max(0, command_end - time.monotonic()))):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                if sum(len(part) for part in chunks.values()) + len(data) > LIMIT:
                    failure = "OUTPUT_BOUND_EXCEEDED"
                    break
                chunks[key.data].extend(data)
            if failure:
                break
        if failure is None:
            while time.monotonic() < command_end:
                if child_exited_unreaped(process.pid):
                    break
                time.sleep(.02)
            else:
                failure = "COMMAND_TIMEOUT"
    except BaseException as error:
        failure = "COMMAND_EXCEPTION:" + type(error).__name__
    finally:
        # Repeated timeout/owner signals are recorded but cannot interrupt the
        # bounded owned-child cleanup and replace its partial receipt.
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            cleanup_handlers[number] = signal.signal(number, lambda signum, frame: cleanup_signals.append(int(signum)))
        # Keep the direct child UNREAPED until both escalation steps finish.
        # Its reserved PID prevents the owned PGID becoming an unrelated group.
        for number in (signal.SIGTERM, signal.SIGKILL):
            try:
                if os.getpgid(process.pid) != process.pid:
                    failure = "OWNED_COMMAND_GROUP_CHANGED"
                    break
                os.killpg(process.pid, number)
                signals.append(int(number))
            except ProcessLookupError:
                pass
            except BaseException as error:
                failure = "OWNED_GROUP_TERMINATION_EXCEPTION:" + type(error).__name__
                # The still-unreaped owned direct PID cannot have been reused.
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if number == signal.SIGTERM:
                time.sleep(min(.1, max(0, envelope_end - time.monotonic())))
        try:
            process.wait(timeout=max(0, min(2, envelope_end - time.monotonic())))
        except subprocess.TimeoutExpired:
            failure = "CHILD_REAP_UNCERTAIN"
        if selector is not None:
            selector.close()
        process.stdout.close()
        process.stderr.close()
        for number, handler in cleanup_handlers.items():
            signal.signal(number, handler)
    group_absent = False
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        group_absent = True
    if not group_absent:
        failure = "CHILD_GROUP_RELEASE_UNCERTAIN"
    if cleanup_signals:
        failure = "OWNER_OR_DEADLINE_SIGNAL_DURING_CLEANUP"
    raw = {key: bytes(value) for key, value in chunks.items()}
    receipt = {"name": name, "argv": list(argv), "source": before,
        "exit_code": process.returncode, "failure": failure, "elapsed_seconds": time.monotonic() - start,
        "pid": process.pid, "pgid": process.pid, "reaped": process.returncode is not None,
        "group_absent": group_absent, "owner_signals": signals,
        "cleanup_signals": cleanup_signals,
        "stdout_bytes": len(raw["stdout"]), "stderr_bytes": len(raw["stderr"]),
        "stdout_sha256": sha(raw["stdout"]), "stderr_sha256": sha(raw["stderr"])}
    return receipt, raw


def child_exited_unreaped(pid):
    if not all(hasattr(os, key) for key in ("waitid", "WNOWAIT", "WEXITED", "WNOHANG", "P_PID")):
        raise CollectionError("Linux waitid WNOWAIT is required for safe command cleanup")
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def root_fuser_prerequisites():
    """Check native-host prerequisites, not adversarial-container host attestation."""
    record = {"contract": "native-ssh-host-root-fuser-prerequisites-v1",
        "native_ssh_host_contract_still_required": True, "initial_host_attested": False,
        "effective_root": os.geteuid() == 0, "errors": []}
    for name in ("uid", "gid"):
        try:
            rows = [line.split() for line in bounded_read("/proc/self/" + name + "_map", 4096).decode("ascii").splitlines() if line.strip()]
            record["full_initial_" + name + "_map"] = rows == [["0", "0", "4294967295"]]
        except (OSError, CollectionError, UnicodeError):
            record["full_initial_" + name + "_map"] = False
            record["errors"].append(name + "_MAP_UNAVAILABLE")
    try:
        status = bounded_read("/proc/self/status", 65536).decode("ascii")
        matches = re.findall(r"^CapEff:\s*([0-9a-fA-F]{1,16})\s*$", status, re.MULTILINE)
        if len(matches) != 1:
            raise CollectionError("missing effective capabilities")
        effective = int(matches[0], 16)
        record["effective_capabilities_hex"] = matches[0].lower()
        record["cap_sys_ptrace"] = bool(effective & (1 << 19))
        record["cap_dac_override_or_read_search"] = bool(effective & ((1 << 1) | (1 << 2)))
    except (OSError, CollectionError, UnicodeError, ValueError):
        record["effective_capabilities_hex"] = None
        record["cap_sys_ptrace"] = record["cap_dac_override_or_read_search"] = False
        record["errors"].append("EFFECTIVE_CAPABILITIES_UNAVAILABLE")
    for kind in ("pid", "user"):
        try:
            current, visible_init = (os.readlink("/proc/" + who + "/ns/" + kind) for who in ("self", "1"))
            if not all(re.fullmatch(kind + r":\[[0-9]+\]", value) for value in (current, visible_init)):
                raise CollectionError("malformed namespace identity")
            record[kind + "_namespace_self"] = current
            record[kind + "_namespace_visible_pid1"] = visible_init
            record[kind + "_namespace_matches_visible_pid1"] = current == visible_init
        except (OSError, CollectionError):
            record[kind + "_namespace_matches_visible_pid1"] = False
            record["errors"].append(kind.upper() + "_NAMESPACE_UNAVAILABLE")
    record["passed"] = all(record[key] is True for key in ("effective_root", "full_initial_uid_map", "full_initial_gid_map",
        "cap_sys_ptrace", "cap_dac_override_or_read_search", "pid_namespace_matches_visible_pid1", "user_namespace_matches_visible_pid1"))
    return record


def local_metadata():
    account = pwd.getpwnam("developer")
    swaps = bounded_read("/proc/swaps").decode().splitlines()
    memory = dict(line.split(":", 1) for line in bounded_read("/proc/meminfo").decode().splitlines() if ":" in line)
    lock = LOCK.lstat()
    if stat.S_ISLNK(lock.st_mode) or not stat.S_ISREG(lock.st_mode):
        raise CollectionError("canonical lock is not a regular non-link file")
    filesystem = os.statvfs("/")
    devices = []
    for render in ("renderD128", "renderD129"):
        device = Path("/sys/class/drm") / render / "device"
        canonical = device.resolve(strict=True)
        bdf = canonical.name
        if bdf not in DEVICES or not str(canonical).startswith("/sys/devices/"):
            raise CollectionError("unexpected sysfs render binding")
        character = (Path("/dev/dri") / render).lstat()
        major, minor = map(int, bounded_read(device.parent / "dev", 64).decode().strip().split(":"))
        if not stat.S_ISCHR(character.st_mode) or (os.major(character.st_rdev), os.minor(character.st_rdev)) != (major, minor):
            raise CollectionError("render character-device/sysfs binding mismatch")
        unique_id = bounded_read(canonical / "unique_id", 256).decode().strip()
        vendor = bounded_read(canonical / "vendor", 64).decode().strip()
        device_id = bounded_read(canonical / "device", 64).decode().strip()
        if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{1,32}", unique_id) or vendor != "0x1002" or device_id != "0x7551":
            raise CollectionError("invalid sysfs AMD identity")
        historical_match = unique_id == RETAINED_SYSFS[bdf]
        devices.append({"render": "/dev/dri/" + render, "bdf": bdf, "sysfs_bdf": bdf,
            "device_major": major, "device_minor": minor,
            "vendor": vendor, "device_id": device_id, "unique_id_raw": unique_id, "retained_sysfs_binding_matches": historical_match,
            "retained_hip_uuid": DEVICES[bdf] if historical_match else None,
            "fresh_native_uuid_observed": False, "uuid_binding_provenance": SYSFS_PROVENANCE})
    return {"hostname": socket.gethostname(), "boot_id": boot_id(),
        "kernel_release": os.uname().release, "operator_euid": os.geteuid(),
        "developer_uid": account.pw_uid, "developer_gid": account.pw_gid,
        "swap_active": len(swaps) != 1, "available_memory_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
        "free_disk_bytes": filesystem.f_bavail * filesystem.f_frsize,
        "canonical_lock": {"path": str(LOCK), "uid": lock.st_uid, "gid": lock.st_gid,
            "mode": stat.S_IMODE(lock.st_mode), "links": lock.st_nlink, "device": lock.st_dev,
            "inode": lock.st_ino, "lock_acquired": False,
            "unresolved_record_present": os.path.lexists(str(LOCK) + ".unresolved.json")}, "devices": devices}


def sanitize(name, raw):
    if name == "holders":
        diagnostic = raw["stderr"].decode("ascii")
        for device in ("/dev/kfd:", "/dev/dri/renderD128:", "/dev/dri/renderD129:"):
            diagnostic = diagnostic.replace(device, "")
        if diagnostic.strip():
            raise CollectionError("holder stderr refuses all-user visibility")
        text = raw["stdout"].decode("ascii")
        if text.strip() and not re.fullmatch(r"\s*\d+(?:\s+\d+)*\s*", text):
            raise CollectionError("unrecognized holder output")
        return {"pids": sorted(set(int(value) for value in text.split())), "visibility": "native-host-root-fuser-prerequisites-checked"}
    lines = raw["stdout"].decode("ascii").splitlines()
    if name == "pods":
        total, active = 0, 0
        if len(lines) < 2 or lines[0] not in ("BEGIN-POD-INVENTORY\tList", "BEGIN-POD-INVENTORY\tPodList") or lines[-1] != "END-POD-INVENTORY":
            raise CollectionError("missing/truncated Pod inventory framing")
        for line in lines[1:-1]:
            phase, regular, initial, claims, overhead = line.split("\t")
            if phase not in ("Pending", "Running", "Unknown", "Succeeded", "Failed"):
                raise CollectionError("unknown workload phase")
            if phase in ("Succeeded", "Failed"):
                continue
            active += 1
            if claims or not regular or not re.fullmatch(r"(?:[0-9]{0,6}/[0-9]{0,6},)*", regular + initial):
                raise CollectionError("DRA or malformed container allocation is unknown")
            for item in (regular + initial).strip(",").split(","):
                requests, limits = item.split("/")
                total += max(int(requests or "0"), int(limits or "0"))
            if overhead and not re.fullmatch(r"[0-9]{1,6}", overhead):
                raise CollectionError("invalid GPU overhead")
            total += int(overhead or "0")
        return {"scope": "all-namespaces-active-workloads", "active_workload_count": active,
            "conservative_gpu_requests": total, "unrelated_workload_names_retained": False}
    if name == "node":
        node, uid, kernel, gpu, boot = lines[0].split("\t")
        conditions = dict(line.split("\t") for line in lines[1:])
        if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uid) or len(conditions) != len(lines) - 1 or not re.fullmatch(r"[0-9A-Za-z._+-]{1,128}", kernel):
            raise CollectionError("malformed node identity")
        return {"name_matches": node == TARGET, "uid": uid,
            "ready": conditions.get("Ready") == "True", "pressure_free": all(conditions.get(key) == "False" for key in ("MemoryPressure", "DiskPressure", "PIDPressure")),
            "kernel_release": kernel, "allocatable_gpu": gpu, "boot_id": boot}
    if len(lines) != 8:
        raise CollectionError("policy metadata missing")
    uid, name, selector_fields, label, expressions, kinds, ingress, egress = lines
    return {"name_matches": name == POLICY, "uid": uid,
        "selected_metadata_sha256": sha(raw["stdout"]), "matches_retained_deny_contract": uid == POLICY_UID and name == POLICY and selector_fields == "1" and label == "1\tbaseline" and expressions == "0" and sorted(kinds.strip(",").split(",")) == ["Egress", "Ingress"] and ingress == egress == "0"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-owner-metadata", action="store_true")
    parser.add_argument("--expected-self-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.collect_owner_metadata or not args.output or not re.fullmatch(r"[0-9a-f]{64}", args.expected_self_sha256 or ""):
        parser.error("explicit collection switch, source hash and fixed-prefix fresh output are required")
    started, work_end = time.monotonic(), time.monotonic() + 50
    self_raw = bounded_read(Path(__file__).resolve(), 256 * 1024)
    source = self_stat()
    if sha(self_raw) != args.expected_self_sha256 or os.geteuid() != 0 or source.st_uid != 0 or source.st_mode & 0o022 or source.st_nlink != 1 or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise CollectionError("source/root/isolated-stdlib execution precondition failed")
    output, output_fd = create_output(args.output)
    report = {"schema": 1, "scope": "owner-run-availability-metadata-only-v1", "source_sha256": sha(self_raw),
        "start_utc": utc(), "status": "INCOMPLETE", "not_current_authority": True,
        "launch_authorized": False, "qualification": False, "gpu_work_performed": False,
        "release_authority": False, "commands": [], "observations": {}, "failures": []}
    previous, timer_owned = {}, False
    def owner_signal(number, frame):
        report["owner_signal"] = int(number)
        raise CollectionError("owner signal")
    def deadline_signal(number, frame):
        report["work_deadline_exceeded"] = True
        raise CollectionError("finite collection deadline")
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, owner_signal)
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise CollectionError("existing process timer refuses nested collector")
        previous[signal.SIGALRM] = signal.signal(signal.SIGALRM, deadline_signal)
        timer_owned = True
        signal.setitimer(signal.ITIMER_REAL, max(.001, work_end - time.monotonic()))
        report["local"] = local_metadata()
        report["root_fuser_prerequisites"] = root_fuser_prerequisites()
        if not report["root_fuser_prerequisites"]["passed"]:
            report["failures"].append("NATIVE_ROOT_FUSER_VISIBILITY_PREREQUISITES_FAILED")
            raise CollectionError("root fuser visibility prerequisites failed")
        report["kubeconfig_metadata"] = kubeconfig_metadata()
        for name in COMMANDS:
            if time.monotonic() + 14 > work_end:
                report["failures"].append("COMPLETE_COMMAND_ENVELOPE_DOES_NOT_FIT")
                break
            receipt, raw = run_command(name, work_end)
            report["commands"].append(receipt)
            if report.get("owner_signal") or report.get("work_deadline_exceeded") or receipt.get("cleanup_signals"):
                report["failures"].append("OWNER_SIGNAL")
                break
            expected = (0, 1) if name == "holders" else (0,)
            if receipt["failure"] or receipt["exit_code"] not in expected or name != "holders" and raw["stderr"].strip():
                report["failures"].append(name + ":COMMAND_OR_VISIBILITY_FAILURE")
                continue
            try:
                sanitized = sanitize(name, raw)
                if name == "holders" and (receipt["exit_code"] == 0) != bool(sanitized["pids"]):
                    raise CollectionError("holder status and PID evidence disagree")
                report["observations"][name] = sanitized
            except (CollectionError, ValueError, KeyError, IndexError, TypeError, UnicodeError):
                report["failures"].append(name + ":MALFORMED_OR_UNAVAILABLE_METADATA")
        local, facts = report["local"], report["observations"]
        lock = local["canonical_lock"]
        if local["hostname"] != TARGET or (local["developer_uid"], local["developer_gid"]) != (1000, 1000) or local["swap_active"] or (lock["uid"], lock["gid"], lock["mode"], lock["links"]) != (1000, 1000, 0o600, 1) or lock["unresolved_record_present"]:
            report["failures"].append("TARGET_SWAP_OR_LOCK_UNAVAILABLE")
        if local["available_memory_bytes"] < 46 * 1024**3 or local["free_disk_bytes"] < 64 * 1024**3:
            report["failures"].append("RESOURCE_HEADROOM_UNAVAILABLE")
        if facts.get("holders", {}).get("pids") or facts.get("pods", {}).get("conservative_gpu_requests", 1) != 0:
            report["failures"].append("GPU_BUSY_OR_ALLOCATION_UNKNOWN")
        node = facts.get("node", {})
        if not node.get("name_matches") or not node.get("ready") or not node.get("pressure_free") or node.get("kernel_release") != local["kernel_release"] or node.get("boot_id") != local["boot_id"] or node.get("allocatable_gpu") != "2" or not facts.get("policy", {}).get("matches_retained_deny_contract"):
            report["failures"].append("NODE_OR_POLICY_UNAVAILABLE")
        if len(local["devices"]) != 2 or {row["bdf"] for row in local["devices"]} != set(DEVICES) or any(not row.get("retained_sysfs_binding_matches") for row in local["devices"]):
            report["failures"].append("SYSFS_BINDING_MISSING_OR_CHANGED")
        report["fresh_native_uuid_observed"] = False
        if len(facts) != 4:
            report["failures"].append("REQUIRED_METADATA_MISSING")
        if not report["failures"]:
            report["status"] = "OWNER_METADATA_COLLECTED"
    except BaseException as error:
        # Never publish raw exception text: commands can embed credential errors.
        report["failures"].append("COLLECTION_EXCEPTION:" + type(error).__name__)
    finally:
        if timer_owned:
            signal.setitimer(signal.ITIMER_REAL, max(.001, started + 60 - time.monotonic()))
        try:
            report["end_boot_id"] = boot_id()
            if report["end_boot_id"] != report.get("local", {}).get("boot_id"):
                report["status"] = "INCOMPLETE"
                report["failures"].append("BOOT_CHANGED_OR_MISSING")
        except (CollectionError, OSError, ValueError, UnicodeError):
            report["status"] = "INCOMPLETE"
            report["failures"].append("END_BOOT_UNAVAILABLE")
        report["end_utc"] = utc()
        report["collection_span_seconds"] = time.monotonic() - started
        if report["collection_span_seconds"] > 60:
            report["status"] = "INCOMPLETE"
            report["failures"].append("COMPLETE_COLLECTION_ENVELOPE_EXCEEDED")
        fd = None
        try:
            fd = os.open("REPORT.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
            raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
            position = 0
            while position < len(raw):
                position += os.write(fd, raw[position:])
            os.fsync(fd)
        finally:
            try:
                if fd is not None:
                    os.close(fd)
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
                if timer_owned:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                for number, handler in previous.items():
                    signal.signal(number, handler)
    print(str(output / "REPORT.json"))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
