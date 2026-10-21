"""Finite Linux host-process ownership primitives; no platform or GPU operations.

The caller pins the canonical lock and supplies independently observed namespace,
device and allocation release evidence. Process-group absence alone is not release.
"""
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid


class OwnerError(RuntimeError):
    pass


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _bounded(path, limit=65536):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, limit + 1)
        if len(raw) > limit:
            raise OwnerError("process observation size bound exceeded")
        return raw
    finally:
        os.close(fd)


def _directory(path, uid, gid):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise OwnerError("directory must have a canonical absolute path")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or (info.st_uid, info.st_gid) != (uid, gid):
        raise OwnerError("directory ownership or private mode mismatch")
    return path


def checked_script(path, expected_sha256):
    if not _digest(expected_sha256):
        raise OwnerError("invalid script digest")
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise OwnerError("script must have a canonical absolute path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 2 * 1024 * 1024:
            raise OwnerError("unsafe or oversized owned script")
        raw = os.read(fd, 2 * 1024 * 1024 + 1)
        after = os.fstat(fd)
        if info != after or _sha(raw) != expected_sha256:
            raise OwnerError("owned script identity mismatch")
    finally:
        os.close(fd)
    return str(path)


@dataclass(frozen=True)
class ProcessIdentity:
    boot_id: str
    pid: int
    ppid: int
    pgid: int
    sid: int
    start_ticks: int
    uid: int
    gid: int
    pid_namespace: str
    command_sha256: str
    state: str

    @property
    def running(self):
        return self.state not in ("Z", "X", "x")

    def same_instance(self, other):
        keys = ("boot_id", "pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace")
        return isinstance(other, ProcessIdentity) and all(getattr(self, key) == getattr(other, key) for key in keys)


def read_process(pid, proc_root="/proc"):
    """Read one Linux process identity; None means observed absence, not a zombie."""
    if type(pid) is not int or pid <= 0:
        raise OwnerError("invalid PID")
    root, directory = Path(proc_root), Path(proc_root) / str(pid)
    try:
        raw = _bounded(directory / "stat")
        right = raw.rfind(b")")
        if right < 0 or int(raw.split(b"(", 1)[0]) != pid:
            raise OwnerError("malformed Linux process stat")
        fields = raw[right + 2:].split()
        status = dict(line.split(":", 1) for line in _bounded(directory / "status").decode().splitlines() if ":" in line)
        uid_values = [int(x) for x in status["Uid"].split()]
        gid_values = [int(x) for x in status["Gid"].split()]
        if len(uid_values) != 4 or len(gid_values) != 4 or len(set(uid_values)) != 1 or len(set(gid_values)) != 1:
            raise OwnerError("process credential transition or malformed status")
        command = _bounded(directory / "cmdline")
        value = ProcessIdentity(_bounded(root / "sys/kernel/random/boot_id", 256).decode().strip(), pid,
            int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]), uid_values[0], gid_values[0],
            os.readlink(directory / "ns/pid"), _sha(command), fields[0].decode("ascii"))
        # Detect reuse or mutation while reading the multiple proc files.
        again = _bounded(directory / "stat")
        second = again[again.rfind(b")") + 2:].split()
        if int(second[19]) != value.start_ticks:
            raise OwnerError("process identity changed during read")
        return value
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, IndexError, UnicodeError) as error:
        raise OwnerError("malformed Linux process identity") from error


def descendants(known, proc_root="/proc"):
    """Identify descendants by observed authenticated parentage, including setsid."""
    current = {}
    for old in known:
        value = read_process(old.pid, proc_root)
        if value is not None and old.same_instance(value):
            current[value.pid] = value
    entries = list(Path(proc_root).iterdir())
    if len(entries) > 65536:
        raise OwnerError("process inventory bound exceeded")
    candidates = []
    for entry in entries:
        if entry.name.isdecimal():
            try:
                raw = _bounded(entry / "stat")
                fields = raw[raw.rfind(b")") + 2:].split()
                candidates.append((int(entry.name), int(fields[1])))
            except (FileNotFoundError, PermissionError):
                continue
            except (ValueError, IndexError) as error:
                raise OwnerError("malformed descendant inventory") from error
    found = []
    for _ in range(512):
        added = False
        for pid, ppid in candidates:
            parent = current.get(ppid)
            if pid in current or parent is None:
                continue
            # Do not read commands/credentials of unrelated workstation jobs.
            value = read_process(pid, proc_root)
            if value is not None and value.ppid == parent.pid and value.boot_id == parent.boot_id and value.uid == parent.uid and value.gid == parent.gid and value.start_ticks >= parent.start_ticks:
                current[value.pid] = value
                found.append(value)
                added = True
        if not added:
            return found
    raise OwnerError("descendant inventory bound exceeded")


def signal_authenticated(identity, number, identity_reader=read_process):
    """Pin the process with a Linux pidfd; numeric-PID-only fallback is forbidden."""
    value = identity_reader(identity.pid)
    if value is None or not identity.same_instance(value) or not value.running:
        return None
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise OwnerError("authenticated signaling requires Linux pidfd support")
    try:
        fd = os.pidfd_open(identity.pid, 0)
    except ProcessLookupError:
        return None
    try:
        value = identity_reader(identity.pid)
        if value is None or not identity.same_instance(value) or not value.running:
            return None
        try:
            signal.pidfd_send_signal(fd, number, None, 0)
        except ProcessLookupError:
            return None
    finally:
        os.close(fd)
    return {"pid": identity.pid, "start_ticks": identity.start_ticks,
        "signal": int(number), "observed_monotonic": time.monotonic()}


@dataclass(frozen=True)
class Deadlines:
    work_end: float
    term_end: float
    kill_end: float
    reap_end: float

    def validate(self, now):
        values = (now, self.work_end, self.term_end, self.kill_end, self.reap_end)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values) or not now < self.work_end < self.term_end < self.kill_end < self.reap_end:
            raise OwnerError("invalid or exhausted complete process envelope")


@dataclass(frozen=True)
class SupervisionResult:
    root_process: ProcessIdentity | None
    observed_processes: tuple
    exit_code: int | None
    failure: str | None
    cleanup_errors: tuple
    survivors: tuple
    child_reaped: bool
    owner_signal: int | None
    escalation_receipts: tuple = ()

    @property
    def process_cleanup_observed(self):
        return self.child_reaped is True and not self.cleanup_errors and not self.survivors


# This interpreter performs no target work until the parent persists its identity.
_GATE = "import os,sys; f=int(sys.argv[1]); b=os.read(f,1); os.close(f); b==b'G' or sys.exit(125); os.execv(sys.argv[2],sys.argv[2:])"


def supervise(argv, *, script_path, script_sha256, deadlines, on_child, stdout, stderr,
              identity_reader=read_process, descendant_reader=descendants,
              clock=time.monotonic, resource_check=lambda: None, signal_process=signal_authenticated):
    """Supervise a gated child; caller journals launch-intent before calling.

    on_child persists the authenticated child identity before the target executes.
    External namespace/device release is deliberately not asserted here.
    """
    deadlines.validate(clock())
    script = checked_script(script_path, script_sha256)
    if not isinstance(argv, (list, tuple)) or not argv or any(type(x) is not str or not x or "\x00" in x for x in argv) or script not in argv or not Path(argv[0]).is_absolute():
        raise OwnerError("invalid frozen command or missing bound script")
    child, gate_read, gate_write, root = None, None, None, None
    known, cleanup_errors, survivors, escalations = [], [], [], []
    failure, owner_signal, reaped = None, None, False
    previous = {}

    def observe():
        nonlocal known
        added = descendant_reader(tuple(known))
        if len(known) + len(added) > 512:
            raise OwnerError("owned descendant bound exceeded")
        for value in added:
            if not isinstance(value, ProcessIdentity):
                raise OwnerError("invalid descendant identity")
            if not any(value.same_instance(old) for old in known):
                known.append(value)

    def signal_owned(number):
        for old in reversed(known):
            receipt = signal_process(old, number, identity_reader)
            if receipt is not None:
                escalations.append(receipt)

    def interrupted(number, frame):
        nonlocal owner_signal
        owner_signal = number

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        gate_read, gate_write = os.pipe()
        child = subprocess.Popen([sys.executable, "-I", "-B", "-S", "-c", _GATE, str(gate_read), *argv],
            pass_fds=(gate_read,), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True)
        os.close(gate_read)
        gate_read = None
        root = identity_reader(child.pid)
        if root is None or not root.running or root.pgid != child.pid or root.sid != child.pid:
            raise OwnerError("gated child identity unavailable")
        known.append(root)
        on_child(root)
        # Reject a substituted owned script immediately before releasing the gate.
        checked_script(script_path, script_sha256)
        if clock() >= deadlines.work_end or owner_signal is not None:
            raise OwnerError("deadline or signal before gate release")
        os.write(gate_write, b"G")
        os.close(gate_write)
        gate_write = None
        while True:
            observe()
            code = child.poll()  # Polling is also the required owned-child reap.
            if code is not None:
                reaped = True
                if code != 0:
                    failure = "child exited nonzero"
                break
            resource_check()
            if owner_signal is not None:
                failure = "owner signal"
                break
            if clock() >= deadlines.work_end:
                failure = "work deadline"
                break
            time.sleep(min(0.02, max(0.001, deadlines.work_end - clock())))
    except BaseException as error:
        failure = f"{type(error).__name__}: {str(error)[:512]}"
    finally:
        for fd in (gate_read, gate_write):
            if fd is not None:
                os.close(fd)
        if child is not None:
            try:
                observe()
            except Exception as error:
                cleanup_errors.append(f"descendant observation: {type(error).__name__}: {str(error)[:256]}")
            for number, end in ((signal.SIGTERM, deadlines.term_end), (signal.SIGKILL, deadlines.kill_end)):
                try:
                    signal_owned(number)
                    while clock() < end:
                        observe()
                        code = child.poll()
                        reaped = code is not None
                        active = [old for old in known if (value := identity_reader(old.pid)) is not None and old.same_instance(value) and value.running]
                        if not active and reaped:
                            break
                        time.sleep(min(0.02, max(0.001, end - clock())))
                except Exception as error:
                    cleanup_errors.append(f"owned termination: {type(error).__name__}: {str(error)[:256]}")
            try:
                if not reaped:
                    child.wait(timeout=max(0, deadlines.reap_end - clock()))
                    reaped = True
                for old in known:
                    value = identity_reader(old.pid)
                    # An unreaped descendant zombie is not observed process absence.
                    if value is not None and old.same_instance(value):
                        survivors.append(old)
            except Exception as error:
                cleanup_errors.append(f"owned reap/postcheck: {type(error).__name__}: {str(error)[:256]}")
        for number, handler in previous.items():
            signal.signal(number, handler)
    return SupervisionResult(root, tuple(known), None if child is None else child.returncode,
        failure, tuple(cleanup_errors), tuple(survivors), reaped, owner_signal, tuple(escalations))


@dataclass(frozen=True)
class ReleaseEvidence:
    root: str
    session_sha256: str
    boot_id: str
    arm: str
    child_pid: int
    child_start_ticks: int
    lock_device: int
    lock_inode: int
    observed_monotonic: float
    process_absent: bool
    pid_namespace_absent: bool
    devices_released: bool
    allocation_released: bool
    target_policy_healthy: bool


@dataclass(frozen=True)
class IndependentReleaseEvidence:
    root: str
    session_sha256: str
    boot_id: str
    lock_device: int
    lock_inode: int
    observed_monotonic: float
    process_absent: bool
    pid_namespace_absent: bool
    devices_released: bool
    allocation_released: bool
    target_policy_healthy: bool
    independent_lock_released: bool


class CanonicalLease:
    """Hold existing FD9 continuously; retain uncertainty in a durable v2 journal."""
    def __init__(self, lock_path, root, session_sha256, boot_id, *, expected_uid=1000, expected_gid=1000, owner_identity=None):
        self.lock_path = Path(lock_path)
        self.root = _directory(root, expected_uid, expected_gid)
        self.parent = _directory(self.lock_path.parent, expected_uid, expected_gid)
        if not self.lock_path.is_absolute() or self.lock_path.resolve() != self.lock_path or not _digest(session_sha256) or not isinstance(boot_id, str) or not boot_id:
            raise OwnerError("invalid lease identity")
        self.uid, self.gid, self.session, self.boot = expected_uid, expected_gid, session_sha256, boot_id
        self.owner_identity = owner_identity
        self.directory_identities = [(path.stat().st_dev, path.stat().st_ino) for path in (self.parent, self.root)]
        self.lock_identity = None
        self.journal = self.lock_path.with_name(self.lock_path.name + ".unresolved.json")
        self.fd, self.record, self.journal_identity, self.closed_monotonic = None, None, None, None

    def _lock_identity(self, descriptor=None):
        fd = self.fd if descriptor is None else descriptor
        if fd is None:
            raise OwnerError("lease is not held")
        info, named = os.fstat(fd), self.lock_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or (info.st_uid, info.st_gid) != (self.uid, self.gid) or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino) or stat.S_ISLNK(named.st_mode):
            raise OwnerError("canonical lock identity changed")
        _directory(self.parent, self.uid, self.gid)
        _directory(self.root, self.uid, self.gid)
        if [(path.stat().st_dev, path.stat().st_ino) for path in (self.parent, self.root)] != self.directory_identities or self.lock_identity is not None and (info.st_dev, info.st_ino) != self.lock_identity:
            raise OwnerError("canonical directory or held-lock inode substituted")
        return info

    def acquire(self, inherited_fd=None):
        if self.fd is not None or os.path.lexists(self.journal):
            raise OwnerError("existing lease or unresolved ownership")
        if inherited_fd is None:
            try:
                os.fstat(9)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                raise OwnerError("FD9 already occupied; explicit inherited authentication required")
            fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.dup2(fd, 9, inheritable=False)
            except BaseException:
                os.close(fd)
                raise
            finally:
                if fd != 9 and self._fd_open(fd):
                    os.close(fd)
            self.fd = 9
        else:
            if inherited_fd != 9:
                raise OwnerError("only inherited FD9 is supported")
            self.fd = inherited_fd
        try:
            info = self._lock_identity()
            self.lock_identity = (info.st_dev, info.st_ino)
            # A separate open description must be refused, including for inherited FD9.
            contender = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    fcntl.flock(contender, fcntl.LOCK_UN)
                    raise OwnerError("FD9 does not authenticate a held exclusive lock")
            finally:
                os.close(contender)
            # A blocked contender alone could be another owner's lock. This call
            # succeeds only for our same open description, never that other owner.
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.path.lexists(self.journal):
                raise OwnerError("unresolved ownership appeared")
            owner = self.owner_identity if self.owner_identity is not None else read_process(os.getpid())
            if not isinstance(owner, ProcessIdentity) or owner.pid != os.getpid() or owner.boot_id != self.boot or (owner.uid, owner.gid) != (self.uid, self.gid) or not owner.running:
                raise OwnerError("current owner identity is unavailable or mismatched")
            self.record = {"version": 2, "kind": "host-process-benchmark", "token": uuid.uuid4().hex,
                "root": str(self.root), "session_sha256": self.session, "boot_id": self.boot,
                "owner_pid": os.getpid(), "uid": self.uid, "gid": self.gid,
                "owner": asdict(owner),
                "lock_device": info.st_dev, "lock_inode": info.st_ino, "state": "reserved", "arms": []}
            self._write(initial=True)
            return self
        except BaseException:
            # Never unlock an inherited descriptor on failed authentication.
            if inherited_fd is None and self.fd is not None:
                os.close(self.fd)
            self.fd = None
            raise

    @staticmethod
    def _fd_open(fd):
        try:
            os.fstat(fd)
            return True
        except OSError:
            return False

    def _write(self, initial=False):
        self._lock_identity()
        if not initial:
            self._verify_journal()
        temporary = self.journal if initial else self.parent / (".owner-journal-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            raw = _json(self.record)
            if len(raw) > 256 * 1024:
                raise OwnerError("journal size bound exceeded")
            offset = 0
            while offset < len(raw):
                offset += os.write(fd, raw[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        if not initial:
            self._verify_journal()
            os.replace(temporary, self.journal)
        directory_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        info = self.journal.lstat()
        self.journal_identity = (info.st_dev, info.st_ino, _sha(self.journal.read_bytes()))

    def _verify_journal(self):
        fd = os.open(self.journal, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info, raw = os.fstat(fd), os.read(fd, 256 * 1024 + 1)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or (info.st_uid, info.st_gid) != (self.uid, self.gid) or (info.st_dev, info.st_ino, _sha(raw)) != self.journal_identity:
                raise OwnerError("unresolved ownership identity changed")
        finally:
            os.close(fd)

    def begin_arm(self, arm, argv, script_path, script_sha256):
        self._lock_identity()
        self._verify_journal()
        expected = "bf16" if not self.record["arms"] else "kv4"
        if arm != expected or len(self.record["arms"]) >= 2 or self.record["state"] not in ("reserved", "released") or self.record.get("campaign_failed"):
            raise OwnerError("arm order, single-use or release gate refused")
        checked_script(script_path, script_sha256)
        if not isinstance(argv, (list, tuple)) or not argv or any(type(x) is not str or not x for x in argv) or str(script_path) not in argv:
            raise OwnerError("invalid frozen command")
        self.record["arms"].append({"arm": arm, "argv": list(argv), "argv_sha256": _sha(_json(list(argv))),
            "script": str(script_path), "script_sha256": script_sha256, "intent_monotonic": time.monotonic()})
        self.record["state"] = "launch-intent"
        self._write()

    def record_child(self, identity):
        if self.record["state"] != "launch-intent" or not isinstance(identity, ProcessIdentity) or identity.boot_id != self.boot or (identity.uid, identity.gid) != (self.uid, self.gid) or not identity.running:
            raise OwnerError("child identity does not match launch intent")
        self.record["arms"][-1]["child"] = asdict(identity)
        self.record["state"] = "running"
        self._write()

    def finish_arm(self, result, evidence):
        self._lock_identity()
        row = self.record["arms"][-1]
        expected = (str(self.root), self.session, self.boot, row["arm"], row.get("child", {}).get("pid"), row.get("child", {}).get("start_ticks"), self.record["lock_device"], self.record["lock_inode"])
        if self.record["state"] != "running" or not isinstance(result, SupervisionResult) or not result.process_cleanup_observed or result.root_process is None or not isinstance(evidence, ReleaseEvidence):
            raise OwnerError("missing observed process and external release evidence")
        observed = (evidence.root, evidence.session_sha256, evidence.boot_id, evidence.arm, evidence.child_pid, evidence.child_start_ticks, evidence.lock_device, evidence.lock_inode)
        flags = (evidence.process_absent, evidence.pid_namespace_absent, evidence.devices_released, evidence.allocation_released, evidence.target_policy_healthy)
        if observed != expected or any(type(value) is not bool or not value for value in flags) or type(evidence.observed_monotonic) not in (int, float) or not math.isfinite(evidence.observed_monotonic) or not row["intent_monotonic"] <= evidence.observed_monotonic <= time.monotonic() or not result.root_process.same_instance(ProcessIdentity(**row["child"])):
            raise OwnerError("release evidence is unbound, stale or incomplete")
        row["release"] = asdict(evidence)
        row["exit_code"] = result.exit_code
        row["failure"] = result.failure
        self.record["state"] = "released"
        self._write()
        if result.failure is not None or result.exit_code != 0:
            self.record["campaign_failed"] = True
            self._write()

    def close(self):
        if self.fd is None:
            return
        try:
            self._lock_identity()
            self._verify_journal()
            if self.record["state"] == "released":
                # Keep the refusal record across the independent lock/device
                # observation. A failed final check must not create a clean slate.
                self.record["state"] = "pending-independent"
                self._write()
            elif self.record["state"] == "reserved" and not self.record["arms"]:
                self.journal.unlink()
                directory_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
            self.closed_monotonic = time.monotonic()

    def confirm_closed(self, evidence):
        """Finalize only after fresh independent observations, under a new probe lock."""
        if self.fd is not None or self.closed_monotonic is None or self.record["state"] != "pending-independent" or not isinstance(evidence, IndependentReleaseEvidence):
            raise OwnerError("independent finalization is not admissible")
        expected = (str(self.root), self.session, self.boot, self.record["lock_device"], self.record["lock_inode"])
        actual = (evidence.root, evidence.session_sha256, evidence.boot_id, evidence.lock_device, evidence.lock_inode)
        flags = (evidence.process_absent, evidence.pid_namespace_absent, evidence.devices_released,
            evidence.allocation_released, evidence.target_policy_healthy, evidence.independent_lock_released)
        if actual != expected or any(type(value) is not bool or not value for value in flags) or type(evidence.observed_monotonic) not in (int, float) or not math.isfinite(evidence.observed_monotonic) or not self.closed_monotonic <= evidence.observed_monotonic <= time.monotonic():
            raise OwnerError("independent release is stale, unbound or incomplete")
        fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
        removed, original_journal = False, None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = self._lock_identity(fd)
            if (info.st_dev, info.st_ino) != (evidence.lock_device, evidence.lock_inode):
                raise OwnerError("verification lock identity mismatch")
            self._verify_journal()
            original_journal = _bounded(self.journal, 256 * 1024)
            self.journal.unlink()
            removed = True
            directory_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.record["state"] = "closed"
        except BaseException as original_error:
            if removed:
                # Directory-fsync failure is uncertain deletion, not accepted
                # release. Restore only our exact record; never replace another.
                try:
                    restore = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    try:
                        offset = 0
                        while offset < len(original_journal):
                            offset += os.write(restore, original_journal[offset:])
                        os.fsync(restore)
                    finally:
                        os.close(restore)
                    directory_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    info = self.journal.lstat()
                    self.journal_identity = (info.st_dev, info.st_ino, _sha(original_journal))
                except BaseException as restoration_error:
                    raise OwnerError("UNRESOLVED RECORD LOSS: deletion failed and durable refusal restoration failed: " +
                        type(restoration_error).__name__ + ": " + str(restoration_error)[:256]) from original_error
            raise
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
