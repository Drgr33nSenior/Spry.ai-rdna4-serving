"""CPU ownership contracts; proc/hardware evidence is explicitly synthetic.

Real POSIX child tests exercise the gate, polling, signals and reaping. Their
portable ps observation is a test boundary, not Linux /proc or GPU qualification.
"""
from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("host_process_owner", ROOT / "tools/host_process_owner.py")
owner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = owner
SPEC.loader.exec_module(owner)


def identity(pid, ppid=1, ticks=10, state="S", pgid=None, sid=None):
    return owner.ProcessIdentity("fixture-boot", pid, ppid, pid if pgid is None else pgid,
        pid if sid is None else sid, ticks, os.getuid(), os.getgid(), "pid:[fixture]", "a" * 64, state)


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)
        self.lock = self.root / "diagnostic.lock"
        self.lock.touch(mode=0o600)
        self.script = self.work / "child.py"
        self.script.write_text("print('complete', flush=True)\n")
        self.script_sha = hashlib.sha256(self.script.read_bytes()).hexdigest()
        self.command = [sys.executable, "-I", "-B", "-S", str(self.script)]
        self.leases = []

    def tearDown(self):
        for lease in self.leases:
            if lease.fd is not None:
                try:
                    lease.close()
                except Exception:
                    if lease.fd is not None:
                        os.close(lease.fd)
                        lease.fd = None
        self.temporary.cleanup()

    def lease(self):
        value = owner.CanonicalLease(self.lock, self.work, "b" * 64, "fixture-boot",
            expected_uid=os.getuid(), expected_gid=os.getgid(), owner_identity=identity(os.getpid()))
        self.leases.append(value)
        return value

    def running(self, arm="bf16"):
        lease = self.lease().acquire()
        lease.begin_arm(arm, self.command, self.script, self.script_sha)
        child = identity(12345)
        lease.record_child(child)
        return lease, child

    def released(self, lease, child, failure=None, exit_code=0):
        result = owner.SupervisionResult(child, (child,), exit_code, failure, (), (), True, None)
        record = lease.record
        evidence = owner.ReleaseEvidence(str(self.work), "b" * 64, "fixture-boot", record["arms"][-1]["arm"],
            child.pid, child.start_ticks, record["lock_device"], record["lock_inode"], time.monotonic(),
            True, True, True, True, True)
        return result, evidence

    def independent(self, lease):
        return owner.IndependentReleaseEvidence(str(self.work), "b" * 64, "fixture-boot",
            lease.record["lock_device"], lease.record["lock_inode"], time.monotonic(),
            True, True, True, True, True, True)

    def test_existing_private_lock_contention_and_reserved_release(self):
        lease = self.lease().acquire()
        self.assertEqual(lease.fd, 9)
        raw = json.loads(lease.journal.read_bytes())
        self.assertEqual(raw["version"], 2)
        self.assertEqual(raw["kind"], "host-process-benchmark")
        self.assertEqual(raw["owner"]["pid"], os.getpid())
        contender = os.open(self.lock, os.O_RDWR)
        with self.assertRaises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease.close()
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(contender)
        self.assertFalse(lease.journal.exists())

    def test_existing_foreign_unresolved_including_symlink_refuses(self):
        journal = self.lock.with_name("diagnostic.lock.unresolved.json")
        journal.write_text('{"foreign":true}')
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()
        self.assertEqual(journal.read_text(), '{"foreign":true}')
        journal.unlink()
        journal.symlink_to(self.work / "not-created")
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()

    def test_inherited_unlocked_description_cannot_borrow_another_holder(self):
        holder = os.open(self.lock, os.O_RDWR)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        other = os.open(self.lock, os.O_RDWR)
        os.dup2(other, 9)
        if other != 9:
            os.close(other)
        try:
            with self.assertRaises(BlockingIOError):
                self.lease().acquire(inherited_fd=9)
        finally:
            os.close(9)
            os.close(holder)

    def test_inherited_held_description_authenticates_and_releases(self):
        fd = os.open(self.lock, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.dup2(fd, 9)
        if fd != 9:
            os.close(fd)
        lease = self.lease().acquire(inherited_fd=9)
        lease.close()
        fd = os.open(self.lock, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(fd)

    def test_exact_arm_order_single_use_and_bound_release(self):
        lease, child = self.running()
        self.assertEqual(lease.record["state"], "running")
        result, evidence = self.released(lease, child)
        for invalid in (replace(evidence, arm="kv4"), replace(evidence, child_start_ticks=11),
                        replace(evidence, devices_released=1), replace(evidence, observed_monotonic=0)):
            with self.assertRaises(owner.OwnerError):
                lease.finish_arm(result, invalid)
        lease.finish_arm(result, evidence)
        with self.assertRaises(owner.OwnerError):
            lease.begin_arm("bf16", self.command, self.script, self.script_sha)
        lease.begin_arm("kv4", self.command, self.script, self.script_sha)
        lease.record_child(child)
        result, evidence = self.released(lease, child)
        lease.finish_arm(result, evidence)
        with self.assertRaises(owner.OwnerError):
            lease.begin_arm("kv4", self.command, self.script, self.script_sha)
        lease.close()
        self.assertTrue(lease.journal.exists())
        lease.confirm_closed(self.independent(lease))
        self.assertFalse(lease.journal.exists())

    def test_failed_arm_stops_but_observed_release_allows_safe_abort(self):
        lease, child = self.running()
        lease.finish_arm(*self.released(lease, child, "nonzero", 2))
        with self.assertRaises(owner.OwnerError):
            lease.begin_arm("kv4", self.command, self.script, self.script_sha)
        lease.close()
        self.assertTrue(lease.journal.exists())
        lease.confirm_closed(self.independent(lease))
        self.assertFalse(lease.journal.exists())

    def test_independent_failure_keeps_durable_refusal_and_probe_releases(self):
        lease, child = self.running()
        lease.finish_arm(*self.released(lease, child))
        lease.close()
        self.assertEqual(json.loads(lease.journal.read_bytes())["state"], "pending-independent")
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()
        evidence = self.independent(lease)
        for bad in (replace(evidence, process_absent=False), replace(evidence, devices_released=False),
                    replace(evidence, independent_lock_released=False), replace(evidence, observed_monotonic=0)):
            with self.assertRaises(owner.OwnerError):
                lease.confirm_closed(bad)
            self.assertTrue(lease.journal.exists())
        holder = os.open(self.lock, os.O_RDWR)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(BlockingIOError):
            lease.confirm_closed(self.independent(lease))
        self.assertTrue(lease.journal.exists())
        os.close(holder)
        lease.confirm_closed(self.independent(lease))
        self.assertFalse(lease.journal.exists())
        contender = os.open(self.lock, os.O_RDWR)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(contender)

    def test_independent_directory_fsync_failure_restores_exact_refusal(self):
        lease, child = self.running()
        lease.finish_arm(*self.released(lease, child))
        lease.close()
        before = lease.journal.read_bytes()
        actual_fsync, failed = owner.os.fsync, []
        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode) and not failed:
                failed.append(True)
                raise OSError("fixture directory fsync failure")
            return actual_fsync(fd)
        with patch.object(owner.os, "fsync", side_effect=fsync):
            with self.assertRaisesRegex(OSError, "fixture directory fsync"):
                lease.confirm_closed(self.independent(lease))
        self.assertEqual(lease.journal.read_bytes(), before)
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()
        lease.confirm_closed(self.independent(lease))
        self.assertFalse(lease.journal.exists())

    def test_uncertain_cleanup_or_ambiguous_intent_retains_journal(self):
        lease, child = self.running()
        result, evidence = self.released(lease, child)
        with self.assertRaises(owner.OwnerError):
            lease.finish_arm(replace(result, survivors=(child,)), evidence)
        lease.close()
        self.assertTrue(lease.journal.exists())
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()

    def test_journal_substitution_lock_link_and_source_substitution_refuse(self):
        lease = self.lease().acquire()
        lease.journal.write_text('{}')
        with self.assertRaises(owner.OwnerError):
            lease.begin_arm("bf16", self.command, self.script, self.script_sha)
        with self.assertRaises(owner.OwnerError):
            lease.close()
        self.script.write_text("raise RuntimeError('substituted')\n")
        with self.assertRaises(owner.OwnerError):
            owner.checked_script(self.script, self.script_sha)
        alias = self.root / "lock-alias"
        os.link(self.lock, alias)
        with self.assertRaises(owner.OwnerError):
            self.lease().acquire()

    def test_held_lock_or_private_root_inode_substitution_refuses(self):
        lease = self.lease().acquire()
        original = self.root / "original-work"
        self.work.rename(original)
        self.work.mkdir(mode=0o700)
        try:
            with self.assertRaises(owner.OwnerError):
                lease.begin_arm("bf16", self.command, self.script, self.script_sha)
            with self.assertRaises(owner.OwnerError):
                lease.close()
        finally:
            self.work.rmdir()
            original.rename(self.work)
        # Independent inode substitution cannot make a newly opened FD9 stand
        # in for the continuously held original description.
        lease.journal.unlink()
        second = self.lease().acquire()
        replacement = self.root / "replacement"
        replacement.touch(mode=0o600)
        replacement.replace(self.lock)
        with self.assertRaises(owner.OwnerError):
            second._lock_identity()

    def proc(self, pid, ppid, ticks, state="S", pgid=None):
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
        (proc / "sys/kernel/random/boot_id").write_text("fixture-boot\n")
        folder = proc / str(pid)
        folder.mkdir(exist_ok=True)
        (folder / "ns").mkdir(exist_ok=True)
        if not os.path.lexists(folder / "ns/pid"):
            (folder / "ns/pid").symlink_to("pid:[fixture]")
        fields = [state, str(ppid), str(pid if pgid is None else pgid), str(pid)] + ["0"] * 16
        fields[19] = str(ticks)
        (folder / "stat").write_text(str(pid) + " (name ) with spaces)) " + " ".join(fields))
        (folder / "status").write_text("Uid:\t" + "\t".join([str(os.getuid())] * 4) + "\nGid:\t" + "\t".join([str(os.getgid())] * 4) + "\n")
        (folder / "cmdline").write_bytes(b"fixture\0child\0")
        return proc

    def test_linux_stat_parentheses_zombie_and_start_tick_binding(self):
        proc = self.proc(100, 1, 77, "Z")
        value = owner.read_process(100, proc)
        self.assertEqual(value.start_ticks, 77)
        self.assertFalse(value.running)
        self.assertIsNone(owner.read_process(101, proc))
        self.assertFalse(value.same_instance(replace(value, start_ticks=78)))
        self.assertTrue(value.same_instance(replace(value, state="S", command_sha256="c" * 64)))

    def test_authenticated_descendant_cross_session_and_reused_parent_refusal(self):
        proc = self.proc(100, 1, 10)
        self.proc(101, 100, 11)
        self.proc(102, 101, 12)
        self.proc(103, 999, 13)
        root = owner.read_process(100, proc)
        found = owner.descendants((root,), proc)
        self.assertEqual([row.pid for row in found], [101, 102])
        self.assertNotEqual(found[0].sid, root.sid)
        self.assertEqual(owner.descendants((replace(root, start_ticks=9),), proc), [])

    def test_signaling_rejects_stale_pid_and_zombie_and_records_owned_action(self):
        child = identity(12345)
        with patch.object(owner.os, "kill") as kill:
            for observed in (None, replace(child, start_ticks=11), replace(child, state="Z")):
                self.assertIsNone(owner.signal_authenticated(child, signal.SIGTERM, lambda pid: observed))
            kill.assert_not_called()
            pidfd = os.open(os.devnull, os.O_RDONLY)
            with patch.object(owner.os, "pidfd_open", return_value=pidfd, create=True) as opened, patch.object(owner.signal, "pidfd_send_signal", create=True) as send:
                receipt = owner.signal_authenticated(child, signal.SIGTERM, lambda pid: child)
                opened.assert_called_once_with(child.pid, 0)
                send.assert_called_once_with(pidfd, signal.SIGTERM, None, 0)
            kill.assert_not_called()
            self.assertEqual(receipt["start_ticks"], child.start_ticks)
            pidfd = os.open(os.devnull, os.O_RDONLY)
            reads = iter([child, replace(child, start_ticks=11)])
            with patch.object(owner.os, "pidfd_open", return_value=pidfd, create=True), patch.object(owner.signal, "pidfd_send_signal", create=True) as send:
                self.assertIsNone(owner.signal_authenticated(child, signal.SIGTERM, lambda pid: next(reads)))
                send.assert_not_called()

    def test_completed_zombie_is_polled_and_reaped_before_classification(self):
        child = identity(12345)
        class ExitedChild:
            pid = 12345
            returncode = None
            polls = 0
            def poll(self):
                self.polls += 1
                self.returncode = 0
                return 0
        process = ExitedChild()
        retained_gate = []
        def spawn(*args, **kwargs):
            # An explicit stub child retains the pipe endpoint just as Popen's
            # pass_fds would; otherwise the parent GO write fails before polling.
            retained_gate.append(os.dup(kwargs["pass_fds"][0]))
            return process
        reads = []
        observed_zombies = []
        def read(pid):
            reads.append(pid)
            if len(reads) == 1:
                return child
            return None if process.returncode is not None else replace(child, state="Z")
        def scan(known):
            if known and process.returncode is None:
                observed_zombies.append(read(child.pid).state)
            return []
        now = time.monotonic()
        with patch.object(owner.subprocess, "Popen", side_effect=spawn), patch.object(owner.os, "kill") as kill:
            result = owner.supervise(self.command, script_path=self.script, script_sha256=self.script_sha,
                deadlines=owner.Deadlines(now + 1, now + 2, now + 3, now + 4), on_child=lambda row: None,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, identity_reader=read,
                descendant_reader=scan)
        for fd in retained_gate:
            os.close(fd)
        self.assertGreaterEqual(process.polls, 1)
        self.assertTrue(result.child_reaped)
        self.assertEqual(result.exit_code, 0)
        self.assertIsNone(result.failure)
        self.assertTrue(result.process_cleanup_observed)
        self.assertEqual(observed_zombies, ["Z"])
        kill.assert_not_called()

    @staticmethod
    def portable_cpu_identity(pid):
        """Explicit CPU-only /proc substitute for the real child tests on macOS."""
        result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "ppid=,stat="],
            capture_output=True, text=True, timeout=1, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parent, state = result.stdout.split()
        try:
            return identity(pid, int(parent), ticks=pid, state=state[0], pgid=os.getpgid(pid), sid=os.getsid(pid))
        except ProcessLookupError:
            return None

    @staticmethod
    def portable_cpu_signal(value, number, reader):
        """Declared macOS-only CPU fixture boundary, not the live Linux signaler."""
        current = reader(value.pid)
        if current is not None and value.same_instance(current) and current.running:
            try:
                os.kill(value.pid, number)
            except ProcessLookupError:
                return None
            return {"pid": value.pid, "start_ticks": value.start_ticks,
                "signal": int(number), "observed_monotonic": time.monotonic()}
        return None

    def run_child(self, source, *, fail_callback=False, resource=False, bad_descendants=False):
        self.script.write_text(source)
        digest = hashlib.sha256(self.script.read_bytes()).hexdigest()
        gated = []
        marker = self.work / "marker"
        def on_child(value):
            self.assertFalse(marker.exists())
            gated.append(value)
            if fail_callback:
                raise owner.OwnerError("fixture durable write failed")
        def resources():
            if resource:
                raise owner.OwnerError("fixture resource overflow")
        def scan(known):
            if bad_descendants:
                raise owner.OwnerError("fixture descendant inventory unavailable")
            return []
        start = time.monotonic()
        with (self.work / "stdout").open("wb") as stdout, (self.work / "stderr").open("wb") as stderr:
            result = owner.supervise(self.command, script_path=self.script, script_sha256=digest,
                deadlines=owner.Deadlines(start + .35, start + .65, start + .95, start + 1.25),
                on_child=on_child, stdout=stdout, stderr=stderr,
                identity_reader=self.portable_cpu_identity, descendant_reader=scan, resource_check=resources,
                signal_process=self.portable_cpu_signal)
        self.assertIsNone(self.portable_cpu_identity(result.root_process.pid))
        self.assertLess(time.monotonic() - start, 1.7)
        return result, gated, marker

    def test_actual_gated_child_success_and_nonzero_poll_reap(self):
        result, gated, marker = self.run_child("from pathlib import Path\nPath(__file__).with_name('marker').write_text('ran')\n")
        self.assertEqual(len(gated), 1)
        self.assertTrue(marker.exists())
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.process_cleanup_observed)
        self.assertFalse(result.escalation_receipts)
        marker.unlink()
        result, _, _ = self.run_child("raise SystemExit(7)\n")
        self.assertEqual(result.exit_code, 7)
        self.assertIsNotNone(result.failure)
        self.assertTrue(result.process_cleanup_observed)

    def test_actual_child_timeout_resource_signal_and_gate_write_failure_cleanup(self):
        for options in ({}, {"resource": True}, {"fail_callback": True}):
            with self.subTest(options=options):
                result, _, marker = self.run_child("import time\ntime.sleep(4)\n", **options)
                self.assertIsNotNone(result.failure)
                self.assertTrue(result.child_reaped)
                self.assertFalse(result.survivors)
                self.assertFalse(marker.exists())
                if not options.get("fail_callback"):
                    self.assertTrue(result.escalation_receipts)
        previous = signal.getsignal(signal.SIGTERM)
        result, _, _ = self.run_child("import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(4)\n")
        self.assertEqual(result.owner_signal, signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertTrue(result.process_cleanup_observed)

    def test_descendant_cleanup_failure_retains_error_not_release(self):
        result, _, _ = self.run_child("print('partial',flush=True)\n", bad_descendants=True)
        self.assertTrue(result.cleanup_errors)
        self.assertFalse(result.process_cleanup_observed)
        self.assertTrue((self.work / "stdout").exists())

    def test_deadlines_and_script_source_refuse_before_launch(self):
        now = time.monotonic()
        for deadline in (owner.Deadlines(now - 1, now + 1, now + 2, now + 3),
                         owner.Deadlines(now + 1, now + 1, now + 2, now + 3),
                         owner.Deadlines(now + 1, float("nan"), now + 2, now + 3)):
            with self.assertRaises(owner.OwnerError):
                deadline.validate(now)
        with self.assertRaises(owner.OwnerError):
            owner.checked_script(self.script, "f" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
