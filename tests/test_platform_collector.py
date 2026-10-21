"""CPU-only collector composition: declared root, filesystem and command stubs.

No root execution, kubectl, fuser, device discovery or native library is invoked.
"""
import copy
import hashlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collect_benchmark_platform", ROOT / "tools/collect_benchmark_platform.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
BOOT = "12345678-1234-1234-1234-123456789abc"


def metadata():
    return {"hostname": collector.TARGET, "boot_id": BOOT, "kernel_release": "fixture-kernel",
        "operator_euid": 0, "developer_uid": 1000, "developer_gid": 1000, "swap_active": False,
        "available_memory_bytes": 64 * 1024**3, "free_disk_bytes": 128 * 1024**3,
        "canonical_lock": {"uid": 1000, "gid": 1000, "mode": 0o600, "links": 1,
            "device": 1, "inode": 2, "unresolved_record_present": False, "lock_acquired": False},
        "devices": [{"render": "/dev/dri/" + render, "bdf": bdf, "unique_id_raw": unique,
            "retained_sysfs_binding_matches": True, "retained_hip_uuid": collector.DEVICES[bdf],
            "fresh_native_uuid_observed": False} for render, (bdf, unique) in zip(
                ("renderD128", "renderD129"), collector.RETAINED_SYSFS.items())]}


def outputs():
    return {"pods": b"BEGIN-POD-INVENTORY\tPodList\nRunning\t/,\t\t\t\nEND-POD-INVENTORY\n",
        "node": (collector.TARGET + "\t87654321-4321-4321-4321-cba987654321\tfixture-kernel\t2\t" + BOOT +
            "\nReady\tTrue\nMemoryPressure\tFalse\nDiskPressure\tFalse\nPIDPressure\tFalse\n").encode(),
        "policy": (collector.POLICY_UID + "\n" + collector.POLICY + "\n1\n1\tbaseline\n0\nIngress,Egress,\n0\n0\n").encode(),
        "holders": b""}


def visibility_inputs():
    return {"/proc/self/uid_map": b"         0          0 4294967295\n",
        "/proc/self/gid_map": b"         0          0 4294967295\n",
        "/proc/self/status": b"Name:\tprivate-fixture-not-for-report\nCapEff:\t0000000000080002\n",
        "/proc/self/ns/pid": "pid:[4026531836]", "/proc/1/ns/pid": "pid:[4026531836]",
        "/proc/self/ns/user": "user:[4026531837]", "/proc/1/ns/user": "user:[4026531837]"}


class CollectorTests(unittest.TestCase):
    def main_case(self, *, local=None, payloads=None, command_fail=None, stderr=None, ending_boot=BOOT, alarm=False, visibility=None):
        local, payloads = copy.deepcopy(local or metadata()), copy.deepcopy(payloads or outputs())
        seen = []
        proc = visibility_inputs()
        proc.update(visibility or {})
        read_actual, link_actual = collector.bounded_read, collector.os.readlink
        def observe(path, *args):
            value = proc.get(str(path))
            if isinstance(value, BaseException):
                raise value
            return value if value is not None else read_actual(path, *args)
        def readlink(path):
            value = proc.get(str(path))
            if isinstance(value, BaseException):
                raise value
            return value if value is not None else link_actual(path)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "evidence"
            def output(value):
                self.assertEqual(value, "/var/tmp/fp8-benchmark-metadata.fixture123")
                folder.mkdir(mode=0o700)
                return folder, os.open(folder, os.O_RDONLY)
            def run(name, work_end):
                seen.append(name)
                self.assertGreater(work_end, time.monotonic())
                return {"name": name, "exit_code": 1 if name == "holders" else 0,
                    "failure": command_fail if name == "pods" else None,
                    "source": {"sha256": "f" * 64}, "group_absent": True, "reaped": True}, {
                        "stdout": payloads[name], "stderr": (stderr or {}).get(name, b"")}
            source_stat = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o444, st_nlink=1)
            digest = hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest()
            def read_local():
                if alarm:
                    signal.raise_signal(signal.SIGALRM)
                return local
            with patch.object(collector.os, "geteuid", return_value=0), patch.object(collector, "self_stat", return_value=source_stat), \
                    patch.object(collector, "create_output", side_effect=output), patch.object(collector, "local_metadata", side_effect=read_local), \
                    patch.object(collector, "kubeconfig_metadata", return_value={"contents_read_by_collector": False}), \
                    patch.object(collector, "run_command", side_effect=run), patch.object(collector, "boot_id", return_value=ending_boot), \
                    patch.object(collector, "bounded_read", side_effect=observe), patch.object(collector.os, "readlink", side_effect=readlink):
                code = collector.main(["--collect-owner-metadata", "--expected-self-sha256", digest,
                    "--output", "/var/tmp/fp8-benchmark-metadata.fixture123"])
            report = json.loads((folder / "REPORT.json").read_bytes())
            self.assertFalse(report["launch_authorized"])
            self.assertFalse(report["qualification"])
            self.assertFalse(report["release_authority"])
            self.assertTrue(report["not_current_authority"])
            self.assertEqual(stat.S_IMODE((folder / "REPORT.json").stat().st_mode), 0o600)
            self.assertNotIn("private-fixture-not-for-report", json.dumps(report))
            self.assertLess(report["collection_span_seconds"], 60)
            return code, report, seen

    def test_actual_main_complete_metadata_never_grants_authority(self):
        code, report, seen = self.main_case()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "OWNER_METADATA_COLLECTED")
        self.assertEqual(seen, ["pods", "node", "policy", "holders"])
        self.assertFalse(report["fresh_native_uuid_observed"])
        self.assertFalse(report["local"]["canonical_lock"]["lock_acquired"])
        self.assertTrue(report["root_fuser_prerequisites"]["passed"])
        self.assertFalse(report["root_fuser_prerequisites"]["initial_host_attested"])
        self.assertEqual(report["observations"]["holders"]["visibility"], "native-host-root-fuser-prerequisites-checked")

    def test_capless_mapped_denied_or_different_namespace_root_refuses(self):
        cases = ({"/proc/self/status": b"CapEff:\t0000000000000000\n"},
            {"/proc/self/status": b"CapEff:\t0000000000080000\n"},
            {"/proc/self/status": b"CapEff:\t0000000000000002\n"},
            {"/proc/self/uid_map": b"0 1000 1\n"}, {"/proc/self/gid_map": b"0 0 65536\n"},
            {"/proc/self/uid_map": FileNotFoundError()}, {"/proc/self/status": PermissionError()},
            {"/proc/self/ns/pid": "pid:[123]"}, {"/proc/self/ns/user": "user:[123]"},
            {"/proc/1/ns/pid": PermissionError()}, {"/proc/self/ns/user": "unbound"})
        for visibility in cases:
            with self.subTest(visibility=visibility):
                code, report, seen = self.main_case(visibility=visibility)
                self.assertEqual(code, 1)
                self.assertEqual(seen, [])
                self.assertEqual(report["status"], "INCOMPLETE")
                self.assertFalse(report["root_fuser_prerequisites"]["passed"])
                self.assertIn("NATIVE_ROOT_FUSER_VISIBILITY_PREREQUISITES_FAILED", report["failures"])

    def test_read_search_capability_alternative_keeps_native_host_limit(self):
        code, report, _ = self.main_case(visibility={"/proc/self/status": b"CapEff:\t0000000000080004\n"})
        self.assertEqual(code, 0)
        observed = report["root_fuser_prerequisites"]
        self.assertTrue(observed["cap_sys_ptrace"])
        self.assertTrue(observed["cap_dac_override_or_read_search"])
        self.assertTrue(observed["native_ssh_host_contract_still_required"])
        self.assertFalse(observed["initial_host_attested"])

    def test_default_cli_hash_uid_and_unsafe_source_refusal(self):
        with self.assertRaises(SystemExit):
            collector.main([])
        with self.assertRaises(collector.CollectionError):
            collector.main(["--collect-owner-metadata", "--expected-self-sha256", "0" * 64,
                "--output", "/var/tmp/fp8-benchmark-metadata.fixture123"])
        digest = hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest()
        with patch.object(collector.os, "geteuid", return_value=1000), self.assertRaises(collector.CollectionError):
            collector.main(["--collect-owner-metadata", "--expected-self-sha256", digest,
                "--output", "/var/tmp/fp8-benchmark-metadata.fixture123"])

    def test_literal_private_output_path_guard_and_existing_entry_refusal(self):
        for value in ("/tmp/fp8-benchmark-metadata.fixture123", "/var/tmp/arbitrary", "relative", "/var/tmp/../tmp/fp8-benchmark-metadata.fixture123"):
            with self.assertRaises(collector.CollectionError):
                collector.create_output(value)
        fake = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
        with patch.object(collector.Path, "lstat", return_value=fake), patch.object(collector.os, "open", return_value=123), \
                patch.object(collector.os, "close"), patch.object(collector.os, "mkdir", side_effect=FileExistsError):
            with self.assertRaises(FileExistsError):
                collector.create_output("/var/tmp/fp8-benchmark-metadata.fixture123")

    def test_busy_allocation_swap_unresolved_and_missing_device_refuse(self):
        cases = []
        for key, value in (("swap_active", True), ("available_memory_bytes", 1), ("free_disk_bytes", 1)):
            local = metadata()
            local[key] = value
            cases.append((local, None))
        local = metadata()
        local["canonical_lock"]["unresolved_record_present"] = True
        cases.append((local, None))
        local = metadata()
        local["devices"] = local["devices"][:1]
        cases.append((local, None))
        payload = outputs()
        payload["pods"] = b"BEGIN-POD-INVENTORY\tList\nRunning\t/2,\t\t\t\nEND-POD-INVENTORY\n"
        cases.append((None, payload))
        payload = outputs()
        payload["holders"] = b"123 456\n"
        cases.append((None, payload))
        for local, payload in cases:
            with self.subTest(local=local, payload=payload):
                code, report, _ = self.main_case(local=local, payloads=payload)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "INCOMPLETE")

    def test_permission_failure_command_timeout_and_missing_metadata_refuse(self):
        for options in ({"stderr": {"holders": b"permission denied secret-fixture"}},
                        {"command_fail": "COMMAND_TIMEOUT"},
                        {"payloads": {**outputs(), "pods": b""}},
                        {"payloads": {**outputs(), "node": b""}},
                        {"payloads": {**outputs(), "policy": b""}}):
            code, report, _ = self.main_case(**options)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertNotIn("secret-fixture", json.dumps(report))

    def test_pod_framing_empty_list_limits_init_overhead_and_dra(self):
        def sanitize(text):
            return collector.sanitize("pods", {"stdout": text.encode(), "stderr": b""})
        empty = "BEGIN-POD-INVENTORY\tList\nEND-POD-INVENTORY\n"
        self.assertEqual(sanitize(empty)["conservative_gpu_requests"], 0)
        text = "BEGIN-POD-INVENTORY\tPodList\nRunning\t/2,\t1/1,\t\t1\nEND-POD-INVENTORY\n"
        self.assertEqual(sanitize(text)["conservative_gpu_requests"], 4)
        for bad in ("", "BEGIN-POD-INVENTORY\tList\n", empty.replace("List", "SecretList"),
                    text.replace("Running", "invented"), text.replace("\t\t1", "\tx\t1"),
                    text.replace("/2,", "/2.0,")):
            with self.assertRaises((collector.CollectionError, ValueError)):
                sanitize(bad)

    def test_boot_change_or_stale_node_boot_refuses(self):
        code, report, _ = self.main_case(ending_boot="new-boot")
        self.assertEqual(code, 1)
        self.assertIn("BOOT_CHANGED_OR_MISSING", report["failures"])
        payload = outputs()
        payload["node"] = payload["node"].replace(BOOT.encode(), b"stale-node-boot")
        self.assertEqual(self.main_case(payloads=payload)[0], 1)

    def test_real_work_watchdog_propagates_and_preserves_partial_report(self):
        previous = signal.getsignal(signal.SIGALRM)
        code, report, seen = self.main_case(alarm=True)
        self.assertEqual(code, 1)
        self.assertEqual(seen, [])
        self.assertTrue(report["work_deadline_exceeded"])
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_fixed_queries_do_not_request_secret_bearing_objects(self):
        self.assertEqual(set(collector.COMMANDS), {"pods", "node", "policy", "holders"})
        for argv in collector.COMMANDS.values():
            self.assertTrue(argv[0].startswith("/usr/bin/"))
            self.assertNotIn("sudo", argv)
            self.assertNotIn("json", argv)
        pod = collector.PODS_OUTPUT
        self.assertNotIn("metadata.name", pod)
        self.assertNotIn(".env", pod)
        self.assertNotIn(".command", pod)
        self.assertIn("resourceClaims", pod)
        self.assertIn("BEGIN-POD-INVENTORY", pod)

    def test_actual_bounded_command_timeout_overflow_partial_and_cleanup(self):
        class PipeChild:
            def __init__(self, data, exited=True):
                self.pid, self.returncode, self.exited, self.descendant = 987654, None, exited, True
                self.stdout = tempfile.TemporaryFile()
                self.stderr = tempfile.TemporaryFile()
                self.stdout.write(data)
                self.stdout.seek(0)
            def wait(self, timeout):
                if self.descendant:
                    raise AssertionError("direct child was reaped before surviving descendant cleanup")
                self.returncode = 0
                return 0
        # selectors cannot monitor regular files on all hosts; explicit selector
        # fixture preserves production capture/overflow and finally/reap bodies.
        class Selector:
            def __init__(self): self.keys = []
            def register(self, file, event, kind): self.keys.append(SimpleNamespace(fileobj=file, data=kind))
            def unregister(self, file): self.keys = [key for key in self.keys if key.fileobj != file]
            def get_map(self): return self.keys
            def select(self, timeout): return [(key, 1) for key in self.keys]
            def close(self): pass
        for raw, exited in ((b"partial\n", True), (b"x" * (collector.LIMIT + 1), True), (b"", False)):
            process = PipeChild(raw, exited)
            killed = []
            def kill(pid, number):
                if not number:
                    raise ProcessLookupError()
                self.assertIsNone(process.returncode)
                killed.append(number)
                if number == signal.SIGKILL:
                    process.descendant = False
            clock = [0]
            def tick():
                clock[0] += .1
                return clock[0]
            with patch.object(collector, "source_identity", return_value={"sha256": "a" * 64}), \
                    patch.object(collector.subprocess, "Popen", return_value=process), \
                    patch.object(collector.selectors, "DefaultSelector", Selector), \
                    patch.object(collector, "child_exited_unreaped", return_value=exited), \
                    patch.object(collector.os, "getpgid", return_value=process.pid), \
                    patch.object(collector.os, "killpg", side_effect=kill), \
                    patch.object(collector.time, "monotonic", side_effect=tick), patch.object(collector.time, "sleep"):
                receipt, capture = collector.run_command("pods", 100)
            self.assertEqual(receipt["reaped"], True)
            self.assertTrue(receipt["group_absent"])
            self.assertEqual(killed, [signal.SIGTERM, signal.SIGKILL])
            if len(raw) > collector.LIMIT:
                self.assertEqual(receipt["failure"], "OUTPUT_BOUND_EXCEEDED")
            elif not exited:
                self.assertEqual(receipt["failure"], "COMMAND_TIMEOUT")
            else:
                self.assertIsNone(receipt["failure"])
                self.assertEqual(capture["stdout"], raw)

    def test_kubeconfig_metadata_rejects_user_control_without_reading_contents(self):
        unsafe = SimpleNamespace(st_uid=1000, st_mode=stat.S_IFREG | 0o600)
        with patch.object(collector.Path, "lstat", return_value=unsafe), patch.object(collector, "bounded_read") as read:
            with self.assertRaises(collector.CollectionError):
                collector.kubeconfig_metadata()
            read.assert_not_called()

    def test_source_identity_ignores_atime_but_not_content_identity(self):
        info = Path(collector.__file__).stat()
        fields = {name: getattr(info, name) for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns", "st_atime_ns")}
        changed = dict(fields, st_atime_ns=fields["st_atime_ns"] + 100)
        self.assertEqual(collector.stable_stat(SimpleNamespace(**fields)), collector.stable_stat(SimpleNamespace(**changed)))
        changed["st_mtime_ns"] += 100
        self.assertNotEqual(collector.stable_stat(SimpleNamespace(**fields)), collector.stable_stat(SimpleNamespace(**changed)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
