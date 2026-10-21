"""CPU contract fixtures. Root, procfs and platform observations are substitutions."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_release_observer", ROOT / "tools/benchmark_release_observer.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)
BOOT = "12345678-1234-1234-1234-123456789abc"
SELF_SHA = hashlib.sha256(Path(observer.__file__).read_bytes()).hexdigest()


def process(pid, namespace="pid:[100]", ticks=10):
    return {"boot_id": BOOT, "pid": pid, "ppid": 1, "pgid": pid, "sid": pid, "start_ticks": ticks,
        "uid": 1000, "gid": 1000, "pid_namespace": namespace, "command_sha256": "a" * 64, "state": "S"}


def request(mode="release"):
    now = time.monotonic()
    admission = mode in ("admission", "locked-admission")
    return {"schema": 1, "mode": mode, "root": "/home/developer/fp8-kv-benchmark.fixture123",
        "session_sha256": "b" * 64, "boot_id": BOOT, "lock_device": 1, "lock_inode": 2,
        "arm": "bf16" if mode == "release" else None,
        "root_process": None if admission else process(100),
        "observed_processes": [] if admission else [process(100), process(101, "pid:[101]", 11)],
        "journal_sha256": None if mode == "admission" else "c" * 64,
        "issued_monotonic": now, "deadline_monotonic": now + 75, "native_ssh_host_contract": True}


class ObserverTests(unittest.TestCase):
    def platform(self, *, visibility=True, command_failure=None):
        # Functions are declared platform boundaries; no native commands run.
        devices = {"0000:c3:00.0": "39373864-6465-3038-6163-303837666533",
            "0000:23:00.0": "34663466-6561-3362-3832-346261366439"}
        local = {"hostname": "threadripper-ai", "boot_id": BOOT, "swap_active": False,
            "developer_uid": 1000, "developer_gid": 1000, "available_memory_bytes": 64 * 1024**3,
            "free_disk_bytes": 128 * 1024**3, "kernel_release": "fixture-kernel",
            "canonical_lock": {"device": 1, "inode": 2, "uid": 1000, "gid": 1000, "mode": 0o600, "links": 1},
            "devices": [{"bdf": bdf, "sysfs_bdf": bdf, "render": "/dev/dri/renderD" + str(128 + index),
                "retained_sysfs_binding_matches": True, "retained_hip_uuid": value,
                "fresh_native_uuid_observed": False} for index, (bdf, value) in enumerate(devices.items())]}
        facts = {"node": {"name_matches": True, "ready": True, "pressure_free": True,
                "kernel_release": "fixture-kernel", "boot_id": BOOT, "allocatable_gpu": "2"},
            "policy": {"matches_retained_deny_contract": True}, "holders": {"pids": []},
            "pods": {"conservative_gpu_requests": 0}}
        calls = []
        def run(name, end):
            calls.append(name)
            return {"failure": command_failure, "reaped": True, "group_absent": True,
                "exit_code": 1 if name == "holders" else 0, "source": {"sha256": "d" * 64}}, {"stdout": b"", "stderr": b""}
        return NS(TARGET="threadripper-ai", DEVICES=devices, COMMANDS=dict.fromkeys(("pods", "node", "policy", "holders")),
            root_fuser_prerequisites=lambda: {"passed": visibility, "initial_host_attested": False},
            local_metadata=lambda: copy.deepcopy(local), kubeconfig_metadata=lambda: {"contents_read_by_collector": False},
            run_command=run, sanitize=lambda name, raw: copy.deepcopy(facts[name]), boot_id=lambda: BOOT), calls

    def collect(self, req=None, *, visibility=True, command_failure=None, proc=None):
        req = copy.deepcopy(req or request())
        platform, calls = self.platform(visibility=visibility, command_failure=command_failure)
        actual_process_release = observer.process_release
        def release(value, deadline):
            return actual_process_release(value, deadline, proc)
        # Source permission/parent checks are a declared root-owned-filesystem
        # substitution. Actual source digest is still computed and compared.
        def source(path, expected):
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            observer.require(actual == expected, "CPU fixture source substitution")
        with tempfile.TemporaryDirectory() as directory:
            proc = proc or directory
            with patch.object(observer, "source", side_effect=source), patch.object(observer.os, "geteuid", return_value=0), \
                    patch.object(observer, "load_collector", return_value=platform), \
                    patch.object(observer, "check_journal", return_value={"authenticated": True, "lock_acquired": False}), \
                    patch.object(observer, "process_release", side_effect=release):
                snapshot = observer.collect_snapshot(req, expected_observer_sha256=SELF_SHA, deadline_monotonic=req["deadline_monotonic"])
        self.assertFalse(snapshot["qualification"])
        self.assertFalse(snapshot["launch_authorized"])
        return snapshot, calls

    def test_release_and_independent_complete_snapshots_satisfy_actual_reader(self):
        adapter_spec = importlib.util.spec_from_file_location("observer_test_adapter", ROOT / "tools/benchmark_owner_adapter.py")
        adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(adapter)
        for mode in ("release", "independent"):
            req = request(mode)
            snapshot, calls = self.collect(req)
            self.assertEqual(snapshot["status"], "OBSERVATION_COMPLETE")
            self.assertEqual(calls, ["pods", "node", "policy", "holders"])
            result = NS(root_process=NS(**req["root_process"]), observed_processes=[NS(**row) for row in req["observed_processes"]])
            adapter._platform_facts(snapshot, [result])
            self.assertFalse(snapshot["fresh_native_uuid_observed"])
            self.assertEqual(snapshot["collector_source_sha256"], SELF_SHA)

    def test_admission_remains_metadata_only_and_no_lock_observation_claim(self):
        snapshot, _ = self.collect(request("admission"))
        self.assertEqual(snapshot["status"], "OBSERVATION_COMPLETE")
        self.assertEqual(snapshot["scope"], "host-benchmark-admission-v1")
        self.assertEqual(snapshot["checked_processes"], [])
        self.assertFalse(snapshot["journal"]["lock_acquired"])

    def test_exact_request_shape_root_freshness_and_duplicate_identity_refusals(self):
        variations = []
        for key, value in (("root", "/etc"), ("schema", True), ("session_sha256", "bad"),
                           ("native_ssh_host_contract", False), ("deadline_monotonic", 1), ("arm", "other")):
            req = request()
            req[key] = value
            variations.append(req)
        req = request()
        req["observed_processes"].append(req["observed_processes"][0])
        variations.append(req)
        req = request()
        del req["journal_sha256"]
        variations.append(req)
        req = request()
        req["observed_processes"][1]["uid"] = 0
        variations.append(req)
        for req in variations:
            with self.assertRaises(observer.ObservationError):
                observer.validate_request(req, req["deadline_monotonic"])

    def test_visibility_and_command_failure_cannot_become_release(self):
        for options in ({"visibility": False}, {"command_failure": "COMMAND_TIMEOUT"}):
            snapshot, calls = self.collect(**options)
            self.assertEqual(snapshot["status"], "INCOMPLETE")
            self.assertTrue(snapshot["failures"])
            self.assertNotIn("holders", snapshot)
            if options.get("visibility") is False:
                self.assertEqual(calls, [])

    def proc(self, root, row):
        folder = Path(root) / str(row["pid"])
        (folder / "ns").mkdir(parents=True)
        (folder / "ns/pid").symlink_to(row["pid_namespace"])
        fields = [row["state"], str(row["ppid"]), str(row["pgid"]), str(row["sid"])] + ["0"] * 16
        fields[19] = str(row["start_ticks"])
        (folder / "stat").write_text(str(row["pid"]) + " (fixture ) comm) " + " ".join(fields))
        (folder / "status").write_text("Name: private-not-retained\nUid: 1000 1000 1000 1000\nGid: 1000 1000 1000 1000\n")

    def test_actual_proc_named_identity_and_namespace_survivor_refusal(self):
        for substitute in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                row = process(101, "pid:[101]", 12 if substitute else 11)
                self.proc(directory, row)
                req = request()
                if substitute:
                    with self.assertRaises(observer.ObservationError):
                        observer.process_release(req, time.monotonic() + 1, directory)
                else:
                    snapshot, _ = self.collect(req, proc=directory)
                    self.assertEqual(snapshot["status"], "INCOMPLETE")
                    self.assertEqual(snapshot["pid_namespace_members"], {"pid:[101]": [101]})
                    self.assertEqual(snapshot["authenticated_processes_present"], [[BOOT, 101, 11]])
                    self.assertNotIn("private-not-retained", json.dumps(snapshot))

    def journal(self, req):
        return {"version": 2, "kind": "host-process-benchmark", "token": "a" * 32,
            "root": req["root"], "session_sha256": req["session_sha256"], "boot_id": BOOT,
            "uid": 1000, "gid": 1000, "owner_pid": 500, "owner": process(500),
            "lock_device": 1, "lock_inode": 2, "state": "reserved" if req["mode"] == "locked-admission" else "running" if req["mode"] == "release" else "pending-independent",
            "arms": [] if req["mode"] == "locked-admission" else [{"arm": "bf16", "child": req["root_process"]}]}

    def journal_case(self, req, record, *, present=True):
        raw = observer.canonical_json(record)
        req["journal_sha256"] = hashlib.sha256(raw).hexdigest()
        info = NS(st_mode=stat.S_IFDIR | 0o700, st_uid=1000, st_gid=1000)
        actual_resolve = observer.Path.resolve
        def resolve(path, *args, **kwargs):
            return path if str(path) == req["root"] else actual_resolve(path, *args, **kwargs)
        local = self.platform()[0].local_metadata()
        with patch.object(observer.Path, "resolve", resolve), patch.object(observer.Path, "lstat", return_value=info), \
                patch.object(observer, "read_regular", return_value=raw), patch.object(observer.os.path, "lexists", return_value=present), \
                patch.object(observer, "process_metadata", return_value={key: value for key, value in process(500).items() if key not in ("boot_id", "command_sha256")}):
            return observer.check_journal(req, local)

    def test_actual_journal_binding_foreign_state_and_admission_refusals(self):
        for mode in ("release", "independent"):
            req = request(mode)
            self.assertTrue(self.journal_case(req, self.journal(req))["authenticated"])
        for key, value in (("root", "/foreign"), ("session_sha256", "d" * 64), ("boot_id", "other"),
                           ("uid", 0), ("state", "reserved"), ("lock_inode", 3)):
            req = request()
            record = self.journal(req)
            record[key] = value
            with self.assertRaises(observer.ObservationError):
                self.journal_case(req, record)
        req = request("admission")
        with self.assertRaises(observer.ObservationError):
            self.journal_case(req, {}, present=True)
        self.assertFalse(self.journal_case(req, {}, present=False)["present"])

    def test_provisional_then_locked_admission_authenticates_only_own_reservation(self):
        provisional = request("admission")
        self.assertFalse(self.journal_case(provisional, {}, present=False)["present"])
        locked = request("locked-admission")
        journal = self.journal(locked)
        bound = self.journal_case(locked, journal)
        self.assertTrue(bound["authenticated"])
        self.assertEqual(bound["state"], "reserved")
        snapshot, _ = self.collect(locked)
        self.assertEqual(snapshot["scope"], "host-benchmark-locked-admission-v1")
        self.assertEqual(snapshot["status"], "OBSERVATION_COMPLETE")
        self.assertEqual(snapshot["checked_processes"], [])
        for key, value in (("root", "/foreign"), ("session_sha256", "d" * 64),
                           ("state", "running"), ("arms", [{"arm": "bf16"}])):
            changed = copy.deepcopy(journal)
            changed[key] = value
            with self.assertRaises(observer.ObservationError):
                self.journal_case(request("locked-admission"), changed)
        wrong = request("locked-admission")
        wrong["journal_sha256"] = None
        with self.assertRaises(observer.ObservationError):
            observer.validate_request(wrong, wrong["deadline_monotonic"])

    def test_bounded_regular_file_source_hash_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "fixture"
            file.write_bytes(b"{}"); file.chmod(0o600)
            self.assertEqual(observer.read_regular(file, 2, os.getuid(), os.getgid(), 0o600), b"{}")
            with self.assertRaises(observer.ObservationError):
                observer.read_regular(file, 1, os.getuid())
            alias = Path(directory) / "alias"
            alias.symlink_to(file)
            with self.assertRaises(OSError):
                observer.read_regular(alias, 2, os.getuid())
        with self.assertRaises(observer.ObservationError):
            observer.json_bytes(b'{"duplicate":1,"duplicate":2}')
        with self.assertRaises(observer.ObservationError):
            observer.json_bytes(b'{"value":NaN}')
        self.assertEqual(hashlib.sha256((ROOT / "tools/collect_benchmark_platform.py").read_bytes()).hexdigest(), observer.COLLECTOR_SHA256)

    def test_explicit_cli_and_arbitrary_root_path_refusal(self):
        with self.assertRaises(observer.ObservationError):
            observer.main([])
        with self.assertRaises(observer.ObservationError):
            observer.main(["--collect-owner-snapshot", "--request", "/etc/REQUEST.json", "--output", "/etc/SNAPSHOT.json",
                "--request-sha256", "a" * 64, "--expected-self-sha256", SELF_SHA])

    def test_actual_cli_binds_request_and_exclusive_private_output(self):
        req = request()
        raw = observer.canonical_json(req)
        directory = Path("/var/tmp/fp8-benchmark-observer.fixture123")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            named = target.stat()
            info = NS(st_mode=stat.S_IFDIR | 0o700, st_uid=0, st_dev=named.st_dev, st_ino=named.st_ino)
            actual_resolve, actual_open = observer.Path.resolve, observer.os.open
            def resolve(path, *args, **kwargs):
                return path if path == directory else actual_resolve(path, *args, **kwargs)
            def opened(path, flags, *args, **kwargs):
                return actual_open(target if Path(path) == directory else path, flags, *args, **kwargs)
            snapshot = {"schema": 1, "status": "OBSERVATION_COMPLETE", "qualification": False}
            with patch.object(observer.Path, "resolve", resolve), patch.object(observer.Path, "lstat", return_value=info), \
                    patch.object(observer, "read_regular", return_value=raw), patch.object(observer, "collect_snapshot", return_value=snapshot) as collect, \
                    patch.object(observer.os, "open", side_effect=opened):
                code = observer.main(["--collect-owner-snapshot", "--request", str(directory / "REQUEST.json"),
                    "--output", str(directory / "SNAPSHOT.json"), "--request-sha256", hashlib.sha256(raw).hexdigest(),
                    "--expected-self-sha256", SELF_SHA])
                self.assertEqual(code, 0)
                self.assertEqual(collect.call_args.kwargs["expected_observer_sha256"], SELF_SHA)
                with self.assertRaises(FileExistsError):
                    observer.main(["--collect-owner-snapshot", "--request", str(directory / "REQUEST.json"),
                        "--output", str(directory / "SNAPSHOT.json"), "--request-sha256", hashlib.sha256(raw).hexdigest(),
                        "--expected-self-sha256", SELF_SHA])
            output = json.loads((target / "SNAPSHOT.json").read_bytes())
            self.assertEqual(output["request_file_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertFalse(output["qualification"])

    def test_source_hash_and_expired_process_observation_refusal(self):
        with patch.object(observer.Path, "resolve", lambda value: value), patch.object(observer.Path, "stat", return_value=NS(st_uid=0, st_mode=0o444)), \
                patch.object(observer, "read_regular", return_value=b"declared-source"):
            with self.assertRaises(observer.ObservationError):
                observer.source(Path("/root/observer.py"), "0" * 64)
            observer.source(Path("/root/observer.py"), hashlib.sha256(b"declared-source").hexdigest())
        with self.assertRaises(observer.ObservationError):
            observer.process_release(request(), time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
