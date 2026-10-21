"""Inert controller composition; no root, remote, runtime or GPU effects.

Synthetic spec/model/native payload pins are explicit CPU file substitutes.
Production fixed pins and disabled execution remain asserted independently.
"""
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = load("controller_fixture", ROOT / "tools/benchmark_physical_controller.py")
release_observer = load("controller_release_contract", ROOT / "tools/benchmark_release_observer.py")
fixtures = load("controller_benchmark_fixtures", ROOT / "tests/test_model_benchmark.py")
BOOT = "12345678-1234-1234-1234-123456789abc"


class LinuxTargetPath(type(Path())):
    """Only translate unavailable Linux target path resolution on macOS."""
    def resolve(self, *args, **kwargs):
        if str(self).startswith("/home/developer/fp8-kv-benchmark."):
            return self
        return super().resolve(*args, **kwargs)


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class LeaseFixture:
    """Explicit ownership-effect substitute; actual lease is tested separately."""
    def __init__(self, package, work):
        self.root, self.session, self.boot = Path(package["target_root"]), package["session_sha256"], BOOT
        self.lock_path = work / "diagnostic.lock"
        self.lock_path.write_bytes(b"")
        self.journal = work / "diagnostic.lock.unresolved.json"
        info = self.lock_path.stat()
        self.record = {"state": "reserved", "arms": [], "lock_device": info.st_dev, "lock_inode": info.st_ino}
        self.fd, self.closed_monotonic = None, None
    def save(self):
        self.journal.write_bytes(raw(self.record))
        info = self.journal.stat()
        self.journal_identity = (info.st_dev, info.st_ino, sha(self.journal.read_bytes()))
    def _verify_journal(self):
        if self.journal_identity[2] != sha(self.journal.read_bytes()):
            raise ValueError("fixture journal changed")
    def acquire(self):
        self.fd = 9
        self.save()
    def begin_arm(self, arm, argv, script_path, script_sha256):
        self.record["state"] = "launch-intent"
        self.record["arms"].append({"arm": arm, "intent_monotonic": time.monotonic(),
                                   "argv": list(argv), "script_sha256": script_sha256})
        self.save()
    def record_child(self, child):
        self.record["state"] = "running"
        self.record["arms"][-1]["child"] = asdict(child)
        self.save()
    def finish_arm(self, result, evidence):
        self.record["state"] = "released"
        self.save()
    def close(self):
        self.fd = None
        self.record["state"] = "pending-independent"
        self.closed_monotonic = time.monotonic()
        self.save()
    def confirm_closed(self, evidence):
        self.record["state"] = "closed"
        self.journal.unlink()


class ControllerTests(unittest.TestCase):
    def test_oci_proof_binds_every_exact_field_and_current_check_is_still_required(self):
        record = controller.retained_verifier.proof_record()
        controller.require_retained_base_binding(record)
        for name, value in record.items():
            changed = dict(record)
            changed[name] = '0' * 64 if isinstance(value, str) else True
            with self.subTest(field=name), self.assertRaises(ValueError):
                controller.require_retained_base_binding(changed)
        with self.assertRaises(ValueError):
            controller.require_retained_base_binding({**record, 'extra': True})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.boundaries = ExitStack()
        self.addCleanup(self.boundaries.close)
        self.boundaries.enter_context(patch.object(controller.adapter, "Path", LinuxTargetPath))
        self.package, self.runtime, self.model, self.retained = [self.base / name
            for name in ("package", "runtime", "model", "retained")]
        for folder in (self.package / "review", self.runtime, self.model, self.retained):
            folder.mkdir(parents=True)
        for name in controller.SOURCE_NAMES:
            (self.package / "review" / name).write_bytes((ROOT / "tools" / name).read_bytes())
        config = b'{"declared":"CPU-only-model-fixture"}'
        model_rows = []
        for name in sorted(["config.json"] + [f"inert-{index:02d}.bin" for index in range(80)]):
            data = config if name == "config.json" else name.encode()
            (self.model / name).write_bytes(data)
            model_rows.append({"name": name, "size": len(data), "sha256": sha(data)})
        model_raw = raw(model_rows)
        (self.package / "review/MODEL-INVENTORY.json").write_bytes(model_raw)
        stack = b'{"scope":"inert-common-stack-fixture"}'
        (self.package / "review/COMMON-STACK-RECEIPT.json").write_bytes(stack)
        native = b"not a native shared library; never load"
        self.binary_name, self.proof_name = "native/inert.so", "proof/build.json"
        (self.runtime / "native").mkdir()
        (self.runtime / self.binary_name).write_bytes(native)
        (self.runtime / "proof").mkdir()
        proof = {"schema": 2, "serving_composition_complete": True,
            "qualification": "build-reviewed-not-serving-qualified", "binary_sha256": sha(native),
            "stack_receipt_sha256": sha(stack)}
        (self.runtime / self.proof_name).write_bytes(raw(proof))
        for member in controller.RUNTIME_REQUIRED:
            path = self.runtime / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# inert CPU source boundary; never import\n")
        spec = fixtures.fixture()
        spec.update(worker_seconds=690, config_path="/mnt/models/config.json",
            package_root="/mnt/composition/image-source/python/sglang",
            stack_receipt="/mnt/review/COMMON-STACK-RECEIPT.json", stack_receipt_sha256=sha(stack),
            build_proof_sha256=sha(raw(proof)))
        spec["server_args"]["model_path"] = "/mnt/models"
        self.spec_raw = raw(spec)
        (self.package / "review/BENCHMARK-SPEC2.json").write_bytes(self.spec_raw)
        target, session = "/home/developer/fp8-kv-benchmark.fixture123", "a" * 64
        entry = target + "/review/model_benchmark.py"
        resource = {"schema": 1, "scope": controller.resources.SCOPE, "root": target,
            "session_sha256": session, "benchmark_spec_sha256": sha(self.spec_raw),
            "gate_path": target + "/review/benchmark_resource_gate.py",
            "gate_sha256": sha((ROOT / "tools/benchmark_resource_gate.py").read_bytes()),
            "entry_path": entry, "entry_sha256": sha((ROOT / "tools/model_benchmark.py").read_bytes()),
            "bwrap_sha256": "b" * 64, "units": {}, "workload_argv": {}, "workload_argv_sha256": {}}
        for arm in ("bf16", "kv4"):
            argv = controller.adapter.bwrap_argv(arm, retained_root=self.retained,
                composition=self.runtime, review=target + "/review", model=self.model,
                output=target + "/arms/" + arm, entry=entry, spec_sha256=sha(self.spec_raw),
                bounded_scratch=True)
            resource["units"][arm] = "spry-fp8-benchmark-" + session[:16] + "-" + arm + ".scope"
            resource["workload_argv"][arm] = list(argv)
            resource["workload_argv_sha256"][arm] = controller.resources.argv_sha256(list(argv))
        (self.package / "review/RESOURCE-MANIFEST.json").write_bytes(raw(resource))
        self.value = {"schema": 1, "scope": "matched-benchmark-controller-candidate-v1",
            "experiment_mode": "model-benchmark-v1", "launch_authorized": False,
            "proposed_complete_seconds": 2400, "target_root": target, "boot_id": BOOT,
            "session_sha256": session, "package_root": str(self.package), "retained_root": str(self.retained),
            "runtime_root": str(self.runtime), "model_root": str(self.model),
            "package_files": self.files(self.package), "runtime_files": self.files(self.runtime),
            "build_proof_file": self.proof_name, "native_binary_file": self.binary_name,
            "model_inventory_file": "review/MODEL-INVENTORY.json", "spec_sha256": sha(self.spec_raw),
            "resource_manifest_sha256": sha(raw(resource))}
        self.manifest = self.base / "MANIFEST.json"
        self.refresh()
        for name, value in (("SPEC_SHA256", sha(self.spec_raw)), ("CONFIG_SHA256", sha(config)),
                            ("MODEL_INVENTORY_SHA256", sha(model_raw)), ("BINARY_SHA256", sha(native))):
            self.boundaries.enter_context(patch.object(controller, name, value))
        self.prepared = controller.validate_manifest(self.manifest, self.pin, deadline=time.monotonic() + 30)
        self.lease = LeaseFixture(self.value, self.base)
        self.observed, self.supervised = [], []

    @staticmethod
    def files(root):
        return {str(path.relative_to(root)): sha(path.read_bytes()) for path in sorted(root.rglob("*")) if path.is_file()}

    def refresh(self):
        self.manifest.write_bytes(raw(self.value))
        self.pin = sha(self.manifest.read_bytes())

    def identity(self, pid, namespace):
        return controller.adapter.owner.ProcessIdentity(BOOT, pid, 1, pid, pid, 10,
            1000, 1000, namespace, "c" * 64, "S")

    def supervisor(self, argv, **kwargs):
        arm = self.lease.record["arms"][-1]["arm"]
        self.supervised.append(arm)
        child = self.identity(1000 + len(self.supervised), "pid:[100]")
        private = self.identity(2000 + len(self.supervised), "pid:[" + str(200 + len(self.supervised)) + "]")
        kwargs["on_child"](child)
        return controller.adapter.owner.SupervisionResult(child, (child, private), 0, None, (), (), True, None)

    def observer(self, request, **kwargs):
        self.assertLessEqual(request["deadline_monotonic"] - request["issued_monotonic"], 75)
        self.assertEqual(kwargs["deadline_monotonic"], request["deadline_monotonic"])
        self.observed.append(request["mode"])
        rows = request["observed_processes"]
        scope = {"locked-admission": "host-benchmark-locked-admission-v1", "release": "host-benchmark-release-v1",
                 "independent": "host-benchmark-independent-release-v1"}[request["mode"]]
        value = {key: request[key] for key in ("root", "session_sha256", "boot_id", "lock_device", "lock_inode")}
        value.update(schema=1, scope=scope, status="OBSERVATION_COMPLETE", failures=[],
            collector_source_sha256=kwargs["expected_observer_sha256"], metadata_source_sha256=controller.METADATA_SHA256,
            observed_monotonic=time.monotonic(), checked_processes=[list(row) for row in sorted({
                (one["boot_id"], one["pid"], one["start_ticks"]) for one in rows})],
            authenticated_processes_present=[], pid_namespace_members={one["pid_namespace"]: [] for one in rows
                if one["pid_namespace"] != "pid:[100]"},
            holders={"visibility": "all-users-platform", "exit_code": 1, "stdout": "", "stderr": ""},
            allocation={"scope": "all-namespaces-active-workloads", "gpu_requests": 0},
            target={"hostname": "threadripper-ai", "uid": 1000, "gid": 1000, "boot_id": BOOT,
                "swap_active": False, "node_ready": True, "pressure_free": True, "expected_policy_unchanged": True},
            devices=[{"bdf": bdf, "uuid": uuid, "render": "/dev/dri/renderD" + str(128 + index), "sysfs_bdf": bdf}
                     for index, (bdf, uuid) in enumerate(controller.adapter.DEVICES.items())],
            journal={"authenticated": True, "state": self.lease.record["state"],
                     "sha256": request["journal_sha256"]})
        if request["mode"] == "release":
            value.update(arm=request["arm"], child_pid=request["root_process"]["pid"],
                         child_start_ticks=request["root_process"]["start_ticks"])
        return value

    def resource_receipt(self, arm):
        frozen = self.prepared["resource_manifest"]
        return {"schema": 1, "scope": controller.resources.SCOPE, "arm": arm,
            "root": self.value["target_root"], "session_sha256": self.value["session_sha256"],
            **{key: frozen[key] for key in ("gate_sha256", "entry_sha256", "bwrap_sha256")},
            "workload_argv_sha256": frozen["workload_argv_sha256"][arm], "actual_cgroup_verified": True,
            "limits": {"memory.max": 38 * 1024**3, "memory.swap.max": 0, "pids.max": 512},
            "cpu_max": [2400000, 100000], "memory_events": {"max": 0, "oom": 0, "oom_kill": 0},
            "pids_events": {"max": 0}, "scratch_quota_verified": False,
            "rlimits": {"core": [0, 0], "file_size": [128 << 20, 128 << 20]},
            "cgroup_path": controller.resources.USER_MANAGER + "/app.slice/" + frozen["units"][arm],
            "initial_usage": {"memory.current": 1000000, "pids.current": 4}}

    def inner(self, arm):
        rows = fixtures.completed_records()
        for row in rows:
            row.update(arm=arm, qualification=False, serving_quality_qualified=False)
        return {"schema": 1, "scope": controller.adapter.readout.bench.SCOPE, "arm": arm,
            "status": "MODEL_BENCHMARK_COMPLETE", "spec_sha256": sha(self.spec_raw), "timed_out": False,
            "exits": [0, 0], "groups_observed_absent": [True, True], "parent_error": None,
            "cleanup_errors": [], "rank_parse_errors": [], "qualification": False,
            "serving_quality_qualified": False, "http_throughput_measured": False,
            "independent_release_required": True, "ranks": rows,
            "samples": controller.adapter.readout.bench.summarize(rows, arm)}

    def invoke(self, **changed):
        args = dict(manifest_path=self.manifest, manifest_sha256=self.pin,
            grant={"schema": 1, "scope": "inert-controller-cpu-composition", "launch_authorized": False,
                "maximum_starts": 0, "proposed_complete_seconds": 2400, "manifest_sha256": self.pin},
            lease=self.lease, observer=self.observer, read_inner_result=self.inner,
            read_resource_receipt=self.resource_receipt, resource_check=lambda: None,
            streams={arm: (None, None) for arm in ("bf16", "kv4")}, supervisor=self.supervisor)
        args.update(changed)
        return controller.run_controller(**args)

    def test_actual_controller_positive_is_inert_and_never_launch_ready(self):
        report = self.invoke()
        self.assertEqual(report["status"], "LOCAL_CONTROLLER_COMPOSITION_PASSED", report)
        self.assertFalse(report["qualification"])
        self.assertFalse(report["physical_ready"])
        self.assertFalse(report["launch_authorized"])
        self.assertTrue(report["remaining_gaps"])
        self.assertEqual(self.observed, ["locked-admission", "release", "release", "independent"])
        self.assertEqual(self.supervised, ["bf16", "kv4"])
        # Real monotonic progression catches callback-overhead budget reopening.
        self.assertLess(report["observer_requests"][0]["issued_monotonic"],
                        report["admission"]["observed_monotonic"])
        self.assertEqual(report["source_preparation_seconds"], 105)

    def test_session_transitions_complete_through_real_release_request_validator(self):
        histories = []
        def supervise(*args, **kwargs):
            original = self.supervisor(*args, **kwargs)
            worker = original.observed_processes[-1]
            before = replace(worker, pgid=original.root_process.pgid, sid=original.root_process.sid)
            history = (original.root_process, before, worker)
            histories.append(history)
            return replace(original, observed_processes=history)
        def observe(request, **kwargs):
            release_observer.validate_request(request, request["deadline_monotonic"])
            self.assertEqual(len({row["pid"] for row in request["observed_processes"]}),
                             len(request["observed_processes"]))
            if request["mode"] == "release":
                legacy = dict(request, observed_processes=[asdict(row) for row in histories[-1]])
                with self.assertRaisesRegex(release_observer.ObservationError, "duplicate"):
                    release_observer.validate_request(legacy, request["deadline_monotonic"])
            return self.observer(request, **kwargs)
        report = self.invoke(supervisor=supervise, observer=observe)
        self.assertEqual(report["status"], "LOCAL_CONTROLLER_COMPOSITION_PASSED", report)
        self.assertFalse(report["qualification"])
        self.assertEqual(self.observed, ["locked-admission", "release", "release", "independent"])
        for arm, history in zip(report["campaign"]["arms"], histories):
            self.assertEqual(list(arm["supervision"]["observed_processes"]), [asdict(row) for row in history])

    def test_release_projection_rejects_pid_reuse_credentials_boot_or_namespace_changes(self):
        root = self.identity(1001, "pid:[100]")
        worker = self.identity(2001, "pid:[201]")
        for changes in ({"start_ticks": 11}, {"boot_id": "different"}, {"uid": 0},
                        {"gid": 0}, {"pid_namespace": "pid:[999]"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "conflicting"):
                controller.release_process_inventory((root, worker, replace(worker, **changes)), root)
        with self.assertRaisesRegex(ValueError, "root missing"):
            controller.release_process_inventory((worker,), root)
        with self.assertRaisesRegex(ValueError, "bound"):
            controller.release_process_inventory((worker,) * 513, root)
        latest = replace(worker, ppid=2, pgid=3001, sid=3001, state="Z", command_sha256="d" * 64)
        rows = controller.release_process_inventory((root, worker, latest), root)
        self.assertEqual(rows, [asdict(root), asdict(latest)])
        # The exact journal root is not replaced by a later metadata snapshot.
        self.assertEqual(controller.release_process_inventory((root, replace(root, state="Z")), root),
                         [asdict(root)])

    def test_missing_changed_extra_linked_sources_and_model_members_refuse(self):
        for target in (self.model / "inert-00.bin", self.runtime / self.binary_name,
                       self.package / "review/model_benchmark.py"):
            saved = target.read_bytes()
            target.write_bytes(saved + b"changed")
            self.assertEqual(self.invoke()["status"], "INCOMPLETE")
            target.write_bytes(saved)
        extra = self.model / "EXTRA"
        extra.write_bytes(b"extra")
        self.assertEqual(self.invoke()["status"], "INCOMPLETE")
        extra.unlink()
        extra.symlink_to(self.model / "config.json")
        self.assertEqual(self.invoke()["status"], "INCOMPLETE")
        self.assertEqual(self.supervised, [])

    def test_missing_resource_report_stops_second_arm_and_retains_failure(self):
        report = self.invoke(read_resource_receipt=lambda arm: {})
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(self.supervised, ["bf16"])
        self.assertTrue(report["campaign"]["failure"])

    def test_missing_inner_result_does_not_complete(self):
        report = self.invoke(read_inner_result=lambda arm: {})
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(self.supervised, ["bf16"])
        self.assertFalse(report["qualification"])

    def test_uncertain_cleanup_keeps_unresolved_and_stops_second_arm(self):
        def uncertain(request, **kwargs):
            value = self.observer(request, **kwargs)
            if request["mode"] in ("release", "independent"):
                value["authenticated_processes_present"] = [request["root_process"]]
            return value
        report = self.invoke(observer=uncertain)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(self.supervised, ["bf16"])
        self.assertTrue(report["campaign"]["unresolved"])
        self.assertFalse(report["qualification"])

    def test_observer_refusal_and_expired_identity_deadline_prevent_start(self):
        report = self.invoke(observer=lambda *args, **kwargs: {"status": "INCOMPLETE"})
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(self.supervised, [])
        with self.assertRaises(ValueError):
            controller.validate_manifest(self.manifest, self.pin, deadline=time.monotonic() - 1)

    def test_production_pins_and_no_live_default_remain_explicit(self):
        self.assertEqual(controller.METADATA_SHA256, "c33222c324bf841b32d71544450bbf8b418954d373b005f1d032c860f065b4c5")
        self.assertIn("physical grant", " ".join(controller.GAPS))
        self.assertEqual(self.invoke(grant={})["status"], "INCOMPLETE")
        self.assertEqual(self.invoke(supervisor=controller.adapter.owner.supervise)["status"], "INCOMPLETE")
        self.assertNotIn("torch", sys.modules)

    def physical_grant(self):
        now = time.monotonic()
        parent = {"pid": 900, "start_ticks": 5, "uid": 0, "boot_id": BOOT}
        manifest = dict(self.value, schema=2)
        grant = {"schema": 2, "scope": "one-matched-fp8-kv-benchmark", "model_start_authorized": True,
            "maximum_starts": 2, "arm_order": ["bf16", "kv4"], "cumulative_ceiling_seconds": 2400,
            "fixed_reserve_seconds": 300, "received_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "received_monotonic": now, "boot_id": BOOT, "manifest_sha256": self.pin,
            "package_sha256": controller.mapping_sha256(manifest["package_files"]),
            "source_manifest_sha256": controller.mapping_sha256({name: pin for name, pin in manifest["package_files"].items()
                if name.endswith(".py")}), "transport_parent": parent}
        return grant, manifest, parent

    def test_physical_grant_binds_real_parent_receipt_sources_and_exact_counts(self):
        grant, manifest, parent = self.physical_grant()
        self.assertEqual(controller.validate_physical_grant(grant, manifest, self.pin, parent), grant)
        for key, bad in (("maximum_starts", True), ("arm_order", ["kv4", "bf16"]),
                         ("package_sha256", "0" * 64), ("fixed_reserve_seconds", 299),
                         ("received_monotonic", time.monotonic() - 106)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.validate_physical_grant(dict(grant, **{key: bad}), manifest, self.pin, parent)
        with self.assertRaises(ValueError):
            controller.validate_physical_grant(grant, manifest, self.pin, dict(parent, uid=1000))
        with self.assertRaises(ValueError):
            controller.main([])

    def test_actual_physical_entry_refuses_missing_retained_base_before_lease_or_model(self):
        grant, manifest, parent = self.physical_grant()
        manifest_path, grant_path = self.base / "PHYSICAL-MANIFEST.json", self.base / "INERT-GRANT.json"
        manifest_path.write_bytes(raw(manifest))
        manifest_pin = sha(manifest_path.read_bytes())
        grant["manifest_sha256"] = manifest_pin
        grant_path.write_bytes(raw(grant))
        class InertRootRPC:
            def authenticate_parent(self):
                return parent
        with ExitStack() as stack:
            for name in ("getuid", "geteuid", "getgid", "getegid"):
                stack.enter_context(patch.object(controller.os, name, return_value=1000))
            after = stack.enter_context(patch.object(controller, "_run_physical_after_retained_closure"))
            with self.assertRaisesRegex(ValueError, "NOT_READY.*base-rootfs"):
                controller.run_physical_controller(manifest_path=manifest_path, manifest_sha256=manifest_pin,
                    grant_path=grant_path, grant_sha256=sha(grant_path.read_bytes()), rpc=InertRootRPC())
            after.assert_not_called()
        self.assertEqual(self.supervised, [])
        self.assertNotIn("torch", sys.modules)

    def test_actual_seal_has_exact_members_no_self_reference_and_external_charge(self):
        root = self.base / "seal-fixture"
        root.mkdir()
        (root / "observed.txt").write_bytes(b"explicit local CPU evidence\n")
        now = time.monotonic()
        receipt = controller.seal_result(root, deadline=now + 10, received_monotonic=now,
            received_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        self.assertEqual(receipt["member_count"], 1)
        self.assertEqual(receipt["manifest_sha256"], sha((root / "FINAL-ARTIFACTS.sha256").read_bytes()))
        self.assertTrue(root.with_name(root.name + ".closeout.json").is_file())
        self.assertNotIn(b"FINAL-ARTIFACTS", (root / "FINAL-ARTIFACTS.sha256").read_bytes())
        self.assertGreaterEqual(receipt["charged_seconds"], receipt["elapsed_seconds"])
        with self.assertRaises(ValueError):
            controller.seal_result(root, deadline=now + 10, received_monotonic=now, received_utc="unused")

    def test_live_namespace_join_reads_real_function_with_declared_proc_files(self):
        child = self.identity(1001, "pid:[200]")
        proc = self.base / "proc"
        (proc / "1001").mkdir(parents=True)
        (proc / "1001/status").write_bytes(b"Name:\tdeclared-cpu-fixture\nNSpid:\t1001 7\n")
        with patch.object(controller.adapter.owner, "read_process", return_value=child):
            row = controller.observe_namespace_process(child, proc_root=proc)
            self.assertEqual(row["nspid"], [1001, 7])
            (proc / "1001/status").write_bytes(b"NSpid:\t999 7\n")
            with self.assertRaises(ValueError):
                controller.observe_namespace_process(child, proc_root=proc)
        with patch.object(controller.adapter.owner, "read_process", side_effect=[child, None]):
            (proc / "1001/status").write_bytes(b"NSpid:\t1001 7\n")
            self.assertIsNone(controller.observe_namespace_process(child, proc_root=proc))

    def test_counterbalanced_seal_preserves_explicit_ceiling_and_model_default(self):
        # Real local filesystem sealing, with only the elapsed clock substituted.
        root = self.base / 'counterbalanced-seal'
        root.mkdir()
        (root / 'cpu.txt').write_text('synthetic elapsed time, no physical run\n')
        receipt = controller.seal_result(root, deadline=3000, received_monotonic=0,
            received_utc='2026-10-15T00:00:00Z', clock=lambda: 2500.25, complete_seconds=3000)
        self.assertEqual(receipt['charged_seconds'], 2501)
        self.assertEqual(receipt['member_count'], 1)
        for index, kwargs in enumerate(({}, {'complete_seconds': 2999},
                                       {'complete_seconds': True}, {'complete_seconds': 3000, 'deadline': 3001})):
            target = self.base / ('seal-refusal-' + str(index))
            target.mkdir()
            arguments = dict(deadline=3000, received_monotonic=0,
                             received_utc='fixture', clock=lambda: 2500.25)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                controller.seal_result(target, **arguments)
            self.assertFalse((target / 'FINAL-ARTIFACTS.sha256').exists())

    def test_final_resource_reader_requires_bound_gate_and_clear_actual_events(self):
        gate = self.identity(1001, "pid:[100]")
        result = controller.adapter.owner.SupervisionResult(gate, (gate,), 0, None, (), (), True, None)
        initial = self.resource_receipt("bf16")
        initial.update(phase="initial", benchmark_spec_sha256=controller.SPEC_SHA256,
            gate_process={"pid": gate.pid, "start_ticks": gate.start_ticks},
            cgroup_identity={"device": 20, "inode": 30})
        final = {key: initial[key] for key in ("schema", "scope", "arm", "root", "session_sha256",
            "benchmark_spec_sha256", "gate_sha256", "entry_sha256", "bwrap_sha256", "workload_argv_sha256",
            "gate_process", "cgroup_identity")}
        final.update(phase="final", complete=True, child_exit=0, child_reaped=True, owner_signals=[],
            qualification=False, only_wrapper_remains=True, cgroup_release_proven=False,
            resource_boundaries_clear=True, memory_events={"max": 0, "oom": 0, "oom_kill": 0},
            pids_events={"max": 0}, cgroup_events={"populated": 1, "frozen": 0},
            final_usage={"memory.current": 100, "memory.peak": 200, "pids.current": 1})
        snapshots = [dict(ordinal=i, started_monotonic=10.0 + i * .25,
            finished_monotonic=10.01 + i * .25, complete=True,
            gate_process=initial["gate_process"], cgroup_identity=initial["cgroup_identity"],
            limits=initial["limits"], cpu_max=initial["cpu_max"],
            memory_events=final["memory_events"], pids_events=final["pids_events"],
            cgroup_events=final["cgroup_events"], usage=final["final_usage"],
            processes=[str(initial["gate_process"]["pid"])], threads=[str(initial["gate_process"]["pid"])],
            only_wrapper_remains=True) for i in (0, 1)]
        final["cgroup_observation"] = {"snapshots": snapshots, "quiescence": {
            "schema": 1, "started_monotonic": 10., "deadline_monotonic": 12.,
            "finished_monotonic": 10.26, "maximum_seconds": 2., "sample_interval_seconds": .2,
            "sample_limit": 12, "sample_bytes_limit": 4096, "required_consecutive": 2,
            "consecutive": 2, "complete": True}}
        controller.checked_physical_resources({"initial": initial, "final": final}, self.prepared, "bf16", result)
        for key, bad in (("complete", False), ("child_exit", 1), ("owner_signals", [15]),
                         ("gate_process", {"pid": 1002, "start_ticks": 10}),
                         ("memory_events", {"max": 1, "oom": 0, "oom_kill": 0})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.checked_physical_resources({"initial": initial, "final": dict(final, **{key: bad})},
                    self.prepared, "bf16", result)
        for mutate in (lambda f: f.pop("cgroup_observation"),
                       lambda f: f["cgroup_observation"]["snapshots"].pop(),
                       lambda f: f["cgroup_observation"]["snapshots"][0]["usage"].update({"pids.current": 2}),
                       lambda f: f["cgroup_observation"]["snapshots"][0].update(threads=["99"]),
                       lambda f: f["cgroup_observation"]["snapshots"][0].update(started_monotonic=12.),
                       lambda f: f["cgroup_observation"]["quiescence"].update(maximum_seconds=3.),
                       lambda f: f["cgroup_observation"]["snapshots"][0].update(cpu_max=[1, 1])):
            changed = copy.deepcopy(final)
            mutate(changed)
            with self.assertRaises(ValueError):
                controller.checked_physical_resources({"initial": initial, "final": changed},
                    self.prepared, "bf16", result)

    def test_crash_diagnostics_transport_and_controller_completion_are_separate(self):
        artifacts = load("inert_v2_artifacts", ROOT / "tools/benchmark_artifacts.py")
        bench = controller.adapter.readout.bench
        binding = dict(arm="bf16", session_sha256=self.value["session_sha256"],
                       spec_sha256=controller.SPEC_SHA256)
        for completed in (False, True):
            output = self.base / ("progress-complete" if completed else "progress-crash")
            output.mkdir()
            for rank in (0, 1):
                with patch.object(bench.os, "getpid", return_value=1000 + rank):
                    progress = bench.RankProgress(output, rank, **binding)
                try:
                    for phase in bench.PROGRESS_PHASES[:bench.PROGRESS_PHASES.index("ready") + 1]:
                        progress.record(phase)
                    for ordinal, item in enumerate(bench.request_plan({})):
                        progress.record("batch_intent", **item, ordinal=ordinal)
                        if not completed:
                            break
                        progress.record("batch_complete", **item, ordinal=ordinal)
                    if completed:
                        progress.record("cleanup")
                        progress.record("rank_finished", success=True)
                finally:
                    progress.close()
                (output / f"rank-{rank}.log").write_bytes(b"retained native output\x00\xff")
            (output / "owned-workers.json").write_bytes(raw({"pids": [1000, 1001], "worker_seconds": 690}))
            stage = artifacts.stage_artifacts(output, binding)
            sink = io.BytesIO()
            artifacts.export_bundle(stage, binding, sink)
            destination = output / "received"
            receipt = artifacts.receive_bundle(sink.getvalue(), b"", destination, binding)
            self.assertFalse(receipt["all_members_present"])
            diagnostic = controller.diagnostic_readout(destination, binding, artifacts)
            self.assertTrue(diagnostic["valid"], diagnostic)
            self.assertFalse(diagnostic["qualification"])
            if completed:
                controller.require_complete_diagnostics(diagnostic)
            else:
                self.assertEqual(diagnostic["rank_progress"][0]["intent_requests"], 1)
                self.assertEqual(diagnostic["rank_progress"][0]["completed_requests"], 0)
                with self.assertRaises(ValueError):
                    controller.require_complete_diagnostics(diagnostic)
            inner = {"rank_progress": diagnostic["rank_progress"], "ranks": [None, None]}
            self.assertTrue(controller.diagnostic_readout(destination, binding, artifacts, inner)["valid"])
            inner["rank_progress"] = []
            self.assertFalse(controller.diagnostic_readout(destination, binding, artifacts, inner)["valid"])
            path = destination / "diagnostics.json"
            original = path.read_bytes()
            for mutate in (lambda d: d["binding"].update(arm="kv4"),
                           lambda d: d["logs"]["0"].update(tail_sha256="0" * 64),
                           lambda d: d["logs"]["0"].update(tail_offset=1),
                           lambda d: d["logs"]["0"].update(original_bytes=True)):
                meta = json.loads(original)
                mutate(meta)
                path.write_bytes(raw(meta))
                bad = controller.diagnostic_readout(destination, binding, artifacts)
                self.assertFalse(bad["valid"])
                with self.assertRaises(ValueError):
                    controller.require_complete_diagnostics(bad)
            path.write_bytes(original)
            progress_file = destination / "rank-0-progress.jsonl"
            progress_file.write_bytes(progress_file.read_bytes() + b'{"torn":')
            bad = controller.diagnostic_readout(destination, binding, artifacts)
            self.assertFalse(bad["valid"])
            self.assertIsNotNone(bad["rank_progress"][0]["intent_requests"])
            progress_file.unlink()
            bad = controller.diagnostic_readout(destination, binding, artifacts)
            self.assertFalse(bad["valid"])
            self.assertIsNone(bad["rank_progress"][0]["intent_requests"])

    def test_native_reader_joins_both_real_typed_records_to_observed_host_processes(self):
        helper = load("inert_native_proof_validator", ROOT / "tools/benchmark_gpu_identity.py")
        identity_dir = self.base / "identities"
        identity_dir.mkdir()
        devices = [{"index": i, "bdf": bdf, "uuid": uuid, "architecture": "gfx1201", "vram_bytes": 32 << 30}
            for i, (bdf, uuid) in enumerate(controller.adapter.readout.bench.PHYSICAL_DEVICES.items())]
        proof = {"library": str(helper.LIBRARY), "sha256": helper.LIBRARY_SHA256, "bytes": helper.LIBRARY_BYTES,
            "no_load_handle": True, "symbol": "hipDeviceGetPCIBusId", "header_sha256": helper.HEADER_SHA256,
            "loaded_inode": 1, "loaded_device": 1}
        rows, joins, observed = [], [], []
        for rank in (0, 1):
            host = self.identity(1001 + rank, "pid:[200]")
            observed.append(host)
            process = {"pid": 7 + rank, "start_ticks": 10, "boot_id": BOOT,
                       "pid_namespace": "pid:[200]", "uid": 1000, "gid": 1000}
            native = {"schema": 1, "scope": "benchmark-native-preweight-identity-v1", "rank": rank,
                "gpu_id": rank, "arm": "bf16", "session_sha256": self.value["session_sha256"],
                "spec_sha256": controller.SPEC_SHA256, "helper_sha256": controller.GPU_IDENTITY_SHA256,
                "before_model_config": True, "before_model_runner": True, "qualification": False,
                "devices": devices, "hip": proof, "process": process}
            (identity_dir / f"rank-{rank}-identity.json").write_bytes(raw(native))
            rows.append({"rank": rank, "native_identity": native})
            joins.append({"host_process": asdict(host), "nspid": [host.pid, process["pid"]],
                          "observed_monotonic": time.monotonic()})
        inner = {"physical_artifacts": True, "session_sha256": self.value["session_sha256"],
                 "native_identity_verified": True, "ranks": rows}
        result = controller.adapter.owner.SupervisionResult(observed[0], tuple(observed), 0, None, (), (), True, None)
        controller.checked_native_records(identity_dir, inner, arm="bf16", session_sha256=self.value["session_sha256"],
            joins=joins, result=result)
        with self.assertRaises(ValueError):
            controller.checked_native_records(identity_dir, inner, arm="bf16", session_sha256=self.value["session_sha256"],
                joins=joins[:1], result=result)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
