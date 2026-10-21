"""Source-bound unprivileged benchmark controller and inert CPU composition.

Physical execution requires a separately authenticated human-root socket parent;
there is no sudo, root observer fallback, runtime repair, or retry here.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time


def _adjacent(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _adjacent("spry_controller_adapter", "benchmark_owner_adapter.py")
resources = _adjacent("spry_controller_resources", "benchmark_resource_gate.py")
retained_verifier = _adjacent("spry_retained_runtime_verifier", "verify_benchmark_runtime.py")
SPEC_SHA256 = "67bd5c3188c1bb3af92d8b30f6f4b8e472ad2c8c685a9aa116de6e3f131a4f5a"
MODEL_INVENTORY_SHA256 = "892e95cdabf87bbf96298e258e5889e2572acf21659121d811ca9a09eac09986"
CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
BINARY_SHA256 = "d09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171"
METADATA_SHA256 = "c33222c324bf841b32d71544450bbf8b418954d373b005f1d032c860f065b4c5"
SOURCE_NAMES = ("benchmark_physical_controller.py", "benchmark_owner_adapter.py",
    "host_process_owner.py", "benchmark_resource_gate.py", "benchmark_release_observer.py",
    "collect_benchmark_platform.py", "model_benchmark.py", "analyze_model_benchmark.py")
PACKAGE_NAMES = {"review/" + name for name in SOURCE_NAMES} | {
    "review/BENCHMARK-SPEC2.json", "review/RESOURCE-MANIFEST.json",
    "review/MODEL-INVENTORY.json", "review/COMMON-STACK-RECEIPT.json"}
LIVE_SOURCE_NAMES = SOURCE_NAMES + ("benchmark_gpu_identity.py", "benchmark_artifacts.py",
                                   "prepare_benchmark_package.py", "benchmark_host_transport.py",
                                   "verify_benchmark_runtime.py")
LIVE_PACKAGE_NAMES = {"review/" + name for name in LIVE_SOURCE_NAMES} | {
    "review/BENCHMARK-SPEC2.json", "review/RESOURCE-MANIFEST.json", "review/MODEL-INVENTORY.json",
    "review/COMMON-STACK-RECEIPT.json", "review/COMPOSITION-FACTS.json"}
FACTS_SHA256 = "539441ba42ac450d7b03cc47dc5462beda174937931e2a0f2fd832ac102cc5a5"
GPU_IDENTITY_SHA256 = "ccd819f15d2df5e13acd1f7575d6a296be6ac5c9f1096658d34894e9cf3989bb"
PROOF_NAME = "image-source/python/sglang/srt/layers/attention/ultraquant-bf16-build.json"
BINARY_NAME = "image-source/python/sglang/srt/layers/attention/libuq_bf16_v2.so"
RUNTIME_SCOPES = ("image-source/python/sglang", "codec", "runtime")
RUNTIME_REQUIRED = {"image-source/python/sglang/" + name for name in (
    "srt/model_executor/model_runner.py", "srt/model_executor/pool_configurator.py",
    "srt/layers/attention/ultraquant_contract.py", "srt/layers/attention/ultraquant_backend.py",
    "srt/layers/attention/ultraquant_evidence.py", "srt/mem_cache/unified_radix_cache.py")}
FIELDS = {"schema", "scope", "experiment_mode", "target_root", "boot_id", "session_sha256",
    "package_root", "package_files", "retained_root", "runtime_root", "runtime_files", "build_proof_file",
    "native_binary_file", "model_root", "model_inventory_file", "spec_sha256",
    "resource_manifest_sha256", "launch_authorized", "proposed_complete_seconds"}
GAPS = ["authenticated human root-observer transport and outer termination/reap",
        "native pre-weight GPU UUID/BDF/render verification through final entry",
        "final target-path/import closure and private-interpreter proof",
        "final bounded-scratch source-count/output-size and aggregate-space proof",
        "final scoped resource-event and cgroup-release observation after each arm",
        "separately authorized physical grant, one-use receipt and verified final sealing"]


def release_process_inventory(processes, root_process):
    """Project observation history to unique PIDs without merging substitutions.

    setsid/exec can change group, session, command, parent and state without
    creating a new process. Keep the full history in SupervisionResult; only
    this read-only release request uses one record per immutable identity.
    The journal's exact root record remains the cleanup binding.
    """
    need(len(processes) <= 512, "release process history bound exceeded")
    mutable = {"ppid", "pgid", "sid", "command_sha256", "state"}
    selected = {}
    for process in processes:
        need(isinstance(process, adapter.owner.ProcessIdentity), "invalid release process identity")
        row = asdict(process)
        previous = selected.get(process.pid)
        need(previous is None or all(previous[key] == value for key, value in row.items()
             if key not in mutable), "conflicting release process PID identity")
        selected[process.pid] = row
    if root_process is not None:
        need(root_process in processes, "journal root missing from process history")
        selected[root_process.pid] = asdict(root_process)
    return [selected[pid] for pid in sorted(selected)]


def need(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value):
    need(type(value) is str, "relative member string required")
    path = PurePosixPath(value)
    need(value and not path.is_absolute() and str(path) == value
         and not any(part in (".", "..", ".idea", "keeper-env") for part in value.split("/")),
         "unsafe member")
    return value


def _canonical(value):
    need(type(value) is str and value.startswith("/") and "\x00" not in value,
         "canonical absolute path required")
    path = Path(value)
    need(str(path) == value and path.resolve(strict=True) == path and len(path.parents) >= 3
         and not any(part in (".idea", "keeper-env") for part in path.parts), "unsafe source root")
    return path


def _stamp(info):
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode", "st_nlink",
        "st_size", "st_mtime_ns", "st_ctime_ns"))


def _hash(path, deadline, clock):
    need(clock() < deadline, "identity work deadline")
    path = Path(path)
    need(path.resolve(strict=True) == path and not path.is_symlink(), "source link refused")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular singly linked source required")
        digest = hashlib.sha256()
        while True:
            need(clock() < deadline, "identity work deadline")
            data = os.read(fd, 1 << 20)
            if not data:
                break
            digest.update(data)
        need(_stamp(before) == _stamp(os.fstat(fd)) == _stamp(path.lstat()), "source changed while hashing")
        return digest.hexdigest(), before.st_size, _stamp(before)
    finally:
        os.close(fd)


def _tree(root, inventory, deadline, clock):
    need(type(inventory) is dict and 1 <= len(inventory) <= 32768, "bounded exact inventory required")
    for name, digest in inventory.items():
        _relative(name)
        need(_digest(digest), "inventory digest required")
    found, stamps = set(), {}
    for path in root.rglob("*"):
        need(clock() < deadline, "identity work deadline")
        need(not path.is_symlink(), "tree link refused")
        if path.is_dir():
            continue
        name = str(path.relative_to(root))
        need(name in inventory, "extra inventory member")
        digest, _, stamp = _hash(path, deadline, clock)
        need(digest == inventory[name], "inventory hash mismatch")
        found.add(name)
        stamps[str(path)] = stamp
    need(found == set(inventory), "missing inventory member")
    return stamps


def validate_manifest(path, expected_sha256, *, deadline, clock=time.monotonic):
    """Authenticate complete local inputs; no runtime import or remote query."""
    value, _ = adapter.readout.read_json(path, expected_sha256)
    live = value.get("schema") == 2
    need(set(value) == FIELDS | ({"runtime_scopes", "retained_base"} if live else set()) and
         value["schema"] in (1, 2) and
         value["scope"] == ("matched-benchmark-physical-package-v2" if live else
                            "matched-benchmark-controller-candidate-v1") and
         value["experiment_mode"] == "model-benchmark-v1" and value["launch_authorized"] is False
         and value["proposed_complete_seconds"] == 2400, "preparation-only controller manifest")
    need(re.fullmatch(r"/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}", value["target_root"])
         and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["boot_id"])
         and _digest(value["session_sha256"]), "target/session binding")
    package = _canonical(value["package_root"])
    names = LIVE_PACKAGE_NAMES if live else PACKAGE_NAMES
    need(set(value["package_files"]) == names, "exact package source/assets required")
    if live:
        require_retained_base_binding(value.get('retained_base'))
        # The review directory is immutable; receipts/arms live outside this
        # exact subtree and therefore cannot invalidate package membership.
        stamps = _tree(package / "review", {name.removeprefix("review/"): pin
            for name, pin in value["package_files"].items()}, deadline, clock)
    else:
        stamps = _tree(package, value["package_files"], deadline, clock)
    for name in (LIVE_SOURCE_NAMES if live else SOURCE_NAMES):
        actual = Path(__file__).resolve().with_name(name)
        need(_hash(actual, deadline, clock)[0] == value["package_files"]["review/" + name],
             "executed author/source differs from package")
    need(value["package_files"]["review/collect_benchmark_platform.py"] == METADATA_SHA256,
         "metadata collector identity")
    spec_path = package / "review/BENCHMARK-SPEC2.json"
    need(value["spec_sha256"] == SPEC_SHA256 == value["package_files"]["review/BENCHMARK-SPEC2.json"],
         "frozen benchmark spec identity")
    spec, _ = adapter.readout.read_json(spec_path, SPEC_SHA256)
    adapter.readout.bench.validate_spec(spec)
    need(spec["worker_seconds"] == 690 and spec["config_path"] == "/mnt/models/config.json"
         and spec["package_root"] == "/mnt/composition/image-source/python/sglang"
         and spec["stack_receipt"] == "/mnt/review/COMMON-STACK-RECEIPT.json"
         and spec["server_args"]["model_path"] == "/mnt/models", "unchanged inner paths/work bound")
    resource_path = package / "review/RESOURCE-MANIFEST.json"
    need(value["resource_manifest_sha256"] == value["package_files"]["review/RESOURCE-MANIFEST.json"],
         "resource manifest binding")
    resource_manifest = resources.read_manifest(resource_path, value["resource_manifest_sha256"])
    need(resource_manifest["root"] == value["target_root"] and
         resource_manifest["session_sha256"] == value["session_sha256"] and
         resource_manifest["benchmark_spec_sha256"] == SPEC_SHA256 and
         resource_manifest["gate_sha256"] == value["package_files"]["review/benchmark_resource_gate.py"] and
         resource_manifest["entry_sha256"] == value["package_files"]["review/model_benchmark.py"],
         "resource/package/source binding")
    runtime = _canonical(value["runtime_root"])
    retained = _canonical(value["retained_root"])
    for arm in ("bf16", "kv4"):
        expected = adapter.bwrap_argv(arm, retained_root=retained, composition=runtime,
            review=Path(value["target_root"]) / "review", model=value["model_root"],
            output=Path(value["target_root"]) / "arms" / arm,
            entry=Path(value["target_root"]) / "review/model_benchmark.py",
            spec_sha256=SPEC_SHA256, bounded_scratch=True,
            **({"physical_artifacts": True, "session_sha256": value["session_sha256"],
                "artifacts_sha256": value["package_files"]["review/benchmark_artifacts.py"]} if live else {}))
        need(resource_manifest["workload_argv"][arm] == list(expected), "exact read-only bounded-scratch projection")
    need(RUNTIME_REQUIRED <= set(value["runtime_files"]), "complete executable runtime source closure required")
    if live:
        need(value["runtime_scopes"] == list(RUNTIME_SCOPES), "fixed complete runtime closure scopes")
        need(value["package_files"]["review/COMPOSITION-FACTS.json"] == FACTS_SHA256 and
             value["package_files"]["review/benchmark_gpu_identity.py"] == GPU_IDENTITY_SHA256,
             "authenticated composition/native identity source")
        facts_raw = adapter.readout.bench.checked_file(package / "review/COMPOSITION-FACTS.json",
            FACTS_SHA256, maximum=16 << 20)
        facts = json.loads(facts_raw, object_pairs_hook=adapter.readout._object,
                           parse_constant=adapter.readout._constant)
        retained_identity = retained_verifier.verify_retained(retained, facts, deadline)
        bound = {"image-source/python/sglang/" + name: pin for name, pin in facts["runtime_files"].items()}
        bound.update(facts["native_sources"])
        bound.update({PROOF_NAME: spec["build_proof_sha256"], BINARY_NAME: BINARY_SHA256})
        need(value["runtime_files"] == bound and not any("__pycache__" in name.split("/") or
             name.endswith(".pyc") for name in bound), "complete cache-free effective source closure")
        for scope in RUNTIME_SCOPES:
            scoped = {name[len(scope) + 1:]: pin for name, pin in bound.items() if name.startswith(scope + "/")}
            stamps.update(_tree(runtime / scope, scoped, deadline, clock))
        need(value["build_proof_file"] == PROOF_NAME and value["native_binary_file"] == BINARY_NAME,
             "existing native proof/library paths")
    else:
        stamps.update(_tree(runtime, value["runtime_files"], deadline, clock))
    proof_name, binary_name = _relative(value["build_proof_file"]), _relative(value["native_binary_file"])
    need(proof_name in value["runtime_files"] and binary_name in value["runtime_files"] and
         value["runtime_files"][binary_name] == BINARY_SHA256, "existing runtime native/build proof required")
    proof, _ = adapter.readout.read_json(runtime / proof_name, value["runtime_files"][proof_name])
    need(proof.get("schema") == 2 and proof.get("serving_composition_complete") is True and
         proof.get("qualification") == "build-reviewed-not-serving-qualified" and
         proof.get("binary_sha256") == BINARY_SHA256 and
         proof.get("stack_receipt_sha256") == value["package_files"]["review/COMMON-STACK-RECEIPT.json"]
         == spec["stack_receipt_sha256"] and value["runtime_files"][proof_name] == spec["build_proof_sha256"],
         "unchanged common-stack/build proof")
    inventory_name = _relative(value["model_inventory_file"])
    need(inventory_name == "review/MODEL-INVENTORY.json" and
         value["package_files"][inventory_name] == MODEL_INVENTORY_SHA256, "retained model inventory identity")
    raw = adapter.readout.bench.checked_file(package / inventory_name, MODEL_INVENTORY_SHA256)
    model_rows = json.loads(raw)
    need(type(model_rows) is list and len(model_rows) == 81 and all(type(row) is dict
         and set(row) == {"name", "size", "sha256"} for row in model_rows), "complete 81-file model inventory")
    model_files = {row["name"]: row["sha256"] for row in model_rows}
    need(len(model_files) == 81 and [row["name"] for row in model_rows] == sorted(model_files)
         and model_files.get("config.json") == CONFIG_SHA256, "model names/config identity")
    model = _canonical(value["model_root"])
    stamps.update(_tree(model, model_files, deadline, clock))
    need(all(type(row["size"]) is int and row["size"] >= 0 and
             (model / row["name"]).stat().st_size == row["size"] for row in model_rows), "model payload sizes")
    return {"manifest": value, "spec": spec, "spec_path": str(spec_path),
            "resource_manifest": resource_manifest, "resource_path": str(resource_path), "stamps": stamps,
            **({'retained_identity': retained_identity} if live else {})}


def verify_unchanged(prepared):
    for path, stamp in prepared["stamps"].items():
        need(not Path(path).is_symlink() and _stamp(Path(path).lstat()) == stamp, "authenticated input changed")
    for path, previous in prepared.get('retained_identity', {}).get('stamps', {}).items():
        need(retained_verifier.stamp(Path(path).lstat()) == previous, 'authenticated retained runtime changed')


def checked_resource_receipt(receipt, prepared, arm):
    value, frozen = prepared["manifest"], prepared["resource_manifest"]
    expected = {"schema": 1, "scope": resources.SCOPE, "arm": arm, "root": value["target_root"],
        "session_sha256": value["session_sha256"], "gate_sha256": frozen["gate_sha256"],
        "entry_sha256": frozen["entry_sha256"], "bwrap_sha256": frozen["bwrap_sha256"],
        "workload_argv_sha256": frozen["workload_argv_sha256"][arm], "actual_cgroup_verified": True}
    need(type(receipt) is dict and all(type(receipt.get(key)) is type(item) and receipt.get(key) == item
         for key, item in expected.items()), "resource receipt/source binding")
    need(receipt.get("limits") == {"memory.max": 38 * 1024**3, "memory.swap.max": 0, "pids.max": 512},
         "actual resource limits differ")
    need(receipt.get("cgroup_path") == resources.USER_MANAGER + "/app.slice/" + frozen["units"][arm]
         and receipt.get("rlimits") == {"core": [0, 0], "file_size": [128 << 20, 128 << 20]}
         and receipt.get("scratch_quota_verified") is False, "resource path/limit observation differs")
    usage = receipt.get("initial_usage", {})
    need(type(usage.get("memory.current")) is int and 0 < usage["memory.current"] < 38 * 1024**3
         and type(usage.get("pids.current")) is int and 0 < usage["pids.current"] <= 512,
         "invalid initial scoped usage")
    cpu = receipt.get("cpu_max")
    need(type(cpu) is list and len(cpu) == 2 and all(type(item) is int for item in cpu)
         and 1000 <= cpu[1] <= 1000000 and cpu[0] == 24 * cpu[1], "actual CPU quota differs")
    need(all(receipt.get("memory_events", {}).get(key) == 0 for key in ("max", "oom", "oom_kill"))
         and receipt.get("pids_events", {}).get("max") == 0, "resource/OOM failure")


def _run_bound_controller(*, manifest_path, manifest_sha256, grant, lease, observer,
                   read_inner_result, read_resource_receipt, resource_check, streams,
                   supervisor, clock=time.monotonic, physical_context=None, prepare_streams=None):
    """Exercise the candidate composition with explicitly supplied CPU effects.

    This slice accepts only an inert preparation grant. It does not expose a
    live physical mode or invent an authenticated root observation channel.
    """
    started = clock() if physical_context is None else physical_context["received_monotonic"]
    report = {"schema": 1, "scope": "benchmark-physical-controller-candidate-v1",
        "status": "INCOMPLETE", "qualification": False, "physical_ready": False,
        "launch_authorized": False, "native_run": False, "remaining_gaps": list(GAPS),
        "proposed_complete_seconds": 2400, "admission_seconds": 180,
        "source_preparation_seconds": 105, "locked_admission_seconds": 75,
        "arm_envelope_seconds": 795, "collection_reserve_seconds": 330,
        "seal_reserve_seconds": 300, "failure": None, "observer_requests": [], "resource_receipts": []}
    try:
        if physical_context is None:
            need(type(grant) is dict and grant == {"schema": 1, "scope": "inert-controller-cpu-composition",
                "launch_authorized": False, "maximum_starts": 0, "proposed_complete_seconds": 2400,
                "manifest_sha256": manifest_sha256}, "unfunded proposal permits inert CPU composition only")
            need(supervisor is not adapter.owner.supervise, "inert composition cannot use the live supervisor")
        else:
            need(grant == physical_context and grant["schema"] == 2, "validated physical context required")
            report["remaining_gaps"] = []
            report["scope"] = "matched-physical-model-benchmark-v2"
        need(all(callable(item) for item in (observer, read_inner_result, read_resource_receipt,
             resource_check, supervisor)), "complete explicit owner effects required")
        prepared = validate_manifest(manifest_path, manifest_sha256, deadline=started + 105, clock=clock)
        value, results = prepared["manifest"], []
        checked_spec, _ = adapter.readout.read_json(prepared["spec_path"], SPEC_SHA256)
        report["instrumented_timings"] = checked_spec.get("diagnostic_profile") is not None
        if report["instrumented_timings"]:
            report["profile_method"] = adapter.readout.bench.PROFILE_METHOD
        need(physical_context is None or value["schema"] == 2, "physical execution requires physical package schema")
        need(str(lease.root) == value["target_root"] and lease.session == value["session_sha256"]
             and lease.boot == value["boot_id"], "lease/package target binding")
        observer_sha = value["package_files"]["review/benchmark_release_observer.py"]

        def observe(mode, current_lease, arm, result, end):
            now = clock()
            # Adapter already reserved the complete phase. Callback/transport
            # overhead consumes that phase rather than reopening another 75 s.
            phase_end = min(end, now + 75)
            need(phase_end - 75 <= now < phase_end - 15,
                 "observation work/termination reserve exhausted")
            info = Path(current_lease.lock_path).lstat()
            history = result.observed_processes if result else ()
            if mode == "independent":
                history = tuple(item for one in results for item in one.observed_processes)
            rows = release_process_inventory(history, result.root_process if result else None)
            journal_sha = None
            if current_lease.journal.exists():
                journal_sha = hashlib.sha256(adapter.readout.bench.read_bounded_file(
                    current_lease.journal, 256 << 10)).hexdigest()
            request = {"schema": 1, "mode": mode, "root": str(current_lease.root),
                "session_sha256": current_lease.session, "boot_id": current_lease.boot,
                "lock_device": info.st_dev, "lock_inode": info.st_ino, "arm": arm,
                "root_process": asdict(result.root_process) if result else None,
                "observed_processes": rows, "journal_sha256": journal_sha,
                "issued_monotonic": phase_end - 75, "deadline_monotonic": phase_end,
                "native_ssh_host_contract": True}
            report["observer_requests"].append(request)
            snapshot = observer(request, expected_observer_sha256=observer_sha, deadline_monotonic=phase_end)
            need(clock() <= phase_end and type(snapshot) is dict and snapshot.get("status") == "OBSERVATION_COMPLETE"
                 and snapshot.get("failures") == [] and snapshot.get("collector_source_sha256") == observer_sha
                 and snapshot.get("metadata_source_sha256") == METADATA_SHA256,
                 "observer missing, unbound, failed or deadline-exhausted")
            if physical_context is not None and mode in ("release", "independent"):
                released = snapshot.get("root_resource_release", {})
                expected_arms = [arm] if mode == "release" else [row["arm"] for row in current_lease.record["arms"]]
                need(type(released) is dict and type(released.get("scopes")) is list and
                     [row.get("arm") for row in released["scopes"]] == expected_arms,
                     "independent root cgroup release inventory required")
                for row in released["scopes"]:
                    expected_path = "/sys/fs/cgroup" + resources.USER_MANAGER + "/app.slice/" + prepared["resource_manifest"]["units"][row["arm"]]
                    need(row.get("cgroup_path") == expected_path and row.get("absent") is True and
                         type(row.get("observed_monotonic")) in (int, float) and
                         now <= row["observed_monotonic"] <= clock(), "cgroup release unproved or stale")
            return snapshot

        def admission(current_lease, *, deadline_monotonic):
            need(deadline_monotonic <= started + 180, "locked admission exceeds complete preparation bound")
            snapshot = observe("locked-admission", current_lease, None, None, deadline_monotonic)
            need(snapshot.get("root") == value["target_root"] and
                 snapshot.get("boot_id") == value["boot_id"], "admission target differs")
            report["admission"] = snapshot
            verify_unchanged(prepared)
            return snapshot

        def supervise(argv, **kwargs):
            report["native_run"] = physical_context is not None
            result = supervisor(argv, **kwargs)
            results.append(result)
            arm = lease.record["arms"][-1]["arm"]
            try:
                receipt = read_resource_receipt(arm)
                report["resource_receipts"].append(receipt)
                if physical_context is None:
                    checked_resource_receipt(receipt, prepared, arm)
                else:
                    checked_physical_resources(receipt, prepared, arm, result)
            except Exception as error:
                result = replace(result, failure="resource receipt: " + type(error).__name__)
                results[-1] = result
            return result

        frozen_arms = tuple(adapter.FrozenArm(arm,
            resources.scope_argv(prepared["resource_path"], value["resource_manifest_sha256"], arm),
            prepared["resource_manifest"]["gate_path"], prepared["resource_manifest"]["gate_sha256"])
            for arm in ("bf16", "kv4"))
        campaign = adapter.run_campaign(lease=lease, arms=frozen_arms,
            spec_path=prepared["spec_path"], spec_sha256=SPEC_SHA256,
            collector_sha256=observer_sha, active_deadline=started + 2100,
            collect_release=lambda current, arm, result, end: observe("release", current, arm, result, end),
            read_inner_result=read_inner_result,
            independent_postcheck=lambda current, end: observe("independent", current, None, results[-1], end),
            streams=streams, resource_check=resource_check, supervisor=supervise, clock=clock,
            arm_envelope_seconds=795, collection_reserve_seconds=330, admission_check=admission,
            prepare_streams=prepare_streams)
        report["campaign"] = campaign
        verify_unchanged(prepared)
        need(campaign.get("adapter_complete") is True and campaign.get("unresolved") is False
             and campaign.get("independent_release_verified") is True, "candidate campaign incomplete")
        report["status"] = ("MATCHED_PHYSICAL_BENCHMARK_COMPLETE" if physical_context is not None
                            else "LOCAL_CONTROLLER_COMPOSITION_PASSED")
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)[:1024]
    report["elapsed_seconds"] = clock() - started
    return report


def run_controller(**kwargs):
    """Preserved inert CPU composition; cannot provide a physical context."""
    return _run_bound_controller(**kwargs)


def mapping_sha256(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()


GRANT_FIELDS = {"schema", "scope", "model_start_authorized", "maximum_starts", "arm_order",
    "cumulative_ceiling_seconds", "fixed_reserve_seconds", "received_utc", "received_monotonic",
    "boot_id", "manifest_sha256", "package_sha256", "source_manifest_sha256", "transport_parent"}


def validate_physical_grant(grant, manifest, manifest_sha256, parent_identity, *,
                            clock=time.monotonic, utc_now=lambda: datetime.now(timezone.utc)):
    need(type(grant) is dict and set(grant) == GRANT_FIELDS and grant["schema"] == 2 and
         grant["scope"] == "one-matched-fp8-kv-benchmark" and grant["model_start_authorized"] is True and
         type(grant["maximum_starts"]) is int and grant["maximum_starts"] == 2 and
         grant["arm_order"] == ["bf16", "kv4"] and grant["cumulative_ceiling_seconds"] == 2400 and
         grant["fixed_reserve_seconds"] == 300, "exact separate two-start physical grant required")
    need(manifest["schema"] == 2 and grant["manifest_sha256"] == manifest_sha256 and
         grant["boot_id"] == manifest["boot_id"] and
         grant["package_sha256"] == mapping_sha256(manifest["package_files"]) and
         grant["source_manifest_sha256"] == mapping_sha256({name: pin for name, pin in manifest["package_files"].items()
             if name.endswith(".py")}), "physical grant source/package/boot binding")
    parent = grant["transport_parent"]
    need(type(parent) is dict and set(parent) == {"pid", "start_ticks", "uid", "boot_id"} and
         all(type(parent[key]) is int for key in ("pid", "start_ticks", "uid")) and parent["pid"] > 1 and
         parent["start_ticks"] > 0 and parent["uid"] == 0 and parent["boot_id"] == grant["boot_id"] and
         parent_identity == parent, "authenticated human-root parent required")
    received = grant["received_monotonic"]
    need(type(received) in (int, float) and math.isfinite(received) and
         received <= clock() < received + 105, "source/admission receipt window expired")
    need(type(grant["received_utc"]) is str and grant["received_utc"].endswith("Z"), "original UTC receipt required")
    instant = datetime.fromisoformat(grant["received_utc"].replace("Z", "+00:00"))
    need(instant.tzinfo is not None and abs((utc_now() - instant).total_seconds() - (clock() - received)) <= 3,
         "UTC/monotonic receipt disagreement")
    return grant


def checked_physical_resources(receipts, prepared, arm, result, *, expected_spec_sha256=None):
    expected_spec_sha256 = SPEC_SHA256 if expected_spec_sha256 is None else expected_spec_sha256
    need(_digest(expected_spec_sha256), 'authenticated resource spec identity required')
    need(type(receipts) is dict and set(receipts) == {"initial", "final"}, "complete gate resource observations required")
    checked_resource_receipt(receipts["initial"], prepared, arm)
    final = receipts["final"]
    initial = receipts["initial"]
    # Final resource monitoring is collected while the gate remains alive in the
    # actual scope, then root independently proves disappearance after reaping.
    need(type(final) is dict and initial.get("phase") == "initial" and final.get("phase") == "final" and
         all(final.get(key) == initial.get(key) for key in ("schema", "scope", "arm", "root", "session_sha256",
             "benchmark_spec_sha256", "gate_sha256", "entry_sha256", "bwrap_sha256", "workload_argv_sha256",
             "gate_process", "cgroup_identity")) and
         initial.get("benchmark_spec_sha256") == expected_spec_sha256 and final.get("complete") is True and
         type(final.get("child_exit")) is int and final["child_exit"] == 0 and
         final.get("child_reaped") is True and final.get("owner_signals") == [] and
         final.get("qualification") is False and final.get("only_wrapper_remains") is True and
         final.get("cgroup_release_proven") is False and final.get("resource_boundaries_clear") is True,
         "final resource monitor incomplete")
    gate, cgroup = initial.get("gate_process"), initial.get("cgroup_identity")
    need(type(gate) is dict and set(gate) == {"pid", "start_ticks"} and
         all(type(gate[key]) is int and gate[key] > 0 for key in gate) and
         type(cgroup) is dict and set(cgroup) == {"device", "inode"} and
         all(type(cgroup[key]) is int and cgroup[key] > 0 for key in cgroup) and
         any(item.pid == gate["pid"] and item.start_ticks == gate["start_ticks"]
             for item in result.observed_processes), "resource gate process/cgroup identity unbound")
    need(all(final.get("memory_events", {}).get(key) == 0 for key in ("max", "oom", "oom_kill")) and
         final.get("pids_events", {}).get("max") == 0, "final resource boundary/OOM failure")
    need(result.process_cleanup_observed, "owned process release incomplete")
    usage = final.get("final_usage", {})
    need(type(usage.get("memory.current")) is int and type(usage.get("memory.peak")) is int and
         0 < usage["memory.current"] <= usage["memory.peak"] <= 38 * 1024**3 and
         usage.get("pids.current") == 1 and final.get("cgroup_events", {}).get("populated") == 1 and
         final.get("cgroup_events", {}).get("frozen") == 0, "final scoped population/memory differs")
    observed = final.get("cgroup_observation", {})
    q, samples = observed.get("quiescence", {}), observed.get("snapshots", [])
    need(type(q) is dict and q.get("schema") == 1 and q.get("complete") is True and
         q.get("maximum_seconds") == resources.FINAL_QUIESCENCE_SECONDS and
         q.get("sample_interval_seconds") == resources.FINAL_SAMPLE_INTERVAL_SECONDS and
         q.get("sample_limit") == resources.FINAL_SAMPLE_LIMIT and
         q.get("sample_bytes_limit") == resources.FINAL_SAMPLE_BYTES and
         q.get("required_consecutive") == q.get("consecutive") == 2 and
         all(type(q.get(key)) in (int, float) and math.isfinite(q[key]) for key in
             ("started_monotonic", "deadline_monotonic", "finished_monotonic")) and
         q["deadline_monotonic"] == q["started_monotonic"] + resources.FINAL_QUIESCENCE_SECONDS and
         q["started_monotonic"] <= q["finished_monotonic"] < q["deadline_monotonic"] and
         type(samples) is list and 2 <= len(samples) <= resources.FINAL_SAMPLE_LIMIT,
         "bounded quiescence observations required")
    previous_finish = None
    consecutive = 0
    for index, sample in enumerate(samples):
        need(type(sample) is dict and len(json.dumps(sample, sort_keys=True).encode()) <= resources.FINAL_SAMPLE_BYTES
             and sample.get("ordinal") == index and sample.get("complete") is True and
             sample.get("gate_process") == gate and sample.get("cgroup_identity") == cgroup and
             sample.get("limits") == initial["limits"] and sample.get("cpu_max") == initial["cpu_max"],
             "quiescence snapshot identity/limits")
        start, finish = sample.get("started_monotonic"), sample.get("finished_monotonic")
        need(all(type(v) in (int, float) and math.isfinite(v) for v in (start, finish)) and
             q["started_monotonic"] <= start <= finish < q["deadline_monotonic"] and
             (previous_finish is None or start - previous_finish >= resources.FINAL_SAMPLE_INTERVAL_SECONDS - 1e-8),
             "quiescence snapshot timing")
        previous_finish = finish
        sample_usage = sample.get("usage", {})
        need(all(sample.get("memory_events", {}).get(key, 0) == 0 for key in
             ("max", "oom", "oom_kill", "oom_group_kill")) and
             all(key in sample.get("memory_events", {}) for key in ("max", "oom", "oom_kill")) and
             sample.get("pids_events", {}).get("max") == 0 and
             sample.get("cgroup_events") == {"populated": 1, "frozen": 0} and
             all(type(sample_usage.get(key)) is int for key in ("memory.current", "memory.peak", "pids.current")) and
             0 < sample_usage["memory.current"] <= sample_usage["memory.peak"] <= resources.MEMORY_BYTES and
             0 < sample_usage["pids.current"] <= resources.TASKS, "quiescence resource boundary")
        for members in (sample.get("processes"), sample.get("threads")):
            need(type(members) is list and 0 < len(members) <= resources.TASKS and
                 all(type(pid) is str and re.fullmatch(r"[1-9][0-9]{0,9}", pid) and
                     int(pid) <= 2147483647 for pid in members) and len(set(members)) == len(members),
                 "quiescence membership")
        only = (sample["processes"] == sample["threads"] == [str(gate["pid"])] and
                sample_usage["pids.current"] == 1)
        need(sample.get("only_wrapper_remains") is only, "quiescence count/membership disagreement")
        consecutive = consecutive + 1 if only else 0
        need(consecutive < 2 or index == len(samples) - 1, "extra quiescence snapshots")
    need(consecutive == 2 and q["finished_monotonic"] == previous_finish and
         all(final.get(key) == samples[-1].get(key) for key in
             ("gate_process", "cgroup_identity", "memory_events", "pids_events", "cgroup_events")) and
         final["final_usage"] == samples[-1]["usage"], "final/quiescence mismatch")


def observe_namespace_process(identity, *, proc_root=Path("/proc"), clock=time.monotonic):
    """Retain an authenticated live host/private PID join, not an inferred PID."""
    before = adapter.owner.read_process(identity.pid, str(proc_root))
    if before is None or not identity.same_instance(before):
        return None
    raw = adapter.readout.bench.read_bounded_file(Path(proc_root) / str(identity.pid) / "status", 65536)
    rows = [line.split(b":", 1)[1].split() for line in raw.splitlines() if line.startswith(b"NSpid:")]
    need(len(rows) == 1 and 1 <= len(rows[0]) <= 32 and all(value.isdigit() for value in rows[0]),
         "actual bounded NSpid mapping unavailable")
    nspid = [int(value) for value in rows[0]]
    need(nspid[0] == identity.pid and all(value > 0 for value in nspid), "host/private PID mapping malformed")
    after = adapter.owner.read_process(identity.pid, str(proc_root))
    if after is None:
        return None
    need(before.same_instance(after), "process changed during namespace observation")
    return {"host_process": asdict(before), "nspid": nspid, "observed_monotonic": clock()}


def checked_native_records(directory, inner, *, arm, session_sha256, joins, result):
    need(inner.get("physical_artifacts") is True and inner.get("session_sha256") == session_sha256 and
         inner.get("native_identity_verified") is True and len(inner.get("ranks", [])) == 2,
         "physical native identity completion required")
    identities = []
    for rank in (0, 1):
        path = Path(directory) / f"rank-{rank}-identity.json"
        raw = adapter.readout.bench.read_bounded_file(path, 64 << 10)
        value, _ = adapter.readout.read_json(path, hashlib.sha256(raw).hexdigest())
        adapter.readout.bench.validate_rank_identity(value, rank=rank, arm=arm,
            session_sha256=session_sha256, spec_sha256=SPEC_SHA256)
        need(inner["ranks"][rank].get("rank") == rank and
             inner["ranks"][rank].get("native_identity") == value, "rank/result/native file mismatch")
        process = value["process"]
        matching = [row for row in joins if row["nspid"][-1] == process["pid"] and
            all(row["host_process"][key] == process[key] for key in
                ("boot_id", "start_ticks", "pid_namespace", "uid", "gid"))]
        need(matching and any(any(item.pid == row["host_process"]["pid"] and
             item.start_ticks == process["start_ticks"] and item.boot_id == process["boot_id"]
             for item in result.observed_processes) for row in matching),
             "native rank lacks observed authenticated live host PID join")
        identities.append(value)
    need(identities[0]["process"] != identities[1]["process"], "duplicate native worker identity")
    return identities


def diagnostic_readout(directory, binding, artifacts, inner=None):
    """Retain crash evidence and prefix counts before applying success gates."""
    bench = adapter.readout.bench
    directory = Path(directory)
    value = {"schema": 1, "binding": binding, "valid": False,
             "qualification": False, "errors": [], "rank_progress": [], "logs": {}}
    for rank in (0, 1):
        value["rank_progress"].append(bench.read_progress(directory, rank=rank, arm=binding["arm"],
            session_sha256=binding["session_sha256"], spec_sha256=binding["spec_sha256"]))
    try:
        meta = artifacts._json(bench.read_bounded_file(directory / "diagnostics.json", 64 << 10))
        need(type(meta) is dict and set(meta) == {"schema", "binding", "logs", "errors"} and
             type(meta["schema"]) is int and meta["schema"] == 1 and meta["binding"] == binding and
             meta["errors"] == [] and type(meta["logs"]) is dict and set(meta["logs"]) == {"0", "1"},
             "diagnostic metadata binding")
        value["logs"] = meta["logs"]
        for rank in (0, 1):
            log = meta["logs"][str(rank)]
            need(type(log) is dict and set(log) == {"exists", "missing", "original_bytes", "full_sha256",
                "tail_file", "tail_bytes", "tail_sha256", "tail_offset", "truncated"} and
                log["tail_file"] == f"rank-{rank}.log.tail" and type(log["exists"]) is bool and
                log["missing"] is (not log["exists"]) and type(log["truncated"]) is bool,
                "log metadata shape")
            if not log["exists"]:
                need(all(log[key] is None for key in ("original_bytes", "full_sha256", "tail_bytes",
                    "tail_sha256", "tail_offset")) and not log["truncated"] and
                    not (directory / log["tail_file"]).exists(), "missing log metadata")
                raise ValueError("rank log unavailable")
            need(all(type(log[key]) is int for key in ("original_bytes", "tail_bytes", "tail_offset")) and
                 0 <= log["original_bytes"] <= 128 << 20 and
                 log["tail_bytes"] == min(log["original_bytes"], 2 << 20) and
                 log["tail_offset"] == log["original_bytes"] - log["tail_bytes"] and
                 log["truncated"] is (log["tail_offset"] != 0) and
                 bench.valid_digest(log["full_sha256"]), "log length/hash binding")
            raw = bench.read_bounded_file(directory / log["tail_file"], 2 << 20)
            need(len(raw) == log["tail_bytes"] and hashlib.sha256(raw).hexdigest() == log["tail_sha256"] and
                 (log["truncated"] or log["tail_sha256"] == log["full_sha256"]), "log tail identity")
        workers = artifacts._json(bench.read_bounded_file(directory / "owned-workers.json", 64 << 10))
        need(type(workers) is dict and set(workers) == {"pids", "worker_seconds"} and
             workers["worker_seconds"] == 690 and type(workers["pids"]) is list and
             len(workers["pids"]) == 2 and all(type(pid) is int and pid > 1 for pid in workers["pids"]) and
             len(set(workers["pids"])) == 2, "worker diagnostic inventory")
        for rank, row in enumerate(value["rank_progress"]):
            need(row["valid"] and row["pid"] == workers["pids"][rank], "progress identity/inventory")
        if inner is not None:
            need(inner.get("rank_progress") == value["rank_progress"], "stored progress differs from raw records")
            for rank, row in enumerate(value["rank_progress"]):
                native = inner.get("ranks", [None, None])[rank]
                if type(native) is dict and "native_identity" in native:
                    need(native["native_identity"]["process"]["pid"] == row["pid"],
                         "progress/native worker identity")
        value["valid"] = True
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        value["errors"].append(type(error).__name__ + ": " + str(error)[:512])
    return value


def require_complete_diagnostics(value):
    need(value.get("valid") is True and value.get("qualification") is False and
         len(value.get("rank_progress", [])) == 2 and all(
             row["valid"] and row["finished"] and row["counts_complete"] and
             row["intent_batches"] == row["completed_batches"] == 30 and
             row["intent_requests"] == row["completed_requests"] == 42
             for row in value["rank_progress"]), "complete rank diagnostic evidence required")


def _publish(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    need(len(raw) <= 8 << 20, "bounded controller result")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            need(written > 0, "short result write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def seal_result(root, *, deadline, received_monotonic, received_utc, clock=time.monotonic,
                complete_seconds=2400):
    """Seal the stopped UID1000 tree; closeout stays outside the sealed tree."""
    need(type(complete_seconds) is int and complete_seconds in (2400, 3000) and
         deadline <= received_monotonic + complete_seconds, "fixed seal envelope required")
    root = _canonical(str(root))
    need(clock() < deadline, "seal deadline exhausted")
    manifest = root / "FINAL-ARTIFACTS.sha256"
    need(not manifest.exists(), "sealed result cannot be amended")
    members, total = {}, 0
    for path in sorted(root.rglob("*")):
        need(clock() < deadline and not path.is_symlink(), "seal cutoff/link refusal")
        if path.is_dir():
            continue
        name = _relative(str(path.relative_to(root)))
        pin, size, _ = _hash(path, deadline, clock)
        total += size
        need(total <= 512 << 20, "complete retained session exceeds 512MiB")
        members[name] = pin
    raw = "".join(pin + "  " + name + "\n" for name, pin in members.items()).encode()
    fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(fd, raw[offset:])
            need(count > 0 and clock() < deadline, "seal write/deadline failure")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)
    stamps = _tree(root, dict(members, **{"FINAL-ARTIFACTS.sha256": hashlib.sha256(raw).hexdigest()}), deadline, clock)
    seal_mtime = manifest.stat().st_mtime_ns
    need(root.stat().st_mtime_ns <= seal_mtime and all(Path(path).stat().st_mtime_ns <= seal_mtime for path in stamps),
         "result member/root newer than seal")
    completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elapsed = clock() - received_monotonic
    need(math.ceil(elapsed) <= complete_seconds and clock() <= deadline, "receipt-to-verified-seal ceiling exceeded")
    closeout = {"schema": 1, "root": str(root), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "member_count": len(members), "verified_completion_utc": completed, "received_utc": received_utc,
        "elapsed_seconds": elapsed, "charged_seconds": math.ceil(elapsed), "qualification": False}
    _publish(root.with_name(root.name + ".closeout.json"), closeout)
    return closeout


def require_retained_base_binding(record=None):
    """Bind the verified OCI derivation; current verification remains mandatory."""
    expected = retained_verifier.proof_record()
    need(type(record) is dict and set(record) == set(expected) and
        all(type(record[name]) is type(value) and record[name] == value for name, value in expected.items()),
        'NOT_READY: authenticated expanded base-rootfs inventory/check is absent or substituted')


def run_physical_controller(*, manifest_path, manifest_sha256, grant_path, grant_sha256, rpc):
    """Run real owner code only after the human parent's authenticated grant.

    CPU tests patch explicit operating-system/module boundaries; no effect
    substitution is offered by the physical CLI or accepted from its payload.
    """
    need((os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (1000, 1000, 1000, 1000),
         "physical workload owner must be UID/GID1000")
    manifest, _ = adapter.readout.read_json(manifest_path, manifest_sha256)
    grant, _ = adapter.readout.read_json(grant_path, grant_sha256)
    validate_physical_grant(grant, manifest, manifest_sha256, rpc.authenticate_parent())
    # The retained proof authenticates the effective SGLang tree and built
    # prefix, but not the expanded base containing Torch and the stdlib. OCI
    # layer provenance alone is not a fresh expanded-tree identity check.
    # Do not let a complete transport/grant silently bypass that missing gate.
    require_retained_base_binding(manifest.get('retained_base'))
    return _run_physical_after_retained_closure(manifest_path=manifest_path, manifest_sha256=manifest_sha256,
        grant_path=grant_path, grant=grant, manifest=manifest, rpc=rpc)


def _run_physical_after_retained_closure(*, manifest_path, manifest_sha256, grant_path, grant, manifest, rpc):
    """Unexposed projection; no caller reaches this before retained closure exists."""
    root = _canonical(manifest["target_root"])
    need(root == Path(manifest["package_root"]) and Path(manifest_path) == root / "MANIFEST.json" and
         Path(grant_path) == root / "GRANT.json" and Path(__file__).resolve() == root / "review/benchmark_physical_controller.py",
         "exact fresh physical root/entry/grant paths required")
    artifacts = _adjacent("spry_physical_artifacts", "benchmark_artifacts.py")
    output = root / "results"
    output.mkdir(mode=0o700, exist_ok=False)
    lease = adapter.owner.CanonicalLease("/home/developer/model-serving.izR2m1O9/diagnostic.lock",
        root, manifest["session_sha256"], manifest["boot_id"])
    captured, parsed, caps, supervised_results = {}, {}, {}, {}
    artifact_receipts, diagnostic_observations = {}, {}
    joins = {arm: {} for arm in ("bf16", "kv4")}
    known = {arm: [] for arm in ("bf16", "kv4")}
    with ExitStack() as cleanup:
        streams = {}
        def prepare_streams():
            # Pipes are created only after acquire. No sentinel ever occupies9.
            need(lease.fd == 9 and not streams, "capture requires fresh held canonical FD9")
            for arm in ("bf16", "kv4"):
                cap = artifacts.CapturePipes(output / (arm + "-capture"))
                caps[arm] = cap
                cleanup.callback(lambda cap=cap: cap.finish() if not cap.finished else None)
                streams[arm] = (cap.stdout, cap.stderr)

        def supervised(argv, **kwargs):
            arm = lease.record["arms"][-1]["arm"]
            result = None
            try:
                result = adapter.owner.supervise(argv, **kwargs)
            finally:
                try:
                    captured[arm] = caps[arm].finish()
                except Exception as error:
                    captured[arm] = {"capture_complete": False, "failure": type(error).__name__}
                    if result is None:
                        raise
                    result = replace(result, failure="bounded stream finalization failed")
            supervised_results[arm] = result
            return result

        def resource_check():
            for cap in caps.values():
                cap.check()
            if lease.record and lease.record.get("arms"):
                arm = lease.record["arms"][-1]["arm"]
                child = lease.record["arms"][-1].get("child")
                if child:
                    if not known[arm]:
                        known[arm].append(adapter.owner.ProcessIdentity(**child))
                    added = adapter.owner.descendants(tuple(known[arm]))
                    for item in added:
                        if not any(item.same_instance(old) for old in known[arm]):
                            known[arm].append(item)
                    need(len(known[arm]) <= 512, "live process join inventory exceeded")
                    for item in known[arm]:
                        record = observe_namespace_process(item)
                        if record is not None:
                            key = (item.pid, item.start_ticks)
                            old = joins[arm].get(key)
                            joins[arm][key] = {"first": old["first"] if old else record, "last": record}

        def parse(arm):
            if arm in parsed:
                return parsed[arm]
            need(captured[arm].get("capture_complete") is True, "bounded stream capture incomplete")
            stdout = adapter.readout.bench.read_bounded_file(output / (arm + "-capture/stdout.raw"), artifacts.STREAM_LIMIT)
            stderr = adapter.readout.bench.read_bounded_file(output / (arm + "-capture/stderr.raw"), artifacts.STREAM_LIMIT)
            binding = {"session_sha256": manifest["session_sha256"], "arm": arm, "spec_sha256": SPEC_SHA256}
            bundle = artifacts.receive_bundle(stdout, stderr, output / (arm + "-artifacts"), binding)
            artifact_receipts[arm] = bundle
            destination = output / (arm + "-artifacts")
            diagnostic_observations[arm] = diagnostic_readout(destination, binding, artifacts)
            _publish(output / (arm + "-diagnostics.json"), diagnostic_observations[arm])
            resources_raw = [line[len(b"SPRY_RESOURCE_V1 "):] for line in stdout.splitlines()
                             if line.startswith(b"SPRY_RESOURCE_V1 ")]
            need(len(resources_raw) == 2 and all(len(raw) <= 65536 for raw in resources_raw), "exact initial/final resource records required")
            records = [artifacts._json(raw) for raw in resources_raw]
            need([row.get("phase") for row in records] == ["initial", "final"], "resource record phase order")
            inner_path = output / (arm + "-artifacts") / "result.json"
            inner_raw = adapter.readout.bench.read_bounded_file(inner_path, 8 << 20)
            inner, _ = adapter.readout.read_json(inner_path, hashlib.sha256(inner_raw).hexdigest())
            # Failed arms retain raw tails and prefix counts above even when
            # final rank-result files or completion records are unavailable.
            need(bundle.get("all_members_present") is True, "complete bound raw result bundle required")
            diagnostics = diagnostic_readout(destination, binding, artifacts, inner)
            require_complete_diagnostics(diagnostics)
            checked_native_records(inner_path.parent, inner, arm=arm, session_sha256=manifest["session_sha256"],
                joins=[record for pair in joins[arm].values() for record in pair.values()],
                result=supervised_results[arm])
            parsed[arm] = (inner, {"initial": records[0], "final": records[1]})
            return parsed[arm]

        report = _run_bound_controller(manifest_path=manifest_path, manifest_sha256=manifest_sha256,
            grant=grant, lease=lease, observer=rpc.observe, read_inner_result=lambda arm: parse(arm)[0],
            read_resource_receipt=lambda arm: parse(arm)[1], resource_check=resource_check,
            streams=streams,
            supervisor=supervised, physical_context=grant, prepare_streams=prepare_streams)
        report["stream_captures"] = captured
        report["artifact_receipts"] = artifact_receipts
        report["diagnostic_observations"] = diagnostic_observations
        report["live_namespace_joins"] = {arm: list(rows.values()) for arm, rows in joins.items()}
        _publish(output / "CONTROLLER-RESULT.json", report)
    seal_result(root, deadline=grant["received_monotonic"] + 2400,
        received_monotonic=grant["received_monotonic"], received_utc=grant["received_utc"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Human-parent-bound UID1000 matched FP8 benchmark controller")
    parser.add_argument("--execute-owner-benchmark", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--grant-json")
    parser.add_argument("--grant-sha256")
    parser.add_argument("--observer-fd", type=int)
    args = parser.parse_args(argv)
    need(args.execute_owner_benchmark and all((args.manifest, args.manifest_sha256,
         args.grant_json, args.grant_sha256)) and args.observer_fd is not None,
         "NOT_READY: separate grant and authenticated inherited root socket required")
    grant, _ = adapter.readout.read_json(args.grant_json, args.grant_sha256)
    transport = _adjacent("spry_physical_root_transport", "benchmark_host_transport.py")
    rpc = transport.RpcClient(args.observer_fd, grant, args.manifest_sha256)
    report = run_physical_controller(manifest_path=args.manifest, manifest_sha256=args.manifest_sha256,
        grant_path=args.grant_json, grant_sha256=args.grant_sha256, rpc=rpc)
    return 0 if report["status"] == "MATCHED_PHYSICAL_BENCHMARK_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
