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
http_integration = _adjacent("spry_http_serving_integration", "serving_integration.py")
mtp_contract = _adjacent("spry_mtp_contract", "mtp_contract.py")
transport = _adjacent("spry_physical_root_transport", "benchmark_host_transport.py")
SUPPORTED_RUN_SHA256 = '48aab0c783f27717ae44be9753e064a08cb8c33bf0c175c5bcb302348868b7ab'
SUPPORTED_SETTINGS_SHA256 = '099148986068e3d2a01761036f78e4d76d3c366402bfe3ff1e95f5ce327c12c4'
SPEC_TEMPLATE_SHA256 = 'ac1b7afaf10e248d944a0d441ffa9abe30d72ed134860b93a7ec69aae09a07fd'
MODEL_INVENTORY_SHA256 = "892e95cdabf87bbf96298e258e5889e2572acf21659121d811ca9a09eac09986"
CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
if mtp_contract.precision_reference_enabled():
    MODEL_INVENTORY_SHA256 = mtp_contract.precision_profile()['inventory_sha256']
    CONFIG_SHA256 = mtp_contract.target_config_sha256()
BINARY_SHA256 = "1ca7efb0cf9220adb16a56e30e4cd2931b5d49edbe3277fc15d68e629500cd7b"
METADATA_SHA256 = "c33222c324bf841b32d71544450bbf8b418954d373b005f1d032c860f065b4c5"
SOURCE_NAMES = ("benchmark_physical_controller.py", "benchmark_owner_adapter.py",
    "host_process_owner.py", "benchmark_resource_gate.py", "benchmark_release_observer.py",
    "collect_benchmark_platform.py", "model_benchmark.py", "analyze_model_benchmark.py")
PACKAGE_NAMES = {"review/" + name for name in SOURCE_NAMES} | {
    "review/BENCHMARK-SPEC2.json", "review/RESOURCE-MANIFEST.json",
    "review/MODEL-INVENTORY.json", "review/COMMON-STACK-RECEIPT.json"}
LIVE_SOURCE_NAMES = SOURCE_NAMES + ("benchmark_gpu_identity.py", "benchmark_artifacts.py",
                                   "benchmark_host_transport.py",
                                   "verify_benchmark_runtime.py", "validation_comparison.py")
LIVE_SOURCE_NAMES += ('callback_profile.py', 'stall_trace.py')
LIVE_SOURCE_NAMES += ('graph_screen.py', 'graph_integration.py', 'graph_outer.py', 'graph_full_cuda_graph_backend.py', 'context_tokenizer.py')
LIVE_SOURCE_NAMES += ('serving_contract.py', 'serving_client.py', 'serving_reader.py', 'serving_integration.py',
                      'serving_projection.py', 'serving_entry.py', 'serving_launcher.py', 'worker_entry.py',
                      'serving_runtime.py', 'serving_graph_observer.py', 'mtp_contract.py', 'serving_parity.py')
if mtp_contract.j01_probe_enabled() or mtp_contract.precision_reference_enabled():
    # The HTTP entry launches native model_runner/sampler paths from its source
    # map; capture_sampler is used only by the separate historical layer screen.
    LIVE_SOURCE_NAMES = tuple(name for name in LIVE_SOURCE_NAMES if name != 'capture_sampler.py') + ('j01_probe.py',)
if mtp_contract.verify_attention_chain_enabled():
    # This native HTTP diagnostic never installs the historical logits overlay.
    # Exchange it for the adapter, preserving the transport member ceiling.
    LIVE_SOURCE_NAMES = tuple(name for name in LIVE_SOURCE_NAMES if name != 'capture_logits_processor.py') + ('verify_decode_adapter.py',)
if mtp_contract.norm_candidate_enabled() or mtp_contract.full_enabled_benchmark():
    # The HTTP norm mode authenticates the stock hybrid backend directly.
    # Reclaim only its inactive historical overlay for the candidate asset.
    LIVE_SOURCE_NAMES = tuple(name for name in LIVE_SOURCE_NAMES if name != 'capture_hybrid_linear_attn_backend.py')
if mtp_contract.full_enabled_benchmark():
    LIVE_SOURCE_NAMES = tuple(name for name in LIVE_SOURCE_NAMES if name != 'capture_qwen3_5.py')
    # This HTTP mode has no callback/stall instrumentation; retain these files
    # in authoring provenance while preserving the 64-member transport ceiling.
    LIVE_SOURCE_NAMES = tuple(name for name in LIVE_SOURCE_NAMES
                             if name not in ('callback_profile.py', 'stall_trace.py'))
LIVE_SOURCE_NAMES += ("layernorm_p3_candidate.py", "qwen3_5_candidate.py", "d3b_head_adapter.py", "d3b_combined_model_runner.py")
LIVE_SOURCE_NAMES += ('observer_health.py', 'target_kv_metadata.py',
                      'kv_graph_join.py', 'request_receipts.py')
LIVE_SOURCE_NAMES += ('p1_prefill.py', 'p1_fp8_kernels.py')
LIVE_SOURCE_NAMES += ('d3b_target_head_adapter.py', 'd3b_head_launches.py', 'd3b_head_fp8_ops.py', 'd3b_head_fp8_kernels.py')
LIVE_PACKAGE_NAMES = {"review/" + name for name in LIVE_SOURCE_NAMES} | {
    "review/BENCHMARK-SPEC2.json", "review/RESOURCE-MANIFEST.json", "review/MODEL-INVENTORY.json",
    "review/COMMON-STACK-RECEIPT.json", "review/COMPOSITION-FACTS.zlib"}
LIVE_PACKAGE_NAMES.update({'review/CONTEXT-PROMPTS.zlib', 'review/TOKENIZER-CONTRACT.json', 'review/CONTROL-CASE.zlib'})
LIVE_PACKAGE_NAMES.update({'review/HTTP-BINDING-control.json', 'review/HTTP-BINDING-mtp.json'})
LIVE_PACKAGE_NAMES.add('review/DRAFT-MODEL.json')
LIVE_PACKAGE_NAMES.update(mtp_contract.package_assets())
LIVE_PACKAGE_NAMES.difference_update('review/' + name for name in mtp_contract.PACKED_REQUEST_NAMES if name != 'FINAL-REQUESTS.zlib')
INDEPENDENT_MODE = 'independent-bf16-reference-v1'


def live_source_names(mode):
    return LIVE_SOURCE_NAMES + (('independent_runner.py',) if mode == INDEPENDENT_MODE else ())


def live_package_names(mode):
    if mode == INDEPENDENT_MODE:
        # The independent runner embeds its complete 427-token input. The
        # historical graph-context asset is never read by this explicit mode.
        return (LIVE_PACKAGE_NAMES - {'review/CONTEXT-PROMPTS.zlib'}) | {
            'review/independent_runner.py', 'review/INDEPENDENT-REFERENCE.json'}
    return set(LIVE_PACKAGE_NAMES)


FACTS_SHA256 = "d414db807ccec0f7c285be99cd040c86e0230521b03996f64f9bc4f9392d3072"
GPU_IDENTITY_SHA256 = "7324842c4f779749d561dc586168bd923e84e6900d17e3d22c140647bcd133df"
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


def retained_proof_attention_reader(spec):
    """Select the reader authenticated by the retained common-stack proof.

    The HTTP screen shares the graph-shell/native serving composition with the
    32K retrieval screen, although it intentionally does not enable that
    screen's capture-mode runtime behaviour. Keep this provenance decision
    separate from ``native_capture_mode``.
    """
    need(type(spec) is dict and type(spec.get("experiment_mode")) is str and
         type(spec.get("attention_reader")) is dict, "retained proof reader spec")
    if spec["experiment_mode"] in (
        "native-model-context-retrieval-32k-v1",
        "native-http-fp8-sprint01-v1",
    ):
        return adapter.readout.bench.comparison().REFERENCE
    return spec["attention_reader"]


def http_projection_contract(spec):
    """Return the explicit bwrap input for the DFlash2 screen."""
    if spec.get("experiment_mode") != "native-http-fp8-sprint01-v1":
        return None
    value = spec.get("http_serving")
    need(type(value) is dict and value.get("mode") == "native-http-fp8-sprint01-v1",
         "HTTP projection contract missing")
    return value


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


def authenticate_draft_model(package, package_files, *, deadline, clock=time.monotonic):
    """Bind the external two-file draft tree to the sealed package inventory."""
    name = 'review/DRAFT-MODEL.json'
    if mtp_contract.precision_reference_enabled() and not mtp_contract.full_enabled_benchmark():
        need(name not in package_files and all(mtp_contract.contract_for_arm(arm)['enabled'] is False
             for arm in ('control', 'mtp')), 'ordinary reference must exclude draft model and execution')
        return {}
    need(_digest(package_files.get(name)), 'DFlash2 draft package identity required')
    draft, _ = adapter.readout.read_json(Path(package) / name, package_files[name])
    need(draft == mtp_contract.draft_model_binding(), 'DFlash2 draft package/source binding differs')
    source, _ = adapter.draft_model_mount(draft)
    root = _canonical(source)
    before = root.lstat()
    need(stat.S_ISDIR(before.st_mode), 'DFlash2 draft directory required')
    rows = draft.get('files')
    need(type(rows) is list and len(rows) == 2 and all(type(row) is dict and
         set(row) == {'name', 'sha256', 'size'} and type(row['size']) is int and
         row['size'] > 0 for row in rows), 'DFlash2 exact two-file draft inventory required')
    inventory = {row['name']: row['sha256'] for row in rows}
    need(set(inventory) == {'config.json', 'model.safetensors'} and
         {path.name for path in root.iterdir()} == set(inventory), 'DFlash2 draft tree membership differs')
    stamps = _tree(root, inventory, deadline, clock)
    need(all(stamps[str(root / row['name'])][4] == row['size'] for row in rows),
         'DFlash2 draft payload sizes differ')
    config, _ = adapter.readout.read_json(root / 'config.json', inventory['config.json'])
    need(config == draft.get('config'), 'DFlash2 draft config metadata differs')
    need(_stamp(before) == _stamp(root.lstat()) and all(_stamp(Path(path).lstat()) == stamp
         for path, stamp in stamps.items()), 'DFlash2 draft changed while authenticating')
    stamps[str(root)] = _stamp(before)
    return stamps


def validate_manifest(path, expected_sha256, *, deadline, clock=time.monotonic, configuration_pins=None):
    """Authenticate complete local inputs; no runtime import or remote query."""
    value, _ = adapter.readout.read_json(path, expected_sha256)
    live = value.get("schema") == 2
    need(set(value) == FIELDS | ({"runtime_scopes", "retained_base", "logical_package_inventory", "run_config"} if live else set()) and
         value["schema"] in (1, 2) and
         value["scope"] == ("matched-benchmark-physical-package-v2" if live else
                            "matched-benchmark-controller-candidate-v1") and
         value["experiment_mode"] in ("model-benchmark-v1", "v4-repeatability-control-v1",
                                       "v4-first-divergence-capture-v1", "native-repeatability-capture-v1",
                                       "native-decode-layer-capture-v1", "native-prefill-row-audit-v1",
                                       "native-async-prefill-layer-capture-v1", "native-model-context-retrieval-32k-v1",
                                       "native-http-fp8-sprint01-v1", INDEPENDENT_MODE) and
         value["launch_authorized"] is False
         and type(value["proposed_complete_seconds"]) is int
         and 0 < value["proposed_complete_seconds"] <= transport.TOTAL_SECONDS, "preparation-only controller manifest")
    need(re.fullmatch(r"/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}", value["target_root"])
         and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["boot_id"])
         and _digest(value["session_sha256"]), "target/session binding")
    package = _canonical(value["package_root"])
    names = live_package_names(value['experiment_mode']) if live else PACKAGE_NAMES
    need(set(value["package_files"]) == names, "exact package source/assets required")
    if live:
        require_retained_base_binding(value.get('retained_base'))
        mtp_contract.validate_logical_package_inventory(value.get('logical_package_inventory'))
        # The review directory is immutable; receipts/arms live outside this
        # exact subtree and therefore cannot invalidate package membership.
        stamps = _tree(package / "review", {name.removeprefix("review/"): pin
            for name, pin in value["package_files"].items()}, deadline, clock)
    else:
        stamps = _tree(package, value["package_files"], deadline, clock)
    for name in (live_source_names(value['experiment_mode']) if live else SOURCE_NAMES):
        actual = Path(__file__).resolve().with_name(name)
        need(_hash(actual, deadline, clock)[0] == value["package_files"]["review/" + name],
             "executed author/source differs from package")
    need(value["package_files"]["review/collect_benchmark_platform.py"] == METADATA_SHA256,
         "metadata collector identity")
    spec_path = package / "review/BENCHMARK-SPEC2.json"
    need(live and value['experiment_mode'] == 'native-http-fp8-sprint01-v1',
         'configured W5 HTTP manifest required')
    configured = verify_configured_spec(configuration_pins, value, deadline=deadline, clock=clock)
    spec, spec_sha256 = configured['spec'], configured['spec_sha256']
    timing = configured['owner_timing']
    session_timing = configured['session_timing']
    stamps.update(configured['stamps'])
    adapter.readout.bench.validate_spec(spec)
    need(value["experiment_mode"] == spec["experiment_mode"], "manifest/spec experiment mode")
    independent = spec['experiment_mode'] == INDEPENDENT_MODE
    if independent:
        need(live and mtp_contract.precision_reference_enabled() and
             mtp_contract.precision_profile()['precision'] == 'bf16',
             'independent reference requires admitted genuine BF16 checkpoint')
        row = spec['independent_reference']
        need(value['package_files'].get(row['path']) == row['sha256'],
             'independent manifest/package binding')
        runner, independent_manifest = adapter.readout.bench.independent_runtime(spec)
        need(value['package_files'].get('review/independent_runner.py') ==
             independent_manifest['runner']['sha256'], 'independent runner/package binding')
        bindings = independent_manifest['bindings']
        need(bindings['runtime_facts']['sha256'] == value['package_files']['review/COMPOSITION-FACTS.zlib'] and
             bindings['gpu_guard']['sha256'] == value['package_files']['review/benchmark_gpu_identity.py'],
             'independent authenticated native helper bindings')
    elif spec["experiment_mode"] == "native-http-fp8-sprint01-v1":
        http = spec.get("http_serving")
        context = mtp_contract.phase_config()[0]
        need(spec.get('context_length') == spec.get('physical_slots') == context and
             spec.get('server_args', {}).get('context_length') == context and
             http.get('client_seconds') == session_timing.client_seconds, 'final phase context/client binding')
        expected_devices = {
            "0000:c3:00.0": {"uuid": "39373864-6465-3038-6163-303837666533", "render": "/dev/dri/renderD128"},
            "0000:23:00.0": {"uuid": "34663466-6561-3362-3832-346261366439", "render": "/dev/dri/renderD129"},
        }
        http_fields = {"schema", "mode", "bindings", "client_seconds", "expected_devices"}
        if mtp_contract.settings()['phase'] in mtp_contract.FAIL_FAST_PHASES or mtp_contract.precision_reference_enabled():
            http_fields.add('diagnostic_environment')
            need(http.get('diagnostic_environment') == {
                arm: mtp_contract.expected_environment(arm) for arm in ('control', 'mtp')},
                'frozen diagnostic environment differs')
        need(type(http) is dict and set(http) == http_fields and
             http["schema"] == 1 and http["mode"] == spec["experiment_mode"] and
             type(http["client_seconds"]) is int and 1 <= http["client_seconds"] <= 2040 and
             type(http["bindings"]) is dict and set(http["bindings"]) == {"control", "mtp"} and
             http["expected_devices"] == expected_devices,
             "frozen HTTP serving binding contract")
        for logical, row in http["bindings"].items():
            need(type(row) is dict and set(row) == {"path", "sha256"} and
                 row["path"] == "review/HTTP-BINDING-" + logical + ".json" and _digest(row["sha256"]) and
                 value["package_files"].get(row["path"]) == row["sha256"], "HTTP binding template package identity")
        projection = adapter.readout.bench._GRAPH_OUTER.validate_projection(spec["graph_projection"])
        need(all(value["package_files"].get("review/" + name) == pin for name, pin in projection["helpers"].items()),
             "HTTP graph-shell helper package binding")
        need(value["package_files"].get("review/" + projection["overlay"]["file"]) == projection["overlay"]["sha256"] ==
             spec["graph_runtime_sources"]["backend"]["sha256"], "HTTP graph backend overlay binding")
        stamps.update(authenticate_draft_model(package, value['package_files'], deadline=deadline, clock=clock))
    elif spec["experiment_mode"] == "native-model-context-retrieval-32k-v1":
        projection = adapter.readout.bench._GRAPH_OUTER.validate_projection(spec["graph_projection"])
        need(all(value["package_files"].get("review/" + name) == pin for name, pin in projection["helpers"].items()), "graph private helper package binding")
        need(value["package_files"].get("review/" + projection["overlay"]["file"]) == projection["overlay"]["sha256"] == spec["graph_runtime_sources"]["backend"]["sha256"], "read-only graph backend overlay binding")
    else:
        stall = spec.get('stall_trace')
        need(type(stall) is dict and stall['helper_sha256'] == value['package_files']['review/stall_trace.py'], 'stall trace helper binding')
        callback = spec.get('callback_profile')
        need(type(callback) is dict and callback['helper_sha256'] == value['package_files']['review/callback_profile.py'] and callback['capture_helper_sha256'] == value['package_files']['review/divergence_capture.py'], 'callback helper/counter source binding')
    comparison = spec.get("validation_comparison")
    if comparison is not None:
        module = adapter.readout.bench.comparison()
        module.validate_spec(spec)
        for name, row in comparison['overlay_files'].items():
            need(value["package_files"].get("review/" + name) == row["sha256"],
                 "exact candidate overlay input binding")
    need(spec["worker_seconds"] == timing.worker_seconds and spec["config_path"] == "/mnt/models/config.json"
         and spec["stack_receipt"] == "/mnt/review/COMMON-STACK-RECEIPT.json"
         and (independent or spec.get('package_root') == '/mnt/composition/image-source/python/sglang' and
              spec.get('server_args', {}).get('model_path') == '/mnt/models'), "unchanged inner paths/work bound")
    resource_path = package / "review/RESOURCE-MANIFEST.json"
    need(value["resource_manifest_sha256"] == value["package_files"]["review/RESOURCE-MANIFEST.json"],
         "resource manifest binding")
    resource_manifest = resources.read_manifest(resource_path, value["resource_manifest_sha256"])
    need(resource_manifest["root"] == value["target_root"] and
         resource_manifest["session_sha256"] == value["session_sha256"] and
         resource_manifest["benchmark_spec_sha256"] == spec_sha256 and
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
            spec_sha256=spec_sha256, bounded_scratch=True,
            **({"physical_artifacts": True, "session_sha256": value["session_sha256"],
                "artifacts_sha256": value["package_files"]["review/benchmark_artifacts.py"]} if live else {}))
        if comparison is not None:
            expected = adapter.bwrap_argv(arm, retained_root=retained, composition=runtime,
                review=Path(value["target_root"]) / "review", model=value["model_root"],
                output=Path(value["target_root"]) / "arms" / arm,
                entry=Path(value["target_root"]) / "review/model_benchmark.py",
                spec_sha256=spec_sha256, bounded_scratch=True, physical_artifacts=True,
                session_sha256=value["session_sha256"],
                artifacts_sha256=value["package_files"]["review/benchmark_artifacts.py"],
                comparison_contract=comparison)
        if spec["experiment_mode"] in ("native-model-context-retrieval-32k-v1", "native-http-fp8-sprint01-v1"):
            expected = adapter.bwrap_argv(arm, retained_root=retained, composition=runtime,
                review=Path(value["target_root"]) / "review", model=value["model_root"],
                output=Path(value["target_root"]) / "arms" / arm, entry=Path(value["target_root"]) / "review/model_benchmark.py",
                spec_sha256=spec_sha256, bounded_scratch=True, physical_artifacts=True,
                session_sha256=value["session_sha256"], artifacts_sha256=value["package_files"]["review/benchmark_artifacts.py"],
                graph_contract=spec["graph_projection"],
                http_contract=http_projection_contract(spec))
        if independent:
            expected = adapter.bwrap_argv(arm, retained_root=retained, composition=runtime,
                review=Path(value['target_root']) / 'review', model=value['model_root'],
                output=Path(value['target_root']) / 'arms' / arm,
                entry=Path(value['target_root']) / 'review/model_benchmark.py',
                spec_sha256=spec_sha256, bounded_scratch=True, physical_artifacts=True,
                session_sha256=value['session_sha256'],
                artifacts_sha256=value['package_files']['review/benchmark_artifacts.py'],
                independent_contract=spec['independent_reference'])
        expected = mtp_contract.replay_serving_sandbox_argv(expected, value['package_files'])
        need(resource_manifest["workload_argv"][arm] == list(expected), "exact read-only bounded-scratch projection")
    need(RUNTIME_REQUIRED <= set(value["runtime_files"]), "complete executable runtime source closure required")
    if live:
        need(value["runtime_scopes"] == list(RUNTIME_SCOPES), "fixed complete runtime closure scopes")
        need(value["package_files"]["review/COMPOSITION-FACTS.zlib"] == mtp_contract.FACTS_STORED['sha256'] and
             value["package_files"]["review/benchmark_gpu_identity.py"] == GPU_IDENTITY_SHA256,
             "authenticated composition/native identity source")
        facts_raw = mtp_contract.composition_facts_bytes()
        need(hashlib.sha256(facts_raw).hexdigest() == FACTS_SHA256, 'original decoded composition facts')
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
    retained_reader = None if independent else retained_proof_attention_reader(spec)
    need(proof.get("schema") == 2 and proof.get("serving_composition_complete") is True and
         proof.get("qualification") == "build-reviewed-not-serving-qualified" and
         proof.get("binary_sha256") == BINARY_SHA256 and
         (independent or proof.get("attention_reader") == retained_reader) and
         proof.get("stack_receipt_sha256") == value["package_files"]["review/COMMON-STACK-RECEIPT.json"]
         == spec["stack_receipt_sha256"] and value["runtime_files"][proof_name] == spec["build_proof_sha256"],
         "unchanged common-stack/build proof")
    inventory_name = _relative(value["model_inventory_file"])
    need(inventory_name == "review/MODEL-INVENTORY.json" and
         value["package_files"][inventory_name] == MODEL_INVENTORY_SHA256, "retained model inventory identity")
    raw = adapter.readout.bench.checked_file(package / inventory_name, MODEL_INVENTORY_SHA256)
    model_rows = json.loads(raw)
    expected_count = mtp_contract.precision_profile()['files'] if mtp_contract.precision_reference_enabled() else 81
    need(type(model_rows) is list and len(model_rows) == expected_count and all(type(row) is dict
         and set(row) == {"name", "size", "sha256"} for row in model_rows), "complete bound model inventory")
    model_files = {row["name"]: row["sha256"] for row in model_rows}
    need(len(model_files) == expected_count and [row["name"] for row in model_rows] == sorted(model_files)
         and model_files.get("config.json") == CONFIG_SHA256, "model names/config identity")
    model = _canonical(value["model_root"])
    if mtp_contract.precision_reference_enabled():
        profile = mtp_contract.precision_profile()
        need(str(model) == profile['model_root'] and
             sum(row['size'] for row in model_rows) == profile['bytes'],
             'precision reference model root/size binding differs')
    stamps.update(_tree(model, model_files, deadline, clock))
    need(all(type(row["size"]) is int and row["size"] >= 0 and
             (model / row["name"]).stat().st_size == row["size"] for row in model_rows), "model payload sizes")
    if independent and independent_manifest.get('torch_candidate') is not None:
        candidate = independent_manifest['torch_candidate']
        rows = runner.candidate_inventory(candidate)
        wheel = Path(candidate['wheel']['path'])
        stamps.update(_tree(wheel.parent, {wheel.name: candidate['wheel']['sha256']}, deadline, clock))
        stamps.update(_tree(_canonical(candidate['root']), {name: row['sha256'] for name, row in rows.items()}, deadline, clock))
        if candidate.get('sdk') is not None:
            sdk = candidate['sdk'];sdk_rows = runner.sdk_inventory(candidate, rows)
            source = Path(sdk['manifest']['path']);digest, size, stamp = _hash(source, deadline, clock)
            need(digest == sdk['manifest']['sha256'] and size == sdk['manifest']['bytes'], 'SDK inventory source pin')
            stamps[str(source)] = stamp
            stamps.update(_tree(_canonical(sdk['root']), {name: row['sha256'] for name, row in sdk_rows.items()}, deadline, clock))
    if mtp_contract.settings().get('serving_stack') is not None:
        stack=mtp_contract.serving_stack();guard=_adjacent('serving_stack_guard','benchmark_gpu_identity.py')
        candidate=stack['torch_candidate'];rows=guard.candidate_inventory(candidate)
        wheel=Path(candidate['wheel']['path']);stamps.update(_tree(wheel.parent,{wheel.name:candidate['wheel']['sha256']},deadline,clock))
        stamps.update(_tree(_canonical(candidate['root']),{n:r['sha256'] for n,r in rows.items()},deadline,clock))
        sdk=candidate['sdk'];sdk_rows=guard.sdk_inventory(candidate,rows)
        source=Path(sdk['manifest']['path']);digest,size,stamp=_hash(source,deadline,clock)
        need(digest==sdk['manifest']['sha256'] and size==sdk['manifest']['bytes'],'serving SDK manifest')
        stamps[str(source)]=stamp
        stamps.update(_tree(_canonical(stack['runtime_root']),{n:r['sha256'] for n,r in sdk_rows.items()},deadline,clock))
        for name,mode in stack['execution_modes'].items():
            need((Path(stack['runtime_root'])/name).stat().st_mode & 0o777 == mode,'SDK tool executable mode')
        compat=stack['compatibility'];stamps.update(_tree(_canonical(compat['sitepackages']),{n:r['sha256'] for n,r in compat['files'].items()},deadline,clock))
    if mtp_contract.settings().get('serving_stack') is not None:
        tc=stack['toolchain'];data=guard.strict_json(guard.checked(tc['manifest']))
        need(set(data)=={'schema','root','sdk_version','source','files'} and data['schema']==1 and data['root']==tc['root'] and data['sdk_version']==candidate['sdk']['version'],'development inventory identity')
        rows=data['files'];need(type(rows) is dict and 1<=len(rows)<=65536 and sum(row['bytes'] for row in rows.values())<=20<<30,'development inventory bounds')
        for name,row in rows.items():
            need(set(row)=={'bytes','sha256','mode'} and row['mode'] in (0o444,0o555),'development file schema')
            guard.binding(dict(path=tc['root']+'/'+name,bytes=row['bytes'],sha256=row['sha256']),allow_empty=True)
        stamps.update(_tree(_canonical(tc['root']),{name:row['sha256'] for name,row in rows.items()},deadline,clock))
        for name,row in rows.items():need((Path(tc['root'])/name).stat().st_mode & 0o777 == row['mode'],'development file mode')
        digest,size,stamp=_hash(Path(tc['manifest']['path']),deadline,clock);need(digest==tc['manifest']['sha256'] and size==tc['manifest']['bytes'],'development manifest current identity');stamps[tc['manifest']['path']]=stamp
    return {"manifest": value, "spec": spec, "spec_path": str(spec_path), "spec_sha256": spec_sha256,
            "owner_timing": timing, "session_timing": session_timing,
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
                   supervisor, clock=time.monotonic, physical_context=None, prepare_streams=None,
                   capture_directories=None):
    """Exercise the candidate composition with explicitly supplied CPU effects.

    This slice accepts only an inert preparation grant. It does not expose a
    live physical mode or invent an authenticated root observation channel.
    """
    started = clock() if physical_context is None else physical_context["received_monotonic"]
    report = {"schema": 1, "scope": "benchmark-physical-controller-candidate-v1",
        "status": "INCOMPLETE", "qualification": False, "physical_ready": False,
        "launch_authorized": False, "native_run": False, "remaining_gaps": list(GAPS),
        "proposed_complete_seconds": 6360, "admission_seconds": 180,
        "source_preparation_seconds": 105, "locked_admission_seconds": 75,
        "arm_envelope_seconds": 2490, "collection_reserve_seconds": 330,
        "seal_reserve_seconds": 300, "failure": None, "observer_requests": [], "resource_receipts": []}
    try:
        if physical_context is None:
            need(type(grant) is dict and grant == {"schema": 1, "scope": "inert-controller-cpu-composition",
                "launch_authorized": False, "maximum_starts": 0, "proposed_complete_seconds": 6360,
                "manifest_sha256": manifest_sha256}, "unfunded proposal permits inert CPU composition only")
            need(supervisor is not adapter.owner.supervise, "inert composition cannot use the live supervisor")
        else:
            need(capture_directories is None, 'physical capture paths are owner-derived')
            need(grant == physical_context and grant["schema"] == 2, "validated physical context required")
            report["remaining_gaps"] = []
            report["scope"] = "matched-physical-model-benchmark-v2"
        need(all(callable(item) for item in (observer, read_inner_result, read_resource_receipt,
             resource_check, supervisor)), "complete explicit owner effects required")
        prepared = validate_manifest(manifest_path, manifest_sha256,
                                     deadline=started + transport.SOURCE_PREPARATION_SECONDS, clock=clock,
                                     configuration_pins=physical_context)
        spec_sha256 = prepared['spec_sha256']
        timing = prepared['owner_timing']
        session_timing = prepared['session_timing']
        report.update(proposed_complete_seconds=session_timing.total_seconds,
                      admission_seconds=session_timing.admission_seconds,
                      source_preparation_seconds=session_timing.source_preparation_seconds,
                      seal_reserve_seconds=session_timing.seal_reserve_seconds,
                      arm_envelope_seconds=timing.arm_envelope_seconds,
                      locked_admission_seconds=timing.locked_admission_seconds,
                      collection_reserve_seconds=timing.collection_reserve_seconds)
        value, results = prepared["manifest"], []
        if physical_context is not None:
            prepared['stamps'].update(authenticate_configuration_files(
                grant, value, deadline=started + session_timing.source_preparation_seconds, clock=clock))
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
            need(deadline_monotonic <= started + session_timing.admission_seconds,
                 "locked admission exceeds complete preparation bound")
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
                message = str(error)[:512]
                report.setdefault("failure_observations", []).append({"stage": "resource_receipt",
                    "arm": arm, "exception_type": type(error).__name__, "message": message,
                    "prior_supervision_failure": result.failure})
                result = replace(result, failure="resource receipt: " + type(error).__name__ + ": " + message)
                results[-1] = result
            return result

        frozen_arms = tuple(adapter.FrozenArm(arm,
            resources.scope_argv(prepared["resource_path"], value["resource_manifest_sha256"], arm),
            prepared["resource_manifest"]["gate_path"], prepared["resource_manifest"]["gate_sha256"])
            for arm in ("bf16", "kv4"))
        campaign = adapter.run_campaign(lease=lease, arms=frozen_arms,
            spec_path=prepared["spec_path"], spec_sha256=spec_sha256,
            collector_sha256=observer_sha, active_deadline=started + session_timing.active_seconds,
            collect_release=lambda current, arm, result, end: observe("release", current, arm, result, end),
            read_inner_result=read_inner_result,
            independent_postcheck=lambda current, end: observe("independent", current, None, results[-1], end),
            streams=streams, resource_check=resource_check, supervisor=supervise, clock=clock,
            timing=timing, admission_check=admission,
            prepare_streams=prepare_streams)
        report["campaign"] = campaign
        verify_unchanged(prepared)
        need(campaign.get("adapter_complete") is True and campaign.get("unresolved") is False
             and campaign.get("independent_release_verified") is True, "candidate campaign incomplete")
        if prepared['spec'].get('experiment_mode') == INDEPENDENT_MODE:
            independent_arms = {arm: read_inner_result(arm) for arm in ('bf16', 'kv4')}
            report.update(experiment_mode=INDEPENDENT_MODE, qualification=False,
                source_promotion=False, serving_quality_qualified=False, http_throughput_measured=False,
                independent_reference={arm: dict(status=inner['status'],
                    model_execution_performed=inner['model_execution_performed'],
                    result_canonical_sha256=hashlib.sha256((json.dumps(inner, sort_keys=True,
                        separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest())
                    for arm, inner in independent_arms.items()})
        elif prepared["spec"].get("experiment_mode") == "native-http-fp8-sprint01-v1":
            arms = {arm: read_inner_result(arm) for arm in ("bf16", "kv4")}
            campaign["experiment_mode"] = prepared["spec"]["experiment_mode"]
            campaign["http_bindings"] = {arm: value["binding"] for arm, value in arms.items()}
            report.update(_http_serving_report(arms, campaign))
            report.update(experiment_mode=prepared["spec"]["experiment_mode"], qualification=False,
                          source_promotion=False, serving_quality_qualified=False,
                          http_throughput_measured=True)
        elif prepared["spec"].get("experiment_mode") == "native-model-context-retrieval-32k-v1":
            graph_directories = {arm: Path(value["target_root"]) / "results" / (arm + "-artifacts") for arm in ("bf16", "kv4")}
            report["graph_potential"] = adapter.readout.analyze(prepared["spec"], spec_sha256,
                read_inner_result("bf16"), read_inner_result("kv4"), graph_directories)
            report.update(experiment_mode=prepared["spec"]["experiment_mode"], qualification=False, source_promotion=False)
        if prepared["spec"].get("validation_comparison") is not None:
            left, right = (read_inner_result(arm) for arm in ("bf16", "kv4"))
            directories = (capture_directories if capture_directories is not None else
                {arm: Path(value['target_root']) / 'results' / (arm + '-artifacts') for arm in ('bf16', 'kv4')})
            analysis = adapter.readout.analyze(prepared["spec"], spec_sha256, left, right,
                directories if adapter.readout.bench.capture_mode(prepared['spec']) else None)
            report.update(experiment_mode=prepared["spec"]["experiment_mode"],
                arm_labels=prepared["spec"]["validation_comparison"]["arm_labels"],
                exact_output_gate=analysis["exact_output_gate"],
                telemetry_available=analysis["telemetry_available"], source_promotion=False)
            if adapter.readout.bench.capture_mode(prepared['spec']):
                report['capture'] = analysis['capture']
                report['capture_complete'] = analysis['capture']['records_complete'] and campaign['independent_release_verified']
            if 'prefill_row_audit' in analysis:
                report['prefill_row_audit'] = analysis['prefill_row_audit']
            if adapter.readout.bench.native_capture_mode(prepared['spec']):
                report['native_repeatability_gate'] = analysis['native_repeatability_gate']
                need(analysis['native_repeatability_gate']['passed'] is True,
                     'native comparison repeatability mismatch')
            need(analysis["exact_output_gate"]["passed"] is True,
                 'native comparison numerical mismatch' if adapter.readout.bench.native_capture_mode(prepared['spec'])
                 else 'compact comparison numerical mismatch')
        report["status"] = ("MATCHED_PHYSICAL_BENCHMARK_COMPLETE" if physical_context is not None
                            else "LOCAL_CONTROLLER_COMPOSITION_PASSED")
    except Exception as error:
        report["failure"] = type(error).__name__ + ": " + str(error)[:1024]
    report["elapsed_seconds"] = clock() - started
    return report


def mapping_sha256(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()


GRANT_FIELDS = {"schema", "scope", "model_start_authorized", "maximum_starts", "arm_order",
    "cumulative_ceiling_seconds", "fixed_reserve_seconds", "received_utc", "received_monotonic",
    "boot_id", "manifest_sha256", "package_sha256", "source_manifest_sha256", "transport_parent",
    "run_config_sha256", "host_sha256", "protected_set_sha256"}

PROTECTED_SOURCE_NAMES = (
    'benchmark_artifacts.py', 'benchmark_gpu_identity.py', 'benchmark_host_transport.py',
    'benchmark_owner_adapter.py', 'benchmark_physical_controller.py',
    'benchmark_release_observer.py', 'benchmark_resource_gate.py',
    'collect_benchmark_platform.py', 'host_process_owner.py')


def _configuration_file(path, expected_sha256, maximum, *, deadline, clock, stamps, json_object=True):
    """Read bounded, canonical approval inputs without importing their contents."""
    need(_digest(expected_sha256) and clock() < deadline, 'configuration pin/deadline required')
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'canonical configuration file required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum,
             'bounded singly linked configuration file required')
        chunks, remaining = [], before.st_size
        while remaining:
            need(clock() < deadline, 'configuration authentication deadline')
            chunk = os.read(fd, min(1 << 20, remaining))
            need(chunk, 'short configuration input')
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b''.join(chunks)
        need(path.resolve(strict=True) == path and
             _stamp(before) == _stamp(os.fstat(fd)) == _stamp(path.lstat()), 'configuration input changed')
        need(hashlib.sha256(raw).hexdigest() == expected_sha256, 'configuration/source digest mismatch')
        stamps[str(path)] = _stamp(before)
    finally:
        os.close(fd)
    need(clock() < deadline, 'configuration authentication deadline')
    if not json_object:
        return raw
    value = json.loads(raw, object_pairs_hook=adapter.readout._object, parse_constant=adapter.readout._constant)
    need(type(value) is dict and raw == (json.dumps(value, sort_keys=True, separators=(',', ':'),
         allow_nan=False) + '\n').encode(), 'canonical configuration JSON object required')
    return value


def authenticate_configuration_files(grant, manifest, *, deadline, clock=time.monotonic):
    """Bind independently approved run, host, menu and protected source bytes."""
    need(type(grant) is dict and all(_digest(grant.get(key)) for key in
         ('run_config_sha256', 'host_sha256', 'protected_set_sha256')), 'three configuration approval pins required')
    root = _canonical(manifest['package_root'])
    need(Path(__file__).resolve(strict=True) == root / 'review/benchmark_physical_controller.py',
         'configuration must bind the executed controller root')
    run_name = manifest.get('run_config')
    need(type(run_name) is str and re.fullmatch(r'config/runs/[a-z0-9][a-z0-9-]{0,63}/run.json', run_name),
         'exact run configuration location required')
    stamps = {}
    protected = _configuration_file(root / 'PROTECTED-SHA256.json', grant['protected_set_sha256'],
        64 << 10, deadline=deadline, clock=clock, stamps=stamps)
    names = {'review/' + name for name in PROTECTED_SOURCE_NAMES}
    need(set(protected) == names and all(_digest(pin) for pin in protected.values()),
         'exact protected file set required')
    for name, pin in protected.items():
        need(manifest['package_files'].get(name) == pin, 'protected set/package binding differs')
        _configuration_file(root / name, pin, 1 << 20, deadline=deadline, clock=clock,
                            stamps=stamps, json_object=False)
    run = _configuration_file(root / run_name, grant['run_config_sha256'], 256 << 10,
        deadline=deadline, clock=clock, stamps=stamps)
    need(type(run.get('schema')) is int and run['schema'] == 1
         and run.get('run_id') == Path(run_name).parent.name, 'run configuration identity differs')
    host, menu = run.get('host'), run.get('menu')
    need(type(host) is dict and set(host) == {'file', 'sha256'}
         and type(host['file']) is str and re.fullmatch(r'config/host/[a-z0-9][a-z0-9-]{0,63}\.json', host['file'])
         and host['sha256'] == grant['host_sha256'], 'run/host approval binding differs')
    need(type(menu) is dict and set(menu) == {'file', 'sha256'} and menu['file'] == 'menu.json'
         and _digest(menu['sha256']), 'run/menu binding differs')
    host_value = _configuration_file(root / host['file'], grant['host_sha256'], 64 << 10,
        deadline=deadline, clock=clock, stamps=stamps)
    need(set(host_value) == {'schema', 'devices', 'rank_indices', 'power_cap_watts'}
         and type(host_value['schema']) is int and host_value['schema'] == 1,
         'host configuration fields differ; boot ID belongs in the approval')
    _configuration_file((root / run_name).parent / menu['file'], menu['sha256'], 1 << 20,
        deadline=deadline, clock=clock, stamps=stamps)
    # Include immutable parent directories so a later config/review symlink or
    # directory replacement is refused by the existing verify_unchanged calls.
    for name in tuple(stamps):
        parent = Path(name).parent
        while parent != root:
            need(parent.resolve(strict=True) == parent and parent.is_dir(), 'configuration parent changed')
            stamps[str(parent)] = _stamp(parent.lstat())
            parent = parent.parent
    verify_unchanged({'stamps': stamps})
    need(clock() < deadline, 'configuration authentication deadline')
    return stamps


def render_configured_spec(template, run, host, package_files):
    """Reproduce the supported W5 spec from a fingerprinted template and inputs.

    Only the listed slots can vary with configuration/source bindings. Hashing
    the remainder authenticates every other field, including unknown fields.
    The pinned run is a temporary compatibility boundary until all consumers
    support configuration changes; owner ceilings remain separate literals.
    """
    need(mapping_sha256(run) == SUPPORTED_RUN_SHA256, 'unsupported run configuration')
    need(mapping_sha256(host) == run['host']['sha256'], 'configured host identity differs')
    expected = json.loads(json.dumps(template, allow_nan=False))
    slots = [(('worker_seconds',), run['limits']['client_seconds'] + run['limits']['worker_startup_seconds']),
             (('context_length',), run['profile']['context_length']),
             (('physical_slots',), run['profile']['context_length']),
             (('gpu_ids',), host['rank_indices']),
             (('http_serving', 'client_seconds'), run['limits']['client_seconds']),
             (('http_serving', 'expected_devices'), host['devices'])]
    slots += [(('server_args', key), value) for key, value in run['profile'].items()]
    sources = [(('http_serving', 'bindings', arm, 'sha256'), 'HTTP-BINDING-' + arm + '.json')
               for arm in ('control', 'mtp')]
    sources += [(('graph_projection', 'helpers', name), name) for name in
                ('context_tokenizer.py', 'graph_integration.py', 'graph_outer.py', 'graph_screen.py')]
    for path, name in sources:
        pin = package_files.get('review/' + name)
        need(_digest(pin), 'configured spec source pin required')
        slots.append((path, pin))
    for path, value in slots:
        parent = expected
        for key in path[:-1]:
            need(type(parent) is dict and key in parent, 'spec template shape differs')
            parent = parent[key]
        need(type(parent) is dict and path[-1] in parent, 'spec template slot missing')
        parent[path[-1]] = None
    need(mapping_sha256(expected) == SPEC_TEMPLATE_SHA256, 'fixed spec template differs')
    for path, value in slots:
        parent = expected
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
    return expected


def verify_configured_spec(grant, manifest, *, deadline, clock=time.monotonic):
    """Authenticate inputs and compare complete spec bytes before owner effects."""
    stamps = authenticate_configuration_files(grant, manifest, deadline=deadline, clock=clock)
    initial_stamps = dict(stamps)
    root = _canonical(manifest['package_root'])
    run = _configuration_file(root / manifest['run_config'], grant['run_config_sha256'],
        256 << 10, deadline=deadline, clock=clock, stamps=stamps)
    host = _configuration_file(root / run['host']['file'], grant['host_sha256'],
        64 << 10, deadline=deadline, clock=clock, stamps=stamps)
    files = manifest['package_files']
    settings_pin = files.get('review/FINAL-SETTINGS.json')
    need(settings_pin == SUPPORTED_SETTINGS_SHA256, 'supported W5 settings required')
    _configuration_file(root / 'review/FINAL-SETTINGS.json', settings_pin, 256 << 10,
        deadline=deadline, clock=clock, stamps=stamps)
    for name in ('HTTP-BINDING-control.json', 'HTTP-BINDING-mtp.json', 'context_tokenizer.py',
                 'graph_integration.py', 'graph_outer.py', 'graph_screen.py'):
        _configuration_file(root / 'review' / name, files.get('review/' + name),
            64 << 10 if name.endswith('.json') else 2 << 20, deadline=deadline, clock=clock,
            stamps=stamps, json_object=name.endswith('.json'))
    pin = manifest.get('spec_sha256')
    need(_digest(pin) and pin == files.get('review/BENCHMARK-SPEC2.json'), 'spec/package identity differs')
    spec = _configuration_file(root / 'review/BENCHMARK-SPEC2.json', pin, 64 << 10,
        deadline=deadline, clock=clock, stamps=stamps)
    expected = render_configured_spec(spec, run, host, files)
    # Canonical input was already required above. The final digest comparison
    # checks all bytes, so bool/int aliases, field omissions and extra fields fail.
    need(mapping_sha256(expected) == pin, 'configuration-generated spec differs')
    session_timing = transport.SessionTiming.from_limits(run['limits'])
    need(type(manifest.get('proposed_complete_seconds')) is int
         and manifest['proposed_complete_seconds'] == session_timing.total_seconds,
         'manifest/session timing differs')
    need(expected['worker_seconds'] == session_timing.worker_seconds, 'spec/session worker timing differs')
    timing = adapter.CampaignTiming(worker_seconds=session_timing.worker_seconds,
        owner_cleanup_seconds=session_timing.owner_cleanup_seconds,
        arm_release_seconds=session_timing.arm_release_seconds,
        locked_admission_seconds=session_timing.locked_admission_seconds,
        collection_reserve_seconds=session_timing.collection_reserve_seconds)
    verify_unchanged({'stamps': initial_stamps})
    verify_unchanged({'stamps': stamps})
    need(clock() < deadline, 'configured spec verification deadline')
    return {'spec': expected, 'spec_sha256': pin, 'stamps': stamps, 'owner_timing': timing,
            'session_timing': session_timing}


def validate_physical_grant(grant, manifest, manifest_sha256, parent_identity, *,
                            clock=time.monotonic, utc_now=lambda: datetime.now(timezone.utc)):
    need(type(grant) is dict and set(grant) == GRANT_FIELDS and grant["schema"] == 2 and
         grant["scope"] == "one-matched-fp8-kv-benchmark" and grant["model_start_authorized"] is True and
         type(grant["maximum_starts"]) is int and grant["maximum_starts"] == 2 and
         grant["arm_order"] == ["bf16", "kv4"] and type(grant["cumulative_ceiling_seconds"]) is int
         and 0 < grant["cumulative_ceiling_seconds"] <= transport.TOTAL_SECONDS
         and type(grant["fixed_reserve_seconds"]) is int
         and grant["fixed_reserve_seconds"] == transport.SEAL_RESERVE, "exact separate two-start physical grant required")
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
         received <= clock() < received + transport.SOURCE_PREPARATION_SECONDS, "source/admission receipt window expired")
    need(type(grant["received_utc"]) is str and grant["received_utc"].endswith("Z"), "original UTC receipt required")
    instant = datetime.fromisoformat(grant["received_utc"].replace("Z", "+00:00"))
    need(instant.tzinfo is not None and abs((utc_now() - instant).total_seconds() - (clock() - received)) <= 3,
         "UTC/monotonic receipt disagreement")
    configured = verify_configured_spec(grant, manifest,
        deadline=received + transport.SOURCE_PREPARATION_SECONDS, clock=clock)
    timing = configured['session_timing']
    need(grant['cumulative_ceiling_seconds'] == timing.total_seconds
         and grant['fixed_reserve_seconds'] == timing.seal_reserve_seconds,
         'grant/configured session timing differs')
    timing.require_start_window(received, clock=clock)
    return grant


def checked_physical_resources(receipts, prepared, arm, result):
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
         initial.get("benchmark_spec_sha256") == prepared['spec_sha256'] and final.get("complete") is True and
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
    try:
        raw = adapter.readout.bench.read_bounded_file(Path(proc_root) / str(identity.pid) / "status", 65536)
    except (FileNotFoundError, ProcessLookupError):
        # A process can exit after the authenticated identity read but before
        # the separate NSpid status read. Re-observe it: absence is safe, but
        # a live or replaced PID must remain a refusal rather than a guessed
        # namespace mapping.
        after = adapter.owner.read_process(identity.pid, str(proc_root))
        if after is None:
            return None
        if not identity.same_instance(after): raise ValueError(("process changed during namespace observation stage=status_missing pid=" + str(identity.pid) + " fields=" + ",".join(k for k in ("boot_id", "pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace") if getattr(identity, k) != getattr(after, k)) + " values=" + ";".join(k + ":" + repr(getattr(identity, k))[:24] + "->" + repr(getattr(after, k))[:24] for k in ("boot_id", "pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace") if getattr(identity, k) != getattr(after, k)))[:500])
        raise ValueError("live process status unavailable during namespace observation")
    rows = [line.split(b":", 1)[1].split() for line in raw.splitlines() if line.startswith(b"NSpid:")]
    need(len(rows) == 1 and 1 <= len(rows[0]) <= 32 and all(value.isdigit() for value in rows[0]),
         "actual bounded NSpid mapping unavailable")
    nspid = [int(value) for value in rows[0]]
    need(nspid[0] == identity.pid and all(value > 0 for value in nspid), "host/private PID mapping malformed")
    after = adapter.owner.read_process(identity.pid, str(proc_root))
    if after is None:
        return None
    if not before.same_instance(after): raise ValueError(("process changed during namespace observation stage=status_read pid=" + str(identity.pid) + " fields=" + ",".join(k for k in ("boot_id", "pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace") if getattr(before, k) != getattr(after, k)) + " values=" + ";".join(k + ":" + repr(getattr(before, k))[:24] + "->" + repr(getattr(after, k))[:24] for k in ("boot_id", "pid", "pgid", "sid", "start_ticks", "uid", "gid", "pid_namespace") if getattr(before, k) != getattr(after, k)))[:500])
    return {"host_process": asdict(before), "nspid": nspid, "observed_monotonic": clock()}


def checked_native_records(directory, inner, *, arm, session_sha256, spec_sha256, joins, result):
    need(inner.get("physical_artifacts") is True and inner.get("session_sha256") == session_sha256 and
         inner.get("native_identity_verified") is True and len(inner.get("ranks", [])) == 2,
         "physical native identity completion required")
    identities = []
    for rank in (0, 1):
        path = Path(directory) / f"rank-{rank}-identity.json"
        raw = adapter.readout.bench.read_bounded_file(path, 64 << 10)
        value, _ = adapter.readout.read_json(path, hashlib.sha256(raw).hexdigest())
        adapter.readout.bench.validate_rank_identity(value, rank=rank, arm=arm,
            session_sha256=session_sha256, spec_sha256=spec_sha256)
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


def checked_independent_records(spec, inner, *, arm, session_sha256, joins, result,
                                artifacts_sha256, spec_sha256):
    """Join the one real native Python owner to live host observations."""
    adapter.readout.bench.checked_independent_result(spec, inner, arm=arm,
        session_sha256=session_sha256, spec_sha256=spec_sha256)
    native = inner.get('native_identity')
    need(type(native) is dict and native.get('before_weights') is True and
         native.get('helper_sha256') == GPU_IDENTITY_SHA256, 'independent preweight identity required')
    process = native.get('process')
    adapter.readout.bench.validate_native_process(process)
    helper = adapter.readout.bench.authenticated_adjacent('benchmark_gpu_identity.py', GPU_IDENTITY_SHA256)
    adapter.readout.bench.validate_physical_details(helper, native.get('devices'), native.get('hip'))
    command = ['/mnt/work/build-venv/bin/python3', '-I', '-B', '/mnt/review/model_benchmark.py',
        '--execute-reviewed-benchmark', '--spec', '/mnt/review/BENCHMARK-SPEC2.json',
        '--spec-sha256', spec_sha256, '--output', '/mnt/output/benchmark', '--arm', arm,
        '--physical-artifacts', '--session-sha256', session_sha256,
        '--artifacts-sha256', artifacts_sha256]
    command_sha = hashlib.sha256(b'\0'.join(value.encode() for value in command) + b'\0').hexdigest()
    matching = [row for row in joins if type(row) is dict and type(row.get('nspid')) is list and
        len(row['nspid']) >= 2 and row['nspid'][-1] == process['pid'] and
        type(row.get('host_process')) is dict and
        all(row['host_process'].get(key) == process[key] for key in
            ('boot_id', 'start_ticks', 'pid_namespace', 'uid', 'gid')) and
        row['host_process'].get('command_sha256') == command_sha]
    need(matching and len({row['host_process']['pid'] for row in matching}) == 1 and any(
        any(item.pid == row['host_process']['pid'] and item.start_ticks == process['start_ticks'] and
            item.boot_id == process['boot_id'] and item.command_sha256 == command_sha and
            item.pid_namespace == process['pid_namespace'] for item in result.observed_processes)
        for row in matching), 'independent native Python lacks authenticated live host process join')
    return native


def http_expected_binding(root, spec, *, owner_arm, session_sha256, spec_sha256):
    """Read a frozen source template before accepting a runtime readiness gate."""
    need(_digest(spec_sha256) and mapping_sha256(spec) == spec_sha256, 'verified HTTP spec identity required')
    logical = mtp_contract.owner_arm_map()[owner_arm]
    row = spec["http_serving"]["bindings"][logical]
    path = Path(root) / row["path"]
    raw = adapter.readout.bench.read_bounded_file(path, 64 << 10)
    need(hashlib.sha256(raw).hexdigest() == row["sha256"], "HTTP binding template changed")
    value = json.loads(raw, object_pairs_hook=adapter.readout._object,
                       parse_constant=adapter.readout._constant)
    value = mtp_contract.expand_http_template(value)
    expected_devices = spec["http_serving"]["expected_devices"]
    need(type(value) is dict and set(value) == {"schema", "mode", "arm", "source_manifest_sha256", "port",
         "binding_template", "worker_template", "speculation"} and value["schema"] == 1 and
         value["mode"] == "native-http-fp8-sprint01-v1" and value["arm"] == logical and
         value["port"] == {"control": 18087, "mtp": 18088}[logical] and _digest(value["source_manifest_sha256"]) and
         mtp_contract.validate_contract(value["speculation"]) == mtp_contract.contract_for_arm(logical) and
         type(value["binding_template"]) is dict and type(value["worker_template"]) is dict and
         value["binding_template"].get("speculation") == value["speculation"] and
         value["binding_template"].get("source_manifest_sha256") == value["source_manifest_sha256"] and
         value["worker_template"].get("source_manifest_sha256") == value["source_manifest_sha256"] and
         value["binding_template"].get("arm") == logical and value["worker_template"].get("arm") == logical and
         value["binding_template"].get("session_sha256") is None and value["binding_template"].get("spec_sha256") is None and
         value["worker_template"].get("session_sha256") is None and value["worker_template"].get("spec_sha256") is None and
         type(value["binding_template"].get("sources")) is dict and type(value["worker_template"].get("sources")) is dict and
         value["binding_template"]["sources"] == value["worker_template"]["sources"] and
         hashlib.sha256(json.dumps(value["binding_template"]["sources"], sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest() == value["source_manifest_sha256"] and
         value["worker_template"].get("expected_devices") == expected_devices,
         "HTTP binding template schema")
    return {"arm": logical, "source_manifest_sha256": value["source_manifest_sha256"],
            "session_sha256": session_sha256, "spec_sha256": spec_sha256}


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
             workers["worker_seconds"] == 2280 and type(workers["pids"]) is list and
             len(workers["pids"]) == 2 and all(type(pid) is int and pid > 1 for pid in workers["pids"]) and
             len(set(workers["pids"])) == 2, "worker diagnostic inventory")
        for rank, row in enumerate(value["rank_progress"]):
            need(row["valid"] and row["pid"] == workers["pids"][rank], "progress identity/inventory")
        if binding.get('experiment_mode') == 'native-decode-layer-capture-v1':
            capture_contract = bench.comparison().decode_layer_capture_contract()
            helper = bench.authenticated_adjacent('divergence_capture.py', capture_contract['capture']['helper_sha256'])
            value['operation_markers'] = []
            for rank in (0, 1):
                native = artifacts._json(bench.read_bounded_file(directory / f'rank-{rank}-identity.json', 64 << 10))
                bench.validate_rank_identity(native, rank=rank, **{k: binding[k] for k in ('arm', 'session_sha256', 'spec_sha256')})
                expected = bench.capture_binding(dict(experiment_mode=binding['experiment_mode'],
                    validation_comparison=capture_contract), native)
                value['operation_markers'].append(helper.read_operation(directory / f'rank-{rank}-operation.json', expected))
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
    if value.get('binding', {}).get('experiment_mode') == 'native-decode-layer-capture-v1':
        need(len(value.get('operation_markers', [])) == 2 and all(row['phase'] == 'capture_closed'
            and row['complete'] is True for row in value['operation_markers']), 'operation markers incomplete')
    need(value.get("valid") is True and value.get("qualification") is False and
         len(value.get("rank_progress", [])) == 2 and all(
             row["valid"] and row["finished"] and row["counts_complete"] and
             row["intent_batches"] == row["completed_batches"] == 7 and
             row["intent_requests"] == row["completed_requests"] == 7
             for row in value["rank_progress"]), "complete rank diagnostic evidence required")


def _http_serving_report(arms, campaign):
    """Terminal reporting boundary used by the real controller and CPU regression."""
    need(type(arms) is dict and set(arms) == {'bf16', 'kv4'}, 'both HTTP arms required')
    accepted = {arm: http_integration.accept_with_owner(arms[arm], campaign, owner_arm=arm)
                for arm in ('bf16', 'kv4')}
    return dict(http_serving_report_format='retained-complete-http-wire-reference-v1',
                http_serving={arm: _http_wire_reference(accepted[arm], owner_arm=arm)
                              for arm in ('bf16', 'kv4')})


def _http_wire_reference(value, *, owner_arm):
    """Call only after the unchanged accept_with_owner; keep raw evidence intact."""
    if (owner_arm not in ('bf16','kv4') or type(value) is not dict or
            value.get('status') != 'SERVING_SCREEN_COMPLETE' or
            value.get('wire_complete') is not True or
            value.get('qualification') is not False or
            value.get('independent_release_required') is not False or
            value.get('arm') != {'bf16':'control','kv4':'mtp'}[owner_arm]):
        raise ValueError('complete owner-validated composed HTTP wire required')
    rows=value.get('rows')
    settings = mtp_contract.settings()
    if settings.get('quality_screen') == 'normal-stop-fp8-mxfp4-v1':
        planned = settings['requests']
        if (type(rows) is not list or len(rows) != len(planned) or
                any(type(row) is not dict or type(row.get('ordinal')) is not int or
                    row['ordinal'] != expected['ordinal'] or row.get('case') != expected['case'] or
                    row.get('request_body_sha256') != expected['sha256'] or
                    row.get('status') != 'NORMAL_STOP_COMPLETE' or row.get('finish_reason') != 'stop'
                    for row, expected in zip(rows, planned, strict=True))):
            raise ValueError('exact source-bound natural-stop response menu required')
        request_count = response_count = len(planned)
        cancellation_count = 0
    else:
        if (type(rows) is not list or len(rows)!=53 or
                [r.get('ordinal') for r in rows] != list(range(53)) or
                rows[48].get('status')!='CANCELLED_COMPLETE' or
                any(r.get('status')=='CANCELLED_COMPLETE' for i,r in enumerate(rows) if i!=48)):
            raise ValueError('exact 52-response/one-cancel menu required')
        request_count, response_count, cancellation_count = 53, 52, 1
    binding=value.get('binding')
    if (type(binding) is not dict or
            set(binding) != {'arm','session_sha256','spec_sha256','source_manifest_sha256'} or
            binding['arm']!=value['arm'] or any(type(binding[k]) is not str or len(binding[k])!=64 or
                any(c not in '0123456789abcdef' for c in binding[k])
                for k in ('session_sha256','spec_sha256','source_manifest_sha256'))):
        raise ValueError('exact source/session/arm wire binding required')
    # Streaming canonical hash avoids another complete transcript allocation.
    digest=hashlib.sha256(); size=0
    encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False)
    for text in encoder.iterencode(value):
        raw=text.encode();size+=len(raw);digest.update(raw)
    digest.update(b'\n');size+=1
    return dict(schema=1,scope='retained-complete-http-wire-reference-v1',
        owner_arm=owner_arm,logical_arm=value['arm'],binding=dict(binding),
        status=value['status'],wire_complete=True,qualification=False,
        independent_release_required=False,independent_authority_claimed=False,
        canonical_wire_sha256=digest.hexdigest(),canonical_wire_bytes=size,
        request_count=request_count,response_count=response_count,cancellation_count=cancellation_count,
        retained_artifacts={name:'results/'+owner_arm+'-artifacts/'+name for name in
            ('http-client-bundle.json','native-http-bundle.json','result.json')},
        recompute='unchanged read_http_artifacts and accept_with_owner, then compare this reference')


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


def seal_result(root, *, timing, received_monotonic, received_utc, clock=time.monotonic):
    """Seal the stopped UID1000 tree; closeout stays outside the sealed tree."""
    need(type(timing) is transport.SessionTiming, 'explicit configured session timing required')
    timing = transport.SessionTiming(**asdict(timing))
    _, deadline = timing.deadlines(received_monotonic)
    need(received_monotonic <= clock(), 'original receipt is in the future')
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
    need(math.ceil(elapsed) <= timing.total_seconds and clock() <= deadline,
         "receipt-to-verified-seal ceiling exceeded")
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


def require_complete_artifact_members(bundle, spec, directory, inner, artifacts):
    """Only authenticated, untriggered signal-event pairs may be absent."""
    if spec is not None and spec.get('experiment_mode') == 'native-http-fp8-sprint01-v1':
        need(bundle.get('all_members_present') is True, 'complete bound HTTP result bundle required')
        return
    if (spec is None or spec.get('experiment_mode') not in (
            'native-prefill-row-audit-v1', 'native-async-prefill-layer-capture-v1')
            or type(spec.get('stall_trace')) is not dict):
        need(bundle.get('all_members_present') is True, 'complete bound raw result bundle required')
        return
    trace = adapter.readout.bench.read_stall_trace_evidence(spec, directory, inner, require_complete=True)
    need(trace.get('complete') is True, 'complete authenticated stall trace required')
    optional = set()
    for row in trace['workers']:
        if (row.get('complete') is True and row.get('status') == 'not_triggered'
                and row.get('attempted') is False and row.get('signal_sent') is False):
            optional.update(f"rank-{row['rank']}-trace-{kind}.json" for kind in ('intent', 'outcome'))
    members = bundle.get('members')
    need(type(members) is dict and set(members) == set(artifacts.member_limits(bundle['binding'])),
         'exact authenticated artifact member inventory required')
    missing = {name for name, member in members.items() if member.get('exists') is not True}
    need(missing <= optional, 'complete bound raw result bundle required')


def _run_physical_after_retained_closure(*, manifest_path, manifest_sha256, grant_path, grant, manifest, rpc):
    """Unexposed projection; no caller reaches this before retained closure exists."""
    root = _canonical(manifest["target_root"])
    need(root == Path(manifest["package_root"]) and Path(manifest_path) == root / "MANIFEST.json" and
         Path(grant_path) == root / "GRANT.json" and Path(__file__).resolve() == root / "review/benchmark_physical_controller.py",
         "exact fresh physical root/entry/grant paths required")
    configured = verify_configured_spec(grant, manifest,
        deadline=grant['received_monotonic'] + transport.SOURCE_PREPARATION_SECONDS)
    spec_sha256 = configured['spec_sha256']
    artifacts = _adjacent("spry_physical_artifacts", "benchmark_artifacts.py")
    raw_transport = (manifest['experiment_mode'] == artifacts.HTTP_MODE and
                     http_integration._mtp_contract().raw_attention_capture_enabled())
    output = root / "results"
    output.mkdir(mode=0o700, exist_ok=False)
    lease = adapter.owner.CanonicalLease("/home/developer/model-serving.izR2m1O9/diagnostic.lock",
        root, manifest["session_sha256"], manifest["boot_id"])
    captured, parsed, caps, supervised_results = {}, {}, {}, {}
    artifact_receipts, diagnostic_observations, partial_resource_records = {}, {}, {}
    joins = {arm: {} for arm in ("bf16", "kv4")}
    known = {arm: [] for arm in ("bf16", "kv4")}
    with ExitStack() as cleanup:
        streams = {}
        def prepare_streams():
            # Pipes are created only after acquire. No sentinel ever occupies9.
            need(lease.fd == 9 and not streams, "capture requires fresh held canonical FD9")
            for arm in ("bf16", "kv4"):
                raw_options = {}
                if raw_transport:
                    raw_options = dict(raw_binding=dict(session_sha256=manifest["session_sha256"],
                        arm=arm, spec_sha256=spec_sha256, experiment_mode=artifacts.HTTP_MODE,
                        artifact_transport=artifacts.RAW_TRANSPORT),
                        raw_destination=output / (arm + "-artifacts"))
                cap = artifacts.CapturePipes(output / (arm + "-capture"),
                    stream_limit=artifacts.stream_limit(manifest['experiment_mode']), mode=manifest['experiment_mode'],
                    **raw_options)
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
            limit = artifacts.stream_limit(manifest['experiment_mode'])
            stdout = adapter.readout.bench.read_bounded_file(output / (arm + "-capture/stdout.raw"), limit)
            stderr = adapter.readout.bench.read_bounded_file(output / (arm + "-capture/stderr.raw"), limit)
            binding = {"session_sha256": manifest["session_sha256"], "arm": arm, "spec_sha256": spec_sha256}
            if manifest['experiment_mode'] in ('v4-first-divergence-capture-v1', 'native-repeatability-capture-v1',
                                               'native-decode-layer-capture-v1', 'native-prefill-row-audit-v1',
                                               'native-async-prefill-layer-capture-v1', 'native-model-context-retrieval-32k-v1',
                                               'native-http-fp8-sprint01-v1', INDEPENDENT_MODE):
                binding['experiment_mode'] = manifest['experiment_mode']
            if raw_transport:
                binding['artifact_transport'] = artifacts.RAW_TRANSPORT
                bundle = captured[arm].get('raw_transport')
                need(type(bundle) is dict and bundle.get('binding') == binding and
                     bundle.get('transport_valid') is True, 'raw frame transport incomplete')
            else:
                bundle = artifacts.receive_bundle(stdout, stderr, output / (arm + "-artifacts"), binding)
            artifact_receipts[arm] = bundle
            destination = output / (arm + "-artifacts")
            http_mode = manifest['experiment_mode'] == 'native-http-fp8-sprint01-v1'
            independent_mode = manifest['experiment_mode'] == INDEPENDENT_MODE
            if not http_mode and not independent_mode:
                diagnostic_observations[arm] = diagnostic_readout(destination, binding, artifacts)
                _publish(output / (arm + "-diagnostics.json"), diagnostic_observations[arm])
            resources_raw = [line[len(b"SPRY_RESOURCE_V1 "):] for line in stdout.splitlines()
                             if line.startswith(b"SPRY_RESOURCE_V1 ")]
            need(len(resources_raw) == 2 and all(len(raw) <= 65536 for raw in resources_raw), "exact initial/final resource records required")
            records = [artifacts._json(raw) for raw in resources_raw]
            need([row.get("phase") for row in records] == ["initial", "final"], "resource record phase order")
            partial_resource_records[arm] = {"initial": records[0], "final": records[1],
                "complete_result_accepted": False, "qualification": False}
            inner_path = output / (arm + "-artifacts") / "result.json"
            inner_raw = adapter.readout.bench.read_bounded_file(inner_path, 8 << 20)
            inner, _ = adapter.readout.read_json(inner_path, hashlib.sha256(inner_raw).hexdigest())
            # Failed arms retain raw tails and prefix counts above even when
            # final rank-result files or completion records are unavailable.
            capture_spec = None
            if manifest['experiment_mode'] in ('v4-first-divergence-capture-v1', 'native-repeatability-capture-v1',
                                               'native-decode-layer-capture-v1', 'native-prefill-row-audit-v1',
                                               'native-async-prefill-layer-capture-v1', 'native-model-context-retrieval-32k-v1',
                                               'native-http-fp8-sprint01-v1', INDEPENDENT_MODE):
                capture_path = root / 'review/BENCHMARK-SPEC2.json'
                capture_raw = adapter.readout.bench.read_bounded_file(capture_path, 8 << 20)
                need(hashlib.sha256(capture_raw).hexdigest() == spec_sha256,
                     'frozen capture spec identity')
                capture_spec = json.loads(capture_raw, object_pairs_hook=adapter.readout._object,
                                          parse_constant=adapter.readout._constant)
                need(capture_spec.get('experiment_mode') == manifest['experiment_mode'],
                     'frozen capture mode differs from manifest')
            require_complete_artifact_members(bundle, capture_spec, destination, inner, artifacts)
            if independent_mode:
                checked_independent_records(capture_spec, inner, arm=arm,
                    session_sha256=manifest['session_sha256'], spec_sha256=spec_sha256,
                    joins=[record for pair in joins[arm].values() for record in pair.values()],
                    result=supervised_results[arm],
                    artifacts_sha256=manifest['package_files']['review/benchmark_artifacts.py'])
                diagnostic_observations[arm] = dict(schema=1, mode=INDEPENDENT_MODE, owner_arm=arm,
                    status=inner['status'], qualification=False, native_process_join_verified=True)
                _publish(output / (arm + '-diagnostics.json'), diagnostic_observations[arm])
            elif http_mode:
                expected = http_expected_binding(root, capture_spec, owner_arm=arm,
                    session_sha256=manifest["session_sha256"], spec_sha256=spec_sha256)
                inner = http_integration.read_http_artifacts(destination,
                    extraction_root=output / (arm + '-http-client-tree'), expected=expected, owner_arm=arm,
                    joins=[record for pair in joins[arm].values() for record in pair.values()],
                    supervision=supervised_results[arm])
                diagnostic_observations[arm] = {"schema": 1, "mode": manifest['experiment_mode'],
                    "owner_arm": arm, "status": inner["status"], "qualification": False}
                _publish(output / (arm + "-diagnostics.json"), diagnostic_observations[arm])
            else:
                diagnostics = diagnostic_readout(destination, binding, artifacts, inner)
                checked_native_records(inner_path.parent, inner, arm=arm, session_sha256=manifest["session_sha256"],
                    spec_sha256=spec_sha256,
                    joins=[record for pair in joins[arm].values() for record in pair.values()],
                    result=supervised_results[arm])
            if manifest["experiment_mode"] == "native-model-context-retrieval-32k-v1":
                adapter.readout.bench.read_graph_evidence(capture_spec, destination, inner)
            if manifest['experiment_mode'] in ('v4-first-divergence-capture-v1', 'native-repeatability-capture-v1',
                                               'native-decode-layer-capture-v1', 'native-prefill-row-audit-v1',
                                               'native-async-prefill-layer-capture-v1'):
                # capture_arm first authenticates the format-4 binary captures, then
                # invokes the row-audit reader for this mode's sidecar records.
                adapter.readout.capture_arm(capture_spec, inner, destination)
            if not http_mode and not independent_mode:
                require_complete_diagnostics(diagnostics)
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
        report["partial_resource_records"] = partial_resource_records
        report["live_namespace_joins"] = {arm: list(rows.values()) for arm, rows in joins.items()}
        _publish(output / "CONTROLLER-RESULT.json", report)
    seal_result(root, timing=configured['session_timing'],
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
    rpc = transport.RpcClient(args.observer_fd, grant, args.manifest_sha256)
    report = run_physical_controller(manifest_path=args.manifest, manifest_sha256=args.manifest_sha256,
        grant_path=args.grant_json, grant_sha256=args.grant_sha256, rpc=rpc)
    return 0 if report["status"] == "MATCHED_PHYSICAL_BENCHMARK_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
