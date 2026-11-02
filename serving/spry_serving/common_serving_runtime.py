"""One bounded native-HTTP arm under the existing benchmark owner.

The module deliberately does not import Torch or SGLang at import time.  It
only materializes an already authenticated template and calls the native HTTP
entry.  Scheduler processes remain owned by native ``launch_server``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable

import mtp_contract
import serving_client
import serving_contract
import serving_entry


MODE = "native-http-fp8-sprint01-v1"
ARM_MAP = {"bf16": "control", "kv4": "mtp"}
_PORTS = {"control": 18087, "mtp": 18088}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_GDN_SOURCE_ROLES = {"gdn_backend", "gdn_triton", "gdn_packed", "gdn_verify", "gdn_prefill",
                     "hybrid_backend", "memory_pool"}


class ServingRuntimeError(ValueError):
    """A serving-screen binding cannot safely start native HTTP."""


def _load_spry_common(caller, pin, error):
    """Authenticate common source before any shared guard can execute."""
    import hashlib
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import os
    from pathlib import Path
    import stat

    def check(value, message):
        if not value:
            raise error('shared common source: ' + message)

    origin = Path(caller)
    path = origin.with_name('spry_common_canon.py')
    if origin.name.startswith('common_') and origin.parent.name in ('spry_serving', 'spry_evidence'):
        path = origin.resolve().parents[1] / 'spry_common/selected.py'
    elif (origin.parent.name == 'review' and origin.parent.parent.name == 'common-closure'
          and origin.parent.parent.parent.name == 'proposals'):
        path = origin.resolve().parents[3] / 'spry_common/selected.py'
    check(path.is_absolute() and path.resolve(strict=True) == path, 'canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 4096, 'bounded regular file required')
        raw = os.read(descriptor, 4097)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        check(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(descriptor))
              == stamp(path.lstat()), 'file changed during read')
    finally:
        os.close(descriptor)
    check(hashlib.sha256(raw).hexdigest() == pin, 'digest differs')
    # Use the checked bytes, ignoring ambient modules and cached bytecode.
    loader = SourceFileLoader('_spry_common', str(path))
    loader.get_code = lambda name: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader('_spry_common', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_SPRY_COMMON_SHA256 = 'c2cb7702d6ae482b0f9d70b6b838af3d0b9eb312140ff09d001df0880a9e6bed'
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, ServingRuntimeError)


def _require(value: bool, message: str) -> None:
    return _SPRY_COMMON.need(value, message, error=ServingRuntimeError, prefix='')


_canonical = _SPRY_COMMON.canonical_bare


def _digest(value: Any) -> str:
    return _SPRY_COMMON.digest(_canonical(value))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_regular(raw: str) -> Path:
    path = Path(raw)
    _require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
             "canonical regular file required")
    _require(stat.S_ISREG(path.lstat().st_mode), "regular template file required")
    return path


def _read_template(row: dict[str, Any], *, arm: str) -> tuple[dict[str, Any], str]:
    _require(type(row) is dict and set(row) == {"path", "sha256"}, "template pin differs")
    _require(type(row["sha256"]) is str and _HEX.fullmatch(row["sha256"]) is not None,
             "template digest invalid")
    raw_path = row["path"]
    _require(type(raw_path) is str, "template path invalid")
    if Path(raw_path).is_absolute():
        # Unit composition may use a canonical temporary source.  The
        # physical package uses the explicit relative form below.
        path = _canonical_regular(raw_path)
    else:
        expected = f"review/HTTP-BINDING-{arm}.json"
        _require(raw_path == expected, "relative template path differs")
        path = _canonical_regular(str(Path("/mnt") / raw_path))
    _require(_file_digest(path) == row["sha256"], "template digest differs")
    value = mtp_contract.expand_http_template(json.loads(path.read_text(encoding="utf-8")))
    fields = {"schema", "mode", "arm", "source_manifest_sha256", "port", "binding_template", "worker_template", "speculation"}
    _require(type(value) is dict and set(value) == fields and value["schema"] == 1 and value["mode"] == MODE
             and value["arm"] == arm and value["port"] == _PORTS[arm],
             "template mode or arm differs")
    binding, worker = value["binding_template"], value["worker_template"]
    _require(type(binding) is dict and type(worker) is dict and binding.get("schema") == worker.get("schema") == 1
             and binding.get("mode") == worker.get("mode") == MODE and binding.get("arm") == worker.get("arm") == arm
             and binding.get("session_sha256") is None and worker.get("session_sha256") is None
             and binding.get("spec_sha256") is None and worker.get("spec_sha256") is None
             and binding.get("worker_manifest_sha256") is None, "template has live execution state")
    _require(binding.get("source_manifest_sha256") == worker.get("source_manifest_sha256")
             == value["source_manifest_sha256"] and binding.get("sources") == worker.get("sources"),
             "template source maps differ")
    _require(value["speculation"] == binding.get("speculation")
             and _bound_mtp_contract(binding)["logical_arm"] == arm,
             "template MTP contract differs")
    return value, row["sha256"]


def _write_exclusive(path: Path, value: Any) -> str:
    raw = _canonical(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _fresh_directory(path: Path) -> Path:
    _require(path.is_absolute() and path.parent.is_dir() and not path.parent.is_symlink(),
             "fresh owned output parent required")
    path.mkdir(mode=0o700)
    _require(path.resolve() == path and not path.is_symlink(), "fresh owned output differs")
    return path


def _source(template: dict[str, Any], name: str) -> dict[str, Any]:
    source = template.get("sources", {}).get(name)
    _require(type(source) is dict and {"path", "sha256"} <= set(source),
             "missing pinned source: " + name)
    return dict(source)


def _bound_mtp_contract(template: dict[str, Any]) -> dict[str, Any]:
    """Use only the source-loaded contract helper pinned by this template."""
    source = _source(template, "mtp_contract")
    path = _canonical_regular(source["path"])
    loaded_spec = getattr(mtp_contract, "__spec__", None)
    _require(getattr(mtp_contract, "__file__", None) == str(path)
             and loaded_spec is not None and loaded_spec.origin == str(path),
             "loaded MTP contract origin differs")
    _require(_file_digest(path) == source["sha256"], "loaded MTP contract source changed")
    return mtp_contract.validate_contract(template.get("speculation"))


def _materialize(
    template: dict[str, Any], *, template_sha256: str, session_sha256: str,
    spec_sha256: str, output: Path, deadline: float, expected_devices: dict[str, str],
) -> dict[str, Any]:
    native, client = _fresh_directory(output / "native"), _fresh_directory(output / "client")
    binding_template, worker_template = template["binding_template"], template["worker_template"]
    sources = binding_template["sources"]
    _require(type(sources) is dict
             and template.get("source_manifest_sha256") == _digest(sources),
             "template source map differs")
    for source in sources.values():
        _require(type(source) is dict and type(source.get("path")) is str
                 and type(source.get("sha256")) is str and _HEX.fullmatch(source["sha256"]) is not None,
                 "template source pin differs")
        path = _canonical_regular(source["path"])
        _require(_file_digest(path) == source["sha256"], "template source changed")
    _require(type(expected_devices) is dict and len(expected_devices) == 2
             and all(type(key) is str and type(value) is dict and set(value) == {"uuid", "render"}
                     and type(value["uuid"]) is str and type(value["render"]) is str
                     for key, value in expected_devices.items()),
             "expected device binding differs")
    _require(worker_template.get("expected_devices") == expected_devices,
             "worker template device binding differs")
    _require(type(binding_template.get("bytecode_masks")) is list and len(binding_template["bytecode_masks"]) == mtp_contract.bytecode_mask_count(binding_template['arm']),
             "five bytecode masks required")
    serving_entry.validate_bytecode_masks(binding_template["bytecode_masks"], arm=binding_template['arm'])
    worker = dict(worker_template)
    _require(worker.get("evidence_dir") == str(native), "worker evidence target differs")
    worker.update(session_sha256=session_sha256, spec_sha256=spec_sha256)
    _require(type(worker.get("preweight")) is dict and worker["preweight"].get("deadline_monotonic") is None,
             "worker preweight template differs")
    worker["preweight"] = dict(worker["preweight"], deadline_monotonic=deadline)
    for name in ("gpu_identity", "graph_observer"):
        _require(sources.get(name) == worker["preweight"].get(name),
                 "preweight helper source map differs: " + name)
    _require(type(worker.get("arm_config")) is dict
             and worker["arm_config"].get("backend_source_sha256") == _source(binding_template, "backend")["sha256"],
             "worker backend source differs")
    _require(worker["arm_config"].get("disable_cuda_graph") is False,
             "target graph must remain enabled")
    _require(binding_template.get("backend_source_sha256") == _source(binding_template, "backend")["sha256"],
             "entry backend source differs")
    worker_path = Path(binding_template["worker_manifest"])
    _require(worker_path == native / "WORKER-MANIFEST.json", "worker manifest target differs")
    worker_sha256 = _write_exclusive(worker_path, worker)
    binding = dict(binding_template)
    _require(binding.get("evidence_dir") == str(native), "entry evidence target differs")
    _require(_port(binding["canonical_server_cli"]) == template["port"], "entry port differs from template")
    binding.update(session_sha256=session_sha256, spec_sha256=spec_sha256,
                   evidence_dir=str(native), worker_manifest=str(worker_path),
                   worker_manifest_sha256=worker_sha256)
    binding_path = native / "SERVING-BINDING.json"
    binding_sha256 = _write_exclusive(binding_path, binding)
    return {"template_sha256": template_sha256, "entry_binding": binding,
            "entry_binding_path": binding_path, "entry_binding_sha256": binding_sha256,
            "worker_manifest_path": worker_path, "worker_manifest_sha256": worker_sha256,
            "native_dir": native, "client_dir": client, "worker_template": worker_template}


def _port(argv: list[str]) -> int:
    _require(type(argv) is list and argv.count("--port") == 1, "one canonical HTTP port required")
    index = argv.index("--port")
    _require(index + 1 < len(argv) and argv[index + 1].isdigit(), "HTTP port differs")
    port = int(argv[index + 1])
    _require(1 <= port <= 65535, "HTTP port range")
    return port


def _proc_live(worker: dict[str, Any]) -> dict[str, Any]:
    pid = worker["pid"]
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        start_ticks = int(fields[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "start_ticks": -1, "boot_id": "", "alive": False}
    return {"pid": pid, "start_ticks": start_ticks, "boot_id": boot_id, "alive": True}


def _hip_sampling_ready(material: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]] | None:
    mtp_contract.settings()
    paths = sorted(material['native_dir'].glob('HIP-SAMPLING-COMPONENT-*'))
    _require(not paths, 'HIP sampling records appeared outside the candidate phase')
    return None


def _gdn_required(sources: dict[str, Any]) -> bool:
    _require(type(sources) is dict, "GDN source map differs")
    if not (_GDN_SOURCE_ROLES - {"gdn_backend", "memory_pool"}) & set(sources):
        return False
    _require(_GDN_SOURCE_ROLES <= set(sources), "complete GDN source roles required")
    return True


def _read_gdn_capture(path: Path) -> dict[str, Any]:
    path = _canonical_regular(str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                 0 < before.st_size <= 4 << 20, "bounded single-link GDN capture required")
        parts, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(block, "short GDN capture read")
            parts.append(block)
            remaining -= len(block)
        after, named = os.fstat(descriptor), path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        _require(all(getattr(before, field) == getattr(after, field) == getattr(named, field)
                     for field in fields), "GDN capture changed during read")
    finally:
        os.close(descriptor)

    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "GDN capture duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(b"".join(parts), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ServingRuntimeError("nonfinite GDN capture JSON")))
    _require(type(value) is dict, "GDN capture object required")
    return value


def _gdn_ready(material, workers, observer, snapshot, *, startup=None):
    """Join actual native captures and their snapshots before an HTTP send."""
    binding, result = material["entry_binding"], []
    sources = binding["sources"]
    _require(callable(getattr(observer, "validate_gdn_record", None)) and
             callable(getattr(observer, "canonical", None)), "GDN observer validator missing")
    rows = snapshot.get("ranks") if type(snapshot) is dict else None
    _require(type(snapshot) is dict and snapshot.get("schema") == 1 and
             snapshot.get("arm") == binding["arm"] and type(rows) is list and len(rows) == 2 and
             [row.get("rank") for row in rows if type(row) is dict and type(row.get("rank")) is int] == [0, 1],
             "GDN snapshot rank binding differs")
    for rank, worker in enumerate(workers):
        capture = _read_gdn_capture(material["native_dir"] / f"graph-captures-rank-{rank}.json")
        _require(type(capture.get("rank")) is int and capture.get("rank") == rank and
                 capture.get("arm") == binding["arm"] and
                 capture.get("pid") == worker["pid"], "GDN capture worker binding differs")
        observer.validate_gdn_record(capture.get("gdn"), binding["arm"], rank, worker["pid"], sources=sources)
        _require(capture["gdn"]["speculation"] == {
            "algorithm": binding["speculation"]["speculative_algorithm"],
            "draft_tokens": binding["speculation"]["speculative_num_draft_tokens"]},
            "GDN live speculation differs from binding")
        identity, backend = capture.get("backend_identity"), sources.get("backend")
        _require(type(identity) is dict and type(backend) is dict and
                 type(backend.get("module")) is str and type(backend.get("path")) is str and
                 type(backend.get("sha256")) is str and _HEX.fullmatch(backend["sha256"]) is not None and
                 all(identity.get(key) == backend.get(key) for key in ("module", "path", "sha256")) and
                 identity.get("class_name") == "FullCudaGraphBackend" and
                 identity.get("model_runner_role") == "target", "GDN graph backend source identity differs")
        bound = {"rank": rank, "pid": worker["pid"], "backend_identity": identity,
                 "gdn_sha256": hashlib.sha256(observer.canonical(capture["gdn"])).hexdigest()}
        _require(all(rows[rank].get(key) == value for key, value in bound.items()),
                 "GDN snapshot startup identity differs")
        if startup is not None:
            _require(bound == startup[rank], "GDN startup identity changed before request")
        result.append(bound)
    return result


def _allocator_ready(material, gate, observer):
    """Require both immutable startup records before constructing the HTTP client."""
    binding, native = material['entry_binding'], material['native_dir']
    workers = serving_client.validate_gate(gate, {name: binding[name] for name in
        ('arm', 'source_manifest_sha256', 'session_sha256', 'spec_sha256')})
    required = {f'GPU-ALLOCATOR-rank-{rank}.json' for rank in (0, 1)}
    _require({path.name for path in native.glob('GPU-ALLOCATOR*')} == required,
             'allocator startup observation membership differs')
    _require(callable(getattr(observer, 'validate_allocator_record', None)), 'allocator validator missing')
    records = []
    for rank, worker in enumerate(workers):
        record = _read_gdn_capture(native / f'GPU-ALLOCATOR-rank-{rank}.json')
        observer.validate_allocator_record(record, binding=binding, worker=worker,
            preweight=_read_gdn_capture(native / f'worker-rank-{rank}-preweight.json'),
            expected_devices=material['worker_template']['expected_devices'])
        records.append(record)
    return records


def _client_callback(
    material: dict[str, Any], *, deadline: float, client_seconds: int,
    client_factory: Callable[..., Any], holder: dict[str, Any],
) -> Callable[[dict[str, Any]], Any]:
    _require(type(client_seconds) is int and 1 <= client_seconds <= 2040,
             "bounded configured client duration required")
    binding = material["entry_binding"]
    expected = {key: binding[key] for key in ("arm", "source_manifest_sha256", "session_sha256", "spec_sha256")}
    port = _port(binding["canonical_server_cli"])
    observer_source = material["worker_template"]["preweight"]["graph_observer"]
    gdn_startup = gdn_workers = None
    group_snapshots = {}

    def request_observer(phase: str, ordinal: int, case: dict[str, Any]) -> None:
        _require(phase in ("before", "after") and type(ordinal) is int and ordinal >= 0
                 and type(case) is dict and type(case.get("case")) is str,
                 "request graph-observer binding differs")
        observer = _load_private_helper(observer_source)
        group = next(g for g in mtp_contract.kv_request_groups() if ordinal in g['ordinals'])
        key = (phase, group['id']) if group is not None else None
        if key is not None and key in group_snapshots:
            snapshot, observed = group_snapshots[key]
        else:
            snapshot = (observer.snapshot(material["native_dir"], binding["arm"], sources=binding["sources"])
                        if gdn_startup is not None else observer.snapshot(material["native_dir"], binding["arm"]))
            if gdn_startup is not None:
                _gdn_ready(material, gdn_workers, observer, snapshot, startup=gdn_startup)
            observed = time.monotonic()
            if key is not None:
                group_snapshots[key] = (snapshot, observed)
        row = {
            "schema": 1,
            "mode": MODE,
            "arm": binding["arm"],
            "ordinal": ordinal,
            "case": case["case"],
            "body_sha256": _digest(case.get("body")),
            "phase": phase,
            "observed_monotonic": observed,
            "snapshot": snapshot,
            "instrumented_latency": True,
            "qualification": False,
        }
        if group is not None:
            row.update(group_id=group['id'], group_ordinals=group['ordinals'])
        before = _bb_write_observation(material["native_dir"], row)
        if phase == "after":
            _require(before.get("body_sha256") == row["body_sha256"], "request observer body differs")
            before_ranks = before["snapshot"]["ranks"]
            after_ranks = snapshot["ranks"]
            _require([item["rank"] for item in before_ranks] == [0, 1]
                     and [item["rank"] for item in after_ranks] == [0, 1],
                     "request observer rank ordering differs")
            deltas = [after_ranks[index]["replays"] - before_ranks[index]["replays"] for index in (0, 1)]
            _require(all(delta > 0 for delta in deltas), "native target graph replay did not advance per rank")

    def run(gate: dict[str, Any]) -> Any:
        nonlocal gdn_startup, gdn_workers
        _require(time.monotonic() < deadline, "shared HTTP deadline elapsed before native readiness")
        if mtp_contract.instrumentation_enabled() and _gdn_required(binding.get("sources")):
            _require(binding["sources"].get("graph_observer") == observer_source,
                     "GDN graph observer source map differs")
            gdn_workers = serving_client.validate_gate(gate, expected)
            observer = _load_private_helper(observer_source)
            snapshot = observer.snapshot(material["native_dir"], binding["arm"], sources=binding["sources"])
            gdn_startup = _gdn_ready(material, gdn_workers, observer, snapshot)
        _hip_sampling_ready(material, gate)
        if mtp_contract.instrumentation_enabled() and mtp_contract.settings().get('allocator_snapshot') is True:
            _allocator_ready(material, gate, _load_private_helper(observer_source))
        ready = time.monotonic()
        client_deadline = min(deadline, ready + client_seconds)
        _require(ready < client_deadline, "shared HTTP deadline elapsed during native readiness")
        raw_gate = _canonical(gate)
        _write_exclusive(material["native_dir"] / "NATIVE-READINESS-CALLBACK.json", gate)
        client = client_factory(port=port, output_root=material["client_dir"], absolute_deadline=client_deadline,
                                expected_gate=expected, live_worker_checker=_proc_live,
                                request_observer=request_observer if mtp_contract.instrumentation_enabled() else None)
        holder["summary"] = client.run(gate, hashlib.sha256(raw_gate).hexdigest())
        return holder["summary"]
    return run


def _load_private_helper(source: dict[str, Any]) -> Any:
    """Load the graph-observer from its exact manifest path, never sys.path."""
    _require(type(source) is dict and set(source) == {"path", "sha256", "module"},
             "graph observer source binding differs")
    _require(type(source["module"]) is str and source["module"], "graph observer module differs")
    path = _canonical_regular(source["path"])
    _require(_file_digest(path) == source["sha256"], "graph observer source changed")
    spec = importlib.util.spec_from_file_location(source["module"], path)
    _require(spec is not None and spec.loader is not None, "graph observer loader differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(getattr(module, "__file__", None) == str(path), "graph observer origin differs")
    _require(callable(getattr(module, "snapshot", None)), "graph observer snapshot missing")
    return module


def _result(
    output: Path, *, status: str, arm: str, session_sha256: str, spec_sha256: str,
    source_manifest_sha256: str | None, client_summary: Any, native_gate: Any, parent_error: str | None,
) -> dict[str, Any]:
    value = {"schema": 1, "mode": MODE, "status": status, "arm": arm,
             "logical_arm": mtp_contract.owner_arm_map()[arm], "session_sha256": session_sha256,
             "spec_sha256": spec_sha256, "source_manifest_sha256": source_manifest_sha256,
             "client_summary": client_summary, "native_gate": native_gate,
             "parent_error": parent_error, "qualification": False}
    _write_exclusive(output / "result.json", value)
    return value


def _complete_normal_stop_summary(summary: Any, *, paired_raw=None) -> None:
    serving_contract.validate_kv_summary(summary, paired_raw)


def validate_http_specification(http):
    fields = {"schema", "mode", "bindings", "client_seconds", "expected_devices"}
    if type(http) is dict and 'artifact_limits' in http:
        values = http['artifact_limits']
        _require(type(values) is dict and len(values) == 16
                 and all(type(value) is int and 0 < value < 2**63 for value in values.values()),
                 'bounded HTTP artifact primaries required')
        fields.add('artifact_limits')
    if mtp_contract.precision_reference_enabled():
        fields.add('diagnostic_environment')
        _require(type(http) is dict and http.get('diagnostic_environment') == {
            arm: mtp_contract.expected_environment(arm) for arm in ('control', 'mtp')},
            'HTTP diagnostic environment differs')
    _require(type(http) is dict and set(http) == fields
             and http["schema"] == 1 and http["mode"] == MODE and type(http["bindings"]) is dict
             and set(http["bindings"]) == {"control", "mtp"}
             and type(http["client_seconds"]) is int and 1 <= http["client_seconds"] <= 2040,
             "HTTP specification differs")


def execute(
    args: Any, spec: dict[str, Any], output: Path, *, deadline: float,
    artifacts: Any | None = None,
    client_factory: Callable[..., Any] = serving_client.ServingClient,
    entry_runner: Callable[..., Any] = serving_entry.run_bound,
) -> dict[str, Any]:
    """Execute a single native HTTP arm through the existing native callback seam."""
    # Artifact export is owned by the outer artifact module.  Keeping its
    # authenticated object in this signature prevents a second owner or a
    # silent unbound export path here.
    _require(artifacts is not None, "authenticated outer artifact helper required")
    _require(getattr(args, "arm", None) in ARM_MAP and type(spec) is dict,
             "explicit owner arm and HTTP specification required")
    _require(type(deadline) in (int, float) and math.isfinite(deadline) and time.monotonic() < deadline,
             "finite shared HTTP deadline required")
    _require(not hasattr(client_factory, "betterbench_owner_mode"),
             "retired BetterBench client is not supported")
    http = spec.get("http_serving")
    validate_http_specification(http)
    client_seconds = http["client_seconds"]
    worker_seconds = spec.get("worker_seconds")
    _require(type(worker_seconds) is int and client_seconds + 240 <= worker_seconds <= 2280,
             "bounded HTTP worker/client timing required")
    _require(deadline <= time.monotonic() + worker_seconds, "shared HTTP deadline exceeds configured worker budget")
    _require(type(getattr(args, "session_sha256", None)) is str and _HEX.fullmatch(args.session_sha256) is not None
             and type(getattr(args, "spec_sha256", None)) is str and _HEX.fullmatch(args.spec_sha256) is not None,
             "session/spec identities required")
    logical = mtp_contract.owner_arm_map()[args.arm]
    template = None
    material = None
    summary = gate = None
    holder: dict[str, Any] = {}
    try:
        template, template_sha256 = _read_template(http["bindings"][logical], arm=logical)
        material = _materialize(template, template_sha256=template_sha256, session_sha256=args.session_sha256,
                                spec_sha256=args.spec_sha256, output=output, deadline=deadline,
                                expected_devices=http["expected_devices"])
        entry_runner(material["entry_binding"],
                     client=_client_callback(material, deadline=deadline, client_seconds=client_seconds,
                                             client_factory=client_factory, holder=holder))
        summary = holder.get("summary")
        _require(summary is not None, "native callback returned without client summary")
        _require(not (material["native_dir"] / "POSTWARMUP-REFUSAL.json").exists(),
                 "post-warmup callback refusal retained")
        _complete_normal_stop_summary(summary,
            paired_raw=None if mtp_contract.instrumentation_enabled() else (material['client_dir']/'paired'/'RUN-SUMMARY.json').read_bytes())
        gate = json.loads((material["native_dir"] / "NATIVE-READINESS-CALLBACK.json").read_text(encoding="utf-8"))
        return _result(output, status="HTTP_ARM_COMPLETE", arm=args.arm, session_sha256=args.session_sha256,
                       spec_sha256=args.spec_sha256, source_manifest_sha256=template["source_manifest_sha256"],
                       client_summary=summary, native_gate=gate, parent_error=None)
    except BaseException as error:
        try:
            gate_path = material["native_dir"] / "NATIVE-READINESS-CALLBACK.json" if material else Path("/")
            gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
        except (OSError, ValueError, json.JSONDecodeError):
            gate = None
        _result(output, status="HTTP_ARM_FAILED", arm=args.arm, session_sha256=args.session_sha256,
                spec_sha256=args.spec_sha256, source_manifest_sha256=(template or {}).get("source_manifest_sha256"),
                client_summary=summary, native_gate=gate,
                parent_error=type(error).__name__ + ": " + str(error)[:1024])
        raise


"""BetterBench-only durable two-record request journals; no witness is dropped."""
def _bb_write_observation(native, row):
    phase, ordinal = row['phase'], row['ordinal']
    _require(phase in ('before', 'after') and type(ordinal) is int and 0 <= ordinal < 176,
             'bounded BetterBench observation ordinal/phase')
    path = native / f'request-{ordinal:02d}.jsonl'
    raw = _canonical(row) + b'\n'
    _require(len(raw) <= 8192, 'BetterBench observation line bound')
    before = prior = None
    if phase == 'after':
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            prior = os.fstat(descriptor)
            _require(stat.S_ISREG(prior.st_mode) and prior.st_nlink == 1 and prior.st_uid == os.getuid()
                     and 0 < prior.st_size <= 8192, 'owned single before observation')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                old = stream.read(8193)
            before = json.loads(old)
            _require(_canonical(before) + b'\n' == old and before.get('phase') == 'before'
                     and before.get('ordinal') == ordinal and before.get('body_sha256') == row['body_sha256'],
                     'canonical matching BetterBench before observation')
        finally:
            os.close(descriptor)
    flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if phase == 'before' else os.O_APPEND)
    descriptor = os.open(path, flags, 0o600)
    try:
        current = os.fstat(descriptor)
        _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and current.st_uid == os.getuid(),
                 'owned regular BetterBench journal')
        if prior is not None:
            _require((current.st_dev, current.st_ino, current.st_size) ==
                     (prior.st_dev, prior.st_ino, prior.st_size), 'BetterBench before journal changed')
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return before
