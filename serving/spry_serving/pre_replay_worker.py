"""Manifest-bound scheduler entry for the native HTTP server.

This module does no SGLang import at module import time.  The native scheduler
is selected and checked in the spawned worker immediately before delegation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import stat
import sys
import time
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Callable
from types import CodeType


class WorkerEntryError(ValueError):
    """A manifest, source, or worker-identity precondition failed."""


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ENV_PATH = "FP8_HTTP_WORKER_MANIFEST"
_ENV_SHA256 = "FP8_HTTP_WORKER_MANIFEST_SHA256"
_MTP_SOURCE_ROLES = {
    "model_config", "mtp_model", "mtp_worker", "tp_worker", "model_loader",
    "speculative_hook", "serving_hook", "spec_info", "environ",
}
_DFLASH_SOURCE_ROLES = {
    "dflash_model", "dflash_worker", "dflash_utils", "dflash_info",
    "dflash_info_v2", "draft_worker_common",
}
_SOURCE_DIGEST_SCOPE = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def identity_digest(value: Any) -> str:
    """Use the wire-contract JSON identity, including its final newline."""
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest()


def _uncached_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_identity(path: Path) -> tuple:
    info = _safe_regular(path).lstat()
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _readonly_source(path: Path) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, 'ST_RDONLY', 0))
    except (OSError, AttributeError):
        return False


def _file_digest(path: Path) -> str:
    scope = _SOURCE_DIGEST_SCOPE
    if scope is None or scope['pid'] != os.getpid() or str(path) not in scope['paths']:
        return _uncached_file_digest(path)
    cache = scope['digests']
    identity = _source_file_identity(path)
    if not _readonly_source(path):
        cache.pop(path, None)
        return _uncached_file_digest(path)
    previous = cache.get(path)
    if previous is not None and previous[0] == identity:
        return previous[1]
    value = _uncached_file_digest(path)
    _require(_source_file_identity(path) == identity and _readonly_source(path),
             'source changed while hashing')
    cache[path] = (identity, value)
    return value


@contextmanager
def _source_digest_scope(manifest):
    """Reuse source digests only within this worker and its read-only view."""
    global _SOURCE_DIGEST_SCOPE
    _require(_SOURCE_DIGEST_SCOPE is None, 'fresh source digest scope required')
    sources = manifest.get('sources', {})
    paths = (frozenset(row['path'] for row in sources.values()
                       if type(row) is dict and type(row.get('path')) is str)
             if type(sources) is dict else frozenset())
    _SOURCE_DIGEST_SCOPE = dict(pid=os.getpid(), paths=paths, digests={})
    try:
        yield
    finally:
        _SOURCE_DIGEST_SCOPE = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerEntryError(message)


def _safe_regular(path: Path) -> Path:
    _require(path.is_absolute(), "manifest path must be absolute")
    resolved = path.resolve(strict=True)
    _require(resolved == path, "manifest path must be canonical")
    mode = path.lstat().st_mode
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), "manifest path must be a regular non-link")
    return path


def _read_manifest() -> dict[str, Any]:
    raw_path = os.environ.get(_ENV_PATH)
    expected = os.environ.get(_ENV_SHA256)
    _require(isinstance(raw_path, str) and raw_path, "worker manifest path missing")
    _require(isinstance(expected, str) and _DIGEST.fullmatch(expected) is not None, "worker manifest digest missing")
    path = _safe_regular(Path(raw_path))
    _require(_file_digest(path) == expected, "worker manifest digest mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, "worker manifest must be an object")
    return value


def _validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "mode", "arm", "session_sha256", "spec_sha256",
        "source_manifest_sha256", "sources", "expected_ranks", "expected_devices", "evidence_dir", "arm_config", "preweight",
    }
    _require(set(value) == fields, "worker manifest fields differ")
    _require(value["schema"] == 1 and value["mode"] in (
        "native-http-final-verification-v1", "native-http-fp8-sprint01-v1"), "worker manifest mode differs")
    _require(value["arm"] in ("control", "mtp"), "worker manifest arm differs")
    for key in ("session_sha256", "spec_sha256", "source_manifest_sha256"):
        _require(type(value[key]) is str and _DIGEST.fullmatch(value[key]) is not None, f"invalid {key}")
    _require(type(value["expected_ranks"]) is list and value["expected_ranks"] == [0, 1], "expected TP ranks differ")
    _require(type(value["expected_devices"]) is dict and len(value["expected_devices"]) == 2 and
             all(type(key) is str and type(item) is dict and set(item) == {"uuid", "render"}
                 and type(item["uuid"]) is str and type(item["render"]) is str
                 for key, item in value["expected_devices"].items()),
             "expected devices differ")
    _require(type(value["sources"]) is dict and _digest(value["sources"]) == value["source_manifest_sha256"], "source manifest digest mismatch")
    _require(type(value["arm_config"]) is dict and set(value["arm_config"]) == {"disable_cuda_graph", "backend_source_sha256", "shared_config_sha256"}, "arm config differs")
    _require(value["arm_config"]["disable_cuda_graph"] is False, "target graph must remain enabled")
    _require(all(type(value["arm_config"][key]) is str and _DIGEST.fullmatch(value["arm_config"][key]) is not None for key in ("backend_source_sha256", "shared_config_sha256")), "arm config hashes invalid")
    _require(type(value["evidence_dir"]) is str and value["evidence_dir"], "evidence directory missing")
    evidence = Path(value["evidence_dir"])
    _require(evidence.is_absolute() and evidence.resolve() == evidence and evidence.is_dir() and not evidence.is_symlink(), "evidence directory invalid")
    preweight = value["preweight"]
    _require(type(preweight) is dict and set(preweight) == {"deadline_monotonic", "gpu_identity", "graph_observer"},
             "preweight binding differs")
    _require(type(preweight["deadline_monotonic"]) in (int, float) and preweight["deadline_monotonic"] > 0,
             "preweight deadline differs")
    for name, module_name in (("gpu_identity", "benchmark_gpu_identity"),
                              ("graph_observer", "serving_graph_observer")):
        source = preweight[name]
        _require(type(source) is dict and set(source) == {"path", "sha256", "module"}
                 and source["module"] == module_name, "preweight helper differs")
        path = _safe_regular(Path(source["path"]))
        _require(_file_digest(path) == source["sha256"], "preweight helper source changed")
        _require(value["sources"].get(name) == source, "preweight helper source manifest differs")
    scheduler = value["sources"].get("scheduler")
    helper = value["sources"].get("worker_entry")
    launcher = value["sources"].get("serving_launcher")
    model_config = value["sources"].get("model_config")
    mtp_contract = value["sources"].get("mtp_contract")
    _require(type(helper) is dict and set(helper) == {"path", "sha256"}, "worker helper source pin differs")
    _require(type(launcher) is dict and set(launcher) == {"path", "sha256"}, "launcher helper source pin differs")
    _require(type(scheduler) is dict and set(scheduler) == {"module", "target", "path", "sha256"}, "scheduler source pin differs")
    _require(all(type(scheduler[key]) is str and scheduler[key] for key in ("module", "target", "path")), "scheduler source fields invalid")
    _require(_DIGEST.fullmatch(scheduler["sha256"]) is not None, "scheduler hash invalid")
    _require(type(model_config) is dict and {"module", "path", "sha256"} <= set(model_config)
             and model_config["module"] == "sglang.srt.configs.model_config", "model config source pin differs")
    _require(type(mtp_contract) is dict and {"path", "sha256"} <= set(mtp_contract),
             "MTP contract source pin differs")
    roles = _MTP_SOURCE_ROLES | (_DFLASH_SOURCE_ROLES if value["mode"] == "native-http-fp8-sprint01-v1" else set())
    for role in roles:
        source = value["sources"].get(role)
        _require(type(source) is dict and {"module", "path", "sha256"} <= set(source)
                 and type(source["module"]) is str and source["module"].startswith("sglang."),
                 "MTP source role differs: " + role)
    for key, source in value["sources"].items():
        _require(type(key) is str and type(source) is dict, "source row invalid")
        _require(set(source) >= {"path", "sha256"}, "source pin incomplete")
        _require(type(source["path"]) is str and type(source["sha256"]) is str and _DIGEST.fullmatch(source["sha256"]) is not None, "source pin invalid")
        path = _safe_regular(Path(source["path"]))
        _require(_file_digest(path) == source["sha256"], f"source hash mismatch: {key}")
    _require(Path(helper["path"]) == Path(__file__).resolve() and helper["sha256"] == _file_digest(Path(__file__).resolve()), "worker helper origin differs")
    return value


def _source_selection(source: dict[str, str]) -> Callable[..., Any]:
    expected = Path(source["path"])
    spec = importlib.util.find_spec(source["module"])
    _require(spec is not None and type(spec.loader) is SourceFileLoader, "scheduler loader is not SourceFileLoader")
    _require(spec.origin == str(expected), "scheduler origin differs")
    _require(_file_digest(expected) == source["sha256"], "scheduler source changed")
    module = importlib.import_module(source["module"])
    _require(getattr(module, "__file__", None) == str(expected), "loaded scheduler file differs")
    loaded_spec = getattr(module, "__spec__", None)
    _require(loaded_spec is not None and loaded_spec.origin == str(expected) and type(loaded_spec.loader) is SourceFileLoader, "loaded scheduler spec differs")
    target = getattr(module, source["target"], None)
    _require(callable(target), "native scheduler target missing")
    return target


def _native_module(source: dict[str, str]) -> Any:
    """Import one source-pinned native module after its origin is verified."""
    expected = Path(source["path"])
    spec = importlib.util.find_spec(source["module"])
    _require(spec is not None and type(spec.loader) is SourceFileLoader and spec.origin == str(expected),
             "native module origin differs")
    _require(_file_digest(expected) == source["sha256"], "native module source changed")
    module = importlib.import_module(source["module"])
    loaded_spec = getattr(module, "__spec__", None)
    _require(getattr(module, "__file__", None) == str(expected) and loaded_spec is not None
             and loaded_spec.origin == str(expected) and type(loaded_spec.loader) is SourceFileLoader,
             "loaded native module differs")
    return module


def _process_start_ticks() -> int:
    text = Path("/proc/self/stat").read_text(encoding="ascii")
    close = text.rfind(")")
    _require(close >= 0, "invalid proc stat")
    fields = text[close + 2:].split()
    # /proc stat field 22 is index 19 after fields 1 and 2.
    _require(len(fields) > 19, "short proc stat")
    value = int(fields[19])
    _require(value > 0, "invalid process start ticks")
    return value


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    _require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is not None, "invalid boot identity")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = _canonical(value)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise WorkerEntryError("worker identity already exists") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)


def _worker_record(manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    scheduler = manifest["sources"]["scheduler"]
    record = {
        "schema": 1,
        "mode": manifest["mode"],
        "arm": manifest["arm"],
        "rank": rank,
        "pid": os.getpid(),
        "start_ticks": _process_start_ticks(),
        "boot_id": _boot_id(),
        "session_sha256": manifest["session_sha256"],
        "spec_sha256": manifest["spec_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "arm_config": manifest["arm_config"],
        "sources": manifest["sources"],
        "helper_sha256": _file_digest(Path(__file__).resolve()),
        "scheduler_source": {"module": scheduler["module"], "path": scheduler["path"], "sha256": scheduler["sha256"]},
    }
    _require(_process_start_ticks() == record["start_ticks"], "process identity changed while recording")
    record["identity_sha256"] = identity_digest(record)
    return record


def _source_module(source: dict[str, str]):
    """Load a manifest-pinned private helper without an ambient search path."""
    path = _safe_regular(Path(source["path"]))
    _require(_file_digest(path) == source["sha256"], "private helper source changed")
    existing = sys.modules.get(source["module"])
    if existing is not None:
        # model_benchmark imports the observer first.  Replacing it here would
        # split the observer's module-level state from the backend import that
        # already captured that object.  Reuse only an identically pinned,
        # source-loaded instance; a substituted ambient module is refused.
        loaded_spec = getattr(existing, "__spec__", None)
        _require(
            getattr(existing, "__file__", None) == str(path)
            and loaded_spec is not None
            and loaded_spec.origin == str(path)
            and type(loaded_spec.loader) is SourceFileLoader,
            "preloaded private helper origin differs",
        )
        return existing
    spec = importlib.util.spec_from_file_location(source["module"], path)
    _require(spec is not None and type(spec.loader) is SourceFileLoader, "private helper loader differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[source["module"]] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(source["module"], None)
        raise
    _require(getattr(module, "__file__", None) == str(path), "private helper origin differs")
    return module



def _validate_precision_config(contract_module):
    """Authenticate the actual checkpoint metadata before weight construction."""
    profile = contract_module.precision_profile()
    config_path = _safe_regular(Path(contract_module.MODEL_PATH) / 'config.json')
    _require(0 < config_path.stat().st_size <= 128 << 10 and
             _file_digest(config_path) == profile['config_sha256'],
             'precision reference target config file differs')
    config = json.loads(config_path.read_bytes())
    quantization = config.get('quantization_config')
    _require((quantization is None) if profile['precision'] == 'bf16' else
             (type(quantization) is dict and quantization.get('quant_method') ==
              ('quark' if profile['precision'] == 'mxfp4' else 'fp8')),
             'precision reference checkpoint quantization differs')
    if profile['precision'] == 'mxfp4':
        _require(config['text_config'].get('mamba_ssm_dtype') == 'float32',
                 'MXFP4 recurrent temporal state must remain FP32')
    return config


def _mtp_preflight(manifest: dict[str, Any], rank: int, server_args: Any) -> dict[str, Any]:
    """Resolve the native speculative config before Scheduler can construct weights.

    ``ModelConfig.from_server_args`` reads pinned configuration metadata only;
    this boundary never creates a model runner or loads checkpoint tensors.
    """
    contract_source = dict(manifest["sources"]["mtp_contract"], module="mtp_contract")
    contract_module = _source_module(contract_source)
    _require(manifest['mode'] == contract_module.settings()['mode'], "speculative experiment mode differs")
    contract = contract_module.validate_contract(contract_module.contract_for_arm(manifest["arm"]))
    _require(os.environ.get(contract_module.DRAFT_GRAPH_ENV) == "1", "draft graph environment differs")
    expected_env = contract_module.expected_environment(manifest['arm'])
    for key, value in expected_env.items():
        _require(os.environ.get(key) == value, 'diagnostic environment differs: ' + key)
    contract_fields = (
        "speculative_algorithm", "enable_multi_layer_eagle",
        "speculative_draft_model_path", "speculative_draft_model_revision",
        "speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens",
        "speculative_draft_model_quantization", "speculative_dflash_block_size",
    )
    observed_fields = (
        "quantization",
        "_speculative_draft_quantization_explicitly_set",
        "speculative_use_rejection_sampling", "speculative_accept_threshold_single",
        "speculative_accept_threshold_acc",
    )
    # ServerArgs retains parsed input.  The native overrides resolver is the
    # effective configuration that ModelConfig itself consumes, so evidence
    # must describe that view rather than recopying the parser namespace.
    model_source = manifest["sources"]["model_config"]
    module = _native_module(model_source)
    resolver = getattr(module, "resolving_view", None)
    _require(callable(resolver), "native resolved-config constructor missing")
    cfg = resolver(server_args)
    pool_config = None
    if contract_module.memory_pool_config(manifest['arm']) is not None:
        pool_config = contract_module.validate_memory_pool_config(
            {key: getattr(cfg, key, None) for key in contract_module.memory_pool_config(manifest['arm'])},
            manifest['arm'])
    cache_config = None
    if contract_module.cache_config() is not None:
        cache_config = contract_module.validate_cache_config(
            {key: getattr(cfg, key, None) for key in contract_module.cache_config()})
        _require(os.environ.get('AMD_SERIALIZE_KERNEL') == expected_env.get('AMD_SERIALIZE_KERNEL'),
                 'kernel serialization differs from the bound prefix-cache diagnostic')
    fields = contract_fields + observed_fields
    resolved = {name: getattr(cfg, name, None) for name in fields}
    for name in contract_fields:
        if name == 'speculative_draft_model_quantization' and not contract['enabled']:
            _require(resolved[name] is None or resolved[name] == 'fp8',
                     'native disabled draft quantization differs')
            continue
        _require(resolved[name] == contract[name], "native MTP field differs: " + name)
    _require(resolved["quantization"] == contract_module.target_quantization(), "native target quantization differs")
    if contract_module.precision_reference_enabled():
        contract_module.validate_precision_loading_policy(
            {key: getattr(cfg, key, None) for key in contract_module.PRECISION_LOADING_FIELDS},
            encoded_extra_config=True)
        _validate_precision_config(contract_module)
    explicit = resolved['_speculative_draft_quantization_explicitly_set']
    _require((explicit is None or explicit is False) if not contract['enabled'] else
             explicit is contract['speculative_draft_quantization_explicit'],
             "native draft quantization intent differs")
    draft_summary: dict[str, Any] = contract_module.expected_draft_summary('none')
    if contract["enabled"]:
        if contract['speculative_kind'] == 'dflash2':
            config_path = _safe_regular(Path(contract['speculative_draft_model_path']) / 'config.json')
            _require(config_path.stat().st_size < 32768 and
                     _file_digest(config_path) == contract['draft_model_config_sha256'],
                     'DFlash2 draft config file differs')
            _require(json.loads(config_path.read_bytes()) == contract['draft_model_config'],
                     'DFlash2 draft config dictionary differs')
        model_config = getattr(module, "ModelConfig", None)
        _require(model_config is not None and callable(getattr(model_config, "from_server_args", None)),
                 "native ModelConfig constructor missing")
        draft = model_config.from_server_args(
            server_args,
            model_path=contract["speculative_draft_model_path"],
            model_revision=contract["speculative_draft_model_revision"],
            is_draft_model=True,
        )
        hf_config = getattr(draft, "hf_config", None)
        text_config = getattr(draft, "hf_text_config", None)
        architectures = getattr(hf_config, "architectures", None)
        architecture = architectures[0] if type(architectures) is list and architectures else None
        draft_summary = {
            "is_draft_model": bool(getattr(draft, "is_draft_model", False)),
            "architecture": architecture,
            "num_nextn_predict_layers": getattr(hf_config, "num_nextn_predict_layers", None),
            "text_num_nextn_predict_layers": getattr(text_config, "num_nextn_predict_layers", None),
            "quantization": getattr(draft, "quantization", None),
            "is_draft_quantization_explicit": getattr(draft, "is_draft_quantization_explicit", None),
        }
        if contract['speculative_kind'] == 'dflash2':
            dflash_fields = ('num_hidden_layers', 'hidden_size', 'num_key_value_heads',
                             'sliding_window', 'layer_types', 'num_target_layers', 'dflash_config')
            draft_summary.update({name: getattr(hf_config, name, None) for name in dflash_fields})
            draft_summary['dtype'] = str(getattr(draft, 'dtype', None))
            _require(draft_summary == contract_module.expected_draft_summary('dflash2'),
                     'native DFlash2 draft config differs')
        else:
            _require(draft_summary == contract_module.expected_draft_summary('mtp'),
                     "native one-step Qwen MTP rewrite differs")
    value = {
        "schema": 1,
        "mode": manifest["mode"],
        "arm": manifest["arm"],
        "rank": rank,
        "session_sha256": manifest["session_sha256"],
        "spec_sha256": manifest["spec_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "contract": contract,
        "environment": expected_env,
        "resolved_speculative": resolved,
        "draft_model_config": draft_summary,
        "model_config_source": {key: model_source[key] for key in ("module", "path", "sha256")},
        "before_scheduler": True,
        "qualification": False,
    }
    if cache_config is not None:
        value['resolved_prefix_cache'] = cache_config
    if pool_config is not None:
        value['resolved_memory_pool'] = pool_config
    if contract_module.settings()['phase'] in contract_module.BATCH_PHASES:
        wanted = dict(enable_metrics=True, decode_log_interval=contract_module.expected_decode_log_interval(manifest['arm']))
        observed = {key: getattr(cfg, key, None) for key in wanted}
        _require(all(type(observed[key]) is type(item) and observed[key] == item
                     for key, item in wanted.items()), 'closure native observability configuration differs')
        value['resolved_observability'] = observed
    _atomic_json(Path(manifest["evidence_dir"]) / f"MTP-PREFLIGHT-rank-{rank}.json", value)
    if rank == 0:
        _atomic_json(Path(manifest["evidence_dir"]) / "MTP-PREFLIGHT.json", value)
    return value


def _validate_serving_startup(manifest: dict[str, Any], server_args: Any, contract: Any) -> None:
    """Check effective W5 configuration and overlay selection in both modes."""
    selected = contract.validate_contract(contract.contract_for_arm(manifest['arm']))
    model_config = _native_module(manifest['sources']['model_config'])
    resolver = getattr(model_config, 'resolving_view', None)
    _require(callable(resolver), 'native resolved-config constructor missing')
    cfg = resolver(server_args)
    expected = {name: selected[name] for name in ('speculative_algorithm',
        'speculative_num_steps', 'speculative_eagle_topk', 'speculative_num_draft_tokens',
        'speculative_dflash_block_size', 'enable_multi_layer_eagle')}
    expected.update(tp_size=len(manifest['expected_ranks']), pp_size=1,
        disable_overlap_schedule=contract.cache_config()['disable_overlap_schedule'],
        cuda_graph_backend_decode=selected['target_cuda_graph_backend_decode'],
        cuda_graph_backend_prefill=selected['target_cuda_graph_backend_prefill'],
        cuda_graph_bs_decode=selected['target_cuda_graph_capture_bs'],
        cuda_graph_max_bs_decode=selected['target_cuda_graph_max_bs'])
    for name, value in expected.items():
        observed = getattr(cfg, name, None)
        _require(type(observed) is type(value) and observed == value,
                 'resolved serving startup field differs: ' + name)
        if type(value) is list:
            _require(all(type(item) is int for item in observed),
                     'resolved serving startup graph batch types differ')
    overlays = contract.runtime_overlays(manifest['arm'])
    _require(any(row['key'] == 'model_runner' for row in overlays),
             'selected startup runner overlay required')
    for row in overlays:
        source = manifest['sources'].get(row['key'])
        _require(type(source) is dict and source.get('module') == row['module']
                 and source.get('path') == row['target'] and source.get('sha256') == row['sha256']
                 and source.get('stock_sha256') == row['original_sha256'],
                 'selected startup overlay binding differs: ' + row['key'])
        _require(_file_digest(_safe_regular(Path(source['path']))) == row['sha256'],
                 'selected startup overlay source differs: ' + row['key'])


def _preweight_before_scheduler(manifest: dict[str, Any], rank: int, server_args: Any, *,
                                gpu_id: int, contract: Any, instrumented: bool) -> dict[str, Any]:
    """Run only pinned native identity/config checks before Scheduler."""
    config = manifest["preweight"]
    _require(type(instrumented) is bool and type(rank) is int and rank in (0, 1)
             and type(gpu_id) is int and gpu_id == manifest['expected_ranks'][rank],
             'native rank/index or observation mode differs')
    if time.monotonic() >= config["deadline_monotonic"]:
        raise WorkerEntryError("preweight deadline elapsed")
    _validate_serving_startup(manifest, server_args, contract)
    if time.monotonic() >= config["deadline_monotonic"]:
        raise WorkerEntryError("preweight deadline elapsed during configuration checks")
    gpu_identity = _source_module(config["gpu_identity"])
    import torch

    expected = manifest["expected_devices"]
    render_observation = _render_device_bindings(expected)
    devices, hip = gpu_identity.physical_devices(
        torch, {bdf: row["uuid"] for bdf, row in expected.items()}, config["deadline_monotonic"]
    )
    rank_binding = contract.bind_observed_rank(devices, render_observation, expected,
        manifest['expected_ranks'], rank, gpu_id)
    record = dict(schema=1, mode=manifest["mode"], arm=manifest["arm"], rank=rank,
                  session_sha256=manifest["session_sha256"], spec_sha256=manifest["spec_sha256"],
                  devices=devices, render_bindings=render_observation, hip=hip, rank_binding=rank_binding,
                  before_scheduler=True, qualification=False)
    record["identity_sha256"] = identity_digest(record)
    if not instrumented:
        return record
    _atomic_json(Path(manifest["evidence_dir"]) / f"worker-rank-{rank}-preweight.json", record)
    graph_observer = _source_module(config["graph_observer"])
    graph_observer.install(rank=rank, evidence=manifest["evidence_dir"],
                           backend_sha256=manifest["arm_config"]["backend_source_sha256"], arm=manifest["arm"],
                           sources=manifest["sources"])
    _mtp_preflight(manifest, rank, server_args)
    return record


def _render_device_bindings(expected: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Bind each expected render node to its live sysfs PCI function."""
    observed = []
    for bdf, item in sorted(expected.items()):
        render = Path(item["render"])
        _require(render.is_absolute() and render.parent == Path("/dev/dri")
                 and re.fullmatch(r"renderD[0-9]+", render.name) is not None,
                 "render device path differs")
        sysfs = Path("/sys/class/drm") / render.name / "device"
        target = sysfs.resolve(strict=True)
        _require(target.name.lower() == bdf.lower() and target.is_dir(), "render/sysfs BDF differs")
        observed.append({"bdf": bdf, "render": str(render), "sysfs_bdf": target.name.lower(),
                         "uuid": item["uuid"]})
    _require(len({row["render"] for row in observed}) == 2, "duplicate render binding")
    return observed


def _precision_loading_policy(runner, manifest, contract):
    """Observe effective stock loader configuration and its loaded source classes."""
    sources = contract.precision_loading_sources(manifest['sources'])
    modules = {role: _native_module(source) for role, source in sources.items()}
    resolver = getattr(modules['model_config'], 'resolving_view', None)
    _require(callable(resolver), 'precision loader resolved-config constructor missing')
    cfg = resolver(runner.server_args)
    resolved = {key: getattr(cfg, key, None) for key in contract.PRECISION_LOADING_FIELDS}
    contract.validate_precision_loading_policy(resolved, encoded_extra_config=True)
    get_model = getattr(modules['head_context'], 'get_model', None)
    _require(callable(get_model), 'precision loader runtime model configuration missing')
    actual = get_model()
    runtime = {key: getattr(actual, key, None) for key in contract.PRECISION_LOADING_FIELDS}
    contract.validate_precision_loading_policy(runtime, encoded_extra_config=True)
    _require(runtime == resolved, 'precision loader runtime/resolved policy differs')
    loader, config = runner.loader, runner.load_config
    _require(type(loader) is getattr(modules['model_loader'], 'DefaultModelLoader', None) and
             type(config) is getattr(modules['loader_load_config'], 'LoadConfig', None) and
             loader.load_config is config, 'precision native loader/config identity differs')
    load_format = getattr(modules['loader_load_config'], 'LoadFormat', None)
    _require(load_format is not None and config.load_format is load_format.SAFETENSORS and
             type(config.model_loader_extra_config) is dict and
             config.model_loader_extra_config == contract.precision_loading_policy()['model_loader_extra_config'] and
             config.model_loader_extra_config.get('enable_multithread_load') is False,
             'precision native parsed loader config differs')
    return dict(sources=sources, resolved=resolved, runtime=runtime,
        loader_class=type(loader).__name__, loader_module=type(loader).__module__,
        load_config_class=type(config).__name__, load_config_module=type(config).__module__,
        load_config=dict(load_format=config.load_format.value, model_loader_extra_config=config.model_loader_extra_config),
        loader_config_is_runner_config=True,
        scope='effective stock configuration and loaded classes; cache advice syscall count is not observed')


_MXFP4_SELECTION = None
_MXFP4_SCHEME = None
_MXFP4_SOURCE_CHECK = None
MXFP4_SOURCE_ROLES = (
    'mxfp4_package', 'mxfp4_prefill_reference', 'mxfp4_kernels', 'mxfp4_ops',
    'mxfp4_scheme', 'mxfp4_selection',
    'quark_config', 'quark_mxfp4_scheme',
    'quark_parameter', 'model_loader', 'loader_weight_utils',
)


def _load_post_load_hook(manifest, contract):
    """Authenticate and register the hook before Scheduler can load models."""
    source = manifest['sources'].get('post_load')
    _require(source == contract.POST_LOAD_SOURCE, 'selected post-load source binding differs')
    module = _source_module(source)
    function = getattr(module, 'run', None)
    compiled = SourceFileLoader(source['module'], source['path']).source_to_code(
        Path(source['path']).read_bytes(), source['path'])
    matches = [value for value in compiled.co_consts
               if type(value) is CodeType and value.co_name == 'run']
    _require(inspect.isfunction(function) and len(matches) == 1
             and function.__code__ == matches[0] and function.__globals__ is module.__dict__,
             'post-load startup callable/code/globals changed')
    return module


def _install_mxfp4_selection(manifest, contract):
    """Select the private serialized target scheme only within this worker."""
    global _MXFP4_SELECTION, _MXFP4_SCHEME, _MXFP4_SOURCE_CHECK
    _require(contract.dense_mxfp4_enabled() and _MXFP4_SELECTION is None,
             'fresh MXFP4 selection scope required')
    config = _validate_precision_config(contract)
    contract.mxfp4_sources(manifest['sources'])
    modules = {}
    for role in MXFP4_SOURCE_ROLES:
        source = manifest['sources'][role]
        modules[role] = (_native_module(source) if source['module'].startswith('sglang.')
                         else _source_module(source))
    loader, utils = modules['model_loader'], modules['loader_weight_utils']
    quark, parent = modules['quark_config'], modules['quark_mxfp4_scheme']
    scheme_module, selector = modules['mxfp4_scheme'], modules['mxfp4_selection']
    original = loader.get_quant_config
    _require(original is utils.get_quant_config, 'captured native factory alias differs')
    checks = []
    generated_checks = []
    for role, names in {
        'loader_weight_utils': ('get_quant_config',),
        'quark_config': ('QuarkConfig.from_config', 'QuarkConfig.get_linear_scheme',
                         'QuarkConfig.get_quant_method', 'QuarkLinearMethod.create_weights',
                         'QuarkLinearMethod.process_weights_after_loading', 'QuarkLinearMethod.apply'),
        'quark_mxfp4_scheme': ('QuarkW4A4MXFP4.create_weights',),
        'mxfp4_scheme': ('build_scheme_class',),
        'mxfp4_selection': ('build_config_class', 'Selection.__init__', 'Selection.check',
                           'Selection.__enter__', 'Selection.__exit__'),
        'mxfp4_ops': ('validate_geometry', '_resolve_runtime', 'bind_runtime', 'runtime',
                      'quantize_preallocated', 'gemm_preallocated'),
        'mxfp4_prefill_reference': ('quantizer_geometry',),
    }.items():
        source, module = manifest['sources'][role], modules[role]
        compiled = SourceFileLoader(source['module'], source['path']).source_to_code(
            Path(source['path']).read_bytes(), source['path'])
        for name in names:
            value, code = module, compiled
            for part in name.split('.'):
                value = getattr(value, part)
                matches = [item for item in code.co_consts if type(item) is CodeType and item.co_name == part]
                _require(len(matches) == 1, 'unique MXFP4 source callable required')
                code = matches[0]
            value = getattr(value, 'fn', value)  # Exact Triton JIT Python body, not a replacement kernel.
            checks.append((module, name, getattr(value, '__func__', value), code))

    def source_check():
        _require(utils.get_quant_config is original, 'defining native factory rebound')
        for role, module in modules.items():
            source = manifest['sources'][role]
            _require(sys.modules.get(source['module']) is module and
                     module.__file__ == source['path'] and
                     _file_digest(_safe_regular(Path(source['path']))) == source['sha256'],
                     'MXFP4 loaded source changed: ' + role)
        for module, name, function, code in checks:
            value = module
            for part in name.split('.'):
                value = getattr(value, part)
            value = getattr(value, 'fn', value)
            _require(getattr(value, '__func__', value) is function and inspect.isfunction(function) and
                     function.__code__ == code and function.__globals__ is module.__dict__,
                     'MXFP4 source callable/code/globals changed: ' + name)
        for cls, name, function, code, module in generated_checks:
            _require(cls.__dict__.get(name) is function and function.__code__ == code and
                     function.__globals__ is module.__dict__, 'private MXFP4 class callable changed')

    source_check()
    expected = config['quantization_config']
    specs = expected['global_quant_config']
    import torch
    scheme = scheme_module.build_scheme_class(parent.QuarkW4A4MXFP4, torch,
        modules['mxfp4_ops'], specs['weight'], specs['input_tensors'])
    private = selector.build_config_class(quark.QuarkConfig, scheme, expected)
    for role, factory, cls, names in (
        ('mxfp4_scheme', 'build_scheme_class', scheme,
         ('__init__', 'create_weights', 'process_weights_after_loading', 'apply_weights')),
        ('mxfp4_selection', 'build_config_class', private, ('_get_scheme_from_config',)),
    ):
        source = manifest['sources'][role]
        compiled = SourceFileLoader(source['module'], source['path']).source_to_code(
            Path(source['path']).read_bytes(), source['path'])
        for part in (factory, cls.__name__):
            matches = [item for item in compiled.co_consts if type(item) is CodeType and item.co_name == part]
            _require(len(matches) == 1, 'unique private MXFP4 class body')
            compiled = matches[0]
        for name in names:
            matches = [item for item in compiled.co_consts if type(item) is CodeType and item.co_name == name]
            _require(len(matches) == 1, 'unique private MXFP4 callable body')
            generated_checks.append((cls, name, cls.__dict__[name], matches[0], modules[role]))
    selection = selector.Selection(loader, original, quark.QuarkConfig, private,
        target_root=contract.MODEL_PATH, revision=contract.target_model_revision(),
        expected_quant=expected, source_check=source_check)
    selection.__enter__()
    _MXFP4_SELECTION, _MXFP4_SCHEME, _MXFP4_SOURCE_CHECK = selection, scheme, source_check

    def restore():
        global _MXFP4_SELECTION, _MXFP4_SCHEME, _MXFP4_SOURCE_CHECK
        try:
            selection.__exit__(None, None, None)
        finally:
            _MXFP4_SELECTION = _MXFP4_SCHEME = _MXFP4_SOURCE_CHECK = None
    return restore


def _mxfp4_loaded_weights(model, manifest, rank, contract):
    """Observe all packed parameters without tensor copies or arithmetic."""
    _require(_MXFP4_SELECTION is not None and _MXFP4_SCHEME is not None,
             'MXFP4 loaded witness requires the selected factory')
    _MXFP4_SELECTION.check()
    _MXFP4_SOURCE_CHECK()
    native = _native_module(manifest['sources']['quark_config'])
    parameters = _native_module(manifest['sources']['quark_parameter'])
    linears = _native_module(manifest['sources']['head_linear'])
    named = dict(model.named_modules())
    expected = contract.mxfp4_linear_layout(rank)
    actual = {name for name, module in named.items()
              if isinstance(getattr(module, 'quant_method', None), native.QuarkLinearMethod)}
    _require(actual == {row['path'] for row in expected}, 'complete 304 loaded MXFP4 module inventory')
    rows = []
    for plan in expected:
        layer = named[plan['path']]
        method, scheme = layer.quant_method, layer.scheme
        _require(type(layer) is getattr(linears, plan['class_name']) and
                 type(method) is native.QuarkLinearMethod and type(scheme) is _MXFP4_SCHEME and
                 getattr(layer, '_isolated_mxfp4_loaded', None) is True and
                 getattr(layer, 'bias', None) is None and
                 {name for name, _ in layer.named_parameters(recurse=False)} == {'weight', 'weight_scale'} and
                 list(layer.named_buffers(recurse=False)) == [],
                 'exact loaded MXFP4 class/parameters; no dense expansion or bias')
        tensors = {}
        for name, cls in (('weight', parameters.PackedvLLMParameter),
                          ('weight_scale', parameters.GroupQuantScaleParameter)):
            value = getattr(layer, name)
            _require(type(value) is cls, 'original packed/group parameter class')
            tensors[name] = dict(class_name=cls.__name__, class_module=cls.__module__,
                dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                device=str(value.device), is_cuda=value.is_cuda, contiguous=value.is_contiguous(),
                element_bytes=value.element_size(), elements=value.numel())
        rows.append(dict(path=plan['path'], class_name=type(layer).__name__,
            class_module=type(layer).__module__, method_class=type(method).__name__,
            method_module=type(method).__module__, scheme_class=type(scheme).__name__,
            scheme_module=type(scheme).__module__, loaded=True, **tensors))
    return dict(sources={role: manifest['sources'][role] for role in MXFP4_SOURCE_ROLES},
        modules=rows, selection_calls=dict(_MXFP4_SELECTION.calls),
        selection_active=_MXFP4_SELECTION.active, target_root=contract.MODEL_PATH,
        revision=contract.target_model_revision(), scope='loaded parameter metadata; no numerical equivalence claim')


def _precision_weights(scheduler, manifest, worker, observer, contract, head_record):
    """Read loaded target metadata once; perform no tensor arithmetic or copies."""
    runner = scheduler.tp_worker.model_runner
    model = runner.model
    linear = model.model.layers[0].mlp.down_proj
    source = manifest['sources']['head_linear']
    module = _native_module(source)
    _require(type(linear).__name__ == 'RowParallelLinear' and
             type(linear).__module__ == source['module'] and
             getattr(module, 'RowParallelLinear', None) is type(linear),
             'precision target linear class differs')
    bf16 = contract.precision_profile()['precision'] == 'bf16'
    mxfp4 = contract.dense_mxfp4_enabled()
    role = 'quark_config' if mxfp4 else 'head_unquant' if bf16 else 'fp8_config'
    method_source = manifest['sources'][role]
    method_module = _native_module(method_source)
    method = linear.quant_method
    expected_class = 'QuarkLinearMethod' if mxfp4 else 'UnquantizedLinearMethod' if bf16 else 'Fp8LinearMethod'
    _require(type(method).__name__ == expected_class and type(method).__module__ == method_source['module'] and
             getattr(method_module, expected_class, None) is type(method),
             'precision loaded linear method differs')
    weight = linear.weight
    record = {key: manifest[key] for key in
              ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')}
    record.update(schema=1, **{key: worker[key] for key in ('rank', 'pid', 'start_ticks', 'boot_id')},
        worker_identity_sha256=worker['identity_sha256'], profile=contract.precision_profile(),
        model_config_quantization=runner.model_config.quantization,
        loading_policy=_precision_loading_policy(runner, manifest, contract),
        linear=dict(path='model.layers.0.mlp.down_proj', class_name=type(linear).__name__,
            class_module=type(linear).__module__, method_class=type(method).__name__,
            method_module=type(method).__module__, method_source=method_source,
            weight_dtype=str(weight.dtype), weight_shape=list(weight.shape), weight_device=str(weight.device),
            weight_is_cuda=weight.is_cuda, weight_contiguous=weight.is_contiguous()),
        head_startup=head_record, before_external_generations=True, extra_tensor_operations=0,
        extra_model_forwards=0, qualification=False)
    if mxfp4:
        record['mxfp4'] = _mxfp4_loaded_weights(model, manifest, worker['rank'], contract)
    observer.validate_precision_weights(record, binding=manifest, worker=worker)
    _atomic_json(Path(manifest['evidence_dir']) / f"PRECISION-WEIGHTS-rank-{worker['rank']}.json", record)
    if contract.full_enabled_benchmark():
        enabled=contract.contract_for_arm(manifest['arm'])['enabled'];adapter=(_source_module(manifest['sources']['verify_decode_adapter']).CURRENT if enabled else None)
        proof={key:manifest[key] for key in ('mode','arm','session_sha256','spec_sha256','source_manifest_sha256')}
        proof.update(schema=1,rank=worker['rank'],pid=worker['pid'],worker_identity_sha256=worker['identity_sha256'],enabled=enabled,block_size=contract.contract_for_arm(manifest['arm'])['speculative_dflash_block_size'],before_external_generations=True,extra_tensor_operations=0,extra_model_forwards=0,qualification=False)
        proof.update(adapter_source=manifest['sources']['verify_decode_adapter'] if enabled else None,eligible_host_calls=adapter.eligible_host_calls if adapter else 0,capture_host_calls=adapter.capture_host_calls if adapter else 0,layers=sorted(adapter.layers) if adapter else [],buffers=len(adapter.buffers) if adapter else 0)
        proof.update(metadata_build_host_calls=adapter.metadata_build_host_calls if adapter else 0,metadata_reuse_host_calls=adapter.metadata_reuse_host_calls if adapter else 0,metadata_capture_build_host_calls=adapter.metadata_capture_build_host_calls if adapter else 0)
        proof['metadata_preparation_policy'] = adapter.METADATA_PREPARATION_POLICY if adapter else None
        proof['indices_elements'] = sorted(int(v['indices'].numel()) for v in adapter.buffers.values()) if adapter else []
        context = contract.precision_profile()['context_length']
        _require(proof['indices_elements'] == [8 * (context + 8), 16 * (context + 8)], 'batch-size-1/2 split-KV indices geometry differs')
        _require(not enabled or (adapter.enabled and proof['capture_host_calls']>=32 and proof['eligible_host_calls']>=proof['capture_host_calls'] and len(proof['layers'])==16 and proof['buffers']==2),'batch-size-1/2 split-KV adapter not captured')
        _atomic_json(Path(manifest['evidence_dir']) / f"FULL-ENABLED-rank-{worker['rank']}.json",proof)
    return record


def _allocator_startup_hook(manifest, rank, gpu_id, worker, preweight, target, head_startup=None):
    """Wrap the source-pinned handshake once, outside graph capture and traffic."""
    source = manifest['sources']['scheduler']
    module = sys.modules.get(source['module'])
    cls = getattr(module, 'Scheduler', None)
    original = getattr(cls, '__dict__', {}).get('get_init_info')
    constructor = getattr(cls, '__dict__', {}).get('__init__')
    path = _safe_regular(Path(source['path']))
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == source['sha256'], 'allocator scheduler source changed')
    compiled = SourceFileLoader(source['module'], str(path)).source_to_code(raw, str(path))

    def code(parent, name):
        matches = [item for item in parent.co_consts if type(item) is CodeType and item.co_name == name]
        _require(len(matches) == 1, 'allocator native callable source missing')
        return matches[0]
    _require(inspect.isfunction(target) and module is not None and (target is getattr(module, source['target'], None)) and (target.__globals__ is module.__dict__) and (target.__code__ == code(compiled, source['target'])) and (type(cls) is type) and (cls.__module__ == source['module']) and (cls.__qualname__ == 'Scheduler') and inspect.isfunction(constructor) and (constructor.__globals__ is module.__dict__) and (constructor.__code__ == code(code(compiled, 'Scheduler'), '__init__')) and (target.__globals__.get('Scheduler') is cls) and inspect.isfunction(original) and (original.__globals__ is module.__dict__) and (original.__code__ == code(code(compiled, 'Scheduler'), 'get_init_info')), 'allocator native scheduler class/method identity differs')
    observer = _source_module(manifest['preweight']['graph_observer'])
    _require(callable(getattr(observer, 'validate_allocator_record', None)), 'allocator validator missing')
    _require(type(gpu_id) is int and gpu_id == rank, 'allocator GPU/TP mapping differs')
    called = False
    probe = None

    def restore():
        current = cls.__dict__.get('get_init_info')
        _require(current is wrapper or current is original, 'allocator handshake wrapper replaced')
        if current is wrapper:
            cls.get_init_info = original

    def wrapper(self, *args, **kwargs):
        nonlocal called, probe
        restore()
        _require(not called and type(self) is cls and ('get_init_info' not in self.__dict__), 'allocator duplicate/substituted scheduler handshake')
        called = True
        result = original(self, *args, **kwargs)
        torch = sys.modules.get('torch')
        cuda = getattr(torch, 'cuda', None)
        _require(cuda is not None and cuda.current_device() == gpu_id and (not cuda.is_current_stream_capturing()), 'allocator device/capture boundary differs')
        free, total = cuda.mem_get_info(gpu_id)
        record = {name: manifest[name] for name in ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')}
        record.update(schema=1, rank=rank, pid=os.getpid(), start_ticks=_process_start_ticks(), boot_id=_boot_id(), worker_identity_sha256=worker['identity_sha256'], gpu_index=gpu_id, device=preweight['devices'][gpu_id], scheduler_source=dict(source, class_name='Scheduler', method_name='get_init_info'), counters=dict(allocated_bytes=cuda.memory_allocated(gpu_id), reserved_bytes=cuda.memory_reserved(gpu_id), reported_peak_allocated_bytes=cuda.max_memory_allocated(gpu_id), reported_peak_reserved_bytes=cuda.max_memory_reserved(gpu_id), driver_free_bytes=free, driver_total_bytes=total), scope=observer.ALLOCATOR_SCOPE, after_native_get_init_info=True, before_external_generations=True, hook_restored=cls.__dict__.get('get_init_info') is original, qualification=False)
        observer.validate_allocator_record(record, binding=manifest, worker=worker, preweight=preweight, expected_devices=manifest['expected_devices'])
        _require(_process_start_ticks() == record['start_ticks'], 'allocator process identity changed')
        _atomic_json(Path(manifest['evidence_dir']) / f'GPU-ALLOCATOR-rank-{rank}.json', record)
        contract = _source_module(dict(manifest['sources']['mtp_contract'], module='mtp_contract'))
        head_record = None
        if contract.precision_reference_enabled():
            _require(head_startup is not None, 'precision FP32 head startup witness missing')
            head_record = head_startup.finish(self)
            _precision_weights(self, manifest, worker, observer, contract, head_record)
        helper = _source_module(manifest['sources']['j01_probe'])
        cache_record = helper.draft_kv_startup(self, manifest, worker)
        print('SPRY_DRAFT_KV ' + helper.canonical(cache_record).decode().strip(), flush=True)
        contract.settings()
        if probe is None:
            contract.settings()
        _require(probe is None, 'paired slot probe collision')
        probe = helper.install_lifecycle_slots(self, manifest, worker)
        contract.settings()
        contract.settings()
        return result
    cls.get_init_info = wrapper

    def cleanup():
        restore()
        if probe is not None:
            probe.restore()
    return cleanup


def run_scheduler_process(
    server_args: Any,
    port_args: Any,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Any,
    pipe_writer: Any,
    display_tp_rank: Any = None,
    display_dp_rank: Any = None,
    display_moe_ep_rank: Any = None,
) -> Any:
    """Record the spawned TP scheduler identity then delegate once, unchanged."""
    value = _read_manifest()
    with _source_digest_scope(value):
        return _run_scheduler_process(
            value, server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank, pipe_writer,
            display_tp_rank=display_tp_rank, display_dp_rank=display_dp_rank,
            display_moe_ep_rank=display_moe_ep_rank)


def _run_scheduler_process(value: dict[str, Any], server_args: Any, port_args: Any, gpu_id: int, tp_rank: int, pp_rank: int, dp_rank: Any, pipe_writer: Any, display_tp_rank: Any=None, display_dp_rank: Any=None, display_moe_ep_rank: Any=None) -> Any:
    manifest = _validate_manifest(value)
    _require(type(tp_rank) is int and tp_rank in manifest['expected_ranks'], 'unexpected TP rank')
    _require(pp_rank == 0 and dp_rank is None, 'unsupported native scheduler topology')
    record = _worker_record(manifest, tp_rank)
    worker_path = Path(manifest['evidence_dir']) / f'worker-rank-{tp_rank}.json'
    _atomic_json(worker_path, record)
    contract = _source_module(dict(manifest['sources']['mtp_contract'], module='mtp_contract'))
    instrumented = contract.instrumentation_enabled()
    if instrumented:
        parity = _source_module(dict(manifest['sources']['serving_parity'], module='serving_parity'))
        parity.wait_before_weights(manifest)
    _native_module(manifest['sources']['gemm'])
    preweight = _preweight_before_scheduler(manifest, tp_rank, server_args, gpu_id=gpu_id, contract=contract, instrumented=instrumented)
    target = _source_selection(manifest['sources']['scheduler'])
    head_startup = None
    phase_timing = None
    restore = None
    adapter = None
    restore_target_kv = None
    restore_mxfp4 = None
    graph_observer = None
    try:
        if contract.dense_mxfp4_enabled():
            _load_post_load_hook(manifest, contract)
            restore_mxfp4 = _install_mxfp4_selection(manifest, contract)
        if instrumented:
            graph_observer = _source_module(manifest['preweight']['graph_observer'])
            helper = _source_module(manifest['sources']['j01_probe'])
            restore_target_kv = helper.install_target_kv_witness(manifest, record)
        if instrumented and 'head_features' in manifest['sources']:
            helper = _source_module(manifest['sources']['j01_probe'])
            graph_observer = _source_module(manifest['preweight']['graph_observer'])
            contract.settings()
            norm_capture = None
            contract.settings()
            contract.settings()
            input_capture = None
            phase_timing = helper.PhaseTiming(manifest, record) if contract.settings().get('phase_timing') == 'sprint8-c1-phase-timing-v1' and manifest['arm'] == 'mtp' else None
            if phase_timing is not None and contract.settings().get('target_timing') == 'sprint9-native-target-families-v1':
                helper.TargetGraphHooks(phase_timing, helper.head_module, helper.checked_phase_function)
            head_startup = helper.HeadStartup(manifest, input_capture=input_capture, phase_timing=phase_timing)
        restore = _allocator_startup_hook(manifest, tp_rank, gpu_id, record, preweight, target, head_startup=head_startup) if instrumented and contract.settings().get('allocator_snapshot') is True else None
        if contract.verify_attention_observed(manifest['arm']):
            adapter = _source_module(manifest['sources']['verify_decode_adapter']).install(manifest, record)
        return target(server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank, pipe_writer, display_tp_rank=display_tp_rank, display_dp_rank=display_dp_rank, display_moe_ep_rank=display_moe_ep_rank)
    finally:
        primary = sys.exc_info()[1]
        failures = []
        actions = [adapter.restore] if adapter is not None else []
        actions += [restore] if restore is not None else []
        actions += [restore_target_kv] if restore_target_kv is not None else []
        actions += [restore_mxfp4] if restore_mxfp4 is not None else []
        if head_startup is not None:
            actions.append(head_startup.restore)
            if head_startup.input_capture is not None:
                actions.append(head_startup.input_capture.restore)
        if phase_timing is not None:
            actions.append(phase_timing.restore_all)
        if graph_observer is not None and graph_observer.CURRENT is not None:
            actions.append(lambda: graph_observer.CURRENT.finish(manifest, record))
        for cleanup in actions:
            try:
                cleanup()
            except BaseException as error:
                if primary is not None:
                    primary.add_note('Worker cleanup failed: ' + type(error).__name__)
                else:
                    failures.append(error)
        if failures:
            raise failures[0]
