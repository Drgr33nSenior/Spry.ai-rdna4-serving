"""Source-bound native HTTP entry with normal-stop client hand-off.

This is intentionally an entry boundary, not a new supervisor.  The native
HTTP server owns scheduler children and its existing finalizer remains the
only child-tree cleanup path.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Callable
import mtp_contract

class EntryError(ValueError):
    """The bound HTTP entry cannot safely reach a native runtime call."""
_HEX = re.compile('[0-9a-f]{64}\\Z')
_MODE = 'native-http-fp8-sprint01-v1'
_REQUIRED_SOURCES = {'launcher', 'parser', 'http_server', 'engine', 'scheduler', 'protocol', 'serving_contract', 'serving_client', 'serving_runtime', 'serving_launcher', 'worker_entry', 'gpu_identity', 'graph_observer', 'tokenizer_manager', 'mtp_contract'}
_LOGICAL_ARMS = frozenset(('control', 'mtp'))

def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()

def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise EntryError(reason)

def _canonical_regular(raw: str) -> Path:
    path = Path(raw)
    _require(path.is_absolute(), 'path must be absolute')
    resolved = path.resolve(strict=True)
    _require(resolved == path and (not path.is_symlink()), 'path must be canonical non-link')
    _require(stat.S_ISREG(path.lstat().st_mode), 'path must be a regular file')
    return path

def _canonical_directory(raw: str) -> Path:
    path = Path(raw)
    _require(path.is_absolute(), 'directory must be absolute')
    resolved = path.resolve(strict=True)
    _require(resolved == path and (not path.is_symlink()) and path.is_dir(), 'directory must be canonical non-link')
    return path

def read_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(type(expected_sha256) is str and _HEX.fullmatch(expected_sha256) is not None, 'binding digest invalid')
    checked = _canonical_regular(str(path))
    _require(_file_digest(checked) == expected_sha256, 'binding digest mismatch')
    value = json.loads(checked.read_text(encoding='utf-8'))
    _require(type(value) is dict, 'binding must be an object')
    return validate_binding(value)

def validate_binding(value: dict[str, Any]) -> dict[str, Any]:
    fields = {'schema', 'mode', 'arm', 'session_sha256', 'spec_sha256', 'backend_source_sha256', 'source_prefix', 'private_helper_root', 'sources', 'source_manifest_sha256', 'worker_manifest', 'worker_manifest_sha256', 'canonical_server_cli', 'resolved_contract', 'speculation', 'evidence_dir', 'bytecode_masks'}
    _require(set(value) == fields and value['schema'] == 1 and (value['mode'] == _MODE) and (value['arm'] in _LOGICAL_ARMS), 'binding fields or mode differ')
    for name in ('session_sha256', 'spec_sha256', 'backend_source_sha256'):
        _require(type(value[name]) is str and _HEX.fullmatch(value[name]) is not None, f'binding {name} invalid')
    prefix = _canonical_directory(value['source_prefix'])
    helper = _canonical_directory(value['private_helper_root'])
    evidence = _canonical_directory(value['evidence_dir'])
    _require(type(value['sources']) is dict and set(value['sources']) >= _REQUIRED_SOURCES, 'required source pins missing')
    _require(type(value['source_manifest_sha256']) is str and value['source_manifest_sha256'] == _digest(value['sources']), 'source manifest digest mismatch')
    selected_gemm = mtp_contract.gemm_for_arm(value['arm'])
    _require(value['sources'].get('gemm') == dict(path=selected_gemm['target'], sha256=selected_gemm['sha256'], module='sglang.kernels.ops.gemm.fp8_kernel'), 'final arm GEMM source identity differs')
    for overlay in mtp_contract.runtime_overlays(value['arm']):
        expected_source = dict(path=overlay['target'], sha256=overlay['sha256'], module=overlay['module'], stock_sha256=overlay['original_sha256'])
        if overlay['key'] == 'gdn_verify':
            expected_source = mtp_contract.gdn_source_for_arm(value['arm'])
        if overlay['key'] == 'midpoint_layernorm':
            expected_source = mtp_contract.norm_source_for_arm(value['arm'])
        _require(value['sources'].get(overlay['key']) == expected_source, 'allocation repair source identity differs')
        _require(overlay['bytecode_mask'] in value['bytecode_masks'], 'allocation repair bytecode mask missing')
    if mtp_contract.settings()['phase'] == mtp_contract.GDN_PAIR_PHASE:
        _require(value['sources'].get('gdn_verify') == mtp_contract.gdn_source_for_arm(value['arm']), 'GDN pair original/effective source binding differs')
    if mtp_contract.norm_candidate_enabled():
        _require(value['sources'].get('midpoint_layernorm') == mtp_contract.norm_source_for_arm(value['arm']), 'norm candidate original/effective source binding differs')
    for name, source in value['sources'].items():
        _require(type(name) is str and type(source) is dict and (set(source) >= {'path', 'sha256'}), 'source pin shape differs')
        path = _canonical_regular(source['path'])
        _require(type(source['sha256']) is str and _HEX.fullmatch(source['sha256']) is not None and (_file_digest(path) == source['sha256']), f'source hash mismatch: {name}')
        _require(path.is_relative_to(prefix) or path.is_relative_to(helper), f'source outside bound roots: {name}')
    for name in ('serving_launcher', 'worker_entry'):
        _require(Path(value['sources'][name]['path']).parent == helper, f'helper origin differs: {name}')
    for name in ('launcher', 'parser', 'http_server', 'engine', 'scheduler', 'protocol', 'tokenizer_manager'):
        source = value['sources'][name]
        _require(type(source.get('module')) is str and source['module'].startswith('sglang.'), f'native source module differs: {name}')
    mtp_source = value['sources']['mtp_contract']
    _require(Path(mtp_source['path']).parent == helper, 'MTP contract origin differs')
    loaded_spec = getattr(mtp_contract, '__spec__', None)
    _require(getattr(mtp_contract, '__file__', None) == mtp_source['path'] and loaded_spec is not None and (loaded_spec.origin == mtp_source['path']) and (type(loaded_spec.loader) is SourceFileLoader), 'loaded MTP contract origin differs')
    _require(mtp_contract.validate_contract(value['speculation'])['logical_arm'] == value['arm'], 'binding MTP contract differs')
    _require(type(value['worker_manifest']) is str and type(value['worker_manifest_sha256']) is str and (_HEX.fullmatch(value['worker_manifest_sha256']) is not None), 'worker manifest binding invalid')
    worker_manifest = _canonical_regular(value['worker_manifest'])
    _require(_file_digest(worker_manifest) == value['worker_manifest_sha256'], 'worker manifest digest mismatch')
    _require(type(value['bytecode_masks']) is list and len(value['bytecode_masks']) == mtp_contract.bytecode_mask_count(value['arm']) and all((type(item) is str for item in value['bytecode_masks'])), 'bytecode mask binding differs')
    for raw in value['bytecode_masks']:
        _canonical_directory(raw)
    _require(type(value['canonical_server_cli']) is list and all((type(arg) is str and arg for arg in value['canonical_server_cli'])), 'canonical server CLI invalid')
    _require('--disable-cuda-graph' not in value['canonical_server_cli'], 'deprecated graph flag forbidden')
    required_contract = {'tp_size', 'context_length', 'chunked_prefill_size', 'mem_fraction_static', 'max_running_requests', 'cuda_graph_backend_decode', 'cuda_graph_backend_prefill', 'cuda_graph_bs_decode', 'cuda_graph_max_bs_decode', 'kv_cache_dtype', 'model_config_sha256', 'enable_linear_replayssm', 'enable_linear_replayssm_spec', 'max_mamba_cache_size', 'speculative_draft_kv_cache_dtype'}
    lifecycle = mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES
    reference = mtp_contract.precision_reference_enabled()
    _require(type(value['resolved_contract']) is dict, 'resolved contract fields differ')
    if mtp_contract.serving_seed_required():
        required_contract.add('random_seed')
        cli = value['canonical_server_cli']
        _require(type(value['resolved_contract'].get('random_seed')) is int and value['resolved_contract']['random_seed'] == 12345 and (cli.count('--random-seed') == 1) and (cli[cli.index('--random-seed') + 1:cli.index('--random-seed') + 2] == ['12345']), 'chain diagnostic seed binding differs')
    head_precision = 'enable_fp32_lm_head' in value['resolved_contract']
    if reference:
        _require(head_precision and mtp_contract.settings().get('allocator_snapshot') is True, 'precision reference requires FP32 head and startup evidence')
    if head_precision:
        required_contract.add('enable_fp32_lm_head')
        _require(value['resolved_contract']['enable_fp32_lm_head'] is True and value['canonical_server_cli'].count('--enable-fp32-lm-head') == 1 and ({'head_logits', 'head_embedding', 'head_unquant', 'head_linear', 'head_features', 'head_context', 'head_model', 'j01_probe'} <= value['sources'].keys()), 'FP32 head option/source binding differs')
    else:
        _require('--enable-fp32-lm-head' not in value['canonical_server_cli'] and (not any((name.startswith('head_') for name in value['sources']))), 'unbound FP32 head option or source')
    if lifecycle:
        required_contract |= {'max_total_tokens', 'prefix_cache'}
    if reference:
        required_contract.add('max_total_tokens')
        required_contract |= mtp_contract.PRECISION_LOADING_FIELDS
    mtp_contract.validate_recurrent_capacity_cli(value['canonical_server_cli'])
    contract = value['resolved_contract']
    cli = value['canonical_server_cli']
    _require(contract.get('speculative_draft_kv_cache_dtype') == 'bf16' and cli.count('--speculative-draft-kv-cache-dtype') == 1 and cli[cli.index('--speculative-draft-kv-cache-dtype')+1:cli.index('--speculative-draft-kv-cache-dtype')+2] == ['bf16'], 'explicit BF16 draft KV override binding differs')
    _require(type(contract) is dict and set(contract) == required_contract, 'resolved contract fields differ')
    _require(contract['enable_linear_replayssm'] is False and type(contract['enable_linear_replayssm_spec']) is bool and (contract['enable_linear_replayssm_spec'] is mtp_contract.replay_serving_enabled(value['arm'])) and ('--enable-linear-replayssm' not in value['canonical_server_cli']) and (value['canonical_server_cli'].count('--enable-linear-replayssm-spec') == int(mtp_contract.replay_serving_enabled(value['arm']))), 'ReplaySSM exact arm/CLI binding differs')
    if lifecycle or reference:
        mtp_contract.validate_memory_pool_config({key: contract[key] for key in mtp_contract.memory_pool_config(value['arm'])}, value['arm'])
    if lifecycle:
        mtp_contract.validate_cache_config(contract['prefix_cache'])
    _require(contract['tp_size'] == 2 and contract['context_length'] == mtp_contract.phase_config()[0] and (contract['chunked_prefill_size'] == mtp_contract.prefill_chunk_size(value['arm'])) and (contract['mem_fraction_static'] == mtp_contract.memory_fraction_static()) and (contract['max_running_requests'] == 2) and (contract['kv_cache_dtype'] == 'auto'), 'fixed serving contract differs')
    _require(contract['cuda_graph_backend_decode'] == 'full' and contract['cuda_graph_backend_prefill'] == 'disabled', 'modern graph settings invalid')
    _require(contract['cuda_graph_bs_decode'] == [1, 2] and contract['cuda_graph_max_bs_decode'] == 2, 'modern graph batch settings invalid')
    if reference:
        mtp_contract.validate_precision_loading_policy({key: contract[key] for key in mtp_contract.PRECISION_LOADING_FIELDS})
        mtp_contract.validate_precision_loading_cli(value['canonical_server_cli'])
        _require(contract['model_config_sha256'] == mtp_contract.target_config_sha256(), 'precision reference config digest differs')
        cli = value['canonical_server_cli']
        wanted = mtp_contract.target_quantization()
        _require(cli.count('--quantization') == 1 and cli[cli.index('--quantization') + 1:cli.index('--quantization') + 2] == ['fp8'] if wanted else '--quantization' not in cli, 'precision reference quantization CLI differs')
    _require(type(contract['model_config_sha256']) is str and _HEX.fullmatch(contract['model_config_sha256']) is not None, 'model config digest invalid')
    return value

def _validate_worker_cross_binding(binding: dict[str, Any]) -> None:
    """Reject a separately authenticated worker manifest that binds another arm."""
    manifest_path = _canonical_regular(binding['worker_manifest'])
    _require(_file_digest(manifest_path) == binding['worker_manifest_sha256'], 'worker manifest digest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _require(type(manifest) is dict, 'worker manifest must be an object')
    _require(manifest.get('schema') == 1 and manifest.get('mode') == binding['mode'] and (manifest.get('arm') == binding['arm']) and (manifest.get('session_sha256') == binding['session_sha256']) and (manifest.get('spec_sha256') == binding['spec_sha256']) and (manifest.get('source_manifest_sha256') == binding['source_manifest_sha256']) and (manifest.get('sources') == binding['sources']), 'worker manifest cross-binding differs')
    arm_config = manifest.get('arm_config')
    _require(type(arm_config) is dict and arm_config.get('backend_source_sha256') == binding['backend_source_sha256'] and (arm_config.get('disable_cuda_graph') is False), 'worker manifest graph backend differs')

def _fresh_sglang_and_bytecode(binding: dict[str, Any]) -> None:
    _require(not any((name == 'sglang' or name.startswith('sglang.') for name in sys.modules)), 'SGLang already imported')
    _require(sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'bytecode writes are not disabled')
    validate_bytecode_masks(binding['bytecode_masks'], arm=binding['arm'])

def validate_bytecode_masks(raw_paths: list[str], arm=None) -> None:
    """Nested empty mount points are directories, never cached bytecode.

    The model_executor/runner_backend and layers/attention masks are mounted
    inside two other masks. Permit only those declared child mount points;
    every other entry, including a file or link, is a refusal.
    """
    expected_count = mtp_contract.bytecode_mask_count(arm)
    _require(type(raw_paths) is list and len(raw_paths) == expected_count and all((type(raw) is str for raw in raw_paths)), 'six bytecode masks required')
    masks = {_canonical_directory(raw) for raw in raw_paths}
    _require(len(masks) == expected_count, 'duplicate bytecode mask')
    for mask in masks:
        for child in mask.iterdir():
            _require(child in masks and (not child.is_symlink()) and child.is_dir(), 'bytecode mask has an undeclared entry')

def _bind_import_roots(binding: dict[str, Any]) -> None:
    if mtp_contract.settings().get('serving_stack') is not None:
        stack = mtp_contract.serving_stack(binding['arm'])
        if not stack['is_baseline']:
            _require(Path('/mnt/compat').is_dir(), 'read-only compatibility prefix required')
            if '/mnt/compat' not in sys.path:
                sys.path.insert(0, '/mnt/compat')
    for raw in (binding['private_helper_root'], binding['source_prefix']):
        if raw not in sys.path:
            sys.path.insert(0, raw)

def _verify_native_origins(binding: dict[str, Any]) -> None:
    for name in ('launcher', 'parser', 'http_server', 'engine', 'scheduler', 'protocol', 'tokenizer_manager'):
        source = binding['sources'][name]
        spec = importlib.util.find_spec(source['module'])
        _require(spec is not None and type(spec.loader) is SourceFileLoader and (spec.origin == source['path']), f'native source origin differs: {name}')

def _load_helper(binding: dict[str, Any]):
    source = binding['sources']['serving_launcher']
    _require('serving_launcher' not in sys.modules and 'worker_entry' not in sys.modules, 'private helper already imported')
    _bind_import_roots(binding)
    spec = importlib.util.spec_from_file_location('serving_launcher', source['path'])
    _require(spec is not None and type(spec.loader) is SourceFileLoader, 'helper loader differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules['serving_launcher'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop('serving_launcher', None)
        raise
    _require(Path(module.__file__).resolve() == Path(source['path']), 'helper file differs')
    return module

def _loaded_helper(binding: dict[str, Any]):
    """Use a prior source-loaded helper only when it is this exact binding."""
    launcher = sys.modules.get('serving_launcher')
    worker = sys.modules.get('worker_entry')
    if launcher is None and worker is None:
        return None
    _require(launcher is not None and worker is not None, 'partial private helper import')
    for name, module in (('serving_launcher', launcher), ('worker_entry', worker)):
        source = binding['sources'][name]
        path = _canonical_regular(source['path'])
        _require(getattr(module, '__file__', None) == str(path) and _file_digest(path) == source['sha256'], 'loaded private helper origin differs: ' + name)
    return launcher

@dataclass
class NativeRuntime:
    load_plugins: Callable[[], None]
    prepare_server_args: Callable[[list[str]], Any]
    resolving_view: Callable[[Any], Any]
    get_exec: Callable[[], Any]
    launch_server: Callable[..., Any]
    kill_process_tree: Callable[..., None]
    request_native_stop: Callable[[], None]
    pid: Callable[[], int]

def load_native_runtime(binding: dict[str, Any]) -> NativeRuntime:
    _verify_native_origins(binding)
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.entrypoints.http_server import get_global_state, launch_server
    from sglang.srt.managers.tokenizer_manager import SignalHandler
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    def request_native_stop() -> None:
        """Use the native graceful-stop path without replaying a process signal.

        The native HTTP server installs its own stop hook on the live
        TokenizerManager.  ``SignalHandler.sigterm_handler`` is the native
        consumer that sets graceful exit, sends the scheduler shutdown, and
        invokes that hook.  Calling it directly preserves that lifecycle while
        avoiding Uvicorn's later replay of a captured SIGTERM.
        """
        manager = get_global_state().tokenizer_manager
        SignalHandler(manager).sigterm_handler(15, None)
    return NativeRuntime(load_plugins, prepare_server_args, resolving_view, get_exec, launch_server, kill_process_tree, request_native_stop, os.getpid)

def _verify_resolved(runtime: NativeRuntime, server_args: Any, binding: dict[str, Any]) -> None:
    """Run the native resolution pass without inventing a pre-Engine view.

    ``get_exec().graph.cuda_graph_config`` is published by the native Engine,
    not by the parser stash.  The post-warmup callback checks that real nested
    object before client traffic.  Here we only retain the source-derived
    dispatch rejections that the native launcher itself makes at this stage.
    """
    server_args.resolve_once()
    cfg = runtime.resolving_view(server_args)
    pool = mtp_contract.memory_pool_config(binding['arm'])
    if pool is not None:
        observed = mtp_contract.validate_memory_pool_config({key: getattr(cfg, key, None) for key in pool}, binding['arm'])
        bound = mtp_contract.validate_memory_pool_config({key: binding['resolved_contract'].get(key) for key in pool}, binding['arm'])
        _require(observed == bound, 'native memory-pool binding differs')
    if mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES:
        _require(getattr(server_args, 'enable_metrics', None) is True and type(getattr(server_args, 'decode_log_interval', None)) is int and (server_args.decode_log_interval == mtp_contract.expected_decode_log_interval(binding['arm'])), 'closure native metric/log observations are not enabled')
    if mtp_contract.cache_config() is not None:
        cache = mtp_contract.validate_cache_config({key: getattr(cfg, key, None) for key in mtp_contract.cache_config()})
        _require(cache == binding['resolved_contract'].get('prefix_cache'), 'native prefix-cache binding differs')
        _require(os.environ.get('AMD_SERIALIZE_KERNEL') == mtp_contract.expected_environment(binding['arm']).get('AMD_SERIALIZE_KERNEL'), 'kernel serialization differs from the bound prefix-cache diagnostic')
    _require(not getattr(cfg, 'encoder_only', False) and (not getattr(cfg, 'smg_grpc_mode', False)) and (not getattr(cfg, 'grpc_mode', False)) and (not getattr(cfg, 'use_ray', False)), 'non-HTTP native launch mode refused')
    _require(getattr(server_args, 'tokenizer_worker_num', None) == 1, 'native HTTP screen requires exactly one tokenizer worker')
    _require('SGLANG_RUST_SERVER' not in os.environ, 'Rust native server mode is not supported by this HTTP screen')
    for name in ('tp_size', 'context_length', 'chunked_prefill_size', 'mem_fraction_static', 'max_running_requests', 'kv_cache_dtype', 'speculative_draft_kv_cache_dtype'):
        _require(getattr(cfg, name) == binding['resolved_contract'][name], 'native resolved field differs: ' + name)
    if mtp_contract.precision_reference_enabled():
        observed_policy = mtp_contract.validate_precision_loading_policy({key: getattr(cfg, key, None) for key in mtp_contract.PRECISION_LOADING_FIELDS}, encoded_extra_config=True)
        bound_policy = mtp_contract.validate_precision_loading_policy({key: binding['resolved_contract'].get(key) for key in mtp_contract.PRECISION_LOADING_FIELDS})
        _require(observed_policy == bound_policy, 'native precision loading policy differs')
        _require(getattr(cfg, 'quantization', None) == mtp_contract.target_quantization(), 'native target quantization differs')
    if mtp_contract.target_quantization() == 'fp8':
        _require(cfg.fp8_gemm_runner_backend == 'triton', 'native FP8 GEMM backend differs')
    if mtp_contract.serving_seed_required():
        _require(type(getattr(cfg, 'random_seed', None)) is int and cfg.random_seed == binding['resolved_contract']['random_seed'] == 12345, 'native chain diagnostic random seed differs')
    if 'enable_fp32_lm_head' in binding['resolved_contract']:
        _require(getattr(cfg, 'enable_fp32_lm_head', None) is True, 'native FP32 LM-head option not resolved')
    for field in ('enable_linear_replayssm', 'enable_linear_replayssm_spec'):
        _require(type(getattr(cfg, field, None)) is bool and getattr(cfg, field) is binding['resolved_contract'][field], 'native ReplaySSM flag differs: ' + field)
    contract = mtp_contract.validate_contract(binding['speculation'])
    _require(os.environ.get(mtp_contract.DRAFT_GRAPH_ENV) == '1', 'draft graph environment differs')
    for key, value in mtp_contract.expected_environment(binding['arm']).items():
        _require(os.environ.get(key) == value, 'diagnostic environment differs: ' + key)
    _require(getattr(cfg, 'speculative_algorithm', None) == contract['speculative_algorithm'], 'native speculative algorithm differs')
    _require(getattr(cfg, 'enable_multi_layer_eagle', None) is contract['enable_multi_layer_eagle'], 'native multi-layer EAGLE setting differs')
    if contract['enabled']:
        for field in ('speculative_draft_model_path', 'speculative_draft_model_revision', 'speculative_num_steps', 'speculative_eagle_topk', 'speculative_num_draft_tokens', 'speculative_dflash_block_size'):
            _require(getattr(cfg, field, None) == contract[field], 'native MTP field differs: ' + field)

def _verify_graph_config(config: Any, expected: dict[str, Any]) -> None:
    decode, prefill = (getattr(config, 'decode', None), getattr(config, 'prefill', None))
    _require(getattr(decode, 'backend', None) == 'full' and getattr(prefill, 'backend', None) == 'disabled' and (type(getattr(decode, 'bs', None)) is list) and (getattr(decode, 'bs', None) == [1, 2]) and all((type(value) is int for value in decode.bs)) and (type(getattr(decode, 'max_bs', None)) is int) and (decode.max_bs == 2), 'runtime graph configuration differs')

def _refusal_path(binding: dict[str, Any]) -> Path:
    return Path(binding['evidence_dir']) / 'POSTWARMUP-REFUSAL.json'

def _write_refusal(binding: dict[str, Any], exc: BaseException) -> None:
    path = _refusal_path(binding)
    value = {'schema': 1, 'mode': _MODE, 'status': 'CLIENT_REFUSED', 'error_type': type(exc).__name__, 'error': str(exc)[:512]}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as output:
            output.write(_canonical(value))
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)

def run_bound(binding: dict[str, Any], *, client: Callable[[dict[str, Any]], Any], runtime: NativeRuntime | None=None, launcher: Any | None=None) -> Any:
    """Run one source-bound native HTTP launch; CPU tests inject both boundaries."""
    binding = validate_binding(dict(binding))
    _validate_worker_cross_binding(binding)
    _fresh_sglang_and_bytecode(binding)
    _bind_import_roots(binding)
    if launcher is None:
        launcher = _loaded_helper(binding) or _load_helper(binding)
    if runtime is None:
        runtime = load_native_runtime(binding)
    _require(callable(getattr(launcher, 'launch_native_http', None)), 'launcher seam missing')
    runtime.load_plugins()
    server_args = runtime.prepare_server_args(list(binding['canonical_server_cli']))
    _verify_resolved(runtime, server_args, binding)

    def after_ready(gate: dict[str, Any]) -> None:
        try:
            _verify_graph_config(getattr(getattr(runtime.get_exec(), 'graph', None), 'cuda_graph_config', None), binding['resolved_contract'])
            client(gate)
        except BaseException as exc:
            try:
                _write_refusal(binding, exc)
            finally:
                runtime.request_native_stop()
            raise
        runtime.request_native_stop()
    try:
        return launcher.launch_native_http(server_args, native_launch_server=runtime.launch_server, worker_manifest=Path(binding['worker_manifest']), worker_manifest_sha256=binding['worker_manifest_sha256'], on_ready=after_ready)
    finally:
        runtime.kill_process_tree(runtime.pid(), include_parent=False)

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Bound native HTTP FP8 screen entry')
    parser.add_argument('--execute-serving-screen', action='store_true')
    parser.add_argument('--binding', required=True)
    parser.add_argument('--binding-sha256', required=True)
    namespace = parser.parse_args(argv)
    _require(namespace.execute_serving_screen, '--execute-serving-screen is required')
    read_binding(Path(namespace.binding), namespace.binding_sha256)
    raise EntryError('NON_EXECUTABLE: package entry must supply the source-bound client callback and live-worker checker; direct CLI dispatch is refused')
if __name__ == '__main__':
    raise SystemExit(main())
