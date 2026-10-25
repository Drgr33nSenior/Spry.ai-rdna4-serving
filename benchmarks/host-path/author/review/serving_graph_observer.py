"""Owned host-only native graph evidence; no tensor work or timing claims."""
import hashlib
import inspect
from importlib.machinery import SourceFileLoader
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time
CURRENT = None
MODULE = 'sglang.srt.model_executor.runner_backend.full_cuda_graph_backend'
GDN_MODULES = {'gdn_backend': 'sglang.srt.layers.attention.linear.gdn_backend', 'gdn_triton': 'sglang.srt.layers.attention.linear.kernels.gdn_triton', 'gdn_packed': 'sglang.kernels.ops.attention.fla.fused_recurrent', 'gdn_verify': 'sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent', 'gdn_prefill': 'sglang.kernels.ops.attention.fla.fused_gdn_gating', 'hybrid_backend': 'sglang.srt.layers.attention.hybrid_linear_attn_backend', 'memory_pool': 'sglang.srt.mem_cache.memory_pool'}
GDN_ROLES = frozenset(GDN_MODULES)
CLOSURE_PHASE = 'sprint01-gdn-aligned-lifecycle'
CLOSURE_PHASES = frozenset({CLOSURE_PHASE})
CLOSURE_LAYOUT = '<QQQQ captures,total,bs1,bs2'

def closure_phase():
    import mtp_contract
    return mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES

def expected_alignment(arm):
    import mtp_contract
    if mtp_contract.full_enabled_benchmark():
        return 'beta-bf16'
    if mtp_contract.precision_reference_enabled():
        return 'stock'
    if arm == 'mtp' or closure_phase():
        return 'beta-bf16'
    import mtp_contract
    settings = mtp_contract.settings()
    return 'beta-bf16' if settings['phase'] == mtp_contract.GDN_PAIR_PHASE and settings.get('aligned_gdn_both_arms') is True else 'stock'
FLOAT_DTYPES = ('torch.bfloat16', 'torch.float16', 'torch.float32', 'torch.float64')
GDN_STOCK = '7cad2a1b690b15c8d13f15cf1e458a7fba85f2bca9e47327a9804a1e7ba92943'
GDN_ALIGNED = '0ae8788fc8194505fb6a0ee5e320456f5fb9ebf45d4b83a4d28e2e196ec15b87'
ALLOCATOR_SCOPE = 'startup after native graph setup; allocator peaks since last native reset; not process-lifetime or whole-device peak VRAM'

def validate_allocator_record(record, *, binding, worker, preweight, expected_devices):
    """Pure join of the once-only startup counters and authenticated identities."""
    fields = {'schema', 'mode', 'arm', 'rank', 'pid', 'start_ticks', 'boot_id', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'worker_identity_sha256', 'gpu_index', 'device', 'scheduler_source', 'counters', 'scope', 'after_native_get_init_info', 'before_external_generations', 'hook_restored', 'qualification'}
    need(type(record) is dict and set(record) == fields, 'allocator record schema')
    need(record['schema'] == 1 and type(record['schema']) is int and (record['qualification'] is False) and (record['scope'] == ALLOCATOR_SCOPE) and (record['after_native_get_init_info'] is True) and (record['before_external_generations'] is True) and (record['hook_restored'] is True), 'allocator startup scope')
    for name in ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256'):
        need(record[name] == binding[name], 'allocator binding identity: ' + name)
    for name in ('rank', 'pid', 'start_ticks'):
        need(type(record[name]) is int and record[name] == worker[name] and (record[name] >= (0 if name == 'rank' else 1)), 'allocator worker identity: ' + name)
    rank = record['rank']
    need(rank in (0, 1) and type(record['gpu_index']) is int and (record['gpu_index'] == rank) and (record['boot_id'] == worker['boot_id']) and (type(record['boot_id']) is str) and record['boot_id'] and (record['worker_identity_sha256'] == worker['identity_sha256']), 'allocator process/device identity')
    scheduler = binding['sources']['scheduler']
    need(record['scheduler_source'] == dict(scheduler, class_name='Scheduler', method_name='get_init_info'), 'allocator scheduler source binding')
    need(type(preweight) is dict and type(preweight.get('rank')) is int and (preweight.get('rank') == rank) and all((preweight.get(name) == binding[name] for name in ('mode', 'arm', 'session_sha256', 'spec_sha256'))) and (preweight.get('before_scheduler') is True) and (preweight.get('qualification') is False), 'allocator preweight binding')
    devices = preweight.get('devices')
    need(type(devices) is list and len(devices) == 2 and all((type(row) is dict and set(row) == {'index', 'bdf', 'uuid', 'architecture', 'vram_bytes'} and (type(row['index']) is int) and (row['index'] == index) and (type(row['bdf']) is str) and (type(row['uuid']) is str) and (type(row['architecture']) is str) and (row['architecture'].split(':')[0] == 'gfx1201') and (type(row['vram_bytes']) is int) and (0 < row['vram_bytes'] < 1 << 64) for index, row in enumerate(devices))) and (len({row['bdf'] for row in devices}) == 2) and ({row['bdf']: row['uuid'] for row in devices} == {bdf: row['uuid'] for bdf, row in expected_devices.items()}) and (type(record['device']) is dict) and (canonical(record['device']) == canonical(devices[rank])), 'allocator physical device binding')
    counters = record['counters']
    names = {'allocated_bytes', 'reserved_bytes', 'reported_peak_allocated_bytes', 'reported_peak_reserved_bytes', 'driver_free_bytes', 'driver_total_bytes'}
    need(type(counters) is dict and set(counters) == names and all((type(value) is int and 0 <= value < 1 << 64 for value in counters.values())), 'allocator finite nonnegative integer counters')
    need(0 < counters['driver_total_bytes'] == devices[rank]['vram_bytes'] and counters['driver_free_bytes'] <= counters['driver_total_bytes'] and (counters['allocated_bytes'] <= counters['reserved_bytes'] <= counters['reported_peak_reserved_bytes'] <= counters['driver_total_bytes']) and (counters['allocated_bytes'] <= counters['reported_peak_allocated_bytes'] <= counters['reported_peak_reserved_bytes']), 'allocator counter consistency')
    need(len(canonical(record)) <= 8192, 'allocator record bound')
    return record

def validate_precision_weights(record, *, binding, worker):
    """Join a bounded loaded-weight witness to the authenticated source profile."""
    import mtp_contract
    need(mtp_contract.precision_reference_enabled(), 'precision weight witness outside reference phase')
    fields = {'schema', 'mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'rank', 'pid', 'start_ticks', 'boot_id', 'worker_identity_sha256', 'profile', 'model_config_quantization', 'loading_policy', 'linear', 'head_startup', 'before_external_generations', 'extra_tensor_operations', 'extra_model_forwards', 'qualification'}
    need(type(record) is dict and set(record) == fields and (type(record['schema']) is int) and (record['schema'] == 1) and (record['profile'] == mtp_contract.precision_profile()), 'precision weight witness schema/profile')
    for key in ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256'):
        need(record[key] == binding[key], 'precision weight binding identity: ' + key)
    for key in ('rank', 'pid', 'start_ticks', 'boot_id'):
        need(type(record[key]) is type(worker[key]) and record[key] == worker[key], 'precision weight worker identity: ' + key)
    need(record['worker_identity_sha256'] == worker['identity_sha256'] and record['before_external_generations'] is True and (type(record['extra_tensor_operations']) is int) and (record['extra_tensor_operations'] == 0) and (type(record['extra_model_forwards']) is int) and (record['extra_model_forwards'] == 0) and (record['qualification'] is False) and (record['model_config_quantization'] == mtp_contract.target_quantization()), 'precision weight runtime provenance')
    loading = record['loading_policy']
    fields = {'sources', 'resolved', 'runtime', 'loader_class', 'loader_module', 'load_config_class', 'load_config_module', 'load_config', 'loader_config_is_runner_config', 'scope'}
    need(type(loading) is dict and set(loading) == fields and (loading['sources'] == mtp_contract.precision_loading_sources(binding['sources'])) and (loading['loader_class'] == 'DefaultModelLoader') and (loading['loader_module'] == binding['sources']['model_loader']['module']) and (loading['load_config_class'] == 'LoadConfig') and (loading['load_config_module'] == binding['sources']['loader_load_config']['module']) and (loading['loader_config_is_runner_config'] is True) and (loading['scope'] == 'effective stock configuration and loaded classes; cache advice syscall count is not observed'), 'precision loaded loader/config source identity differs')
    resolved_policy = mtp_contract.validate_precision_loading_policy(loading['resolved'], encoded_extra_config=True)
    runtime_policy = mtp_contract.validate_precision_loading_policy(loading['runtime'], encoded_extra_config=True)
    need(resolved_policy == runtime_policy and loading['resolved'] == loading['runtime'] and (type(loading['load_config']) is dict) and (set(loading['load_config']) == {'load_format', 'model_loader_extra_config'}) and (loading['load_config']['load_format'] == 'safetensors') and (type(loading['load_config']['model_loader_extra_config']) is dict) and (loading['load_config']['model_loader_extra_config'] == resolved_policy['model_loader_extra_config']) and (loading['load_config']['model_loader_extra_config'].get('enable_multithread_load') is False), 'precision native loader configuration differs')
    linear = record['linear']
    fields = {'path', 'class_name', 'class_module', 'method_class', 'method_module', 'method_source', 'weight_dtype', 'weight_shape', 'weight_device', 'weight_is_cuda', 'weight_contiguous'}
    bf16 = mtp_contract.precision_profile()['precision'] == 'bf16'
    role = 'head_unquant' if bf16 else 'fp8_config'
    need(type(linear) is dict and set(linear) == fields and (linear['path'] == 'model.layers.0.mlp.down_proj') and (linear['class_name'] == 'RowParallelLinear') and (linear['class_module'] == binding['sources']['head_linear']['module']) and (linear['method_class'] == ('UnquantizedLinearMethod' if bf16 else 'Fp8LinearMethod')) and (linear['method_module'] == binding['sources'][role]['module']) and (linear['method_source'] == binding['sources'][role]) and (linear['weight_dtype'] in (('torch.bfloat16',) if bf16 else ('torch.float8_e4m3fn', 'torch.float8_e4m3fnuz'))) and (type(linear['weight_shape']) is list) and (linear['weight_shape'] == [5120, 8704]) and all((type(size) is int for size in linear['weight_shape'])) and (linear['weight_device'] == 'cuda:' + str(worker['rank'])) and (linear['weight_is_cuda'] is True) and (type(linear['weight_contiguous']) is bool), 'precision loaded linear/weight differs')
    need(type(record['head_startup']) is dict, 'precision head startup evidence missing')
    return record

def alignment_mode(sources):
    row = sources['gdn_verify']
    mode = row.get('alignment')
    if mode is not None:
        need(mode in ('stock', 'beta-bf16') and row.get('stock_sha256') == GDN_STOCK and (row['sha256'] == (GDN_STOCK if mode == 'stock' else GDN_ALIGNED)), 'GDN original/effective alignment binding')
    return mode

def selected_functions(sources, method='target_verify'):
    if method == '_replayssm_fold_target_verify':
        result = selected_functions(sources, 'target_verify')
        result['target_verify'] = dict(module=sources['gdn_backend']['module'], path=sources['gdn_backend']['path'], name='GDNAttnBackend._replayssm_fold_target_verify')
        return result
    if method == 'packed_decode':
        return {'packed_decode': dict(module=sources['gdn_triton']['module'], path=sources['gdn_triton']['path'], name='TritonGDNKernel.packed_decode'), 'wrapper': dict(module=sources['gdn_packed']['module'], path=sources['gdn_packed']['path'], name='fused_recurrent_gated_delta_rule_packed_decode'), 'kernel': dict(module=sources['gdn_packed']['module'], path=sources['gdn_packed']['path'], name='fused_recurrent_gated_delta_rule_packed_decode_kernel')}
    return {'target_verify': dict(module=sources['gdn_triton']['module'], path=sources['gdn_triton']['path'], name='TritonGDNKernel.target_verify'), 'wrapper': dict(module=sources['gdn_verify']['module'], path=sources['gdn_verify']['path'], name='fused_sigmoid_gating_delta_rule_update'), 'kernel': dict(module=sources['gdn_verify']['module'], path=sources['gdn_verify']['path'], name='fused_sigmoid_gating_delta_rule_update_kernel')}

def gdn_enabled(sources):
    enabled = bool(sources and GDN_ROLES - {'memory_pool', 'gdn_backend'} & sources.keys())
    need(not enabled or GDN_ROLES <= sources.keys(), 'incomplete GDN source roles')
    return enabled

def gdn_sources(sources):
    need(gdn_enabled(sources), 'GDN source contract required')
    result = {}
    for role, module in GDN_MODULES.items():
        row = sources[role]
        need(row.get('module') == module and type(row.get('path')) is str and (type(row.get('sha256')) is str) and (len(row['sha256']) == 64) and all((c in '0123456789abcdef' for c in row['sha256'])), 'GDN source row')
        result[role] = {key: row[key] for key in ('module', 'path', 'sha256')}
        if role == 'gdn_verify' and alignment_mode(sources) is not None:
            result[role].update(alignment=row['alignment'], stock_sha256=row['stock_sha256'])
    return result

def tensor_metadata(value):
    torch = sys.modules.get('torch')
    parameter = getattr(getattr(torch, 'nn', None), 'Parameter', None)
    need(torch is not None and type(value) in (torch.Tensor, parameter), 'GDN native Tensor/Parameter required')
    shape = list(value.shape)
    strides = list(value.stride())
    offset = value.storage_offset()
    need(0 < len(shape) <= 6 and len(shape) == len(strides) and all((type(n) is int and 0 < n < 1 << 40 for n in shape)) and all((type(n) is int and 0 <= n < 1 << 50 for n in strides)) and (type(offset) is int) and (0 <= offset < 1 << 60), 'GDN tensor metadata bound')
    return dict(dtype=str(value.dtype), shape=shape, stride=strides, offset=offset)

def validate_gdn_record(record, arm, rank, pid, sources=None):
    import mtp_contract
    fold = mtp_contract.replay_serving_enabled(arm)
    pins = gdn_sources(sources)
    aligned_pair = alignment_mode(pins) is not None
    fields = {'schema', 'arm', 'rank', 'pid', 'sources', 'classes', 'pool', 'flags', 'warmups', 'scope', 'qualification', 'speculation'}
    if fold:
        fields.add('replay_pool')
    need(type(record) is dict and set(record) == fields | ({'selected_functions'} if aligned_pair else set()), 'GDN record schema')
    method = '_replayssm_fold_target_verify' if fold else 'packed_decode' if record['speculation'].get('algorithm') is None else 'target_verify'
    if aligned_pair:
        import mtp_contract
        wanted = mtp_contract.contract_for_arm(arm)
        need(record['speculation'] == dict(algorithm=wanted['speculative_algorithm'], draft_tokens=wanted['speculative_num_draft_tokens']), 'GDN arm speculation binding')
        need(record['selected_functions'] == selected_functions(pins, method) and alignment_mode(pins) == expected_alignment(arm), 'GDN selected callable source/arm binding')
    need(record['schema'] == 1 and record['arm'] == arm and (record['rank'] == rank) and (type(record['rank']) is int) and (rank in (0, 1)) and (type(pid) is int) and (pid > 0) and (type(record['pid']) is int) and (record['pid'] == pid) and (record['qualification'] is False) and (record['scope'] == 'target warmup1 only'), 'GDN identity')
    need(record['sources'] == pins, 'GDN evidence source binding')
    need(record['classes'] == {'hybrid': 'HybridLinearAttnBackend', 'linear': 'GDNAttnBackend', 'dispatcher': 'GDNKernelDispatcher', 'decode': 'TritonGDNKernel', 'verify': 'TritonGDNKernel', 'pool': 'MambaPool'}, 'GDN class contract')
    pool = record['pool']
    need(type(pool) is dict and set(pool) == {'temporal', 'layer_ids', 'offsets'}, 'GDN pool schema')
    temporal, layer_ids, offsets = (pool['temporal'], pool['layer_ids'], pool['offsets'])

    def metadata(row, dimensions):
        need(type(row) is dict and set(row) == {'dtype', 'shape', 'stride', 'offset'} and (row['dtype'] in FLOAT_DTYPES) and (type(row['shape']) is list) and (len(row['shape']) == dimensions) and (type(row['stride']) is list) and (len(row['stride']) == dimensions) and all((type(n) is int and 0 < n < 1 << 40 for n in row['shape'])) and all((type(n) is int and 0 <= n < 1 << 50 for n in row['stride'])) and (type(row['offset']) is int) and (0 <= row['offset'] < 1 << 60), 'GDN dtype/shape metadata')
    metadata(temporal, 5)
    need(temporal['shape'][1] == 20, 'Sprint7 matched physical recurrent slots including padding')
    need(type(layer_ids) is list and 0 < len(layer_ids) <= 256 and all((type(n) is int and 0 <= n < 512 for n in layer_ids)) and (len(set(layer_ids)) == len(layer_ids)) and (len(layer_ids) == temporal['shape'][0]), 'GDN pool layers')
    need(offsets == [temporal['offset'] + n * temporal['stride'][0] for n in range(len(layer_ids))] and len(set(offsets)) == len(offsets), 'GDN layer offsets')
    if aligned_pair:
        need(len(layer_ids) == 48, 'GDN pair requires all 48 linear layers')
    flags = record['flags']
    need(type(flags) is dict and set(flags) == {'enable_linear_replayssm', 'enable_linear_replayssm_spec', 'replayssm_spec_fold', 'replayssm_is_kda', 'cache_base_present', 'rawv_present', 'd_present', 'k_present', 'g_present', 'supports_packed_decode'} and all((type(v) is bool for v in flags.values())), 'GDN ReplaySSM flags')
    warmups = record['warmups']
    need(type(warmups) is list and len(warmups) == 2 and ({w.get('size') for w in warmups} == {1, 2}), 'GDN two unique warmup buckets')
    speculation = record['speculation']
    need(type(speculation) is dict and set(speculation) == {'algorithm', 'draft_tokens'} and (speculation['algorithm'] is None and speculation['draft_tokens'] is None or (speculation['algorithm'] == 'DFLASH' and type(speculation['draft_tokens']) is int and (speculation['draft_tokens'] == 8))), 'GDN live speculation')
    method = '_replayssm_fold_target_verify' if fold else 'packed_decode' if speculation['algorithm'] is None else 'target_verify'
    if aligned_pair:
        import mtp_contract
        allowed_route = method == ('_replayssm_fold_target_verify' if fold else 'target_verify') if mtp_contract.full_enabled_benchmark() else method == 'packed_decode' if mtp_contract.precision_reference_enabled() else method == 'target_verify' or (mtp_contract.greedy_quality_draft_ab() and arm == 'control')
        expected_flags = {name: False for name in flags if name != 'supports_packed_decode'}
        if fold:
            expected_flags.update(enable_linear_replayssm_spec=True, replayssm_spec_fold=True, rawv_present=True, g_present=True)
        need(allowed_route and all((flags[name] is value for name, value in expected_flags.items())), 'GDN pair selected route')
        if fold:
            replay = record['replay_pool']
            need(type(replay) is dict and set(replay) == {'rawv', 'rawk', 'g', 'beta', 'snapshots_absent', 'fold_source'}, 'GDN replay pool schema')
            need(replay['snapshots_absent'] is True and replay['fold_source'] == sources['replay_fold'], 'GDN replay allocation/source binding')
            slots = temporal['shape'][1]
            for name, shape, dtype in (('rawv', [48, slots, 24, 8, 128], 'torch.bfloat16'), ('rawk', [48, slots, 8, 8, 128], 'torch.bfloat16'), ('g', [48, slots, 24, 8], 'torch.float32'), ('beta', [48, slots, 24, 8], 'torch.float32')):
                metadata(replay[name], len(shape))
                need(replay[name]['shape'] == shape and replay[name]['dtype'] == dtype, 'GDN persistent-slot raw-ring geometry')
    for row in warmups:
        fields = {'size', 'phase', 'warmup_step', 'method', 'calls', 'offsets', 'b', 'state', 'branch', 'arguments', 'restored'}
        need(type(row) is dict and set(row) == fields | ({'A_log', 'dt_bias', 'a', 'selected_branch', 'branch_basis'} if aligned_pair else set()) and (type(row['size']) is int) and (row['phase'] == 'warmup') and (row['warmup_step'] == 1) and (row['method'] == method) and (type(row['calls']) is int) and (row['calls'] == len(layer_ids)) and (row['offsets'] == offsets) and (row['restored'] is True), 'GDN calls/phase/restoration')
        metadata(row['b'], 2)
        metadata(row['state'], 4)
        if aligned_pair:
            for name, dimensions in (('A_log', 1), ('dt_bias', 1), ('a', 2)):
                metadata(row[name], dimensions)
            need(row['a']['dtype'] == row['b']['dtype'] == 'torch.bfloat16' and row['a']['shape'] == row['b']['shape'] and (row['A_log']['shape'] == row['dt_bias']['shape'] == [row['b']['shape'][1]]) and (row['selected_branch'] == ('legacy_packed_decode' if method == 'packed_decode' else 'bf16_sigmoid_beta' if alignment_mode(pins) == 'beta-bf16' else 'fp32_exp_beta')) and (row['branch_basis'] == ('inferred from live packed arguments and authenticated packed_decode source; not device constexpr capture' if method == 'packed_decode' else 'inferred from live b dtype and authenticated target_verify constants; not device constexpr capture')), 'GDN pair live dtype/branch evidence')
        need(row['b']['shape'][0] == row['size'] * (speculation['draft_tokens'] or 1) and row['state']['dtype'] == temporal['dtype'] and (row['state']['shape'] == temporal['shape'][1:]) and (row['state']['stride'] == temporal['stride'][1:]) and (row['state']['offset'] == offsets[0]), 'GDN live operand/pool consistency')
        args = row['arguments']
        if method == 'packed_decode':
            need(type(args) is dict and set(args) == {'replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_write_pos', 'replayssm_force_flush'} and all((type(v) is bool for v in args.values())), 'GDN packed args')
            branch = 'replayssm_decode' if all((args[k] for k in ('replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_write_pos'))) else 'legacy_packed_decode'
            need(row['branch'] == branch and flags['supports_packed_decode'], 'GDN packed branch')
            need(branch == 'legacy_packed_decode', 'GDN observed ReplaySSM source route is not covered')
        elif fold:
            need(type(args) is dict and args == dict(retrieve_parent_token=False, intermediate_states_buffer=False, cache_ring=True, ring_steps=8) and (row['branch'] == 'aligned_ring_target_verify'), 'GDN aligned ring verify arguments')
        else:
            need(type(args) is dict and set(args) == {'retrieve_parent_token', 'intermediate_states_buffer', 'cache_steps'} and (type(args['retrieve_parent_token']) is bool) and (args['intermediate_states_buffer'] is True) and (type(args['cache_steps']) is int) and (args['cache_steps'] == 8) and (row['branch'] == 'generic_target_verify'), 'GDN verify args/branch')
            if aligned_pair:
                need(args['retrieve_parent_token'] is False, 'GDN pair tree route differs')
    need(len(canonical(record)) <= 65536, 'GDN record bound')
    return record

class GDNObservation:

    def __init__(self, runner, sources, arm, rank, on_failure=None, on_commit=None):
        self.on_failure = on_failure
        import mtp_contract
        self.fold = mtp_contract.replay_serving_enabled(arm)
        self.sources = gdn_sources(sources)
        self.alignment = alignment_mode(self.sources)
        for row in self.sources.values():
            module = sys.modules.get(row['module'])
            spec = getattr(module, '__spec__', None)
            path = Path(row['path'])
            need(module is not None and isinstance(getattr(spec, 'loader', None), SourceFileLoader) and (spec.origin == str(path)) and (module.__file__ == str(path)) and (path.resolve() == path) and (hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']), 'GDN loaded source identity')

        def checked(value, role, name):
            cls = type(value)
            need(cls.__name__ == name and cls.__module__ == self.sources[role]['module'] and (getattr(sys.modules[cls.__module__], name, None) is cls), 'GDN loaded class identity')
            return value
        hybrid = checked(runner.attn_backend, 'hybrid_backend', 'HybridLinearAttnBackend')
        linear = checked(hybrid.linear_attn_backend, 'gdn_backend', 'GDNAttnBackend')
        self.dispatcher = checked(linear.kernel_dispatcher, 'gdn_backend', 'GDNKernelDispatcher')
        checked(self.dispatcher.decode_kernel, 'gdn_triton', 'TritonGDNKernel')
        checked(self.dispatcher.verify_kernel, 'gdn_triton', 'TritonGDNKernel')
        checked(self.dispatcher.tree_verify_kernel, 'gdn_triton', 'TritonGDNKernel')
        if self.alignment is not None:
            triton_module = sys.modules[self.sources['gdn_triton']['module']]
            packed = runner.server_args.speculative_algorithm is None
            role = 'gdn_packed' if packed else 'gdn_verify'
            method = 'packed_decode' if packed else 'target_verify'
            function = 'fused_recurrent_gated_delta_rule_packed_decode' if packed else 'fused_sigmoid_gating_delta_rule_update'
            verify_module = sys.modules[self.sources[role]['module']]
            wrapper = getattr(verify_module, function, None)
            kernel = getattr(getattr(verify_module, function + '_kernel', None), 'fn', None)
            target = getattr(type(self.dispatcher.decode_kernel if packed else self.dispatcher.verify_kernel), method)
            need(inspect.isfunction(target) and inspect.isfunction(wrapper) and inspect.isfunction(kernel) and (target.__code__.co_filename == self.sources['gdn_triton']['path']) and (target.__globals__.get(function) is wrapper) and (getattr(triton_module, function) is wrapper) and (wrapper.__code__.co_filename == kernel.__code__.co_filename == self.sources[role]['path']) and (wrapper.__globals__.get(function + '_kernel') is getattr(verify_module, function + '_kernel')), 'GDN selected callable source identity')
        self.subject = linear if self.fold else self.dispatcher
        if self.fold:
            target = type(linear)._replayssm_fold_target_verify
            need(inspect.isfunction(target) and target.__code__.co_filename == self.sources['gdn_backend']['path'], 'GDN native ring route source')
        need(not hasattr(linear, '_replay_serving_commit_observer') and callable(on_commit), 'one bounded native commit observer')
        linear._replay_serving_commit_observer = on_commit
        pool = checked(runner.req_to_token_pool.mamba_pool, 'memory_pool', 'MambaPool')
        need(linear.req_to_token_pool is runner.req_to_token_pool, 'GDN pool route')
        temporal = tensor_metadata(pool.mamba_cache.temporal)
        layer_ids = list(pool.mamba_layer_ids)
        need(0 < len(layer_ids) <= 256 and len(temporal['shape']) == 5, 'GDN layer count')
        flags = {name: getattr(pool, name) for name in ('enable_linear_replayssm', 'enable_linear_replayssm_spec', 'replayssm_spec_fold', 'replayssm_is_kda')}
        flags.update(cache_base_present=pool.replayssm_cache_base is not None, rawv_present=pool.mamba_cache.replayssm_rawv is not None, d_present=pool.mamba_cache.replayssm_d is not None, k_present=pool.mamba_cache.replayssm_k is not None, g_present=pool.mamba_cache.replayssm_g is not None, supports_packed_decode=self.dispatcher.supports_packed_decode)
        self.record = dict(schema=1, arm=arm, rank=rank, pid=os.getpid(), sources=self.sources, classes=dict(hybrid='HybridLinearAttnBackend', linear='GDNAttnBackend', dispatcher='GDNKernelDispatcher', decode='TritonGDNKernel', verify='TritonGDNKernel', pool='MambaPool'), pool=dict(temporal=temporal, layer_ids=layer_ids, offsets=[temporal['offset'] + n * temporal['stride'][0] for n in range(len(layer_ids))]), flags=flags, warmups=[], scope='target warmup1 only', qualification=False, speculation=dict(algorithm=runner.server_args.speculative_algorithm, draft_tokens=runner.server_args.speculative_num_draft_tokens))
        if self.fold:
            self.record['replay_pool'] = dict(rawv=tensor_metadata(pool.mamba_cache.replayssm_rawv), rawk=tensor_metadata(pool.mamba_cache.replayssm_rawk), g=tensor_metadata(pool.mamba_cache.replayssm_g), beta=tensor_metadata(pool.mamba_cache.replayssm_beta), snapshots_absent=pool.mamba_cache.intermediate_ssm is None, fold_source=dict(sources['replay_fold']))
        self.active, self.saved = (None, {})
        if self.alignment is not None:
            self.record['selected_functions'] = selected_functions(self.sources, '_replayssm_fold_target_verify' if self.fold else 'packed_decode' if self.record['speculation']['algorithm'] is None else 'target_verify')

    def restore(self):
        for name, wrapper in self.saved.items():
            need(self.subject.__dict__.get(name) is wrapper, 'GDN wrapper replaced')
            delattr(self.subject, name)
        self.saved = {}
        if self.active is not None:
            self.active['restored'] = True

    def begin(self, size):
        need(not self.saved and self.active is None and (size not in [row['size'] for row in self.record['warmups']]), 'GDN warmup duplicate')
        method = '_replayssm_fold_target_verify' if self.fold else 'packed_decode' if self.record['speculation']['algorithm'] is None else 'target_verify'
        original = getattr(self.subject, method)
        need(method not in self.subject.__dict__ and inspect.ismethod(original) and (original.__func__ is type(self.subject).__dict__[method]) and (original.__func__.__code__.co_filename == self.sources['gdn_backend']['path']), 'GDN original method')
        signature = inspect.signature(original)
        self.active = dict(size=size, phase='warmup', warmup_step=1, method=method, calls=0, offsets=[], b=None, state=None, branch=None, arguments=None, restored=False)
        self.record['warmups'].append(self.active)

        def wrapper(*args, **kwargs):
            try:
                bound = signature.bind(*args, **kwargs)
                values = dict(bound.arguments)
                values.update(values.pop('kwargs', {}))
                row = self.active
                if self.fold:
                    values.update(A_log=values['layer'].A_log, dt_bias=values['layer'].dt_bias)
                b, state = (tensor_metadata(values['b']), tensor_metadata(values['ssm_states']))
                operands = {name: tensor_metadata(values[name]) for name in ('A_log', 'dt_bias', 'a')} if self.alignment is not None else {}
                offset = state['offset']
                state['offset'] = self.record['pool']['offsets'][0]
                need(row['calls'] < len(self.record['pool']['layer_ids']) and offset not in row['offsets'], 'GDN extra or duplicate layer call')
                if method == 'packed_decode':
                    arguments = {name: values.get(name) is not None for name in ('replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_write_pos', 'replayssm_force_flush')}
                    branch = 'replayssm_decode' if all((arguments[name] for name in ('replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_write_pos'))) else 'legacy_packed_decode'
                elif self.fold:
                    arguments = dict(retrieve_parent_token=values.get('retrieve_parent_token') is not None, intermediate_states_buffer=False, cache_ring=True, ring_steps=values['layer_cache'].replayssm_rawv.shape[-2])
                    branch = 'aligned_ring_target_verify'
                else:
                    arguments = dict(retrieve_parent_token=values.get('retrieve_parent_token') is not None, intermediate_states_buffer=values.get('intermediate_states_buffer') is not None, cache_steps=values.get('cache_steps'))
                    branch = 'generic_target_verify'
                if row['calls']:
                    need(({k: v for k, v in row['b'].items() if k != 'offset'}, row['state'], row['branch'], row['arguments']) == ({k: v for k, v in b.items() if k != 'offset'}, state, branch, arguments), 'GDN inconsistent layer metadata')
                    need(all(({k: v for k, v in row[name].items() if k != 'offset'} == {k: v for k, v in operand.items() if k != 'offset'} for name, operand in operands.items())), 'GDN inconsistent gate metadata')
                else:
                    row.update(b=b, state=state, branch=branch, arguments=arguments)
                    row.update(operands)
                    if self.alignment is not None:
                        need(b['dtype'] == 'torch.bfloat16' and (method == 'packed_decode' and branch == 'legacy_packed_decode' or ((method == 'target_verify' and arguments['intermediate_states_buffer'] or (self.fold and arguments['cache_ring'])) and (not arguments['retrieve_parent_token']))), 'GDN guarded candidate branch preconditions')
                        row['selected_branch'] = 'legacy_packed_decode' if method == 'packed_decode' else 'bf16_sigmoid_beta' if self.alignment == 'beta-bf16' else 'fp32_exp_beta'
                        row['branch_basis'] = 'inferred from live packed arguments and authenticated packed_decode source; not device constexpr capture' if method == 'packed_decode' else 'inferred from live b dtype and authenticated target_verify constants; not device constexpr capture'
                row['calls'] += 1
                row['offsets'].append(offset)
                return original(*args, **kwargs)
            except BaseException:
                try:
                    self.restore()
                except BaseException:
                    pass
                if self.on_failure is not None:
                    try:
                        self.on_failure()
                    except BaseException:
                        pass
                raise
        setattr(self.subject, method, wrapper)
        self.saved[method] = wrapper

    def finish(self):
        need(self.active is not None, 'GDN warmup missing')
        self.restore()
        self.active['offsets'].sort()
        self.active = None

def need(value, message):
    if not value:
        raise ValueError('HTTP native graph evidence: ' + message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'

class Observer:

    def __init__(self, rank, evidence, backend_sha256, arm, sources=None):
        need(type(rank) is int and rank in (0, 1) and (arm in ('control', 'mtp')), 'rank/arm')
        self.rank, self.arm, self.pin = (rank, arm, backend_sha256)
        self.root = Path(evidence)
        self.events, self.identity = ([], None)
        self.target_backend = None
        self.sources = sources if gdn_enabled(sources) else None
        self.gdn = None
        self.captures = self.replays = self.commits = 0
        self.commit_identity = self.commit_callable = self.commit_code = None
        self.closure = closure_phase()
        import mtp_contract
        self.phase = mtp_contract.settings()['phase']
        self.commit_limit = mtp_contract.commit_observation_limit()
        self.dflash = self.closure
        self.bs1_replays = self.bs2_replays = 0
        counter_bytes = 32 if self.closure else 16
        path = self.root / f'graph-counter-rank-{rank}.bin'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
        try:
            os.ftruncate(fd, counter_bytes)
            self.counter = mmap.mmap(fd, counter_bytes)
        finally:
            os.close(fd)
        self.write_counter()
        self.commit_counter = None
        if mtp_contract.host_path_cleanup_enabled(arm):
            path = self.root / f'replay-commit-counter-rank-{rank}.bin'
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
            try:
                os.ftruncate(fd, 8)
                self.commit_counter = mmap.mmap(fd, 8, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)
        self.write_commit_counter()
        self.publish()

    def write_counter(self):
        self.counter[:] = struct.pack('<QQQQ', self.captures, self.replays, self.bs1_replays, self.bs2_replays) if self.closure else struct.pack('<QQ', self.captures, self.replays)

    def write_commit_counter(self):
        if self.commit_counter is not None:
            struct.pack_into('<Q', self.commit_counter, 0, self.commits)
            return
        path = self.root / f'replay-commit-counter-rank-{self.rank}.bin'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 384)
        try:
            need(os.write(fd, struct.pack('<Q', self.commits)) == 8, 'bounded commit counter write')
        finally:
            os.close(fd)

    def commit_complete(self, hybrid, helper):
        need(self.target_backend is not None and hybrid is self.target_backend._cuda_graph_runner.model_runner.attn_backend, 'native completed commit backend identity')
        if self.commit_identity is None:
            import mtp_contract
            fold = mtp_contract.replay_serving_enabled(self.arm)
            source = self.sources['replay_fold' if fold else 'replay_scatter']
            name = 'commit_gdn_replayssm_fold_after_verify' if fold else 'scatter_mamba_states_after_mtp_verify'
            module = sys.modules.get(source['module'])
            spec = getattr(module, '__spec__', None)
            need(inspect.isfunction(helper) and helper.__name__ == name and (helper.__module__ == source['module']) and (module is not None) and (getattr(module, name, None) is helper) and isinstance(getattr(spec, 'loader', None), SourceFileLoader) and (spec.origin == module.__file__ == helper.__code__.co_filename == source['path']) and (hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']), 'completed native commit helper source identity')
            self.commit_identity = dict(source, name=name)
            self.commit_callable, self.commit_code = (helper, helper.__code__)
            self.publish()
        need(helper is self.commit_callable and helper.__code__ is self.commit_code, 'completed native commit helper substituted')
        need(self.commits < self.commit_limit, 'finite commit observation count')
        self.commits += 1
        self.write_commit_counter()

    def publish(self):
        path = self.root / f'graph-captures-rank-{self.rank}.json'
        value = dict(schema=1, rank=self.rank, arm=self.arm, pid=os.getpid(), backend_identity=self.identity, events=self.events, counter_layout='<QQ captures,replays', native_graph_api_observations=True, qualification=False, gpu_time_measured=False)
        if self.closure:
            value.update(phase=self.phase, counter_layout=CLOSURE_LAYOUT, replay_batch_binding='original-forward-batch.batch_size=runner.raw_bs=shape.size')
        if self.sources is not None:
            value['gdn'] = None if self.gdn is None else self.gdn.record
            value['commit_source'] = self.commit_identity
        if hasattr(self, 'token_probe'):
            value['token_probe'] = self.token_probe
            need(len(canonical(value)) < 65536, 'J01 combined capture byte budget')
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(path, flags, 384)
        try:
            os.write(fd, canonical(value))
        finally:
            os.close(fd)

    def checked_backend(self, backend, shape):
        cls = type(backend)
        module = sys.modules.get(MODULE)
        spec = getattr(module, '__spec__', None)
        need(cls.__module__ == MODULE and cls.__name__ == 'FullCudaGraphBackend' and (module is not None) and (getattr(module, cls.__name__, None) is cls) and isinstance(getattr(spec, 'loader', None), SourceFileLoader), 'loaded native backend class/source')
        path = Path(spec.origin)
        need(str(path) == module.__file__ and path.resolve() == path and (hashlib.sha256(path.read_bytes()).hexdigest() == self.pin), 'effective backend source hash')
        need(type(shape.size) is int and shape.size in (1, 2), 'actual native graph shape')
        runner = getattr(backend, '_cuda_graph_runner', None)
        model_runner = getattr(runner, 'model_runner', None)
        need(getattr(model_runner, 'is_draft_worker', None) is False, 'draft graph callback refused')
        if self.closure:
            algorithm = model_runner.spec_algorithm
            need(algorithm.is_none() is False and algorithm.is_dflash() is True and (getattr(runner, 'capture_bs', None) == [1, 2]) and (getattr(runner, 'ragged_verify_mode', None) is False) and (getattr(runner, 'require_mlp_tp_gather', None) is False) and (type(getattr(runner, 'captured_req_width', None)) is int) and (runner.captured_req_width == 8), 'fixed DFlash8 request-count buckets required')
        if self.target_backend is None:
            self.target_backend = backend
        need(backend is self.target_backend, 'target backend substituted')
        if self.identity is None:
            self.identity = dict(module=MODULE, path=str(path), sha256=self.pin, class_name=cls.__name__, model_runner_role='target')
        return shape.size

    def capture_forward(self, backend, shape, phase, warmup_step=None):
        role = getattr(getattr(getattr(backend, '_cuda_graph_runner', None), 'model_runner', None), 'is_draft_worker', None)
        need(type(role) is bool, 'native graph worker role missing')
        if role:
            return
        if phase == 'capture' and self.gdn is not None and self.gdn.saved:
            self.gdn.restore()
        need(len(self.events) < 6, 'capture count')
        size = self.checked_backend(backend, shape)
        need(phase == 'warmup' and type(warmup_step) is int and (warmup_step in (0, 1)) or (phase == 'capture' and warmup_step is None), 'native capture phase')
        if self.sources is not None:
            if phase == 'warmup' and warmup_step == 1:
                need(self.events and self.events[-1] == dict(size=size, phase='warmup', warmup_step=0), 'GDN warmup0 must precede observed warmup1')
                if self.gdn is None:
                    self.gdn = GDNObservation(backend._cuda_graph_runner.model_runner, self.sources, self.arm, self.rank, on_failure=self.publish, on_commit=self.commit_complete)
                self.gdn.begin(size)
            elif phase == 'capture':
                need(self.gdn is not None, 'GDN observer warmup missing')
                self.gdn.finish()
        self.events.append(dict(size=size, phase=phase, warmup_step=warmup_step))
        self.captures += int(phase == 'capture')
        self.write_counter()
        self.publish()

    def backend_replayed(self, backend, shape, static_forward_batch):
        role = getattr(getattr(getattr(backend, '_cuda_graph_runner', None), 'model_runner', None), 'is_draft_worker', None)
        need(type(role) is bool, 'native replay worker role missing')
        if role:
            return
        need(backend is self.target_backend, 'target replay backend substituted')
        if self.closure:
            runner = backend._cuda_graph_runner
            raw_bs = getattr(static_forward_batch, 'batch_size', None)
            need(self.identity is not None and type(shape.size) is int and (shape.size in (1, 2)) and (type(raw_bs) is int) and (raw_bs == shape.size) and (type(getattr(runner, 'raw_bs', None)) is int) and (runner.raw_bs == raw_bs), 'observed native replay must bind the actual unpadded request count')
            self.bs1_replays += int(raw_bs == 1)
            self.bs2_replays += int(raw_bs == 2)
        else:
            need(self.identity is not None and shape.size == 1, 'observed singleton native replay')
        self.replays += 1
        self.write_counter()

def install(rank, evidence, backend_sha256, arm, sources=None):
    global CURRENT
    need(CURRENT is None, 'one observer per native worker')
    CURRENT = Observer(rank, evidence, backend_sha256, arm, sources=sources)
    return CURRENT

def snapshot(evidence, arm, sources=None):
    result = []
    closure = closure_phase()
    for rank in (0, 1):
        root = Path(evidence)
        raw = (root / f'graph-counter-rank-{rank}.bin').read_bytes()
        need(len(raw) == (32 if closure else 16) and raw == (root / f'graph-counter-rank-{rank}.bin').read_bytes(), 'stable bounded native counter')
        counts = struct.unpack('<QQQQ' if closure else '<QQ', raw)
        captures, replays = counts[:2]
        startup = json.loads((root / f'graph-captures-rank-{rank}.json').read_bytes())
        need(startup['rank'] == rank and startup['arm'] == arm, 'rank/arm capture binding')
        if closure:
            need(startup.get('phase') == __import__('mtp_contract').settings()['phase'] and startup.get('counter_layout') == CLOSURE_LAYOUT and (startup.get('replay_batch_binding') == 'original-forward-batch.batch_size=runner.raw_bs=shape.size') and (counts[2] + counts[3] == replays), 'closure native counter layout/count binding')
        expected = sorted(((size, phase, step) for size in (1, 2) for phase, step in (('warmup', 0), ('warmup', 1), ('capture', None))))
        actual = sorted(((row['size'], row['phase'], row['warmup_step']) for row in startup['events']))
        identity = startup['backend_identity']
        need(captures == 2 and replays > 0 and (actual == expected) and (type(identity) is dict) and (identity.get('model_runner_role') == 'target'), 'both target native graph capture buckets and replay required')
        row = dict(rank=rank, pid=startup['pid'], captures=captures, replays=replays, backend_identity=startup['backend_identity'])
        if closure:
            row.update(bs1_replays=counts[2], bs2_replays=counts[3])
        if gdn_enabled(sources) or 'gdn' in startup:
            validate_gdn_record(startup.get('gdn'), arm, rank, startup['pid'], sources=sources)
            row['gdn_sha256'] = hashlib.sha256(canonical(startup['gdn'])).hexdigest()
        commit_path = root / f'replay-commit-counter-rank-{rank}.bin'
        import mtp_contract
        fold = mtp_contract.replay_serving_enabled(arm)
        expected_commit = dict(sources['replay_fold' if fold else 'replay_scatter'], name='commit_gdn_replayssm_fold_after_verify' if fold else 'scatter_mamba_states_after_mtp_verify')
        need(startup.get('commit_source') == expected_commit, 'observed completed native commit source')
        commit_raw = commit_path.read_bytes()
        need(len(commit_raw) == 8 and commit_raw == commit_path.read_bytes(), 'stable bounded commit counter')
        row['completed_commits'] = struct.unpack('<Q', commit_raw)[0]
        need(row['completed_commits'] <= mtp_contract.commit_observation_limit(), 'finite completed commits')
        result.append(row)
    return dict(schema=1, arm=arm, observed_monotonic=time.monotonic(), ranks=result, native_graph_api_observations=True, gpu_time_measured=False)
