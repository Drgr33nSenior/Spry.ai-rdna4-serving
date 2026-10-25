"""Frozen source excerpts for metadata-only validation regressions.

Retained 2026-10-25 worker startup producer and strict reader. These are test
inputs, not active candidate validation or native/GPU qualification. Function
bodies below are verbatim; imports and runtime dependencies are supplied by the
test harness. No private paths or runtime packages are required.
"""

SOURCE_PROVENANCE = {
    'corrected': ('static-width-readiness-proof-correction-20261025-v1/candidate/worker_entry.py', '45bc64fe0174058ee01e60e69d6b95fcf59487ab8157e2a9a6df80b0c7b55585'),
    'original': ('static-v5-approved-execution-preparation-20261025-v1/approved-author-v2/review/worker_entry.py', '22dbe8b191b23b6b0df6abac823212bb3ff115218a1de9c46975b5ff70f771ee'),
    'reader': ('static-v5-approved-execution-preparation-20261025-v1/approved-author-v2/review/serving_integration.py', 'e5d150af78e3b03a32e100d0695bd36866d80d1534e14c4f8acdfcf105c1f41a'),
}

# Original consumed producer, retained verbatim as source data.
ORIGINAL_PRECISION_WEIGHTS = '''def _precision_weights(scheduler, manifest, worker, observer, contract, head_record):
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
    role = 'head_unquant' if bf16 else 'fp8_config'
    method_source = manifest['sources'][role]
    method_module = _native_module(method_source)
    method = linear.quant_method
    expected_class = 'UnquantizedLinearMethod' if bf16 else 'Fp8LinearMethod'
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
    observer.validate_precision_weights(record, binding=manifest, worker=worker)
    _atomic_json(Path(manifest['evidence_dir']) / f"PRECISION-WEIGHTS-rank-{worker['rank']}.json", record)
    if contract.full_enabled_benchmark():
        enabled=contract.contract_for_arm(manifest['arm'])['enabled'];adapter=(_source_module(manifest['sources']['verify_decode_adapter']).CURRENT if enabled else None)
        proof={key:manifest[key] for key in ('mode','arm','session_sha256','spec_sha256','source_manifest_sha256')}
        proof.update(schema=1,rank=worker['rank'],pid=worker['pid'],worker_identity_sha256=worker['identity_sha256'],enabled=enabled,block_size=contract.contract_for_arm(manifest['arm'])['speculative_dflash_block_size'],before_external_generations=True,extra_tensor_operations=0,extra_model_forwards=0,qualification=False)
        proof.update(adapter_source=manifest['sources']['verify_decode_adapter'] if enabled else None,eligible_host_calls=adapter.eligible_host_calls if adapter else 0,capture_host_calls=adapter.capture_host_calls if adapter else 0,layers=sorted(adapter.layers) if adapter else [],buffers=len(adapter.buffers) if adapter else 0)
        proof.update(metadata_build_host_calls=adapter.metadata_build_host_calls if adapter else 0,metadata_reuse_host_calls=adapter.metadata_reuse_host_calls if adapter else 0,metadata_capture_build_host_calls=adapter.metadata_capture_build_host_calls if adapter else 0)
        proof['indices_elements'] = sorted(int(v['indices'].numel()) for v in adapter.buffers.values()) if adapter else []
        _require(bool(proof['indices_elements']) and all(n == 8 * contract.precision_profile()['context_length'] for n in proof['indices_elements']), '64K split-KV indices geometry differs')
        _require(not enabled or (adapter.enabled and proof['capture_host_calls']>0 and proof['eligible_host_calls']>=proof['capture_host_calls'] and len(proof['layers'])==16 and proof['buffers']>0),'split-KV adapter not captured')
        _atomic_json(Path(manifest['evidence_dir']) / f"FULL-ENABLED-rank-{worker['rank']}.json",proof)
    return record
'''

# corrected:_require SHA256 9e9bd3ad80e6cbe021c24f84ada68374761935e789ab8c2ffff0ec3829b56a81
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerEntryError(message)


# corrected:_full_enabled_indices SHA256 0b1ea303f5ea1dd2ab380ce7ebbc7a624e8ab0d50650b6c390a95f3c5a212ac1
def _full_enabled_indices(adapter, arm_contract, context_length, device):
    """Authenticate existing graph-owned index descriptors; no device operations."""
    width = arm_contract['speculative_num_draft_tokens']
    _require(type(width) is int and width in (4, 8) and
             type(arm_contract['speculative_dflash_block_size']) is int and
             arm_contract['speculative_dflash_block_size'] == width and
             arm_contract['speculative_algorithm'] == 'DFLASH' and
             type(context_length) is int and context_length in (32768, 65536) and
             type(device) is str and device.startswith('cuda:'),
             'split-KV exact arm width/context/device contract differs')
    _require(adapter is not None and type(adapter.buffers) is dict and adapter.buffers and
             type(adapter.buffer_widths) is dict and
             set(adapter.buffer_widths) == set(adapter.buffers),
             'split-KV graph-owned buffer/width inventory differs')
    expected = width * context_length
    elements, ranges = [], []
    for key, buffers in adapter.buffers.items():
        _require(type(key) is int and key > 0 and type(adapter.buffer_widths[key]) is int and
                 adapter.buffer_widths[key] == width and type(buffers) is dict and
                 'indices' in buffers, 'split-KV captured buffer width differs')
        indices = buffers['indices']
        count, pointer = indices.numel(), indices.data_ptr()
        _require(type(count) is int and count == expected and
                 tuple(indices.shape) == (expected,) and tuple(indices.stride()) == (1,) and
                 str(indices.dtype) == 'torch.int64' and indices.element_size() == 8 and
                 indices.is_cuda is True and str(indices.device) == device and
                 indices.is_contiguous() is True,
                 'split-KV width-aware indices geometry/dtype/device differs')
        _require(type(pointer) is int and pointer > 0 and pointer % 8 == 0 and
                 all(pointer + count * 8 <= start or end <= pointer for start, end in ranges),
                 'split-KV graph-owned indices pointer/alias differs')
        ranges.append((pointer, pointer + count * 8))
        elements.append(count)
    return sorted(elements)


# corrected:_precision_weights SHA256 f2d406b8bfa66b8007c2b18a1e6d5cc0bae11de1ab0e3426d2e982c83608ac75
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
    role = 'head_unquant' if bf16 else 'fp8_config'
    method_source = manifest['sources'][role]
    method_module = _native_module(method_source)
    method = linear.quant_method
    expected_class = 'UnquantizedLinearMethod' if bf16 else 'Fp8LinearMethod'
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
    observer.validate_precision_weights(record, binding=manifest, worker=worker)
    _atomic_json(Path(manifest['evidence_dir']) / f"PRECISION-WEIGHTS-rank-{worker['rank']}.json", record)
    if contract.full_enabled_benchmark():
        enabled=contract.contract_for_arm(manifest['arm'])['enabled'];adapter=(_source_module(manifest['sources']['verify_decode_adapter']).CURRENT if enabled else None)
        proof={key:manifest[key] for key in ('mode','arm','session_sha256','spec_sha256','source_manifest_sha256')}
        proof.update(schema=1,rank=worker['rank'],pid=worker['pid'],worker_identity_sha256=worker['identity_sha256'],enabled=enabled,block_size=contract.contract_for_arm(manifest['arm'])['speculative_dflash_block_size'],before_external_generations=True,extra_tensor_operations=0,extra_model_forwards=0,qualification=False)
        proof.update(adapter_source=manifest['sources']['verify_decode_adapter'] if enabled else None,eligible_host_calls=adapter.eligible_host_calls if adapter else 0,capture_host_calls=adapter.capture_host_calls if adapter else 0,layers=sorted(adapter.layers) if adapter else [],buffers=len(adapter.buffers) if adapter else 0)
        proof.update(metadata_build_host_calls=adapter.metadata_build_host_calls if adapter else 0,metadata_reuse_host_calls=adapter.metadata_reuse_host_calls if adapter else 0,metadata_capture_build_host_calls=adapter.metadata_capture_build_host_calls if adapter else 0)
        proof['indices_elements'] = _full_enabled_indices(adapter, contract.contract_for_arm(manifest['arm']), contract.precision_profile()['context_length'], str(weight.device))
        _require(bool(proof['indices_elements']), 'split-KV indices buffers absent')
        _require(not enabled or (adapter.enabled and proof['capture_host_calls']>0 and proof['eligible_host_calls']>=proof['capture_host_calls'] and len(proof['layers'])==16 and proof['buffers']>0),'split-KV adapter not captured')
        _atomic_json(Path(manifest['evidence_dir']) / f"FULL-ENABLED-rank-{worker['rank']}.json",proof)
    return record


# reader:_need SHA256 af4a0cd486edba401500840c3294d7f7dfc93be5033ed72fa2bc4d0898204b54
def _need(value, message):
    if not value:
        raise ValueError('HTTP integration: ' + message)


# reader:validate_full_enabled_record SHA256 592e2db0d53450fbac78485484ff583d906260ffb34b8064a8826a0ebc2ee315
def validate_full_enabled_record(record, binding, worker):
    fields = {'mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'schema', 'rank', 'pid', 'worker_identity_sha256', 'enabled', 'block_size', 'before_external_generations', 'extra_tensor_operations', 'extra_model_forwards', 'qualification', 'adapter_source', 'eligible_host_calls', 'capture_host_calls', 'layers', 'buffers'}
    fields.update({'metadata_build_host_calls', 'metadata_reuse_host_calls', 'metadata_capture_build_host_calls'})
    fields.add('indices_elements')
    _need(type(record) is dict and set(record) == fields and (record['schema'] == 1), 'full-enabled witness schema')
    for key in ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256'):
        _need(record[key] == binding[key], 'full-enabled witness source/session binding')
    _need(record['rank'] == worker['rank'] and record['pid'] == worker['pid'] and (record['worker_identity_sha256'] == worker['identity_sha256']), 'full-enabled witness worker identity')
    enabled = _mtp_contract().contract_for_arm(binding['arm'])['enabled']
    _need(record['enabled'] is enabled and record['before_external_generations'] is True and (record['qualification'] is False) and (record['extra_tensor_operations'] == record['extra_model_forwards'] == 0), 'full-enabled witness scope')
    for key in ('eligible_host_calls', 'capture_host_calls', 'buffers'):
        _need(type(record[key]) is int and record[key] >= 0, 'full-enabled witness count')
    if enabled:
        _need(record['block_size'] == _mtp_contract().contract_for_arm(binding['arm'])['speculative_num_draft_tokens'] and record['adapter_source'] == binding['sources']['verify_decode_adapter'] and (0 < record['capture_host_calls'] <= record['eligible_host_calls']) and (record['buffers'] > 0) and (type(record['layers']) is list) and (len(record['layers']) == len(set(record['layers'])) == 16), 'split-KV adapter absent or not captured')
    else:
        _need(record['block_size'] is record['adapter_source'] is None and record['layers'] == [] and (record['buffers'] == record['capture_host_calls'] == record['eligible_host_calls'] == 0), 'ordinary baseline unexpectedly enabled')
    counts = [record[key] for key in ('metadata_build_host_calls', 'metadata_reuse_host_calls', 'metadata_capture_build_host_calls')]
    _need(all((type(n) is int and n >= 0 for n in counts)), 'metadata reuse witness counts')
    if enabled:
        _need(type(record['indices_elements']) is list and len(record['indices_elements']) == record['buffers'] and binding.get('resolved_contract', {}).get('context_length') in (32768, 65536) and all(type(n) is int and n == _mtp_contract().contract_for_arm(binding['arm'])['speculative_num_draft_tokens'] * binding['resolved_contract']['context_length'] for n in record['indices_elements']), '64K split-KV indices geometry differs')
        _need(record['metadata_capture_build_host_calls'] == 1 and record['capture_host_calls'] == 16 and (record['metadata_build_host_calls'] >= 1) and (record['metadata_reuse_host_calls'] >= 15) and (record['metadata_build_host_calls'] + record['metadata_reuse_host_calls'] == record['eligible_host_calls']), 'metadata not reused across captured attention layers')
    else:
        _need(counts == [0, 0, 0], 'ordinary reference reported metadata reuse')
    return record


