"""CPU metadata validation shared by the proposed producer and offline reader.

No tensor conversion, scalar read, allocation, model import or device call.
The caller must bind the resulting record to the existing worker/session record.
"""
import math


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
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, ValueError)


need = _SPRY_COMMON.need


def tensor_metadata(tensor):
    shape = list(tensor.shape)
    need(len(shape) == 3 and all(type(n) is int and n > 0 for n in shape), 'native 3D cache shape')
    result = dict(shape=shape, dtype=str(tensor.dtype), device=str(tensor.device),
                  device_type=tensor.device.type, pointer=tensor.data_ptr(),
                  elements=tensor.numel(), element_bytes=tensor.element_size())
    validate_tensor(result)
    return result


def validate_tensor(row):
    need(set(row) == {'shape', 'dtype', 'device', 'device_type', 'pointer', 'elements', 'element_bytes'},
         'exact tensor metadata fields')
    need(type(row['shape']) is list and len(row['shape']) == 3 and
         all(type(n) is int and n > 0 for n in row['shape']), 'tensor shape')
    need(row['device_type'] == 'cuda' and type(row['device']) is str and
         type(row['pointer']) is int and row['pointer'] > 0, 'device/pointer metadata')
    need(type(row['elements']) is int and row['elements'] == math.prod(row['shape']) and
         type(row['element_bytes']) is int and row['element_bytes'] in (1, 2), 'tensor payload metadata')


def startup(scheduler, *, target_fp8, target_cache_dtype=None):
    target = scheduler.tp_worker.model_runner
    draft = scheduler.model_worker.draft_model_runner
    need(type(target_fp8) is bool and target is not draft, 'distinct target/draft runner identities')
    result = {}
    pointers = set()
    for name, runner in (('target', target), ('draft', draft)):
        fp8 = name == 'target' and target_fp8
        pool = getattr(runner.token_to_kv_pool, 'full_kv_pool', runner.token_to_kv_pool)
        need(runner.is_draft_worker is (name == 'draft'), 'target/draft runner role')
        need(str(runner.dtype) == 'torch.bfloat16', 'compute dtype remains BF16')
        dtype = 'torch.float8_e4m3fn' if fp8 else 'torch.bfloat16'
        storage = 'torch.uint8' if fp8 else 'torch.bfloat16'
        need(str(runner.kv_cache_dtype) == str(pool.dtype) == dtype and
             str(pool.store_dtype) == storage, 'resolved pool/public/backing dtype')
        tag = runner.kv_cache_dtype_str
        need(tag == ('bf16' if name == 'draft' else target_cache_dtype or ('fp8_e4m3' if fp8 else 'auto')),
             'resolved cache tag; source mapping still requires native selector read')
        sides = {}
        for side in ('k_buffer', 'v_buffer'):
            buffers = getattr(pool, side)
            need(type(buffers) in (list, tuple) and 1 <= len(buffers) <= 64, 'bounded native cache inventory')
            rows = [tensor_metadata(tensor) for tensor in buffers]
            for row in rows:
                need(row['dtype'] == storage and row['element_bytes'] == (1 if fp8 else 2),
                     'backing tensor dtype/width')
                need(row['pointer'] not in pointers, 'K/V/layer/target/draft cache storage aliases')
                pointers.add(row['pointer'])
            shapes = {}
            for row in rows:
                key = tuple(row['shape'])
                shapes[key] = shapes.get(key, 0)+1
            sides[side] = dict(buffers=len(rows), dtype=storage, element_bytes=(1 if fp8 else 2),
                shape_counts=[dict(shape=list(shape), count=count) for shape, count in sorted(shapes.items())],
                pointers=[row['pointer'] for row in rows],
                payload_bytes=sum(row['elements']*row['element_bytes'] for row in rows))
        need(sides['k_buffer']['buffers'] == sides['v_buffer']['buffers'], 'K/V layer inventory')
        result[name] = dict(compute_dtype=str(runner.dtype), kv_cache_dtype=dtype, kv_cache_dtype_str=tag,
                            store_dtype=storage, pool_type=type(pool).__name__, **sides)
    validate_startup(result, target_fp8=target_fp8, target_cache_dtype=target_cache_dtype)
    return result


def validate_startup(result, *, target_fp8, target_cache_dtype=None):
    need(set(result) == {'target', 'draft'} and type(target_fp8) is bool, 'both runner records')
    need(target_cache_dtype in (None, 'bf16') and (target_cache_dtype is None or not target_fp8),
         'explicit BF16 target cache profile')
    pointers = set()
    for name, runner in result.items():
        fp8 = name == 'target' and target_fp8
        storage = 'torch.uint8' if fp8 else 'torch.bfloat16'
        need(runner['compute_dtype'] == 'torch.bfloat16' and
             runner['kv_cache_dtype'] == ('torch.float8_e4m3fn' if fp8 else 'torch.bfloat16') and
             runner['store_dtype'] == storage and runner['kv_cache_dtype_str'] ==
             ('bf16' if name == 'draft' else target_cache_dtype or ('fp8_e4m3' if fp8 else 'auto')), 'offline target/draft precision isolation')
        for side in ('k_buffer', 'v_buffer'):
            row = runner[side]
            need(type(row['buffers']) is int and 1 <= row['buffers'] <= 64 and
                 len(row['pointers']) == row['buffers'] and row['dtype'] == storage and
                 row['element_bytes'] == (1 if fp8 else 2), 'offline native inventory')
            count = size = 0
            for group in row['shape_counts']:
                shape = group['shape']
                need(len(shape) == 3 and all(type(n) is int and n > 0 for n in shape) and
                     type(group['count']) is int and group['count'] > 0, 'shape histogram')
                count += group['count']
                size += math.prod(shape)*group['count']*row['element_bytes']
            need(count == row['buffers'] and size == row['payload_bytes'], 'offline payload accounting')
            for pointer in row['pointers']:
                need(type(pointer) is int and pointer > 0 and pointer not in pointers, 'offline cache storage isolation')
                pointers.add(pointer)
        need(runner['k_buffer']['buffers'] == runner['v_buffer']['buffers'], 'offline K/V inventory')
    return result


def route(backend, q, kb, vb, layer, *, route_name, capture):
    need(backend.bf16_query_fp8_target_kv is True, 'target-only FP8 route')
    outer_pool = backend.token_to_kv_pool
    pool = getattr(outer_pool, 'full_kv_pool', outer_pool)
    storage_layer = layer.layer_id
    need(type(storage_layer) is int and storage_layer >= 0, 'native global layer')
    if pool is not outer_pool:
        # Match source-bound HybridLinearKVPool.get_{key,value}_buffer:
        # global layer -> dense full-attention layer -> inner pool offset.
        mapping = getattr(outer_pool, 'full_attention_layer_id_mapping', None)
        need(type(mapping) is dict and 1 <= len(mapping) <= 64 and
             all(type(key) is int and key >= 0 and type(value) is int and value >= 0
                 for key, value in mapping.items()) and
             len(set(mapping.values())) == len(mapping) and storage_layer in mapping,
             'native full-attention layer mapping')
        storage_layer = mapping[storage_layer]
    need(type(pool.start_layer) is int and pool.start_layer >= 0, 'native inner start layer')
    index = storage_layer - pool.start_layer
    need(0 <= index < len(pool.k_buffer) == len(pool.v_buffer) <= 64, 'native local layer')
    record = dict(route=route_name, capture=capture, layer=layer.layer_id,
                  q=tensor_metadata(q), k_view=tensor_metadata(kb), v_view=tensor_metadata(vb),
                  k_backing=tensor_metadata(pool.k_buffer[index]),
                  v_backing=tensor_metadata(pool.v_buffer[index]),
                  k_scale=layer.k_scale_float, v_scale=layer.v_scale_float,
                  scale_origin='actual resolved layer.k_scale_float and layer.v_scale_float; checkpoint/default provenance not determined here',
                  extra_tensor_operations=0, cpu_tensor_copies=0)
    validate_route(record)
    return record


def validate_route(record):
    need(record['route'] in ('extend', 'decode', 'verify_decode') and type(record['capture']) is bool,
         'route/capture category')
    for name in ('q', 'k_view', 'v_view', 'k_backing', 'v_backing'):
        validate_tensor(record[name])
    need(record['q']['dtype'] == 'torch.bfloat16' and record['q']['shape'][1:] == [12, 256], 'BF16 Q route')
    for side in ('k', 'v'):
        view, backing = record[side+'_view'], record[side+'_backing']
        need(view['dtype'] == 'torch.float8_e4m3fn' and backing['dtype'] == 'torch.uint8' and
             view['element_bytes'] == backing['element_bytes'] == 1 and
             view['shape'] == backing['shape'] and view['shape'][1:] == [2, 256] and
             view['pointer'] == backing['pointer'] and view['device'] == backing['device'] == record['q']['device'],
             'raw uint8 to public FP8 view alias')
        value = record[side+'_scale']
        need(type(value) is float and math.isfinite(value) and value > 0, 'actual independent positive K/V scale')
    need(record['extra_tensor_operations'] == record['cpu_tensor_copies'] == 0, 'metadata-only observation')
    return record


def validate_bound_routes(records, manifest, workers):
    """Validate route records with existing owner-authenticated worker identities.

    Host records alone do not finish request or graph-replay joins. Those remain
    required consumers of the existing request/graph journals before acceptance.
    """
    need(set(workers) == {0, 1}, 'two expected rank identities')
    sequences = {0: 0, 1: 0}
    external = set()
    for row in records:
        rank = row.get('rank')
        need(type(rank) is int and rank in workers, 'unexpected rank')
        for key in ('arm','session_sha256','spec_sha256','source_manifest_sha256'):
            need(row[key] == manifest[key], 'source/request scope identity')
        worker = workers[rank]
        need(row['worker_identity_sha256'] == worker['identity_sha256'] and
             all(row[key] == worker[key] for key in ('pid','start_ticks','boot_id')), 'worker lifecycle identity')
        need(row['helper_source'] == manifest['sources']['j01_probe'] and
             row['backend_source'] == manifest['sources']['triton_backend'], 'producer source identity')
        need(type(row['sequence']) is int and row['sequence'] == sequences[rank] and sequences[rank] < 128,
             'bounded contiguous per-rank sequence')
        sequences[rank] += 1
        need(row['stage'] in ('startup','external'), 'known lifecycle stage')
        validate_route(row['evidence'])
        need(row['evidence']['q']['device'] == 'cuda:'+str(rank), 'selected TP rank/device mapping')
        if row['stage'] == 'external':
            external.add(rank)
    need(external == {0,1}, 'both ranks external host call evidence required')
    return dict(host_metadata_complete=True, request_join_complete=False,
                graph_replay_join_complete=False, numerical_acceptance=False,
                serving_acceptance=False, records_by_rank=sequences)
