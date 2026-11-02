"""One immutable FP8 head allocation shared by target and draft adapters.

Importing this module performs no native imports or device work. The worker
authenticates it before any model is loaded.
"""
import hashlib
import re

N, K = 124160, 5120
_CACHE = '_spry_shared_fp8_head'
HASH_CHUNK_BYTES = 8 << 20


def _finite_weight(weight, torch):
    """Scan bounded chunks, combine on device, and read one startup scalar."""
    finite = torch.ones((), dtype=torch.bool, device=weight.device)
    for row in range(0, N, 128):
        torch.logical_and(finite, torch.isfinite(weight[row:row + 128]).all(), out=finite)
    if not finite.item():
        raise ValueError('nonfinite D3b BF16 head weight')


def shared_fp8_head(head, torch, ops, *, expected_device):
    """Publish only a fully quantized pair; refuse stale weight bindings."""
    if type(expected_device) is not int or expected_device not in (0, 1):
        raise ValueError('shared FP8 head requires logical device zero or one')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('shared FP8 head installation must precede graph capture')
    weight = head.weight
    if (tuple(weight.shape) != (N, K) or weight.dtype != torch.bfloat16
            or not weight.is_contiguous() or weight.requires_grad
            or weight.device.type != 'cuda' or weight.device.index != expected_device):
        raise ValueError('shared FP8 head weight geometry/device differs')
    try:
        version = weight._version
    except RuntimeError as error:
        raise ValueError('shared FP8 head requires a versioned weight') from error
    quantize = ops.quantize_weight_preallocated
    function = getattr(quantize, '__func__', quantize)
    identity = (weight.data_ptr(), version, type(ops), function, expected_device)
    cache = getattr(head, _CACHE, None)
    if cache is not None:
        if type(cache) is not tuple or len(cache) != 4 or cache[0] is not weight or cache[1] != identity:
            raise RuntimeError('shared FP8 head binding changed; use a fresh model and recapture')
        return cache[2], cache[3], False
    _finite_weight(weight, torch)
    b8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((N // 128, K // 128), dtype=torch.float32, device=weight.device)
    ops.quantize_weight_preallocated(weight, b8, scales, expected_device=expected_device)
    setattr(head, _CACHE, (weight, identity, b8, scales))
    return b8, scales, True


def tensor_sha256(value, torch):
    """Hash contiguous tensor bytes in bounded CPU chunks, outside capture."""
    if torch.cuda.is_current_stream_capturing() or not value.is_contiguous():
        raise ValueError('startup tensor digest requires contiguous bytes outside capture')
    raw = value.detach().view(torch.uint8).reshape(-1)
    result = hashlib.sha256()
    for offset in range(0, raw.numel(), HASH_CHUNK_BYTES):
        result.update(memoryview(raw[offset:offset + HASH_CHUNK_BYTES].cpu().numpy()))
    return result.hexdigest()


def verify_shared_fp8_head(head, torch, ops, *, expected_device):
    """Instrumented startup only: compare a transient second quantization."""
    b8, scales, created = shared_fp8_head(head, torch, ops, expected_device=expected_device)
    if created:
        raise RuntimeError('shared head proof requires the installed target allocation')
    other_b8 = torch.empty_like(b8)
    other_scales = torch.empty_like(scales)
    try:
        ops.quantize_weight_preallocated(head.weight, other_b8, other_scales,
                                        expected_device=expected_device)
        equal = (torch.equal(b8.view(torch.uint8), other_b8.view(torch.uint8))
                 and torch.equal(scales.view(torch.uint8), other_scales.view(torch.uint8)))
        if not equal:
            raise ValueError('transient FP8 head quantization bytes differ')
    finally:
        del other_b8, other_scales
    return dict(equal=True, transient_quantizations=1,
                fp8_sha256=tensor_sha256(b8, torch), scales_sha256=tensor_sha256(scales, torch))


def validate_memory_recovery(record, *, rank):
    """Require complete startup byte and sharing proofs; no tensor operations."""
    def need(value, message):
        if not value:
            raise ValueError('memory recovery: ' + message)

    def pointer(value):
        return type(value) is int and 0 < value < 1 << 64

    def sha(value):
        return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

    need(type(rank) is int and rank in (0, 1), 'rank')
    need(type(record) is dict and set(record) == {'schema', 'policy', 'device', 'head', 'packed',
         'sharing', 'before_graph_capture', 'predicted_released_payload_bytes'}, 'fields')
    need(type(record['schema']) is int and record['schema'] == 1
         and record['policy'] == 'shared-head-packed-views-v1'
         and type(record['device']) is int and record['device'] == rank
         and record['before_graph_capture'] is True
         and type(record['predicted_released_payload_bytes']) is int
         and record['predicted_released_payload_bytes'] == 1711668800, 'scope')
    head = record['head']
    need(type(head) is dict and set(head) == {'equal', 'transient_quantizations',
         'fp8_sha256', 'scales_sha256'} and head['equal'] is True
         and type(head['transient_quantizations']) is int and head['transient_quantizations'] == 1
         and sha(head['fp8_sha256']) and sha(head['scales_sha256']), 'transient head bytes')
    sharing = record['sharing']
    need(type(sharing) is dict and set(sharing) == {'target', 'draft'}, 'head sharing')
    target, draft = sharing['target'], sharing['draft']
    need(type(target) is dict and type(draft) is dict and target == draft
         and set(target) == {'fp8_id', 'scales_id', 'fp8_pointer', 'scales_pointer'}
         and all(pointer(value) for value in target.values())
         and target['fp8_id'] != target['scales_id']
         and target['fp8_pointer'] != target['scales_pointer'], 'same head objects and storage')
    rows = record['packed']
    need(type(rows) is list and len(rows) == 48, 'all GDN layers')
    bases = set()
    fields = {'weight_offset_bytes', 'scale_offset_bytes', 'weight_base_pointer', 'scale_base_pointer',
              'weight_pointer', 'scale_pointer', 'p1_identity_rebound', 'equal_at_startup',
              'rebound_payload_bytes', 'weight_sha256', 'scale_sha256'}
    for row in rows:
        need(type(row) is dict and set(row) == {'qkvz', 'ba'}, 'both projection shards')
        for name, width, offset in (('qkvz', 8192, 0), ('ba', 48, 8192)):
            shard = row[name]
            need(type(shard) is dict and set(shard) == fields, 'packed shard fields')
            need(shard['equal_at_startup'] is True and shard['p1_identity_rebound'] is (name == 'qkvz')
                 and type(shard['rebound_payload_bytes']) is int
                 and shard['rebound_payload_bytes'] == width * (2560 + 160)
                 and sha(shard['weight_sha256']) and sha(shard['scale_sha256']), 'packed shard bytes')
            for kind, stride in (('weight', 2560), ('scale', 160)):
                base, view, displacement = (shard[kind + suffix] for suffix in
                                             ('_base_pointer', '_pointer', '_offset_bytes'))
                need(pointer(base) and pointer(view) and base % 256 == view % 256 == 0
                     and type(displacement) is int and displacement == offset * stride
                     and view == base + displacement, 'packed view offset/alignment')
                need(base == row['qkvz'][kind + '_base_pointer'], 'shared packed buffer')
        pair = (row['qkvz']['weight_base_pointer'], row['qkvz']['scale_base_pointer'])
        need(pair[0] != pair[1] and pair not in bases, 'distinct layer buffers')
        bases.add(pair)
    return record
