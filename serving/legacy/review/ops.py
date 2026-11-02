"""Preallocated isolated kernel calls; no model, install, owner or execution CLI.

The surrounding component must authenticate sources and CPU operand bytes before
allocation. These functions check metadata without reading device values. Warm
imports/JIT before capture; call on each card's explicit capture stream.
"""

from .prefill_reference import quantizer_geometry

CONFIG = {
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
    'GROUP_SIZE_M': 1, 'NUM_KSPLIT': 1, 'cache_modifier': None,
    'matrix_instr_nonkdim': 16, 'num_stages': 1, 'num_warps': 4, 'waves_per_eu': 1,
}


def validate_geometry(m, n, k):
    if any(type(x) is not int for x in (m, n, k)):
        raise ValueError('integer matrix dimensions required')
    if not 1 <= m <= 2048 or (n,k) not in ((17408,5120),(5120,8704),(8192,5120),(7168,5120),(5120,3072),(48,5120),(8240,5120)):  # 8240 = packed QKVZ 8192 + BA 48 (verify rows only)
        raise ValueError('undeclared matrix shape')


def runtime(expected_device):
    if type(expected_device) is not int or expected_device not in (0, 1):
        raise ValueError('declared card 0 or 1 required')
    import torch
    import triton
    from . import kernels
    if torch.cuda.current_device() != expected_device:
        raise ValueError('wrong current card')
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip' or target.arch != 'gfx1201':
        raise ValueError('this isolated experiment admits only gfx1201 HIP')
    return torch, triton, kernels


def _tensor(value, shape, dtype, device, stride):
    if tuple(value.shape) != shape or value.dtype != dtype or tuple(value.stride()) != stride:
        raise ValueError('tensor shape/dtype/stride mismatch')
    if value.device.type != 'cuda' or value.device.index != device:
        raise ValueError('tensor card mismatch')


def _distinct_storage(*values):
    pointers = [value.untyped_storage().data_ptr() for value in values]
    if not all(pointers) or len(set(pointers)) != len(pointers):
        raise ValueError('component tensors must have separate nonempty storage')


def quantize_preallocated(x, packed, scales, *, expected_device):
    torch, triton, kernels = runtime(expected_device)
    m, k = x.shape
    geometry = quantizer_geometry(m,k)
    _tensor(x, (m, k), torch.bfloat16, expected_device, (k, 1))
    _tensor(packed, (m, k // 2), torch.uint8, expected_device, (k // 2, 1))
    # Preserve upstream dynamic quantizer's transposed scale allocation layout.
    _tensor(scales, (m, k // 32), torch.uint8, expected_device, (1, m))
    _distinct_storage(x, packed, scales)
    grid = geometry.pop('grid')
    return kernels._dynamic_mxfp4_quant_kernel[grid](
        x, packed, scales, *x.stride(), *packed.stride(), *scales.stride(),
        M=m, N=k, MXFP4_QUANT_BLOCK_SIZE=32, SCALING_MODE=0, **geometry,
    )


def gemm_preallocated(x, weight, x_scales, weight_scales, y, *, expected_device):
    torch, triton, kernels = runtime(expected_device)
    m, packed_k = x.shape
    n, weight_packed_k = weight.shape
    k = packed_k * 2
    validate_geometry(m, n, k)
    if weight_packed_k != packed_k:
        raise ValueError('packed reduction dimension mismatch')
    _tensor(x, (m, k // 2), torch.uint8, expected_device, (k // 2, 1))
    _tensor(weight, (n, k // 2), torch.uint8, expected_device, (k // 2, 1))
    _tensor(x_scales, (m, k // 32), torch.uint8, expected_device, (1, m))
    _tensor(weight_scales, (n, k // 32), torch.uint8, expected_device, (k // 32, 1))
    _tensor(y, (m, n), torch.bfloat16, expected_device, (n, 1))
    _distinct_storage(x, weight, x_scales, weight_scales, y)
    # Screen 1: rows M<=16 take the 16-row tile of the unchanged kernel body (real-weight
    # component evidence: exact parity on gate/up, qkvz, out, qkv and down).
    # Screen 2: rows M>=1024 take the 128x128 tile with eight warps (P0, global attempt 82:
    # exact parity and about 19 percent lower time at M2047/M2048 on the real gate/up shard).
    # BLOCK_SIZE_K and NUM_KSPLIT stay fixed, so the K accumulation order is unchanged.
    block_m, block_n, warps = (16, 64, 4) if m <= 16 else (128, 128, 8) if m >= 1024 else (64, 64, 4)
    config = dict(CONFIG, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, num_warps=warps, SPLITK_BLOCK_SIZE=k)
    return kernels._gemm_afp4wfp4_kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        x, weight, y, x_scales, weight_scales,
        m, n, packed_k,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        0, y.stride(0), y.stride(1),
        x_scales.stride(0), x_scales.stride(1),
        weight_scales.stride(0), weight_scales.stride(1),
        **config,
    )
