"""INT4 KV micro-benchmark for gfx1201 verify attention (UQ3 investigation, 2026-11-12).

Question: what does reading the target KV cache as 4-bit integers cost on the R9700, against the
FP8 cache served today (patches 0026/0027, hardware fp8 decode)? gfx1201 has INT4/INT8 matrix and
dot instructions but no FP4 conversion, so only integer 4-bit formats are considered.

Variants, all timed the way patch 0021 / check_fp8_kv_dequant.py time verify: 16 layers, each on
its own cache (DRAM-fed), captured in one CUDA graph, median replay / 16:

  prod_bf16  production verify kernel (verify_attention_gfx1151), bf16 cache, its default tile
  prod_fp8   production verify kernel, fp8 e4m3 cache, "pairs" hardware decode (UQ2, served today)
  ctl_bf16   this tool's kernel on a bf16 cache: same structure as the INT4 variants, so the
             INT4 cost is read against it, not against a differently tuned kernel
  i4_bf16    UQ3 layout: signed 4-bit codes (-7..7, two per byte) and one e4m3 scale per 32
             values of a (token, kv head); unpacked and scaled to bf16 for both WMMA bf16 dots
  i4_i8dot   same cache; Q.K^T on the INT8 matrix path: q quantized to int8 per (row, q head),
             K codes sign-extended to int8, int32 dot per 32-value group times its scale;
             P.V as in i4_bf16 (checked, not timed: 4x the control)
  half_g128  split-half layout (byte i = dim i low nibble, dim 128 + i high), one scale per 128
             values folded into the score columns and P: two dots per tile, no interleave shuffle
  half_g32   split-half layout, one scale per 32 values applied per element as it is unpacked
  magic1_*   interleaved codes read as u16 words; v_perm_b32 places each nibble under a 0x43 byte,
             which is the bf16 value 128 + code (no int-to-float conversion); the constant offset
             is removed once per score row (sum of q) and once per output row (sum of P).
             V: one scale per token folded into P. K: one scale per 64 values applied after each
             64-wide dot (magic1_kg64) or one scale per token after the whole dot (magic1_kpt)

Not measured: the Hadamard rotation (once per q row and per output row, not per cached token),
the store path, extend/prefill and decode-only shapes. Correctness: every variant's output is
compared with an fp32 reference on the values that variant reads (kernel error), and the INT4
variants also with the bf16 reference (format error), at an 8k prefix.

Usage: bench_int4_kv.py [--quick] [--shapes 8192,32768x2]   (prints one JSON object per row; needs the SGLang image)
"""
import argparse
import json
import os
import sys

os.environ.setdefault("SGLANG_GFX1201_VERIFY_ATTENTION", "1")
os.environ.setdefault("SGLANG_GFX1201_FP8_KV_DEQUANT", "1")
os.environ.setdefault("SGLANG_GFX1201_FP8_KV_DECODE", "pairs")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

va = None  # the production module, imported on the GPU only (see _production)


def _production():
    global va
    if va is None:
        from sglang.kernels.ops.attention import verify_attention_gfx1151 as module
        va = module
    return va

HQ, HKV, D = 12, 2, 256  # per card at TP2: 24 q heads, 4 kv heads, head_dim 256
GQ = HQ // HKV
T = 8  # DFlash2 block
G = 32  # values per scale
FP8 = torch.float8_e4m3fn
DEV = "cuda"
SCALE = D**-0.5
LAYERS = 16
NSPLIT = 16
FMT_BF16, FMT_I4, FMT_I4_I8DOT = 0, 1, 2


@triton.jit
def _unpack_i4(codes, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[ROWS, COLS // 2] uint8 (value 2i in the low nibble, 2i + 1 in the high) -> [ROWS, COLS] int8 in -8..7."""
    lo = (codes & 15).to(tl.int8) - 8
    hi = (codes >> 4).to(tl.int8) - 8
    return tl.reshape(tl.join(lo, hi), (ROWS, COLS))


@triton.jit
def _split_kernel(q_ptr, qi8_ptr, qs_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, part_ptr, ml_ptr,
                  req_to_token, r2t_stride, req_pool_indices, seq_lens, scale,
                  T: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                  NSPLIT: tl.constexpr, BT: tl.constexpr, HG: tl.constexpr, QC: tl.constexpr,
                  FMT: tl.constexpr):
    """verify_attention_gfx1151's split kernel with a pluggable K/V read (see FMT_*)."""
    GQ: tl.constexpr = HQ // HKV
    NG: tl.constexpr = (GQ + HG - 1) // HG
    HP: tl.constexpr = 4 if HG <= 4 else 8
    R: tl.constexpr = 8 * HP
    GS: tl.constexpr = 32
    u = tl.program_id(0)
    g = u % NG
    j = (u // NG) % HKV
    s = (u // (NG * HKV)) % NSPLIT
    b = u // (NG * HKV * NSPLIT)
    prefix = tl.load(seq_lens + b).to(tl.int32)
    rp = tl.load(req_pool_indices + b).to(tl.int64)
    total = prefix + T
    chunk = tl.cdiv(total, NSPLIT)
    t0 = s * chunk
    t1 = tl.minimum(total, t0 + chunk)
    r = tl.arange(0, R)
    m = r // HP
    h = g * HG + r % HP
    valid = (r % HP < HG) & (h < GQ) & (m < T)
    seq_r = prefix + m + 1
    d = tl.arange(0, D)
    dq = tl.arange(0, QC)
    qrow = (b * T + m) * HQ + j * GQ + h
    qsafe = tl.where(valid, qrow, b * T * HQ + j * GQ)
    q_rows = q_ptr + qsafe[:, None] * D
    bt = tl.arange(0, BT)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    if FMT == 2:
        qscale = tl.load(qs_ptr + qsafe)
    for t in range(t0, t1, BT):
        tt = t + bt
        tok = tl.load(req_to_token + rp * r2t_stride + tl.minimum(tt, t1 - 1))
        sc = tl.zeros((R, BT), tl.float32)
        if FMT == 0:
            base = tl.multiple_of(tok * (HKV * D) + j * D, 16)
            for c in tl.static_range(0, D, QC):
                qc = tl.load(q_rows + c + dq[None, :], volatile=True)
                kc = tl.load(tl.max_contiguous(tl.multiple_of(k_ptr + base[:, None] + c + dq[None, :], [1, 16]), [1, 8]))
                sc = tl.dot(qc, tl.trans(kc), sc)
            v = tl.load(tl.max_contiguous(tl.multiple_of(v_ptr + base[:, None] + d[None, :], [1, 16]), [1, 8]))
        else:
            cbase = tl.multiple_of(tok * (HKV * D // 2) + j * (D // 2), 16)
            sbase = tok * (HKV * D // GS) + j * (D // GS)
            ch = tl.arange(0, GS // 2)
            for c in tl.static_range(0, D, GS):
                codes = tl.load(k_ptr + cbase[:, None] + c // 2 + ch[None, :])
                kq = _unpack_i4(codes, BT, GS)
                ks = tl.load(ks_ptr + sbase + c // GS).to(tl.float32)
                if FMT == 1:
                    qc = tl.load(q_rows + c + tl.arange(0, GS)[None, :], volatile=True)
                    kc = (kq.to(tl.float32) * ks[:, None]).to(tl.bfloat16)
                    sc = tl.dot(qc, tl.trans(kc), sc)
                else:
                    qc8 = tl.load(qi8_ptr + qsafe[:, None] * D + c + tl.arange(0, GS)[None, :], volatile=True)
                    part = tl.dot(qc8, tl.trans(kq))  # int8 x int8 -> int32 (WMMA iu8)
                    sc += part.to(tl.float32) * ks[None, :]
            if FMT == 2:
                sc = sc * qscale[:, None]
            vcodes = tl.load(v_ptr + cbase[:, None] + tl.arange(0, D // 2)[None, :])
            vq = _unpack_i4(vcodes, BT, D).to(tl.float32)
            vs = tl.load(vs_ptr + sbase[:, None] + tl.arange(0, D // GS)[None, :]).to(tl.float32)
            v = tl.reshape(tl.reshape(vq, (BT, D // GS, GS)) * vs[:, :, None], (BT, D)).to(tl.bfloat16)
        sc = sc * scale
        sc = tl.where((tt[None, :] < seq_r[:, None]) & (tt[None, :] < t1), sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    prow = qrow * NSPLIT + s
    tl.store(part_ptr + prow[:, None] * D + d[None, :], acc, mask=valid[:, None])
    tl.store(ml_ptr + prow * 2, m_i, mask=valid)
    tl.store(ml_ptr + prow * 2 + 1, l_i, mask=valid)


@triton.jit
def _combine_kernel(part_ptr, ml_ptr, o_ptr, D: tl.constexpr, NSPLIT: tl.constexpr):
    """Merge one (token row, q head)'s splits into o (bf16); a copy of the pinned combine kernel."""
    row = tl.program_id(0)
    d = tl.arange(0, D)
    mx = tl.full((8,), float("-inf"), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        mx = tl.maximum(mx, tl.load(ml_ptr + (row * NSPLIT + s) * 2, mask=s < NSPLIT, other=float("-inf")))
    mx = tl.max(mx, axis=0)
    num = tl.zeros((D,), tl.float32)
    den = tl.zeros((8,), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        smask = s < NSPLIT
        mm = tl.load(ml_ptr + (row * NSPLIT + s) * 2, mask=smask, other=float("-inf"))
        ll = tl.load(ml_ptr + (row * NSPLIT + s) * 2 + 1, mask=smask, other=0.0)
        w = tl.where(mm == float("-inf"), 0.0, tl.exp(mm - mx))
        acc = tl.load(part_ptr + (row * NSPLIT + s)[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
        num += tl.sum(w[:, None] * acc, axis=0)
        den += w * ll
    tl.store(o_ptr + row * D + d, (num / tl.sum(den, axis=0)).to(o_ptr.dtype.element_ty))


@triton.jit
def _half_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, part_ptr, ml_ptr,
                 req_to_token, r2t_stride, req_pool_indices, seq_lens, scale,
                 T: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 NSPLIT: tl.constexpr, BT: tl.constexpr, HG: tl.constexpr, QCH: tl.constexpr,
                 GPH: tl.constexpr):
    """Split-half INT4 layout: byte i of a (token, kv head) holds dim i (low nibble) and dim
    D/2 + i (high nibble), so each half unpacks to its own tile with no interleave. GPH scale
    groups per half: GPH = 1 folds the K scale into the score columns and the V scale into P
    (no per-element scaling); GPH = 4 (32-value groups) scales elements as they are unpacked."""
    GQ: tl.constexpr = HQ // HKV
    NG: tl.constexpr = (GQ + HG - 1) // HG
    HP: tl.constexpr = 4 if HG <= 4 else 8
    R: tl.constexpr = 8 * HP
    H2: tl.constexpr = D // 2
    GW: tl.constexpr = H2 // GPH
    NS: tl.constexpr = 2 * GPH
    u = tl.program_id(0)
    g = u % NG
    j = (u // NG) % HKV
    s = (u // (NG * HKV)) % NSPLIT
    b = u // (NG * HKV * NSPLIT)
    prefix = tl.load(seq_lens + b).to(tl.int32)
    rp = tl.load(req_pool_indices + b).to(tl.int64)
    total = prefix + T
    chunk = tl.cdiv(total, NSPLIT)
    t0 = s * chunk
    t1 = tl.minimum(total, t0 + chunk)
    r = tl.arange(0, R)
    m = r // HP
    h = g * HG + r % HP
    valid = (r % HP < HG) & (h < GQ) & (m < T)
    seq_r = prefix + m + 1
    qrow = (b * T + m) * HQ + j * GQ + h
    qsafe = tl.where(valid, qrow, b * T * HQ + j * GQ)
    q_rows = q_ptr + qsafe[:, None] * D
    bt = tl.arange(0, BT)
    dc = tl.arange(0, QCH)
    col = tl.arange(0, H2)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc_lo = tl.zeros((R, H2), tl.float32)
    acc_hi = tl.zeros((R, H2), tl.float32)
    for t in range(t0, t1, BT):
        tt = t + bt
        tok = tl.load(req_to_token + rp * r2t_stride + tl.minimum(tt, t1 - 1))
        cbase = tl.multiple_of(tok * (HKV * H2) + j * H2, 16)
        sbase = tok * (HKV * NS) + j * NS
        if GPH == 1:
            slo = tl.zeros((R, BT), tl.float32)
            shi = tl.zeros((R, BT), tl.float32)
            for c in tl.static_range(0, H2, QCH):
                codes = tl.load(tl.max_contiguous(tl.multiple_of(k_ptr + cbase[:, None] + c + dc[None, :], [1, 16]), [1, 16]))
                klo = ((codes & 15).to(tl.int8) - 8).to(tl.bfloat16)
                khi = ((codes >> 4).to(tl.int8) - 8).to(tl.bfloat16)
                qlo = tl.load(q_rows + c + dc[None, :], volatile=True)
                qhi = tl.load(q_rows + H2 + c + dc[None, :], volatile=True)
                slo = tl.dot(qlo, tl.trans(klo), slo)
                shi = tl.dot(qhi, tl.trans(khi), shi)
            ks_lo = tl.load(ks_ptr + sbase).to(tl.float32)
            ks_hi = tl.load(ks_ptr + sbase + 1).to(tl.float32)
            sc = slo * ks_lo[None, :] + shi * ks_hi[None, :]
        else:
            sc = tl.zeros((R, BT), tl.float32)
            for c in tl.static_range(0, H2, GW):
                codes = tl.load(tl.max_contiguous(tl.multiple_of(k_ptr + cbase[:, None] + c + tl.arange(0, GW)[None, :], [1, 16]), [1, 16]))
                ks_lo = tl.load(ks_ptr + sbase + c // GW).to(tl.float32)
                ks_hi = tl.load(ks_ptr + sbase + GPH + c // GW).to(tl.float32)
                klo = (((codes & 15).to(tl.int8) - 8).to(tl.float32) * ks_lo[:, None]).to(tl.bfloat16)
                khi = (((codes >> 4).to(tl.int8) - 8).to(tl.float32) * ks_hi[:, None]).to(tl.bfloat16)
                qlo = tl.load(q_rows + c + tl.arange(0, GW)[None, :], volatile=True)
                qhi = tl.load(q_rows + H2 + c + tl.arange(0, GW)[None, :], volatile=True)
                sc = tl.dot(qlo, tl.trans(klo), sc)
                sc = tl.dot(qhi, tl.trans(khi), sc)
        sc = sc * scale
        sc = tl.where((tt[None, :] < seq_r[:, None]) & (tt[None, :] < t1), sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vcodes = tl.load(tl.max_contiguous(tl.multiple_of(v_ptr + cbase[:, None] + col[None, :], [1, 16]), [1, 16]))
        vlo = (vcodes & 15).to(tl.int8) - 8
        vhi = (vcodes >> 4).to(tl.int8) - 8
        if GPH == 1:
            vs_lo = tl.load(vs_ptr + sbase).to(tl.float32)
            vs_hi = tl.load(vs_ptr + sbase + 1).to(tl.float32)
            acc_lo = acc_lo * alpha[:, None] + tl.dot((p * vs_lo[None, :]).to(tl.bfloat16), vlo.to(tl.bfloat16))
            acc_hi = acc_hi * alpha[:, None] + tl.dot((p * vs_hi[None, :]).to(tl.bfloat16), vhi.to(tl.bfloat16))
        else:
            vs_lo = tl.load(vs_ptr + sbase[:, None] + (col // GW)[None, :]).to(tl.float32)
            vs_hi = tl.load(vs_ptr + sbase[:, None] + GPH + (col // GW)[None, :]).to(tl.float32)
            pb = p.to(tl.bfloat16)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(pb, (vlo.to(tl.float32) * vs_lo).to(tl.bfloat16))
            acc_hi = acc_hi * alpha[:, None] + tl.dot(pb, (vhi.to(tl.float32) * vs_hi).to(tl.bfloat16))
        m_i = m_new
    prow = qrow * NSPLIT + s
    tl.store(part_ptr + prow[:, None] * D + col[None, :], acc_lo, mask=valid[:, None])
    tl.store(part_ptr + prow[:, None] * D + H2 + col[None, :], acc_hi, mask=valid[:, None])
    tl.store(ml_ptr + prow * 2, m_i, mask=valid)
    tl.store(ml_ptr + prow * 2 + 1, l_i, mask=valid)


@triton.jit
def _magic_unpack(h):
    """uint16 [rows, n] (4 nibbles each: byte b holds dims 2b low, 2b + 1 high) -> bf16 [rows, 4n] = 136 + q.

    BF16 bits 0x43 0x0c are exactly 128 + c for a nibble c, so v_perm_b32 builds two values per
    instruction next to a 0x43 constant: and, shift, and, mov, 4 x perm per 8 values. The +136 offset
    (128 + the code bias 8) is removed by the caller once per score row / output row."""
    # $2/$3 first hold the low/high nibble bytes and are overwritten last; $4 holds the 0x43 constant.
    o0, o1, o2, o3, _ = tl.inline_asm_elementwise(
        asm="v_and_b32 $2, 0x0f0f0f0f, $5\n"
            "v_lshrrev_b32 $3, 4, $5\n"
            "v_and_b32 $3, 0x0f0f0f0f, $3\n"
            "v_mov_b32 $4, 0x43434343\n"
            "v_perm_b32 $0, $4, $2, 0x04020400\n"
            "v_perm_b32 $1, $4, $3, 0x04020400\n"
            "v_perm_b32 $2, $4, $2, 0x04030401\n"
            "v_perm_b32 $3, $4, $3, 0x04030401",
        constraints="=&v,=&v,=&v,=&v,=&v,v",
        args=[h],
        dtype=(tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16),
        is_pure=False,
        pack=2,
    )
    a = tl.join(o0, o2)
    b2 = tl.join(o1, o3)
    return tl.reshape(tl.join(a, b2), [h.shape[0], h.shape[1] * 4])


@triton.jit
def _magic1_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, part_ptr, ml_ptr,
                   req_to_token, r2t_stride, req_pool_indices, seq_lens, scale,
                   T: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                   NSPLIT: tl.constexpr, BT: tl.constexpr, HG: tl.constexpr, KPT: tl.constexpr):
    """KPT = 1: one K scale per token (the chunk dots accumulate raw; one correction and scale per tile).
    Magic-unpack INT4 with K scaled per 64-value chunk (after each chunk's dot) and V scaled per
    token (folded into P), so P.V is one dot into one accumulator, as in the bf16 kernel."""
    GQ: tl.constexpr = HQ // HKV
    NG: tl.constexpr = (GQ + HG - 1) // HG
    HP: tl.constexpr = 4 if HG <= 4 else 8
    R: tl.constexpr = 8 * HP
    QC: tl.constexpr = 64
    NKS: tl.constexpr = 1 if KPT else D // QC
    u = tl.program_id(0)
    g = u % NG
    j = (u // NG) % HKV
    s = (u // (NG * HKV)) % NSPLIT
    b = u // (NG * HKV * NSPLIT)
    prefix = tl.load(seq_lens + b).to(tl.int32)
    rp = tl.load(req_pool_indices + b).to(tl.int64)
    total = prefix + T
    chunk = tl.cdiv(total, NSPLIT)
    t0 = s * chunk
    t1 = tl.minimum(total, t0 + chunk)
    r = tl.arange(0, R)
    m = r // HP
    h = g * HG + r % HP
    valid = (r % HP < HG) & (h < GQ) & (m < T)
    seq_r = prefix + m + 1
    qrow = (b * T + m) * HQ + j * GQ + h
    qsafe = tl.where(valid, qrow, b * T * HQ + j * GQ)
    q_rows = q_ptr + qsafe[:, None] * D
    dq = tl.arange(0, QC)
    d = tl.arange(0, D)
    w = tl.arange(0, QC // 4)
    wv = tl.arange(0, D // 4)
    bt = tl.arange(0, BT)
    sg0 = tl.sum(tl.load(q_rows + dq[None, :]).to(tl.float32), axis=1)
    sg1 = tl.sum(tl.load(q_rows + 64 + dq[None, :]).to(tl.float32), axis=1)
    sg2 = tl.sum(tl.load(q_rows + 128 + dq[None, :]).to(tl.float32), axis=1)
    sg3 = tl.sum(tl.load(q_rows + 192 + dq[None, :]).to(tl.float32), axis=1)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    corr = tl.zeros((R,), tl.float32)
    for t in range(t0, t1, BT):
        tt = t + bt
        tok = tl.load(req_to_token + rp * r2t_stride + tl.minimum(tt, t1 - 1))
        wbase = tl.multiple_of(tok * (HKV * D // 4) + j * (D // 4), 8)
        sc = tl.zeros((R, BT), tl.float32)
        for c in tl.static_range(0, D, QC):
            kw = tl.load(tl.max_contiguous(tl.multiple_of(k_ptr + wbase[:, None] + c // 4 + w[None, :], [1, 8]), [1, 8]))
            qc = tl.load(q_rows + c + dq[None, :], volatile=True)
            if KPT:
                sc = tl.dot(qc, tl.trans(_magic_unpack(kw)), sc)
            else:
                sig = sg0 if c == 0 else (sg1 if c == 64 else (sg2 if c == 128 else sg3))
                ks = tl.load(ks_ptr + tok * (HKV * NKS) + j * NKS + c // QC).to(tl.float32)
                sc += (tl.dot(qc, tl.trans(_magic_unpack(kw))) - 136.0 * sig[:, None]) * ks[None, :]
        if KPT:
            ks = tl.load(ks_ptr + tok * HKV + j).to(tl.float32)
            sc = (sc - 136.0 * (sg0 + sg1 + sg2 + sg3)[:, None]) * ks[None, :]
        sc = sc * scale
        sc = tl.where((tt[None, :] < seq_r[:, None]) & (tt[None, :] < t1), sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vs = tl.load(vs_ptr + tok * HKV + j).to(tl.float32)
        pv = (p * vs[None, :]).to(tl.bfloat16)
        vw = tl.load(tl.max_contiguous(tl.multiple_of(v_ptr + wbase[:, None] + wv[None, :], [1, 8]), [1, 8]))
        acc = acc * alpha[:, None] + tl.dot(pv, _magic_unpack(vw))
        corr = corr * alpha + tl.sum(pv.to(tl.float32), axis=1)
        m_i = m_new
    prow = qrow * NSPLIT + s
    tl.store(part_ptr + prow[:, None] * D + d[None, :], acc - 136.0 * corr[:, None], mask=valid[:, None])
    tl.store(ml_ptr + prow * 2, m_i, mask=valid)
    tl.store(ml_ptr + prow * 2 + 1, l_i, mask=valid)


def run_magic1(q, o, cache, r2t, rpi, seq, bt=64, warps=8, hg=6, kpt=False):
    """cache = (k_words u16, v_words u16, k_scales [n, HKV, D/64], v_scales [n, HKV, 1])."""
    rows = q.shape[0]
    part = torch.empty(rows * HQ * NSPLIT, D, device=DEV, dtype=torch.float32)
    ml = torch.empty(rows * HQ * NSPLIT, 2, device=DEV, dtype=torch.float32)
    k, v, ks, vs = cache
    ng = -(-GQ // hg)
    _magic1_kernel[((rows // T) * HKV * ng * NSPLIT,)](q, k, v, ks, vs, part, ml, r2t, r2t.stride(0), rpi, seq, SCALE,
                                                      T=T, HQ=HQ, HKV=HKV, D=D, NSPLIT=NSPLIT, BT=bt, HG=hg,
                                                      KPT=int(kpt), num_warps=warps, num_stages=1)
    _combine_kernel[(rows * HQ,)](part, ml, o, D=D, NSPLIT=NSPLIT, num_warps=4)
    return o


def magic_cache(x, gw):
    """Interleaved INT4 codes as uint16 words plus one e4m3 scale per gw values (dequantized fp32 too)."""
    codes, scales, deq = quantize_i4(x, gw)
    return codes.view(torch.uint16), scales, deq


def run_half(gph, q, o, cache, r2t, rpi, seq, bt=64, warps=8, qch=64, hg=6):
    """One verify call of the split-half INT4 kernel: cache = (k_codes, v_codes, k_scales, v_scales)."""
    rows = q.shape[0]
    part = torch.empty(rows * HQ * NSPLIT, D, device=DEV, dtype=torch.float32)
    ml = torch.empty(rows * HQ * NSPLIT, 2, device=DEV, dtype=torch.float32)
    k, v, ks, vs = cache
    ng = -(-GQ // hg)
    _half_kernel[((rows // T) * HKV * ng * NSPLIT,)](q, k, v, ks, vs, part, ml, r2t, r2t.stride(0), rpi, seq, SCALE,
                                                    T=T, HQ=HQ, HKV=HKV, D=D, NSPLIT=NSPLIT, BT=bt, HG=hg,
                                                    QCH=qch if gph == 1 else (D // 2) // gph, GPH=gph,
                                                    num_warps=warps, num_stages=1)
    _combine_kernel[(rows * HQ,)](part, ml, o, D=D, NSPLIT=NSPLIT, num_warps=4)
    return o


def quantize_i4_half(x, gph):
    """[n, HKV, D] -> split-half layout: codes uint8 [n, HKV, D/2] (dim i low nibble, dim D/2 + i high),
    scales e4m3 [n, HKV, 2 * gph] (low-half groups, then high-half groups), dequantized fp32."""
    n, h2 = x.shape[0], D // 2
    g = x.float().view(n, HKV, 2 * gph, h2 // gph)
    scale = (g.abs().amax(-1, keepdim=True) / 7.0).to(FP8).float().clamp_min(2.0**-9)
    qv = torch.clamp(torch.round(g / scale), -7, 7)
    deq = (qv * scale).view(n, HKV, D)
    u = (qv + 8).to(torch.uint8).view(n, HKV, D)
    codes = (u[..., :h2] | (u[..., h2:] << 4)).contiguous()
    return codes, scale.squeeze(-1).to(FP8).contiguous(), deq


def unpack_half_reference(codes):
    """Host-side mirror of the split-half unpack: uint8 [..., D/2] -> int8 [..., D]."""
    return torch.cat([(codes & 15).to(torch.int8) - 8, (codes >> 4).to(torch.int8) - 8], dim=-1)


def run_variant(fmt, q, o, cache, r2t, rpi, seq, bt=32, warps=4, hg=6, qc=32):
    """One verify call of this tool's kernel: cache = (k, v) bf16, or (k_codes, v_codes, k_scales, v_scales)."""
    rows = q.shape[0]
    part = torch.empty(rows * HQ * NSPLIT, D, device=DEV, dtype=torch.float32)
    ml = torch.empty(rows * HQ * NSPLIT, 2, device=DEV, dtype=torch.float32)
    qi8, qs = q, q.new_empty(1, dtype=torch.float32)
    if fmt == FMT_I4_I8DOT:
        qf = q.float()
        qs = (qf.abs().amax(-1).clamp_min(1e-12) / 127.0).reshape(-1).contiguous()
        qi8 = torch.round(qf / qs.view(rows, HQ, 1)).clamp(-127, 127).to(torch.int8).contiguous()
    if fmt == FMT_BF16:
        k, v = cache
        ks = vs = q.new_empty(1, dtype=torch.float32)
    else:
        k, v, ks, vs = cache
    ng = -(-GQ // hg)
    B = rows // T
    _split_kernel[(B * HKV * ng * NSPLIT,)](q, qi8, qs, k, v, ks, vs, part, ml, r2t, r2t.stride(0), rpi, seq, SCALE,
                                           T=T, HQ=HQ, HKV=HKV, D=D, NSPLIT=NSPLIT, BT=bt, HG=hg, QC=qc, FMT=fmt,
                                           num_warps=warps, num_stages=1)
    _combine_kernel[(rows * HQ,)](part, ml, o, D=D, NSPLIT=NSPLIT, num_warps=4)
    return o


def unpack_i4_reference(codes):
    """Host-side mirror of _unpack_i4 (for the CPU test): uint8 [..., D/2] -> int8 [..., D]."""
    lo = (codes & 15).to(torch.int8) - 8
    hi = (codes >> 4).to(torch.int8) - 8
    return torch.stack([lo, hi], dim=-1).reshape(*codes.shape[:-1], codes.shape[-1] * 2)


def quantize_i4(x, group=G):
    """[n, HKV, D] bf16 -> (codes uint8 [n, HKV, D/2], scales e4m3 [n, HKV, D/group], dequantized fp32)."""
    g = x.float().view(x.shape[0], HKV, D // group, group)
    scale = (g.abs().amax(-1, keepdim=True) / 7.0).to(FP8).float().clamp_min(2.0**-9)
    qv = torch.clamp(torch.round(g / scale), -7, 7)
    deq = (qv * scale).view_as(x.float())
    u = (qv + 8).to(torch.uint8).view(x.shape[0], HKV, D)
    codes = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    return codes, scale.squeeze(-1).to(FP8).contiguous(), deq


def cache(n, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(n, HKV, D, generator=gen, device=DEV).to(torch.bfloat16)
    v = (torch.randn(n, HKV, D, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    return k, v


def inputs(prefix, batch=1, seed=1):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(batch * T, HQ, D, device=DEV, generator=gen) * 2).to(torch.bfloat16)
    r2t = torch.arange(batch * (prefix + T), dtype=torch.int32, device=DEV).view(batch, -1)
    rpi = torch.arange(batch, dtype=torch.int64, device=DEV)
    seq = torch.full((batch,), prefix, dtype=torch.int64, device=DEV)
    return q, r2t, rpi, seq


def reference(q, k, v, visible):
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    kf, vf = k.float(), v.float()
    for row in range(q.shape[0]):
        n = int(visible[row])
        for j in range(HKV):
            qs = q[row, j * GQ:(j + 1) * GQ].float()
            out[row, j * GQ:(j + 1) * GQ] = torch.softmax((qs @ kf[:n, j].T) * SCALE, -1) @ vf[:n, j]
    return out


def rel_err(out, ref):
    return (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


def check(prefix=8192):
    k, v = cache(prefix + T, 1)
    q, r2t, rpi, seq = inputs(prefix)
    visible = [prefix + t + 1 for t in range(T)]
    ref_bf16 = reference(q, k, v, visible)
    row = {"phase": "check", "prefix": prefix}
    row["ctl_bf16_kernel_err"] = rel_err(run_variant(FMT_BF16, q, torch.empty_like(q), (k, v), r2t, rpi, seq), ref_bf16)
    kc, ksc, kd = quantize_i4(k)
    vc, vsc, vd = quantize_i4(v)
    ref_i4 = reference(q, kd, vd, visible)
    for name, fmt in (("i4_bf16", FMT_I4), ("i4_i8dot", FMT_I4_I8DOT)):
        try:
            out = run_variant(fmt, q, torch.empty_like(q), (kc, vc, ksc, vsc), r2t, rpi, seq)
            row[f"{name}_kernel_err"] = rel_err(out, ref_i4)
            row[f"{name}_format_err"] = rel_err(out, ref_bf16)
        except Exception as exc:  # noqa: BLE001
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
    for gph in (1, 4):
        name = f"half_g{(D // 2) // gph}"
        kc, ksc, kd = quantize_i4_half(k, gph)
        vc, vsc, vd = quantize_i4_half(v, gph)
        try:
            out = run_half(gph, q, torch.empty_like(q), (kc, vc, ksc, vsc), r2t, rpi, seq)
            row[f"{name}_kernel_err"] = rel_err(out, reference(q, kd, vd, visible))
            row[f"{name}_format_err"] = rel_err(out, ref_bf16)
        except Exception as exc:  # noqa: BLE001
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
    for kpt in (False, True):
        name = "magic1_kpt" if kpt else "magic1_kg64"
        kc, ksc, kd = magic_cache(k, D if kpt else 64)
        vc, vsc, vd = magic_cache(v, D)
        try:
            out = run_magic1(q, torch.empty_like(q), (kc, vc, ksc, vsc), r2t, rpi, seq, kpt=kpt)
            row[f"{name}_kernel_err"] = rel_err(out, reference(q, kd, vd, visible))
            row[f"{name}_format_err"] = rel_err(out, ref_bf16)
        except Exception as exc:  # noqa: BLE001
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
    return row


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def graph_us(step, reps):
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    ms = timed(graph.replay, reps)
    del graph
    return 1000 * ms / LAYERS


MAGIC_TILES = ((64, 8), (32, 8), (64, 4))
CTL_TILES = ((64, 8, 64), (64, 4, 64), (32, 4, 32))  # production b1 bf16 tile first, its b2 tile last
HALF_TILES = ((64, 8, 64), (64, 4, 64), (32, 4, 32), (32, 8, 64), (64, 8, 32))


def time_row(prefix, batch, reps, sweep):
    va = _production()
    q, r2t, rpi, seq = inputs(prefix, batch)
    n = batch * (prefix + T)
    o = torch.empty_like(q)
    row = {"phase": "verify", "prefix": prefix, "batch": batch}
    bf16 = [cache(n, 100 + layer) for layer in range(LAYERS)]

    def prod(caches, tile):
        return lambda: [va.verify_attention_gfx1151(q, o, kb, vb, r2t, rpi, seq, T, SCALE,
                                                    bt=tile[0], num_warps=tile[1], qc=tile[2]) for kb, vb in caches]

    def best_of(name, make, tiles):
        best = None
        for tile in tiles:
            try:
                us = graph_us(make(tile), reps)
            except Exception as exc:  # noqa: BLE001
                row[f"{name}_{'x'.join(map(str, tile))}_us"] = f"failed: {type(exc).__name__}"
                continue
            row[f"{name}_{'x'.join(map(str, tile))}_us"] = round(us, 2)
            best = us if best is None else min(best, us)
        row[f"{name}_best_us"] = None if best is None else round(best, 2)
        return best

    b1 = va.GFX1201_B1_TILE if batch == 1 else (32, 4, 32)
    row["prod_bf16_us"] = round(graph_us(prod(bf16, b1), reps), 2)
    best_of("ctl_bf16", lambda tile: lambda: [run_variant(FMT_BF16, q, o, c, r2t, rpi, seq, bt=tile[0], warps=tile[1],
                                                          qc=tile[2]) for c in bf16], CTL_TILES if sweep else CTL_TILES[:1])
    fp8 = [(k.to(FP8), v.to(FP8)) for k, v in bf16]
    fp8_tile = va.GFX1201_FP8_B1_TILE if batch == 1 else (32, 4, 32)
    row["prod_fp8_us"] = round(graph_us(prod(fp8, fp8_tile), reps), 2)
    del fp8
    for gph in (1, 4):
        name = f"half_g{(D // 2) // gph}"
        half = []
        for k, v in bf16:
            kc, ksc, _ = quantize_i4_half(k, gph)
            vc, vsc, _ = quantize_i4_half(v, gph)
            half.append((kc, vc, ksc, vsc))
        best_of(name, lambda tile, gph=gph, half=half: lambda: [run_half(gph, q, o, c, r2t, rpi, seq, bt=tile[0],
                                                                            warps=tile[1], qch=tile[2]) for c in half],
                HALF_TILES if sweep else HALF_TILES[:1])
        del half
    for kpt in (False, True):
        name = "magic1_kpt" if kpt else "magic1_kg64"
        mc = []
        for k, v in bf16:
            kc, ksc, _ = magic_cache(k, D if kpt else 64)
            vc, vsc, _ = magic_cache(v, D)
            mc.append((kc, vc, ksc, vsc))
        best_of(name, lambda tile, kpt=kpt, mc=mc: lambda: [run_magic1(q, o, c, r2t, rpi, seq, bt=tile[0], warps=tile[1],
                                                                       kpt=kpt) for c in mc],
                MAGIC_TILES if sweep else MAGIC_TILES[:1])
        del mc
    del bf16
    torch.cuda.empty_cache()
    for name in ("ctl_bf16", "half_g128", "half_g32", "magic1_kg64", "magic1_kpt"):
        best = row.get(f"{name}_best_us")
        if best:
            row[f"{name}_vs_prod_fp8"] = round(best / row["prod_fp8_us"], 3)
            row[f"{name}_vs_prod_bf16"] = round(best / row["prod_bf16_us"], 3)
    row["prod_fp8_vs_prod_bf16"] = round(row["prod_fp8_us"] / row["prod_bf16_us"], 3)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="8k and 32k only, no tile sweep")
    parser.add_argument("--shapes", help="comma list of prefix[xbatch], e.g. 8192,32768,120000,32768x2")
    args = parser.parse_args(argv)
    print(json.dumps({"phase": "device", "name": torch.cuda.get_device_name(0),
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"),
                      "triton": triton.__version__, "torch": torch.__version__}), flush=True)
    print(json.dumps(check()), flush=True)
    if args.shapes:
        shapes = [(int(x.split("x")[0]), int(x.split("x")[1]) if "x" in x else 1) for x in args.shapes.split(",")]
    elif args.quick:
        shapes = [(8192, 1), (32768, 1)]
    else:
        shapes = [(8192, 1), (32768, 1), (65536, 1), (120000, 1), (32768, 2)]
    for prefix, batch in shapes:
        print(json.dumps(time_row(prefix, batch, reps=20, sweep=not args.quick)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
