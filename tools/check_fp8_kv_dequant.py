#!/usr/bin/env python3
"""GPU check for sglang-main patches 0026/0027 (SGLANG_GFX1201_FP8_KV_DEQUANT) on gfx1201.

Run inside the SGLang image built with the series, on an otherwise idle card:

    python3 check_fp8_kv_dequant.py [--quick]

Qwen3.8-27B attention shapes at TP2 (12 q heads, 2 kv heads, head dim 256), random data,
the serving switches (SGLANG_GFX1201_VERIFY_ATTENTION, _B1_TILE). Decode modes
(SGLANG_GFX1201_FP8_KV_DECODE): ``to`` = Triton's fp8 -> bf16 cast (0026), ``hw`` =
v_cvt_f32_fp8 per value, ``pairs`` = v_cvt_f32_fp8 + v_perm_b32 on 16-bit pairs (verify only).

1. Decode: every float8_e4m3fn code through ``hw`` and ``pairs`` is bitwise equal to
   torch's cast (the two NaN codes stay NaN).
2. Kernels: target verify (8 draft tokens, prefixes 1k-123k), extend (8k cached prefix +
   2048-token chunk) and grouped decode (8k, 47k). Each fp8 mode's output must be bitwise
   equal to the ``to`` output (same bf16 operands), and the fp8 kernel error against an fp32
   reference built from the same fp8 values must stay within 1.5x (+1e-3) of the kernel's
   error on a bf16 cache. Stock fp8 (switch off: q and p cast to fp8) is reported only.
3. Verify time as patch 0021 measured it: 16 layers, each on its own KV cache (DRAM-fed),
   captured in one CUDA graph, median replay / 16, each cache type on its own default tile.
   From 8k up the ``pairs`` default must take at most 1.10x the bf16 time. A tile sweep at
   47k / 88k and a two-request batch at 47k are reported only.
4. Extend time (serving tile switches), 16 layers on their own prefix caches, one
   2048-token chunk each, at 8k / 47k / 88k prefixes, plus a tile sweep at 47k: reported only.

Prints one JSON line prefixed RESULT=; exit status 0 only if every check passes.
"""

import argparse
import json
import os
import sys
import time

# Read once at import (lru_cache); set before importing the kernels.
os.environ.setdefault("SGLANG_GFX1201_VERIFY_ATTENTION", "1")
os.environ.setdefault("SGLANG_GFX1201_VERIFY_ATTENTION_B1_TILE", "1")
os.environ.setdefault("SGLANG_GFX1201_FP8_KV_DEQUANT", "1")
# the serving profile's extend tile (0023 over 0016)
os.environ.setdefault("SGLANG_GFX1201_EXTEND_TILE32", "1")
os.environ.setdefault("SGLANG_GFX1201_EXTEND_WARPS8", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from sglang.kernels.ops.attention import gfx1201_fp8_kv  # noqa: E402
from sglang.kernels.ops.attention import verify_attention_gfx1151 as va  # noqa: E402
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd  # noqa: E402
from sglang.kernels.ops.attention import extend_attention as ea  # noqa: E402
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd  # noqa: E402

HQ, HKV, D = 12, 2, 256
GQ = HQ // HKV
T = 8
FP8 = torch.float8_e4m3fn
DEV = "cuda"
SCALE = D**-0.5
LAYERS = 16
KERNEL_RATIO, KERNEL_SLACK, TIME_RATIO = 1.5, 1e-3, 1.10
# fp8 pairs verify tiles (BT, warps, QC) beside the default, and extend (BLOCK_M, BLOCK_N, warps)
SWEEP_TILES = ((64, 8, 64), (64, 4, 32), (64, 2, 64), (128, 4, 64), (32, 4, 64))
EXTEND_TILES = ((64, 64, 4), (64, 64, 8), (32, 64, 4), (64, 32, 4), (32, 32, 8))


def mode(name):
    """Select the decode mode for subsequent launches (read per launch by decode())."""
    os.environ["SGLANG_GFX1201_FP8_KV_DECODE"] = name


def switch(on):
    """Turn the dequant switch on or off for subsequent kernel launches."""
    os.environ["SGLANG_GFX1201_FP8_KV_DEQUANT"] = "1" if on else "0"
    gfx1201_fp8_kv.enabled.cache_clear()


@triton.jit
def _decode_codes(src, out_hw, out_pairs):
    r = tl.arange(0, 16)[:, None]
    c = tl.arange(0, 16)[None, :]
    x = tl.load(src + r * 16 + c)
    tl.store(out_hw + r * 16 + c, gfx1201_fp8_kv.e4m3fn_to_bf16(x))
    h = tl.load(src.to(tl.pointer_type(tl.uint16)) + r * 8 + tl.arange(0, 8)[None, :])
    tl.store(out_pairs + r * 16 + c, gfx1201_fp8_kv.e4m3fn_pairs_to_bf16(h))


def check_codes():
    codes = torch.arange(256, dtype=torch.uint8, device=DEV).view(FP8)
    ref = codes.to(torch.bfloat16)
    row = {}
    for name in ("hw", "pairs"):
        row[name] = torch.empty(256, dtype=torch.bfloat16, device=DEV)
    _decode_codes[(1,)](codes, row["hw"], row["pairs"])
    nan = torch.isnan(ref)
    out = {"nan_codes": int(nan.sum())}
    for name in ("hw", "pairs"):
        got = row[name]
        same = torch.equal(got[~nan].view(torch.int16), ref[~nan].view(torch.int16))
        out[f"{name}_bitwise"] = bool(same and torch.isnan(got[nan]).all())
        out[f"{name}_mismatches"] = int((got[~nan].view(torch.int16) != ref[~nan].view(torch.int16)).sum())
    out["pass"] = out["hw_bitwise"] and out["pairs_bitwise"]
    return out


def cache(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(n, HKV, D, generator=g, device=DEV).to(torch.bfloat16)
    v = (torch.randn(n, HKV, D, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    return k, v


def reference(q, k, v, visible):
    """fp32 attention: q [R, HQ, D] (row r sees the first visible[r] tokens of k/v)."""
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    kf, vf = k.float(), v.float()
    for r in range(q.shape[0]):
        n = int(visible[r])
        for j in range(HKV):
            qs = q[r, j * GQ : (j + 1) * GQ].float()
            s = (qs @ kf[:n, j].T) * SCALE
            out[r, j * GQ : (j + 1) * GQ] = torch.softmax(s, dim=-1) @ vf[:n, j]
    return out


def err(out, ref):
    return (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


def verify_inputs(prefix, seed=1, batch=1):
    """q for ``batch`` requests of T draft tokens; request b owns cache slots
    [b * (prefix + T), (b + 1) * (prefix + T))."""
    q = (torch.randn(batch * T, HQ, D, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)) * 2).to(
        torch.bfloat16
    )
    r2t = torch.arange(batch * (prefix + T), dtype=torch.int32, device=DEV).view(batch, -1)
    rpi = torch.arange(batch, dtype=torch.int64, device=DEV)
    seq = torch.full((batch,), prefix, dtype=torch.int64, device=DEV)
    return q, r2t, rpi, seq


def check_verify(prefix):
    k, v = cache(prefix + T, 1)
    k8, v8 = k.to(FP8), v.to(FP8)
    q, r2t, rpi, seq = verify_inputs(prefix)
    visible = [prefix + t + 1 for t in range(T)]
    row = {"prefix": prefix}
    outs = {}
    for name, kb, vb, m in (("bf16", k, v, "to"), ("fp8_to", k8, v8, "to"), ("fp8_hw", k8, v8, "hw"),
                            ("fp8_pairs", k8, v8, "pairs")):
        mode(m)
        o = torch.empty_like(q)
        if not va.covered(q, o, kb, vb, T, HQ, HKV, D):
            raise SystemExit(f"verify kernel does not cover the {name} cache")
        # every fp8 mode on the fp8 tile, so the outputs compare bit for bit
        tile = va.GFX1201_FP8_B1_TILE if kb.dtype == FP8 else va.GFX1201_B1_TILE
        va.verify_attention_gfx1151(q, o, kb, vb, r2t, rpi, seq, T, SCALE,
                                    bt=tile[0], num_warps=tile[1], qc=tile[2])
        outs[name] = o
    mode("pairs")
    row["bf16_kernel_err"] = err(outs["bf16"], reference(q, k, v, visible))
    row["fp8_kernel_err"] = err(outs["fp8_to"], reference(q, k8, v8, visible))
    row["fp8_vs_bf16_reference_err"] = err(outs["fp8_to"], reference(q, k, v, visible))
    row["hw_bitwise_to"] = torch.equal(outs["fp8_hw"], outs["fp8_to"])
    row["pairs_bitwise_to"] = torch.equal(outs["fp8_pairs"], outs["fp8_to"])
    row["pass"] = (row["fp8_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
                   and row["hw_bitwise_to"] and row["pairs_bitwise_to"])
    return row


def check_extend(prefix=8192, chunk=2048):
    kp, vp = cache(prefix, 2)
    ke, ve = cache(chunk, 3)
    q = torch.randn(chunk, HQ, D, device=DEV).to(torch.bfloat16)
    qo = torch.tensor([0, chunk], dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(prefix, dtype=torch.int64, device=DEV)
    visible = [prefix + i + 1 for i in range(chunk)]
    row = {"prefix": prefix, "chunk": chunk}
    outs = {}
    kp8, vp8 = kp.to(FP8), vp.to(FP8)
    for name, kb, vb, on, m in (("bf16", kp, vp, True, "to"), ("fp8_to", kp8, vp8, True, "to"),
                                ("fp8_hw", kp8, vp8, True, "hw"), ("fp8_stock", kp8, vp8, False, "to")):
        switch(on)
        mode(m)
        o = torch.empty_like(q)
        try:
            run = lambda: extend_attention_fwd(q, ke, ve, o, kb, vb, qo, kv_indptr, kv_indices, None, True,  # noqa: E731
                                               None, chunk, 1.0, 1.0, sm_scale=SCALE)
            run()
        except Exception as exc:  # the stock fp8 dot may not compile on every Triton
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}"
            continue
        outs[name] = o
        ref = reference(q, torch.cat([kb.to(torch.bfloat16), ke]), torch.cat([vb.to(torch.bfloat16), ve]), visible)
        row[f"{name}_kernel_err"] = err(o, ref)
        row[f"{name}_ms"] = timed(run, 10)
    switch(True)
    mode("pairs")
    row["hw_bitwise_to"] = torch.equal(outs["fp8_hw"], outs["fp8_to"])
    row["pass"] = (row["fp8_to_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
                   and row["hw_bitwise_to"])
    return row


def check_decode(prefix, splits=16):
    k, v = cache(prefix, 4)
    q = torch.randn(1, HQ, D, device=DEV).to(torch.bfloat16)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(prefix, dtype=torch.int64, device=DEV)
    row = {"prefix": prefix}
    outs = {}
    k8, v8 = k.to(FP8), v.to(FP8)
    for name, kb, vb, on, m in (("bf16", k, v, True, "to"), ("fp8_to", k8, v8, True, "to"),
                                ("fp8_hw", k8, v8, True, "hw"), ("fp8_stock", k8, v8, False, "to")):
        switch(on)
        mode(m)
        o = torch.empty_like(q)
        logits = torch.empty(1, HQ, splits, D, dtype=torch.float32, device=DEV)
        lse = torch.empty(1, HQ, splits, dtype=torch.float32, device=DEV)
        nsplit = torch.full((1,), splits, dtype=torch.int32, device=DEV)
        try:
            run = lambda: decode_attention_fwd(q, kb, vb, o, kv_indptr, kv_indices, logits, lse, nsplit,  # noqa: E731
                                               splits, SCALE, 1.0, 1.0, enable_lean=False)
            run()
        except Exception as exc:
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}"
            continue
        outs[name] = o
        row[f"{name}_kernel_err"] = err(o, reference(q, kb.to(torch.bfloat16), vb.to(torch.bfloat16), [prefix]))
        row[f"{name}_ms"] = timed(run, 20)
    switch(True)
    mode("pairs")
    row["hw_bitwise_to"] = torch.equal(outs["fp8_hw"], outs["fp8_to"])
    row["pass"] = (row["fp8_to_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
                   and row["hw_bitwise_to"])
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


def graph_us_per_layer(q, caches, r2t, rpi, seq, tile, reps):
    """Median microseconds per layer: LAYERS verify calls on separate caches in one graph."""
    o = torch.empty_like(q)
    kw = {} if tile is None else {"bt": tile[0], "num_warps": tile[1], "qc": tile[2]}

    def step():
        for kb, vb in caches:
            va.verify_attention_gfx1151(q, o, kb, vb, r2t, rpi, seq, T, SCALE, **kw)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    ms = timed(graph.replay, reps)
    del graph
    return 1000 * ms / len(caches)


def time_verify(prefix, sweep, reps, batch=1):
    q, r2t, rpi, seq = verify_inputs(prefix, batch=batch)
    bf16 = [cache(batch * (prefix + T), 100 + layer) for layer in range(LAYERS)]
    fp8 = [(k.to(FP8), v.to(FP8)) for k, v in bf16]
    row = {"prefix": prefix, "batch": batch}
    mode("to")
    row["bf16_us"] = graph_us_per_layer(q, bf16, r2t, rpi, seq, None, reps)
    del bf16
    for m in ("to", "hw", "pairs"):
        mode(m)
        row[f"fp8_{m}_us"] = graph_us_per_layer(q, fp8, r2t, rpi, seq, None, reps)
    if sweep:
        mode("pairs")
        for tile in SWEEP_TILES if batch == 1 else ((64, 4, 64), (64, 8, 64)):
            try:
                row[f"fp8_pairs_{'x'.join(map(str, tile))}_us"] = graph_us_per_layer(q, fp8, r2t, rpi, seq, tile, reps)
            except Exception as exc:
                row[f"fp8_pairs_{'x'.join(map(str, tile))}_us"] = f"failed: {type(exc).__name__}"
    mode("pairs")
    del fp8
    torch.cuda.empty_cache()
    row["pairs_ratio"] = row["fp8_pairs_us"] / row["bf16_us"]
    row["hw_ratio"] = row["fp8_hw_us"] / row["bf16_us"]
    # GB/s of K+V each variant reached (fp8: one byte per value, bf16: two); MB/us x 1000
    mb = 2 * batch * (prefix + T) * HKV * D / 1e6
    row["bf16_gbs"] = round(2 * mb / row["bf16_us"] * 1e3, 1)
    row["fp8_pairs_gbs"] = round(mb / row["fp8_pairs_us"] * 1e3, 1)
    row["fp8_hw_gbs"] = round(mb / row["fp8_hw_us"] * 1e3, 1)
    # below 8k the call is launch-bound and the ratio is noise
    row["pass"] = prefix < 8192 or row["pairs_ratio"] <= TIME_RATIO
    return row


def extend_ms_per_layer(q, ke, ve, caches, qo, kv_indptr, kv_indices, chunk, reps):
    """Median ms per layer: one chunk's extend over LAYERS separate prefix caches (eager)."""
    o = torch.empty_like(q)

    def step():
        for kb, vb in caches:
            extend_attention_fwd(q, ke, ve, o, kb, vb, qo, kv_indptr, kv_indices, None, True, None,
                                 chunk, 1.0, 1.0, sm_scale=SCALE)

    return timed(step, reps) / len(caches)


def time_extend(prefix, sweep, reps, chunk=2048):
    ke, ve = cache(chunk, 3)
    q = torch.randn(chunk, HQ, D, device=DEV).to(torch.bfloat16)
    qo = torch.tensor([0, chunk], dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(prefix, dtype=torch.int64, device=DEV)
    args = (q, ke, ve)
    rest = (qo, kv_indptr, kv_indices, chunk, reps)
    bf16 = [cache(prefix, 200 + layer) for layer in range(LAYERS)]
    fp8 = [(k.to(FP8), v.to(FP8)) for k, v in bf16]
    row = {"prefix": prefix, "chunk": chunk}
    mode("to")
    row["bf16_ms"] = extend_ms_per_layer(*args, bf16, *rest)
    row["fp8_to_ms"] = extend_ms_per_layer(*args, fp8, *rest)
    mode("hw")
    row["fp8_hw_ms"] = extend_ms_per_layer(*args, fp8, *rest)
    if sweep:
        original = ea._get_block_sizes_for_extend_attention
        os.environ["SGLANG_GFX1201_EXTEND_TILE32"] = "0"
        os.environ["SGLANG_GFX1201_EXTEND_WARPS8"] = "0"
        try:
            for bm, bn, warps in EXTEND_TILES:
                ea._get_block_sizes_for_extend_attention = (
                    lambda lq, lv, bm=bm, bn=bn, w=warps: (*original(lq, lv)[:3], bm, bn, w))
                name = f"{bm}x{bn}x{warps}"
                for label, caches, m in (("bf16", bf16, "to"), ("fp8_hw", fp8, "hw")):
                    mode(m)
                    try:
                        row[f"{label}_{name}_ms"] = extend_ms_per_layer(*args, caches, *rest)
                    except Exception as exc:
                        row[f"{label}_{name}_ms"] = f"failed: {type(exc).__name__}"
        finally:
            ea._get_block_sizes_for_extend_attention = original
            os.environ["SGLANG_GFX1201_EXTEND_TILE32"] = "1"
            os.environ["SGLANG_GFX1201_EXTEND_WARPS8"] = "1"
    mode("pairs")
    del bf16, fp8
    torch.cuda.empty_cache()
    row["hw_ratio"] = row["fp8_hw_ms"] / row["bf16_ms"]
    row["pass"] = True  # reported only
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="1k and 8k prefixes only, no tile sweep")
    args = parser.parse_args()
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1201" or not gfx1201_fp8_kv.enabled():
        raise SystemExit(f"needs gfx1201 with SGLANG_GFX1201_FP8_KV_DEQUANT=1 (device {arch})")
    started = time.monotonic()
    prefixes = (1024, 8192) if args.quick else (1024, 8192, 23552, 47104, 88064, 122880)
    result = {"schema": 2, "device": arch, "torch": torch.__version__,
              "shapes": {"hq": HQ, "hkv": HKV, "d": D, "draft_tokens": T, "layers": LAYERS}}
    result["codes"] = [check_codes()]
    print("CODES", json.dumps(result["codes"][0]), flush=True)
    result["verify"] = []
    for p in prefixes:
        result["verify"].append(check_verify(p))
        print("VERIFY", json.dumps(result["verify"][-1]), flush=True)
    result["extend"] = [check_extend()]
    print("EXTEND", json.dumps(result["extend"][0]), flush=True)
    result["decode"] = []
    for p in (8192,) if args.quick else (8192, 47104):
        result["decode"].append(check_decode(p))
        print("DECODE", json.dumps(result["decode"][-1]), flush=True)
    result["timing"] = []
    for p in prefixes:
        result["timing"].append(time_verify(p, sweep=not args.quick and p in (47104, 88064), reps=10 if args.quick else 20))
        print("TIMING", json.dumps(result["timing"][-1]), flush=True)
    if not args.quick:
        result["timing"].append(time_verify(47104, sweep=True, reps=20, batch=2))
        print("TIMING", json.dumps(result["timing"][-1]), flush=True)
    result["extend_timing"] = []
    for p in (8192,) if args.quick else (8192, 47104, 88064):
        result["extend_timing"].append(time_extend(p, sweep=not args.quick and p == 47104, reps=3 if args.quick else 5))
        print("EXTEND_TIMING", json.dumps(result["extend_timing"][-1]), flush=True)
    result["seconds"] = round(time.monotonic() - started, 1)
    parts = ("codes", "verify", "extend", "decode", "timing", "extend_timing")
    result["pass"] = all(r["pass"] for part in parts for r in result[part])
    print("RESULT=" + json.dumps(result), flush=True)
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
