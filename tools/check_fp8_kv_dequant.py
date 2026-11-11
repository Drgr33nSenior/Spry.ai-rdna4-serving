#!/usr/bin/env python3
"""GPU check for sglang-main patch 0026 (SGLANG_GFX1201_FP8_KV_DEQUANT) on one gfx1201 card.

Run inside the SGLang image built with the series, on an otherwise idle card:

    python3 check_fp8_kv_dequant.py [--quick]

Qwen3.8-27B attention shapes at TP2 (12 q heads, 2 KV heads, head dim 256), random data.
For each patched kernel it compares the output on an fp8 (e4m3fn) cache with an fp32
reference computed from the *same* fp8 values, so kernel error is separated from the
storage rounding:

- target verify (``verify_attention_gfx1151``), 8 draft tokens, prefixes 1k-88k;
- extend (``extend_attention_fwd``), an 8k cached prefix plus a 2048-token chunk;
- grouped decode (``decode_attention_fwd``), 8k and 47k prefixes.

Each kernel's error on the fp8 cache must stay within 1.5x (+1e-3) of its own error on a
bf16 cache, i.e. reading fp8 as bf16 adds no kernel error beyond bf16 rounding. The
stock fp8 path (switch off: q and p cast to fp8) is measured for comparison only. Verify
time on the fp8 cache must be at most 1.10x the bf16 time from 8k prefixes up.
Prints one JSON line prefixed RESULT=; exit status 0 only if every check passes.
"""

import argparse
import json
import os
import sys
import time

# Read once at import (lru_cache); set before importing the kernels.
os.environ.setdefault("SGLANG_GFX1201_VERIFY_ATTENTION", "1")
os.environ.setdefault("SGLANG_GFX1201_FP8_KV_DEQUANT", "1")

import torch  # noqa: E402

from sglang.kernels.ops.attention import gfx1201_fp8_kv  # noqa: E402
from sglang.kernels.ops.attention import verify_attention_gfx1151 as va  # noqa: E402
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd  # noqa: E402
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd  # noqa: E402

HQ, HKV, D = 12, 2, 256
GQ = HQ // HKV
T = 8
FP8 = torch.float8_e4m3fn
DEV = "cuda"
SCALE = D**-0.5
KERNEL_RATIO, KERNEL_SLACK, TIME_RATIO = 1.5, 1e-3, 1.10


def switch(on):
    """Turn the dequant switch on or off for subsequent kernel launches."""
    os.environ["SGLANG_GFX1201_FP8_KV_DEQUANT"] = "1" if on else "0"
    gfx1201_fp8_kv.enabled.cache_clear()


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


def check_verify(prefix, reps):
    k, v = cache(prefix + T, 1)
    k8, v8 = k.to(FP8), v.to(FP8)
    q = (torch.randn(T, HQ, D, device=DEV) * 2).to(torch.bfloat16)
    r2t = torch.arange(prefix + T, dtype=torch.int32, device=DEV).view(1, -1)
    rpi = torch.zeros(1, dtype=torch.int64, device=DEV)
    seq = torch.tensor([prefix], dtype=torch.int64, device=DEV)
    visible = [prefix + t + 1 for t in range(T)]
    row = {"prefix": prefix}
    for name, kb, vb in (("bf16", k, v), ("fp8", k8, v8)):
        o = torch.empty_like(q)
        if not va.covered(q, o, kb, vb, T, HQ, HKV, D):
            raise SystemExit(f"verify kernel does not cover the {name} cache")
        run = lambda: va.verify_attention_gfx1151(q, o, kb, vb, r2t, rpi, seq, T, SCALE)  # noqa: E731
        run()
        row[f"{name}_kernel_err"] = err(o, reference(q, kb, vb, visible))
        row[f"{name}_ms"] = timed(run, reps)
        row[f"_{name}_out"] = o.float()
    # fp8 kernel output against the bf16 cache's exact reference: storage + kernel error
    row["fp8_vs_bf16_reference_err"] = err(row.pop("_fp8_out"), reference(q, k, v, visible))
    row.pop("_bf16_out")
    row["time_ratio"] = row["fp8_ms"] / row["bf16_ms"]
    # below 8k the call is launch-bound and the ratio is noise
    row["pass"] = (row["fp8_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
                   and (prefix < 8192 or row["time_ratio"] <= TIME_RATIO))
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
    for name, kb, vb, on in (("bf16", kp, vp, True), ("fp8", kp.to(FP8), vp.to(FP8), True),
                             ("fp8_stock", kp.to(FP8), vp.to(FP8), False)):
        switch(on)
        o = torch.empty_like(q)
        try:
            extend_attention_fwd(q, ke, ve, o, kb, vb, qo, kv_indptr, kv_indices, None, True, None,
                                 chunk, 1.0, 1.0, sm_scale=SCALE)
        except Exception as exc:  # the stock fp8 dot may not compile on every Triton
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}"
            continue
        ref = reference(q, torch.cat([kb.to(torch.bfloat16), ke]), torch.cat([vb.to(torch.bfloat16), ve]), visible)
        row[f"{name}_kernel_err"] = err(o, ref)
    switch(True)
    row["pass"] = row["fp8_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
    return row


def check_decode(prefix, splits=16):
    k, v = cache(prefix, 4)
    q = torch.randn(1, HQ, D, device=DEV).to(torch.bfloat16)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(prefix, dtype=torch.int64, device=DEV)
    row = {"prefix": prefix}
    for name, kb, vb, on in (("bf16", k, v, True), ("fp8", k.to(FP8), v.to(FP8), True),
                             ("fp8_stock", k.to(FP8), v.to(FP8), False)):
        switch(on)
        o = torch.empty_like(q)
        logits = torch.empty(1, HQ, splits, D, dtype=torch.float32, device=DEV)
        lse = torch.empty(1, HQ, splits, dtype=torch.float32, device=DEV)
        nsplit = torch.full((1,), splits, dtype=torch.int32, device=DEV)
        try:
            decode_attention_fwd(q, kb, vb, o, kv_indptr, kv_indices, logits, lse, nsplit, splits,
                                 SCALE, 1.0, 1.0, enable_lean=False)
        except Exception as exc:
            row[f"{name}_kernel_err"] = f"failed: {type(exc).__name__}"
            continue
        row[f"{name}_kernel_err"] = err(o, reference(q, kb.to(torch.bfloat16), vb.to(torch.bfloat16), [prefix]))
    switch(True)
    row["pass"] = row["fp8_kernel_err"] <= KERNEL_RATIO * row["bf16_kernel_err"] + KERNEL_SLACK
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="1k and 8k verify prefixes only, fewer timing reps")
    args = parser.parse_args()
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1201" or not gfx1201_fp8_kv.enabled():
        raise SystemExit(f"needs gfx1201 with SGLANG_GFX1201_FP8_KV_DEQUANT=1 (device {arch})")
    started = time.monotonic()
    prefixes = (1024, 8192) if args.quick else (1024, 8192, 23552, 47104, 88064)
    result = {"schema": 1, "device": arch, "torch": torch.__version__,
              "shapes": {"hq": HQ, "hkv": HKV, "d": D, "draft_tokens": T},
              "verify": [check_verify(p, 10 if args.quick else 30) for p in prefixes],
              "extend": [check_extend()],
              "decode": [check_decode(p) for p in ((8192,) if args.quick else (8192, 47104))]}
    result["seconds"] = round(time.monotonic() - started, 1)
    result["pass"] = all(r["pass"] for part in ("verify", "extend", "decode") for r in result[part])
    for part in ("verify", "extend", "decode"):
        for r in result[part]:
            print(part.upper(), json.dumps(r), flush=True)
    print("RESULT=" + json.dumps(result), flush=True)
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
