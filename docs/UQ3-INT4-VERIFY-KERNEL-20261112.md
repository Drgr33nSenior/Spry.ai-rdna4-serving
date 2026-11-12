# UQ3: INT4 KV in the verify kernel on gfx1201, 12 November 2026

**Result: not worth building. Every 4-bit integer KV layout tried makes target-verify
attention at least 1.6x slower than today's FP8 cache. Only one layout reaches that speed, and
it has the worst 4-bit quality measured. The layouts with acceptable quality cost 4-9x.**
UQ3 and K8V4 stay parked, and UQL (FP8) remains the large-context format.

This follows the real-data quality screen in
[UQ3-PHASE0A-REAL-20261111.md](UQ3-PHASE0A-REAL-20261111.md). The tool is
`tools/bench_int4_kv.py`.

## Why the INT4 hardware does not help

The R9700 (gfx1201) has INT4 matrix instructions: `v_wmma_i32_16x16x16_iu4` and
`v_wmma_i32_16x16x32_iu4`. Both operands of those instructions must be 4-bit.

In attention, one operand of each product is the query (Q.K^T) or the softmax weights (P.V).
Both are bf16. Quantizing either one to 4 bits destroys the scores. The nearest mixed path is
INT8 q x INT4 codes widened to INT8, and it ran at 4x the control (v1, below).

A 4-bit cache therefore saves only memory and bandwidth. Each value must be unpacked to bf16 on
the vector ALU before the bf16 WMMA can use it. There is no FP4 conversion either: the LLVM
gfx1201 assembler rejects it.

## Method

The timing follows patch 0021 / `check_fp8_kv_dequant.py`:

- Qwen3.8-27B TP2 shapes per card: 12 q heads, 2 KV heads, head dim 256, 8 draft tokens.
- 16 layers, each on its own cache, so reads come from DRAM.
- One CUDA graph; the reported time is median replay / 16.
- One R9700, Triton 3.9.0, torch 2.16.0a0+devrocm10.3.0, image
  `sglang-gfx1201@sha256:9fc1a58c…`.
- Runs: Jobs `int4-kv-bench-2d71c54096` (v1), `-92efa1df70` (v2) and `-a119874640` (v3), each
  in a short pause of the live service.

Correctness was checked at an 8k prefix against an fp32 reference on the same quantized values.

| Variant | Kernel error |
| --- | --- |
| control (bf16) | 4.2e-4 |
| i4 v1 | 2.3e-4 |
| half_g128 | 4.3e-4 |
| half_g32 | 2.3e-4 |
| magic1_kg64 | 3.9e-4 |
| magic1_kpt | 3.5e-4 |

All of these are bf16 rounding. The format error against the bf16 reference is 0.069-0.070 for
every 4-bit layout.

## Results (µs per layer, best tile of each variant)

| Shape | prod bf16 | **prod FP8 (UQ2, served)** | control bf16 | i4 G32 v1 | half G128 | half G32 | magic K G64 | **magic per-token** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8k | 50.5 | **46.8** | 56.2 | 153.6 | 306.2 | 427.9 | 138.9 | **79.5 (1.70x)** |
| 32k | 134.2 | **116.6** | 161.2 | 564.3 | 1118.8 | 1556.5 | 465.4 | **269.3 (2.31x)** |
| 64k | 262.9 | **232.4** | 331.5 | 1105.9 | 2202.9 | 3003.9 | 915.1 | **528.2 (2.27x)** |
| 120k | 464.2 | **413.8** | 600.5 | 1962.9 | 3925.2 | 5458.4 | 1659.8 | **969.4 (2.34x)** |
| 32k x2 | 260.6 | **229.3** | 256.2 | 669.5 | 1192.2 | 1600.5 | 598.2 | **367.4 (1.60x)** |

The ratio in brackets is against prod FP8. The prod columns come from the v3 run. v1 used the
interleaved G32 layout and its timings come from the v1 run. Production's own variation between
runs was under 2%.

The layout quality comes from the real capture (output SQNR / attention KL against FP8):

| Layout | Output SQNR | Attention KL vs FP8 |
| --- | --- | --- |
| FP8 (served) | 31.7 dB | 1x |
| G32 (K and V) | 23.1 dB | 6.1x |
| K G64 / V per-token | 21.6 dB | 7.4x |
| per-token (K and V) | 20.9 dB | **10.2x** |

## What each iteration showed

Instruction counts come from compiling offline for gfx1201 (all instructions, VGPR spills):

| Kernel (tile) | Instructions | WMMA | Bit ops | Conversions | Spill slots |
| --- | --- | --- | --- | --- | --- |
| control bf16 (64x8) | 2,179 | 96 | 198 | 0 | 2 |
| i4 G32 v1 (32x8) | 6,949 | 48 | 1,849 | 320 | 0 |
| half G128 (32x8) | 7,994 | 48 | 2,096 | 390 | 32 |
| half G32 (32x8) | 14,292 | 48 | 2,958 | 528 | 16 |
| magic K G64 (32x8) | 4,227 | 48 | 415 | 2 | 68 |
| **magic per-token (64x8)** | **3,855** | 96 | 379 | 8 | **0** |

1. **v1 (interleaved G32):** about 4.7x FP8. The cost is ALU work, not bandwidth: integer-to-
   float conversions, nibble shuffles from `tl.join`/`tl.reshape`, and a per-element scale.
2. **v2 (split halves, so no interleave shuffle):** slower, at 5.2-9.5x. Each half needs its own
   accumulator, and two `[R, 128]` fp32 accumulators push the kernel into register spills.
3. **v3 (`v_perm_b32` "magic" unpack):** a nibble placed under a 0x43 byte *is* the bf16 value
   128 + n. That removes the conversions, and the offset is subtracted once per row. With one
   scale per token for K and V, the loop keeps the control's single accumulator. That variant
   has no spills and 1.8x the control's instructions, which matches its measured 1.6x the
   control. Any finer K scale needs a per-chunk multiply on the score tile, and the spills
   return (G64: 2.6x slower again).

At batch 1, my control kernel is 1.11-1.29x slower than production bf16 at the same tile, because
production carries patch 0027's tuning. If the INT4 kernel got the same tuning, the per-token
variant would still be about 1.8x FP8 at 120k.

## Cost in serving terms

At a 120k context the per-token variant adds about 555 µs per full-attention layer. Over the 16
full-attention layers, that is about **+8.9 ms per target-verify step**. The FP8 attention cost
it replaces is 6.6 ms.

That figure covers verify only. Decode-only and extend/prefill attention, and the KV store
path, would each need their own INT4 kernels.

The gain would be 2x the tokens per byte of FP8: ~127k tokens per shared-mode conversation
instead of ~82k (~202k vs ~107k with the bounded draft pool). That gain comes at the lowest
quality of any layout screened.

## Decision and reopen conditions

**Parked.** No layout is both fast enough (within ~1.3x of FP8) and close to FP8 in quality.

Reopen only if one of these becomes true:

- the target gains a mixed bf16 x int4 matrix instruction, or a packed 4-bit to bf16
  conversion;
- end-to-end acceptance and quality runs show that per-token INT4 KV is tolerable for this
  model (no such evidence exists; the attention KL says otherwise).

## Reproduce

| Step | Command |
| --- | --- |
| Unit tests (CPU) | `make test-int4-bench` |
| GPU run (workstation) | Render with workstation `infrastructure/ai-stack/serving/kernel_check.py <image> --serving <copy> --tool int4-bench`, then run the Job with `~/ai-stack/private/kernel-job-run.sh <src> <job.json>`. The script pauses chat and pictures for ~15 min and always restores them. |
| Shorter run | `--shapes 8192,32768x2` |
