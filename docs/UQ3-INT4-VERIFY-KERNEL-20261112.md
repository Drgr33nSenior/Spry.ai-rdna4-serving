# UQ3: INT4 KV in the verify kernel on gfx1201, 12 November 2026

**Result: not worth building.** The decision rests on two independent failures.

- **Quality.** Every 4-bit-K layout distorts attention 4.9x FP8 or worse, against the plan's 3x
  flag, and loses 8-11 dB of output accuracy
  ([UQ3-PHASE0A-REAL-20261111.md](UQ3-PHASE0A-REAL-20261111.md)). No kernel changes this.
- **Speed.** As written, the fastest INT4 verify kernel is 1.7-2.3x the FP8 time, and it uses
  the worst-quality layout (one scale per token). Layouts with smaller scale groups cost
  2.9-13x.

UQ3 stays parked. K8V4 also stays parked, by inference: it was not timed (see below). UQL (FP8)
remains the large-context format.

Fable 5.2 reviewed this work and signed it off on 12 November 2026, with the corrections that
are now included in this document. The tools are `tools/bench_int4_kv.py` (kernels and timing)
and `tools/uq3_group_mixes.py` (scale-group quality).

## Why the INT4 hardware does not help directly

The R9700 (gfx1201) has INT4 matrix instructions: `v_wmma_i32_16x16x16_iu4` and
`v_wmma_i32_16x16x32_iu4`. `llvm-mc -mcpu=gfx1201` accepts them only with 4-bit data in both
operands; mixed widths are rejected. `swmmac` and `v_dot8` are 4-bit on both sides too. There
is no `v_cvt_f32_fp4` and no `v_pk_sub_bf16`.

In attention, one operand of each product is the query (Q.K^T) or the softmax weights (P.V),
and both are bf16. A plain 4-bit cache therefore saves only memory and bandwidth: each value
must be unpacked to bf16 on the vector ALU before the bf16 WMMA can use it. The INT8-q path
(q in int8, codes widened to int8) ran at 4x the control (v1).

One route was not tried: split q into nibble planes, so Q.K^T runs on the 4-bit WMMA with no
unpack. It is listed under the reopen conditions.

## Method

Timing follows patch 0021 / `check_fp8_kv_dequant.py`:

- Qwen3.8-27B TP2 shapes per card: 12 q heads, 2 KV heads, head dim 256, 8 draft tokens.
- 16 layers, each on its own cache, so reads come from DRAM.
- One CUDA graph; the reported time is median replay / 16.

| Item | Value |
| --- | --- |
| Hardware and software | One R9700, Triton 3.9.0, torch 2.16.0a0+devrocm10.3.0, image `sglang-gfx1201@sha256:9fc1a58c…` |
| Production kernels | Run at their own split count (32 on gfx1201) |
| This tool's kernels | 16 splits |
| Runs | Jobs `int4-kv-bench-2d71c54096` (v1), `-92efa1df70` (v2) and `-a119874640` (v3), each in a short pause of the live service |

**Correctness check.** The check compares each kernel at an 8k prefix with an fp32 reference on
the same quantized values:

| Variant | Kernel error |
| --- | --- |
| control (bf16) | 4.2e-4 |
| i4 v1 | 2.3e-4 |
| half_g128 | 4.3e-4 |
| half_g32 | 2.3e-4 |
| magic1_kg64 | 3.9e-4 |
| magic1_kpt | 3.5e-4 |

The check has these limits:

- **Degenerate input.** In these runs the queries and the cache came from the same random seed,
  so q was a multiple of K for half the heads, and those heads' attention was one-hot. The
  logged "format error" (0.069-0.070) is therefore only the V quantization step, not an
  attention metric, and is not used here. The tool now seeds q separately.
- **Size of the errors.** The errors are relative to a maximum output of about 2.2. They rule
  out ordering or offset bugs, which would give errors of order 1. They do not show that the
  INT4 kernels are as accurate as bf16 rounding.
- **Not checked.** The check ran each variant's default tile, not its timed-best tile. It did
  not cover the cancellation in `acc - 136 * corr` at long, diffuse contexts.
- **Unpack derivation.** The `v_perm_b32` unpack (selectors 0x04020400 / 0x04030401, pack=2
  element order) and the offset corrections were also derived by hand in review and found
  correct.

## Results (µs per layer, best tile of each variant)

| Shape | prod bf16 | **prod FP8 (UQ2, served)** | control bf16 | i4 G32 v1 | half G128 (v2) | half G32 (v2) | magic K G64 | **magic per-token** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8k | 50.5 | **46.8** | 56.2 | 153.6 | 306.2 | 427.9 | 138.9 | **79.5 (1.70x)** |
| 32k | 134.2 | **116.6** | 161.2 | 564.3 | 1118.8 | 1556.5 | 465.4 | **269.3 (2.31x)** |
| 64k | 262.9 | **232.4** | 331.5 | 1105.9 | 2202.9 | 3003.9 | 915.1 | **528.2 (2.27x)** |
| 120k | 464.2 | **413.8** | 600.5 | 1962.9 | 3925.2 | 5458.4 | 1659.8 | **969.4 (2.34x)** |
| 32k x2 | 260.6 | **207.0** | 256.2 | 669.5 | 1192.2 | 1600.5 | 598.2 | **367.4 (1.78x)** |

The ratio in brackets is against prod FP8. Where each column comes from:

- **prod bf16, control and magic:** v3.
- **i4 G32:** v1.
- **half:** v2.
- **prod FP8, batch 1:** v3.
- **prod FP8, 32k x2:** v1. In v2 and v3 the bench gave FP8 the (32, 4, 32) tile at batch 2
  (229.3 µs), but patch 0027 uses (64, 4, 64) for B <= 2, so those runs flattered INT4. The tool
  is now fixed.

Production bf16 at 32k x2 also differed between runs (289.8 µs in v1, 260.6 in v3). Batch-1
production cells agreed within 2.6%.

**Ratios against FP8:**

| Variant | Ratio |
| --- | --- |
| magic per-token | 1.70-2.34x |
| magic K G64 | 2.9-4.0x |
| i4 G32 (v1) | 3.2-4.8x |
| half G128 | 5.8-9.6x |
| half G32 | 7.7-13.3x |

Quality on the real capture (`tools/uq3_group_mixes.py`, output SQNR / attention KL against
FP8; all INT4 rows rotated with random-sign Hadamard, whose kernel cost is not timed here):

| Layout | Output SQNR | Attention KL vs FP8 |
| --- | --- | --- |
| FP8 (served) | 31.7 dB | 1x |
| G32 (K and V) | 23.1 dB | 6.1x |
| K G64 / V per-token | 21.6 dB | 7.4x |
| per-token (K and V) | 20.9 dB | **10.2x** |
| K FP8 / V per-token | 23.7 dB | 1x |

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

1. **v1 (interleaved G32): 3.2-4.8x FP8.** The cost is ALU work, not bandwidth:
   integer-to-float conversions, nibble shuffles from `tl.join`/`tl.reshape`, and a
   per-element scale.
2. **v2 (split halves, so no interleave shuffle): slower.** Each half needs its own
   accumulator, and two `[R, 128]` fp32 accumulators push the kernel into register spills.
3. **v3 (`v_perm_b32` "magic" unpack).** A nibble placed under a 0x43 byte *is* the bf16
   value 128 + n. That removes the conversions, and the offset is subtracted once per row.
   - **One scale per token for K and V:** the loop keeps the control's single accumulator. The
     kernel has no spills, 1.8x the control's instructions, and measures 1.6x the control.
   - **A finer K scale:** it needs a per-chunk multiply on the score tile, and the spills
     return. K G64 is 1.6-1.75x the per-token variant.

**Headroom not taken (unmeasured).** In review, the per-token kernel's extra loop instructions
were mostly wait/delay hints and Triton's software bf16-to-fp32 conversion for `corr`. A
bitcast-and-shift would replace that conversion. Raising the tool's 16 splits to production's
32 is another lever. Fable's estimate for a tuned per-token kernel is about 1.4-1.6x FP8:
still above the ~1.3x bar, and still the worst-quality layout.

**K8V4 (FP8 K, INT4 V) was not timed.** An offline compile of a V-only magic loop comes to
2,900 loop instructions, against 2,943 for full magic per-token. The V side carries nearly all
of the overhead, so K8V4 would not be meaningfully faster than the per-token row.

## Cost in serving terms

At a 120k context the per-token variant adds about 555 µs per full-attention layer. Over the 16
full-attention layers, that is about **+8.9 ms per target-verify step**, on top of 6.6 ms of FP8
attention. A total verify-step time was not measured, so this is not expressed as a share of
the step.

That figure covers verify only. Decode-only and extend/prefill attention, and the KV store
path, would each need their own INT4 kernels.

The gain would be about 2x the tokens per byte of FP8: ~127k tokens per shared-mode
conversation instead of ~82k (~202k vs ~107k with the bounded draft pool).

## Decision and reopen conditions

**Parked.** No layout is both fast enough (within ~1.3x of FP8) and close to FP8 in quality.

Reopen if any of these becomes true:

- **End-to-end evidence.** Acceptance-rate and quality runs show that this model tolerates
  4-bit K at about 6x attention KL. This is the real gate, and no such evidence exists.
- **New hardware support.** The target gains a mixed bf16 x int4 matrix instruction, or a
  packed 4-bit to bf16 conversion.
- **The nibble-plane route (untested).** Split int8 q as 16 * hi + lo and run Q.K^T on
  `v_wmma_i32_*_iu4` directly; the signedness flags are set per operand.
  - **Limits:** V stays on the unpack path, because P.V contracts over non-contiguous tokens.
    It needs a hand-written kernel (Triton cannot emit the instruction): weeks of work.
  - **Review estimate:** about 1.3-1.5x FP8 at roughly 22.5 dB / 6.1x KL (K G32, V
    per-token). Applied to K alone with V in FP8, it would run near FP8 speed, but give only
    1.33x the capacity at 6x KL.

## Reproduce

| Step | Command |
| --- | --- |
| Unit tests (CPU) | `make test-int4-bench` |
| Quality mixes (CPU) | `python3 -I tools/uq3_group_mixes.py capture-20261111232550.pt` |
| GPU run (workstation) | Render with workstation `infrastructure/ai-stack/serving/kernel_check.py <image> --serving <copy> --tool int4-bench`, then run the Job with `~/ai-stack/private/kernel-job-run.sh <src> <job.json>`. The script pauses chat and pictures for ~15 min and always restores them. |
| Shorter run | `--shapes 8192,32768x2` |
