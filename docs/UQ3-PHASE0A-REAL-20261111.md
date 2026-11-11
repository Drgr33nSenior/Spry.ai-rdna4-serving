# UQ3 Phase 0a on real Q/K/V, 11 November 2026

**Result: K8V4 (keys stay FP8, values 4-bit) is the format to test end to end next. Full
4-bit UQ3 stays a stretch candidate, because its key error distorts attention about 6x as
much as today's FP8.** This replaces the proxy ranking
([UQ3-PHASE0A-PROXY-20261111.md](UQ3-PHASE0A-PROXY-20261111.md)): real attention is far
more tolerant than the proxy, and the ranking of formats holds.

## Capture

| Item | Value |
| --- | --- |
| Job | workstation `render.py capture`, arm Y (BF16 KV, so earlier layers are free of FP8 rounding), image `sglang-gfx1201@sha256:9fc1a58c…` |
| Prompt | 33,923 tokens of real text (the workstation repo's `docs/*.md`) as one chat turn; the first **16,384** captured |
| Tensors | TP rank 0 (2 of 4 KV heads, 12 of 24 query heads), all 16 full-attention layers, post QK-norm and RoPE; K, V `[16384, 2, 256]`; the 32 query rows ending at token 16,384 |
| File | `capture-20261111232550.pt`, 729,266,425 B, sha256 `ef8dbb42…80bc4` (owner's workstation) |
| Scoring | `tools/uq3_fakequant.py --capture` (workstation CPU, 16 s) |

## Results (means over 16 layers)

| Format | Bytes/token/card | K dB | V dB | logit σ (nats) | attention KL | vs FP8 | output dB | worst layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UQ2 (FP8, today) | 16 KiB | 31.5 | 31.5 | 0.064 | 2.97e-03 | 1x | **31.7** | 27 (28.6) |
| **K8V4** (K FP8, V I4G32-RHT) | 12.25 KiB | 31.5 | 20.3 | 0.064 | 2.97e-03 | **1x** | **25.5** | 7 (23.1) |
| K8, V I4G16-RHT | 12.5 KiB | 31.5 | 21.3 | 0.064 | 2.97e-03 | 1x | 26.3 | 55 (24.0) |
| **UQ3** = I4G32-RHT (K and V) | 8.5 KiB | 20.2 | 20.3 | 0.199 | 1.82e-02 | **6.1x** | **23.1** | 55 (18.8) |
| UQ3, V unrotated | 8.5 KiB | 20.2 | 18.7 | 0.199 | 1.82e-02 | 6.1x | 22.2 | 55 (17.9) |
| K I4G16-RHT, V I4G32 | 8.75 KiB | – | – | – | 1.45e-02 | 4.9x | 22.6 | 55 (18.9) |
| I4G16-RHT (K and V) | 9.0 KiB | 21.3 | 21.3 | 0.177 | 1.45e-02 | 4.9x | 24.1 | 55 (20.5) |
| I4G32, plain Hadamard | 8.5 KiB | 20.2 | 20.2 | 0.199 | 1.86e-02 | 6.3x | 23.0 | 55 (19.0) |
| I4G32, no rotation | 8.5 KiB | 18.8 | 18.7 | 0.239 | 2.78e-02 | 9.4x | 21.1 | 55 (17.3) |
| I4 per-vector-RHT | – | 18.0 | 18.1 | 0.257 | 3.02e-02 | 10.2x | 20.9 | 55 (17.2) |
| MXFP4-RHT | 8.5 KiB | 18.7 | 18.8 | 0.248 | 3.39e-02 | 11.4x | 20.0 | 55 (15.8) |

UQ3 output dB per layer (3 … 63): 23.1 22.1 23.9 21.6 21.8 21.3 23.9 23.6 23.2 22.3 24.9 22.8
21.8 18.8 25.1 30.2. The worst layers by attention KL against FP8 are 55 (7.4x), 23 (8.3x) and
15 (9.7x). Layer 55 is weakest on output for every 4-bit-K format.

## Findings

1. **Key precision drives attention distortion.** Every format that keeps K in FP8 has
   exactly today's logit noise (0.064 nats) and attention KL. Every 4-bit-K format sits at
   4.9-11x. The plan flags anything above 3x.
2. **K8V4 costs output accuracy only through V:** 25.5 dB vs 31.7, with an unchanged
   attention pattern. A G16 V scale gains another 0.8 dB for +0.25 KiB.
3. **Rotation is required for 4-bit:** +2.0 dB output, KL from 9.4x to 6.1x.
   **Random signs and plain Hadamard tie on real data too,** so keep the free signs.
4. **V rotation gains 0.9 dB,** just under the plan's 1 dB keep threshold. K8V4 still
   rotates V. If V's inverse rotation proves costly in the kernel, dropping it is acceptable.
5. **INT4 beats MXFP4 by 3.1 dB on real data,** wider than on the proxy (1.7 dB).
   A per-vector scale is clearly worse.

## Capacity (shared mode, per conversation, same 3.28 GiB per card)

| Format | Draft pool as today | With the bounded draft pool (plan #32) |
| --- | --- | --- |
| UQ2 (today) | 81,888 | ~107,500 |
| **K8V4** | **~99,700** | **~140,400** |
| UQ3 | ~127,400 | ~202,400 |

Large mode is already at the model's 262,144-token limit. Smaller formats there only
add concurrent full-length conversations.

## Decision and next steps

- **Next, Phase 0c / Phase 2:** a fake-quant write path in the KV cache (quantize and
  dequantize in place; every value is exact in BF16, so existing kernels read it) for
  **K8V4**, with UQ3 as a second arm. Measure teacher-forced KLD, long-context retrieval,
  the quality menu and DFlash acceptance against UQ2.
- **Pair K8V4 with the bounded draft pool** for about 1.7x shared-mode context. UQ3 is worth
  pursuing only if Phase 2 shows its 6x attention distortion doesn't cost real quality.
- **Kernel risk is smaller for K8V4:** the hot Q·K path keeps today's hardware FP8 decode.
  Only the V reader is new.

## Limits

- **One sample:** one prompt (English docs), 16,384 keys, 32 query rows, rank 0's heads.
  Code, chat and very long (128k-262k) contexts are not covered.
- **Not token quality:** "attention KL" is per-head softmax divergence, not token KLD.
  Phase 2 measures task quality.
