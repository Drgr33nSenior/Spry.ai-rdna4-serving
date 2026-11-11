# UQ3 Phase 0a: offline KV format ranking (proxy), 11 November 2026

Phase 0a of the UQ3 plan scores candidate 4-bit KV formats on attention before any kernel
is written. Real Q/K/V captures from the 16 full-attention layers don't exist yet, so this
run uses a **proxy**: K and Q come from the checkpoint's real per-layer `q_norm`/`k_norm`
gains and real partial RoPE, while token content and V are synthetic. The proxy ranks
formats. It does not measure model quality.

## Inputs

| Input | Source |
| --- | --- |
| Norm gains | `q_norm`/`k_norm` of layers 3, 7, …, 63 (BF16), range-fetched from `amd/Qwen3.8-27B-Quark-AWQ-MXFP4@5233554c` by `tools/uq3_fetch_norms.py` (header and 32 vectors only) |
| K, Q | Student-t (df 5) tokens → GemmaRMSNorm `x̂·(1+w)` with the real gains → RoPE on dims 0-63, θ = 1e7 |
| V | Synthetic: Student-t (df 4) with log-normal channel scales (σ 0.75) |
| Shape | 4,096 keys, 32 query rows, per card: 2 KV heads and 12 query heads per layer |

The real gains are far from flat. Layer 31's K channel scales `1+w` range from −0.15 to 2.9.
That is the fixed outlier pattern a 4-bit K has to absorb.

## Results

`python3 tools/uq3_fakequant.py --proxy norms.json --tokens 4096 --queries 32` (workstation CPU, 7 s).
Means over 16 layers:

| Format | K dB | V dB | logit σ (nats) | attention KL | output dB | worst layer |
| --- | --- | --- | --- | --- | --- | --- |
| Y (BF16) | 55.6 | 55.6 | 0.004 | 1.0e-05 | 45.6 | 23 (43.8) |
| **UQ2 (FP8, today)** | 31.5 | 30.5 | **0.061** | **2.6e-03** | **21.3** | 11 (20.1) |
| I4G32, no rotation | 18.2 | 16.5 | 0.280 | 3.9e-02 | 11.3 | 43 (10.3) |
| I4G32, plain Hadamard | 20.3 | 20.6 | 0.221 | 2.4e-02 | 14.5 | 43 (13.4) |
| **UQ3 = I4G32-RHT** | 20.2 | 20.6 | **0.222** | **2.4e-02** | **14.3** | 55 (13.3) |
| UQ3 with V unrotated | 20.2 | 16.5 | 0.222 | 2.4e-02 | 13.0 | 35 (12.2) |
| I4G16-RHT | 21.3 | 21.6 | 0.196 | 1.9e-02 | 15.4 | 43 (14.4) |
| I4 per-vector-RHT | 18.0 | 18.6 | 0.286 | 4.0e-02 | 12.1 | 43 (10.8) |
| MXFP4-RHT | 18.7 | 18.9 | 0.266 | 3.6e-02 | 12.6 | 31 (11.4) |
| **K8V4** (K FP8, V I4G32-RHT) | 31.5 | 20.6 | **0.061** | **2.6e-03** | **18.2** | 11 (17.5) |

UQ3 output SQNR per layer (3 … 63): 16.2, 15.9, 14.2, 14.5, 14.2, 14.5, 13.5, 13.7, 13.4, 14.4,
13.5, 13.6, 13.4, 13.3, 17.3, 13.3. No single layer stands out.

## Findings

1. **Rotation is required for 4-bit K.** It improves output by 3.0 dB and cuts logit noise
   from 0.28 to 0.22 nats against the real `k_norm` outlier pattern.
2. **Random signs and plain Hadamard tie on the proxy.** That is expected, because proxy
   tokens have zero mean, and a mean direction is what random signs protect against. A real
   capture decides. Keep signs on (they cost nothing).
3. **V rotation gains 1.3 dB of output,** which meets the plan's 1 dB keep threshold. But
   proxy V is synthetic, so a real capture decides.
4. **INT4 beats MXFP4 by 1.7 dB** at the same size, confirming the INT4 choice for gfx1201.
   A per-vector scale loses 2.2 dB. G16 gains 1.1 dB for +0.5 KiB per token per card
   (9.0 vs 8.5 KiB).
5. **The gap to UQ2 is large.** UQ3's attention KL is about 9x UQ2's, and its logit noise is
   0.22 vs 0.06 nats. The plan flags anything above 3x UQ2. On this proxy, **K8V4** keeps
   UQ2's K noise exactly, costs 3 dB of output, and stores 12.25 KiB per token: about
   1.7x shared-mode context with the bounded draft pool, against about 2.4x for UQ3.
6. **No layer needs special treatment.** The layers 3 / 63 FP8 knob isn't indicated, though
   layer 59 is the most tolerant.

## Limits

- **Synthetic content:** token content and V are synthetic. Real attention is sparser and
  peakier, which can either hide or amplify logit noise.
- **Attention KL is not token KLD:** "attention KL" is the per-head softmax divergence, not
  teacher-forced token KLD. Phase 2 measures the latter.
- **Single run:** one seed, 4,096 keys. Long-context behaviour (128k-262k) is not covered.

## Next (owner-run, GPU)

1. **Capture real tensors:** Q/K/V after QK-norm and RoPE, for the 16 layers, on a few long
   prompts, saved as `{layer: {"q": [Tq, Hq, 256], "k": [N, Hkv, 256], "v": [N, Hkv, 256]}}`.
   Then run `tools/uq3_fakequant.py --capture FILE.pt` to settle findings 2, 3 and 5 on real data.
2. **If UQ3's attention KL stays above 3x UQ2 on the capture,** move to K8V4 (the plan's
   ladder), then to the Phase 0c fake-quant write path in large mode for teacher-forced KLD.
