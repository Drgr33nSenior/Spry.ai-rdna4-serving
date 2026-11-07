# MXFP4 candidate: fused Gemma RMSNorm overlay (P3), 28 October 2026

This record extends the [tile-geometry candidate](MXFP4-TILE-GEOMETRY-CANDIDATE.md). The dense
MXFP4 serving candidate (published AMD Quark AWQ MXFP4 Qwen3.8-27B checkpoint, BF16 target and
draft KV, DFlash width 8, TP2) now carries one more change: every Gemma RMSNorm runs a private
fused Triton kernel instead of the torch-native seven-node chain. The owner accepted precision for
this base and promoted it as the private candidate package base on 28 October 2026. Nothing here
changes the official FP8 selection, the installed runtime or production defaults; nothing is
deployed. The kernel and the serving package are not distributed with this repository.

## What changed

The stock `sglang/srt/layers/layernorm.py` is mounted with two edits: a module-level fused kernel
(BF16 input, weight and output; FP32 statistics; the residual sum rounded to BF16 before the
statistics and returned, as the native chain does; `(1 + w)` applied after normalization; one
program per row) and a BF16-guarded early route in `GemmaRMSNorm.forward_hip`. Non-BF16 inputs fall
through to the unchanged stock branches. GEMM arithmetic, launch policy, requests and limits are
those of the tile-geometry candidate. The protected transport helper admits the one new member
under an owner-approved exact diff; seven other protected helpers are byte-identical.

## Component result (operation 87)

Real checkpoint norm weights (a layer input norm, the final norm, an attention q-norm) with captured
and seeded activations, 48 contexts and 96 cases on both gfx1201 cards, exact-order comparison
against the native chain:

- Maximum output difference 1 BF16 ulp (2 of 1,721,344 compared basis elements); FP32 variance
  relative difference at most 3.0e-7 against a 2^-20 bound; residual outputs bitwise equal.
- Zero spills; wave32; VGPR counts 8, 66 and 74 across the three specializations.
- Timing: hidden-size norms 5 to 15 times faster, attention-head norms at 2,047 rows 7 to 9
  times faster. The predeclared gate (3x at small row counts, 2x at 2,047 rows, worst candidate
  against best baseline) was missed on 5 of 48 contexts: four head-norm contexts at 16 rows
  (1.8x, launch-bound at a 6 to 7 microsecond replay floor) and one residual context with a single
  noisy batch. The owner chose to proceed to serving with the kernel as measured and record the
  miss as a limitation; the gate was not relaxed.

## Serving screen (operation 90)

Same four frozen 2,047-token requests, 64 forced greedy tokens each, request 1 traced after a
warm-up request, compared with the sealed tile-geometry run on the same boot. All seven predeclared
gates passed. Per rank:

| Per verify cycle (rank 0; rank 1 within 0.1 ms) | stock norms | fused norm | delta |
| --- | ---: | ---: | ---: |
| GPU span | 52.9 ms | 46.4 ms | -6.5 ms |
| kernel time | 41.4 ms | 39.4 ms | -2.0 ms |
| graph nodes | 3,030 | 1,741 | -1,289 |

The span saving exceeds the kernel-time saving: the remainder is per-node dispatch latency of HIP
graph replay (about 3.4 microseconds per node, measured on the sealed traces), which the fusion
removes with the nodes. Prefill kernel time fell 61 ms (1,284 to 1,223 ms). Traced-request server
e2e 1.853 to 1.821 s. The outputs match the stock-norm run for 19 tokens and then diverge into an
equivalent rendering; on this single prompt draft acceptance fell from 5.8 to 4.9 tokens per cycle,
which is why the full-menu gate below was required before promotion.

## Full-model quality/repeat gate (operation 91)

One package, two model starts, 106 requests (53 per start), post-release native CPU grading of
104 full-wire decodes: **74 of 74 natural answers QUALITY_PASS** (37 per start), 30 forced-cap
responses ungraded by design, two state-only cancellations. Both starts agree on content,
reasoning, finish reason and token count on all 53 cases. Independent release verified.

Draft acceptance on the 37 natural answers: 5.04 tokens per cycle against 5.27 for the stock-norm
run of the same menu (one run each). The answers differ textually from the stock-norm run on every
case, as expected under changed norm arithmetic, and all grade correct.

| Recorded metric | stock norms (op 83), start 1 / 2 | fused norm (op 91), start 1 / 2 |
| --- | ---: | ---: |
| C1 output tok/s | 62.49 / 62.20 | 117.29 / 116.61 |
| C1 median request latency | 16.38 s / 16.46 s | 8.73 s / 8.78 s |
| C2 aggregate output tok/s | 77.34 / 77.15 | 69.13 / 69.05 |
| C2 median individual latency | 26.48 s / 26.55 s | 29.64 s / 29.66 s |

These timing groups force 1,024 output tokens past the natural stop. The continuation is degenerate
text whose draft acceptance differs between runs (C1 5.99 against 3.21, C2 1.97 against 2.41
accepted tokens per cycle), so the swing in both directions tracks acceptance on degenerate text,
not kernel speed. They are descriptive only; the serving screen above is the speed evidence.

## Limitations retained

- The component timing gate was missed on head-norm contexts at 16 rows and one noisy residual
  context; the kernel was carried to serving under the owner's decision, not a changed threshold.
- The overlay wrapper (reshape, buffer allocation, launch) first ran on hardware in the serving
  screen; the kernel body had run in the component.
- The serving screen is one prompt; the server's first-token field was unusable in that run and
  prefill-finished timestamps were compared instead.
- Acceptance length is measured once per base on 37 natural answers.
- The grading replay consumed a layout-corrected derived copy of the sealed run root (package
  directory name and seal record shape); GPU evidence bytes were unchanged and re-verified against
  the owner-made seals. The sealed run root's execution summary carries a defective
  whole-body-hash agreement line; a correction record holds the text-level 53 of 53.

## Closeout

Operations 87, 90 and 91 charged 168 + 168 + 1,243 owner seconds; cumulative charges for
operations 78 to 91 are 4,299 owner and 4,412 caller seconds, every operation under a one-use
allowance with no automatic retry and all failures retained. The candidate pointer now names the
fused-norm package; the official FP8 selection is unchanged. Work in flight on the prefill path is
not part of this record. [Provenance](../provenance/MXFP4-P3-FUSED-NORM-20261028.json) records
package identities, seals, gates and charges.
