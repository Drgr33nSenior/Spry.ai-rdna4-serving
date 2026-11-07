# MXFP4 tile-geometry candidate: launch policy, quality gate and A8 retirement

This candidate serves the published AMD Quark AWQ MXFP4 Qwen3.8-27B checkpoint on two
Radeon AI PRO R9700s (TP2) with BF16 target and draft KV caches through a private MXFP4
component that is not distributed with this repository. It is a private candidate: **not
selected, not deployed, not qualified**. The official FP8 target remains the selection. The
owner accepted its numerical behaviour on 2026-10-28 against the retained natural-answer
oracle; that acceptance is not token equality with the FP8 selection.

## Launch policy under test

Three changes to how the unchanged AITER-derived FP4 GEMM kernel is launched, with the
same BLOCK_SIZE_K and K-split so the accumulation order is unchanged:

- rows M up to 16 take the sixteen-row tile (verification-cycle and small-batch projections);
- rows M of 1,024 and above take the 128 by 128 tile with eight warps (prefill projections);
- the opt-in A8 route (per-token FP8 activations with the unroll4 FP8 WMMA gate/up kernel) is
  retired and its modules are removed from the package.

Component evidence on real shards, exact same-quantized parity in every context: the sixteen-row
tile cut median time 43 to 51 percent across four small-M layouts; the 128 by 128 tile cut
combined time 17.9 to 19.6 percent at M2047 and M2048; the unroll4 FP8 kernel was 1.8 to 3.3
percent slower than the sixteen-row MXFP4 tile on the same worker.

## Serving screens

The screening workload is one model start and four serial requests with 2,047-token prompts
and 64 forced tokens, one request traced with profiler markers.

| Screen | Change | Outcome |
| --- | --- | --- |
| 1 (op 81) | sixteen-row tile for M up to 16 | token-exact; verify cycle 15 ms faster; e2e 7.5 percent lower |
| 3 (op 82) | A8 route inactive | zero unroll4 dispatches; tokens diverge from the A8-era baseline at token 16 by design (the token-exact gate was mis-specified and reported as failed) |
| 2 (op 83) | 128 by 128 prefill tile | token-exact to Screen 3; prefill GEMM 18.8 percent lower on both ranks; kernel-sum gate missed on rank 1 because the cold traced request compiled new Triton specializations (reported as failed) |
| 4 v1 (op 84) | trace window moved to a warm request | failed before any request on a trace-runtime ownership error; allowance consumed |
| 4 v2 (op 85) | same, corrected | all gates: prefill kernel sum 1283 / 1283 ms versus 1,522 / 1,526 ms (15.7 / 15.9 percent lower); collectives back at baseline |
| 5 v2 (op 86) | A8 modules removed from the package | all gates; token-exact to Screen 2; within noise of Screen 4 v2 |

Warm end-to-end latency on the four-request workload fell from 2.30 s (operation 80, A8-era
baseline) to 1.82 s. These are server-reported latencies on a diagnostic workload, not a
service benchmark.

## Quality and repeat gate

Operation 83 ran two model starts and the frozen 106-request menu (53 per start) on the
candidate base that carries the launch policy, then the retained post-release native CPU
grading: 104 full-wire decodes, **74 of 74 natural answers passed** (37 per start), 30
forced-cap responses ungraded by design, two state-only cancellations. The two starts agree
token for token on all 53 cases. Independent release was verified.

| Recorded metric | Earlier MXFP4 B64 | A8 unroll4 (partial run) | This candidate, start 1 / start 2 |
| --- | ---: | ---: | ---: |
| C1 output tok/s | 40.38 | 35.88 | 62.49 / 62.2 |
| C1 median request latency | 25.34 s | 28.55 s | 16.38 s / 16.46 s |
| C2 aggregate output tok/s | 53.15 | 53.23 | 77.34 / 77.15 |
| C2 median individual latency | 38.52 s | 38.47 s | 26.48 s / 26.55 s |

Descriptive server-reported rates on the fixed menu (2,047-token prompts, 1,024 forced output
tokens). They are not a speed selection and not comparable with the FP8 sprint numbers, which
use different workloads.

## Limitations retained

- The quality gate graded the base before the A8 modules were removed. The promoted cleanup
  base has byte-identical GEMM, kernel and scheme sources and was token-exact in its serving
  screen; the gate was not repeated on it.
- The grading replay consumed a layout-corrected derived copy of the sealed run root; the GPU
  evidence bytes were unchanged and re-verified against the owner-made seals.
- The cost-diagnostic screens retain no per-module scheme witness for the cleanup base.
- The protected transport helper was changed for the cleanup (member table, flat staging,
  member limit) under an owner-approved exact diff.

## Closeout

Cumulative charges for operations 78 to 86: owner 2722 s, caller
2794 s. Every operation ran under a one-use allowance with no automatic
retry; failures are retained. No campaign is prepared. UltraQuant KV integration remains deferred.
[Provenance](../provenance/MXFP4-TILE-GEOMETRY-CANDIDATE-20261028.json) records package identities,
seals, gates and charges.
