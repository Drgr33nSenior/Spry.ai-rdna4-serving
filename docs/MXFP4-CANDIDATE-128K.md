# MXFP4 candidate: 128K profile, stock endurance, and C4 prefill

The private MXFP4 candidate from [Phase 1](MXFP4-CANDIDATE-PHASE1.md) now serves a 128K context profile in which two
users can each hold a full-length conversation at once. With stock attention, long-document validation and a
61-minute endurance run passed every graded answer, with no GPU faults. The owner then adopted C4 extend attention
after a serving A/B measured 23.3% less cold-prefill time, accepting a failed warm/branch latency gate. **C4 endurance
has not been tested.** The public FP8 selection is unchanged, and nothing was deployed.

Evidence: [`MXFP4-CANDIDATE-128K-20261031.json`](../provenance/MXFP4-CANDIDATE-128K-20261031.json). Operations
167–170 and 174 ran on two Radeon AI PRO R9700 cards at TP2, with **235 W** power caps. Phase 1 ran at 300 W.

## Profile

| Setting | Value |
| --- | --- |
| Context length | 131,040 tokens |
| KV pool | 262,144 tokens, two full conversations |
| Running requests | 2 |
| Static memory fraction | 0.92 (stock 128K feasibility: 28.82 of 31.47 GB per card) |
| Prefill chunk | 2,048 tokens |
| Recurrent-state slots | 19 |
| Prefix reuse | radix cache |

The candidate retains Phase 1's seven overlays, including the packed QKVZ+BA route. C4 is the eighth overlay:
the guarded gfx1201, head-dimension-256 extend-attention path changes from `(64,64,4,1)` to `(64,64,8,1)`
(block M, block N, warps, stages). Other shapes and architectures retain their existing path. Target and draft
KV remain BF16; the BF16 DFlash drafter stays at width 8. Two corrections went into the 128K profile:

- **Verify-adapter index buffer.** The index buffer is now sized as `rows × (context + 8)`. The previous
  `rows × context` sizing was latent and could overflow only when a request came within four tokens of the context
  limit.
- **Commit-observation limit.** It is now doubled, so the observer has at least 2× headroom over the longest run.

## Long-document validation (B1, stock attention)

Two fresh starts each served 13 greedy requests:

- ~120K-token documents, each asked cold, as an exact repeat and as a branch question;
- a concurrent pair of ~60K-token questions;
- one near-limit request, alone and as a concurrent pair.

| Check | Result |
| --- | --- |
| Graded answers | 20/20 |
| Near-limit requests | 6/6 intact: 4,096 output tokens, prompt plus output 131,037, decoded text equal to the wire |
| Prefix reuse | 118,784 cached tokens on repeat and branch questions (99%); 0 on cold questions |
| Two users at full length | 262,074 tokens resident at once, with no retractions |
| Faults | None |

The harness controller refused this operation's formal acceptance after serving. Its post-run rule required a natural
stop on every row, while the near-limit rows were designed to stop at their output cap. The owner accepted the run as
diagnostic evidence after native CPU grading. The two earlier B1 attempts stopped before any request was served; both
were packet defects (a 32K benchmark spec and a 4 MiB parity-report cap) and are retained.

At 235 W:

- **Cold ~120K document:** prefill took about 97 s, averaging ~1,240 tok/s. The rate falls from ~2,300 tok/s in the
  first 32K to ~790 tok/s past 96K, as attention over the growing context costs more.
- **Repeat or branch question on the cached document:** 9–11 s per request.
- **Decode at ~120K context:** ~107 tok/s for one user.

## Endurance (W5 v2, stock attention)

Two fresh starts each served 176 natural-stop requests: the W1 menu nine times plus five B1 rows. That is about
61 minutes of serving. In-process instrumentation was **off** (production mode); pass and fail came only from client,
server-log and host evidence.

| Check | Result |
| --- | --- |
| Graded answers | 352/352 |
| B1 rows token-equal to the B1 run (gate) | 10/10 |
| W1 rows token-equal to the Phase 1 packed run (information) | 171/171 in both starts |
| Kernel log | 0 amdgpu/KFD/DRM fault lines across the window |
| Host samples | 402/402, constant boot ID, 235 W throughout |
| Temperatures | Junction max 95 °C, memory max 87 °C |
| VRAM drift | 2 MiB across the hour (limit 256 MiB) |
| Throughput drift | Under 0.3% (limit 10%) |

The W1 equality shows that production mode and the 128K settings did not change any output token on those rows.
This operation preceded C4 adoption and does not establish C4 endurance.

## C4 serving A/B and adoption (operation 174)

Two fresh starts ran the same 16-request menu at 235 W: stock attention first, C4 second. Both used the 128K
production profile with in-process instrumentation off. All 32 natural-stop answers passed native CPU grading.
All 16 paired final answers matched; full output tokens matched in 9/16 pairs, including all six W1 short rows.
The seven long-context pairs differed in reasoning.

| Measurement | Stock | C4 | Change |
| --- | ---: | ---: | ---: |
| Mean cold ~120K prefill, two documents | 96.5 s | 74.0 s | −23.3% time |
| Mean cold request, including completion | 105.07 s | 83.34 s | −20.7% time |
| Warm A request | 10.83 s | 9.72 s | −10.2% time |
| Branch request | 9.11 s | 9.72 s | +6.7% time |
| Warm B request | 9.01 s | 9.49 s | +5.4% time |
| Concurrent ~60K pair makespan | 71.81 s | 62.32 s | −13.2% time |
| Short W1 aggregate HTTP throughput, combined TP2 | 158.44 tok/s | 158.95 tok/s | +0.3% rate |

**The A3 warm/branch latency gate failed.** It permits at most 5% higher latency; branch and warm B exceeded
that limit. Both produced 691 output tokens with C4 versus 634 with stock: 57 extra reasoning tokens and the
same final answer. Unequal generated work prevents this comparison from isolating kernel speed. A1 correctness,
A2 cold prefill, and A4 short-task throughput passed. The short W1 rate includes prefill and HTTP completion;
it is a different metric and request subset from the server decode rates below.

The owner accepted the A3 exception and adopted start 2 as the private candidate on 2026-11-01. The original
`ADOPTION_CRITERIA_NOT_MET` verdict and 5% threshold remain unchanged. The matched stock arm, start 1 in the
same packet, is retained as rollback. C4 is not bitwise-equivalent to stock.

All seven external criteria passed: 93/93 host samples, 235 W caps, temperature and VRAM limits, clean kernel
and fatal server logs, and successful process exits. Maximum sampled junction temperature was 93 °C.
Independent GPU and CPU release passed. No compilation was logged after readiness.

This was one fixed stock-then-C4 order, with two cold documents per start and prefill logs at 1 s resolution.
Warm prefill rounded to 0 s, so no warm-prefill ratio is claimed. The failed initial SSH preflight and original
JSON report writer are retained. A local report correction changed only metric-key serialization; it repeated
neither GPU work nor native grading and changed no validation threshold.

## Decode throughput on the W1 menu

These historical measurements precede C4 adoption. They are server-reported decode rates, combined across both cards.
The mean acceptance length is identical in every row: 6.95 tokens per verify step at one user and 7.09 at two.

| Run | Conditions | 1 user | 2 users |
| --- | --- | ---: | ---: |
| FP8 selected profile | 300 W | ~163.5 tok/s | ~254 tok/s |
| MXFP4 candidate + packed route | 300 W, instrumented, 32K | ~198 tok/s | ~380 tok/s |
| MXFP4 candidate, stock attention, 128K profile | 235 W, production mode, 128K | ~185.5 tok/s | ~356.6 tok/s |

The last row changes the power cap, the instrumentation and the profile at the same time, so this run cannot separate
their effects. These rates exclude prefill and HTTP overheads.

## Limits

- The two-request limit is unchanged. Batch sizes 3–8 have not run on this stack.
- Phase 1's unexplained GPU memory fault (1 in 12 sampled starts) has not recurred, but it remains unresolved.
- C4 attention tuning and the bounded serving A/B are complete; C4 endurance remains untested.
- The selected Torch 2.14 build has no Kineto GPU profiling support. No in-process GPU trace was available.
- All results come from bounded menus at one or two concurrent requests, not from production traffic.
