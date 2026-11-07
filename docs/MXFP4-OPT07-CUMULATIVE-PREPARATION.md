# OPT-07 cumulative optimization preparation

**Operation 215 completed: F1 CORRECT and RETAINED_POSITIVE at 235 W; released and parked.** Operation 214 remains component USEFUL. The approved four-arm run passed on both cards. OPT-00 through OPT-06 keep their signed-off CPU dispositions, and all 21 original/Exa references remain in the register. No runtime selection or adoption changed.

The immediate composition is **verification prefetch plus cap32**. It now passes the physical component at 235 W. Gated norm plus MXFP4 quantization is retained as a conditional third member after its positive synthetic component; production qualification remains outstanding. No serving gain is established.

## Four-arm component result — operation 214

The configured power caps were 235 W on both cards before and after the sequential work. These are cap readbacks, not power-draw or energy measurements.

| Card | Stock cap16 | Prefetch cap16 | Stock cap32 | Prefetch + cap32 | Combined saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 22.986 ms | 20.093 ms | 22.574 ms | 19.720 ms | 14.21% |
| 1 | 22.915 ms | 20.059 ms | 22.496 ms | 19.678 ms | 14.13% |

These primary-case medians cover the split helper plus 16 attention layers at a 119,990-token prefix and 8 verification rows. They are not request latency. Combined gaps exceed the frozen noise bounds on both cards. All long cases improve by 13.19–14.43%; the short-case medians worsen by 1.00% and 0.09%, within the observed ranges. No resolved combined regression occurs against any other arm.

Cap32's extra primary reduction over prefetch is 1.86% and 1.90%, positive but unresolved against the timing spread. The combined gain is measured; cap32's independent added benefit remains uncertain. The original standalone cap32 verdict is unchanged.

All 128 layer references, 384 candidate comparisons to FP32, bitwise prefetch checks at both caps, poison/write masks, binary identity and resource gates passed. Peak allocated/reserved bytes were 4,466,673,664 / 4,596,957,184 per card; allocations returned to zero. The component took 17.14 s, the caller 50.21 s, and independent release is recorded at checkpoint 581.

Serving remains **HOLD_FOR_EVIDENCE** under the frozen spending rule. Complete or explicitly exclude F1, then authenticate the 235 W serving control, request-wide route/dependency counts, ordinary-decode and KV-capacity witnesses, and economic assessment before proposing a serving packet. No further GPU work is authorised.

## Retained benefits and baseline figures

These are within-operation comparisons. The historical power settings and measurement boundaries differ, so the rows cannot be combined into a speedup forecast. Component times cover 16 sequential layers; they are not request latency.

| Candidate and case | Baseline → candidate, card 0 | Baseline → candidate, card 1 | Disposition |
| --- | ---: | ---: | --- |
| Verification prefetch, 119,990-token sequential prefix, operation 211 at 300 W | 20.756 → 18.201 ms | 20.872 → 18.191 ms | Component USEFUL; serving benefit remains unqualified |
| Cap32, same prefix case, operation 213 at 235 W | 22.655 → 22.068 ms | 22.597 → 22.110 ms | Positive primary signal inside the observed ranges; standalone NOT_USEFUL |
| Cap32, concurrent-row case, operation 213 at 235 W | 22.028 → 21.337 ms | 22.050 → 21.382 ms | Median gaps exceed the observed ranges; retain for composition |
| C4 accumulator fold, prefix 16,384/query 2,048, operation 190 | 192.187 → 181.893 ms | 193.047 → 182.823 ms | Long-prefix benefit retained |
| Same C4 candidate, zero prefix/query 2,048 | 11.260 → 11.849 ms | 11.389 → 11.893 ms | Resolved regression blocks the unchanged candidate from the bundle |

Operation 213 had no resolved regression and passed all per-layer references on both cards. Its fixed 10% standalone rule remains unchanged. The C4 accumulator result also remains NOT_USEFUL; a future source or routing repair needs its own mechanism and validation. Its 235 W record is a post-run observation, not a continuous power trace.

Operation 212's historical prefetch serving observations were smaller: warm C1 was 9.024 → 8.763 s and warm C2 was 8.319 → 8.089 s. The balanced latency reduction was 0.767%, below the serving threshold. That run lacked the required production route witness, so it does not establish an accepted serving gain. OPT-00 repaired the witness on CPU; the new route must emit and authenticate it physically.

F1's reviewed component completed as operation 215. Its synthetic exactness, real-tensor compiler identity, component graph lifetime, cleanup and 235 W power gates pass on both cards. Production engagement and representative model activations remain unmeasured. The lossless codec remains a CPU storage result without a qualified GPU decoder.

## F1 norm/quantizer component — operation 215

**RETAINED_POSITIVE; standalone NOT_USEFUL under the unchanged all-case 10% rule.** Every median improved and no resolved regression occurred. Both T8 inclusive gains exceed noise. T16 inclusive gains on both cards and the card-0 T16 boundary gain remain unresolved.

| Scope across 48 synthetic layers | Card 0 baseline → fused | Card 1 baseline → fused | Median saving |
| --- | ---: | ---: | --- |
| T8 norm + quantizer | 0.272242 → 0.185841 ms | 0.274641 → 0.185520 ms | 31.74% / 32.45%; resolved on both |
| T8 including unchanged GEMM | 2.009869 → 1.914970 ms | 2.055948 → 1.962307 ms | 4.72% / 4.55%; resolved on both |
| T16 norm + quantizer | 0.290521 → 0.193320 ms | 0.289821 → 0.193081 ms | 33.46% / 33.38%; card 0 unresolved |
| T16 including unchanged GEMM | 2.007329 → 1.922509 ms | 2.052448 → 1.952428 ms | 4.23% / 4.87%; unresolved on both |

These are captured component times, not request latency or tokens per second. The unchanged GEMM dilutes the boundary saving. The T16 inclusive baseline ranges are 0.264400 and 0.270201 ms, larger than the gaps of 0.084820 and 0.100020 ms. All samples are retained.

Both cards passed 26 bit-pattern cases, 96 timing-layer checks, two graph cases and exact compiler pins. Across both cards that is 52 cases, 192 timing-layer checks, four graph cases and 80 compiler artifact instances. Timing activations are reused across layers and T8 is a prefix of T16; the layer checks validate the workload rather than increase independent numerical coverage. The device API count was 32 on both cards, with four rows per norm block for T8 and T16.

Peak allocated/reserved bytes were 848,331,264 / 1,117,782,016 per card and allocation returned to zero. The component took 9.66 s and the caller 43.17 s. Configured 235 W cap observations bracket both card intervals; no energy or power-draw saving is established. Independent release is recorded at checkpoint 583. One invocation was consumed, with zero model starts and no retry.

F1 remains on the cumulative shortlist, conditional on representative activations, actual per-layer/both-rank production engagement, native TP2 reduction, C2/cancel/recurrent lifecycle and serving economics. No adoption or serving bundle is qualified. Do not add F1's percentages to operation 214's attention percentages. Serving remains HOLD_FOR_EVIDENCE.

Detailed F1 result (`RESULT.md`, private evidence) includes medians, raw ranges, every sample and release evidence. The approved owner decision (`PHASE2-NORM-QUANT-OWNER-DECISION-v1.md`, private evidence) and sealed packet remain unchanged. Their stale donor interpretation, and the retained summary's q/k/v wording, are descriptive artifacts; the authoritative experiment mode and numerical contract are F1 norm/quantizer.

## Combined compiler proof

The native CPU run reproduced all 80 artifact instances across four cases, both arms and both stages. The candidate stage1 is the reviewed prefetch binary; the reduction is the cap32 binary. Baseline reproduction completed before candidate compilation. No GPU devices were exposed, no compiled object was loaded and no kernel launched. Module and linecache restoration passed after normal and forced-exception paths.

The composition needs an explicit resource contract. The cap32-only comparator requires unchanged stage1 code and at most 77 scalar registers. Prefetch instead uses 100, as already reviewed in its own packet. The new contract binds exact constituent binaries and resources: baseline stage1 245 VGPR/77 SGPR, prefetch stage1 236/100, both reductions 13/33, zero spills/private scratch, and 16 KiB/zero dynamic LDS for stage1/stage2. The sealed split-only comparator is unchanged. This distinction must remain visible in the component decision.

Local verification passed eight composition checks, four recipe checks and the 43-check public preflight. The finite native compile exited cleanly in 8.96 seconds. These checks establish compiler compatibility, not combined numerical correctness or speed.

## Four-arm packet preparation

Opus signed off the CPU reconciliation and composed proof. The derived harness compares stock16, prefetch16, stock32 and prefetch32, so a combined gain cannot hide cap32's incremental effect. Prefetch must remain bitwise-neutral at both caps. NaN poisoning and exact active/empty partial-slot masks apply before every layer's correctness call.

The generated owner requires read-only 235 W power-cap readbacks before and after component work on both cards; these are configured limits, not measured draw or a continuous trace. No power setting is changed. Exact constituent resource checks, all-layer FP32 bounds and independent release remain required. Six mirrored timing rounds balance every pair's order. The original 10% screen stays reported; a separate cumulative rule retains positive primary medians on both cards only when the combination has no resolved regression against the baseline or either constituent. Positive signals inside noise are not proven gains.

The revised harness uses six fixed mirrored warmup rounds before each case's six measured rounds. Warmup values are excluded from scoring, with no adaptive discard. The projection charges the extra work within the unchanged limits. All 45 focused harness tests passed locally and on native Python; all 12 owner checks and three caller checks passed. The native CPU sandbox reproduced all 160 artifact instances. Kernel bytes and measured-sample rules are unchanged. The coarse planning estimate was 67.74 seconds per card against the 280-second limit; operation 214 subsequently completed both cards within that limit.

Both reviews are complete. Earlier refused preparations remain sealed. Only a timing-only REJECT of the combined arm, with all hard gates passed, opens the prefetch-only fallback; P/B must also have no resolved regression on any case and positive primary gaps on both cards. A missing positive cap32 increment on at least one card is reported but does not itself open that branch. A serving pilot is still held until authenticated request-wide route counts, control/work evidence and the conditional F1 disposition support the unchanged balanced serving threshold. A component pass alone will not spend the serving allowance.

The final owner decision (`PHASE2-VERIFY-COMPOSED-OWNER-DECISION-v1.md`, private evidence) binds packet `8cbb9eb079596044`, both Opus reviews, all seals and the final fallback clarification. Its one invocation was approved and consumed as operation 214. The packet and decision remain immutable; release is recorded at checkpoint 581.

## Proposed testing sequence

1. Completed: the approved four-arm component passed at 235 W as operation 214. Retain its USEFUL result and unresolved cap32 increment. The one-use allowance is consumed; do not repeat it.
2. Completed: F1 passed its hard component gates as operation 215 and remains RETAINED_POSITIVE. Its standalone NOT_USEFUL verdict stays unchanged. Preserve all timing samples; no retry. Production engagement, representative activations and full serving lifecycle remain separate qualification gates.
3. Apply the preregistered spending rule before proposing a fresh 235 W B0 control against the largest qualified compatible bundle. Preserve the selected model, precision, context, thinking profile and all existing quality/work/noise/regression gates. Cap3 remains an explicit comparison-runtime deviation in both arms, not a newly adopted optimization. Authenticate the production route on both ranks before requests.
4. Prepare a separately approved reverse-order confirmation only if the pilot passes. A null or failed result is retained. An ablation needs a specific observed interaction; no combination sweep is planned.

The planning reservation remains at most two component envelopes plus one serving A/B: 12,280 seconds. This is neither predicted duration nor authority, and it does not renew the expired four-hour campaign. Affected lifecycle checks are separately frozen and costed before adoption. Every physical packet still requires its exact approval.

## Complete lead reconciliation

| IDs | Area | Current disposition and next evidence |
| --- | --- | --- |
| L01, G2 | MXFP4 operand packing | Follow-up layouts remove shared-memory work but fail the frozen register gate. Parked with reviewed load-wait caveats. |
| L02, R1 | Recurrent-state materialization | Deferred pending a flush, generation and consumer protocol; no state update is moved. |
| L03, H2 | Host submission, graph dispatch and queues | Build source already contains the dispatch work. Historical draft-to-target submission is early; post-target/C2 exposure and effective settings remain unmeasured. |
| L04, H3 | TP communication, NUMA, helpers and cgroups | No placement change justified. Need actual rank/helper placement and exposed collective dependencies under load. |
| L05 | C4 int32 indices | Tested variant failed the unchanged compiler resource gate. Excluded. |
| L06 | C4 index-ahead | Tested variant failed the unchanged resource gate. Excluded. |
| L07 | C4 explicit address/layout work | Deferred pending exact lane ownership and proof of a removable exposed reload. |
| L08 | Verification partitioning | Composition USEFUL in operation 214; cap32's additional benefit over prefetch remains inside noise. Serving held. |
| L09 | Integer MXFP4 reduction | Whole-reduction association, exceptional scales and hardware equivalence remain unproved. |
| L10 | Other RDNA4 GEMM scheduling | No survivor. Need a new layout/lifetime mechanism or GEMM-specific wait evidence. |
| L11 | Prefill host metadata copies | CPU-feasible, but no exposed request cost; no standalone GPU run. |
| L12, D1 | Compact draft cache and vocabulary shortlist | Deferred pending selected C1/C2 coverage, graph-safe ownership and committed-token economics. |
| M1 | Complete speculative-round graph | Deferred pending an exposed boundary and complete TP/cancel/state lifecycle contract. Device-side gaps remain a hypothesis. |
| F1 | Gated norm plus MXFP4 quantization | Operation 215 CORRECT/RETAINED_POSITIVE: T8 inclusive gain resolved; T16 inclusive unresolved. Production qualification pending. |
| G3 | Lossless tile codec | CPU storage-only result; LZ4 weights do not shrink, Snappy has a format/data bound, and scale/entropy consumers remain deferred. |
| H4 | Native cache tree/session retention | Deferred pending authenticated node events and attributed host time. Cap3's A/A cache-hit evidence is not a speed comparison. |

Earlier EXP2, N32 GEMM, bit-decode, two-query attention and scale-prefetch variants remain excluded by their recorded resource failures. No correctness or lifecycle failure becomes eligible merely by joining a bundle. Already-selected C4 tiling, FP8 heads, packed projections, GDN copy fusion, P1 prefill and P3 norm remain in B0 and are not counted again.

The [canonical sprint register](../provenance/MXFP4-OPTIMIZATION-SPRINT-PLAN-20261103.json) holds current membership. The [preparation record](../provenance/MXFP4-OPT07-20261103.json) binds the sealed reconciliation and CPU compiler proof. No serving improvement is claimed for the composition until it is measured against its fresh control.

## Combined serving source preparation

The [combined preparation](MXFP4-OPT07-COMBINED-SERVING-PREPARATION.md) retains all three members and prepares private cap32 routing without changing native decode. Twenty focused CPU checks and the public preflight pass. Production attachment and the complete both-rank witness remain unimplemented. The reproducible economic sensitivity check retains HOLD_FOR_EVIDENCE. Opus 5.5 agreed to CPU signoff with the recurrent-branch scope correction recorded. The [request-level cost model and control-only observation spec](MXFP4-OPT07-REQUEST-COST-MODEL.md) have CPU signoff, with the reviewed input-validation and retention corrections applied and tested. The bundle model is expected to remain inconclusive; other leads must justify the shared diagnostic. No owner packet, GPU authority or serving gain follows from this preparation.
