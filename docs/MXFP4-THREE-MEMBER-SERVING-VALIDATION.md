# Three-member cap16 serving validation

Latest result, 6 November: operation232 independently passes the final
unprofiled A/B. Result review `dc14aa7b…` accepts the frozen KEEP classification,
quality, matched work and release with no blockers. Its case-balanced latency
score gain is 3.810953%; this is not aggregate throughput. The matched
profiling requirement was completed by operation229 and its reviewed CPU
successor. The owner selected this tested mtp arm, retained control as rollback
and parked the programme. No further GPU work is authorized; see the
[Qwen4 handoff](QWEN4-SERVING-HANDOFF.md).

The operation226/227 evidence below records the earlier confirmation stage.

Operation227 passed the reversed-order confirmation at the configured 235 W cap. The balanced score
rose by **3.7394%**, above the unchanged 3% threshold. The frozen readout returns
**CONFIRMED_KEEP_REVERSED_ORDER**, accepted by independent result review from
raw native receipts.
Operation226's control-first A/B independently passed at **3.2652%**, accepted by
independent review as ELIGIBLE_FOR_FINAL_CONFIRMATION. No pooled verdict is used.
At that stage selection and adoption were unchanged, with matched profiling
still required. The latest result above supersedes that pending step.

The candidate combines verification prefetch, gated-norm/MXFP4 quantizer fusion
and gate/up M8/M16 wave0. Both arms use private verification cap16; ordinary decode
also stays cap16. Cap32, Down-M16, BW-03 and a fifth member are excluded. The
experiment preserves TP2, precision, context and usable KV/recurrent capacity.
Both arms share the disclosed cap3 trimming and LLVM_PATH deviations.

## Reversed confirmation: operation227

The candidate ran first, through physical label `bf16`; the control ran second,
through `kv4`. Role-aware grading and timing projection authenticate that mapping.

| Case | Baseline median (s) | Candidate median (s) | Less request latency | Resolution |
| --- | ---: | ---: | ---: | --- |
| Long C1 cold | 83.108 | 81.983 | +1.35% | Within noise |
| Long C1 warm | 9.597 | 9.097 | +5.21% | Resolved gain |
| Long C2 cold | 61.174 | 60.520 | +1.07% | Within noise |
| Long C2 warm | 8.814 | 8.406 | +4.63% | Resolved gain |
| Short C1 | 13.683 | 12.939 | +5.44% | Resolved gain |
| Short C2 | 14.778 | 14.211 | +3.84% | Resolved gain |

## Control-first screen: operation226

| Case | Baseline median (s) | Candidate median (s) | Less request latency | Resolution |
| --- | ---: | ---: | ---: | --- |
| Long C1 cold | 82.383 | 82.540 | -0.19% | Within noise |
| Long C1 warm | 9.599 | 9.115 | +5.04% | Resolved gain |
| Long C2 cold | 60.872 | 60.714 | +0.26% | Within noise |
| Long C2 warm | 8.826 | 8.423 | +4.57% | Resolved gain |
| Short C1 | 13.660 | 12.972 | +5.04% | Resolved gain |
| Short C2 | 14.777 | 14.170 | +4.11% | Resolved gain |

Each median uses three scored repeats per arm. C2 request latency averages its
two slots within each repeat. The balanced score uses request-latency ratios;
group-wall time supplies noise and regression vetoes only. It is not an isolated
decode-throughput or kernel-speed figure. No case
has a resolved regression under the unchanged comparator.

The independent review distinguishes the balanced objective from aggregate
throughput: operation227's total measured group-wall time fell about 2.26%, with
57.22 to 58.54 output tokens/s over that menu. Resolved warm/short gains hold in
both run orders; long-cold changes remain within noise and depend on order.
Even with both long-cold ratios set to zero gain, the review's sensitivity
calculation gives a 3.3186% balanced gain. That calculation is report-only and
does not replace the frozen comparison.

The sealed readout README, chart and producing script incorrectly say group-wall
time enters the score. This paragraph supersedes that wording without changing
the immutable artifacts. The reader's `fastest_fully_passing` field is an
eligibility label under the old rule, not a final selection.

## Correctness and release

- All 30 control streams and speculative counters reproduce the retained op224
  control reference.
- All 30 same-job candidate streams match their control token IDs, reasoning/final
  counts, verify rounds, accepted and proposed drafts, and acceptance histograms.
- All 60 natural-stop quality grades pass, including long-answer coverage.
- The declared numerical, graph/eager, C2 recovery, capacity, retention and
  independent-release gates pass. Bounded probes cover 48 F1 layers and 64 GEMM
  routes on each rank. Both ranks report device API count 32; no T8 fallback occurs.
- The existing observers read the configured 235 W limit on both cards.
  Checkpoint608 independently releases operation226; checkpoint610 releases
  operation227. Both invocations are consumed and parked, with no remaining
  allowance or retry. The finalization campaign itself is still in progress.

The protected operation's four-member/private-cap32 labels are inherited schema
descriptions. The pinned package and observed routes require three members at
cap16. A separate scope correction is retained without editing protected records.

## Remaining work at operation227 (historical)

This section records the earlier plan. Operation229 and its reviewed successor
now satisfy the diagnostic requirement under the limits in the Qwen4 handoff.

Independent review accepts operation227. Its successful finite readout checks
the role-specific compiler records, numerical witnesses, token/speculative work,
quality and independent release. The readout's legacy next-action text mentions
selection; the owner's later instruction requires full matched profiling first.

The separate matched baseline/candidate profiling packet must close timestamp
calibration, full graph membership and candidate capture on both ranks, then bind
the runtime, available counters, overhead, interpretation and finite limits.
It needs its own exact GPU decision. Instrumented latency is diagnostic; use the
unprofiled results for final selection. Review the diagnostic before choosing
further optimisation work, then document, select and park through the existing
workflow and finish the Qwen4 handoff.

The reversed preparation passed 93 canonical-location tests, public43 and one
device-free native Python 3.14.4 run of 129 tests, with 1188 input records matched.
Its registrar proof refused 58 mutations. The owner approved the exact packet and
protected-helper exception; fresh preflight passed before the single invocation.
No preparation record or approval can be reused for another run.

These results do not establish universal F1 equivalence, an individual member's
share of the gain, energy savings or production qualification. The resource
samples do not establish DRAM throughput, instruction stalls or their cause.
The owner used the default `amd-smi monitor` display. Its installed source maps
`MEM%` to `average_umc_activity`, an activity percentage rather than physical GB/s.
Operation227's bounded 10 s sampler retained 149 rows with no failed, malformed
or dropped rows. Both cards have 143 readable observations and six EBUSY reads.
Whole-run GPU-busy medians are 100%; sysfs `mem_busy_percent` medians are 16% and
15%. That sysfs value is not asserted equal to AMD-SMI's field. Observations span
loading, warmup and measured traffic; they are not request-aligned attribution.
Raw gpu_metrics records are 120 bytes with header `78000103`; no fields or
structure revision have been qualified. Separate power reads include spikes;
they do not establish energy use or a throttle cause. These data remain
report-only and do not replace the full profiling step.

The [finalization record](../provenance/MXFP4-FINALIZATION-20261105.json) binds the
sealed execution, parent readout, original native evidence and checkpoint.
Operation225's hostname refusal, operation224's INCONCLUSIVE verdict and
operation217's NOT_COMPARABLE verdict remain unchanged.
