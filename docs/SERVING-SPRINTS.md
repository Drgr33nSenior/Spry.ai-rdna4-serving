# Current serving sprint list

Updated 22 October 2026. This list reconciles the retained sprint evidence with
the Astra review and the owner decision to keep metadata reuse. Keep the current
Torch 2.14 / ROCm SDK 10.2 stack, official FP8 target weights, BF16 attention KV,
FP32 recurrent state and head, aligned GDN, graphs and DFlash2 block eight.
Keep [metadata reuse](METADATA-REUSE.md) and M2047 stage 1 active in the
current session candidate. The owner accepted session use on 21 October and
deferred further baseline-variant diagnosis unless concerning results recur.
The exact-output gate remains incomplete. See the
[current selection](../provenance/CURRENT-SESSION-SELECTION.json).

The pre-Sprint 7 optimization screens are retained in the
[screening list](OPTIMIZATION-CANDIDATES.md). The owner requested the next
optimization cycle on 22 October. Sprints 8–10 below implement the
[reviewed optimization plan](NEXT-OPTIMIZATION-PLAN.md).

The earlier sprint results apply to their recorded source and workload. They
do not qualify every feature in the current combined candidate. An item can
finish with a decision to park it; there is no need to repeat a slower screen.

| Sprint | Retained result or decision | Remaining work |
| --- | --- | --- |
| 0 — boundaries | Current-profile bounded cancellation and recovery passed in Sprint 7. | Maximum-capacity and endurance coverage remain outside this screen. |
| 1 — representative performance and quality | Short C1 benchmarks retained; both-profile natural-stop task and sustained-final-output screens passed. | Broader task coverage and statistically supported performance claims remain open. |
| 2 — exact prefix reuse | Current DFlash profiles passed cold/warm, branching, pressure and cancellation final-answer checks. Slot bookkeeping passed; whole output IDs matched 42/52 cache pairs. | Numerical/exact-output equivalence remains open under owner disposition. |
| 3 — memory and context | Aligned ReplaySSM component and production bridge passed. Both-order matched-capacity timing retained 1.63 GiB/card startup allocation saving; current lifecycle slots account for 19 usable/20 physical per rank. | Numerical-state equivalence and 32K/64K DFlash requests remain unqualified. |
| 4 — prefill | M2047 passed all 360 exact component comparisons on the current Torch 2.14 / SDK 10.2 stack. The geometric-mean kernel speedup was 1.43 times. | Both serving orders showed about 4.9 percent lower median latency. Owner selected session use; exact-output gate remains incomplete. Revisit the baseline variant if concerning results recur. |
| 5 — speculation | Keep block eight. The fresh current-stack screen with equivalent attention paths measured 1.8474 s at width eight versus 2.5137 s at width four. Keep metadata reuse. | Width outputs differed and task quality was not graded. Adaptive policy remains unimplemented and untested after its prerequisite audit. Current-profile metadata lifecycle coverage passed in the bounded Sprint 7 screen. |
| 6 — communication and host work | Earlier review selected no change. Draft selection already uses compact maxima and IDs. | Current C1 event screen measured ~0.157 ms gather within a ~43.9 ms target graph per rank. Retain existing settings; no shard-max implementation selected for this scope. |
| 7 — combined qualification | Bounded screens complete for snapshot and aligned ring-fold: matched capacity/both orders, greedy quality, 12272-token inputs, actual C2, reuse/branch/pressure, four cancellations/repeated starts and 8/8 publisher-sampled answers. | Exact-output exception remains open. No general, distribution or production qualification. See closeout. |
| 8 — cost attribution | Complete. Corrected packet passed 107 CPU checks, two starts,8/8 final answers and independent release;126 completed samples/card. Target graph spans about 83% of sampled long-request cycles. | Retain missing windows, reasoning variation and the unresolved numerical contract. See the attribution record. |
| 9 — one measured optimization | [Complete](SPRINT9-TARGET-GRAPH.md). Classification and native M8 gate/up component gate passed. | Stage 1 passed 96 exact component comparisons and cut synthetic component median time 6.61–13.06%; Sprint 10 subsequently completed matched serving validation. |
| 10 — matched serving validation | [Complete](SPRINT10-M8-SERVING.md): 0.35–0.60% lower median latency in both orders; greedy 8/8, sampled 8/8, lifecycle 52/52 plus cancellation/C2/slots. M8 stage one selected for the next experimental launch. | Full-output/state equivalence, broader coverage and endurance remain separate work. Installed runtime unchanged. |
| 11 — remaining projection tuning | [Bounded trials closed](SPRINT11-PROJECTION-TUNING.md). Five components passed 480 exact checks; four serving campaigns completed. Tile32 failed output/work parity; the 3072 candidate retained a positive timing signal but failed its predeclared confirmation work gate. | At that closeout all candidates were parked, with the reference at about 111.4 combined TP2 output tokens/s. Prepared B/B quality packets had not launched. See the subsequent experimental disposition below. |
| 12 — native unrolled projection | [Stage3 configuration parked](SPRINT12-UNROLLED-PROJECTION.md). Unrolled compilation requires 197,120 shared-memory bytes versus a 65,536-byte limit. The allocation guard stopped before GPU1; release verified. | No candidate parity or performance result. Another configuration requires a separate reviewed component. |
| 13 — lower-stage unrolled projection | [Component closed](SPRINT13-UNROLLED-STAGE1.md): 96 exact comparisons passed on both cards. Stage one fits at 16 KiB shared memory, but takes 47.08–52.21% more time than common stage three. | Park; no serving campaign or source change. Compiler reports 57 spills versus zero; a smaller M tile is a possible next component. |
| 14 — smaller unrolled tile | [Component passed](SPRINT14-UNROLLED-TILE16.md): 96 exact checks; tile16/stage1 has zero reported spills and 25.41–29.84% lower component time. | Both serving orders passed output/work gates at 1.74–1.79% lower median latency and 115.12–115.15 combined TP2 output tokens/s. Greedy8/8, sampled8/8, lifecycle52/52 plus cancellations/C2/slots passed. Astra approved next experimental launch; prior numerical findings and S11 failed gates remain. |
| 15 — smaller unrolled gate/up tile | [Closed and parked](SPRINT15-UNROLLED-GATEUP.md): 96 exact component checks pass; serving outputs/work match in both orders. | Latency −0.0494% forward, +0.1316% reverse: no repeatable serving gain. Keep Sprint 14; refresh the short cost profile before more tuning. |
| 16 — selected-profile cost refresh | [Complete](SPRINT16-CURRENT-PROFILE.md): 8/8 final answers, 4/4 full output pairs, eight complete windows per rank. | Projection methods about 29.2 ms/46.5 ms target graph; next proposed component is GDN qkvz `(8,8192,5120)`. No campaign prepared; Sprint 14 retained. |
| 17 — GDN input projection | [Complete](SPRINT17-GDN-INPUT.md): 0.54–0.64% less latency, 115.71–115.73 combined tokens/s; all eight output/work pairs match. | Greedy8/8, sampled8/8, lifecycle52/52 plus cancellations, actual C2, slots/release; Astra approved next experimental launch. Preserve Sprint 14 rollback and numerical limits. |
| 18 — attention QKV projection | [Closed, option unselected](SPRINT18-ATTENTION-QKV.md): 96 exact component checks; both serving median gates pass; greedy8/8 final answers. | Median latency −0.1948–0.2237%, pooled mean +0.0574%; retain timing outlier and reasoning variation. Astra recommends keeping Sprint 17. Sampled/lifecycle preparations remain unlaunched. |
| 19 — aligned ring-fold with current kernels | [Both serving orders complete](SPRINT19-MATCHED-RINGFOLD.md): all eight output/work pairs match; median latency falls 3.83348–3.87926%, with 1.62558 GiB less startup allocation per card at matched capacity. | Greedy and sampled 8/8, lifecycle 52/52 final answers and two cancellations pass, with C2/slots/release. Lifecycle full outputs match 24/26 pairs. Astra supports ring-fold selection for the next experimental session only; paired snapshot control is retained for rollback. |
| 20 — selected-profile classifier refresh | [Complete](SPRINT20-CURRENT-PROFILE.md): 8/8 final answers, 4/4 full-output pairs and eight complete windows per rank. | Gate/up forward median 11.932–11.939 ms led to Sprint 21. |
| 21 — common M16 gate/up | [Component and serving orders complete](SPRINT21-COMMON-M16.md): 96 exact component comparisons and 8/8 matched output/work pairs; median latency falls 0.38064% and 0.28794%. | Candidate 120.304/120.368 combined TP2 tokens/s, with one excluded warmup and three measured requests per start; pooled mean latency −0.29775% descriptive. Greedy and sampled repeated-candidate screens each passed 8/8 finals and 4/4 full-output pairs with release; the sampled route was verified, not its distribution. Lifecycle passed 52/52 finals, two cancellations, C2, slots and release; 24/26 full cache outputs and 26/26 final texts matched. M16 is selected for the next experimental session only, with prior GEMM rollback and ring-fold unchanged. [Provenance](../provenance/SPRINT21-COMMON-M16-20261022.json). |
| 22 — common M16 GDN output | [Complete and selected for the next experimental session](SPRINT22-GDN-OUTPUT-M16.md): 96 exact component checks; both serving orders match all eight output/work pairs at 0.814–0.892% lower median latency and 121.175–121.243 combined TP2 tokens/s. | Greedy and fresh sampled screens each passed 8/8 answers and 4/4 full-output pairs. Lifecycle passed 52/52 normal answers, two cancellations, actual C2, slots and release; 24/26 full cache outputs and 26/26 final texts match, retaining the same index-149 variations as Sprint 21. Astra approved M16 GDN output with prior GEMM rollback and ring-fold unchanged. Preserve the initial sampled observation failure and read-only collection recovery; no numerical resolution or endurance qualification claimed. |
| 23 — common M16 attention QKV | [Closed, candidate parked](SPRINT23-ATTENTION-COMMON-M16.md): 96 exact comparisons passed at unchanged stage three; three cases faster, GPU 0 random 2.60% slower. | The all-four-cases timing gate failed. One model-free attempt, independent release, zero model starts and no serving or quality runs. Sprint 22 GEMM and ring-fold remain selected; propose a short cost refresh before another tuning candidate. |
| 24 — selected-profile cost refresh | [Complete with retained output variations](SPRINT24-CURRENT-PROFILE.md): 8/8 natural final answers and eight selected timing windows per rank; projection forwards about 27.2 ms within a 44.1 ms target graph. | Gate/up remains largest at 11.84 ms per rank. Full outputs/work match 1/4, final text/IDs 3/4; no overhead or speedup claim for unmatched work. Independent release passed; selected profile unchanged. Propose one N64 gate/up component, retaining M16/stage1; uncompiled and not queued. |
| 25 — common gate/up N64 | [Complete and parked](SPRINT25-GATEUP-N64.md): 96 exact component comparisons; all four medians 0.540–0.980% lower. Forward serving matches 4/4 output/work pairs, but median latency is 0.04241% higher. | Required improvement gate failed; no reverse, retry or candidate quality screens. Independent release passed; selected GEMM and ring-fold unchanged. Native sampled acceptance uses default-device RNG draws whose per-request state was not recorded; cause and earlier numerical finding remain unresolved. At closeout, bounded RNG/input evidence was proposed. |
| 26 — sampled acceptance investigation | [Bounded comparisons complete](SPRINT26-SAMPLER-STATE.md): two full traces, 8,328 observed events and 8/8 matched output/work pairs. | Cross-phase review found one differing start of eight; its values and cause remain unresolved. Synchronous instrumentation can perturb timing. No causal, numerical-equivalence or performance claim. Selected profile unchanged; no GPU campaign queued. |
| 27 — gate/up cost isolation | [Complete; bounded review approved](SPRINT27-GATEUP-COST.md): 8/8 answers, 4/4 matching full-output/work pairs and eight windows per rank. | GEMM dominates four sampled gate/up sites; propose a stage-two component for the current shape. No source selection, bandwidth-saturation or performance claim. |
| 28 — gate/up stage two | [Component parked; review approved](SPRINT28-GATEUP-STAGE2.md): 96 exact comparisons passed, but two of four timing cases were slower. | No serving preparation, retry or source selection. Sprint 29 completed CPU examination of compiler artifacts and scheduling. |
| 29 — gate/up compiler review | [CPU review complete; proposal reviewed](SPRINT29-COMPILER-REVIEW.md): retained stage-one/stage-two artifacts examined, with an exact-shape scalar specialization proposed. | No candidate implementation, compile, GPU work or speedup claim; selected profile unchanged. |
| 30 — gate/up static scalars | [Component parked; review approved](SPRINT30-GATEUP-STATIC-SCALARS.md): 96 exact comparisons and four compiler proofs passed, but one timing case was slower. | No serving preparation, retry or source selection. A bounded warp-count component is a possible later lead after source and resource review. |

The owner subsequently requested [experimental 3072 selection](M8-3072-EXPERIMENTAL.md).
Its task/cache screens passed: greedy 8/8, sampled 8/8 and lifecycle 52/52 plus
two cancellations, actual C2 and slot accounting. Astra approved selection for
the next experimental launch. The descriptive candidate rate is 113.09–113.27
combined TP2 output tokens/s. Sprint 11's failed matched-work results and the
unknown variation cause remain unchanged; installed runtime is unchanged.

Sprint 7's planned bounded screens are complete. Its ordered gates, outcomes
and limits are in the [qualification plan](SPRINT7-QUALIFICATION.md) and
[closeout](../provenance/SPRINT7-CLOSEOUT-20261021.json). The exact-output
exception remains recorded under the owner's experimental-session disposition.

## ReplaySSM source decision

The existing speculative ReplaySSM path removes the full intermediate SSM
snapshot allocation, historically 1.69 GiB per rank. It adds rings and residual
buffers; net savings have not been measured for this candidate.

The pinned current backend dispatches replay verification to a separate kernel
instead of the aligned recurrent verifier. The replay kernel computes beta as
an unrounded FP32 sigmoid expression. The selected recurrent verifier rounds
beta through BF16 on this path. Replay verification also uses cumulative gates,
dot products and chunked reconstruction; compact commit reconstructs state
from the stored updates. These operations do not preserve the sequential
arithmetic merely because the checkpoint stays FP32.

Park direct flag enablement under the current arithmetic contract. This is a
source compatibility decision, not an observed native failure or a finding
that ReplaySSM cannot work. A revised candidate needs native output and state
checks for every accepted prefix, rejection, ring wrap, slot reuse, radix
tracking and graph replay before a memory or timing comparison.

Source identities and the audit result are in
[`REPLAYSSM-SOURCE-AUDIT.json`](../provenance/REPLAYSSM-SOURCE-AUDIT.json).

The later isolated candidate uses aligned sequential raw-ring recording and the
production all-layer fold, with explicit physical-slot sizing and memory
accounting. It passed the component bridge and bounded C1 serving screen.
This does not enable the original compact path or replace the installed
runtime. Carry the revised profile into the broader Sprint 7 checks recorded
in the [screening list](OPTIMIZATION-CANDIDATES.md).

## M2047 prefill tail: current-stack component pass

The source extension adds M2047 to the existing five-pair HIP/BF16 one-stage
guard. The native kernel, scale policy, tail mask and other dispatch guards
stay unchanged. The current baseline and candidate have the same source
identities as the earlier component comparison. CPU checks execute the actual
host selection function with declared substitutes; they do not compile or
time a GPU kernel.

The historical synthetic kernel speedup was 1.358–1.590 times, with 20 cases
and 360 exact comparisons across both GPUs. This is not a serving speedup.
The retained shape-frequency evidence predates the current combined stack.

The fresh current-stack component check passed all 360 exact comparisons across
20 cases, with 320 timed launches and 40 warm-ups. Stage 1 was 1.311–1.575 times
faster than stage 3; the geometric mean was 1.429. Both GPUs were tested
sequentially with synthetic inputs. The owned run took 47.81 seconds, including
runtime verification, numerical checks and observed release. It started no
models. Independent release and both evidence seals verified. The existing
wrapper's Python 3.14 regex SyntaxWarning remains in the retained logs.

See the [current-stack component record](../provenance/M2047-COMPONENT-20261021.json)
for shapes, timings, compiler resources, memory and evidence identities.

Both matched serving orders are complete. Median request times were 1.9487 s
(stage 3) versus 1.8536 s (stage 1) in forward order, and 1.9454 s versus
1.8501 s in reverse order. All five changed shapes reached both ranks. Rates
are combined TP2 figures. The workload used 2,047 input tokens and 128 forced
output tokens, C1, low reasoning, identical pools and DFlash block eight.

Seven of eight paired output prefixes matched, including five of six measured
pairs. Candidate outputs matched across all eight requests. One reverse-order
baseline response first differed at output index 102 and used 23 verification
cycles instead of 25. Keep its timing and response in the record. The exact
output gate remains incomplete; the cause is unknown. Aggregate counters do
not distinguish numerical variation from a state or cache defect.

The owner selected the [source patch](../patches/sglang-gfx1201-prefill-stages.patch)
for current-session use after the repeatability screen. Keep the unresolved
variant in the record and revisit it if concerning results recur. This selection
does not mark the exact-output gate or Sprint 7 complete. See the
[serving record](../provenance/M2047-SERVING-20261021.json) for every sample,
shape witnesses and evidence identities. No installed runtime changed.

## Baseline repeatability follow-up

Two fresh starts used the identical stage-3 baseline and the original four
requests per start. All eight response prefixes matched the historical common
output. Each request used 25 verification graph replays per rank, with zero
recaptures. Speculative histograms agreed with the reported counters. Native
source, runtime, weight, HTTP and independent release checks passed. No new
hooks, flush intervention or stack change was introduced.

The earlier variation was not reproduced in this finite screen. Across the
original baseline requests and this follow-up, 15 of 16 matched the common
output. This does not establish determinism or identify the cause of the
remaining variant. The M2047 exact-output gate stays incomplete. The owner
selected current-session use and deferred further diagnosis. If concerning
results recur, the proposed diagnostic is a bounded host trace at
`Scheduler.process_batch_result`, with
request output-ID and verification-counter deltas. It must delegate the
native method unchanged and avoid GPU tensor copies. That trace has not been
implemented or run. See the
[repeatability record](../provenance/BASELINE-REPEATABILITY-20261021.json).

## Current-session activation and Sprint 6 census

The owner selected M2047 stage 1 for current-session runs. Two fresh starts
used it in both logical arms with metadata reuse and the current precision,
stack, graphs and pools. Both ranks exercised all five changed M2047 shapes.
The 28 CPU acceptance checks passed. All eight native requests completed, with
25 verification replays per rank and zero recaptures in each request. Source,
runtime, weights, wire and independent release checks passed. The finite
workload was released; no persistent server or installed default was replaced.

Seven outputs matched the historical common output exactly. One response
differed only at output index 124: `Required` versus `required`. The record
retains the exact mismatch; case-insensitive text equality is descriptive, not
a numerical pass. Keep M2047 selected under the owner's session decision and
revisit if concerning results recur. The exact-output gate stays incomplete.

Sprint 6 now has native startup gather geometry on both ranks. A C1
verification block has an FP32 `[8, 124160]` local shard, 3,973,120 bytes
(3.79 MiB), and an `[8, 248320]` result, 7,946,240 bytes (7.58 MiB) per rank.
The pinned multimem helper used its native FP32 TP-all-gather fallback. The
observer delegated the native call once and restored before traffic. No extra
tensor operations, model forwards or synchronizations were added. Startup
bucket-two evidence does not qualify concurrent C2 traffic.

These tensor sizes do not measure link traffic or exposed latency. Retain the
Sprint 6 no-tuning decision; implement target shard-max only if attribution
justifies the change and its full-logit consumers and greedy semantics are
covered. Complete the remaining optimization screens before selecting the configurations
for Sprint 7. Keep M2047 enabled in the session reference.
See the [session and census record](../provenance/SESSION-M2047-SPRINT6-20261021.json).

## Sprint 6 current C1 timing screen

The [head/gather timing screen](OPTIMIZATION-CANDIDATES.md) is complete.
Each rank retained 99 external-request cycle intervals, with gather medians
around 0.157 ms inside a 43.9 ms target graph. Four control/profile output pairs
matched. No extra model forwards or per-step synchronizations were added.
Keep existing communication settings for this scope. Complete the remaining
ReplaySSM serving integration separately from the reference profile's Sprint 7
qualification. The width comparison and adaptive prerequisite audit are now
recorded in the [optimization screening list](OPTIMIZATION-CANDIDATES.md).

## Optional: FP8 target attention KV

Status: deferred optional work, outside the current Sprint 7 qualification path.
The owner chose to keep the tested configuration and avoid broad changes now.
The model weights remain FP8; target and draft attention caches remain BF16,
with `--speculative-draft-kv-cache-dtype bf16` explicit.

This separate work item would compare BF16 and FP8 target attention KV while
keeping the drafter BF16 and the selected snapshot reference otherwise matched.
It needs supported calibration and scale loading, FP8/non-unit-scale support
in the split adapter, and a reviewed schedule and evidence layout for four
fresh starts. The native FP8 path also changes internal query arithmetic, so
any result must disclose the complete supported cache path.

The feasibility estimate is 90–150 minutes of CPU implementation and review,
not a launch-ready flag change. Replan and budget this item separately if the
owner chooses to pursue it. Retain the existing feasibility evidence and
matched workload, correctness, timing, capacity and release requirements. No
implementation pass or GPU campaign is scheduled for this optional item.

The bounded Sprint 7 screens completed with BF16 target/draft caches. FP8
target KV remains a separate deferred option.

## Optimization sprints 8–10

Sprint 8 has three bounded milestones. See the
[attribution preparation record](SPRINT8-COST-ATTRIBUTION.md) for source and test evidence.

1. **8.1 — host attribution groundwork:** complete on CPU. Add the frozen
   request/cycle bindings and an optional host recorder to a fresh private
   sampling observer. Keep its validation, authenticated delegate and normal
   installation unchanged. Retain at most 128 rows per rank across the four
   requests; the model-call bound remains 36864.
2. **8.2 — device integration and packet preparation:** CPU preparation complete. The packet installs graph
   events before capture, authenticates their replay generations, adds eager
   acceptance/commit intervals and joins host rows to actual verifier cycles.
   Enabled startup/cleanup fixtures, exact package/binding limits and independent
   implementation/packet review passed. The subsequent native failure and
   correction are recorded in milestone 8.3.
3. **8.3 — native attribution and decision:** complete after the event-device fix.
   The fresh control/timing packet passed 8/8 final answers and independent
   release, with 126 completed samples/card. The target graph is the largest
   measured interval. Coverage gaps and one reasoning-output difference remain
   explicit; the screen establishes no serving speedup.

Sprint 8 exits with enough authenticated timing to select one exposed cost, or
an explicit finding that attribution is incomplete and a bounded follow-up is
needed. It does not exit merely because a profiler ran.

Sprint 9 follows the target-graph branch selected by the
[Sprint 8 attribution](SPRINT8-COST-ATTRIBUTION.md). First check installed trace
compatibility and prepare a bounded eight-cycle trace on both ranks, or selective
events if needed. Classify projections/quantization, attention/GDN and layer
communication before selecting one component. Preserve the numerical contract;
an approximation or changed tie policy requires a separate owner decision.
Native component parity and a measured potential gain are its exit criteria.

Sprint 10 separates fixed output timing from natural-stop quality. A component
win must survive matched serving and reverse order, plus the affected request,
cache and cancellation checks. A small reproducible gain is acceptable; there
is no five-percent floor. Retain failures and unresolved baseline variation.

Keep the current Torch/SDK, FP8 weights, BF16 target/draft caches, thinking,
M2047, metadata reuse, width eight, flags and pools throughout the first packet.
Do not fold FP8 target KV, adaptive speculation or C2 throughput tuning into it.

## Packaging and endurance follow-up

Owner direction, 22 October: defer reliability and endurance testing until the
tuning process is exhausted. Continue Sprint 11 first; retain the bounded
correctness gates for each candidate.

A reproducible serving package and longer mixed-workload/endurance checks remain
separate work before deployment. The repository remains a source prototype;
these bounded profile screens do not supply a complete installable runtime or
production qualification. Define that scope and its acceptance criteria separately.

Keep the unresolved exact-output variation visible under the owner's current
experimental-session disposition. General numerical equivalence remains a
separate open question. Reopen diagnosis if concerning outputs or runtime
faults recur. FP8 target attention KV remains optional and deferred. Adaptive
speculation and target shard-max also remain separate unimplemented candidates;
they are not required follow-up kernel changes merely because Sprint 7 ends.

Sprint 31: [W2 and W8 parked](SPRINT31-GATEUP-WARPS.md). W2 failed its serving median gate; independent review approved the closeout. Sprint 32 subsequently completed a bounded 32K screen with owner-accepted lexical exceptions and a retained strict quality failure. A separately reviewed 64K screen followed in Sprint 33. UltraQuant is parked and the BF16-versus-FP8 weight comparison is deferred; optional FP8 target-KV and adaptive speculation feasibility follow later, with endurance after selection.

Sprint 32: [32K long-context screen](SPRINT32-LONG-CONTEXT.md). All 14 requests stopped normally; strict SRE grading passed 2/12, with owner-accepted lexical exceptions and a retained retrospective audit. The 64K adapter preflight and a separate device screen followed in Sprint 33.

Sprint 33: [64K long-context screen](SPRINT33-LONG-CONTEXT64.md). All 14 responses stopped normally; both frozen SRE gates scored 10/12 and failed, with two retained format failures. Independent review approved the conservative closeout.
