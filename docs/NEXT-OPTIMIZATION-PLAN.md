# Further serving optimization plan

Progress: [Sprint 8 attribution](SPRINT8-COST-ATTRIBUTION.md) and
[Sprint 9.1 classification](SPRINT9-TARGET-GRAPH.md) are complete. The native
classification passed all eight final-answer gates and independent release.
Projection methods dominate its measured target windows; MLP gate/up projections
are the largest subgroup. Sprint 9.2 passed 96 exact comparisons on both cards with 6.61–13.06% lower
component median time. [Sprint 10](SPRINT10-M8-SERVING.md) completed both serving
orders and affected quality/lifecycle gates; M8 stage one is selected for the
next experimental profile launch.
The owner then deferred reliability testing until tuning is exhausted.
[Sprint 11](SPRINT11-PROJECTION-TUNING.md) closed five projection component
comparisons and four serving campaigns. At that closeout, all candidates were parked. The 3072
candidate retained a consistent timing signal but failed its predeclared
forward confirmation work gate; affected B/B quality packets had not launched.
The Sprint 10 reference measured about 111.4 combined TP2
output tokens/s.
The owner then directed [Sprint 12](SPRINT12-UNROLLED-PROJECTION.md): one
model-free common-versus-unrolled comparison at the exact M8 down shape,
with the selected stage depth and tiles unchanged.
That one attempt rejected the stage-three unrolled configuration: shared-memory
demand exceeded the hardware limit, and no candidate timing was produced.
Independent release passed; no retry followed. The owner then requested a
separate [3072 experimental assessment](M8-3072-EXPERIMENTAL.md). Its task and
cache screens passed all 68 normal answers, cancellations, actual C2 and slot
accounting. Astra approved the 3072 source for the next experimental launch.
Its 113.09–113.27 combined TP2 output tokens/s remains descriptive: preserve
the failed work gates and unknown variation cause.
The next owner-directed candidate, [Sprint 13](SPRINT13-UNROLLED-STAGE1.md),
completed one model-free comparison of common stage three against unrolled
stage one. All 96 exact checks passed, and stage one fits shared memory, but
it takes 47.08–52.21% more time. Park that configuration. Compiler-reported
spills suggest a smaller unrolled M tile as a possible next component; no
additional campaign is prepared or reserved.
The owner then directed [Sprint 14](SPRINT14-UNROLLED-TILE16.md), reducing the
unrolled M tile to 16. It passed 96 exact checks and reduced component time
25.41–29.84%, with zero compiler-reported spills. Both serving orders passed
all output/work gates and reduced median latency 1.74–1.79%, reaching
115.12–115.15 combined TP2 output tokens/s. Greedy8/8, sampled8/8 and
lifecycle52/52 final answers passed, with cancellation, actual C2 and slot/release
checks. Astra approved the source for the next experimental launch. Earlier
numerical findings and Sprint 11 failed gates remain unchanged; no further
GPU campaign is prepared.
The owner then directed [Sprint 15](SPRINT15-UNROLLED-GATEUP.md): one
exact gate/up comparison of common tile64/stage1 against unrolled tile16/stage1.
The component passed 96 exact checks with 1.22–2.62% lower median time. Both
serving orders passed output/work gates, but latency changed −0.0494% forward
and +0.1316% in reverse. Park the candidate; Sprint 14 remains selected. Refresh
one bounded short cost profile on that selected source before choosing another
tuning lever.
[Sprint 16](SPRINT16-CURRENT-PROFILE.md) completed the selected-source classifier
refresh: 8/8 final answers, 4/4 full output pairs and eight complete windows per
rank. Long-window projection methods total about 29.2 ms within a 46.5 ms target
graph on each card. GDN input qkvz takes about 4.57 ms and has no selected M8
override. Propose one exact `(8,8192,5120)` component comparing common tile64/stage3
with unrolled tile16/stage1. No next campaign is prepared or reserved. Retain the
6.5–7.6% observed instrumentation cost, uncovered windows and numerical limits.
The original planning direction below remains the scope reference.

[Sprint 17](SPRINT17-GDN-INPUT.md) tested that exact qkvz shape: all 96 component
comparisons passed with 17.0–20.9% less kernel time. Both serving orders matched
all eight output/work pairs and reduced median latency 0.54–0.64%, reaching
115.71–115.73 combined TP2 output tokens/s. Greedy8/8, sampled8/8 and
lifecycle52/52 final answers passed, plus cancellations, actual C2, slot accounting
and release. Astra approved qkvz tile16/stage1 for the next experimental launch
after verifying all 18 evidence roots and exact patch application. Retain Sprint
14 rollback, numerical findings and deferred reliability. No GPU campaign is queued.

The owner next directed [Sprint 18](SPRINT18-ATTENTION-QKV.md): one attention
QKV `(8,7168,5120)` component comparing common tile64/stage3 with unrolled
tile16/stage1 against the selected Sprint 17 source. Preparation passed 43 CPU
checks and the bounded parameter review. The owner reaffirmed the existing
scoped sudo procedure. The component passed 96 exact checks and reduced kernel
time 16.25–18.25% across all four cases. The Astra-reviewed forward pilot passed
all output/work gates at 0.2237% lower median latency; reverse order passed at
0.1948% lower median latency. Retain one slower measured candidate request.
The pooled mean rose 0.0574%. Astra recommended leaving the option unselected.
Greedy completed 8/8 final answers with 4/4 final texts and 3/4 full output pairs matching;
retain the reasoning variant. Sampled/lifecycle preparations remain unlaunched.
Sprint 17 remained selected at the Sprint 18 closeout.

[Sprint 19](SPRINT19-MATCHED-RINGFOLD.md) completed both matched serving orders:
all eight output/work pairs match, median latency fell 3.87926% forward and
3.83348% in reverse, and startup allocation fell 1.62558 GiB per card at equal
capacity. Greedy and publisher-sampled quality each passed 8/8 final answers;
sampled full outputs matched two of four pairs. Lifecycle passed 52/52 normal
final answers, both cancellations, C2, slots and release; 24/26 cache full-ID
pairs and all final-text pairs match. Astra final review supports selecting
aligned ring-fold for the next experimental session only; the Sprint 17 GEMM is
unchanged and the paired snapshot control remains available for rollback.
Earlier numerical findings, Sprint 11 failed gates and the owner experimental-
session disposition remain; endurance stays deferred.

[Sprint 20](SPRINT20-CURRENT-PROFILE.md) completed its two-start,
eight-request classifier refresh on the selected GEMM and ring-fold profile:
8/8 final answers, 4/4 full-output pairs, eight complete windows per rank and
independent release. Gate/up forward medians were 11.939/11.932 ms on ranks
0/1. These nested intervals include quantization and waits, not isolated GEMM
time. [Sprint 21](SPRINT21-COMMON-M16.md) then passed 96 exact component
comparisons for common M16/stage1 gate/up `(8,17408,5120)`. Both matched serving
orders passed 8/8 output/work pairs and reduced median latency by 0.38064% and
0.28794%; candidate rates were 120.304 and 120.368 combined TP2 tokens/s, with one
excluded warmup and three measured requests per start. The
pooled mean latency change was −0.29775%, descriptively. Greedy and sampled repeated-candidate
screens each passed 8/8 final answers and 4/4 full-output pairs, with independent
release. The sampled route was verified, but its distribution was not. The
lifecycle screen passed 52/52 final answers, two cancellations, actual C2, slots
and release. Cache full outputs matched 24/26 pairs and all 26 final texts matched.
Astra selected the M16 GEMM for the next experimental session only, retaining the
previous GEMM for rollback and ring-fold unchanged. Endurance remains deferred. See the [Sprint 21 provenance](../provenance/SPRINT21-COMMON-M16-20261022.json).

[Sprint 22](SPRINT22-GDN-OUTPUT-M16.md) tested common M16/stage1 at GDN output
`(8,5120,3072)`. All 96 component checks and eight serving output/work pairs
passed. Both serving orders reduced median latency by 0.814–0.892%, reaching
121.175–121.243 combined TP2 tokens/s; the pooled mean latency fell 0.96771%,
descriptively. Greedy passed all eight answers and four full-output pairs.
The initial sampled campaign stopped during second-arm process-identity
observation; its four first-arm HTTP responses remain ungraded. The owner then
authorized completion with diagnostic messages added to existing refusals,
preserving all identity decisions and read ordering. Fresh sampled validation
passed 8/8 answers and 4/4 full-output pairs. Lifecycle passed 52/52 normal
answers, two cancellations, actual C2, slots and release. The same two cache
reasoning variations at index 149 remain; all 26 final-text pairs match.
A timed-out evidence transfer was recovered through a separately reviewed,
bounded compressed collection, with no GPU rerun. Astra approved M16 GDN output
for the next experimental session, keeping the previous GEMM for rollback and
ring-fold unchanged. The initial observation cause and earlier numerical
findings remain unresolved. Endurance stays deferred; no further GPU campaign
is queued.

[Sprint 23](SPRINT23-ATTENTION-COMMON-M16.md) compared common M64 and M16 at
attention QKV `(8,7168,5120)`, holding stage three fixed. All 96 component
comparisons passed. Three timing cases improved 1.64–8.03%, but GPU 0's random
case was 2.60% slower. The all-four-cases timing gate failed, so the candidate
is parked without serving or quality runs. Independent release passed after
one model-free attempt. The selected Sprint 22 GEMM and ring-fold remain
unchanged. At that closeout, a short cost refresh on the selected profile was
proposed before choosing another tuning lever.
Earlier numerical findings and deferred endurance remain.

[Sprint 24](SPRINT24-CURRENT-PROFILE.md) completed that short refresh on the
selected Sprint 22 GEMM and ring-fold. All eight natural final answers,
sampling-route checks and independent release passed. Each rank completed its
eight selected device windows, including four long-case windows. Projection
forward medians remain 27.234/27.206 ms within 44.111/44.115 ms target graphs;
gate/up remains largest at 11.839/11.845 ms. These inclusive, overlapping
intervals do not isolate GEMM time or establish whole-request fractions.
Full outputs and verification work matched only 1/4 pairs; final text and IDs
matched 3/4. Preserve the passing but different coding final and the ledger
reasoning variations. Unmatched work supports no overhead or speedup claim.

At the Sprint 24 closeout, the proposed component was common gate/up
`(8,17408,5120)` N128-to-N64, retaining
M16, K128, stage one, group 32, four warps and 128-wide block scales. Source
indexing supported the candidate; compilation, exactness and timing were
unverified at that point. Keep the selected profile and
earlier numerical limits; endurance remains deferred.

[Sprint 25](SPRINT25-GATEUP-N64.md) completed that component and one forward
serving pilot. All 96 component comparisons passed, with medians 0.540–0.980%
lower in all four cases. Serving output IDs and verification cycles matched
all four pairs, including warmup, but candidate median latency was 0.04241%
higher. The required lower-median gate failed. Park N64 with no reverse,
retry or candidate quality screens; retain the selected Sprint 22 GEMM and
ring-fold. This small single-order timing difference establishes no significant
slowdown. Independent release passed.

[Sprint 26](SPRINT26-SAMPLER-STATE.md) examined the sampled path using
first-acceptance observations and two full traces. The same-ring and
snapshot/ring-fold full traces each matched every observed event and all four
output/work pairs under synchronous observation. Earlier intermittent
first-acceptance and output differences remain unresolved; the first two
phases captured host-prefix metadata, not the actual GPU prefix. No numerical
resolution or ring-fold causal conclusion follows. The bounded campaign is
complete; its cause remains unknown. No blind repeat is queued without a
source-backed discriminator. If the variation recurs naturally, retain the
actual GPU prefix and lossless values. Preserve the selected profile and defer
endurance.

[Sprint 27](SPRINT27-GATEUP-COST.md) measured activation quantization and
GEMM at four selected gate/up sites. GEMM accounted for median per-window
shares of 91.06%/91.00% on ranks 0/1; independent review approved the results.
At Sprint 27 closeout, the proposed component changed the current common
M16/N128/K128 gate/up shape from stage one to stage two, keeping the kernel
body and scales unchanged. Sprint 28 completed and parked that component.
The four-site observation does not describe all 64 gate/up calls.

[Sprint 28](SPRINT28-GATEUP-STAGE2.md) completed the proposed stage-two
component: all 96 exact comparisons passed, but two of four timing cases
were slower. The candidate is parked and independent review approved the closeout. The
subsequent Sprint 29 CPU examination covered the retained compiler artifacts
and scheduling; no GPU campaign or source change follows automatically.

[Sprint 29](SPRINT29-COMPILER-REVIEW.md) reviewed the retained stage-one and
stage-two compiler artifacts on CPU. Its proposed next candidate specializes
five runtime scalars only for the current exact gate/up shape through a guarded
Triton JIT wrapper forwarding literals to the unchanged native body. Preserve
strides, masks, wrapped rows and dot/scale/add order; require compiled IR to
prove constant propagation. At Sprint 29 closeout it had not been implemented
or compiled. Sprint 30 subsequently passed exactness and compiler proof but parked the candidate after
one timing case was slower. Independent review approved the Sprint 30 parking disposition.

[Sprint 30](SPRINT30-GATEUP-STATIC-SCALARS.md) proved the intended scalar
specialization and passed 96 exact component comparisons. One of four timing
cases was slower, so the candidate is parked. At Sprint 30 closeout, a bounded warp-count component was a proposed next lead.
Sprint 31 completed that screen and parked both candidates.

For future comparisons, carry the separately selected recurrent profile in both
arms unless recurrence is the stated A/B variable. A GEMM hash alone does not
identify the recurrent profile. Optimize warmed C1 latency on the current model
and numerical profile first; keep aggregate C2 throughput as a separate
objective. The historical attribution plan below prepares no launch or GPU
reservation. Retain the selected greedy validation lane separately.
The numerical mismatch remains unresolved under the owner's session-use
disposition. This plan does not expand that numerical contract.

## Evidence and source findings

The [cost analysis](CURRENT-COST-ANALYSIS.md) establishes that generation is the
dominant service phase in the latest long sampled requests. It does not rank
current GPU subphases. Earlier short-C1 device timing places about 95% of its target graph
outside the output head and final gather. Layer collectives remain unmeasured;
the small final gather does not rule them out.

The pinned implementation has two distinct vocabulary selections:

| Path at C1, block eight | Expected matrix on each rank | Selection |
| --- | --- | --- |
| Draft candidate selection | 7 by 124160 local scores, then 7 by 32 merged scores | k=16; live warning confirms the large local selection uses `torch.topk` |
| Sampled target probability preparation | 8 by 248320 global logits | k=20 for the frozen publisher recipe, followed by top-p over those candidates and dense zero/scatter output |

The target builder already applies top-k before top-p. Eliminating a supposed
full-vocabulary top-p sort is not a supported opportunity. Its dense FP32
probability output has 7,946,240 payload bytes per rank at this shape; allocation,
zeroing, normalization and rejection sampling need separate timing.

Existing alternatives are not qualified replacements. The AOT HIP DSA kernel
is fixed to k=2048 and returns indices. The JIT fast-top-k wrapper supports
k=512/2048 and unspecified order. The Triton top-k/LSE helper supports the small
k=16 draft shape but computes extra LSE, uses scratch and smaller `torch.topk`,
and has H100-tuned geometry. At target k=20, its non-power-of-two guard falls
back to full `torch.topk` plus LSE. It is not a tiled target-k=20 alternative.
Ties, finite sentinels and masked tails need review.
No new library installation is proposed.

## 1. Historical attribution packet

Reuse the frozen four-task publisher sampling requests from the successful
screen: inputs 226/246/8176/8176, per-request total output cap 8192, temperature 1.0,
top-p 0.95, top-k=20, neutral penalties, thinking enabled at low effort. Freeze
the exact bodies, input IDs, source hashes and scoring rules before preparation.
Use the selected snapshot model path in both arms: one current observer control
and one timing arm. Do not mix in ring-fold as another changing variable.

Keep Torch 2.14/SDK 10.2, FP8 target weights, explicit BF16 draft KV, BF16 target
KV, existing head/state arithmetic, DFlash block 8, metadata reuse, M2047 stage 1,
graph buckets [1,2], context 16384, KV slots 16416 and 19 usable/20 physical recurrent
slots. Preserve all current runtime flags and owner controls.

Record the following intervals on both ranks:

| Parent interval | Nested attribution |
| --- | --- |
| Draft graph | Draft backbone, vocabulary projection, local top-k=16, small TP candidate exchange/merge and selector walk |
| Target verification | Complete graph first; output head and final vocabulary gather as disjoint subintervals |
| Acceptance | Target scaling/top-k=20/top-p, dense target-probability construction, draft-probability scatter, rejection sampler and cleanup |
| Commit | Target recurrent-state commit, target-hidden append to draft KV and next-step bookkeeping |
| CPU/request timeline | Scheduler work, observer validation/hashing, event queries and blocking waits |

First reuse the proven HIP-event approach. Install event objects and wrappers
before graph capture; an after-capture Python wrapper cannot time individual
captured kernels. Query only completed earlier work, outside capture, without
per-cycle synchronization or an extra model forward. Keep parent and nested
intervals separate. Python return times measure host work/enqueue, not GPU
completion. Do not add ranks or overlapping intervals into a wall-time total.

Freeze 32 cycle indices per request using the retained snapshot counts
62/188/357/386: one-based cycle `9 + floor(j * (C_ref - 10) / 31)` for
`j=0..31`. This spans cycle nine through the reference penultimate cycle.
Bind each case and exact request SHA256 to the schedule in the
[machine-readable plan](../provenance/POST-SPRINT7-OPTIMIZATION-PLAN-20261022.json).
Report actual covered cycle and sequence/prefix-length ranges on both ranks.
Retain unreached or uncollected indices as missing and any later tail as
uncovered. Restrict conclusions to measured windows; do not extend generation,
add a replay or synchronize to fill the schedule.

Cap retained timing records at 128 per rank. Keep the existing admitted
model-commit limit of 36864 separate; the old profiler's 128-replay guard cannot
serve this packet unchanged. Captured event commands execute on each replay
even when no record is retained. Measure that overhead and authenticate event
reuse against request/cycle replay generations before interpreting intervals.

Compose timing with the existing sampling witness and its authenticated native
delegate. Do not overwrite that witness or wrap native bodies before their
source checks. Install graph events before capture; attach eager wrappers only
after startup authentication. CPU checks must cover the actual enabled startup,
one delegate call, return values, native exceptions, restoration and cleanup.
Retain the complete output-head/cache startup dependency closure.

The sampled witness currently hashes and validates input IDs on every selector
acceptance call. Time it in place. Keep it in both arms; do not weaken evidence
collection or call a reduction in diagnostic overhead a model speedup. Mark
short requests with insufficient completed samples as incomplete attribution.

Reuse the existing two-start/four-request-per-start owner schedule. The selected
owner ceiling is 5400 seconds, with 1800-second worker and 1740-second client
limits; it is not a new ten-minute hard deadline. Preparation and actual bindings
must still pass the 64-member, 16 MiB archive, 32 MiB combined, 256-argv and 64 KiB
binding ceilings. No limit increase is part of this plan. The previous sampled
packet took 381 seconds for its native owner; allow roughly 6–10 minutes as an
estimate, not a promised deadline. Preparation effort is estimated at 45–90
minutes, with source authentication and graph-lifetime tests before admission.
If it exceeds that estimate, report the concrete gap before broadening scope.

Acceptance: source/runtime/cache/worker/HTTP/release checks pass; counters join
the same requests; both ranks have enough completed event samples to describe
each reported phase. All final answers retain the existing task checks. Record
instrumentation overhead descriptively, preserving output lengths and all ID
variations. This is attribution, not a speedup benchmark. A greedy probe is a
separate baseline and cannot attribute this sampled acceptance path.

## 2. Follow the largest measured cost

Choose one branch after attribution. Prefer existing implementations and buffer
reuse before writing a new kernel. A measured interval that includes waits is
not automatically removable work.

| Measured result | Smallest next experiment | Main correctness boundary |
| --- | --- | --- |
| Draft or target vocabulary top-k is material | Component comparison of native `torch.topk` and one compatible HIP/Triton candidate at the exact k=16/local or k=20/global shape | Exact values and ordered IDs for supported inputs; shard offsets, padded vocab, ties, NaNs/infinities, dtype and capture-buffer lifetimes |
| Target probability allocation/scatter is material | Reuse correctly sized dense scratch and unchanged normalization operations | Zero every required element; preserve top-k-first/top-p semantics, cleanup-before-view-reuse and C1/C2 lifetimes; no sparse distribution rewrite initially |
| Target graph dominates | One short kernel/API trace to classify FP8 projections/quantization, GDN, full attention and layer collectives | Trace overlap and waiting explicitly; report per-rank timelines, not summed kernel durations as a request fraction |
| FP8 projections dominate the target body | One M8-specific launch/stage candidate for the observed shapes | Preserve block scales, rounding, reductions, masks and checkpoint bytes; M2047 prefill success does not qualify M8 |
| GDN or recurrent commit dominates | Compare the previously screened snapshot and aligned ring-fold options using matched capacity and timed verify-plus-commit | Preserve the accepted sequential arithmetic and exact component/state tests; do not enable the original compact replay kernel |
| Layer collectives or graph dispatch gaps dominate | One supported, source-reviewed communication/queue change at a time | Keep topology and numerical flags fixed until an isolated comparison supports a change; final-gather timing is insufficient evidence |
| Observer or scheduler CPU work dominates | Separate diagnostic overhead from actual scheduling; then remove only proved redundant production work | Preserve request ownership, cancellation, cache identity and evidence checks; no performance claim from weaker validation |

A kernel trace is a second diagnostic only if coarse timing cannot select a
branch. Prefer existing compatible tooling with HIP-graph attribution. Bound
capture to at most eight complete verifier cycles, both ranks, and separate
traced timing from clean benchmarking. [AMD documents graph-attribution support by output format](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html).
Check the installed tool/version and graph support first; do not install a
profiler or broaden runtime dependencies here.
Use selective additional events if compatible tracing is unavailable. Freeze
artifact limits before launch rather than producing an unrestricted trace.

Evaluate benefit using exposed non-overlapping time, not a source warning or
kernel sum. Conditionally, halving a 10% exposed cost gives about 5.3% whole-run
speedup; halving a 30% cost gives about 17.6%. These are Amdahl examples, not
predictions. Small reproducible gains remain useful when implementation and
maintenance cost are low; there is no arbitrary five-percent acceptance floor.

## 3. Validate only the selected candidate

First pass a bounded component test on both GPUs with native timing and exact
retained operands or declared synthetic inputs. Keep unique-score, tie, masked
vocabulary, dtype, suffix, canary and graph-replay cases relevant to the change.
[PyTorch documents that tied `topk` indices are not stable](https://docs.pytorch.org/docs/stable/generated/torch.topk.html). Equal selected
values alone are insufficient for candidate-ID parity. If tie/NaN policy cannot
match the pinned reference, leave that candidate parked pending an explicit
numerical-contract decision. Do not silently assign a new tie policy.

A component win earns one matched serving pilot. Use fixed output work for
latency measurement and natural stops for task quality. Keep output lengths,
reasoning/final segments and sampling modes explicit. A lower latency caused by
fewer generated tokens is not a kernel speedup. Check the same runtime route,
metadata reuse, graph replay, acceptance and commit witnesses in both arms.

If the pilot supports the candidate, freeze one implementation, run reverse
order and complete the affected greedy and sampled quality/reuse/cancellation
checks. Define that candidate's exact packets and finite budget after its
scope is known. Do not reserve a multi-candidate sweep now. Preserve the current
baseline variation separately; new task failures, faults or materially new
unexplained changes require investigation. No production promotion follows.

## Parked work and next deliverable

Keep FP8 target KV deferred. Keep width eight: width four was slower, and the
long tasks already emit about 7.4–7.5 tokens per verification cycle. Adaptive
speculation remains a separate policy project. The small output-gather interval
does not justify starting target shard-max now. Approximate normalization fusion,
precision changes, altered sampling/RNG behavior and new graph/overlap policies
are outside the first packet. C2 throughput tuning follows as a separate metric.

Sprint 10 has now completed both matched benchmark orders; see its
[tracking record](SPRINT10-M8-SERVING.md). Affected candidate quality and lifecycle
screens passed after independent review. The M8 source is selected for the next
experimental profile launch; no installed runtime or production default changed.
The owner selected further projection tuning as Sprint 11. That bounded study
closed without a new source selection. Preserve its failed gates and the parked
status at that closeout. The subsequent owner-requested
[3072 experimental assessment](M8-3072-EXPERIMENTAL.md) has separate task/cache
evidence. Packaging and endurance remain separate deferred work.

A source-only audit also identified the existing unrolled-kernel selector's
missing return. Its intent and performance remain unverified. The isolated
[Sprint 12 component](SPRINT12-UNROLLED-PROJECTION.md) rejected the stage-three
configuration before any candidate launch. Another stage depth or tile needs a
separate reviewed component; do not enable the alternate route globally on a
source finding alone.

Sprint 31 parked W8 at the component gate and W2 after its serving median gate failed. Independent review approved the closeout; the selected profile remains unchanged. Sprint 32 completed a bounded 32K BF16-KV screen; the owner accepted lexical exceptions while the strict quality gate remains failed. That led to the separately reviewed Sprint 33 64K screen described below. UltraQuant is parked and the [BF16 versus FP8 weight comparison](BF16-FP8-WEIGHT-COMPARISON.md) is deferred. FP8 target-KV and adaptive speculation remain optional later feasibility work; endurance follows selection.

Sprint 33 completed the bounded 64K screen: both frozen SRE gates scored 10/12 and failed, and the two format failures remain. Independent review approved the conservative closeout. Those findings remain available as later optimization leads. The owner has frozen the current Qwen3.8 baseline and prioritized Qwen3.8-Flash-Next feasibility, then a baseline, then separately reviewed thread-inspired optimizations. Shape-specific patches do not transfer automatically.

Qwen3.8-Flash-Next [feasibility preflight](QWEN-NEXT-FEASIBILITY.md) found the official FP8 tensors exceed combined physical RAM and VRAM. The checkpoint and memory route remain undecided; the baseline and later optimizations are design only.
