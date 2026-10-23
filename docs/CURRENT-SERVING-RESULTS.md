# Current serving verification

These results describe bounded private runs on two Radeon AI PRO R9700 GPUs.
They do not turn this source prototype into an installable or qualified serving
distribution. The runs used a complete separately pinned SGLang runtime.
Machine configuration, model weights, native binaries and raw logs are excluded.

## FP8 thinking checks, 2026-10-21

The published Qwen3.8-27B-FP8 checkpoint passed the fixed J01 ledger task in two
fresh starts at low reasoning and two at the highest published effort, `xhigh`.
Every answer rejected a debit of 31 plus a fee of 4 against a balance of 34.
Each pair returned identical output token IDs. Low effort produced 425 output
tokens per start; xhigh produced 334. Exact prompt IDs and final-answer decoding
were checked against the checkpoint's published tokenizer.

The xhigh worker timeout was 1800 seconds. Requests completed in about 14 seconds;
the timeout includes separate startup work and was not a measured request time.
The earlier wrong answer was not reproduced on this short prompt. Long-context
quality and the combined speculative path were outside these checks.

The stack was PyTorch `2.14.0+git2e66fe81.rocm10.2.0`, ROCm SDK
`10.2.0.dev20261020+git535d2990`, HIP `7.17.26416` and RCCL `2.32.3`.
Private startup compatibility work rebuilt the retained SGLang kernel and pinned
TorchVision source against that Torch build, restored SDK tool executable modes
in a byte-identical runtime copy, exposed the matching HIP development SDK and
redistributed scratch space to provide 512 MiB of shared memory. The original
sealed inputs and installed defaults were preserved.

## Matched previous/current stack benchmark

One ordered previous/current pair and one current/previous pair used identical
requests: 451 input tokens, exactly 128 output tokens, low thinking and concurrency
one. Each stack had two fresh starts, each with one excluded warm-up and three
measured requests. All 16 requests completed, with zero cached tokens and zero
retractions. All output token vectors matched.

| Stack | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| PyTorch 2.13 / ROCm SDK 10.1 | 24.59 | 5.2062 s | 5.1949–5.2224 s |
| PyTorch 2.14 / ROCm SDK 10.2 | 24.57 | 5.2096 s | 5.1922–5.2158 s |

The combined median rate changed by -0.07%. The two orders showed opposite
changes of about 0.3%. No meaningful performance regression was detected in this
screen. This small sample does not establish equivalence for other workloads.

Both arms used the same FP8 target, BF16 attention cache, FP32 recurrent state
and head, greedy sampling, seed 12345, Triton attention, decode graphs and
hipBLASLt preference. **DFlash2 and the split K/V decode adapter were disabled.**
This is a whole-stack comparison, not an attribution to Torch, HIP or RCCL alone.

The rate is output tokens divided by local non-stream HTTP request-to-terminal
time. It includes prefill and protocol completion. It is not isolated decode
throughput, TTFT or ITL. All throughput figures describe one request served by
both GPUs together in TP2; they are not per-card rates. `ignore_eos=true` forces
the fixed output count; these
truncated responses were not graded as completed task answers. Long contexts,
concurrent traffic, sustained load, power and thermal behavior remain untested.

## Combined DFlash2 and split K/V benchmark

The combined path completed on the current stack with the pinned BF16 DFlash2
drafter at block size 8, the retained per-row native split-decode adapter and
aligned GDN, together with the FP8 target, FP32 head, GEMM optimization and
decode graphs.

| Current-stack configuration | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| Ordinary FP8 serving | 24.50 | 5.2246 s | 5.2237–5.2273 s |
| DFlash2 + split K/V + aligned GDN | 86.68 | 1.4766 s | 1.4753–1.4808 s |

The median rate increased **3.54×** on this workload. Each arm had one excluded
warm-up and three measured requests, using the same 451-input/128-output-token
workload described above. All eight requests completed with zero cache hits and
zero retractions. All four output-token pairs matched exactly.

Both target ranks recorded 16 adapter capture calls across all 16 attention
layers, with an allocated adapter buffer set. Native validators authenticated
graph replay and speculative verification. The four combined-arm responses
reported 100 verification passes, 700 proposed drafts and 412 correct drafts;
these counters are not completion-token totals.

This is one ordinary/full arm order. It measures the whole combined path and
does not isolate each optimization. The layer-normalization candidate and
FP4/KV4 cache were excluded. The truncated outputs were not graded as completed
answers. The HTTP timing and workload limits described above apply.

The stock verification split-KV kernel is gated to another GPU family in the
retained source. On R9700, the experimental adapter invokes native decode
attention with the split selector applied separately to each verification row.
This is separate from the four-bit KV cache prototype.

### Memory observations from the combined-path benchmark

The retained telemetry includes a host-memory peak for each complete arm and
GPU startup figures for each rank. The figures below use GiB (2^30 bytes).
SGLang labels its binary-unit GPU figures as `GB` in the logs. Both ranks
reported the same rounded GPU values.

| Host-memory metric, whole arm | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Cgroup memory peak, including startup and warm-up | 18.32 GiB | 23.69 GiB |
| Configured cgroup memory limit | 38 GiB | 38 GiB |

These peaks are Linux cgroup v2 `memory.peak`, including charged page cache,
tmpfs and child processes. They are not process RSS or GPU VRAM. Swap was
disabled. Neither arm recorded a memory-limit event, OOM or OOM kill.

| GPU startup metric, per GPU | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Target weight-load free-memory delta | 14.51 GiB | 14.51 GiB |
| Drafter weight-load free-memory delta | — | 2.16 GiB |
| Target BF16 KV pool, K + V | 0.25 + 0.25 GiB | 0.25 + 0.25 GiB |
| Drafter BF16 KV pool, K + V | — | 0.08 + 0.08 GiB |
| Recurrent-state pool slots | 93 | 19 |
| Recurrent convolution / SSM state | 0.13 / 6.61 GiB | 0.03 / 1.41 GiB |
| Speculative intermediate SSM / convolution state | — | 1.69 / 0.01 GiB |
| Target graph-capture memory delta | 0.29 GiB | 0.38 GiB |
| Drafter graph-capture memory delta | — | 0.10 GiB |
| Available GPU memory after graph capture | 9.11 GiB | 10.13 GiB |

Each attention KV pool had 16,416 token slots. The runtime sized the recurrent
pool differently for speculative execution, which also requires intermediate
state. The greater free GPU memory in the combined arm therefore does not
establish better memory efficiency at equal capacity. Both configurations
completed the measured concurrency-one workload.

GPU figures are rounded startup observations, taken before external requests.
They are not a complete additive memory accounting or a measured GPU peak.
No continuous VRAM samples, Torch allocator peaks, per-request host peaks or
per-process RSS were retained. Exact host peak bytes, per-rank GPU figures and
capture hashes are included in the sanitised JSON linked below.

### Metadata reuse validation, 2026-10-21

A subsequent candidate reuses causal metadata across the 16 attention layers
while preserving all eight singleton split selections. Both GPU ranks recorded
one metadata build during capture across 16 eligible attention calls. All eight
requests completed, with zero cached tokens and zero retractions. All four
ordinary/candidate output pairs matched, as did all four comparisons against
the earlier combined-path outputs.

| Current validation configuration | Median output tokens/s | Median request time |
| --- | ---: | ---: |
| Ordinary FP8 reference | 24.47 | 5.2302 s |
| Combined path with metadata reuse | 87.29 | 1.4663 s |

The workload and timing metric match the short C1 screen above. These are
combined TP2 rates, not per-card rates. The approximately 0.70 percent increase
over the earlier 86.68 tokens/s is descriptive: there was no matched
old-adapter/new-adapter comparison in reversed order. No additional throughput
gain is established. Source, native/wire, raw timing and independent cleanup
checks passed. See [metadata reuse](METADATA-REUSE.md) and its
[sanitised validation record](../provenance/VERIFY-METADATA-REUSE.json) for
capture counters, evidence identities and scope limits.

### M2047 prefill-tail component validation, 2026-10-21

The one-stage M2047 candidate passed all 360 exact comparisons across 20
synthetic cases on the current Torch 2.14 / SDK 10.2 stack. Both GPUs were tested
sequentially. Stage 1 was 1.311–1.575 times faster than stage 3, with a geometric
mean of 1.429 across the cases. There were 320 timed launches and 40 warm-ups.
Structured outputs matched an analytical oracle; random outputs matched the
stage-3 baseline. Input immutability and finite-output checks passed.

The owned run took 47.81 seconds and its native payload took 29.70 seconds.
Independent release and both evidence seals verified. These are component
kernel measurements, not TP2 serving rates or a request-level speedup. Matched
serving A/B is recorded below. See the [component record](../provenance/M2047-COMPONENT-20261021.json)
and [current sprint list](SERVING-SPRINTS.md).

### M2047 matched serving A/B, 2026-10-21

Both orders showed about 4.9 percent lower median request latency for stage 1.
Forward medians were 1.9487 s versus 1.8536 s; reverse medians were 1.9454 s
versus 1.8501 s. These are combined TP2 measurements at C1 with 2,047 input
tokens and 128 forced output tokens. Each model start had one excluded warm-up
and three measured requests. The two orders used four starts and 16 requests.
All five changed M2047 shapes were observed on both ranks.

The exact output gate is incomplete. Seven of eight paired prefixes matched;
the candidate was repeatable in eight of eight requests, while one baseline
response first differed at output index 102. Its 23 verification cycles differ
from the common 25-cycle path. The record preserves that response and timing.
These observations do not establish clean performance attribution or task
quality. The owner subsequently selected M2047 for current-session use, with
the unresolved variation retained for review if concerning results recur.
The current stack and installed runtime remain unchanged. See the
[sanitised serving record](../provenance/M2047-SERVING-20261021.json) and
[current sprint list](SERVING-SPRINTS.md).

### Baseline repeatability follow-up, 2026-10-21

Two fresh stage-3 baseline starts replayed the same four requests each. All
eight outputs matched the historical common prefix, with 25 verification
replays per rank and no graph recapture. Measured request medians were 1.9490 s
and 1.9445 s. Warm-up outputs were compared but excluded from timing.
Source, runtime, weight, HTTP and independent release checks passed.

The earlier variant was not reproduced. This finite screen does not identify
its cause or complete the M2047 exact-output gate. The owner subsequently
selected current-session use; see the
[current selection](../provenance/CURRENT-SESSION-SELECTION.json).
The stack and installed runtime remain unchanged. See the
[repeatability record](../provenance/BASELINE-REPEATABILITY-20261021.json) for
all samples, speculative histograms and per-rank graph deltas.

### Current-session M2047 activation and Sprint 6, 2026-10-21

M2047 stage 1 is selected for this session. Two fresh starts with both arms
enabled completed eight requests. Measured request medians were 1.8488 s and
1.8525 s, approximately 69.23 and 69.10 output tokens/s combined TP2. Each
start excluded one warm-up from timing. All five changed M2047 shapes reached
both ranks. All requests used 25 verification replays per rank and no
recaptures. The 28 CPU checks and source/runtime/weight/wire/release checks
passed. The current stack and installed defaults remain unchanged.

Seven outputs matched the common prefix exactly. One differed only in the
capitalization of `Required` at output index 124. The exact mismatch remains
in the record, and the owner selected continued session use. These forced
128-token outputs are not a task-quality qualification.

The startup census observed a 3.79 MiB FP32 local shard and 7.58 MiB gathered
result per rank for a C1 verification block. Exposed latency and link-level
traffic were not measured. Retain current communication settings and continue
Sprint 7 combined-profile qualification. See the
[session and census record](../provenance/SESSION-M2047-SPRINT6-20261021.json).

### Remaining optimization screens, 2026-10-21

The fresh width comparison retained DFlash8: median request latency was
1.8474 s (69.29 tokens/s combined TP2) versus 2.5137 s (50.92 tokens/s) at
width four. Both used equivalent singleton split attention and metadata reuse.
The output prefixes differed; this was a forced-output C1 screen with no task
grading. All eight requests and independent release checks completed.

The aligned ReplaySSM component passed all 1,032 exact comparisons in sixteen
cases on both GPUs. It is ready for serving integration, not full-model
qualification. Component storage formulas and timings do not establish
whole-model VRAM savings or a serving speedup. Adaptive-policy prerequisites
were audited; no adaptive policy was implemented or GPU-tested.

Keep M2047 stage 1 + metadata reuse + DFlash8 as the Sprint 7 reference.
Aligned ReplaySSM remains a separate integration candidate. See the
[screening results and limits](OPTIMIZATION-CANDIDATES.md).

## Evidence identities and limits

| Completed observation | Private local evidence manifest SHA256 |
| --- | --- |
| FP8 low/xhigh verification | `780765826ccbbdc5bce91064dcf74eb17b3adee78c785100ce705802235e685e` |
| Previous/current HTTP benchmark | `47f91427c597fdec16f374686bcd4e17ed338a3c383a4a5b0c6cd8e4ef433fd0` |
| Combined-path native benchmark | `ff2061124636c6894ea1e037173dd272783d47869ce89b78448f8216e116f3af` |
| Metadata-reuse native validation | `2dc999e0d649c59bb43bf29ed30d1ef05d90fe4a688146aa054965ce3fb05fe5` |

Sanitised measurements and source/native evidence identities are retained in
[SERVING-RESULTS-20261021.json](../provenance/SERVING-RESULTS-20261021.json).

Source, native worker, HTTP, loaded-weight, graph, resource and independent
release checks passed for those completed observations. The benchmark timings
were independently recomputed from retained monotonic timestamps. These hashes
identify private evidence; they are not portable runtime/build proofs or launch
authority. Optional TVM Torch C DLPack and Python 3.14 TorchScript warnings remain.
No production deployment or broad quality qualification was performed.

## Aligned ReplaySSM serving screen, 2026-10-21

The aligned ring-fold candidate passed 43 CPU checks and eight native C1
requests after independent Astra implementation review. All four paired
output-ID vectors matched exactly. Both arms used DFlash8, singleton split
attention with metadata reuse, M2047 stage one and the current stack and
precision. Each request used 2,047 input tokens and 128 forced output tokens.

Median request latency was 1.8490 s for the snapshot reference and 1.8042 s
for ring-fold, a 2.42% reduction. Combined TP2 output rate was 69.23 versus
70.95 tokens/s. This is one order, with one excluded warmup and three measured
requests per start. Per-request native commits, graph replays and verifier
cycles all matched at 25 on each rank. Independent release passed.

Automatic recurrent capacity increased from 19 to 74 usable slots per card.
Startup Torch allocation rose from 20.53 to 23.01 GiB per card. That screen did
not measure matched-capacity savings or reverse order; the follow-up below
covers both. Carry this candidate into broader Sprint 7 qualification
alongside the current reference. Normal-stop quality, longer contexts, C1/C2,
sampling, reuse and repeated lifecycles remain incomplete.
See the [serving record](../provenance/REPLAYSSM-SERVING-20261021.json)
and [screening list](OPTIMIZATION-CANDIDATES.md).

## Sprint 7 matched-capacity comparison, 2026-10-21

Both execution orders completed at 19 usable recurrent slots per card, with
45 CPU checks per packet and sixteen native requests across four starts.
Ring-fold reduced median request latency by 2.32% and 2.47% in the two orders.
Startup Torch allocation fell from 20.53 to 18.90 GiB per card in both orders.
The 2 MiB residual between the measured allocation change and the main tensor
calculation remains unassigned. These counters describe startup after graph
setup, not whole-device peak VRAM.

Seven of eight output pairs matched exactly. One reference response repeated
the earlier capitalization variant at token 124; all eight candidate outputs
matched the historical common output. All requests used 25 verification
replays and completed commits per rank, with no recaptures. Independent
release passed. These short C1 timings are descriptive combined-TP2 figures;
normal-stop quality and broader lifecycle qualification remain incomplete.
See the [two-order record](../provenance/SPRINT7-MATCHED-CAPACITY-20261021.json)
and [remaining Sprint 7 gates](SPRINT7-QUALIFICATION.md).

## Explicit BF16 draft KV and owner disposition, 2026-10-21

The explicit `--speculative-draft-kv-cache-dtype bf16` override completed two
fresh GPU starts and eight C1 requests. Both ranks confirmed actual BF16 target
and draft K/V buffers in both starts. All eight forced 128-token outputs matched
the historical common response. Source, weight, HTTP, graph and independent
release checks passed after 38 CPU package checks and Astra implementation review.

Median full HTTP output rates were 69.15 and 69.31 tokens/s for combined TP2.
There was no flag-off control, isolated decode measurement or task-quality grade.
This screen did not test FP8 target KV or establish a mismatch fix.

The owner selected this configuration for the current session and directed us
to move on, backtracking if more strange behavior occurs. The mismatch diagnosis
is closed as unresolved. Earlier capitalization variants and the later stage-3
variation from token 16 remain retained; exact reproducibility and broader task
quality are not qualified. Continue with the selected snapshot reference and
explicit BF16 draft override in both future KV arms. No installed default or
persistent server was changed. See the [override validation and owner decision](../provenance/BF16-DRAFT-KV-OVERRIDE-20261021.json).

The owner deferred FP8 target attention KV to a [separate optional work item](SERVING-SPRINTS.md#optional-fp8-target-attention-kv). Keep the current FP8 weights and BF16 target/draft caches. The next active work is Sprint 7 normal-stop quality and sustained-output validation.


## Combined low-thinking task screen, 2026-10-21

The selected M2047 stage 1 + metadata reuse + DFlash8 snapshot passed four tasks
in each of two fresh starts with thinking enabled at low effort. All eight
responses stopped naturally and passed applicable final-answer task and format
checks. The two ledger finals contained 763 and 821 tokens and matched all 28
expected events, including the earlier J01/H17 fee-boundary cases. All four
output-ID pairs matched exactly. Independent GPU resource release was verified.

Ledger inputs were 12272 tokens, with a 4096-token total output cap inside the
existing 16K context. Inert filler was reduced; task facts and oracles stayed
fixed. This is a bounded greedy-profile result, not a mode-only comparison,
publisher-recipe qualification or general quality guarantee. Ring-fold and
other Sprint 7 gates remain pending. See the
[thinking quality record](../provenance/SPRINT7-THINKING-QUALITY-20261021.json)
and [validation skill](../.agents/skills/aithreadripper-serving-validation/SKILL.md).

## Sprint 7 matched thinking quality, 2026-10-21

Snapshot and ring-fold each passed four natural-stop tasks in one fresh start:
8/8 applicable task, format and sustained-final-output checks. All final texts
and token segments matched the two prior selected-profile starts. Two of four
full output-ID pairs matched; the ledger reasoning paths differed. Exact-output
equivalence and the earlier mismatch remain unresolved.

Both profiles observed 19 usable/20 physical recurrent slots per card and BF16
target/draft attention caches. Ring-fold again reduced startup Torch allocation
by 1.63 GiB per card. Zero request recaptures and equal per-rank commit/replay/verify
counts were authenticated, along with independent release. This is a bounded
quality result with one start per profile and no speedup or repeatability claim.
See the [record](../provenance/SPRINT7-RINGFOLD-THINKING-QUALITY-20261021.json).
C2, reuse/branching, cancellation and repeated lifecycle coverage remain pending.

## Both-order thinking lifecycle screen, 21 October 2026

Both profiles completed four fresh starts and 108 planned requests in both
orders. All 104 normal final answers were correct, and four active streaming
cancellations completed with correct recovery. Each start showed real batch-two
graph traffic on both ranks. Verifier counts, singleton/batch-two replays and
completed commits reconcile. All active request rows were released; free plus
evictable cached slots accounted for all 19 usable recurrent slots per rank.

The schedule covered prefix-boundary reuse and branches, cache pressure,
victim recovery, cancellation, concurrency and final replay. Integer allocator
observations establish slot bookkeeping, not numerical state equivalence or
per-node cache ownership. Full output IDs matched 42/52 cache comparisons; all
52 final-answer comparisons passed. Retain the ten reasoning-output differences
and the unresolved exact-output contract. Independent release passed after both
packets. These instrumented runs establish bounded lifecycle behavior and task
quality; their timings are not performance claims. See the
[lifecycle record](../provenance/SPRINT7-LIFECYCLE-20261021.json).

## Sprint 7 sampled screen and closeout, 21 October 2026

The final publisher-recipe screen passed 8/8 natural final answers on snapshot
and ring-fold. It used low thinking, temperature 1.0, top-p 0.95 and top-k 20.
All 16 external request/rank witnesses joined a successful native DFlash
selector-sampling return to the frozen request. All coding finals passed the
unchanged Linux grader; all ledger finals passed the full event oracles and
exceeded 512 final tokens. Independent release passed.

The ledger inputs were 8176 tokens, with an 8192-token total output cap and a
16-token reserve: 16384 tokens in total, within the selected context. Only inert
filler was reduced. This is a small functional sampling screen, not a distribution test
or a performance benchmark. See the [sampled result](../provenance/SPRINT7-SAMPLING-20261021.json).

The [bounded Sprint 7 closeout](../provenance/SPRINT7-CLOSEOUT-20261021.json)
retains the snapshot reference, the separately screened ring-fold option and
the unresolved exact-output exception. General numerical equivalence and
production qualification remain incomplete. No persistent server or installed
default was replaced.

## Offline cost attribution, 2026-10-22

The [current cost analysis](CURRENT-COST-ANALYSIS.md) uses the latest sampled
quality telemetry, a separate greedy counter crosscheck and the earlier
short-C1 device timing. Generation is the dominant service phase on the long
requests. Current GPU shares for drafting, target verification and commit are
not measured. The draft selector's large-vocabulary top-k fallback is a concrete
untimed candidate. No models were started and no new speedup was established.

## Sprint 8 native cost attribution, 22 October 2026

The corrected snapshot control/timing packet passed all eight natural final
answers, authenticated 126 completed samples per card and verified independent
release. It preserved the selected stack, flags, FP8 weights and BF16 caches.
All final-answer texts matched; one held-out reasoning length differed by eight
tokens. The prior numerical mismatch remains unresolved.

On the long requests, the target graph took about 44.85 ms within a 54.06–54.09 ms
sampled cycle on each rank, about 83% of that interval. About 42.64 ms remained
after the output head and final gather. Draft local top-k was only 0.068–0.071 ms.
These are per-rank instrumented windows, with missing and uncovered cycles
retained. They do not establish a whole-request share or speedup. Sprint 9 first
classifies the target graph through a bounded trace before choosing a component.
See the [attribution and coverage record](SPRINT8-COST-ATTRIBUTION.md).

## Sprint 9 target graph classification, 22 October 2026

The two-start diagnostic passed 8/8 final-answer gates and independent release,
with eight completed timing samples per card. In four long-request samples per
rank, projection methods total a median 30.757/30.730 ms within a 48.334/48.339 ms
target graph. MLP gate/up projections total 13.775/13.777 ms across layers. These
nested, instrumented intervals select an M8 gate/up component comparison; they
are not a clean serving benchmark. Three of four full output-ID pairs match;
all final answers match. The held-out reasoning variation remains recorded.
See [Sprint 9 tracking](SPRINT9-TARGET-GRAPH.md) for coverage gaps, overhead and
remaining gates. The stack, runtime flags and selected numerical profile remain
unchanged; no new optimization is qualified.

The M8 stage-depth component subsequently passed 96/96 exact comparisons,
including 24 graph replays across both cards. Stage 1 reduced eager component
median time by 6.61–13.06% across four structured/random cases. The first attempt's
capture-stream harness failure remains sealed; the corrected run passed both
cards and independent release. The experimental patch is recorded but remains
uninstalled. Sprint 9's component exit is complete after independent review;
Sprint 10 subsequently completed matched serving validation below. See the
[component record](../provenance/SPRINT9-M8-COMPONENT-20261022.json).


## Sprint 10 matched M8 serving, 22 October 2026

M8 gate/up stage one is selected for the next experimental profile launch after
Astra review. On fixed 2,047-input/1,024-output C1 work, median client latency
fell 0.6023% forward and 0.3479% in reverse order. Combined TP2 rates rose
0.6059% and 0.3491%. All six measured pairs match output IDs and 164 verifier
cycles; each start's first request is excluded. Existing observers remain equal
and additional classification timing events are disabled.

Repeated candidate screens passed greedy 8/8 and publisher-sampled 8/8 final-answer
gates, then lifecycle 52/52 natural answers plus two cancellations/recoveries,
actual C2 and slot accounting. Greedy finals all match between starts; two
reasoning sequences differ. Lifecycle full outputs match 24/26 cache pairs and
all 26 finals match. The numerical mismatch remains unresolved; this is bounded
qualification on the current snapshot/BF16 profile. Startup allocation is
20.52785 GiB/card, reserved 20.80078 GiB/card. Installed runtime and production
defaults remain unchanged. See [Sprint 10](SPRINT10-M8-SERVING.md) and its
[result index](../provenance/SPRINT10-M8-SERVING-20261022.json).


## Sprint 14 smaller unrolled tile, 22 October 2026

The exact untuned M8 down projection now selects native unrolled tile16/stage1
for the next experimental session launch, after Astra review. Both matched
orders passed all eight output-ID pairs and 164 verifier cycles per request.
Median client latency fell 1.7442% forward and 1.7851% in reverse, reaching
115.1192 and 115.1496 combined TP2 output tokens/s across both cards.
The workload uses C1, 2,047 input tokens and 1,024 forced output tokens with
low thinking. Each start's first request is excluded. Rates include prefill;
they are not decode-only or final-answer throughput.

Natural-stop screens passed greedy8/8, sampled8/8 and lifecycle52/52 final
answers, plus two cancellations/recoveries, actual C2, 19-slot accounting and
independent release. Greedy3/4 and cache24/26 full-output pairs matched; all
paired final answers matched. Sampling route execution passed on both ranks,
without establishing sampling-distribution equivalence. Earlier numerical
findings and Sprint 11 failed work gates remain unresolved under the owner's
experimental-session disposition. Reliability remains deferred; installed
runtime and production defaults are unchanged. Retain the previous selected
source for rollback. See [Sprint 14](SPRINT14-UNROLLED-TILE16.md) and its
[evidence index](../provenance/SPRINT14-UNROLLED-TILE16-20261022.json).

## Sprint 17 GDN input projection, 22 October 2026

The exact untuned M8 GDN qkvz projection `(8,8192,5120)` now selects native
unrolled tile16/stage1 for the next experimental session launch, after Astra
review. Both matched orders passed all eight output-ID pairs and 164 verifier
cycles per request. Median client latency fell 0.5435% forward and 0.6425% in
reverse, reaching 115.7085 and 115.7343 combined TP2 output tokens/s across both
cards. These rates include prefill on C1 work with 2,047 input tokens and 1,024
forced output tokens, low thinking and one excluded warmup per start. They are
not isolated decode or final-answer throughput measurements.

Greedy8/8, sampled8/8 and lifecycle52/52 normal final answers passed, plus two
cancellations/recoveries, actual C2, 19-slot accounting and independent release.
Greedy and sampled each matched all four full output pairs. Lifecycle matched
24/26 full cache-output pairs and all 26 final-answer pairs; retain the two
`concurrent-a` reasoning variants. Sampling route execution was verified on both
ranks without establishing distribution equivalence.

The current stack, FP8 weights, BF16 target/draft caches, FP32 state/head, flags,
DFlash8, metadata reuse and earlier selected routes remain unchanged. Preserve
Sprint 14 source for rollback, earlier numerical findings and Sprint 11 failed
gates. Reliability remains deferred; installed runtime and persistent defaults
are unchanged. See [Sprint 17](SPRINT17-GDN-INPUT.md) and its
[evidence index](../provenance/SPRINT17-GDN-INPUT-20261022.json).

## Sprint 18 attention QKV, 22 October 2026

The exact `(8,7168,5120)` unrolled tile16/stage1 option remains unselected after
Astra review. Component checks passed 96/96 with 16.25–18.25% less kernel time.
Both matched serving median gates passed at 0.1948–0.2237% lower latency; all
eight output pairs and 164 cycles/request matched. Candidate rates were
116.0174–116.0459 combined TP2 output tokens/s on the same C1, 2047-input/
1024-forced-output low-thinking workload, with one excluded warmup/start.
Rates include prefill; they are not decode-only or final-answer rates.

One reverse measured candidate request was 1.3016% slower. Including every sample,
the pooled mean was 0.0574% higher. Overall service benefit is inconclusive;
these descriptive results do not replace the selected Sprint 17 benchmark.
Greedy passed 8/8 final answers, with 4/4 paired final texts and 3/4 full output pairs
matching. Retain the audit-threshold reasoning variant. Prepared sampled and
lifecycle screens were not launched. Current source, stack, flags and precision
remain selected; reliability stays deferred. See [Sprint 18](SPRINT18-ATTENTION-QKV.md)
and its [evidence index](../provenance/SPRINT18-ATTENTION-QKV-20261022.json).


## Sprint 19 matched ring-fold, 22 October 2026

Both orders matched eight full-output ID pairs and 164 verification cycles
per request per rank. Ring-fold measured 120.3602 versus 115.6911 combined TP2 output
tokens/s forward and 120.2798 versus 115.6689 in reverse; median latency fell
3.87926% and 3.83348%, respectively. All six measured candidate requests were
faster, without a statistical significance claim. These C1 rates include
prefill on 2047-input/1024-forced-output, greedy low-thinking requests, with one
excluded warmup and three measured requests per start.

With 19 usable and 20 physical recurrent slots per card, startup allocation
after graph setup fell from 22,041,610,240 to 20,296,157,184 bytes on each
card in both orders: 1.62558 GiB per card. This is not whole-device or
process-lifetime peak memory. Geometry predicts 1,747,550,208 bytes of savings;
the 2,097,152-byte residual is unassigned. The saving is attributed to recurrent
ring-fold; the split-KV computational adapter retains BF16 caches.

Greedy natural-stop quality passed 8/8 final answers, with 4/4 full-output ID
pairs and 4/4 final-text pairs matching across profiles. The two long ledger
finals reached 763 and 821 tokens, above the 512-token gate. Source, native,
HTTP, cache, work and release evidence was authenticated. Publisher-sampled
quality passed 8/8 final answers, with 4/4 final-text and 2/4 full-output pairs
matching. Two ledger reasoning outputs differed; the native selector-sampling
route was verified on both ranks and arms without establishing distribution
equivalence. Lifecycle passed 52/52 normal final answers, two cancellations,
actual C2, slot bookkeeping and release. Of 26 cache-output pairs, 24 matched
full IDs and all 26 matched final text; both arms retain the `concurrent-a`
versus `victim-cold-a` reasoning difference at index 149. Ten starts covered 86
requests: 16 forced-output, 68 natural-stop and two cancellations. All 15
evidence roots passed closeout verification. Astra final review supports
selecting aligned ring-fold for the next experimental session only. The Sprint
17 GEMM remains unchanged and the paired snapshot control is retained for
rollback. Earlier
numerical findings, Sprint 11 failed gates and the owner experimental-session
disposition remain. Reliability and
endurance remain deferred, no GPU campaign is queued, and installed runtime and
persistent defaults remain unchanged. This is not production, numerical, state
or distribution qualification. See [Sprint 19](SPRINT19-MATCHED-RINGFOLD.md).


## Sprint 20 selected-profile cost refresh, 22 October 2026

Two starts and eight sampled low-thinking requests completed on the selected
Sprint 17 GEMM plus Sprint 19 ring-fold profile. All eight natural final answers
passed, all four full-output ID pairs matched, and verification-cycle counts
matched in each pair. Both ranks collected all eight scheduled device windows;
source, native, HTTP, cache, graph and independent release checks passed.

Four long-request windows per rank place projection methods at 27.550/27.516 ms
within a 44.445/44.446 ms target graph. Gate/up remains the largest projection
group at 11.939/11.932 ms; down takes 5.668/5.654 ms and GDN qkvz 4.384/4.377 ms.
These forward intervals include quantization, launches, waits and enclosed
collectives. Nested groups overlap; do not add groups or ranks into request time.

Instrumented client elapsed time increased by 2.83%, 8.27%, 6.52% and 7.02% with
matched outputs and verification work. One control-first order does not isolate
causal overhead. Coverage remains sparse, with 13/42/64/150 later cycles beyond
the last sampled window. This attribution does not establish a whole-request
bottleneck or a new serving speed. The historical Sprint 19 benchmark was about
120.3 combined TP2 tokens/s under its fixed-work envelope.

The follow-on common M16 component and serving comparison are recorded below.
See [Sprint 20](SPRINT20-CURRENT-PROFILE.md) and its
[evidence index](../provenance/SPRINT20-CURRENT-PROFILE-20261022.json).

## Sprint 21 common M16 gate/up, 22 October 2026

The common M16/stage1 candidate passed 96 exact component comparisons at
`(8,17408,5120)`. Both matched serving orders passed all eight output/work
pairs. Median latency fell 0.38064% and 0.28794%; candidate rates were 120.304
and 120.368 combined TP2 output tokens/s. The pooled mean latency change was
−0.29775%, a descriptive result. The historical Sprint 19 benchmark was about
120.3 combined TP2 tokens/s under its fixed-work envelope.

Each performance start had one excluded warmup and three measured requests.
Greedy and sampled quality used candidate GEMM and ring-fold in both arms; each
passed 8/8 final answers and 4/4 full-output pairs, with independent release.
The sampled route was verified, but its distribution was not. The 54-request
lifecycle screen passed 52/52 finals, two cancellations, C2, slots and release;
24/26 full cache outputs and 26/26 final texts matched. Both arms retained the
Sprint 19 first difference at token 149 for the same cache comparison, without a
proven cause. Astra selected M16 for the next experimental session only; the prior
GEMM remains rollback and ring-fold is unchanged. Earlier numerical findings and
deferred endurance remain. See [Sprint 21](SPRINT21-COMMON-M16.md) and its
[provenance](../provenance/SPRINT21-COMMON-M16-20261022.json).

## Sprint 22 GDN output candidate, 22 October 2026

The common M16/stage1 candidate at `(8,5120,3072)` passed 96 exact component
checks and all eight serving output/work pairs in both orders. Median latency
fell 0.89178% forward and 0.81403% in reverse, with candidate rates of
121.175–121.243 combined TP2 output tokens/s under the same fixed-work envelope.
These are finite C1 results for the source selected for the next experimental
session; the previous Sprint 21 source remains available for rollback.

Greedy validation passed 8/8 normal-stop answers and 4/4 full-output pairs.
The sampled campaign stopped when the supervisor reported that a process changed
during namespace observation at second-arm startup. The interrupted arm then
lacked its expected artifact bundle. Four first-arm HTTP responses remain
ungraded, and lifecycle validation was not launched in that initial attempt.
Independent release passed without survivors or unresolved cleanup.

The owner subsequently authorized completion. Diagnostic messages were added
to existing identity refusals without changing acceptance or read order.
Fresh sampled validation passed 8/8 normal answers and 4/4 full-output pairs;
the native sampling route was verified. Lifecycle passed 52/52 normal answers,
two cancellations, actual C2, slots and release. Cache full outputs matched
24/26 pairs and all 26 final texts matched, retaining the same two index-149
reasoning variations as Sprint 21. A local transfer timeout was recovered with
a reviewed compressed read-only collection; the GPU run was not repeated.
Astra approved the smaller tile for the next experimental session with prior
GEMM rollback and ring-fold unchanged. The initial observation cause, earlier
numerical findings and deferred endurance remain. See [Sprint 22](SPRINT22-GDN-OUTPUT-M16.md) and its
[provenance](../provenance/SPRINT22-GDN-OUTPUT-M16-20261022.json).

## Sprint 23 attention QKV common M16, 22 October 2026

The model-free `(8,7168,5120)` comparison changed only the common kernel's
M tile from 64 to 16, retaining stage three. All 96 exact comparisons passed,
including 24 graph comparisons. Timing was mixed: three cases improved
1.64–8.03%, while GPU 0's random case took 2.60% more time. This fails the
declared requirement for a lower median in all four cases, so the candidate
is parked. The run completed in 23.60 seconds with independent release and
zero model starts; no serving or quality campaign followed.

The selected Sprint 22 GEMM and ring-fold remain unchanged. Its measured
121.18–121.24 combined TP2 tokens/s was the qualified Sprint 22 result; Sprint 23
establishes no new serving rate or timing cause. See [Sprint 23](SPRINT23-ATTENTION-COMMON-M16.md)
and its [provenance](../provenance/SPRINT23-ATTENTION-COMMON-M16-20261022.json).

## Sprint 24 selected-profile cost refresh, 22 October 2026

The refresh retained the Sprint 22 GEMM and ring-fold profile. Two starts and
eight sampled low-thinking requests passed all eight natural final-answer
checks, sampling-route checks and independent release. Each rank completed
all eight selected target timing windows, including four long-case windows.

Long-window projection forward medians were 27.234/27.206 ms within target
graphs of 44.111/44.115 ms on ranks 0/1. Gate/up remained the largest projection
at 11.839/11.845 ms. These intervals include quantization, launches, waits and
enclosed collectives; overlapping groups and ranks are not additive.

Only 1/4 full-output and verification-work pairs matched; 3/4 final texts and
final ID pairs matched. One coding final changed while retaining its functional
pass. The other two differing outputs had exact final answers but different
reasoning. The three unmatched pairs cannot measure instrumentation overhead.
Retain all differences, sparse coverage and the unresolved numerical finding.
This is no new serving-speed result or source selection.

At Sprint 24 closeout, the proposed component was common M16 gate/up with N64
instead of N128, holding K128, stage one and block scales fixed. It was
source-feasible but not yet compiled or prepared. See [Sprint 24](SPRINT24-CURRENT-PROFILE.md)
and its [provenance](../provenance/SPRINT24-CURRENT-PROFILE-20261022.json).

## Sprint 25 gate/up N64, 23 October 2026

The common gate/up N64 candidate passed all 96 exact component comparisons
and reduced component medians by 0.540–0.980% across both GPUs and input
patterns. M16, K128, stage one, group 32, four warps and the original 128-wide
block scales remained fixed.

The forward serving pilot matched all four output-ID and verification-work
pairs, including warmup, at 164 cycles per request. Both ranks authenticated
the intended N128/N64 dispatch. With one warmup excluded and three measured
requests per start, N128's median was 8.464984 s, or 120.968929 combined TP2
tokens/s; N64's was 8.468574 s, or 120.917646 tokens/s. Candidate median
latency was 0.04241% higher. This small single-order result failed the required
improvement gate; it does not establish a significant slowdown. N64 is parked
without reverse, retry, natural-stop candidate screens or source selection.
Independent release passed. The selected Sprint 22 GEMM, ring-fold and retained
121.18–121.24 combined TP2 tokens/s result remain unchanged.

An offline source audit found that the native sampled accept path makes two
default-device RNG draws before its downstream deterministic kernel call.
The retained server seed does not identify the RNG state entering each request.
That state and matched selector inputs/draws remain unobserved, so this finding
establishes neither the cause of Sprint 24's variations nor numerical resolution.
At the Sprint 25 closeout, a small diagnostic of those inputs was proposed.
See [Sprint 25](SPRINT25-GATEUP-N64.md) and its
[provenance](../provenance/SPRINT25-GATEUP-N64-20261023.json).

## Sprint 26 sampled acceptance investigation, 23 October 2026

Two full instrumented traces completed: one repeated ring-fold and one
snapshot versus ring-fold. Across both, all 8,328 observed events, eight
full-output/work pairs and 16 functional answers matched; independent release
passed. Synchronous CPU copies can perturb timing. A cross-phase review found one differing start out of eight, with its raw
probability values unavailable. Earlier first-acceptance differences and a later
sampled branch remain unresolved, and these finite
comparisons do not prove numerical equivalence or ring-fold causality. The
selected `b101` GEMM and ring-fold profile remain unchanged. The historical qualified
Sprint 22 result was 121.18–121.24 combined TP2 tokens/s.
See [Sprint 26](SPRINT26-SAMPLER-STATE.md) and its
[provenance](../provenance/SPRINT26-SAMPLER-STATE-20261023.json).

## Sprint 27 gate/up cost isolation, 23 October 2026

The selected-profile diagnostic completed eight functional answers and four
matching full-output/work pairs, with eight complete device windows per rank.
Independent review approved the bounded results. Across four selected gate/up sites in
long windows, GEMM had a median per-window share of 91.06%/91.00% on ranks 0/1;
activation quantization had 3.22%/3.24%. The residual includes unattributed
work and event effects. This sampled scope is not a new speed benchmark or an
extrapolation to all 64 calls. At Sprint 27 closeout, the proposed next component changed stage one
to stage two for the current gate/up shape; Sprint 28 completed and parked it. See
[Sprint 27](SPRINT27-GATEUP-COST.md) and its
[provenance](../provenance/SPRINT27-GATEUP-COST-20261023.json).

## Sprint 28 gate/up stage two component, 23 October 2026

All 96 model-free exact comparisons passed, but stage two was slower in two
of four device/input timing cases. The candidate failed the all-case timing
gate and is parked without serving preparation or source selection. Independent
review approved the bounded closeout. The proposed CPU examination of retained
compiler artifacts completed in Sprint 29; it does not authorize a GPU run. The selected profile and
latest clean Sprint 22 serving benchmark remain unchanged. See
[Sprint 28](SPRINT28-GATEUP-STAGE2.md) and its
[provenance](../provenance/SPRINT28-GATEUP-STAGE2-20261023.json).

## Sprint 29 gate/up compiler review, 23 October 2026

A CPU-only review of the retained Sprint 28 compiler artifacts identified a
possible exact-shape scalar specialization. Sprint 29 did not compile or run a
candidate; Sprint 30 subsequently tested and parked it. Static instruction
counts do not establish a bottleneck or gain. Independent engineering review
approved the CPU findings and proposal. The selected profile and latest clean
Sprint 22 serving benchmark remain unchanged. See
[Sprint 29](SPRINT29-COMPILER-REVIEW.md) and its
[provenance](../provenance/SPRINT29-COMPILER-REVIEW-20261023.json).

## Sprint 30 gate/up static scalars, 23 October 2026

The model-free candidate passed 96 exact comparisons and all four conservative
compiler proofs, but GPU 0 random-input median latency was 0.709% higher. The
all-case timing gate failed; the candidate is parked without serving
preparation or source selection. Independent review approved the bounded results and parking. See
[Sprint 30](SPRINT30-GATEUP-STATIC-SCALARS.md) and its
[provenance](../provenance/SPRINT30-GATEUP-STATIC-SCALARS-20261023.json).

## Sprint 31 warp screen

W2 passed the component gate, but its forward serving median was 8.482985 s versus 8.481692 s for W4; the lower-median gate failed. The control measured 120.73 combined TP2 tokens/s and W2 120.71. All four output and cycle pairs matched. W2 and W8 are parked; independent review approved the closeout. This small sample does not establish a general slowdown. The selected profile remains unchanged. See [Sprint 31](SPRINT31-GATEUP-WARPS.md).

## Sprint 32 long-context screen

Two selected-profile 32K starts completed 14 normal responses and seven identical output-ID pairs. Strict SRE grading passed 2/12; the owner accepted two lexical exceptions for current experimental use, while the strict failure remains recorded. A retrospective audit found all 12 factual-field sets matched under those exceptions. Actual outputs did not reach the full context-plus-output boundary. See [Sprint 32](SPRINT32-LONG-CONTEXT.md). A separately reviewed 64K device screen followed in Sprint 33.

## Sprint 33 64K long-context screen

Two repeated starts completed 14 normal stops and seven equal output-ID and verifier-cycle pairs. The literal and factual SRE gates each scored 10/12 and failed; two answers used a string for an action-order array that the prompt had not explicitly requested. Both failures remain. Long inputs were 59,999 tokens and outputs were 567–849 tokens, so the full output boundary was not exercised. [Sprint 33](SPRINT33-LONG-CONTEXT64.md) records memory, release and scope; independent review approved the conservative closeout.

## Frozen Qwen3.8 baseline

The selected experimental profile and its bounded results are recorded in the [frozen baseline](QWEN38-FROZEN-BASELINE.md). The 64K grading failures and earlier numerical variation remain unresolved; the next phase is Qwen3.8-Flash-Next feasibility.

## Qwen3.8-Flash-Next feasibility

The [metadata and source preflight](QWEN-NEXT-FEASIBILITY.md) found that fully resident official FP8 tensors exceed combined physical RAM and VRAM. Source paths exist for ROCm, but device execution is unverified. The checkpoint decision and a separately reviewed baseline remain pending; the frozen 27B selection is unchanged.
