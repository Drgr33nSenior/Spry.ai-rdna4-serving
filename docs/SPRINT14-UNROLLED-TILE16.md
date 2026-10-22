# Sprint 14: smaller unrolled down-projection tile

Status: the component and both matched serving orders passed. Candidate latency
was 1.74–1.79% lower. All 68 normal final answers, cancellations, actual C2
and slot/release gates passed. Astra approved the source for the next
experimental session launch.
The selected source is `f90f4779c13c…`; retain `7c8fa3cb3316…` for rollback.

[Sprint 13](SPRINT13-UNROLLED-STAGE1.md) showed that unrolled stage one
fits shared memory but takes 47.08–52.21% more time with an M tile of 64.
The compiler reported 57 spills. Test an M tile of 16 as a separate candidate;
do not assume that fewer tile rows will remove spills or improve performance.

Compare common tile64/stage3 against native unrolled tile16/stage1 at exactly
`(M,N,K)=(8,5120,8704)`. Keep N/K tiles at 128, group size 32 and four warps.
Kernel definitions, FP8 operands, FP32 scales/accumulation and BF16 output stay
unchanged. This is a bundled kernel, stage and tile comparison against the
selected route; it does not isolate any one change. K is divisible by the
unrolled step of 512, excluding partial unrolled steps.

The private dispatcher changes only this exact untuned HIP/BF16 shape. Preserve
tuned configurations, other shapes, platforms and dtypes, including the
selected 3072, gate/up and prefill overrides. Leave the global selector's
missing return unchanged. Evidence applies only to the two R9700 gfx1201 cards.

Reuse the reviewed component harness and all nine protected helpers. This
parameter-only change received a scoped implementation review and 37 CPU
checks, including wrong-tile and wrong-stage refusal. No new Astra review was
requested. Fresh compiler/boot/config observations passed. Require a fresh
execution copy to match the tested source before launch.

Require 96 exact comparisons, including 24 graph replays, and 64 alternating
eager timed launches across both cards and structured/random inputs. Retain
the independent structured oracle, random baseline equality, poisoning,
canaries, input hashes, explicit per-device graph streams, allocation guard,
compiler metadata and independent release checks. No tolerance relaxation or
automatic retry follows a failed bounded campaign.

Keep the 1,200-second active, 600-second worker, 30-second cleanup and
1,500-second total limits, with the 256 MiB device-tensor ceiling. No model
starts, installed-runtime changes or source selection are part of this
component. Preserve the current stack, BF16 caches, flags, pools and unresolved
numerical contract for any later serving comparison. Reliability remains
deferred by owner direction.

Advance to separately reviewed serving preparation only if all exact checks
pass and the candidate median time is lower in every case on both cards.
Otherwise retain and park the result. See the
[evidence index](../provenance/SPRINT14-UNROLLED-TILE16-20261022.json).

## Component result and serving gate

Both source and fresh execution copy passed 37 CPU checks. The component
completed all four cases, 96 exact comparisons and 64 alternating eager timed
launches, with independent release. It starts no models. The compiler reports
153 registers, zero spills and 16 KiB shared memory for tile16/stage1.
The earlier tile64/stage1 configuration reported 256 registers and 57 spills.
These are compiler resource reports, not measured occupancy or a causal
attribution of timing changes.

| Card | Pattern | Common tile64/stage3 median, ms | Unrolled tile16/stage1 median, ms |
| --- | --- | --- | --- |
| 0 | Structured | 0.080980 | 0.057540 |
| 0 | Random | 0.079760 | 0.059160 |
| 1 | Structured | 0.080860 | 0.060310 |
| 1 | Random | 0.084175 | 0.059060 |

The result meets the predeclared component advancement rule. It does not
establish serving throughput or task quality. Prepare one forward matched
serving pilot with two starts and eight requests: 2,047 input tokens, 1,024
forced output tokens, low thinking and greedy sampling. Exclude each start's
first request; compare three measured requests per arm. Preserve all existing
flags, precision, caches, pools, observers and resource limits.

Both arms add the same metadata-only down-shape witness. Require the selected
baseline to use common tile64/stage3 and the candidate to use unrolled
tile16/stage1 on both ranks. Keep selected 3072 and gate/up at common
tile64/stage1. The inherited witness's `native` field records whether the
selected kernel is the common implementation; it is false for the unrolled
down shape. Source binding and witness checks must agree.

Preparation passed 22 packet, 35 startup and 10 reader checks. An inherited
startup fixture initially expected the older baseline. Its exact expectation
now uses the sealed selected component source plus the same witness extension;
the failed receipt and original fixture remain retained. Runtime code, request
bodies and acceptance gates were not changed to address that fixture failure.

Stop after the pilot if any measured output-ID or verifier-count pair differs,
or if candidate median client latency is not lower. Retain the result without
an automatic retry. A clean pilot win requires reverse-order and affected
quality/cache validation before selection. Prior numerical and matched-work
exceptions remain unresolved; the owner's acceptance of 3072 does not waive
these gates for the new candidate.

## Both-order serving result

Both orders passed all eight output-ID pairs and matched 164 verifier cycles
per request, including warm-ups. Both ranks reported the expected down route
and preserved the selected 3072 and gate/up routes. Independent release passed.
Each order had two starts and eight requests; the first request per start was
excluded from timing. These are median client request times, including prefill,
for C1 with 2,047 input tokens and 1,024 forced output tokens.

| Order | Reference, s | Candidate, s | Reference, output tokens/s | Candidate, output tokens/s | Latency change |
| --- | --- | --- | --- | --- | --- |
| Reference then candidate | 9.053030 | 8.895131 | 113.1113 | 115.1192 | −1.7442% |
| Candidate then reference | 9.054416 | 8.892783 | 113.0940 | 115.1496 | −1.7851% |

Rates are combined TP2 service throughput across both cards. These forced-output
requests do not grade natural-stop task quality or measure decode-only speed.
The small matched screen does not establish general performance or resolve
earlier numerical variation.

The first forward readout refused the author's five-field seal metadata because
the inherited reader expected its three identity fields. All content hashes
and membership verified. A narrow wrapper validates the exact metadata schema
and preserves the original full seal verification; 13 checks and Astra review
passed. Retain the refusal and original reader. Reverse readout used the same
corrected reader and passed directly. New quality authorization records use the
original three-field identity schema. The quality readers retain all checks,
with their exact GEMM identity pin updated to the tested candidate.

The quality packets substitute only the tested candidate source and its source
bindings into the existing screens. Both logical arms use the candidate; these
are bounded task/cache checks against fixed oracles and historical references,
not a contemporaneous quality A/B. Frozen requests, grading and owner controls
remain unchanged. The generated controller specification digest changes with
the candidate-containing specification; all other helper bytes and controller
logic remain unchanged. Preparation initially refused an overstrict byte
comparison, before sealing or upload; the corrected check permits only that
exact specification-digest substitution.

Greedy offline authentication first refused the old source pin inherited from
the previous candidate. Native execution and release had completed successfully.
The corrected greedy, sampled and lifecycle readers require the new exact pin;
all other authentication code is byte-identical. Isolated checks accept that
pin and reject the old and arbitrary pins. All three reader suites passed again.
The original greedy reader and failed receipt remain retained; only readout
resumed, with no GPU rerun or gate relaxation.


## Natural-stop task and lifecycle results

Greedy and publisher-sampled screens each passed eight of eight final answers
and normal stops across two fresh candidate starts. Each lane uses four unique
tasks repeated twice. Greedy requests keep thinking enabled at low effort,
temperature zero and top-k one. The sampled lane retains temperature 1.0,
top-k 20 and top-p 0.95; its authenticated native selector route passed on both
ranks for every external request. This does not test sampling-distribution
equality.

Three of four greedy full-output pairs match. The remaining pair differs in
reasoning length, 301 versus 329 tokens; all four final-answer pairs match
exactly. All four sampled full-output pairs match. No output difference is
removed from the evidence. Final answers were decoded after the authenticated
closing thinking marker and checked against the existing fixed task oracles,
including isolated execution of generated Python. Forced-output timing results
remain separate from these natural-stop quality checks.


Lifecycle validation passed 52 of 52 normal final answers across two fresh
candidate starts, two cancellations with recovery, two actual C2 groups and
both native group and slot accounting. All 26 paired cache final answers match;
24 full-output pairs match. Preserve the two full-output differences. Each
concurrent batch-two path uses native attention rather than the singleton split
adapter. Numerical state equality and general repeatability remain unqualified.

All campaigns passed source/runtime/worker/weight, wire and independent release
checks. Sprint 14 started ten models and made 86 requests: 16 fixed-work timing
requests and 70 quality/lifecycle requests, including two cancellations.
The component itself started no models. Reliability and endurance remain
deferred by owner direction. No installed runtime, persistent server or
production default changed.

The selected [source patch](../patches/sglang-gfx1201-m8-down-unrolled-tile16-serving.patch)
applies to the exact previously selected `7c8fa3cb3316…` source after the existing 3072
patch. Applying it in a fresh private copy reproduces the serving-tested
`f90f4779c13c…` bytes. Astra independently verified all 17 seals, result pins,
exact patch application and the source-pin correction, with no blocking findings.
The source is selected for the next experimental session launch. Retain the
previous selected source as rollback, with all earlier numerical and matched-work
exceptions unchanged.
