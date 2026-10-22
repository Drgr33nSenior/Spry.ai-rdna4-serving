# Optimization screening before Sprint 7

Owner decision, 21 October 2026: work through Astra's optimization suggestions
before selecting one or more configurations for Sprint 7. Keep M2047 and
metadata reuse in the current session reference. Keep Torch 2.14 / ROCm SDK
10.2, model weights and recurrent-state precision unchanged. Further diagnosis
of the recorded output variations remains deferred unless concerning results
recur.

A source audit, a component pass and a serving pass establish different facts.
Record each separately. An incompatible or blocked candidate is not GPU-tested.
Do not call all suggestions tested while a row still lacks the stated evidence.

| Candidate | Evidence already retained | Remaining screen |
| --- | --- | --- |
| Metadata reuse | CPU lifecycle/replay checks and bounded native C1 pass; one captured build across sixteen attention layers per rank. Session selected. | Additional causal speed gain remains unproven. Preserve current selection; broader lifecycle coverage belongs to the eventual Sprint 7 profile. |
| M2047 stage 1 | 360 exact component comparisons, serving comparisons in both orders and session activation screen. Session selected, with exact-output exceptions recorded. | Retain the owner decision and full result history when selecting the combined profile. |
| Fixed speculation width | Fresh current-stack C1 comparison used equivalent singleton split attention and metadata reuse at widths four and eight. Width four had 36.07% higher median latency. | Keep width eight. Different output prefixes prevent an exact-output equivalence claim; task quality was not graded. |
| Adaptive verification | Active selector and graph sources authenticated. Ten CPU cases exercised the actual native width guards. Retained telemetry lacks per-position confidence. | Park the unimplemented policy. It needs joined lattice scores and prefix labels, diverse evaluation inputs, and graph support for changing widths. This was not a GPU policy test. |
| ReplaySSM | The aligned per-layer component passed 1,032 exact comparisons; a separate production all-layer fold bridge passed 1,808 exact finite comparisons on both GPUs. Matched-capacity C1 serving completed both orders, with 1.63 GiB lower startup allocation per card and 2.32–2.47% lower latency. | Seven of eight output pairs matched; retain the known reference capitalization exception. Carry as a second Sprint 7 profile; broader reuse, quality and lifecycles remain incomplete. Direct enablement of the original compact path remains parked. |
| Target shard maxima | Native current-path FP32 gather geometry observed on both ranks. | Timing screen complete: gather median ~0.157 ms within ~43.9 ms target graph per rank (~0.36%). No shard-max implementation selected for this short C1 path. Longer context/C2 remains unmeasured. |

## Screening order and selection

1. Measure the current head and gather intervals. Keep instrumented timings
   separate from native request benchmarks. Retain rank and request joins,
   missing samples and output differences.
2. Test the smallest compatible ReplaySSM component without loading a model.
   Compare aligned verification plus accepted-prefix fold against snapshots;
   measure both memory and added work. Preserve every mismatch.
3. Close the width-four attention compatibility gap, then use one matched
   current-stack screen. Assess adaptive features separately; truncating a
   width-eight trace does not predict a native width-four drafter.
4. Select the supported configurations from the measured results. Bring those
   configurations into Sprint 7 for normal-stop quality, context, C1/C2,
   sampling, cache reuse, cancellation and repeated lifecycle coverage.

No candidate is selected for production by this work. Current session selection
is recorded in [CURRENT-SESSION-SELECTION.json](../provenance/CURRENT-SESSION-SELECTION.json).

## Shortlist after the bounded screens

Keep the current M2047 stage 1, metadata reuse and DFlash8 configuration as the
Sprint 7 reference, with the selected stack, precision and pools unchanged.
Retain the recorded output exceptions; broader qualification is incomplete.

The aligned ReplaySSM serving candidate passed its bounded C1 integration
screen. Carry it as a second Sprint 7 profile alongside the current reference.
Matched-capacity memory and both execution orders are now recorded in the
[Sprint 7 results](SPRINT7-QUALIFICATION.md). Complete broader quality/lifecycle
checks before qualification. Keep width four
and shard-max changes out of the current shortlist. Adaptive policy work
remains unimplemented and untested; its prerequisite audit does not qualify it.

## Current-stack width comparison

The final packet passed 48 CPU checks and completed eight native requests.
Both arms used M2047 stage 1, the same precision, pools and frozen prompt,
and per-row singleton split attention with metadata reuse. Source, runtime,
loaded-weight, HTTP, graph and independent release checks passed.

| Configuration | Median request latency | Combined TP2 output rate | Verification cycles/request |
| --- | ---: | ---: | ---: |
| DFlash8 | 1.8474 s | 69.29 tokens/s | 25 |
| DFlash4 | 2.5137 s | 50.92 tokens/s | 39 |

One warm-up per fresh start is excluded from these medians. Width four was
36.07% slower in latency and 26.51% lower in throughput in this finite,
one-order C1 screen. The two arms each produced a consistent 128-token prefix,
but those prefixes differed at 109 positions, starting at index 16. Preserve
that difference without inferring its cause or grading task quality.

Two earlier packets were incomplete: one launch-command width mismatch and
one historical width-eight evidence check. Both refused before external
candidate requests and released independently. The final packet rendered
both width flags from the contract and validated native startup records for
both widths. See the [full width record](../provenance/WIDTH48-CURRENT-STACK-20261021.json).

## Aligned ReplaySSM component

The revised component keeps the aligned sequential verifier and native exact
fold. Its only kernel-source change extends BF16 beta alignment to ring-only
recording. All 1,032 output/state/ring comparisons passed exactly across
sixteen cases: both GPUs sequentially, batches one and two, structured and
random inputs, and BF16/FP32 raw rings. Tests include committed prefixes zero
through eight, poisoned rejected tails, slot-zero/null/permuted indices,
rotated ring fixtures, graph repeats and chained cycles. Thirty CPU checks
passed. Native owner duration was 30.88 seconds, with independent release.

The synthetic per-layer snapshot allocation was 12 MiB at batch one or
24 MiB at batch two. The three-slot raw ring used 201,216 bytes in BF16 or
397,824 bytes in FP32. These storage formulas are not full-model VRAM savings.
Measured Torch allocator peaks for the complete test fixtures were
28,651,008–43,135,488 bytes.

The baseline component took 1.67–2.29 times the ring component's event time.
The baseline commit uses Torch gather/index_copy rather than the fused serving
commit, so this does not establish a serving speedup. The initial packet
stopped during second-GPU graph capture after eight exact cases; the corrected
packet used explicit per-device capture streams and completed both GPUs.
See the [component record](../provenance/REPLAYSSM-ALIGNED-COMPONENT-20261021.json).

## Production all-layer fold bridge

The production commit uses a different native kernel and launch from the
per-layer component above. A separate bridge passed 35 CPU checks and all
1,808 exact finite comparisons across eight cases on both GPUs. It covers two
layers, physical slot mappings, unequal accepted prefixes, optional tracking,
poisoned rejected tails, graph replay and chained cycles. Peak Torch device
allocation was 97.21 MiB within the 256 MiB fixture limit. Independent owner
release and retained source/device evidence passed.

This closes the production-fold numerical prerequisite. It does not execute
GPU convolution rollback or qualify full-model serving, radix lifecycles,
quality or memory savings. No component timing comparison was made. See the
[production-fold record](../provenance/REPLAYSSM-PRODUCTION-FOLD-BRIDGE-20261021.json).

## Bounded ReplaySSM serving screen

Astra reviewed the implementation at max reasoning. The corrected package
passed 43 CPU checks, including the unchanged owner archive validator, and
completed two starts with four requests each. Both arms retained FP8 weights,
FP32 recurrent state/head, BF16 KV/draft, DFlash8, metadata reuse and M2047
stage one. Each request had 2,047 input tokens and 128 forced output tokens.
One warmup per start is excluded from the medians.

| Configuration | Median request latency | Combined TP2 output rate | Usable recurrent slots per card | Startup Torch allocation per card |
| --- | ---: | ---: | ---: | ---: |
| Snapshot reference | 1.8490 s | 69.23 tokens/s | 19 | 20.53 GiB |
| Aligned ring-fold | 1.8042 s | 70.95 tokens/s | 74 | 23.01 GiB |

The observed latency fell 2.42% and output rate rose 2.48% in this one-order
screen. All four paired output vectors matched exactly. Every request had
25 verifications, graph replays and completed native commits per rank, with
no recapture. Both processes released independently.

Automatic sizing converted the removed snapshot allocation into more
persistent capacity. Startup allocation therefore rose by 2.48 GiB per card.
This screen does not establish total VRAM savings at matched capacity. The
main tensor formula leaves 938,496 bytes of the per-card allocation delta
unassigned; preserve that residual without assuming its cause. Startup peaks
also do not establish process-lifetime or whole-device peaks.

The implementation fixes the native model guard, persistent-slot ring sizing,
accepted-prefix commit and ring/convolution budget. It retains convolution
rollback and the following PLE update. The first packet refused before grant
at the unchanged archive member ceiling; its evidence is retained. The fresh
packet reclaimed inactive assets and passed the real archive validator before
launch. Installed runtime and current session reference remain unchanged.
See the [complete serving record](../provenance/REPLAYSSM-SERVING-20261021.json).

## Adaptive-policy prerequisite audit

The active selector computes candidate-lattice scores, then writes one-hot
proposal rows for greedy decoding. Those rows are not confidence estimates.
The retained request records contain acceptance totals and histograms but no
per-cycle lattice scores. Both native graph runners reject a width different
from their capture width and fall back to eager execution.

A future policy needs pre-one-hot conditional scores joined to accepted-prefix
labels, evaluation across diverse prompts, and explicit graph/capacity/metadata
integration. Labels after the first rejection are invalid for the realized
history. Truncating a width-eight trace does not predict a native width-four
drafter or save width-eight graph computation. No adaptive policy was
implemented or GPU-tested. See the [audit](../provenance/ADAPTIVE-FEASIBILITY-20261021.json).

## Target head/gather screen result

The current-stack packet passed 35 CPU controls and completed both native arms
with authenticated source, runtime, weights, HTTP and independent release. All
four output pairs matched, including warm-up. Each rank retained 99 complete
external-request cycle intervals: 25 for each of the first three requests and
24 for the final request. The final replay103 was not collected before exit;
no extra replay or synchronization was added to obtain it.

| Instrumented per-rank interval | TP0 median | TP1 median |
| --- | ---: | ---: |
| Whole target verification graph | 43.8688 ms | 43.8677 ms |
| FP32-output head computation | 2.0513 ms | 2.0512 ms |
| TP gather completion | 0.1568 ms | 0.1566 ms |

Median per-cycle gather/graph fractions were 0.3575 and 0.3569 percent. These
are instrumented device-stream intervals, including waiting. They are not
pure transport times, whole-request fractions or independent quantities to
add across ranks. Existing communication settings remain selected for the
screened C1, 2,047-input/128-forced-output workload. The gather is too small in
this screen to justify a target shard-max integration. This does not settle
long-context or C2 behavior. See the
[complete timing record](../provenance/HEAD-GATHER-TIMING-20261021.json).
