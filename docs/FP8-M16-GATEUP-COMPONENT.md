# M16 gate/up component

**Component-tested on both cards; unselected.** The exact C2 gate/up candidate
reduced six kernel median latencies by **7.78–12.05%**. All 144 exact comparisons
passed, including 36 poisoned-output graph checks. Astra independently reviewed
the raw result, compiler artifacts and resource release. The subsequent serving
screen closed as **inconclusive and unselected**, as recorded below.

The compound change selects common M16/stage1 instead of common M64/stage3 only
for `(M,N,K)=(16,17408,5120)` during target C2 graph capture. N128/K128, group32,
four warps, ordered K128 dot/scale/add operations, masks and BF16 output remain
unchanged. This does not isolate a stage-only effect or change every M8 guard.

| Operand pattern | Card 0, control → candidate (ms) | Card 1, control → candidate (ms) |
| --- | ---: | ---: |
| Structured | 0.19592 → 0.17500 | 0.20222 → 0.18252 |
| Seeded random | 0.18364 → 0.16540 | 0.18292 → 0.16868 |
| Adversarial | 0.18098 → 0.16524 | 0.19988 → 0.17580 |

The 96 warmed eager event intervals alternate execution order. Candidate latency
was lower in 45 of 48 paired repetitions; five of six timing ranges overlap.
Candidate relative median absolute deviations reached 9.83%, and the observed
effect depends on execution order. Raw spikes remain retained. This bounded
synthetic component result is not a confidence interval, controlled-cache result
or full-model throughput gain.

Compiler reports show 256 → 114 registers, 49,664 → 16,384 shared-memory bytes and
zero spills. Maximum allocated/reserved tensor memory was 90,842,624/92,274,688
bytes per card. Payload/owner durations were 6.11/26.13 seconds. Independent
process, namespace, device, scope and journal release passed. The existing
invalid-escape `SyntaxWarning` remains in stderr.

The [experimental patch](../patches/sglang-gfx1201-m16-gateup-component.patch)
reproduces the measured component source from b101. It requires an authenticated
target-capture observer scope; it is not an installation recipe. The serving
comparison adds an arm predicate and uses the same numerical kernel bodies.
Exact source, runtime, raw timings and compiler identities are in
[provenance](../provenance/FP8-M16-GATEUP-COMPONENT-20261023.json).

## Serving screen

The control/candidate order completed with authenticated M64/S3 versus M16/S1
gate/up dispatch on both ranks. All 38 wire decodes joined; all eight natural
checks on the frozen two-case corpus passed. Six measured C1 responses matched
output IDs and verifier work, with a candidate/control median latency ratio of
1.000844. Every timed C2 group differed in output and verifier work.

| Instrumented service throughput, combined TP2 | Control | Candidate |
| --- | ---: | ---: |
| C1 | 120.116 tok/s | 119.823 tok/s |
| C2 | 173.776 tok/s | 130.297 tok/s |

The third candidate C2 group needed 363 batch-two replays/commits per rank,
versus 172 for control, and took 23.5138 versus 11.7387 seconds. There were no
singleton tails in that group. These unequal-work timings establish neither a
causal kernel regression nor a hardware-jitter explanation. All timings and
output differences remain retained.

Astra independently reproduced authentication, strict grading and the summary
exactly. Both independent releases passed with no unresolved record or cleanup
errors. The initial decoder refusal occurred before transport because copied
tokenizer metadata was missing; exact sealed donor files restored that metadata
without changing reader implementations.

The mandatory matching predicate failed in the first order. Reverse order is
**NOT RUN**: it cannot repair that failed conjunct. No two-order result is
claimed. Lifecycle, cancellation, cache/slot isolation and combined qualification
remain **NOT RUN** for this unselected candidate. Exact results and seals are in
[serving provenance](../provenance/FP8-M16-GATEUP-SERVING-20261023.json).

Rollback is
non-adoption: selected b101, BF16 target/draft KV, aligned ring-fold, fixed eight,
low thinking and 2048-token chunks remain unchanged. Historical numerical
findings and production-qualification limits remain open.
