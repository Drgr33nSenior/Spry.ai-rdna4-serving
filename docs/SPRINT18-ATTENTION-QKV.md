# Sprint 18: attention QKV projection

Status: closed with the option unselected, after Astra review. Both predeclared
serving median gates pass, but overall service benefit remains inconclusive.
Keep selected Sprint 17 source `3f28660b26d0…`. Greedy quality passed 8/8 final
answers; prepared sampled and lifecycle packets were not launched. Installed
runtime and persistent defaults remain unchanged.

## Exact component scope

Compare `(M,N,K)=(8,7168,5120)`: selected common tile64/stage3 versus native
unrolled tile16/stage1. The pinned attention configuration includes its output
gate. At TP2/DCP1, local Q/gate, K and V widths are 6144, 512 and 512. This is
a source/configuration derivation, not a live model weight-shape witness.
Sprint 11's mixed stage-only trial used the common kernel in both arms; this
comparison also changes the kernel and M tile.

The candidate adds one dispatcher branch for untuned HIP/BF16, block128 and
the expected common configuration. It preserves selected down and GDN qkvz
unrolled routes, gate/up and GDN output stage one, prefill overrides, the stack,
caches, flags and owner limits. Native kernel bodies and nine protected helpers
remain unchanged. The inherited guard has no explicit architecture check;
qualification is limited to the two existing R9700 gfx1201 cards.

All 43 CPU checks and the bounded parameter review passed. The model-free run
passed four structured/random cases across both cards: 96 finite/exact checks,
including 24 graph comparisons, and 64 alternating eager timings. Each device
used an explicit capture stream. Analytical structured oracles, random baseline
equality, canaries and operand immutability remained required without tolerance.

Kernel time fell 16.25–18.25% across all four cases. The owner completed in
22.80 seconds, with zero model starts and 37,916,672 peak allocated device bytes.
Cards ran sequentially with synthetic operands. These component results do not
establish TP2 serving speed or model quality. Independent release passed.

## Matched serving results

Astra approved one forward pilot after 22 packet, 35 startup and 14 reader
checks, plus six independent shape tests. Reverse order passed 23 packet and
35 startup checks; its identical reader reused the 14 checks. Parent review
confirmed the order-only change and its generated specification binding.

Both arms received the same addition to the once-per-shape metadata witness.
The reader required attention QKV on both ranks in both arms, with its expected
kernel, tile and stage. Existing down or GDN witnesses could not substitute.
Source, stack, weights, BF16 caches, workers, wire and release authenticated.

| Order | Reference median | Candidate median | Median latency change | Candidate combined TP2 rate |
| --- | ---: | ---: | ---: | ---: |
| Reference, candidate | 8.846055 s | 8.826263 s | −0.2237% | 116.0174 tokens/s |
| Candidate, reference | 8.841318 s | 8.824091 s | −0.1948% | 116.0459 tokens/s |

All eight output-ID pairs matched, including warmups. Every request used 164
verification cycles, graph replays and completed commits per rank. Workload:
C1, 2047 input tokens, 1024 forced output tokens, greedy sampling and low
thinking. Exclude the first request per start; retain three measured requests
per start. Rates include prefill and represent the combined two-card service,
not isolated decode or final-answer throughput.

Five of six measured candidate requests were faster. The third reverse
candidate request took 8.963737 seconds versus 8.848564 for its reference:
1.3016% slower, with matching output and cycles. Retain this timing outlier.
Forward mean latency fell 0.2101%; reverse mean rose 0.3250%. The pooled means
were 8.844015 seconds for reference and 8.849088 for candidate, a 0.0574%
increase. Both predeclared median gates remain PASS. These small samples do not
establish statistical significance, a general service benefit or a causal
regression. No hardware explanation is established.

## Bounded quality and disposition

The already active greedy screen completed two fresh candidate starts and eight
natural-stop requests. All eight final-answer gates passed. All four paired
final texts matched; three of four full output-ID pairs matched. Retain the
`quality16k-audit-threshold` reasoning variation beginning at index 553:
2093 versus 2126 reasoning tokens, followed by identical 763-token finals.
Both held-out finals contained 821 tokens. All four long ledger responses
therefore exceeded the required 512 final-answer tokens.

Preparation retained the Sprint 17 requests, input IDs, task oracles and grading
rules. Checks passed: greedy 18 packet/35 startup/15 reader, sampled 18/35/16,
and lifecycle 65 packet/12 reader. The greedy reader additionally passed three
existing strict author-seal metadata checks. Independent release passed.

Astra independently verified both sealed serving results and recommended
leaving the option unselected. The small typical-latency signal gives
inconclusive evidence of an overall service benefit; further qualification cost
is not justified under the owner's usage constraint. The greedy screen finished
normally; the prepared sampled/lifecycle screens remain sealed and unlaunched,
with zero model starts or requests. No candidate sampling, C2, reuse/cancellation
or lifecycle qualification is claimed.

The [optional candidate patch](../patches/sglang-gfx1201-m8-attention-qkv-unrolled-tile16-serving.patch)
reproduces the exact tested serving candidate from selected Sprint 17 source
in a CPU fixture. Retain it for a separately scoped future assessment.

## Evidence limits and retained preparation refusals

The reverse parent review initially refused byte equality for the generated
controller specification hash. Exact comparison confirmed that only its bound
hash changed. Lifecycle preparation initially omitted the retained native
prompt-geometry report; restoring the identical sealed donor file made all 65
checks pass. Original refusals remain. No request or acceptance rule changed.

The completed greedy run's authorization carried a five-field author closeout,
while its inherited reader expected three identity fields. Preserve the original
reader refusal as a sealed root. A fresh reader reused the already reviewed
strict metadata adapter, requiring matching identity, UTC metadata and false
qualification. Three negative/compatibility checks passed. Authentication,
decoding and grading then completed without a model rerun. The future local
preparation helper now emits the legacy identity-only authorization field;
full author closeouts remain in provenance. Executed evidence was not modified.

Actual scope: one model-free component and three two-start model campaigns,
six model starts, 16 forced-work requests and eight natural-stop requests.
Preserve earlier numerical findings, Sprint 11 failed work gates and the owner
session-use disposition. Full state/distribution equivalence remains unverified;
reliability remains deferred. No GPU campaign is queued.

See the [evidence index](../provenance/SPRINT18-ATTENTION-QKV-20261022.json).
