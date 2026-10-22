# Sprint 11: remaining projection tuning

Status: bounded trials closed; no new source selected. Five native component
candidates passed all 480 exact comparisons. Four serving campaigns completed
with independent release. Tile32 failed output/work parity. The 3072 stage-one
candidate showed a consistent timing signal, but its predeclared forward
confirmation failed the clean-work gate. All candidates are parked.
The selected serving source remains `cafc15b03fbb…`; its recorded throughput
remains about 111.4 output tokens/s for the combined TP2 service.

The owner directed tuning to continue before reliability and endurance testing.
Preserve bounded correctness checks for each optimization; defer extended
reliability campaigns. The earlier numerical mismatch remains unresolved under
the existing experimental-session disposition.

## Current measured throughput

The selected Sprint 10 configuration measured **111.38–111.44 output tokens/s
for the combined two-GPU TP2 service**, at concurrency one. Forward median was
111.44082 tokens/s; reverse median was 111.38218 tokens/s. Each fresh start had
one excluded warmup and three measured requests.

These requests used 2,047 input tokens and 1,024 forced output tokens, thinking
enabled at low effort and greedy sampling. Output counts include reasoning.
The denominator is complete client request time, including prefill and HTTP
completion. These are not per-card or isolated decode rates. Short-output and
long-context natural-stop results use different workloads and are not directly
comparable. See the [Sprint 10 results](SPRINT10-M8-SERVING.md).

## First candidate: M8 down-projection stage depth

Authenticated Sprint 10 logs show missing R9700 tuned configurations for the
five recorded projection N/K pairs. Gate/up already uses the selected stage-one
override. The first comparison used the MLP down projection
`(M,N,K)=(8,5120,8704)`: selected default stage three versus stage one.
It retained tiles `(64,128,128)`, four
warps, block scales, kernel body, arithmetic and output dtype unchanged.

The candidate adds this exact shape to the existing untuned HIP/BF16 M8 guard.
It preserves gate/up stage one, every existing M2047 override, tuned entries
including default-valued entries, other shapes and non-HIP/other-dtype routes.
The inherited guard does not explicitly check architecture; qualification is
limited to the two existing R9700 gfx1201 cards.

The source audit verified the selected baseline, component-template and native
dispatch evidence seals. A fresh read-only host observation confirmed the
current compiler-source hashes and absence of the five device-specific tuned
configurations. No native runtime was imported on the Mac.

## Component and serving gates

The first packet reuses the corrected model-free component owner. Test both
cards sequentially with structured and random synthetic operands: four cases,
72 eager comparisons and 24 graph-replay comparisons. Use eight alternating
timed launches per variant/case. Retain the analytical structured oracle,
random baseline equality, output poisoning, canaries, input immutability and
explicit per-device capture streams. Graph replay checks exactness; its
performance is not benchmarked by this packet.

Keep the existing 1,200-second active, 600-second worker and 1,500-second total
envelope, with 30 seconds for worker cleanup. The device-tensor ceiling remains
256 MiB. Protected owner/identity/resource helpers and the current stack remain
unchanged. A component result establishes neither task quality nor serving gain.

Advance a measured component win to fixed-work matched serving and reverse
order, followed by the affected bounded quality/cache checks. If it fails parity
or provides no useful gain, retain the result and choose the next bounded shape
or configuration. Do not adopt a blanket stage override or reserve a sweep.

Preserve FP8 weights, BF16 target/draft KV, FP32 head/state, snapshot, DFlash8,
metadata reuse, existing M2047 and M8 gate/up overrides, pools, graphs and flags.
Ring-fold combination, FP8 target KV, adaptive speculation and C2 throughput
remain separate candidates. Installed runtime and production defaults remain
unchanged.

The [preparation index](../provenance/SPRINT11-PROJECTION-TUNING-20261022.json)
records source hashes, donor seals, exact limits, CPU checks and pending gates.

The owner explicitly reaffirmed the existing scoped sudo approval for the
component owner, source protection, independent observer and evidence collection.
Continue through that finite procedure without requesting the same approval again.

## First component result

Down projection `(8,5120,8704)` passed all 96 exact comparisons and independent
release. Stage one was 0.65–8.65% slower across four synthetic cases on the two
cards, so retain stage three. The owned run took 22.79 seconds and started no
models. This is a component decision; current serving throughput is unchanged.

The `(8,8192,5120)` stage-depth comparison also passed 96 exact comparisons
and independent release in 20.69 seconds. Stage one ranged from 1.98% slower
to 1.85% faster; this mixed small result is parked. Its sealed dispatch proof
retains inherited down-projection wording. Its explicit shape, source, driver,
tests and native results bind `(8,8192,5120)`; this correction leaves the sealed
evidence unchanged.

## Tile32 component result

Gate/up tile32 at fixed stage one passed all 96 exact comparisons. It reduced
synthetic component median time by 0.33–1.10% across four cases on both cards;
the geometric-mean speedup was 1.00844 times. Independent release passed after
22.42 seconds. This earned a fixed-work matched serving pilot.
The pilot retains 2,047 input tokens and 1,024 forced output tokens, one excluded
warmup and three measured samples per start. It passed 21 packet checks and
35 startup checks; the reader passed seven focused checks.

The pilot completed two starts and eight requests, with authenticated source,
runtime, HTTP, cache, counters and independent release. Reference and candidate
medians were 111.19075 and 110.88663 output tokens/s. These rates are descriptive:
the clean speed comparison was ineligible. Reference measured requests each
used 164 verification cycles; candidate requests used 166, 163 and 165.
Two of three measured output-ID pairs matched. The remaining pair first differed
at index 124 (`2483` to `12296`) and differed at 691 positions across the full
1,024-token continuation. That full difference is not only capitalization.

Park tile32. The pilot establishes neither a causal slowdown nor a serving
gain. Astra reviewed this disposition. No candidate quality, reverse-order or
endurance campaign followed this unsupported pilot, and no source was selected.
The earlier numerical cause remains unresolved.

## Remaining component results

Each comparison below passed 96 exact checks and 64 alternating timed launches
on the two cards, with independent release. Inputs were synthetic, and cards
ran sequentially. Positive time reduction means the candidate was faster.

| Exact shape `(M,N,K)` | Change | Component time reduction across four cases | Decision |
| --- | --- | --- | --- |
| `(8,5120,8704)` | Stages 3 to 1 | -8.65% to -0.65% | Park; retain stage 3 |
| `(8,8192,5120)` | Stages 3 to 1 | -1.98% to +1.85% | Park mixed result |
| `(8,17408,5120)` | Row tile 64 to 32; stage 1 | +0.33% to +1.10% | Park after ineligible serving comparison |
| `(8,7168,5120)` | Stages 3 to 1 | -6.22% to +3.94% | Park mixed result |
| `(8,5120,3072)` | Stages 3 to 1 | +2.13% to +9.80% | Park after predeclared forward confirmation failed its work gate |

The 3072 component geometric-mean speedup was 1.05980 times. Its owned run took
21.55 seconds and started no models. Compiler reports changed from 256 to 238
registers and 49,664 to 16,384 shared-memory bytes; both variants reported zero
spills. These reports do not measure hardware occupancy.

The second serving pilot changes only the exact untuned `(8,5120,3072)` stage
depth. Both arms retain row tile 64, gate/up stage one and the existing M2047
overrides. Both receive the same metadata-only witness extension so the reader
can require the actual changed shape on both ranks. It retains the same frozen
2,047-input/1,024-output thinking-low requests. Preparation passed 22 packet,
35 startup and eight reader checks; Astra approved the bounded two-start pilot.

That forward pilot completed eight requests and independent release in
338.76 seconds. All four output pairs matched. Reference and candidate medians
were 9.200448 and 9.040335 seconds, or 111.29892 and 113.27013 output tokens/s.
These are descriptive results: median latency was 1.7403% lower and throughput
was 1.7711% higher. Both matched-cycle measured pairs were faster; the remaining
candidate request was also faster but used 165 cycles versus 164. Retain all
three samples. The aggregate clean-comparison gate remains failed.

Astra reviewed the authenticated result and recommended one reverse-order
exploratory confirmation. Reversing the order does not retroactively qualify
the forward run or alone justify source selection. The reverse packet changed
only the frozen execution order, retaining requests, source pair and limits.
Its inherited startup fixture initially expected the forward order; that failed
receipt is retained, and the corrected reverse-order fixture preserves all
capacity and native-binding checks.

The reverse run passed its clean-work gate. All four output pairs matched,
and every request used 164 verification cycles. Astra then required one
predeclared forward confirmation before the existing affected B/B quality
packets. The stop condition was explicit: park on any physical, authentication,
output or work-gate failure, with no further automatic confirmation.

| 3072 comparison | Reference tokens/s | Candidate tokens/s | Descriptive median latency change | Clean-work gate |
| --- | --- | --- | --- | --- |
| Original forward | 111.29892 | 113.27013 | -1.7403% | Fail: candidate measured cycles 164/165/164 versus 164/164/164 |
| Exploratory reverse | 111.10351 | 113.09484 | -1.7608% | Pass: all requests 164 cycles |
| Predeclared forward confirmation | 111.35889 | 113.25937 | -1.6780% | Fail: reference measured cycles 167/164/164 versus 164/164/164 |

All 12 output pairs matched across these three comparisons. Seven of nine
measured cycle-count pairs matched. Keep every sample, including both failed
comparisons. The consistent 113.09–113.27 tokens/s signal remains descriptive;
the matched-work requirement in both orders did not pass. Astra verified the
confirmation and the stop-and-park decision. No causal serving gain or task
quality qualification is claimed for this candidate.

The greedy, sampled and lifecycle B/B packets passed CPU preparation but never
launched. Their conditional approval depended on a clean confirmation. Greedy
and sampled admission correctly refused while the confirmation owned the
diagnostic marker; those receipts remain. The lifecycle source donor lacked
the admission helper held by its native-run donor. The missing-file refusal,
identical helper addition, refreshed checks and unused launcher are retained.
No admission refusal was bypassed.

Astra also corrected an inherited sampled-reader label from a single "timed
arm" to separate results for both repeated candidate starts. A fixture with one
passing start and one failing start confirms that reporting and the unchanged
global gate. Those prepared readers collected no native results.

Sprint 11 completed nine physical campaigns: five model-free component runs
and four two-start serving campaigns, with 32 fixed-work requests in total.
All physical campaigns and independent release checks passed. Local fixture and
receipt corrections remain in the evidence. No installed runtime, persistent
server or production default changed. Extended reliability remains deferred.

## Next source opportunity

A CPU source audit found that the pinned HIP
`use_w8a8_block_fp8_matmul_unrolledx4` helper computes its workgroup predicate
without returning it. The actual helper/selector AST chose the common kernel
in four declared CPU cases. This is a source observation, with a substituted
core count, not a native device or performance result. The intent is unknown.

The next bounded tuning candidate is an exact-shape comparison of the common
kernel and the existing native unrolled kernel. Do not enable the alternate
route globally on this evidence. No such implementation or GPU run has started.
Reliability and endurance remain deferred by owner direction.
