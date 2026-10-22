# Sprint 15: smaller unrolled gate/up tile

Status: complete; park the candidate. All component and serving output/work
checks passed, but the tiny forward timing improvement did not hold in reverse.
Sprint 14 source `f90f4779c13c…` remains selected.

The retained Sprint 9 profile identified MLP gate/up as the largest projection
group. Sprint 14's unrolled tile16 improvement applies to down projection only.
This experiment tests whether the same tile/kernel combination helps gate/up;
the earlier down result does not predict a gain at this different geometry.

Compare common tile64/stage1 against native unrolled tile16/stage1 at exactly
`(M,N,K)=(8,17408,5120)`. Both keep N/K tiles128, group32 and four warps.
Kernel bodies, scales, accumulation and BF16 output remain unchanged. The new
dispatch guard applies only to the exact untuned HIP/BF16 shape and expected
selected configuration. Preserve tuned entries, other shapes/platforms/dtypes,
the selected down route, 3072 and prefill overrides. Leave the global selector's
missing return unchanged. Scope remains the two existing R9700 gfx1201 cards.

Reuse the sealed Sprint 14 component harness and all nine protected helpers.
The initial shape and baseline-stage changes received a scoped parent review.
The admission correction received separate Astra review before the native
component. Require fresh execution checks and source/package identity binding;
the corrected suite contains 40 checks.

Run four structured/random cases across both cards sequentially, with 96 exact
comparisons including 24 graph replays, and 64 alternating eager timed launches.
Keep the independent structured oracle, exact random baseline comparison,
poisoning, canaries, input immutability, per-device capture streams, compiler
resource reports and independent release. No tolerance relaxation or automatic
retry follows a failed attempt.

Keep the existing 1,200-second active, 600-second worker, 30-second cleanup and
1,500-second total envelope, with the 256 MiB device-tensor limit. This component
starts no models and changes no installed runtime or selected source.

Advance only if all exact checks pass and candidate median time is lower in
every case on both cards. Otherwise retain and park the result. A component win
earns separately reviewed matched serving, reverse order and affected quality
validation before selection. Preserve the current stack, flags, caches and
numerical contract. Reliability remains deferred by owner direction.

See the [evidence index](../provenance/SPRINT15-UNROLLED-GATEUP-20261022.json).


## Admission refusal and correction

The initial source and fresh execution copy each passed 37 CPU checks. Launch
then refused during locked admission because the root observer still required
Sprint 14's mode name. No GPU arm or supervision began. The root dispatcher was
reaped, no owned scope remained and the unresolved marker was absent. The
ordinary independent-release success field remains false; this admission-only
refusal is retained under its separate failure disposition and seals.

The corrected packet changes only the root observer's exact mode literal and
adds three actual-entry CPU checks. The current mode delegates once; the old
mode, an arbitrary mode and an altered root-source closure refuse before source
or observer execution. All 40 checks pass. Source/kernel arithmetic and the nine
protected helper files are unchanged. Retain the first attempt and use a new
nonce after independent review, sealing and fresh execution checks. Do not
rerun the failed nonce or interpret this harness refusal as a kernel result.


## Component result and serving preparation

Astra approved the exact root-observer correction before the fresh-nonce attempt.
The corrected author and execution copy each passed 40 CPU checks. Native work
completed all four cases, 96 exact comparisons including 24 graph replays, and
64 alternating eager timed launches, with independent release. Candidate median
time was lower in every case; the range was 1.22–2.62%. The owned run took
23.00 seconds and started no models. These synthetic component measurements do
not establish a serving gain or isolate the kernel and tile contributions.

The next pilot compares the selected Sprint 14 source against only the gate/up
route change, retaining the selected down route, 3072 and prefill overrides.
Both arms already have the same shape witness. Require common tile64/stage1
for reference gate/up and unrolled tile16/stage1 for candidate gate/up on both
ranks. Down remains unrolled tile16/stage1 in both arms; 3072 remains common
tile64/stage1. The inherited witness's `native` flag means common-kernel use.

Preparation passed 22 packet, 35 startup and 14 reader checks, plus launcher
preparation and read-only admission. The packet retains 2,047 input tokens,
1,024 forced output tokens, low thinking and greedy sampling, with two starts
and eight requests. Exclude each start's first request from timing. Preserve
all output-ID, verifier-count, source/cache/worker/wire and release gates. Stop
and park on a measured output/work mismatch or no lower median request latency.
Only a clean win permits separately prepared reverse order and affected quality
validation. Sprint 14 remains selected.


## Forward pilot result

The pilot completed two starts and eight requests with authenticated sources,
workers, weights, cache witnesses, wire results and independent release. All
four output-ID pairs matched, and every request used 164 verifier cycles.
The measured comparison passed its exact-output and equal-work gates.

Reference and candidate median client times were 8.898470 and 8.894075 seconds,
respectively: 0.0494% lower candidate latency. Combined TP2 rates were 115.07596
and 115.13283 output tokens/s. Each start's first request is excluded; three
requests per arm remain in the timing summary. This small difference alone
is insufficient to establish a repeatable gain. Reverse order remains required.
The source pair, frozen requests, witnesses, reader and controls are identical
in the reverse packet; only execution order changes.

The reverse startup fixture initially required forward owner order and refused
before launch. Its corrected assertion requires the exact reverse mapping;
capacity and native-binding checks remain unchanged. The failed receipt and
original fixture are retained. Reverse preparation passed 23 packet, 35 startup
and 14 reader checks, plus launcher preparation and read-only admission. This
order-only change received a parent review that reuses Astra's integration
review. No extra Astra turn was requested for reversing order.


## Final disposition: park

The reverse campaign completed two starts and eight requests, with independent
release. All four output pairs matched and every request used 164 verifier
cycles. The reference and candidate median client times were 8.885689 and
8.897385 seconds, respectively. Candidate latency was 0.1316% higher; combined
TP2 rates were 115.24148 and 115.08999 output tokens/s.

| Order | Reference, output tokens/s | Candidate, output tokens/s | Candidate latency change |
| --- | ---: | ---: | ---: |
| Reference then candidate | 115.07596 | 115.13283 | −0.0494% |
| Candidate then reference | 115.24148 | 115.08999 | +0.1316% |

The predeclared both-order improvement rule failed. These small differences do
not establish a repeatable serving gain or a general slowdown. Park the gate/up
candidate and keep Sprint 14 selected. All eight paired outputs across the two
orders matched, with equal verifier work; this is not a new output mismatch.
No extra confirmation or affected-quality campaign follows the failed timing
gate. The component win remains valid within its synthetic-input scope.

Sprint 15 completed one GPU component and two two-start serving campaigns,
with 16 fixed-work requests. The earlier admission-only refusal started no GPU
worker. Both successful serving campaigns and the corrected component verified
independent release. The parent final audit verified all ten source/native/reader
seals, with the initial refusal and fixture correction retained. No new Astra
closeout review was needed to apply its previously reviewed stop/park rule.

Installed runtime, persistent defaults and the selected source are unchanged.
Prior numerical findings remain unresolved, and reliability testing stays
deferred. Refresh the short classification profile on the selected Sprint 14
source before choosing another tuning lever. This follow-up is diagnostic;
it does not promote the parked gate/up candidate.
