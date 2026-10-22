# Sprint 10 M8 matched serving validation

Status: complete. Both matched benchmark orders and all affected bounded
candidate screens passed. Astra verified the results and supports M8 stage one
for the next experimental profile launch. Keep the prior source for rollback.
The installed runtime and production defaults remain unchanged.

Compare the selected M8 stage depth three with the candidate stage depth one
only for the exact untuned gate/up shape `(8,17408,5120)`. Preserve M2047 stage
one, the current Torch 2.14 / SDK 10.2 stack, FP8 weights, BF16 target and draft
KV, FP32 head/state, snapshot path, metadata reuse, DFlash block eight, graph
buckets and capacity. Both sources carry the same metadata-only M8 dispatch
witness. Additional Sprint 8/9 timing events are disabled in both arms.

Each fresh start receives the same 2,047-token prompt with 1,024 forced output
tokens, low thinking and greedy sampling (temperature zero, top-p one, top-k
one). Exclude its first request as warmup and retain all three measured samples.
Input messages and token IDs match the sealed historical fixed-work donor;
only output cap and cache namespace changed. Forced output does not establish
normal-stop task quality. Report throughput for the combined TP2 service.

The packet passed 56 focused checks and Astra prelaunch review. All eight
protected helpers remain unchanged. The HTTP packet replaces an unused legacy
context asset with the second GEMM source, preserving all transport ceilings:
64 members, 16,148,480 archive bytes, 32,397,250 combined bytes, 255 argv entries
per arm and 65,411/65,383-byte bindings. The original asset remains in private
source provenance. Source and numerical-kernel identity checks remain active.

Authenticate source, runtime, worker, weight/cache, HTTP and resource-release
evidence before interpreting timing. Require 1,024 outputs for every request,
actual M8 dispatch on both cards, and per-request graph/commit/verification
joins. Retain ID and cycle differences; equal output lengths alone do not
establish equal verification work. The supported pilot proceeded to reverse order and the affected greedy/sampled
quality and lifecycle checks. The existing numerical mismatch remains unresolved under the owner’s
session-use disposition. No installed or production default changes.

The [result index](../provenance/SPRINT10-M8-SERVING-20261022.json) records source
identity, all outcomes, retained limitations and the experimental selection.

## Matched benchmark result

| Order | Reference median | Candidate median | Latency change | Combined TP2 rate change |
| --- | ---: | ---: | ---: | ---: |
| Reference, candidate | 9.2444 s | 9.1887 s | −0.6023% | +0.6059% |
| Candidate, reference | 9.2257 s | 9.1936 s | −0.3479% | +0.3491% |

All six measured pairs have identical 1,024-token output IDs and 164 verifier,
replay and commit cycles per rank. No request recaptured graphs. The excluded
reverse reference warmup used 165 cycles versus 164 for the candidate despite
matching output IDs; retain this observation. M8 dispatch is confirmed at stage
three or one as assigned on both ranks; M2047 remains stage one in both arms.
Both campaigns passed source/native/HTTP checks and independent resource release.
These small samples establish an observed gain in both orders, not a general
performance guarantee. The benchmarks include the same existing observers.

The following qualification lanes use two fresh candidate/snapshot starts each:
greedy natural stops, publisher-sampled natural stops, and the existing 27-request
lifecycle schedule. They retain frozen requests and oracles, C2, reuse, branch,
pressure, cancellation and recovery checks. Compare quality to historical sealed
reference evidence; these are repeated candidate screens, not contemporaneous
quality A/B comparisons. No numerical-equivalence conclusion follows.

## Affected quality and lifecycle result

| Frozen lane | Fresh candidate starts | Observed result |
| --- | ---: | --- |
| Greedy, low thinking | 2 | 8/8 natural-stop final-answer gates; four unique tasks per start |
| Publisher sampled, low thinking | 2 | 8/8 natural-stop final-answer gates; four unique tasks per start |
| Reuse, branch, pressure, cancellation and C2 | 2 | 52/52 correct natural-stop answers, two successful cancellations/recoveries, actual C2 and slot accounting |

Both quality starts in each lane use the candidate and snapshot configuration.
The selected greedy ledger inputs remain 12,272 tokens with a 4,096-token output
cap. The publisher-sampled ledger inputs remain 8,176 with an 8,192-token cap.
Both preserve thinking at low effort. Code runs only through the retained
network-free Linux grader after authenticated final-segment extraction.

All four greedy final-text pairs match. Two whole output-ID pairs differ in
reasoning: running totals uses 309/320 reasoning tokens and the threshold ledger
2,126/2,121. The other two pairs match, including both 3,977-token held-out
responses. All sampled output pairs differ; all answers pass their task checks.
These observations retain the existing unresolved numerical contract.

Lifecycle cache comparisons match 24/26 full output-ID pairs and 26/26 final
answers. Both differing pairs compare concurrent-a with its cold singleton;
the first difference is output index 149. The first C2 group has 41 batch-two
replays and no singleton replay, with request counts 41/41. The second has
41 batch-two plus three singleton replays, with request counts 44/41. Weighted
request accounting passes on both ranks. All active requests are released;
each card finishes with eight free and 11 evictable cached slots, accounting
for all 19 usable slots. This does not establish numerical state equality.

Sampled-start Torch allocation is 20.52785 GiB per card, with 20.80078 GiB
reserved, in both starts. These are startup allocator observations before
external generation; per-request and whole-device peaks are not measured here.
All five native campaigns passed independent release. No GPU campaign is queued.

## Source selection and evidence

The [serving patch](../patches/sglang-gfx1201-m8-gateup-stages-serving.patch)
reproduces the exact tested candidate from the recorded instrumented reference.
The result index binds input/output and patch hashes. The original Sprint 9
component patch remains as historical evidence. The dispatch guard checks HIP,
shape, kernel and default configuration; validation covers these two R9700
gfx1201 cards. It does not explicitly guard the device architecture.

Retain Torch 2.14 / SDK 10.2, FP8 weights, BF16 target/draft KV, FP32 head/state,
M2047 stage one, metadata reuse, DFlash block eight, snapshot, graph buckets and
matched capacity. Keep FP8 target KV and ring-fold as separate optional work.
Full-output/state equivalence, broad task quality, sampling-distribution equality,
endurance and production qualification remain open. The earlier numerical
mismatch is not resolved by this sprint.

Packet/startup checks passed 56/56 for each benchmark order, 53/53 for each
quality lane and 65/65 for lifecycle. Readout checks passed, including injected
local-evidence refusals, cap/missing-final grading guards and counter accounting.
Astra independently verified the final results and the patch bytes. Selected
source, native and readout seals verify; the result index retains their pins.

One sampled author seal was invalidated when a read-only remote admission check
wrote three local receipts after sealing. All original hashes matched. The
original remains preserved; the actual campaign used a fresh reviewed recovery
snapshot sealed after the receipts, with unchanged payload and no consumed nonce.
The reader's local-cache-proof boundary and inherited timing label were corrected
without a GPU rerun. A condensed review message's C2 counts were corrected to
match the already correct detailed native and public arrays.
