# Sprint 7 qualification

Status: bounded Sprint 7 screens complete. Exact-output equivalence remains open
under the owner's experimental-session disposition. This closes the planned
screens; it does not establish general or production qualification. See the
[closeout](../provenance/SPRINT7-CLOSEOUT-20261021.json).

Use the [project validation skill](../.agents/skills/aithreadripper-serving-validation/SKILL.md) when resuming this work.

Sprint 7 carries two profiles: the selected M2047 stage-one, metadata-reuse,
DFlash8 reference and the aligned ReplaySSM ring-fold candidate. Keep the
current Torch 2.14 / ROCm SDK 10.2 stack, FP8 target, BF16 KV/draft, FP32
recurrent state/head and existing owner/resource controls. Installed defaults
and the selected reference stay unchanged during these experiments.

The initial ReplaySSM screen matched all four output pairs and reduced median
request latency by 2.42%. Automatic recurrent capacity grew from 19 to 74
usable slots per card, and startup allocation increased. That result does not
establish memory savings at equal capacity or broader serving qualification.
See the [screening record](../provenance/REPLAYSSM-SERVING-20261021.json).

| Gate | Required evidence | Status |
| --- | --- | --- |
| Matched capacity and both orders | Explicit 19 usable recurrent slots per card in both profiles; observe 20 physical slots including padding on both ranks. Match context16384, KV tokens16416, request limit2 and the frozen C1 workload. Run candidate/reference, then reference/candidate; retain every request and startup allocation. | Evidence complete. Seven of eight output pairs matched; retain the known reference capitalization exception. Exact-output gate incomplete. |
| Normal-stop quality and sustained output | Use task-specific answer checks and full completion at natural stop; include longer outputs. Record output IDs, finish reasons, speculative counters and any differences. | Selected snapshot low-thinking screen passed: 8/8 natural final answers and task checks; ledger finals exceed512 tokens. Matched snapshot/ring-fold thinking screen also passed 8/8; all final segments match prior selected starts. Full reasoning outputs differ. |
| Longer contexts and C1/C2 | Establish exact input IDs and context/output headroom. Exercise actual concurrent batch two; record the native attention route, graph buckets and capacity. Warmup alone is insufficient. | C1 ledger inputs 12272 and natural outputs up to3977 observed. Both-order lifecycle packets now show actual C2 on both ranks, with request/graph/commit accounting. |
| Cache reuse and branching | Measure cold/warm prefix reuse and branching with cached-token and state/slot evidence. Check outputs against fresh requests. | Bounded both-order screen passed final-answer oracles and slot bookkeeping, including checkpoint boundaries 4095/4096/4097 and cache pressure. Full reasoning outputs vary. |
| Cancellation and repeated lifecycle | Cancel active work through the supported API, check recovery requests, and verify slot/resource release. Exercise repeated starts and request lifecycles within fixed bounds. | Four cancellations and recovery passed across four fresh starts. All active requests released; free plus cached slots account for all 19 usable slots per rank. |
| Sampling | Use the published thinking recipe, prove actual native sampled verification and grade complete final answers on both profiles. | Passed 8/8 natural final answers on both profiles. All 16 external request/rank joins show successful native selector sampling. Distribution equivalence is outside this screen. |

## First gate

Use two fresh two-start packets. Reverse order comes first because the previous
screen ran reference then candidate. Each start has one excluded warmup and
three measured requests with 2,047 input tokens, 128 forced output tokens and
low reasoning. Rates describe the complete TP2 service, while allocator
figures are per rank/card. Forced output does not grade normal-stop quality.

Bind `--max-mamba-cache-size 19` into the CLI, resolved native configuration,
worker preflight and retained evidence. Require the live and offline GDN pool
records to show exactly 20 physical state slots. The model implementation,
weights, attention adapter and precision remain identical to the accepted
ReplaySSM source. Preserve the reference/candidate flag difference.

Run the focused CPU suite and the unchanged archive validator before launch.
Verify the actual request/output hashes, graph replays, completed native
commits, metadata reuse, allocator records and independent release. Keep all
samples, including warmup and exceptions. Report each order separately before
combining results. Do not combine the earlier unequal-capacity comparison
with these packets as a balanced fixed-capacity benchmark.

## Matched-capacity results, 21 October 2026

Both packets passed 45 CPU checks and completed four fresh starts and sixteen
requests in total. Both ranks observed 20 physical recurrent slots in every
start. Source, runtime, weights, HTTP and independent release checks passed.

| Execution order | Reference median | Ring-fold median | Latency change | Rate change |
| --- | --- | --- | --- | --- |
| Candidate, then reference | 1.8481 s | 1.8052 s | -2.32% | +2.37% |
| Reference, then candidate | 1.8500 s | 1.8044 s | -2.47% | +2.53% |

These are descriptive C1 observations with three measured requests per start.
Rates describe the complete TP2 service. The sample does not establish
statistical significance or representative task quality.

Startup Torch allocation fell from 22,041,610,240 to 20,296,157,184 bytes on
each card in both orders: 20.53 to 18.90 GiB, a 1.63 GiB reduction. The main
tensor calculation predicts 1,747,550,208 fewer bytes; the observed reduction
is 1,745,453,056 bytes. The remaining 2,097,152 bytes are unassigned. These
startup counters do not measure process-lifetime or whole-device peak VRAM.

All eight candidate responses matched the historical common output. One
reference response reproduced the prior capitalization variant at token 124;
the full output-ID hash matches the earlier session variant. Seven of eight
paired responses matched exactly. All requests retained 25 verifier cycles,
graph replays and completed commits per rank, with zero recaptures. Keep the
exception and the owner's deferred-diagnosis decision. Do not mark exact
equivalence or task quality as passed. See the
[two-order record](../provenance/SPRINT7-MATCHED-CAPACITY-20261021.json).

The next gate is normal-stop task quality and sustained output for both
profiles. The selected reference and installed runtime remain unchanged.

## Decisions

Mark each gate only after its own evidence passes. Preserve failed packets
and unexplained output differences. The earlier baseline variation remains
deferred under the owner's decision unless concerning results recur. Adaptive
policy is unimplemented; width four and target shard maxima remain outside
these profiles. This sprint introduces no toolchain update or deployment.

## Owner disposition after mismatch diagnosis, 2026-10-21

The owner selected continued current-session use with the explicit
`--speculative-draft-kv-cache-dtype bf16` override and directed backtracking if
more strange behavior occurs. Its GPU screen completed two starts and eight
requests, with BF16 target/draft buffers confirmed on both ranks and eight
outputs matching the historical common response. See the
[validation record](../provenance/BF16-DRAFT-KV-OVERRIDE-20261021.json).

Close the mismatch diagnosis as unresolved; retain the originating-operation
question, the capitalization variants and the broader stage-3 difference from
token 16. This owner disposition permits moving on with the experimental
session. Exact-output and task-quality gates remain incomplete. Reopen the
investigation if new strange outputs, a concerning quality regression or a
runtime/GPU fault appears. Stage 3 is not a demonstrated fix.

The owner subsequently deferred FP8 target attention KV to a
[separate optional work item](SERVING-SPRINTS.md#optional-fp8-target-attention-kv).
Continue normal-stop task quality and sustained output with the selected
snapshot and explicit BF16 draft cache. The FP8 model weights remain selected;
both attention caches remain BF16. Ring-fold stays a separate qualification
candidate. No FP8 cache implementation or campaign is scheduled at this stage.

## Nonthinking control and profile correction, 21 October 2026

The selected request profile requires `enable_thinking=true` with
`reasoning_effort=low`. Earlier matched native BF16 tests reproduced the J01
failure without thinking and passed with low thinking. FP8 ordinary serving
also passed J01 in two low-effort starts and two xhigh starts. See the
[prior thinking evidence](CURRENT-SERVING-RESULTS.md#fp8-thinking-checks-2026-10-21).
Those short-task passes do not qualify longer contexts or combined DFlash.

The subsequent eight-response screen incorrectly reused a nonthinking corpus.
Its requests set `enable_thinking=false` and `reasoning_effort=none`. Runtime
flags and kernels matched the selected source, but the request profile did not.
Retain it as a nonthinking control outside the selected request profile. It does
not evaluate thinking-enabled quality or establish a regression in that profile.
The original interpretation as selected-profile quality failure is superseded.

All eight control responses stopped naturally; all four output-ID pairs matched.
The ledger responses contained 766 and 815 output tokens per start. Both
running_totals answers passed. Four ledger answers failed the frozen checks,
including the J01/H17 affordability errors. Two unique_last answers used
`sort(key=lambda ...)` and remained ungraded under the unchanged sandbox.
These control results and their sealed raw evidence remain intact. See the
[corrected control record](../provenance/SPRINT7-SELECTED-QUALITY-20261021.json).

The next quality packet must enable thinking at low effort. Recompute native
prompt IDs and verify context headroom and an output budget that includes both
reasoning and the final answer. Grade the decoded final answer, and count final
answer tokens separately from reasoning for the sustained-output check. Preserve
the selected stack, kernels, BF16 caches and owner/resource controls. Combined
DFlash quality with thinking, ring-fold quality and other Sprint 7 gates were
pending at that correction. The thinking-enabled result follows below. No GPU
campaign was launched solely for the documentation correction.

## Thinking-enabled selected snapshot, 21 October 2026

The corrected packet enabled thinking at low effort on the selected combined
DFlash8 profile. All eight responses stopped naturally and passed every
applicable task, final-answer format and sustained-final-output check. Four
unique tasks ran in two identical fresh starts; all four output-ID pairs matched.

| Task | Input tokens | Reasoning tokens | Final-segment tokens | Total output tokens |
| --- | ---: | ---: | ---: | ---: |
| running_totals | 226 | 309 | 41 | 352 |
| unique_last | 246 | 786 | 95 | 883 |
| threshold ledger | 12272 | 2126 | 763 | 2891 |
| heldout ledger | 12272 | 3154 | 821 | 3977 |

Counts matched across starts. Each 28-event ledger trace matched its complete
oracle, including correct rejection at J01 and H17 with unchanged balance and
zero fee. Both coding functions passed the unchanged Linux sandbox and format
checks. The pinned tokenizer decoded final answers after the closing thinking
token and verified full output decoding against HTTP text. Reasoning did not
count toward the 512-token sustained-final-output requirement.

The 16384-token context, runtime flags, numerical kernels, BF16 caches and finite
resource controls stayed selected. Ledger filler shrank from about 15344 to 12272
input tokens to reserve 4096 output tokens plus 16 internal tokens. Facts,
questions, oracles and evidence-placement rules stayed unchanged. This context
change and the larger cap prevent a mode-only causal comparison with the old
nonthinking control.

The packet passed 46 CPU checks. Independent review passed 8 packet and 4 readout
checks; the local readout suite passed 4 behavior tests. Source/native/HTTP,
weight, startup-cache and release authentication passed; there were zero request
graph recaptures. The finite launcher completed in about 408.0 seconds. See the
[thinking quality record](../provenance/SPRINT7-THINKING-QUALITY-20261021.json).

This passes the selected-snapshot screen only. The greedy sampling profile
differs from the publisher's sampled recipe. The longest response used 3977 of 4096
output tokens, leaving 119 tokens. Ring-fold thinking quality, C2, sampling, reuse
and lifecycle gates remain pending. No general numerical-equivalence, mismatch
resolution or performance-improvement claim follows. Next, apply this corrected
thinking corpus to the separate ring-fold candidate with matched capacity.

## Matched snapshot and ring-fold thinking quality, 21 October 2026

The matched packet passed all eight task checks at natural stop: four tasks on
one fresh snapshot start and one fresh ring-fold start. Both profiles used
19 usable recurrent slots and 20 physical slots per card. The request bodies
and input vectors exactly reused the corrected thinking corpus. All final-answer
texts and token segments matched both earlier selected-snapshot starts.

Only two of four full output-ID pairs matched. The two ledger responses diverged
within reasoning at token indices 378 and 493, while their complete final traces
remained correct. Preserve these differences and the unresolved exact-output
gate. This result does not resolve the mismatch or qualify repeatability.

Startup Torch allocation again fell by 1,745,453,056 bytes per card with ring-fold.
Both ranks retained zero request recaptures and equal native commit, replay and
verifier counts. Source, weights, HTTP, BF16 cache startup and independent release
checks passed. The packet passed 63 CPU checks and independent preparation/readout
reviews; the native launcher completed in 395.8 seconds. No latency improvement
claim follows from this quality run. See the
[matched thinking record](../provenance/SPRINT7-RINGFOLD-THINKING-QUALITY-20261021.json).

Actual C2, cache reuse/branching with state-slot evidence, cancellation/recovery
and repeated lifecycle checks remain pending. Their current-stack packet is in
CPU preparation. Publisher-sampling coverage also remains separate. The selected
reference and installed defaults remain unchanged.

## Thinking lifecycle results, 21 October 2026

Two packets ran both profile orders, with four fresh starts and 108 requests.
All 104 normal requests stopped naturally and passed their exact final-answer
oracles. Four active streaming cancellations completed, followed by correct
recovery answers. Both ranks observed real batch-two graph traffic in all four
starts. The C2 split adapter is ineligible at batch two; these requests used
the native attention route. Per-request verifier counts reconcile with singleton
and batch-two replays and completed native commits.

The schedule covered cold/warm reuse, branches at cached-prefix boundaries
4095/4096/4097, six 8191-token pressure prompts, victim recovery, cancellation,
concurrent requests and final replay. Source-bound integer slot observations
accounted for 19 usable slots at the final quiescent point. All active request
rows were released. Remaining cached slots were evictable, with none protected.
This establishes bookkeeping; it does not compare numerical state values or
prove per-node cache ownership.

All 52 final-answer cache comparisons passed. Whole output IDs matched 42/52
pairs; ten reasoning-output differences remain recorded. Preserve the unresolved
exact-output gate and the owner's experimental-session disposition. Independent
release passed after each packet. Instrumented timing is not a performance
result. See the [lifecycle record](../provenance/SPRINT7-LIFECYCLE-20261021.json).

The publisher sampling screen below completed the remaining bounded GPU work.

## Why thinking helps these tasks

Thinking mode changes the model's generation procedure: it produces reasoning
tokens before the final answer. Those tokens provide extra autoregressive
computation for tracking balances, checking constraints and revising a plan.
This is a plausible explanation for the local ledger improvement, not a
demonstrated diagnosis of every nonthinking failure. The published model
supports both modes; [model guidance](../.agents/skills/aithreadripper-serving-validation/references/model-guidance.md)
separates its recommendations and external reports from local evidence.

Matched short J01 controls failed without thinking and passed with thinking.
The corrected combined-profile thinking corpus passed its full final-answer
oracles. Its long-prompt comparison also reduced inert filler and increased
the output budget, so it is not a mode-only causal experiment. Thinking does
not repair a GPU kernel, prove nonthinking universally broken, or resolve
unexplained numerical output differences.

## Publisher sampling and bounded closeout, 21 October 2026

Both profiles completed the official thinking recipe: temperature 1.0, top-p 0.95,
top-k 20 and neutral declared penalties. All eight answers stopped naturally and
passed their applicable functional, format and sustained-final-output checks.
The four ledger finals had 763 or 821 final tokens, excluding reasoning. Each of
the eight external requests produced a source/worker/request-bound successful
native selector-sampling return on both ranks. Independent release passed.

Ledger inputs were 8176 tokens with an 8192-token total output cap and 16-token
reserve inside the unchanged 16384-token context. Task facts, questions and
oracles were preserved; inert filler was reduced. The separate greedy screen
retains 12272-token ledger inputs. This sampled screen uses one fresh start per
profile and four tasks. It does not establish distribution equivalence or a
performance gain. See the [sampling record](../provenance/SPRINT7-SAMPLING-20261021.json).

The planned bounded screens are complete. Retain the snapshot reference and
keep aligned ring-fold as a separately screened experimental option. The prior
matched-capacity timing and 1.63 GiB/card startup allocation saving remain bounded
observations. Full-output variation and numerical-state equivalence remain open
under the existing owner disposition. Installed defaults and toolchain stay at
the selected versions. See [follow-up scope](SERVING-SPRINTS.md#after-sprint-7);
no Sprint 8 or deployment is defined by this closeout.
