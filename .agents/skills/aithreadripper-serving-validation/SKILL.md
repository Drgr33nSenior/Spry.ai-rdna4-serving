---
name: aithreadripper-serving-validation
description: Resume AIThreadripper RDNA4 serving validation using the selected thinking-enabled profile, sealed evidence and sprint gates. Use for serving experiments and session handoffs, not workstation installation.
---

# AIThreadripper serving validation

Resolve the project root as `../../..` from this skill. Read
[current selection](../../../provenance/CURRENT-SESSION-SELECTION.json),
[Sprint 7](../../../docs/SPRINT7-QUALIFICATION.md) and the relevant section of
[current results](../../../docs/CURRENT-SERVING-RESULTS.md) before choosing work.
Use these maintained records for current settings; do not infer selection from
the newest scratchpad directory. Resolve the private evidence workspace from
the session handoff. For its existing launch/readout procedure, read
[evidence workflow](references/evidence-workflow.md).

## Preserve the request profile

The selected quality profile enables thinking with
`chat_template_kwargs.enable_thinking=true` and `reasoning_effort=low`.
Nonthinking ledger failures were already demonstrated. Matched BF16 controls
failed without thinking and passed J01 with low thinking; FP8 ordinary serving
passed that short task at low and xhigh effort. These are narrow observations,
not general quality guarantees. For external model-specific claims and sampling
recommendations, read [model guidance](references/model-guidance.md).
Do not repeat the nonthinking screen as selected
profile validation or describe its failure as a new selected-profile regression.

Before building a packet, verify the actual frozen request bodies against the
selected profile. Thinking changes the rendered prompt and input IDs: recompute
them with the pinned published tokenizer. Check input + total output cap +
internal reserve against the selected context. Budget for reasoning and the
complete final answer. Reduce inert filler when necessary; retain task facts,
oracles and evidence placement rules, and disclose the context change.

Grade the final segment decoded after the authenticated closing thinking token.
Bind full decoded output to retained HTTP text. Count reasoning tokens and final
answer tokens separately. A cap hit, unfinished thinking, unsupported generated
code or absent final answer cannot become a task pass. Execute model-produced
Python only through the existing network-free Linux grader. Preserve separate
functional, format, ungraded and sustained-final-output results.

## Preserve experiment meaning

Use the selected stack, precision, flags, kernels, pools and owner controls.
FP8 target **weights** do not imply FP8 attention KV. Both attention caches remain
BF16; preserve the explicit BF16 draft-KV override. FP8 target KV is a deferred
optional work item. Read the selected recurrent profile separately from the
GEMM source: the same GEMM can run with snapshot or aligned ring-fold. Carry
that profile into both arms unless it is the stated A/B variable. Keep the
original compact `--enable-linear-replayssm` flag disabled.

The split-KV verification adapter changes attention computation; it does not
compress the BF16 cache. Attribute ring-fold savings to recurrent-state storage
at matched usable capacity. Startup allocator counters do not establish
whole-device peak memory or a memory advantage over another serving engine.
See [Sprint 19](../../../docs/SPRINT19-MATCHED-RINGFOLD.md) for the measured scope.

Identical repeated outputs establish observed repeatability, not correctness or
a causal speedup. Forced fixed-token benchmarks do not grade normal completion.
Report TP2 throughput as combined service throughput and allocator figures per
rank/card. The earlier mismatch is unresolved under the owner's session-use
disposition; do not call it fixed or erase its outliers.

## Carry the session forward

After a bounded run, retain failures and receipts, authenticate source/native/
HTTP/resource-release evidence, then update the relevant provenance and sprint
record. Record the exact request profile, unique task count, repeat count,
remaining gates and next bounded action. Keep obsolete interpretations as
explicit corrections without modifying sealed evidence. Update this skill only
when a verified finding changes future decisions; keep run-specific facts in
the linked records. Skill use supplies no new launch, deployment or publication
authority beyond the user's current task.
