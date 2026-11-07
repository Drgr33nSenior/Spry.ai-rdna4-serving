# Sprint 37: FP8 target-KV feasibility

Retain BF16 target and draft KV. The source feasibility audit is complete; the matched FP8 serving experiment is blocked by the current query-arithmetic and campaign contracts. No GPU work ran, and no runtime source or default changed. This does not establish that FP8 KV is unsupported on gfx1201.

The audit verified the original direction and four predecessor seals, then read seven installed source files against their pinned hashes. It preserved the selected FP8 target weights, BF16 draft weights and cache, FP32 recurrent state and head arithmetic, aligned ring-fold, DFlash8, metadata reuse, b101 W4 and 2048-token prefill chunks. Existing numerical findings remain unresolved under the owner's experimental-session disposition.

## Compatibility findings

| Surface | Source finding | Required next work |
| --- | --- | --- |
| Native cache storage and scaling | The pool divides K and V by their respective scales before casting. Native attention restores K scaling in the score multiplier and V scaling in the output. The checkpoint scale loader supports separate scalar K/V scales and a native unit-scale fallback when scales are absent. | Observe the actual loaded scales, representation and clipping/range behavior. Do not assume calibration is required, or treat fallback scales as calibrated values. |
| Split verifier | The selected adapter accepts only BF16 Q/K/V/output/cache tensors and unit scales. Changing the target cache flag alone makes it ineligible. | Extend the adapter and its evidence contract without weakening shape, causal-row, suffix-write, graph-lifetime or draft-BF16 checks. |
| Query arithmetic | Generic grouped decode and cached-prefix extend paths cast Q to the cache dtype. With an FP8 cache, this changes BF16 query arithmetic. The existing BF16-query workaround is gated to gfx1250, not gfx1201. | Prepare a narrow BF16-query FP8-cache candidate and test its actual decode and prefix-read dispatch. A guard-only adapter change does not meet the original contract. |
| Prefill | The generic extend path includes FP8 cache reads and descales. Separate gfx95/Kimi optimizations do not establish support for this Qwen configuration. | Verify native compilation, writes, prefix reuse, scales and output on gfx1201. Generic support is not disproved by the specialized-path guards. |
| Runtime witnesses | Current entry and evidence checks require target `auto` cache dtype and BF16 target/draft cache buffers. | Update the target-specific producer, live consumer and offline reader together; preserve the explicit BF16 draft override. |

This is more than a flag change, but the native scale/dequantization code already exists. No new backend port has been shown necessary. No FP8 speed, memory, correctness or device-compilation result follows from this source audit.

## Campaign and workload gaps

The original direction requires four fresh starts, BF16/FP8 then FP8/BF16, inside one campaign with a 4,570-second owner limit. The protected owner and its adapter accept exactly two starts. Their selected reservation is `75 + 2×2010 + 330 = 4425` seconds. Four starts with those same envelopes would require at least 8,445 seconds; two separate owners reserve 8,850 seconds. These are configured bounds, not predicted run times. Neither alternative fits the original schedule unchanged.

The original natural-stop cap is 1,024 output tokens in a 16,384-token context. The prior thinking ledger answers used 2,891–3,977 output tokens, and Sprint 36 used approximately 26K-token inputs with a 4,096-token cap. Neither corpus can be reused unchanged. Prepare new short thinking tasks, exact native tokenizer IDs and frozen checks, including the required conversations. Retain fixed-output timing as a separate, explicitly truncated workload. The existing nonstream transport can report HTTP rate with TTFT and decode-only rate marked unmeasured, as the original direction permits.

## Decision and bounded follow-up

The 1,200-second preparation and 5,080-second physical limits were not expanded. No admission, GPU reservation, model start, calibration or native device test occurred. The original direction requires an actionable closeout when preparation becomes substantial; this result stops at that boundary.

The proposed next FP8 work is a separate compatibility stage: at most 3,600 seconds of source implementation, focused CPU checks and review, followed only after review by a component campaign capped at 1,200 seconds including admission, both-card checks, release and sealing. These are proposed ceilings, not an execution grant or a promise that the work fits. Cover distinct nonunit K/V scales, BF16 query/output arithmetic, causal rows, suffix writes, prefix reuse and graph-buffer lifetimes. Stop without serving if any required gate remains open.

Before the full serving comparison, resolve the four-start owner schedule and freeze a fitting thinking corpus under a separately reviewed scope and budget. Do not modify the protected owner or reduce its safety reserves during admission. Adaptive-speculation prerequisite analysis remains the next independent tuning item.

See the [machine-readable result](../provenance/SPRINT37-FP8-KV-FEASIBILITY-20261023.json). Reliability testing and Next-model work retain their previous deferred status.
