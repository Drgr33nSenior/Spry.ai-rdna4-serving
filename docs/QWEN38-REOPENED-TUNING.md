# Reopened Qwen3.8 tuning

The owner reopened the remaining Qwen3.8-27B tuning work on 23 October 2026. The [frozen baseline](QWEN38-FROZEN-BASELINE.md) remains the reference. Flash-Next, lower-bit weight checkpoints and alternate engines remain parked while the owner waits for Qwen4 dense 27B. No release date or final configuration is assumed.

Keep the selected Torch 2.14 / ROCm SDK 10.2 stack, official FP8 target weights, BF16 target and draft KV, explicit BF16 draft override, DFlash8, metadata reuse, b101 W4 and aligned ring-fold. Preserve FP32 recurrent state and head arithmetic, 19 usable/20 physical recurrent slots, thinking enabled at low effort, and the existing owner and resource controls. Each candidate changes one stated variable against that reference.

| Sprint | Work | First bounded gate |
| --- | --- | --- |
| 35 | Long-document GPU prefix reuse | Prepare a 32K C1 screen with a shared approximately 26K-token document, exact repeats, a changed question suffix and an isolated-salt control. Require native tokenizer identities, actual cache-hit counts, complete typed answers, timings and independent release. The [first bounded screen passed](SPRINT35-LONG-PREFIX-REUSE.md): 12 normal decoded responses, 10/10 strict SRE answers and 24,576 cached tokens on warm requests. No runtime change was required. |
| 36 | Prefill chunk sizing | [Both-order screen complete](SPRINT36-PREFILL-CHUNKS.md): 12 normal finals and 8/8 strict SRE answers. Retain 2048; 4096 used fewer chunks but had higher observed times and allocator peaks, with unmatched generated work. |
| 37 | FP8 target KV feasibility | Audit scales, query arithmetic and the split adapter. Keep draft KV BF16. Preserve the original matched-test direction and budgets; this is a separate capacity/efficiency experiment, not a weight change. |
| 38 | Adaptive speculation | Establish usable pre-selection scores and acceptance telemetry before implementing a policy. Keep DFlash8 as the reference and test the sampling, graph and slot-lifetime contract. |
| 39 | C2 throughput and scheduling | Fund both requests' context capacity, observe actual batch-two execution, and measure service throughput separately from C1. Preserve native fallback, cancellation and cache ownership. |
| 40 | UltraQuant KV feasibility, conditional | Reuse the prior codec assessment only if cache capacity or traffic warrants the integration cost. The retained eager path is not a drop-in graph/DFlash backend and does not compress model weights. |
| 41 | BF16-weight control, conditional | Reconcile loader, host and per-card memory before any comparison. Use a separate quality/performance contract; fit and stability superiority remain unproved. |

Following the Sprint 36 source audit, both ranks observed full M2048 chunks using stage 1 and M4096 chunks using stage 3. The latter bypasses the selected M2016/2047/2048 specialization. An M4096 stage-1 route remains an unselected source lead requiring a separate experiment. Sprint 37 next audits optional FP8 target KV under the original matched-test scope and budgets.

Sprint 35 used two fresh starts with six requests each: warmup, cold document A, exact warm repeat A, a shared-document branch, an isolated cold copy of that branch, and its warm repeat. All long requests allow at most 26,000 input tokens, 4,096 total output tokens and a 32-token internal reserve within context 32,768. The token pool is 32,800. New prompts explicitly declare JSON arrays, numeric types and action codes. These are new predeclared checks; Sprint 32 and 33 failures remain unchanged.

Report observed cached tokens and their fraction of the actual shared prefix. A partial hit does not establish reuse of the whole document. Preserve every generated-token and verifier-cycle difference. If generated work differs, timing is descriptive rather than a matched-output speedup. The first screen measures existing behavior; a source finding alone does not select a new kernel or cache policy.

Rejected kernel candidates and their failed gates remain recorded. Reconsider them when new source evidence or a changed workload justifies it. Reliability and endurance remain after tuning, as previously directed. New concerning task results, numerical behavior or GPU faults follow the existing investigation procedure. See the [reopened work record](../provenance/QWEN38-REOPENED-TUNING-20261023.json).
