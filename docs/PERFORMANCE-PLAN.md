# FP8 performance work order

Owner decision: 2026-10-15. Follow this order for the FP8 implementation. Keep
the official FP8 weights, cache formats, model configuration and numerical
acceptance gates fixed unless a later experiment explicitly changes them.
KV4 refers to cache storage, not FP4 model weights.

## 1. Establish the cache-path comparison

The [decoder and validator candidate](CACHE-PATH.md) completed its native cache
comparison on 2026-10-15. Use that measured library as the baseline for the
grouped-reader change. Repeat the [native cache comparison](CACHE-GPU-TEST.md)
with fresh candidate build evidence before preparing a changed model benchmark.

Compare baseline and candidate on both cards, with the same cache inputs and
operation order. Require exact retained decoder, cache and attention results,
malformed-input refusal, output canaries and observed cleanup. Report complete
attention-call times, including validation. Keep kernel-test timing distinct
from full-model latency and throughput. Do not combine profiled operation sums
into an assumed wall-time fraction.

## 2. Audit work and reuse decoded values

Check whether launch grids, validation and cache reads follow active token
counts or allocated capacity. Preserve bounds, initialization and output
semantics when eliminating empty work. An audit that finds no excess work is
not evidence of a defect.

The 2026-10-15 source audit found that the current BF16 v2 reader and validator
already use active per-row lengths. No allocated-capacity scan defect was
found. Six query heads independently decode the same KV head; prefill rows
also revisit overlapping causal prefixes. These are source observations, not
measured bandwidth or a sixfold speedup prediction.

The decoder/validator native comparison has now passed the retained exact-output
checks on both cards. Its synchronized cache-call median gains were 1.36–1.55x
for protected BF16 K/V and 1.04–1.05x for compact KV4. The current source candidate
adds compact-only grouped decoding; these measurements do not apply to that new
candidate. See [cache-path status and boundaries](CACHE-PATH.md).

Investigate decoding each KV value once for the query heads that share it.
Prefer a candidate that preserves each head's ordered FP32 dot products,
softmax and accumulation. Account for synchronization, LDS traffic, register
pressure and temporary storage; fewer source operations need not be faster.
Keep validation intact. Moving validation requires proof that the checked
properties remain valid across append, reuse, eviction and replay.

Measure the current reader, candidate reader and strongest working dense path
at matched contexts and batches. Distinguish component timing from complete
calls and model measurements. Do not interpret greater feasible batch capacity
as lower single-request latency.

## 3. Evaluate fusion in the full-model path

After cache measurements, inspect normalization, activation quantization,
packing and GEMM boundaries. Consider producer/consumer fusion only when the
measured cost is material. Preserve the current block-scale policy, rounding
and checkpoint bytes. A changed reduction order needs explicit numerical
review even when the mathematical expression is unchanged.

## 4. Investigate recurrent state and TP2 communication

Measure Gated DeltaNet state traffic and tensor-parallel synchronization in a
full-model run before proposing a change. These operations are absent from the
standalone cache test and cannot explain its timings.

Check existing SGLang implementations before starting a port. ReplaySSM is a
candidate only if recurrent-state traffic is significant. State quantization,
including LeapQuant, needs a separate approximate-numerics and quality contract.
Communication changes must retain uncompressed values and undergo numerical
verification; mathematical equivalence does not imply bitwise equality.

## Research basis and limits

The research scan covered approximately 2026-09-15 through 2026-10-15, with one
near-window AMD report. These sources suggest experiments, not local results.

| Source | Useful idea | Important limit |
| --- | --- | --- |
| [D-Quant, 17 September](https://arxiv.org/html/2609.19880v1) | Decode once and reuse data during attention. | Its lossy format is not adopted. Small-batch runs were slower than BF16; the 3.5x claim compares maximum feasible batches. |
| [Meta low-precision FA4 report, 16 September](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/) | Fuse quantization into producers to remove intermediate global-memory passes. | Blackwell and MXFP8 implementation, not the current AMD block-FP8 numerical contract. |
| [AMD gfx90a report, 14 September](https://arxiv.org/html/2609.15627v1) | Audit active work versus allocated cache capacity. | Different model and architecture; its kernel/resident/HTTP gains are not interchangeable. |
| [ByteCross, 27 September](https://arxiv.org/html/2609.33889v1) | Match workload and use a strong baseline. | Sparse computation changes semantics; reported gain varies greatly with baseline choice. |
| [ReplaySSM, 9 October](https://arxiv.org/html/2610.13778v1) | Buffer updates and avoid complete recurrent-state writes each token. | High-batch NVIDIA results do not establish batch-one RDNA4 gains or bitwise equality. |
| [LeapQuant, 29 September](https://arxiv.org/abs/2609.38166) | Periodic state quantization with high-precision buffered updates. | Approximate state arithmetic, not an approved precision change. |
| [T-CCL, 5 October](https://arxiv.org/html/2610.07098v1) | Reduce execution resources and synchronization consumed by collectives. | NVIDIA TMA implementation; serving results used eager execution, not graphs. |

No source establishes a fourfold improvement for this workstation. A fourfold
whole-workload gain requires removing at least 75 percent of baseline wall time,
even if the optimized portion becomes instantaneous. Measure that fraction
before estimating an overall ceiling.

## Execution and rollback

Use fresh, identity-bound packages and the existing finite owners, admission
checks and independent release evidence. The agent may prepare and verify the
unprivileged compiler path. A human starts the privileged workstation observer
through the generated one-use launcher. Do not reuse consumed packages, install
dependencies, change device settings or overwrite the installed runtime.

Rollback is non-adoption: retain the unchanged baseline library, model and
runtime. Research candidates do not alter serving defaults. GPU execution,
model qualification, quality and speed require their own observed evidence.
