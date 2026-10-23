# Sprint 36: prefill chunk sizing

Retain the selected **2048-token prefill chunk**. The completed 2048→4096 and 4096→2048 screens returned 12 normal decoded responses and passed all 8 strict SRE checks across two repeated task families. The 4096 configuration used fewer chunks but had higher observed long-request times and allocator peaks. Generated work differed, so these are descriptive instrumented HTTP observations; they do not establish a matched speedup or general performance ranking.

## Requests and observed time

Four fresh starts each served one excluded warmup and two cold SRE requests. The exact bodies, published-tokenizer IDs and strict typed oracles came from [Sprint 35](SPRINT35-LONG-PREFIX-REUSE.md): 2040 input tokens for warmup, 25991 for the incident question and 25978 for the runbook. All 12 comparison responses reported zero cached tokens.

| Execution order | Task | Chunk 2048 | Chunk 4096 |
| --- | --- | ---: | ---: |
| 2048 then 4096 | Incident | 15.377 s | 16.956 s |
| 2048 then 4096 | Runbook | 13.419 s | 15.137 s |
| 4096 then 2048 | Incident | 15.421 s | 15.724 s |
| 4096 then 2048 | Runbook | 13.453 s | 15.141 s |

Duration is local non-stream HTTP request-to-terminal time for the combined TP2 service. It includes uncached prefill, generation, protocol completion, observer overhead and any remaining compilation. Each start used an isolated Triton cache; exact bulk-shape compilation warmness was not separately established. Observer overhead, TTFT, native prefill device time and isolated decode were not measured.

| Order | Task | 2048 output tokens / verifier cycles | 4096 output tokens / verifier cycles |
| --- | --- | ---: | ---: |
| 2048 then 4096 | Incident | 673 / 105 | 673 / 103 |
| 2048 then 4096 | Runbook | 523 / 79 | 504 / 84 |
| 4096 then 2048 | Incident | 673 / 105 | 519 / 80 |
| 4096 then 2048 | Runbook | 523 / 79 | 504 / 84 |

The first-order incident output IDs matched, while verifier work differed. Both runbook pairs and the reverse-order incident differed in full output IDs and work. All corresponding 2048 outputs and work matched across orders. The 4096 incident also changed between its two starts. Every graded final answer met its frozen oracle; keep all reasoning and work differences in the record. Facts were near the front before filler, so this does not establish dispersed-fact or broad long-context quality. Previous Sprint 32/33 failures and numerical findings remain unchanged.

## Actual chunk geometry and dispatch

Both ranks authenticated complete request coverage and restored the two observer hooks. Long requests used 12 full chunks plus a tail at 2048, and 6 full chunks plus the same tail at 4096. Tail lengths were 1415 and 1402 tokens. Logical lengths, forward-entry rows and actual consumed GEMM M were recorded separately; they matched in these samples.

Each rank observed 3328 block-FP8 GEMM calls per long request at 2048 and 1792 at 4096. Including warmup, the totals were 27 successful prefill forwards / 6912 credited calls versus 15 / 3840. These counts cover the observed block-FP8 routine, not every model operation or GPU kernel. Observer invocation counts differed, and the hooks were restored after complete prefill coverage.

Actual full-chunk M2048 calls used the selected stage-1 configuration. M4096 calls used stage 3; both tails used stage 3. The observed callable was `_w8a8_block_fp8_matmul`, with FP8 operands, FP32 scales and BF16 output under the unchanged b101 source. Input and scale dtypes are retained separately in provenance. Larger chunks bypassed the existing M2016/2047/2048 specialization. The M4096 dispatch is a separate tuning lead; attribution requires another experiment.

## Memory scope

Both orders and both ranks reproduced these maximum observed cumulative allocator peak counters at successful prefill host return:

| Per-rank counter | Chunk 2048 | Chunk 4096 | Difference |
| --- | ---: | ---: | ---: |
| Peak allocated | 20.075 GiB | 20.425 GiB | +359.125 MiB |
| Peak reserved | 20.408 GiB | 20.869 GiB | +472 MiB |

The observer added no synchronization, counter reset, driver-free query or tensor operation. These observations do not establish end-of-request, process-lifetime or whole-device peak VRAM. At the final prefill observation, current allocated memory differed by only +0.0913 MiB; reserved memory differed by +472 MiB. Preserve that distinction from peak allocated.

Fresh per-start host cgroup peaks, including startup, page cache, tmpfs and children, were 20.253/22.187 GiB for the 2048→4096 order and 28.119/20.573 GiB for 4096→2048. The host variation remains recorded. These are not process RSS or GPU memory. No start recorded an OOM or memory-limit event; the existing 38 GiB host cap and zero-swap control remained in force.

## Closure and retained failure

The numerical kernel source, stack, precision, pools and installed defaults did not change. Both profiles kept official FP8 target weights, BF16 target and draft KV with the explicit BF16 draft override, FP32 recurrent state and head arithmetic, aligned ring-fold, DFlash8, metadata reuse and 19 usable/20 physical recurrent slots per rank. Concurrency was one, context 32768, pool 32800, total output cap 4096, thinking enabled at low effort, greedy generation and natural stop.

An initial first-arm campaign completed three normal responses and both observers, then the integration validator rejected the new telemetry at a legacy J01-only guard. Its 193.5-second failed campaign remains sealed, with its answers ungraded and excluded from this comparison. The 4096 arm and reverse order did not start in that packet. Independent release passed with no cleanup error or unresolved ownership.

The correction explicitly admits Sprint 36 observations and requires full source, worker, request-ID and real-wire validation after wire parsing. Fourteen composed regressions use sealed native evidence, reproduce the original refusal and preserve malformed-record and unrelated-mode refusals. Both fresh successor packets passed 129 acceptance checks, verified 297 tested hashes and retained 26 passing reader checks each. All package ceilings and the eight protected helpers stayed unchanged. The successful campaigns took 332.5 and 324.8 seconds, including startup and closure; these are not request durations.

Independent source/native/HTTP/observer/answer/release checks passed. The [provenance record](../provenance/SPRINT36-PREFILL-CHUNKS-20261023.json) retains exact observations, source and result hashes, output/work differences and failed preparation roots. Keep 2048 selected. Reliability and endurance remain after tuning; [Sprint 37](QWEN38-REOPENED-TUNING.md) next addresses optional FP8 target-KV feasibility with the draft cache kept BF16 and the original matched-test budget preserved.
