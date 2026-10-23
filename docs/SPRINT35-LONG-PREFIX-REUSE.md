# Sprint 35: long-document prefix reuse

The first reopened Qwen3.8-27B screen passed its bounded cache and answer checks on 23 October 2026. Both fresh starts reused 24,576 tokens for exact repeats and a changed question over a shared document. All 12 responses stopped normally and decoded; all 10 graded SRE responses passed the predeclared strict typed JSON check. Those responses cover two task families with repeated requests.

This screen exercises existing behavior against the [frozen reference](QWEN38-FROZEN-BASELINE.md). It does not select a new kernel or cache policy. The selected b101 GEMM, aligned ring-fold, FP32 recurrent state and head arithmetic, official FP8 target weights, BF16 target and draft KV, explicit BF16 draft override, DFlash8 and metadata reuse remain in use. Thinking is enabled at low effort with greedy generation and natural stop.

## Requests and observations

Each start served the six requests below sequentially at concurrency one. Context length was 32,768, the token pool was 32,800, and the output cap was 4,096 tokens. Prefill chunks remained 2,048 with the `no_buffer` cache strategy. Native tokenization measured a 25,890-token common prefix between the incident and runbook questions.

| Request | Input tokens | Cached tokens | Output tokens | Verifier cycles | Start 1 elapsed | Start 2 elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Excluded warmup | 2,040 | 0 | 33 | 6 | 0.980 s | 0.979 s |
| Cold incident A | 25,991 | 0 | 673 | 105 | 15.364 s | 15.420 s |
| Exact repeat A | 25,991 | 24,576 | 673 | 105 | 6.357 s | 6.352 s |
| Changed question, shared document and salt A | 25,978 | 24,576 | 523 | 79 | 4.904 s | 4.898 s |
| Same changed question, isolated salt B | 25,978 | 0 | 523 | 79 | 13.395 s | 13.460 s |
| Exact repeat with salt B | 25,978 | 24,576 | 523 | 79 | 4.881 s | 4.881 s |

Counts were identical in both starts. Cached tokens were 94.56% of incident input and 94.60% of runbook input. The observed cached count was 94.92% of the separately measured common-prefix length. This count comparison does not establish reuse of the whole document or separately attributed physical cache bytes.

All six corresponding output-token vectors and verifier-cycle counts matched across starts. Within each start, cold incident versus its repeat, isolated runbook versus its repeat, and isolated runbook versus the shared-document question also matched. The `control` and `mtp` labels in the private evidence identify two starts of the same full configuration.

Elapsed time is local non-stream HTTP request-to-terminal duration, including uncached prefill, generation and protocol completion. These are individual request observations in one fixed order per start. They are not a benchmark speedup, TTFT or isolated decode rate. Both GPUs served each request together in TP2.

## Scope and closure

The two task families require incident facts and runbook actions, typed arrays and integer thresholds. Facts appear near the front before filler. This does not test facts distributed through the document or broader long-context reliability. The new explicit type requirements do not regrade the recorded Sprint 32 or Sprint 33 failures. Earlier numerical findings and the owner's session-use disposition remain unchanged.

Both starts reported the same startup allocator values on each rank: 19.565 GiB allocated, 19.854 GiB reserved and 11.070 GiB driver free. These observations follow graph setup; they are not whole-workload peak VRAM. Recurrent capacity remained 19 usable and 20 physical slots per rank.

One physical campaign completed in 358.9 seconds. Source identities, finite receipts, CPU-only answer decoding and independent resource release were verified. A local preparation interpreter refusal and an offline reader path error were retained. The corrected reader used byte-identical reviewed source and required no GPU rerun. No installed defaults changed. Reliability and endurance remain deferred.

The [provenance record](../provenance/SPRINT35-LONG-PREFIX-REUSE-20261023.json) contains exact counts, timings, hashes and limitations. [Sprint 36](QWEN38-REOPENED-TUNING.md) next evaluates prefill chunk sizing with actual dispatch and memory witnesses.
