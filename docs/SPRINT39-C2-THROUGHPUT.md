# Sprint 39: sequential and concurrent request pairs

The first bounded C2 throughput screen passed. Two concurrent requests delivered about **173 output tokens/s combined across both GPUs**, compared with about **120 tokens/s for sequential requests**. Individual requests took longer under C2. These are instrumented HTTP service rates; the generated token paths and verifier work differed.

The selected numerical sources, kernels, runtime defaults and resource limits did not change. Both fresh starts used official FP8 target weights, BF16 target and draft attention KV, BF16 draft weights, FP32 recurrent state and head arithmetic, fixed DFlash8, metadata reuse, b101 W4, aligned ring-fold and 2048-token prefill chunks. Thinking remained enabled at low effort with greedy sampling.

## Measured service rates

Each measured request had 2047 input tokens and exactly 1024 forced output tokens. Each row aggregates three two-request groups. The second fresh start reversed the C1/C2 group order.

| Fresh start | Submission mode | Combined TP2 tokens/s | Median individual request latency | Individual latency range |
| --- | --- | ---: | ---: | ---: |
| 1: C1 then C2 | C1, sequential pair | 120.18 | 8.502 s | 8.493–8.531 s |
| 1: C1 then C2 | C2, concurrent pair | 172.98 | 11.763 s | 11.710–12.045 s |
| 2: C2 then C1 | C1, sequential pair | 119.95 | 8.517 s | 8.505–8.544 s |
| 2: C2 then C1 | C2, concurrent pair | 172.75 | 11.871 s | 11.744–11.950 s |

A group rate divides its total completion tokens by the interval from the earliest request preparation to the last received response. Each aggregate divides 6144 tokens by the sum of its three group intervals. It does not average per-request rates. C1 includes the gap between its sequential submissions. All rates include prefill and local HTTP/receipt overhead; none is a per-card rate, TTFT or isolated decode measurement.

The campaign executed 38 requests: 24 measured fixed-output requests, six excluded warmup requests and eight natural completions. Each start warmed C1 and C2 separately. The first measured groups remain in the results; the warmups do not prove that every kernel was compilation-warm. Nineteen frozen request bodies, each with a distinct salt and request ID, were reused across two independent server/cache processes. Every execution reported zero cached tokens and zero retractions.

## Execution and quality evidence

All ten C2 groups, including warmup and quality groups, showed actual batch-two graph replays on both ranks. All C1 groups showed singleton execution. Per-request verifier counts reconcile with graph replays and completed commits; no singleton tails occurred in these C2 groups. Both-rank slot checks accounted for all requests and the 19 usable recurrent slots. HTTP overlap alone did not satisfy the C2 gate.

All eight natural completions passed exact final-answer checks after authenticated tokenizer decoding. They repeat two retained short-answer cases in C1 and C2 across both starts. Each final answer used 22 tokens. Forced-output responses were excluded from answer-quality grading. This establishes the small screen, not broad quality or sustained final-output reliability.

All twelve measured C1-versus-C2 request comparisons differed at output token index 16, counting from zero. C1 used 164 verifier calls per request; C2 used 172–177. None met the exact-output-and-work gate. Across starts, 11 of 19 corresponding output/work pairs matched, including all four natural-quality requests. The C2 warmups and measured outputs retained variation. The historical numerical findings remain unresolved; this run supplies no new waiver or numerical-equivalence claim.

Within each start, natural case A also differed between C1 and C2 at token index 149: 242 versus 221 output tokens and 43 versus 41 verifier calls. Its final answer matched. Case B matched in full output IDs and verifier work. Both observations repeated in the second start.

## Capacity and interpretation

The configured context was 32768 and the token pool was 32800. With a 16-token reserve per request, the timing pair required at most 6174 tokens; the quality pair required 13002 tokens, using 5461 inputs and a 1024-output cap per request. This does not demonstrate two full 32K contexts at once.

After graph setup, each rank reported 19.565 GiB allocated and 19.854 GiB reserved by the Torch allocator in both starts. These startup values are not whole-device or campaign peak VRAM.

The retained slot observer reads bounded integer allocator indices outside graphs. Its synchronization cost can differ with batching and was not measured separately. Therefore, use the table as descriptive instrumented service throughput, not an intrinsic kernel speedup or a comparison with the earlier uninstrumented 121 tok/s result.

The observed tradeoff favors C1 for individual latency and C2 for aggregate throughput when both requests fit the pool. No new scheduler policy or default was installed. Broader traffic mixes, long concurrent contexts and endurance remain separate work. UltraQuant KV and BF16-weight controls remain conditional feasibility items.

The single GPU campaign completed with independent resource release. The first offline reader omitted the live result's derived native-group field before exact controller comparison. Its refusal is sealed and retained. A fresh reader recomputed that field from authenticated observations; both actual starts passed, while altered evidence still failed. No GPU rerun was used. Source preparation passed 75 checks; the corrected reader passed 28 focused checks and six retained-data composition checks. See the [provenance record](../provenance/SPRINT39-C2-THROUGHPUT-20261023.json) for exact results and evidence identities.
