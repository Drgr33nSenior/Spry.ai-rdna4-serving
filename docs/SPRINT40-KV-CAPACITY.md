# Sprint 40: BF16 64K capacity and UltraQuant KV decision

The bounded 64K C1 screen passed all **10 long-document answer checks** using the selected FP8 model weights and BF16 target/draft KV caches. Maximum recorded cumulative Torch allocator peaks were **21.403 GiB allocated and 21.734 GiB reserved per card**. Keep UltraQuant KV integration deferred for this workload: the BF16 path fits, while the retained mixed-cache adapter needs graph/speculative integration work.

## Requests and quality

Two fresh starts each executed a short warmup and five long requests. The incident and runbook documents used 59,993 and 59,980 input tokens, with a shared prefix of 59,892 tokens. Their questions explicitly specify JSON types and canonical action codes, reusing Sprint 35's facts and oracles. Facts precede the archive filler; the question follows it. This tests two task families in one evidence layout, with repeats and shared-document branches.

All twelve responses stopped naturally and decoded against the wire text. All ten long responses passed the unchanged strict JSON-value/type checks for this new corpus. Across starts, all six full output-ID and verifier-count pairs matched. Cold, warm and isolated controls also matched their corresponding full output IDs and verifier counts within each start.

The configured context was 65,536, the token pool 65,568, and the output cap 4,096 with a 32-token reserve. Actual long outputs were 516 or 525 tokens. This establishes the frozen roughly 60K-input C1 screen; it does not establish the full output allowance, arbitrary 64K quality, two simultaneous full contexts or endurance.

The earlier [Sprint 33](SPRINT33-LONG-CONTEXT64.md) already demonstrated resident 64K operation. Its original 10/12 strict result and two format failures remain unchanged. Sprint 40 uses explicitly typed questions and a different evidence layout; it does not regrade those failures or resolve the numerical investigation.

## Cache reuse and elapsed time

| Request | Input tokens | Cached tokens | Output tokens | Start 1 HTTP time | Start 2 HTTP time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incident, cold | 59993 | 0 | 516 | 35.216 s | 35.406 s |
| Incident, exact repeat | 59993 | 59392 | 516 | 5.237 s | 5.258 s |
| Runbook, shared-document branch | 59980 | 59392 | 525 | 5.136 s | 5.166 s |
| Runbook, isolated cold namespace | 59980 | 0 | 525 | 34.792 s | 34.871 s |
| Runbook, exact repeat | 59980 | 59392 | 525 | 5.090 s | 5.100 s |

Warm requests reused about 99% of their actual input, with a partial hit on the shared prefix. These are descriptive individual HTTP intervals including prefill, generation and observer overhead. They are not TTFT, isolated decode measurements or a throughput benchmark. The earlier Sprint 39 short-request rates remain separate.

## Memory measurements

Both ranks in both starts reproduced the same maximum recorded cumulative allocator peaks:

| Per-card observation | Value |
| --- | ---: |
| Startup Torch allocation after graph setup | 20.893 GiB |
| Cumulative peak allocated through request-row release | 21.403 GiB |
| Cumulative peak reserved through request-row release | 21.734 GiB |
| Driver-free completion snapshots, rank 0 | 8.781–8.861 GiB |
| Driver-free completion snapshots, rank 1 | 8.959–9.039 GiB |

The source-authenticated hook delegates native request-row release, then records allocator counters once for each frozen request. All 24 request/rank records join the native worker, input-ID hash, cache salt and actual response ID. Unrelated server warmups pass through before text-ID validation. The observer adds no tensor operation, model forward, peak reset or explicit device synchronization; query and serialization overhead are unmeasured.

The peak counters include startup and preceding requests. Driver-free values are point observations after release. Neither measures whole-device peak VRAM or proves that the snapshot's free memory was available throughout generation.

The existing [32K prefill observations](SPRINT36-PREFILL-CHUNKS.md#memory-scope) reached 20.075 GiB peak allocated and 20.408 GiB peak reserved per card with chunk 2048. Their window ends at successful prefill host returns. Preserve that different scope when comparing them with Sprint 40's request-release observations.

## UltraQuant decision

At 65,536 nominal target-cache slots, the retained geometry has 16 full-attention layers, two KV heads per TP rank and head dimension 256. BF16 K/V payload is 2.000 GiB per card. Protecting four layers in BF16 and encoding twelve with G32 E2M1/UE8M0 cells projects 0.898 GiB, a **1.102 GiB target-payload saving**. This is a calculation, not a measured compressed allocation. It excludes reserved/padding rows, other scale metadata, rotations/workspace, draft KV, recurrent state, weights and allocator costs.

The pinned retained source includes the private Qwen BF16 mixed pool and source adapter. It explicitly accepts ordinary eager extend/decode, rejects speculative metadata and graph capture/replay, and copies metadata to the CPU. Its score path can allocate a 256 MiB workspace. Preserving the selected split-attention, fixed DFlash8, graph replay, cache ownership and rollback would require a codec port and separate component/quality work.

The successful BF16 fit supports deferring that integration at the tested capacity. No quantized-KV GPU comparison or codec performance claim was made. Revisit compression if a larger context, retained-document pool or concurrent workload establishes a capacity need.

## Closeout

The existing optional [64K adapter guard](../patches/sglang-dflash-splitkv-context64.patch), b101 W4, aligned ring-fold, 19 usable/20 physical recurrent slots, 2048 prefill chunks, thinking-low greedy profile and explicit BF16 draft-KV override were preserved. Selected numerical sources, installed defaults and resource limits did not change. The single campaign completed in 440.04 seconds with independent resource release and no retry.

Preparation passed six tokenizer fixtures, 61 source/package checks and 29 reader checks. Independent review verified the frozen packet and final readout. [Provenance](../provenance/SPRINT40-KV-CAPACITY-20261023.json) records exact source, native, reader and tokenizer identities. Sprint 41 remains a conditional BF16-weight memory audit; reliability/endurance stays deferred until tuning is exhausted.
