# Sprint 19: aligned ring-fold on the selected kernels

Status: both serving orders matched eight full-output ID pairs and 164
verification cycles per request per rank. Ring-fold reduced median latency by
3.87926% forward and 3.83348% in reverse. Greedy and publisher-sampled quality
each passed all eight final answers. Lifecycle quality is active. Ring-fold
remains unselected pending lifecycle and final review; the Sprint 17 snapshot
and source `3f28660b26d0…` stay selected.

Compare snapshot and aligned ring-fold with identical selected kernels, current
Torch 2.14 / ROCm SDK 10.2, FP8 target weights, BF16 target/draft attention KV,
FP32 recurrent state/head, metadata reuse and DFlash block eight. Enable
`--enable-linear-replayssm-spec` in the candidate only. Keep the original
`--enable-linear-replayssm` flag disabled. The aligned helper sources match the
reviewed Sprint 7 implementation; this requires current serving validation.

The first pilot has two fresh starts and eight requests: 2047 input tokens,
1024 forced output tokens, C1, low thinking and greedy sampling. Exclude the
first request in each start and retain three measured requests. Both arms have
19 usable recurrent slots per card; require 20 physical slots including padding
in the native witnesses. Report combined TP2 service throughput and startup
allocator counters per card. These counters do not measure whole-device peak
VRAM. Forced output does not grade natural-stop answers.

Authenticate the current source, runtime, worker, weight, wire, cache, graph,
commit and resource-release evidence. Require matched measured output IDs and
verification work before interpreting a speed comparison. Preserve every
sample, failure and output difference. Both serving orders completed with
independent release. Astra approved the conditional serial quality campaign after both orders and release. No source
promotion follows from the performance results alone.

## Matched serving results

| Order | Snapshot, combined TP2 tokens/s | Ring-fold, combined TP2 tokens/s | Median latency change |
| --- | ---: | ---: | ---: |
| Forward | 115.6911 | 120.3602 | −3.87926% |
| Reverse | 115.6689 | 120.2798 | −3.83348% |

All six measured ring-fold requests were faster than the snapshot results. These
observations do not establish statistical significance. The rates include prefill and are not decode-only or final-answer throughput.
Each start used one excluded warmup and three measured requests.

Both profiles used 19 usable recurrent slots and 20 physical slots per card.
Startup allocated memory after graph setup fell from 22,041,610,240 to
20,296,157,184 bytes on each card in both orders: 1,745,453,056 bytes
(1.62558 GiB) saved per card, or about 3.25 GiB combined. Constructor geometry
predicts 1,747,550,208 bytes; the remaining 2,097,152 bytes are unassigned. These counters are not
whole-device or process-lifetime peak memory. The recurrent ring-fold change
accounts for the measured allocation difference; this is not an engine
comparison. The split-KV computational adapter retains BF16 caches and does not
compress the cache.

The selected GEMM SHA-256 is
`3f28660b26d00938e907441be64e5e9a74153194058ed9e91b52f8808f1a1390`
in both arms. The candidate's only flag difference is
`--enable-linear-replayssm-spec`; the original compact flag remains off in both.

## Quality and preparation

The two-start greedy campaign passed all eight natural-stop final answers.
All four full-output ID pairs and four final-text pairs match across profiles;
reasoning and final token counts match in each pair: 309/41, 786/95,
2126/763 and 3154/821. Both long ledger finals met the 512-token gate, at 763
and 821 tokens. Source, native, HTTP, cache, work and release evidence was
authenticated. This is bounded greedy quality evidence, not general equivalence.

Publisher-sampled quality passed all eight natural-stop final answers. All four
final-text pairs and two of four full-output ID pairs match across profiles.
The threshold and held-out ledger reasoning differed first at token indices 137
and 2; their reasoning counts were 2429/2227 and 2394/2535, respectively.
Both profiles produced 763 and 821 final tokens for those ledger cases. The
native selector-sampling route was verified on both ranks and arms; this does
not establish distribution equivalence.

Lifecycle quality is active. Local packet/startup/reader checks passed 18/35/21
for greedy, 18/35/22 for sampled, and 65 packet plus 18 reader checks for
lifecycle. These checks do not replace native quality evidence.

Preparation retained refusals for fixed-1024 predicates and the exact package
inventory of 62 files and 63 archive members, with matched fixtures. Reverse
bootstrap was restored. A quality-test typo and reader metadata fixture
compatibility were corrected, and the lifecycle tokenizer reference was
restored byte-for-byte from its sealed donor before launch.

Sprint 7's 128-output results do not establish current 1024-output performance.
The earlier numerical findings and owner experimental-session disposition
remain. Full state and distribution equivalence remain unverified; no new
production qualification is claimed. Reliability and endurance remain deferred. Installed
runtime and persistent defaults are unchanged. See the [evidence index](../provenance/SPRINT19-MATCHED-RINGFOLD-20261022.json).
