# Sprint 31: gate/up warp screen

**Status: `SPRINT31_COMPLETE_WARP_CANDIDATES_PARKED`; independent review approved.** The model-free component compared the selected four-warp common kernel with two- and eight-warp candidates. W2 passed its component gate; W8 is parked. [Provenance](../provenance/SPRINT31-GATEUP-WARPS-20261023.json)

The candidates change only the warp assignment for the untuned HIP/BF16 gate/up shape `(8, 17408, 5120)`. The common kernel uses M16/N128/K128 tiles, group 32 and stage one. Other dispatch and kernel bodies stay unchanged. CPU tests must establish dispatcher behavior separately; the native component invokes the common kernel directly.

Each card ran structured and random cases sequentially. For each case, all six ordering permutations repeat twice, giving 12 timed launches per configuration, six appearances in each pairwise order and four in each position. Each configuration has one warmup and three graph checks per case. All 192 exact comparisons (156 eager and 36 graph) and 144 timed launches passed their planned execution checks.

The baseline must be valid. Each candidate has an independent gate: all four cases and 64 candidate checks must match the 64 baseline checks exactly, compiler warp/stage/resource evidence and raw artifacts must be complete, all four candidate medians must be lower, and release must pass. A candidate compile or numerical refusal is retained without preventing assessment of the other candidate. A failed gate parks that candidate; it does not authorize a retry, model run or selection.

| Device | Input | W4 median (ms) | W2 median (ms) | W8 median (ms) |
|---|---|---:|---:|---:|
| 0 | Structured | 0.183520 | 0.181381 | 0.185320 |
| 0 | Random | 0.182880 | 0.181300 | 0.185261 |
| 1 | Structured | 0.183220 | 0.181640 | 0.186081 |
| 1 | Random | 0.182681 | 0.182080 | 0.186600 |

W2 reduced median kernel time by 1.1655%, 0.8642%, 0.8623% and 0.3290% in those cases; W8 was slower in all four. Reported W4/W2/W8 registers were 114/170/92, with 16 KiB shared memory, zero spills and stage one for each. This synthetic component does not establish a serving gain. Independent review approved W2 eligibility for separately reviewed serving preparation. The W2 forward pilot subsequently failed its serving median gate. CPU checks passed: 33 targeted and 53 full component checks; pilot preflight 1, procfs 10, startup 22, replay 18 and reader 22, with 290 tested hashes. Earlier harness corrections remain recorded.

The forward pilot used two starts and eight fixed-output requests, with one excluded warmup and three measured requests per arm. All four full-output and verification-cycle pairs matched, including warmups: each request produced 1,024 tokens and 164 cycles. The affected shape and unchanged dispatches were authenticated on both ranks and arms, together with source, runtime, cache, FP32 head arithmetic, ring-fold capacity and release.

| Arm | Median latency (s) | Combined TP2 tokens/s |
|---|---:|---:|
| W4 control | 8.481692 | 120.730630 |
| W2 candidate | 8.482985 | 120.712232 |

W2 median latency was 0.01524% higher and rate 0.01524% lower in this small sample. It failed the lower-median gate, so W2 is parked; this does not establish a significant or general slowdown. Native elapsed time was 319.922 seconds. No retry, reverse, new quality screen or source promotion follows. Independent review approved the bounded closeout.

The component limits were: 1,200 seconds active, 600 seconds per worker, 30 seconds cleanup, 300 seconds sealing and 1,500 seconds total; 24 CPUs, 38 GiB memory, 512 tasks, two threads and 320 MiB tmpfs. Output limits are 16 MiB per file, 32 MiB aggregate and 128 members. The separate serving envelope used a 5,400-second owner limit, 1,800-second worker limit, 1,740-second client limit and 5,480-second native wrapper limit. Fresh read-only host and configuration observations are sealed in the provenance record.

The selected `b101` GEMM and aligned ring-fold profile remain unchanged, with FP8 weights, BF16 KV caches and FP32 head arithmetic and recurrent state. The Sprint 31 control pilot measured 120.73 combined TP2 tokens/s; the qualified Sprint 22 range of 121.18–121.24 is historical. The Sprint 26 numerical finding remains unresolved; reliability and endurance testing remain deferred.

At Sprint 31 closeout, the proposed sequence began with a matched BF16-versus-FP8 weight assessment. The current priority is the 32K BF16-KV quality and context baseline, then 64K if fit and quality support it; the weight comparison is deferred. UltraQuant is parked. FP8 target-KV and adaptive speculation remain optional later feasibility work, with endurance after selection. These are future work, not launched or qualified by this screen. See [weight comparison readiness](BF16-FP8-WEIGHT-COMPARISON.md). A historical BF16 host-memory gate failed despite one completed HTTP response.
