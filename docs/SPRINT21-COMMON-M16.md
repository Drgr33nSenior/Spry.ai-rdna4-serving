# Sprint 21: common M16 gate/up

Status: `SPRINT21_COMPLETE_SELECTED_FOR_NEXT_EXPERIMENTAL_SESSION`. Astra approved the common M16 candidate for the next experimental session only after both serving orders and affected quality screens passed. The aligned ring-fold profile remains unchanged; the previous GEMM remains available for rollback. Installed runtime and persistent defaults are unchanged.

The candidate changes the common kernel's M tile from 64 to 16 only for untuned HIP/BF16 gate/up shape `(8, 17408, 5120)`. Both variants use the same common kernel, N128/K128, group 32, four warps and stage 1. This is distinct from Sprint 11's common M32 and Sprint 15's unrolled M16 experiments.

## Component results

| Device | Input | M64 median, ms | M16 median, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | Structured | 0.190311 | 0.185680 | 2.4334% |
| 0 | Random | 0.189960 | 0.185785 | 2.1978% |
| 1 | Structured | 0.188330 | 0.183855 | 2.3759% |
| 1 | Random | 0.191215 | 0.186591 | 2.4182% |

The component passed 96 exact comparisons: 72 eager and 24 graph comparisons. It used 64 alternating eager timing launches. Structured input used an analytical oracle; random output was compared exactly with the M64 baseline. Compiler records report 238 registers for M64 and 114 for M16, no spills, and 16 KiB shared memory for both. The synthetic inputs ran sequentially per card, not simultaneously under TP2, and were not retained model operands.

The 105-member component source seal is `92f262d678b8b28c55543be6d619bc853b2b48313cc26c48fd1ca75539992323`; the 190-member native seal is `27de5fb6ac9f7ff626e976062ecf31d949094e1b5c8e063fd2af22c4652fec10`.

## Matched serving results

Each order completed two starts and eight requests. Each start used one excluded warmup and three measured requests with 2,047 input tokens, 1,024 forced output tokens, concurrency one and low thinking. Both arms used ring-fold, 19 usable and 20 physical recurrent slots per card, with the original compact replay flag disabled. The control used the previously selected `3f28660b26d00938e907441be64e5e9a74153194058ed9e91b52f8808f1a1390` GEMM; the candidate used `ceec5726fa20815395d2b7f6b44b22fe379a60ef1fb64f3dde89954ae04ee892`.

| Order and arm | Median latency, s | Min–max, s | Combined TP2 output tokens/s |
| --- | ---: | ---: | ---: |
| Forward M64 control | 8.544294 | 8.526720–8.546679 | 119.8461 |
| Forward M16 candidate | 8.511771 | 8.503251–8.523952 | 120.3040 |
| Reverse M16 candidate | 8.507248 | 8.505762–8.508489 | 120.3679 |
| Reverse M64 control | 8.531815 | 8.530665–8.532786 | 120.0214 |

Median latency fell 0.38064% forward and 0.28794% reverse; throughput rose 0.38210% and 0.28878%, respectively. The pooled mean latency decrease was 0.29775%, descriptively. All eight full-output/work pairs matched, including warmups; each request used 164 verifier cycles. Both ranks authenticated the expected common M64 and M16 gate/up dispatches, and both orders passed independent release.

Before the forward launch, the packet passed 22 startup, 18 replay and 20 reader checks. Its archive has 64 members and 16,117,760 bytes; total transport accounting is 32,335,203 bytes. Both argument vectors contain 255 entries; control and candidate bindings are 65,463 and 65,423 bytes. Existing limits are unchanged. Preparation refusals and corrections remain in the evidence history.

## Affected quality and selection

Each quality screen used M16 with ring-fold on both starts. Greedy and publisher-sampled screens each passed 8/8 normal final answers and matched 4/4 full-output pairs, with independent release. The native sampling route was verified; the sampling distribution was not. Lifecycle passed 52/52 normal final answers, two cancellations, actual C2, slot bookkeeping and independent release. Cache comparisons matched 24/26 full outputs and 26/26 final texts. Both arms retained the `concurrent-a` versus `victim-cold-a` first difference at token 149 seen in Sprint 19; its cause is not established.

Across ten starts, the campaign issued 86 requests: 16 fixed-work, 68 normal-answer and two cancellation requests. The selected source is `review/GEMM-tail.py` in the 349-member `fp8-sprint21-greedy._jmxsimr` author, seal `f308b513e2ef73b3b4c6baefe1e20d1edb47a6b4095531f29ace0319322e1690`. The [selected session patch](../patches/sglang-gfx1201-m8-gateup-common-tile16-serving.patch) has SHA-256 `01f6cf5c2b1b7689e62dd026446196d6f0a1afaae0b104fc709d99a6f8948f92`; its exact-application proof has seal `8af95e70eed631ea00b486be70f9363658e562f39c286ecd26076ececf016ba8`. The [provenance record](../provenance/SPRINT21-COMMON-M16-20261022.json) includes the source, native, reader and final-review evidence.

Serving dispatch evidence is derived from live Python and source configuration; it does not establish device constexpr values or occupancy counters. Sprint 21 claims no additional VRAM saving. Earlier numerical and Sprint 11 gate findings remain unresolved; sampling-distribution and production qualification are not established, and endurance is deferred. No GPU campaign is active or queued.
