# Sprint 24: Current profile refresh

Sprint 24 recorded sampled cost windows for the selected Sprint 22 GEMM and ring-fold profile. Eight natural-stop requests passed their task checks. The two starts produced different outputs and verification work in three of four pairs, so these results do not establish instrumentation overhead or a speedup. The selected profile is unchanged. [Provenance](../provenance/SPRINT24-CURRENT-PROFILE-20261022.json)

The campaign used two starts, eight sampled requests and low thinking. It completed 8 of 8 device windows on each rank, including four long-case windows per rank. Independent release passed; there was no native fault or retry. The native launch command lasted 384.068 seconds. The retained Sprint 22 clean C1 result of 121.18–121.24 tokens/s combined across two tensor-parallel ranks is not a Sprint 24 speed measurement.

Four distinct tasks ran once per start. Input lengths were 226, 246, 8176 and 8176 tokens, with an 8192-token total output cap, temperature 1.0, top-p 0.95 and top-k 20. Thinking remained enabled at low effort; source, precision, caches and timing schedule matched the frozen profile.

| Long-window median, ms | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Verification cycle | 53.2106 | 53.2026 |
| Target graph | 44.1106 | 44.1146 |
| Projection | 27.2339 | 27.2055 |
| Gate/up projection | 11.8391 | 11.8448 |
| MLP | 18.4688 | 18.4674 |
| GDN | 10.3686 | 10.3405 |
| Attention | 4.3184 | 4.3020 |
| All-reduce | 3.2507 | 3.3343 |

These are overlapping forward intervals, not additive costs or fractions of a whole request. Projection intervals include quantization, launches, waits and enclosed collectives; they are not isolated GEMM timings. The long medians use completed samples from the two long cases. Target samples leave 13, 42, 64 and 150 subsequent cycles across the four cases. The separate host schedule leaves 65 held-out tail cycles uncovered; threshold cycles 344 and 356 were beyond its actual stop at 341. Completing the selected device windows does not cover every request cycle. Comparisons to Sprint 20 are historical sampled-window differences, not causal speedups.

Full output IDs and verification-cycle counts matched in 1 of 4 pairs; final text and final IDs matched in 3 of 4. The unique-last case first differed at full-output index 148 in reasoning and produced different final code; both versions passed the functional checks. The two ledger cases first differed in reasoning at indices 16 and 2, with identical final answers. All three unmatched pairs had different verification counts. Every timed full output matched both Sprint 20 arms, while three current control outputs matched neither. This comparison establishes neither global novelty nor a cause. The earlier numerical finding remains unresolved, and sampling distribution equivalence is not established.

The CPU preparation retained fixture-related failures, a parent static-inventory assumption corrected before native preparation, and a pre-execution analysis syntax error. Corrections did not alter the sealed native run. Independent review verified the source, native and reader seals and recomputed the reported medians. The final analysis is sealed.

A proposed next component experiment is common gate/up N64 for shape `(8, 17408, 5120)`, retaining M16, K128, stage 1, group 32, four warps and 128-wide block scales. Source inspection indicates feasibility; it has not been compiled, verified, prepared or queued, and no new execution is authorized by this record. Endurance remains deferred.
