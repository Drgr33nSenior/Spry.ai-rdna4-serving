# Sprint 27: gate/up cost isolation

**Status: `SPRINT27_COMPLETE_BOUNDED_COST_CLASSIFICATION`.** The two-start diagnostic completed and independent result review approved the bounded findings. It used the selected `b101` GEMM and aligned ring-fold profile. The proposed next component changes pipeline stages only; it has not been prepared or run. [Provenance](../provenance/SPRINT27-GATEUP-COST-20261023.json)

Both starts used four frozen sampled tasks with low thinking, FP8 target weights, BF16 caches and FP32 recurrent state and language-model head. Control added no new events; the instrumented arm captured activation quantization and GEMM intervals at MLP indices 0, 21, 42 and 63. Arithmetic, RNG behavior and the selected source remained unchanged. The native command took 385.174 seconds. All eight answers stopped normally and passed functional checks; independent release passed. All four full-output and work pairs matched, with 62, 188, 341 and 450 verification cycles and 333, 1,101, 2,590 and 3,352 output tokens respectively.

All eight planned device windows per rank completed, including four long windows per rank. The four sites were observed at active C1; C2 bucket capture supplied geometry only. Each observed FP8 weight had shape `[17408, 5120]`, an 85 MiB payload and 21,760 bytes of scales per rank and call. The table shows medians across the four long windows for the **sum of the four selected layers**, in milliseconds.

| Interval | Rank 0 | Rank 1 |
|---|---:|---:|
| Parent gate/up | 0.800066 | 0.798065 |
| Activation quantization | 0.025800 | 0.025820 |
| GEMM | 0.728785 | 0.725965 |
| Signed parent minus children | 0.045580 | 0.046081 |

The medians of the per-window shares were 3.220% and 3.239% for quantization, 91.060% and 91.000% for GEMM, and 5.707% and 5.768% for the residual. These are not ratios of the independently calculated medians. GEMM dominates these four measured sites; quantization does not. The positive residual contains unattributed work and event effects, and is not automatically removable.

The four weight payloads divided by summed GEMM time give nominal 489.19 and 491.09 GB/s per card, about 76.4% and 76.7% of the [AMD-advertised 640 GB/s](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/radeon-ai-pro-r9700-datasheet.pdf). This is not measured VRAM traffic, saturation or available headroom. The sample does not extrapolate to the 64 inclusive gate/up calls or establish a new serving-speed result.

The proposed next component compares the current common HIP/BF16 gate/up `(8,17408,5120)` with one versus two pipeline stages, retaining M16/N128/K128, group 32, four warps, the kernel body and block scales. Compilation, exact output parity, compiler resource checks and timings on both GPUs are required. No candidate has been compiled, prepared or run, and no gain is promised. The bounded Sprint 11 review did not find this exact test; it does not establish that the test was never attempted elsewhere.

The initial local staging attempt stopped before upload because an older donor authorization omitted the sampled-lane field; the failure and reviewed recovery are retained. CPU readout fixture failures and their corrections are also retained. Final source, reader, grading and release checks passed. Sprint 26's numerical variation remains unresolved; no numerical source, production default or selected profile changed. The latest clean serving benchmark remains 121.18–121.24 combined TP2 tokens/s from Sprint 22. Endurance remains deferred.
