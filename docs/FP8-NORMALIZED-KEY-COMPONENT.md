# FP32 normalized-key reuse component

Decision: **rejected and unselected**. The exact state checks passed on both GPUs, but every measured cell missed the frozen benefit gate. Astra authenticated the source, actual results and independent release.

The candidate stores the existing FP32 normalized key in a fresh ring, then reuses it during the accepted-prefix fold. It preserves the normalization expression, state-update order, gate arithmetic and BF16 beta rounding. It adds 15 MiB per rank for the selected 48-layer, 20-slot geometry and does not remove materialized state.

## Native results

| Check | Result |
| --- | --- |
| Exact key audits | 268 passed |
| Correctness coverage | 192 prefix/track cases, 76 physical-slot cases, 216 graph replays, 256 continuation windows, 32 singleton-tail windows |
| Captured cost | 48 layers, two private slots; eight timing cells and 128 timed runs |
| Graph/compiler joins | 392 captured full48 native launches joined to warmup HSACO identities |
| Compiler resources | Forward/fold: 203/213 registers, zero reported spills; fold normalization removed, wider key accesses retained |
| Peak memory, each card | 179,772,928 B allocated; 184 MiB reserved |
| Host peak | 1,611,464,704 B |
| Duration | Component 135.195 s; entry 141.285 s; owner 159.488 s |

Total median gains are positive for an improvement and negative for a regression:

| Work | GPU0 | GPU1 |
| --- | ---: | ---: |
| C1, accepted 4 | −1.090% | −1.888% |
| C1, accepted 8 | −1.116% | −1.343% |
| C2, accepted 4 | +2.939% | −0.037% |
| C2, accepted 8 | −4.039% | +0.343% |

GPU0 C2/4 saves 55.86 µs, but misses the 3% threshold, has a 0.337% slower fold and does not reproduce on GPU1. GPU1 C2/8 saves 6.62 µs. Every benefit cell fails; the forward-regression gate passes. Raw C2 variation and all individual samples remain retained.

The events measure captured recurrent work and the commit fold. CPU checkpoint restores, finiteness checks and hashes are outside those events. These are component timings, not model throughput or measured DRAM traffic.

## Validation boundary

Source review caught an oversized temporary in the initial audit: full-checkpoint `isfinite` would exceed the fixed GPU allocation limit. The corrected audit copies each result to CPU once and uses that copy for finiteness and hashing. The numerical kernels and resource limits remain unchanged. The failed preparation and source-executed allocation reproduction remain recorded.

Correctness covers synthetic one-layer/full20-slot execution. Cost covers captured full48/two-slot execution. Actual model operands, production radix, scheduler, convolution and circular-offset behavior are **NOT RUN**. Per-window ring overwrite does not establish circular-offset coverage.

Independent release verified six workload processes absent, an empty private namespace, owned-cgroup absence in both root observations, no GPU holders, zero allocations and a closed journal. There were no cleanup errors or unresolved ownership.

This result is **source-validated, component-tested and hardware-validated** within that boundary. Serving is **NOT RUN**. The candidate closes without promotion or retry. Deeper state algebra is not advanced: this result supplies no evidence to justify its precision and integration costs under the current contract.

The [provenance record](../provenance/FP8-NORMALIZED-KEY-COMPONENT-20261024.json) retains all eight timing cells. Native evidence is `scratchpad/fp8-normkey-v2-run.7vynmfox`, SHA256 `c8d368d2bd7a6d90fc63f04a3fe30d17828c3e6194a42890d10c44fa634355bd` (193 members). Source evidence is sealed with `1810a4ea9c20bd10f0d56e073e0bf5ee70593f60ce31a269d4ac72cd71302985` (51 members).
