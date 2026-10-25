# C2 singleton metadata component

**Component-tested on both cards; serving-tested in both startup orders; unselected because the benefit gate failed.** The geometric C2 throughput gain was **1.983077%**, below the frozen **2%** minimum. Selected defaults remain unchanged.

## Serving result

Each order completed two arms and sixty requests. Fourteen native final-answer observations across four unique inputs/oracles, two isolated code graders and independent GPU/CPU release passed in each order. The original AB screen had independent Astra review; its single-order benefit was provisional. The later BA result closes the declared benefit screen:

| Startup order | C2 aggregate throughput change | C1 median request-latency change |
| --- | ---: | ---: |
| Control then candidate, AB | +3.063820% | +0.094148% |
| Candidate then control, BA | +0.913667% | −0.024164% |
| Geometric C2 result | **+1.983077%; below 2%** | — |

BA control/candidate medians were 174.78/176.38 aggregate output tok/s and 11.72/11.61 seconds individual latency. These descriptive service rates include prefill and HTTP completion across both GPUs. They are not per-card or isolated decode rates; TTFT and inter-token latency were not measured. All six timed C1 output/work pairs matched in each order; all six timed C2 pairs differed. The result therefore establishes no matched-work kernel speedup.

The shared validation menu was strengthened in BA after timing. Both ranks observed successful native batch-size-two sampled-selector returns for the two sampled requests, as well as the singleton route. This extends the earlier singleton-only sampled observation; it does not establish sampling-distribution correctness. Operational cancellation, survivor, slot and cache-isolation checks passed, but numerical survivor/state equivalence was not established. New held-out task families were **NOT RUN** after the benefit prerequisite failed. Four unique short-answer oracles and repeated code checks do not establish broad task accuracy.

The derived disposition is `C2-REOPENED-AB-BA-DISPOSITION-20261025.json`, SHA256 `c7cd967d97627958778b34eae131eaa433759764de598ea4bfdacca71046dd54`, under the private programme root `scratchpad/fp8-remaining-optimizations.w3m3xpg7`. The BA native root is `fp8-c2-metadata-reopened-ba-native.30537332e92e3cea`; its independent GPU-release and native-quality/CPU-release receipts have SHA256 `f00a4ff74694213878908704aab47f798ccd5957155443350c65110e8f15a08b` and `2f2c810301ef07f06e4bb75258a406097c1f7ab400749f6f3b883f0989200213`. These are evidence identifiers, not public filesystem links. The [current programme provenance](../provenance/REMAINING-FP8-OPTIMIZATIONS-20261025.json) reconciles the later serving result without rewriting the historical component record.

## Component result

The component passed 34 cases, 68 captures, 1,768 checked graph replays and 1,312 timings. Astra independently replayed its reader, authenticated compiler and timing evidence, and verified release.

The isolated metadata candidate replaces sixteen launches with one grid of sixteen programs. Each program calls the complete native singleton policy with its original parameters. It preserves the row policy rather than applying the native sixteen-sequence heuristic. All 144 paired timings favoured fusion: median ratios were 0.18469–0.18757, with nonoverlapping ranges and consistent execution-order subsets. Metadata time fell from approximately 54.2–54.8 µs to 10.0–10.3 µs.

The complete attention comparison includes the same cache writes, metadata, attention and reductions. Its baseline is selected native C2 extend. Its candidate is the two-request split adapter with fused singleton metadata. These full-route gains cannot be attributed solely to metadata fusion.

| Prefix lengths | Card 0 native → candidate, ms | Card 1 native → candidate, ms |
| --- | ---: | ---: |
| 0 / 31 | 0.05060 → 0.04896 | 0.05114 → 0.04924 |
| 31 / 2040 | 0.45464 → 0.07896 | 0.40184 → 0.07134 |
| 2040 / 2040 | 0.39658 → 0.09040 | 0.39550 → 0.09040 |
| 8192 / 1024 | 1.48524 → 0.15022 | 1.47716 → 0.14764 |
| 26000 / 127 | 4.69527 → 0.33428 | 4.65495 → 0.33220 |

These are warmed component event medians. Short-context steady latency improved 3.24% and 3.72%, with 24/24 paired wins per card. Short-context medians after the declared 64 MiB zero-write passed the aggregate regression gate, with ratios 0.99545 and 1.00737. Preserve the second card's 0.30928 ms native outlier and order sensitivity: one order subset reached 1.02158. This cell does not establish a stable flush speedup or proven cold-cache behavior.

Candidate outputs matched the selected singleton reference exactly in the declared checks. Native C2 extend matched its own eager reference. Those two references are not bitwise identical: observed maximum absolute differences reached 0.0078125. This is an observation, not a general accuracy bound or full-model numerical qualification.

Both cards retained gfx1201 AMDGCN and AMDGPU ELF metadata artifacts. Fused/singleton metadata used 21/22 registers, no spills, four warps and two stages. These are integer kernels; they establish no WMMA mechanism for full attention. Full source, compatibility, graph poisoning, live lengths, canaries, causal isolation, slot permutation and stream checks passed. The compiler guard covers source, capture, key and parameter state; SDK dispatch relies on exact pre-import admission.

Peak allocated device memory was 261,237,248 bytes, below 256 MiB. Host peak was 7,429,144,576 bytes. The owner completed in 50.46 seconds and retained 2,163,023 evidence bytes. Independent release found no remaining owned processes, private namespace, scope, GPU holders or allocations; cleanup errors were empty.

The [component provenance](../provenance/FP8-C2-SINGLETON-METADATA-20261024.json) retains every component case median/range, source pin, seal and review limitation. Its historical scope is separate from the later serving disposition. Original failed attempts remain in the append-only evidence; successful readouts do not replace them. The [current programme record](REMAINING-FP8-OPTIMIZATIONS.md) records the retained baseline and outstanding final qualification. Rollback is non-adoption. This candidate is neither selected nor deployed and supplies no production qualification.
