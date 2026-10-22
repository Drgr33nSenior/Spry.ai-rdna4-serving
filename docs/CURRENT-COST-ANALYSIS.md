# Current serving cost analysis

The newer [Sprint 8 device attribution](SPRINT8-COST-ATTRIBUTION.md) now places
the target verification graph at about 83% of sampled long-request cycle
intervals. The offline analysis below is retained as its historical input;
it does not contain the new device measurements.

The pre-Sprint 8 telemetry identified repeated generation as the dominant
service phase. It does not identify the dominant GPU kernel or split current
time among drafting, target verification, acceptance and commit. This offline
analysis started no models and changed no runtime settings.

## Latest request evidence

The packet available for this offline analysis used the published sampling recipe, low thinking, FP8 target
weights, BF16 target/draft caches and the current Torch/SDK. Four tasks ran once
on each profile. These were instrumented quality requests, not a balanced
performance benchmark. Snapshot and ring-fold outputs can have different lengths.

| Observation | Retained result | Interpretation |
| --- | --- | --- |
| Long requests | 21.83–30.98 seconds, with 8176 input tokens | Whole HTTP request times, not decode-only timing |
| Time after decode was already underway | At least 75.75–85.72% of each long request's reported server interval | Conservative bound from the first decode log and response timestamp; includes CPU work, GPU work, waiting and observer overhead |
| Client minus reported server interval | 2.27–5.68 ms | Small difference; does not rule out CPU work inside the server |
| Long-task emitted tokens per verifier cycle | 7.43–7.54 out of a block of eight | Limited headroom from accepting more tokens per cycle alone |
| Retractions and request graph recaptures | Zero | No observed retraction or recapture explanation for these requests |

The timestamp bound accounts for the logs' one-second resolution. It begins
after generation is already underway and therefore understates that phase.
It is not a GPU occupancy measurement, a time-to-first-token measurement, or a
pure decode interval. The log's input/output throughput counters span reporting
intervals; they are not isolated prefill or per-request decode benchmarks.

Holding emitted output counts fixed, perfect eight-token packing would reduce
verifier-cycle counts by at most 5.60–6.99% on these four long requests. This is
an arithmetic bound on this workload, not a speedup estimate: cycle costs and
later proposals can change. The separate greedy 12272-input-token corpus shows
similar emitted/cycle ratios, 7.39–7.55. Coding tasks have lower ratios. Keep the
greedy and sampled workloads separate.

## Earlier direct GPU timing

The earlier instrumented 2047-input/128-forced-output C1 screen measured 74
complete non-warmup replays per rank. Its target graph median was 43.87 ms;
the output-head interval was 2.051 ms and its vocabulary gather 0.157 ms.
The paired samples place about 94.97% of that target-graph interval outside
the two disjoint head/gather intervals. These per-rank intervals include
waiting. Do not add ranks, transfer those timings onto the latest requests,
or treat the output gather as all tensor-parallel communication.

This narrows the expensive region inside that target graph to the remaining
model body and its waits. It does not rank linear projections, GDN recurrent
work, full attention or layer collectives. The earlier screen also predates the
explicit BF16 draft-cache override. Its graph timing is supporting evidence,
not a current long-context cost decomposition.

## Draft selector candidate

Both ranks in both latest starts report that FlashInfer top-k is unavailable.
The source-pinned DFlash implementation uses `torch.topk` for local vocabulary
selection, followed by a smaller merge top-k. At C1/block eight, the local
selection sees seven rows across 124160 vocabulary entries per rank; the merge
sees seven rows across 32 entries. The selector is included in the draft graph.

This is a concrete path to time. The source warning's claim of roughly halved
throughput is not a local measurement and supplies no twofold speedup estimate.
A CUDA-specific replacement cannot be assumed to work on this HIP stack.

## Next attribution step

Preserve the latest sampled profile to explain its acceptance costs. Record
separate device intervals for draft graph, candidate projection/top-k, target
graph, acceptance and commit, plus host intervals for scheduling and observers.
The sampling witness currently validates and hashes input IDs on every selector
acceptance call; its CPU cost must be isolated. A greedy probe would be a
separate baseline and would not attribute sampled acceptance.

No such profiling packet was prepared or launched by this analysis. Its owner,
source bindings and finite resource budget need the existing launch procedure.
The unresolved numerical mismatch and selected reference remain unchanged.

See the [machine-readable analysis](../provenance/POST-SPRINT7-COST-ANALYSIS-20261022.json) and
[earlier head/gather evidence](../provenance/HEAD-GATHER-TIMING-20261021.json).

The [further optimization plan](NEXT-OPTIMIZATION-PLAN.md) adds the pinned source
audit, fixed timing-sample schedule and conditional experiments. It starts with
phase attribution; it does not select a new kernel or change this analysis.
