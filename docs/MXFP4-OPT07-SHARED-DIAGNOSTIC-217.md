# Operation 217: shared diagnostic collected and released

Operation 217 completed both sequential TP2 starts at the configured 235 W cap.
The caller ran from 00:01:46 to 00:25:49 UTC on 4 November 2026 (1,442.69 seconds).
All 60 requests completed normally and passed the unchanged answer-quality check.
Independent owner release is authenticated. The single GPU allowance is spent.

The result is **CAPTURE_AND_QUALITY_COMPLETE_WORK_NOT_COMPARABLE**. This was a
control-only diagnostic, with no optimisation candidate attached. It establishes
neither a serving speedup nor approval to adopt a change. Serving optimisation
spend remains on HOLD.

## Evidence obtained

Each rank recorded 3,779 batches in each start, with no observer errors and
successful hook restoration. Start 1 recorded metadata only. Start 2 exported
all seven planned windows on both ranks: one long-C1 cold prefill and six decode
windows. The native CPU reader validated all 14 captures, device bindings,
annotations and HTTP joins.

The new traces have zero overlapping kernel records within each recorded
stream. Their interval union equals their duration sum in every window.
This observation does not establish clock alignment, a critical path, utilization
or request-level execution cost. The earlier rank-0 timestamp anomaly is absent
in these samples; its original cause is still unknown.

No serving-cgroup OOM, OOM-kill or memory-limit event was recorded in either
start. Host-memory peaks were 17,854,435,328 and 16,583,557,120 bytes, below the
40,802,189,312-byte per-start limit. These are cgroup host-memory observations,
not GPU VRAM or whole-host memory measurements. Grafana investigation remains
parked at the owner's request.

## Why the starts remain not comparable

Both ranks report the same differences. C2 member order differs in 1,085 batches:
930 short-C2 cold and 155 long-C2 cold. The first difference is sequence 969,
where one start admits request 11 and the other admits request 10; the following
prefill order and batch slots are reversed. Slot-indexed output counts,
acceptance counts and two extend-length records consequently also differ.

All 30 per-request output-token hashes, completion counts, cache counts and
speculative counters match between starts. Separately extracting each request's
ordered observed history also produces equal histories. This explanatory
normalization is not used by the acceptance gate. The frozen exact batch-work
comparison remains **NOT_COMPARABLE**. Profiled latency is excluded from
performance scoring regardless, and the request-cost model remains coverage-only.

## Repairs and validation

The CPU reproduction identified an observer compatibility defect: native
SGLang uses an exact array('q') for output IDs, while the observer accepted only
a list. The repair accepts those two exact container types, reads their length,
and preserves all numerical, lifecycle and completeness checks. It also records
bounded failure predicates. The source/package test suite passed 122 checks
locally and under native Python 3.14.4; public preflight passed 43 checks. Opus
accepted the repair before the fresh operation.

The GPU operation succeeded, but the original post-release CPU replay then
refused the new raw-frame transport because it expected a legacy V2 stdout
bundle. The original failure and checkpoint remain retained unchanged.
A fresh offline template replays authenticated payloads through the original
raw-frame receiver and preserves all source, resource, release and HTTP checks.
Ten focused tests passed, and Opus accepted that repair. An independent byte
accounting addendum also matches both recorded wire sizes.

Original wire ordering is authenticated by the live sealed receiver receipts;
it is not independently reconstructed from stripped stdout. This limitation is
explicit. No GPU retry or second generation was needed for the recovery.

The local diagnostic reader refused macOS's address-space-limit operation before
analysis. The same reader then ran in the established Linux CPU-only sandbox
with the original 2 GiB address-space and 120-second CPU limits. Preparatory
mount and file-audit refusals are retained. The final readout exited cleanly.
One separate native CPU grading invocation validated all 60 retained answers in
33.72 seconds, within the 320-second limit. No tolerance or resource limit changed.

## CPU bandwidth investigation

The separate sealed bandwidth input audit reused the existing 93-tile sample;
it read no checkpoint or GPU data. The old records retain separate code and
scale histograms, not aligned pairs, so they cannot determine H(C|S).

An unconditioned three-magnitude palette has no ideal size headroom on those
samples, even before metadata. This does not reject a scale-conditioned palette.
The next useful CPU step is a representative, held-out joint code/scale census
by layer, projection, shard and tile, followed by complete layout accounting.
All 21 original leads, the three cumulative candidates and BW-01 through BW-04
remain recorded. No prior rejected candidate has been relabelled as a gain.

## Reproduction and limits

Run the retained analysis without frameworks, remote access or GPU use:

```sh
/opt/local/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -I -B -S analyze_closeout.py
```

Its output is ANALYSIS.json. SHARED-READOUT.json is the authenticated native
diagnostic output; QUALITY-RESULT.json carries the separate quality verdict.
The run uses the disclosed profiling-twin Torch and cap-3 Mamba experiment in
both starts. It does not qualify the selected serving binary or establish F1
engagement. No serving selection, power setting, installation or adoption changed.


Evidence roots, seals and source pins are in [the operation provenance](../provenance/MXFP4-OPT07-SHARED-DIAGNOSTIC-217-20261104.json). Run the reproduction command from its CPU recovery root.
