# Sprint 9 target graph classification

Status: Sprint 9 complete. Classification selected M8 gate/up projections;
the native component passed exact parity and measured potential gain. Independent
review approved both readouts. [Sprint 10](SPRINT10-M8-SERVING.md) subsequently
completed matched serving and affected quality validation. Sprint 8 selected
this branch because the target graph took
about 83% of the sampled long-request cycle time. See the
[Sprint 8 result](SPRINT8-COST-ATTRIBUTION.md) and
[optimization plan](NEXT-OPTIMIZATION-PLAN.md).

## Diagnostic scope

Reuse the four sampled tasks, low thinking, natural stops and 8192-token total
output cap. Run the selected snapshot in both starts; instrument the second
start only. Preserve Torch 2.14, ROCm SDK 10.2, FP8 target weights, BF16 target and
draft KV, FP32 head/state, DFlash8, singleton split K/V, metadata reuse, M2047
stage one, graph buckets [1,2] and matched capacity. The unresolved numerical
mismatch retains the owner's experimental-session disposition.

The installed SDK includes rocprofv3 1.4.2 with graph attribution and selected
regions. Integrating its finalization with the existing worker cleanup and
bounded artifact transport needs further lifecycle work. This packet uses the
plan's selective-event fallback. No profiler installation or limit increase is
part of the run. The installed source audit and packet are sealed separately.

The hooks time native MLP, GDN and full-attention boundaries, projection methods
and TP2 all-reduces. Source hashes and fixed declaring methods authenticate the
callables. Ordered discovery and capture transcripts, including calling context,
must match across both graph buckets. Capture-only model/group hooks are removed
before external requests. Native return values and exceptions are preserved.
Projection and communication intervals can nest inside model-family intervals;
never add these groups together or add the ranks into request wall time. An
all-reduce interval includes waits and does not measure pure transport cost.

Retain two frozen cycles per task on each rank: 22/49, 54/146, 98/277 and 106/300.
That is at most eight device samples per rank. The original 32-point host schedule
per task remains unchanged and independently validated. Captured timing event
nodes execute on every graph replay, including unsampled cycles. This is an
instrumented classification run; clean performance measurement remains separate.
Report actual sampled prefixes, unreached indices and trailing unmeasured cycles.

## Preparation evidence

The packet passed 119 CPU checks: 39 public recorder/reader checks, 11 actual
startup/cleanup integration checks and 69 package checks. The reader passed 15
checks. Astra found and verified fixes for callable substitution, reordered calls
and lost validation of host rows outside the smaller device sample. Independent
review approved the bounded two-start packet and reader; native results were
pending at approval.

| Limit | Actual | Ceiling |
| --- | ---: | ---: |
| Archive members | 64 | 64 |
| Archive bytes | 16,721,920 | 16,777,216 |
| Combined transport bytes | 33,538,130 | 33,554,432 |
| Workload argv entries per arm | 255 | 256 |
| Control/profile binding bytes | 65,411 / 65,391 | 65,536 each |

All eight protected helpers and six numerical source files match the sealed
Sprint 8 source. The serving diagnostic owner, worker and client time limits remained 5400, 1800
and 1740 seconds. No model, stack, precision, runtime flag or installed default changes.

## Native classification, 22 October 2026

Both starts completed in 394.1 seconds. Source, native, HTTP, worker, startup
cache and independent resource-release checks passed. All eight responses stopped
naturally and passed their functional, format and sustained-final-output gates.
Both ranks retained eight COMPLETE event samples, two per task, with 561 measured
parts. Astra independently verified the grouped sums, coverage and final readout.
See the [sealed result index](../provenance/SPRINT9-TARGET-GRAPH-20261022.json).

The table uses medians of four long-request samples per rank: two from each
ledger request. Rows describe instrumented method intervals and can include
quantization, kernels, launch effects and waits.

| Interval, milliseconds per sampled cycle | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Complete verifier cycle | 57.660 | 57.610 |
| Target verification graph | 48.334 | 48.339 |
| All projection methods | 30.757 | 30.730 |
| MLP family | 20.888 | 20.872 |
| MLP gate/up projections, summed across layers | 13.775 | 13.777 |
| GDN family | 11.468 | 11.428 |
| Full-attention family | 4.603 | 4.580 |
| All-reduce methods | 3.559 | 3.644 |
| Output head | 2.051 | 2.052 |
| Final vocabulary gather | 0.156 | 0.158 |

Projection and communication intervals overlap the model families. Do not add
these rows or combine ranks. The gate/up total spans 64 methods across layers;
it is neither one kernel duration nor isolated GEMM time. This supports a narrow
M8 gate/up launch experiment before top-k, probability scratch or communication
changes. No optimization or serving-speed claim follows from classification.

The target samples cover cycles 22/49, 54/146, 98/277 and 106/300. Their prefix ranges
are 332–489, 559–1032, 8829–10253 and 8862–10386 tokens on both ranks. All indices were
reached. The actual trailing unmeasured cycles are 13, 42, 64 and 198, respectively;
earlier cycles and gaps between selected indices are also unmeasured. Four
long-request samples per rank do not establish a whole-request phase share.

All four final-answer pairs match. Three full output-ID pairs match. The held-out
ledger first differs at output index 130 and uses 2529 versus 2936 reasoning tokens,
450 versus 498 verifier cycles, and 821 final-segment tokens in both starts. Retain
this reasoning variation under the existing unresolved numerical contract.
Astra found no new task-quality failure that warrants reopening the owner-closed
investigation. This does not establish numerical equivalence.

The three equal-ID pairs took 4.30%, 6.84% and 5.12% longer in the instrumented start.
The held-out pair took 16.25% longer while generating 407 additional reasoning
tokens and running 48 more verifier cycles. These one-order observations show
material diagnostic overhead; they do not quantify clean serving performance or
show that the selected uninstrumented configuration regressed.

## Sprint 9.2 component gate

Test only the native M8 gate/up FP8 shape `(8,17408,5120)`, comparing stage depths
3 and 1 with otherwise identical kernel source, tiles, scales and arithmetic.
Keep the existing M2047 dispatcher behavior. The selected native logs report the
default configuration on both ranks; the exact R9700 tuned-config path is also
absent in the fresh read-only observation. Bind the selected dispatcher and
configuration before the component launch; kernel-source equality alone is
insufficient.

Use the current-stack model-free component owner, latest protected helpers,
structured and random synthetic inputs, alternating eager timings, graph replay,
output canaries and readonly operand hashes on both cards. Preserve its finite
1200-second active, 600-second worker and 1500-second total envelope, with 30 seconds
for worker cleanup. Runtime inventory verification and payload execution share
one worker deadline. There are no model starts or serving requests in this test.

Exact native component parity and a measured potential gain remain the Sprint 9
exit. A component win earns the matched serving pilot in Sprint 10; it does not
promote a source patch or installed default. A failed candidate remains retained.

## Component result, 22 October 2026

The corrected component campaign passed all 96 exact comparisons: 72 eager
warmup/timed comparisons and 24 graph-replay comparisons across both cards.
Four cases covered structured and random operands at `(8,17408,5120)`; each stage
had eight alternating timed launches per case. Output poisoning preceded each
checked operation outside its timing interval/capture. Output canaries remained
intact, operand hashes matched, and graph records bound the correct stream device.
The owner completed in 22.5 seconds, including runtime verification and independent
release; payload execution took 4.65 seconds. There were no survivors or unresolved
ownership records.

| Card | Synthetic pattern | Stage 3 median, ms | Stage 1 median, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | Structured | 0.2014 | 0.1751 | 13.06% |
| 0 | Random | 0.1843 | 0.1722 | 6.61% |
| 1 | Structured | 0.1986 | 0.1801 | 9.28% |
| 1 | Random | 0.2004 | 0.1833 | 8.50% |

These are eager component event intervals on each card separately, with eight
samples per median. They include submission effects. Graph replay established
exactness; its performance was not benchmarked. Synthetic inputs are not retained
model operands. No end-to-end serving gain or statistical significance follows.
Compiler-reported shared memory fell from 49,664 to 16,384 bytes and registers
from 256 to 238, with zero reported spills. These reports do not measure hardware
occupancy or full driver memory. Peak allocated Torch tensors were 86.33 MiB in
this component test, not the serving model's memory use.

The first component campaign is retained as an incomplete harness failure.
Card 0 completed both patterns; card 1 passed eager checks and failed during graph
capture. The pinned Torch context manager caches one process-wide default capture
stream, which remained on card 0. The correction passes an explicitly allocated,
device-checked stream. The failed campaign retained 66 exact comparisons and
verified independent release. Its evidence is sealed separately; no result was
discarded or promoted from that partial run. The inherited invalid-escape warning
in the component owner's regex remains in the logs.

The [experimental patch](../patches/sglang-gfx1201-m8-gateup-stages.patch) applies
only to the selected source hash. It changes stage 3 to stage 1 for the exact
untuned M8 gate/up shape and preserves every tuned configuration, including
entries whose values equal the defaults. Existing M2047 handling is unchanged.
A clean application check reproduces the tested candidate bytes. The patch is
an artifact for the current gfx1201 profile; it is not an installed default or
a general HIP optimization. See the
[component evidence](../provenance/SPRINT9-M8-COMPONENT-20261022.json).

Astra verified all comparisons, timings, release evidence and source pins,
and accepted the Sprint 9 component exit. Sprint 10 must test a clean matched serving
pilot and reverse a promising result, then complete affected quality, sampling,
cache and cancellation checks. Retain the selected source until those gates pass.

## Evidence recovery note

After native authentication and readout, a local preparation import generated
`__pycache__/seal_evidence.cpython-311.pyc` in the earlier sealed source folder.
Its full membership seal therefore fails the final audit. All 357 original
members retain their recorded hashes; the extra bytecode matches the compiled
original helper. The original folder is preserved, and a fresh source snapshot
contains every original member plus an explicit recovery record. The native
artifacts, classification results and later component evidence are unchanged.
Use the recovery seal in the result index for future source inspection. The
consumed launch nonce cannot be reused.
