# Sprint 13: lower-stage unrolled down projection

Status: the component passed all 96 exact comparisons but was slower in every
case. Park this configuration. Independent release passed; no serving test follows.
The selected serving source remains `7c8fa3cb3316…`.

The [Sprint 12](SPRINT12-UNROLLED-PROJECTION.md) unrolled stage-three
configuration required 197,120 shared-memory bytes per block against a
65,536-byte hardware limit. Test stage one as a separate configuration. Do not
assume that reducing stage depth guarantees compilation or a speedup.

## One bounded component

Compare the selected common kernel at stage three with the existing native
unrolled kernel at stage one, at exactly `(M,N,K)=(8,5120,8704)`. Both use
tiles `(64,128,128)`, group size 32 and four warps. Preserve FP8 operands,
FP32 block scales and accumulation, and BF16 output. K is divisible by the
unrolled step of 512; this packet excludes partial unrolled steps.

This comparison changes the kernel and stage depth together. It measures that
configuration against the selected route, without attributing a result to
unrolling alone. Extract both kernel definitions unchanged from the sealed
selected source. Bind actual compiled kernel names and stage depth to each
variant; a declared label cannot satisfy those checks.

The private dispatcher changes only the exact untuned HIP/BF16 down shape.
Preserve tuned entries, other shapes, platforms and dtypes, including the
selected 3072, gate/up and prefill overrides. Leave the global selector's
missing return unchanged. Qualification is limited to the two existing R9700
gfx1201 cards; the inherited HIP guard is not architecture-specific.

Test both cards sequentially with structured and random operands. Require all
96 exact comparisons, including 24 graph replays, and 64 alternating eager
timed launches. Keep the analytical structured oracle, random baseline
equality, output poisoning, canaries, input hashes, explicit per-device capture
streams and device-handover allocation guard. Retain compiler refusals and
partial results; do not retry a failed bounded campaign automatically.

Reuse the protected owner helpers and the existing 1,200-second active,
600-second worker, 30-second cleanup and 1,500-second total limits. The
device-tensor ceiling remains 256 MiB. No model starts, installed-runtime
changes or source selection are part of this component. Preserve the current
stack, precision, flags, pools and unresolved numerical contract for any later
serving comparison. Extended reliability remains deferred by owner direction.

Advance to matched serving preparation only if all exact checks pass and the
candidate median event time is lower in every case on both cards. A mixed
timing result does not justify a model campaign. Otherwise retain and park the
result. Serving and affected validation require their own reviewed packets.

## Result

Astra approved the bounded preparation. Both the author and fresh execution
copy passed 36 CPU checks with no native runtime imported on the Mac. The
owned GPU attempt took 23.31 seconds and completed all four cases on both
cards, with 96 exact comparisons and 64 alternating timed launches.

| Card | Pattern | Common stage 3 median, ms | Unrolled stage 1 median, ms |
| --- | --- | --- | --- |
| 0 | Structured | 0.081220 | 0.120900 |
| 0 | Random | 0.080860 | 0.123080 |
| 1 | Structured | 0.082975 | 0.122040 |
| 1 | Random | 0.081280 | 0.123200 |

The candidate takes 47.08–52.21% more time across these synthetic cases. It
fails the predeclared advancement rule, so no model campaign is prepared.
The stage-one kernel compiled with 16 KiB shared memory per block, compared
with 48.5 KiB for the common baseline. Compiler reports show `n_regs=256`
for both, with `n_spills=57` for the unrolled kernel and `0` for the common
kernel. This suggests register pressure as a future tuning lead; it does not
measure occupancy or prove the cause of the slowdown.

The device allocator peak was 44,731,392 bytes, within the 256 MiB ceiling.
Source, runtime, compiled names/stages, retained artifacts and independent
release checks passed. Both evidence roots are sealed. The earlier stage-three
refusal remains retained. No installed runtime, stack or selected source changed.

A smaller unrolled M tile is a possible next component candidate. It needs
separate preparation and review; no further campaign is reserved. Synthetic
component equality does not establish model-state or task-quality equivalence.
See the [evidence index](../provenance/SPRINT13-UNROLLED-STAGE1-20261022.json).
