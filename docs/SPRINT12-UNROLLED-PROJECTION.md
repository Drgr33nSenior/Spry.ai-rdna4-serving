# Sprint 12: exact-shape unrolled projection

Status: the reviewed stage-three configuration is parked after one failed
component attempt. The unrolled kernel exceeded the GPU shared-memory limit;
the worker stopped before the second card. Independent release passed.
At this attempt, the selected serving source was
`cafc15b03fbb…`, with the current stack, precision, flags and pools.

The owner directed this tuning lead after the
[Sprint 11 closeout](SPRINT11-PROJECTION-TUNING.md). Sprint 11's 3072 option
was parked: both forward comparisons had unequal verification-cycle work
despite exact output matches. That result does not identify the variation's
cause or establish a candidate quality regression. Preserve its failed gates.
The owner subsequently requested a separate
[experimental selection assessment](M8-3072-EXPERIMENTAL.md).

## One component comparison

The pinned HIP helper `use_w8a8_block_fp8_matmul_unrolledx4` computes its
workgroup predicate without returning it. The current selector therefore uses
the common kernel. The helper's intent and the alternate kernel's performance
remain unverified. This packet leaves that helper unchanged.

Compare the existing native common and unrolled kernels at exactly
`(M,N,K)=(8,5120,8704)`, the observed untuned MLP down-projection shape.
Both variants use tiles `(64,128,128)`, group size 32, four warps and stage
depth three. Preserve FP8 input tensors, FP32 block scales and accumulation,
and BF16 output. The K dimension is divisible by the unrolled kernel's
four-block step, 512, so this comparison excludes a partial unrolled step.

The private candidate changes only the kernel choice for that exact untuned
HIP/BF16 shape and configuration. Tuned entries, other shapes, dtypes and
platforms keep their prior route. Gate/up stage one and the existing prefill
overrides remain unchanged. The inherited HIP guard is not architecture-specific;
any result applies only to the two tested R9700 gfx1201 cards.

The component driver extracts both native definitions from the sealed selected
source without changing their ASTs. Variant zero binds the common kernel;
variant one binds the unrolled kernel. Actual compiled names and stage depth
must match the declared variant. A label cannot substitute for kernel identity.

## Gates and limits

Reuse the finite model-free owner and protected helpers. Test both cards
sequentially with structured and random synthetic operands. Require the
structured analytical oracle and random baseline equality, output poisoning,
canaries, input immutability and explicit per-device graph capture streams.
A complete pass has 72 eager and 24 graph-replay exact comparisons, plus 64
alternating eager timing launches. Graph replay checks correctness; it is not
a graph performance benchmark.

The actual attempt completed two GPU0 cases before stopping. The common
baseline passed 24 exact checks. The unrolled candidate could not launch:
compilation required 197,120 shared-memory bytes per block, exceeding the
65,536-byte hardware limit. It produced no candidate output or timing result.
The subsequent `previous-device tensors remain live` guard stopped the worker
before GPU1. The within-worker allocation-liveness cause remains unverified;
the independent observer confirmed process/resource release after exit.

The owned attempt took 24.75 seconds, exited nonzero and retained its partial
report and both remote seals. This is not a 96-comparison pass, and it supplies
no unrolled parity or speedup claim. Astra confirmed the rejection. No automatic
retry followed. A different stage depth or tile would need a separate reviewed
component; this result does not establish that every unrolled configuration is
infeasible.

Keep 1,200 active seconds, a 600-second worker, 30-second cleanup, 1,500 total
seconds and the existing CPU, RAM, process and scratch ceilings. The declared
device-tensor ceiling remains 256 MiB. No model starts or installed-runtime
change are part of this component packet. Preserve every compilation or
numerical refusal; a rejected candidate receives no speedup claim or automatic
retry.

Final preparation passed 35 CPU checks with no native runtime imported on the
Mac. Fresh read-only observations confirmed the current boot, compiler-source
identities and absence of the relevant device-specific tuned configurations.
The existing scoped sudo approval remains applicable to this bounded owner
procedure.

A component win would require separate matched serving and affected quality
validation before selection. Component equality does not establish model-state
equivalence or task quality. Extended reliability remains deferred by owner
direction. See the [evidence index](../provenance/SPRINT12-UNROLLED-PROJECTION-20261022.json).
