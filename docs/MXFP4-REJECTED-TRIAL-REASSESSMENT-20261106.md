# Rejected-trial reassessment, 6 November 2026

Status: **three-member cap16 candidate selected; programme parked for Qwen4**.

Final disposition, 6 November: the owner selected operation232's tested mtp
arm, retained control as rollback and parked the optimisation programme. No
more GPU work is queued. The v6-QB follow-up remains parked without a device
correctness or speed verdict; cap32 is dropped and BW-03 remains future-only.
Use the [Qwen4 handoff](QWEN4-SERVING-HANDOFF.md) for the actual selection,
tested limits and resume procedure. The pending-selection states below are
historical.

The accepted review's mechanical conditions are closed. One read-only
observation and the unchanged writer appended checkpoint 619, preserving 618
and the original failed result. Independent release is authenticated, the
owner is idle and GPU allowance remains 0. No GPU work was repeated. v6-QB
stopped at the exact memory-accounting guard before correctness probes or
timing; TP0's outcome remains unknown. The allocator-counter mismatch and SSH
stall cause are unproven. Park this optional lead without a performance or
numerical verdict. Final selection of the confirmed three-member cap16 bundle
and programme parking remain separate pending steps.

The reviewed four-step recovery authenticated release without repeating the
workload. TP1's retained witness identifies the exact memory-accounting guard;
TP0 left no witness. Correctness probes and timing were not reached. The
separate CPU reconciliation is prepared, but checkpoint618 remains unchanged
until independent review and a fresh scope/journal absence observation pass.
No GPU retry or selection change is authorized. Earlier accounts are historical.

The owner approved the exact witness-enabled attempt. Its physical-owner SSH
stopped responding with exit255, and no local correctness or timing result was
produced. Checkpoint618 records release unverified. The host is reachable again
on the same boot; an unsealed record claims cleanup after stage-A failure and
stageB not started. Authenticate that evidence before accepting release. The
allowance is consumed, and no workload retry or selection change is authorized.
The earlier preparation entries below are historical.

Independent final delta review `601addd9…` accepts the packet with no blockers.
Readiness `473c0d98…` binds that exact review. Final PLAN `fccaf0c4…` and the
three campaign scripts pass mechanical closure; the final registration root
is sealed `dae9e087…`. A new owner decision and fresh remote admission remain
required. Any stage-A refusal consumes the invocation and forbids stage B and
retry. No GPU work, model start or registration occurred. Earlier preparation
states below are historical.

Latest preparation, 6 November: after review, the witness repair was integrated
into fresh packet `93dc847c81eb3802` (`47503930…`). Every frame now starts with a
newline, and `PYTORCH_HIP_ALLOC_CONF` is recorded. One device-free native run
passed 28 tests on Python 3.14.4 (`a40d3faf…`). The registration preparation
(`15f8f4d7…`) refuses every consumed op230 pin. The experiment, the memory
guard, the kernels and the timing thresholds are unchanged. An independent
final review, a new owner GPU decision and fresh admission remain. Stage A may
refuse again at the same guard.

Earlier status: v6-QB failure reviewed; CPU witness repair tested, unintegrated. The latest diagnostic
review accepts the corrected CPU readout. Its output matches the required
SHA-256 `88b6e713d754ba26ec75b3a5cb63343616c50c53084a18414f1842a98f2a22d2`.
The original STAGE2_INCOMPLETE result remains unchanged. Final selection and
programme parking remain pending.

Operation230 consumed the authorized invocation and stopped at stage A's exact
steady-padding allocation check. Real-input probes and component timing did not
run; stage B was forbidden. Independent release passes at checkpoint616.
The actual failed allocation deltas were not published, so the underlying
cause remains unproven. No numerical, capacity or performance conclusion follows.
There is no retry, adoption or selection change. Independent review accepts
the failure and release. Only TP1 has a retained refusal; TP0's result is
unknown. The +1,966,080 B message states the expected delta. The monitor was
complete; its final validator refused child137 because it required exit0.
The approved CPU repair adds a bounded stderr witness before the unchanged
guard. All 18 tests pass through the executed capture code. This is an
unintegrated candidate without independent repair review or native proof,
and the old missing measurements remain unavailable. The earlier preparation
states below remain historical.

Update after CPU follow-up: independent review accepts the dispositions below
and the completed native CPU fact collection. The single approved invocation
is consumed. The owner subsequently authorized one finite GPU correctness and
speed test for v6-QB, accepting the narrow A6 recommendation for this test only.
The composed packet and native proof pass independent review with findings.
Final plan/grant construction passes the CPU registrar proof. The exact
helper-exception answer and fresh admission remain open; no GPU work has run.

The audit covered the 25-ID MXFP4 reconciliation, the historical serving-sprint
index, and the detailed compiler, projection, chunk-size and HRX records cited
below. This was a document and receipt-summary audit, with no old benchmark
rerun or new native work. It found one current MXFP4 policy-sensitive candidate,
one runtime candidate with a fixed source blocker, and two conditional groups.
It did not establish that recent bugs caused any old performance failure.

## Shortlist

| Priority | Candidate | Original reason it stopped | Why reconsideration is justified | Smallest next step |
| --- | --- | --- | --- | --- |
| 1 | **MXFP4 v6 weight prepacking**, L01/G2 | Compiler-used VGPR count exceeded the frozen baseline ceiling; it never reached GPU timing. | Independent review found the same encoded hardware VGPR allocation. The count gate did not demonstrate a hardware allocation increase or a slowdown. Current profiling identifies decode GEMMs as a major cost. | CPU-only all-shape allocation and load-dependency audit; propose an explicit resource contract before any new measurement. |
| 2 | **Updated HRX runtime**, L03/H2-related | Old runtime had a PyTorch event-recording blocker; its legacy-path microbenchmark showed no GPU dispatch benefit. | The assessed upstream revision implements the missing event operation and contains native-path recording changes. This is a changed prerequisite, not evidence that the old benchmark was wrong. | Isolated source/compatibility assessment of events, actual RCCL APIs and the intended backend. Decide whether exposed overhead warrants a new finite experiment. |
| 3, conditional | **M4096 prefill stage-1 route plus chunk-size reassessment** | The larger-chunk serving screen had unequal generated work and bypassed the M2048 specialization, so it could not establish a causal ranking. | A separate M4096 component already passed exact checks and showed a synthetic timing benefit, but never reached serving. This is an unqualified integration opportunity, not a rejected component. | Confirm the exact current prefill route and model shapes, then assess a matched-work TTFT/peak-memory design. Keep 2048 selected meanwhile. |
| 4, low priority | **Small-positive historical FP8 projection candidates**: Sprint 18 attention QKV, Sprint 25 N64 gate/up, Sprint 31 W2 gate/up | Components passed, but serving effects were tiny, inconsistent or insufficient to justify the remaining qualification cost. | These are not demonstrated general slowdowns. A prospective cumulative assessment could retain small component benefits. No specific repaired bug has been tied to these outcomes. | Check source/shape compatibility first. They target historical FP8 kernels and are not patches for the current MXFP4 GEMM. Prefer a future FP8/Qwen4 qualification if applicable. |

### 1. Weight prepacking: strongest current-deployment candidate

V6 removes shared-memory weight conversion and barrier work while preserving
the operand conversion and carried accumulation sequence at source/IR level.
The original compiler-count failure remains valid under its original contract.
The [signoff amendment](../provenance/MXFP4-FOLLOWUP-SPRINTS-20261103.json)
records why that count is not itself a hardware allocation threshold.

V6 also waits for weight loads much sooner and has more load-count waits.
Those costs could erase the saving. Reopening requires all-shape headers,
runtime allocation assumptions, exact packing/tail tests and load-dependency
evidence. Do not waive the gate retrospectively or rerun an unchanged packet.
The wave-hint approval is not a waiver for v6. See
[the CPU screen](MXFP4-FOLLOWUP-SPRINTS.md).

### 2. HRX: a fixed blocker, with uncertain performance relevance

The [retained upstream assessment](HRX-UPSTREAM-REASSESSMENT-20261105.md)
identifies the event fix in a pinned 5 November source snapshot. This audit did
not refresh upstream again. RCCL compatibility and gfx1201 native-path benefit
remain unqualified. The old legacy-path result still stands.

The new profile shows that the host mostly waits inside the observed decode
rounds. Faster host submission alone therefore has weak support. Instrumented
graph gaps do not establish an unprofiled runtime opportunity. Keep the active
HIP runtime unchanged while assessing this candidate.

### 3. Prefill: a confounded comparison and an unused component win

[Sprint 36](SPRINT36-PREFILL-CHUNKS.md) changed both chunk geometry and the GEMM
route. Outputs and verification counts differed, so complete-request durations
cannot isolate the chunk effect. The separate
[M4096 component](../provenance/FP8-M4096-STAGE1-COMPONENT-20261023.json)
passed 720 exact comparisons and was faster across its synthetic cases. It
establishes neither serving TTFT nor quality, retained-model exactness or peak
VRAM. Current MXFP4 prefill composition must be checked afresh; the historical
FP8 route cannot be assumed to transfer. Decode remains the stronger current
profile target.

### 4. Historical small gains: reserve options

[Sprint 18](SPRINT18-ATTENTION-QKV.md) passed both serving median gates but kept
an adverse outlier and incomplete quality/lifecycle coverage.
[Sprint 25](SPRINT25-GATEUP-N64.md) and
[Sprint 31 W2](SPRINT31-GATEUP-WARPS.md) each failed a one-order lower-median
gate by a very small amount. Their outputs and work matched; no known bug
invalidates those measurements. This supports reviewing their economics, not
calling them recovered winners. Any new acceptance rule must be frozen before
new measurements and preserve numerical and regression gates.

## Keep closed or separate

- **Already recovered:** verification prefetch, local F1 fusion and gate/up
  wave-hint removal now form the confirmed three-member cap16 bundle. Do not
  duplicate their earlier harness repairs or standalone-threshold reconsideration.
- **Still valid failures:** scalar FP4 decoding with spills, C4 int32/index-ahead
  spill increases, the oversized unrolled stage-3 configuration, and specific
  codec size/coverage failures. This audit found no changed premise for them.
- **No known fix for old parity failures:** Sprint 11 tile32 output differences
  and other retained reasoning/rounding counterexamples remain unresolved. The
  later 3072 experimental selection already addressed that branch separately.
- **Measured regressions without a changed mechanism:** Sprint 13, Sprint 23,
  Sprint 28, Sprint 30 and Sprint 31 W8 remain parked. Tiny serving effects in
  Sprint 15 did not hold in reverse order; it is not a priority reopening.
- **Deferred numerical work:** native FP8/BF8 arithmetic with MXFP4 storage,
  integer reduction and SWMMAC still need their production-equivalence proofs.
  Profiling does not remove scale-range or accumulation-order obligations.
- **Owner dispositions:** cap32 stays dropped. BW-03 stays an open research
  option, without a measured gain. BW-04, collective fusion, cache-tree and
  state-protocol work retain their original evidence requirements.

## Diagnostic limits that carry forward

GEMM shares of 66–70% short / about 45% long refer to the VERIFY span; whole-GPU-
round shares are 52–57% / 38–39%. Host-round profiler inflation spans 2.9–5.9%
long and 4.9–9.9% short. The original table used two-round means; individual
RCCL rounds reach 4.04 ms and include peer waiting. Round-1 calibration widths
are 66.3–92.8 microseconds, with a twice-width claim threshold. The exact short
attention change is +1.7% to +3.3%. Clock/GEMM correlation remains a hypothesis
from request-group samples, not kernel-aligned clock readings.

Four prefill calibration chains and 22 of 40 resource windows remain unresolved.
Hardware counters, decoded gpu_metrics, cross-rank clocks and kernel geometries
are unavailable. Inventory matching is not broader physical completeness.
These are future research limits, not blockers identified by the result review.
Selection still uses the separately passing op226/op227 results, never pooled
or replaced by instrumented timing. No benchmark, deployment or selection is
authorised by this shortlist.

## CPU follow-up and next finite decision

The follow-up passed 26 local tests and independent review. General v6 packing
would change tensors shared with mid-M and prefill consumers, while gate/up
already uses wave0. Those routes are parked. Only the existing verification-only
QKVZ+BA copy advances to a CPU compiler-fact collection step. Its approximately
19.7% share is of instrumented candidate MXFP4 GEMM time, not total round time;
it is 8.4–12.1% of summed candidate kernel time. No speedup is predicted.

The new frozen reader mechanically checks both scale and NaN-mask mappings,
including the review's type-consistent wrong-broadcast counterexample. Eighteen
focused tests, the retained-control CLI and nine unchanged input seals close
the review condition without another full review. Reader root
`v6-qb-scale-map-reader-repair-20261106-v1` has seal `a9fdabb0…`.

The owner approved the exact five-compile CPU invocation. A fresh read-only
preflight passed, the caller exited 0, and the frozen readout passed both M8/M16
specializations. Execution `1e3e928f…` and readout `7edb7e70…` remain sealed.
Independent result review `47d6b7e3…` accepts the facts with no blockers.
The allowance is consumed; no retry remains. These are static compiler facts;
device correctness and speed are untested.

M8 and M16 are two specializations with one binary per arm. G1 must obtain the
dynamic LDS size (baseline 4096 B, candidate 1024 B), four warps, grid 129,
constants, attributes and signature from the authenticated wrapper, rather than
reconstructing these launch inputs from the frozen reader's output.

The owner's GPU-test authorization accepts A6 only for this verify-mode QKVZ+BA
M8/M16 K5120 test with the frozen wave1 launch. A6 makes used VGPR count reported
but non-binding, while preserving the exact descriptor, float-mode, LDS,
spill/scratch and allocation-opcode predicates in the frozen contract. The old
109-used-VGPR gate remains unchanged outside this test.

The new CPU preparation is sealed at `3981530e…` (110 members), with 65 local
tests passing. It implements the packer/launcher (G1), memory accounting (G2),
component corpus, frozen classification and readout. Review findings were repaired
in `5acdea52…` with 35 affected tests passing. The composed packet `b7301234…`
and native proof `92a9ddc8…` pass independent review `9d6b76b8…`.
Real-input qualification needs one TP2 model start to observe all 48 GDN layers
at M8/M16 on both ranks. The proposed route follows that with a zero-start
component correctness/timing phase. Final registration root `c74fa026…` passes
the CPU registrar proof (one positive case, 13 mutation refusals). The exact
owner helper-exception answer and fresh remote admission remain unfinished.
Stage A has no power-cap readback or graph replay; later in-process compiler
hash drift remains a refusal risk. Component timing may be cache-resident and
is no serving-speed claim. No registration, GPU or model start occurred. This authorization
allows one test with no retry and no serving adoption or selection change.

HRX holds for actual startup RCCL and unprofiled dispatch-gap evidence. Eight of
the 18 decode event-recording calls may be instrumentation. P1 shares the old
M4096 configuration but uses a copied kernel, so the historical component gain
does not transfer; all per-chunk activations grow with the chunk. Both FP8 decode
GEMMs are heads, so the old projection patches remain inapplicable. These held
items do not block final selection and parking of the confirmed three-member
cap16 bundle.
