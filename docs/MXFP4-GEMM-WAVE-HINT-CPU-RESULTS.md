# MXFP4 gate/up wave-hint CPU follow-up

The owner's three-case CPU follow-up passed its revised contract. The single
invocation completed in 8.58 seconds including transport, with the child reaped
and no owned process group remaining. Its allowance is consumed and work is
parked. Opus 5.5 accepts the CPU signoff with no blocking findings.

There was no GPU execution, model start, power adjustment, deployment or
selection change. GPU correctness and performance remain unmeasured. The
gate/up candidate is suitable for considering a separate component preparation;
it is not GPU-ready or adoptable.

| Case | Baseline compiler pins | Revised CPU result | Used VGPRs, baseline → candidate |
| --- | --- | --- | --- |
| Down-M16, negative control | All five match | PASS | 109 → 90 |
| Gate/up M16 | All five match | PASS | 109 → 90 |
| Gate/up M8 | All five match | PASS | 109 → 90 |

Only `waves_per_eu` changes, from 1 to 0. Each pair has byte-identical TTIR and
TTGIR; LLVM IR differs only by deletion of the hint attribute. The approved
whole-function and K-loop instruction/operand checks pass, including wait
operands and counts. Exact IR and instruction multisets do not prove device
numerical equality or dataflow isomorphism.

Each case retains 41 SGPRs, 4,096 bytes of dynamic LDS, zero spills and zero
fixed private/scratch bytes. Wave32, 128-thread workgroups, four warps and one
stage are unchanged. The candidate removes the register padding, and its
descriptor passes the approved field checks. The two gate/up shapes share one
machine binary per arm, although their earlier Triton IR differs. Down-M16 uses
the same instruction schedule, with a different loop-count instruction. Thus
the negative control varies grid size and K-loop length; it is not an
independent schedule.

The trade-off remains: each candidate adds 137 dependency-delay instructions
and removes early register release. Code grows by 540 bytes. Compiler-reported
occupancy increases from 5 to 16, but this is a limit, not measured active waves
or a speedup ratio. Only gate/up's 272 programs exceed the inferred baseline
capacity; the 80-program down-M16 case remains a negative control.

Twenty focused local tests and the required 43-check public preflight passed.
The native compiler and comparator ran under the pinned Python 3.14.4 runtime;
the unit suite ran on local Python 3.11. All 30 compiler artifacts are retained.
The offline reader re-executes the same pinned gate implementation and
reproduces the result exactly. Opus independently checked the quantities with
a separate parser and reran all 20 focused tests.

The accepted register-masked instruction comparison can admit semantically
different operand use or scheduling. It carries no proof of numerical
correctness; the future device bitwise checks are mandatory. Assembly and
HSACO are authenticated compiler outputs, but the gate does not prove their
instruction-by-instruction equivalence. Future device tests must bind the
actual HSACO hashes.

The [provenance record](../provenance/MXFP4-GEMM-WAVE-HINT-APPROVED-CPU-20261104.json)
binds the approval, frozen preparation, receipt, artifacts and review status.
The [original experiment](MXFP4-GEMM-WAVE-HINT-SCREEN.md) remains rejected under
its original rule. The v6 layout, codec, integer and SWMMAC dispositions remain
unchanged, as do the original 21 leads and BW-01–04. Serving spend stays HOLD.

Any subsequent component needs a gate/up-only host route, mixed-M fallback and
graph/lifetime checks, both-card compiler identities, unchanged bitwise and
no-tolerance numerical gates, poisoned outputs, matched 235 W timing and
independent release. GPU execution needs separate exact approval.
