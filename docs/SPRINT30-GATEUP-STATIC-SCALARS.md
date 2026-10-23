# Sprint 30: gate/up static scalars

**Status: `SPRINT30_COMPLETE_CANDIDATE_PARKED`; independent review approved.** The model-free candidate passed all exact comparisons and compiler-specialization checks, but was slower in one of four timing cases. The all-case timing gate failed; no serving preparation, retry or source selection follows. The selected `b101` GEMM and aligned ring-fold profile remain unchanged. [Provenance](../provenance/SPRINT30-GATEUP-STATIC-SCALARS-20261023.json)

The candidate is an exact-shape guarded `@triton.jit` wrapper that forwards literal `M=8`, `N=17408`, `K=5120`, `group_n=128` and `group_k=128` to the unchanged selected common kernel body. It retains the strides, masks, wrapped M16 input rows and dot/scale/add order; the launch uses M16/N128/K128, group 32, four warps and stage one. CPU tests covered dispatcher and fallback behavior; the native component exercised the kernels. This does not authenticate the full candidate serving route. If revisited, serving instrumentation must explicitly authenticate the wrapper route.

Both cards were tested sequentially with structured and random inputs. All 96 comparisons were exact: 72 eager and 24 graph checks, with 64 alternating eager timed launches. Structured checks used an independent oracle; random checks used the selected baseline. All four candidate TTGIR proofs passed: the five scalars were unused or removed, the K loop ran from 0 to 40 in steps of one with one dot, and no integer division or remainder remained in the loop.

| Device | Input | Baseline median (ms) | Static scalar median (ms) |
|---|---|---:|---:|
| 0 | Structured | 0.186176 | 0.184301 |
| 0 | Random | 0.184800 | 0.186111 |
| 1 | Structured | 0.185495 | 0.184681 |
| 1 | Random | 0.185995 | 0.185430 |

The candidate was 0.709% slower for device 0 random input. TTGIR, AMDGCN and HSACO hashes differed for both variants. Reported resources remained 114 registers, 16 KiB shared memory and zero spills. The specialization was demonstrated, but a consistent timing benefit was not. These synthetic component results do not establish serving gain, occupancy or bandwidth effects.

The physical interval was 22.018 seconds, including 4.466 seconds of payload work; the outer interval was 26.463 seconds. No models started, and independent release passed. The inherited `common.py` invalid-escape `SyntaxWarning` was retained; execution and analysis exited zero. The 32 targeted and 52 full/composed CPU checks passed.

A bounded warp-count component is a possible next lead after source and resource review; no count or gain is established and no component is queued. The latest clean serving result remains 121.18–121.24 combined TP2 tokens/s from Sprint 22. The Sprint 26 numerical finding remains unresolved; reliability and endurance testing remain deferred.
