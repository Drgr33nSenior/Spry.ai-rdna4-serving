# Sprint 28: gate/up stage two component

**Status: `SPRINT28_COMPLETE_CANDIDATE_PARKED`; independent review approved.** The model-free stage-two candidate passed all exact comparisons but was slower in two of four timing cases. It failed the all-case timing gate, so no serving preparation or run follows. The selected `b101` GEMM and aligned ring-fold profile remain unchanged. [Provenance](../provenance/SPRINT28-GATEUP-STAGE2-20261023.json)

The candidate changed only `num_stages` from one to two in the existing common HIP/BF16 gate/up guard for `(8,17408,5120)`. M16/N128/K128, group 32, four warps, the kernel body and block scales were unchanged. Both cards were tested sequentially with structured and random inputs. All 96 comparisons were exact: 72 eager checks, including warmup, and 24 graph checks. Structured checks used an independent oracle; random checks used the selected stage-one baseline. The component performed 64 alternating timed launches. CPU tests proved the guarded candidate dispatcher; the native component invoked the same selected common kernel with stage-one and stage-two launch parameters.

| Device | Input | Stage one median (ms) | Stage two median (ms) |
|---|---|---:|---:|
| 0 | Structured | 0.220265 | 0.196801 |
| 0 | Random | 0.187311 | 0.187660 |
| 1 | Structured | 0.185355 | 0.187211 |
| 1 | Random | 0.191985 | 0.189841 |

TTGIR, AMDGCN and HSACO hashes differed for both variants in every case. Reported registers changed from 114 to 106 and shared memory from 16,384 to 18,944 bytes; reported spills remained zero. Missing resource attributes remain unknown. Graph replay checked exactness; timings used eager GPU events. These synthetic inputs are not retained model operands, and no occupancy or bandwidth counters were collected. The results do not establish a TP2 serving gain or model-quality equivalence.

The physical component interval was 22.770 seconds, including 4.602 seconds of payload work; the outer interval was 26.941 seconds. No models started, and independent release passed. A runtime `common.py` invalid-escape `SyntaxWarning` was retained; execution and analysis exited zero. The 29 targeted and 49 full CPU checks passed.

The candidate is parked without retry, source selection or stack change. A bounded CPU examination of the retained compiler artifacts and scheduling is the proposed next step; it does not authorize a GPU run. The latest clean serving result remains 121.18–121.24 combined TP2 tokens/s from Sprint 22. The separate Sprint 26 numerical finding remains unresolved, and reliability and endurance testing remain deferred.
