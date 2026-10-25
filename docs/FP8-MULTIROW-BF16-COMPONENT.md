# Two-row BF16 attention component

Decision: **rejected and unselected**. Both GPUs passed the exact correctness checks, but the tested implementation failed its performance gate. Astra independently verified the source, raw results and release.

This component paired two adjacent causal rows in stage one, sharing a KV tile when their split geometry and physical prefix matched. It kept the native reduction and BF16 cache writes. The reference used the current native grouped BF16 decode. The selected serving configuration was not changed.

## Actual scope and results

| Check | Result |
| --- | --- |
| Coverage | Two R9700s; nine prefixes from 0 to 59,984; three operand patterns; 54 cases |
| Correctness | 108 graph captures and 4,536 checked replays passed exact BF16 output and written FP32 partial/LSE comparisons |
| Timing | 3,456 device-event measurements; alternating order within eight blocks |
| Candidate/reference median ratios | 1.009301–1.236214 across 108 case/condition cells; every median was slower |
| Frozen benefit gate | Failed: all 18 required long-prefix steady cells missed the improvement threshold; 106/108 cells exceeded the 2% regression limit |
| Variation | Candidate won 99/1,728 matched samples, with one tie. Sample ranges overlapped in 32/54 steady and 26/54 preceding-write cells |
| Compiler resources, both cards | Native: 245 registers, zero spills. Pair: 256 registers, 21 spills |
| Memory | Peak allocated 204,737,024 B; reserved 205,520,896 B; host peak 2,268,049,408 B; all case-final allocated counts zero |
| Duration | Payload 209.458 seconds; owner 229.238 seconds, within the declared bounds |

Timing includes captured cache writes, stage one and native reduction. Metadata preparation and correctness audits are outside the events. A preceding 64 MiB zero write is a stated condition, not proof of cold-cache or DRAM-traffic behavior.

Prefixes 0, 31 and 127 use the native fallback because their actual per-row split counts differ. Initial prefixes 511 and above satisfy the shared-prefix conditions. Deliberately incompatible prefix phases correctly fall back. The result does not show that every cell executes the shared branch.

The spills may help explain the result, but their causal contribution was not isolated. Static instruction counts include fallback branches and do not count executed traffic. The reduction artifact retains two different resource views: its assembly descriptor says `next_free_vgpr 241`, while HSA metadata says `vgpr_count 13`, matching the live compiled-kernel value. Both observations remain recorded.

## Validation and disposition

Source preparation passed 62 CPU checks. Independent review rechecked the corrected reader, compiler and ownership contracts before execution. Actual graph checks include poisoned output/partial/LSE storage, future tokens, foreign prefixes, slot remapping, cache writes and buffer lifetimes. No error tolerance was substituted for the exact reference.

Independent release authenticated nine process identities and private-namespace absence, four scope-absence observations, empty all-user GPU-holder checks, zero allocations, and journal closure followed by root-verified absence. There were no cleanup errors or unresolved ownership. A nonfatal optional Torch DLPack-extension warning remains in the evidence.

This is **source-validated, component-tested and hardware-validated** within the stated model-free scope. Full-model quality, scheduler cancellation, eight-row integration and serving throughput are **NOT RUN**. This rejection does not establish that every possible eight-row design would fail. The failed benefit gate closes this implementation without a retry or promotion.

The [provenance record](../provenance/FP8-MULTIROW-BF16-COMPONENT-20261024.json) contains all case medians and exact identities. Retained native evidence is `scratchpad/fp8-w4-v2-run.9toi6lsj`, sealed with SHA256 `44716d8d7d0aebd6c7a403c5c47cb7260ee8f36e73337ef7384411909d13292f` (205 members). Its source seal is `27bbbcd5af94a498505f2f5a222c12cf9b4a4d60a2273e37b138c209e0ed7a0d` (78 members).
