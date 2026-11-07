# OPT-03 attention screening and split-cap GPU result

**Operation 213 completed on both cards — 3 November 2026.** The approved cap16/cap32 component passed correctness but is NOT_USEFUL under the frozen timing contract. The invocation is consumed and parked, with independent release verified. No retry, model start, adoption or selection change occurred.

| Lead | Result | Disposition |
| --- | --- | --- |
| C4 int32 indices | Narrower gathers, but spill count 102 → 109 and private storage 380 → 404 bytes. | Reject this variant. |
| C4 next-index prefetch | Spill count 102 → 142 and private storage 380 → 560 bytes; new loop scratch stores. | Reject this variant. |
| C4 address/V/head layout | No proved mapping that removes the costly address reloads. | Deferred, not disproved. |
| Verification split cap 16 → 32 | Both stages pass compiler limits on all four frozen cases. | Operation 213: correct; primary saving below 10% and inside observed timing spread. Park. |

The failed C4 changes were not combined. Earlier two-query, EXP2 and accumulator-fold outcomes remain unchanged.

The split-cap candidate preserves the selected BF16 kernel source. It changes the partial-buffer capacity, stage 1 grid and stage 2 reduction limit. Stage 1 stays byte-identical at 245 VGPRs, 77 SGPRs, zero spills and 16 KiB LDS. Stage 2 remains at 13 VGPRs, 33 SGPRs, zero spills and zero LDS. All five baseline artifacts for both stages reproduced before candidate compilation.

The actual host wrapper produces matching buffer, stride and grid records on Mac and native CPU Python. It refuses mismatched partial capacity before either launch. The selected serving adapter still requires a maximum of 16, so a global flag change alone is not a valid implementation.

The potential benefit is shorter serial work per program. The cost is twice as many programs and twice the partial-buffer capacity, with more partial reductions. A runtime core count of 32 and the cap16 split counts were already observed on both cards in the retained verification-prefetch campaign. Using that count, the model reduces the maximum long-case loop from 235 to 118 iterations. Operation 213 confirmed core count 32 and actual split count 32 on both cards. Long-row spans changed from 7520 to 3776; the sixteen-row case changed from 3776 to 1888. Short-row spans changed from 64 to 32, with four versus nine empty slots.

On the short case, modeled active partial writes plus reduction reads rise from about 20% to 38% of the logical KV bytes. On the long case they remain below 0.5%. These are data-volume estimates, not measured memory traffic or speed. Short requests may regress.

The reviewed component packet implements a configuration contract, actual split-count and both-stage binary witnesses, and positive-length admission. Every layer must pass its own FP32 reference check under the unchanged error limits, including the fixed 1.02 mean-error bound. Each case and arm receives fresh NaN-filled partial buffers, with explicit empty-slot checks. All cap-dependent allocations and launches derive from one arm cap, without changing the backend's ordinary decode setting. Changed partitioning changes FP32 association, so candidate bitwise equality to baseline is not claimed. Timing includes the split helper and both attention stages. The existing timing/no-regression gates and separate serving/lifecycle qualification remain required.

CPU validation before execution: 16 focused CPU tests and the 43-check public preflight passed. Opus independently reran the focused suite and reproduced the readout. Two additional signoff tests cover unused grid slots across all positive supported lengths and the shared source span expression. The newly cited campaign seal also verifies. No GPU work occurred during that CPU screen. The original sealed report is preserved; the core-count correction and stronger component requirements are recorded in a separate signoff root.

Component preparation passed 31 local and native source tests, 10 native owner tests and three caller tests. The actual native loader reproduced all five compiler artifacts for each arm, stage and frozen case, with source-guard restoration on success and exception. A stale donor boot ID refused the first source-check preparation before tests; that refusal is retained, and a fresh preparation passed with the authenticated current boot. Opus independently reran the 31 source, 10 owner and three caller tests and returned AGREE with no blockers. The owner then approved the unchanged packet, and operation 213 ran it once. Measured component timing follows below.

The final decision binds the outer seal, completed review, resolution and registration conditions. A resolved short-case regression is a final NOT_USEFUL result under this packet's rules. No resolved regression occurred. Every case had zero candidate layers bitwise-equal to baseline, so the review's all-equal output-write investigation condition did not trigger. No numerical or timing threshold changed.

## Operation 213

| Card | Case | Baseline cap16 (ms) | Candidate cap32 (ms) | Median saving |
| --- | --- | ---: | ---: | ---: |
| 0 | 119,990 / 8 rows, sequential | 22.6553 | 22.0684 | +2.59% |
| 0 | 119,990 / 8 rows, even–odd | 22.6387 | 22.0896 | +2.43% |
| 0 | 59,993 + 59,980 / 16 rows | 22.0285 | 21.3367 | +3.14% |
| 0 | 724 / 8 rows | 0.7371 | 0.7360 | +0.15% |
| 1 | 119,990 / 8 rows, sequential | 22.5968 | 22.1097 | +2.16% |
| 1 | 119,990 / 8 rows, even–odd | 22.6366 | 22.1049 | +2.35% |
| 1 | 59,993 + 59,980 / 16 rows | 22.0497 | 21.3819 | +3.03% |
| 1 | 724 / 8 rows | 0.7449 | 0.7491 | -0.57% |

Each median uses six unprofiled device-event samples, with alternating AB/BA order. Each interval contains the per-row split helper and sixteen verification-attention layers, including both stages. Inputs are seeded synthetic BF16 tensors for one TP2 rank shard; these values are not serving latency.

The primary sequential case saved 2.59% and 2.16%, below the fixed 10% threshold. Its median gaps were 0.587 and 0.487 ms, smaller than the largest observed ranges of 1.901 and 1.661 ms. The other long cases showed lower medians with gaps above their ranges, but do not replace the primary gate. The short case showed no resolved regression. No serving gain is established.

All 128 per-layer FP32 checks passed, with the mean-error factor fixed at 1.02. Baseline self-repeat, fresh empty-partial checks and both-stage compiler identity gates passed on both cards. Peak allocated/reserved memory was 4.44/4.55 GB per card, below the 6 GiB observed-peak vetoes. Allocated bytes returned to zero; the outer owner released independently at checkpoint 579.

The component took 11.90 s and the caller 44.07 s. The read-only post-component observation recorded 235 W power caps on both cards; the experiment changed no power setting. The owner confirmed restoration to 235 W before the 12:14:52 UTC caller start. These paired baselines are authoritative for this comparison. Earlier 300 W measurements are not substitutes.

The standalone candidate is parked as NOT_USEFUL. Under the owner's later cumulative-gains amendment, retain cap32 for the post-sprint bundle evaluation: its primary timing signal is unresolved, its other long-case median gaps exceed their observed ranges, and correctness/resource checks passed without a resolved regression. This is eligibility for a new combined experiment, not a revised verdict or serving qualification. OPT-04 recurrent-state and small-fusion CPU work is next. See [the sprint plan](MXFP4-OPTIMIZATION-SPRINT-PLAN.md) and [provenance](../provenance/MXFP4-OPT03-20261103.json). No further GPU invocation, adoption or renewed Hyperloom campaign is authorized by this record.
