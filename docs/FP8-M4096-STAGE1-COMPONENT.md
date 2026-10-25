# M4096 stage-one component

**Component-tested and serving-tested in the scopes below; unselected.** Stage one reduced isolated M4096 kernel medians by **24.63–41.34%** and showed a narrower TTFT benefit against stage three at 4K chunks. The improved 4K profile was then **2.7–2.8% slower** than the selected-equivalent 2K profile, with higher allocator peaks. Keep selected b101 and 2,048-token chunks.

The candidate changes stage three to stage one only for the five observed M4096 FP8 projection shapes. It preserves M64/N128/K128, group 32, four warps, scales, ordered dot/scale/add, masks, BF16 output and existing C1 routes. The [experimental patch](../patches/sglang-gfx1201-m4096-stage1-experimental.patch) applies to selected b101 source, not arbitrary upstream source. It is not selected or installed.

## Component measurements

All 720 exact comparisons passed, including 180 graph checks. Independent Astra review reproduced the component result and verified resource release.

Each card ran the five shapes sequentially with structured, seeded random and adversarial operands. Eight warmed eager timings alternated stage order. Graph checks were separate, with poisoned output before each of three correctness replays per variant. Structured cases use an independent oracle; random and adversarial cases require exact equality with selected stage three.

The table shows random-input medians in milliseconds. These are isolated kernel event intervals, not request or prefill times.

| N × K, with M=4096 | Card 0, stage 3 → 1 | Card 1, stage 3 → 1 |
| --- | ---: | ---: |
| 17408 × 5120 | 6.4970 → 4.4276 | 6.8147 → 4.8639 |
| 5120 × 8704 | 2.8077 → 1.9486 | 2.8218 → 1.9686 |
| 8192 × 5120 | 3.4367 → 2.4651 | 3.3942 → 2.4917 |
| 7168 × 5120 | 2.8419 → 1.6670 | 2.8335 → 2.0232 |
| 5120 × 3072 | 1.0400 → 0.7037 | 1.0292 → 0.7123 |

All 240 paired repetitions favoured stage one. Every observed candidate timing range was below its corresponding control range. The largest relative median absolute deviation was 8.16%; this remains one bounded component campaign with uncontrolled cache coldness. The component result alone establishes no cold-prefill, TTFT, decode-throughput or full-model gain.

Compiler reports show 256 → 238 registers, 49,664 → 16,384 shared-memory bytes and zero spills. These are compiler resource reports, not measured occupancy or physical memory traffic.

## Resources and identities

Maximum observed Torch allocated/reserved memory was 254,434,304 / 257,949,696 bytes per card, within the 256 MiB component bound. Host peak was 9,543,520,256 bytes under the unchanged 38 GiB owner cap. Payload and owner durations were 112.98 and 131.81 seconds. Output files, process/namespace/scope observations and independent release passed their existing bounds. The existing invalid-escape `SyntaxWarning` remains in raw stderr.

The active runtime was Torch `2.14.0+git2e66fe81.rocm10.2.0`, ROCm SDK `10.2.0.dev20261020+git535d2990` and HIP `7.17.26416`. The retained lower-base Torch 2.13 metadata does not identify the consumed overlay. Exact source, SDK, toolchain, device, package and evidence identities are in [provenance](../provenance/FP8-M4096-STAGE1-COMPONENT-20261023.json).

Preparation retained two independently found reader gaps and their fixes: exact timing iteration/order validation and consumed-runtime joins. The corrected complete-result fixture passed, including mutations that reproduce the old acceptance. The 56-check suite and a fresh execution-copy suite passed before device execution. The first new fixture's missing output directory and its correction remain recorded.

## Serving comparisons

The exact serving candidate GEMM SHA256 is `3038555586696f65423a7d6c2bc2227add9acbdff3ac05f787be49de9fcd7cec`. Its dispatcher changes only the five M4096 shapes above. Removing that guard reconstructs selected b101's AST; M2048, M1415/M1402 tails and C1/C2 routes remain equivalent. Both serving comparisons used thinking with low reasoning, BF16 target/draft KV, FP32 state/head arithmetic and fixed DFlash8.

At fixed 4,096-token chunks, the stage-only comparison retained this signal:

| Order and evidence class | SRE TTFT, stage 3 → 1 | Retrieval TTFT, stage 3 → 1 |
| --- | ---: | ---: |
| AB, authenticated forensic readout; original operation failed | 10.673569 → 9.509156 s (−10.9093%) | 10.691286 → 9.514317 s (−11.0087%) |
| BA, successful physical operation and strict final readout | 10.747255 → 9.456717 s (−12.0081%) | 10.750768 → 9.473383 s (−11.8818%) |

All sixteen frozen natural-answer checks passed across these orders. BA measured pairs and AB retrieval had identical generated IDs and verifier work; AB SRE differed. Both ranks recorded fourteen excluded warm-up and fourteen measured prefill forwards per arm, with zero measured Triton specialization misses, compilation attempts or failed forwards. Actual source/shape/configuration joins authenticated stage-three/stage-one M4096 routes and unchanged tails. Timed per-GEMM observation was removed.

The original AB controller remained **INCOMPLETE** after its final identity-envelope reader refused `complete wire and exact MTP mapping required`. An isolated integration correction attached the already-authenticated arm identity; a corrected offline reader then validated genuine retained evidence. Neither correction changes the original failed status. The strict formal comparator still requires a successful original AB outer report, so the **formal stage-comparison gate remains false**. Clean BA is a separate successful operation, not a repair of AB.

A separately declared, unselected current-baseline assessment then changed only chunks, using exact `303855…` source in both arms. The 2K control exercises the source-equivalent b101 routes. The useful-effect gate required at least 2% **and** 20 ms lower TTFT on each task:

| Measured task | Input tokens | Selected-equivalent 2K TTFT | Improved 4K TTFT | 4K slower |
| --- | ---: | ---: | ---: | ---: |
| SRE | 25,991 | 9.260415 s | 9.521374 s | 2.8180%, 260.959 ms |
| Retrieval/tool plan | 25,978 | 9.281089 s | 9.532386 s | 2.7076%, 251.297 ms |

All eight natural-answer checks and the complete physical/final-reader/source/route/capacity gates passed. The benefit gate failed, so no reverse-order chunk packet or run followed. This bounded AB screen closes the tested 4K lead; it does not estimate a robust performance distribution. Both ranks recorded 26 warm-up/26 measured forwards at 2K and 14/14 at 4K, with zero measured compilation misses, attempts or failures.

TTFT here is the client interval to the first complete positive generated-ID event. It includes HTTP, prefill, first draft/verification and buffering. The runs used warmed Triton specializations and cold KV under the frozen cache contract; they do not establish all lazy-load or compilation costs. These are TP2 service intervals, not pure GPU-prefill time, per-card rates or decode throughput.

The chunk comparison's generated work differed: SRE emitted 554/511 tokens with 86/80 verifier cycles; retrieval emitted 435/451 with 76/78. Total completion time is therefore not a matched-work kernel comparison. The natural-stop checks cover complete literal JSON answers/tool plans, not external tool execution or broad task accuracy.

| Per-rank allocator peak | Selected-equivalent 2K | Improved 4K |
| --- | ---: | ---: |
| Allocated bytes | 21,513,001,984 | 21,931,351,040 |
| Reserved bytes | 21,913,141,248 | 22,420,652,032 |

The 4K profile added 418,349,056 allocated and 507,510,784 reserved bytes per rank. Both profiles passed the unchanged resource envelope. Allocator peaks are not complete device-memory accounting. Independent GPU/process/scope release and zero OOM events passed.

## Evidence and disposition

The [component provenance](../provenance/FP8-M4096-STAGE1-COMPONENT-20261023.json) remains a historical component record. The [current programme provenance](../provenance/REMAINING-FP8-OPTIMIZATIONS-20261025.json) records the later dispositions. Principal retained identities are:

| Private evidence identifier | SHA256 seal or result |
| --- | --- |
| `fp8-m4096-reopened-ab-native.30c9488c8f13117a`, 181 members; original failed | `d13b23daf0de670133579872d23b343c9f6f458391296f84c600ba8bda35cb83` |
| `fp8-m4096-reopened-ba-native.483f430a1edf56f4`, 181 members; clean BA | `7f71a4fc836cc60bc43dd5fd4bb33f0d1f288f8e5729330dbef3358f8a222a22` |
| `CHUNK-DEPENDENCY-GATE-ANALYSIS-NOT-AUTHORITY.json`; formal gate false | `49b6f2d29c83cc72b48f57463a8acb61468b59a96c50f1f00585779f9570efff` |
| `fp8-m4096-current-baseline-profile-native.db8e1b23cb85aeff`, 181 members | `4ef23b350ff30316d1736a139fb2b37f17d37a11b43255fbe9c00defd28f3587` |
| `ACTUAL-CURRENT-BASELINE-RESULT.json`; useful-effect rejection | `ff90e891b400353144c7a67037184633cb6437bedf919a653e9c425fb3970e00` |

Analysis records reside under `scratchpad/fp8-remaining-optimizations.w3m3xpg7/m4096-reopen-native-preparation-20261025-v1`. These labels identify private evidence without linking public readers to local filesystem paths. The stage scope retains 1,382 charged seconds; the independent chunk comparison charged 407 separately. No retry or campaign-limit enlargement followed either result.

The older 95-token control failure with no closing thinking delimiter remains preserved and is mandatory in final qualification; its cause remains unproven. It was not regraded by these new held-out cases. Broader combined/state/cancellation/long-continuation qualification, that exact historical case, BetterBench and endurance remain **NOT RUN** here. The current [programme record](REMAINING-FP8-OPTIMIZATIONS.md) tracks their separate final-review gates.

Disposition: the component benefit is retained in its narrow scope, the formal stage comparison is incomplete, and the tested 4K serving profile is rejected against current 2K. Rollback is non-adoption. Selected b101, 2,048-token chunks, BF16 target/draft caches, aligned ring-fold, low thinking and fixed DFlash8 remain unchanged. No candidate was selected or deployed; historical numerical findings and production qualification remain unresolved by this work.
