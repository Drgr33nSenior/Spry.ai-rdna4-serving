# Sprint 20: selected-profile classifier refresh

Status: `SPRINT20_COMPLETE_CURRENT_PROFILE_ANALYZED`. The two-start,
eight-request classifier refresh passed all eight natural final answers, all
four full-output ID pairs, and independent release. All eight scheduled device
windows completed on each rank. The owner duration was 385.30 seconds.
The control and timing arms used the selected Sprint 17 GEMM
`3f28660b26d00938e907441be64e5e9a74153194058ed9e91b52f8808f1a1390`
and the Sprint 19 aligned ring-fold recurrent profile. Timing is enabled only
in the timing arm. No new kernel or source promotion is proposed.

The packet retains the Sprint 16 publisher-sampled four-task requests and
schedule, low thinking, FP8 target weights, BF16 target and draft attention KV,
FP32 recurrent state, DFlash block eight and existing runtime limits. Both arms
use 19 usable recurrent slots and 20 physical slots per card. The original
compact replay flag remains disabled.

Before launch, local CPU preparation passed 51 startup, 69 packet and 22 reader
checks. The source package has 64 archive members and a 16,721,920-byte archive;
combined resource accounting is 33,539,891 bytes. Workload argument vectors have
255 entries per arm, and control and timing bindings are 65,443 and 65,423 bytes.
The existing archive, payload, argument and binding limits remain unchanged.
Preparation refusals and corrections remain in the private receipt history.
A reader-only result metadata correction derives the ring-fold quality flag from
authenticated stack rows; grading rules and the retained status string are unchanged.

## Current long-window medians

| Interval, ms | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Cycle | 53.559 | 53.531 |
| Target graph | 44.445 | 44.446 |
| Projection forwards | 27.550 | 27.516 |
| MLP | 18.587 | 18.557 |
| GDN | 10.536 | 10.510 |
| Attention | 4.372 | 4.348 |
| All-reduce | 3.252 | 3.363 |
| Draft graph | 7.238 | 7.129 |
| Accept | 0.343 | 0.348 |

The gate/up, down and GDN qkvz projection forward medians were 11.939/11.932,
5.668/5.654 and 4.384/4.377 ms on ranks 0/1. These are nested forward
intervals, including quantization, launches, waits and enclosed collectives;
they are not isolated GEMM times or whole-request fractions and are not
additive. Each rank completed all eight scheduled windows, but gaps between
samples and tails of 13, 42, 64 and 150 verifier cycles after the last sampled
window remain unmeasured.

With equal full output IDs and verification cycles, measured client time was
2.83%, 8.27%, 6.52% and 7.02% higher in the timed arm for the four requests.
This is descriptive instrumentation overhead from one control-first order. It
is not a speed benchmark or a causal improvement claim. The historical Sprint 19
rate was about 120.3 combined TP2 output tokens/s.

The follow-on [Sprint 21](SPRINT21-COMMON-M16.md) common M16/stage1 component
passed 96 exact comparisons. Both matched serving orders passed 8/8 output/work
pairs, with 0.38064% and 0.28794% lower median latency. The candidate rates were
120.304 and 120.368 combined TP2 tokens/s; pooled mean latency fell 0.29775%,
descriptively, with one excluded warmup and three measured requests per start.
Greedy and sampled repeated-candidate screens each passed 8/8 final answers and
4/4 full-output pairs, with independent release. The sampled route was verified,
but its distribution was not. Lifecycle passed 52/52 finals, two cancellations,
C2, slots and release; 24/26 full cache outputs and 26/26 final texts matched.
Astra selected the M16 GEMM for the next experimental session only, keeping the
prior GEMM as rollback and the ring-fold profile unchanged. The refresh
does not establish production, numerical, state or sampling-distribution
qualification. Reliability and endurance remain deferred; installed runtime and
persistent defaults are unchanged. See the
[evidence index](../provenance/SPRINT20-CURRENT-PROFILE-20261022.json).
