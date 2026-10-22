# Sprint 17: GDN input projection

Status: the component and both matched serving orders passed. Median serving
latency fell 0.54–0.64%, reaching 115.71–115.73 combined TP2 output tokens/s.
Greedy 8/8, sampled 8/8 and lifecycle 52/52 final answers passed, along with
cancellation, actual C2, slot accounting and release checks. Astra approved
qkvz tile16/stage1 for the next experimental session launch. Retain Sprint 14
as rollback; installed runtime and persistent defaults remain unchanged.

[Sprint 16](SPRINT16-CURRENT-PROFILE.md) measured GDN input qkvz methods at about
4.57 ms per long sampled cycle on each card. This sprint compared common M
tile64/stage3 with native unrolled M tile16/stage1 at `(M,N,K)=(8,8192,5120)` only.
It preserved selected down, gate/up, 3072 and prefill routes, the stack, FP8
weights, BF16 caches, FP32 state/head, flags, DFlash8 and metadata reuse.

Fresh pinned source/configuration inspection confirmed the TP2 geometry and
found all five expected tuned-config paths absent. This is a source/configuration
derivation, not a live model qkvz weight-shape witness. The selected dispatcher
AST with CPU stubs confirms common tile64/stage3 for this untuned shape. The
candidate adds one guarded exact-shape branch; native kernel bodies and all
nine protected owner/transport/resource/identity helpers remain byte-identical.

The author CPU suite passed 43 distinct checks across 46 invocations: three
imported dispatcher tests execute twice during discovery. The fresh execution
clone passed the same checks. Parent review covered the exact shape and stage
substitution in the previously reviewed component machinery; no new independent
Astra review was needed for this bounded parameter change.

The model-free attempt completed four structured/random cases across both cards,
96 finite/exact comparisons including 24 graph checks, and 64 alternating eager
timing launches, using an explicit capture stream for each card. It retained exact
structured analytical oracles and random equality against the selected common
baseline. Tolerances were not relaxed and no nonce was retried.

| Card | Pattern | Common tile64/stage3 | Unrolled tile16/stage1 | Less time |
| --- | --- | ---: | ---: | ---: |
| 0 | Structured | 0.067720 ms | 0.056220 ms | 16.98% |
| 0 | Random | 0.069480 ms | 0.054980 ms | 20.87% |
| 1 | Structured | 0.067600 ms | 0.054080 ms | 20.00% |
| 1 | Random | 0.066220 ms | 0.054260 ms | 18.06% |

The owner window took 22.84 seconds with zero model starts. Peak Torch device
allocation was 42,128,384 bytes. Compiler reports show 256 versus 153 registers,
zero spills in both variants, and 49,664 versus 16,384 shared-memory bytes.
These reports do not measure occupancy or total driver memory. Synthetic
component data and sequential card execution do not establish a TP2 serving gain.

The forward serving packet kept the existing four frozen requests per start:
2,047 input tokens and 1,024 forced output tokens with low thinking. The first
request was excluded from timing; all output and verification-cycle gates were
retained. Qkvz was added to the existing once-per-shape metadata witness equally
in both arms, so the reader required the exact changed route on each rank.
Native kernel bodies, runtime flags and owner controls remained unchanged.

## Matched serving results

Astra approved one forward pilot after checking the exact source substitution,
identical witness addition, unchanged requests and controls, and reader coverage
of qkvz on both ranks in both arms. Preparation passed 22 packet, 35 startup and
14 reader checks. Reverse order passed 23 packet and 35 startup checks; its
byte-identical reader reused the 14 passing forward checks. Parent review
confirmed that only execution order changed before a fresh one-use launch.

| Order | Reference median | Candidate median | Latency change | Candidate combined TP2 rate |
| --- | ---: | ---: | ---: | ---: |
| Reference, candidate | 8.89818 s | 8.84982 s | −0.5435% | 115.7085 tokens/s |
| Candidate, reference | 8.90507 s | 8.84785 s | −0.6425% | 115.7343 tokens/s |

All eight output-ID pairs match, including warmups. Every request retained 164
verification cycles, graph replays and completed commits per rank. Qkvz dispatch
was witnessed on both cards in both arms, with the expected stage and tile;
selected down, gate/up, 3072 and prefill routes remained intact. Source, current
stack, weights, BF16 cache and independent release checks passed. Timing uses
three measured requests per start; retain this bounded sample without a
statistical-significance or normal-stop quality claim.

## Affected quality and cache gates

The screens reused the existing candidate/candidate tasks, input IDs, request
profiles and oracles, with two fresh starts per screen. Parent review verified
the exact candidate source substitution, unchanged control helpers and frozen
requests. Each authenticator changed only its required GEMM source identity.

| Screen | Observed result | CPU packet / startup / reader checks |
| --- | --- | --- |
| Greedy | 8/8 complete final answers; 4/4 full output pairs and final texts match. | 18 / 35 / 15 |
| Publisher sampled | 8/8 complete final answers; 4/4 full output pairs and final texts match; native selector route verified on both ranks. | 18 / 35 / 16 |
| Cache/lifecycle | 52/52 normal final answers; two cancellations/recoveries; two actual C2 groups, slot accounting and release pass. | 65 / included in packet / 12 |

Greedy and sampled screens each used four tasks twice, with thinking enabled at
low effort. Greedy input counts were 226/246/12272/12272 with total output cap
4096, temperature 0 and top-k/top-p 1. Sampled input counts were
226/246/8176/8176 with cap 8192, temperature 1, top-p 0.95 and top-k 20.
Both long tasks met the 512-final-token requirement in every repeat.

Lifecycle full outputs match 24/26 cache pairs; all 26 final-answer pairs match.
The two `concurrent-a` full-output variants first differ at token 149, the same
case/index pattern retained in Sprint 14. Preserve those differences and the
existing numerical contract. These screens do not establish full-state or
sampling-distribution equivalence.

All five model campaigns authenticated the selected source, runtime, weights,
BF16 caches, worker identities, wire data and independent release. Total work
was ten model starts and 86 requests: 16 fixed-work requests, 68 natural final
answers and two cancellations, plus the separate model-free component.

## Retained preparation refusals and limits

A read-only precheck for the queued greedy packet initially refused the active
reverse campaign's lease journal. The original receipts are retained. After
reverse release authenticated, all three unchanged prechecks passed. No quality
GPU launch occurred during the refusal, and no owner gate was relaxed.

The greedy offline grader also refused a missing frozen `GRADER-CHECKS.json`
before scoring any answer. The exact sealed prerequisite was restored after its
source hashes matched. Grading resumed with unchanged code and rules, retaining
the failed receipt; no model rerun or sealed-root edit occurred. The sampled
reader received the same prerequisite before its launch.

The component retained its 1,200-second active owner, 600-second worker,
300-second seal reserve and 256 MiB device-allocation guard. All model campaigns
retained their existing finite owner, worker, client, resource and output limits.
No limit or acceptance gate was expanded.

## Experimental selection

Astra independently verified all 18 sealed source/native/reader roots, five
result hashes and exact patch application before approval. The
[source patch](../patches/sglang-gfx1201-m8-qkvz-unrolled-tile16-serving.patch)
applies to selected Sprint 14 source `f90f4779c13c…` and produces the exact tested
source `3f28660b26d0…`. Its SHA256 is
`26aed66729d59909a4c597c22f56475acf74ffcca479e63f50fa358c3528abc6`.

Select that source for the next experimental launch only. Preserve the old f90
source for rollback, the current stack and all earlier selected routes. No GPU
campaign is queued and no Git commit or push was made in this sprint.

Sprint 11's common-kernel stage1-only result at this shape was mixed; preserve
that result. This candidate also changes the kernel and M tile. Earlier numerical
findings and failed work gates remain unresolved under the owner's session-use
disposition. Reliability remains deferred until tuning is exhausted.

See the [evidence index](../provenance/SPRINT17-GDN-INPUT-20261022.json).
