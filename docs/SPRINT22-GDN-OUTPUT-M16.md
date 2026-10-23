# Sprint 22: common M16 GDN output projection

The model-free component passed all 96 exact comparisons and used 7.36–18.33% less kernel time in all four cases. Both matched serving orders passed. The initial quality screening stopped during the sampled attempt. The separately authorized completion passed its bounded quality gates. The b101 source is selected for the next experimental session, with ceec available for rollback and the ring-fold profile unchanged.

The candidate changes only the untuned HIP/BF16 common-kernel GDN output shape `(8,5120,3072)` from M64 to M16. Both variants retain stage one, N128/K128, group 32 and four warps. The selected gate/up M16 route and all other kernel bodies are unchanged.

| GPU | Pattern | M64 median, ms | M16 median, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | structured | 0.045061 | 0.036801 | 18.3309% |
| 0 | random | 0.044260 | 0.036600 | 17.3068% |
| 1 | structured | 0.051425 | 0.047640 | 7.3602% |
| 1 | random | 0.044120 | 0.037060 | 16.0016% |

The 96 comparisons comprise 72 eager and 24 graph comparisons; timing used 64 alternating eager launches. Structured inputs used an analytical oracle; random outputs were compared exactly with the previously selected M64 baseline. Both cards were tested sequentially with synthetic inputs. These component timings are not TP2 serving measurements. The wider GPU1 structured timing is retained; its cause is unknown.

Compiler records report 238 versus 114 registers, zero spills and 16 KiB shared memory. Device tensors peaked at 16,890,368 bytes within the unchanged 256 MiB cap. The physical component completed in 23.04 seconds with independent release and no model starts. All 45 composed CPU checks passed. Fresh same-boot compiler and tuned-config absence checks were joined to four authenticated Sprint 21 selected-source dispatch records; group size is source-derived metadata.

The baseline GEMM is `ceec5726fa20815395d2b7f6b44b22fe379a60ef1fb64f3dde89954ae04ee892`; the candidate is `b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069`. The [provenance record](../provenance/SPRINT22-GDN-OUTPUT-M16-20261022.json) binds the source, native and host-observation seals.

Both serving orders used 2,047 input tokens, 1,024 forced output tokens, concurrency one, greedy low thinking, and one excluded warmup plus three measured requests per start. Both arms retained selected gate/up M16, ring-fold, 19 usable/20 physical recurrent slots and BF16 caches. Each order used two starts and eight requests. Combined TP2 output rates are shown below.

| Order | Candidate rate, tokens/s | Median latency change | Full output and work pairs |
| --- | ---: | ---: | ---: |
| Control first | 121.1752 | −0.8918% | 4/4 |
| Candidate first | 121.2428 | −0.8140% | 4/4 |

All eight pairs matched full output IDs and verification work, with 164 verification cycles per request per rank. The pooled mean latency change was −0.9677%, a descriptive result from these starts. Both orders passed independent release. These measurements do not qualify the source for production.

The repeated-candidate greedy screen passed all eight normal answers and task checks, with 4/4 full output pairs matching and independent release verified. A premature grading command failed because the decoded finals were not yet present; the unchanged CPU decoder and grader then passed. No model retry occurred.

The initial sampled attempt stopped during second-arm startup when the supervisor reported that a process changed during namespace observation. The owner terminated the arm; the missing bounded artifact bundle was a secondary consequence. The first arm established one model start and completed four HTTP responses, which were not graded. A second model start was not established. Independent release passed with no surviving processes. This was a harness failure; the underlying process-observation cause remains undetermined. Lifecycle was CPU prepared but did not launch in the initial attempt. No automatic retry was made.

In the initial attempt, five GPU attempts included one model-free component; seven model starts were established. There were 28 completed HTTP responses: 16 fixed-work, eight greedy and four ungraded sampled responses. There were no cancellations. Astra approved parking the candidate at that closeout pending complete quality screening. The previously selected ceec source remains the rollback; its Sprint 21 benchmark was 120.304–120.368 combined TP2 tokens/s under the same fixed-work protocol.

The three repeated-candidate quality packets passed CPU preparation. Greedy and sampled passed 18 packet, 35 startup and 22/23 reader checks respectively; lifecycle passed 65 source and 19 reader checks. Both arms use the candidate and ring-fold, so these screens are not a baseline comparison. A later user-authorized completion prepared fresh sampled and lifecycle packets.

The completion added diagnostics to the two existing process-identity mismatch refusals. It preserved the identity decisions and read order, made no change to the resource handler, and passed 10 focused procfs checks per packet. The fresh sampled screen passed all eight normal answers and task checks, with 4/4 full output pairs matching and independent release verified. The native sampling route and recipe were verified; sampling-distribution equivalence was not established. The fresh lifecycle screen passed 52/52 normal finals, two cancellations, actual concurrency-two group accounting, slot checks and independent release. Of 26 cache comparisons, 24 matched full output IDs and all 26 matched final text. In each arm, concurrent-a first differed from victim-cold-a at index 149, the same retained pattern as Sprint 21. The later sampled pass does not establish the cause of the initial observation failure. Across the completed bounded evidence set, 10 starts served 86 requests: 16 fixed, 68 normal and two cancellations. Including the initial incomplete attempt, there were 11 established model starts, 90 requests and seven GPU attempts, one model-free. The initial second sampled start was not established; its absence is not proven.

The lifecycle model run completed, but its first local collection timed out after 60 seconds. The 42,663,936-byte partial archive and failed receipt were retained. One independently reviewed compressed read-only collection recovered the existing evidence, with 50,667,520 raw and 8,531,734 compressed workload bytes. Source, archives and both collection receipts were bound to the recovery; no GPU rerun or limit increase occurred. The recovery reader passed 27 distinct CPU checks, with eight affected checks rerun.

One retained CPU preparation refusal occurred before component packaging: the fresh observation copy encountered an existing compiler-source directory. The existing files were checked against the sealed observations and retained unchanged. A separate preflight comparison initially traversed a generated donor package; the corrected comparison passed before the serving build. Neither involved a native retry. The selected next-session [GDN output patch](../patches/sglang-gfx1201-m8-gdnout-common-tile16-serving.patch) has an exact-apply proof. The selected source is the lifecycle author’s `review/GEMM-tail.py` with the b101 hash above.

The earlier Sprint 11 3072 work-parity failure remains recorded. No hardware cause or numerical resolution is established. Reliability and endurance remain deferred. Astra verified 18 top-level seals and two remote seals for the initial closeout. Astra’s sealed final review approved selection for the next experimental session. The deeper observation-failure cause remains undetermined. No installed runtime or persistent default changed, and no production, numerical, state or distribution qualification is claimed.
