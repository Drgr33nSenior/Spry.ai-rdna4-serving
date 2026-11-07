# Current serving verification

These results describe bounded private runs on two Radeon AI PRO R9700 GPUs.
They do not turn this source prototype into an installable or qualified serving
distribution. The runs used a complete separately pinned SGLang runtime.
Machine configuration, model weights, native binaries and raw logs are excluded.

## Split-K private selection, 2026-11-07

The owner selected operation234's logical mtp arm (physical kv4, start 2): the
three-member cap16 profile plus S4 down/out at M8/M16. Operation232's selected
profile is the immediate rollback. The programme remains parked. This changes
the private candidate pointer, not the installed runtime or launch authority.

The frozen verdict remains **INCONCLUSIVE_PARK**. C1 verify-round rate rose
3.91%, below the unchanged 5% bar. Observed C1 throughput was 96.29 to 104.34
tokens/s (+8.37%), and C2 was 172.81 to 194.52 (+12.56%). One start per arm,
baseline-first order and different sampled work prevent a matched-token or
causal effect-size claim. No task-quality oracle ran.

Startup checks found no value differences across 31,457,280 elements (signed
zeros treated as equal), and all 1,152 exact-oracle cells passed. S4 adds
1.25 MiB FP32 scratch per card and a reducer launch per eligible GEMM. Runtime
compiler hashes differ from the native proof; debug paths are an unproven
explanation. Both pre/post configured caps were 235 W. Release is authenticated
at checkpoint 625, with zero allowance and no retry. The parent independently
recomputed the headline rates and checked release and ledger accounting.
See [the selection record](../provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json).

## BetterBench repeat accepted, 2026-11-07

Operation233 completed two starts of the selected cap16 candidate. Independent
review accepts the retained measurement and release at checkpoint 623. Mean
combined TP2 rates were 99.01 tokens/s at C1 and 178.95 tokens/s at C2, with
configured cap readbacks of 235 W per card before and after. BetterBench has no
answer-quality oracle. The older W2 comparison is descriptive: runtime,
instrumentation, context and configured caps differ. See the
[measurement record](MXFP4-SELECTED-CAP16-BETTERBENCH.md) for client overhead,
variation and prefill findings. The selected candidate is unchanged.

The later split-K comparison and owner selection are recorded above. The
operation233 measurement and its historical selection state remain unchanged.

## Final three-member sign-off result, 2026-11-06

Final selection: the owner adopted operation232's logical mtp candidate
(physical bf16, start 1), retained control as rollback and parked the
optimisation programme for Qwen4. This is private experimental selection of
the tested profile, with its cap3/LLVM_PATH deviations and startup numerical
probes retained. No deployment or additional GPU work occurred. The private
pointer is `4e19fc56…`; the old pointer `fe92916c…` is preserved. Approval,
promotion and parking pins are in the finalization record. The public
historical FP8 selection remains unchanged.

Operation232 completed and independently released at checkpoint 621
(`d22b84a7…`). The frozen readout is **CONFIRMED_KEEP_REVERSED_ORDER / KEEP**:
3.810953% case-balanced request-latency score gain, all 60 natural-stop quality
grades passing, all 30 paired token streams and speculative counters matching,
and control reproduction against op224. There are no comparator vetoes. Both
pre/post configured cap readbacks were 235 W. Independent result review
accepts the result with no blockers (`dc14aa7b…`). The owner subsequently
approved the final experimental selection and parking described above; the
GPU invocation itself made no selection change and no retry is authorized.
The execution status
`FOUR_MEMBER_SERVING_MEASURED_PARKED` and `four_member_contract` are legacy
schema labels: this run has three active members at cap16, with no active cap32.

The score is distinct from aggregate throughput. Measured group-wall time
falls 2.1236%; combined TP2 output throughput is 57.1629 tokens/s for control
and 58.4032 tokens/s for the candidate. Group wall supplies comparator vetoes
only. This run passes independently; no pooling with op226/op227. Readout
`d8802e8e…`, execution `10b53b84…` and post-run evidence `4784fd97…` are sealed.
The earlier running and preparation accounts below are historical.

Operation232 is registered at checkpoint 620 (`970e1e6c…`) and its caller is
running. Exact owner approval of PLAN `7b661096…` is preserved in authorization
root `73205f0f…`; fresh remote admission passed and both pre-run configured cap
readbacks were 235 W. No completed quality, performance or release result exists
yet. The invocation is consumed; no retry, adoption or selection change is
authorized. The preparation account below is historical.

No new serving result exists yet. A fresh unprofiled quality and benchmark A/B
is prepared on CPU only. It compares the selected baseline with the confirmed
three-member cap16 bundle, candidate first. The rule is op227's, with unchanged
gates and no pooling. The packet changes only nonce, path and digest bytes. The
reader has one repaired literal for the current controller closure. Independent
CPU review `3fd67474…` accepts the packet with no blockers. Final PLAN
`7b661096…` is sealed in registration root `be6c2864…`; readiness `f0f857fa…`
binds the exact review. Exact owner binding to that PLAN and helper clause,
then fresh admission, remain before GPU execution. Nothing is registered or
invoked, and the ledger remains at checkpoint 619 / next index 232. The
op226 and op227 results stand unchanged.

## v6-QB follow-up, 2026-11-06

Operation231 is closed as **FAILED_WORKLOAD_RELEASE_RECONCILED_PARKED**.
Independent review accepts the recovered failure and release; the exact
observation fix and mechanical conditions pass. Checkpoint 619 (`26a83efa…`)
authenticates release, leaves 618 unchanged and keeps GPU allowance 0. The owner
is idle. The v6-QB trial is parked with no correctness, speed or serving verdict.
The allocator-counter mismatch and SSH stall cause remain unproven. No retry,
adoption or selection change occurred; the confirmed cap16 bundle is unchanged.
Final selection and programme parking remain pending.

Reviewed post-run recovery now authenticates owner evidence and independent
release in root `222aa738…`, with no repeated workload. The original failed
caller and checkpoint618 stay unchanged. TP1's complete retained witness shows
the exact memory-accounting check refused; TP0 left no witness. No correctness
or speed verdict exists. The separate CPU reconciliation is prepared and awaits
independent review before updating the unresolved ledger. No GPU retry is
authorized. The transport account below is historical.

Operation231's caller failed after the physical-owner SSH stopped responding
(exit255). No local correctness or timing result exists. The pinned close
recorded release UNVERIFIED at checkpoint618. A later unprivileged status read
found the same boot, no owner journal and an unsealed record claiming stage-A
failure and cleanup, with stageB skipped. Those observations require retention
authentication before release can be accepted. The attempt is consumed; no
caller, owner or GPU retry is permitted. CPU recovery assessment is next.

Independent final delta review `601addd9…` accepts witness packet
`93dc847c81eb3802` with no blockers. Readiness `473c0d98…` binds that exact
review. Final PLAN `fccaf0c4…` and the three campaign scripts pass the reviewed
mechanical closure checks; the final registration root is sealed `dae9e087…`.
The new exact owner decision and fresh remote admission remain open. If stage A
refuses for any reason, the single invocation is consumed, stage B does not run
and there is no retry. No new GPU work, model start or registration occurred.

Latest preparation, 6 November: a new witness packet, `93dc847c81eb3802`, is
prepared and inert. Nothing ran on the GPU, no model started and nothing was
registered.

- **Witness review.** A focused review of the CPU witness repair added a
  leading newline to every frame, so another writer's unterminated output cannot
  hide a frame. It also added `PYTORCH_HIP_ALLOC_CONF` to the recorded
  variables. The revised serving source is `8ebb6cf4…`, in root `dc9b3411…`.
- **Packet.** The re-composed packet `47503930…` (package `539388fd…`) changes
  only that source. Every other difference is a regenerated build token.
- **Tests.** 69 local tests pass, and so do public43 and an exact reproduction
  of op230's stage-A package.
- **Native proof.** One device-free run on Python 3.14.4 passed 28 tests
  (root `a40d3faf…`).
- **Registration.** The CPU registrar proof (`15f8f4d7…`) refuses 30 mutations,
  including every consumed op230 pin.

The final review is accepted; a new owner GPU decision and fresh admission are
still required. The allocated-memory guard is unchanged. **Stage A may refuse
again;** the witness then records the observed deltas and the failing check,
but does not fix the cause.

Operation230 finished **RELEASED_WORKLOAD_FAILED / PARK_STAGE_A_REFUSED**.
TP1 refused stage A's exact steady-padding allocation check during repacking,
before real-input probes or component timing. Stage B never started. The
failed raw allocation deltas were not published, so the cause remains open;
this establishes no capacity failure, numerical mismatch or speed result.
Retained output and independent release are authenticated at checkpoint616.
The single invocation is consumed; no retry or selection change exists.
Independent review accepts the failed result and release. TP0's check outcome
is unknown. The +1,966,080 B message gives the expected delta, not a measured
one. The resource monitor reported complete; its validator refused the nonzero
child exit. The approved CPU-only follow-up adds a bounded stderr witness
before the unchanged memory guard; 18 tests pass through the executed capture
code. The candidate is unintegrated and has no independent repair review or
native proof. It cannot recover the old measurements. Earlier preparation
states below remain historical, and the op226/op227 results are unchanged.

The single approved five-compile CPU invocation and frozen readout passed.
Independent result review accepts the facts with no blockers. M8/M16 are two
specializations with one binary per arm; device correctness and performance
remain untested. The CPU allowance is consumed and the selected profile is
unchanged. The owner subsequently authorized one finite GPU correctness and
speed test; this accepts the narrow A6 recommendation for this test only.
The composed packet and native environment-projection proof now pass independent
review with findings. The parent filled the final plan and readiness and passed
the CPU registrar proof: one positive case and 13 refused mutations. The exact
protected-helper clause answer and fresh admission remain required. No GPU or
model start has run under this authorization. Stage A is untimed, lacks a power-cap
readback and captures without replaying the v6 serving graph. Stage B brackets
its component timing with cap reads; its weights may be cache-resident. Neither
phase establishes a serving-speed or selection result.
The [reassessment](MXFP4-REJECTED-TRIAL-REASSESSMENT-20261106.md)
records the proposed narrow register rule and remaining packer, launcher,
memory and GPU gates. This optional follow-up does not block final selection
and parking of the confirmed three-member cap16 bundle.

## MXFP4 matched profiling, 2026-11-06

Operation229 completed both matched profiling starts at the configured 235 W
cap. All 60 natural-stop grades pass, and the readout returns SAME_REQUEST_WORK
and MATCHED_PROFILE_CAPTURES_COMPLETE. Independent release passes at
checkpoint614. The retained captures cover all seven windows on both ranks of
both starts. This is diagnostic evidence, not a new serving-speed verdict.

The original offline analysis remains STAGE2_INCOMPLETE. Independent review
confirmed a reader-symbol mismatch: DRAFT's memset was dispatched as
fillBufferUnAligned. A separately labelled CPU successor applies the reviewed
one-line mapping and two-line retained start-clock repair. Its output matches
the required `88b6e713…` hash exactly; all nine source seals still verify. All 96
replays match their inventories, and all 24 decode windows have bounded clock
chains. Four prefill chains and 22/40 resource windows remain unresolved.
Inventory matching does not prove broader physical completeness.

The diagnostic is sufficient for finalization under the review's stated limits.
GEMM's 66–70% short / about 45% long shares refer to VERIFY; whole-GPU-round
shares are 52–57% / 38–39%. Profiling inflates host rounds by 2.9–5.9% long and
4.9–9.9% short, so unprofiled dispatch-gap cost is unknown. Clock/GEMM causation,
physical GB/s and hardware stalls remain unproved. The owner requested
[a historical reopening shortlist](MXFP4-REJECTED-TRIAL-REASSESSMENT-20261106.md)
for review before further optimisation. No trial, selection or retry changed.
See the
[finalization record](../provenance/MXFP4-FINALIZATION-20261105.json).

## MXFP4 finalization attempt, 2026-11-05

Operation227 passed the reversed-order three-member cap16 confirmation at the
configured 235 W cap: the
balanced score improved by **3.7394%**, all 30 same-job token streams and
speculative counters matched, and all 60 natural-stop quality grades passed.
The control reproduced the retained reference. Independent release passed at
checkpoint610. Independent result review accepts CONFIRMED_KEEP_REVERSED_ORDER
and reproduces the score from raw native receipts. Operation226's control-first A/B separately
passed at 3.2652%, accepted by independent review, with release at checkpoint608.
Neither order relies on a pooled result. Selection is unchanged. The separate
matched diagnostic is now complete within its reviewed limits above. See the
[baseline/candidate results](MXFP4-THREE-MEMBER-SERVING-VALIDATION.md).

The balanced objective scores request latency. Group-wall time supplies noise
and regression vetoes, not score weight. Aggregate measured group-wall time fell
about 2.26%; that is a different metric from the 3.7394% balanced score gain.

The bounded passive sampler retained 149 rows. Both cards' whole-run GPU-busy
medians are 100%, while sysfs memory-busy medians are 16% and 15%. These
activity observations are not physical bandwidth or attribution to instruction,
memory or power stalls. The full profiling step remains open.

Operation225 registered the approved three-member cap16 A/B but stopped at its
first pre-launch hostname identity check. No upload, model start, GPU kernel,
quality grade or timing sample ran. Its frozen result class is INCOMPLETE.
Separate source and remote-absence evidence reconciles the unstarted native
owner at checkpoint606; the original failure and checkpoint605 remain retained.
The allowance is consumed, no retry ran, and the selected profile is unchanged.
Final qualification and the resource diagnostic remain open. See the
[finalization record](../provenance/MXFP4-FINALIZATION-20261105.json).

## FP8 thinking checks, 2026-10-21

The published Qwen3.8-27B-FP8 checkpoint passed the fixed J01 ledger task in two
fresh starts at low reasoning and two at the highest published effort, `xhigh`.
Every answer rejected a debit of 31 plus a fee of 4 against a balance of 34.
Each pair returned identical output token IDs. Low effort produced 425 output
tokens per start; xhigh produced 334. Exact prompt IDs and final-answer decoding
were checked against the checkpoint's published tokenizer.

The xhigh worker timeout was 1800 seconds. Requests completed in about 14 seconds;
the timeout includes separate startup work and was not a measured request time.
The earlier wrong answer was not reproduced on this short prompt. Long-context
quality and the combined speculative path were outside these checks.

The stack was PyTorch `2.14.0+git2e66fe81.rocm10.2.0`, ROCm SDK
`10.2.0.dev20261020+git535d2990`, HIP `7.17.26416` and RCCL `2.32.3`.
Private startup compatibility work rebuilt the retained SGLang kernel and pinned
TorchVision source against that Torch build, restored SDK tool executable modes
in a byte-identical runtime copy, exposed the matching HIP development SDK and
redistributed scratch space to provide 512 MiB of shared memory. The original
sealed inputs and installed defaults were preserved.

## Matched previous/current stack benchmark

One ordered previous/current pair and one current/previous pair used identical
requests: 451 input tokens, exactly 128 output tokens, low thinking and concurrency
one. Each stack had two fresh starts, each with one excluded warm-up and three
measured requests. All 16 requests completed, with zero cached tokens and zero
retractions. All output token vectors matched.

| Stack | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| PyTorch 2.13 / ROCm SDK 10.1 | 24.59 | 5.2062 s | 5.1949–5.2224 s |
| PyTorch 2.14 / ROCm SDK 10.2 | 24.57 | 5.2096 s | 5.1922–5.2158 s |

The combined median rate changed by -0.07%. The two orders showed opposite
changes of about 0.3%. No meaningful performance regression was detected in this
screen. This small sample does not establish equivalence for other workloads.

Both arms used the same FP8 target, BF16 attention cache, FP32 recurrent state
and head, greedy sampling, seed 12345, Triton attention, decode graphs and
hipBLASLt preference. **DFlash2 and the split K/V decode adapter were disabled.**
This is a whole-stack comparison, not an attribution to Torch, HIP or RCCL alone.

The rate is output tokens divided by local non-stream HTTP request-to-terminal
time. It includes prefill and protocol completion. It is not isolated decode
throughput, TTFT or ITL. All throughput figures describe one request served by
both GPUs together in TP2; they are not per-card rates. `ignore_eos=true` forces
the fixed output count; these
truncated responses were not graded as completed task answers. Long contexts,
concurrent traffic, sustained load, power and thermal behavior remain untested.

## Combined DFlash2 and split K/V benchmark

The combined path completed on the current stack with the pinned BF16 DFlash2
drafter at block size 8, the retained per-row native split-decode adapter and
aligned GDN, together with the FP8 target, FP32 head, GEMM optimization and
decode graphs.

| Current-stack configuration | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| Ordinary FP8 serving | 24.50 | 5.2246 s | 5.2237–5.2273 s |
| DFlash2 + split K/V + aligned GDN | 86.68 | 1.4766 s | 1.4753–1.4808 s |

The median rate increased **3.54×** on this workload. Each arm had one excluded
warm-up and three measured requests, using the same 451-input/128-output-token
workload described above. All eight requests completed with zero cache hits and
zero retractions. All four output-token pairs matched exactly.

Both target ranks recorded 16 adapter capture calls across all 16 attention
layers, with an allocated adapter buffer set. Native validators authenticated
graph replay and speculative verification. The four combined-arm responses
reported 100 verification passes, 700 proposed drafts and 412 correct drafts;
these counters are not completion-token totals.

This is one ordinary/full arm order. It measures the whole combined path and
does not isolate each optimization. The layer-normalization candidate and
FP4/KV4 cache were excluded. The truncated outputs were not graded as completed
answers. The HTTP timing and workload limits described above apply.

The stock verification split-KV kernel is gated to another GPU family in the
retained source. On R9700, the experimental adapter invokes native decode
attention with the split selector applied separately to each verification row.
This is separate from the four-bit KV cache prototype.

### Memory observations from the combined-path benchmark

The retained telemetry includes a host-memory peak for each complete arm and
GPU startup figures for each rank. The figures below use GiB (2^30 bytes).
SGLang labels its binary-unit GPU figures as `GB` in the logs. Both ranks
reported the same rounded GPU values.

| Host-memory metric, whole arm | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Cgroup memory peak, including startup and warm-up | 18.32 GiB | 23.69 GiB |
| Configured cgroup memory limit | 38 GiB | 38 GiB |

These peaks are Linux cgroup v2 `memory.peak`, including charged page cache,
tmpfs and child processes. They are not process RSS or GPU VRAM. Swap was
disabled. Neither arm recorded a memory-limit event, OOM or OOM kill.

| GPU startup metric, per GPU | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Target weight-load free-memory delta | 14.51 GiB | 14.51 GiB |
| Drafter weight-load free-memory delta | — | 2.16 GiB |
| Target BF16 KV pool, K + V | 0.25 + 0.25 GiB | 0.25 + 0.25 GiB |
| Drafter BF16 KV pool, K + V | — | 0.08 + 0.08 GiB |
| Recurrent-state pool slots | 93 | 19 |
| Recurrent convolution / SSM state | 0.13 / 6.61 GiB | 0.03 / 1.41 GiB |
| Speculative intermediate SSM / convolution state | — | 1.69 / 0.01 GiB |
| Target graph-capture memory delta | 0.29 GiB | 0.38 GiB |
| Drafter graph-capture memory delta | — | 0.10 GiB |
| Available GPU memory after graph capture | 9.11 GiB | 10.13 GiB |

Each attention KV pool had 16,416 token slots. The runtime sized the recurrent
pool differently for speculative execution, which also requires intermediate
state. The greater free GPU memory in the combined arm therefore does not
establish better memory efficiency at equal capacity. Both configurations
completed the measured concurrency-one workload.

GPU figures are rounded startup observations, taken before external requests.
They are not a complete additive memory accounting or a measured GPU peak.
No continuous VRAM samples, Torch allocator peaks, per-request host peaks or
per-process RSS were retained. Exact host peak bytes, per-rank GPU figures and
capture hashes are included in the sanitised JSON linked below.

### Metadata reuse validation, 2026-10-21

A subsequent candidate reuses causal metadata across the 16 attention layers
while preserving all eight singleton split selections. Both GPU ranks recorded
one metadata build during capture across 16 eligible attention calls. All eight
requests completed, with zero cached tokens and zero retractions. All four
ordinary/candidate output pairs matched, as did all four comparisons against
the earlier combined-path outputs.

| Current validation configuration | Median output tokens/s | Median request time |
| --- | ---: | ---: |
| Ordinary FP8 reference | 24.47 | 5.2302 s |
| Combined path with metadata reuse | 87.29 | 1.4663 s |

The workload and timing metric match the short C1 screen above. These are
combined TP2 rates, not per-card rates. The approximately 0.70 percent increase
over the earlier 86.68 tokens/s is descriptive: there was no matched
old-adapter/new-adapter comparison in reversed order. No additional throughput
gain is established. Source, native/wire, raw timing and independent cleanup
checks passed. See [metadata reuse](METADATA-REUSE.md) and its
[sanitised validation record](../provenance/VERIFY-METADATA-REUSE.json) for
capture counters, evidence identities and scope limits.

### M2047 prefill-tail component validation, 2026-10-21

The one-stage M2047 candidate passed all 360 exact comparisons across 20
synthetic cases on the current Torch 2.14 / SDK 10.2 stack. Both GPUs were tested
sequentially. Stage 1 was 1.311–1.575 times faster than stage 3, with a geometric
mean of 1.429 across the cases. There were 320 timed launches and 40 warm-ups.
Structured outputs matched an analytical oracle; random outputs matched the
stage-3 baseline. Input immutability and finite-output checks passed.

The owned run took 47.81 seconds and its native payload took 29.70 seconds.
Independent release and both evidence seals verified. These are component
kernel measurements, not TP2 serving rates or a request-level speedup. Matched
serving A/B is recorded below. See the [component record](../provenance/M2047-COMPONENT-20261021.json)
and [current sprint list](SERVING-SPRINTS.md).

### M2047 matched serving A/B, 2026-10-21

Both orders showed about 4.9 percent lower median request latency for stage 1.
Forward medians were 1.9487 s versus 1.8536 s; reverse medians were 1.9454 s
versus 1.8501 s. These are combined TP2 measurements at C1 with 2,047 input
tokens and 128 forced output tokens. Each model start had one excluded warm-up
and three measured requests. The two orders used four starts and 16 requests.
All five changed M2047 shapes were observed on both ranks.

The exact output gate is incomplete. Seven of eight paired prefixes matched;
the candidate was repeatable in eight of eight requests, while one baseline
response first differed at output index 102. Its 23 verification cycles differ
from the common 25-cycle path. The record preserves that response and timing.
These observations do not establish clean performance attribution or task
quality. The owner subsequently selected M2047 for current-session use, with
the unresolved variation retained for review if concerning results recur.
The current stack and installed runtime remain unchanged. See the
[sanitised serving record](../provenance/M2047-SERVING-20261021.json) and
[current sprint list](SERVING-SPRINTS.md).

### Baseline repeatability follow-up, 2026-10-21

Two fresh stage-3 baseline starts replayed the same four requests each. All
eight outputs matched the historical common prefix, with 25 verification
replays per rank and no graph recapture. Measured request medians were 1.9490 s
and 1.9445 s. Warm-up outputs were compared but excluded from timing.
Source, runtime, weight, HTTP and independent release checks passed.

The earlier variant was not reproduced. This finite screen does not identify
its cause or complete the M2047 exact-output gate. The owner subsequently
selected current-session use; see the
[FP8 session selection](../provenance/CURRENT-SESSION-SELECTION.json), now historical.
The stack and installed runtime remain unchanged. See the
[repeatability record](../provenance/BASELINE-REPEATABILITY-20261021.json) for
all samples, speculative histograms and per-rank graph deltas.

### Current-session M2047 activation and Sprint 6, 2026-10-21

M2047 stage 1 is selected for this session. Two fresh starts with both arms
enabled completed eight requests. Measured request medians were 1.8488 s and
1.8525 s, approximately 69.23 and 69.10 output tokens/s combined TP2. Each
start excluded one warm-up from timing. All five changed M2047 shapes reached
both ranks. All requests used 25 verification replays per rank and no
recaptures. The 28 CPU checks and source/runtime/weight/wire/release checks
passed. The current stack and installed defaults remain unchanged.

Seven outputs matched the common prefix exactly. One differed only in the
capitalization of `Required` at output index 124. The exact mismatch remains
in the record, and the owner selected continued session use. These forced
128-token outputs are not a task-quality qualification.

The startup census observed a 3.79 MiB FP32 local shard and 7.58 MiB gathered
result per rank for a C1 verification block. Exposed latency and link-level
traffic were not measured. Retain current communication settings and continue
Sprint 7 combined-profile qualification. See the
[session and census record](../provenance/SESSION-M2047-SPRINT6-20261021.json).

### Remaining optimization screens, 2026-10-21

The fresh width comparison retained DFlash8: median request latency was
1.8474 s (69.29 tokens/s combined TP2) versus 2.5137 s (50.92 tokens/s) at
width four. Both used equivalent singleton split attention and metadata reuse.
The output prefixes differed; this was a forced-output C1 screen with no task
grading. All eight requests and independent release checks completed.

The aligned ReplaySSM component passed all 1,032 exact comparisons in sixteen
cases on both GPUs. It is ready for serving integration, not full-model
qualification. Component storage formulas and timings do not establish
whole-model VRAM savings or a serving speedup. Adaptive-policy prerequisites
were audited; no adaptive policy was implemented or GPU-tested.

Keep M2047 stage 1 + metadata reuse + DFlash8 as the Sprint 7 reference.
Aligned ReplaySSM remains a separate integration candidate. See the
[screening results and limits](OPTIMIZATION-CANDIDATES.md).

## Evidence identities and limits

| Completed observation | Private local evidence manifest SHA256 |
| --- | --- |
| FP8 low/xhigh verification | `780765826ccbbdc5bce91064dcf74eb17b3adee78c785100ce705802235e685e` |
| Previous/current HTTP benchmark | `47f91427c597fdec16f374686bcd4e17ed338a3c383a4a5b0c6cd8e4ef433fd0` |
| Combined-path native benchmark | `ff2061124636c6894ea1e037173dd272783d47869ce89b78448f8216e116f3af` |
| Metadata-reuse native validation | `2dc999e0d649c59bb43bf29ed30d1ef05d90fe4a688146aa054965ce3fb05fe5` |

Sanitised measurements and source/native evidence identities are retained in
[SERVING-RESULTS-20261021.json](../provenance/SERVING-RESULTS-20261021.json).

Source, native worker, HTTP, loaded-weight, graph, resource and independent
release checks passed for those completed observations. The benchmark timings
were independently recomputed from retained monotonic timestamps. These hashes
identify private evidence; they are not portable runtime/build proofs or launch
authority. Optional TVM Torch C DLPack and Python 3.14 TorchScript warnings remain.
No production deployment or broad quality qualification was performed.

## Aligned ReplaySSM serving screen, 2026-10-21

The aligned ring-fold candidate passed 43 CPU checks and eight native C1
requests after independent Astra implementation review. All four paired
output-ID vectors matched exactly. Both arms used DFlash8, singleton split
attention with metadata reuse, M2047 stage one and the current stack and
precision. Each request used 2,047 input tokens and 128 forced output tokens.

Median request latency was 1.8490 s for the snapshot reference and 1.8042 s
for ring-fold, a 2.42% reduction. Combined TP2 output rate was 69.23 versus
70.95 tokens/s. This is one order, with one excluded warmup and three measured
requests per start. Per-request native commits, graph replays and verifier
cycles all matched at 25 on each rank. Independent release passed.

Automatic recurrent capacity increased from 19 to 74 usable slots per card.
Startup Torch allocation rose from 20.53 to 23.01 GiB per card. That screen did
not measure matched-capacity savings or reverse order; the follow-up below
covers both. Carry this candidate into broader Sprint 7 qualification
alongside the current reference. Normal-stop quality, longer contexts, C1/C2,
sampling, reuse and repeated lifecycles remain incomplete.
See the [serving record](../provenance/REPLAYSSM-SERVING-20261021.json)
and [screening list](OPTIMIZATION-CANDIDATES.md).

## Sprint 7 matched-capacity comparison, 2026-10-21

Both execution orders completed at 19 usable recurrent slots per card, with
45 CPU checks per packet and sixteen native requests across four starts.
Ring-fold reduced median request latency by 2.32% and 2.47% in the two orders.
Startup Torch allocation fell from 20.53 to 18.90 GiB per card in both orders.
The 2 MiB residual between the measured allocation change and the main tensor
calculation remains unassigned. These counters describe startup after graph
setup, not whole-device peak VRAM.

Seven of eight output pairs matched exactly. One reference response repeated
the earlier capitalization variant at token 124; all eight candidate outputs
matched the historical common output. All requests used 25 verification
replays and completed commits per rank, with no recaptures. Independent
release passed. These short C1 timings are descriptive combined-TP2 figures;
normal-stop quality and broader lifecycle qualification remain incomplete.
See the [two-order record](../provenance/SPRINT7-MATCHED-CAPACITY-20261021.json)
and [remaining Sprint 7 gates](SPRINT7-QUALIFICATION.md).

## Explicit BF16 draft KV and owner disposition, 2026-10-21

The explicit `--speculative-draft-kv-cache-dtype bf16` override completed two
fresh GPU starts and eight C1 requests. Both ranks confirmed actual BF16 target
and draft K/V buffers in both starts. All eight forced 128-token outputs matched
the historical common response. Source, weight, HTTP, graph and independent
release checks passed after 38 CPU package checks and Astra implementation review.

Median full HTTP output rates were 69.15 and 69.31 tokens/s for combined TP2.
There was no flag-off control, isolated decode measurement or task-quality grade.
This screen did not test FP8 target KV or establish a mismatch fix.

The owner selected this configuration for the current session and directed us
to move on, backtracking if more strange behavior occurs. The mismatch diagnosis
is closed as unresolved. Earlier capitalization variants and the later stage-3
variation from token 16 remain retained; exact reproducibility and broader task
quality are not qualified. Continue with the selected snapshot reference and
explicit BF16 draft override in both future KV arms. No installed default or
persistent server was changed. See the [override validation and owner decision](../provenance/BF16-DRAFT-KV-OVERRIDE-20261021.json).

The owner deferred FP8 target attention KV to a [separate optional work item](SERVING-SPRINTS.md#optional-fp8-target-attention-kv). Keep the current FP8 weights and BF16 target/draft caches. The next active work is Sprint 7 normal-stop quality and sustained-output validation.


## Combined low-thinking task screen, 2026-10-21

The selected M2047 stage 1 + metadata reuse + DFlash8 snapshot passed four tasks
in each of two fresh starts with thinking enabled at low effort. All eight
responses stopped naturally and passed applicable final-answer task and format
checks. The two ledger finals contained 763 and 821 tokens and matched all 28
expected events, including the earlier J01/H17 fee-boundary cases. All four
output-ID pairs matched exactly. Independent GPU resource release was verified.

Ledger inputs were 12272 tokens, with a 4096-token total output cap inside the
existing 16K context. Inert filler was reduced; task facts and oracles stayed
fixed. This is a bounded greedy-profile result, not a mode-only comparison,
publisher-recipe qualification or general quality guarantee. Ring-fold and
other Sprint 7 gates remain pending. See the
[thinking quality record](../provenance/SPRINT7-THINKING-QUALITY-20261021.json)
and [validation skill](../.agents/skills/aithreadripper-serving-validation/SKILL.md).

## Sprint 7 matched thinking quality, 2026-10-21

Snapshot and ring-fold each passed four natural-stop tasks in one fresh start:
8/8 applicable task, format and sustained-final-output checks. All final texts
and token segments matched the two prior selected-profile starts. Two of four
full output-ID pairs matched; the ledger reasoning paths differed. Exact-output
equivalence and the earlier mismatch remain unresolved.

Both profiles observed 19 usable/20 physical recurrent slots per card and BF16
target/draft attention caches. Ring-fold again reduced startup Torch allocation
by 1.63 GiB per card. Zero request recaptures and equal per-rank commit/replay/verify
counts were authenticated, along with independent release. This is a bounded
quality result with one start per profile and no speedup or repeatability claim.
See the [record](../provenance/SPRINT7-RINGFOLD-THINKING-QUALITY-20261021.json).
C2, reuse/branching, cancellation and repeated lifecycle coverage remain pending.

## Both-order thinking lifecycle screen, 21 October 2026

Both profiles completed four fresh starts and 108 planned requests in both
orders. All 104 normal final answers were correct, and four active streaming
cancellations completed with correct recovery. Each start showed real batch-two
graph traffic on both ranks. Verifier counts, singleton/batch-two replays and
completed commits reconcile. All active request rows were released; free plus
evictable cached slots accounted for all 19 usable recurrent slots per rank.

The schedule covered prefix-boundary reuse and branches, cache pressure,
victim recovery, cancellation, concurrency and final replay. Integer allocator
observations establish slot bookkeeping, not numerical state equivalence or
per-node cache ownership. Full output IDs matched 42/52 cache comparisons; all
52 final-answer comparisons passed. Retain the ten reasoning-output differences
and the unresolved exact-output contract. Independent release passed after both
packets. These instrumented runs establish bounded lifecycle behavior and task
quality; their timings are not performance claims. See the
[lifecycle record](../provenance/SPRINT7-LIFECYCLE-20261021.json).

## Sprint 7 sampled screen and closeout, 21 October 2026

The final publisher-recipe screen passed 8/8 natural final answers on snapshot
and ring-fold. It used low thinking, temperature 1.0, top-p 0.95 and top-k 20.
All 16 external request/rank witnesses joined a successful native DFlash
selector-sampling return to the frozen request. All coding finals passed the
unchanged Linux grader; all ledger finals passed the full event oracles and
exceeded 512 final tokens. Independent release passed.

The ledger inputs were 8176 tokens, with an 8192-token total output cap and a
16-token reserve: 16384 tokens in total, within the selected context. Only inert
filler was reduced. This is a small functional sampling screen, not a distribution test
or a performance benchmark. See the [sampled result](../provenance/SPRINT7-SAMPLING-20261021.json).

The [bounded Sprint 7 closeout](../provenance/SPRINT7-CLOSEOUT-20261021.json)
retains the snapshot reference, the separately screened ring-fold option and
the unresolved exact-output exception. General numerical equivalence and
production qualification remain incomplete. No persistent server or installed
default was replaced.

## Offline cost attribution, 2026-10-22

The [current cost analysis](CURRENT-COST-ANALYSIS.md) uses the latest sampled
quality telemetry, a separate greedy counter crosscheck and the earlier
short-C1 device timing. Generation is the dominant service phase on the long
requests. Current GPU shares for drafting, target verification and commit are
not measured. The draft selector's large-vocabulary top-k fallback is a concrete
untimed candidate. No models were started and no new speedup was established.

## Sprint 8 native cost attribution, 22 October 2026

The corrected snapshot control/timing packet passed all eight natural final
answers, authenticated 126 completed samples per card and verified independent
release. It preserved the selected stack, flags, FP8 weights and BF16 caches.
All final-answer texts matched; one held-out reasoning length differed by eight
tokens. The prior numerical mismatch remains unresolved.

On the long requests, the target graph took about 44.85 ms within a 54.06–54.09 ms
sampled cycle on each rank, about 83% of that interval. About 42.64 ms remained
after the output head and final gather. Draft local top-k was only 0.068–0.071 ms.
These are per-rank instrumented windows, with missing and uncovered cycles
retained. They do not establish a whole-request share or speedup. Sprint 9 first
classifies the target graph through a bounded trace before choosing a component.
See the [attribution and coverage record](SPRINT8-COST-ATTRIBUTION.md).

## Sprint 9 target graph classification, 22 October 2026

The two-start diagnostic passed 8/8 final-answer gates and independent release,
with eight completed timing samples per card. In four long-request samples per
rank, projection methods total a median 30.757/30.730 ms within a 48.334/48.339 ms
target graph. MLP gate/up projections total 13.775/13.777 ms across layers. These
nested, instrumented intervals select an M8 gate/up component comparison; they
are not a clean serving benchmark. Three of four full output-ID pairs match;
all final answers match. The held-out reasoning variation remains recorded.
See [Sprint 9 tracking](SPRINT9-TARGET-GRAPH.md) for coverage gaps, overhead and
remaining gates. The stack, runtime flags and selected numerical profile remain
unchanged; no new optimization is qualified.

The M8 stage-depth component subsequently passed 96/96 exact comparisons,
including 24 graph replays across both cards. Stage 1 reduced eager component
median time by 6.61–13.06% across four structured/random cases. The first attempt's
capture-stream harness failure remains sealed; the corrected run passed both
cards and independent release. The experimental patch is recorded but remains
uninstalled. Sprint 9's component exit is complete after independent review;
Sprint 10 subsequently completed matched serving validation below. See the
[component record](../provenance/SPRINT9-M8-COMPONENT-20261022.json).


## Sprint 10 matched M8 serving, 22 October 2026

M8 gate/up stage one is selected for the next experimental profile launch after
Astra review. On fixed 2,047-input/1,024-output C1 work, median client latency
fell 0.6023% forward and 0.3479% in reverse order. Combined TP2 rates rose
0.6059% and 0.3491%. All six measured pairs match output IDs and 164 verifier
cycles; each start's first request is excluded. Existing observers remain equal
and additional classification timing events are disabled.

Repeated candidate screens passed greedy 8/8 and publisher-sampled 8/8 final-answer
gates, then lifecycle 52/52 natural answers plus two cancellations/recoveries,
actual C2 and slot accounting. Greedy finals all match between starts; two
reasoning sequences differ. Lifecycle full outputs match 24/26 cache pairs and
all 26 finals match. The numerical mismatch remains unresolved; this is bounded
qualification on the current snapshot/BF16 profile. Startup allocation is
20.52785 GiB/card, reserved 20.80078 GiB/card. Installed runtime and production
defaults remain unchanged. See [Sprint 10](SPRINT10-M8-SERVING.md) and its
[result index](../provenance/SPRINT10-M8-SERVING-20261022.json).


## Sprint 14 smaller unrolled tile, 22 October 2026

The exact untuned M8 down projection now selects native unrolled tile16/stage1
for the next experimental session launch, after Astra review. Both matched
orders passed all eight output-ID pairs and 164 verifier cycles per request.
Median client latency fell 1.7442% forward and 1.7851% in reverse, reaching
115.1192 and 115.1496 combined TP2 output tokens/s across both cards.
The workload uses C1, 2,047 input tokens and 1,024 forced output tokens with
low thinking. Each start's first request is excluded. Rates include prefill;
they are not decode-only or final-answer throughput.

Natural-stop screens passed greedy8/8, sampled8/8 and lifecycle52/52 final
answers, plus two cancellations/recoveries, actual C2, 19-slot accounting and
independent release. Greedy3/4 and cache24/26 full-output pairs matched; all
paired final answers matched. Sampling route execution passed on both ranks,
without establishing sampling-distribution equivalence. Earlier numerical
findings and Sprint 11 failed work gates remain unresolved under the owner's
experimental-session disposition. Reliability remains deferred; installed
runtime and production defaults are unchanged. Retain the previous selected
source for rollback. See [Sprint 14](SPRINT14-UNROLLED-TILE16.md) and its
[evidence index](../provenance/SPRINT14-UNROLLED-TILE16-20261022.json).

## Sprint 17 GDN input projection, 22 October 2026

The exact untuned M8 GDN qkvz projection `(8,8192,5120)` now selects native
unrolled tile16/stage1 for the next experimental session launch, after Astra
review. Both matched orders passed all eight output-ID pairs and 164 verifier
cycles per request. Median client latency fell 0.5435% forward and 0.6425% in
reverse, reaching 115.7085 and 115.7343 combined TP2 output tokens/s across both
cards. These rates include prefill on C1 work with 2,047 input tokens and 1,024
forced output tokens, low thinking and one excluded warmup per start. They are
not isolated decode or final-answer throughput measurements.

Greedy8/8, sampled8/8 and lifecycle52/52 normal final answers passed, plus two
cancellations/recoveries, actual C2, 19-slot accounting and independent release.
Greedy and sampled each matched all four full output pairs. Lifecycle matched
24/26 full cache-output pairs and all 26 final-answer pairs; retain the two
`concurrent-a` reasoning variants. Sampling route execution was verified on both
ranks without establishing distribution equivalence.

The current stack, FP8 weights, BF16 target/draft caches, FP32 state/head, flags,
DFlash8, metadata reuse and earlier selected routes remain unchanged. Preserve
Sprint 14 source for rollback, earlier numerical findings and Sprint 11 failed
gates. Reliability remains deferred; installed runtime and persistent defaults
are unchanged. See [Sprint 17](SPRINT17-GDN-INPUT.md) and its
[evidence index](../provenance/SPRINT17-GDN-INPUT-20261022.json).

## Sprint 18 attention QKV, 22 October 2026

The exact `(8,7168,5120)` unrolled tile16/stage1 option remains unselected after
Astra review. Component checks passed 96/96 with 16.25–18.25% less kernel time.
Both matched serving median gates passed at 0.1948–0.2237% lower latency; all
eight output pairs and 164 cycles/request matched. Candidate rates were
116.0174–116.0459 combined TP2 output tokens/s on the same C1, 2047-input/
1024-forced-output low-thinking workload, with one excluded warmup/start.
Rates include prefill; they are not decode-only or final-answer rates.

One reverse measured candidate request was 1.3016% slower. Including every sample,
the pooled mean was 0.0574% higher. Overall service benefit is inconclusive;
these descriptive results do not replace the selected Sprint 17 benchmark.
Greedy passed 8/8 final answers, with 4/4 paired final texts and 3/4 full output pairs
matching. Retain the audit-threshold reasoning variant. Prepared sampled and
lifecycle screens were not launched. Current source, stack, flags and precision
remain selected; reliability stays deferred. See [Sprint 18](SPRINT18-ATTENTION-QKV.md)
and its [evidence index](../provenance/SPRINT18-ATTENTION-QKV-20261022.json).


## Sprint 19 matched ring-fold, 22 October 2026

Both orders matched eight full-output ID pairs and 164 verification cycles
per request per rank. Ring-fold measured 120.3602 versus 115.6911 combined TP2 output
tokens/s forward and 120.2798 versus 115.6689 in reverse; median latency fell
3.87926% and 3.83348%, respectively. All six measured candidate requests were
faster, without a statistical significance claim. These C1 rates include
prefill on 2047-input/1024-forced-output, greedy low-thinking requests, with one
excluded warmup and three measured requests per start.

With 19 usable and 20 physical recurrent slots per card, startup allocation
after graph setup fell from 22,041,610,240 to 20,296,157,184 bytes on each
card in both orders: 1.62558 GiB per card. This is not whole-device or
process-lifetime peak memory. Geometry predicts 1,747,550,208 bytes of savings;
the 2,097,152-byte residual is unassigned. The saving is attributed to recurrent
ring-fold; the split-KV computational adapter retains BF16 caches.

Greedy natural-stop quality passed 8/8 final answers, with 4/4 full-output ID
pairs and 4/4 final-text pairs matching across profiles. The two long ledger
finals reached 763 and 821 tokens, above the 512-token gate. Source, native,
HTTP, cache, work and release evidence was authenticated. Publisher-sampled
quality passed 8/8 final answers, with 4/4 final-text and 2/4 full-output pairs
matching. Two ledger reasoning outputs differed; the native selector-sampling
route was verified on both ranks and arms without establishing distribution
equivalence. Lifecycle passed 52/52 normal final answers, two cancellations,
actual C2, slot bookkeeping and release. Of 26 cache-output pairs, 24 matched
full IDs and all 26 matched final text; both arms retain the `concurrent-a`
versus `victim-cold-a` reasoning difference at index 149. Ten starts covered 86
requests: 16 forced-output, 68 natural-stop and two cancellations. All 15
evidence roots passed closeout verification. Astra final review supports
selecting aligned ring-fold for the next experimental session only. The Sprint
17 GEMM remains unchanged and the paired snapshot control is retained for
rollback. Earlier
numerical findings, Sprint 11 failed gates and the owner experimental-session
disposition remain. Reliability and
endurance remain deferred, no GPU campaign is queued, and installed runtime and
persistent defaults remain unchanged. This is not production, numerical, state
or distribution qualification. See [Sprint 19](SPRINT19-MATCHED-RINGFOLD.md).


## Sprint 20 selected-profile cost refresh, 22 October 2026

Two starts and eight sampled low-thinking requests completed on the selected
Sprint 17 GEMM plus Sprint 19 ring-fold profile. All eight natural final answers
passed, all four full-output ID pairs matched, and verification-cycle counts
matched in each pair. Both ranks collected all eight scheduled device windows;
source, native, HTTP, cache, graph and independent release checks passed.

Four long-request windows per rank place projection methods at 27.550/27.516 ms
within a 44.445/44.446 ms target graph. Gate/up remains the largest projection
group at 11.939/11.932 ms; down takes 5.668/5.654 ms and GDN qkvz 4.384/4.377 ms.
These forward intervals include quantization, launches, waits and enclosed
collectives. Nested groups overlap; do not add groups or ranks into request time.

Instrumented client elapsed time increased by 2.83%, 8.27%, 6.52% and 7.02% with
matched outputs and verification work. One control-first order does not isolate
causal overhead. Coverage remains sparse, with 13/42/64/150 later cycles beyond
the last sampled window. This attribution does not establish a whole-request
bottleneck or a new serving speed. The historical Sprint 19 benchmark was about
120.3 combined TP2 tokens/s under its fixed-work envelope.

The follow-on common M16 component and serving comparison are recorded below.
See [Sprint 20](SPRINT20-CURRENT-PROFILE.md) and its
[evidence index](../provenance/SPRINT20-CURRENT-PROFILE-20261022.json).

## Sprint 21 common M16 gate/up, 22 October 2026

The common M16/stage1 candidate passed 96 exact component comparisons at
`(8,17408,5120)`. Both matched serving orders passed all eight output/work
pairs. Median latency fell 0.38064% and 0.28794%; candidate rates were 120.304
and 120.368 combined TP2 output tokens/s. The pooled mean latency change was
−0.29775%, a descriptive result. The historical Sprint 19 benchmark was about
120.3 combined TP2 tokens/s under its fixed-work envelope.

Each performance start had one excluded warmup and three measured requests.
Greedy and sampled quality used candidate GEMM and ring-fold in both arms; each
passed 8/8 final answers and 4/4 full-output pairs, with independent release.
The sampled route was verified, but its distribution was not. The 54-request
lifecycle screen passed 52/52 finals, two cancellations, C2, slots and release;
24/26 full cache outputs and 26/26 final texts matched. Both arms retained the
Sprint 19 first difference at token 149 for the same cache comparison, without a
proven cause. Astra selected M16 for the next experimental session only; the prior
GEMM remains rollback and ring-fold is unchanged. Earlier numerical findings and
deferred endurance remain. See [Sprint 21](SPRINT21-COMMON-M16.md) and its
[provenance](../provenance/SPRINT21-COMMON-M16-20261022.json).

## Sprint 22 GDN output candidate, 22 October 2026

The common M16/stage1 candidate at `(8,5120,3072)` passed 96 exact component
checks and all eight serving output/work pairs in both orders. Median latency
fell 0.89178% forward and 0.81403% in reverse, with candidate rates of
121.175–121.243 combined TP2 output tokens/s under the same fixed-work envelope.
These are finite C1 results for the source selected for the next experimental
session; the previous Sprint 21 source remains available for rollback.

Greedy validation passed 8/8 normal-stop answers and 4/4 full-output pairs.
The sampled campaign stopped when the supervisor reported that a process changed
during namespace observation at second-arm startup. The interrupted arm then
lacked its expected artifact bundle. Four first-arm HTTP responses remain
ungraded, and lifecycle validation was not launched in that initial attempt.
Independent release passed without survivors or unresolved cleanup.

The owner subsequently authorized completion. Diagnostic messages were added
to existing identity refusals without changing acceptance or read order.
Fresh sampled validation passed 8/8 normal answers and 4/4 full-output pairs;
the native sampling route was verified. Lifecycle passed 52/52 normal answers,
two cancellations, actual C2, slots and release. Cache full outputs matched
24/26 pairs and all 26 final texts matched, retaining the same two index-149
reasoning variations as Sprint 21. A local transfer timeout was recovered with
a reviewed compressed read-only collection; the GPU run was not repeated.
Astra approved the smaller tile for the next experimental session with prior
GEMM rollback and ring-fold unchanged. The initial observation cause, earlier
numerical findings and deferred endurance remain. See [Sprint 22](SPRINT22-GDN-OUTPUT-M16.md) and its
[provenance](../provenance/SPRINT22-GDN-OUTPUT-M16-20261022.json).

## Sprint 23 attention QKV common M16, 22 October 2026

The model-free `(8,7168,5120)` comparison changed only the common kernel's
M tile from 64 to 16, retaining stage three. All 96 exact comparisons passed,
including 24 graph comparisons. Timing was mixed: three cases improved
1.64–8.03%, while GPU 0's random case took 2.60% more time. This fails the
declared requirement for a lower median in all four cases, so the candidate
is parked. The run completed in 23.60 seconds with independent release and
zero model starts; no serving or quality campaign followed.

The selected Sprint 22 GEMM and ring-fold remain unchanged. Its measured
121.18–121.24 combined TP2 tokens/s was the qualified Sprint 22 result; Sprint 23
establishes no new serving rate or timing cause. See [Sprint 23](SPRINT23-ATTENTION-COMMON-M16.md)
and its [provenance](../provenance/SPRINT23-ATTENTION-COMMON-M16-20261022.json).

## Sprint 24 selected-profile cost refresh, 22 October 2026

The refresh retained the Sprint 22 GEMM and ring-fold profile. Two starts and
eight sampled low-thinking requests passed all eight natural final-answer
checks, sampling-route checks and independent release. Each rank completed
all eight selected target timing windows, including four long-case windows.

Long-window projection forward medians were 27.234/27.206 ms within target
graphs of 44.111/44.115 ms on ranks 0/1. Gate/up remained the largest projection
at 11.839/11.845 ms. These intervals include quantization, launches, waits and
enclosed collectives; overlapping groups and ranks are not additive.

Only 1/4 full-output and verification-work pairs matched; 3/4 final texts and
final ID pairs matched. One coding final changed while retaining its functional
pass. The other two differing outputs had exact final answers but different
reasoning. The three unmatched pairs cannot measure instrumentation overhead.
Retain all differences, sparse coverage and the unresolved numerical finding.
This is no new serving-speed result or source selection.

At Sprint 24 closeout, the proposed component was common M16 gate/up with N64
instead of N128, holding K128, stage one and block scales fixed. It was
source-feasible but not yet compiled or prepared. See [Sprint 24](SPRINT24-CURRENT-PROFILE.md)
and its [provenance](../provenance/SPRINT24-CURRENT-PROFILE-20261022.json).

## Sprint 25 gate/up N64, 23 October 2026

The common gate/up N64 candidate passed all 96 exact component comparisons
and reduced component medians by 0.540–0.980% across both GPUs and input
patterns. M16, K128, stage one, group 32, four warps and the original 128-wide
block scales remained fixed.

The forward serving pilot matched all four output-ID and verification-work
pairs, including warmup, at 164 cycles per request. Both ranks authenticated
the intended N128/N64 dispatch. With one warmup excluded and three measured
requests per start, N128's median was 8.464984 s, or 120.968929 combined TP2
tokens/s; N64's was 8.468574 s, or 120.917646 tokens/s. Candidate median
latency was 0.04241% higher. This small single-order result failed the required
improvement gate; it does not establish a significant slowdown. N64 is parked
without reverse, retry, natural-stop candidate screens or source selection.
Independent release passed. The selected Sprint 22 GEMM, ring-fold and retained
121.18–121.24 combined TP2 tokens/s result remain unchanged.

An offline source audit found that the native sampled accept path makes two
default-device RNG draws before its downstream deterministic kernel call.
The retained server seed does not identify the RNG state entering each request.
That state and matched selector inputs/draws remain unobserved, so this finding
establishes neither the cause of Sprint 24's variations nor numerical resolution.
At the Sprint 25 closeout, a small diagnostic of those inputs was proposed.
See [Sprint 25](SPRINT25-GATEUP-N64.md) and its
[provenance](../provenance/SPRINT25-GATEUP-N64-20261023.json).

## Sprint 26 sampled acceptance investigation, 23 October 2026

Two full instrumented traces completed: one repeated ring-fold and one
snapshot versus ring-fold. Across both, all 8,328 observed events, eight
full-output/work pairs and 16 functional answers matched; independent release
passed. Synchronous CPU copies can perturb timing. A cross-phase review found one differing start out of eight, with its raw
probability values unavailable. Earlier first-acceptance differences and a later
sampled branch remain unresolved, and these finite
comparisons do not prove numerical equivalence or ring-fold causality. The
selected `b101` GEMM and ring-fold profile remain unchanged. The historical qualified
Sprint 22 result was 121.18–121.24 combined TP2 tokens/s.
See [Sprint 26](SPRINT26-SAMPLER-STATE.md) and its
[provenance](../provenance/SPRINT26-SAMPLER-STATE-20261023.json).

## Sprint 27 gate/up cost isolation, 23 October 2026

The selected-profile diagnostic completed eight functional answers and four
matching full-output/work pairs, with eight complete device windows per rank.
Independent review approved the bounded results. Across four selected gate/up sites in
long windows, GEMM had a median per-window share of 91.06%/91.00% on ranks 0/1;
activation quantization had 3.22%/3.24%. The residual includes unattributed
work and event effects. This sampled scope is not a new speed benchmark or an
extrapolation to all 64 calls. At Sprint 27 closeout, the proposed next component changed stage one
to stage two for the current gate/up shape; Sprint 28 completed and parked it. See
[Sprint 27](SPRINT27-GATEUP-COST.md) and its
[provenance](../provenance/SPRINT27-GATEUP-COST-20261023.json).

## Sprint 28 gate/up stage two component, 23 October 2026

All 96 model-free exact comparisons passed, but stage two was slower in two
of four device/input timing cases. The candidate failed the all-case timing
gate and is parked without serving preparation or source selection. Independent
review approved the bounded closeout. The proposed CPU examination of retained
compiler artifacts completed in Sprint 29; it does not authorize a GPU run. The selected profile and
latest clean Sprint 22 serving benchmark remain unchanged. See
[Sprint 28](SPRINT28-GATEUP-STAGE2.md) and its
[provenance](../provenance/SPRINT28-GATEUP-STAGE2-20261023.json).

## Sprint 29 gate/up compiler review, 23 October 2026

A CPU-only review of the retained Sprint 28 compiler artifacts identified a
possible exact-shape scalar specialization. Sprint 29 did not compile or run a
candidate; Sprint 30 subsequently tested and parked it. Static instruction
counts do not establish a bottleneck or gain. Independent engineering review
approved the CPU findings and proposal. The selected profile and latest clean
Sprint 22 serving benchmark remain unchanged. See
[Sprint 29](SPRINT29-COMPILER-REVIEW.md) and its
[provenance](../provenance/SPRINT29-COMPILER-REVIEW-20261023.json).

## Sprint 30 gate/up static scalars, 23 October 2026

The model-free candidate passed 96 exact comparisons and all four conservative
compiler proofs, but GPU 0 random-input median latency was 0.709% higher. The
all-case timing gate failed; the candidate is parked without serving
preparation or source selection. Independent review approved the bounded results and parking. See
[Sprint 30](SPRINT30-GATEUP-STATIC-SCALARS.md) and its
[provenance](../provenance/SPRINT30-GATEUP-STATIC-SCALARS-20261023.json).

## Sprint 31 warp screen

W2 passed the component gate, but its forward serving median was 8.482985 s versus 8.481692 s for W4; the lower-median gate failed. The control measured 120.73 combined TP2 tokens/s and W2 120.71. All four output and cycle pairs matched. W2 and W8 are parked; independent review approved the closeout. This small sample does not establish a general slowdown. The selected profile remains unchanged. See [Sprint 31](SPRINT31-GATEUP-WARPS.md).

## Sprint 32 long-context screen

Two selected-profile 32K starts completed 14 normal responses and seven identical output-ID pairs. Strict SRE grading passed 2/12; the owner accepted two lexical exceptions for current experimental use, while the strict failure remains recorded. A retrospective audit found all 12 factual-field sets matched under those exceptions. Actual outputs did not reach the full context-plus-output boundary. See [Sprint 32](SPRINT32-LONG-CONTEXT.md). A separately reviewed 64K device screen followed in Sprint 33.

## Sprint 33 64K long-context screen

Two repeated starts completed 14 normal stops and seven equal output-ID and verifier-cycle pairs. The literal and factual SRE gates each scored 10/12 and failed; two answers used a string for an action-order array that the prompt had not explicitly requested. Both failures remain. Long inputs were 59,999 tokens and outputs were 567–849 tokens, so the full output boundary was not exercised. [Sprint 33](SPRINT33-LONG-CONTEXT64.md) records memory, release and scope; independent review approved the conservative closeout.

## Frozen Qwen3.8 baseline

The selected experimental profile and its bounded results are recorded in the [frozen baseline](QWEN38-FROZEN-BASELINE.md). The 64K grading failures and earlier numerical variation remain unresolved. The owner has parked Flash-Next and lower-bit checkpoint work to wait for Qwen4 dense 27B.

## Qwen3.8-Flash-Next feasibility

The [metadata and source preflight](QWEN-NEXT-FEASIBILITY.md) found that fully resident official FP8 tensors exceed combined physical RAM and VRAM. Source paths exist for ROCm, but device execution is unverified. The [candidate search](QWEN-NEXT-WEIGHT-CANDIDATES.md) is retained as research. No checkpoint or engine is selected and no experiment is queued; the frozen 27B selection is unchanged.

## Reopened Qwen3.8 tuning

The owner reopened the [remaining Qwen3.8 work](QWEN38-REOPENED-TUNING.md). [Sprint 35](SPRINT35-LONG-PREFIX-REUSE.md) completed two fresh starts with 12 normal decoded responses and 10/10 strict SRE answers across two repeated task families. Warm requests reused 24,576 tokens; cold requests reported zero. Cold long-request durations were 13.4–15.4 seconds and warm durations 4.9–6.4 seconds. Output IDs and verifier work matched in all corresponding comparisons. These individual HTTP timings are descriptive, not a benchmark speedup. The selected sources, BF16 caches and thinking-enabled profile remain the reference. The following Sprint 36 comparison is complete. Flash-Next and alternate engines remain parked.

## Prefill chunk sizing, Sprint 36

The [both-order screen](SPRINT36-PREFILL-CHUNKS.md) completed four fresh starts, 12 normal decoded finals and 8/8 strict SRE answers across two repeated task families. Retain 2048-token chunks. The 4096 configuration reduced long-request chunk counts from 13 to 7, but observed request times were higher in both orders and cumulative peak allocator counters increased by 359.125 MiB allocated and 472 MiB reserved per rank. Current allocated memory at the final prefill return differed by only 0.0913 MiB. All long cross-configuration pairs had unequal generated work; HTTP timings include instrumentation and possible compilation and do not isolate prefill. The initial postrun guard failure and its ungraded responses remain sealed separately; the corrected consumer passed real-evidence regression and native validation. The selected kernels, BF16 caches, precision, capacity and installed defaults remain unchanged.

## FP8 target-KV feasibility, Sprint 37

The [source audit](SPRINT37-FP8-KV-FEASIBILITY.md) found native FP8 scale handling, but the generic FP8 cache path changes query arithmetic and the selected split adapter requires BF16 cache tensors. The protected two-start owner cannot represent the original four-start campaign within its declared budget. Retain BF16; no GPU test, performance result or runtime change occurred. Compatibility work, a fitting thinking corpus and a schedule decision remain separate prerequisites.

## Adaptive-speculation prerequisites, Sprint 38

The [offline analysis](SPRINT38-ADAPTIVE-SPECULATION.md) reused 1,041 distinct sampled verification cycles across four Sprint 26 tasks. It joined exact proposal probabilities to 6,335 accepted drafts and 229 first rejections, leaving 723 later positions unlabeled. Selected-token probability ranked the observed labels with AUC 0.9768; this is neither answer accuracy nor a speedup. Repeated starts and ranks corroborated the data without increasing the independent sample. Fixed DFlash8 remains selected. Greedy confidence, policy generalization and variable-width graph/state support remain separate work. No new GPU run occurred.

## Sequential and concurrent pairs, Sprint 39

The [C2 screen](SPRINT39-C2-THROUGHPUT.md) completed two fresh starts with reversed group order: 38 authenticated responses, including 24 measured fixed-output requests and 8/8 passing natural-answer checks across two repeated cases. C1 aggregates were 120.18 and 119.95 combined TP2 tok/s; C2 aggregates were 172.98 and 172.75. Median individual latency increased from 8.50–8.52 s to 11.76–11.87 s. Actual batch-two execution was verified on both ranks.

These are instrumented HTTP rates for 2047 input and 1024 forced output tokens per measured request. Every C1/C2 timing comparison had unequal output IDs and verifier work, so no matched-work or intrinsic kernel speedup is claimed. Startup Torch allocation was 19.565 GiB per card, with 19.854 GiB reserved; this is not whole-device peak memory. All selected numerical sources and runtime defaults remain unchanged. The initial offline acceptance-join refusal was corrected against the same sealed native run; independent release passed and no GPU retry occurred.

## BF16 64K capacity, Sprint 40

The [64K capacity screen](SPRINT40-KV-CAPACITY.md) completed twelve decoded natural responses, with 10/10 strict typed long-document checks across two repeated cases. Inputs were 59,993/59,980 tokens; actual long outputs were 516/525 tokens. All six cross-start output-ID and verifier-count pairs matched. Warm requests reused 59,392 prompt tokens and took 5.09–5.26 s, compared with 34.79–35.41 s for isolated cold requests. Timings are descriptive HTTP intervals.

Maximum recorded cumulative Torch peaks were 21.403 GiB allocated and 21.734 GiB reserved per card, including startup. Completion-time driver-free snapshots ranged from 8.781 to 9.039 GiB; whole-device peak memory was not measured. BF16 target/draft KV fits this bounded C1 workload. Defer UltraQuant's graph/speculative codec port: the retained mixed layout projects 1.102 GiB/card of target-payload savings at nominal 64K before workspace and other costs. Defaults and numerical sources remain unchanged; Sprint 33 failures and the unresolved numerical findings remain recorded.

## Remaining FP8 programme final checkpoint, 25 October 2026

The [programme closeout](REMAINING-FP8-OPTIMIZATIONS.md) retains the b101/2K/fixed-eight baseline; no new runtime optimisation passed all selection gates. Final bounded QA passed 14 native answer checks and two isolated code graders. The [completed BetterBench checkpoint](../benchmarks/results/20261025-final-baseline/final-summary.html) passed 106 requests across two fresh starts, source/native/owner checks and independent release. Mixed-workload C1 rates were 81.4–87.3 output tok/s and C2 aggregate rates 133.1–134.9 across both GPUs, including prefill/HTTP completion and instrumentation. Work differs from the earlier fixed-output tests, so no matched regression or gain is claimed. The two earlier failed benchmark attempts remain recorded. Endurance and production qualification remain NOT RUN; no service was deployed.


## MXFP4 verification-prefetch serving A/B, 3 November 2026

Operation 212 completed two sequential starts and passed all 60 natural-answer
grades with independent release. The frozen comparator returned **REVERT**:
the balanced speed ratio improved by 0.773%, below the required 3%. This is
equivalent to a 0.767% geometric-mean latency reduction. All completion-token
counts matched across corresponding requests, and no case regressed beyond
its predeclared noise band.

| Case | Baseline median (s) | Candidate median (s) | Less latency |
| --- | ---: | ---: | ---: |
| Long C1 cold | 77.028 | 77.555 | −0.68% |
| Long C1 warm | 9.024 | 8.763 | 2.89% |
| Long C2 cold | 56.947 | 57.066 | −0.21% |
| Long C2 warm | 8.319 | 8.089 | 2.76% |
| Short C1 cold | 12.888 | 12.903 | −0.12% |
| Short C2 cold | 13.984 | 14.000 | −0.11% |

Each case has three samples. C2 reports the median of concurrent pairs' mean
request latency. Only the warm cases improved beyond the frozen 2% noise floor.
Both arms used MXFP4 weights, BF16 caches, selected C4 extend attention, the
explicit cap-3 cache experiment and observed 300 W caps. Older 235 W serving
measurements are not a matched baseline.

**The promised graph-route proof is incomplete.** Production mode skipped the
startup writer and reader that would establish candidate calls in both 8-row
and 16-row graphs and zero calls in control. The retained source, HTTP, quality
and release evidence remains valid, but the complete routing claim is unverified.
This corrects the preparation review's assertion that this witness gate would
run in the timed packet. The runner's original result is preserved; the parent
marks the experiment ineligible for adoption.

The [provenance record](../provenance/MXFP4-VERIFY-PREFETCH-SERVING-20261103.json)
binds the sealed measurements and gap diagnosis. The operation is parked with
zero GPU allowance. No retry, adoption, selected-source change or deployment
occurred.

## Four-member qualification attempts, 4 November 2026

Operations 221 and 222 at 235 W both stopped in control graph warmup, before
HTTP qualification or scored requests. The allocator-predicate repair in 222
advanced beyond221, then the strict gate/up M16 compiler-record check refused.
Both runs were retained and independently released; no serving improvement or
quality verdict exists, and no adoption or selection changed. Both timing
sample counts are zero. See the [failure and CPU hold](MXFP4-FOUR-MEMBER-SERVING-VALIDATION.md).

CPU follow-up to operation 222 reproduced compiler-cache reuse across component
and serving source paths. The repaired proof and real CPU-tensor binder retain
fresh serving-context artifacts; all ten binaries preserve machine code and
resources, while five full hashes change only in debug sections. Forty-one
focused tests and public43 pass. Opus review is pending; no new GPU run, serving
gain or adoption is implied. See [the current four-member validation](MXFP4-FOUR-MEMBER-SERVING-VALIDATION.md).

Opus 5.5 subsequently accepted the CPU cache repair. Fresh packet
ac6c108f39cbae77 is sealed and awaits exact owner approval. Its preparation
passes 60 local checks, 59 native CPU tests, public43, exact inert package/caller
reproduction and six registrar refusal mutations. No registration or GPU
execution occurred, and no serving gain is established. The new decision (`FOUR-MEMBER-SERVING-OWNER-DECISION-v3.md`, private evidence) binds the proposed one-invocation scope.

## Operation 223 export refusal and CPU repair

Operation 223 served control requests, then refused export with encoded bundle
overflow. The candidate never started, no validated A/B timings or grades were
recovered, and independent release passed at checkpoint 601. Opus 5.5 at xhigh
fixed the derived bound and added a build-time guard. Fresh packet
a4777122b5277d13 passes 70 local checks, 69 native CPU tests, public43 and exact
owner/caller/registrar checks. It has no GPU approval or registration. See the
[current validation](MXFP4-FOUR-MEMBER-SERVING-VALIDATION.md).

## Operation 224: completed four-member serving A/B

Operation 224 completed approved packet a4777122b5277d13: two sequential
TP2 starts at configured 235 W per card, control then candidate. The result is
INCONCLUSIVE and parked. The raw balanced speedup score is 1.016601
(baseline 1.000000; +1.66%), below the frozen 3% gate. All six cases exceed the
2% generated-work limit, and long C2 triggers no-regression vetoes. No causal
serving gain or energy saving is established.

| Case | Baseline (s) | Four-member bundle (s) | Latency change |
| --- | ---: | ---: | ---: |
| Long C1 cold | 82.467 | 82.955 | +0.59% |
| Long C1 warm | 9.607 | 9.391 | -2.25% |
| Long C2 cold | 60.915 | 61.670 | +1.24% |
| Long C2 warm | 8.842 | 9.324 | +5.46% |
| Short C1 | 13.713 | 12.675 | -7.57% |
| Short C2 | 14.828 | 13.843 | -6.64% |

Each case has three scored samples per arm. C2 is the median of concurrent
pairs' mean request latency; positive change means slower. Long C1 produced
737 versus 721 tokens; long C2 pairs produced 804/538 versus 998/563; short
requests produced 2300 versus 2244. Final-segment token counts match within
each case, while reasoning lengths differ. These raw times cannot isolate
the kernel effect.

Both ranks passed representative F1/GEMM exactness probes across all 48/64
layers at 8 and 16 rows, and the startup capture witnesses passed. Candidate
verification used cap32; ordinary decode stayed cap16. All 60 natural-stop
answers passed grading and long-answer coverage. These are bounded serving
checks, not whole-model bitwise equivalence or production qualification.

The native launch took 1499.01 s; the finite invocation took 1582.07 s.
Export and readback succeeded. Source/HTTP replay, native seals and independent
release authenticate. Checkpoint 603 records no active owner or reservation
and zero remaining GPU allowance. No retry, adoption, selection change or
Hyperloom campaign followed.

The sealed readout (`REPORT.md`, private evidence) includes exact values, pins, token counts and the reproduced comparator.
The four members remain research candidates. Further serving spend stays HOLD.
