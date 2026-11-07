# MXFP4 optimisation sprint plan

**Current queue, 4 November:** see [remaining gates](MXFP4-REMAINING-GATES.md). The gate/up wave-hint CPU follow-up passed; its narrow host/launch gate is now CPU-signed-off. The full GPU component and serving qualification remain pending. The historical sprint results below are preserved.

**New backlog addition:** [effective memory bandwidth](MXFP4-EFFECTIVE-BANDWIDTH-LEADS.md). BW-01 and BW-02 refine the existing entropy and integer leads; BW-03 adds certified output-head screening; BW-04 adds graphics-compression feasibility. Start with a representative checkpoint entropy/scale census. All original 21 references and cumulative candidates remain recorded.

**Operation 214 completed and parked — 3 November 2026.** OPT-00 through OPT-06 keep their reviewed CPU exits. [OPT-07](MXFP4-OPT07-CUMULATIVE-PREPARATION.md) passed its four-arm attention component at 235 W: prefetch + cap32 reduced primary component time by 14.21% and 14.13% on the two cards, with no resolved per-case regression. Cap32's added benefit over prefetch remains inside noise. All hard gates and independent release passed. Serving remains HOLD_FOR_EVIDENCE; no adoption, selection change or further GPU allowance follows.

F1 gated-norm/MXFP4 fusion completed as operation 215 at 235 W: exact component gates passed and all timing medians improved. It is RETAINED_POSITIVE for cumulative evaluation, while the strict standalone verdict remains NOT_USEFUL. Including GEMM, T8 saved 4.72% / 4.55% above noise; T16 saved 4.23% / 4.87% inside noise. Independent release passed; the one-use approval is consumed. Production qualification and serving gains remain unestablished. See the F1 result (`RESULT.md`, private evidence).

The objective is balanced C1/C2 natural-completion latency with no per-case regression. Keep the selected model, precision, tensor parallelism and context capacity. Screen every lead cheaply; send only justified candidates to physical tests. A documented rejection, duplicate or lack of measurable opportunity is a valid sprint result.

**Owner amendment — cumulative gains.** After OPT-00 through OPT-06 have dispositions, OPT-07 will test a compatible combination of changes with smaller performance or efficiency benefits. Missing a standalone usefulness threshold does not discard a change from that later experiment. Preserve each original verdict, distinguish measured benefits from unresolved signals, and evaluate the combined effect against a fresh control.

The [coverage and budget record](../provenance/MXFP4-OPTIMIZATION-SPRINT-PLAN-20261103.json) maps all twelve original backlog entries and all nine deep-search entries. Those are 21 references, with deliberate overlap, not 21 independent optimisations. It also records deferred work and previous failures.

## Shared reference and prerequisites

Freeze a reference called **B0** from the adopted C4 MXFP4 composition. Preserve its selected FP8 target/draft-head overlays, packed projection, copy fusion, P3 normalization, aligned ring-fold and BF16 caches. Inspect the consumed overlays before inferring a cost from stock SGLang code.

For new balanced serving comparisons, use the operation-212 **control source and cache policy**, with the explicit cap-3 MAMBA experiment in both arms, at the owner's restored **235 W caps on both cards**. Operation 212 ran at 300 W; its absolute timings cannot serve as this control. The selected pointer records 235 W and does not include cap-3. Bind the cache-policy deviation explicitly; do not describe it as global adoption or change the power setting. Re-authenticate the actual boot, source, runtime and observations when preparing a packet. If they differ, resolve the baseline before timing.

Keep the existing six-case menu: long C1 cold/warm, long C2 cold/warm, short C1 cold and short C2 cold. Each start has 30 requests, including warmups, and three observations per case. Preserve prompts, natural completion, thinking at low effort, warm-cache minima, DFlash8, TP2, context131040, pool262144, request limit2 and chunks2048.

Operation212 is parked and its prefetch candidate is unselected. Its production route witnesses were missing. **Repair and exercise the real production producer-to-reader path before any new serving candidate**, with missing, wrong-arm and wrong-geometry records refusing. Existing numerical/component evidence remains useful; the old run cannot become complete retrospectively.

## Sprint map

These identifiers are a new planning series and do not renumber the historical serving sprints.

| Sprint | Work grouped together | Reviewable exit |
| --- | --- | --- |
| OPT-00 | Baseline, source census, evidence reuse and test readiness | One baseline manifest, lead matrix and working production witness contract |
| OPT-01 | HIP graphs/queues, CPU placement, scheduler and TP2 transport | A bounded runtime hypothesis, or a justified no-op; shared profile only if needed |
| OPT-02 | Weight packing, lossless coding, integer algebra and GEMM layout | Exact representations and CPU compiler evidence for surviving candidates |
| OPT-03 | C4 index width/prefetch/layout and verification partitioning | Separate prefill and verification component candidates |
| OPT-04 | Recurrent commit and remaining small fusions | One removable pass with numerical and state-lifecycle proof |
| OPT-05 | Draft vocabulary/head cost and existing draft graph routes | A shortlist design with a credible acceptance-versus-cost case |
| OPT-06 | Cache tree and complete speculative-round capture | Separately costed structural designs; no automatic large port |
| OPT-07 | Cumulative integration of compatible large and small benefits | Fresh paired serving evidence, affected lifecycle checks and an owner decision |

OPT-00 precedes every physical proposal. OPT-02 through OPT-05 can share evidence and progress independently after that. OPT-06 consumes the interface and lifecycle findings from OPT-01/04/05; it does not require those candidates to win. Per the owner's amendment, freeze the OPT-07 cumulative bundle after OPT-00 through OPT-06 have dispositions. A deferred or rejected branch counts as a disposition; it need not be forced into an implementation.

## OPT-00 — baseline and evidence readiness

Build one matrix containing the consumed source, dispatch condition, relevant shapes, existing optimisation, remaining work, known cost and missing evidence for every lead.

Reuse authenticated compiler artifacts, source seals and unchanged test receipts. Complete source imports, actual producer/reader/caller/collector interfaces and production-mode route checks must be exercised under the declared runtime. Keep Mac work standard-library-only; framework imports and compilation belong in the established device-free native CPU sandbox.

Carry the repairs already learned from GPU failures: runtime device counts, actual split helper behavior, module/linecache restoration, retained primary tracebacks and isolated BLAS workspace cleanup. Test success and failure closeout without replacing real guards with mocks that hide leakage. Bound packaging, preparation time, archive membership and output retention before registration.

Create candidate contracts before measuring candidates. Existing families keep their frozen oracles and resource gates. New head/state/fusion/codec families need an explicit reference and acceptance contract reviewed before GPU use; a generic attention tolerance cannot qualify them.

**Stop:** source identity, producer/reader behavior or release evidence is unresolved. No serving slot is spent to discover a CPU-reproducible interface defect.

## OPT-01 — host, runtime and communication

Bundle the H2/H3 audits with the existing host/TP2 backlog:

- Determine whether the installed HIP already contains flat graph-packet dispatch and which path the captured graphs use. Inspect effective hardware queue configuration.
- Map scheduler and ROCr helper threads to permitted cores, CCD/NUMA locality and memory placement. Check cgroup limits, ancestor throttling, pressure and active CPU frequency policy.
- Identify actual collective sizes, dependency waits and transport. Read why overlap scheduling and custom all-reduce are disabled.
- Audit the already-present draft sampler/cache routes. Keep H1's two prefill metadata copies deprioritized unless new evidence shows an exposed cost.

If retained evidence cannot answer a named decision, prepare **one shared diagnostic profile**. Collect bounded request-phase markers, both-rank graph/collective windows, draft/target/head/commit timing, accepted-token counters and CPU scheduling/pressure data. The same capture should answer questions for OPT-02/04/05/06 where feasible. Pin instrumentation and output bounds. Do not sum overlapping intervals into whole-request percentages or use profiled time as a speed verdict.

Configuration candidates are separate hypotheses: queue count, affinity policy or scheduler overlap are not toggled together without individual supporting evidence. Check them first through source/CPU tests; use a matched serving slot when physical latency is required. Do not disable IOMMU/ACS, relax cgroups or change BIOS/power/security controls as a tuning shortcut.

**Stop:** a feature is already active, the supposed delay is hidden, no harmful placement/throttling exists, or a change would starve helpers or serialize useful C2/TP2 work.

## OPT-02 — representation, mathematics and GEMM

Share the retained GEMM shape census, operand truth tables, baseline compiler proxy and resource reader across four branches.

1. **G2 packing first.** Prove a same-size inverse mapping of packed weights and E8M0 scales. Identify a specific runtime conversion or gather that disappears. Preserve activation quantization, scale association, arithmetic and selected shape dispatch.
2. **G3 entropy screen.** On bounded CPU samples, measure code/scale entropy and actual tile-code size including offsets and tables. Prove exact byte recovery. Do not build a GPU codec unless the saved traffic plausibly exceeds decoding and synchronization cost.
3. **Integer reduction proof.** Continue finite-precision analysis of underflow, overflow, signed zero, special scales and accumulation order. Real-number algebra or a serial reference does not prove the selected WMMA behavior.
4. **Other scheduling.** Require a distinct dependency/layout mechanism. MI350 packing and pipeline methods are references, not permission to assume scaled MFMA, transpose reads or CDNA resource limits on RDNA4.

For a surviving mechanism, reproduce the exact baseline first, then compile the candidate on one representative shape before expanding to all affected M8/M16 families. Report registers, spills/private segment, LDS, loads, waits and instruction changes. Freeze a small candidate set; allow a source revision only when a concrete compiler result identifies a correctable mechanism. Stop allocation rerolls and blind parameter sweeps.

**Physical route:** eligible candidates compete for the shared component slots. Entropy coding and integer algebra remain CPU-only until their stronger proof obligations pass.

## OPT-03 — attention as two related test families

For C4, examine int32 index storage, index-ahead, address rematerialization and V/head layout using one source/ISA map. Include the once-per-batch index conversion in total cost; do not hide a GPU synchronization or conversion inside setup. Prove index range and retain the required address-arithmetic width. Unsupported, negative or overflow cases must keep the established fallback.

Where both mechanisms survive independently, compile baseline, int32-only, prefetch-only and their composition. A combined candidate needs its own resource and correctness proof; fewer index registers might help prefetch, but that is not assumed.

For verification attention, examine split/partition strategy through the actual per-row selector and both kernel stages. Cover long rows8, long rows16 with unequal lengths, short prefixes, empty splits and retained index orders. Changed reduction order uses the established numerical oracle; bitwise-neutral data movement still requires bitwise equality.

C4 cases additionally cover zero-prefix, long-prefix, boundary/tail and masked behavior. Carry short/zero-prefix regression risk into serving eligibility. A compile win at the main shape is insufficient.

**Bundling:** share fixtures, compiler tooling and input generation. Extend and verification remain separate component families and packet closures. A new multi-arm runner is not assumed to exist.

**Stop:** a frozen compiler/numerical gate fails, wrapper overhead removes the benefit, or a resolved regression appears. The failed two-query and EXP2 experiments remain closed.

## OPT-04 — recurrent state and small fusions

First trace selected ring-fold acceptance, commit, next read and cache publication. Use DeltaLog and MI350 GDN fusion to locate a remaining transfer, not to replace the recurrence speculatively.

Choose one boundary: accepted-state application, convolution/recurrence, gated-norm/quantization, or a comparable remaining pass. Preserve intermediate rounding where the contract requires it. GDN single-token decode code is not automatically valid for DFlash verification.

Build CPU lifecycle fixtures for zero/all/partial acceptance, rejection rollback, different C2 accept lengths, slot reuse, chunk boundaries, prefix hits, cancellation, termination and forced exceptions. Compare state and outputs under the declared reference. A deferred representation must flush before every consumer that requires dense state.

All-reduce/normalization/quantization fusion is a separate subcase. It depends on OPT-01 transport and ordering evidence and must preserve the selected collective semantics. It is excluded from the first local-fusion bundle if those facts remain unknown.

**Physical route:** one minimal surviving candidate, followed by state-sensitive serving/lifecycle qualification. Do not combine a new recurrence with a new collective in its first test.

## OPT-05 — draft-only vocabulary reduction

Audit the **selected FP8 head adapters** and their graph coverage; stock dense-head presence is not proof of the active cost. Share this audit with the already-present compact draft-cache and TP greedy max/id route checks.

Model fixed-address shortlist buffers, per-request ownership, refresh rules, global token IDs, TP shard ties, empty/low-coverage lists and fallback. The target retains its full vocabulary and verifier. Pruned draft proposals may differ; accepted final behavior must satisfy the frozen target contract.

Use retained suitable logits or token evidence for a preliminary coverage screen. If such evidence is absent, say so: token frequency alone cannot establish acceptance. Any required future capture belongs in the single shared profile proposal when its volume and privacy bounds fit.

A candidate must save total round time after gathering, shortlist maintenance and acceptance loss. Evaluate accepted tokens per round divided by round time. Changed acceptance is an expected outcome here; unchanged final work/quality remains mandatory.

**Physical route:** head/index correctness and cost screen first, then target-output/acceptance checks before serving timing. Do not mistake a faster shortlisted GEMM for a serving win.

## OPT-06 — structural cache and graph work

Keep two independent subprojects behind the shared lifecycle interfaces.

**Cache core:** replay retained radix-tree event sequences on CPU and compare lookup, eviction, lock and MAMBA slot behavior. Measure tree CPU cost before a Rust port. Session-aware retention must preserve the fixed workload, request isolation and explicit cap-3 policy. If no relevant tree cost or eviction problem exists, park it. Do not introduce artificial priming.

**Complete speculative round:** map the MI350 same-model graph design onto our TP2, C2 and precision contract. Start with one exposed graph boundary. Prove per-request acceptance counters, state publication, finite graph exit, cancellation points and error recovery. Persistent grid synchronization requires an RDNA4 residency/deadlock analysis.

**Physical route:** full graph behavior needs a dedicated serving/lifecycle qualification proposal; a model-free component cannot qualify it. Cost these projects after CPU feasibility. They are outside the first-wave cap unless explicitly substituted. They are not bundled together in an initial experiment.

## OPT-07 — test cumulative gains without losing attribution

A sprint can share preparation and harnesses without combining source changes. Keep one cumulative-candidate register in the plan's provenance. At each sprint exit, retain every smaller observed latency, memory, traffic or energy benefit with its baseline, sample spread, affected cases, correctness/resource results, remaining qualification and interaction risks. Record the metric actually measured: fewer instructions or allocated bytes do not establish lower energy or request latency.

Before the final bundle, reconcile the earlier experiment records as well as OPT-00 through OPT-06. Keep three distinctions: measured benefit below the standalone threshold; positive but unresolved signal; and blocked or failed evidence. A correctness, resource, lifecycle or resolved-regression failure cannot become eligible just by joining a bundle. Repair and validate the specific defect first. Already-selected changes belong in B0 and are not counted again.

Build the largest compatible, validated set that fits one finite serving pair. There is **no arbitrary two-change limit**. Source conflicts, an unqualified structural change, memory, preparation or retention limits can require smaller compatible sets; state every exclusion and do not extend the owner limits. A primary timing signal inside its spread may enter as an explicitly uncertain experimental member when the other gates pass. It is not labelled a demonstrated individual gain.

- Use a same-time B0 control and B0+bundle candidate at matched 235 W. Recheck composed source, dispatch, resources and numerical behavior. Changes to the same kernel require a new combined native compile and oracle screen.
- State the interaction hypothesis. Do not add component percentage gains or infer each change's serving contribution from the bundle.
- Keep runtime-only changes separate from a first unqualified state/graph change. A runtime candidate uses a serving slot; it has no faithful model-free latency proxy.
- Run one two-start pilot. If it clears all gates, prepare a separate reversed-order pair. Only surviving bundles earn that second cost.
- On failure or a null result, retain and park. Propose one targeted ablation only when the retained evidence identifies a specific unresolved interaction; no exhaustive combination sweep or automatic repeat.
- Complete the minimum affected cache/cancel/state lifecycle menu before adoption. Its packet and budget are explicit and separate from the unchanged timing menu.

Both serving orders keep the frozen comparator: three observations per case, 2% noise floor, at most 10% spread, at most 2% completion-token difference, at least 3% balanced speed-ratio gain and no per-case regression. All natural answers, warm-cache minima, route witnesses and independent release must pass. A null result remains null; bundling does not lower a gate.

Current component families retain their source-bound rules, including the 10% primary saving screen and range/regression checks where specified. Operation 213 remains NOT_USEFUL under that rule. The owner's amendment changes eligibility for a **new cumulative experiment**, not the original result: a change that missed the standalone performance gate can be considered without relaxing correctness, resource or no-regression gates. The new bundle contract must name these members and their uncertainty before approval and measurement.

Initial register: cap32 is retained for cumulative evaluation after its successful correctness/resource screen, with the primary timing signal still unresolved. Verification prefetch remains a separate lead with incomplete serving route evidence; OPT-00 repaired the witness on CPU, but physical route proof is still required before accepting a new bundle. The lossless codec's CPU storage reduction remains an efficiency lead, not a runnable GPU saving. These cases are tracked in provenance so “parked” does not mean “lost.”

## Spending and batching proposal

These are conservative **reservation envelopes**, not predicted execution durations or permission. The old four-hour Hyperloom allowance is expired.

| Unit | Existing execution limits used for planning | Reservation |
| --- | --- | ---: |
| C: model-free component | Owner1500s, caller2190s, worker600s, each card280s; cards sequential; observed allocated/reserved peak veto6GiB/card | 2310s, including120s planning overhead |
| AB: serving pair | Two TP2 starts; owner6360s, caller7160s, worker2280s/start, HTTP2040s/start, 60 post-release grades within320s | 7660s overall |
| P: shared profile | Exact instrumentation, retention and time projection still need preparation; conservatively reserve a serving-sized envelope | 7660s placeholder |

The extra120s for C does not extend any worker or caller. No new packet may silently inherit limits from a different family. If a proposed bundle does not fit the existing CPU preparation, execution or retention bounds, split it before asking for execution approval.

**Recommended first physical batch:** at most **2C + 1AB = 12,280s (3h24m40s)** reserved, after CPU screening. If the evidence is insufficient, substitute P for AB; that batch diagnoses costs and produces no serving gain verdict.

**Optional first-wave ceiling:** **4C + 2AB + 1P = 32,220s (8h57m)** across separately approved batches, with at most six model starts. This is a proposal, not a renewal of the old campaign. All physical candidates share those slots; the cap is not repeated for each sprint. If a host A/B consumes a slot, fewer remain for a combined pilot or reversal.

Structural work, additional ablations, affected lifecycle qualification and endurance are not hidden in these totals. Estimate the exact required follow-up before committing it. A candidate without a remaining confirmation/lifecycle budget stays unqualified.

Reserve the planned serving pilot for the final cumulative bundle rather than spending one serving pair per small gain. Include any combined compiler/component checks in its costed preparation. This amendment adds no invocation or time allowance; the final packet still requires its exact approval.

To control engineering and review credits, first give every branch one bounded evidence/CPU screen and rank the survivors. Reuse unchanged receipts and stop compiler failures promptly. The owner now requires one consolidated Opus 5.5 review before sprint signoff, replacing the earlier Fable preference. Do not commission separate reviews for every idea or automatically switch provider on a quota failure.

## Coverage and retained dispositions

| Prior backlog entry | Sprint |
| --- | --- |
| 1. MXFP4 prepacking/lookup | OPT-02, G2 |
| 2. Recurrent commit fusion | OPT-04, R1 |
| 3. Host scheduler/submission | OPT-01; structural continuation OPT-06 |
| 4. TP2 communication/NUMA | OPT-01; collective fusion OPT-04 |
| 5. C4 int32 indices | OPT-03 |
| 6. C4 index-ahead | OPT-03 |
| 7. C4 address/V/head layout | OPT-03 |
| 8. Verification partitioning | OPT-03 |
| 9. Integer MXFP4 group reduction | OPT-02, CPU proof |
| 10. Other RDNA4 GEMM scheduling | OPT-02, distinct mechanism only |
| 11. H1 metadata copies | OPT-01, retained low priority |
| 12. Existing draft cache/TP greedy graph | OPT-01/05 dispatch audit |

The nine Exa entries map as follows: H2/H3→OPT-01; G2/G3→OPT-02; R1/F1→OPT-04; D1→OPT-05; M1/H4→OPT-06. The G2/R1 entries refer to existing work, not extra experiments. MI350 references inform G2, F1, R1 and M1; their hardware-specific constants do not transfer.

Preserve negative results for N32 GEMM, EXP2 reorder, two-query attention, scalar bitdecode and scale-prefetch, plus the parked accumulator and verification-prefetch serving outcomes. An old failure can be revisited only with a new explicit mechanism and decision, not as an undocumented retry.

Adaptive width, trained/tree drafters, sparse attention, requantization, FP8 KV, new model architectures and a renewed autonomous Hyperloom campaign remain separate scope decisions. Endurance stays owner-deferred; that does not remove the bounded lifecycle obligations of a new candidate.

## Review output and execution boundary

Every sprint returns a short disposition: **already present, rejected on CPU, evidence missing, ready for a component packet, ready for serving, or parked after physical measurement**. Include absolute baseline and candidate measurements, sample ranges, work/quality, route, memory and release status when measurements exist.

OPT-00 through OPT-06 are signed off. [OPT-02 CPU screening](MXFP4-OPT02-REPRESENTATION-SCREEN.md) produced no GPU-ready candidate. [OPT-03](MXFP4-OPT03-ATTENTION-SCREEN.md) completed its cap16/cap32 component as operation 213. All 128 layer/reference checks passed, but primary savings of 2.59% and 2.16% missed the fixed 10% gate and did not exceed the observed timing ranges. No resolved regression occurred. Keep cap32 standalone NOT_USEFUL and retain it in the cumulative register; no serving gain or adoption follows. [OPT-04](MXFP4-OPT04-RECURRENT-FUSION-SCREEN.md) retains a gated-norm/MXFP4 candidate whose host preparation and isolated component packet now pass review. The component, owner and caller suites pass 28, 12 and 3 checks. Operation 215 then passed real-device compiler, exactness, graph and release gates on both cards. Its positive component timing is retained with T16 uncertainty; representative activations and production engagement remain outstanding. [OPT-05](MXFP4-OPT05-DRAFT-HEAD-SCREEN.md) is signed off as deferred pending selected-workload coverage and graph evidence. OPT-07 now retains prefetch plus cap32 as the immediate composition, with norm/quantizer fusion as a conditional third member. The composed attention component passed as operation 214 at 235 W. Its serving pilot remains held by the pre-registered route, control-work and economic evidence requirements; F1 now has a retained positive component disposition; serving qualification remains separate. The old C4 accumulator benefit remains recorded, but its resolved zero-prefix regression excludes the unchanged source. See the [complete reconciliation](MXFP4-OPT07-CUMULATIVE-PREPARATION.md). OPT-01 distinguishes the selected DFlash2 selector from the generic greedy sampler and keeps opt-in compact worker-cache activation open in OPT-05/06. This document schedules no GPU invocation, owner index, installation, publication or adoption. Each surviving physical packet still carries exact identities, one-use authority, existing protected observer/staging only, finite limits, no automatic retry and independent release.

Sources: sealed Exa research report (`RESULT.md`, private evidence), [current CPU screen](../provenance/MXFP4-CPU-LEAD-SCREEN-20261103.json), [operation212 result](../provenance/MXFP4-VERIFY-PREFETCH-SERVING-20261103.json), [validation workflow](../.agents/skills/aithreadripper-serving-validation/references/evidence-workflow.md).
