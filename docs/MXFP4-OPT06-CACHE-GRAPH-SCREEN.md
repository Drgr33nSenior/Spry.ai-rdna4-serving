# OPT-06 cache and graph CPU screen

OPT-06 is signed off after the Opus 5.5 review. The native cache-tree rewrite and whole-round graph implementation remain deferred pending evidence of a removable cost. This is a CPU exit, not GPU qualification or a rejection of either idea.

All 21 optimization references remain in the lead register. This sprint changes no runtime, selection, power setting or numerical gate. It uses existing evidence and local standard-library Python only.

## Cache retention

The analyzer reproduces all 20 previously retained conditional pool-model runs. These are the original combinations of path cap, completion-leaf assumption and C2 admission order; this sprint performs no new parameter search.

At cap 3, the model's cached-token outcomes match all 60 HTTP request outcomes from operation 204: 42 cold requests have zero cached tokens, six warm C1 requests have 118,784, and twelve warm C2 requests have 59,392. Operation 204 used two identical cap-3 arms. It qualified that experimental cache policy; it did not compare its speed against the selected unlimited-path policy or adopt it globally.

The cache rules explain why this distinction matters:

- A warm match requires both FULL and MAMBA device state.
- The match limit is prompt length minus one.
- Trimming preserves locked nodes, forks, device leaves and the current tail. Cap 3 is therefore a soft retention target, not a hard bound on all live states.
- Trimming a recurrent checkpoint does not erase the FULL KV component.

Tests execute the retained matching, prefix-limit and trimming functions through their AST with inert objects. A separate model checks namespace isolation, checkpoint locking and eviction. Neither supplies an exact replay of the native radix tree.

The consumed records contain no complete timestamped node/lock/eviction stream and no attributed cache-tree host self time. Cached-token agreement corroborates the capacity explanation, not the exact LRU history or a prospective Rust/C++ speedup. H4 remains deferred until those measurements justify a rewrite.

## Graph boundary

The analyzer identifies the draft and target graph groups in eight retained operation-183 decode windows using kernel correlation and source ordering. These are four sampled C1 windows on each rank, not a current 235 W serving baseline.

| Quantity | Median across eight rank/windows |
| --- | ---: |
| Draft graph completion to target graph start | 204.842 µs |
| Union of recorded kernel execution within that interval | 116.633 µs |
| Interval not covered by recorded kernels | 88.332 µs |
| Target graph API completion before draft graph completion | 5,358.608 µs |

Each median is computed independently; the median rows are not additive. Every window contains twenty intervening kernel dispatches. The target graph API returned before the draft graph finished in all eight windows.

This evidence does not establish late host submission as the bottleneck at this boundary. It also does not prove that every larger graph would be useless: an outer graph could change dispatch gaps or buffer handling, and post-target, cross-round and C2 costs remain unresolved. Time without a recorded kernel is unattributed; it is not automatically GPU idle, exposed CPU cost or removable latency. Long overlapping HIP API durations are not added to kernel time.

## State and lifetime contract

The retained decode path orders draft, target verification, acceptance, recurrent/convolution commit, optional committed-length publication, and draft-KV append. Host result processing then resolves token/acceptance arrays and request completion/cache actions. Publication happens before draft-KV append, so the next round cannot treat published length alone as proof that its draft inputs are ready.

The inert round model covers per-request generations, different C2 lengths and accept counts, TP agreement, zero/partial/full acceptance, commit before publication, draft-KV readiness, finite rounds, cancellation, failure and drain before reuse. Source-order tests and this model do not prove live tensor, stream, collective or numerical correctness.

A persistent grid would additionally need native occupancy and cross-rank progress evidence. The recorded API core count of 32 is not a residency proof. A model-free component cannot qualify an entire serving graph.

The compact draft index view remains off. Its model declares a window of 2,048; effective backend enforcement remains unverified. A shorter index view must not be reported as physical KV-pool savings. Recurrent dense-state consumers and flush/generation obligations remain part of the deferred design.

## Verification and next action

The final focused suite passes 16 tests. The analyzer verifies eleven input seals, fifteen copied source files and the data pins, reproduces the conditional cache outcomes, and reads the eight retained traces. Twelve source copies match selected-role pins; three cache/result dependencies come from retained root-cause evidence and are explicitly not a fresh native observation. Earlier fixture-development failures are retained.

No framework imports, native operations, model starts or GPU launches occurred. Shared runtime and public-preflight sources are unchanged; the earlier public43 receipt is not presented as a new run.

The private CONTRACTS.md defines the missing event fields, bounded capture requirements, TP/state/lifetime predicates and separate physical qualification. Merge those evidence requirements into the planned shared host diagnostic instead of spending a separate campaign on each hypothesis. The collector itself is not implemented by this sprint.

Next, reconcile the cumulative candidate register. Preserve the cap-32 attention signal and verification-prefetch evidence with their original verdicts and outstanding composition/route checks. Gated-norm/quantizer fusion remains a CPU compiler survivor that still needs device correctness and host integration. This audit neither adds a new bundle member nor removes those earlier leads.

## Review disclosures

The reviewer reran all 16 tests and reproduced the analysis. FULL KV retention during trimming is a source-reading claim: the test stubs the cascade. Device-only matching is supplied to the test; the selected route must establish that condition. Static ordering is conditional: greedy acceptance assumes all-greedy sampling and draft graph execution, while state commit and publication have their own guards.

The retained B0-pinned environ.py defaults TP sync to all at line 1393 (SHA256 2085881c2aa46abd90fc43af6b8f4672a0e3377f18eb2393d165a7a5d9e98a6d). The effective setting remains unobserved. The reviewer separately found all twenty eager-kernel launches completed before draft completion in every window. Rank-dependent gaps and negative same-stream gaps remain unexplained; these observations do not establish a speedup.

Before implementing M1, extend the abstract model to define in-flight cancellation, event-backed drain, collective sequence and output/prefix agreement, terminal trimming, generation checks on readers, capacity/page limits and C2 batch shrink. Pin the vocabulary bound. Before H4 work, re-pin the three retained dependencies from the selected image. A future analyzer must assert selected-role hashes against B0; this review checked them manually. These gaps do not invalidate the CPU deferral, but they prevent using its toy models as an implementation proof.

The original audit remains sealed. The separate opt-06-signoff-v1 record carries all nine review notes and this corrected public report.

Evidence: private optimization-continuation-v1/opt-06-cache-and-graph-cpu-v1; public digest record: provenance/MXFP4-OPT06-20261103.json.
