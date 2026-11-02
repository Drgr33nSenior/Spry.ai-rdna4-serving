# Preserve draft attention during captured-map preparation

Operation 205 stopped before readiness or any HTTP request. Target verification
graph capture completed on both ranks, then draft graph capture raised
`ValueError: captured verify metadata is incompatible`. The metadata hook patches
the native backend class, which both the target and draft use. The draft graph
also uses target-verify forward mode, so that mode alone cannot identify the
backend that the split-decode adapter supports.

The pinned target uses 12 query heads and two KV heads per rank. The pinned
DFlash2 draft uses 16 and four, with a different head dimension. The correction
applies strict captured-map admission only to the target head geometry already
required by the adapter. Other head geometries keep native initialization and
attention. Their exceptions still propagate. The target keeps its captured view
per backend and batch size, and still refuses missing or incompatible metadata.
No response acceptance rule, kernel, device read or dependency changes.

The earlier adapter remains versioned. Tests retain its draft-capture refusal
and exercise the corrected draft capture, replay and graph-forward path at both
batch sizes. A draft instance of the same patched class leaves the target's
captured map intact. The nine existing captured-view tests also run against the
correction, including failed recapture and preparation.

Operation 205's original caller returned its failure receipt. Independent GPU
release passed, and checkpoint 561 is parked. It produced zero responses and was
ineligible for CPU grading. No retry ran. See the
[retained result](../../measurements/native-qualification/GPU-205-RESULT.json).

The next instrumented candidate is `e2011102000000a1`. All 1,070 public tests,
43 boundary tests, six exact packet/caller checks and 48 native CPython 3.14.7
source-only checks pass. The native checks include the actual pinned initializer
through the target and draft paths, both complete archives and their transports,
worker materialization, draft preflight and startup producer/reader. Tensor,
fill-helper and native admission substitutes are explicit. The checks deny GPU
devices, native serving imports, network and subprocesses within native Python.
Both final author packages and approval requests reproduce byte-for-byte.

Only the adapter differs from operation 205. All ten protected entries, the
other 60 Python files, five tensor-storage sources, settings and 176 frozen
requests remain identical. Earlier failed preparation and inert-baseline source
checks are retained in [verification](VERIFICATION.json). The final baseline
uses the retained startup-corrected source; the candidate bytes did not change.

The candidate needs fresh exact approval. Its paired baseline is an inert source
artifact, with no automatic execution. Native response recovery, full-cleanup
qualification, production measurement and final closeout remain open. No GPU
memory or performance gain has been established.
