# OPT-07 shared diagnostic capture and readout

Status: **CPU review complete with parent corrections; owner integration remains.**

The shared diagnostic now has fixed-window capture control, bounded trace
export, HTTP count joins and cross-start work comparison. All six findings in
the [observer-core review](MXFP4-OPT07-SHARED-OBSERVER-CORE.md) are corrected.
This stage has no serving entrypoint or executable owner packet. It adds no
measured performance gain. Serving spend remains on hold.

## Core corrections

The writer retains its summary digest and returns it through hook restoration.
Capture-observer restoration now returns a receipt with both the core and
capture digests. A capture-file write failure returns the completed core pin,
an absent capture pin and an explicit incomplete status.
Readers must receive that digest from the authenticated owner receipt. They
must not manufacture a pin by hashing the input they are about to accept.
A foreign result cannot discard a pending selected batch. Closed observers
disable retained hook references. Rank joins require matching menu digests;
trace joins require the graph flag to agree with the phase. Both producer and
reader check contiguous prefill coverage before decode.

The metadata allocation is now 3.75 MiB per rank/start, including its summary
reserve. Four streams use 15 MiB. Another 256 KiB per stream remains reserved
for capture reports and other metadata, within the 16 MiB total. This corrects
the earlier eight-MiB-per-stream limit. The owner must account for every file
in the remaining reserve before packet preparation.

## Capture and joins

`shared_capture.py` implements seven fixed windows on start 2: one middle
prefill chunk for long-C1-cold, plus one two-round decode window in each of
the six cases, using repetition zero only. Decode capture starts after two completed rounds with
positive output progress for every member. Initial C2 singleton rounds do not
count as settling. Both C2 requests must stay active throughout capture.
Missing membership or a failed capture cannot select a later replacement.
After a capture failure, later pending windows are explicitly labelled
NOT_ATTEMPTED_AFTER_CAPTURE_FAILURE. An aborted window is stopped without
export or compression. Its annotation closes before profiler cleanup, and a
cleanup interrupt cannot replace an existing native exception.

Start 1 activates no profiler. Start 2 uses the existing Kineto interface and
adds no explicit device synchronization. Native profiler start/stop may
synchronize internally. Upstream PyTorch source supports this concern, but the
installed source has not yet been verified. Profiled latency is not scored.
Prompt hashing also adds CPU work, including work before the first recorded
host timestamp; observer overhead remains unmeasured.

The backend requires a fresh, dedicated 32 MiB tmpfs per rank before constructing
the profiler. It checks the mount table and live capacity. No mount was created
in this stage. CPU tests substitute the mount proof; they do not establish a
native filesystem bound. The existing compressed limits remain below 768 KiB
per prefill trace and below 320 KiB per decode trace, with a 3.5 MiB aggregate
limit per rank. Overflow retains an incomplete result without retry.

`shared_readout.py` replays window selection and joins writer-pinned capture
reports, trace files, annotations, graph correlations and the owner-supplied
worker/device identity. Its HTTP join checks native RIDs, request/input hashes,
visible output counts, cache hits, retractions and draft counters against the
existing HTTP summary. The existing owner source, wire, quality and independent
release validators must run first. These helpers do not replace them.

The HTTP join now also requires an independently owner-pinned receipt that binds
the summary to the start nonce, session, menu and control identity. Both ranks
must use the same summary and receipt; the two starts must use distinct HTTP
evidence. The owner receipt publisher remains to be integrated.

Cross-start comparison checks complete round inventories and response-token
digests. It does not align device clocks or establish that timings transfer
between starts. Readout functions consume the in-memory result of the core
reader; that result's integer-keyed count map is not a JSON transport contract.

The trace census reports duration sums separately from interval unions and
compares symbol durations across ranks without pairing kernel instances. The
comparison now consumes successful capture readouts for the same start and
fixed window, and requires distinct device UUIDs. The reader checks the capture
schema and extra capture files. The producer requires one directory for the
capture report and traces.
It reproduces all eight retained op183 decode traces, including rank-0 overlap
within a reported stream and the absence of that overlap on rank 1. It assigns
no cause, utilization, critical path or execution cost to those observations.

## Verification and limits

All **69 focused CPU tests pass** from disposable copies: 26 core, 15 trace,
18 capture and 10 readout tests. They cover native-call preservation, failure
and interrupt cleanup, fixed triggers, bounded export, digest and identity
refusals, HTTP mismatches and cross-start differences. Framework objects and
trace events in the integration fixtures are synthetic. No Torch, Triton or
SGLang import, SSH, native operation or GPU launch ran.
The added cases include multi-chunk prefill, initial C2 singleton rounds,
failure in the second captured round, interrupted cleanup, both-digest handoff
and reused or mismatched HTTP receipts. Removing the HTTP pairing guard or the
abort-path export skip in disposable copies causes the four targeted regression
tests to fail as expected.

The analyzer reproduces the retained trace census and two metadata planning
scenarios byte for byte. A padded estimate based on op204 is 3,228,668 bytes
per stream against the 3,932,160-byte journal cap. This assumes comparable
outputs and 2048-token prefill chunks except the final chunk. It is a planning
check, not proof of live fit. The prior 43-check public preflight is unchanged
evidence; it was not rerun and does not validate the new integration.

Opus reviewed the sealed v1 capture/readout implementation and accepted its CPU
scope with one medium HTTP-join defect and smaller findings. The parent applied
and tested the corrections in a fresh v2 root. The reviewed v1 roots remain
unchanged. A complete owner packet still needs its own review; these checks do
not qualify a physical diagnostic.

## Remaining owner-integration gates

1. Bind the real frozen menu, selected image, decorated scheduler callables,
   worker identity and installed profiler source. Verify installation timing
   and module/linecache restoration against the unchanged outer guard.
2. Add the declared runtime/environment inventory and identify the selected
   recurrent branch. Record F1 geometry and eligibility for all 48 layers on
   both ranks without attaching a candidate or reading extra GPU tensors.
3. Consume the raw tmpfs plan, publish producer digests through authenticated
   owner receipts, and connect the existing collector, HTTP validators and
   failure-aware independent release path. Exercise the complete consumed
   closure under the declared native Python.
4. Freeze two-start runtime, retention and closeout budgets. Review the full
   packet and request exact GPU approval only after those checks pass.

The bundle cost model remains **INCONCLUSIVE_HOLD**. The diagnostic is intended
to investigate host, collective, speculative and timestamp leads; changing
prefix lengths still prevent the current component calibration from proving a
request-level bundle gain. All 21 leads and all three cumulative members remain
retained. Model, precision, context, selection and power settings are unchanged.

The [provenance record](../provenance/MXFP4-OPT07-SHARED-CAPTURE-INTEGRATION-20261103.json)
binds this report to the implementation, tests and sealed evidence.
