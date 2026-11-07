# OPT-07 shared diagnostic observer core

Status: **Opus accepted the CPU core with six required fixes; parent corrections applied.**

Opus reviewed the sealed v1 core. Its six findings are corrected and tested in
the [capture integration](MXFP4-OPT07-SHARED-CAPTURE-INTEGRATION.md), which also
adds fixed-window control and further joins. The v1 evidence remains unchanged.

This stage adds the observation and file-reading core for the control-only
shared diagnostic. It does not create an executable serving packet. No model
start, GPU launch, SSH operation, power change or selection change ran.

The [request cost model](MXFP4-OPT07-REQUEST-COST-MODEL.md) remains
INCONCLUSIVE_HOLD. The diagnostic is justified by host scheduling, collectives,
speculative phases, source inventory and the timestamp question. This stage
neither changes the bundle spending rule nor claims another performance gain.

## Implemented boundary

`shared_diagnostic.py` wraps the selected non-overlap scheduler's `run_batch`
and `process_batch_result` methods. It calls each original method once and
returns the original result. It records request membership, CPU output counts,
existing CPU draft counters, host prefill extents and four host timestamps.
Decode prefix lengths remain unknown; the observer does not fetch a device
value to fill that gap.

The caller supplies the exact expected original callables, a worker/session
identity and the frozen menu. The helper checks their identities but does not
authenticate the installed serving image. That binding is still an owner task.
Prompt IDs are hashed on first observation and are not retained. Request IDs,
cache salts and counts remain private evidence. Observer errors disable the
observer without skipping or repeating native work. Native exceptions and
interrupts propagate. Scope exit restores owned hooks; a foreign replacement
is left intact and makes the observation incomplete.

Each rank/start writes exclusive JSONL chunks with a one-MiB chunk limit, a
64-KiB event/report limit and a 3.75-MiB journal budget including report reserve.
The four streams reserve another 256 KiB each for other metadata, keeping the
total within 16 MiB. Rotation can create up to four chunks. The reader authenticates the
summary digest supplied by its caller, each chunk, exact file membership,
request continuity and the session identity. A COMPLETE result means only that
this metadata stream completed and its hooks were restored. It does not mean
that HTTP quality, profiling, independent release or the diagnostic passed.

`shared_trace_io.py` exports gzip streams with the inherited strict compressed
limits: below 768 KiB for prefill and below 320 KiB for decode. It enforces a
3.5-MiB compressed budget per rank and a 32-MiB expanded-file bound. Overflow
leaves a bounded, incomplete export. An attempt marker prevents repeating the
same window. The reader refuses truncated gzip data, trailing members, changed
bytes and oversized expansion.

The raw-file check is **not a producer-side filesystem limit**. The module
provides an inert plan for two sized raw tmpfs mounts. The owner has not consumed
that plan, so every export explicitly reports
`producer_hard_bound_verified: false`.

The annotation reader joins one prefill marker or two adjacent decode markers
to observed request rounds, then to same-process graph-launch API records and
their kernel correlations. It refuses foreign markers, changed C2 membership,
missing or ambiguous correlations and graph launches outside the fixed markers.
Its verdict is JOINED_CPU_GRAPH_API_ONLY. Kernel timestamps need not fall inside
the CPU annotation. The join proves neither GPU dependencies nor critical-path
cost. It must not be used to explain the historical rank-0 timestamp anomalies.

## Source and CPU checks

The source audit authenticates 11 retained files. It extracts the native
`_resolve_spec_v2_tokens` function and runs it with CPU stubs, confirming that
native processing publishes the draft counters consumed by the observer. It
also extracts the recurrent fold-admission and cuTeDSL predicates. FP32 temporal
state rules out the latter branch under the tested source conditions. The live
pool fields and the selected branch remain unobserved.

The result-processor dependency must still be authenticated against the selected
image. Extracted functions have their decorators removed. These checks execute
neither a real scheduler nor a framework and do not validate device transfer or
synchronization behavior.

The original **35 focused tests passed** from disposable copies. They cover completed and
incomplete journals, native-call preservation, interruption and restoration,
foreign hooks, byte limits, request/rank mismatches and trace ambiguity. Two
mutation probes fail when the summary-digest check or the unowned-graph check
is removed. The public preflight passed all **43 checks**. Its full output was
observed in the tool session; the evidence retains the final output chunk and a
receipt that discloses this partial log retention.

Fixtures use short synthetic prompts and controlled CPU counters. They do not
prove production menu coverage, profiler output schema, overhead, filesystem
behavior in the owner sandbox or finite runtime fit.

## Work remaining at the v1 boundary

This list records the original boundary. The linked integration report records
which items now have CPU implementations and which still block an owner packet.

1. Bind the real menu, selected image, worker/PID identity and loaded callables.
   Recheck the result-processor dependency and restore the module/linecache
   scope before the unchanged outer guard runs.
2. Implement the fixed seven-window controller: two completed decode rounds
   after two committed rounds, both C2 requests active, one attempt per window,
   no substitution. The annotation reader does not enforce the settling trigger.
3. Integrate the profiler lifecycle and the two raw tmpfs limits without adding
   a device read or synchronization solely for observation. Establish supported
   annotation/correlation fields from the pinned profiler implementation.
4. Add both-rank runtime/environment inventory and recurrent-branch/F1 eligibility
   witnesses. The current core does not attach a candidate or observe GDN tensors.
5. Join final output counts to authenticated HTTP results. Add cross-rank duration
   comparisons and timestamp-overlap checks without aligning rank clocks or
   treating kernel start order as causality.
   Compare both starts' full work inventories before any proposed transfer.
6. Measure or conservatively bound metadata volume, observer cost, capture/decode
   duration and closeout within the finite owner envelope. Integrate failed
   closeout, collection and independent release, then prepare an exact decision.

All 21 leads and the three cumulative members remain retained. Their
classifications and the serving-spend hold are unchanged. The new core is not
GPU-ready and creates no execution authority.
