# Sprint 8: serving cost attribution

Status: the corrected native campaign passed. All eight final answers passed,
and 126 completed timing samples per card authenticated. Independent release
passed. Sprint 8 selects the target verification graph for deeper attribution
in Sprint 9. The selected runtime and model configuration are unchanged; this
instrumented screen establishes no serving speedup.

The [sprint list](SERVING-SPRINTS.md#optimization-sprints-810) divides the
[reviewed plan](NEXT-OPTIMIZATION-PLAN.md) into attribution, one optimization
and matched serving validation. The existing numerical mismatch remains
unresolved under the owner's experimental-session disposition.

## 8.1: host recorder and sampling observer

The [recorder](../prototype/serving/sampling_host_timing.py) retains host elapsed
intervals before native sampled acceptance, inside that native call and after
its return. Native-call elapsed time can include device waits. It is not GPU
execution time or CPU consumption. The intervals include timing bookkeeping
and exclude subsequent record assembly; they are instrumented observations.

A fresh private derivative composes it with the source-authenticated
`SamplingRoute`. The route still validates and hashes the complete prompt on
every call, delegates native acceptance once and emits the same successful
route witness. The normal installer still passes no recorder. The candidate
adds no tensor operations, model forwards, synchronization or RNG calls.

The recorder binds each request's body SHA256, input-ID SHA256 and cache salt
to the existing 32-cycle schedule. Four tasks share a maximum of 128 retained
rows per rank. The 36864-call observation limit remains separate. Readiness
traffic cannot consume external-request samples; another request ID cannot
silently replay a frozen task. Snapshots retain failures, missing indices,
actual host prefix ranges and uncovered tails. They do not establish a
whole-request bottleneck.

The canonical class lives in this repository. The private preparation script
embeds that class into the existing helper to avoid adding an archive member
or source-binding role. It checks class equality and limits changes to the
existing `SamplingRoute` definition. Native authentication, head/gather startup,
BF16-cache startup and the default installer retain their previous bodies.
The complete packet and its transport checks are recorded in milestone 8.2.

Run the public CPU suite with `make test-sampling-host-timing`. The private
integration suite exercises the actual modified route with declared CPU
fixtures. It also runs the prior route authentication and control-mode cases.
These checks do not execute an enabled native model startup or qualify graphs.
Source pins, counts and review are in the
[preparation record](../provenance/SPRINT8-ATTRIBUTION-PREPARATION-20261022.json).

## 8.2: device integration and prepared packet

The [event recorder](../prototype/serving/device_phase_timing.py) keeps separate
backends, shape buckets and replay generations for draft and target graphs.
Graph event objects initialize before capture. Eager event allocation now waits
until capture finishes on the selected device. Capture-time hooks restore after
startup; captured events execute on every replay. A fresh outer completion
event follows each retained replay. Before any later event reuse, the observer
collects the sample or marks it incomplete, including on unsampled calls.
No per-cycle synchronization or extra model forward is added.

The packet measures draft graph, candidate construction, projection, local
k=16 selection and selector walk; target graph, output head and gather; and
eager acceptance, target probabilities, recurrent commit, draft-KV append and
next-input preparation. Candidate construction includes the small candidate
exchange and merge. Remaining draft-graph time is not pure backbone time.

Target probabilities form one measured group: scaling, global k=20 selection,
top-p/softmax and dense allocation/scatter. The observer wraps the existing
DSpark helper alias after authenticating its exact function and source. It
does not patch Torch globally or time k=20 alone. Finer attribution follows
only if this group warrants it. Preserve the sampler's normalization, cleanup
and RNG behavior.

The existing host recorder binds the frozen request hash and prefix to each
cycle. The producer requires one draft replay, one target replay, native
sampling and one completed commit. The
[readout helper](../prototype/serving/phase_timing_readout.py) validates individual
pending/collected/host joins and interval bounds after the campaign reader has
authenticated the raw logs, workers and host summary. It does not replace
campaign authentication or prove aggregate coverage.

Initialization and cleanup are transactional. Collection or reporting failures
cannot prevent attempts to restore the owned hooks. Native exceptions retain
their identity. The actual enabled worker path, head/gather startup and required
head/cache witness calls passed CPU fixtures. Device identity, native classes,
CUDA and external scheduler effects are declared substitutes in those tests;
they do not prove native startup.

All 104 focused CPU checks passed: 15 event/readout, 11 host regression, nine
runtime/startup and 69 package/retained-component checks. Run the public suites
with `make test-device-phase-timing test-sampling-host-timing`. Astra independently
verified the final source, packet, tested hashes and all eight protected helpers.
The selected numerical source files remain unchanged. Earlier preparation and
fixture failures remain in separate private roots.

| Prepared transport | Actual | Ceiling |
| --- | ---: | ---: |
| Archive members | 64 | 64 |
| Archive bytes | 16711680 | 16777216 |
| Combined transport bytes | 33514674 | 33554432 |
| Argv entries, each arm | 255 | 256 |
| Binding bytes, control/timing | 65411 / 65391 | 65536 each |

The control binding has 125 bytes of remaining capacity. Rebuild and recheck
all limits after any packet edit. Both arms select snapshot mode with FP8
weights and BF16 target/draft KV. The inherited `bf16`/`kv4` owner labels do not
indicate a cache-precision comparison. The second arm adds timing only.

The owner authorized the agent to start the staged, hash-verified launcher.
The first campaign and its failed timing arm are retained in milestone 8.3.
The initial preparation remains immutable. See the
[sealed device preparation record](../provenance/SPRINT8-DEVICE-PREPARATION-20261022.json).

## 8.3: measurement gate

Use the existing sampled four-task recipe and the selected snapshot path in
both starts: observer control, then phase timing. Keep low thinking and natural
stops. The owner ceiling remains 5400 seconds, with 1800-second worker and
1740-second client limits. The previous packet's roughly 6–10-minute estimate
does not replace those limits.

Authenticate source, workers, precision, HTTP, counters and independent release.
Grade all eight final answers. Report timing overhead, actual cycle/prefix
coverage and every output difference. Keep nested intervals, ranks, sampled
and greedy profiles distinct. Select Sprint 9 only from enough completed
samples; retain incomplete attribution explicitly.

### First native attempt and correction

The two-start campaign stopped after 383.81 seconds. The control arm produced
four responses; they have not been graded. The timing arm completed startup and
failed on rank 1 at selected cycle 9 with `RuntimeError: Event device`. No valid
completed phase samples were collected. This campaign cannot establish phase
costs, observer overhead or task-quality equivalence.

The event recorder constructed and recorded its eager events before the native
model runner selected the worker GPU. Its graph events initialized later during
capture. At the first selected sample, rank 1 attempted to record an eager event
bound to the earlier device. A CPU fixture that enforces first-record device
binding reproduces the retained error with the old class and passes with the
correction.

The correction defers eager event allocation until capture finishes, pins the
device selected during graph preparation and rejects later device changes.
All 38 focused checks passed: 12 event, five phase-join, 11 host and 10 runtime
checks. The runtime suite includes the actual observer/head/startup composition
with a simulated rank-1 device transition. These are CPU substitutes. Astra
reviewed the correction and independently passed the 12 event tests; native
validation remains pending.

The sealed failure analysis verified source and stream hashes, protected
closeout metadata and independent release. No owned processes or resource
scopes survived. Successful-campaign readers remain blocked for this run. Their
separate 10-check offline suite and review do not turn a partial campaign into
a benchmark. The known numerical mismatch remains unresolved.

See the [failure and correction record](../provenance/SPRINT8-NATIVE-FAILURE-20261022.json).
The next bounded step is to build and review a fresh one-use packet, recheck its
transport limits and validate the correction on both GPUs. At the first closeout, no automatic retry
was launched and no replacement packet was prepared. The owner subsequently
authorized the corrected campaign below.

### Corrected native result, 22 October 2026

The fresh two-start packet passed 107 focused CPU checks and Astra's package
review, then completed in 383.88 seconds. Both arms kept the selected snapshot,
FP8 weights, BF16 target/draft KV, FP32 head/state, current flags, metadata reuse
and DFlash8. Thinking remained enabled at low effort, with the frozen publisher
sampling recipe and natural stops. The corrected eager events ran successfully
on both GPUs. Source, runtime, worker, cache, HTTP, counters and independent
release checks passed. No persistent server or installed default changed.

All eight final answers passed their task and format checks. The ledger final
segments contained 763 and 821 tokens, above the 512-token requirement. All four
final-answer texts matched across arms. Full output IDs matched three pairs;
the held-out ledger had 2529 versus 2521 reasoning tokens and 450 versus 448
verifier cycles. Preserve that variation and the unresolved numerical contract.

Each rank retained 126 completed samples: 32, 32, 30 and 32 by request. No
retained sample was not-ready or left uncollected. The threshold ledger stopped
at cycle 341, leaving scheduled cycles 344 and 356 unreached. Its final retained
index was 333, so its last eight actual cycles are also unmeasured. The producer
reports zero `uncovered_tail_cycles` here because that field counts only cycles
beyond the frozen schedule, not beyond the last retained sample. The held-out ledger
continued through cycle 448, leaving 63 cycles after the final sampled index
385 uncovered. Cycles 1–8 and intervals between scheduled samples remain outside
the measurement. The retained record gives every cycle and prefix range.

The following medians use 62 completed samples from the two long requests on
each card. They describe instrumented windows, not whole-request shares.

| Interval, milliseconds | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Complete verifier cycle | 54.063 | 54.090 |
| Target verification graph | 44.850 | 44.852 |
| Target graph after subtracting head and final gather | 42.641 | 42.642 |
| Target output head, nested in target graph | 2.051 | 2.051 |
| Final vocabulary gather, nested in target graph | 0.157 | 0.157 |
| Draft graph | 7.281 | 7.182 |
| Draft vocabulary projection, nested in draft graph | 2.131 | 2.131 |
| Draft local top-k=16, nested in draft graph | 0.068 | 0.071 |
| Acceptance | 0.349 | 0.354 |
| Target probability group, nested in acceptance | 0.229 | 0.226 |
| Recurrent commit | 0.298 | 0.347 |
| Draft-KV append | 0.627 | 0.617 |

The target graph spans a median 83.04%/83.02% of the sampled cycle interval.
Its remainder includes projections, quantization, GDN, full attention, layer
collectives and possible waits. It is not a pure compute or removable-cost
measurement. Keep nested intervals and ranks separate; do not add table rows.

The three equal-ID request pairs changed latency by -0.086%, +0.103% and
+0.221%. The held-out pair changed by -0.324%, with fewer reasoning tokens and
cycles in the timing arm. These are descriptive one-order observations. They do
not isolate instrumentation overhead or establish a model speedup. The host
observer's before-native median on the long requests was about 0.52–0.54 ms;
that diagnostic work remains enabled in the selected observer path.

Sprint 9 starts by classifying the target verification graph. Check the existing
installed trace tool's version and graph support, then prepare a reviewed trace
of at most eight complete verifier cycles on both ranks with frozen artifact
and time limits. Use selective events if compatible tracing is unavailable.
No trace has launched and no new kernel is selected. Draft local top-k and target
probability scratch are not the first optimization targets from this screen.
See the [authenticated result](../provenance/SPRINT8-COST-ATTRIBUTION-20261022.json).
