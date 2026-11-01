# D2 proposal: configured worker and client timing

This increment connects the active W5 worker and HTTP client to timing values
in the authenticated spec. It extends the
[session-timing proposal](../d2-session-timing/README.md). This is configuration
cleanup; no inference throughput or latency improvement has been measured.

[SOURCE.patch](SOURCE.patch) records the canonical `model_benchmark.py` and
`serving_runtime.py` changes, which the source builder consumes.
[D2.patch](D2.patch) contains the separate two-line protected controller change.
It binds the client duration to the verified session plan instead of the phase
table. The builder still emits the original nine protected helpers.
[PROPOSAL.json](PROPOSAL.json) pins both source changes and the proposed
protected set. Complete D2 review and native launch remain separate gates.

## Behavior

The baseline keeps a 2,280-second worker budget and a 2,040-second client
budget. Both consumers require exact integers, at least one client second,
at least 240 seconds between client and worker budgets, and the original
ceilings. Invalid values refuse before runtime preparation.

The entry point snapshots the validated worker budget before loading runtime
helpers. The runtime snapshots the client budget before preparing the native
entry. After readiness checks, the client receives the earlier of the worker
deadline and readiness time plus the configured client budget. If readiness
uses the remaining worker time, execution refuses before creating the client
or writing a readiness callback record. Socket timeouts continue to use that
same absolute deadline.

The HTTP result and evidence schemas are unchanged. Readout continues to bind
evidence to authenticated source and spec hashes. The controller's separate
legacy diagnostic check still uses its original worker literal; the active
HTTP path does not call it. Its removal belongs to later compatibility cleanup.

## Verification and limits

Public tests cover baseline, shorter and minimum budgets; strict types,
reserves and ceilings; snapshot behavior; expired readiness; socket expiry;
and the existing result/export behavior. Private checks authenticate all 313
donor members, build fresh copies and block native imports. The actual HTTP
import chain accepts baseline and shorter timing for both arms. Native
readiness, clocks and HTTP request execution use CPU substitutes; source
bytes and the actual client socket-deadline method remain unchanged.

The complete proposed controller also runs both 176-row response replays,
including export, receive, readout and refusal of length stops. The inherited
owner, transport, archive and seal checks run against the new composition.
Contract checks rebuild the canonical source and compare deterministic output,
source bindings, active overlays, resource arguments and retained entry behavior.

```sh
python3 -I -B -S serving/tests/test_worker_timing.py
python3 -I -B -S serving/tests/check_worker_timing_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
python3 -I -B -S serving/tests/check_contract_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
make test
```

Logs are under `.test-build/serving-cleanup/worker-timing/`.
[VERIFICATION.json](VERIFICATION.json) records the observed results and exact
source hashes. No GPU or native serving run has occurred.

Shorter plans are supported at the tested consumer boundaries. Whole-run
configuration mutations still refuse under the pinned W5 configuration.
Artifact limits, per-row stop policy, observed device/rank binding and fresh
packet identity remain open. The archive remains at 80 members. The measured
file-count margin remains 1.59 times, below the required two times; instrumented
health receipts and all-cap/doubled-menu evidence still need validation.
