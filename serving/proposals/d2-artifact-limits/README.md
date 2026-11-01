# D2 proposal: configured HTTP artifact limits

This proposal extends [worker timing](../d2-worker-timing/README.md) with an
immutable `HTTPArtifactLimits` plan. [D2.patch](D2.patch) changes the protected
artifact helper and controller. [PROPOSAL.json](PROPOSAL.json) pins the exact
sources. The active builder still emits the original nine protected helpers.

The plan validates 16 primary byte and file partitions. Exact positive integers
are required. Every value must remain within its existing protected ceiling;
derived values cannot be supplied. The encoded payload must fit the stream
budget, and durable partitions must fit the session budget. The arithmetic
matches `spry_common.limits.derive` for the retained W5 configuration.

The actual pack, stage, export, receive and pipe-capture APIs accept the plan.
They enforce tree file counts, per-member bytes, aggregate bytes and stream
limits. A corrupted plan or oversized input refuses before staging or receive
output is created. Existing callers retain the baseline policy and wire bytes.
The controller derives the plan from authenticated configuration and passes
it to physical capture and receive. No owner ceiling is raised.

This is a consumer migration. Binding changed limits into worker export,
readout and all outer durable partitions remains open. The pinned whole-run
configuration still refuses mutations. Instrumented/raw-frame transport has
not been qualified with changed limits. Do not infer launch readiness from
the presence of these APIs.

## Verification

Public tests exercise strict types and ceilings, immutable snapshots, bounded
trees, actual pack/export/receive round trips, early refusal and actual pipe
overflow. Private checks run the complete authenticated CPU composition,
including both 176-row response replays and the unchanged release validators.

The in-memory HTTP fixture now holds each concurrent response until both real
client threads enter `getresponse`. This removes an OS scheduling race in the
stand-in. Client clocks, recorded evidence and the production overlap validator
are unchanged. Earlier failed attempts remain in the log directory.

```sh
python3 -I -B -S serving/tests/test_artifact_limits.py -v
python3 -I -B -S serving/tests/check_artifact_limits_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
make test
```

[VERIFICATION.json](VERIFICATION.json) records exact source and log hashes.
Logs are under `.test-build/serving-cleanup/artifact-limits/`. No native serving
or GPU operation is performed. The measured file-count margin remains below
the required two times. No inference performance gain has been measured.
