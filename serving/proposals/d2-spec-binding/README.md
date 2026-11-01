# D2 proposal: verify the configured W5 spec

Status: **isolated protected-source proposal; not consumed by the builder**.
This extends the [cleaned inventory proposal](../d2-cleaned-inventory/README.md).
[D2.patch](D2.patch) replaces the controller's global `SPEC_SHA256` dependency
with verification of the complete canonical spec and explicit hash arguments.
[PROPOSAL.json](PROPOSAL.json) pins the controller, patch and composed protected
set. The transport and other eight protected helpers retain their preceding
bytes. The package still has 80 archive members.

## Reproduction rule

The verifier authenticates the independently approved run, host, menu and
protected set. It reads the spec and its six referenced source/template files
through bounded, no-follow readers, checks their package hashes, and retains
file and directory stamps for subsequent rechecks.

The spec has 19 configurable/source-bound slots:

| Spec fields | Source |
| --- | --- |
| `worker_seconds` | `client_seconds + worker_startup_seconds` |
| `context_length`, `physical_slots` | Run profile context length |
| `gpu_ids`, expected devices | Host rank indices and device map |
| HTTP `client_seconds` | Run client deadline |
| Seven profile fields in `server_args` | Run profile |
| Two HTTP template hashes | Authenticated package members |
| Four graph helper hashes | Authenticated package members |

The verifier replaces these slots with `null` and checks the remainder against
a protected template fingerprint derived from the sealed W5 spec. This binds
every other field, including paths, safety flags and native-source references.
It fills the slots from configuration and authenticated package hashes, then
compares the complete canonical result with the spec's approved hash. Unknown,
missing, noncanonical or altered fields cannot pass by recomputing package
hashes. Boolean/integer aliases also refuse.

The supported run and `FINAL-SETTINGS.json` remain pinned to the W5 baseline.
This is a compatibility boundary while runtime consumers still contain literal
limits and settings. A reapproved configuration mutation still refuses; this
proposal does not complete the general configuration-mutation gate. The owner
ceilings, stop policy and device-observation checks remain unchanged.

Grant validation and actual pre-launch use this verifier. The physical entry
verifies the spec before creating results or constructing the lease. Resource
receipts, campaign arguments, HTTP bindings and artifact parsing use the
verified hash explicitly. Input changes during or after verification refuse.
No controller global is patched or populated with the generated hash.

## CPU verification

Public tests reproduce the original sealed spec byte for byte and exercise
source substitutions, changed configuration, altered specs with recomputed
hashes, strict JSON, deadlines and input-change detection. Private checks load
the complete proposed controller from authenticated fresh copies.

The private pre-launch comparison uses the retained external-host substitutes.
The configured controller passes spec reproduction and reaches the same absent
workstation runtime build proof as the original W5 controller. This is not a
complete native pre-launch. Wrong specs refuse before that boundary and before
lease acquisition.

The retained CPU producer needed a small API adapter: its five reads of the old
constant now hash the verified spec, and its HTTP-binding call supplies that
hash explicitly. [FIXTURE-ADAPTER.patch](FIXTURE-ADAPTER.patch) records that diff.
The complete controller verifies configuration before replay and rechecks its
inputs afterward. Controller functions and attributes are unchanged during
post-run tests. Native/process observations remain CPU substitutes.

```sh
python3 -I -B -S serving/tests/test_spec_binding_proposal.py
python3 -I -B -S serving/tests/check_spec_binding_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

Logs are under `.test-build/serving-cleanup/spec-binding/`. Remaining work
includes configurable runtime consumers, fresh packet identity, per-row stop
policy, device binding, instrumented health-receipt composition, all-cap and
doubled-menu capacity, native validation and private publication. The source
builder continues to emit the original nine protected helpers.

[VERIFICATION.json](VERIFICATION.json) records eleven public tests, nine private
checks and the full `make test` result, including 178 focused tests. Both 176-row
arms pass expected-response and retained-envelope replay. Length stops,
oversized members and changed source files refuse. The minimum measured margin
remains 1.59 times, below the required two times.

The first refusal loop reused a fixture-modified controller; each case now
loads a fresh one. A subsequent full run was interrupted after test discovery
also selected an imported legacy TestCase. The import is corrected and the final
nine-check run passes. Earlier logs remain intact.
