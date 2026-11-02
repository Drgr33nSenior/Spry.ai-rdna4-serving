# Protected configuration migration map

Status: preparation for D1/D2. This document maps the remaining migration.
The [first isolated D1 proposal](d1-controller-inventory/README.md) contains an
exact deletion-only patch. It is not the complete D1 or a request for approval.
The [first D2 proposal](d2-config-binding/README.md) adds configuration approval
bindings. The [transport extension](d2-transport-binding/README.md) adds matching
approval arguments, archive authentication, staging and grant fields. These
proposals do not complete the migration.
The [D1 package proposal](d1-package-pruning/README.md) removes five inactive
members and supplies a separate composition with D2. The retained package then
fits with configuration. The [cleaned inventory proposal](d2-cleaned-inventory/README.md)
removes three more legacy/authoring names and registers the four extracted
helpers. The complete cleaned test composition fits at 80 archive members.
The [spec-binding proposal](d2-spec-binding/README.md) now verifies complete
canonical spec reproduction for the pinned W5 configuration and passes its hash
explicitly through the controller. The [owner-timing proposal](d2-owner-timing/README.md)
passes a validated timing plan from that verifier to the owner adapter, under
the existing ceilings and cleanup reserves.
The [session-timing proposal](d2-session-timing/README.md) supplies a shared
runtime timing plan to the transport and controller, binds manifest/grant/seal
totals, and retains the original receipt for all deadline calculations.
The [worker-timing proposal](d2-worker-timing/README.md) connects the active
worker and client to the authenticated spec and binds the controller's client
check to that session plan. It preserves the HTTP evidence schema and existing
time ceilings. These changes do not establish an inference performance gain.
The [artifact-limits proposal](d2-artifact-limits/README.md) validates configured
HTTP partitions and applies them to the pack/export/receive/capture APIs. The
controller supplies its authenticated plan to capture and receive. Worker
export configuration, readout and outer durable partition binding remain open.
The [stop-policy proposal](d2-stop-policy/README.md) unifies completion acceptance
across the client, reader, summaries and protected report. The retained quality
packet still refuses cap work and needs a matching menu before cap runs can be
enabled. The [host-binding proposal](d2-host-binding/README.md) validates observed
rank/BDF/UUID/render identity in both serving modes and uses authenticated host
devices in the controller.
The [artifact-binding proposal](d2-artifact-binding/README.md) projects the
verified primaries into the spec and supplies one immutable plan to worker
export and readout. The [power observer](../launcher/README.md) reads the pinned
host map and cap. The [durable-limits proposal](d2-durable-limits/README.md) now
accounts for both stopped evidence trees, the two manifests and outside
closeouts under the configured partitions. Its source-copy margin is about
1.73 times, below the required two-times gate. The [quality source](../quality/README.md)
projects explicit natural/cap menus and excludes cap work from grading; native
grading packet assembly and wider run support remain open. Fresh caller assembly
also remains open. The [boot-approval proposal](d2-approval-boot/README.md) now
supports boot-independent schema 3 manifests with an explicit observed boot
check before workload creation. The [review renderer](../launcher/PROJECTION.md)
prepares fresh manifest and caller source templates. It does not emit accepted
launch inputs; caller activation remains pending.
The nine protected files consumed by the builder remain unchanged.
Their authenticated donor hashes are in [SOURCE.json](../SOURCE.json), under
`protected_source_files`.

The owner approved D1 and D2 in principle in
`OWNER-CLEANUP-DECISIONS-20261031-v1.json`. That decision requires exact owner
approval of the produced diffs and ceilings. Prepare and test those diffs before
requesting that approval.

The owner subsequently approved the presented controller configuration-binding
proposal and continued implementation. The [approval record](OWNER-APPROVAL-20261101.json)
pins that proposal. The new transport diff and the complete migration remain
separate review work; no source proposal authorizes native execution.

## D1: removal only

Use the retained `cap_rules_protected.py` classifications to locate candidates.
Trace each candidate through the W5 controller, adapter, export and readout paths
before removing it. The current source bundle still contains compatibility
readers and legacy package allowlists; a static unused-mode label alone does not
prove that an allowlist member can be removed.

The D1 diff must contain removals only. Update the proposed package inventory
separately, and test the real W5 pre-launch and post-run paths with that inventory.
Do not combine config loading, changed ceilings or stop-policy changes with D1.

The package proposal deletes five source lines and omits five files: the old
capture runner, divergence and prefill-row helpers, and two historical review
build proofs. Both W5 arms retain the combined runner and identical overlay
hashes and sandbox arguments. This applies to the retained `both` draft-head
selection; it does not qualify other historical runner selections.

## D2: configuration consumers

| Current consumer | Configuration or derived value | Required preservation |
| --- | --- | --- |
| `benchmark_physical_controller.SPEC_SHA256`, `validate_manifest`, grant checks | Canonical run hash, host hash, protected-set hash, and reproduced spec bytes | Wrong config, host, helper or generated spec must refuse before launch. A hash alone must not replace generation checks. |
| `benchmark_host_transport.TOTAL_SECONDS`, `SEAL_RESERVE` | Derived `total_seconds`; primary `seal_reserve_seconds` | Keep the owner maximum as a protected literal. A config cannot raise it. |
| `benchmark_owner_adapter.INNER_WORK`, `ARM_WORK`, `ARM_ENVELOPE`, `run_campaign` | Derived worker, arm-work and arm-envelope seconds | Preserve cleanup, reaping, locked admission and collection deadlines. |
| Controller admission, preparation, active and seal deadlines | Named primary reserves and derived active/total seconds | Preserve receipt-to-seal accounting and stop-on-failure behavior. |
| Active model entry and HTTP client callback | Spec `worker_seconds` and `http_serving.client_seconds` | Keep worker/client ceilings, startup reserve and the shared absolute deadline. Bind controller validation to the verified session plan. |
| `benchmark_artifacts.HTTP_MEMBER_LIMITS`, `HTTP_STREAM_LIMIT`, `HTTP_SESSION_MAX_BYTES` | Primary member/stream partitions and derived durable bytes | Keep owner ceilings in protected code and preserve oversized-member refusal. |
| Device checks in controller, adapter, collector and GPU identity helper | Host `devices` and `rank_indices` | Bind observed HIP indices and the exact BDF/UUID/render set. Do not infer a BDF-to-rank assignment from sorted JSON keys. |
| Caller `power_observe.CAP` (outside the nine-file set) | Host `power_cap_watts`, converted to microwatts | Preserve observation-only behavior and the owner's cap ceiling. Include this caller change in review. |
| Controller `_http_wire_reference` | Each menu row's `stop_policy` | Preserve natural-row refusal of length stops. Define and test the exact cap-row result rule before enabling cap rows. |

The shared arithmetic is implemented in
[limits.py](../spry_common/limits.py). With the retained primaries it reproduces
worker 2280, arm work 2430, arm envelope 2490, active 6060, total 6360 and caller
7160 seconds. It also reproduces the protected exporter's 496.5 MiB durable
partition total under its 512 MiB ceiling.

The 390-second `campaign_margin_seconds` is an accounting remainder:
`6060 - (2 × 2490 + 180 + 105 + 75 + 330)`. The named stage values come from the
retained controller; the remainder preserves its existing envelope. It is not
a measured stage duration or extra permission to extend a run.

## Evidence available and required

The current tests prove canonical config pins, strict schema/type checks,
deterministic builds, identical configured/unconfigured review bytes, unchanged
protected sources, and equality
with the retained exporter's byte arithmetic. Changed profile, host, menu,
policy, instrumentation and time inputs refuse under the pinned W5 baseline.
This refusal is a temporary compatibility boundary, not the design's T3 test
of supported config changes.

The source builder now refreshes the resource manifest's source and argument
hashes after rendering the spec and HTTP templates. The actual resource gate
accepts that manifest, and both command lines match the retained owner adapter
and replay projection. The renderer preserves the donor's sandbox, root and
session; it does not yet generate a fresh packet identity or support changed
runtime configuration. The controller's literal spec pin still needs D2.

The isolated D2 controller now requires the three configuration approval pins,
checks run/host/menu and protected-helper file bindings, and includes those
inputs in the before-launch rechecks. It preserves all prior grant checks and
the literal spec pin. The transport extension supplies those pins from explicit
human-entry arguments and stages the four authenticated configuration files.
The retained archive has 80 members; adding those four files exceeds its
unchanged 80-member ceiling and correctly refuses. The subsequent D1 package
proposal reduces that configured archive to 79 members. The cleaned-inventory
proposal combines three further removals with the four extracted helpers and
fits at exactly 80. Real pre-launch inventory/source checks pass, then the
unchanged spec literal correctly refuses the generated spec. The subsequent
spec-binding proposal replaces that literal check with canonical reproduction:
a fixed template fingerprint binds all unconfigured fields, and configuration
plus authenticated package hashes supply the remaining slots. It reaches the
retained external build-proof boundary. The original cleaned-inventory replay
used a spec-pin seam; the new proposal's replay calls the real verifier and
passes the hash explicitly. Runtime configuration mutations, full native
pre-launch and complete D2 remain open.

The owner-timing slice now exercises the actual lease, journal and release
validators with CPU substitutes for child execution and platform observations.
The original and proposed adapters retain identical baseline deadlines and
success/failure decisions. An adapter-only shorter worker plan tightens its
deadlines; the whole-run configuration verifier still refuses that mutation.
The session-timing slice now covers transport total/seal accounting and
controller stage deadlines. Its derived 5,385-second remaining-owner check
replaces the stale 1,995-second transport literal. Whole-run mutations still
refuse; the stage reserves retain their exact W5 values. The worker-timing slice
now consumes spec durations and verifies the actual HTTP import chain and
socket expiry. Complete response replays retain source/spec bindings and the
existing evidence schema. Later slices cover artifact consumer APIs, per-row
completion and observed device/rank startup/readout binding. Fresh source-review
identities and all 11 active overlay patches are reproducible. Caller power
binding, outer artifact partitions, fresh launch manifests, quality-menu policy
and complete configuration mutation support remain open.

Before the complete D2 can be reviewed:

1. Implement config loading and authentication in proposal copies of every
   affected consumer. Include imported shared modules in the proposed protected
   set, while keeping boot ID in the one-use approval.
2. Call the actual controller `validate_manifest` and full post-run chain on CPU.
   Compare decisions with the retained helpers, including wrong spec, oversized
   member and stop-policy violations.
3. Generate real-menu, worst-case and doubled-menu evidence. Prove the required
   byte, member and observer margins. The current settings-size check covers
   only `FINAL-SETTINGS.json`.
4. Extend the retained-template resource renderer to fresh packet identities and
   supported config changes. Generate the complete package inventory from the config.
   Test each supported mutation through every dependent artifact and consumer.

These steps remain open. The D1 and D2 slices are prepared separately; no
protected patch has been applied to the builder's consumed sources. The
canonical worker/client changes are consumed by the builder. The controller slice's
source-review approval does not complete or approve the whole migration.
