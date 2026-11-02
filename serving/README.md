# Serving cleanup: source implementation

This directory holds the observer safety and structural cleanup increments from
`CLEANUP-HANDOFF-20261031-v1.md`. The owner selected this repository for the
cleanup source. These files are not a launch packet or a new serving selection.

The latest runtime increment is the
[verify metadata preparation](proposals/pre-replay/README.md).
The adapter prepares causal maps before forward-graph capture or replay;
its startup witness and reader require the new preparation policy. It retains
the decode frontends and the [runner post-load hook](proposals/post-load/README.md).
The candidate has 78 members under the existing
80-member ceiling. The [source reference proposal](proposals/source-references/README.md)
reuses unchanged protected files and produces an identical package without
adding review copies. The runtime builds on the
[closed HTTP package](proposals/closed-http/README.md), which removes disabled
feature APIs, legacy diagnostics and three unused payloads, and the
[shared helper cleanup](proposals/common-helpers/README.md), the
[startup source digest proposal](proposals/startup-digests/README.md), the
[MXFP4 startup runtime binding](proposals/runtime-binding/README.md) and the
[configured request and evidence proposal](proposals/configuration-mutations/README.md).
The retained A/B pilot predates these changes. Replay metadata, startup scans
and memory cleanup remain; [CLEANUP-PROGRESS.json](CLEANUP-PROGRESS.json) records
the gates. The new sources still need native serving qualification.

## Implemented

The W5 v2 lifecycle observer can exhaust its 8,192-event allowance after an
allocator operation succeeds. Its old assertion then escapes into serving.

`legacy/review/j01_probe.py` now separates observation from all five wrapped
allocator/cache operations. Observation errors disable collection and log once.
The event counter saturates. Each original operation still runs once, returns
its original result, and propagates its own exceptions. Startup source and
pool checks remain fatal. Process-control exceptions are not swallowed.

On cleanup, the observer restores its wrappers and publishes one health receipt
per rank. Publication uses an exclusive pending file and a non-overwriting
hard link. A failed publication retains the pending file. The reader requires
healthy receipts bound to the run, worker and helper source; missing, failed,
noncanonical or incomplete receipts cannot pass. The unchanged retained
exporter preserves these files and rejects unfinished multiple-link files.

Graph capture, replay and commit callbacks now contain observation failures.
Their counters saturate at their declared bounds. The GDN wrapper observes
metadata separately from its original forward call, so observation failure
cannot skip or retry that call. Original forward exceptions still propagate.
Replay identity failures disable observation instead of raising into replay.

The target-KV observer buffers at most 128 records and 1 MiB, with an 8 KiB
per-record cap. It flushes the journal during cleanup, then publishes health.
Its optional inference callback no longer asserts that an observer exists.
The graph observer also publishes health when the worker's existing cleanup
path runs. Receipt validation joins graph/commit counts to the binary counters
and target-KV counts/bytes to the journal. Failed, stale or missing evidence is
rejected. The shared `observer_health.py` is pinned by both importing modules;
changing or omitting it refuses their import.

`legacy/review/serving_integration.py` requires lifecycle receipts in instrumented
composed runs and graph/target-KV receipts in instrumented runs. Production
readback still follows its existing path. This is an
intentional evidence-format change: assemble the new producer and reader
together. Historical packets keep their original readers.

### Structural cleanup

The retired BetterBench client and its unused terminal validators are removed
from `j01_probe.py`. The reader no longer imports the probe at module startup
for BetterBench. The copied `serving_runtime.py` no longer loads that client,
runs its preflight, or dispatches its callback. It rejects factories with a
`betterbench_owner_mode` attribute before template loading or native work.
The ordinary client's gate, callback arguments, summary validation and error
propagation retain their previous behavior.

Three active embedded sources are now adjacent files: `target_kv_metadata.py`,
`kv_graph_join.py` and `request_receipts.py`. All three began as byte-for-byte
extractions from the sealed donor. The later evidence-capacity change compresses
request receipt fields and groups logical control records.
Their consumers pin each file separately and execute the checked source bytes,
without consulting cached bytecode. Missing, changed or linked helpers refuse
loading. The probe is about 41 KB, down from 255 KB.

The model, graph and HTTP readout modules now load authenticated source bytes
through standard module loaders. Each loader retains its own checked bytes;
cached bytecode or later changes to the named file cannot replace them.
The HTTP import chain still registers dependencies in order and removes its
new registrations if any import fails. The reader's temporary contract import
still restores the previous module on success or failure. The receipt writer
retains its lock, write-once fields, byte limits and atomic publication.

`mtp_contract.py` now permits only `sprint01-paired-serving`. The donor contained
25 phases; the 24 inactive phases and their settings, request-history,
speculation and overlay branches are removed. Standalone `none` and `mtp`
speculative kinds now refuse validation. Both selected arms retain DFlash8,
the existing checkpoints, BF16 caches, capacities and source pins. The contract
is about 83 KB, down from 109 KB.

Legacy phase markers and flag functions remain where current consumers still
reference them. They cannot select a retired phase. Their removal needs the
corresponding consumer migration. `on` and `production` are accepted;
omission now selects `production`, as the owner decided. Explicit `on` is
required for evidence collection. The proposed production view now masks the
six private evidence helpers; native verification remains a Phase 1 gate.

`model_benchmark.py` now launches only the W5 HTTP path. The obsolete per-rank
runner, process observation/wait loop and independent-reference dispatch are
removed. The module is about 34 KB, down from 86 KB. The entry point validates
the HTTP specification before loading artifact helpers or creating scratch
output. Retired modes refuse explicitly. The reviewed execution flag, physical
identity, deadline, private output permissions and export-on-failure behavior
remain required. Controller/readout compatibility functions and the pinned
graph-shell projection remain until their consumers migrate; the protected
helpers are unchanged.

The transitional modules come from the authenticated W5 v2 package, apart from
the new shared `observer_health.py`. Their donor hashes, extraction hashes and
seal are in [SOURCE.json](SOURCE.json). Tests import the complete edited modules;
the flat `legacy/review` layout remains until the builder supplies the full
source closure. Edit these repository sources; the initial scratch copies are
superseded.

### Deterministic source assembly

`build.py` assembles a **source review bundle**, with no launch manifest or
approval. It authenticates all 313 members of the sealed W5 v2 donor, copies
the retained package, and replaces modules with the canonical repository
sources. The five previously missing HTTP/readout modules are now canonical
copies; startup identity and stop-policy changes are recorded in the subsequent
proposals. The builder executes no donor Python code.

The builder refreshes `ALLOCATION-OVERLAY.json`, both HTTP binding templates,
`BENCHMARK-SPEC2.json` and `RESOURCE-MANIFEST.json`. It includes all four extracted/shared helpers in
the source maps and recomputes the worker aliases and source-manifest hashes.
Native source pins, original overlay hashes, settings and request assets remain
unchanged. W5's active overlay selection also remains unchanged; retained
inactive overlay files still have their own valid pins.

```sh
python3 -I -B -S serving/build.py \
  --donor /absolute/path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --output /absolute/path/to/new-source-review
```

The output must be new and outside the donor and `serving` trees. The builder
refuses changed seals, missing members, linked sources and protected overrides.
It checks inputs and renders before creating output. A write failure retains
the incomplete directory; choose another new destination after investigation.

The bundle contains 75 review members, the retained nine-helper
`PROTECTED-SHA256.json`, and `SOURCE-BUILD.json`. The latter records every payload
hash, static imports, references to changed source pins, and remaining gates.
Two builds from identical inputs have identical file bytes. The protected
controller still contains the donor's spec hash; the report identifies that
reference. Changing it requires the planned protected-helper migration.
Eight inactive donor files are omitted after complete seal authentication;
their hashes are recorded in `SOURCE-BUILD.json`. The selected HTTP mode and
`both` draft-head selection are required. Legacy compatibility readers remain
until their protected consumers migrate. Resource-manifest source bindings are generated for the retained W5
sandbox template. Fresh launch manifests, package limits and runtime config
consumers remain pending. Source-review sessions can use an explicit packet ID.

The resource renderer checks the donor's spec, entry, gate and artifact pins,
both argument hashes, exact invocation tails, entry mounts and session/unit
bindings before rendering. It refreshes the source pins and argument hashes,
and adds the six production evidence-file masks. Existing native mounts,
isolation options, environment variables and scratch sizes remain unchanged.
It preserves the retained remote root and session. This source-review
template does not authorize reuse of that session or create a launch packet.
With `--packet-id` and a run directory, the builder derives a fresh session from
the packet ID and authenticated run hash, then renders matching roots, arguments
and unit names. It still emits no launch manifest or execution approval.

With `--upstream-root`, the builder reproduces all 11 active overlays from the
[pinned effective base and patch series](patches/README.md). Forward and reverse
proofs match every retained overlay hash. Two patched builds and the retained
overlay path have identical payload bytes. The pinned base includes retained
runner/KV changes; it is not an unmodified release tree.

### W5 configuration baseline

`config/runs/w5-v2/run.json` binds the host, the 176-row menu, model/stack assets,
profile, overlay order and primary limits. Every menu row declares `stop_policy`;
this retained baseline uses `natural` for all rows. The host file records the
device set, HIP rank indices and expected 235 W power cap. Boot ID remains an approval
field and is rejected in the host config.

Pass the absolute run directory before the builder options to use it:

```sh
python3 -I -B -S serving/build.py /absolute/repository/serving/config/runs/w5-v2 \
  --donor /absolute/path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --output /absolute/path/to/new-config-review
```

The three local schemas are closed: unknown fields, wrong types and supplied
derived limits refuse. The standard-library validator implements only their
declared keyword subset and rejects unsupported keywords. Config identity uses
SHA-256 of `canonical_line()` (sorted compact JSON with one trailing newline);
source-manifest identity continues to use `canonical_bare()`.

`spry_common/limits.py` derives worker, arm, active, total and caller times, plus
the durable-byte total. It reproduces the retained 6,360-second total and
496.5 MiB byte accounting. The 390-second campaign margin is the arithmetic
remainder after the named stages, not a measured duration. The loader checks
the authenticated protected ceilings without executing those helpers. Only
the settings file currently has a verified twofold byte margin; complete
evidence capacity remains unverified.

Configured builds include the canonical run/host/menu files and
`DERIVED-LIMITS.json`. Every review-file byte matches the existing source build.
Profile, host, menu, instrumentation and policy changes currently refuse against
the pinned W5 baseline. General config mutation support requires the remaining
consumer migration; this is not the completed T3 gate. See the
[protected migration map](proposals/PROTECTED-CONFIG-MIGRATION.md) for the exact
consumer boundaries and required proof. It is not an applied D1/D2 diff.

### First protected-source proposal

The [first D1 proposal](proposals/d1-controller-inventory/README.md) removes
47 lines of obsolete controller inventory logic and an unused CPU wrapper.
Its exact patch contains deletions only. The active W5 verification-attention
branch, selected package inventory, remaining function bodies and ceilings
are preserved. The source builder continues to emit all nine original protected
helpers. Further D1 pruning, D2 and exact owner approval remain open.

The [first D2 proposal](proposals/d2-config-binding/README.md) adds checks for
the approved run-config, host and protected-set hashes. It checks referenced
menu and helper bytes, then includes these inputs in before-launch rechecks.
The existing spec literal and owner checks remain. The owner approved this
presented source proposal and continued implementation; the exact context and
hashes are in the [approval record](proposals/OWNER-APPROVAL-20261101.json).

The [transport extension](proposals/d2-transport-binding/README.md) now accepts
the three independent approval hashes, authenticates the archived configuration,
stages it through no-follow directory handles and emits matching grant fields.
The retained archive contains 80 members; its four new configuration files would
exceed the unchanged 80-member ceiling. Package pruning must precede complete
transport integration. Runtime consumers and generated-spec checks remain open.
Both proposals are isolated from the builder's consumed sources.

The [D1 package-pruning proposal](proposals/d1-package-pruning/README.md) removes
five inactive package files with five deleted source lines and no additions.
It preserves both arms' active overlay hashes and sandbox argument lists.
Its separate D2 composition fits the retained package and configuration into
79 archive members. That slice alone would require 83 members with the four
extracted/shared helpers.

The [cleaned-inventory proposal](proposals/d2-cleaned-inventory/README.md)
removes the three remaining authoring/legacy-profiler names and adds the four
extracted helpers to the controller inventory. The builder now omits all eight
traced files. A fresh test composition with the proposed protected sources
contains exactly 80 archive members, including configuration. The actual
controller still refuses its generated spec at the unchanged literal check;
config-to-spec verification remains required.

The subsequent [spec-binding proposal](proposals/d2-spec-binding/README.md)
reproduces the complete canonical spec from the pinned W5 run, host, fixed
template and authenticated source hashes. It removes the controller's global
spec pin and passes the verified hash through its consumers. This isolated
controller reaches the same external build-proof boundary as the retained W5
controller. Arbitrary configuration changes remain unsupported while runtime
consumers retain literal limits and settings.

The [owner-timing proposal](proposals/d2-owner-timing/README.md) supplies an
immutable timing plan from verified configuration to the actual owner adapter.
It preserves the worker/work/envelope ceilings, cleanup reserves and signal
offsets. Invalid plans refuse before lease acquisition. A shorter adapter-only
plan tightens deadlines; whole-run mutations still refuse until the remaining
transport, controller and worker consumers migrate.

The [session-timing proposal](proposals/d2-session-timing/README.md) now connects
transport and controller accounting to the verified plan. It binds manifest,
grant and seal totals, retains the original receipt time, and replaces the
stale 1,995-second transport start check with the derived 5,385-second W5 owner
budget.

The [worker-timing proposal](proposals/d2-worker-timing/README.md) connects the
active worker and client to validated spec values under the existing ceilings.
It prevents readiness work from extending the client beyond the worker deadline.
The canonical entry/runtime changes are consumed by the source builder; the
two-line protected controller change remains isolated. This is configuration
cleanup. No inference throughput or latency improvement has been measured.

The [artifact-limits proposal](proposals/d2-artifact-limits/README.md) adds a
validated immutable plan to the actual HTTP pack, export, receive and capture
APIs. The controller supplies authenticated limits to capture and receive.
Worker export configuration, readout and outer durable partitions still need
that binding. The existing byte and file ceilings remain protected.

The [artifact-binding proposal](proposals/d2-artifact-binding/README.md) now
projects the primaries into the authenticated spec with `--configured-artifacts`.
The worker passes one frozen plan to all export operations; both readout modes
enforce the projected encoded, decoded and tree limits. The controller verifies
the projection and passes its plan to readout. This option requires the proposed
protected consumers.

The [durable-limits proposal](proposals/d2-durable-limits/README.md) binds the
outer partitions across the workload and protected transport trees, including
both seal manifests and outside closeouts. It also removes the root transport's
two direct source-execution sites. All owner ceilings remain unchanged. Combined
source copies fit their partition with about 1.73 times margin; further source
reduction is required to meet the two-times evidence gate.

The [stop-policy proposal](proposals/d2-stop-policy/README.md) connects the client,
reader, summaries and protected reporting to one per-row completion rule. A
declared cap row can finish naturally or reach its exact cap; cap-length work
remains ungraded. The retained natural-answer quality packet correctly refuses
that work. The [quality source](quality/README.md) now projects a configured menu
without retaining cap-work answer oracles. It preserves natural baseline bytes,
checks full wire decodes, and excludes cap rows from answer grades. Native grading
packet assembly and whole-run mutation support remain open.

The [boot-approval proposal](proposals/d2-approval-boot/README.md) moves current
boot identity out of the schema 3 source manifest and into the separate approval.
The transport checks the observed boot before workload creation. Existing
schema 2 packets keep their pinned boot. The [proposal renderer](launcher/PROJECTION.md)
prepares the fresh manifest, configuration and caller dependencies with no
accepted launch files or execution authority. Caller activation remains open.

The [production import cleanup](proposals/production-imports/README.md) removes
unconditional parity imports from the model, runtime and launcher. The
authenticated production chain loads seven helpers; instrumented mode retains
all eight. Both modes retain source authentication and import rollback. Evidence
mount isolation is still pending.

The [host-binding proposal](proposals/d2-host-binding/README.md) binds observed
HIP indices to BDF, UUID and render/sysfs identity. Production now performs the
pre-weight identity check without importing observers at that boundary or
writing an extra pre-weight evidence record. The controller uses authenticated
host devices. Complete production mount/import isolation remains open.
The [canonical power observer](launcher/README.md) also reads the pinned host
map and cap. Its remote execution is mocked in tests. Fresh caller generation
must still supply its dependencies and configuration binding.

## Verify

The standard-library suite uses native integer-tensor and allocator stubs. It
does not import Torch, SGLang or Triton, and it performs no GPU work.

```sh
make test-serving-cleanup
```

The private packet checks authenticate the original 313-member seal, import
fresh copies, reproduce the old cap failure, and exercise the unchanged
protected exporter. Supply the retained W5 v2 preparation root explicitly:

```sh
python3 -I -B -S serving/tests/check_retained_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_contract_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_protected_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_config_binding_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_transport_binding_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_package_pruning_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_cleaned_inventory_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_spec_binding_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_owner_timing_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_session_timing_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_worker_timing_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_artifact_limits_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_stop_policy_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_host_binding_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_artifact_binding_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_quality_policy_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 -v
python3 -I -B -S serving/tests/check_patch_series_packet.py --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 --upstream-root /path/to/pinned-effective-base -v
```

Observed locally: the full repository suite passes on macOS Python 3.14.8,
including 412 focused tests and 849 Python unit tests overall. Thirty-seven
private composition checks pass on that interpreter, covering
the complete cleaned inventory, grant/spec verification, actual owner deadlines
and release decisions, configured sealing, worker/client deadlines and both
response replays. The fifteen contract checks pass again with the edited
canonical serving sources, including 134 byte-identical output comparisons.
Seventeen patch-series checks also pass, including deterministic builds, all
11 exact overlays, fresh identities and actual resource/owner projections.
The one intentional comparison difference verifies that omitted instrumentation
now selects production; both explicit modes retain their previous behavior.
Earlier protected-source and
retained-packet results remain recorded with their exact source pins.

The quality composition's 27 checks pass within the complete composition run.
The initial patch-order failure and its targeted correction remain recorded.
Both actual serving-wire replays preserve the natural input bytes. The declared
cap replay retains both complete wires with no answer oracle. CPU tokenizer
stand-ins also reproduce all 352 retained natural decoded answers and grades.
Eleven focused quality tests cover mixed, all-cap and doubled menus. This does
not establish the complete serving pipeline's doubled-menu capacity.

Eight focused boot-approval tests and five proposal-rendering tests pass. Two
additional complete-input renderer checks prove byte-identical output in two
destinations, the actual 80-member archive/spec binding, and the staged caller's
pinned host read. They retain the external runtime-proof refusal and execute
no SSH or native workload.
Historical proposal checkers require their recorded canonical source versions.
Use the latest checker for the current composition; later source changes can
make an earlier proposal's pin checks refuse.
Tests cover callback failures,
counter saturation, exception
preservation, source substitution, receipt/counter joins, and the worker's real
cleanup dispatch with native seams stubbed. The lifecycle test completes 9,000
allocator calls while retaining exactly 8,192 events. Target-KV tests admit
exactly 128 distinct records before saturation. The Triton optional-call change
has a static AST check; it has not been executed in the native runtime.
The structural tests cover changed/missing/linked sources, stale bytecode,
the 176-row composed evidence join and its cancellation row, invalid metadata,
ordinary client success/failure, and early refusal of the retired client.
Retained-packet checks verify the extracted bytes against the sealed originals
and compare bytes from the old and new composed-record validators. Runtime
callback tests use CPU substitutes for adjacent contracts and native entry
points; they do not establish the complete runtime import closure.
Import tests also cover process-control exceptions, registry rollback,
authenticated bytes surviving a later file change, and concurrent receipt
writes. The legacy graph route retains its existing source-pin refusal.

The contract comparison suite authenticates the donor seal and uses two fresh
copies. It checks 135 byte-identical outputs, including both logical arms,
ten settings variants, package membership, overlays, sandbox arguments and
template expansion. It also checks refusal of all 24 inactive phases, malformed
settings, changed request assets and changed overlay sources. These comparisons
use the donor's assets; source-build checks separately assemble the edited modules.
An additional check imports the real retained contract/client/reader chain,
then repeats it with the cleaned contract and client. All 176 frozen request
bodies and their physical receipt files match byte for byte. The retained
exporter also produces identical output for old and new receipt envelopes.
These are request-receipt checks, not a complete response/evidence-size gate.
The real entry point also matches the sealed runner's dispatch, exit behavior,
result bytes and export calls for both W5 arms across five completion/failure
cases. Native execution and export are replaced with CPU substitutes in these
checks. The unchanged protected owner adapter produces identical full sandbox
arguments and selects the same retained proof reader with the cleaned module.
The ordinary contract tests use three pinned donor JSON fixtures and synthetic
request bytes. They test settings and hash checks without the private corpus.

Source-build checks compare two complete output trees, validate every private
binding pin and worker alias, and confirm unchanged active overlays. They load
the actual eight-module HTTP chain and the readout chain for both arms while
blocking native imports. Test-only templates project `/mnt/review` onto a local
directory; source bytes stay unchanged. This proves the CPU import boundary,
not native startup, worker execution or launch readiness.

The generated resource manifest passes the retained resource gate's real
`read_manifest` validator. Both complete command lines match the unchanged
owner adapter followed by the contract's replay projection, using the built
spec and source hashes. Gate checks refuse stale argument hashes, a changed arm
and a changed session. Builder tests also reject inconsistent source pins,
duplicate flags, altered output paths and missing entry mounts before creating
output. These tests do not execute the sandbox or native worker.

The isolated D1 comparison runs the retained 176-row response replay and its
existing envelope through both controllers' HTTP post-run chain for both arms.
Both accept the same rows and refuse length stops and oversized members.
Pre-launch checks reach the same missing workstation build proof; they use
substitutes for external host inputs. These checks cover the retained W5 sources
with one proposed controller change.

The complete cleaned inventory now passes both production response replays,
including export, receive, readout and length-stop refusal. With the spec-binding
proposal, the real verifier authenticates configuration and reproduces the spec
before replay. The retained CPU producer is adapted to pass the verified hash
explicitly; no controller global or function is replaced for post-run. Pre-launch
reaches the same external build-proof boundary as the retained controller. Archive inspection,
staging and synthetic grant authentication pass at 80 members; member 81 and
missing/changed extracted helpers refuse. Full native pre-launch, instrumented
health-receipt composition, all-cap and doubled-menu evidence remain open.
The measured file-count margin is 1.59 times, below the design's two-times requirement.

The owner-timing comparison uses the real lease and release validators with
synthetic child/platform observations. Both adapters retain identical baseline
deadlines and decisions for completion and four failure cases. A verified
timing plan completes both synthetic arms. Six invalid-input cases refuse
before lease acquisition. An adapter-only shorter plan tightens deadlines;
the full configuration verifier still rejects changed runs.

Session-timing tests retain the original receipt through preparation, grant
validation and sealing. The transport refuses an expired receipt before
creating a workload root or launching a process. The complete controller
accepts the matching configured grant and writes its seal and closeout under
that plan. Shortened plans pass isolated transport and sealing checks;
whole-run mutations still refuse under the pinned W5 configuration.

Worker/client tests now cover configured durations, startup reserves, unchanged
ceilings, expired readiness and socket timeouts. They load the actual HTTP
module chain with fresh source pins and exercise both baseline and shorter
durations for both arms. The complete controller replays preserve the HTTP
evidence schema and source/spec bindings. Whole-run mutation support still
requires the remaining configuration consumers and fresh packet identity.

Current durable-accounting logs are in `../.test-build/serving-cleanup/durable-limits/`.
Current import-isolation logs are in `../.test-build/serving-cleanup/production-imports/`.
Boot-approval and fresh proposal logs are in `../.test-build/serving-cleanup/approval-boot/`.
Quality-policy logs are in `../.test-build/serving-cleanup/quality-policy/`.
Earlier artifact-binding logs remain in `../.test-build/serving-cleanup/artifact-binding/`.
Caller-power logs are in `../.test-build/serving-cleanup/caller-power/`.
Patch-series logs remain in `../.test-build/serving-cleanup/patch-series/`.
Fresh-identity logs are in `../.test-build/serving-cleanup/fresh-identity/`.
Host-binding logs are in `../.test-build/serving-cleanup/host-binding/`.
Stop-policy logs are in `../.test-build/serving-cleanup/stop-policy/`.
Artifact-limit logs remain in `../.test-build/serving-cleanup/artifact-limits/`.
Worker-timing logs remain in `../.test-build/serving-cleanup/worker-timing/`.
Session-timing logs remain in `../.test-build/serving-cleanup/session-timing/`.
Owner-timing logs remain in `../.test-build/serving-cleanup/owner-timing/`.
Spec-binding logs remain in `../.test-build/serving-cleanup/spec-binding/`.
Inventory logs remain in `../.test-build/serving-cleanup/cleaned-inventory/`.
Package-pruning logs remain in `../.test-build/serving-cleanup/package-pruning/`.
Transport logs remain in `../.test-build/serving-cleanup/transport-binding/`.
Controller configuration logs remain in `../.test-build/serving-cleanup/config-binding/`.
Resource-manifest logs remain in `../.test-build/serving-cleanup/resource-manifest/`.
Proposal logs remain in `../.test-build/serving-cleanup/protected-d1/`.
Configuration logs remain in `../.test-build/serving-cleanup/config/`.
The retained-packet checks remain recorded under `../.test-build/serving-cleanup/imports/`.
Earlier logs and preparation scratch files remain retained.

## Remaining gates

The complete cleanup is **not finished**. The current proposals have CPU proof
for observer containment, production evidence masks, resolved startup settings,
configured requests and limits, source imports, bounded transport, closed HTTP
consumers and the runner post-load hook. The default builder retains its
original protected set. Native adoption needs the exact protected diff and a
fresh one-use approval.

Implementation still required:

1. Finish the remaining common helper consumers and inactive phase markers.
2. Consolidate startup tensor scans and scalar synchronizations.
3. Share the FP8 head and rebind QKVZ/BA shards as packed-buffer views. Preserve
   P1 identity checks and keep recovered memory as headroom.

Native qualification still requires live production mount/import isolation,
greedy token equality, selected answer quality, startup byte equality, memory
sampling per BDF, and independent resource release. The configured and
instrumented full-menu capacity gates also remain. Source-only and finite CPU
fixtures do not establish these native results.

The [patch series](patches/README.md) reproduces all 11 active overlays against
the pinned retained effective base. The [post-load proposal](proposals/post-load/README.md)
replaces inline runner installation with the startup hook. Historical packets
retain their own source and readers; selecting an earlier sealed package is
the rollback path. No sealed root or installed runtime is edited in place.

The [production measurement preparation](measurements/production-cleanup/README.md)
stages matched W5 baseline and cleanup source reviews. Both use production
instrumentation mode and stock attention. The review excludes C4 and profilers
from the comparison and grants no execution authority. The later
[GPU preparation](measurements/gpu-run/README.md) seals separate baseline and
cleanup sources, verifies caller failure paths and W5 capacity, and passes the
native CPU preflight. Their later native results are recorded below.

Baseline A subsequently completed two starts and 352 responses. Its complete
owner/native/HTTP/release replay passed. The first native CPU grading attempt
stopped before decoding because the inherited template mixed older timing limits
with the W5 profile. The approved correction decoded and passed all 352 natural
answers and required long-answer coverage. Independent CPU release passed.
Cleanup B subsequently completed its approved two starts and 352 responses.
Complete replay, all 352 natural-answer checks, long-answer coverage and
independent GPU/CPU release passed. Two local replay handoffs required correction
before CPU dispatch; the native sources and limits remained unchanged. All
352 A/B rows match for prompt IDs, output IDs, speculative work and cache work.
The [pilot result](measurements/gpu-run/CLEANUP-B-RESULT.json) records observed
throughput changes from -0.080% to -0.175%. It establishes no repeatable
performance change. The operation is parked with no remaining native allowance.
This closes the fixed W5 comparison, while the broader cleanup gates above
remain open.
