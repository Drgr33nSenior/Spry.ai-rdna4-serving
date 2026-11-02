# Serving proposal index

Each directory retains a source increment or a control used during the cleanup.
Its `PROPOSAL.json` selects exact files and hashes; its verification record
describes what had passed at that point. Older “pending” states belong to those
versions. The [completed sign-off](../measurements/native-qualification/CLEANUP-SIGNOFF.md)
records the final bounded result.

Use the [serving guide](../README.md) for current work and the
[history](../HISTORY.md) for the implementation log. A proposal directory or an
old packet ID supplies no new launch authority.

## Find an increment by purpose

| Concern | Starting points |
| --- | --- |
| Protected configuration contract | [Migration map](PROTECTED-CONFIG-MIGRATION.md), [spec binding](d2-spec-binding/README.md), [configuration binding](d2-config-binding/README.md), [boot approval](d2-approval-boot/README.md) |
| Configured budgets and artifacts | [Worker timing](d2-worker-timing/README.md), [owner timing](d2-owner-timing/README.md), [session timing](d2-session-timing/README.md), [durable limits](d2-durable-limits/README.md), [artifact limits](d2-artifact-limits/README.md) |
| Shared source and smaller packages | [Common helpers](common-helpers/README.md), [shared closure](common-closure/README.md), [phase closure](phase-closure/README.md), [closed HTTP package](closed-http/README.md), [source references](source-references/README.md) |
| Production isolation and startup | [Production imports](production-imports/README.md), [production isolation](production-isolation/README.md), [startup settings](production-startup/README.md), [startup proofs](startup-proof-policy/README.md), [draft preflight](selected-draft-preflight/README.md) |
| Runtime and source identity | [Runtime binding](runtime-binding/README.md), [startup digest cache](startup-digests/README.md), [source fingerprints](source-fingerprints/README.md), [preweight pins](preweight-source-pins/README.md), [bootstrap staging](bootstrap-common-stage/README.md) |
| Stock runner and memory recovery | [Post-load hook](post-load/README.md), [metadata preparation](pre-replay/README.md), [shared head and packed views](shared-memory/README.md) |
| Replay and host-readback corrections | [Captured metadata](replay-metadata-scope/README.md), [target-only hook](target-replay-scope/README.md), [authenticated host scope](host-readout-scope/README.md) |
| Evidence and campaign assembly | [Capacity](evidence-capacity/README.md), [final source pins](final-template-pins/README.md), [campaign tooling](final-campaign/README.md), [qualification preparation](final-qualification/README.md) |

## Controls and source selection

The [post-load control](post-load-control/README.md) and
[pre-memory control](pre-memory-startup/README.md) are bounded comparison
baselines. They do not select the final cleanup's memory and replay changes.

The final operation-207 source was prepared through the
[host-readout increment](host-readout-scope/README.md). That record preserves the
complete selected source, including the earlier memory and target-replay work.
Its pre-launch verification is a snapshot; the later
[native result](../measurements/native-qualification/GPU-207-RESULT.json) and
sign-off supply the completed validation.

Select a reviewed proposal explicitly during preparation. Similar filenames,
default CLI choices and the newest directory name are not substitutes for the
recorded source pins. Historical sealed sources and failed attempts remain
available for traceability and the documented rollback path.
