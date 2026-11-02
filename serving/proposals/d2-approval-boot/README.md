# Boot identity in the approval

This isolated proposal extends the durable-limits proposal. The canonical source
builder still emits the original nine protected helpers. Exact complete protected
review and a fresh native approval remain required.

A schema 3 physical manifest omits `boot_id`. Its scope is
`matched-benchmark-physical-package-v3`. The root transport requires an explicit
`--approved-boot-id` and compares it with the observed host boot before creating
the workload directory. A stale or missing approval cannot stage or start work.

After that check, `bind_manifest_boot` creates an in-memory schema 2 view for the
existing runtime contract. It changes no archive or manifest bytes. The grant
retains the hash of the original schema 3 manifest and binds the approved boot
to the authenticated root parent. The controller checks that parent, the
configuration pins, the original receipt and its existing deadlines. All release,
resource, start-count and external runtime-proof gates remain required.

The same source archive can therefore receive a fresh approval after a reboot.
Legacy schema 2 manifests keep their embedded boot binding. An explicit approval
cannot silently replace that legacy binding. Schema 3 manifests with an embedded
boot claim refuse, even when the archive and manifest hashes are updated.

Eight focused CPU tests cover the archive, grant and human-entry refusal paths.
Thirty private composition checks pass, including both serving-wire replays,
natural and cap quality projections, durable sealing, two fresh boot bindings
against identical manifest bytes, and the unchanged external-proof refusal.
The grants in those tests are CPU objects. No approval was issued and no native
workload or root transport was invoked.

`serving/propose_packet.py` now renders a fresh manifest proposal and the caller
source dependencies. It authenticates the complete donor and every selected
protected source pin. It emits `MANIFEST-PROPOSAL.json`,
`LAUNCHER-BINDINGS-TEMPLATE.json` and an unfunded `APPROVAL-TEMPLATE.json`.
It emits no accepted launch manifest, authorization file or launch script.
The proposed package has exactly 80 members. The caller's pinned host file,
power observer and retained connection/command modules are staged for review.

The renderer remains separate from `build.py`. Caller activation and native
grading assembly remain pending. See [VERIFICATION.json](VERIFICATION.json) and
[the renderer verification](../../launcher/PROJECTION-VERIFICATION.json).
