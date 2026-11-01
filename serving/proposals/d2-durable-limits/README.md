# Configured durable evidence limits

This isolated proposal extends the artifact-binding proposal. The source builder
still emits the original nine protected helpers. This proposal has no execution
authority and does not complete D2.

`benchmark_artifacts.audit_durable_trees` accounts for the workload tree, the
protected transport tree and their outside closeout or refusal files. It enforces
the configured source-copy, observer-copy, outer-output, summary and small-metadata
partitions. Per-arm checks cover output streams, the three received members,
decoded client files and metadata. Producer and readout checks retain the native
tree limits; the controller does not extract a separate native tree.

Every file contributes once to the complete session byte count. Complete controller
and transport reports contribute to both the summary and observer partitions.
This conservative charge includes embedded observer copies without trusting a
report to identify its own overhead. Links, overlapping roots, changed files and
duplicate pending members refuse accounting.

The controller audits before hashing and includes its pending seal manifest before
exclusive publication. The root transport checks combined archive and source
copies before staging, derives both outer stream limits from configuration, and
audits both stopped trees before publishing its seal. Each outside closeout must
fit the remaining summary and complete-session space. All retained owner ceilings,
deadlines, release checks and one-use rules remain in force.

The root transport now imports its authenticated source bytes with a standard
module loader. It excludes source reopening and cached bytecode, registers modules
for dataclasses, and restores the previous registration if import fails. Its two
direct `exec(compile(...))` sites are removed.

The focused tests exercise real temporary files, seals and pipe-budget arithmetic.
The private packet checks authenticate all 313 donor members, replay both serving
arms and verify the actual archive, two source trees, both seals and both
closeouts. They do not execute a native workload or the physical root entry point.
See [VERIFICATION.json](VERIFICATION.json) for the observed results.

The composed source copies occupy 19,372,202 bytes of their 32 MiB partition.
That fits the ceiling but gives about 1.73 times margin. It does not satisfy the
required two-times evidence margin. Remaining source and runner cleanup must
resolve this gate; no ceiling was raised to make it pass.
