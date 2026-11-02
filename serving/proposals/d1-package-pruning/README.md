# D1 proposal: prune the retained W5 package

Status: **isolated proposal; not approved or consumed by the builder**.
This extends the [first D1 proposal](../d1-controller-inventory/README.md) and
provides a separate composition with the controller and transport D2 proposals.
It applies to the retained W5 configuration with `d3b_draft_head_arm=both`.
Other historical runner selections are outside this comparison.

[D1.patch](D1.patch) contains five deleted source lines and no added lines.
The controller drops the old capture inventory group and the two historical
build-proof entries. The transport drops its stale requirement for the old
capture runner. All function bodies, imports and owner ceilings are unchanged.

The selected review inventory falls from 79 to 74 members. These five files are
omitted from fresh test copies; their sealed originals remain intact:

- `capture_model_runner.py`: both arms select `d3b_combined_model_runner.py`.
- `divergence_capture.py`: the old comparison/capture modes load this helper;
  the selected HTTP spec has no `validation_comparison`.
- `prefill_row_audit.py`: the historical prefill-row audit mode loads this helper.
- `CAPTURE-BUILD-PROOF.json`: historical comparison overlays require this proof.
- `STOCK-BUILD-PROOF.json`: the retired graph-context preflight reads this proof.

The selected hybrid backend remains in the contract's package assets. Both arms'
actual overlay files, hashes and complete sandbox argument lists remain identical.
The controller still checks the external retained-runtime build proof; removing
the two review-directory proof files does not remove that check.

[PACKAGE-INVENTORY.json](PACKAGE-INVENTORY.json) records the 74-member D1 review
inventory. [PROPOSAL.json](PROPOSAL.json) pins both changed sources, the patch,
the inventory and the proposed protected sets.

## D2 composition

[D2.patch](D2.patch) applies the previously prepared configuration-binding changes
after this D1 patch. The exact resulting files are under [integration/review](integration/review).
The source checks prove that the only differences from the earlier D2 copies are
the same D1 deletions. The previous controller approval names an earlier hash;
this newly composed source set remains a separate review artifact.

With the composed D2 helpers, the retained package contains 74 review members,
one manifest and four configuration files: **79 archive members under the
unchanged 80-member ceiling**. The CPU integration check exercises real archive
inspection and local staging, then validates the transport-generated grant with
the complete controller. An 81-member archive still refuses.

This result covers the retained-source package. The cleaned source
bundle adds four extracted/shared helpers. Applying only these removals to that
bundle's review files, with the same four configuration inputs and a manifest,
would produce 83 archive members. This projection excludes both review-only
reports, `SOURCE-BUILD.json` and `DERIVED-LIMITS.json`. At least three more
members must be removed or migrated before the
complete cleaned packet fits. The subsequent
[D2 cleaned-inventory proposal](../d2-cleaned-inventory/README.md) supplies those
three removals and the four helper entries. This earlier proposal and its
recorded evidence remain unchanged. Neither protected proposal is installed independently.

## Verification

Five source checks, ten private package checks and the full `make test` suite
pass on macOS Python 3.11.6. [VERIFICATION.json](VERIFICATION.json) records the
exact sources, acceptance decisions, refusals and retained-envelope margins.

```sh
python3 -I -B -S serving/tests/test_package_pruning_proposal.py
python3 -I -B -S serving/tests/check_package_pruning_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

The private checker authenticates all 313 sealed donor members, then works on
fresh copies. It compares exact inventories, remaining file bytes, overlay hashes,
both owner argument lists, pre-launch decisions and refusal cases. It runs the
actual retained HTTP producer, exporter, receiver and controller post-run checks
for 176 rows in each arm, for both the expected-response replay and retained
envelope. Length stops and oversized artifact members must still refuse.

Pre-launch comparison stops at the same unavailable workstation build proof.
External host/runtime checks use the retained CPU substitutes. The integrated
grant is synthetic and stays in temporary test data. No root observer, lease,
native model or GPU starts. The replay does not exercise the whole cleaned
source bundle or its new observer receipts; the all-cap, doubled-menu and
two-times evidence-margin gates remain open.

Local logs are retained under
`.test-build/serving-cleanup/package-pruning/`. Source assembly continues to emit
the original protected helper set until the complete reviewed migration is ready.
