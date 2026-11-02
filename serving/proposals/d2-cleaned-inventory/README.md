# D2 proposal: complete the cleaned W5 inventory

Status: **isolated protected-source proposal; not consumed by the builder**.
The source builder now omits eight inactive donor files. With this proposed
controller and the prior composed transport, the cleaned package contains
75 review members, one manifest and four configuration files: **80 members**.
The archive ceiling remains 80.

This [D2.patch](D2.patch) extends the composed controller from
[D1 package pruning](../d1-package-pruning/README.md). It removes the authoring,
batch-telemetry and cache-profile names from the inventory and includes
`observer_health.py`, `target_kv_metadata.py`, `kv_graph_join.py` and
`request_receipts.py`. Function bodies, imports, limits and the frozen spec
check remain unchanged. [PROPOSAL.json](PROPOSAL.json) pins the exact diff and
composed nine-helper set. The transport bytes are reused from the preceding
proposal.

The three removals share source lines with active inventory entries, so this
is explicitly a D2 inventory change. It does not alter either earlier D1 patch
or claim to be deletion-only. Config-generated inventories remain later work.

## Dependency scope

The earlier five removals remain: the old capture runner, divergence and
prefill-row helpers, and the two historical review build proofs. The three
additional omissions are:

- `prepare_benchmark_package.py`: authoring only; the runtime has no consumer.
- `batch_telemetry.py`: loaded by the legacy comparison summary path.
- `cache_profile.py`: loaded by legacy cache-attribution readers. The active
  `mtp_contract.cache_profile()` method is separate and remains unchanged.

The selected HTTP path uses `checked_http_arm` and `serving_integration`.
It does not call the legacy comparison summary or cache-attribution readers.
Compatibility functions still exist for the protected consumers awaiting
migration; omitting their helpers does not qualify those historical modes.

The builder requires the retained W5 HTTP mode, `sprint01-paired-serving`,
the `both` draft-head selection, and no legacy comparison/cache-profile spec.
It authenticates all 313 donor members, including the eight omitted files,
before rendering. It records the omitted hashes in `SOURCE-BUILD.json` and
leaves every donor file intact. Active runtime overlay bytes remain identical.

Source builds still contain the original nine protected helpers. Only fresh
CPU test copies combine the proposed controller, prior transport and cleaned
sources. The archive test excludes the two review-only reports,
`SOURCE-BUILD.json` and `DERIVED-LIMITS.json`. It creates a synthetic manifest
and grant in temporary test storage; these are not launch artifacts.

## Verification boundary

The private checker exercises actual archive inspection and staging at 80
members, grant authentication, and rejection of member 81. It also rejects
missing or changed extracted helpers before staging. The actual pre-launch
controller passes inventory/source checks and then refuses with
`frozen benchmark spec identity`: its retained spec literal does not match the
builder's generated spec. This refusal remains required until D2 verifies
that the spec is reproduced from the approved configuration.

The cleaned production HTTP replay exercises client, producer, exporter,
receiver, reader and protected post-run acceptance for both 176-row arms.
For these post-run tests only, an in-memory seam sets the controller's spec
pin to the generated spec hash. The seam is never written to source and is
not used for the pre-launch test. Native/process observations use the retained
CPU substitutes. This proves post-run composition separately from the still
incomplete pre-launch/configuration path. It does not qualify new instrumented
health receipts, native execution, all-cap or doubled-menu evidence.

```sh
python3 -I -B -S serving/tests/test_cleaned_inventory_proposal.py
python3 -I -B -S serving/tests/check_cleaned_inventory_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

Local logs are under `.test-build/serving-cleanup/cleaned-inventory/`.
Six private checks pass across the initial run and a focused archive/grant retry.
The first grant test used the remote package root instead of the temporary
mirror path; the fixture is corrected and both attempts are retained.
[VERIFICATION.json](VERIFICATION.json) records the exact inventory, both replay
results, test-source versions and corrected check. The minimum observed evidence
margin is 1.59 times, below the required two times.
Complete config/spec consumers, capacity margins, native verification and
private publication remain open.
