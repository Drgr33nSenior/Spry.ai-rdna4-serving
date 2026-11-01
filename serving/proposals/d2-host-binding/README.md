# D2 proposal: observed rank and device binding

This increment extends [stop policy](../d2-stop-policy/README.md).
[SOURCE.patch](SOURCE.patch) records the canonical startup and readout changes.
[D2.patch](D2.patch) replaces the protected controller's copied device map with
its authenticated host configuration. The original nine protected helpers are
still emitted by the builder.

Before scheduler entry, production and instrumented modes now check the pinned
HIP library and physical device identities. Previously the pre-weight check
ran only with instrumentation. Rank binding joins the configured HIP index to
the observed BDF, UUID and render/sysfs identity. JSON key order does not assign
a rank. Invalid indices, duplicate observations, wrong identities and expired
deadlines refuse before scheduler delegation.

Production performs the startup invariant checks without importing the graph
observer or writing a pre-weight evidence record. Instrumented mode retains
its startup record and observer/preflight dispatch. Its record now contains
`rank_binding`; the new readout requires an exact typed match to the observed
devices and pinned worker manifest. Historical packets retain their own reader.

## Verification and limits

Eight public tests use CPU substitutes for Torch/HIP and sysfs observations.
They exercise the actual scheduler entry, prove identity checking precedes one
delegation, and check production observer absence at that boundary. Readout
refuses stale BDFs, UUIDs and boolean/integer aliases.

The complete CPU packet suite passes 21 checks, including both baseline response
replays, the declared-cap consumer fixture, actual owner release decisions and
configured sealing. The actual resource gate and import chain remain covered.
No physical device is queried by these tests.

```sh
python3 -I -B -S serving/tests/test_host_binding.py -v
python3 -I -B -S serving/tests/check_host_binding_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

[VERIFICATION.json](VERIFICATION.json) records exact pins and results. Logs are
under `.test-build/serving-cleanup/host-binding/`. Complete production mount/import
isolation, caller power binding, instrumented evidence composition, native token
identity and launch approval remain open.
