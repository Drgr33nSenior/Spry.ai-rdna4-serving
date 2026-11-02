# D2 proposal: artifact limits from the authenticated spec

This increment extends [host binding](../d2-host-binding/README.md). The builder's
`--configured-artifacts` option projects the 16 artifact primaries from the run
configuration into `http_serving.artifact_limits`. The protected controller
reproduces that field from its verified plan, then compares the complete spec
hash. Altered, missing or extra policy fields refuse. No sandbox configuration
file or runtime dependency is added.

[SOURCE.patch](SOURCE.patch) records the canonical builder, spec loader, worker
entry and readout changes. [D2.patch](D2.patch) records the protected controller
change. The original nine protected helpers remain the builder's default output;
this proposal must be composed and reviewed before launch. Historical review
builds omit the new projection unless the option is supplied.

The worker snapshots the validated flat primaries before loading helpers,
constructs the protected immutable plan before scratch creation, and passes that
same object to pack, stage and export. A missing configured helper or invalid
owner limit refuses before scratch preparation. Changed downstream mappings
cannot extend the export plan. Configured raw-frame export remains unsupported.

Both readout modes accept the verified primaries. They enforce encoded member
sizes, decoded member sizes, tree bytes and file counts. Configured client
limits are checked before extraction. Existing independent reader bounds still
apply, so a policy cannot raise those bounds. The controller passes its verified
plan into readout. Outer durable partitions remain a separate open consumer.

## Verification

Six public tests cover the actual worker entry with CPU runtime/export stand-ins,
snapshot behavior, early refusal and bounded readout. The private packet fixture
passes the spec-derived plan through the actual pack, stage, export, receive
and readout APIs. Its exact adaptation is in
[FIXTURE-ARTIFACT-BINDING.patch](FIXTURE-ARTIFACT-BINDING.patch). Production
validators are not replaced.

For owner decision equivalence, the retained owner receives its original spec
shape with only the new generated field omitted. The proposed owner receives
the new shape. Both execute their actual validation and release logic. Earlier
failed fixture attempts remain retained.

```sh
python3 -I -B -S serving/tests/test_artifact_binding.py -v
python3 -I -B -S serving/tests/check_artifact_binding_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

[VERIFICATION.json](VERIFICATION.json) records the observed results and exact
source hashes. Whole-run configuration mutations, outer durable partitions,
complete instrumented evidence and native validation remain open. No inference
performance improvement is claimed.
