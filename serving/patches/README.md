# Reproduce the active W5 overlays

The [lock](../upstream/sglang-0.5.21.lock) binds all 11 active runtime and graph/GEMM
overlays to their input, patch and output hashes. These are the retained image's
effective base files. Its runner and KV configurator already contain retained
changes; an unmodified SGLang release is not a substitute for those exact bytes.

Supply a source directory with the lock's `python/sglang/...` paths. The builder
reads those files, checks their hashes, and applies each patch to a temporary
copy with `/usr/bin/patch`, zero fuzz. It checks the resulting bytes against the
authenticated serving package. Neither upstream inputs nor sealed donors are
modified. The existing operating-system patch utility is a build-time tool;
no production dependency is added.

```sh
python3 -I -B -S serving/build.py /absolute/repository/serving/config/runs/w5-v2 \
  --donor /absolute/path/to/sealed-w5-preparation \
  --upstream-root /absolute/path/to/pinned-effective-base \
  --packet-id fedcba0987654321 \
  --output /absolute/path/to/new-source-review
```

The packet ID is an explicit deterministic input. Choose a fresh ID for each
planned session. It changes the workload root, session arguments, unit names
and their resource hashes. It does not create a launch manifest or approval.
Without `--upstream-root`, the compatibility path retains authenticated donor
overlay bytes. With it, every generated payload byte matches that path.

The CPU proof applies and reverses all 11 patches. Builder checks compare two
patched builds and an equivalent retained-overlay build, then run the actual
resource gate and owner argument projection. Invalid input hashes, patch hashes,
extra targets, path traversal, wrong order and failed context refuse. The
builder publishes no partial review bundle when rendering fails.

```sh
python3 -I -B -S serving/tests/test_patch_series.py -v
python3 -I -B -S serving/tests/check_patch_series_packet.py \
  --packet /path/to/sealed-w5-preparation \
  --upstream-root /path/to/pinned-effective-base \
  --report /path/to/new-report.json -v
```

[VERIFICATION.json](VERIFICATION.json) records the exact proof. The previous
runtime-only series is retained under `.test-build/serving-cleanup/patch-series/`.
Phase 5a reproduction is separate from replacing the runner or moving adapter
work before replay. Those Phase 5b changes still need implementation and native
token-identity validation.
