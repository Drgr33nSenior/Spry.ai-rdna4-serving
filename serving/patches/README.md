# Reproduce the historical 0.5.21 W5 overlays

Status, 9 November 2026: this series is historical. The workstation AI-stack
pipeline now applies the separate [`sglang-main`](sglang-main/README.md) series
to SGLang `main`. The `--upstream-root` input below is the retained 0.5.21
image's effective base. The 2026-11 workstation reinstall lost that image, so
this reproduction cannot be rerun from this repository alone.

The [lock](../upstream/sglang-0.5.21.lock) binds all 11 retained 0.5.21 runtime and graph/GEMM
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
runtime-only series was kept in a local build directory that is not in this
repository. This reproduction record covers Phase 5a. The later runner post-load hook and
metadata preparation before replay are separate increments. They were included
in the completed [cleanup validation](../measurements/native-qualification/CLEANUP-SIGNOFF.md).
Their earlier preparation and failure records remain in the
[implementation history](../HISTORY.md).
