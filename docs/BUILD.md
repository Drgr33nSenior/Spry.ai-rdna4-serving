# Build guide

The serving image is built by the workstation repository, not here. This
repository supplies the SGLang patch series that the build applies. This page
explains how the two connect, then keeps the build notes for the historical
SGLang 0.5.21 runtime.

## Current build

The workstation's
[AI-stack pipeline](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/README.md)
builds every component from upstream source, pinned in its
[`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock).

| Component | Pinned version (refreshed 2026-11-08) |
| --- | --- |
| SGLang | `main` at `fcb32746` (0.5.23.post957) |
| Patch series | This repository's [`sglang-main` series](../serving/patches/sglang-main/README.md), pinned by content hash `142d0ae7…` |
| ROCm | TheRock 10.3.0-dev (2026-11-08), HIP 7.18 |
| PyTorch | 2.16.0a0 nightly, built from `main` for that ROCm |
| Triton | 3.9.0, the commit PyTorch pins |
| Python | 3.14 (cp314) |
| Target | gfx1201, with SGLang's kernels built ahead of time |

The build runs these stages on the workstation:

1. `refresh.sh` moves the pins to the current upstream heads and recomputes
   the patchset hash from a checkout of this repository.
2. `sync.sh` copies the pipeline and this patch series to the workstation.
3. The `rocm`, `torch` and `sglang` stages build wheels. The `sglang` stage
   applies the patch series in name order and stops if any patch fails.
4. The `image` stage installs the wheels into the serving image, which is then
   referenced by its digest.

The serving flags and environment are in the workstation's
[serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json).
Its `qualification` list records which image and patchset each arm was
qualified on. Arm Y was qualified on 2026-11-10, and arms UQS and UQL on
2026-11-11. Rebuilding the image or changing the series needs a new
qualification.

To change a patch, follow the steps in the
[patch series guide](../serving/patches/sglang-main/README.md#change-a-patch).

## Historical builds

Everything below describes the SGLang 0.5.21 build. The November 2026
workstation reinstall removed its base image and built trees, so it cannot be
reproduced from this repository.

| Goal | Entry point | Required external inputs |
| --- | --- | --- |
| Assemble cleaned serving source for review | [Serving guide](../serving/README.md) and [caller projection](../serving/launcher/PROJECTION.md) | Authenticated sealed donor, selected proposal and configured assets |
| Reproduce the 0.5.21 overlays | [Patch series](../serving/patches/README.md) | Exact effective base files from the lock; that base image is lost |
| Build the KV4 native library | The KV4 build contract below | Complete pinned Linux toolchain, runtime sources and independent build review |

A source review bundle, a native library and an accepted runtime/build proof
are different outputs. The runtime pins below describe the recorded FP8/KV4
composition. Do not run historical experiment controllers or treat test
fixtures as proof.

### Shared runtime (0.5.21)

The [qualified native FP8 baseline](FP8.md) has a separate exact image/runtime
identity and stock BF16 KV path. Its profile is historical evidence, not an
image installer or a KV4 build receipt. Keep its graph-enabled qualification
separate from the eager comparison plans and the KV4 source overlay below.

Use SGLang 0.5.21 at commit
`e00930c5489053f26d86b179cee0d087f846acbb`. Obtain and verify the complete upstream
source separately. `patches/common-runtime.patch` contains the five shared
repairs; `common-runtime/SOURCE-MAP.json` binds their stock and effective hashes.
Apply them only to matching stock bytes. Then review the KV4 patch against the
same effective source. Do not apply either patch with fuzz or to a guessed image.

The recorded private base manifest/config digests and stack versions are in
`provenance/STACK-PINS.json`. That record is historical: it pins Torch 2.13,
ROCm SDK 10.1 and the FP8 model, not the later op234 stack. The hash-pinned Python build/runtime dependency
lists are in `common-runtime/locks/`. They are not a wheelhouse, Cargo vendor
closure, complete serving installation or permission to replace base packages.
The accepted private dependency exception was fsspec 2026.9.0 to 2026.7.0; it does
not authorise other Torch, Triton or ROCm replacements.

Build the AOT and serving wheels offline with authenticated sources and Cargo
locks/vendor inventory. Keep build products outside the source. The retained
private build used the AOT `setup_rocm.py bdist_wheel` path and the serving
`pyproject_other.toml` projection with `python -m build --wheel --no-isolation`.
Run dependency and effective-provenance checks afterward. A ModelRunner import
can query GPUs; it is not a device-free check.

### KV4 native build

The maintained producer is `prototype/scripts/build_bf16_v2.py`; the source
inventory and static protocol helper is `bf16_source_identity.py`. The collector
is `collect_bf16_toolchain.py`. The build driver requires explicit source, stock,
baseline/hash, fresh output, absolute tool identities and a future absolute
deadline. Its `--help` lists the arguments. It requires a complete 425-second
internal envelope, with outer collection and sealing time reserved separately.

Keep gfx1201, C++17, `-O2`, `-Wall`, `-Wextra`, `-Werror`,
`-ffp-contract=off` and `-fno-fast-math`. Do not substitute a host target or use
fast-math to make a build pass. Inspect both v1 and v2 exported symbols, actual
gfx1201 device code and architecture metadata. A v1 WMMA instruction does not
establish acceleration in the scalar BF16 v2 reader.

The producer emits unreviewed build facts. Independent review must bind the
actual tools, outputs, source and inspection evidence before an accepted schema2
proof can be assembled. Neither this repository nor a producer self-issues that
review. A fresh complete serving composition is still required by runtime gates.

Stock decode JIT must resolve the actual packaged SDK, not assume `/opt/rocm`.
The tested base used `/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel`
for `ROCM_HOME`/`ROCM_PATH`; that path was inside the base image lost in the
2026-11 reinstall and is historical. Cap configurable application and compiler
pools at two, keep caches private, and bind any JIT artifacts separately from
the KV4 library.
No native build or model test is performed by the CPU test commands.

The retained `tools/model_smoke.py` is the inner two-rank test entry only. It
requires an explicit execution switch, a pinned spec and complete build proof.
Do not run it without a separately reviewed outer supervisor that verifies
physical device handover, all model files, available memory/disk, deadlines and
independent release. No workstation SSH controller or private proof is published.
