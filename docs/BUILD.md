# Build and source-assembly guide

Choose the path that matches your work:

| Goal | Entry point | Required external inputs |
| --- | --- | --- |
| Build the current serving image | [Current build](#current-build) and the [`sglang-main` series](../serving/patches/sglang-main/README.md) | The workstation AI-stack pipeline and its `stack.lock` |
| Assemble cleaned serving source for review | [Serving guide](../serving/README.md) and [caller projection](../serving/launcher/PROJECTION.md) | Authenticated sealed donor, selected proposal and configured assets |
| Reproduce the historical 0.5.21 overlays | [Patch series](../serving/patches/README.md) | Exact effective base files from the lock; that base image is lost |
| Build the separate KV4 native library | The KV4 build contract below | Complete pinned Linux toolchain, runtime sources and independent build review |
| Run local CPU checks | [Verification](VERIFICATION.md) | Existing configured Python and C++ compiler |

A source review bundle, a native library and an accepted runtime/build proof
are different outputs. Follow the selected path through its own checks. The
runtime pins below describe the recorded FP8/KV4 composition; they are not a
universal installation recipe for every later experiment.

This repository does not install dependencies, pull an image or start a GPU.
Use an isolated owned output directory and a separately reviewed execution plan.
Do not run historical experiment controllers or treat test fixtures as proof.

## Current build

Status, 9 November 2026: the serving image is built from SGLang upstream `main`
at `fcb3274646e43c16d70fd83806ebb3a5134d5e81` (about v0.5.23 plus 957 commits)
by the workstation repository's
[AI-stack pipeline](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/README.md). Its
[`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock) pins Python 3.14
(cp314), PyTorch 2.16.0a0 built from `main`, TheRock ROCm 10.3.0.dev20261108
(HIP 7.18), Triton 3.9.0 and sglang-kernel 0.4.10 built ahead of time for
gfx1201. The pipeline applies the
[`sglang-main` patch series](../serving/patches/sglang-main/README.md), which
`stack.lock` pins by content hash rather than by a commit of this repository.

The serving flags and environment are in the workstation
[serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json);
it is unqualified. The sections below describe the historical SGLang 0.5.21
build. The 2026-11 workstation reinstall removed its base image and built
trees, so that build cannot be reproduced from this repository as documented.

## Shared runtime (historical 0.5.21)

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

## KV4 native build

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
