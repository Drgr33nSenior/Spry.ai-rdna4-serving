# SGLang main patch series

Patches the workstation AI-stack pipeline applies to SGLang `main` before it
builds the gfx1201 serving image. The pipeline's `stack.lock` pins the SGLang
commit and this repository's commit; the build applies every `*.patch` here in
name order with `git apply` (no fuzz) and stops if any patch fails.

| Patch | Purpose | Last applied cleanly to |
| --- | --- | --- |
| `0001-gfx1201-aot-build.patch` | Allow gfx1201 in the AOT kernel build, require an explicit `AMDGPU_TARGET`, and build with wave32 (`SGLANG_WAVEFRONT_SIZE=32`). Ported unchanged from the 0.5.21 common-runtime overlay. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |

The `sglang-0.5.21/` series targets the retained 0.5.21 image's effective
files and is kept as history. Port a change from it only after checking that
upstream `main` still lacks it; upstream now ships DFlash v2, Qwen3.5 and the
Quark MXFP4 schemes. When upstream absorbs a patch, delete it here and note
the upstream commit in the commit message.
