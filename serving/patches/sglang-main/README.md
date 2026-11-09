# SGLang main patch series

Patches the workstation AI-stack pipeline applies to SGLang `main` before it
builds the gfx1201 serving image. The pipeline's `stack.lock` pins the SGLang
commit and this repository's commit; the build applies every `*.patch` here in
name order with `git apply` (no fuzz) and stops if any patch fails.

| Patch | Purpose | Last applied cleanly to |
| --- | --- | --- |
| `0001-gfx1201-aot-build.patch` | Allow gfx1201 in the AOT kernel build, require an explicit `AMDGPU_TARGET`, and build with wave32 (`SGLANG_WAVEFRONT_SIZE=32`). Ported unchanged from the 0.5.21 common-runtime overlay. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0002-rocm-aot-cxx20.patch` | Compile the ROCm AOT kernels as C++20. Torch 2.14+ headers stop with `#error` below C++20; upstream's CUDA build already uses C++20, but `setup_rocm.py` still passes `-std=c++17`. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0003-outlines-python314.patch` | Skip `outlines` on Python 3.14 in `runtime_common`. Its pinned `outlines_core==0.1.26` has no 3.14 wheel and current `outlines` requires Python below 3.14. SGLang imports it only for `--grammar-backend outlines`; xgrammar stays the default, as in the qualified 0.5.21 runtime, which shipped no outlines. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0004-gfx1201-mxfp4-without-aiter.patch` | Serve dense Quark MXFP4 on gfx1201, which has no AITER build. Adds `gfx1201_mxfp4_kernels.py`, the qualified op234 runtime's AITER-derived Triton quantizer and FP4 GEMM (MIT, from ROCm/aiter `32214c29`, byte-identical, sha256 `5571b106…`), and `gfx1201_mxfp4.py` with that runtime's launch geometry. When `aiter` is not installed and the GPU is gfx1201, `quark_w4a4_mxfp4.py` uses them; AITER-only routes raise `NotImplementedError`. The MXFP4 and INT4-FP8 MoE schemes import without AITER. Upstream otherwise imports AITER at module load on HIP, which stops the scheduler from importing. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |

The `sglang-0.5.21/` series targets the retained 0.5.21 image's effective
files and is kept as history. Port a change from it only after checking that
upstream `main` still lacks it; upstream now ships DFlash v2, Qwen3.5 and the
Quark MXFP4 schemes. When upstream absorbs a patch, delete it here and note
the upstream commit in the commit message.
