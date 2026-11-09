# SGLang main patch series

Patches the workstation AI-stack pipeline applies to SGLang `main` before it
builds the gfx1201 serving image. The build applies every `*.patch` here in
name order with `git apply` (no fuzz) and stops if any patch fails.

The pipeline's [`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock)
pins the SGLang commit and this series' content hash,
`SGLANG_PATCHSET_SHA256`, not a commit of this repository.
[`patchset-hash.sh`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/patchset-hash.sh)
computes that hash over the `*.patch` files in name order. After adding,
removing or editing a patch, rerun `patchset-hash.sh` on this directory and
update `SGLANG_PATCHSET_SHA256` in `stack.lock` (`refresh.sh` also recomputes
it when it moves every pin). `sync.sh` and the sglang build stage stop when
the series and the pinned hash differ.

| Patch | Purpose | Last applied cleanly to |
| --- | --- | --- |
| `0001-gfx1201-aot-build.patch` | Allow gfx1201 in the AOT kernel build, require an explicit `AMDGPU_TARGET`, and build with wave32 (`SGLANG_WAVEFRONT_SIZE=32`). Ported unchanged from the 0.5.21 common-runtime overlay. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0002-rocm-aot-cxx20.patch` | Compile the ROCm AOT kernels as C++20. Torch 2.14+ headers stop with `#error` below C++20; upstream's CUDA build already uses C++20, but `setup_rocm.py` still passes `-std=c++17`. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0003-outlines-python314.patch` | Skip `outlines` on Python 3.14 in `runtime_common`. Its pinned `outlines_core==0.1.26` has no 3.14 wheel and current `outlines` requires Python below 3.14. SGLang imports it only for `--grammar-backend outlines`; xgrammar stays the default, as in the qualified 0.5.21 runtime, which shipped no outlines. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0004-gfx1201-mxfp4-without-aiter.patch` | Serve dense Quark MXFP4 on gfx1201, which has no AITER build. Adds `gfx1201_mxfp4_kernels.py`, the qualified op234 runtime's AITER-derived Triton quantizer and FP4 GEMM (MIT, from ROCm/aiter `32214c29`, byte-identical, sha256 `5571b106…`), and `gfx1201_mxfp4.py` with that runtime's launch geometry. When `aiter` is not installed and the GPU is gfx1201, `quark_w4a4_mxfp4.py` uses them; AITER-only routes raise `NotImplementedError`. The MXFP4 and INT4-FP8 MoE schemes import without AITER; `ON_GFX950` stays defined on every HIP host. Upstream otherwise imports AITER at module load on HIP, which stops the scheduler from importing. | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |
| `0005-pillow13.patch` | Require Pillow 13 (`>=13.0.0,<14`) in `runtime_base`. Pillow 12.3.0 has five CRITICAL CVEs fixed in 13.0.0 (CVE-2026-112393, -112398, -112403, -112900, -96639), which blocked the image's scan policy. Upstream caps `<13` because its Rust image pipeline, used only by the inkling and deepseek_v41 processors, reproduces Pillow 12 bicubic. Qwen3.8's processor does not use it: `pixel_values` and decoded JPEG/PNG/WebP pixels were bit-identical under Pillow 12.3.0 and 13.0.0 (2026-11-09). | `fcb3274646e43c16d70fd83806ebb3a5134d5e81` |

The `sglang-0.5.21/` series targets the retained 0.5.21 image's effective
files and is kept as history. Port a change from it only after checking that
upstream `main` still lacks it; upstream now ships DFlash v2, Qwen3.5 and the
Quark MXFP4 schemes. When upstream absorbs a patch, delete it here and note
the upstream commit in the commit message.

## Carried and not carried from op234

Status, 9 November 2026: the image built with this series is unqualified.
Op234's measured quality and throughput were taken on its SGLang 0.5.21
runtime and do not carry over.

| Op234 component | In this series |
| --- | --- |
| gfx1201 AOT build and wave32 (0.5.21 common-runtime overlay) | Carried as `0001` |
| AITER-derived MXFP4 quantizer and FP4 GEMM (`kernels.py`, sha256 `5571b106…`) | Carried byte-identical as `0004` |
| Eleven 0.5.21 W5 overlays (`sglang-0.5.21/`) | Not carried |
| Private-package performance members: S4 split-K, verify prefetch, F1 fusion, wave hint, cap16, packed QKVZ, P1 prefill | Not carried |
| C4 extend attention | Not carried |
| Ring-fold/ReplaySSM spec path | Not carried |
| KV4/UltraQuant library | Not carried |
