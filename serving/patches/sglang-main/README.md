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
| `0006-gfx1201-mxfp4-verify-gemm.patch` | gfx1201 MXFP4 GEMM launches for DFLASH verify rows (M8/M16), each off unless its variable is `1`. `SGLANG_GFX1201_SPLITK_DOWNOUT`: op234's S4 split-K for down (5120x8704) and output (5120x3072) shards, four K partitions into a shared 1.25 MiB FP32 buffer and a fixed-order reducer; falls back to one partition if the buffer would be allocated inside graph capture. `SGLANG_GFX1201_GATE_UP_WAVE0`: `waves_per_eu=0` for gate/up (17408x5120). Measured on R9700 (2026-11-09): split-K down 88 to 67 us, output 34 to 30 us per call, outputs bitwise equal; wave0 2 us slower, so leave it off. | SGLang `main` `fcb32746` |
| `0007-gfx1201-packed-gdn-in-proj.patch` | `SGLANG_GFX1201_PACKED_GDN_IN_PROJ=1`: one MXFP4 quantize and GEMM for the 48 GDN QKVZ (8192 rows) and BA (48 rows) input projections on M8/M16 rows (op234 operation 164). The original parameters become views of the packed rows, so no weight is duplicated. Outputs are bitwise equal to the separate projections; measured 101 to 76 us (M8) and 92 to 76 us (M16). Also passes `prepare_before_cuda_graph_capture` through the dense `Qwen3_5ForConditionalGeneration` wrapper, which upstream omits, so pre-capture hooks reach the language model. | SGLang `main` `fcb32746` |
| `0008-dflash-gdn-replayssm-commit.patch` | Lets `--enable-linear-replayssm-spec` run DFLASH on GDN models. Upstream refuses it (`kv_cache_configurator.py:971`) because DFLASH commits through the backend's KDA-only ReplaySSM handler. This routes DFLASH's GDN commit through `commit_mamba_states_after_verify`, upstream's GDN compact-replay commit (cursor advance, checkpoint fold, conv rollback, track snapshots), and relaxes the guard for DFLASH with KDA or GDN. DSPARK stays KDA-only. | SGLang `main` `fcb32746` |
| `0009-gfx1201-fp8-lm-head.patch` | `SGLANG_GFX1201_FP8_LM_HEAD=1`: a block-128 FP8 copy of the dense BF16 LM head (op234 D3b), built before graph capture and used for target verify rows (8, 16; FP32 output) and the draft's candidate projection (7, 14; BF16 output). One copy serves both because the draft borrows the target head. Uses upstream's per-token-group quantizer and block FP8 matmul with op234's small-row tile. Measured 2.04 ms to 1.12 ms per call (124160x5120 shard). Lossy: on random data max logit error about 2% of the logit range and top-1 agreement 93 to 100%, so qualify with the bench quality phase. | SGLang `main` `fcb32746` |
| `0010-gfx1201-gdn-norm-quant.patch` | `SGLANG_GFX1201_GDN_NORM_QUANT=1`: fused GDN output gated RMSNorm and MXFP4 quantization (op234 F1) on 8- and 16-token rows. The kernel is op234's copy of the FLA gated-norm forward with its store replaced by the vendored AITER MXFP4 quantizer. It rounds to BF16 before quantizing, so `out_proj` receives the same packed values and scales the unfused path would produce. It is the MXFP4 counterpart of upstream's FP8 tuple path in the same function. Measured on R9700 (24 heads x 128): 54.5 to 21.1 us per layer at 8 tokens, bitwise equal. | SGLang `main` `fcb32746` |
| `0011-gfx1201-route-markers.patch` | Logging only: one INFO line per shape when the split-K verify GEMM (0006) or the fused GDN norm-quant (0010) route runs, and, when 0010 is enabled but skipped for a decode- or verify-sized input, one line per shape giving the shapes, dtypes, strides and `out_proj` scheme it saw. Neither route logged before, so a bench could not show whether it engaged. No behaviour change. |
| `0012-gfx1201-verify-attention.patch` | `SGLANG_GFX1201_VERIFY_ATTENTION=1`: run DFLASH/EAGLE target verify through upstream's gfx1151 split-KV verify kernel (`verify_attention_gfx1151`, written for Qwen3.8 shapes) on gfx1201, with 32 splits instead of gfx1151's 20, and log its route. Upstream gates the kernel to gfx115x, so verify on gfx1201 ran `extend_attention_fwd`, which walks the whole prefix serially. On the R9700 with TP2 Qwen3.8 shapes (12 q heads, 2 kv heads, head dim 256, 8 draft tokens) it takes 0.029 / 0.047 / 0.100 / 0.249 / 0.449 ms per layer at 1k / 8k / 23.5k / 47k / 88k prefixes against 0.255 / 1.64 / 4.38 / 8.98 / 16.7 ms, max abs difference 1e-3 (BF16 rounding; not bitwise). Also drops the triton backend's now-unused `use_gfx115_kernels` import. |

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
| S4 split-K down/out, gate/up wave hint | `0006`, opt-in |
| Packed QKVZ+BA GDN projection (operation 164) | `0007`, opt-in |
| D3b FP8 target and draft heads | `0009`, opt-in, one shared FP8 copy |
| F1 gated-norm and MXFP4 quantize fusion | `0010`, opt-in |
| Verify prefetch / split-KV adapter, cap16, P1 prefill | Not carried yet: the adapter matters only at long context; measure it with the bench's long-context phase first |
| C4 extend attention | Not carried |
| Ring-fold/ReplaySSM spec path | `0008` enables upstream's GDN compact replay for DFLASH instead of op234's fold |
| P3 fused Gemma RMSNorm | Not needed: upstream's `minimax_m3_rmsnorm` fused kernel serves gfx1201 without AITER |
| KV4/UltraQuant library | Not carried |
