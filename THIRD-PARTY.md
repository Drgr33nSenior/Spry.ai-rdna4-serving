# Third-party notices

The root MIT licence applies to original project work. It does not replace the
terms or copyright notices in copied third-party files.

## SGLang and vLLM-derived source

`prototype/image-source/`, `common-runtime/overlay/`, the SGLang patches and stock
source fixtures contain SGLang code and adaptations. Retain their file-level
copyright and SPDX notices. The Apache-2.0 terms are included in
`LICENSES/SGLang-Apache-2.0.txt`. Some SGLang files identify vLLM contributors and
adaptation URLs; these notices remain unchanged.

The current `serving/patches/sglang-main/` series targets SGLang `main` at
commit `fcb3274646e43c16d70fd83806ebb3a5134d5e81` from
[sgl-project/sglang](https://github.com/sgl-project/sglang).
`prototype/image-source/`, `common-runtime/overlay/`,
`serving/patches/sglang-0.5.21/` and the stock fixtures use the historical
SGLang 0.5.21 baseline, commit `e00930c5489053f26d86b179cee0d087f846acbb`.
`prototype/image-source/` is a modified, selected source overlay. It adds the
private KV4 contract, backend and ownership integration. `patches/sglang-kv4.patch`
and the provenance records identify modifications and added files.
`common-runtime/overlay/` contains the five separately recorded shared runtime
repairs, not a complete upstream checkout.

Historical fixtures retain their original source bytes and identities. They can
contain earlier prototype modifications and are not the current serving source.
Their source/import hashes are recorded in `provenance/SOURCE-IMPORT.json`.

## AMD AITER-derived kernels

`serving/patches/sglang-main/0004-gfx1201-mxfp4-without-aiter.patch` adds
`gfx1201_mxfp4_kernels.py`, a Triton MXFP4 activation quantizer and FP4 GEMM
derived from [ROCm/aiter](https://github.com/ROCm/aiter) commit
`32214c295ea2eec1055bdb3467d2eee7289257be`. The file keeps its
`SPDX-License-Identifier: MIT` line and the notice
`Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`
The MIT terms are included in `LICENSES/AITER-MIT.txt`.

Patches 0010, 0019 and 0020 reuse the vendored AITER MXFP4 quantizer from 0004
inside new kernels. Their new files carry
`SPDX-License-Identifier: Apache-2.0`.

## Flash Linear Attention (FLA)-derived kernel

`serving/patches/sglang-main/0010-gfx1201-gdn-norm-quant.patch` adds
`gfx1201_gdn_norm_quant.py`. Its docstring says the kernel body is a copy of
SGLang's in-tree FLA gated layer-norm forward
(`kernels/ops/attention/fla/layernorm_gated.py`), with the store replaced by the
AITER quantizer. The new file carries only `SPDX-License-Identifier: Apache-2.0`.
It has no separate FLA copyright line. Upstream
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) is
MIT-licensed. Whether SGLang's copy carries an FLA notice that should be kept
here was not checked against the SGLang source; check it before the next
release of this series.

## BetterBench

`benchmarks/betterbench/upstream/` and `benchmarks/betterbench/native-v2/upstream/`
contain [BetterBench](https://github.com/GGZ14/BetterBench) v0.6.1, commit
`f72bb9011a830fd98e82ab89d01e549efa76e9c5`, under the Apache-2.0 licence. Each
copy keeps the upstream `LICENSE` file. The native-v2 client carries an
explicit vendor patch (`vendorclient.patch`); the other files in
`benchmarks/betterbench/` are this project's adapter, under the root MIT licence.

## Qwen config fixture

The small configuration fixture is from `Qwen/Qwen3.8-27B-FP8`, model revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its retained licence is
`prototype/tests/fixtures/Qwen3.8-27B-FP8.LICENSE`; copies inside historical
fixtures retain the same licence. No model weight files are distributed.

## Research attribution

The local four-bit cache format and rotation investigation were informed by
[UltraQuant v3](https://arxiv.org/html/2606.20474v3). The local implementation is
not claimed to be the authors' bit-exact implementation. Source-derived
comparison vectors do not imply execution of an external Python or Triton path.

RDNA4 matrix-core investigation used the primary
[AMD RDNA4 matrix-core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).
The current BF16 reader is a software-unpack reference path; paper CDNA4 results
and unrelated v1 WMMA instructions are not evidence of native FP4 acceleration.
