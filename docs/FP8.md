# Native FP8 baseline

The official Qwen3.8-27B-FP8 checkpoint passed its frozen C1/C2 compatibility
contract at 4096 tokens on two Radeon AI PRO R9700 GPUs. This is a native
SGLang FP8 baseline with BF16 remaining tensors and stock BF16 K/V storage.
It does not use the KV4 overlay or convert weights to FP4.

## Retained implementation

[The qualified profile](../profiles/fp8-bf16-kv-4k-qualified.json) binds the exact
image, private interpreter, runtime versions, model revision and source pins.
The shared build/import repairs already retained under `common-runtime/` remain
separate from the KV4 source overlay. This addition does not change those files,
SGLang arithmetic or the installed runtime.

The historical image manifest is
`f2f0d8b134b7650b4ead961fac6c23751e11d98757fe3f7233be11c817948945`, with config
`f10ff719d78e659dff0e5c02ecae61b3aad45a8d06d296210614417d344bfa06`.
It used `/opt/workstation/sglang/bin/python3`, not the base image's generic
Python. SGLang 0.5.21, sglang-kernel 0.4.7, Torch 2.13.0+rocm10.1.0,
HIP 7.16.26385, Triton 3.8.0+git669b31ac.rocm10.1.0 and RCCL 2.30.7 are pinned.

The model revision is `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
[The model inventory](../provenance/FP8-MODEL-INVENTORY.json) retains all 81
payload names, sizes and SHA256 values: 30,890,049,597 bytes in total.
The checkpoint selects serialized blockwise FP8 weights. No forced alternative
quantization, mixed-V codec, KV4 storage or speculative decoder was enabled.

The tested settings were TP2, context 4096, memory fraction 0.80, maximum two
requests, Triton full/GDN attention, `no_buffer`, and disabled overlap and custom
all-reduce. Native defaults resolved to chunk 4096 and page size 1. Both ranks
captured graphs for batches `[1,2]`; the parsed graph maximum was 48. The native
KV pool had 182559 slots per rank: context length is not total KV capacity.

## Evidence and limits

[The sanitised summary](../provenance/FP8-BASELINE-SUMMARY.json) records the
bounded result and original 375-member evidence seal. Raw archives, credentials,
machine paths, weights, caches and native binaries are not distributed here.
Historical identity pins are not current device ownership or a transferable
runtime/build review receipt.

The same-flush 128/512 repeats matched generated IDs and aligned logprobs
exactly. All 512 teacher logprobs across four document cells matched between C1
and concurrently offered C2, giving per-cell and overall ratios of 1.0.
Eight gated functional tasks passed per phase; two code tasks stayed `UNGRADED`.
Actual batch-2 decode was observed. Teacher scoring forwards stayed singleton,
so the result does not establish batched-scoring invariance.

The attempt used 56 generations, including one intended cancelled stream,
12 flushes, one abort and one health control. Cancellation, recovery, exact-UID
disposal and independent release passed. Native parent drain/exit was observed;
independently graceful distributed-rank destruction remains unqualified.

This result does not establish 32K support, general task quality, BF16 fidelity,
deployment readiness, KV4 compatibility or a speed ranking. The original generic
`qualification=false` flags remain unchanged. The earlier old-stack 512-repeat
failure remains a failed result; this new-stack pass does not explain it.
RCCL graph-usage warnings, default FP8 GEMM configurations and lazy compilation
remain limitations. Request walls are not pure kernel timings.

## Separate comparison contracts

`profiles/fp8-bf16-kv.json` remains the **eager** control plan for a matched KV4
comparison. `profiles/fp8-kv4.json` remains the experimental KV4 profile.
Neither inherits the new graph-enabled baseline's qualification. Their matched
serving benchmark is still unrun. Do not use the new baseline's timings as a
control for different prompts, graphs, cache geometry or concurrency.

## Local verification and future use

Run the new metadata checks with an existing Python 3.11+ interpreter:

```sh
make test-fp8 PYTHON=/path/to/python3
```

These checks validate retained identities and scope; they do not import Torch
or SGLang, load a native library, query devices or rerun the model. The existing
112-case KV4 CPU suite remains unchanged.

This checkout is still a source prototype, not an installable image. A future
launch requires a complete reviewed runtime/build, the exact model inventory,
fresh conflict-free two-GPU admission, finite deadlines and independently
verified cleanup. Do not execute historical controllers or treat this profile
as authority to launch. Keep the image unpublished and retain all previous
models, images, caches and evidence. Rollback is non-adoption of this profile.
