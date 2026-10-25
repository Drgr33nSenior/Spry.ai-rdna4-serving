# EAGLE-4 and lossy communication investigation

25 October 2026. Status: **source-validated feasibility only**. New component,
serving and hardware tests: **NOT RUN**. No dependency, precision, selected
configuration or deployment changed.

The preceding [host cleanup comparison](HOST-PATH-CLEANUP.md) completed and was
published as `b0149ecd2ddcb68bdab35bce28ae114340e3aa04`. It did not establish a
useful speed improvement. This investigation is separate from that measured A/B.

## Dispositions

| Candidate | Finding | Disposition |
| --- | --- | --- |
| EAGLE-4 | Matching draft exists; released kernel admission excludes R9700 `gfx1201` | **PARKED by owner**; no installation or model test |
| SGLang QuickReduce compression | Consumed wrapper admits `gfx94`/`gfx95`, not `gfx1201` | **Unsupported by current admission**; no flag-only test |
| SGLang `enable_quant_communications` | Quantized implementation is NPU-only; other devices use ordinary reduction | **Not an AMD lossy implementation** |
| RCCLX FP8 collectives | Separate implementation with Instinct evidence; current R9700 build and route unqualified | **Unselected; portability and exposure gates unmet** |
| FlashInfer quantized all-reduce | Requires SM90+ and NVSwitch | **Unsupported on these GPUs** |
| Echo 4-bit communication | A100 research evidence, additional compensation state; no verified R9700 integration | **Research lead only** |

Lossy tensor-parallel communication has **not** been tested or ruled out on this
workstation. Earlier collective work measured neither a lossy candidate nor the
current full-model exposed transport cost. Its model-free traces and corrected
offline parser remain limited evidence, not a successful serving comparison.

## EAGLE-4: retained findings, now parked

The source examined was `eagle-hq/eagle-4` at
`60bab6b469e795d47bda82b16243b82bb7b6206f`. Its
[hardware admission](https://github.com/eagle-hq/eagle-4/blob/60bab6b469e795d47bda82b16243b82bb7b6206f/eagle_harness/kernels/arch.py)
accepts AMD `gfx90a`, `gfx942` and `gfx950`; it rejects `gfx1201`.
The released AMD recipe uses a patched SGLang engine, a Python 3.10 ABI and
ROCm 7.2 dependencies. Our Python 3.14/Torch 2.14/SDK 10.2 environment is a
different integration target. Changing the architecture allowlist would not
establish correct kernels, tree attention or recurrent-state commits.

The [Qwen3.8-27B draft](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4/tree/f58a5985223040552ad404fba98f0dd27f5f676a)
is distinct from DFlash2. Published model-card comparisons include DFlash2 on
four B300 GPUs; the AMD recipe uses BF16 target/state and MI355X TP1. Neither
qualifies our FP8 target, FP32 state/head, R9700 TP2 path. No accuracy regression
was measured here. The owner's accuracy concern remains unverified; the owner
subsequently directed that EAGLE-4 be parked. No further EAGLE-4 work is queued.

## Communication paths checked

A bounded read-only observation retrieved the current QuickReduce wrapper from
the workstation and authenticated its parent `parallel_state.py` against the
retained source identity. No framework imports, model starts or GPU access
occurred. The wrapper SHA256 is
`ca29d8d012aa612f9349bd66cca47d9362aa0709101b6c0386be50fade080838`.

The wrapper checks architecture before reading the compression mode. Therefore
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, `INT6` or `INT4` cannot activate it on
R9700 as written. It defaults to `NONE`; it also defaults to casting BF16 inputs
to FP16 when active. The existing native-BF16 two-rank compressed thresholds are
8 MiB; the FP16-cast thresholds are 1–2 MiB. An upstream
[Q8/FP16 scale report](https://github.com/sgl-project/sglang/issues/40084)
describes severe low-amplitude error on another model and Instinct system.
That report is a reason to include small-value and cancellation operands in a
future codec test, not an explanation of our historical mismatches.

RCCLX is a separate source implementation. At TorchComms revision
`44c4a0fe7d3519ef0ec6c547689a7790cae3b494`, its FP8 all-reduce quantizes inputs,
reduces in float, requantizes the sum and returns the requested output type.
Its buffer pool must exist before graph capture. The
[published AMD evidence](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/)
targets MI300/MI350 and large messages, at least 16 MB. The source flag
`RCCL_LOW_PRECISION_ENABLE` does not prove that our installed RCCL 2.32.3 contains
this extension or that SGLang's selected communicator calls it. Neither was
established; no library replacement or flag change was made.

[FlashInfer's FP8 collective](https://docs.flashinfer.ai/generated/flashinfer.comm.quantized_all_reduce.html)
requires NVIDIA SM90+ and NVSwitch. It is not a portable R9700 implementation.
[Echo](https://arxiv.org/html/2610.14277v1) reports large-message and first-token
benefits on A100. Its compensation averages reset each forward pass and span
reduction sites within that pass. An implementation here would need graph-safe
reset and row ownership, not persistent feedback across unrelated requests.
No Echo implementation was compiled or evaluated here.

## Where a future test could pay

The retained target configuration has hidden width 5,120. An ordinary BF16
activation of shape `[rows, 5120]` has these sizes per collective input:

| Rows | Example | Input bytes |
| ---: | --- | ---: |
| 8 | C1 width-eight verification | 81,920 (80 KiB) |
| 16 | C2 width-eight verification | 163,840 (160 KiB) |
| 2,048 | Selected prefill chunk | 20,971,520 (20 MiB) |

These are shape-derived estimates, not captured collective shapes, total wire
traffic or measured bottlenecks. Padding, other tensors and the actual
communicator can change the comparison. They favour investigating prefill first;
compression overhead can outweigh savings for small decode messages.

Historical all-reduce intervals included waiting for the other rank. They do
not separate rank-arrival skew from transport or establish an end-to-end upper
bound. The small final vocabulary gather is not all communication.

The next useful gate is an exact-route observation at current prefill and
verification shapes, followed by a both-card component comparison only if
transport is exposed. A codec candidate must preserve BF16 input/output and
avoid an incidental FP16 cast, account for scales and extra buffers, and test
outliers, small values, opposing signs, eager execution and graph replay.
Model tests additionally require an explicit numerical-error acceptance contract
and the existing task/tool/long-context checks. Previous tolerance of numerical
variation does not authorize a new lossy transport contract. A port, model test
and selection remain **NOT RUN**; no performance gain is claimed.

[Source inventory and disposition](../provenance/EAGLE4-LOSSY-COMMUNICATION-20261025.json)
retain exact source identities and local evidence locations. The selected
runtime and rollback remain those in the host cleanup report. No device process
or reservation remains from this investigation.
