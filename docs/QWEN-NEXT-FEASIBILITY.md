# Qwen3.8-Flash-Next feasibility

The official FP8 checkpoint cannot fit fully resident in the combined physical host RAM and GPU memory observed on this workstation. Its indexed tensors total 185,502,232,570 bytes, versus 135,129,354,240 bytes of combined physical memory before runtime allocations. The official BF16 checkpoint has 359,999,963,128 indexed tensor bytes. These are metadata figures, not measured loading or runtime memory. NVMe offload and lower-bit checkpoints have not been proved viable or impossible; the owner's checkpoint and memory-route decision remains pending.

The existing campaign host limit is 40,802,189,312 bytes and was not raised. The observation recorded zero swap and 954,838,409,216 bytes of free disk. Disk capacity alone does not prove an offload path fits the host limit or performs acceptably. No Next weights were loaded, no GPU model run occurred, and the installed stack was not changed.

The pinned SGLang 0.5.21 runtime files contain model support and explicit ROCm JIT and HIP Triton QSA decode paths. Source presence is not proof of compilation, correctness or performance on gfx1201. The source assessment also found a host-page-table capability guard in the file-backed PLE path; it was not bypassed or tested. The [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [Qwen README](https://github.com/QwenLM/Qwen3.8-Flash-Next), and [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next) describe the model and supported configurations. The cookbook lists AMD MI350X/MI355X, not the R9700. See the [provenance record](../provenance/QWEN-NEXT-FEASIBILITY-20261023.json).

Before the owner parked the work, a first baseline was designed but not packaged or run: two starts, one warmup and four typed-answer tasks per start, C1, greedy natural stop and no speculation. If reopened, the prior design would first verify the checkpoint, loader, tokenizer and resource admission, then expand context after baseline and memory evidence. Possible later work would profile costs, measure cold/warm shared-prefix reuse, audit prefill dispatch, examine compatible embedded MTP within the source's top-k 1 and draft-width-at-most-4 constraints, and review slot reuse. None of those is a launch authorization. The frozen 27B DFlash8 and its shape-specific patches do not automatically transfer to Next.

The [Qwen3.8 27B baseline](QWEN38-FROZEN-BASELINE.md) remains frozen, including its unresolved numerical findings and quality limitations. Independent review approved the conservative source and metadata assessment; no baseline was executed. This feasibility assessment makes no Next quality, capacity, speed or production claim.

## UltraQuant assessment

UltraQuant addresses KV-cache storage, not model-weight compression. Even with hypothetical zero KV storage, the official FP8 weights exceed the observed combined physical RAM and VRAM by 50,372,878,330 bytes before runtime allocations. A separate weight representation or weight/PLE offload route still needs a supported loader and a proven resource fit.

The unmodified public UltraQuant v1 CPU admission contract refuses the Next configuration because it supports only conventional `LlamaForCausalLM`. This is an expected admission result, not a GPU failure. The earlier private Qwen 27B KV-cache path is distinct and has not been verified for Next's QSA, PLE or graphs. No admission guard changed, no Next weight format was selected and no accuracy or speed result was measured. See the [assessment record](../provenance/QWEN-NEXT-ULTRAQUANT-ASSESSMENT-20261023.json) and the [UltraQuant paper](https://arxiv.org/abs/2606.20474v3).

## Lower-bit weight candidates

A [separate candidate assessment](QWEN-NEXT-WEIGHT-CANDIDATES.md) records a Radiance-format 4-bit expert checkpoint identified during the search and compares GGUF and SGLang-oriented alternatives. The owner has parked this work; local fit and quality remain unverified, and no checkpoint or engine was selected.
