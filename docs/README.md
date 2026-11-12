# Documentation guide

Start here to find the right page. This folder holds two kinds of document:

- **Living guides**, kept up to date. They are listed below.
- **Dated records** of experiments, kept unchanged once written. The
  [history index](HISTORY-INDEX.md) lists them and says what replaced each one.

If you are new, read the [project README](../README.md) first. It explains what
runs today and how this repository relates to the workstation repository.

## Pick a reading path

| Your goal | Start with | Then read |
| --- | --- | --- |
| Understand the project | [README](../README.md) | [Architecture](ARCHITECTURE.md) |
| Change or add a patch | [Patch series](../serving/patches/sglang-main/README.md) | [Verification](VERIFICATION.md) |
| Understand how the image is built | [Build guide](BUILD.md) | Workstation [AI-stack pipeline](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/README.md) |
| Run the local checks | [Verification](VERIFICATION.md) | — |
| See the latest KV-cache research | [INT4 verify-kernel record](UQ3-INT4-VERIFY-KERNEL-20261112.md) | [Draft-pool plan](DFLASH-BOUNDED-DRAFT-POOL-PLAN-20261111.md) and its note in the [history index](HISTORY-INDEX.md#current-records-sglang-main-era) |
| Find an older experiment | [History index](HISTORY-INDEX.md) | The record and its `provenance/` summary |
| Work as an agent | [AGENTS.md](../AGENTS.md) | — |

## Living guides

| Guide | Covers |
| --- | --- |
| [README](../README.md) | What the project is, what runs today, the repository map |
| [AGENTS.md](../AGENTS.md) | Rules for agents, including the immutability rule |
| [Architecture](ARCHITECTURE.md) | How a request is served, and how weights, cache and recurrent state differ |
| [Patch series](../serving/patches/sglang-main/README.md) | Each SGLang patch, its switch and which arms use it |
| [Build guide](BUILD.md) | How the workstation builds the image from this series |
| [Verification](VERIFICATION.md) | CPU checks, patch-series checks and GPU check Jobs |
| [Third-party notices](../THIRD-PARTY.md) | Licences of copied code |
| [History index](HISTORY-INDEX.md) | Every older record, by era |

## Glossary

| Term | Meaning |
| --- | --- |
| R9700, gfx1201 | The AMD Radeon AI PRO R9700 GPU, 32 GB, and its LLVM target name (RDNA4) |
| TP2 | Tensor parallelism over two GPUs. The model is split in half, one half per card. Throughput figures are for both cards together |
| MXFP4 | A 4-bit floating-point weight format with a shared scale per block of 32 values. The live model's weights use it |
| FP8, BF16 | 8-bit and 16-bit floating-point formats. Weight format and cache format are chosen separately |
| KV cache, FP8 KV | The attention keys and values kept for each token. "FP8 KV" stores them in FP8 and patches 0026 and 0027 read them back as bf16 |
| GDN | Gated DeltaNet, the recurrent layers in the hybrid Qwen model. Their state is stored separately from the KV cache |
| DFlash, DFlash2 | SGLang's speculative decoding. A small draft model proposes a block of 8 tokens and the target model verifies them in one pass |
| Verify | The target model's pass that checks the draft tokens |
| C1, C2 | One or two requests in flight at once, as BetterBench reports them |
| BetterBench | The throughput benchmark used here, version 0.6.1. It has no answer-quality oracle |
| Profile, arm | The workstation's serving profile is a JSON file of server flags. An arm is a named variant of those flags, such as Y or UQS |
| Mode | What the workstation runs on its two GPUs. Shared mode runs arm UQS beside image generators; large mode runs arm UQL alone |
| Y | The arm qualified on 2026-11-10: the selected gfx1201 speed patches on, BF16 KV cache |
| UQS, UQL | The live arms, qualified on 2026-11-11: arm Y plus an FP8 KV cache. UQS gives 81,888 tokens per conversation; UQL gives 262,112 |
| UQ, UQ2, UQ3, K8V4 | Names for KV-cache compression work. UQ2 is the FP8 cache now live. UQ3 is 4-bit integer KV, and K8V4 keeps keys in FP8 with 4-bit values; both were parked on 2026-11-12 |
| KV4, UltraQuant | The earlier 4-bit KV prototype on SGLang 0.5.21, informed by the UltraQuant v3 paper. Historical |
| op207, op234, "operation N" | Numbered GPU runs on the old 0.5.21 runtime. Op207 signed off the W5 cleanup. Op232–234 chose the final MXFP4 profile on 6–7 November |
| W5 | The serving configuration used for the 0.5.21 cleanup (`serving/config/runs/w5-v2`) |
| cap16, S4, F1, C4, P1 | Names of op-era optimisations: cap16 caps verify attention at 16 KV splits; S4 is a four-way split-K for the down and output GEMMs; F1 fuses GDN norm and quantize; C4 is 8-warp prefill attention; P1 is a block-FP8 prefill route. Several are now patches in the series |
| Sealed evidence | A retained directory whose file list and hashes are fixed by a manifest. Never edited |
| Qualified | The owner accepted an arm after a bench Job and the quality menu on a specific image. Recorded in the profile's `qualification` list |

## What is not in this repository

Model weights, credentials, runtime images and raw private archives are kept
elsewhere. A passing CPU test here does not prove anything about a GPU run.
