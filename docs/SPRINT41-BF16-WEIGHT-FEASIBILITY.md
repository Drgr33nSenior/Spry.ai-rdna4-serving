# Sprint 41: BF16-weight feasibility

**Source-validated; hardware validation NOT RUN.** Retain the official FP8 target weights. The unchanged BF16 control exceeds the selected static device-memory budget before all mandatory allocations are counted. No BF16 model was launched, and no source, runtime default or precision contract changed.

The audit reconciled the checkpoint headers, active language layers, tensor-parallel loader, skipped MTP weights, resident scale buffers and existing memory records. Independent Astra review verified those sources and the arithmetic. The installed host still reports 65,148,308 kB of Linux-visible RAM; the planned 128 GB upgrade is not installed.

Replacing the 400 active language projection weights with BF16 adds exactly **12,160,512,000 bytes (11.325 GiB) per rank**, after removing their resident FP32 block scales. Embedding and language-model-head weights are already BF16; FP32 head arithmetic does not imply an FP32 resident weight copy.

Each card reports 34,208,743,424 bytes of VRAM. Even the optimistic ceiling of 92% of raw VRAM is only 29.311 GiB. The actual configurator subtracts 8% of pre-load free memory and the multimodal reservation, so this ceiling is generous.

| Context / token pool | Conservative BF16 resident lower bound per card |
| --- | ---: |
| 16,384 / 16,416 | 31,904,732,912 bytes / 29.714 GiB |
| 32,768 / 32,800 | 32,609,375,984 bytes / 30.370 GiB |
| 65,536 / 65,568 | 34,018,662,128 bytes / 31.682 GiB |

These bounds include the target projection weights, embedding/head, a lower bound for vision and BF16 draft weights, recurrent/convolution storage, aligned rings and BF16 target/draft KV with padding. They exclude other target weights, rotary caches, multimodal reservation, graphs, workspaces, allocator and driver overhead. Automatic token-pool reduction would invalidate the matched comparison and is not accepted.

This is a source-budget exclusion under the unchanged DFlash8, 19-usable/20-physical-slot profile. It is not an observed GPU allocation failure or proof that every smaller BF16 configuration is impossible. Host fit under the unchanged 38 GiB owner cap remains unmeasured. A smaller-capacity comparison would need a separate design and would not establish BF16 stability superiority.

[Provenance](../provenance/SPRINT41-BF16-WEIGHT-FEASIBILITY-20261023.json) retains the exact byte arithmetic and source seals. The [remaining optimization programme](REMAINING-FP8-OPTIMIZATIONS.md) incorporates this disposition before device candidates. Reliability and endurance remain deferred until tuning closes.
