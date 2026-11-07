# Effective memory bandwidth follow-up

Status: the joint code/scale census and CPU storage screens received Opus 5.5 signoff on 4 November 2026. CPU analysis tools are implemented; no GPU decoder, serving change or measured performance gain is claimed. This workstream supplements the original 21-lead register and preserves every prior disposition.

The owner supplied a detailed investigation brief for Radeon AI PRO R9700 / gfx1201, naming SGLang 0.5.23 and Qwen3.8-27B. Treat those names as requested target labels until they are reconciled with the actual checkpoint, selected source pins and consumed runtime. Do not infer execution precision from a checkpoint name.

## What is new, and what is already covered

| ID | Lead | Relationship to existing work | First decision |
| --- | --- | --- | --- |
| BW-01 | Scale-conditioned, lossless FP4 storage | Extends G3 and MX-03 beyond the earlier 93-tile codec sample | Does conditional structure survive held-out tiles and pay for the complete layout? |
| BW-02 | Integer execution of existing quantized operands | Extends L09 and MX-04; keeps their counterexamples and blocked production-equivalence verdict | Can a supported gfx1201 lowering preserve the actual kernel contract? |
| BW-03 | Certified output-head screening and shared-row shifts | Related to OPT-05/D1, but distinct from heuristic draft shortlisting | Can bounds certify the selected head's rounded result cheaply enough at C1 and C2? |
| BW-04 | Fixed-function graphics compression for weights | New exploratory branch | Can a compression-capable gfx1201 image preserve exact reads and reduce measured traffic? |

Recurrent-state replay remains L02/R1. Revisit it only if measured state traffic or speculative execution exposes a removable cost. Do not create a duplicate project. The owner retains MXFP4 for its memory advantage; this brief does not reopen an FP8 conversion.

## BW-00: baseline and census first

Reuse OPT-00, the selected pointer, launch witnesses, MX-00 through MX-06 and their reviews. Inspect existing branches and implementation ownership before changing code. Future code work needs an isolated branch or checkout and must preserve the active deployment.

Record checkpoint ID/revision, quantization metadata, weight and activation formats, layouts and shard boundaries, GPU topology/TP configuration, SGLang commit, ROCm/Triton/compiler, selected patches and launch arguments. Trace the actual operator dispatch. The retained packed-projection work is evidence about its pinned route, not proof that every projection uses W4A4.

Stream a representative census across layers, projection families, TP shards and tiles. Report counts and sampling coverage, H(C), H(C|S), H(C|S, layer), sign/magnitude distributions and scale ranges. Include exceptional scale codes and signed zero. Separate dictionary fitting from held-out evaluation, and report per-tile dispersion and raw-fallback rates. Existing small samples can exercise the analyzer but cannot establish checkpoint-wide compressibility. Activation scale ranges require appropriate retained activation evidence; checkpoint weights alone do not supply them.

Use existing workload/trace evidence to rank decode, concurrent serving, prefill and long-context costs. Label payload-byte estimates, transaction models and hardware-counter measurements separately. Keep physical bandwidth distinct from effective bandwidth computed using uncompressed bytes.

## Retained-input audit, 4 November

The sealed CPU audit authenticated the previous MX-03 records: 93 tiles with
724,992 weights. Those records contain separate code and scale histograms,
but no paired counts. **They cannot determine H(C|S).** This audit performed no
remote access, checkpoint read or GPU work.

The proposed unconditioned three-magnitude palette has no ideal size headroom
on this sample, including when a separate palette is fitted optimistically per
tile. That result does not reject a scale-conditioned dictionary. It does avoid
building a decoder for the unconditioned version.

At that audit, BW-01 still needed a joint census with held-out tiles and
complete layout accounting. That earlier sample is small,
uses selected projections and diagonal tile anchors, and does not establish
checkpoint-wide entropy or activation scale ranges. Its reproduction script and
seal are linked by the provenance record. This is an input audit, not sprint
signoff or a demonstrated performance improvement.

## Joint census decision, 4 November

The follow-up read paired weight and scale tiles from all 496 U8 layer
projections across 64 layers: 3,968 disjoint tiles, with separate training and
held-out sets. This resolves the missing joint statistics for the sample.
Coverage remains sparse; TP-loader shard identity, activation scales and
actual memory transactions are still unmeasured.

The proposed three-magnitude palette requires escapes for 47.159% of held-out
weights. Its ideal 4.66476 bits/weight exceeds the ordinary 4.25, before any
metadata. Marginal and scale-conditioned Huffman payloads save only 3.64% and
4.75% even before framing. Neither Huffman layout selects a tensor under the
complete-layout 5% training gate. These three layouts are parked; broader
conditional compression remains open only with a stronger representation or
access-cost mechanism. No GPU decoder is justified by these results.

The [reviewed census and priorities](MXFP4-JOINT-CENSUS-AND-PRIORITIES.md) bind
the implementation, tests, results and limitations. The fresh op217 trace
census prioritizes packed GEMM for decode and verification attention for long
contexts; it does not claim request-level savings. BW-02, BW-03 and BW-04 keep
their existing proof obligations. The original 21 leads and three cumulative
members remain intact, and the serving bundle stays on HOLD.

The subsequent [GEMM wave-hint screen](MXFP4-GEMM-WAVE-HINT-SCREEN.md) found a
possible allocation improvement, but failed its original machine-code equality
gate. Opus narrowed the next hypothesis to gate/up M8/M16, with down-M16 as a
negative control. The owner accepted the revised contract, and the
[single three-case CPU follow-up passed](MXFP4-GEMM-WAVE-HINT-CPU-RESULTS.md).
Gate/up compiler candidates are retained; device correctness and speed remain
unmeasured. The invocation is consumed and parked. This does not reopen the v6
layout failure or authorize GPU work.

## BW-01: representation gate

Prototype an offline encoder and exact inverse before any GPU decoder. One candidate keeps the sign bit, uses a two-bit selector for three common magnitudes plus escape, and stores the original three-bit magnitude for an escape. Dictionaries may depend on layer and existing scale information.

Its ideal cost is **3.25 + 3p bits/weight**, including the unchanged eight-bit scale per 32 weights. Against 4.25 bits/weight, **p < 1/3 is necessary before overhead**, not a sufficient acceptance condition. Charge dictionaries, flags, escape addressing, offsets, tile alignment, padding, raw fallback and actual memory transactions. Preserve independent tile access and exact payload/scale reconstruction, including negative zero.

Consider better layouts if the census supports them. Scale-only compression has limited headroom because scales are 1/17 of ordinary MXFP4 storage. A surviving GPU design must decode directly into its consumer; full-tensor expansion into VRAM does not satisfy this bandwidth hypothesis.

## BW-02: arithmetic gate

For finite E2M1 operands, q=2v is an integer in [-12,12], which does not fit signed INT4. A 32-element dot has |z| <= 4608 and real contribution z × 2^(e_A+e_W-256), under the stated E8M0 convention. Verify both conventions against the consumed quantizer and kernel. Do not add activation quantization.

Keep three obligations separate: decoded-bit equality, real-arithmetic identity, and production floating-point equivalence. Reuse the existing failed examples. Cover the actual reduction tree, incoming accumulator, FMA association, whole-K accumulation, signed zero, NaN scales, overflow, underflow, subnormals and flush modes. Cross-block integer alignment needs explicit overflow and representability bounds. Formal verification is useful only for the modeled semantics; unsupported inputs retain a validated fallback.

Inspect gfx1201 integer instructions and native compiler lowering before selecting INT8 or another route. Preserve the existing numerical/resource contracts; a new contract needs an explicit pre-measurement decision.

## BW-03 and BW-04: targeted alternatives

Certified head screening must bound the **selected production kernel**, including final rounding, ties, TP combination, sampling and logit processors. Shared-row shifts W' = W - 1a^T preserve real softmax probabilities, but do not by themselves preserve rounded logits or the serving token. Keep original-row recomputation and full-head fallback. Count certificate, recomputation and fallback costs; C2 regressions can invalidate a C1 benefit. Approximate head quantization is a separate, disabled-by-default proposal.

For graphics compression, first establish exact image reads, active compression, supported formats/layouts and measured latency/traffic in a standalone experiment. The AMD DCC overview does not establish gfx1201 behavior or compression of ordinary HIP buffers. HIP/Vulkan interoperability and SGLang integration come only after that evidence.

## Prototype and acceptance sequence

1. Freeze the baseline and census; reject representations without realistic net headroom.
2. Implement the smallest exact CPU codec or arithmetic reference with adversarial tests.
3. Compile only surviving kernels; inspect ISA, register allocation, occupancy, loads, waits and decode overhead.
4. Prepare a separately approved component comparing existing packed, storage-only, arithmetic-only and combined paths where each is independently justified.
5. Test every code, extreme scale, cancellation, escape-heavy tile, tail, shape boundary and TP layout. Do not relax tolerances.
6. Integrate a survivor behind an explicit option and diagnosed fallback. Validate loading, batching, graph capture and TP behavior.
7. Use matched checkpoint/input/batch/context/topology/configuration and 235 W controls, fixed warmups, synchronization and alternating runs. Report variation, kernel and request latency, TTFT, inter-token/tail latency, throughput, VRAM and fallback rates as applicable.

Require repeatable benefit beyond noise, the stated correctness contract and no unexplained regression. Preserve negative findings. Deliver exact reproduction commands, code/tests, full storage accounting, baseline/candidate results and remaining limitations. A pending census or prototype is not a demonstrated gain.

## Initial Exa source check

These sources were retrieved through Exa. This is an applicability screen, not a reproduction of their proofs or performance results.

- [Low-precision lossless compression](https://arxiv.org/html/2508.19263) supports investigating nonuniform field distributions. It does not establish compressibility of this checkpoint conditioned on its scales.
- [ZipServ](https://arxiv.org/html/2603.17435) supplies prior art for fused decompression into GEMM and GPU-friendly representations. Its reported NVIDIA results do not predict an MXFP4/gfx1201 gain.
- [Scale placement with integer arithmetic](https://borowiecki.dev/pdf/2610.29445) explicitly treats reduction-tree-dependent floating-point equivalence, with bounded integer models and guarded lowering. Its conditions must be matched to our actual instruction semantics.
- [The Work a Verifier Needs](https://sammausberg.com/research/papers/the_work_a_verifier_needs.pdf) studies production rounding and fallback. Its certified-head argument depends on a tested, unproved kernel-error model; its GH200/Qwen3.5-4B results include concurrency regressions.
- [Softmax reparameterization](https://arxiv.org/html/2609.31291) studies functionally equivalent real-valued heads followed by lossy quantization. It is prior art for the shift, not proof of bit-exact production equivalence.
- [AMD DCC overview](https://gpuopen.com/learn/dcc-overview/) explains lossless render-target compression and access restrictions. Its GCN/RDNA/RDNA2 coverage is insufficient evidence for this RDNA4 deployment.
- [Unweight](https://research.cloudflare.com/nikulin2026/?cf_page=unweight-tensor-compression%2F), found by Exa, is adjacent prior art for palettes, raw exception rows and reconstructive matmul. Its BF16/Hopper scope does not validate a scale-conditioned FP4 codec.

The proposed combinations are hypotheses. No novelty claim is made. One consolidated review is required before sprint signoff; listing these leads does not sign them off or authorize another GPU operation.
