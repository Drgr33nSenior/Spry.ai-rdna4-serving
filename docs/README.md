# Documentation guide

Start here to choose a guide. The repository contains a serving implementation,
an experimental KV4 cache path and a history of bounded measurements. Their
results apply to different source, precision and request profiles.

Status, 9 November 2026: serving now targets SGLang upstream `main` at
`fcb3274646e43c16d70fd83806ebb3a5134d5e81` with the
[`sglang-main` patch series](../serving/patches/sglang-main/README.md). The
workstation repository's
[serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json)
holds the flags and environment; it is unqualified. Operation234's split-K
selection and everything built on its SGLang 0.5.21 runtime are historical.

## Pick a reading path

| Your goal | Start with | Then read |
| --- | --- | --- |
| Get oriented | [Project overview](../README.md) | [Architecture](ARCHITECTURE.md) |
| Work on the cleaned serving source | [Serving guide](../serving/README.md) | [Caller workflow](../serving/launcher/README.md) and [quality guide](../serving/quality/README.md) |
| Run local checks | [Verification](VERIFICATION.md) | [Cleanup test procedure](../serving/measurements/native-qualification/TESTING.md) |
| Assemble or build source | [Build guide](BUILD.md) | [Patch series](../serving/patches/README.md) |
| Review the completed cleanup | [Sign-off](../serving/measurements/native-qualification/CLEANUP-SIGNOFF.md) | [Measurement report](../serving/measurements/native-qualification/GPU-207-MEASUREMENT.html) |
| Find the current serving target | [`sglang-main` series](../serving/patches/sglang-main/README.md) | Workstation [serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json) and [research notes](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/RESEARCH-20261108.md) |
| Find the historical op234 selection | [Split-K selection](../provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json) | [Qwen4 handoff](QWEN4-SERVING-HANDOFF.md), [finalization record](../provenance/MXFP4-FINALIZATION-20261105.json), [current results](CURRENT-SERVING-RESULTS.md) and [sprint index](SERVING-SPRINTS.md) |
| Understand the KV4 prototype | [KV4 status](STATUS.md) | [Cache path](CACHE-PATH.md), [operator guide](KV4-OPERATIONS.md) and [historical benchmark contract](BENCHMARKING.md) |

## Terms you will meet

| Term | Meaning here |
| --- | --- |
| FP8 / MXFP4 weights | How a particular model checkpoint stores its weights. Check the run's model identity. |
| BF16 / KV4 cache | How attention keys and values are stored. This choice is separate from weight precision. |
| GDN | Gated DeltaNet recurrent layers. Their state has separate allocation and rollback rules from attention KV. |
| TP2 | Tensor-parallel execution across two GPUs; reported TP2 throughput is combined service throughput. |
| C1 / C2 | One or two concurrent HTTP requests. This label alone does not prove native batch size. |
| Production mode | The reduced-diagnostic evidence mode in these packets; it is not a deployment approval. |
| Instrumented mode | A packet that retains the specified diagnostic observers and startup witnesses. |
| Packet / campaign | A prepared source author and bindings / the bounded operation that consumes them. |
| Sealed evidence | A retained tree whose membership and file hashes are fixed by its manifest. |

## Which record answers which question?

- **Current serving target:** SGLang `main` with the
  [`sglang-main` patch series](../serving/patches/sglang-main/README.md) and
  the workstation serving profile. It is unqualified; no result in this
  repository applies to it yet.
- **Historical experimental choice:** the
  [split-K selection](../provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json). It names operation234's
  `mtp` arm (physical `kv4`, start 2) and operation232's selected arm as
  rollback. The [finalization record](../provenance/MXFP4-FINALIZATION-20261105.json) holds the operation232
  selection and the parking decision. The [Qwen4 handoff](QWEN4-SERVING-HANDOFF.md)
  describes how that work was parked. This was a private experimental
  selection, not an installed service; the 2026-11 workstation reinstall wiped
  its runtime, and it cannot be rebuilt from this repository as documented. The
  [FP8 session selection](../provenance/CURRENT-SESSION-SELECTION.json) is
  historical and retained unchanged. The newest sprint filename does not choose a
  runtime.
- **Quality scope:** operation207 and operation232 natural-answer grades cover
  the cleanup packet and the rollback arm. BetterBench (operations 233 and 234)
  has no answer-quality oracle, so no result here qualifies the browser or
  coding-agent workflow.
- **Completed cleanup:** use the serving sign-off and its exact source and run
  bindings. It covers eight distinct prompt-message sequences repeated across
  the frozen menu.
- **A historical measurement:** use that experiment's own page and provenance.
  Keep its model, cache, context, output policy, concurrency and timing scope.
- **Earlier implementation decisions:** use the
  [serving history](../serving/HISTORY.md) and [proposal index](../serving/proposals/README.md).
  Pending-state text in those records belongs to that increment.

## More detailed guides

| Subject | Documents |
| --- | --- |
| Build and execution boundaries | [Build](BUILD.md), [host ownership](HOST-OWNER.md), [owner collection](OWNER-COLLECTION.md) |
| Cache implementation | [Cache path](CACHE-PATH.md), [metadata reuse](METADATA-REUSE.md), [GPU test contract](CACHE-GPU-TEST.md), [phase profiling](CACHE-PHASE-PROFILE.md) |
| Recorded baselines | [Native FP8 baseline](FP8.md), [frozen Qwen baseline](QWEN38-FROZEN-BASELINE.md), [MXFP4 Phase 1](MXFP4-CANDIDATE-PHASE1.md), [MXFP4 128K](MXFP4-CANDIDATE-128K.md) |
| Performance investigation | [Current cost analysis](CURRENT-COST-ANALYSIS.md), [profiling](PROFILING.md), [host-path cleanup](HOST-PATH-CLEANUP.md), [optimization candidates](OPTIMIZATION-CANDIDATES.md) |
| Planning and research | [Next optimization plan](NEXT-OPTIMIZATION-PLAN.md) (superseded), [performance plan](PERFORMANCE-PLAN.md), [reopened tuning](QWEN38-REOPENED-TUNING.md), [weight candidates](QWEN-NEXT-WEIGHT-CANDIDATES.md) |
| Experiment history | [Sprint index](SERVING-SPRINTS.md) (historical) and each sprint's linked provenance |
| Historical MXFP4 and FP8 records | [Remaining gates](MXFP4-REMAINING-GATES.md), [OPT-00 baseline and witness](MXFP4-OPT00-BASELINE-AND-WITNESS.md), [OPT-01 host and runtime audit](MXFP4-OPT01-HOST-RUNTIME-AUDIT.md), [OPT-06 cache and graph screen](MXFP4-OPT06-CACHE-GRAPH-SCREEN.md), [OPT-07 shared-owner integration](MXFP4-OPT07-SHARED-OWNER-INTEGRATION.md), [P3 fused-norm candidate](MXFP4-P3-FUSED-NORM-CANDIDATE.md), [FP8 target-KV diagnostic](FP8-TARGET-KV-DIAGNOSTIC-20261026.md) |

Raw private archives, credentials, model weights and complete runtime images are
outside this repository. A public summary or a passing CPU fixture is not a
transferable native build proof or permission to launch a workload.
