# Documentation guide

Start here to choose a guide. The repository contains a serving implementation,
an experimental KV4 cache path and a history of bounded measurements. Their
results apply to different source, precision and request profiles.

## Pick a reading path

| Your goal | Start with | Then read |
| --- | --- | --- |
| Get oriented | [Project overview](../README.md) | [Architecture](ARCHITECTURE.md) |
| Work on the cleaned serving source | [Serving guide](../serving/README.md) | [Caller workflow](../serving/launcher/README.md) and [quality guide](../serving/quality/README.md) |
| Run local checks | [Verification](VERIFICATION.md) | [Cleanup test procedure](../serving/measurements/native-qualification/TESTING.md) |
| Assemble or build source | [Build guide](BUILD.md) | [Patch series](../serving/patches/README.md) |
| Review the completed cleanup | [Sign-off](../serving/measurements/native-qualification/CLEANUP-SIGNOFF.md) | [Measurement report](../serving/measurements/native-qualification/GPU-207-MEASUREMENT.html) |
| Find the selected experiment | [Session selection](../provenance/CURRENT-SESSION-SELECTION.json) | [Current results](CURRENT-SERVING-RESULTS.md) and [sprint index](SERVING-SPRINTS.md) |
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

- **Current experimental choice:** use the session selection and its referenced
  profile. The newest sprint filename does not choose a runtime.
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
| Planning and research | [Next optimization plan](NEXT-OPTIMIZATION-PLAN.md), [performance plan](PERFORMANCE-PLAN.md), [reopened tuning](QWEN38-REOPENED-TUNING.md), [weight candidates](QWEN-NEXT-WEIGHT-CANDIDATES.md) |
| Experiment history | [Sprint index](SERVING-SPRINTS.md) and each sprint's linked provenance |

Raw private archives, credentials, model weights and complete runtime images are
outside this repository. A public summary or a passing CPU fixture is not a
transferable native build proof or permission to launch a workload.
