# Spry.ai RDNA4 Serving

Tools, source overlays and validation evidence for experimental language-model
serving on AMD Radeon AI PRO R9700 GPUs. The project studies model execution,
attention-cache storage and reproducible verification on two GPUs.

Start with the [documentation guide](docs/README.md) for a reading path. This
checkout contains source and tests; model weights, complete runtime images,
native binaries and raw experiment archives are supplied separately.

## What is in this project?

| Area | Purpose | Where to start |
| --- | --- | --- |
| Serving source | Configure and assemble reproducible serving packets, validate responses, and retain release evidence | [Serving guide](serving/README.md) |
| FP8 experiments | Record bounded serving results and subsequent tuning work | [Current results](docs/CURRENT-SERVING-RESULTS.md) and [session selection](provenance/CURRENT-SESSION-SELECTION.json) |
| KV4 prototype | Study four-bit attention-cache storage with a separate native codec and SGLang overlay | [Architecture](docs/ARCHITECTURE.md) and [KV4 status](docs/STATUS.md) |

These areas have separate evidence. FP8 and MXFP4 describe model-weight
precision. BF16 and KV4 describe attention-cache storage. A run with FP8 weights
can use BF16 KV; a KV4 experiment does not imply a model-weight conversion.

## Completed serving cleanup

The serving cleanup passed its bounded production-mode validation in operation
207. Both starts completed, all 352 request and output token vectors matched
the reference, all 352 natural-answer grades passed, and independent GPU/CPU
release was verified. The corpus contains eight distinct prompt-message
sequences repeated in 176 frozen cases per start.

The comparison found about 1.63 GB lower median sampled VRAM use per card in
both starts, with fixed pools. Throughput observations changed by at most 0.16%;
no causal speedup is established. Memory was sampled every 10 seconds, so those
values are not whole-device peaks.

Read the [sign-off and its limits](serving/measurements/native-qualification/CLEANUP-SIGNOFF.md)
or open the [measurement report](serving/measurements/native-qualification/GPU-207-MEASUREMENT.html).
This result covers the tested packet. The active experimental profile is
recorded separately in the session selection; no installed service is changed
by cloning this repository.

## Run the local checks

Use an existing configured Python interpreter and C++17 compiler. The completed
cleanup checks used CPython 3.14. The C++ checks enable AddressSanitizer and
UndefinedBehaviorSanitizer. No Python package installation is needed for these
CPU checks.

From the repository root, for example:

```sh
make test PYTHON=python3.14 CXX=clang++
```

Set `PYTHON` and `CXX` to your installed tools. These checks do not load a model
or start GPU work. Test products go in `.test-build/`. For focused targets and
the difference between fixture checks and native evidence, see
[verification](docs/VERIFICATION.md).

## Find the right guide

| You want to… | Read |
| --- | --- |
| Understand the components and precision choices | [Architecture](docs/ARCHITECTURE.md) |
| Work on the cleaned serving source | [Serving guide](serving/README.md) |
| Understand runtime assembly and build requirements | [Build guide](docs/BUILD.md) |
| Follow a reviewed validation campaign | [Caller workflow](serving/launcher/README.md) and [test procedure](serving/measurements/native-qualification/TESTING.md) |
| Find an experiment or its limitations | [Documentation index](docs/README.md) |

## Repository map

| Directory | Contents |
| --- | --- |
| `serving/` | Configuration, builder, patch series, runtime modules, evidence readers and cleanup results |
| `prototype/` | Native KV4 codec, kernels, SGLang overlay and build-identity tools |
| `common-runtime/`, `patches/` | Pinned shared repairs and the separate KV4 delta |
| `profiles/` | Recorded baseline profiles and comparison plans |
| `tests/` | Regression tests and declared CPU substitutes |
| `provenance/` | Source hashes, pinned revisions, selections and result summaries |
| `docs/`, `benchmarks/` | Guides, research records and benchmark definitions |

The repository is an experimental source project. The public KV4 admission path
remains closed, and a selected source overlay is not a complete SGLang
installation. Native work requires a reviewed runtime, authenticated inputs,
fresh ownership and the applicable one-use approval. See the
[KV4 operator guide](docs/KV4-OPERATIONS.md) for that prototype's separate scope.

## Licences and publication scope

Original project code uses the [MIT licence](LICENSE). Copied and modified
SGLang/vLLM-derived files retain their Apache-2.0 notices and terms. See
[third-party notices](THIRD-PARTY.md).

Model weights, credentials, machine configuration, native binaries, caches and
raw private archives are excluded from publication. Published source and result
summaries do not transfer private build proofs or launch authority.
