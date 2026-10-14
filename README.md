# Spry.ai RDNA4 Serving

Experimental LLM serving on AMD RDNA4 GPUs, with ROCm/SGLang integration, FP8
model support, four-bit KV caching and reproducible verification.

The current prototype keeps official Qwen3.8-27B-FP8 weights fixed and changes
the KV cache. FP8 weights and KV4 storage are separate precision choices; this
is not an MXFP4 weight conversion.

## Status

The private integration passed a real eager TP2 smoke on two Radeon AI PRO R9700
GPUs: initial prefill, continuation, decode and a 13-token prefix-reuse hit.
Both ranks produced finite logits and agreed on sampled tokens. KV, request and
Mamba-state capacities returned to their initial values. The publication CPU
suite checks the retained source contracts without loading Torch, SGLang or a
native library.

This is a source prototype, not an installable serving distribution. The retained
SGLang files are a selected overlay, not the complete upstream package. Public
Qwen KV4 admission remains closed. A fresh installation needs a complete, independently
reviewed runtime/build identity; the historical private smoke does not authorise
a launch or supply a reusable build proof. Quality, capacity and performance
benchmarks remain pending. See [status and evidence](docs/STATUS.md).

## Layout

- `prototype/`: maintained native codec, kernels, SGLang source overlay and v2
  source-identity/build tools.
- `patches/`: image-relative KV4 delta and shared FP8-runtime repairs.
- `profiles/`: explicit baseline and KV4 planning configurations.
- `tests/`: current regressions and declared CPU substitutes. `tests/fixtures/`
  contains historical source snapshots needed to verify compatibility, not a
  second maintained implementation.
- `provenance/`: source hashes, pinned revisions and a sanitised result summary.
- `docs/`: precision, build, verification and benchmark boundaries.

## CPU checks

Use Python 3.11 or later and an existing C++17 compiler. No Python packages are
needed for the CPU contract suite. Select your configured interpreter explicitly:

```sh
make test-cpu PYTHON=/path/to/python3
make test-codec CXX=clang++
```

The C++ target enables ASan/UBSan. Neither target builds HIP, loads the native
library, queries a GPU or starts a model. Test products stay in `.test-build/`.
See [verification](docs/VERIFICATION.md) for fixture and historical-test scope.

`tools/model_smoke.py` retains the reviewed inner GPU-smoke entry. It requires
an explicit execution flag and hash-bound inputs. It is not a standalone launch
controller: an authorised outer supervisor must first verify device ownership,
the complete runtime, model-file identity, resource limits and cleanup.

## Runtime and precision

The selected stack is SGLang 0.5.21 at
`e00930c5489053f26d86b179cee0d087f846acbb`, ROCm SDK 10.1.0, Python 3.14.4,
Torch 2.13.0+rocm10.1.0 and RCCL 2.30.7. These pins describe the tested private
composition, not a claim that this checkout installs it.

The Qwen proposal uses TP2, head dimension 256, 16 full-attention layers and 48
GDN layers. Four full-attention layers retain BF16 storage; twelve use local
packed G32 E2M1/UE8M0 KV4. Compressed Q/K rotation occurs after native MRoPE;
V remains unrotated. BF16 input/output and FP32 computation are distinct from
packed four-bit storage. Local cache bytes are not interchangeable with vLLM
UltraQuant bytes. See [architecture](docs/ARCHITECTURE.md).

## Publication scope and licences

This repository contains source, focused tests, dependency pins and documentation.
It excludes model weights, private keys, machine configuration, native binaries,
runtime review receipts, caches and raw experiment archives. No service or
deployment is started by cloning it.

Original project code uses the [MIT licence](LICENSE). Copied and modified
SGLang/vLLM-derived files retain their Apache-2.0 notices and terms. The Qwen
config fixture has its retained licence. See [third-party notices](THIRD-PARTY.md).
