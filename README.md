# Spry.ai RDNA4 Serving

Experimental LLM serving on AMD RDNA4 GPUs, with ROCm/SGLang integration, FP8
model support, four-bit KV caching and reproducible verification.

Official Qwen3.8-27B-FP8 weights stay fixed. The native baseline uses BF16 KV;
the KV4 prototype changes cache storage. FP8 weights and KV4 storage are separate
precision choices; this is not an MXFP4 weight conversion.

## Status

The separate native FP8/BF16-KV baseline passed bounded C1/C2 compatibility at
4096 tokens on both R9700s. Its graph-enabled profile, exact image/model/source
identities and sanitised evidence are retained in [FP8 baseline](docs/FP8.md).
This does not qualify the KV4 composition or establish which configuration is
fastest. A later short-prompt HTTP comparison found no meaningful regression
between the previous Torch/ROCm stack and the current Torch 2.14 / SDK 10.2 stack.
That comparison had DFlash2 disabled. See [current serving results](docs/CURRENT-SERVING-RESULTS.md)
for the exact scope. A separate DFlash2 + split K/V + aligned GDN benchmark
measured 86.68 versus 24.50 output tokens/s, a 3.54× increase on one short workload.

The selected [Sprint 22 profile](docs/SPRINT22-GDN-OUTPUT-M16.md) measured
121.18–121.24 combined TP2 output tokens/s at C1 with 2047 input and 1024 forced
output tokens and low thinking. Both matched serving orders passed all eight
output/work pairs. Greedy and publisher-sampled quality each passed 8/8 final
answers; lifecycle passed 52/52 normal answers, two cancellations, actual C2,
slot accounting and release. Astra approved common M16 GDN output for the next
experimental session, retaining the previous GEMM for rollback and aligned
ring-fold unchanged. The initial observation failure and existing reasoning
variations remain recorded. Installed runtime and production defaults are unchanged.

The [64K SRE screen](docs/SPRINT33-LONG-CONTEXT64.md) completed 14 normal-stop responses; its literal and factual grading gates each scored 10/12 and failed. The two format failures and the bounded input/output scope remain in the record. Independent review approved the conservative closeout.

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
a launch or supply a reusable build proof. General quality, capacity and performance
qualification remains pending. See [status and evidence](docs/STATUS.md).

## Layout

- `prototype/`: maintained native codec, kernels, SGLang source overlay and v2
  source-identity/build tools.
- `patches/`: image-relative KV4 delta and shared FP8-runtime repairs.
- `profiles/`: the qualified 4K FP8 baseline record and separate eager
  baseline/KV4 comparison plans; none is a launch command.
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
make test-fp8 PYTHON=/path/to/python3
make test-benchmark PYTHON=/path/to/python3
make test-benchmark-readout PYTHON=/path/to/python3
make test-host-owner PYTHON=/path/to/python3
make test-platform-collector PYTHON=/path/to/python3
make test-benchmark-bindings PYTHON=/path/to/python3
make test-codec CXX=clang++
```

The C++ target enables ASan/UBSan. These targets do not build HIP, load the native
library, query a GPU or start a model. Test products stay in `.test-build/`.
See [verification](docs/VERIFICATION.md) for fixture and historical-test scope.

`tools/model_smoke.py` retains the reviewed inner GPU-smoke entry. It requires
an explicit execution flag and hash-bound inputs. It is not a standalone launch
controller: an authorised outer supervisor must first verify device ownership,
the complete runtime, model-file identity, resource limits and cleanup.

The [KV4 operator guide](docs/KV4-OPERATIONS.md) covers benchmark readiness,
ownership, stop conditions and result collection. The current KV4 preparation
is source-only: fresh physical packaging, loading-time instrumentation and
explicit two-GPU handover remain required. The guide does not authorize a run
or supply a physical launcher. Earlier shared FP8 attempts failed; later bounded
HTTP comparisons are documented in [current serving results](docs/CURRENT-SERVING-RESULTS.md).

An [owner-run metadata collector](docs/OWNER-COLLECTION.md) is available for the
missing platform observations. It is not run by the agent and does not start,
reserve or authorize GPU work.

The [FP8 performance work order](docs/PERFORMANCE-PLAN.md) prioritizes the native
cache comparison, active-work and decoded-value reuse, then full-model fusion,
recurrent-state traffic and TP2 communication. Research gains are not local
measurements or changes to the numerical contract.

## Runtime and precision

The historical KV4 composition used SGLang 0.5.21 at
`e00930c5489053f26d86b179cee0d087f846acbb`, ROCm SDK 10.1.0, Python 3.14.4,
Torch 2.13.0+rocm10.1.0 and RCCL 2.30.7. These pins describe the tested private
composition, not a claim that this checkout installs it. The later FP8 serving
checks use Torch 2.14 / SDK 10.2; see the current results for complete versions.

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
