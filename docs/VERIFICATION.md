# Verification guide

This page lists the checks for this repository, from quick CPU tests to GPU
check Jobs on the workstation. Run commands from the repository root.

| Check | Where it runs | Needs |
| --- | --- | --- |
| [CPU test suite](#run-the-cpu-tests) | Any machine | Python 3.14, a C++17 compiler |
| [KV-cache tool tests](#test-the-kv-cache-tools) | Any machine | Python with PyTorch (CPU build is enough) |
| [Patch-series checks](#check-the-patch-series) | Any machine | A clone of SGLang and the workstation repository |
| [GPU check Jobs](#run-the-gpu-check-jobs) | The workstation's K3s | A built image and one free GPU |

None of the CPU checks loads a model or uses a GPU. A passing CPU check is not
evidence about GPU behaviour.

## Run the CPU tests

```sh
make test PYTHON=python3.14 CXX=clang++
```

Set `PYTHON` and `CXX` to your installed tools. Build products go in
`.test-build/`. Most of these tests cover the historical 0.5.21 source; the
table says what each focused target exercises.

| Target | What it exercises | What it does not establish |
| --- | --- | --- |
| `test-serving-cleanup` | Serving configuration, source assembly, receipts, quality policy and caller contracts | A completed native campaign |
| `test-validation-preflight` | Actual public guard, package and collector entry points with declared substitutes | GPU allocation or model quality |
| `test-cpu` | Retained source, integration and metadata contracts | Native GPU arithmetic |
| `test-codec` | C++ CPU codec and numerical checks with ASan/UBSan | HIP device behaviour or speed |
| `test-host-owner` | Local process supervision and mocked platform observations | Current workstation ownership or release |
| `test-benchmark` and `test-benchmark-readout` | Frozen schedules, inner results and measurement arithmetic | A matched physical result |
| `test-fp8` | Recorded FP8 baseline identities and scope | Qualification of a different cache or model profile |

## Test the KV-cache tools

Two newer targets are not part of `make test`. Run them when you change the
matching tool.

| Target | Tests | Tool |
| --- | --- | --- |
| `make test-int4-bench PYTHON=python3` | The host-side INT4 codec that the benchmark kernel reads | `tools/bench_int4_kv.py` |
| `make test-uq3 PYTHON=python3` | The UQ3 fake-quantization harness: codec invariants and both input paths | `tools/uq3_fakequant.py` |

`tools/check_fp8_kv_dequant.py`, the GPU check for patches 0026 and 0027, has
no CPU test. It runs only as a GPU Job.

## Check the patch series

Do this after any change to `serving/patches/sglang-main/`.

1. Confirm the series hash. From the workstation repository, run:

   ```sh
   infrastructure/ai-stack/patchset-hash.sh ../Spry.ai-rdna4-serving/serving/patches/sglang-main
   ```

   Compare the output with `SGLANG_PATCHSET_SHA256` in the workstation's
   `infrastructure/ai-stack/stack.lock`. If you changed a patch, update that
   value. On macOS, where `sha256sum` is missing, this gives the same hash:

   ```sh
   cd serving/patches/sglang-main && shasum -a 256 *.patch | shasum -a 256
   ```

2. Confirm the series applies. In a disposable clone of SGLang checked out at
   the `SGLANG_COMMIT` from `stack.lock`, apply every patch in name order, as
   the workstation's `sglang/build.sh` does:

   ```sh
   for p in /path/to/Spry.ai-rdna4-serving/serving/patches/sglang-main/*.patch; do
     git apply --check "$p" && git apply "$p" || { echo "FAILED: $p"; break; }
   done
   ```

   Each patch is checked after the earlier ones are applied, because later
   patches build on earlier ones.

3. Build the image and run the GPU checks below. The workstation's
   [AI-stack guide](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/README.md)
   has the build steps.

## Run the GPU check Jobs

The workstation renders these Jobs from a checkout of this repository. See its
[AI-stack guide](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/README.md)
for the full commands.

| Job | Rendered by | What it checks |
| --- | --- | --- |
| FP8 KV check | `kernel_check.py IMAGE --serving REPO` | `tools/check_fp8_kv_dequant.py`: the hardware FP8 decode is exact, kernel outputs match, and verify time against a bf16 cache |
| INT4 bench | `kernel_check.py IMAGE --serving REPO --tool int4-bench` | `tools/bench_int4_kv.py`: INT4 KV verify cost (research; parked) |
| Bench | `render.py bench PROFILE IMAGE --serving REPO --arm ARM` | BetterBench throughput, prefill and long-context phases, then the quality menu |

A kernel check Job needs one free GPU. A bench Job needs both GPUs, so the
serving Deployment must be stopped first. Record a qualifying result in the
serving profile, not in this repository.

## Historical: before native work on the 0.5.21 runtime

The checks below served the SGLang 0.5.21 runtime, which was wiped in November
2026. They are kept for readers of the older records.

The [serving test procedure](../serving/measurements/native-qualification/TESTING.md)
added exact-packet checks, the declared native interpreter and complete
retained artifact replay to the public fixture suite. For the completed result,
see the
[cleanup sign-off](../serving/measurements/native-qualification/CLEANUP-SIGNOFF.md).

### Retained verification notes

The following sections preserve the earlier source-suite contracts and sprint
outcomes. Counts such as the 112-case selection refer to that specific suite,
not the complete current `make test` count. Declared substitutes, numerical
tolerances and native limitations remain part of each record.

`make test-cpu` also runs the separate metadata-reuse adapter suite described in
[metadata reuse](METADATA-REUSE.md). Its declared CPU graph/tensor substitutes
do not count as native verification or change the historical 112-case selection.

`make test-benchmark-bindings PYTHON=/path/to/python3` checks the local live
observer, cgroup resource gate and physical-controller candidate. All root,
procfs, systemd, platform and model effects use declared CPU fixtures. These
tests do not run a privileged observer, create a transient unit, enforce native
limits or start a GPU. The controller's physical execution entry stays closed.

`make test-platform-collector PYTHON=/path/to/python3` tests the owner-run
metadata collector with declared proc/sysfs/root/platform substitutes. It
exercises field projection, private output, bounds and failure reporting; it
does not invoke sudo, query a cluster or observe real GPU availability.

`make test-host-owner PYTHON=/path/to/python3` runs the host ownership and
benchmark-adapter composition suites. Real local POSIX children exercise the
gate, polling, timeouts, signals and reaping. Linux procfs, pidfd, namespace,
device and platform observations use declared CPU substitutes on this host.
The tests do not contact the workstation or establish native GPU release.
See [host ownership](HOST-OWNER.md) for the remaining live observer binding.

`make test-fp8 PYTHON=/path/to/python3` separately checks the qualified FP8
baseline's image/runtime/model identities, complete retained inventory and
bounded result. Its mutation fixtures reject altered identities, unsupported
scope and false promotion. It performs no native or model execution. See
[FP8 baseline](FP8.md); these checks are not part of the historical 112 cases.

`make test-cpu PYTHON=/path/to/python3` runs the same declared 112-case selection
as the accepted source handoff. It imports only standard-library helpers and
individual source bodies. It must not import Torch or the SGLang package, load a
native library, query devices or execute a model. Native/tensor/device objects
are labelled CPU substitutes, not numerical GPU evidence.

The retained cases cover config/rank identity, refusal before allocation,
constructor-to-backend wiring, real strided caller layouts, cache ownership,
prefix protection, generations, scheduler/release boundaries and source/build
inventory checks. The legacy GPU executable is called only without its execution
switch to verify refusal. Tests use inert binary bytes and synthetic review
receipts; these never count as accepted build evidence.

The fixtures contain frozen earlier source and stock caller bodies so tests can
exercise actual interfaces and compare baseline traces. They are not another
supported serving version. Path bindings were relocated for a fresh clone;
`provenance/TEST-RELOCATION.json` records those changes. Existing assertions and
numerical tolerances were not relaxed. One historical native snapshot test still
uses its authenticated old native fixture; a separate test checks the four
selected current native hashes.

`make test-codec` builds and runs the two C++17 CPU suites with ASan/UBSan. These
exercise the codec, independent vectors, BF16 bits/RNE, FP32 WHT and canaries.
The predeclared WHT tolerance is `2e-6`; the dot tolerance is
`1e-5 + 2e-7*abs(reference)`; the same-code softmax tolerance is `1e-11`.
Those criteria are not silently applied to different BF16 arithmetic, and lossy
four-bit output is not required to equal BF16/FP8 output exactly. No HIP kernel
is compiled or executed by these CPU suites.

Runtime validation is separate: first bind a complete source/build identity,
then perform an authorised device/model smoke. Benchmark and task-quality gates
follow independently. The historical physical pass is described in `STATUS.md`;
it was not rerun for publication. The source subset here cannot substitute for
that complete installed composition.

### Sprint 8 host attribution

`make test-sampling-host-timing` checks the bounded sampling recorder with CPU
clock/request substitutes. It runs without Torch, SGLang or device access and
is included in `make test`. The private integration suite separately exercises
the actual modified `SamplingRoute`, native source authentication, default-mode
delegation, exception preservation and unchanged startup definitions. These
checks do not qualify native startup, graph event timing or a serving speedup.
See the [Sprint 8 record](SPRINT8-COST-ATTRIBUTION.md).

`make test-device-phase-timing` covers graph role/bucket and event generations,
completion before reuse, explicit missing samples, interval containment and
negative request/source/host/readout joins. The private runtime suite exercises
the enabled worker path with CPU substitutes and failure cleanup. Sprint 8.2
retains 104 passing CPU checks and an independently reviewed initial packet.
Those checks did not establish native startup, timing, quality or release.

Sprint 8's first native packet failed in the timing observer at rank 1/cycle 9.
The device-allocation correction passed 38 focused CPU checks and Astra review;
these include a before/after reproduction of the event-device boundary. The
subsequent corrected packet passed 107 CPU checks and completed both native
arms. All 8 final answers passed, 126 samples/card authenticated, and independent
release passed. Full output IDs match 3/4 pairs; all final texts match. The
[attribution record](../provenance/SPRINT8-COST-ATTRIBUTION-20261022.json) retains
coverage gaps and instrumented-timing limits. Independent release also passed for
the failed campaign. See the
[failure record](../provenance/SPRINT8-NATIVE-FAILURE-20261022.json).

Sprint 9 target-family hooks and reader checks run without native runtime imports:

```sh
make test-target-graph-timing test-device-phase-timing test-sampling-host-timing
```

The [Sprint 9 record](SPRINT9-TARGET-GRAPH.md) separates the successful serving
classification, retained component harness failure and corrected two-card
component result. Component parity and timing do not establish serving speedup.


### Sprint 10 matched M8 serving

The [Sprint 10 record](SPRINT10-M8-SERVING.md) retains both matched orders,
56/56 packet/startup checks per order, 53/53 per quality packet and 65/65 for
lifecycle. Readouts authenticate source/native/HTTP/release evidence and enforce
exact fixed output work, natural-stop final-answer gates, actual C2, cancellation
and slot accounting. Greedy 8/8, sampled 8/8 and lifecycle 52/52 answers passed;
two cancellation/recovery checks passed. The public serving patch applies to
the pinned reference and reproduces the tested candidate hash. Independent
review supports the next experimental selection, with the numerical exception
and production/endurance limits retained.
