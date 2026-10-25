# Lossy prefill communication component

25 October 2026. **Component-tested on both R9700 GPUs; unselected.**
The block256 INT8 composition reduced complete communication time at the
synthetic 2,048-row prefill size. Small decode-sized tensors were slower.
Model answers, actual model activations, graph replay and serving performance
remain **NOT RUN** for this candidate. The selected runtime is unchanged.
The subsequent [selected-model route diagnostic](LOSSY-PREFILL-ROUTE.md)
observed matching large BF16 tensors without applying compression.

## Measured cost

The baseline uses BF16 RCCL all-reduce. The candidate quantizes each rank's
BF16 input into blocks of 256 signed INT8 values plus one FP32 scale, gathers
the packed bytes through RCCL, reconstructs and sums in FP32, and writes BF16.
It does not cast inputs to FP16. Timing includes all candidate kernels and the
collective, with separate events on both devices and host completion timing.

Each shape has four warmup pairs and twelve measured pairs, alternating which
arm runs first. Numerical checks and compilation precede measured pairs.
These are eager, single-process/two-device measurements on synthetic operands,
not the server's two-process communicator or token throughput.

| Shape | Baseline GPU median, cards 0 / 1 | INT8 GPU median, cards 0 / 1 | GPU time change |
| --- | --- | --- | --- |
| 8 × 5,120 | 0.0329 / 0.0301 ms | 0.0814 / 0.0803 ms | 148–166% more time |
| 16 × 5,120 | 0.0341 / 0.0317 ms | 0.0979 / 0.0956 ms | 187–201% more time |
| 2,048 × 5,120 | 0.5774 / 0.5750 ms | 0.4742 / 0.4785 ms | **16.8–17.9% less time** |

For the prefill size, host completion fell from 0.5975 to 0.5001 ms, a 16.3%
reduction. All twelve prefill pairs favoured the candidate on both devices and
the host. Card 0 baseline/candidate ranges were 0.5708–0.5855 /
0.4599–0.4872 ms; card 1 ranges were 0.5714–0.5795 / 0.4590–0.4883 ms.
Small-shape timing outliers remain in the published raw record.

The saving is approximately **0.10 ms per tested prefill collective**. It is
not a 17% model or prefill-latency improvement. No measured collective count,
exposed transport fraction or end-to-end benefit follows from this component.

## Numerical limits

All fifteen cases passed exact CPU checks for quantized codes, FP32 scales,
FP32 reconstruction/BF16 output, both-rank equality, and an analytical error
bound. Cases cover normal values, small values, large outliers, exact opposing
inputs and zeros at all three shapes. Maximum input-normalized RMS error was
0.007186, below the predeclared 0.02 component screen.

**That screen is insufficient for model acceptance.** In the large-outlier
cases, opposing large terms cancel and the codec erases the remaining small
signal. Relative error against the true reduced output is **1.0** at all three
shapes. At the prefill size, absolute RMS error is 1.4084 and maximum absolute
error is 7.3594, despite input-normalized RMS error of only 0.0001596.
Exact agreement with the specified codec does not establish acceptable model
accuracy. No model precision contract has been relaxed.

## Scope and retained failures

Runtime: Torch `2.14.0+git2e66fe81.rocm10.2.0`, HIP `7.17.26416`, ROCm SDK
`10.2.0.dev20261020+git535d2990`, RCCL `2.32.3` (`f92b29c+`), Triton `3.8.0`,
native x86 Python `3.14.4`. Both devices authenticated as `gfx1201`.

The first two attempts stopped before codec execution or timing. The second
retained RCCL warnings identifying an 82,576,044-byte shared-memory request
that exceeded the inherited kernel harness's 16 MiB file limit; its 64 MiB
shared-memory mount was also insufficient. The initial launcher additionally
lacked the selected server's explicit RCCL loopback/InfiniBand/log settings.
Restoring those settings alone did not resolve the failure.

The successful fresh packet used a 128 MiB worker file limit and repartitioned
the same 320 MiB scratch allowance into 64 MiB `/tmp` and 256 MiB `/dev/shm`.
A CPU-only namespace check allocated both exact shared-memory segments first.
The 16 MiB output-file, 32 MiB aggregate output, 38 GiB process-memory and
600-second worker bounds remained enforced. Nine protected review helpers
were unchanged. All three attempts independently released the GPUs.

The successful worker observed 90.47 MiB of Torch tensor allocation per card;
RCCL workspaces, framework contexts and JIT memory are excluded from that
figure. RCCL InfiniBand probe and hostname-resolution warnings remain recorded.
Sixteen focused source/reader tests passed on native Python 3.14.4 before each
packet. No model started, and no graph was captured or replayed.

## Independent review

Astra at max reasoning verified the two evidence seals, source/runtime/rank
joins, all timing pairs and independent release. It supports the narrow
component cost result. That review retained cancellation loss and missing
per-operation serving evidence as applicability gaps. The subsequent route
diagnostic addresses the sampled shape/dispatch gap. Actual-activation error,
exposed cost and candidate graph behavior remain unverified. The review itself
did not run native GPU tests.

## Decision and reproduction

Retain this as a **prefill-only component candidate**. Reject its current eager
composition for small decode messages. Do not select it for serving on this
evidence. The selected-model diagnostic confirms sampled in-place distributed
all-reduce calls at `[2048, 5120]` on both ranks. It does not measure their
exposed cost or make the synthetic communicator identical to the server's.
Actual-activation error and candidate execution evidence must precede any model
adoption, with an explicit numerical acceptance decision.

The source-only [prototype](../prototype/lossy-prefill/codec.py),
[kernels](../prototype/lossy-prefill/kernels.py) and
[CPU tests](../prototype/lossy-prefill/test_codec_contract.py) are inert on
import. They require an authenticated caller, native runtime, physical-device
mapping, finite watchdog and independent release to execute on GPUs. They are
not a standalone serving patch or deployment command.

Run the CPU contract checks without importing Torch or Triton:

```sh
python3 -B -m unittest discover -s prototype/lossy-prefill -p 'test_*.py'
```

[Provenance and raw component measurements](../provenance/LOSSY-PREFILL-COMPONENT-20261025.json)
retain source hashes, all timing pairs, numerical cases, review disposition and
the sealed local evidence paths. Rollback is non-adoption; selected source,
weights, precision, cache settings and runtime pins remain unchanged.
