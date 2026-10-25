# Selected prefill communication route

25 October 2026. **Hardware-observed metadata; validated offline; no candidate selected.**
One selected Qwen3.8-27B-FP8 model start completed four requests on both R9700s.
The model kept its existing BF16 communication. This diagnostic did not inject
the [INT8 communication candidate](LOSSY-PREFILL-COMPONENT.md).

## Observed calls

Each request used an existing frozen low-thinking input vector of 25,991 or
25,978 tokens, a distinct cache salt, greedy generation and a 32-token output
cap. All four responses ended at that cap; none was graded as a complete answer.
The native hook verified the actual input vector and cold first 2,048-token
target prefill chunk on each rank.

The observer recorded the first 16 calls in that chunk: **64 calls per rank,
128 recorded calls across both GPUs**. All recorded calls had these properties:

| Property | Observation |
| --- | --- |
| Tensor | BF16 `[2048, 5120]`, contiguous, stride `[5120, 1]` |
| Payload per rank | 20 MiB |
| Dispatch | In-place path to `torch.distributed.all_reduce` |
| Returned tensor | Same object, data pointer and storage as the input |
| Groups | Attention and ordinary TP use the same group |
| Capture | Outside graph capture |
| Observer errors | Zero |

The two ranks' sampled route, shape and alias sequences matched. Each sample
counted **113 subsequent calls omitted after the record limit**; their shapes,
dtypes and routes were not characterized. This is a prefix sample, not a full
communication trace. Hooks were restored, and the existing target decode graph
replay counters advanced during each request.

## What this establishes

The large tensor size used in the earlier synthetic component test occurs in
the selected model. The actual server reaches the distributed process-group
API, whereas the component test used the single-process, two-device NCCL API.
Any candidate integration must preserve the observed in-place buffer contract.

No tensor values, device timers or extra model forwards were added. The
diagnostic did not measure exposed communication time, compression error on
model activations, candidate graph behavior, TTFT, ITL or comparable throughput.
Non-streaming response timing fields are retained as raw diagnostic output;
they are not used as decode-speed or TTFT measurements.

The earlier **16.8–17.9% component time reduction is still not a serving gain**.
The INT8 codec's adversarial cancellation loss remains unresolved. BF16
communication stays selected; no numerical acceptance contract changed.

## Validation and retained failure

Before launch, 55 focused tests passed and 63 modules compiled on x86 Python
3.14.4. Astra reviewed the source, owner, observer and exact packet. A missing
hosts bind was corrected before staging; earlier CPU fixture failures remain
retained. Eight protected helpers and all selected arithmetic sources stayed
unchanged.

The native worker completed all four requests. GPU/process/cgroup release was
independently verified, with no unresolved owner journal or cleanup error. The
owner operation took 201.98 seconds, including startup and supervision; that
duration is not a performance comparison.

Native stderr remains retained. It includes the optional DLPack extension JIT
failure, Python 3.14 TorchScript warnings, optional model-import warnings and a
Triton block-pointer deprecation warning. These were not suppressed or repaired
as part of the diagnostic.

The original owner result is **failed** because its final reader omitted three
required NumPy preflight files from the expected client-file inventory. A
separate reader correction validates those exact files, including byte equality
with the frozen dependency receipt. The complete retained evidence then passed
offline validation. Five real-data tests passed on native Python 3.14.4,
including tampered preflight data and unexpected-file rejection.
Astra independently reproduced the derived diagnostic from sealed evidence
and verified the correction, sampled calls, restoration and release records.
Its review supports this limited metadata result, with no promotion approval.

No GPU rerun was needed. The original failed record and both evidence seals
remain unchanged. [Provenance](../provenance/LOSSY-PREFILL-ROUTE-20261025.json)
records the original failure, corrected interpretation, source identities and
retained evidence paths. **Lossy model execution, task-quality comparison,
candidate graph validation, serving benefit and deployment: NOT RUN.**
