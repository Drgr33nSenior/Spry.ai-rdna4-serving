# Architecture and precision boundaries

This page explains how serving source becomes a validated result and how model
weights, attention KV and recurrent state differ. The diagrams describe the
repository on 2026-11-02; they are functional views, not an installed deployment.
See the [documentation guide](README.md) for a reading path.

## From source to retained evidence

The cleaned serving path assembles an exact source selection before native
work. A separate reviewed caller owns the bounded workstation operation.
Model weights and the complete runtime stay outside this checkout.

```mermaid
flowchart TB
    accTitle: Serving source to retained evidence
    accDescr: Authenticated source inputs form a reviewed packet for one bounded workstation run, followed by retained artifacts and validated summaries.
    source["Repository source and configuration"]
    donor["Private sealed donor<br/>and pinned effective base files"]
    builder["CPU builder and proposal selection"]
    packet["Source archive and reviewed bindings"]
    source -->|"selected code and profile"| builder
    donor -->|"authenticated inputs"| builder
    builder -->|"deterministic source bytes"| packet
    subgraph workstation["Owned Linux validation workstation — tst"]
        runtime["Bounded SGLang run<br/>Protected owner and TP2 workers on two R9700s"]
    end
    packet -->|"one-use caller, exact approval and strict SSH"| runtime
    archive["Sealed private artifacts"]
    result["Published result summaries"]
    runtime -->|"HTTP, startup and release evidence"| archive
    archive -->|"validated replay and approved CPU grade"| result
```

The final readout combines local CPU replay with an isolated Linux CPU grade
when the approval permits it. The [caller sequence](../serving/launcher/README.md)
shows that ordering and the failure path. CPU checks alone do not establish
native ownership or GPU release.

| Source | Responsibility |
| --- | --- |
| [Serving source](../serving/README.md) | Configuration, assembly, runtime hooks, receipt validation and final-answer grading |
| [Patch series](../serving/patches/README.md) | Reproduce active overlays from their exact effective base files |
| [KV4 prototype](../prototype/) | Separate codec, attention kernels and selected SGLang source overlay |
| [Provenance](../provenance/) | Recorded identities, selections and bounded result summaries |

The diagram is grounded in `serving/build.py`, `propose_packet.py`,
`prepare_gpu_run.py`, `launcher/execute_once.py` and `quality/retained_replay.py`.
The completed packet is recorded in the
[cleanup sign-off](../serving/measurements/native-qualification/CLEANUP-SIGNOFF.md).

## Weights, attention KV and recurrent state

A weight format does not choose a cache format. The completed W5 cleanup uses
BF16 target and draft attention KV. The separate KV4 proposal uses official FP8
model weights with a mixed attention-cache policy. Its public admission remains
closed.

The documented hybrid Qwen model has 64 physical layers. This diagram shows
alternative cache policies; the two full-attention branches are not successive
steps or simultaneously selected modes.

```mermaid
flowchart TB
    accTitle: Attention cache and recurrent-state boundaries
    accDescr: The hybrid model has separate full-attention and recurrent layers. BF16 W5 and the separate mixed KV4 proposal are alternative attention-cache policies.
    model["Hybrid Qwen model<br/>Weight identity is frozen separately"]
    full["16 full-attention layers"]
    gdn["48 GDN recurrent layers"]
    bf16["W5 cleanup<br/>BF16 target and draft KV"]
    mixed["Separate KV4 proposal<br/>4 protected BF16 layers + 12 packed KV4 layers"]
    state["Recurrent-state pool<br/>Separate allocation, snapshots and rollback"]
    model -->|"contains"| full
    model -->|"contains"| gdn
    full -->|"validated cleanup policy"| bf16
    full -.->|"prototype cache policy"| mixed
    gdn -->|"owns state in"| state
```

The layer counts and prototype policy come from the
[model fixture](../prototype/tests/fixtures/qwen3.8-27b-fp8.config.json) and
[cache contract](../prototype/image-source/python/sglang/srt/layers/attention/ultraquant_contract.py).
The W5 policy is explicit in its
[run configuration](../serving/config/runs/w5-v2/run.json).

## KV4 geometry and arithmetic

| Property | Recorded prototype contract |
| --- | --- |
| Full-attention physical IDs | `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63` |
| Protected BF16 IDs | `3,7,59,63`; both K and V remain unrotated BF16 |
| Compact layers | The other twelve full-attention layers use local G32 E2M1/UE8M0 storage, with interleaved 17-byte groups |
| Per-rank TP2 geometry | 12 query heads, two KV heads, head dimension 256 and GQA ratio six |
| Transform | Compressed Q/K use FP32 Walsh-Hadamard transform after native MRoPE; V stays unrotated |
| Native arithmetic | BF16 input/output bits, local unpacking and FP32 accumulation |

This is a hybrid protection proposal, not a paper-qualified layer mapping.
The native ABI does not reinterpret BF16 as FP16 or claim native FP4 arithmetic.
Local packed bytes are not interchangeable with another engine's cache format.

The adapter joins the existing hybrid pool and backend through the parent's
physical-to-compact layer mapping. Cache identity binds rank, process, model,
source/build and policy. The owner protects cached prefixes and validates slot
generations before reads and writes. Failed preparation invalidates stale
metadata; packed bytes cannot be read through the ordinary BF16 path. The
historical Llama/TP1/FP16 guards remain separate.

## Memory accounting

The mixed KV payload is 14,720 bytes per physical token per rank. A 4,096-slot
plan therefore has 60,293,120 payload bytes, plus 14,720 bytes for reserved slot
zero and a four-byte native status element. Metadata, allocator costs and GDN
reservation are additional. The 2 GiB workspace reserve is policy, not measured
use. The bounded Q/K/V contiguous-normalization temporary is at most 32 MiB.

For measured cleanup memory, use the
[operation-207 report](../serving/measurements/native-qualification/CLEANUP-SIGNOFF.md).
Its sampled driver values and startup byte witnesses have different scopes
from the prototype payload calculation above.
