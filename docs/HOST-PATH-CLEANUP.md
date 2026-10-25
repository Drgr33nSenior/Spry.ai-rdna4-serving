# Host observation cleanup, 25 October 2026

The cleanup reduced decode logging and removed per-commit file syscalls from
the candidate path. The two-start BetterBench comparison completed all 106
requests and passed strict source, counter and independent release checks.
It did not establish a useful performance improvement. Keep the implementation
as an opt-in experiment; the selected serving configuration remains unchanged.

| Measure | Control | Cleanup candidate |
| --- | ---: | ---: |
| C1 aggregate output tok/s | 83.12 | 77.85 |
| C1 median individual request seconds | 6.580 | 6.325 |
| C1 output tokens / six-request group | 4,610 | 4,221 |
| C1 API verifier cycles | 1,134 | 1,114 |
| C2 aggregate output tok/s | 132.21 | 135.25 |
| C2 median individual request seconds | 6.146 | 7.015 |
| C2 output tokens / six-request group | 3,851 | 3,774 |
| C2 API verifier cycles, summed over requests | 979 | 913 |
| Decode-batch log lines, whole start | 8,007 | 76 |
| Captured stderr bytes, whole start | 1,944,591 | 82,780 |

Rates include prefill and HTTP completion across both GPUs. The observed
throughput changes were −6.34% at C1 and +2.30% at C2. C1 group wall time fell
from 55.46 to 54.22 seconds while producing fewer tokens. Outputs and verifier
work differed, so neither a causal slowdown nor speedup follows. The predeclared
5% useful-effect threshold was not reached. No reverse-order campaign was run.
The earlier 120/173 tok/s Sprint 39 figures used a different fixed-output
workload and are not this comparison's control.

Measured prefill inputs, excluding warmups, were approximately 1.5K, 5.9K and
11.8K tokens. Client content TTFT remained about 0.45–0.47, 1.71–1.73 and
3.60–3.65 seconds respectively. These are client measurements, not isolated
kernel prefill rates. Cache-token counts were unknown; cold-storage, uncached
prefill and native per-token ITL were not established.

## Change and retained controls

The control opens, truncates, writes and closes an eight-byte commit-counter
file after each target commit. The candidate creates one shared writable
mapping and updates the same `<Q` counter in place. It also sets
`--decode-log-interval 100` instead of `1`. Both ranks reported the intended
configuration with metrics enabled.

Graph replay, completed-commit and API verifier counters remain live. Request
journals, before/after observations, source identities, resource limits and
independent release are retained. The benchmark therefore remains instrumented.
Periodic native summaries and acceptance averages can refresh up to 99 batches
later. The mapping has not been qualified as an atomic publication protocol or
for endurance; the retained run establishes valid observed counter joins only.

All 16 numerical native source roles and eight protected owner helpers are
unchanged. Both arms retain official FP8 target weights, BF16 draft weights and
BF16 target/draft KV, FP32 recurrent state/head, b101 GEMM, aligned ring-fold,
metadata reuse, fixed DFlash width eight, low thinking and 2,048-token chunks.
The context is 32,768 tokens, token pool 32,800, and recurrent capacity remains
19 usable/20 physical slots per rank.

The runtime is Torch `2.14.0+git2e66fe81.rocm10.2.0`, ROCm SDK
`10.2.0.dev20261020+git535d2990`, HIP `7.17.26416`, RCCL `2.32.3`,
SGLang `0.5.21`, and Triton `3.8.0+git669b31ac.rocm10.1.0`.
The native source checks used x86 Python 3.14.4. Linux still reported
65,148,308 KiB of system RAM; the planned upgrade was not assumed installed.

## Verification and limits

The [source bundle](../benchmarks/host-path/README.md) includes the exact diff,
changed modules, predicate source and portable tests. All 15 focused checks
passed on native Python 3.14.4, and 48 complete packet modules compiled without
framework imports or device access. The repository's 43 source preflight checks
also passed. Astra reviewed the source/package and the actual bounded result.

All 47 serial-request native joins per arm passed on both ranks. C2 native
batch counts reconciled with weighted per-request API verifier counts. The
owner completed successfully in 1,072.42 seconds; independent process, namespace,
scope and GPU release passed. No memory-limit or OOM event occurred. Host
cgroup peaks were 32,631,619,584 and 22,116,564,992 bytes. They include startup,
client and file-cache effects; they are not per-card VRAM or a memory-saving
result.

BetterBench has no answer-quality oracle. The prior unchanged-arithmetic QA
passed 14 answer checks and two isolated code checks; those checks were not
rerun here. This client retained API token counts, protocol metadata and
selective terminal frames, not complete content SSE or token histories.
Historical numerical exceptions remain. Endurance and deployment are **NOT RUN**.

An extra offline attempt to read old baseline artifacts with the new observer
correctly refused their different source identity. No old evidence was changed
or regraded, and no physical retry was needed.

[Full measurements and provenance](../benchmarks/results/20261025-host-path/summary.json)
and the [standalone table](../benchmarks/results/20261025-host-path/summary.html)
retain the comparison. The native control and candidate reports are included
unchanged. Rollback is to retain the legacy host profile and the existing
selected runtime; no installed source or persistent service was changed.
