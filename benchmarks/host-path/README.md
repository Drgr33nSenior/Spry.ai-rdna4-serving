# Host observation cleanup

This source bundle records the host-path A/B candidate prepared on 25 October
2026. It changes benchmark observation overhead. It does not change model
arithmetic, weights, attention, recurrent state, speculative width or GPU kernels.

| Logical arm | Commit counter | Decode log interval |
| --- | --- | ---: |
| `control` | Open, truncate, write eight bytes and close after each commit | 1 |
| `mtp` | Create one eight-byte shared mapping; update it after each commit | 100 |

Both arms use the same selected FP8 target, BF16 draft and KV caches, FP32
recurrent state/head, b101 GEMM, aligned ring-fold, DFlash width eight, low
thinking and 2,048-token prefill chunks. The arm names are historical harness
identifiers; both arms enable DFlash.

The candidate retains metrics, graph replay counts, completed-commit counts,
per-request API verifier counts, synchronous before/after observations and
durable request journals. Identity checks, finite bounds and independent
resource release remain required. This is still an instrumented benchmark.

Periodic native summaries and acceptance averages can refresh up to 99 decode
batches later. The independent API/graph/commit counters remain live. Startup
probes already restored their hooks before requests; this change does not
remove them or relax their checks.

[HOST-PATH.patch](HOST-PATH.patch) shows the changes against the authenticated
final baseline packet. The integration hunk applies to its unpacked logical
source, which is losslessly packed for the physical packet. The source inventory
binds the published files to that packet. The eight protected owner helpers and
all 16 numerical native source roles are unchanged.

Run the portable source tests from this directory:

```sh
python3 -B test_host_path.py
python3 -B test_host_contract.py
```

The 14 whole-module tests cover counter visibility, exact layout, syscall removal,
finite limits, source substitution, control behavior and resolved configuration.
One additional test exercises the actual worker and reader predicate blocks;
it is not a full worker startup test. All 15 passed on the workstation's x86
Python 3.14.4. The native source check also compiled 48 complete packet modules
without importing Torch, Triton or SGLang or accessing GPUs.

These files are review and source-test artifacts, not a deployment launcher.
Runtime testing uses the existing separately registered, source-pinned owner
procedure. BetterBench uses the unchanged [native client](../betterbench/native-v2/README.md).
Its task outputs have no answer oracle. Different generated tokens and verifier
work must remain visible when comparing rates.
