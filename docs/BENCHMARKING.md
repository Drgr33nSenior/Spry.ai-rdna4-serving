# Matched native timing screen

Status: no complete matched result is available. The first attempt retained
BF16-arm samples but failed collection/release acceptance. A later attempt
stopped after a rank-0 GPU page fault and abort; KV4 did not start. Its independent
release checks passed. The failing GPU operation remains unidentified.
The timing entry and CPU fixtures are separate from native ownership and platform observations.
Public admission, source promotion and deployment remain closed.

## Fixed comparison

[The screen profile](../profiles/benchmark-fp8-kv4-modelrunner.json) freezes the
first comparison: ordinary BF16 KV, then KV4, with one isolated model load per
arm and no retry or replacement start. Keep the same authenticated private
SGLang 0.5.21 composition, official FP8 weights, TP2, context 4096, 4096 physical
KV slots per rank, page size 1, `no_buffer` and eager execution. Do not use the
graph-enabled qualified FP8 baseline's historical times as the control.

The BF16 arm selects the existing ordinary ModelRunner branch with no private
KV4 request and requires the native BF16 pool without a KV4 owner. The KV4 arm
retains its full composition/build proof, native binary and ownership gates.
Neither branch bypasses a proof or changes weights, runtime arithmetic or the
KV4 codec. Both source/cache identities must be authenticated before execution.

Use these cells in order, with exact frozen token vectors and 32 generated
tokens per request. The two C2 counterparts are distinct frozen requests.

| Cell | Concurrent requests | Input tokens per request | Output tokens per request |
| --- | ---: | ---: | ---: |
| C1-128 | 1 | 128 | 32 |
| C1-1024 | 1 | 1024 | 32 |
| C1-3966 | 1 | 3966 | 32 |
| C2-128 | 2 | 128 | 32 |
| C2-1024 | 2 | 1024 | 32 |

A 4096-token prompt plus output exceeds context. Two near-4K requests exceed
the fixed aggregate pool even when each request fits its context. Do not enlarge
the pool to fill a benchmark cell. Each cell gets one retained warm-up batch and
five measured batches: 30 batches and 42 requests per arm, 84 requests overall.
One prefill produces the first token; 31 decode forwards produce the remaining
31. The requested total is 1920 text forwards per rank across both arms.

Release requests and reset the request cache between batches. Require restored
pool capacities, zero prefix-cache hits and actual batch shapes on both ranks.
These observations do not prove every initial recurrent-state tensor is zero.
Use fresh private compiler/cache directories for each arm; preserve accepted
caches. Retain every sample and any compilation inside a measured operation.

## What the timer measures

The entry uses host wall time with GPU synchronization around ForwardBatch
construction, ModelRunner.forward and sampling. Existing adapter validation,
copies and synchronization remain included. Finite-logit checks, TP token
agreement, result collection and cache reset are outside the operation timer.
Their costs remain in the separately reported instrumented batch wall.

Aggregate each aligned operation with the maximum rank time. Retain individual
warm-up and measured values; report median and range for prefill, decode and
generated-token rates. This is not HTTP request latency, time to first token,
production inter-token latency or scheduler throughput. Instrumentation can
change GPU idle periods and clocks. There is no profiler or kernel-only gain
claim.

The crash-evidence repair records startup phases and batch intent/completion
outside the forward/sample timer. It exports bounded rank-log tails after
worker reaping, including when final rank results are absent. These writes
can affect startup, scheduling and arm wall time; they introduce no additional
GPU operations or requests. Preserve partial counts as lower bounds and missing
counts as unknown. A valid diagnostic transport never makes a failed arm usable
for the matched comparison. See the V2 limits in [host ownership](HOST-OWNER.md).

This first screen gates finite, correctly shaped logits, actual batch size,
TP agreement and effective release. Cross-precision generated differences are
data, not malformed responses. Compare output scores only for matching tokens
and conditioning histories. No retained threshold establishes lossy KV4 model
fidelity; this screen keeps model/task quality unqualified. A quality-qualified
serving comparison needs its own justified numerical and task contract.

One fixed-order pair of loads is a preliminary screen, not an independent-start
speed ranking. Report small wins and variability without a hidden 5% cutoff.
Do not change FP8 GEMM settings, graphs, precision or GDN state to improve a cell.

## Execution boundary and remaining work

`tools/model_benchmark.py` is an inner entry, not an admission controller. Its
explicit execution switch is necessary but insufficient authority. The old
private smoke launcher has process supervision but no canonical FD9 lease or
durable unresolved-session record. The reviewed FP8 owner supplies those
controls but binds cleanup and its reader to exact Pod UID/CRI evidence. A
bwrap process is not a Pod; replacing only that owner's phase could falsely
report no creation after GPU use. Do not fabricate Pod disposal records.

The new `host_process_owner.py` and `benchmark_owner_adapter.py` implement the
host-process contract with focused CPU tests. They hold authenticated FD9,
persist launch intent before child execution, bind process/source identities,
use pidfd-only live signaling and retain the refusal record until independent
closeout succeeds. Namespace evidence must name the actual private namespaces,
not the pre-execution gate's host namespace. See [host ownership](HOST-OWNER.md).

Before admission, bind the adapter into a fresh physical controller and select
an authenticated read-only platform observer. No suitable non-escalating observer
was found in the retained sources. The old complete checks use sudo; ordinary
unprivileged fuser silence is not all-user holder or GPU-allocation proof. Do not
invent a helper, bypass current access policy or substitute CPU observations.
No historical controller may run in place, and no competing workload may be
stopped to obtain the slot.

The proposed complete envelope is 2100 seconds: 120 admission, two arms each
with 240 load/warm-up + 450 measurements + 60 owned cleanup, 180 collection and
independent release, and 300 sealing. This corrects the earlier 1800-second
proposal, which budgeted only one load. No physical clock has opened. If the
complete future owner envelope cannot fit, stop before admission.

Run local fixture checks only:

```sh
make test-benchmark PYTHON=/path/to/python3
make test-benchmark-readout PYTHON=/path/to/python3
make test-host-owner PYTHON=/path/to/python3
make test-benchmark-bindings PYTHON=/path/to/python3
```

CPU substitutes test the entry, schedule, timing aggregation and cleanup calls.
They do not establish native import, physical ownership or GPU release. A future
run must bind final tested source, frozen prompt/spec hashes, complete runtime,
all model files and both physical devices through the reviewed owner interface.

`tools/analyze_model_benchmark.py` reads the frozen spec and two future arm
reports offline. It rechecks the inner records, retains warm-ups, reports the
five measured samples per cell and compares generated scores only before token
histories diverge. Its latency ratios describe that cell and metric, not an
overall serving winner. The analyzer never grants cleanup authority or infers
protected-host release from worker process-group records.

The later owner-run snapshot passed its availability checks. The new local
observer, resource gate and controller candidates are described in
[host ownership](HOST-OWNER.md#live-binding-candidates). They do not make the
benchmark launch-ready. Their complete-envelope proposal is 2400 seconds,
separate from the retained unfunded 2100-second draft. Do not shorten cleanup
or silently reuse an old grant to fit the new observation bounds.

The latest local components include preweight native identity, bounded raw
result export, final resource-event checks and a human-root observation
transport candidate. The OCI-derived expanded-base proof, complete installed
prefix and final source/model preflight now pass. Original and copied observer
evidence are included in the bounded session storage partition. See
[verified runtime closure and human launch](HOST-OWNER.md#verified-runtime-closure-and-human-launch).
The generated one-use wrapper requires a human terminal for privileged host
observation; GPU work remains UID/GID 1000. Preparation supplies no GPU timing,
native resource/release qualification or automatic deployment.
