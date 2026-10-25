# C2 attention and cache-sharing component

Status: **component-tested on both GPUs, independently reviewed, unselected**.
The unconditional two-request split adapter failed its declared benefit gate.
Serving, task quality, scheduler cancellation and lifecycle validation are
**NOT RUN**. The selected singleton adapter and serving configuration remain
unchanged. [Provenance](../provenance/FP8-C2-ATTENTION-COMPONENT-20261024.json)

The isolated test completed 16 cells, 32 graph captures, 1,264 poison-checked
graph replays and 1,024 alternating timing observations. Candidate output bits
matched two selected singleton adapters. Native C2 replay matched its own frozen
eager reference. Native and candidate outputs were not bit-identical: maximum
absolute BF16 differences ranged from 0.0009765625 to 0.0078125. This follows the
declared selected-singleton reference contract; it does not qualify model quality.

The following steady-state medians are milliseconds per isolated C2 call,
including identical declared `index_copy_` cache writes, metadata and attention
with reductions. They are synthetic-operand component observations, not model
latency, output tokens/s or per-card serving throughput.

| Prefix lengths | GPU 0 native / candidate | GPU 1 native / candidate |
| --- | ---: | ---: |
| 0 / 31 | 0.04966 / 0.09124 | 0.05032 / 0.09198 |
| 31 / 2040 | 0.45429 / 0.12306 | 0.39311 / 0.11098 |
| 2040 / 2040 | 0.39683 / 0.13070 | 0.39635 / 0.13002 |
| 8192 / 1024 | 1.48472 / 0.18812 | 1.48038 / 0.18720 |
| 26000 / 127 | 4.67793 / 0.37506 | 4.65202 / 0.37383 |

All 256 paired comparisons in the larger C2 cells favored the candidate. The
shortest case lost all 48 steady pairs and 14 of 16 flush-condition pairs. Its
median ratios of 1.8037–1.9250 fail the unchanged maximum-regression ratio of
1.02. The failed general gate remains failed despite the longer-prefix gains.
All ranges and individual observations remain in the sealed evidence.

A separate address-sharing probe compared eight decode rows sharing one KV
copy with four rows using each of two equal copies. Shared/duplicated median
ratios were 1.0005–1.0148 at 2K, 0.9619–0.9824 at 26K and 0.9753–0.9889 at
60K; some ranges overlap. These graphs contain native decode and reductions;
their KV contents and split metadata were prepared before capture. They do not
include cache-write or split-setup cost, measure DRAM traffic or qualify a new
multirow kernel. The 4 MiB and 64 MiB zero-write conditions do not prove cold
cache.

Peak allocated device memory was 261,237,248 bytes, below the 256 MiB tensor
limit. Host peak was 2,266,279,936 bytes. Final phase allocations were zero and
no OOM or resource event occurred. Payload time was 22.607 seconds; the owner
took 41.829 seconds. Independent release passed with no surviving owned
processes, cleanup errors or unresolved journal. An optional TVM Torch C-DLPack
extension JIT warning was retained; its allocator feature was disabled.

The native seal is `b7d4f917f265279648125e3345780f828f1031339458a5f275413dac1f12bafd`
(155 members), under private evidence root
`scratchpad/fp8-attention-v5-gpu-run.hv67x3sw`. The source seal is
`1da0cef942f1bc8b91bc4acc662952f34ce2bab240e0fb2fd5830c76dfefa87d`.
Astra independently reproduced the reader result and authenticated admission,
runtime/source identities, both physical devices, timing and release joins.

Python branching inside the captured adapter cannot switch routes by live
prefix length during replay. Separate capture keys would need an authenticated
host length source and explicit buffer/pool ownership. Reducing metadata launch
cost is a separate hypothesis; changing sixteen singleton split decisions to a
single sixteen-sequence policy is not established as equivalent.
