# Current serving verification

These results describe bounded private runs on two Radeon AI PRO R9700 GPUs.
They do not turn this source prototype into an installable or qualified serving
distribution. The runs used a complete separately pinned SGLang runtime.
Machine configuration, model weights, native binaries and raw logs are excluded.

## FP8 thinking checks, 2026-10-21

The published Qwen3.8-27B-FP8 checkpoint passed the fixed J01 ledger task in two
fresh starts at low reasoning and two at the highest published effort, `xhigh`.
Every answer rejected a debit of 31 plus a fee of 4 against a balance of 34.
Each pair returned identical output token IDs. Low effort produced 425 output
tokens per start; xhigh produced 334. Exact prompt IDs and final-answer decoding
were checked against the checkpoint's published tokenizer.

The xhigh worker timeout was 1800 seconds. Requests completed in about 14 seconds;
the timeout includes separate startup work and was not a measured request time.
The earlier wrong answer was not reproduced on this short prompt. Long-context
quality and the combined speculative path were outside these checks.

The stack was PyTorch `2.14.0+git2e66fe81.rocm10.2.0`, ROCm SDK
`10.2.0.dev20261020+git535d2990`, HIP `7.17.26416` and RCCL `2.32.3`.
Private startup compatibility work rebuilt the retained SGLang kernel and pinned
TorchVision source against that Torch build, restored SDK tool executable modes
in a byte-identical runtime copy, exposed the matching HIP development SDK and
redistributed scratch space to provide 512 MiB of shared memory. The original
sealed inputs and installed defaults were preserved.

## Matched previous/current stack benchmark

One ordered previous/current pair and one current/previous pair used identical
requests: 451 input tokens, exactly 128 output tokens, low thinking and concurrency
one. Each stack had two fresh starts, each with one excluded warm-up and three
measured requests. All 16 requests completed, with zero cached tokens and zero
retractions. All output token vectors matched.

| Stack | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| PyTorch 2.13 / ROCm SDK 10.1 | 24.59 | 5.2062 s | 5.1949–5.2224 s |
| PyTorch 2.14 / ROCm SDK 10.2 | 24.57 | 5.2096 s | 5.1922–5.2158 s |

The combined median rate changed by -0.07%. The two orders showed opposite
changes of about 0.3%. No meaningful performance regression was detected in this
screen. This small sample does not establish equivalence for other workloads.

Both arms used the same FP8 target, BF16 attention cache, FP32 recurrent state
and head, greedy sampling, seed 12345, Triton attention, decode graphs and
hipBLASLt preference. **DFlash2 and the split K/V decode adapter were disabled.**
This is a whole-stack comparison, not an attribution to Torch, HIP or RCCL alone.

The rate is output tokens divided by local non-stream HTTP request-to-terminal
time. It includes prefill and protocol completion. It is not isolated decode
throughput, TTFT or ITL. `ignore_eos=true` forces the fixed output count; these
truncated responses were not graded as completed task answers. Long contexts,
concurrent traffic, sustained load, power and thermal behavior remain untested.

## Combined DFlash2 and split K/V benchmark

The combined path completed on the current stack with the pinned BF16 DFlash2
drafter at block size 8, the retained per-row native split-decode adapter and
aligned GDN, together with the FP8 target, FP32 head, GEMM optimization and
decode graphs.

| Current-stack configuration | Median output tokens/s | Median request time | Observed request time range |
| --- | ---: | ---: | ---: |
| Ordinary FP8 serving | 24.50 | 5.2246 s | 5.2237–5.2273 s |
| DFlash2 + split K/V + aligned GDN | 86.68 | 1.4766 s | 1.4753–1.4808 s |

The median rate increased **3.54×** on this workload. Each arm had one excluded
warm-up and three measured requests, using the same 451-input/128-output-token
workload described above. All eight requests completed with zero cache hits and
zero retractions. All four output-token pairs matched exactly.

Both target ranks recorded 16 adapter capture calls across all 16 attention
layers, with an allocated adapter buffer set. Native validators authenticated
graph replay and speculative verification. The four combined-arm responses
reported 100 verification passes, 700 proposed drafts and 412 correct drafts;
these counters are not completion-token totals.

This is one ordinary/full arm order. It measures the whole combined path and
does not isolate each optimization. The layer-normalization candidate and
FP4/KV4 cache were excluded. The truncated outputs were not graded as completed
answers. The HTTP timing and workload limits described above apply.

The stock verification split-KV kernel is gated to another GPU family in the
retained source. On R9700, the experimental adapter invokes native decode
attention with the split selector applied separately to each verification row.
This is separate from the four-bit KV cache prototype.

### Memory observations from the combined-path benchmark

The retained telemetry includes a host-memory peak for each complete arm and
GPU startup figures for each rank. The figures below use GiB (2^30 bytes).
SGLang labels its binary-unit GPU figures as `GB` in the logs. Both ranks
reported the same rounded GPU values.

| Host-memory metric, whole arm | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Cgroup memory peak, including startup and warm-up | 18.32 GiB | 23.69 GiB |
| Configured cgroup memory limit | 38 GiB | 38 GiB |

These peaks are Linux cgroup v2 `memory.peak`, including charged page cache,
tmpfs and child processes. They are not process RSS or GPU VRAM. Swap was
disabled. Neither arm recorded a memory-limit event, OOM or OOM kill.

| GPU startup metric, per GPU | Ordinary FP8 | DFlash2 + split K/V + aligned GDN |
| --- | ---: | ---: |
| Target weight-load free-memory delta | 14.51 GiB | 14.51 GiB |
| Drafter weight-load free-memory delta | — | 2.16 GiB |
| Target BF16 KV pool, K + V | 0.25 + 0.25 GiB | 0.25 + 0.25 GiB |
| Drafter BF16 KV pool, K + V | — | 0.08 + 0.08 GiB |
| Recurrent-state pool slots | 93 | 19 |
| Recurrent convolution / SSM state | 0.13 / 6.61 GiB | 0.03 / 1.41 GiB |
| Speculative intermediate SSM / convolution state | — | 1.69 / 0.01 GiB |
| Target graph-capture memory delta | 0.29 GiB | 0.38 GiB |
| Drafter graph-capture memory delta | — | 0.10 GiB |
| Available GPU memory after graph capture | 9.11 GiB | 10.13 GiB |

Each attention KV pool had 16,416 token slots. The runtime sized the recurrent
pool differently for speculative execution, which also requires intermediate
state. The greater free GPU memory in the combined arm therefore does not
establish better memory efficiency at equal capacity. Both configurations
completed the measured concurrency-one workload.

GPU figures are rounded startup observations, taken before external requests.
They are not a complete additive memory accounting or a measured GPU peak.
No continuous VRAM samples, Torch allocator peaks, per-request host peaks or
per-process RSS were retained. Exact host peak bytes, per-rank GPU figures and
capture hashes are included in the sanitised JSON linked below.

## Evidence identities and limits

| Completed observation | Private local evidence manifest SHA256 |
| --- | --- |
| FP8 low/xhigh verification | `780765826ccbbdc5bce91064dcf74eb17b3adee78c785100ce705802235e685e` |
| Previous/current HTTP benchmark | `47f91427c597fdec16f374686bcd4e17ed338a3c383a4a5b0c6cd8e4ef433fd0` |
| Combined-path native benchmark | `ff2061124636c6894ea1e037173dd272783d47869ce89b78448f8216e116f3af` |

Sanitised measurements and source/native evidence identities are retained in
[SERVING-RESULTS-20261021.json](../provenance/SERVING-RESULTS-20261021.json).

Source, native worker, HTTP, loaded-weight, graph, resource and independent
release checks passed for those completed observations. The benchmark timings
were independently recomputed from retained monotonic timestamps. These hashes
identify private evidence; they are not portable runtime/build proofs or launch
authority. Optional TVM Torch C DLPack and Python 3.14 TorchScript warnings remain.
No production deployment or broad quality qualification was performed.
