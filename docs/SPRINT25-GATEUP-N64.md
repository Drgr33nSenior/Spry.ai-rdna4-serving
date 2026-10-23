# Sprint 25: common gate/up N64

The N64 candidate is parked after its fixed-work serving median failed the lower-latency gate. The model-free component passed 96 exact comparisons and was faster in all four component timing cases. The selected Sprint 22 GEMM and ring-fold profile remain unchanged. [Provenance](../provenance/SPRINT25-GATEUP-N64-20261023.json)

The candidate changes only the untuned HIP/BF16 common gate/up shape `(8, 17408, 5120)` from N128 to N64. Both variants retain M16, K128, stage 1, group 32 and four warps. Quantization block scales remain 128 by 128. Other selected routes and kernel bodies are unchanged.

| GPU | Pattern | N128 median, ms | N64 median, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | Structured | 0.187866 | 0.186851 | 0.5403% |
| 0 | Random | 0.186731 | 0.185511 | 0.6533% |
| 1 | Structured | 0.183361 | 0.182341 | 0.5563% |
| 1 | Random | 0.188796 | 0.186946 | 0.9799% |

The 96 comparisons comprise 72 eager and 24 graph checks; timing used 64 alternating eager launches. Structured inputs used an independent analytical oracle; random outputs were compared with the selected N128 baseline. Both GPUs ran sequentially on synthetic inputs. Graph replay checked correctness, and eager launches supplied timing. Event intervals include submission effects; no hardware occupancy or bandwidth counters were collected.

Compiler records report 114 versus 91 registers, no spills, and 16,384 versus 8,192 bytes of shared memory for N128 and N64. Peak allocated device memory was 90,522,112 bytes. The component took 24.1042 physical seconds, started no models and passed independent release. Its component gate required every case to match exactly and have a lower candidate median; the gate passed. The focused and full CPU suites passed 26 and 46 checks.

The forward serving pilot used control then candidate, two starts and eight fixed-work requests, with one excluded warmup and three measured requests per start. Each request used 2,047 input tokens and 1,024 forced output tokens, C1 and low thinking. Both arms retained ring-fold with 19 usable and 20 physical slots. All four output-ID pairs and verification-cycle counts matched, including the warmup; every request had 164 cycles. Affected-shape evidence covered both ranks and arms, all limits passed and independent release was verified.

| Serving median | N128 control | N64 candidate |
| --- | ---: | ---: |
| Client latency, seconds | 8.464984 | 8.468574 |
| Output tokens/s, combined TP2 | 120.968929 | 120.917646 |

The candidate median latency was 0.04241% higher and its rate 0.04239% lower. This small difference failed the predeclared lower-median gate; it does not establish a significant slowdown. The native launch lasted 325.9261 seconds, with no native fault. There was no reverse run, retry, sampled or natural-stop candidate quality screen, or source selection. The matching greedy fixed-work outputs do not resolve the sampled differences or the earlier numerical finding.

Two initial local CPU checks used stale component fixture paths; their failed receipts were retained and the corrected checks passed. A later offline lookup used the wrong launch-receipt filename and was corrected without changing native artifacts.

An offline review of [Sprint 24](SPRINT24-CURRENT-PROFILE.md) found two native selector random draws use the default device generator without an explicit generator or per-request seed. The recorded server seed does not establish the state or draw offset at the first selector call. Actual preceding draw consumption and matched selector inputs and draws were not observed. The sampled differences therefore establish neither an RNG cause nor a numerical error; the earlier numerical finding remains unresolved. A possible future diagnostic must first establish supported access to the relevant RNG state and account for observer effects. It is not authorized by this record.

The selected baseline source is `b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069`; the candidate is `a19b38e86bdd8366ffdb8f35ca7dbf1aa6845f7cfde1aa55f1f60aa606bcb9ef`. This component does not establish serving speed, model quality, numerical resolution or sampling distribution equivalence. Reliability and endurance remain deferred.
