# Sprint 23: common M16 attention QKV component

The component matched all 96 numerical comparisons, but the candidate was slower in one of four timing cases. The all-cases advancement gate therefore parked it. No Sprint 23 serving or quality screen was attempted; the selected Sprint 22 source and ring-fold profile remain unchanged.

The candidate changes only the untuned HIP/BF16 common-kernel attention QKV shape `(8, 7168, 5120)` from M64 to M16. Both variants use stage 3, N128/K128, group 32 and four warps. The selected gate/up and GDN output M16 routes, unrolled routes, other kernel bodies and protected helpers remain unchanged.

| GPU | Pattern | M64 median, ms | M16 median, ms | M16 time change |
| --- | --- | ---: | ---: | ---: |
| 0 | Structured | 0.072265 | 0.066460 | −8.0329% |
| 0 | Random | 0.061340 | 0.062935 | +2.6003% |
| 1 | Structured | 0.066275 | 0.065190 | −1.6371% |
| 1 | Random | 0.062639 | 0.060099 | −4.0549% |

The 96 exact comparisons comprise 72 eager and 24 graph comparisons; timing used 64 alternating eager launches. Structured inputs used an independent analytical oracle; random outputs were compared with the selected M64 baseline. Both cards ran sequentially on synthetic inputs. Graph replay checked correctness, while the timing used eager launches. GPU event intervals include submission effects; no hardware occupancy or bandwidth counters were collected.

Compiler records report 256 versus 123 registers and no spills, with 49,664 versus 37,376 bytes of shared memory for M64 and M16. Peak allocated device memory was 37,916,672 bytes. The component took 23.6047 seconds, started no models and passed independent release. The final composed CPU suite passed 45 checks; 25 focused checks also passed. Fresh attention geometry was derived from source, configuration and TP2 partitioning, not observed from live weights.

The selected baseline is `b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069`; the parked candidate is `f02ef47fc5ed25cfcd3d59eda57bf8769efb2029bdf7293fa797f1802356cd2c`. The [provenance record](../provenance/SPRINT23-ATTENTION-COMMON-M16-20261022.json) binds the source, native and host-observation seals. The 121.18–121.24 combined TP2 tokens/s in [Sprint 22](SPRINT22-GDN-OUTPUT-M16.md) remain its serving result, not a Sprint 23 measurement.

No automatic retry or source selection followed the mixed result. This component does not establish model quality, a hardware cause, or resolution of prior numerical findings. Reliability and endurance remain deferred.
