# FP8 target-KV diagnostic

The retained diagnostic supports keeping **BF16 target and draft KV** selected.
FP8 target KV saved 512.52 MiB of startup cache payload per card, but showed no
service speed benefit in this comparison. Both arms passed the retained answer
checks. FP8 target KV remains unselected and is not deployed.

This is a diagnostic result recovered from retained evidence. The original
serving operation failed final report publication; its CPU grading operation
also failed parent finalization. Independent review accepted the separate local
diagnostic continuation and verified resource release. Neither original
operation has been relabelled successful.

## Service measurements

Both arms used the official FP8 target weights, BF16 draft weights and BF16 draft
KV. The comparison changed target KV from BF16 to FP8. It retained low thinking,
DFlash width eight, FP32 recurrent state/head arithmetic and 2,048-token prefill
chunks. C1 means one concurrent request; C2 means two.

| Target KV | C1 median tok/s (range) | C1 median request latency | C2 aggregate median tok/s (range) | C2 median individual latency |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 119.98 (119.40–120.13) | 8.526 s | 172.45 (171.67–173.49) | 11.840 s |
| FP8 | 113.49 (113.35–113.65) | 9.018 s | 141.81 (141.38–148.10) | 14.441 s |

Rates include prefill and instrumented HTTP completion across **both GPUs**.
They are not per-card or isolated decode rates. Each cell summarizes three
timed groups in one BF16-then-FP8 order. All 12 timed response paths and their
verifier work differed between arms. The observed median changes, −5.41% at C1
and −17.77% at C2, therefore do not isolate kernel cost. TTFT, ITL and queueing
were not separately measured.

Target-cache startup payload fell by 537,411,584 bytes per card. This is not a
whole-device or campaign peak-memory saving. One awake observation confirmed
300 W limits on both cards; continuous power and thermal behavior were not
measured, so these data do not establish a thermal explanation.

## Answer checks and evidence limits

Native decoding matched all 104 retained responses to their full HTTP text.
Natural final-answer checks passed 37/37 per arm. Each arm reused 16 distinct
input-token sequences and nine expected answer strings; the 74 checks are not
74 unique tasks or a general accuracy benchmark. Thirty forced-output responses
were ungraded. Two cancellation cases supplied state evidence rather than
answer grades.

The serving report duplicated response data and exceeded its unchanged 8 MiB
publication limit after both request menus completed. The original workload
seal, final report and complete historical process joins remain absent. During
CPU grading, all decodes and grades completed before a missing dynamic-module
registration caused `KeyError: 'reviewed_cpu_host'` in parent finalization.
The owner authorized diagnostic-only grading; a subsequent local continuation
checked the retained outputs without rerunning native work. Astra reviewed the
metrics, continuation and independent CPU/GPU release.

Prospective source fixes and their focused checks are retained separately.
They do not repair either historical operation or qualify FP8 KV for selection.
Endurance testing remains **NOT RUN** under its existing deferral.

The [provenance record](../provenance/FP8-TARGET-KV-DIAGNOSTIC-20261026.json)
contains precise measurements and hashes for the private retained evidence.
Those local evidence paths are not included in this repository.
