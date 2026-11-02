# Cleanup sign-off

The cleanup implementation and bounded validation are complete for the selected
profile and frozen corpus. Operation 207 ran packet `e2011102000000a3` once in
production mode. The exact tested source author contains 148 members and 61
Python modules. The runtime source matches commit `5cc29f4`; the later receipt
codec correction affects offline comparison only.

| Gate | Observed result |
| --- | --- |
| Original campaign | Both starts complete; caller exit 0 in 4,009.68 seconds |
| Regression | All 352 raw requests and output token vectors match B186 |
| Quality | All 352 natural grades and long-answer coverage pass |
| Release | Independent GPU and CPU release verified |
| Complete work comparison | 352 matches; all 186 groups retained |
| Throughput | Observed change −0.16% to +0.06%; no causal speedup established |
| Memory | About 1.63 GB lower median sampled VRAM use per card in both starts |
| Final public checks | 1,082 Python tests in 89 suites plus C++ sanitizers |

The corpus has eight distinct prompt-message sequences in 176 frozen cases per
start. It establishes this bounded result, not broad quality or endurance.
Memory values are external sysfs point samples every 10 seconds within HTTP
windows; they do not establish whole-device peaks. Fixed pools remain unchanged.

The retained operation-206 GPU startup records validate shared head objects,
transient head-byte equality and all 48 packed layers on both ranks through the
prospective corrected CPU reader. Operation 206 remains failed and partial.
No second-start instrumented witness set is claimed. Every earlier failure and
the separate numerical mismatch disposition remain retained.

The offline timing analyzer initially lacked its authenticated receipt contract.
Its correction passes 18 focused tests and complete absent/hostile-module
replays with byte-identical reports. No native work was repeated for that fix.

See [machine-readable sign-off](CLEANUP-SIGNOFF.json),
[native result](GPU-207-RESULT.json), [measurement](GPU-207-MEASUREMENT.html),
[readout verification](GPU-207-READOUT-VERIFICATION.json) and
[test procedure](TESTING.md). No cleanup step or native allowance remains.
The source PR stays in draft; no deployment or source promotion is included.
