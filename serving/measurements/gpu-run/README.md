# Production cleanup GPU measurement

Baseline A, operation 184, and cleanup B, operation 186, each completed two
starts and 352 responses. Both passed all 352 natural-answer checks and required
long-answer coverage. Complete owner/native/HTTP replay and independent GPU and
CPU release passed. See the [baseline result](BASELINE-A-RESULT.json) and
[cleanup result](CLEANUP-B-RESULT.json). The private `READINESS.json` points to
the sealed evidence. Earlier verification files describe preparation only.

All 352 rows match exactly for prompt IDs, output IDs, speculative work, cache
work, sampling parameters and finish reasons. All forty fixed input files and
both native execution contracts match. The pilot retained 186 groups; its
primary timing totals use the 110 measured timing groups.

| Arm | HTTP concurrency | A tokens/s | B tokens/s | Observed change |
| --- | --- | ---: | ---: | ---: |
| control | 1 | 157.667 | 157.456 | -0.134% |
| control | 2 | 254.866 | 254.662 | -0.080% |
| mtp | 1 | 157.527 | 157.251 | -0.175% |
| mtp | 2 | 254.845 | 254.506 | -0.133% |

These rates combine both GPUs and use summed output tokens divided by summed
HTTP group walls. They include prefill and completion. Warmup and non-timing
groups remain in the evidence but are excluded from these totals. HTTP
concurrency does not prove native batch size. Request latency and every group
observation are retained in the comparison JSON and HTML.

This A-then-B pilot shows no performance improvement. The small observed
differences do not establish a repeatable slowdown or speedup. Reverse-order
confirmation has not run. Follow the
[measurement protocol](../production-cleanup/README.md) for interpretation.

Both campaigns used production instrumentation mode on the `tst` host, stock W5
attention, the same request assets, all eleven original overlays, TP2, low
thinking, DFlash8 and 235 W caps. C4, profilers and Hyperloom launch delegation
were excluded. No serving selection, deployment or publication changed.

B's first offline replay omitted the configured spec hash when calling the
resource validator. The next replay exposed a second adapter error: the artifact
receiver requires a typed limit plan, while the HTTP reader requires its
dictionary projection. Both failures occurred before CPU dispatch and remain
sealed. The corrected local reader passes both complete B replay and legacy A
replay; A's resulting CPU input is byte-identical to its original input. Native
GPU sources, CPU grading sources, quality rules and limits remain unchanged.
B's single approved CPU grade then passed in 32.124 seconds. No native retry ran.

The local suite passed 862 Python tests, including 425 cleanup tests. Thirteen
comparison tests cover pooled arithmetic, work disagreement, response identity,
invalid counters, output separation and report labels. Syntax and symbol-table
checks passed for 152 modules. The unchanged target preflight passed 110 CPU
source checks and five CPU profile checks on CPython 3.14.7. Those preflight
checks used no GPU or native serving imports; the later physical and grading
results establish this fixed W5 run's observed behavior.

The baseline preserves the original protected set and capacity gates. Its
minimum retained evidence margin is 1.59, above its original 1.5 floor. The
cleanup proposal requires 2.0: current W5 fixtures reach 2.226 for evidence and
2.195 for source copies. These checks cover expected and retained-envelope
work, not all-cap or doubled menus.

Both one-use allowances are consumed. Operation 186 is parked with no active
caller, GPU owner or CPU owner. No further native execution is authorized.
Every new GPU operation needs a fresh exact approval. The current grading
profile requires the reviewed boot; a reboot requires new preparation and
review. Admission checks do not reserve the host.

The cleanup remains incomplete. Configuration mutations, all-cap and
doubled-menu validation, remaining shared modules and compatibility APIs,
runner hooks, memory recovery and private publication remain open. Future
configurations and refactors need their own verification.
