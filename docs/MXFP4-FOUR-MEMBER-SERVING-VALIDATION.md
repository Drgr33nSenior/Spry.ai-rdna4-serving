# Four-member serving validation

Operation 224 completed approved packet a4777122b5277d13: two sequential
TP2 starts at configured 235 W per card, control then candidate. The result is
INCONCLUSIVE and parked. The raw balanced speedup score is 1.016601
(baseline 1.000000; +1.66%), below the frozen 3% gate. All six cases exceed the
2% generated-work limit, and long C2 triggers no-regression vetoes. No causal
serving gain or energy saving is established.

| Case | Baseline (s) | Four-member bundle (s) | Latency change |
| --- | ---: | ---: | ---: |
| Long C1 cold | 82.467 | 82.955 | +0.59% |
| Long C1 warm | 9.607 | 9.391 | -2.25% |
| Long C2 cold | 60.915 | 61.670 | +1.24% |
| Long C2 warm | 8.842 | 9.324 | +5.46% |
| Short C1 | 13.713 | 12.675 | -7.57% |
| Short C2 | 14.828 | 13.843 | -6.64% |

Each case has three scored samples per arm. C2 is the median of concurrent
pairs' mean request latency; positive change means slower. Long C1 produced
737 versus 721 tokens; long C2 pairs produced 804/538 versus 998/563; short
requests produced 2300 versus 2244. Final-segment token counts match within
each case, while reasoning lengths differ. These raw times cannot isolate
the kernel effect.

Both ranks passed representative F1/GEMM exactness probes across all 48/64
layers at 8 and 16 rows, and the startup capture witnesses passed. Candidate
verification used cap32; ordinary decode stayed cap16. All 60 natural-stop
answers passed grading and long-answer coverage. These are bounded serving
checks, not whole-model bitwise equivalence or production qualification.

The native launch took 1499.01 s; the finite invocation took 1582.07 s.
Export and readback succeeded. Source/HTTP replay, native seals and independent
release authenticate. Checkpoint 603 records no active owner or reservation
and zero remaining GPU allowance. No retry, adoption, selection change or
Hyperloom campaign followed.

The sealed readout (`REPORT.md`, private evidence) includes exact values, pins, token counts and the reproduced comparator.
The four members remain research candidates. Further serving spend stays HOLD.
The earlier attempts and CPU preparation below are historical.

## Output and acceptance divergence (CPU readout)

Operation224's arms sent byte-identical greedy requests. Each candidate stream first
differs inside reasoning, at output token 3–172 across the five unique prompts, and
every final-answer segment is token-identical. Eight retained starts in five
operations, including the prefetch-only candidate at cap16, reproduce all 30 control
token streams and speculative counters. The changed verify-round counts therefore
compare different reasoning texts; they do not measure an acceptance-mechanism
change. Only cap32 has device evidence of non-bitwise output at the serving
geometries (operation213: 0 of 16 layers equal in every case); F1 keeps an
unproven residual. The INCONCLUSIVE verdict is unchanged. A cap-only ablation is
prepared on CPU only; see the acceptance-divergence investigation (`README.md`, private evidence).

## Earlier attempts and preparation

Operation 223 is FAILED_PARKED after the control start refused evidence export
with encoded bundle overflow. The candidate never started. No validated baseline
or candidate timing samples or quality grades were recovered. Independent
release passed at checkpoint 601; no retry, adoption or selection change.

CPU reproduction shows the encoded upper bound is 67,117,086 bytes while the
inner bundle cap is 66,060,288 bytes. The check always refuses, including an
empty bundle or six bytes of member data. The outer stream cap is 68,157,440
bytes. This accounting defect should have been caught before launch. A bounded
CPU repair and complete export/capture/readback tests are next; no GPU allowance
remains. Opus 5.5 at xhigh has now fixed the derived HTTP line bound and added
a build-time consistency check. The fresh packet a4777122b5277d13 passes 70
local checks, 69 native CPU tests, public43, exact owner/caller reproduction and
six registrar mutations. Native and Mac export-budget results agree. The stream,
session, staging, kernel, compiler, numerical and serving gates are unchanged.
The inner receiver bound is derived from the encoded member allowance.

The new owner decision (`FOUR-MEMBER-SERVING-OWNER-DECISION-v4.md`, private evidence)
binds the proposed fresh invocation. No GPU approval, registration or launch
exists for it. Transport proof does not qualify real four-member receipt
semantics or live numerics. All earlier records below remain historical.

Operation 222, the one authorised retry, is FAILED_PARKED at 235 W. The repaired
F1 allocation predicate advanced during the first control graph warmup. The
next guard refused a compiler-record mismatch for gate/up M16 at wave hint1.
No qualification HTTP probes, benchmark requests or quality grades ran. The
candidate serving arm did not start. Baseline and candidate timing sample
counts are both zero; serving gain and production qualification are unmeasured.

The caller retained the failure in 134.85 s; native launch took 127.24 s. Collection,
failed-closeout observation and independent release passed. Checkpoint599 parks
the run with zero remaining allowance. The resource record reports zero OOM
events. No adoption, selection or power-setting change occurred.

The live compiler record was not retained. A subsequent CPU investigation
reproduced a defect in the expected pins: historical calibration warmed Triton's
cache, and later serving-path compiles reused the component's debug-path binary.
The source filename is absent from that cache key. Fresh code generation changes
five of ten bundle binaries, affecting gate/up and candidate attention. Their
machine code, non-debug ELF sections and resource metadata remain identical;
only debug sections differ. This identifies a CPU proof defect consistent with
the refusal, not the exact unretained binary from operation 222.

The repaired proof forces fresh code generation and validates source locations.
The real pinned Triton binder, using CPU tensors through the unchanged serving
GEMM body, reproduces all four fresh GEMM records with unchanged specialization
inputs. The device-free native proof retains ten binaries and 50 artifacts.
Six new regressions plus 35 bundle tests pass, as do the 43 public checks.
Diagnostic refusal logging also retains expected and observed records. The
runtime exact-hash and numerical gates are unchanged. These are CPU results;
live GPU specialization, graph behavior and qualification remain unmeasured.

Opus 5.5 accepts the CPU repair with no blockers to fresh owner preparation.
Packet `ac6c108f39cbae77` is now sealed and awaits exact owner approval. Its
preparation passes 60 local checks, 59 native CPU tests and public43. Actual
inert admission reproduces all package and caller pins, and the registrar's
CPU check refuses six mutations. A generated bounded-capture test preserves
the complete compiler refusal record; native scheduler logging remains untested.

Further GPU spend is on HOLD. The
sealed compiler repair (`README.md`, private evidence),
its accepted review and the derived packet are bound in the
fresh owner decision (`FOUR-MEMBER-SERVING-OWNER-DECISION-v3.md`, private evidence).
No third GPU attempt is authorised. CPU preparation is not physical qualification.

Operation 221 remains a separate failed, sealed attempt. It stopped earlier
because F1 interpreted symmetric-allocation eligibility as enabled allocation.
The corrected dependency combines eligibility with native enabled state and
authenticates the allocator call path. It still refuses active symmetric
allocation. This repair passed 108 local, 52 native and 43 public checks and a
focused Opus 5.5 review before operation222. Those receipts cover the executed
packet, not the later diagnostic-only edit.

The four research members remain verification prefetch, private verification
cap32, F1 gated norm/quantizer fusion and gate/up M8/M16 hint removal. Down-M16
and BW-03 are excluded. The selected checkpoint, precision, BF16 caches,
context capacity, thinking profile and TP2 topology remain fixed. Common cap3
and LLVM_PATH are explicit launch deviations. The original 3% balanced gate,
noise/equal-work controls, per-case no-regression rule and quality gates remain.
Component wins are not serving gains. All failed evidence is preserved.
