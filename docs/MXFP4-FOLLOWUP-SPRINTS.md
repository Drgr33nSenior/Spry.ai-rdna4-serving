# MXFP4 follow-up sprints

The [effective-bandwidth follow-up](MXFP4-EFFECTIVE-BANDWIDTH-LEADS.md) adds a representative scale-conditioned entropy census, guarded integer execution, certified output-head screening and a separate graphics-compression feasibility branch. These are pending leads, reconciled with G3/L09/D1 and the completed screens below. No prior rejection, numerical gate or GPU allowance is changed.

Opus signed off the CPU screens as a bounded no-survivor. None of the seven representative compiler candidates passes the frozen resource contract. This series follows the Exa search into GEMM layouts, lossless representation and arithmetic. It preserves the original 21-lead register and the separate OPT-07 attention packet.

| Sprint | Work | Status |
| --- | --- | --- |
| MX-00 | Authenticate baseline, launches and prior failures; freeze contracts. | Five baseline artifacts reproduced before every candidate compile |
| MX-01 | Packed-weight layout that removes runtime conversion. | Six variants reduce shared memory, but exceed the register gate |
| MX-02 | Exact compressed-scale decoding inside GEMM. | Rejected: SGPR and shared-memory growth |
| MX-03 | Bit-plane and entropy-format cost screen. | LZ4 tested; Snappy bounded by format/data; no GPU decoder selected |
| MX-04 | Exact operand lookup and arithmetic proof; SWMMAC applicability. | Lookup/arithmetic deferred; SWMMAC needs its own numerical contract |
| MX-05 | Consolidated review and final candidate preparation. | CPU signoff complete; no passing candidate |
| MX-06 | MXFP4 versus FP8 memory and hardware assessment. | Owner retains MXFP4; FP8 comparison deferred |

A final candidate must have pinned sources, CPU tests and matching native compiler evidence before a GPU decision. A rejected or unresolved lead stays recorded. No performance gain, GPU readiness or adoption is implied by this plan.

The private execution plan (`PLAN.md`, private evidence) freezes the numerical, resource and review conditions. The objective remains balanced C1/C2 latency without regressions at the selected model, precision and context. Physical comparisons will use the owner's restored 235 W caps. No GPU allowance is granted or renewed.

The last weight-layout variant reduces dynamic shared memory from 4096 to 1024 bytes and barrier pairs from four to two, but uses 127 vector registers against the frozen 109 ceiling. It has no spills and the compiler's occupancy field is unchanged. These are compiler facts, not a speed result or a passing gate.

Opus found that the baseline and candidates have the same encoded hardware register allocation: the 109 ceiling is a compiler-count policy here. It remains unchanged. The candidate also waits on weight loads much sooner and has more load-count waits, so the reduced shared-memory work does not establish a net benefit. The variant stays parked.

On the same 93 sampled tiles, LZ4 makes weight payload larger and chooses raw fallback throughout. It compresses scales. Snappy has no native-library benchmark; a format/data bound rules out smaller independent blocks for these sampled weights. Prefix/rANS decoding remains deferred, with no rANS size result or GPU implementation.

MXFP4 saves **10.31 GiB (35.86%) of checkpoint tensor payload** relative to the retained FP8 checkpoint. An even TP2 split gives about **5.15 GiB per card**, a planning estimate rather than current resident-memory measurement. The owner prioritised that capacity and deferred the FP8 comparison. No small FP8 speed gain has been established.

The detailed CPU results (`RESULT.md`, private evidence) retain failed screens, all codec additions and the SWMMAC numerical questions. Current focused suites total 29 passing tests; the public preflight passed 43 checks. No GPU operations or model starts ran.

The signoff and amendments (`SIGNOFF.md`, private evidence) record the review findings without changing the sealed results. Opus independently reran all 29 focused tests. SWMMAC still needs a separate numerical decision; no tolerance is relaxed. Attention composition and the conditional norm/quantizer fusion remain the next routes.
