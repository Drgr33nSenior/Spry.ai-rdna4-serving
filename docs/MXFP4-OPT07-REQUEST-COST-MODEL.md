# OPT-07 request cost model and control observation plan

Status: **CPU signed off, with parent corrections applied. Serving spend remains on hold.**

Opus 5.5 agreed to the original CPU stage. This revision closes its design and
input-validation findings. Opus reviewed v1; the parent implemented and tested
these corrections in a fresh v2 root. The sealed v1 root and review are unchanged.

No owner packet, candidate attachment, native operation or GPU run was created.
All 21 leads and all three cumulative component members remain retained.

## Spending decision

The shared diagnostic must be justified by the other leads: host scheduling,
collectives, speculative phases, source/branch inventory and the rank-0
timestamp question. The bundle cost model is expected to remain
**INCONCLUSIVE_HOLD** and will report coverage only.

Attention calibration covers single prefix lengths. Generation changes those
lengths, so an exact match cannot cover a whole request. This revision adds no
equivalence rule, interpolation or new calibration. An inconclusive model cannot
justify spending on another serving run.

The arithmetic remains a conditional sensitivity calculation. It assumes that
the component gap transfers unchanged and is fully exposed once per eligible
target-verify round. It is neither a prediction nor an upper bound. The
unresolved F1 t16 median gap remains an explicitly optimistic input; the output
identifies credits whose gap does not exceed the measured sample spread.

The fixed six-case score, 1.03 threshold, separate card scenarios, C2 mean-then-
median aggregation and negative gaps are unchanged. Even a positive synthetic
scenario grants no execution authority and proves no serving gain.

## Input correction

Each credited row now names its kind, sequence, members, row count and component
routes. Only target-verify rows are accepted. Copies of a shared C2 round must
agree, and every declared member must contain that round.

The row count must equal eight times the active membership. Known attention
labels must match the long/short case and geometry; F1 t8/t16 must match the row
count. Staggered target admission and singleton tails are representable, but a
request cannot disappear and rejoin within its observed interval. A future
adapter must still authenticate the real admission and finalization endpoints.

Unknown shapes remain unknown. The model collects all missing fields across the
observed input and reports coverage. It distinguishes calibrated attention,
admitted F1, proven stock fallback and unknown rows. Coverage counts request-
round pairs: a shared C2 batch appears once for each affected request. No live
evidence adapter exists; curated descriptors are not treated as authenticated
runtime facts.

The revised contract SHA-256 is
`dcac968bbfade6d6ce2e7cf4f1d667893f6aaa7aef686718825d4ea6b88bee98`.
This changes descriptor validation and disclosure, not numerical tolerance,
timing acceptance or execution authority.

## Trace interpretation

The parser still reproduces the eight historical operation-183 decode windows.
It keeps per-rank sums and interval unions separate and uses decimal timestamp
arithmetic. Chrome JSON ts/dur values are microseconds regardless of the display
unit. See the [Perfetto format documentation](https://perfetto.dev/docs/getting-started/other-formats)
and [maintainer clarification](https://github.com/google/perfetto/issues/983).

Each window includes 74 kernels outside the two known graph correlations.
They remain in the census. Tied start times use stable file order for the
descriptive successor count; that count is not dependency evidence.

Opus independently confirmed the rank-0 overlaps and the lack of those overlaps
in the sampled rank-1 windows. Timestamp artifacts are a plausible explanation
for the rank-0 norm/quantizer ordering anomalies, not an established diagnosis.
The inferred quantizer/GEMM dependency has not been bound to graph edges.

The diagnostic now requires per-symbol duration comparisons across ranks.
Start-order adjacency must not be used for causal attribution on a rank with
reported-stream overlap. Historical rank-0 duration sums must not be used as
execution costs. Neither rank's census establishes request savings or GPU
utilization.

## Diagnostic specification

The design keeps two sequential starts of the same control at the configured
235 W cap on both cards. It makes no power changes. Selected C4, stock cap16
verification and stock norm/quantization remain in place. Cap3 is an explicit
comparison-only cache setting.

Start 1 runs the unchanged 30-request menu with bounded CPU inventory and no GPU
tracing. Its observer overhead is unmeasured. Start 2 repeats the menu and adds
seven fixed windows on both ranks: one middle prefill and decode for all six
cases. Profiled latency is not scored.

The byte measures are now explicit and bound to the retained profiler source:

- Prefill trace export: gzip size strictly below 768 KiB per file.
- Decode trace export: gzip size strictly below 320 KiB per file.
- Expanded JSON: at most 32 MiB per file.
- Compressed traces: at most 3.5 MiB per rank and 7 MiB in aggregate.
- Coverage report: at most 64 KiB per rank.

Two prefill and twelve decode file limits sum to 5.25 MiB. The outer 160 MiB
export partition remains a ceiling; unused space cannot enlarge the inherited
per-file limits or authorize extra captures. Raw temporary staging must be
bounded separately. Producer enforcement, bounded decompression and failed
closeout still require implementation and review.

The original OPT-01 candidate rules still apply. The source must identify the
selected recurrent branch and record both-rank eligibility. This observation
does not prove F1 attachment, engagement or representative numerical equality.
All prior graph, TP2, C2, cancellation, recurrent-ordering and quality gates
remain.

## Validation and next action

The revised focused suite passes **36 tests**. Five explicit regression probes
show that v1 accepted malformed descriptors as plausible and v2 refuses each.
The retained-trace result remains INCONCLUSIVE_HOLD.

The unchanged public preflight's 43 passing checks are reused from v1; no native
boundary was exercised. No framework import, SSH, installation, Git mutation or
GPU operation ran. The report, source diff, test logs, contract revisions and
input seals are retained in the fresh evidence root.

Next is CPU implementation of the shared observers and authenticated joins,
justified by the other leads. Before any GPU decision, that implementation must
prove restoration, byte limits, supported profiler behavior and finite owner
budgets, then receive its implementation review.

Private evidence: `opt07-request-cost-model-v2` in the existing optimization
continuation directory. The public provenance record binds its seal and the
original Opus review.
