# Sprint 16: selected-profile cost refresh

Status: complete. All eight final-answer gates passed, all four full output-ID
and final-text pairs matched, and both cards completed all eight device timing
windows. Sprint 14 source `f90f4779c13c…` remains selected. Independent resource
release passed. No next GPU campaign is prepared or reserved.

Sprint 15's small component improvement did not produce a both-order serving
gain. This refresh reused the reviewed Sprint 9 control versus instrumented snapshot
packet, with the selected Sprint 14 GEMM source in both starts. This is a
diagnostic; it does not benchmark a new optimization.

The classifier, event hooks, eight device windows per rank, host schedule,
four sampled tasks, task oracles and request bodies remained unchanged.
Thinking stayed at low effort with the frozen publisher sampling recipe.
Source bindings and exact fixtures required the sealed selected source.
Kernel bodies, other runtime helpers, caches, flags, pools and owner limits
remained unchanged. The parent reviewed this source substitution; the existing
independent classifier review was reused. No new Astra review was needed.

The packet retained the 5,400-second owner, 1,800-second worker and 1,740-second client limits.
The packet has 64 archive members, 16,721,920 archive bytes, 33,539,309 combined
transport bytes, 255 arguments per arm and 65,411/65,391-byte bindings. These
fit the existing limits without expansion.

Preparation passed 46 startup, 69 packet and 15 reader checks, plus three isolated
source-pin boundary checks. The unchanged 39 public classifier checks were
reused, not rerun as new work. Initial assertions refused fixture layout,
missing frozen provenance and old GEMM source expectations; a later fixture
indentation check also refused. Original files and failed receipts remain in
evidence. Exact fixtures were corrected before launch; runtime, requests,
grading and acceptance gates were not relaxed. All failures preceded GPU work.

Two starts completed eight natural-stop requests in a 383.53-second owner
window. Source, runtime, worker, weights, cache, wire and release evidence
authenticated. This classification does not benchmark a new optimization.

## Completed long-request windows

These medians use four completed samples per rank: two windows from each long
request. Units are milliseconds per sampled verification cycle. Projection hooks
time linear-layer `forward` methods, including quantization, launches, waits and
any enclosed collectives; these are not isolated GEMM times. Nested groups
overlap. Do not add groups or ranks into request wall time.

| Interval or group | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Complete cycle | 55.824 | 55.910 |
| Target graph | 46.499 | 46.500 |
| All projection methods | 29.217 | 29.213 |
| MLP family | 20.057 | 20.057 |
| GDN family | 10.868 | 10.852 |
| Full-attention family | 4.376 | 4.361 |
| All-reduce methods | 3.480 | 3.512 |
| Output head | 2.051 | 2.051 |
| Final gather | 0.157 | 0.158 |
| Draft graph | 7.200 | 7.366 |
| Acceptance | 0.340 | 0.346 |

Projection subgroup medians:

| Projection | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| MLP gate/up | 13.437 | 13.438 |
| MLP down | 5.642 | 5.643 |
| GDN `in_proj_qkvz` | 4.574 | 4.572 |
| GDN `out_proj` | 2.265 | 2.257 |
| Attention `qkv_proj` | 1.351 | 1.355 |
| GDN `in_proj_ba` | 1.196 | 1.194 |
| Attention `o_proj` | 0.748 | 0.745 |

Gate/up remains largest, but Sprint 15's replacement did not improve serving
repeatably. Down and GDN output already use selected M8 overrides. The next
sizeable projection without a selected M8 override is GDN input qkvz. These
windows do not establish a whole-request compute or memory bottleneck.

## Coverage and observer cost

Each rank completed the same selected cycles. All selected indices were reached;
earlier cycles, gaps and the listed tails remain outside device attribution.

| Task | Device cycles | Prefix-token range | Later uncovered cycles |
| --- | --- | --- | ---: |
| Running totals | 22, 49 | 332–489 | 13 |
| Unique last | 54, 146 | 559–1032 | 42 |
| Audit threshold | 98, 277 | 8829–10253 | 64 |
| Held-out audit | 106, 300 | 8831–10347 | 150 |

The separate host schedule retained 32 points per request and rank. It does not
fill device-timing gaps. Full outputs, including reasoning, matched for this
packet; general reasoning repeatability remains unverified.

Instrumented client elapsed time increased by 7.30%, 7.59%, 6.53% and 6.72% for
the four tasks respectively. Output counts and verification work matched.
Retain this observer cost when interpreting graph intervals. One fixed
control-first order cannot isolate causal overhead or provide a clean service
speed benchmark.

## Next bounded lead

Propose one model-free comparison at GDN input shape `(M,N,K)=(8,8192,5120)`:
current common M tile64/stage3 versus native unrolled M tile16/stage1. Preserve
every other selected f90 route. Sprint 11's common-kernel stage1-only experiment
at this shape was mixed; this proposal changes the kernel and M tile too.

The shape is derived from authenticated model source, configuration and TP2:
key width `16×128=2048`, value width `48×128=6144`, and merged output partitions
`[2048,2048,6144,6144]`, giving local output width 8192 and input width 5120.
This is a source/configuration derivation, not a live qkvz weight-shape witness.
The selected dispatcher AST, exercised with CPU stubs and no tuned entry,
confirms common tile64/stage3 at this shape. A future component must bind fresh
live dispatch and configuration absence, pass finite/exact and graph checks on
both ranks, then show a timing gain before any serving comparison. This sprint
implements and launches no such candidate.

Full-state and sampling-distribution equivalence remain unverified. Preserve
the unresolved numerical findings, owner session-use disposition and Sprint 11
failed work gates. Reliability remains deferred until tuning is exhausted;
this result does not qualify production use or change installed defaults.

See the [evidence index](../provenance/SPRINT16-CURRENT-PROFILE-20261022.json).
