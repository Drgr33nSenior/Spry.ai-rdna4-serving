# Bandwidth census: palette and Huffman layouts parked; decode GEMM remains the priority

The CPU census completed across all 64 transformer layers and all 496 U8 layer
projection tensors in the pinned checkpoint. It sampled 3,968 disjoint tiles,
covering 8,126,464 FP4 values: half for training and half held out. The reader
fetched 4,317,184 payload bytes, used no framework or GPU, and finished its
counting work in 0.375 seconds (0.776 seconds through the finite transport).

The checkpoint is amd/Qwen3.8-27B-Quark-AWQ-MXFP4 at revision
5233554c5fa56afda40150556b95573c2d7d29c0. Its historical model-file hash is bound
through the selected inventory; current size, mtime, header and before/after
file identity passed. The entire 19.8 GB file was not re-hashed.
The private selected pointer still matches B0. BASELINE.json distinguishes its
settings from operation217's profiling-twin Torch and cap-3 Mamba experiment.

## Storage decisions

The suggested sign bit plus three magnitudes and escape layout fails before
metadata. Training-only, per-tensor/per-scale dictionaries need escapes for
47.159% of held-out weights. Its ideal size is 4.66476 bits/weight, including
unchanged scales, against ordinary MXFP4's 4.25. None of the 496 tensors reaches
the necessary one-third escape bound. Even fitting each test tensor's
scale-conditioned dictionary on its own test data, which is an optimistic
oracle and not an admissible evaluation, fails that bound for every tensor.

Therefore do not implement this palette's GPU decoder. This conclusion does
not depend on the chosen record alignment or the 5% development threshold.

The marginal held-out code entropy is 3.78713 bits/code. Pooled H(C|S) is
3.76032; conditioning within individual tensors gives an empirical weighted
3.72476. These are finite-sample, zero-order entropy estimates, not full-model
compression guarantees or universal lower bounds for correlated representations.

A second, explicitly post-census hypothesis sizes canonical Huffman codes.
All 16 original codes remain representable through training pseudocounts;
unseen scales use a training-marginal codebook. Marginal and conditioned books
take 3.84532 and 3.79817 bits/code on held-out data, before scale bytes and
framing. The independent 8x256 tile format charges a 16-byte header, uint32
offset, 128-byte alignment, full dictionary and raw fallback. Neither variant
selects any tensor under the predeclared 5% training rule.

Even with all framing and dictionary costs removed, the held-out payload savings
are only 3.64% (marginal) and 4.75% (conditioned), including unchanged scales.
Both are below 5%. The empirical zero-order entropies correspond to ideal
code-only payload savings of about 5.01–6.48%, before framing and decoding cost.
Those estimates are sample-dependent and can be optimistic with sparse counts;
they are not a bound on formats exploiting correlations or compressing scales.

Park these Huffman layouts. This is a code-length/footprint screen, not a
production encoder, exact model round trip or measured memory-transaction
result. Larger tiles or a different addressing scheme remain separate
hypotheses; no decoder speed or bandwidth saving is claimed. Earlier LZ4,
Snappy, zlib and scale-only results remain unchanged.

## What the new traces support

Operation217 supplies one prefill chunk and two decode rounds per window on
each rank. The following ranges are the per-rank shares of recorded kernel
duration over the corresponding windows. Ranks are never added together.

| Recorded decode workload | MXFP4 GEMM | Verification attention stage 1 |
| --- | ---: | ---: |
| Long C1/C2, cold and warm | 42.35–43.03% | 32.44–34.51% |
| Short C1/C2 | 63.54–66.14% | 0.55–0.87% |

For the sampled decode windows, packed GEMM is the broadest remaining
target; verification attention is a long-context target. The small recurrent
fold, activation-quantization and norm observations do not justify another
fusion sweep. RCCL records remain visible but include waiting; their duration
does not identify link traffic or establish a communication optimization.

The single sampled prefill chunk has a different mix: MXFP4 GEMM is about 0.85%,
block-FP8 matmul about 22%, and MXFP4-to-FP8 expansion about 4.1%. Its large
unclassified group includes hipBLASLt `Cijk_*` GEMMs; those symbols also occur
in decode. The MXFP4 column therefore does not represent all GEMM work.
These samples do not establish whole-request prefill costs.

The parser checks every raw trace digest against the authenticated readout and
reproduces its exact kernel count and duration sum. Family membership uses
explicit kernel names; ambiguous _fwd_kernel records stay in other. PROFILE.json
retains those names and all unclassified records. No hardware-counter traffic,
physical bandwidth, utilization, critical-path saving or request-level speedup
is inferred. The strict cross-start NOT_COMPARABLE result and serving-bundle
HOLD remain unchanged.

## Remaining decisions

- Keep MXFP4 and its memory benefit. No FP8 replacement experiment is proposed.
- Keep the three cumulative members (verification prefetch, private cap32 and
  F1 fusion) for later combination. These data do not clear the serving spend.
- Keep the weight-layout v6 lead parked under its original failed compiler
  contract. Its unchanged hardware allocation is evidence for a possible new
  contract, not permission to relax the current gate or launch it.
- Integer arithmetic remains blocked on production floating-point equivalence;
  exact real arithmetic alone does not clear its earlier counterexamples.
- Certified output-head screening and graphics-image compression remain
  separate CPU research leads. Neither is implemented or made GPU-ready here.
- Preserve all 21 original leads and BW-01 through BW-04. BW-01 now has paired
  held-out evidence and a rejected palette plus two rejected Huffman layouts, rather than
  missing conditional statistics.

The next useful GEMM work is a bounded instruction/layout plan tied to exposed
load cost, with any allocation-contract change decided before measurement.
Further codecs need a stronger representation or access-cost mechanism than
these two screens provide. Do not spend a GPU operation on either failed layout.

## Validation and limitations

Nine census tests, four decision tests and three Huffman tests pass locally.
The existing public preflight passes 43 checks. The generated native payload
contains the exact source and input digests and retains a clean exit, reaped
child and absent process group. CPU limits were 30 seconds and 512 MiB, with a
60-second transport. No service, power setting, model file or kernel changed.
A first local reproduction check compared integer-keyed Python dictionaries with
JSON string keys and refused; exact serialized-byte comparisons then reproduced
all three outputs. The refusal is retained and no result was edited.

Each tensor contributes four train and four test tiles, stratified over row and
K halves. This spans layers and projections, but remains sparse sampling and
does not prove TP-loader shard identity or characterize spatial correlations.
The observed scale range 116–126 is weight-sample evidence only; no activation
scales were read. Negative-zero and exceptional-scale handling are tested
synthetically; these values were absent from this checkpoint sample.

Reproduce from this root with isolated Python 3.11: test_census.py,
test_decisions.py and test_huffman.py; then analyze.py, huffman_sizing.py and
profile_census.py. Their retained outputs are RESULT.json, HUFFMAN-SIZING.json
and PROFILE.json. Do not rerun native/run_file_audit.py to reproduce analysis:
it consumes the single-use native marker and is unnecessary.

## Review and evidence

Opus 5.5 approved this CPU study with no blockers, independently passing all
16 focused tests and reproducing the three analysis outputs byte for byte.
This public report includes its two wording additions: framing-free Huffman
costs and the prefill scope. The original sealed report is unchanged. The
provenance record binds the review, original evidence and a separate signoff.
No new GPU candidate or deployment-level improvement is claimed.
