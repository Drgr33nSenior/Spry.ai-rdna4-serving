# Combined candidate: CPU source preparation, serving held

The three-member candidate remains verification prefetch, private cap32, and
F1 gated RMSNorm plus MXFP4 quantization. The component results are preserved.
The new CPU work prepares the attention adapter and closes the source-layout
question for one F1 recurrent branch under its declared input geometry. It does not install the
candidate or finish production qualification.

## What changed

The private BF16 verification path now supplies 32 directly to the authenticated
native singleton split helper and uses that cap for both partial buffers and the
attention call. The native backend remains cap16. Control and FP8-target routes
retain their native split selection. Metadata reuse and failure cleanup are
unchanged. The helper is authenticated during construction, outside the hot path.

F1's reviewed host route, model addition and scheme addition are byte-identical
to host-preparation-v2. The selected projection splitter allocates contiguous
BF16 z, the audited fold-path recurrent function allocates contiguous output,
and the model flattens both at the F1 boundary. The proof covers one of three
recurrent target-verify branches; it does not establish which branch the selected
pool flags choose. Future integration must identify and assert that branch, its
output allocation, and actual per-layer engagement. These are source constraints,
not observations of production tensors. Attachment and the complete producer/reader/collector
witness are explicitly still unimplemented. No owner packet has been built.

## Serving economics

The historical serving baseline below is operation212 at 300 W. It is retained
context, not a valid control for the proposed 235 W bundle. That run also lacked
the required production route proof.

| Case | Historical control | Prefetch-only arm |
| --- | ---: | ---: |
| Long C1 cold | 77.028 s | 77.555 s |
| Long C1 warm | 9.024 s | 8.763 s |
| Long C2 cold | 56.947 s | 57.066 s |
| Long C2 warm | 8.319 s | 8.089 s |
| Short C1 cold | 12.888 s | 12.903 s |
| Short C2 cold | 13.984 s | 14.000 s |

The score gate is 3% gain in the balanced speed ratio, equivalent to about 2.91%
uniform latency reduction. If only the two long warm cases improve and the other
four stay unchanged, each warm case needs about 8.49% latency reduction to reach
that gate. The historical warm reductions were about 2.8–2.9%.

Operation214's combined attention saving is 1.13 times its prefetch-only saving
for the long C1 geometry and 1.10 times for C2. Operation215's F1 inclusive median
gap averages 0.0943 ms for T8 and 0.0924 ms for T16 per 48 synthetic layers. The
historical sampled decode kernel sum is 61.104 ms per rank/window; it is neither
whole-request time nor a critical-path attribution.

| Conditional sensitivity case | Balanced score gain |
| --- | ---: |
| Scale only historical warm gains, then generously apply F1 to all request time | 1.23% |
| Also credit cold long cases with the same absolute saving as warm cases | 1.37% |

These numbers are neither forecasts nor upper bounds. They mix historical power
settings and measurement boundaries, assume transferable ratios, omit the small
unresolved short-attention losses, and do not establish interactions. They show
why this evidence does not yet support a 3% serving spend; they do not prove the
bundle cannot help. ECONOMICS.json preserves every assumption and input.

## Verification and disposition

Fifteen source/adapter tests and five arithmetic/evidence tests pass on isolated
Mac Python 3.11.6. The 43-check public preflight passes. Eight input seals verify.
The first adapter test run exposed a missing out_cache_loc field in the fixture;
the full failed log is retained. No runtime repair was needed for that failure.
During inspection, an integer cap field was removed from the buffer dictionary
because the inherited snapshot expects only tensors. Shape checks now enforce
the same policy without changing retention's tensor-only contract; the prior
source and passing final regression test are retained.

No native framework import, SSH, GPU operation, model start, Git mutation,
selection change or power mutation occurred. Local tests establish source and
metadata behavior only. Production attachment, representative exactness, real
compiler identities, graph lifetime, both-rank readiness and lifecycle remain
physical or integration gates listed in CONTRACT.md.

Disposition: **HOLD_FOR_EVIDENCE**, with all three candidates retained. Opus 5.5
agreed to CPU signoff. The branch-selection wording above resolves its condition.
Future both-rank witnesses must also bind the split helper identity and permit
cap32 only in the private BF16 candidate path. No GPU decision or approval is requested
from this preparation. The next useful measurement should share the already
planned diagnostic boundary and explain request-level cost and route coverage.
