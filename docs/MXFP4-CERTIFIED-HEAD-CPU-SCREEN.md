# Certified output-head CPU screen

Status: prototype implemented and 25 focused tests pass. Opus 5.5 accepts
HOLD_NO_FIFTH_MEMBER as the CPU stop. This work has not produced a fifth GPU
candidate. On 5 November the owner retained BW-03 as a future research option
and dropped cap32. The current serving bundle has three members; the selected
deployment remains unchanged.

The selected target output head uses FP8 weights, block-128 scales and FP32
logits for verification rows M8/M16. The existing certified-head research
screens a BF16 head with an INT8 copy. Its nominal weight-traffic saving does
not transfer directly to our already-FP8 head; an INT8 arithmetic benefit also
remains unmeasured.

This pass implements a smaller INT4 screening representation, an encoder and
validated decoder, exact-rational residual intervals, both-rank winner
selection, a whole-call fallback model and a conditional cost gate. The INT4
representation is lossy screening data. It does not replace the original
FP8 weights or weaken the original numerical contract.

Residual bounds follow from the triangle inequality. Production certification
also needs a valid bound on the selected gfx1201 kernel's accumulation and
scale-rounding error. That bound is absent, so the prototype returns full-head
fallback. With an explicitly supplied mathematical bound, its conclusions are
labelled model-only and never production-qualified. Ties, overlapping intervals,
missing ranks and unsupported sampling or logit processing also fall back.

The tests cover every finite FP8 code, signed zero, NaN refusal, nibble order,
tails, corrupted metadata, residual containment, both-rank joins, FP32 rounding
counterexamples, source identity and single whole-call fallback. Synthetic
wide-margin examples exercise the model; they establish no representative
screening coverage or performance benefit.

The format counts headers, residual metadata and row padding. The screening
copy adds resident memory because the original FP8 and shared BF16 heads must
remain for fallback. No memory transactions, decode overhead, GPU lowering,
register use or request latency were measured. The cost gate remains
inconclusive without screen, resolution and whole-call fallback measurements.

Retain BW-03 as a tested research prototype while completing the three-member
cap16 bundle's profiling and finalization. Its next research gate is measuring
representative head/input margins and certificate coverage. A future candidate
also needs a valid gfx1201 rounding contract, qualified consumer eligibility and
a concrete total-cost case. The owner's interest does not establish a measured
gain or qualify BW-03 for the current packet. No GPU run, model start, adoption
or selection change occurred in this CPU screen.

BW-01's tested palette/Huffman layouts remain parked; stronger representations
stay open. BW-02 retains its production-equivalence obligations. BW-04 remains
a separate graphics-compression experiment. All 21 original references and
BW-01–04 remain in the [current register](MXFP4-REMAINING-GATES.md).

Primary research: [production rounding and certificates](https://sammausberg.com/research/papers/the_work_a_verifier_needs.pdf),
[shared-row reparameterization](https://arxiv.org/html/2609.31291) and
[exact MIPS versus matrix multiplication](https://arxiv.org/abs/1706.01449).
Their performance results are not gfx1201 evidence. This prototype makes no
novelty claim. Reproduction commands, source pins and limitations are retained
in the private evidence linked by the provenance record.

The independent review found no unsound winner acceptance within the declared
CPU model. It recommends measuring certificate coverage on real operands before
any GPU lowering; the current bound may be too loose to be useful. A future
port must require validated row preparation, reuse the exact production
activation-quantization buffers, and count graph dispatch and fallback costs.
The sealed CPU report remains unchanged; this is an unsealed review addendum.
