# Recurrent gate reuse: advancement rejected

**Partial component execution; incomplete and unselected.** The candidate stores the already computed FP32 exponential gate for the paired commit fold. It adds no buffer or launch and does not change the selected runtime. Independent review authenticated the source, partial results and resource release.

The run stopped after the first card's C1 suite exceeded its declared allocation envelope: 152,307,200 bytes allocated against an upper bound of 130,790,328 bytes. It remained below the absolute 268,435,456-byte limit. Reserved memory peaked at 176,160,768 bytes. This is a failed envelope check, not an out-of-memory event.

Before that check, the first suite recorded 48 exact cases, six graph captures, 54 replay checks, 32 timings and two untimed references. C2, the second card, long continuation and singleton-tail coverage did not run. These partial results do not make the complete component reader pass.

| Accepted prefix | Reference pair median, ms | Candidate pair median, ms | Candidate/reference |
| --- | ---: | ---: | ---: |
| 4 | 0.276566 | 0.289935 | 1.048339 |
| 8 | 0.274690 | 0.281475 | 1.024701 |

Both observed pair medians missed the declared 3% and 2 µs benefit gate. Accepted-prefix four also had a verification median ratio of 1.098152, above the global 1.02 prerequisite. Timing ranges overlap broadly; only 3/8 and 4/8 paired comparisons favoured the candidate. These are warm eager wrapper event intervals; they do not establish captured serving kernel costs. This supports rejecting advancement under the declared gates, without claiming a universal kernel regression. The unrun C2 cases prevent a conclusion about the complete existential benefit predicate.

The protected owner and independent observer verified release, with six supervised process identities absent, no GPU holders or allocations, no cleanup errors and a closed ownership journal. Host peak was 729,501,696 bytes. The physical command lasted 32.40 seconds. No serving requests ran and no service-speed or task-quality result follows.

The [provenance record](../provenance/FP8-GATE-REUSE-COMPONENT-20261024.json) identifies the exact source and retained evidence. Rollback is non-adoption. Normalised-key reuse and deeper state algebra remain separate hypotheses; this result does not qualify or reject them. Existing FP32 state arithmetic and historical numerical findings remain unchanged.
