# Sprint 38: adaptive-speculation prerequisites

The first evidence gate is complete. Retained sampled proposal probabilities distinguish accepted drafts from first rejections in the four examined tasks. Keep fixed DFlash8 selected: no adaptive policy, graph change or new GPU run was made.

Sprint 38 reused the [Sprint 26 acceptance trace](SPRINT26-SAMPLER-STATE.md), which contains more evidence than was available to the earlier [adaptive audit](../provenance/ADAPTIVE-FEASIBILITY-20261021.json). Both recorded starts used the selected b101 GEMM, aligned ring-fold and BF16 target/draft caches. The requests enabled low thinking with temperature 1, top-p 0.95 and top-k 20 in the historical 16K context. This is a sampled profile; it does not provide confidence scores for the selected greedy requests or validate the later 32K SRE workload.

## Reused evidence and labels

Source, native, reader and analysis seals passed verification. Rank 0 retained exact sparse float32 draft probabilities, candidate paths and post-TP acceptance counts. The other rank and repeated start matched all compared identities. Use one start and rank as the primary cohort: **1,041 cycles across four distinct task requests**, with 3,123 corroborating records. The 4,164 total records are not independent examples.

Each width-eight verification contains an anchor and seven proposed tokens. The anchor and target-generated bonus are excluded from draft labels. Positions before the recorded acceptance count are accepted; the next position is the first rejection, if one exists. Later positions remain unlabeled because their proposed history was rejected. The commit span is derived as acceptance count plus one from the pinned native contract; it is not a separately captured commit-length tensor.

The 7,287 proposal rows contain 6,335 accepted labels, 229 first-rejection labels and 723 unlabeled rows. Every retained probability row has 16 positive entries, matching the selector configuration; the largest row-mass error is `2.06749e-7`. Four terminal cycles have shortened emitted spans. Their native acceptance labels are reported separately and excluded from any future utility interpretation.

## Descriptive signal

Here, `q` is the probability assigned to the proposed token within its chosen predecessor's 16-candidate distribution. It is not a calibrated probability of target acceptance. The analysis also calculated row maximum, top-two probability margin and entropy. No threshold was fitted.

| Task | Cycles | Accepted drafts | First rejections | Unlabeled tail | Selected-token q AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Running totals | 62 | 271 | 35 | 128 | 0.9264 |
| Unique last | 188 | 913 | 91 | 312 | 0.9290 |
| Threshold ledger | 341 | 2249 | 39 | 99 | 0.9889 |
| Held-out ledger | 450 | 2902 | 64 | 184 | 0.9877 |
| Total | 1041 | 6335 | 229 | 723 | 0.9768 |

AUC describes how well the feature ranks accepted versus first-rejected proposals in these traces. It is not classification accuracy, answer quality or a predicted speed gain. Excluding terminal cycles gives AUC 0.9769. Neighboring proposals and cycles are correlated; no independent-sample confidence interval or generalization claim follows. The task named “held-out ledger” comes from the existing corpus; this analysis did not hold it out for policy evaluation.

| Selected-token q | Accepted | First rejected | Acceptance fraction among labeled rows |
| --- | ---: | ---: | ---: |
| Below 0.25 | 30 | 47 | 39.0% |
| 0.25 to below 0.50 | 36 | 44 | 45.0% |
| 0.50 to below 0.75 | 83 | 47 | 63.8% |
| 0.75 to below 0.90 | 60 | 34 | 63.8% |
| 0.90 to below 0.99 | 221 | 45 | 83.1% |
| 0.99 through 1.00 | 5905 | 12 | 99.8% |

These bins were declared before extraction. They describe the observed accepted-prefix and first-rejection rows; the 723 later rows were not reclassified as failures. The original trace synchronized and copied tensors at every call, so its timing cannot support performance conclusions.

## Runtime constraints and decision

Six installed source files were read against current pins. The five selector, worker and graph identities from the earlier audit are unchanged. The active worker supports sampled selector acceptance on HIP. Its greedy route replaces probabilities with one-hot rows, so greedy pre-selection confidence remains unavailable in this dataset.

The native adaptive flag accepts EAGLE/EAGLE3 and rejects DFLASH. Both graph runners bind a captured request width. The selected split-attention adapter requires eight-row inputs, and the aligned ring-fold allocation predicate requires width eight. A width4/8 policy therefore needs separate graph, buffer, attention and state-commit integration. Reducing acceptance after a width-eight forward does not recover its compute cost.

The sampled feature is available after drafting. The current trace records acceptance after verification. A future policy must state whether it uses previous-cycle features before drafting or chooses a verification length after drafting; either design needs its own cost measurements and correctness checks. Truncating a width-eight trace cannot predict native width-four proposals or performance.

Retain adaptive speculation as a research candidate with a useful sampled signal. Before a policy experiment, validate a frozen feature rule on new task families, establish any required greedy pre-one-hot observation, and review width transitions, graph replay, accepted-prefix commits, convolution rollback, cache reuse and slot ownership. A passive sampled observer is the smaller telemetry follow-up if new data is needed; it must use bounded device buffers and defer host reads until request completion. No such observer or policy was installed here.

Twelve focused CPU tests passed. Independent review checked the extraction, labels, exact probability hashes, duplicate handling and reported AUC. The historical numerical findings remain unresolved. [Provenance](../provenance/SPRINT38-ADAPTIVE-SPECULATION-20261023.json) records the pins and counts. Sprint 39 C2 throughput/scheduling remains the next independent tuning item; reliability testing stays deferred until tuning is exhausted.
