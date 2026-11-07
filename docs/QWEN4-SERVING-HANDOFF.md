# Serving handoff for Qwen4

Status, 7 November 2026: **complete, selected and parked**. The owner selected
operation234's mtp arm (physical kv4, start 2), adding S4 split-K down/out to the
three-member cap16 profile. Operation232's selected profile is the immediate
rollback. The programme stays parked until Qwen4 work is explicitly resumed.
No further GPU run is queued. Start with
[the current selection](../provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json),
[the finalization record](../provenance/MXFP4-FINALIZATION-20261105.json),
[current results](CURRENT-SERVING-RESULTS.md) and [AGENTS.md](../AGENTS.md).

## Current split-K scope

S4 applies only to bound down/out weights at M8/M16, N5120 and K8704 or K3072,
including rare eligible prefill calls. It uses fixed-order FP32 reduction, one
BF16 conversion, 1.25 MiB scratch per card and one additional reducer launch per
eligible GEMM. Keep the exact source/layout guards and startup numerical gates.
Requalify all of these for a new model.

The frozen verdict remains INCONCLUSIVE_PARK (+3.91% C1 verify-round rate,
below 5%). Owner selection does not turn that result into a benchmark pass.
Startup comparisons showed no value differences on the tested inputs; signed
zeros are treated as equal. Sampled work differed and BetterBench has no task
quality oracle. Runtime/compiler-proof hashes differ for an unresolved reason.
Actual runtime hashes and all limitations are retained in the selection record.
Pointer `2fda6127…` selects S4; `4e19fc56…` is preserved for rollback. Consumed
nonce `d87ac31a2ee705e3` grants no relaunch. The installed runtime is unchanged.

## Operation232 rollback and earlier evidence

The confirmed bundle adds three changes to the existing MXFP4 baseline:

1. Verification attention prefetch.
2. F1 gated-norm/MXFP4 quantizer fusion.
3. Removal of the gate/up GEMM wave hint for the admitted M8/M16 shapes.

Both arms use cap16. Cap32, v6-QB, Down-M16 and BW-03 are excluded. The package
retains the previously accepted baseline overlays; this bundle does not replace
the complete serving stack.

The rollback donor is operation232, nonce `2229e2386a020dfa`. Its logical `mtp`
arm is physical `bf16`, start 1. Control is physical `kv4`, start 2. The archive
pin is `171dac93…`, the candidate verification kernel is `ed26c67b…` and the
compiler-pin file is `3cf410c0…`. Full identities and private evidence roots
are in the finalization record. Its control remains available through the
preserved pointer `4e19fc56…`, which now supplies rollback for S4. The older
pointer `fe92916c…` is also preserved. The historical FP8 selection is unchanged.

The run independently passed the frozen comparator with a 3.810953% balanced
request-latency score gain. All 60 natural-stop quality grades passed; all 30
paired token streams, text and speculative counters matched, and control
reproduced op224. Independent release is authenticated at checkpoint 621.
Op226 and op227 also passed separately; do not pool their scores.

This is a bounded thinking-low, greedy, C1/C2, TP2 result at a configured 235 W
cap per card. Context is 131040, with explicit BF16 target/draft KV and
DFlash width 8. Both arms retain the cap3 recurrent-path and LLVM_PATH
deviations and startup numerical probes. F1 is not proven universally bitwise.
No broad quality, endurance, production or installed-runtime claim follows.

The balanced score uses request latency. Group wall supplies noise and
regression vetoes only. Operation232's descriptive aggregate group-wall
reduction is 2.1236%; combined TP2 output throughput is 57.1629 versus 58.4032
tokens/s. These are different measures.

## Profiling findings to carry forward

Operation229's matched baseline/candidate captures and reviewed CPU successor
provide the diagnostic evidence. The original stage2 remains INCOMPLETE; the
separate successor output is pinned `88b6e713…`. It joins 96 graph replays to
their inventories and bounds all 24 decode windows. Inventory matching does
not prove broader physical completeness.

- MXFP4 GEMMs account for 66–70% of short-context VERIFY time and about 45%
  at long context. Whole GPU-round shares are 52–57% and 38–39% respectively.
- Long-context attention also matters. RCCL timing includes peer wait.
- High GPU busy with low memory activity does not identify a bandwidth limit.
  Unpack/scale dependencies and clock or power pressure remain hypotheses;
  there are no qualified hardware counters proving them.
- Profiling inflates host rounds by 2.9–5.9% at long context and 4.9–9.9%
  at short context. Unprofiled dispatch-gap cost remains unknown.
- Four prefill clock chains, 22 of 40 resource windows, decoded gpu_metrics,
  cross-rank clock alignment and launch geometry remain unresolved. None
  should be silently treated as measured.

## Reuse and requalification

Reuse the owner/lease boundary, finite receipts, request/response hash joins,
thinking-aware grading, independent release, failure retention and matched
baseline/candidate comparison. Keep frozen evidence immutable. Reuse compiler
and native receipts only when their consumed inputs and environment still match.

When the next model is available, pin its exact revision, configuration,
tokenizer and runtime before porting changes. Requalify model shapes, layer
counts, scales, numerical contracts, padding/layout readers, capacity and
recurrent-state behaviour. Recheck all real import paths and startup fixtures;
plain stand-ins previously hid ABCMeta and resolved-library-name failures.

Port each member only where its guards apply. Establish device correctness
before timing, then use identical requests and generated work for comparison.
An arithmetic-order change needs its own numerical and serving decision.
No old approval or consumed nonce authorizes a future launch.

## Parked leads

| Lead | Disposition and condition for revisiting |
| --- | --- |
| Cap32 | Dropped by owner; preserve historical trajectory changes. |
| v6-QB | Parked without device correctness or speed results. Startup stopped at an exact allocator-counter guard; its cause and the later SSH stall remain unproven. Revisit the accounting contract before another device attempt. |
| BW-03 | Future research option; outside the confirmed bundle. |
| HRX update | Hold pending RCCL startup compatibility and an unprofiled estimate of dispatch-gap cost. |
| Larger P1 prefill | Hold; extending beyond 2048 rows needs a new numerical and memory contract. |
| Historical FP8 options | Reserve for a future FP8/model path. Native FP8/BF8 with MXFP4 storage retains its arithmetic and operand-equivalence obligations. |

Preserve the current rollback and all failed evidence. Do not restart a server,
update the runtime, rerun a packet or change power while resuming this handoff.
First identify the next bounded question and obtain any new launch authority.
