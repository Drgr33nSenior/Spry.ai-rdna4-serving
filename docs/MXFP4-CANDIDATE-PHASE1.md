# MXFP4 candidate: Phase 1 benchmark and follow-up levers

The private MXFP4 candidate is faster than the selected FP8 profile in every serving
comparison below. It passed every graded answer and every release gate. One sampled
start in twelve hit a GPU memory fault that remains unexplained. The owner chose the
candidate as the private serving direction. The public FP8 selection is unchanged,
and nothing was deployed.

Evidence: [`MXFP4-CANDIDATE-PHASE1-20261031.json`](../provenance/MXFP4-CANDIDATE-PHASE1-20261031.json).
Operations 149–165 ran on two Radeon AI PRO R9700 cards (TP2, 300 W caps).

## Candidate

Qwen3.8-27B with Quark AWQ MXFP4 layer weights, a BF16 DFlash drafter at width 8,
and BF16 target and draft KV. It is served on SGLang with at most two concurrent
requests and CUDA graphs for batch sizes 1 and 2. Seven overlays sit on the MXFP4
base:

| Overlay | Effect |
| --- | --- |
| P3 fused norm | One exact kernel per RMSNorm chain |
| P1 prefill route | Exact FP4→FP8 weight expansion for 1,024–2,048-row prefill chunks |
| FP8 target head | FP8 output layer with FP32 logits on verify rows |
| FP8 drafter head | FP8 drafter output layer |
| GDN layout fusion | Removes redundant copies in the recurrent layers |
| Batch-size-1/2 verify adapter | Singleton-per-row split-KV verify attention |
| Packed QKVZ+BA | One quantization and one N=8240 GEMM per GDN layer at verify rows |

The FP8 heads are the only precision change; the owner accepted it on 2026-10-30.
The packed route is bitwise-identical to the separate path.

## Natural-stop menu (W1)

Four operations ran the same 19-request menu in two fresh starts each: FP8 before,
FP8 with the verify adapter, the candidate, and FP8 after. All 152 answers passed
strict grading. The two FP8 runs agree within 3%. A cell counts as useful only when
the change is at least 5% favourable against both FP8 runs.

| Family | Cell | FP8 before | FP8 after | Candidate | Candidate outcome |
| --- | --- | ---: | ---: | ---: | --- |
| ledger | C1 median latency | 2.73 s | 2.73 s | 2.35 s | −14%, useful |
| ledger | C2 tok/s | 129.7 | 126.4 | 172.9 | +33% to +37%, useful |
| SRE | C1 median latency | 32.11 s | 32.17 s | 29.22 s | −9%, useful |
| SRE | C2 tok/s | 207.8 | 206.3 | 308.7 | +49%, useful |
| config | C1 median latency | 13.62 s | 13.64 s | 13.21 s | −3%, not clear |
| config | C2 tok/s | 247.6 | 247.4 | 319.7 | +29%, useful |

The verify adapter on FP8 alone was useful in all three C2 cells (+9% to +17%) and
neutral at C1.

## Quality menu (W3)

FP8 ran the candidate's 106-request quality menu. That menu has 74 graded answers,
~16K and ~28K contexts, forced C1/C2 timing and a cancellation. Both FP8 and the
candidate passed 74/74.

| Cell | FP8 | Candidate | Change |
| --- | ---: | ---: | --- |
| C1 tok/s | 120.1 | 130.2 | +8% |
| C2 tok/s | 163.6 | 233.3 | +43% |
| C1 median latency | 8.52 s | 7.87 s | −8% |
| C2 median latency | 11.85 s | 8.76 s | −26% |

## BetterBench v0.6.1 (W2, sampled)

Each lane ran 53 calls per start at temperature 0.7, top-p 0.95 and top-k 20.
BetterBench has no correctness oracle, so these figures are descriptive only.
Figures are combined TP2 output tok/s, averaged over each lane's starts.

| Lane | C1 | C2 |
| --- | ---: | ---: |
| FP8 (operation 154) | 83.4 | 131.4 |
| FP8 + adapter (operation 155) | 80.8 | 138.5 |
| FP8 baseline, 2026-10-25 | 84.3 | 134.0 |
| Candidate (8 clean starts, operations 158–161) | 94.9 | 175.3 |

Prefill rates and TTFT at 2K, 8K and 16K were the same within 1% across the FP8
lanes.

## GPU fault

Operation 156 ran the candidate on BetterBench. Start 1 completed all 53 calls. Start
2 hit `HSA_STATUS_ERROR_MEMORY_FAULT` in the base MXFP4 GEMM (`_gemm_afp4wfp4_kernel`)
during batch-size-1 decode at about 1,050 tokens. One card stayed hung until the
owner reset it. That decode shape had already run thousands of times cleanly.

- **Isolation (operation 157).** Start 1 ran with the candidate-only overlays off,
  start 2 with the full candidate. Neither faulted, so the result is inconclusive.
- **Rate (operations 158–161).** Eight further sampled starts under operation 156's
  conditions all completed. That makes 1 fault in 12 sampled candidate starts.

The fault is treated as rare and remains unresolved.

## Levers tested after Phase 1

- **Prompt lookup.** A greedy counterfactual replay found the drafter already
  averages 6.96 of 8 tokens per verify cycle. A per-cycle oracle would save only
  0.4–2.3% of cycles, so it was not built.
- **Sampling temperature** (operations 162–163). Acceptance was 4.21 tokens per
  cycle at 0.7, 4.22 at 0.6 and 4.32 at 0.3, with throughput changing 2–3%. The gap
  from greedy acceptance comes from workload content, not temperature.
- **Packed QKVZ+BA.**
  - Component (operation 164): bitwise-exact on all 48 layers and both cards. The
    48-layer graph fell from about 5.4 ms to 4.45 ms, a saving of 18.1–19.8 µs per
    layer. That missed the preset 20 µs gate, likely because 129 workgroups run on
    32 compute units.
  - Serving (operation 165): 38/38 graded, with all 38 token paths equal to the
    candidate without the route. All six cells moved about 2% in the favourable
    direction.
  - The owner promoted it into the candidate.

## Not run or deferred

- **C4/C8 serving.** The two-request limit is built into the in-worker graph observer
  and validators. Batch-size 3–8 verify paths have not run on this stack.
- **Single-card serving.** The FP8 heads are built for a two-way vocabulary split,
  and full-stack memory on one card is marginal.
- **60-minute endurance.** It is pending and will be rebuilt on the promoted
  candidate.

## Limits

- All timings are combined TP2 HTTP service rates from bounded menus, not isolated
  kernel speedups.
- Cold-cache requests were used throughout, so multi-turn prefix reuse is not
  reflected.
- Only greedy W1 and W3 answers were graded; sampled BetterBench has no quality
  oracle.
