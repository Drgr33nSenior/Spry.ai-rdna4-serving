# BetterBench repeat on the selected cap16 candidate

Status, 7 November 2026: **operation 233 completed and released; independent result review accepts the result.** Both starts
completed all 53 requests. The configured caps read 235 W on both cards before and after the run. The invocation is
consumed, selection is unchanged and the optimisation programme remains parked.
Record: [MXFP4-SELECTED-CAP16-BETTERBENCH-20261106.json](../provenance/MXFP4-SELECTED-CAP16-BETTERBENCH-20261106.json).

The owner asked: "OK, let's run betterbench again to see what the difference is." The sealed intent allows one
invocation with at most two starts of the selected candidate, with no automatic retry. This is a separate
measurement. The optimisation programme stays parked, and the selection (operation 232's logical mtp arm,
pointer `4e19fc56…`) is unchanged.

## Planned measurement

One invocation runs two identical starts of the selected candidate: verification prefetch, F1 gated-norm/MXFP4
quantizer fusion and gate/up wave-hint removal, all at cap16. Each start sends the same 53 BetterBench v0.6.1 calls
as the W2 runs (operations 154–163). The client, corpus, profile and upstream files are byte-identical to W2's.

| Phase | Calls | Settings |
| --- | ---: | --- |
| Categories | 32 | 8 × (1 warm-up + 3 measured), cap 4096 |
| Prefill | 9 | nominal 2K/8K/16K × (1 warm-up + 2 measured), cap 16, temperature 0 |
| Concurrency | 12 | 6 at C1, 6 at C2 |

Sampling is temperature 0.7, top-p 0.95, top-k 20 and seed 1729. Every request uses thinking at low effort, and
every request has its own cache salt.

## What differs from the older W2 runs

| Item | W2 candidate (ops 158–161) | This repeat |
| --- | --- | --- |
| Runtime | 31 Oct full-stack candidate | selected stack + three members, cap16 |
| Worker path | instrumented (observers on) | production |
| Server context | 32,768 | 131,040 |
| Configured power cap | 300 W | 235 W |
| C2 global counter join | available | unavailable (no observers) |

W2 averaged 94.88 tok/s at C1 and 175.28 tok/s at C2 over eight starts (combined TP2 output). Per-start ranges were
92.2–98.5 and 158.0–188.3. These figures come from a different runtime and configuration. Any comparison is
descriptive only: it cannot attribute a difference to the three members, and it does not pool with operation 232's
greedy confirmation score.

## Why production mode

The selected bundle's startup witness runs only on the production worker handshake. A W2-style instrumented run
would therefore not exercise the selected bundle's startup contract, so this repeat runs in production mode. The W2 BetterBench reader joins each request to
per-request native observations, which exist only in instrumented mode. The prepared adaptation reports that join as
unavailable rather than synthesising it. All readiness, worker, startup-route and bundle-probe checks still run.

## Prepared so far

- An authenticated discovery record of the W2 roots. It recomputes the published W2 means from the raw readouts.
- A transform that re-derives the W2 BetterBench edits for the selected production stack. It changes the same nine
  files as W2, including one protected helper: the controller's BetterBench wire branch and its spec pin.
- A test-only build of that transform on operation 232's consumed package. It must never be registered.
- Fifteen focused tests on the generated code. They include the production reader on real operation-161 BetterBench
  data, which reproduces that run's speculation totals.

## Final preparation (CPU only)

The final private root `selected-cap16-betterbench-final-preparation-20261106-v1` (seal `2457576e…`) carries stage 1 forward:

- **Fresh package.** The unchanged canonical builders produced a fresh A/B reference (nonce `a5f29c1716b5347c`, never
  seen). The BetterBench transform was applied to it, giving archive `51fd2d82…` and manifest `86b3610b…`. Both physical
  starts are logical mtp; there is no control start.
- **Launcher.** It is derived from the reference with seven asserted substitutions. The caller reads the configured
  235 W cap on both cards before and after the launch.
- **Readout.** It runs the template's unchanged replay, then W2's unchanged arithmetic. Production request joins and the
  C2 global counter join are recorded as unavailable.
- **Coordinator.** It uses the operations 154–163 campaign pattern. One entry validates the inputs, constructs the grant
  and writes the registration, then makes one checkpoint write and runs one caller. One finish step closes the run. It
  records release as unverified unless sealed retained evidence shows independent release.
- **Protected helper.** Only the controller changes: its BetterBench branch and spec pin. The other eight protected
  helpers are unchanged.
- **CPU tests.** The suites pass:
  - stage-1 suite on the fresh build: 15/15;
  - final-packet suite: 6/6, including the end-to-end chain on real op232 production evidence and the real W2
    53-request tree;
  - registrar proof through the pre-ledger seam: 6/6;
  - public validation preflight: 43/43.
  The admission preflight passed locally; that is not admission.
- **Native proof.** Attempt 1 failed in the proof harness: its sandbox guard refused a process-spawn call that Python
  makes internally on Linux. Its failed evidence stays unchanged. After the parent checked the repair, corrected
  attempt 2 passed once on Python 3.14.4 in 8.45 seconds. The same executable allowlist now covers `os.posix_spawn`;
  the payload, sandbox and limits are unchanged. The proof reproduced the NumPy dependency receipt, compiled 80 files,
  completed all 53 client calls against an isolated mock server and exercised both starts through the production
  readout chain. Its sealed root is `selected-cap16-betterbench-native-proof-2-20261106-v1` (`be07813c…`). Mock timings
  and rebased retained worker evidence are not model, quality or GPU results.

## Run state

Independent review `58dc3855…` accepts stage 1, the final packet and native proof attempt 2, with no blockers.
Readiness `2add8918…` binds that exact review and the native proof. The unchanged finalizer ran once and generated
PLAN `e2e33fca…`; only the four permitted fields changed. Registration preparation (`f4881d24…`) and readiness
(`90de2740…`) are sealed. The selected pointer is unchanged.

Authorization `6167b490…` preserves the owner's reply `approved` against that proposal. Fresh remote admission
(`63201248…`) passed on boot `cf30344d…`; the registrar measured an admission age of 104.99 seconds. The entry
wrote checkpoint 622 (`533b04bc…`) for operation 233. The caller and the pinned finish ran once. Finish records
`SELECTED_CAP16_BETTERBENCH_COMPLETE_RELEASED_PARKED` at checkpoint 623 (`7f64b5fe…`), with all gates passing,
independent release and zero remaining allowance. Result review `60545a89…` accepts the raw measurements and
success-path release with no blockers. This measurement is closed. Do not invoke again, retry, or change selection.

The packet review disclosed untested sequential reuse of port 18088, a readout whose parts passed without a full
BetterBench-shaped operation-root test, and native fixtures that reused one start's worker evidence. The actual run
and result review close those gaps: both starts succeeded, the complete readout ran, and both real worker bindings
and startup witnesses matched the selected candidate. The client retains a 32768-token prompt/output envelope while
the server context is 131040. Production evidence still cannot prove the sampled selector branch ran on every
request. Release was authenticated from protected closeout and independent observer phases on the success path.
Keep the sealed packet unchanged.

## Accepted descriptive result

| Combined TP2 output | Start 1 | Start 2 | Mean |
| --- | ---: | ---: | ---: |
| C1, tokens/s | 99.51 | 98.50 | 99.01 |
| C2, tokens/s | 177.92 | 179.98 | 178.95 |

Relative to the authenticated W2 means, these means are 4.35% higher at C1 and 2.09% higher at C2. This is a
descriptive comparison under the different configurations listed above, not a causal speedup or a resolved effect
size. The two starts generated different sampled work.

The C1 group clock includes client journal and owner-ACK bookkeeping. This cost about 43 ms per request here,
versus 97–102 ms in W2's instrumented runs. Replacing the new run's measured overhead with W2's mean gives a
diagnostic C1 difference of about **+3.56%**, rather than +4.35%. This calculation does not replace the frozen
readout or isolate serving speed; the runtime, context, cap and generated work still differ. C2 overhead was not
decomposed because its requests overlap.

W2's observed ranges were 92.21–98.52 tokens/s at C1 and 158.03–188.27 at C2. Both new C2 starts lie within that
range; one new C1 start is just above W2's maximum. Two new starts against eight older starts do not resolve an
effect size.

Reported prefill rates were lower:

| Nominal target / actual prompt, approx. | W2 tokens/s | Selected tokens/s | Difference |
| --- | ---: | ---: | ---: |
| 2K / 1.5K | 3716.1 | 3553.7 | −4.37% |
| 8K / 6.0K | 3639.9 | 3501.6 | −3.80% |
| 16K / 11.8K | 3333.8 | 3293.2 | −1.22% |

These are client TTFT-based prompt-rate proxies. Cached-token fields are null, so uncached prefill is unproven.
The lower configured power cap is a possible contributor, not an established cause.

All 106 request protocols completed. Start 1 generated 37,155 tokens and start 2 generated 34,628. Each start's nine
prefill probes stopped at their deliberate 16-token cap. Two other responses in start 1 reached 4096 tokens: one
prose and one reasoning prompt; neither was marked as truncated during reasoning. Start 2 had no non-prefill length
caps. Completion and these labels are not answer-quality grades.

Cache salts are unique within each start and repeat across starts; fresh server processes provide the separation.
The retained fault scan covers amdgpu/KFD/DRM lines, not MCE or OOM-killer lines. Cgroup OOM, OOM-kill and maximum
events were zero on both starts.

The native run (`1a9f69e3…`, 533 members), readout (`3bb54401…`), finish (`49904a55…`), invocation (`0cf5b260…`),
campaign (`562effb7…`), parent supervision (`7ffccf85…`) and post-run checks (`4332e75c…`) are sealed. Parent checks
recomputed the concurrency rates from each group's output tokens and wall time and verified all 13 consumed roots.
No reader was rerun, and no further native or GPU work occurred after the one invocation.

Independent result review is `OPUS-OP233-SELECTED-CAP16-BETTERBENCH-RESULT-REVIEW-v1.md`, SHA-256
`60545a8998b6a08a998b5b89310097d6f1bcde5f7000a78b883c850dba751fe4`. It checked all 17 operation-233 roots and
eight W2 roots before and after its probes, recomputed the raw journals and authenticated release. The final PLAN
root holds preparation records; the actual campaign `REGISTRATION.json` is `546c03c2…` at checkpoint 622.

## Limits

- Sampled decoding has no GPU evidence on the selected bundle: operation 232's menu and startup probes are greedy.
  This run would be the first sampled use. A failure is retained, with no retry.
- BetterBench has no answer-quality oracle. Report normal stops versus length caps, token counts and speculation
  counters, but make no quality claim. Operation 232's 60/60 greedy grades remain separate evidence.
- Two starts show observed repeatability, not a resolved effect size.
