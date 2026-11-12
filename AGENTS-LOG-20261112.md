# RDNA4 serving continuation and Qwen4 handoff

Status, 9 November 2026: serving now targets SGLang upstream `main` at
`fcb3274646e43c16d70fd83806ebb3a5134d5e81` with the
[`sglang-main` patch series](serving/patches/sglang-main/README.md). Flags and
environment are in the workstation
[serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json)
and its [research notes](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/RESEARCH-20261108.md);
the profile is unqualified. The 2026-11 workstation reinstall wiped the op234
runtime (SGLang 0.5.21 `e00930c5`, Torch 2.14 / ROCm SDK 10.2, the private
package, owner sandbox and base image). Op234 and the op232 rollback cannot be
rebuilt from this repository as documented. The 7 November status below and
every statement that the installed runtime is unchanged are historical.

Status, 7 November 2026: **complete — operation234's three-member cap16 + S4 candidate selected; programme parked for Qwen4**.
On 7 November the owner requested private experimental adoption of operation234's
logical mtp arm (physical kv4, start 2). Operation232's selected three-member
cap16 arm is the immediate rollback. The frozen split-K verdict remains
INCONCLUSIVE_PARK: the +3.91% C1 verify-round rate missed the unchanged 5% bar.
Selection is an owner decision, not a reclassification of that result. Read
[the split-K selection](provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json) and
[the finalization record](provenance/MXFP4-FINALIZATION-20261105.json) before acting.
This file is a task handoff; it does not replace the workstation guardrails or
grant authority to rerun an old packet.

Separate measurement, 6 November: the owner asked for one bounded BetterBench
repeat on the selected candidate (two identical starts, one invocation, no
retry). The programme stays parked and selection is unchanged. The final CPU
packet is sealed. On 7 November, corrected native proof attempt 2 passed on
Python 3.14.4; attempt 1 remains a retained harness failure. Independent review
`58dc3855…` accepts the packet. The owner approved final PLAN `e2e33fca…` and
the exact controller exception; authorization `6167b490…` preserves the reply.
Fresh remote admission passed, and operation233 completed both 53-request starts.
The pinned finish recorded complete/released/parked at checkpoint 623
(`7f64b5fe…`); both pre/post configured cap readings were 235 W. Independent
result review `60545a89…` accepts the sealed readout (`3bb54401…`) and release
with no blockers. This closes the measurement. The raw C1/C2 differences versus
W2 are +4.35%/+2.09%, descriptive only. Matching C1 client bookkeeping reduces
its diagnostic difference to about +3.56%; reported prefill rates are 1.2–4.4%
lower. Runtime, instrumentation, configured cap and context differ from W2.
Registrar admission age was 104.99 s. The invocation is consumed; do not retry
or change selection. See
[the BetterBench repeat](docs/MXFP4-SELECTED-CAP16-BETTERBENCH.md).

Separate experiment, 7 November: operation234 completed both 53-request starts
and independently released at checkpoint 625. C1/C2 throughput changed
+8.37%/+12.56%, with different sampled work; these are not matched-token speedups.
Startup checks found no value differences across 31,457,280 elements, treating
signed zeros as equal, and 1,152 exact-oracle cells passed. BetterBench has no
answer-quality oracle. S4 adds 1.25 MiB scratch per card and a reducer launch per
eligible GEMM. Runtime compiler hashes differ from the native proof; the cause
is unresolved. Preserve these limits and all cap3/LLVM_PATH and startup gates.
The owner subsequently selected the candidate despite INCONCLUSIVE_PARK.
No retry, deployment or additional GPU authority follows.

## Resume safely
- Apply the [serving-validation skill](.agents/skills/aithreadripper-serving-validation/SKILL.md).
- For current serving work, read the
  [`sglang-main` series](serving/patches/sglang-main/README.md) and the
  workstation serving profile linked above. The records below are historical.
- Read [the split-K selection](provenance/MXFP4-SPLITK-S4-SELECTION-20261107.json) (historical selection and rollback),
  [the finalization record](provenance/MXFP4-FINALIZATION-20261105.json) (operation232 selection and parking),
  [the Qwen4 handoff](docs/QWEN4-SERVING-HANDOFF.md) and
  [current results](docs/CURRENT-SERVING-RESULTS.md). The
  [MXFP4 128K candidate](provenance/MXFP4-CANDIDATE-128K-20261031.json),
  [FP8 session selection](provenance/CURRENT-SESSION-SELECTION.json) and
  [remaining gates](docs/MXFP4-REMAINING-GATES.md) are historical records.
- Use [private evidence workflow](.agents/skills/aithreadripper-serving-validation/references/evidence-workflow.md).
  Keep sealed roots immutable and use bytecode-disabled imports.
- Preserve uncommitted user work. Do not stage, commit, merge or push without a request.

## Current objective
Status, 9 November 2026: this objective is historical. The runtime it describes
was wiped; see the status note at the top of this file.

Current private pointer `2fda6127…` selects operation234 mtp with S4 only on
M8/M16 down/out (including rare eligible prefill calls). Pointer `4e19fc56…`
is preserved byte-for-byte for operation232 rollback. The programme remains
parked, and the installed runtime and historical FP8 selection are unchanged.
The remaining 6 November closeout and running accounts below are historical.

Final closeout, 6 November: the owner explicitly selected operation232's
tested three-member cap16 mtp arm and parked the optimisation programme until
Qwen4 work is explicitly resumed. The private pointer is `4e19fc56…`; its
predecessor is preserved byte-for-byte as `fe92916c…`. Approval `b1cc0c8d…`,
promotion `7cdaadc8…` and parking record `2c369da8…` use the existing C4 workflow.
The public FP8 session-selection file and installed runtime are unchanged.

The winner adds verification prefetch, F1 norm/quantizer fusion and gate/up
wave-hint removal. Both arms use cap16. Keep the exact tested cap3/LLVM_PATH
deviations, mandatory startup numerical probes and scope limits. Independent
result review accepts the quality, work, score and release evidence. Control
is rollback; v6-QB remains parked, cap32 dropped and BW-03 future-only.
No work remains in this programme. Resume from the
[Qwen4 handoff](docs/QWEN4-SERVING-HANDOFF.md), preserving sealed evidence and
requalifying new model/runtime/shapes before any separately authorized launch.
All pending-decision and running entries below are historical.

Latest review, 6 November: independent result review `dc14aa7b…` accepts
operation232 with no blockers. Its independent raw-receipt score, re-grade of
all 60 answers, work/route checks and release joins agree with the frozen
readout. No GPU work remains. The owner must decide whether to adopt op232's
logical mtp arm (physical bf16, start 1), keep its control arm as rollback and
park the programme. The last GPU authorization excluded adoption and selection
changes, so the current pointer remains unchanged. Follow the existing C4
promotion/parking workflow; do not create another launch or approval framework.
The execution status FOUR_MEMBER_SERVING_MEASURED_PARKED and four_member_contract
are legacy labels, with three active members and no cap32. Read the concise
[Qwen4 handoff](docs/QWEN4-SERVING-HANDOFF.md) before resuming future work.

Latest result, 6 November: operation232 completed both unprofiled starts. The
frozen reader returns CONFIRMED_KEEP_REVERSED_ORDER / KEEP with a 3.810953%
case-balanced request-latency score gain. All 60 natural-stop quality grades
pass; all 30 paired token streams and speculative counters match, and the
control reproduces op224. No comparator veto exists. Both pre/post configured
cap readbacks were 235 W. Independent release is authenticated at checkpoint
621 (`d22b84a7…`), with the owner idle and allowance 0. No retry or new GPU work.

The execution (`10b53b84…`, 552 members), native run (`3f4d686c…`, 335),
invocation (`a32ff603…`, 5), readout (`d8802e8e…`, 8) and post-run root
(`4784fd97…`, 10) are sealed. The reader config changes only the two FUTURE
fields. The score is not aggregate throughput: measured group-wall reduction
is 2.1236%, with combined TP2 throughput 57.1629 to 58.4032 tokens/s. Do not pool
this run with op226/op227 or cite the TWO-ORDER report's mean. The reader's
fastest_fully_passing field is eligibility only; selection remains unchanged.
Next: independent result review, then final selection and programme parking
through the existing owner workflow. Earlier running entries are historical.

Latest execution, 6 November: the owner explicitly approved final PLAN
`7b661096…` and the exact helper clause. Authorization root `73205f0f…`
preserves the answer; TASK-AUTHORIZATION is `87366cd1…`. Fresh remote admission
passed on boot `cf30344d…` at checkpoint 619 / next index 232. The single entry
registered operation232 at checkpoint 620 (`970e1e6c…`); its finite caller is
running. Both pre-run configured cap readbacks were 235 W. No completed owner,
quality, performance or release result exists yet. Do not invoke again or change
canonical sources, selection or adoption. After completion, authenticate retained
evidence and release, seal the execution, fill only the two pending readout
fields, run the pinned reader and arrange an independent result review.
The earlier preparation and pending-approval entries below are historical.

Latest gate, 6 November: independent review `3fd67474…` accepts the fresh
sign-off packet with no blockers. Readiness `f0f857fa…` binds that exact review.
The reviewed finalizer generated PLAN `7b661096…`; only the three permitted
fields changed, and the ledger remains idle at checkpoint 619 / next index 232.
Final registration root `be6c2864…` and readiness root `ede8b23d…` are sealed.
No approval record, registration or GPU invocation exists. The owner authorized
the test and then replied "Approved" before the final PLAN was generated;
the reviewed entry still requires approval of the concrete PLAN and helper
clause. Follow NEXT.json in the final registration root after that binding.
Fresh remote admission must pass within 1800 seconds of entry. Confirm index
232 and absent CHECKPOINT-620 immediately before entry, and keep canonical
sources unchanged through readout. No pooling or selection change is allowed.
The preparation steps below are historical.

Latest preparation, 6 November: the owner wrote "OK, let's run another GPU
quality and benchmark before signing off." (intent root `2e490824…`). One fresh
unprofiled A/B of the selected baseline against the confirmed three-member cap16
bundle, candidate first as op227, is CPU-prepared as nonce `2229e2386a020dfa`.
Preparation root `final-three-member-signoff-preparation-20261106-v1`
(`289b13c1…`) replays op227's sealed route at checkpoint 619 / index 232. Only
nonce, path and digest bytes change in the packet; op227 limits are unchanged.
Two necessary deltas need review:
- the live closure `3721a6b7` (the additive op228 branch in `build_owner.py`);
- a one-literal reader repair (`a54246e6…`), without which the old reader would
  refuse after GPU spend.

Registration-proof root `final-three-member-signoff-registration-proof-20261106-v1`
(`ecda5848…`) ran the real registrar/owner chain through grant construction and
stopped before the checkpoint write. Read-only remote discovery found boot
`cf30344d…` unchanged; it is not admission. Nothing is registered or approved.
Next, in order:
1. an independent review (`REVIEW-HANDOFF.md` in the registration-proof root);
2. a readiness record, then `finalize_plan.py`;
3. the owner's exact answer on the final PLAN and the unchanged helper clause;
4. a fresh `--remote` admission within 1800 s of the single entry.

There is no pooling with op226/op227, and selection is unchanged. The entries
below are earlier states.

Latest closeout, 6 November: recovered-result/reconciliation review `df05cfd1…`
accepts the failed result and authenticated release. Its exact two-line
observation repair is sealed `072a3811…`; the writer is unchanged. Five tests,
the guarded check-only and old/fixed audit emulations pass. One fresh
unprivileged observation verified boot, journal absence and both exact scopes.
The reviewed writer appended checkpoint 619 (`26a83efa…`), leaving 618 and all
failed roots unchanged. The owner is idle, next index 232, GPU allowance 0.
Reconciliation root `65fca55d…`, observation `f73ad977…` and parent closure
`f4f5e9a…` are sealed. Measured owner, caller and itemized recovery charges were
added once. No GPU work or original caller/close was repeated.

**Park v6-QB.** TP1's exact allocator-accounting guard refused; TP0's result is
unknown. Correctness probes and timing were not reached. Why allocator counters
differed and why SSH stalled remain unproven. No retry, adoption or selection
change is authorized. The confirmed three-member cap16 results remain eligible.
Final profile selection and programme parking remain separate pending steps;
use their owner workflow before changing the selection pointer. For Qwen4,
carry the evidence, compiler/source identities, real-fixture regressions and
failure-retention rules forward, then requalify model shapes and numerical
contracts. Keep cap32 dropped and BW-03 future-only. Earlier states below are
historical and do not grant new run authority.

Latest recovery, 6 November: review `20f35016…` accepted frozen post-run
retention. The four original steps ran once under unchanged deadlines; no
caller, owner or GPU work was repeated. Recovery root `222aa738…` (314 members)
authenticates both owner evidence and independent release. The failed caller
and original checkpoint618 remain unchanged. Diagnostic parent `31a4446b…`
retains a complete TP1 memory witness: +335,872 allocated bytes versus the
+1,966,080-byte expectation, with42 failed layer deltas. TP0 has no witness.
This proves which exact guard refused, not why allocator counters differ.
Real-input qualification and timing were not reached. No retry is authorized.
The separate reconciliation preparation `0738eb5e…` (17 members) passes five
CPU tests with34 refused mutations. It has performed no SSH or checkpoint
write. Independent result/reconciliation review and then one fresh read-only
scope/journal observation are required before appending a release checkpoint.
Checkpoint618 still records UNVERIFIED; do not rerun its original close.
Earlier transport and preparation states below retain their chronology.

Latest failure, 6 November: operation231's physical-owner SSH exited255 after
69.5 seconds with a server-not-responding message. The caller failed, leaving
no OPERATOR-RESULT or local results tree. The pinned close ran once and recorded
FAILED / UNVERIFIED release at checkpoint618 (`ecaa548e…`). Its exit0 is not
a workload pass. Local partial-native (`1f68676d…`,45), campaign (`932150f1…`,14)
and invocation (`aa3b4609…`,19) roots are sealed and must remain unchanged.
The host is reachable again on the same boot, with no owner journal. A bounded
unprivileged status read found an unsealed owner record claiming stage-A
exit137, stageB not started, no cleanup errors and independent release. This
is not authenticated retained release, and no first scheduler error or memory
delta is established. Do not restart the caller/owner or run any GPU work.
Independently assess recovery of only the unchanged post-run retention protocol
under the original absolute deadlines, then authenticate release and reconcile
the unresolved checkpoint. No retry, adoption or selection change is authorized.
Earlier running and preparation entries below are historical.

Latest execution, 6 November: the owner supplied both exact clauses and
accepted PLAN `fccaf0c4…`. The verbatim answer is in authorization root
`1b747273…`. Fresh remote admission `f29cfb2a…` passed at checkpoint616/index231.
The reviewed entry registered operation231 at checkpoint617 (`97a969bb…`) and
its single caller is running. The allowance is consumed. Do not invoke again,
retry, adopt or change selection. After completion, authenticate retained output,
verify independent release and close through the pinned close script. On a
memory refusal, re-verify W2 and decoder `33e94b9c…`, then inspect the retained
witness diagnostically. A partial witness is not complete evidence. The earlier
preparation records below are historical.

Latest gate, 6 November: independent final delta review `601addd9…` accepts the
witness packet with no blockers. Readiness `473c0d98…` binds that exact report.
The unchanged finalizer generated PLAN `fccaf0c4…` in
`v6-qb-gpu-final-registration-93dc847c81eb3802-20261106-v1`
(`dae9e087…`, six members). Mechanical closure verified only the permitted
PLAN substitutions, all three byte-identical campaign scripts and both exact
clauses. The ledger remains idle at checkpoint 616 / next index 231; a fresh read-only host check
confirms boot `cf30344d…`. No GPU, model, registration or checkpoint write ran.
The owner must supply the new exact answer; no answer was written on their
behalf. If stage A refuses for any reason, including the unchanged memory
guard, the single invocation is consumed, stage B does not run and there is
no retry. Fresh remote admission within 1800 seconds remains mandatory.
The presentation and remaining commands are in
`v6-qb-gpu-owner-preparation-93dc847c81eb3802-20261106-v1`.
Earlier preparation states below retain their chronology.

Latest preparation, 6 November: in response to the owner's "OK, let's have claude
prepare up until you're ready to run on the GPU again.", CPU and native
preparation are complete. Nothing ran on the GPU, no model started, nothing was
registered and no checkpoint was written. Four fresh sealed roots exist:

- **Witness review, `op230-memory-witness-review-repair-20261106-v1` (`dc9b3411…`).**
  A focused review of the parent repair added a leading newline to every
  frame, because unterminated foreign output could hide one, and recorded
  `PYTORCH_HIP_ALLOC_CONF`. The serving source is `8ebb6cf4…` and 25 tests
  pass. The same sub-agent then integrated this source, so the final review
  must cover these fixes.
- **Packet, `v6-qb-gpu-composed-packet-witness-20261106-v1` (`47503930…`).**
  - Nonce `93dc847c81eb3802`, package `539388fd…`, boot `cf30344d…`.
  - The copied tools reproduce op227's frozen pins and op230's stage-A package
    exactly. After token normalization, only the serving source differs.
  - Only the controller `SPEC_SHA256` line changes among protected helpers.
    Limits are unchanged apart from +3,087 static bytes; the witness adds at
    most 512 KiB of stderr inside the 12 MiB cap.
  - 69 tests and public43 pass. R4 compile-proof reuse is sound.
- **Native proof, `v6-qb-witness-native-proof-20261106-v1` (`a40d3faf…`).** One
  device-free Python 3.14.4 run passed 28 tests (25 packaged-witness and
  3 held-loader). All 126 payload files compile, and the pinned `libc10`
  libraries contain all three allocator variable names.
- **Registration preparation, `v6-qb-gpu-registration-prep-93dc847c81eb3802-20261106-v1`
  (`15f8f4d7…`).** The prefilled PLAN observes checkpoint 616 and next index
  231. `finalize_plan.py` is provided. The CPU registrar proof has 1 positive
  case and 30 refused mutations, covering every consumed op230 pin. The
  admission decision logic passes 1 case and refuses 9 faults. A local-only
  check is not admission.

The next steps, in order: an independent final delta review
(`REVIEW-HANDOFF.md`), a readiness record, `finalize_plan.py`, the owner's
exact new decision quoting both clauses and the final PLAN hash, a fresh
`--remote` admission, then the single entry. Follow `NEXT.json` in the
registration root. **The memory guard is unchanged, so stage A may refuse
again;** the witness records the observed deltas only. The paragraphs below
are earlier states.

Latest CPU follow-up, 6 November: independent result review `9421be48` accepts
operation230's failure accounting and release. Only TP1 has a retained refusal;
TP0's check outcome is unknown. The +1,966,080 B value was expected, not measured.
The resource monitor reported complete; its final validator refused child137
because it required exit0. No capacity or numerical cause is established.
The approved CPU-only repair is in
`op230-memory-witness-cpu-repair-20261106-v1`: 18 tests pass through the exact
executed CapturePipes, including total-only and per-layer-only refusals. It adds
a bounded stderr witness before the unchanged guard, with worker joins and raw
snapshots. Allocator environment values do not establish initialized settings.
The candidate is unintegrated, independently unreviewed and not native-proven.
The canonical adapter, executed packet and protected helpers remain unchanged.
No GPU retry is authorized; the old measurements remain unavailable. Final
selection of the confirmed three-member cap16 bundle remains a separate step.

Latest result, 6 November: operation230 finished RELEASED_WORKLOAD_FAILED /
PARK_STAGE_A_REFUSED. Both rank processes loaded the target/draft and KV pools;
TP1 refused v6qb_serving._attach's exact steady-padding allocation
check before warm(), capture, real-input probes or component timing. Stage B
was forbidden. No correctness or speed verdict exists. Raw failed allocation
deltas were not published, so allocator-accounting, lifetime and other causes
remain unproven; do not infer a capacity or numerical failure.
The reviewed close script authenticated retained data and independent release
at checkpoint616 (c79a1a3d). Native seal d0ae0a7f has 358 members, campaign seal
732a4a1f has 311, and invocation seal c70c5749 has 7. The invocation is consumed.
No retry, adoption or selection change is authorized. The completed independent
review and CPU repair are recorded above; original op226/op227 results stand.
Stage A had no cap readback, and B never reached its power reads.
Earlier running and preparation entries below are historical.

Latest execution, 6 November: the owner supplied the exact helper exception
clause and accepted final PLAN 9f86beef. The answer is preserved verbatim in
sealed authorization root 7cadad05. Fresh remote admission d93041e0 passed at
checkpoint614/index230. The reviewed single entry registered operation230 at
checkpoint615 (3333590b) and its finite caller is running. The invocation is
consumed; do not retry, call the bare driver, change selection or claim success.
After the caller finishes, authenticate retained output, verify independent
release and close through the reviewed script, then obtain result review.
The older preparation and pending-approval paragraphs below are historical.

Latest gate, 6 November: independent packet review `9d6b76b8…` accepts the
composed packet with findings and requires no byte repairs. The parent filled
the missing plan/grant fields and readiness record, then exercised unchanged
validate/spend, actual driver authority, caller field reads and pure source
checks on synthetic fixtures. One positive case and 13 refused mutations pass,
with no real checkpoint write, target command or caller launch.
The final registration root is
`v6-qb-gpu-final-registration-abcf6687ab36a28e-20261106-v1`
(`c74fa026…`, 170 members). PLAN is `9f86beef…`; readiness is `85a6c19c…`.
The actual campaign contains only the three byte-identical reviewed scripts.
The test intent remains authorized, but the sealed registrar also requires an
owner answer quoting the exact protected-helper exception clause. That answer
and fresh remote admission within 1800 s of entry remain open. No GPU/model,
registration, adoption or selection action ran.

Disclose the accepted limits: stage A is untimed and has no 235 W cap readback;
stage B reads caps before card 0 and after card 1. Native proof does not cover
later SGLang environment changes, so a hash refusal after model load consumes
the invocation and forbids B. Stage A captures but does not replay the v6 graph,
sends zero benchmark requests and proves no natural-stop quality result.
Component timing may be cache-resident and is no serving-speed claim.

Latest preparation, 6 November: packet `abcf6687ab36a28e` is composed in
`v6-qb-gpu-composed-packet-20261106-v1` (`b7301234…`, 1793 members).
The single native R4 environment-projection proof passes in
`v6-qb-composed-native-proof-20261106-v1` (`92a9ddc8…`, 16 members).
It reproduces metadata.hash, specialization, artifacts and resource facts under
both child environment projections; in-process SGLang environment changes remain
covered by the runtime check. Forty-four composed CPU tests, 73 tests on an exact
repair copy and public43 pass. The parent verified all three new root seals.
Independent review must cover the composed owner, serving overlay, registration,
retention/readout path and the earlier bounded repair delta.

One CanonicalLease holds across stage A (one TP2 probes-only model start) and
stage B (zero-start component, cards 0 then 1). Stage-A child resource release
precedes B; owner close and independent release occur once. Limits are owner
3930 s (3630 s active), caller 4845 s, stage A 2490 s and stage B 600 s worker /
280 s per card. The exact protected-helper proposal `e0501ac3…` changes only
benchmark_physical_controller.py's canonical SPEC_SHA256 literal. Its required
exact clause is not covered by an invented owner quotation. The existing test
intent is recorded; final exception approval, PLAN/readiness binding and fresh
admission remain open. No GPU, model start, registration or selection change ran.

The older composition-pending states below retain their chronology.

Latest owner instruction, 6 November: "OK, I'll authorise a GPU correctness and
speed test run". The separate intent record preserves that reply verbatim at
`v6-qb-gpu-test-owner-intent-20261106-v1/OWNER-INTENT.json` (`c6362bf2…`).
It is interpreted as accepting the preceding narrow A6 recommendation for this
test only, with every other frozen predicate preserved. It does not change the
general resource policy, old rejections or selected profile. One finite GPU
invocation is authorized, with no retry; no GPU entry has run.

CPU preparation is sealed at `v6-qb-gpu-test-preparation-20261106-v1`
(`3981530e…`, 110 members). The packer, launcher, memory accounting, component
corpus and frozen readout pass 65 local tests. Independent CPU review
(`a4b71af9…`) accepts the stage with findings. The parent repaired the module
collision, retained-record checks and primary-error cleanup ordering in
`v6-qb-gpu-preparation-review-repair-20261106-v1`
(`5acdea52…`, 67 members). All 35 affected tests pass; ten input seals and
eight file pins remain unchanged. Kernel bytes and timing thresholds are unchanged.
The composed packet review must cover this bounded repair delta. Native proof
must still reproduce metadata.hash under both exact child environment projections.
No executable GPU packet exists yet. Real-input qualification needs one TP2
model start to observe both verification sizes at all 48 GDN layers on both
ranks; no retained activations close this obligation. The proposed route then
runs the zero-start component correctness and timing phase. Composition must
fit these phases into one invocation through the existing owner boundary,
with exact limits, retained failures and independent release. Final-byte native
CPU qualification, packet review and fresh admission remain open. Preparation
and the recorded intent do not permit skipping these gates.

The earlier A6/preparation states below are historical. The existing native CPU
allowance remains consumed, and the new intent does not permit repeating it.

Latest native result, 6 November: the owner approved the exact v6-QB CPU
proposal (reply `aproved`, preserved verbatim). The fresh preflight passed;
the single caller exited 0 in 8.625 seconds with five CPU compiles, matching
all five baseline gate pins. Both M8/M16 pairs pass the frozen reader's
resource, scalar arithmetic, scale/mask and byte-ownership predicates.
Execution seal is `1e3e928f…` (13 members); readout seal is `7edb7e70…`
(17 members). Both are immutable. The allowance is consumed; no retry.
This is CPU static evidence, not device correctness or a speed measurement.
A6 remains unadopted, GPU G1–G3 remain open, and selection is unchanged.
Independent result review accepts the facts with no blockers (`47d6b7e3…`).
M8/M16 are two specializations with one binary per arm. Future G1 preparation
must take dynamic LDS (baseline 4096 B, candidate 1024 B), four warps, grid 129,
constants, attributes and signature from the authenticated wrapper. The frozen
reader does not expose all of these launch inputs.
The next owner decision is whether to adopt A6 for this exact verify-only
QKVZ+BA scope: report used VGPRs without treating their count as a limit, while
retaining every other frozen descriptor/resource predicate. If adopted,
separately requested preparation must close the packer/launcher identity and
transient/cache-memory gates before any finite GPU correctness/timing decision.
Final selection and parking of the confirmed bundle can proceed independently.
Earlier preparation paragraphs below are historical and supply no new authority.

Latest CPU gate, 6 November: the rejected-trial follow-up and delta reviews
accept the narrow verify-only QKVZ+BA case. The parent mechanically closed the
scale/mask mapping condition in fresh root
`v6-qb-scale-map-reader-repair-20261106-v1` (`a9fdabb0…`, 39 members).
All 18 tests and the real retained-control CLI pass; nine input seals remain
unchanged. The old comparator accepts the review's wrong-broadcast case and
the repaired one refuses it. TARGET, RECIPE and PREPARATION are unchanged.
No further full review is required under the accepted mechanical conditions.

The proposed native CPU decision is sealed in
`v6-qb-native-cpu-preparation-20261106-v1` (`0f604cdf…`, 20 members).
Decision `544febc0…` and PLAN `f7855065…` bind the fresh uninvoked working root,
five CPU compiles, 180/225/240-second limits, 8 GiB address space and 32 MiB
per-file limit. Read-only connection/host/boot, owner-journal, nine bind roots
and 29 explicit compiler/loader source-pin checks pass. This is not a full
rootfs/binary inventory. The owner has not approved this invocation. Do not
run it without the exact reply, fresh preflight and source checks. A6 is not
adopted; the old 109 gate and all GPU/device/timing gates remain unchanged.
Final selection and parking of the confirmed bundle can proceed independently.
The four-group shortlist below is historical; current dispositions are v6
dominant shapes PARK, only v6-QB eligible for this CPU proof, HRX/prefill HOLD,
and historical FP8 projection changes reserved for Qwen4/FP8.

Latest gate, 6 November: the op229 result/diagnostic review accepts the CPU
corrections with no blockers. The one permitted successor run matched output
`88b6e713…` exactly and preserved all nine input seals. Its sealed root is
`op229-profile-reviewed-cpu-successor-20261106-v1` (`922e8d70…`, 18 members).
The original stage2 remains STAGE2_INCOMPLETE. The successor joins all 96 graph
replays; this is inventory matching, not broader physical completeness. Four
prefill calibration chains, 22/40 resource windows, counters, decoded metrics,
cross-rank clocks and shapes remain future research rather than review blockers.

The owner next requested a historical rejection audit and a list for review.
[The shortlist](docs/MXFP4-REJECTED-TRIAL-REASSESSMENT-20261106.md) covers v6
prepacking, updated HRX, conditional M4096 prefill work and small-positive
historical FP8 options. The 25-ID audit is sealed at
`rejected-optimization-reassessment-20261106-v1` (`5a29891c…`, 37 members).
No trial is reopened, no new GPU work is queued and selection is unchanged.
Do not infer that old numerical failures disappeared. Native FP8/BF8 arithmetic
with MXFP4 storage remains deferred on its existing proof obligations; cap32
stays dropped and BW-03 stays a future research option.

Carry the corrected denominators and overhead into future handoffs: GEMM shares
66–70% short / about 45% long are of VERIFY, not the whole GPU round. Whole-round
shares are 52–57% / 38–39%. Host-round profiling inflation is 2.9–5.9% long and
4.9–9.9% short; unprofiled dispatch-gap cost is unknown. Table ranges are
two-round means, individual RCCL rounds reach 4.04 ms and include peer wait,
round-1 clock widths are 66.3–92.8 microseconds, short attention changes
+1.7% to +3.3%, and clock/GEMM correlation is a request-sample hypothesis.
Legacy bf16/mtp means candidate start 1; kv4 means control start 2.

The chronology below retains earlier preparation states. Use this latest gate
and the finalization record for current readiness.

The owner requested: finish the Opus investigation/remediation, test a positive
candidate on GPU, run the final compatible combined benchmark and resource
diagnostic, select the fastest fully passing profile, then independently release
and park. Finish this document with the actual winner and evidence when those
steps complete. Do not describe the current state as adopted or parked.

Operation227 completed the reversed three-member cap16 confirmation with a
3.7394% balanced score gain. The frozen readout returns
CONFIRMED_KEEP_REVERSED_ORDER. All 30 same-job token streams and speculative
counters match, the control reproduces op224, and quality passes 60/60.
Independent release passes at checkpoint610. Independent result review accepts
the frozen classification and recomputes the score from raw native receipts.
Operation226's control-first A/B independently passed at 3.2652%, with the same
work and quality checks and release at checkpoint608. The two orders are not
pooled to decide a pass. No profile is adopted. The diagnostic is now reviewed;
final selection remains pending the owner's shortlist review.

The balanced objective scores request latency; group-wall time supplies noise and
regression vetoes only. Operation227's aggregate measured group-wall reduction is
about 2.26%, distinct from its 3.7394% balanced score gain. Resolved warm/short
gains hold in both orders; long-cold changes stay within noise and are
order-sensitive. Treat the reader's `fastest_fully_passing` as an eligibility
label, not a final selection. The sealed readout's group-wall scoring wording is
superseded by the independent review and this correction; leave it immutable.

Independent review accepts the reversed-order CPU design and repair delta.
The exact six files are integrated: 93 canonical-location tests, public43 and
one device-free native Python 3.14.4 run of 129 tests pass. All 1188 native input
records match. Final packet `daa8a60bd5319c17` is sealed; exact inert admission,
the registrar proof (58 refused mutations) and the remote readiness preflight
pass. The owner approved this exact packet and exception clause, and its single
invocation completed and independently released. The nonce and allowance are
consumed; no retry, adoption or selection change is authorised. The pinned entry
enforced the rule-derived exception clause before spend. No further GPU allowance
exists. The sealed readout, source and release pins are in the finalization record.

After unprofiled confirmation, prepare a separate matched baseline/candidate
profiling packet with all compatible positive members. The 10 s passive sampler
does not close this profiling step. Cover GPU work and host launches, TP waits,
copies, graph boundaries and power/clock observations. Close the timestamp and
graph-coverage prerequisites before spend; keep unavailable counters explicit.
Treat instrumented timings as diagnostic and use the unprofiled comparison for
selection. Review the data before choosing further optimisations. Complete this
step, document the findings, then select and park. Profiling needs its own exact
finite owner decision; no profiling invocation is authorised yet.

The matched profiling CPU preparation is sealed at
`matched-three-member-profiling-cpu-20261105-v1`, with 164 local tests in 13
suites reported passing. Independent review accepts this CPU stage with findings;
it does not sign off a GPU packet. The parent repaired the read-only recipe,
passed six focused regressions and public43, then ran one source-only check.
That check retained the exact label-helper candidate and pinned Torch graph
source, plus HIP function-name/header evidence. Bind their runtime chain before
closing the label gate. The capture-time node-inventory producer, composed
packaging, canonical integration, native Python proof and fresh admission remain
open. Fix review R2–R9 during composition, including D2H-only clock constraints,
exact inventory identity, drift/claim bounds and the prefill-overflow policy.
The kernel metrics header was absent; keep raw gpu_metrics undecoded. No source
check or historical trace closes the physical gates or supplies a GPU allowance.

The composed profiling packet is now an inert preview in
`matched-three-member-profiling-composition-20261105-v1`. It reports 141 passing
local tests; the legacy generated suite passes 5/9 and its four incompatible
fixtures remain unresolved. The full schema-2 two-rank barrier/readout path and
materialized argv limit also remain open. Four proposed protected-helper diffs
need a fresh exact exception. A source-only ABI check verified six exports in
the pinned HIP library but did not capture `dim3`/`hipError_t`, so layout and
kernel-name route qualification remain open. Keep both the preview and this
partial ABI result distinct from a qualified or approved GPU packet.

The later final-CPU qualification root is sealed and independently reviewed:
`matched-three-member-profiling-final-cpu-qualification-20261105-v1`.
Claude resumed after the rate limit and completed 50 local tests, including both
starts through the full two-rank schema-2 path and ported legacy cases. That test
found a real backend-descriptor mismatch; the proposed entry-template fix is in
the current preview. Preserve the superseded preview and its receipts. Argv
materializes at 254 per arm against 256. The ABI qualifier uses exact installed
declarations, native scalar sizes and the x86-64 psABI layout rule; native ctypes
Structure confirmation was blocked by the read guard's lazy-import refusal.
The reviewer accepts the ABI proof; native ctypes confirmation is optional.
Do not repeat the completed native header step or write into this sealed root.
Review found a startup refusal: the library lookup rejected Linux's resolved
`libstdc++.so.6.0.35` filename. The parent repaired it in the fresh sealed root
`matched-profiling-loaded-library-repair-20261105-v1` (18 local tests pass;
both old-source negative controls fail as expected). The kernel-name lookup
method is unchanged. Regenerate the dependent package and protected-helper
hashes before integration and native proof. Check the real mapped library in
G8, then bind the fresh exact exception to the final bytes before registration.
The later integration/native-proof root is sealed at
`matched-three-member-profiling-integration-native-proof-20261105-v1`.
G7 integrates six modified files and ten new modules/tests/readers locally.
All 158 required tests pass locally and in one native Python 3.14.4 CPU proof;
public43 passes. The real rootfs lookup accepts the resolved library and the
old producer refuses the same maps. All six optional native ABI offsets agree.
The native harness runs up to three CPU test children concurrently; this
deviation, the test-path relocation and local fixture repairs need delta review.
The 40 clone-style kernel names remain prospective and fail closed on mismatch.
Independent delta review accepts exact registrable construction with no blockers.
The native harness permits up to three children plus its parent, each with a
3 GiB address-space ceiling; aggregate address space can reach 12 GiB and no
per-child usage is retained. The old generated test stays intact; the new
matched test needs the sealed test helpers named in its reproduction command.
For final construction, check the exact single changed line in each of the two
build-dependent helpers against the build's own shared-entry/spec hashes; do
not rely on a broad hash mask. Check the shared-entry pin in all three consumers.
The other two helper diffs and all nonce-independent files must remain identical.
This is still an inert preview. Registrable construction, the exact protected-
helper exception on its final bytes, fresh admission and the profiling GPU
decision remain open. Disclose that multiple mapped HIP versions refuse startup.
No GPU or selection change occurred.

The registrable profiling packet `a83f1eba93d913fe` is now built and sealed in
`matched-three-member-profiling-final-packet-20261105-v1` (771 members).
Its experiment delta satisfies the reviewed construction conditions; limits
and producer bytes are unchanged. Final-packet checks pass: 15 registration/
preflight/readout tests, 22 end-to-end tests and four canonical matched tests.
The canonical launcher refuses this diagnostic-only serving combination, so
the preparation uses the op216/op217 direct-route pattern. Independent glue
review refused the sealed post-release path: it binds the old stdout-only
replay, calls a baseline-only comparator for candidate/control records, and
sets an unsupported Mac address-space limit. Preserve that refused root.
The fresh `matched-profiling-postrelease-repair-20261105-v1` is now sealed
(137 members). Forty receipted CPU tests and eight retained raw-frame replay
checks meet their expectations. Reader main, finish failure classes, the
three-review approval chain and a mandatory stage-2 offline-analysis command
are covered. Finish uses labelled CPU seams; positive stage-2 graph membership
is synthetic. These tests do not close physical capture or calibration gates.
Independent follow-up review now accepts the repair for exact final registration
derivation, with no blockers. The final wrapper is derived and sealed at
`matched-three-member-profiling-registration-a83f1eba93d913fe-v1`
(`e6f467be…`, 8 members). Mechanical checks verify G/F/L/P0 seals, the three
byte-identical copies, the exact decision substitution and computed plan.
PLAN is `79c2642e…`; decision is `0869c602…`; review is `1591c020…`.
The draft-only note is removed and pending fields are filled. No GPU bytes changed.
The owner decision and fresh remote admission remain open. The exact owner
answer must quote the four-helper clause; keep its approval file outside G/F/L.
Before mandatory stage 2, re-verify F and P0 and compare recorded reader pins.
Use both group directories and check nonzero resolved resource windows and
coverage; the stage-2 status alone does not prove resource completeness.
The frozen finish can mislabel an unexpected post-quality readout fault; retain
the fault for independent result review. The local-only preflight is not
admission; no new GPU allowance exists.
The owner has now authorised the linked exact proposal. The authorization record
preserves the reply `authorised` verbatim and identifies the accepted proposal
as context, rather than inventing a longer owner quotation. Fresh remote
admission passed at checkpoint610/index228. Operation228 registered at
checkpoint611 and then failed during candidate scheduler initialization. The
retained error is `graph_inventory.InventoryError: graph inventory: graph backend
class` at the producer's `FullCudaGraphBackend` class check. Its caller exited 1
after about 126 seconds. The bound finish verified independent release and closed
the failed campaign at checkpoint612. Both configured cap readbacks were 235 W.
The allowance is consumed; do not invoke or retry it again.
Mandatory stage 2 ran after F/P0 verification and returns STAGE2_INCOMPLETE:
there are no completed captures or matching client groups. No benchmark or
quality verdict exists. Preserve the sealed native run, campaign, invocation
and stage-2 roots. CPU-only diagnosis and a regression repair are next; a new
GPU invocation needs a fresh packet and exact decision. The earlier op226/op227
results stand. Final selection and programme parking remain pending.

The op228 CPU repair is sealed at
`op228-graph-backend-class-cpu-repair-20261105-v1` (`65db16ec…`, 33 members).
The retained selected source chain implies `abc.ABCMeta`; the actual runtime
class value was not retained. The old plain-type guard therefore rejects the
selected class. A candidate pins exact ABCMeta, module and qualified name while
preserving the remaining identity checks. Thirty local tests pass using the
exact base/backend source with labelled dependency stubs. The old producer
reproduces the refusal; the old plain-type fixture also fails with the fix.
Independent review now accepts the source repair with no blockers
(`fdadcf4b…`). Keep the reviewed candidate bytes unchanged. The optional exact
canonical-module-name check is not required: the existing path and method-code
checks still authenticate the loaded source.
The later `op228-profiling-integration-native-proof-20261105-v1` is sealed
(`cffc18f7…`, 8,550 members). Only the canonical producer changed, byte-identical
to `7c34c526…`; both builders are unchanged. Fresh fixtures use the exact
selected class, and the unchanged e2e harness gets it through the new support
helper. Seven affected suites pass 61/61 locally and in one native Python
3.14.4 CPU proof; public43 passes. The native proof binds the real `abc.py`,
class metaclass and method-code match without importing frameworks or mapping
HIP. Earlier unchanged library/demangle/ABI receipts are reused.
The inert preview nonces `eaac3fbaa37c8aa8`, `5c273accad6a3781` and
`5c24618dfde25fc4` must never be registered or uploaded. Builder-generated
dependent pins and helper deltas are recorded. Staged size grows by exactly
403 source bytes; ceilings remain unchanged. The native harness retains the
possible 12 GiB aggregate address-space ceiling. Fixtures, preview delta and
native-harness changes pass focused independent review (`a932d609…`), with no
blockers. Exact registrable preparation may proceed under its conditions 7a–7h:
fresh identities, unchanged producer, exact archive and staged sizes, rematerialized
caps and literal helper-pin checks. Do not reuse the a83f wrapper or admission.
Carry two wording corrections into final construction: the e2e harness is from
the final-CPU root (`b17e12f2…`), and raw native stdout `936718ef…` is authoritative;
the summary's legacy flag has no meaning.
Fresh packet `9ed7506ed1c0cd54` is built and sealed in
`op228-profiling-final-packet-20261105-v1` (`39649d2e…`). Its wrapper is
`op228-profiling-registration-9ed7506ed1c0cd54-v1` (`39c410ce…`), with PLAN
`b6a83653…` and decision `be4627fb…`. The exact helper exception is `c26db25a…`.
All accepted construction conditions pass; 64 pre-seal and 17 post-seal CPU
tests are retained. The native 61-test proof is reused, not rerun.
The parent narrow review (`0cc28d92…`) accepts the fresh chain with no blockers:
all three current review hashes are fixed; preflight predicates are unchanged
and name op228's released roots; PLAN recomputes exactly, finish is unchanged,
and the bound reader differs only in CONFIG. The proposal remains PENDING and
all future execution roots are absent. Actual remote admission must still pass
at checkpoint612/index229 within 1800 seconds of the single entry. Present the
sealed decision and proposed approval text; no further GPU invocation is yet
authorised. The a83f nonce stays consumed. No adoption or selection change.

The owner subsequently approved this exact linked proposal. Its separate
authorization root preserves the reply `approved` and the accepted proposal
context. Fresh remote admission passed at checkpoint612/index229, and the
bound entry registered operation229 at checkpoint613. The configured cap
readbacks were 235 W on both cards. Both starts completed; all 60 natural-stop
quality grades pass and the cross-start readout returns SAME_REQUEST_WORK.
The bound finish reports SUCCEEDED / MATCHED_PROFILE_CAPTURES_COMPLETE and
verified independent release at checkpoint614. All seven windows per rank per
start are retained. The native, campaign, invocation and offline-analysis roots
are sealed and verified. The allowance is consumed; do not retry the entry.
The mandatory offline analysis returns STAGE2_INCOMPLETE: every decode window
has one graph node without a dispatch record (24 membership refusals). Resources
align for 18 of 40 windows from 149 point samples; 22 windows remain unresolved.
Capture metadata completeness is not full graph coverage. Investigate the node
and GPU/resource attribution from retained data on CPU, without dropping the node
or weakening the frozen gate. No new GPU allowance exists. Independent result
review, final selection and programme parking remain pending.

The initial three-member cap16 A/B passed native compiler proof, 104 native CPU tests,
independent review and the final registrar proof. The owner approved packet
`c759eeb5179094dd`; operation225 consumed the invocation but stopped at the first
pre-launch hostname assertion. No upload, model start or GPU kernel ran, and no
benchmark result exists. The frozen classification is INCOMPLETE. Original
checkpoint605 remains unchanged; separate absence and caller-source evidence
supports the no-native-owner reconciliation at checkpoint606. The attempt is
parked, with no retry or selection change. Finalization remains incomplete.

After hostname restoration, fresh packet `f6ebaaf7b67fb166` passed delta review,
the final registrar CPU proof, exact owner approval and reviewed remote admission,
and ran once as operation226. The immutable source/preflight pins and full remote
status were checked. That nonce and allowance are now consumed. A future packet
must recheck current source, boot, ownership and index state; the old preflight
hard-codes checkpoint606/index226 and cannot admit another attempt.
Do not change the host, relax its guard, reuse the consumed nonce or rerun its
allowance. Use the finalization record for all restoration and failure evidence.
The owner dropped cap32 on 5 November in favour of cap16. Preserve its evidence
and lead ID, but do not schedule further cap32 work without a new owner request.

## Findings that must carry forward
- The current bundle has three members: verification prefetch, gated-norm/MXFP4
  quantizer fusion, and gate/up GEMM wave-hint removal, with cap16 in both arms.
  Cap32 is dropped by owner decision; its earlier results remain historical.
- Keep BW-03 certified output-head screening open as a future research option,
  as requested by the owner. It is not a bundle member or a measured speedup.
  Before GPU lowering, measure representative operand margins and certificate
  coverage, qualify production rounding and consumer eligibility, and establish
  the complete screening/fallback cost. Do not start another campaign automatically.
- Operation224 completed quality and lifecycle checks but remains INCONCLUSIVE.
  Its arms produced different reasoning streams while final answers matched.
  Changed verification counts do not isolate an acceptance-efficiency change.
- Cap32 has non-bitwise device evidence. It is the leading explanation for the
  changed trajectory; the individual-member causal attribution is still unproven.
  F1 is bitwise-intended, with universal production-input equivalence still open.
- Operation217 remains NOT_COMPARABLE. Preserve its work gate and all differences.
- Static ISA counts suggest substantial MXFP4 unpack/scale work. They are not
  dynamic instruction counts, measured stalls, matrix saturation or a speed bound.
- The owner used the default `amd-smi monitor` display. The retained installed
  source maps `MEM%` to `average_umc_activity` and `GFX%` to
  `average_gfx_activity`. These activity percentages do not establish physical
  GB/s. The passive inventory found both cards suspended with EBUSY sensor
  reads; it observed no live clocks or gpu_metrics structure revision. Keep
  10 s caller-side sampling bounded and report-only. Operation227 retained 149
  rows, with 143 readable observations and six EBUSY observations per card.
  Whole-run GPU-busy medians are 100%; sysfs `mem_busy_percent` medians are
  16%/15%, not asserted equal to AMD-SMI's field. Raw gpu_metrics is 120 bytes
  with header `78000103`; its fields and revision are not yet qualified.
  Repeated reads of the same power sensor disagree and include spikes. These
  point samples do not establish cap residency, throttling or energy. Use
  "configured 235 W cap"; do not infer sustained draw from the limit.
- Nine exact Kineto source blobs match the retained build inventory. GPU activity
  and runtime APIs use different timestamp conversion paths. Calibration values,
  error bounds and complete graph coverage remain open; a CPU annotation bound
  is not a GPU/runtime-API calibration.
- Native BF8/E5M2 WMMA with MXFP4 storage is an unqualified operand-decoding lead.
  It needs exact supported-domain conversion, scale/exception fallback and
  production accumulation proof. Do not include it among measured positives.
- The [HRX upstream reassessment](docs/HRX-UPSTREAM-REASSESSMENT-20261105.md)
  confirms our event-recording blocker is fixed in upstream source. An isolated
  runtime re-evaluation is worth considering after profiling; RCCL compatibility
  and gfx1201 benefit remain unqualified. The HIP pins it assumed belonged to
  the wiped SGLang 0.5.21 runtime and are historical; current pins are in the
  workstation [`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock).
  Do not automatically launch another campaign or substitute HRX.

## Preserve experiment meaning
Keep the explicit thinking-enabled low-effort request profile, frozen tokenizer,
model revision, precision, TP2 topology, usable KV/recurrent capacity and
configured 235 W cap where the packet requires them. Disclose the common cap3,
LLVM_PATH and profiling-runtime deviations. Keep ordinary decode unchanged
unless it is the declared variable.

Use same-job baseline/candidate evidence and preserve quality, numerical, output/
work, noise, per-case regression, retention and release gates. Forced equal-token
work is not normal-completion quality. Instrumented latency is diagnostic.
A mechanism classification cannot be converted into a serving-performance pass.

All 25 lead IDs remain in the queue, including overlaps, closed variants and
deferred ideas. Do not reopen rejected tile, codec, numerical or C4 variants
without the recorded reopening evidence. No fifth member is qualified.

## Source integrity incident
An Opus test rewrote byte-identical EXPORT-BUDGET-REPORT.json in the historical
packet-v4. Its original seal now fails the timestamp predicate. Preserve that
original failure. Do not backdate or modify the old root.

The documented content-exact recovery is
scratchpad/mxfp4-ultraquant-feasibility-20261026-v1/fp8-fastpath-research-v2/hyperloom-sglang-implementation-20261102-v1/optimization-continuation-v1/packet-v4-source-recovery-20261105-v1
(seal 024efa0110a5c0dc253ba5f36db40510a48081ff70d15e74be4a5dabea4d9729).
It includes 367 members and a recovery note. Recovery supplies source evidence,
not nonce reuse, execution authority or a repaired historical owner completion.
Tests must write only into fresh mutable fixtures and refuse writes to sealed roots.

## Qwen4 restart
When the target checkpoint is available, discover its exact revision, architecture,
quantization, executed operators, layouts, topology and launch configuration.
Do not infer its compatibility from a model name or copy old gains into its record.

Reusable work includes gfx1201 instruction/layout research, exact operand
references, profiling/identity/retention tooling and finite-owner controls.
Requalify GEMM shapes, attention layouts and split counts, fusion boundaries,
recurrent state, cache policy, draft/head compatibility and quality on the new
checkpoint. Retain the owner's memory-capacity preference; do not silently
replace compressed weights with a larger representation.

After finalization is PARKED, do not automatically start another campaign.
Read the actual final selection and remaining gates, then prepare the newly
requested Qwen4 work through the existing owner workflow.
