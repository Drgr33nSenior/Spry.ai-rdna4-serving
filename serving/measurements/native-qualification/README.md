# Native cleanup qualification

The current cleanup remains unqualified. Reference operation 198 completed
352 responses whose request and output token vectors all match passing B186.
Collection and independent GPU release passed. Its original local caller was
interrupted, so complete operation replay and conditional CPU grading remain
unestablished. See [GPU-198-RESULT.json](GPU-198-RESULT.json).

Production operation 201 used candidate `e201110200000099`: the same 61 Python
sources as operation 196 and all 176 B186 request bodies. Its first warmup
produced exactly operation 196's 3,318 token IDs, including two closing thinking
markers. The unchanged response guard refused it. Both ranks passed production
readiness; the original finite caller completed with its receipt. Collection,
power restoration, host-fault checks and independent GPU release passed. The
second arm and conditional CPU grade did not run; checkpoint 551 is parked. See
[GPU-201-RESULT.json](GPU-201-RESULT.json) and the
[response comparison](PRODUCTION-RESPONSE-COMPARISON.json).

Instrumentation alone does not explain this failure. Approved pre-memory
operation 202 then ran the earlier stage without head sharing or packed
parameter rebinding. Its first warmup produced exactly the same invalid
3,318-token vector. Both ranks passed readiness, the original finite caller
completed, and collection, power, host-fault and independent release checks
passed. No second arm, CPU grade or retry ran; checkpoint 553 is parked. See
[GPU-202-RESULT.json](GPU-202-RESULT.json) and its
[token comparison](GPU-202-TOKEN-COMPARISON.json).

The [control before replay metadata](../../proposals/post-load-control/README.md)
reverses only that increment's adapter, startup producer and reader relative to
operation 202. Its other 57 Python files, nine protected helpers and 176 frozen
requests match. This comparison needs fresh exact one-use approval.
The separate Hyperloom session is investigating zero warm C2 cached tokens in
operation 200. That failure has not been shown to explain the thinking-marker
regression here. No GPU memory or performance improvement has been measured.

The approved GPU-free CPU tensor check passed with independent release in
18.17 seconds. Real CPU tensors established shared head objects, packed view
offsets, clone equality, parameter/loader metadata preservation and P1 stale
identity refusal. The head geometry is compact and ops are stubbed with explicit
logical-device substitutes. This does not prove native quantizer arithmetic.
See [CPU-STRUCTURE-RESULT.json](CPU-STRUCTURE-RESULT.json).

Approved operation 191 then refused during protected bootstrap staging before
any model start. The common helper was authenticated and staged, but the root
transport's exact trusted set omitted it. The refusal and failed collection are
retained. An independent read-only observer verified the absent workload root,
ownership journal and held lock, with zero GPU device holders. The global ledger
is parked at checkpoint 529. No retry or quality grade ran. See
[GPU-191-RESULT.json](GPU-191-RESULT.json).

The [bootstrap staging correction](../../proposals/bootstrap-common-stage/README.md)
adds the exact common helper to authentication, membership and byte accounting.
Its fresh packet needs a new one-use approval. GPU startup equality, selected
quality, greedy token identity, BDF memory and production comparison are still
open. No recovered GPU memory or performance improvement has been measured.

Operation 192 passed the corrected root stage and launched one worker. The
worker refused a stale GPU identity source digest before model construction;
no second arm, responses or quality grade ran. Evidence collection and
independent GPU release passed, and checkpoint 531 is parked. See
[GPU-192-RESULT.json](GPU-192-RESULT.json).

The [final template pin correction](../../proposals/final-template-pins/README.md)
refreshes source bindings after protected replacements. Its CPU checks execute
the actual worker authentication loop for both modes and arms. Runtime and
protected Python bytes are unchanged. A fresh exact GPU approval is required.

Operation 193 passed root staging and the main source checks, then refused a
stale nested GPU identity digest in the preweight configuration before model
construction. No response, second arm or quality grade ran. Collection and
independent GPU release passed; checkpoint 533 is parked. See
[GPU-193-RESULT.json](GPU-193-RESULT.json).

The [nested preweight correction](../../proposals/preweight-source-pins/README.md)
refreshes both nested helper references. The earlier CPU check stopped before
this guard. The expanded check runs the actual template reader, complete worker
materializer and written worker cross-binding in both modes and arms. All worker
and protected Python bytes remain unchanged. Fresh packet `e201110200000091`
requires exact one-use approval; GPU acceptance and production measurement are
still pending.

Operation 194 passed source authentication and reached both native scheduler
processes. Each refused before Scheduler construction because the preweight
path requested the retired `none` draft summary. No response, second arm or
quality grade ran. Collection and independent GPU release passed; checkpoint
535 is parked. See [GPU-194-RESULT.json](GPU-194-RESULT.json).

The [selected draft correction](../../proposals/selected-draft-preflight/README.md)
removes that eager call. Complete preflight tests use the real selected contract
and metadata-only native substitutes, reproduce the failure and pass both arms
and ranks in both corrected modes. Actual draft metadata equality and all
existing refusal checks remain in place. Fresh packet `e201110200000095`
requires one-use approval. GPU acceptance and production measurement remain
pending.

Operation 195 reached model/cache initialization and logged draft verify graph
capture completion on both ranks. It then failed during startup proof export
because the worker read a module constant from the adapter instance. No requests,
second arm or quality grade ran. The native HTTP export is absent; raw captures,
owner records and independent release remain retained. Collection and release
passed; checkpoint 537 is parked. See [GPU-195-RESULT.json](GPU-195-RESULT.json).

The [proof policy correction](../../proposals/startup-proof-policy/README.md)
reads the constant from the authenticated adapter module. Complete producer and
reader tests reproduce the old failure and pass both arms and ranks in both
corrected modes, using declared native metadata substitutes.

Approved operation 196 passed native readiness. Both ranks passed the existing
full-enabled startup reader: head byte equality and shared objects validated,
with all 48 packed layers and 96 shards per rank. Its first warmup response then
contained two closing thinking markers. The contract requires exactly one and
correctly refused the response. One of 352 requests ran; no valid complete
response, second arm or CPU grade followed. Collection and independent GPU
release passed; checkpoint 539 is parked. See [GPU-196-RESULT.json](GPU-196-RESULT.json).

The four historical A184/B186 warmup controls used the exact same request and
produced identical valid token vectors. The current response differs; the cause
is unresolved. Startup byte/structure checks do not prove end-to-end arithmetic
or quality. The candidate remains unqualified. No GPU memory or performance
improvement has been measured.

The [passing-reference replay](../reference-replay/README.md) ran the exact B186
source and frozen inputs in production mode as operation 198. All 352 retained
vectors match B186. The original caller interruption remains a separate
failure; its conditional CPU grade did not run. The reference is excluded from
final performance comparison. Response acceptance checks remain unchanged.
