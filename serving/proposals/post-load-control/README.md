# Control before replay metadata preparation

Pre-memory operation 202 reproduced operations 196 and 201 exactly: its first
warmup returned 3,318 token IDs and two closing thinking markers. This occurred
without shared head allocation or packed parameter rebinding. The response
source remains unresolved. See [operation 202](../../measurements/native-qualification/GPU-202-RESULT.json).

This proposal retains the post-load cleanup stage, before the replay-metadata
increment. Compared with operation 202, only `verify_decode_adapter.py`,
`worker_entry.py` and `serving_integration.py` change. The other 57 Python files,
the post-load hook, native execution contract, nine protected helpers and all
176 frozen requests remain identical. The adapter prepares metadata within the
explicit forward and records that work in captured graphs, as in the earlier
stage. This is a diagnostic reversal, not a numerical fix.

The only startup correction removes the retired `none` draft-summary call from
the exact post-load worker. Original and corrected hashes are required. The
builder still admits the prior pre-memory correction only under its existing
exact hashes. Other proposals retain their original bytes.

The source-only proof helper declares the older witness format explicitly and
executes its complete producer and original reader. That reader still requires
two metadata builds during capture. Current-format checks retain the module
policy requirement and reject a legacy declaration. Tests retain the original
startup refusal and both readers' negative cases. No native response acceptance
rule changes.

Baseline `ca3598712e150f21` is the selected next control. Six local package/caller
checks and 21 native CPython 3.14.7 source-only checks pass. The native check uses
both exact archives and their own root transports, complete materialization,
draft preflight and startup producer/reader. Native metadata, tensor objects and
root admission are explicitly substituted. GPU access, native serving imports,
network and subprocesses are denied inside the check.

The paired current candidate is an inert preparation artifact. It is not
scheduled for execution. A baseline result can narrow the regression to the
replay-metadata increment or an earlier stage. It cannot establish a performance
improvement or qualify the current cleanup.

Approved operation 203 completed both starts and all 352 responses. Every raw
request/output vector matches B186; native CPU quality and long-answer coverage
pass. The original caller completed, and GPU/CPU releases are verified. Checkpoint
557 is parked. See [the result](../../measurements/native-qualification/GPU-203-RESULT.json)
and the [CPU correction](../replay-metadata-scope/README.md). The correction needs
its own native qualification and fresh exact approval. The earlier
[source verification](VERIFICATION.json) remains retained.
