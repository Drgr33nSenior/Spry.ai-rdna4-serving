# Keep replay preparation bound to captured metadata

Eager prefill replaces the native backend's `forward_metadata` object. Native
CUDA graph replay refills the graph buffers without restoring that object. The
cleanup adapter used the current object to decide whether to prepare its causal
KV map. After prefill, that decision can reject the eager metadata and leave the
captured map at its startup values. The graph then reads stale map buffers.

The CPU reproduction executes the pinned native replay initializer and the
actual old and cleaned adapters with explicit tensor and graph substitutes.
After an eager metadata replacement, the old adapter refreshes its captured map
and the cleaned adapter keeps its previous map. Both batch sizes reproduce this
failure. This establishes a lifecycle defect; it does not prove native GPU
arithmetic or complete response recovery.

The corrected adapter retains one captured metadata view per backend and captured
batch size. Replay preparation uses that view to fill the same stable map buffers.
Eager execution keeps its current metadata path. A missing or incompatible view
refuses replay; a failed recapture removes its prior view. Restore clears the
references and restores the original methods. The correction adds no device
reads, logging, dependency, kernel or acceptance-rule change.

Only `verify_decode_adapter.py` differs from instrumented operation 196. The
other 60 Python files, ten protected entries, five tensor-structure sources,
settings and all 176 B186 request bodies remain identical. The corrected module
is a new versioned source; the earlier adapter and its failing case are retained.
The startup witness format and `outside-forward-graph-v1` policy stay fixed.

Earlier control 203 completed 352 responses with all request/output vectors
matching B186. Native CPU quality and long-answer coverage pass, across eight
distinct prompt-message sequences. Its original finite caller and independent
GPU/CPU releases passed. The corrected full cleanup remains unqualified.

Validation passes 1,055 public tests, 43 boundary tests, six complete local
packet/caller checks and 31 native CPython 3.14.7 source-only checks. The native
check includes the pinned initializer reproduction, graph-size switching,
failed recapture and preparation, both exact instrumented archives, complete materialization,
draft preflight and startup producer/reader. Native metadata, tensor objects and
root admission are explicitly substituted. GPU access, native serving imports,
network and subprocesses are denied inside that check.

Candidate `e2011102000000a0` is an instrumented qualification packet. It has no
registration or launch authority. Production measurement and final cleanup
qualification remain open. The paired baseline is an inert preparation artifact;
it is not an automatic additional run. See [verification](VERIFICATION.json).
