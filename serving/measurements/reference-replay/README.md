# Passing-reference replay

Operation 196 passed native startup validation but failed on its first warmup
response: two closing thinking markers were present, where the contract
requires exactly one. The same request produced an identical valid token
vector in all four retained A184/B186 control responses. The response difference
is real; its cause remains unresolved. Instrumentation and cleanup stages have
changed since that pilot, so the comparison does not isolate a cause.

Reference packet `e20111020000009b` restores the exact passing B186 production
source. All 62 Python files, nine fixed native inputs and nine protected
inventory entries match B186 byte for byte. See
[MATCHED-REFERENCE.json](MATCHED-REFERENCE.json) and
[RESPONSE-COMPARISON.json](RESPONSE-COMPARISON.json). Raw requests and outputs
remain in the private evidence workspace.

The unfunded packet requests one invocation, two starts and 352 frozen requests,
with 6,360 owner seconds and 7,160 caller seconds. One eligible post-release CPU
decode and grade is capped at 320 seconds. No automatic retry, promotion or
deployment is included. TP2, DFlash8, low thinking, BF16 target/draft KV, aligned
GDN, eleven overlays, fixed pools and 235 W caps retain their selected values.
Hyperloom and profilers are disabled.

This replay checks whether the earlier reference still reproduces its passing
behavior. It is excluded from the final performance comparison. The full
frozen workload avoids changing the old source or adding a diagnostic menu;
the existing client stops at a failed warmup. A reference pass alone would not
identify the cause of the candidate failure or qualify the current cleanup.

Six local packet checks pass for both variants. Six native CPython 3.14.7 checks
pass for the selected reference; all 62 modules also pass AST, symbol-table and
compilation checks. The first native fixture failed on a Mac-only path. The
second fixture relocates its connection-file lookup to the staged, byte-identical
source. All assertions remain unchanged. GPU access, native serving imports,
network and subprocesses are denied, with no denied events. External dispatch
is substituted in the one-use CPU fixture. See [VERIFICATION.json](VERIFICATION.json).

The source author is sealed. No registration, model start, GPU work, native
decoding or quality grading has run for this packet. Fresh exact one-use
approval is pending. The current candidate remains unqualified; GPU memory
savings and performance improvement remain unmeasured.
