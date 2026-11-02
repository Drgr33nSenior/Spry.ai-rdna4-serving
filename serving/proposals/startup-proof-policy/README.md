# Startup proof policy source

Operation 195 reached model and cache initialization and logged completed draft
verify graph capture on both ranks. Startup then failed while exporting proof:
the exporter read `METADATA_PREPARATION_POLICY` from the live `Adapter` instance,
although the constant belongs to its module. The server did not reach readiness;
no requests, second arm or quality grade ran. Collection and independent GPU
release passed; checkpoint 537 is parked. The failure remains sealed.

The worker now loads the authenticated adapter module once, obtains its current
instance, and reads the policy from the module. The adapter, tensor code,
metadata preparation, native configuration and proof reader are unchanged.
The runtime patch changes only `_precision_weights`.

The [regression suite](../../tests/test_startup_proof.py) executes the complete
proof producer through the real full-enabled reader. It covers both arms and
ranks, rejects invalid capture counts, layer coverage, indices geometry and
metadata policy, and confirms that an instance attribute cannot replace the
module policy. The [packet checker](../../tests/check_startup_proof.py) reproduces
the retained failure, then passes all four arm/rank combinations in both modes.
Private module authentication and JSON writes remain real.

Native linear classes, precision loading policy, loaded-weight validation,
tensor metadata and adapter lifecycle/buffers are explicit CPU substitutes.
These tests do not establish native weight equality, graph behavior, model
quality, GPU arithmetic or memory recovery.

Validation: 1,029 Python tests across 85 suites plus C++ sanitizers; 14 focused
native CPython 3.14.7 checks with GPU access, native serving imports, network and
subprocesses denied; six complete campaign checks per mode. Complete worker
materialization and the exact archive root-entry prefix also pass. See
[VERIFICATION.json](VERIFICATION.json).

Only `worker_entry.py` changes among 61 worker/protected Python files. All ten
protected inventory entries, kernels, requests, settings, pools and ceilings
retain their values. The five computation sources covered by the passed real
CPU tensor proof retain their hashes; reuse remains limited to tensor structure.

Approved packet `e201110200000098` ran as operation 196. Native readiness and
both ranks' startup byte/structure proofs passed. Its first warmup response
contained two closing thinking markers, so the unchanged response contract
stopped the run. Collection and independent release passed; no second arm or
CPU grade ran. See [the result](../../measurements/native-qualification/GPU-196-RESULT.json).

The B186 production [reference replay](../../measurements/reference-replay/README.md)
completed two native starts and reproduced all 352 request/output token vectors.
Its original local caller was interrupted; separate collection and independent
GPU release passed, but no conditional CPU grade ran. The response difference
remains unresolved. Same-source production candidate `e201110200000099` awaits
one-use approval. Candidate greedy token identity, selected quality, BDF memory,
production comparison and final closeout remain pending. No memory or
performance improvement has been measured.
