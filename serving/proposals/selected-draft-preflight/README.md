# Selected draft preflight

Operation 194 passed source authentication and reached both native scheduler
processes. Each process refused before Scheduler construction because the
selected preweight path eagerly requested the retired `none` draft summary.
No response, second arm or quality grade ran. Collection and independent GPU
release passed; checkpoint 535 is parked. The failure remains sealed.

The correction removes that one unused initialization call. The enabled path
still resolves the actual draft metadata and requires exact equality with the
selected DFlash2 summary. Draft configuration, quantization intent, loading
policy, memory pools, observability and metadata checks remain in place. The
contract still refuses retired speculative kinds.

The [regression suite](../../tests/test_draft_preflight.py) executes the complete
preflight with the real selected contract and JSON writer. It covers both arms
and ranks, and refuses changed draft identity, architecture, dimensions, dtype,
quantization, width, file digest and explicit quantization intent. The
[packet checker](../../tests/check_draft_preflight.py) reproduces the retained
failure and passes all four arm/rank combinations in both corrected modes.

Native resolver and ModelConfig objects, the draft file digest and stock
precision-source validation are explicit metadata-only substitutes. These tests
do not establish native ModelConfig behavior, Scheduler construction, model
quality, GPU arithmetic or memory recovery.

Validation: 1,021 Python tests across 84 suites plus C++ sanitizers; 14 focused
native CPython 3.14.7 checks with GPU access, native serving imports, network and
subprocesses denied; six complete campaign checks per mode. The exact new
archive passes the root-entry prefix check. Complete worker materialization
passes for both arms in both modes. See [VERIFICATION.json](VERIFICATION.json).

Only `worker_entry.py` changes among the 61 worker/protected Python files. Its
runtime patch removes one line. All ten protected inventory entries, kernels,
requests, settings, pools and ceilings are unchanged. The five computation
sources covered by the real CPU tensor proof retain their hashes; reuse is
limited to that tensor-structure result.

Fresh instrumented packet `e201110200000095` requires one-use approval. GPU
startup byte proofs, greedy token identity, selected quality, BDF memory,
production comparison and final evidence closeout remain pending. No memory
or performance improvement has been measured.
