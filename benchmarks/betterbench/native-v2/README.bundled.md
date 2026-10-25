# BetterBench selected-profile checkpoint

This directory prepares a bounded BetterBench performance checkpoint for the selected Qwen3.8-27B profile. The adapter preserves thinking with `enable_thinking=true` and `reasoning_effort=low` on every request path. It does not start a model server. Native execution still needs a reviewed owner packet, authenticated worker readiness and independent resource release.

The unmodified upstream files are pinned to [BetterBench v0.6.1](https://github.com/GGZ14/BetterBench/tree/f72bb9011a830fd98e82ab89d01e549efa76e9c5), commit `f72bb9011a830fd98e82ab89d01e549efa76e9c5`. [UPSTREAM.json](UPSTREAM.json) records the exact bytes, SHA256 and Git blob identity of its runtime, corpus, configurations, licence and mock self-test. [The upstream Apache-2.0 licence](upstream/LICENSE) applies to those files. No upstream source file was changed.

The client uses the existing native NumPy package. Upstream requires NumPy >=1.24. The native preflight records the exact installed version, module path, source and binary inventory, and interpreter hash. The benchmark refuses a changed dependency receipt. There is no dependency installation step.

## Declared workload

[profile.json](profile.json) is the active configuration. Each arm sends exactly 53 generation requests:

| Phase | Requests | Scope |
| --- | ---: | --- |
| Categories | 32 | Eight categories, one warmup and three measured passes each |
| Prefill | 9 | Approximate depths 2K, 8K and 16K; one warmup and two measured passes each |
| Concurrency | 12 | Six requests at C1 and six at C2 |

The pinned corpus contains 29 prompts. Non-prefill output caps are changed from upstream's 120–900 tokens to 4096 tokens to provide thinking headroom. Prefill probes retain a 16-token cap and measure prompt processing; they do not establish answer completion. Sampling is temperature 0.7, top-p 0.95, top-k 20 and seed 1729. Fresh prompt nonces remain enabled.

The native context is bounded to 32768 tokens. Synthetic prefill depths are approximate; they leave substantial context headroom. The socket inactivity timeout is at most 120 seconds per request. The supervisor enforces a separate 2400-second wall limit, followed by at most five seconds each for TERM and KILL reaping. The owner must fit startup, both arms and release inside its admitted schedule. These ceilings do not predict that a run will complete.

Three measured passes are a low-sample checkpoint. This is not the default or full BetterBench workload, a resolved 1% comparison, or a quality qualification. Preserve the report's thin-sample marks, truncation rates and thinking/answer estimates. Run the existing final-answer quality gates separately. The `paired_ab` request path is regression-tested, but this checkpoint runs single-stream, prefill and C1/C2 phases on one admitted server at a time.

## Preparation and verification

Run the offline tests with an existing Python environment that contains NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -I -B benchmarks/betterbench/test_selected.py
python3 -I -B benchmarks/betterbench/run.py --source-preflight --output SOURCE-CHECK.json
```

The tests use a local mock HTTP server and the actual pinned runner/client. They check every request path, the exact count, C1/C2 concurrency, selected request bodies, failure retention, deadline/count refusals and child termination. A source-only result cannot establish native interpreter or NumPy compatibility.

Inside the existing device-free native CPU sandbox, use the authenticated Linux x86_64 Python 3.14.4 interpreter:

```sh
python3 -I -B benchmarks/betterbench/run.py --native-preflight --output DEPENDENCIES.json
```

Pin that receipt's SHA256 before preparing the owner packet. After the owner authenticates the selected server's readiness, its client callback can invoke:

```sh
python3 -I -B benchmarks/betterbench/run.py \
  --endpoint http://127.0.0.1:18080/v1 --model SERVED_MODEL_NAME \
  --dependency-receipt DEPENDENCIES.json --dependency-sha256 RECEIPT_SHA256 \
  --output FRESH_BENCHMARK_DIRECTORY
```

The port and served model name must come from the authenticated server binding. The adapter permits only an explicit loopback `/v1` endpoint and sends no API key. It imports no Torch or SGLang, probes no local GPU tools and does not invoke upstream's update check.

## Evidence and publication

`requests.jsonl` retains exact request bodies, their hashes, per-request results and failures as each request finishes. The request journal is capped at 16 MiB. `terminal.json` records child completion, timeout and reaping. Successful completion also writes upstream-compatible `results.json`, a standalone `results.html` and `summary.json`. The adapter caps its final result artifacts at 32 MiB. An interrupted or refused run retains its partial files and has no successful terminal result. There is no automatic retry.

The client terminal is not proof of model-worker release. Keep the existing source, weight, worker, readiness and independent release receipts. Bind the adapter/profile, complete corpus, native dependency receipt, selected source and server settings to the result. Review and sanitize private dependency paths and native receipts before publication. Publish the reviewed JSON/HTML with the bounded workload, sample and truncation limits, upstream attribution and separate quality results. No benchmark or publication has run as part of this preparation.
