# Verification metadata reuse candidate

Decision, 21 October 2026: retain metadata reuse in the experimental candidate
and use it in subsequent candidate comparisons. The owner accepted the bounded
validation result. An additional throughput gain remains unproven. This
decision does not change an installed runtime or complete Sprint 7 qualification.

`prototype/serving/verify_decode_adapter.py` derives from the adapter used in
the published combined-path benchmark. It preserves the native attention call,
cache writes, eligibility checks and eight independent singleton split-selector
calls. It changes when metadata is prepared, not attention arithmetic.

The original adapter prepares identical causal lengths, packed read indices and
split counts in each of the 16 attention layers. The candidate prepares them
once per explicitly initialized forward and shares those buffers across eligible
layers. Index construction reads the request-to-token table, not layer-specific
KV values; the native per-layer KV writes remain before attention.

The adapter wraps the authenticated eager and out-of-graph metadata initializers
and the inherited in-graph no-op hook. It refuses installation if that inherited
hook has effects. Every initializer invalidates the previous scope, including
failure paths. Reuse requires the same backend, batch, native metadata object,
prefix-pointer tensor, request-index tensor, translator, stream and sizing
policy. Unsupported calls invalidate reuse. Without an explicit initializer,
the adapter retains the original preparation behavior.

Warm-up preparation cannot satisfy graph capture: the capture state is part of
the reuse check. Capture records preparation on the first eligible layer.
Every graph replay executes those device operations again against live prefix
and request buffers. This does not cache device values between verification
steps. The graph retains all eight singleton selectors; applying the native
selector once to eight rows changes its split policy and is not supported.

The focused CPU tests compare the original and candidate adapter with declared
tensor, stream, selector, index-builder and graph substitutes. They exercise
growing and shrinking prefixes through 32,768 causal tokens, reversed slot maps,
live-input replay, stable buffers, changed inputs/streams, unsupported calls,
initializer failure and hook restoration. They check the metadata preparation
count falls from 16 builds and 128 selector calls to one build and eight selector
calls per simulated 16-layer forward. They do not establish native arithmetic,
GPU synchronization, throughput or broad model quality.

The frozen original adapter in `tests/fixtures/verify-metadata/` is a regression
oracle, not a second maintained implementation. Source identities and the bounded
native validation result are recorded in
[`VERIFY-METADATA-REUSE.json`](../provenance/VERIFY-METADATA-REUSE.json).

Run these checks without a GPU runtime:

```sh
python3 -I -B -S tests/test_verify_metadata_reuse.py
make test PYTHON=python3 CXX=clang++
```

## Bounded native validation, 21 October 2026

The frozen candidate passed native validation on both R9700 ranks. Source,
runtime, model and worker identities, graph replay, HTTP records, resource
limits and independent release evidence passed authentication. Both remote
evidence manifests and the protected audit export verified against their seals.

Each rank recorded 16 eligible attention calls during capture and one captured
metadata build. The startup host counters were three metadata builds and 45
reuse observations, covering startup warm-up and capture. These counters do not
increment on each graph replay. The captured device preparation still executes
on every replay against live inputs.

All eight requests completed with zero cached tokens and zero retractions.
All four ordinary/candidate output-token pairs matched, including the excluded
warm-up. Candidate outputs also matched all four earlier combined-path outputs.
This checks truncated token prefixes, not complete-answer quality or exact
equality of every hidden activation and recurrent state.

The workload used TP2, concurrency one, low thinking, 451 input tokens and
exactly 128 output tokens. Each arm had one excluded warm-up and three measured
requests. The median local non-stream HTTP rate was 87.29 output tokens/s for
the candidate and 24.47 for ordinary serving. These rates cover both GPUs
working together and include prefill and HTTP completion.

The earlier combined path measured 86.68 tokens/s. The approximately 0.70 percent
difference comes from separate runs, without a matched old-adapter/new-adapter
comparison in reversed order. It does not establish an additional throughput
gain from metadata reuse.

Native coverage remains a short-context C1 screen. The CPU tests through 32,768
tokens do not establish native long-context coverage. Broader quality,
concurrency, lifecycle and performance qualification remain separate work.
The original adapter and evidence remain retained. No installed runtime or
serving default was promoted.
