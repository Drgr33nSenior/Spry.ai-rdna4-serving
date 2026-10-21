# FP8 cache performance attribution

The diagnostic mode measures selected forwards from the existing matched
Qwen3.8-27B-FP8 benchmark. Both arms retain TP2, eager execution, 4096 context,
the same requests and the existing owner and resource controls. The arms use
BF16 KV and KV4 respectively. This mode adds observation; it changes no runtime
kernel, codec or model weight.

Set this exact additional field in the frozen benchmark spec:

```json
"diagnostic_profile": {"schema": 1, "mode": "bounded-attribution-v1"}
```

Bind the new spec hash throughout the generated package, resource commands and
controller. The source-package generator must project its `SPEC_SHA256` constant
to that hash. A changed spec cannot reuse a consumed package or its grant.

Each rank profiles the prefill and first decode forward of sample 1 in C1-128
and C1-3966: four existing forwards per rank and arm. The complete schedule still
contains 42 requests and 960 forwards per rank per arm. There are no additional
requests, warmups or graph captures.

The entry uses the installed Torch CPU/GPU profiler and the standard-library
`cProfile` wall timer. It retains exact selected source names, call counts,
inclusive and exclusive wall durations, and all captured GPU intervals with a
deduplicated name table. Labels identify batch construction, model forward and
sampling. The reader recomputes category totals and interval unions. Summed
kernel durations can overlap. `Native.call` wall time includes GPU waiting.
These values do not measure bandwidth, hardware occupancy or CPU cycle counts.

The profile requires both ranks, all four windows, the bound worker/source
identities, actual GPU events and the KV4 reader/validator kernels. Missing or
unsupported GPU activity leaves the diagnostic incomplete. Collection failures
retain available partial records and use the existing workload cleanup path.
Every diagnostic result keeps `qualification=false`.

Each window is limited to 256 KiB and each rank's complete profile to 1 MiB.
The existing 2 MiB rank reader, 8 MiB arm result and V2 artifact transport limits
remain in force. Profiling overhead affects the selected operations and can
affect later timing. The offline reader labels the run as instrumented and
reports `OFFLINE_ATTRIBUTION_COMPLETE`; use the separate unprofiled run as the
performance baseline.

Run the focused CPU checks using the configured project interpreter:

```sh
make test-benchmark-profile PYTHON=/path/to/configured/python3
```

The fixtures substitute Torch, GPU and transport observations while exercising
the entry, source/result validators and controller acceptance. Native tracing
requires the fresh generated owner package and current device admission. The
human launcher performs privileged host observation; the model runs as UID 1000.
No profiler installation or change to device power settings is required by this
diagnostic mode.
