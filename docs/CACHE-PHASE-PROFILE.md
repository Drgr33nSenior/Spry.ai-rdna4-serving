# Candidate cache phase profile

This diagnostic separates the candidate cache attention call into three host-wall
phases. It does not change attention arithmetic. It does not time the decoder
alone, load a model, or measure serving throughput.

The previous ABBA experiment found a repeatable whole-call timing advantage for
the grouped reader. That result did not identify the expensive component. This
diagnostic tests that gap before another optimization.

## Measurement boundaries

| Field | Included operations |
| --- | --- |
| `host_checks_ns` | Descriptor, range, scale, alignment and alias checks; device and property lookup. |
| `validation_status_ns` | Existing status reset, validation kernel, launch check, stream synchronization and four-byte device-to-host status copy. |
| `reader_sync_ns` | Existing reader launch, launch check and final stream synchronization. |
| `total_ns` | Sum of the three phases, from entry to completion of the attention call body. |

All durations use `std::chrono::steady_clock`. The projection adds no HIP calls,
device buffer, kernel launch, device event, allocation or synchronization. The
reader phase includes the query transform, fused decoding, dot products, softmax
and accumulation. It is not isolated GPU execution time.

The existing outer `synchronized_call_ms` timer remains. It includes the complete
ABI call and the fixture's existing extra device synchronization. Profile reads
occur before and after this timer, never inside it. Each measured call must
advance the same-thread profile sequence exactly once. A missing export, stale
record, failed call, incomplete phase record or inconsistent duration rejects
the fixture. Partial failures are not timings.

## Source and experiment identity

`prototype/scripts/profile_cache_source.py` accepts only this uninstrumented
`prototype/runtime/attention.hip` SHA256:

```text
2206e70d233cacd86bea286be36c04a42d504e5807fe05985b2b2fa3319db88d
```

The projector produces a new private file and refuses overwrite. Its reverse
check must recover every original source byte. Do not replace the canonical
runtime file with the generated file. The small profile ABI is defined in
`prototype/runtime/cache_phase_profile.h`.

The explicit experiment mode is `native-cache-phase-profile-v1`. It retains the
`warm-synchronized-output-v2` protocol: two warmups, five measured repetitions,
lengths 128, 1024 and 3966, BF16 and compact KV4 cache arms, and both cards. Each
arm uses baseline/candidate/candidate/baseline order per card. There are 16 native
processes, 240 outer timing samples and 120 candidate phase samples. Each process
uses one card. This is not a TP2 model run or an FP4 weight experiment.

Only the candidate receives `--phase-profile`. The baseline stays byte-identical
to the accepted decoder/validator library. All retained outputs still require
bit-for-bit equality, including reference, warmup and timed outputs. Existing
refusal, device, source, resource and process-release checks remain required.

The new specification binds the original source, projected source, projector,
profile header, uninstrumented candidate provenance and three native binary
hashes. These are declared build inputs until a fresh compiler receipt verifies
the actual build. A specification hash alone does not prove binary provenance.

Legacy modes do not select profiling. The old benchmark analyzer refuses the
profile mode. `tools/analyze_cache_phases.py` requires a completed campaign,
accepted cleanup, both complete arms and the full native result reader. Archive
and source authentication remain separate prerequisites.

## Interpretation

Use per-sample phase shares and retain both candidate occurrences separately.
The phase analyzer does not calculate a baseline speedup. The candidate has
instrumentation and the baseline does not.

- A large validation phase directs the next investigation toward validation,
  its synchronization or status transfer. It does not isolate those operations.
- A large reader phase directs the next investigation toward the fused reader.
  It does not establish that unpacking, memory access or arithmetic is the cause.
- A large host phase points to host checks or device lookup. Native timing is
  still needed to distinguish them.

Host timestamps can affect scheduling and code layout. A disappearing bottleneck
is not a cure. Do not report this instrumented timing as native throughput,
memory bandwidth, a fourfold speedup, BF16 fidelity or model qualification.

## Verification and remaining native work

Local tests use explicit HIP, process and transport substitutes. They cover the
generated host function, phase ordering, original returns, ABI layout, source
recovery, fixture timing placement, stale records and full owner/reader
composition. They do not qualify HIP compilation, GPU behavior or Linux cleanup.

Before a physical attempt:

1. Compile the private projected source and updated fixture through the reviewed
   compiler-only path. Retain the full compiler input/output identities. Keep the
   accepted baseline binary unchanged. Do not install or promote the library.
2. Inspect the native artifacts and bind their hashes in a fresh private package.
   Bind the exact phase specification and final tested owner/reader bytes. Do not
   reuse a consumed package or infer that an old factory supports this mode.
3. Review the new package, then perform current admission through the existing
   human-run owner. Preserve its 3000-second envelope, 300-second seal reserve,
   resource controls, canonical lock and independent release checks.
4. Authenticate the retained result before running the phase analyzer. Keep
   `qualification=false`, including on success.

No physical grant, fresh native binary or launch-ready package is produced by
this local preparation. The next action is compiler-only verification and fresh
package construction, not a model or GPU launch.
