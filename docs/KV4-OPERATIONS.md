# KV4 benchmark operator guide

This guide covers the matched ModelRunner benchmark. It does not authorize a run.
Status as of 2026-10-15: `LOCAL_MATCHED_SOURCE_PREPARED_NATIVE_GATES_PENDING`.
The KV4 preparation is source-only. Public KV4 admission remains closed.

The benchmark controller modules and the `HOST-OWNER.md` and
`OWNER-COLLECTION.md` procedures cited below are not included in this
documentation-only publication. Obtain reviewed, hash-bound copies with the
physical handoff. Missing procedures or executable inputs block a run.

Both arms use official **FP8 model weights**. The first arm uses BF16 KV storage;
the second uses the mixed KV4 implementation. There is no accepted FP4-weight
checkpoint in this package. Do not describe this as an FP4-versus-FP8 weight test.

## 1. Check readiness

The operator must receive a fresh, reviewed physical package and its exact launch
procedure before execution. The published checkout and source-only archive are
not substitutes for that package or the complete installed runtime.

| Gate | Current position | Required before launch |
| --- | --- | --- |
| Frozen workload | Prepared and hash-bound | Verify the specification and exact request vectors |
| KV4 CPU checks | 15 focused checks passed in the preparation | Bind final package bytes and retain applicable checks |
| Native coverage | Earlier small TP2 smoke only | Do not claim that all benchmark cells already pass |
| Loading report | Dedicated interval missing | Add agreed construction/loading and pool/backend-setup timers outside inference timing |
| Physical package | Not assembled for this KV4 preparation | Bind current boot, source, runtime, model, observer and controller identities |
| GPU ownership | No current handover recorded | Obtain an explicit two-device window from the FP8 lead, then pass locked admission |
| Execution authority | Not granted by preparation | Obtain approval for the reviewed package and complete finite envelope |

The FP8 lead must own any matching loading-instrumentation changes to its package.
Do not modify or execute an FP8 package to complete KV4 preparation. Confirm the
cache-only comparison with the owner before physical packaging.

There is no current copy-paste KV4 launch command. In particular:

- `prepare_benchmark_package.py` is a factory module, not a preparation CLI.
  Direct invocation exits `NOT_RUN`.
- `benchmark_host_transport.py` is a physical launcher, not a validation command.
- Do not reuse a consumed wrapper, edit readiness flags, or run a sealed
  controller with its old clock.

The shared FP8 history includes a physical attempt whose BF16 inner arm finished,
but resource and release checks refused completion. KV4 did not start in that
attempt. It is neither a successful pair nor current release evidence. See the
retained failure section in the supplied `HOST-OWNER.md`.

## 2. Verify the handoff

The current local preparation is
`kv4-matched-benchmark-prep-20261015.70I2s7`. Its evidence is private and is not
included in this checkout. The operator must obtain the directory from the lead.

| Input | Expected identity |
| --- | --- |
| Preparation manifest | `91625c1560b7764196011dbc291e453b5c511c4ceb2a6f728bc776565786b4ec`, 132 regular members |
| `payload/review/BENCHMARK-SPEC2.json` | `67bd5c3188c1bb3af92d8b30f6f4b8e472ad2c8c685a9aa116de6e3f131a4f5a` |
| Selected native library | `d09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171` |

For a local read-only hash check, replace the example path with the supplied
absolute evidence path. Run these commands from a normal user shell:

```sh
KV4_PREP_ROOT=/absolute/path/to/kv4-matched-benchmark-prep-20261015.70I2s7
shasum -a 256 "$KV4_PREP_ROOT/FINAL-ARTIFACTS.sha256"
shasum -a 256 "$KV4_PREP_ROOT/payload/review/BENCHMARK-SPEC2.json"
(cd "$KV4_PREP_ROOT" && shasum -a 256 -c FINAL-ARTIFACTS.sha256)
```

Compare the first two digests with the table. Stop on any mismatch. A successful
`shasum -c` checks listed file bytes only. The reviewed package verification must
also check exact membership, unique safe names, no links, and no entries newer
than the manifest, including the root. Do not add outputs inside a sealed root.

Read `HANDOFF.md`, `COMPARABILITY.md`, `COMPARABILITY-IDENTITIES.json`,
`REQUEST-CONTRACT.json`, `MODEL-INVENTORY.json`, `RUNTIME-INVENTORY.json`,
`TESTED-HASHES.json`, and `OUTPUT-AND-ANALYSIS.md`. A source archive is not a
physical-controller archive. The preparation's historical boot placeholder is
not valid for current admission.

The reviewed model is `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Use the supplied inventories for exact
model, tokenizer, image, source, compiler and binary hashes. Do not substitute a
model with the same display name. The selected SGLang 0.5.21/ROCm 10.1 composition
is described in [build guidance](BUILD.md); this checkout does not install it.

## 3. Preserve the workload

The specification is authoritative. These values are a check, not permission to
regenerate nominally equivalent prompts or token vectors.
For this preparation, use this frozen contract rather than the earlier proposal
in the published `BENCHMARKING.md`.

| Setting | Required value |
| --- | --- |
| Arms | Fresh BF16-KV load first; fresh KV4 load second |
| Devices | TP2 on both R9700 GPUs |
| Execution | Eager; no overlap, speculation, MTP or offload |
| Context and pool | Context 4096; 4096 allocatable physical slots plus reserved slot zero |
| Pool policy | Page size 1; memory fraction 0.80; `no_buffer`; chunk setting `-1` |
| Cell order | C1-128, C1-1024, C1-3966, C2-128, C2-1024 |
| Per cell | One warm batch, then five measured batches |
| Per request | 32 generated tokens; temperature 0; `ignore_eos=True` |
| Seed | 592137615 |
| Totals | 30 batches and 42 requests per arm; 84 requests across both arms |

The two arms generate 2688 tokens in total, including warm-up outputs. A batch
contains one prefill and 31 decode forwards. Preserve the exact request IDs,
sampling defaults, vectors and ordering. Use `(session, arm, request ID)` to
distinguish evidence without changing the frozen IDs.

Do not add hidden warm-ups, replace a failed sample, drop a slow sample, or change
a cell to fit available memory. Report a setting mismatch before proposing a
different experiment. Keep warm-up, partial and failed records.

## 4. Arrange ownership and admission

The KV4 lead supplies the reviewed package and finite proposal. The FP8 lead
confirms an exclusive window for both GPUs. The authorized human operator starts
only the exact wrapper in the new handoff. The protected observer supplies
current admission and independent release evidence.

Use the shared canonical lease at
`/home/developer/model-serving.izR2m1O9/diagnostic.lock`. An unresolved
`diagnostic.lock.unresolved.json` blocks admission. Do not create a second KV4
lock, delete an unresolved record, stop another workload, or infer ownership from
idle GPUs or historical release.

The device handover must name BDFs `0000:c3:00.0` and `0000:23:00.0` and match the
package's device identities. Pass the current source/runtime/model check and
locked admission before model startup. Root visibility alone is not sufficient:
the observer must pass its host-namespace and identity checks.

If current platform evidence is missing, follow the supplied
`OWNER-COLLECTION.md` procedure. That read-only collection
does not reserve devices or authorize a model run. Agents must not execute its
privileged wrapper. Do not run an ordinary user-writable script as root.

## 5. Use the finite execution envelope

The current proposal is **2400 seconds**, not an active grant:

| Phase | Maximum seconds |
| --- | ---: |
| Source, runtime and model preflight | 105 |
| Locked admission | 75 |
| BF16-KV arm | 795 |
| KV4 arm | 795 |
| Collection and independent release checks | 330 |
| Final seal and verification | 300 |

Each 795-second arm contains 690 seconds of work, 30 seconds of owned cleanup
(20 TERM, 5 KILL, 5 reap), and 75 seconds of release observation. This is the
split implemented by `benchmark_owner_adapter.py`. It corrects the 15/90 cleanup
and observation prose split in the sealed preparation's `EXECUTION-PROPOSAL.md`;
the total remains unchanged. The sealed document is not amended.

Work includes identity checks, model construction, allocation, warm-ups, measured
requests and worker teardown. Refuse startup if a complete remaining envelope
does not fit. Do not build, install, download or repair sources during a run.
The second arm starts only after the first arm succeeds and release is verified.

Preserve the reviewed resource limits: 38 GiB scoped RAM, no swap, 512 tasks,
24-CPU cgroup quota and configurable application pools capped at two. Preserve
the 46 GiB host-memory admission floor, 8 GiB runtime stop floor and 64 GiB disk
admission requirement. Verify effective limits and events, not only requested
settings. These are limits, not measured peaks.

Private scratch totals 3.5 GiB: 2.75 GiB output tmpfs, 256 MiB shared memory and
512 MiB temporary storage. The durable evidence bound is 458.75 MiB within a
512 MiB partition. Do not enlarge these bounds to rescue a cell. Keep existing
runtime files and caches read-only; use the package's fresh private write paths.

## 6. Stop safely and retain failures

Monitor the reviewed wrapper's status and bounded logs. A timeout, resource
refusal, identity mismatch, invalid sample or uncertain release ends the attempt.
There is no automatic retry or replacement arm.

For an operator interruption, use the reviewed wrapper's signal handling and
wait for bounded owned cleanup. Do not use broad process-name kills or signal a
numeric PID obtained from an old log. Cleanup must authenticate this session's
process identities before TERM/KILL and reaping. An interrupted shell or exited
worker does not prove GPU release.

| Observation | Operator action |
| --- | --- |
| Hash, boot, package or inventory mismatch | Stop before launch; return the mismatch to the lead |
| Existing lease or unresolved record | Stop; request owner reconciliation; do not remove the record |
| Missing host visibility or resource-limit proof | Stop; use the approved collection/review path |
| Timeout, output overflow or invalid sample | Retain partial outputs and cleanup receipts; do not replace samples |
| BF16 arm or its release check fails | Do not start KV4 |
| Processes, namespaces, cgroups or device allocations remain uncertain | Keep the unresolved journal and report release as unverified |

Independent release must check owned process groups, namespaces, cgroups, GPU
clients and allocations. Keep the full ownership transition history. Do not
treat unprivileged holder-query silence as a complete release check. See the
supplied `HOST-OWNER.md` for the authentication and refusal boundaries.

## 7. Collect and interpret results

Retain these five exported files for each arm:

```text
rank-0.json
rank-1.json
result.json
rank-0-identity.json
rank-1-identity.json
```

Also retain the specification, source/model/runtime identities, outer stdout and
stderr, admission/resource observations, full owner journal, cleanup receipts and
independent release report. In-namespace rank logs and intent files are not among
the five exported files; do not promise that ephemeral files survive teardown.

Check exact counts, 32 outputs per request, actual concurrency, finite logits,
TP agreement and restored pool capacities. Preserve all valid timings, including
slow ones. A complete inner `result.json` alone cannot establish campaign success.

Inference timing starts after GPU synchronization and includes
`ForwardBatch.init_new`, `ModelRunner.forward`, sampling and the final GPU
synchronization. Finite checks, TP agreement, result extraction and cache reset
are outside that interval. Use
the maximum rank time for each operation. Report prefill separately from the sum
of 31 decode operations. Decode rate is `31 * concurrency / decode_seconds`.

Report the median, minimum and maximum of all five measured batches per cell.
Keep warm-up results separate. A latency ratio of
`100 * (BF16_median / KV4_median - 1)` is not the same as percentage latency
reduction. Compare selected scores only for matching tokens with matching prior
history, stopping at the first generated-token divergence.

Report model construction/loading and pool/backend setup separately. The current
preparation still needs those timers; do not infer loading time by subtracting
inference from arm wall time. File-hash preflight warms the filesystem cache.
Disclose this, fresh per-arm JIT caches and compilation placement. Keep any JIT
work inside a measured interval in that sample; do not hide it with extra warm-ups.

Seal raw evidence before analysis. Write derived reports outside the sealed tree.
Use the hash-bound `analyze_model_benchmark.py` supplied with the physical handoff
for offline analysis. It checks inner results, not device release. Exchange sealed
results with the FP8 lead only after each campaign's validity and independent
release checks pass.

Both preparations currently bind the same FP8-weight BF16-KV/KV4 pair. If their
runtime, source, kernels or configuration later differ, list those differences
and compare complete implementations. Do not infer HTTP TTFT, production
throughput, general quality, numerical equivalence or an overall winner from
this fixed ModelRunner screen.

## 8. Handoff

Return the status, all failures, package and tested hashes, raw records, effective
resource limits and observed peaks, release result, manifest digest/member count,
verified completion time and receipt-to-seal charge. Keep launch readiness,
sample validity and physical release as separate conclusions. Do not promote
sources or start another campaign automatically.
