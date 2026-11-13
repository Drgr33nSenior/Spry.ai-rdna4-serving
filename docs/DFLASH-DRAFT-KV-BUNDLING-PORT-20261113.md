# DFlash draft KV bundling: port record, 13 November 2026

**Status:** ported and tested on the CPU only. Nothing here has run on a GPU, no
image contains it, and the serving profile is unchanged. A GPU qualification
(below) decides whether it goes live.

Patches `0028` to `0030` of the
[`sglang-main` series](../serving/patches/sglang-main/README.md) port two open
upstream SGLang PRs onto the pinned commit `fcb32746`:

- [#47057](https://github.com/sgl-project/sglang/pull/47057) (`69c84e12`):
  keep sliding-window (SWA) KV only for the window before each cached Mamba
  checkpoint;
- [#47058](https://github.com/sgl-project/sglang/pull/47058) (`1846b83e`):
  `--speculative-draft-kv-bundling`, which sizes the DFlash draft's KV pool by
  Mamba checkpoints instead of giving it one slot per target token.

This replaces the route in the
[11 November plan](DFLASH-BOUNDED-DRAFT-POOL-PLAN-20261111.md), as the owner
decided on 12 November.

## What it buys

The draft (`incoai/Qwen3.8-27B-DFlash2`: 5 layers, all `sliding_attention`,
window 2048) reads only the 2,047 tokens before the committed length. Today it
still gets a KV slot for every target token. With the flag it gets one window
per Mamba slot plus what running requests hold.

| Arm | Draft pool today | Draft pool with the flag | Freed per card | Same bytes as target KV |
| --- | --- | --- | --- | --- |
| UQS (`--max-total-tokens 163840`) | 163,840 tokens, 800 MiB | 43,008 tokens, 210 MiB | 590 MiB | 201,600 target tokens (1.23x), about 100,768 per conversation instead of 81,888 |
| UQL (`--max-total-tokens 524288`) | 524,288 tokens, 2,560 MiB | 43,008 tokens, 210 MiB | 2,350 MiB | 674,688 target tokens (1.29x); the context is already the model's maximum |

The byte figures use 16 KiB per target token and 5 KiB per draft token per
card, as the serving profile records them. They are arithmetic, not
measurements; the startup log of a GPU run is the check.

## Pool size formula

`DraftKVBundlePoolConfigurator` (`pool_configurator.py`) computes

```text
D = M*W' + max(0, C_run(R) - p*R*W', C_run(1) - p*W')
```

capped at one slot per target token. For the FP8 arms:

| Term | Meaning | Value | From |
| --- | --- | --- | --- |
| `W'` | Draft window, page-aligned | 2,047 | `sliding_window` 2048 less the current token; `--page-size 1` |
| `M` | Mamba slots | 19 | `--max-mamba-cache-size 19` |
| `R` | Running requests | 2 | `min(--max-running-requests 2, 19 // 3)`; 3 is the Mamba slots-per-request ratio with `no_buffer` |
| `p` | Mamba slots a running request holds outside the tree | 1 | 1 plus tracking slots, and `no_buffer` has none |
| per request | `W + interval*draft_tokens + page + verify block` | 3,080 | 2,047 + 128 x 8 + 1 + 8; `SGLANG_SWA_EVICTION_INTERVAL` defaults to 128, `--speculative-num-draft-tokens 8` |
| `C_run(2)` | Two requests plus one prefill chunk | 8,209 | 2 x 3,080 + 2,048 + 1; `--disable-overlap-schedule` keeps one chunk in flight |
| `C_run(1)` | One request plus one prefill chunk | 5,129 | 3,080 + 2,048 + 1 |
| `D` | Draft pool | **43,008** | 19 x 2,047 + max(0, 8,209 - 4,094, 5,129 - 2,047) = 38,893 + 4,115 |

`D` does not depend on `--max-total-tokens`: with that flag set, the target
keeps exactly the tokens it names and the draft pool is `D`. Each Mamba slot
costs 2,047 draft tokens, about 10 MiB per card.
`test_gfx1201_draft_kv_bundling.py` computes these numbers through the real
configurator.

## Requirements against the serving profile

Checked against `qwen3.8-27b-mxfp4-tp2.json` (arms UQS and UQL) by reading the
code. A GPU start is the confirmation: every unmet requirement raises at
startup.

| Requirement | Profile | Result |
| --- | --- | --- |
| `--speculative-algorithm` DFLASH or DSPARK | DFLASH | Met |
| CUDA or ROCm device, in-tree platform | ROCm, device `cuda` | Met |
| Target with Mamba-style layers | Qwen3.8-27B (GDN hybrid) | Met |
| Target without sliding-window layers | No SWA layers | Met |
| Draft with MHA attention and every layer `sliding_attention` | DFlash2: 5 of 5 | Met |
| Chunked prefill | `--chunked-prefill-size 2048` | Met |
| `UnifiedRadixCache`, without the session radix cache, external linker, LMCache or FlexKV | The default cache; none of those flags | Met |
| No unified memory, HiSparse, int8 Mamba checkpoints, PD disaggregation or DCP | None set | Met |
| No `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` | Not set | Met |
| No HiCache (this port only, see below) | `--enable-hierarchical-cache` is in `not_set` | Met |
| A particular `--mamba-radix-cache-strategy` | `no_buffer` | No requirement. `no_buffer` gives `p = 1`, the smallest pool. The PR's "buffer mode" exclusion is HiCache's `buffer_only` host mode |

No setting has to change except adding the flag and, to use the freed memory,
`--max-total-tokens` and `--context-length`.

## Build

- **Rust crate.** Both PRs change `rust/sglang-radix-tree`. The gfx1201 build
  does not compile it: `build.sh` copies `pyproject_other.toml`, whose
  `rust-extensions = ["multimodal"]` leaves the tree core out of the wheel.
  The wheel carries no Rust sources either, and PyTorch 2.16 is outside the
  crate's supported range (2.11 to 2.15). SGLang therefore selects the Python
  tree core (`_rust_fallback_reason`), which implements the same policy. The
  startup log line `Using the Python TreeCore: ...` confirms it.
- The Rust changes are kept in `0028` so the patch stays close to the PR. The
  crate builds and its native tests pass with them (below).
- **Series hash.** With `0028` to `0030` the hash is
  `40067ef9288c4cc26d98f1d006c1323806ffb7d0635fa078c27ff37a5868bf74`. The
  workstation's `stack.lock` still pins `142d0ae7...`, so the next build needs
  the lock updated first.
- **With the flag off**, the patches add only no-op hooks to the cache's
  eviction and value-set paths (by reading; not measured). A rebuilt image
  still needs the usual bench and quality run before it replaces the live one.

## How the port differs from the PRs

The PRs are based on SGLang of 29 October. The pin is ten days newer and has
reworked the KV id plumbing, so a few parts changed or dropped out.

| Area | Upstream PR | This port | Why |
| --- | --- | --- | --- |
| Eviction calls | `EvictParams.swa_num_tokens`, `EvictResult.swa_num_tokens_evicted` | `window_num_tokens={"swa": n}`, `window_num_tokens_evicted`, `EvictResult.add` | The pin names window sub-pools (#48007) |
| Draft write slots | `draft_kv_write_loc` helpers called from `dflash_worker_v2.py` and `dspark.py`; `SWAKVPool.set_kv_buffer_prefix_valid` | Left out | The pin already takes each write's window slots from the iteration's id plan (`_draft_write_ids(..., IdSpaceKind.SLIDING_WINDOW)`) and stores through the layer router, which asserts that a window-pool layer got window slots |
| DSPARK fused write | Translates the fused kernel's slot vector | Left out | At the pin the fused kernel is not used when a layer's KV is outside the owner pool; the per-layer path is used instead. DSPARK is not served here |
| Rust binding | Adds `swa_checkpoint_windows` | Adds it after the pin's `mamba_host_only_checkpoint` | Both arguments are needed |
| HiCache | Supported | Rejected at startup (`0030`) | The host-tier paths were rebased over the pin's batched write-back eviction but need GPU transfers to run |
| Tests | `test_hicache_auto_size.py` fixture | Fixture gains `storage`, `window_pool_names`, `window_retention` | The pin reads the retained window from the allocator |

## Kernel audit

The risk: `0029` gives the draft its own window-side pool (43,008 rows)
indexed by window slots, while target slots run to `--max-total-tokens`. Any
code that indexed the draft cache with a target slot would read or write
another request's rows, or past the buffer, without an error.

Every line the 27 earlier patches add was searched for KV buffers and slot ids
(`req_to_token`, `out_cache_loc`, `cache_loc`, `kv_indices`, buffer accessors).
The series computes no slot ids of its own; it has one site that reads them
directly.

| Site | Patch | How it indexes draft KV | Verdict |
| --- | --- | --- | --- |
| `_gfx115_verify_attention` -> `verify_attention_gfx1151` | 0012, 0021, 0026, 0027 | Gathers **target** slot ids straight from `req_to_token` | Safe, and now guarded twice. The draft block runs in target-verify mode on the draft's Triton backend, so this path is reachable in principle. Its caller already skips layers whose KV is in the window pool (and sliding-window layers); `0030` adds the same check inside the wrapper. The draft does not use this kernel today either (its layers have a sliding window), so its speed does not change |
| `extend_attention_fwd` (prefix read) | 0016, 0023, 0026, 0027 | Reads `kv_indices` it is given; the patches change tile, warps and the FP8 decode | Safe. For a sliding-window layer the backend passes `window_kv_indices`, which `update_sliding_window_buffer` builds through the pool's full-to-window table. This is the draft's read path |
| Grouped decode kernel | 0026, 0027 | Reads `kv_indices` it is given | Safe, and the draft never runs in decode mode |
| `gfx1201_fp8_kv.applies` / `decode` | 0026, 0027 | Look at buffer dtypes only | Safe. The bundled draft pool has the same class and dtype as today's, so FP8 draft KV is still read as bf16 |
| Fused draft KV write (`FusedKVMaterializeHelper`) | 0017 | Changes only how the stacked K/V weight is built | Safe. The helper's output is stored per layer through the layer router with both ids. Not used live |
| DFlash GDN replay commit | 0008 | Target Mamba state only | Safe. No draft KV. Not used live |
| DFlash context-KV writes (`_append_target_hidden_*`) | Upstream at the pin | `KVWriteLoc(cache_loc, swa_loc=...)` through `layer_router.layer(id).store` / `store_prefix` | Safe. A window-pool layer without `swa_loc` fails an assertion instead of using the target slot |
| Draft block's own K/V write | Upstream at the pin | `swa_out_cache_loc` from `RunnerIds.write_ids(..., SLIDING_WINDOW)`; in a captured graph, a buffer refilled each replay | Safe. A missing window write loc raises |

What makes the upstream paths take window slots is one fact: the draft
runner's `RunnerIds` has a sliding-window id space, because its pool is an
`SWAKVPool`. Without it, reads would fall back to target ids silently. `0030`
makes a bundled draft's Triton backend refuse to start in that case.

Not covered by reading or by CPU tests: the Triton gather kernel that applies
the table on the GPU (`create_flashinfer_kv_indices_triton` with
`token_mapping`). It is upstream code that hybrid-SWA models already use, but
this stack has not run it. The qualification plan checks it directly.

## Tests

All runs were on the workstation in containers with no GPU device and no
network (`nice -n 19`, `--cpus 12`), using the live image's Python environment
(`sglang-gfx1201:c4ed06e13d29`, PyTorch 2.16.0a0) with the patched tree on
`PYTHONPATH`. `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` in every run.

| Test | Result |
| --- | --- |
| `test/registered/unit/model_executor/test_pool_configurator.py` (includes the PR's configurator and flag-check cases) | 78 passed |
| `test/registered/unit/model_executor/test_gfx1201_draft_kv_bundling.py` (new, `0030`) | 10 passed |
| `test/registered/unit/mem_cache/test_hicache_auto_size.py` | 4 passed |
| `test/registered/unit/mem_cache/test_unified_radix_allocation_eviction.py` | 18 passed |
| `test/registered/unit/mem_cache/test_swa_unittest.py`, on the CPU runner | 20 passed, including the PR's draft-only SWA side case; 4 skipped and 1 error (`test_swa_only_frees_do_not_synchronize`), all needing CUDA |
| `.../unified_radix_cache/test_unified_radix_cache_checkpoint_window.py`, on the CPU runner | 18 cases (9 at each page size): the 5 device-only cases passed at page size 4, and 4 passed and 1 skipped at page size 1 (it needs the extra buffer); the 4 HiCache cases at each page size, including the randomized one, error because host transfers need CUDA |
| `.../unified_radix_cache/test_gfx1201_checkpoint_window_device_only.py` (new, `0030`), on the CPU runner | 2 passed: 6 seeds x 300 random device-only operations at page sizes 1 and 4 |
| `.../unified_radix_cache/test_unified_radix_cache_core.py`, on the CPU runner | 527 passed, 59 skipped |
| `rust/sglang-radix-tree`, `cargo test --locked` (Rust 1.92, PyTorch 2.15.0 CPU, separate container with network) | 965 passed, 1 ignored |
| `test/registered/attention/test_trtllm_mha_graph_metadata.py` | Not run: NVIDIA TRT-LLM backend, needs pytest and FlashInfer |
| `test/registered/spec/dflash/test_dflash_draft_kv_bundling.py` | Not run: needs a GPU and the models |

Upstream gates the unified-radix-cache fixtures on CUDA. The "CPU runner" is
[`tools/run_sglang_unit_on_cpu.py`](../tools/run_sglang_unit_on_cpu.py), which
clears that skip and points the fixtures at the CPU:

```sh
cd test/registered/unit/mem_cache/unified_radix_cache
PYTHONPATH=<sglang>/python:. SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
  python <serving-repo>/tools/run_sglang_unit_on_cpu.py test_gfx1201_checkpoint_window_device_only
```

The other files run as `python <file>` from their own directory. The Rust
crate test follows the crate's README.

What the new tests show:

- The slots the draft writes (`plan.write_ids` for the draft runner, then
  `layer_router.layer(id).store`) are the allocator's mapped window slots, in a
  setup where target and window ids differ and target ids run past the window
  pool. No other row of the pool changes.
- A write without window slots raises.
- The draft's read translation (`ids_in`, and the table the gather kernel is
  given) is the same mapping object, including after `allocator.clear()`.
- The gfx1151 verify wrapper returns before reading for a window-pool layer.
- The draft pool is 43,008 tokens for the serving arms and does not grow with
  `--max-total-tokens`.
- Under random inserts, evictions and locks, every cached Mamba checkpoint
  keeps the SWA KV of its window. The same test fails when the policy is
  switched off, so it does detect a lost window.

## Unresolved

1. **No GPU run.** Startup, the pool sizes in the log, the Triton gather with
   the mapping table, captured decode graphs for batch sizes 1 and 2, and FP8
   draft KV on the window pool are all unverified on gfx1201.
2. **Admission under load.** The scheduler now treats the stack as hybrid SWA.
   Whether two long conversations at the new context limit are admitted without
   stalls or draft-pool exhaustion at the pin is untested. A full pool should
   fail an allocation, not corrupt memory.
3. **Prefix resumes.** A cached prefix now resumes only where a Mamba
   checkpoint still has its window. Radix reuse on long conversations may fall
   back to an earlier checkpoint more often than today; the bench's cache
   report measures it.
4. **HiCache** with the flag is closed in this port. It is not used here.
5. **Upstream may change the PRs** before merging. Drop `0028` to `0030` for
   the merged version when the pin moves past it; keep the two `0030` test
   files if they still apply.
6. **KV usage metrics** will include the draft pool, as the PR notes.

## GPU qualification plan

For a maintenance window. One server holds both GPUs at a time, so the live
service is down throughout. Build one image from the 30-patch series (after
updating `stack.lock`) and run every step on that image, at the configured
235 W cap.

Add two arms to the profile for the run; do not change `UQS`:

- `UQSB0`: `UQS` plus `--speculative-draft-kv-bundling`, same
  `--max-total-tokens 163840` and `--context-length 81888`. It isolates the flag.
- `UQSB`: `UQSB0` with `--max-total-tokens 201600` and
  `--context-length 100768`. The proposed shared-mode arm.

| Step | Arm | What to record | Gate |
| --- | --- | --- | --- |
| 1. Baseline on the new image | `UQS` | Bench Job: `BENCH_RESULT`, `QUALITY_RESULT`, long-context phase; VRAM per card after startup | Matches the 11 November `UQS` qualification within run-to-run spread. This separates the rebuild from the flag |
| 2. Startup | `UQSB0` | The startup log's `Draft KV bundles:` line and the KV pool lines; `Using the Python TreeCore`; VRAM per card after startup | Draft pool 43,008 tokens and about 210 MiB per card; target pool 163,840 tokens; free VRAM about 0.58 GiB per card above step 1 |
| 3. Slot check | `UQSB0` | In a dev pod or a one-off script on the bundled server: for a few live requests, compare the draft backend's `window_kv_indices` with `full_to_swa_index_mapping[req_to_token[row, seq_len-W:seq_len]]`, and check their maximum is below the draft pool size | Equal for every request, eager and under the captured graph |
| 4. Exactness | `UQSB0` against step 1 | Bench with `--greedy`, twice on `UQS` and once on `UQSB0`: per-request output text and accept lengths | The flag changes where draft KV is stored, not its content, so the draft should propose the same tokens. If the two `UQS` runs repeat each other exactly, `UQSB0` must match them on fresh prompts. A difference means the draft read other KV, and stops the run. If `UQS` does not repeat itself, fall back to step 5's gate |
| 5. Accept length | `UQSB0` against step 1 | DFlash accept length (tokens per verify) from `BENCH_RESULT`: fresh prompts, and separately the radix-hit follow-up turns | Fresh prompts within 3% of step 1. Report the resumed-turn figure; a drop there is unresolved item 3 |
| 6. Speed | `UQSB0` against step 1 | C1 and C2 tokens/s, ms per verify step at 32K, 64K and 120K, prefill tokens/s at 2K, 8K and 16K | Within run-to-run spread. The draft's writes and reads gain one table lookup |
| 7. Quality | `UQSB0` | `QUALITY_RESULT` | Same pass, fail and NOT RUN counts as step 1 |
| 8. Larger pool | `UQSB` | Steps 2, 5, 6 and 7 again; the startup log's pool lines; VRAM per card after startup and at peak | Target pool 201,600 tokens; free VRAM per card after startup not below step 1 (about 9 GiB, which ComfyUI needs) |
| 9. Two long conversations | `UQSB` | Two concurrent requests near 100,768 tokens each, then a follow-up turn on each; watch for allocation failures, aborted requests and stalls; cached-token counts from the cache report | Both complete; the follow-up turns reuse their prefix |
| 10. Shared-mode peak | `UQSB` | VRAM per card with both image generators active, as in the `UQS` qualification | No out-of-memory; SGLang's growth after startup stays in the 0.6 to 0.8 GiB measured before |

If steps 2 to 7 pass and step 8 or 9 does not, `UQSB0` is still a valid
result: the same context as today with about 0.58 GiB more free VRAM per card.

For `UQL`, the same flag frees about 2.3 GiB per card at the model's full
context. Qualify it after `UQSB`, as its own arm, by repeating steps 2 and 5 to
9 at `--context-length 262112`.

Propose for shared mode, if the plan passes: `--speculative-draft-kv-bundling`,
`--max-total-tokens 201600`, `--context-length 100768`.

## Worth telling upstream

Nothing was posted. Two observations for the PR authors, if the owner wants to
pass them on:

- #47058 patches only the TRT-LLM backend's encoder-only verify path. On the
  Triton backend the gfx1151 verify kernel reads `req_to_token` directly; it is
  safe only because its call site excludes window-pool layers. A check inside
  `_gfx115_verify_attention` (as in `0030`) would keep it safe if that call
  site changes.
- After #48007, the PR's `draft_kv_write_loc` helpers duplicate what
  `IterationIds.write_ids(..., SLIDING_WINDOW)` already provides, and the
  eviction calls need the window sub-pool names. The rebase in `0028` and
  `0029` shows one way through.
