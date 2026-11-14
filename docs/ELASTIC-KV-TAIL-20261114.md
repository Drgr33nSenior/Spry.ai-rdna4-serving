# Elastic KV tail: Phase B1 record, 14 November 2026

**Status:** built and tested on the CPU only. Nothing here has run on a GPU, no
image contains it, and the serving profile is unchanged. Patches `0031` to
`0033` of the [`sglang-main` series](../serving/patches/sglang-main/README.md)
hold the work. It is off unless one environment switch is set.

This is Phase B1 of the plan the owner approved on 14 November: let the shared
mode's KV pool borrow GPU memory while the picture and video generators are
idle, and give it back before they run. Gate B0 passed the same day (below).
Phase B2 is a GPU window; its first checks are listed at the end.

## What it does

Shared mode serves from a 201,600-token KV pool (arm UQSB) because the video
job needs the rest of the card for about 80 seconds at a time. With the switch
on, the server starts with the large arm's pool size and context, but only the
first 201,600 tokens of every target KV buffer have memory behind them:

| | Tail not backed | Tail backed |
| --- | --- | --- |
| Pool | 201,600 tokens (the base) | Base plus up to 443,958 tail tokens |
| Context the server accepts and advertises | 100,768 | 100,768 plus half the backed tail, at most 262,112 |
| GPU memory | As UQSB, plus a few MiB of larger index tables | 16 KiB more per tail token per card |

A `grow` request maps the tail; a `shrink` request empties and unmaps it. The
draft KV pool stays at 43,008 tokens (`--speculative-draft-kv-bundling`, which
the feature requires).

## Switching it on

One switch, `SGLANG_GFX1201_ELASTIC_KV_BASE_TOKENS`, off when unset or `0`. It
needs one companion value. An elastic arm is the large arm's sizes plus:

| Setting | Value | Meaning |
| --- | --- | --- |
| `--max-total-tokens` | 645558 | The reserved pool: base plus the largest tail |
| `--context-length` | 262112 | The context with the whole tail backed |
| `SGLANG_GFX1201_ELASTIC_KV_BASE_TOKENS` | 201600 | Tokens that are always backed |
| `SGLANG_GFX1201_ELASTIC_KV_BASE_CONTEXT` | 100768 | Context while the tail is not backed |

SGLang still lowers `--max-total-tokens` to what it measures as free at
startup. If it does, the reserved tail is smaller; `GET /kv_tail` reports
`reserved_tail_tokens`.

The server refuses to start with the switch on when it cannot support it: no
draft bundling, a page size other than 1, HiCache, the Rust tree core, DP or
PP, a quantized or non-NHD KV layout, or a pool no larger than the base.

With the switch off, the patches add a few attribute checks (one per
scheduler iteration, one per pool construction, one each in the allocator's
`alloc`, `free` and `available_size`); `TokenToKVPoolAllocator` takes the same
decisions (a test replays random traffic against a copy of the upstream
allocator) and KV buffers are allocated as before.

## Design

### Buffers

SGLang's own arena (`kv_vmm_backing.py`, `cuda_vmm_utils.py`) was read first.
It gives PyTorch a tensor at a fixed address by wrapping a JIT-compiled C++
bump allocator in `torch.cuda.MemPool` and calling `torch.empty` under it, and
it drives CUDA through `cuda.bindings`. Neither part is usable here as it is:
the driver calls are CUDA-only, and `MemPool` under memory pressure on ROCm is
unproven (see the 13 November report). The same file has a second mechanism,
`tensor_from_pointer` (`torch._C._construct_storage_from_data_pointer`: a
storage that does not own its memory), which upstream uses for other
VMM-mapped tensors. The tail uses that, the arena's `KvBufferDesc` byte-span
arithmetic, and HIP through `ctypes`. The serving image's PyTorch
(2.16.0a0, ROCm 10.3) has the function; a CPU test builds a tensor with it.
**Whether it works for a HIP device pointer is the first thing B2 must show.**

Per target KV buffer (16 K and 16 V buffers per rank on the serving arms):

- one `hipMemAddressReserve` range for the whole buffer;
- one base extent, mapped at startup and zeroed, covering slots 0 to the base
  and rounded up to the device granularity;
- one tail extent, from the end of the base extent, mapped by `grow` and
  unmapped by `shrink`. A partial grow maps a shorter extent.

`grow` writes zeros over the whole new extent and reads the first and last
rows back before it enables any slot. Gate B0 saw `hipMemCreate` return in
0.02 ms for 4 GiB, so the memory is probably committed on first touch; the
zero-fill makes a shortage fail inside `grow`, where it can be refused, and not
under a chat.

### Allocator

`TokenToKVPoolAllocator` keeps base slots and tail slots in two free lists. It
hands out a tail slot only while the tail is open and no base slot is free, so
the tail holds what did not fit below it. A count of tail slots in use tells
the shrink when the tail is empty. Freed slots are sorted into the two lists
only while that count is not zero, so a server whose tail is unused or
unbacked pays nothing on the free path. While tail slots are in use, each free
reads the slot ids on the GPU (one synchronisation); B2 measures what that
costs in tokens per second.

The pool, the allocator and every table indexed by slot id are built at the
reserved size, so no table is too small for a tail slot. What changes with the
tail is what the allocator will hand out. The capacity reported to the pool
statistics and the idle leak check is the reserved size less the slots that
are out of circulation.

### Shrink

In this order, on every TP rank at the same scheduler step:

1. Lower the limits to the base for new requests. A running or queued request
   that is still within the base is capped at it, as it would be with the
   feature off. A request already past the base is left alone, and the tail
   stays open until it finishes: a chat is never preempted.
2. Stop handing out tail slots.
3. Evict every idle prefix-cache entry that holds a tail slot, with the
   entries below it in the tree. Their Mamba checkpoints and draft windows go
   with them through the cache's own leaf eviction. Prefixes that sit wholly
   in the base stay cached.
4. Wait for running requests that hold tail slots, rescanning the cache as
   they finish.
5. Clear stale slot ids above the base from the request table, synchronise the
   device, unmap and release each tail extent.
6. Wait until the device reports the released memory free (below).
7. Acknowledge.

If step 1 or 4 outlasts `timeout_s`, the tail is reopened, the limits go back
up, and the answer is `ok: false`, `reason: "timeout"`, `state: "backed"`.

### Release is confirmed, not assumed

Gate B0 found that after `hipMemRelease` and a device synchronise the card's
`mem_info_vram_used` still counted the released extent in about a third of the
rounds, and dropped later. The shrink therefore does not acknowledge when
unmap returns. Each rank reads the card's used memory before the unmap and
polls it (every 20 ms, from the scheduler loop, so chat keeps running) until
it has fallen by the released bytes less a slack of 64 MiB. The source is the
card's sysfs counters when the process can read them, otherwise
`hipMemGetInfo`; the answer names it in `vram_source`. If any rank has not
seen the memory free after `release_timeout_s` (default 10 s), the answer is
`ok: false`, `reason: "release_unconfirmed"`, `vram_confirmed: false`. The tail
is unmapped by then, so the state is `unbacked`; the caller must not start the
media job on that answer.

### Grow

Every rank checks free device memory against the bytes needed plus
`keep_free_mib`; if any rank is short, nothing is mapped. Then every rank
maps and zero-fills. If any rank fails, the ranks that succeeded unmap again
and the answer is `ok: false`, `reason: "map_failed"`. Only then are the tail
slots enabled and the limits raised.

### Ranks stay identical

The control request reaches every rank through the scheduler's request
broadcast, like `release_memory_occupation`. Whatever a rank measures for
itself (free memory, a map error, the release poll, the clock for timeouts) is
exchanged with `all_gather_object` on the TP CPU group at that step, and every
rank acts on the same gathered reports; timeouts follow rank 0's clock. The
cache scan is triggered by counts of scheduler steps and cache sizes, never by
time. The acknowledgement carries `ranks_identical`, false if the ranks
reported different eviction totals or slot counts while draining.

### Limits and the advertised context

With `N` tail tokens backed, the pool is `base + N` and the context is
`min(--context-length, base context + N // max running requests)`. For the
serving arm: 100,768 unbacked, 170,768 with 140,000 tail tokens, 262,112 with
the whole tail. The scheduler's request limits and the tokenizer's context
length (which `/v1/models` reports) follow this. The tokenizer lowers its
context before it forwards a shrink and raises it only after a grow is
acknowledged.

**The gateway.** The workstation's gateway holds a fixed per-conversation
limit from the profile. Unless it changes, conversations stay at 100,768
tokens and a backed tail only adds cache room (fewer prefixes evicted and
prefilled again). To let conversations grow, the gateway has to take its limit
from `context_length` in the `/kv_tail` answers, and drop it when it asks for a
shrink. Nothing in the workstation repository was changed.

Not updated when the tail changes: the scheduler's Prometheus capacity gauges
taken at startup (they show the reserved size), and the multimodal
processors' input limit (it stays at the base).

## Control contract

`POST /kv_tail` with a JSON body, or `GET /kv_tail` for the status. The
endpoint has the same admin authentication level as
`/release_memory_occupation`.

| Request | Fields | Answer |
| --- | --- | --- |
| `{"action": "status"}` or `GET` | none | At once, also while a shrink waits |
| `{"action": "grow"}` | `tail_tokens` (default: the whole reserved tail), `keep_free_mib` (default 1536) | When every rank has mapped, or none has |
| `{"action": "shrink"}` | `timeout_s` (default 300), `release_timeout_s` (default 10) | When every rank has unmapped and seen the memory free, or on a timeout |

HTTP status: 200 when `ok` is true, 400 for a malformed request, 409 when the
request was understood and did not happen.

Every answer has:

| Field | Meaning |
| --- | --- |
| `ok` | The request did what it says |
| `state` | `disabled`, `unbacked`, `backed`, `draining` or `releasing` |
| `message`, `reason` | Text; and on failure one of `busy`, `timeout`, `release_unconfirmed`, `insufficient_vram`, `map_failed`, `already_backed`, `bad_request`, `disabled` |
| `tail_tokens`, `reserved_tail_tokens`, `base_tokens`, `pool_tokens` | Sizes in tokens |
| `context_length`, `base_context_length`, `max_context_length` | What the server accepts now, and its bounds |
| `tail_bytes`, `tail_slots_in_use`, `backend` | Per-rank detail from rank 0 |

A shrink answer adds `waited` (true if it could not finish in the scheduler
step it arrived in), `waited_s`, `waited_for` (`long_requests`,
`tail_slots_in_use`), `evicted_tokens`, `released_bytes`, `vram_confirmed`,
`vram_release_s`, `vram_freed_bytes_by_rank`, `vram_source` and
`ranks_identical`. A grow answer adds `mapped_bytes` and `map_s`. While a
shrink is pending, the status has a `pending` object: `elapsed_s`,
`waiting_for_long_requests`, `waiting_for_tail_slots`,
`waiting_for_vram_release`.

Rules for the caller:

- `shrink` and `grow` are idempotent: a shrink of an unbacked tail and a grow
  to the size already backed answer `ok: true`.
- Start a media job only on a shrink answer with `ok: true`. A 409 with
  `reason: "timeout"` means a chat still needs the tail; ask again or refuse
  the job.
- While a shrink waits, a second `shrink` or `grow` is queued behind it by the
  HTTP server; `status` is not.
- After the job: free the generator's memory, wait for it, then `grow`. A grow
  that answers `insufficient_vram` changed nothing.

## Upstream files touched

Seven existing SGLang files, under the limit of about ten the plan set:

| File | Change |
| --- | --- |
| `mem_cache/allocator/token.py` | The two free lists and the gate |
| `mem_cache/allocator/swa.py` | Reported capacity excludes slots out of circulation |
| `mem_cache/memory_pool.py` | The target pool allocates through the tail owner |
| `managers/io_struct.py` | Four request and answer types |
| `managers/tokenizer_control_mixin.py` | Two communicators, the calls, the context length |
| `managers/scheduler.py` | Install, two dispatcher entries, one call per iteration |
| `entrypoints/http_server.py` | `/kv_tail` |

Two new files: `mem_cache/elastic_kv_tail.py` and the test module.

## Verify kernel

The gfx1151 target-verify kernel (patches 0012, 0021, 0026, 0027) takes a
layer only if `covered()` holds: contiguous K and V of equal shape, and
`k_buffer.numel() < 2**31` because it addresses the cache with int32 offsets.
At the reserved size a buffer is 645,559 rows of 2 kv heads by 256, which is
330,526,208 elements: the same as the large arm that is live today, and well
inside the bound. A tail buffer is one contiguous tensor at a page-aligned
address, so the kernel's 16-byte load hints hold. A test calls the real
`covered()` with tensors of that shape. The fast path is kept.

## Tests

All on the workstation in containers with no GPU device and no network
(`nice -n 19`, `--cpus 8`), in the Python environment of the image
`sglang-gfx1201:4f41979134f3` with the patched tree on `PYTHONPATH` and
`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`. Device memory is
`FakeVmmBackend`: host memory with map bookkeeping, fresh maps filled with
0xA5 and unmapped extents with 0xEE, and an optional lag before released
bytes show as free.

| Test | Baseline (patches 0001 to 0030) | With 0031 to 0033 |
| --- | --- | --- |
| `unified_radix_cache/test_gfx1201_elastic_kv_tail.py` (new), on the CPU runner | Not present | 43 passed |
| `unit/model_executor/test_pool_configurator.py` | 78 passed | 78 passed |
| `unit/model_executor/test_gfx1201_draft_kv_bundling.py` | 10 passed | 10 passed |
| `unit/mem_cache/test_hicache_auto_size.py` | 4 passed | 4 passed |
| `unit/mem_cache/test_unified_radix_allocation_eviction.py` | 18 passed | 18 passed |
| `unit/managers/test_msgpack_ipc_roundtrip.py` | 13 passed | 13 passed |
| `unit/managers/test_prefill_adder.py` | 56 passed | 56 passed |
| `unit/managers/test_scheduler_flush_cache.py` | 7 passed | 7 passed |
| `unit/managers/test_scheduler_init_req_max_new_tokens.py` | 10 passed | 10 passed |
| `unit/managers/test_scheduler_pause_generation.py` | 26 passed | 26 passed |
| `unit/managers/test_scheduler_timeouts.py` | 14 passed | 14 passed |
| `unit/managers/test_tokenizer_config_updates.py` | 10 passed | 10 passed |
| `unit/mem_cache/test_swa_unittest.py`, on the CPU runner | 20 passed, 4 skipped, 1 error (needs CUDA) | The same |
| `unified_radix_cache/test_unified_radix_cache_checkpoint_window.py`, on the CPU runner | Device-only cases pass; the HiCache cases fail or error (they need CUDA): 4 failures, 14 errors, 1 skipped, counting subtests | The same |
| `unified_radix_cache/test_gfx1201_checkpoint_window_device_only.py`, on the CPU runner | 2 passed | 2 passed |
| `unified_radix_cache/test_unified_radix_cache_core.py`, on the CPU runner | 527 passed, 59 skipped | 527 passed, 59 skipped |

The baseline column is the same run on a tree with patches 0001 to 0030 only,
so the two CUDA-only results are not caused by this work.

Also checked:

- **Series.** All 33 patches apply in name order to `fcb32746` with
  `git apply`, and the result is byte-identical to the tree the tests ran on.
  The series hash is
  `bfd43a2ed4eb66f811fe7fc55c731728e1704dc2da6d012a8473af7efca017e4`
  (`40067ef9...` for the first 30, unchanged). `stack.lock` was not touched.
- **Mutations.** Five deliberate breaks of the new code (freed tail slots
  re-enter circulation, no eviction of tail holders, release assumed instead
  of confirmed, a closed tail still handed out, descendants of tail holders
  kept) each fail the new test module.
- **This repository's own tests.** `make -k test` on macOS: every target
  passes except `test-serving-cleanup`, which stops at
  `serving/tests/test_worker_timing.py`
  (`test_pins_and_separate_source_patch_are_exact`). That test fails the same
  way on an untouched checkout of `main`; it belongs to the 0.5.21 runtime's
  records.
- **Not run:** `ruff format` and `isort` on the patched SGLang files (neither
  is in the image or on the Mac). Import order and line length were checked by
  hand.

The new module is
`test/registered/unit/mem_cache/unified_radix_cache/test_gfx1201_elastic_kv_tail.py`,
run with [`tools/run_sglang_unit_on_cpu.py`](../tools/run_sglang_unit_on_cpu.py).
What it shows:

- **Flag off.** Random traffic gives the same slots as a copy of the upstream
  allocator; a pool is built with ordinary buffers; `install` returns nothing.
- **Low slots first.** A tail slot is handed out only when no base slot is
  free, in fixed and random sequences.
- **Tail gating.** A closed tail is never handed out, a freed tail slot does
  not re-enter circulation, and free plus held plus out-of-circulation slots
  always equal the pool.
- **Shrink order.** With idle prefixes in the tail, one locked by a running
  request, and a request holding tail slots of its own: close, evict, wait,
  scrub, unmap, acknowledge, in that order and no earlier.
- **Tail-holding prefixes.** With Full, SWA and Mamba slots and checkpoint
  windows at page size 1, eviction removes exactly the entries that hold tail
  slots and their descendants, returns their Mamba checkpoints and window
  slots, keeps base-only prefixes, and leaves every remaining checkpoint its
  window; 4 seeds of 400 random operations with grow and shrink cycles keep
  the cache's sanity check and the leak-check identity.
- **Two ranks.** Two simulated ranks with a blocking gather end with identical
  free lists, cache contents and limits; a slow release, a release that never
  shows, a failed map and a memory shortage on one rank decide the outcome for
  both.
- **Release confirmation.** A delayed free defers the acknowledgement; a free
  that never shows gives `release_unconfirmed` after the timeout.
- **Buffers.** One extent per buffer at the same address over 50 cycles, base
  checksum unchanged, a failed map leaves nothing mapped.
- **Wiring.** `install` on a stub scheduler, the IPC codec, the tokenizer's
  context length around grow and shrink, and the HTTP handler.

## Not verified

1. **No GPU run.** `HipVmmBackend` has never executed: the `ctypes` layouts
   follow the image's `hip_runtime_api.h` and `driver_types.h`, and a test
   checks their sizes, but no HIP call was made.
2. **A PyTorch tensor over a HIP-mapped address**
   (`_construct_storage_from_data_pointer` with a `cuda` device), including
   Triton kernels and captured graphs reading it.
3. **`pci_bus_id` and friends** on `torch.cuda.get_device_properties` under
   ROCm, and whether the pod can read the card's sysfs counters. Without them
   the release check falls back to `hipMemGetInfo`, which may not show the lag
   B0 saw.
4. **Startup at the reserved size in shared mode.** Whether SGLang's sizing
   accepts 645,558 tokens with the generators loaded, or lowers it.
5. **A real scheduler.** The tail was driven directly and through a stub
   scheduler; no server was started. The per-iteration gather during a shrink
   and the request-limit attributes are exercised only by reading the code.
6. **Stale slot ids.** The request table is scrubbed before the unmap. Whether
   any other GPU buffer keeps a stale tail slot id that a kernel dereferences
   (a padded batch row, a captured graph's index buffer) is unknown; on a GPU
   that would be a page fault, not a wrong answer.
7. **Speed.** The free path's synchronisation while tail slots are in use, and
   the cache scan during a shrink.
8. **HTTP workers.** The context length lives in the one HTTP process the
   serving arms run; several tokenizer workers would each need the update.

## For Phase B2, in this order

1. Flag on, tail unbacked: the server starts; the log shows the base and tail
   sizes; free VRAM per card, graph capture and decode speed match UQSB. This
   also proves items 1 to 4 above.
2. One `grow` and one `shrink` with no traffic: `mapped_bytes`,
   `released_bytes`, `vram_source: "sysfs"`, `vram_confirmed`, and VRAM per
   card before, between and after.
3. A conversation that reaches into the tail, then a shrink: no GPU page fault
   in the kernel log, the conversation's prefix is prefilled again afterwards.
4. The plan's 50 shrink and grow cycles with a base checksum
   (`TailBufferOwner.base_checksum`), under memory pressure from the
   generators so the zero-fill after map meets a nearly full card.
5. Chat speed with tail slots in use against UQSB (item 7).
6. Then the plan's remaining B2 steps.

## Gate B0, for the record

Run by the owner on 14 November on both R9700s (kernel 7.3.4, HIP runtime
71826445): 1,000 rounds per card of create, map, read, fill, verify, unmap and
release of a 256 MiB to 4 GiB extent at the same address, with the base and a
`hipMalloc` buffer checked every round. No corrupt word. Map took 0.02 to
0.08 ms at the median (4 GiB: 40 to 77 ms at the 99th percentile), unmap and
release 0.08 to 0.10 ms. The released extent stayed in `mem_info_vram_used` in
about 32% of rounds and always returned later.
