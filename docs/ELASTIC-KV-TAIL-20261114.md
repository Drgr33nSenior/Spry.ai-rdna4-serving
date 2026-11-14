# Elastic KV tail: Phase B1 record, 14 November 2026

**Status:** built and tested on the CPU only. Nothing here has run on a GPU, no
image contains it, and the serving profile is unchanged. Patches `0031` to
`0033` of the [`sglang-main` series](../serving/patches/sglang-main/README.md)
hold the work. It is off unless one environment switch is set.

This is Phase B1 of the plan the owner approved on 14 November: let the shared
mode's KV pool borrow GPU memory while the picture and video generators are
idle, and give it back before they run. Gate B0 passed the same day (below).
Phase B2 is a GPU window; its first checks are listed at the end.

This record was revised on the day it was written, before anything was
published. The first draft let conversations grow with the tail and made a
shrink wait for running requests. The owner then decided that the tail is
prefix-cache room only and that media never waits for chat, so a shrink now
moves what running requests hold into the base instead of waiting for them.

## What it does

Shared mode serves from a 201,600-token KV pool (arm UQSB) because the video
job needs the rest of the card for about 80 seconds at a time. With the switch
on, the server starts with the large arm's pool size, but only the first
201,600 tokens of every target KV buffer have memory behind them:

| | Tail not backed | Tail backed |
| --- | --- | --- |
| Pool | 201,600 tokens (the base) | Base plus up to 443,958 tail tokens |
| Context a request may use, and the server advertises | 100,768 | 100,768 |
| GPU memory | As UQSB, plus a few MiB of larger index tables | 16 KiB more per tail token per card |

A `grow` request maps the tail; a `shrink` request empties and unmaps it. A
backed tail means fewer cached conversations are evicted and prefilled again.
It does not make a conversation longer. The draft KV pool stays at 43,008
tokens (`--speculative-draft-kv-bundling`, which the feature requires).

## Switching it on

One switch, `SGLANG_GFX1201_ELASTIC_KV_BASE_TOKENS`, off when unset or `0`. It
needs one companion value. An elastic arm is UQSB with a larger pool:

| Setting | Value | Meaning |
| --- | --- | --- |
| `--max-total-tokens` | 645558 | The reserved pool: base plus the largest tail |
| `--context-length` | 100768 | As UQSB |
| `SGLANG_GFX1201_ELASTIC_KV_BASE_TOKENS` | 201600 | Tokens that are always backed |
| `SGLANG_GFX1201_ELASTIC_KV_BASE_CONTEXT` | 100768 | The context requests are held to |

The context in force is the smaller of `--context-length` and the base
context. Keeping `--context-length` at 100,768 also keeps the request table
and the captured graphs' index buffers the size they are in UQSB.

SGLang still lowers `--max-total-tokens` to what it measures as free at
startup. If it does, the reserved tail is smaller; `GET /kv_tail` reports
`reserved_tail_tokens`.

The server refuses to start with the switch on when it cannot support it:

- no draft bundling, a page size other than 1, HiCache, the Rust tree core, DP
  or PP, a quantized or non-NHD KV layout, or a pool no larger than the base;
- the overlap scheduler (a shrink moves KV between forwards, so
  `--disable-overlap-schedule` is required, as the serving arms have it);
- `--enable-prefill-weight-versions` (a per-slot table the migration does not
  rename);
- `max running requests x context` above the base. For the serving arm that is
  2 x 100,768 = 201,536, within 201,600. This is what guarantees that running
  requests fit the base.

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
hands out a tail slot only while the tail is open and no base slot is free. A
count of tail slots in use is kept on the CPU. Freed slots are sorted into the
two lists only while that count is not zero, so a server whose tail is unused
or unbacked pays nothing on the free path. While tail slots are in use, each
free reads the slot ids on the GPU (one synchronisation); B2 measures what
that costs in tokens per second.

Once the cache is warm the base is full of idle prefixes, because the prefix
cache evicts only when no slot is free. New requests then land in the tail.
That is why a shrink has to be able to move them.

The pool, the allocator and every table indexed by slot id are built at the
reserved size, so no table is too small for a tail slot. The capacity reported
to the pool statistics and the idle leak check is the reserved size less the
slots that are out of circulation.

### Shrink: migrate, never wait

All of this happens in the scheduler step that receives the request, between
forwards, on every TP rank:

1. Stop handing out tail slots.
2. Evict the idle prefix-cache entries that hold tail slots. An entry goes
   only if nothing at or below it in the tree is locked, and then the entries
   below it go too. Their Mamba checkpoints and draft windows go with them
   through the cache's own leaf eviction.
3. List the tail slots still in use. They belong to running requests and to
   prefixes those requests have locked. Every one must be named by the request
   table or by a prefix-cache entry; if one is not, the shrink is refused,
   because nothing would know its new name.
4. Make room in the base: evict idle base prefixes, oldest first, until the
   slots to move fit. If they still do not fit, the shrink is refused.
5. Every rank reports whether it can proceed. If any cannot, all reopen the
   tail and answer `ok: false`, `reason: "cannot_migrate"`, `state: "backed"`.
   Nothing a request holds has changed at that point.
6. Migrate: allocate base slots, copy K and V of every target layer from the
   old rows to the new ones (device to device, 16,384 rows at a time), and
   rename the slots wherever a slot id is kept:
   - the request table (`req_to_token`), whose rows are the register of a
     request's slots;
   - the Full value of every prefix-cache entry that named a tail slot;
   - the keys of `full_to_swa_index_mapping`, through which the bundled draft
     pool finds a token's window row. The window rows themselves do not move.

   One rename table is applied to all three, so a slot that a request row and
   a locked prefix both name stays one slot.
7. Verify: no live request's row and no prefix-cache entry names a slot above
   the base, and the allocator holds no tail slot out. Every rank reports the
   result; the answer carries it as `verified_no_tail_refs`. If it is false on
   any rank the tail is **not** unmapped and the answer is `ok: false`,
   `reason: "verify_failed"`.
8. Clear what is left above the base in the request table (rows of finished
   requests), synchronise the device, unmap and release each tail extent.
9. Wait until the device reports the released memory free (below), then
   acknowledge. This is the only part that spans scheduler steps; chat runs
   on during it.

What else stores a target slot id was checked by reading the pin and patches
0028 to 0030:

| Place | Stores | Migration |
| --- | --- | --- |
| `req_to_token` rows | Target slot ids | Renamed |
| Prefix-cache Full values | Target slot ids | Renamed |
| `full_to_swa_index_mapping` (0029) | Window row, keyed by target slot id | Keys moved |
| Prefix-cache SWA values (0028) | Window-pool rows | Not slot ids of the target; unchanged |
| Checkpoint-window bookkeeping (0028) | Per-node counts and flags | No slot ids |
| Mamba checkpoints and request state | Mamba slot ids | Unchanged |
| `ReqKvInfo` | Row index and lengths only | Unchanged |
| KV weight-version table | Per target slot | Not supported; refused at startup |
| Captured graphs' index buffers, a batch's `out_cache_loc` | Slot ids of one forward | Refilled every forward; none is in flight at a step boundary with overlap off |

The last row and the DFlash worker's per-request state are read, not run; B2
checks them with a chat that continues across a shrink.

**Capacity.** Requests are held to the base context whether or not the tail is
backed, and startup checks that `max running requests x context` fits the
base. So what running requests hold always fits the base once idle prefixes
are evicted, and `cannot_migrate` should not occur in the configured arm. It
is still checked on every shrink, not assumed.

**Cost.** A shrink moves 16 KiB per migrated token on each card. The worst
case is two requests at the context limit entirely in the tail: 2 x 100,768
tokens x 16 KiB = 3.07 GiB read and written per card, plus the eviction of
whatever idle prefixes sat in the base. The chat stalls for that long. On the
CPU the copy cannot be timed meaningfully; **B2 must time it** against the
15 s the media request allows. The answer reports `migrated_tokens`,
`migrated_bytes` and `migrate_s`.

### Release is confirmed, not assumed

Gate B0 found that after `hipMemRelease` and a device synchronise the card's
`mem_info_vram_used` still counted the released extent in about a third of the
rounds, and dropped later. The shrink therefore does not acknowledge when
unmap returns. Each rank reads the card's used memory before the unmap and
polls it (every 20 ms, from the scheduler loop) until it has fallen by the
released bytes less a slack of 64 MiB. The source is the card's sysfs counters
when the process can read them, otherwise `hipMemGetInfo`; the answer names it
in `vram_source`. If any rank has not seen the memory free after
`release_timeout_s` (default 10 s), the answer is `ok: false`,
`reason: "release_unconfirmed"`, `vram_confirmed: false`. The tail is unmapped
by then, so the state is `unbacked`; the caller must not start the media job
on that answer.

### Grow

Every rank checks free device memory against the bytes needed plus
`keep_free_mib`; if any rank is short, nothing is mapped. Then every rank
maps and zero-fills. If any rank fails, the ranks that succeeded unmap again
and the answer is `ok: false`, `reason: "map_failed"`. Only then are the tail
slots enabled.

### Ranks stay identical

The control request reaches every rank through the scheduler's request
broadcast, like `release_memory_occupation`. Whatever a rank measures for
itself (free memory, a map error, whether its slots can move, the
verification, the release poll, the clock) is exchanged with
`all_gather_object` on the TP CPU group at that step, and every rank acts on
the same gathered reports; the release timeout follows rank 0's clock. New
slots come from the base free list in its own order and old slots are taken
in ascending order, so identical ranks rename identically. The answer carries
`ranks_identical`, false if the ranks reported different counts.

### Limits

The scheduler's pool size follows the tail (`base + tail`); its request-length
limits and the tokenizer's context length are set to the base context at
startup and never change. `/v1/models` reports the base context. Nothing in
the workstation repository was changed, and the gateway's fixed
per-conversation limit needs no change.

Not updated when the tail changes: the scheduler's Prometheus capacity gauges
taken at startup (they show the reserved size).

## Control contract

`POST /kv_tail` with a JSON body, or `GET /kv_tail` for the status. The
endpoint has the same admin authentication level as
`/release_memory_occupation`.

| Request | Fields | Answer |
| --- | --- | --- |
| `{"action": "status"}` or `GET` | none | At once, also while a shrink waits for the device |
| `{"action": "grow"}` | `tail_tokens` (default: the whole reserved tail), `keep_free_mib` (default 1536) | When every rank has mapped, or none has |
| `{"action": "shrink"}` | `release_timeout_s` (default 10) | When every rank has migrated, unmapped and seen the memory free; or at once when it is refused |

An unknown field is a 400, including the first draft's `timeout_s`: a shrink
has nothing left to wait for but the device.

HTTP status: 200 when `ok` is true, 400 for a malformed request, 409 when the
request was understood and did not happen.

Every answer has:

| Field | Meaning |
| --- | --- |
| `ok` | The request did what it says |
| `state` | `disabled`, `unbacked`, `backed` or `releasing` |
| `message`, `reason` | Text; and on failure one of `cannot_migrate`, `verify_failed`, `release_unconfirmed`, `insufficient_vram`, `map_failed`, `already_backed`, `busy`, `bad_request`, `disabled` |
| `tail_tokens`, `reserved_tail_tokens`, `base_tokens`, `pool_tokens` | Sizes in tokens |
| `context_length` | The base context; constant |
| `tail_bytes`, `tail_slots_in_use`, `backend` | Per-rank detail from rank 0 |
| `waited`, `queued_s` | Grow and shrink only: time spent queued behind an earlier grow or shrink in the HTTP process, and whether it exceeded 0.25 s. Never time spent on a running request |

A shrink answer adds:

| Field | Meaning |
| --- | --- |
| `migrated_tokens`, `migrated_bytes`, `migrate_s` | What was moved to the base, per card, and how long steps 6 to 8 took |
| `evicted_tokens` | Idle prefix tokens dropped: `evicted_tail_tokens` (they sat in the tail) plus `evicted_base_tokens` (dropped to make room) |
| `verified_no_tail_refs` | Step 7 held on every rank |
| `released_bytes`, `vram_confirmed`, `vram_freed_bytes_by_rank`, `vram_source` | The release and its confirmation |
| `elapsed_s` | Scheduler time from the request to the answer, the device wait included |
| `ranks_identical` | The ranks reported the same counts |

A grow answer adds `mapped_bytes`, `map_s` and `elapsed_s`. While a shrink
waits for the device, the status has `pending` with `elapsed_s` and
`waiting_for_vram_release`.

Rules for the caller:

- `shrink` and `grow` are idempotent: a shrink of an unbacked tail and a grow
  to the size already backed answer `ok: true`.
- Start a media job only on a shrink answer with `ok: true`. On a 409 refuse
  the job; `cannot_migrate` and `verify_failed` leave the tail backed,
  `release_unconfirmed` leaves it unbacked with the memory not yet visible.
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
| `managers/tokenizer_control_mixin.py` | Two communicators, the calls, the base context at startup |
| `managers/scheduler.py` | Install, two dispatcher entries, one call per iteration |
| `entrypoints/http_server.py` | `/kv_tail` |

Two new files: `mem_cache/elastic_kv_tail.py` and the test module. The
migration reads and writes prefix-cache entries from the new module; the
cache's own files are not changed.

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
| `unified_radix_cache/test_gfx1201_elastic_kv_tail.py` (new), on the CPU runner | Not present | 48 passed |
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
  `4002e1d4e86660867015f8be50f399ea8a590b35df33279d27e2a4ab56d84737`
  (`40067ef9...` for the first 30, unchanged). `stack.lock` was not touched.
- **Mutations.** Fifteen deliberate breaks of the new code each fail the new
  test module: freed tail slots re-entering circulation, a closed tail handed
  out, idle tail holders not evicted, locks ignored when choosing what to
  evict, KV not copied, the window mapping not carried or not cleared, the
  request table or the prefix cache not renamed, no eviction for room,
  unnamed slots moved anyway, verification skipped, stale ids not cleared,
  release assumed instead of confirmed, and the tail not reopened after a
  refusal.
- **This repository's own tests.** `make -k test` on macOS (run before the
  revision; no tracked file other than this record, the series README and
  the history index has changed): every target passes except
  `test-serving-cleanup`, which stops at
  `serving/tests/test_worker_timing.py`
  (`test_pins_and_separate_source_patch_are_exact`). That test fails the same
  way on an untouched checkout of `main`; it belongs to the 0.5.21 runtime's
  records.
- **Not run:** `ruff format` and `isort` on the patched SGLang files (neither
  is in the image or on the Mac). Import order, unused imports and line length
  were checked by hand.

The new module is
`test/registered/unit/mem_cache/unified_radix_cache/test_gfx1201_elastic_kv_tail.py`,
run with [`tools/run_sglang_unit_on_cpu.py`](../tools/run_sglang_unit_on_cpu.py).
Its cache is the unified radix cache with Full, SWA and Mamba slots and
checkpoint windows at page size 1. Every token a test places has K and V
derived from the token, so a row that was lost, not copied or copied to the
wrong place shows. What it shows:

- **Flag off.** Random traffic gives the same slots as a copy of the upstream
  allocator; a pool is built with ordinary buffers; `install` returns nothing;
  the tokenizer's context is left alone.
- **Low slots first and tail gating.** A tail slot is handed out only when no
  base slot is free; a closed tail is never handed out; free plus held plus
  out-of-circulation slots always equal the pool.
- **Migration.** Two running requests that share a locked prefix, all in the
  tail: one shrink call, no waiting. K and V at the new slots equal the old
  content for each request and for the shared prefix; both request rows and
  the cache name the same new slots; the window mapping follows the tokens and
  the old keys are cleared; the idle prefix in the tail is evicted and the
  base ones stay; afterwards the requests finish and every Full, window and
  Mamba slot returns to its pool.
- **Room.** Idle base prefixes are evicted, oldest first and only as many as
  needed, when the base is full.
- **Refusals.** Running requests that exceed the base, and a tail slot nothing
  names, give `cannot_migrate` with the tail reopened and every row unchanged;
  a rename that misses the cache gives `verify_failed` and no unmap.
- **Tree shapes.** An idle tail prefix goes with the idle prefixes below it; a
  tail prefix with a lock below it is moved; an idle base prefix below a
  locked tail prefix stays.
- **Property test.** 9 seeds of 500 random operations (insert, lock, unlock,
  start a request with or without a cached prefix, finish, grow, shrink).
  After every operation: content of every running request and cached prefix,
  the cache's sanity check, every checkpoint's window, and the leak-check
  identity. After every shrink: no slot above the base in any row or entry.
- **Two ranks.** Two simulated ranks with a blocking gather end with identical
  free lists, request tables, mappings, cache contents and base KV after a
  migration; a rank that cannot migrate, a slow or never-visible release, a
  failed map and a memory shortage on one rank decide the outcome for both.
- **Release confirmation.** A delayed free defers the acknowledgement; a free
  that never shows gives `release_unconfirmed` after the timeout.
- **Buffers.** One extent per buffer at the same address over 50 cycles, base
  checksum unchanged, a failed map leaves nothing mapped, chunked row copy.
- **Wiring.** `install` on a stub scheduler with a running request whose KV is
  in the real pool's tail, the startup refusals, the IPC codec, the
  tokenizer's context and `waited`, and the HTTP handler.

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
   scheduler; no server was started.
6. **That the request table and the prefix cache name every slot a running
   request holds.** If the DFlash worker holds slots of a verify block that
   are in neither at a step boundary, every shrink during a chat answers
   `cannot_migrate` (safe, but media would be refused). If some other buffer
   keeps a copy of a slot id across forwards, the chat would read the old,
   unmapped row after a shrink: a GPU page fault.
7. **Migration time** at real sizes, and the chat stall it causes.
8. **Speed** of the free path while tail slots are in use.

## For Phase B2, in this order

1. Flag on, tail unbacked: the server starts; the log shows the base and tail
   sizes; free VRAM per card, graph capture and decode speed match UQSB. This
   also proves items 1 to 4 above.
2. One `grow` and one `shrink` with no traffic: `mapped_bytes`,
   `released_bytes`, `vram_source: "sysfs"`, `vram_confirmed`, and VRAM per
   card before, between and after.
3. Warm the cache until requests land in the tail (`tail_slots_in_use` above
   zero), then shrink **during** a generation: the answer is `ok: true` with
   `migrated_tokens` above zero, the generation continues with sane text and
   its usual accept length, and the kernel log has no amdgpu page fault. This
   settles item 6.
4. Time the worst case: two requests near 100,768 tokens in the tail, shrink,
   read `migrate_s` and `elapsed_s`, and compare with the 15 s media budget.
5. The plan's 50 shrink and grow cycles with a base checksum
   (`TailBufferOwner.base_checksum`), under memory pressure from the
   generators so the zero-fill after map meets a nearly full card.
6. Chat speed with tail slots in use against UQSB (item 8).
7. Then the plan's remaining B2 steps.

## Gate B0, for the record

Run by the owner on 14 November on both R9700s (kernel 7.3.4, HIP runtime
71826445): 1,000 rounds per card of create, map, read, fill, verify, unmap and
release of a 256 MiB to 4 GiB extent at the same address, with the base and a
`hipMalloc` buffer checked every round. No corrupt word. Map took 0.02 to
0.08 ms at the median (4 GiB: 40 to 77 ms at the 99th percentile), unmap and
release 0.08 to 0.10 ms. The released extent stayed in `mem_info_vram_used` in
about 32% of rounds and always returned later.
