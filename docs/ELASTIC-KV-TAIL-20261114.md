# Elastic KV tail: Phase B1 record, 14 November 2026

**Status:** built and tested on the CPU only. Nothing here has run on a GPU, no
image contains it, and the serving profile is unchanged. Patches `0031` to
`0033` of the [`sglang-main` series](../serving/patches/sglang-main/README.md)
hold the work. It is off unless one environment switch is set.

This is Phase B1 of the plan the owner approved on 14 November: let the shared
mode's KV pool borrow GPU memory while the picture and video generators are
idle, and give it back before they run. Gate B0 passed the same day (below).
Phase B2 is a GPU window; its first checks are listed at the end.

This record was revised twice on the day it was written, before anything
was published. The first draft let conversations grow with the tail and made
a shrink wait for running requests. The owner then decided that the tail is
prefix-cache room only and that media never waits for chat, so a shrink now
moves what running requests hold into the base instead of waiting for them.
An independent review of that version then found the defects listed under
[Review findings](#review-findings); this text describes the code after they
were fixed. The first GPU run then showed that a grow without a size could
never fit; it is now sized to the free memory (see [Grow](#grow)).

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
- an external cache linker or Unlimited-OCR document reuse (both hold KV
  slots outside the request table and the prefix cache);
- running requests that could not all fit the base (next section).

### Capacity margin

A shrink can always move running requests into the base only if they fit it
at their largest. A request holds at most its context plus what decode
allocates ahead of it, and never more than its request-table row can name:

```text
per request = min(row width, context + max(row width - context length, 2 x draft tokens))
margin      = base - max running requests x per request
```

The row width is `--context-length` plus SGLang's row headroom, 4 plus the
draft tokens (`get_req_to_token_extra_context_len`); DFLASH allocates two
verify blocks ahead (`dflash_info_v2.py`). For the arm above, with 8 draft
tokens: row width 100,780, per request min(100,780, 100,768 + max(12, 16)) =
100,780, margin 201,600 - 2 x 100,780 = **40 slots**. The server logs the
margin, reports it as `base_margin_tokens`, and refuses to start when it is
negative. The arm passes.

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
3. Make room in the base: evict idle base prefixes, oldest first, until the
   tail slots in use would fit.
4. List the tail slots still in use, after that eviction. They belong to
   running requests and to prefixes those requests have locked. Every one must
   be named by the row of a running request or a streaming session, or by a
   prefix-cache entry: those are the names that get rewritten. A row left by
   a finished request does not count. If a slot is named by none of them, or
   the slots do not fit the base, the shrink is refused.
5. Every rank reports whether it can proceed and what it would move. If any
   cannot, all reopen the tail and answer `ok: false`,
   `reason: "cannot_migrate"`, `state: "backed"`. If the ranks would move
   different numbers of slots or evicted different amounts, the answer is
   `reason: "rank_mismatch"`. Nothing a request holds has changed at that
   point.
6. Migrate: allocate base slots, copy K and V of every target layer from the
   old rows to the new ones (device to device, 16,384 rows at a time), and
   rename the slots wherever a slot id is kept:
   - the rows of running requests and sessions in the request table
     (`req_to_token`), which are the register of a request's slots;
   - the Full value of every prefix-cache entry that named a tail slot;
   - the keys of `full_to_swa_index_mapping`, through which the bundled draft
     pool finds a token's window row. The window rows themselves do not move.

   One rename table is applied to all three, so a slot that a request row and
   a locked prefix both name stays one slot.
7. Verify: no running request's or session's row and no prefix-cache entry
   names a slot above the base, and the allocator holds no tail slot out.
   Every rank reports the result; the answer carries it as
   `verified_no_tail_refs`. If it is false on any rank the tail is **not**
   unmapped and the answer is `ok: false`, `reason: "verify_failed"`. The
   ranks also compare what they moved and what their allocators and caches
   hold afterwards; if they differ the tail is not unmapped either
   (`rank_mismatch`).
8. Only now clear what is left above the base in the request table. The live
   rows were just checked, so these are entries of finished requests.
   Synchronise the device, unmap and release each tail extent.
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
backed, and startup checks the [margin](#capacity-margin). So what running
requests hold always fits the base once idle prefixes are evicted, and
`cannot_migrate` should not occur in the configured arm. It is still checked
on every shrink, not assumed.

**Cost.** A shrink moves 16 KiB per migrated token on each card. The worst
case is two requests at the context limit entirely in the tail: 2 x 100,768
tokens x 16 KiB = 3.07 GiB read and written per card, plus the eviction of
whatever idle prefixes sat in the base. The chat stalls for that long. A
shrink also reads the prefix cache's slot ids three times (to choose what to
evict, to plan, to verify); each read is one transfer of all cached ids to
the host, not one device round trip per cache entry. On the CPU the copy
cannot be timed meaningfully; **B2 must time it** against the 15 s the media
request allows. The answer reports `migrated_tokens`, `migrated_bytes` and
`migrate_s`.

### A rank that fails

The ranks exchange reports at fixed points of a grow and a shrink. A rank
that raised between two of them would leave its peer waiting for a report
that never comes, so local work never raises: a failure (a device call, a
sysfs read, an assertion) is caught, logged and sent as the rank's report,
and all ranks then take the same branch.

| Failure | State on every rank | Answer |
| --- | --- | --- |
| Reading free memory before a grow | `unbacked`, nothing mapped | 409 `rank_error` |
| Mapping the tail | `unbacked`; ranks that mapped unmap again | 409 `map_failed` |
| Preparing a shrink (evicting, listing) | `backed`, tail reopened; resizing stops (below) | 409 `rank_error` |
| Copying K and V, or reading device memory before the unmap | `backed`, tail reopened; base slots taken for the copy are given back | 409 `rank_error`; a retry works |
| Renaming slots | `backed`, tail reopened and mapped; resizing stops (below) | 409 `rank_error` |
| Unmapping | `unbacked`: no tail slot is handed out again; the failing rank may still have extents mapped (`tail_bytes`) | 409 `rank_error`; a retried shrink, or a grow, unmaps what is left |
| Reading device memory while waiting for the release | `unbacked` | 409 `release_unconfirmed` at the timeout |

A failure inside an eviction or half way through a rename leaves that rank's
bookkeeping in doubt. Its K and V are still valid, because the old rows stay
mapped and the new rows hold copies, but the tail can no longer be resized
safely. Every rank then records a `fault`, shown in the status, and answers
every later grow and shrink with `rank_error` until the server restarts. The
tail stays as it is; chat is not affected, media is refused.

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

The server remembers a release it could not confirm (`release_unconfirmed`
in the status). The next shrink does not answer from that memory: it polls
the device again, against the reading taken before the unmap, and answers
`ok: true` with `rechecked: true` only when the memory shows free, otherwise
`release_unconfirmed` again. A successful grow clears the record.

### Startup self-test

When the pool is built, before anything uses it, the server maps a small tail
extent (8 MiB) at one address 32 times; each time it writes and reads it back
and checks three pieces of live memory next to it (base rows of two KV
buffers and an ordinary allocation). On a kernel without the amdgpu TLB-fence
fix, a remap at the same address overwrites live buffers (ROCm/TheRock#8744);
this is a cheap guard against starting on such a kernel, not a substitute for
Gate B0. If it fails on any rank, the result is logged, the server runs on
the base, and every grow answers `reason: "self_test_failed"`. The time it
takes on a GPU is not measured yet; it is skipped when the switch is off.

### Grow

Every rank reads its free device memory and the ranks exchange the readings.
What may be used is the smallest reading less `keep_free_mib`.

- **Without `tail_tokens`** the tail is sized to fit: the most tokens whose
  extents fit that budget on the tightest rank, which rounds down to whole
  allocation granules and whole tokens, capped at the reserved tail. If that
  is below a floor of 32,768 tokens (512 MiB per card), nothing is mapped and
  the answer is `insufficient_vram` with `fits_tail_tokens`. The first GPU
  run showed why this is the default: after the generators free their
  memory the cards had 7.34 and 7.83 GB free, less than the whole tail
  (7.18 GB) plus the 1.5 GiB kept free, so a fixed full-size grow could never
  succeed.
- **With `tail_tokens`** the size is exact: mapped as asked, or refused with
  `insufficient_vram`.

The size is computed from the gathered readings, so every rank maps the same
number of tokens. Then every rank maps and zero-fills. If any rank fails, the ranks that succeeded unmap again
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
| `{"action": "grow"}` | `tail_tokens` (default: the largest tail that fits every rank), `keep_free_mib` (default 1536) | When every rank has mapped, or none has |
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
| `message`, `reason` | Text; and on failure one of `cannot_migrate`, `verify_failed`, `rank_mismatch`, `rank_error`, `release_unconfirmed`, `insufficient_vram`, `map_failed`, `self_test_failed`, `already_backed`, `busy`, `bad_request`, `disabled` |
| `tail_tokens`, `reserved_tail_tokens`, `base_tokens`, `pool_tokens` | Sizes in tokens |
| `context_length` | The base context; constant |
| `tail_bytes`, `tail_slots_in_use`, `backend` | Per-rank detail from rank 0 |
| `release_unconfirmed` | A release the device has not shown yet; shrink again to check |
| `base_margin_tokens` | The [capacity margin](#capacity-margin) |
| `fault`, `self_test_error` | Present only when resizing has stopped, or never started |
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

A grow answer adds `mapped_bytes`, `sized_to_fit`, `free_bytes_by_rank`,
`map_s` and `elapsed_s`; `tail_tokens` is the size chosen. While a shrink
waits for the device, the status has `pending` with `elapsed_s` and
`waiting_for_vram_release`.

Rules for the caller:

- `shrink` and `grow` are idempotent: a shrink of an unbacked tail, a grow
  without a size while any tail is backed, and a grow to the size already
  backed answer `ok: true`. To change the size of a backed tail, shrink
  first.
- Start a media job only on a shrink answer with `ok: true`. On a 409 refuse
  the job. `cannot_migrate`, `verify_failed` and `rank_mismatch` leave the
  tail backed. `release_unconfirmed` leaves it unbacked with the memory not
  yet visible; shrink again later to check. `rank_error` says in `state`
  which it is.
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
| `unified_radix_cache/test_gfx1201_elastic_kv_tail.py` (new), on the CPU runner | Not present | 71 passed |
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
  `d9af3651b8857720c996ec27c4341437c1ef80a78cc49cf0eb14536a8565f9cd`
  (`40067ef9...` for the first 30, unchanged). `stack.lock` was not touched.
- **Mutations.** Thirty-three deliberate breaks of the new code each fail the
  new test module. Among them: freed tail slots re-entering circulation, a
  closed tail handed out, locks ignored when choosing what to evict, KV not
  copied, the window mapping not carried or not cleared, request rows or the
  prefix cache not renamed, the slots listed before the eviction for room,
  stale rows vouching for a slot, the table scrubbed before the verification,
  a retry answering from memory, rank comparisons skipped, a local failure
  raised between gathers, a fault or a self-test verdict kept to one rank, a
  leaked reservation, session rows ignored, the margin without decode's
  headroom, and a default grow sized to the larger rank, without the floor
  or rounded up a granule. The list is `scripts/mutate.sh` in the job folder, not in this
  repository.
- **This repository's own tests.** `make -k test` on macOS (run before the
  revisions; no tracked file other than this record, the series README and
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
  a finished request's row that still shows such a slot does not vouch for
  it; a rename that misses the cache gives `verify_failed`, no unmap, and no
  entry of the request table zeroed.
- **A rank that fails.** On one of two ranks: the memory read before a grow,
  the memory read before the unmap, the unmap, the eviction while preparing,
  the last step of the rename, and every release poll. In each case both
  ranks answer within the harness's 10 s limit (a rank left waiting would
  break it), end in the same state, and the documented recovery works: a
  retry after a failed read or unmap, a grow after a failed unmap, no further
  resizing after a fault.
- **Rank disagreement.** Ranks that would move different slots move none;
  ranks that differ after moving do not unmap.
- **Release retried.** After `release_unconfirmed` a second shrink polls the
  device again, fails again while the memory is not visible and succeeds only
  when it is.
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
- **Grow sizing.** Without a size, the largest tail that fits is mapped: one
  byte short of a second granule per buffer maps one; the kept-free memory
  comes off first; two ranks size to the smaller device and end identical;
  below the floor nothing is mapped. With a size, one token more than fits is
  refused and the exact size is mapped.
- **Release confirmation.** A delayed free defers the acknowledgement; a free
  that never shows gives `release_unconfirmed` after the timeout.
- **Buffers.** One extent per buffer at the same address over 50 cycles, base
  checksum unchanged, a failed map leaves nothing mapped, chunked row copy; a
  failed reserve, map or tensor construction at startup leaves no mapping and
  no address range behind; with the switch off a pool has the upstream
  shapes, dtypes, strides and zeroed ordinary memory for bf16 and FP8.
- **Self-test.** Passes on a sound fake device with nothing left mapped or
  written; a remap that scribbles on live memory is caught in the first round
  and every grow is then refused, also when only one rank saw it.
- **Wiring.** `install` on a stub scheduler with a running request whose KV is
  in the real pool's tail, a streaming session's row moved like a request's,
  the startup refusals and the margin arithmetic, the IPC codec, the
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
9. **The self-test's duration** on a GPU (it is meant to stay under 2 s), and
   that it passes on the fixed kernel.
10. **`kv_allocated_len`** as the length of a row that covers every slot the
    request holds, DFLASH's two verify blocks included. If it is shorter, the
    slots past it are unnamed and a shrink during that request is refused.

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
4. More shrinks during traffic, each with a check of the text and the kernel
   log:
   - with one request finished whose row 0 held tail ids, then a batch of two
     replayed through the padded batch-2 graph;
   - in the middle of a chunked prefill, between two chunks;
   - while DFLASH v2 holds its over-allocated verify blocks, the generation
     continuing afterwards.
5. Time the worst case: two requests near 100,768 tokens in the tail, shrink,
   read `migrate_s` and `elapsed_s`, and compare with the 15 s media budget.
6. The plan's 50 shrink and grow cycles with a base checksum
   (`TailBufferOwner.base_checksum`), under memory pressure from the
   generators so the zero-fill after map meets a nearly full card.
7. Chat speed with tail slots in use against UQSB (item 8).
8. Then the plan's remaining B2 steps.

Not possible before B2, by design: a real scheduler, worker and graph replay,
any `HipVmmBackend` call, and `_construct_storage_from_data_pointer` on a HIP
pointer.

## Review findings

An independent read-only review of the migration version found no surviving
reference to the tail in the configured arm, and these defects, all fixed in
the code this record describes:

| Finding | Fix |
| --- | --- |
| A rank that raised between two gathers left its peer waiting and its own state half done | Local work reports failures instead of raising; states per failure point are defined [above](#a-rank-that-fails) |
| The list of slots to move was taken before the eviction for room | It is taken after it |
| A finished request's row could vouch for a slot held by something unseen | Only rows of running requests and sessions count; the external linker and document reuse are refused at startup |
| A shrink retried after `release_unconfirmed` answered `ok` without asking the device | The unconfirmed release is remembered and polled again |
| Ranks that disagreed were logged but still unmapped | Both comparisons now refuse the shrink |
| The request table was scrubbed before the verification could fail | Scrubbed only after it passes |
| The startup capacity check ignored decode's allocation ahead | The [margin](#capacity-margin) includes it |
| One device round trip per cache entry, four walks per shrink | One host transfer per walk, three walks |
| A failed start leaked the address range or the base mapping | Cleaned up on every path |
| No guard against a kernel without the TLB-fence fix | The [startup self-test](#startup-self-test) |

## Gate B0, for the record

Run by the owner on 14 November on both R9700s (kernel 7.3.4, HIP runtime
71826445): 1,000 rounds per card of create, map, read, fill, verify, unmap and
release of a 256 MiB to 4 GiB extent at the same address, with the base and a
`hipMalloc` buffer checked every round. No corrupt word. Map took 0.02 to
0.08 ms at the median (4 GiB: 40 to 77 ms at the 99th percentile), unmap and
release 0.08 to 0.10 ms. The released extent stayed in `mem_info_vram_used` in
about 32% of rounds and always returned later.
