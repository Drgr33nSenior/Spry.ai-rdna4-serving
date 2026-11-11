# Bounded DFlash draft KV pool: plan, 11 November 2026

**Status:** plan only, no code. This is the next context-capacity step after UQ2. It frees
about 30% more target KV in both modes without touching the target's format, and it
compounds with UQ3. It needs GPU validation, so it is written as a plan with gates.

## Evidence

| Fact | Source |
| --- | --- |
| Draft `incoai/Qwen3.8-27B-DFlash2@015e7956`: 5 layers, all `sliding_attention`, window 2048, 8 KV heads × 128, block size 8 | its `config.json` |
| The pinned draft passes the window to attention (`sliding_window_size`, `_get_dflash_layer_attention_params`) | `srt/models/dflash.py` at `fcb32746`, lines ~200 and ~347 |
| The pinned pool gives the draft **one KV row per target row**: `scale_kv_cell_size_per_token_for_dflash` folds the draft cost into every token's cell size. No bound. | `srt/model_executor/pool_configurator.py` ~355-377 |
| Upstream PR #36995 (open, +3,510 / −104) adds a compact, window-sized physical draft layout for **exactly Qwen3.8-27B + DFlash2**. Its location kernel `rebuild_compact_physical_draft_req_to_token` is Triton. Validated on CUDA with Mooncake PD only, and it **requires radix cache, HiCache and unified memory to be disabled** | the PR body and files |

The draft only ever reads its last 2,048 rows, plus the 8-token block it's drafting. A
pool sized to that per running request holds everything it reads.

## Payoff (per card, TP2, FP8 target and FP8 draft)

| | Draft KV today | Bounded draft (2 requests × (2048 + 8 + 64 guard)) | Target pool, same budget |
| --- | --- | --- | --- |
| Shared mode (UQS, 3.28 GiB) | 800 MiB | 20.7 MiB | 163,840 → **213,715 tokens (1.30x)**: about 107k per conversation instead of 81,888 |
| Large mode (UQL, 10.06 GiB) | 2,452 MiB | 20.7 MiB | 502,093 → **657,672 tokens (1.31x)**: two full 262,112-token conversations fit, instead of one plus about 240k |

Per token: target 16 KiB, draft 5 KiB (5 layers × K,V × 4 heads per card × 128 × 1 B).
With UQ3 as well, the research's 2.4x shared-mode figure applies.

## The blocker: radix cache

#36995's compact path needs the radix cache off. Here that's not acceptable: radix prefix
reuse is what keeps each new turn of a long conversation from re-prefilling the whole
history (a cold 235K prompt takes 239 s). With a bounded draft pool, a cached prefix has
target KV but **no draft KV for its last window**. Three options:

| Option | What happens on a prefix hit | Cost | Risk |
| --- | --- | --- | --- |
| **A. Rebuild the window** | Re-run the target over the last min(W, prefix) tokens to regenerate the hidden states the draft consumes (`target_layer_ids` 5, 19, 33, 47, 61), then fill the draft window | About 2,048 extra prefill tokens per turn, about 0.6 s at 3,600 tok/s | Most code: a partial-recompute path in the scheduler |
| **B. Cold draft** | The draft starts with an empty window at the hit point | No TTFT cost; DFlash acceptance drops until 2,048 new tokens accumulate | Simplest. Acceptance loss is measurable on the bench (tok/verify) |
| **C. Window per radix node** | Keep a W-row draft window stored with each cached sequence end | Memory per cached branch (about 10 MiB each, FP8) | Most invasive: radix eviction must own draft rows |

**Recommendation:** prototype **B** first. It's a pure layout change, its quality risk is
zero (the target decides every token), and its cost shows up directly as tok/verify on the
existing bench. Move to **A** only if B's acceptance loss on resumed long conversations
outweighs 30% more context.

## Plan

1. **Port the layout:** port the non-PD parts of #36995 as serving patch `0028`:
   - `speculative/dflash_compact_physical_layout.py`;
   - the Triton `rebuild_compact_physical_draft_req_to_token`;
   - the `pool_configurator` fixed-draft accounting (`_dflash_draft_fixed_bytes`, `_subtract_fixed_pool_bytes`);
   - the `server_args` switch.
   - Drop the Mooncake/PD transport and the radix-off requirement, and add option B's empty-window start on a prefix hit.
2. **ROCm check:** confirm the gfx1201 attention backend applies `sliding_window_size` for the draft (`window_left`), since today an unbounded pool would hide a backend that ignores it. Gate: draft logits with the bounded pool are bitwise-equal to the unbounded pool on prompts up to 2,048 tokens, and equal within the existing error rule beyond.
3. **Accounting:** `max_total_num_tokens` rises by the predicted 1.30x at the same `--mem-fraction-static`. Measure **peak VRAM**, not only tensor bytes, in shared mode with both SDXL generators active.
4. **Bench gates (owner-run):**
   - tok/verify within 3% of today on fresh prompts;
   - report tok/verify separately for radix-hit resumes (option B's cost);
   - quality menu at least 11/0/1;
   - no change in ms per verify step at 32k/64k/120k.
5. **Profiles:** a new arm (e.g. `UQSB`/`UQLB`) carrying the larger `--max-total-tokens` and context. The qualification record follows the bench.

## Not in scope

PD disaggregation and Mooncake transfer; HiCache; a 4-bit draft (the research's 1.88x
alternative needs its own reader).
