# Cache-path optimization candidate

This candidate changes KV4 cache decoding and validation for the official FP8
weight path. It does not convert model weights to FP4, change the cache format,
or change the model's precision or serving defaults. Installed runtime and
historical build evidence remain unchanged.

The retained instrumented long-prefill observation identified the custom reader
and cache validation as useful targets: about 4.766 seconds for the reader,
1.121 seconds for read validation and 0.665 seconds for FP8 matmul on rank 0.
These are cumulative recorded operation durations from one partial profile,
not a general throughput result or additive wall-time shares.

## Changes

`prototype/runtime/bf16_math.h` constructs the decoded BF16 bits directly from
the E2M1 nibble and UE8M0 scale. It replaces `ldexpf` followed by a BF16 roundtrip.
The sign, signed zero, subnormal and overflow cases retain explicit handling.
The generic codec remains the compatibility oracle in the CPU test.

`bf16_v2::validate_read` in `prototype/runtime/attention.hip` uses one 256-thread
block per query row. Adjacent lanes check adjacent protected BF16 channels or
packed bytes. One lane per packed group checks its K/V scales. Every nibble,
scale, query value, index and protected cache value remains checked. Span and
slot guards run before any corresponding cache access.

The largest task counts are 67,108,864 protected-channel tasks and 33,554,432
packed-byte tasks per row, within signed 32-bit indexing. Cache offsets still
use `size_t`. Duplicate read indices remain valid. The host still synchronizes
and checks validation status before launching attention. Invalid input cannot
be accepted by omitting validation or by using a cached validation result.

The encoder, Walsh-Hadamard transform, cache layout, C ABI, GDN state, ordered
FP32 dot products, online softmax and attention accumulation order are unchanged.
The decoder/validator candidate retained the 32-thread reader. The subsequent
grouped-reader candidate below changes only compact-cache reader dispatch.

## Measured decoder/validator baseline

The 2026-10-15 native cache comparison completed on both R9700 cards. Four
same-device baseline/candidate comparisons reported identical complete payloads;
the retained hashes, summary readers and cleanup observations passed offline
verification. Five synchronized call samples per cell gave median latency
reductions of 26.3–35.5 percent for protected BF16 K/V and 4.1–4.8 percent for
compact KV4, across lengths 128, 1024 and 3966. These are fixed-order standalone
cache-call measurements, not serving throughput or statistical significance.
No model weights were loaded.

The measured candidate library was
`3bf86042ba225abc279245b41e1baa35c2549def0e8547eaf60199c4b55d8849`,
with attention source
`a11c985bb5495e0445034a4aabe5b89fe1ed22fbd6995d76f9aaf4cda823a84b`.
Use those bytes as the comparison baseline for the next reader change. They are
not a build proof for the changed source.

The protected format stores **both K and V in BF16**. An earlier private readout
incorrectly called it BF16-K/FP8-V. That label correction does not change the
measurements or the original sealed evidence. KV4 denotes compact cache storage,
not conversion of the official FP8 model weights.

## Grouped compact reader candidate

`bf16_v2::read_grouped` uses one 192-thread block per query row and KV head.
Six 32-thread waves handle the six query heads sharing that KV head. One wave
decodes each token's 256 K and 256 V channels once into shared memory. The other
waves reuse those exact FP32 values. The protected BF16 path keeps the original
32-thread reader; validation and its host-side refusal remain unchanged.

Each head keeps its own query, product array, softmax state and eight per-lane
accumulators. The ascending 256-term dot sum, token order, FP32 operation
boundaries and final BF16 rounding remain unchanged. Four uniform per-token
barriers publish decoded values, products and softmax state, then prevent the
next token from overwriting values still in use. Shared storage is 14,408 bytes
per block. No new device allocation or cache format is introduced.

This reduces source-level compact loads/decodes by six for the reader, not for
validation or the whole model. It is not a sixfold performance prediction.
Fewer independent blocks, additional barriers, shared-memory traffic and register
pressure can offset the saved work. Native synchronization, occupancy, exact
outputs and speed require a fresh build and GPU comparison before adoption.

## CPU verification

Use the configured project interpreter and installed C++ compiler:

```sh
make test PYTHON=/absolute/configured/python CXX=/absolute/installed/clang++
```

`test-codec` checks 2,293,760 exact decoder comparisons, both nibble positions,
all scale bytes and channels, mixed adjacent records, signed zeros, subnormals,
overflow and canaries. It also retains the prior BF16, encoder and WHT checks.

`test-cache-validator` extracts the actual original and candidate HIP validator
bodies, then compiles them with explicit CPU thread/block stand-ins. It checks
6,448 baseline/candidate cases with read-only inputs, status canaries and guarded
buffer ends. Coverage includes every query position, every protected channel,
every packed nibble position, all scale bytes, malformed spans/indices, block
tails and supported-limit geometry. This is not native GPU execution.

Both C++ targets use AddressSanitizer and UndefinedBehaviorSanitizer, disabled
FP contraction and disabled fast math. The 112-case historical source suite
remains unchanged in size. Its historical native snapshot still uses the
original fixture; the current-candidate pin test binds the new source hashes.

`test-grouped-reader` compiles phases extracted from the actual original and
grouped reader bodies. A CPU scheduler executes all logical lanes between the
declared barriers, in ascending and reversed order. The test compares 46,885,056
intermediate FP32 values across 64 row/KV-head cases and 15,112 token steps. It
also checks complete BF16 outputs, input snapshots, output canaries and decode
counts. Removing barriers, changing head/dispatch bindings or changing a product
must fail. The arithmetic check uses the same sanitizers and strict compiler
flags. A separate declaration-only check parses the entire HIP source on the
host. Neither check establishes device compilation or GPU memory visibility.

## Native checks required before adoption

CPU equality does not establish HIP denormal behavior, GPU refusal behavior or
speed. Do not substitute the new source hashes into an old accepted build proof.

1. Create fresh candidate source/baseline bindings. Build and inspect with the
   existing `build_bf16_v2.py` interface and its complete 425-second internal
   envelope. Keep exact gfx1201 and strict numerical compiler flags. Compilation
   alone does not require a GPU, but requires the authenticated Linux SDK. The
   inspector must include `read_grouped` as the fifth required v2 kernel, with
   decoded instructions, a unique symbol and no unexpected matrix instructions.
2. Independently bind compiler, source, library, fatbin and device payload hashes.
   Keep original and effective runtime identities separate. Inspect register and
   private-memory changes rather than assuming that fewer source operations are
   faster.
3. In a fresh owned native test, compare old and candidate decoder bits and full
   attention outputs. Include protected and compact paths, signed subnormals,
   zero, boundary scales, noncontiguous indices, both heads and row lengths around
   31/32/33 and 255/256/257. Check malformed cache refusal and unchanged output
   canaries. Preserve admission and independent cleanup controls.
4. Only after native correctness passes, run a matched old/candidate performance
   comparison. Keep FP8 weights, requests, source identities and ownership limits
   fixed. Include long prefill and decode separately. Keep instrumented and
   uninstrumented timings separate.

A fresh composition, proof, spec and package must bind the new library before a
model benchmark. The consumed profile package must not be rerun. The unchanged
stock/BF16-KV runtime must remain available as the rollback path. No GPU speedup,
serving qualification or deployment promotion is claimed by this source change.

The cache-only controller pins the next comparison baseline to `3bf86042...`.
The full-model controller retains `d09bb064...` and its existing build proof.
Changing the standalone comparison baseline does not promote a model runtime.
