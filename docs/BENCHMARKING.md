# Controlled benchmark proposal

This proposal starts no process. The private TP2 smoke passed; public admission
and source promotion remain closed. Benchmark the exact sealed package and
native binary. Preserve official FP8 weights, TP2 and context 4096.

Use a 1800-second envelope: 120 seconds for source/device/model admission,
240 seconds for one cold load and warm-up, 900 seconds for matched measurements,
60 seconds for owned-process termination, 180 seconds for collection and
300 seconds for sealing. Stop measurements if the remaining cleanup and seal
allowance no longer fits. Refuse an unrelated device client; do not take over it.

First bind a stock BF16-KV control to the same common runtime and model. Its
different source/cache identity must be explicit. Do not bypass the KV4 proof to
turn the adapter off. Use isolated runs and the same flush state, request token
fixtures, sampling policy and eager scheduling. Unsupported graphs, speculation,
offload, LoRA and MTP remain out of scope.

Measure warm prefill and decode separately at prompt lengths 128, 1024 and 4096,
with up to 32 decode tokens where the context permits. Include one and two
concurrent requests. Use at least five matched warm samples when the envelope
permits; otherwise report the smaller sample count and incomplete cells. Record
median, range, tokens/second, time-to-first-token and inter-token latency. Exclude
load/JIT warm-up from warm timing but report their cost separately.

Run finite-logit, TP-agreement, prefix-reuse and release checks alongside timing.
Freeze model-level numerical and task-quality criteria before a comparison;
lossy KV4 is not expected to equal BF16 exactly. Do not infer quality from greedy
token equality. Keep a cold-versus-reused-prefix comparison separate from the
different-precision comparison. Record allocator measurements separately from
KV payload arithmetic, host-memory policy and the stock GDN reservation.

Report every measured win, including below 5%, with variability. Do not promote
a noisy single sample as a speedup. Default FP8 GEMM configuration warnings and
per-layer scalar/synchronization costs are follow-up tuning candidates, not
reasons to alter the weights or benchmark gates silently.
