# Status and evidence

Status: `EXPERIMENTAL_SOURCE_WITH_BOUNDED_VERIFICATION`.

The later FP8 thinking checks, previous/current stack HTTP benchmark and one
combined DFlash2/split K/V timing screen are summarised in
[current serving results](CURRENT-SERVING-RESULTS.md). They do not establish
general quality or qualify the four-bit cache prototype.

That status describes the KV4 prototype. The separate native FP8/BF16-KV path
passed bounded 4K C1/C2 compatibility with graph capture `[1,2]` on both ranks.
See [FP8 baseline](FP8.md) and
[`FP8-BASELINE-SUMMARY.json`](../provenance/FP8-BASELINE-SUMMARY.json).
This does not promote the eager KV4 composition, qualify 32K or establish a
matched performance winner. Published profiles are preparation inputs, not
transferable launch authority or evidence of a qualified deployment.

The source import is based on a sealed 1127-member private evidence root with
manifest SHA256
`c5b35b58c9795c3c20a2c7255b1d1eaaf538ea0aede61c5b6e5a004398a33289`.
That seal was verified on 2026-10-14. Raw archives, machine paths, private review
receipts and native binaries are not part of this repository.

## Historical physical observation

The complete private composition built and loaded the official pinned FP8 model
on two Radeon AI PRO R9700 GPUs. Both TP ranks completed these forwards:

| Stage | Greedy token | Logits |
| --- | ---: | --- |
| Initial prefill | 96752 | finite, shape `(1, 248320)` |
| Continuation | 271 | finite, shape `(1, 248320)` |
| Decode | 2 | finite, shape `(1, 248320)` |
| Prefix reuse | 271 | finite, shape `(1, 248320)` |

Both ranks agreed. The reused request matched 13 cached tokens. Available KV,
request and Mamba-state slot counts returned to `[4096, 2, 68]` on both ranks.
Workers exited zero, device clients were absent and all 80 pinned model files
were unchanged. This was one fixed-token correctness/lifecycle smoke, not a
quality, maximum-capacity or performance measurement.

The private installed package contained 5182 source files. Its source aggregate
was `1b10a4b7a1e83dbe2a76d36dd9d44e47315130a0c7432044dd7d174dc30b905c`.
The selected native library SHA256 was
`d09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171`.
The source subset here is not that complete installation or a transferable
accepted build proof.

## Retained limits

Earlier failed CPU fixtures, resource admissions, loaded-config checks and
compiler-path attempts remain in the private evidence. They are not relabelled
as passes. The final publication verification reports its own outcomes separately.

Default FP8 GEMM configuration warnings remain. Transport-to-request identity,
LoRA serving, graphs, speculative execution, offload and MTP are not qualified.
The new mode is not publicly registered. No production deployment, service
benchmark beyond the bounded later HTTP screens, model-quality comparison or
cold-versus-reuse numerical qualification
is implied by this source publication.
