# Precision and ownership

Model weights stay in official FP8 form. The proposed cache layout is independent
of weight precision, although a different weight loader would need separate
integration and qualification.

The fixed Qwen configuration has 64 physical layers: 16 full-attention layers at
IDs `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`, and 48 GDN layers. TP2 gives
12 query heads and two KV heads per rank, with head dimension 256 and GQA six.

Physical layers `3,7,59,63` retain unrotated BF16 K/V. The remaining twelve full
layers use local G32 E2M1/UE8M0 storage with interleaved 17-byte groups. This
protection policy is a hybrid proposal, not a paper-qualified layer mapping.
Compressed Q/K use FP32 WHT after the model's native MRoPE; V stays unrotated.
The native ABI takes BF16 bits, unpacks locally and accumulates in FP32. It does
not reinterpret BF16 as FP16 or claim native FP4 arithmetic.

The private full-attention adapter is injected into the existing hybrid pool and
backend. Physical-to-compact IDs cross the existing parent writer boundary.
GDN allocation, state updates, snapshots and rollback remain separately owned;
recurrent state is not an ordinary KV cache.

Cache identity binds rank, process, model, source/build and cache policy. The
owner protects cached prefixes and validates slot generations before reads and
writes. Preparation failure invalidates stale metadata. Packed bytes cannot be
returned through an ordinary BF16 reader. Historical v1 Llama/TP1/FP16 guards
remain separate and unchanged.

Mixed KV payload is 14720 bytes per physical token per rank. A 4096-slot plan
therefore has 60293120 payload bytes, plus 14720 bytes for reserved slot zero and
a four-byte native status element. Add metadata, allocator costs and stock GDN
reservation separately. The 2 GiB workspace reserve is policy, not measured use.
The bounded Q/K/V contiguous-normalization temporary is at most 32 MiB.

Public mode registration and unsupported execution paths remain refused. A
source checkout, synthetic receipt or passing CPU fixture is not runtime proof
or launch authority.
