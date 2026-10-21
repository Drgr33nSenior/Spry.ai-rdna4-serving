# Native cache comparison

This test compares the accepted and candidate cache libraries on both R9700
cards. It does not load a model or convert FP8 weights to FP4. `bf16` and `kv4`
name the two cache formats, not model weight formats.

The installed serving runtime remains unchanged. Each library runs in a separate
process. The test uses the existing host lock, resource gate, root observer and
independent release checks. A human starts the root observer with the generated
one-use `RUN-ON-WORKSTATION.sh`; an agent must not invoke that wrapper or sudo.

## Legacy test contract

The order is BF16 cache, then KV4 cache. Each arm runs the baseline and candidate
on device 0, followed by the baseline and candidate on device 1. This gives eight
native processes and zero model starts.

Each process checks the actual HIP UUID, PCI address and gfx1201 architecture
before loading a library. Exactly two visible devices are required. Environment
variables that override device visibility or architecture are refused.

The native fixture checks 8,192 decoder cases per process. Attention coverage uses
12 query heads, two KV heads and head size 256, with noncontiguous cache slots.
Contexts include 1, 31, 32, 33, 255, 256, 257, 1,024 and 3,966 tokens. The BF16 arm
checks 18 output rows and 35 refusals. The KV4 arm checks 19 rows and 51 refusals.
Invalid inputs must leave output canaries unchanged. Explicit device buffers are
limited to 64 MiB; this accounting excludes HIP and library internal allocations.

For each device and cache format, the collector compares the complete retained
binary records byte for byte. These include appended cache data and attention
outputs. It records their hashes and comparison counts before removing only its
own temporary binary buffers from the private output filesystem. It retains
child summaries, stdout, stderr, process identities and supervision results.

After its correctness checks, each process records five synchronized attention
call times at lengths 128, 1,024 and 3,966. These calls include validation and
attention. Timings are usable only when all old/candidate comparisons and cleanup
checks pass. Fixed baseline-first ordering can introduce thermal or clock bias.
These measurements do not establish model throughput or end-to-end speedup.

## Counterbalanced measurement candidate

The explicit `native-cache-counterbalanced-v2` mode keeps the same libraries,
cache inputs, shapes, devices, identity checks and cleanup. It changes only the
measurement protocol. Legacy v1 packages retain their original behavior.

Each cache arm runs baseline, candidate, candidate, baseline on each card in
four fresh processes. Both cards still run sequentially, not as concurrent
TP2 workers. The full schedule has 16 native processes and zero model starts.
The source-bound spec requires `timing_protocol=warm-synchronized-output-v2`,
`timing_warmups=2`, five measured repeats, and the exact four-variant order.

For each length (128, 1024 and 3966), the fixture retains one reference output,
two explicit warmup outputs and all five measured outputs. Each output must be
finite, preserve canaries and match its reference exactly. The collector also
compares complete output records from all four occurrences on the same card.
It retains one reference binary and one current binary at a time, below the
existing 64 MiB output-filesystem bound. The normal byte comparison then removes
only these owned temporary buffers. A failure retains available partial files.

Output reset and a device synchronization occur before each timer. The timer
covers the validated attention API call and its final device synchronization.
Status checks, output download, comparison and persistence occur after timing.
Those operations can still affect the next call's cache/history. The result is
whole-call latency, not isolated kernel latency or model throughput.

The offline `tools/analyze_cache_benchmark.py` reader keeps all 240 timing samples,
per-occurrence medians/ranges, both AB and BA adjacent comparisons, and later-to-
earlier drift for each variant. Authenticate the retained archive separately
before using this readout. Two occurrences per variant are not ten independent
experiments. ABBA does not remove nonlinear thermal or history effects. Do not
subtract the BF16-control improvement as a causal correction.

The proposed v2 envelope is 3000 seconds, including 300 for sealing. Each arm
retains eight complete 105-second child envelopes inside 900 seconds of work,
then 30 seconds of termination/collection and 75 seconds of release observation.
Admission, independent postchecks and root transport remain mandatory. Existing
v1 and full-model envelopes remain 2400 seconds; arbitrary ceilings are refused.

This candidate needs a newly compiled fixture and a fresh source-bound package
before a human launch. Do not use an old fixture hash or rerun the consumed
grouped-reader package. No new native timing result follows from CPU tests.

## Preparation and limits

Compile with the authenticated Linux HIP SDK, strict numerical flags and
`--offload-arch=gfx1201`. The compiler runs without GPU devices. Bind the source,
compiler, executable, both libraries and retained runtime in a fresh package.
Do not reuse a consumed package or substitute hashes in historical evidence.

The native owner retains the existing 2,400-second complete envelope, including
300 seconds for sealing. Each cache arm has a 480-second work limit within a
585-second envelope. Each native child has a 105-second complete envelope.
Resource controls remain 24 CPU, 38 GiB RAM, no swap and bounded private temporary
filesystems. The owner refuses conflicts and unresolved ownership records.

Run offline checks with the configured interpreter:

```sh
make test-cache-gpu PYTHON=/absolute/configured/python
```

This target tests the actual fixture source with declaration-only HIP substitutes
and exercises the owner contract with explicit CPU substitutes. It never runs a
GPU workload. Native execution still requires the prepared human launcher.

If a native test fails, preserve the evidence and unresolved ownership record.
Do not rerun its one-use package. A passing native test permits assessment of a
fresh matched model benchmark; it does not promote the candidate or qualify
serving.
