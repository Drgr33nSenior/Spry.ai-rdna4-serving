# Shared FP8 head and packed projection views

This proposal shares one FP8 head allocation between the target and drafter.
It also rebinds the QKVZ and BA parameters to their packed buffers. It preserves
parameter objects, classes, loader metadata, compute frontends and per-call
head binding guards. P1 checks the recorded original identity and exact weight
and scale equality before accepting the new storage identity.

The post-load hook packs all 48 GDN layers, installs the target head, and then
lets the draft runner attach its proxy to the same FP8 tensors. It returns
unreferenced allocator blocks to the driver before graph capture. Token pools,
request limits, precision, sampling and owner ceilings stay fixed.

The head finite scan uses chunk-sized scratch and combines reductions on the
device before one scalar read. The drafter adds no persistent FP8 allocation.
The predicted tensor-payload reduction is 1,711,668,800 bytes per card. This is
an allocation calculation, not measured memory recovery or an inference speedup.

Instrumented startup also compares a transient second head quantization and
hashes the head and packed buffers. The precision witness joins these results
to matching target/draft tensor identities and all 48 packed layers. Its reader
rejects missing layers, wrong offsets, stale sharing and the old witness format.
Production skips the diagnostic quantization and hashes. It retains the exact
packed-shard equality checks and the serving guards.

The consumed package has 79 members under the existing 80-member ceiling.
Protected changes bind the exact new source pins in the transport and add the
shared helper to the controller inventory. The other seven protected files
retain their bytes. Missing, changed or substituted sources refuse admission.
The worker authenticates the helper source, callable code and globals before
loading Scheduler.

Source-only regression and packet checks use standard-library stand-ins. They
do not establish real tensor aliasing, native arithmetic or GPU replay. A
separate real CPU tensor test is prepared but remains unexecuted. Native
startup byte checks, greedy token identity, selected answer quality and memory
sampling by BDF also remain pending. Each GPU packet requires a fresh one-use
approval. This proposal supplies no execution allowance.

See [PROPOSAL.json](PROPOSAL.json) for source pins and
[VERIFICATION.json](VERIFICATION.json) for observed results and remaining gates.
Use the previous sealed package for rollback; do not edit an installed runtime.
