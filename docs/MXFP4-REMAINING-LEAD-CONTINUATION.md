# Remaining leads: BW-04 CPU reference and diagnostic handoff

Status: Opus accepted v1 for CPU signoff. This fresh v2 contains parent fixes
for A1-A6, with focused verification; the delta has no independent review.
No GPU readiness, deployment change or new launch authority is claimed.

This continuation preserves all 21 original references and BW-01 through BW-04:
25 IDs in 18 distinct groups. RECONCILIATION.json lists each previous status,
current disposition and next evidence gate. It does not erase failed variants
or treat evidence-dependent deferrals as disproved.

The only new implementation is a local CPU reference for BW-04. No canonical
serving module, owner, observer, selected pointer or sealed input was changed.

## BW-04 implementation

source/image_layout.py provides two reversible layouts for integer transfer
texels: unchanged packed bytes and four bit planes over eight FP4 codes.
Each texel carries four bytes in a proposed R8G8B8A8_UINT format. The code
preserves every nibble, including negative zero, and leaves scale bytes intact.
A direct-read reference reconstructs one requested packed byte from one texel;
it does not expand a full weight tensor in VRAM.

Canonical metadata now records the layout mode. Both readers refuse a mode
mismatch, including legacy metadata without a mode. The full decoder refuses
inconsistent geometry, nonzero row/image padding and bad byte counts. Direct
reads assume the complete immutable image passed upload validation once. The
future upload receipt must bind the payload and its mode-bearing metadata. The default planning limit of 16,384 is a
supplied CPU limit, not a discovered device capability. Geometry fits both
retained 44,564,480-byte gate/up and down shard extents without transfer padding.
No full shard was packed or read in this stage. Tightly packed transfer bytes
do not establish VkMemoryRequirements, rowPitch, optimal tiling or VRAM use.

This is a reversible representation, not a software compression codec. It
has zero intrinsic payload saving. A useful result requires the hardware to
compress the data and the consumer to retain that benefit. Bit-plane inversion
adds instructions; its lowering, register/layout cost and latency are unknown.

Twenty focused layout tests pass, including all 65,536 combinations in byte
positions 0-1, every byte value in each texel position, signed-zero/code coverage,
all scale-byte codes, row/image tails, geometry refusal and direct reads. New
regressions refuse both cross-mode directions and missing-mode metadata. Eight
reconciliation tests pass, including latest-gate preservation, separate
cumulative-member coverage and dropped-ID/tampered-input refusals. The finite local receipts show clean exits, reaped children and
absent process groups. AST parsing passes. No framework import, native run,
new checkpoint read, driver call or GPU operation occurred.

## What Exa established

AMD's older [DCC overview](https://gpuopen.com/learn/dcc-overview/) describes
lossless compression and recommends coherent high-order bits for packed data.
Its RDNA/RDNA2 restrictions are not a description of every GFX12 path.

More relevant [AMD GFX12 SDMA source](https://github.com/GPUOpen-Drivers/pal/blob/dev/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp)
contains compression-mode handling for images and buffers, a default unsigned
32-bit buffer format, and comments about reading compressed data via a buffer
descriptor. Its default compressed-buffer mode permits decompression reads
while disabling compression writes. This is a concrete upstream buffer lead,
not evidence that the selected HIP/KFD allocations have DCC enabled.

The [Vulkan compression flags](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCompressionFlagBitsEXT.html)
distinguish default lossless behaviour from fixed-rate modes. The reference
does not opt into fixed-rate compression. DEFAULT disallows fixed-rate modes;
it does not establish that lossless compression is active. Exact UINT reads
must still be verified on the selected driver and format.

A [Linux GFX12 change](https://github.com/torvalds/linux/commit/9dff2bb709e6fbd97e263fd12bf12802d2b5a0cf)
disables dma-buf P2P for allocations carrying the GFX12 DCC flag. It is not proof
of the running kernel's behaviour. Any future prototype must keep each rank's
weight allocation local and qualify sharing rather than assume TP2 pointers
are interchangeable.

SOURCES.json pins retained Exa-normalised primary-source text. Those are
retrieval snapshots, not exact installed driver source. The legacy Exa MCP
endpoint refused authentication; the bundled Exa app worked. No standard-search
fallback or install was used.

## Next hardware gate, still unprepared and unauthorised

0. Select the allocation/consumer route first: a GFX12-DCC GEM BO, KFD/HIP
   allocation, dma-buf import or Vulkan image. Determine which route can keep
   weights compressed while the actual HIP consumer reads them. Image extent,
   format and the CPU image layout apply only to the image route.
1. Identify the running kernel, graphics driver/ICD, BO flags, alignment and
   supported allocation/import/descriptor controls. Upstream support does not
   establish the installed configuration.
2. Identify and authenticate the compressing initial-write path. The retained
   PAL default disables compression writes for compressed buffers; an upload
   that only decompresses or writes uncompressed bytes cannot prove the lead.
3. Prove compression attributes for the exact GPU-VA mapping and descriptor
   used by the consumer, from the installed driver construction path and
   authenticated allocation/mapping metadata. Upstream Linux
   gmc_v12_0_get_vm_pte sets AMDGPU_PTE_DCC when the BO carries
   AMDGPU_GEM_CREATE_GFX12_DCC. This supports checking both BO and PTE
   construction; it does not establish independently configurable per-mapping
   DCC or the selected HIP/KFD path. MAPPING-AUDIT.json retains the source join.
4. Establish exact reads of original weights and scales, all codes and tails.
   Validate the complete upload once and bind payload bytes, mode-bearing
   metadata, source identity and allocation/mapping/descriptor identities.
5. Measure active lossless compression and physical transactions against a
   matched compression-disabled control. Charge metadata, alignment, padding,
   any extra residency and write/setup work.
6. Measure consuming work directly, including inverse-layout instructions,
   occupancy and access cost. A staging round trip or full expansion is not
   an execution-bandwidth improvement.
7. Only a passing standalone test justifies interoperability and SGLang work.
   Rank-local allocations and any sharing need explicit qualification.
   No driver setting, kernel control or protected observer is changed here.

The required standalone GPU harness, hardware-counter route and finite owner
packet do not yet exist. No speedup, compression ratio or traffic reduction
is measured. Do not call BW-04 a fifth bundle member.

## Other leads and final diagnostics

The 20261104 status and next gate for every ID are now copied alongside the
20261103 status, with production_qualified=false. The original codecs and
compiler variants retain their closed screens.
Integer execution retains its production-rounding counterexamples.
Certified heads retain HOLD_NO_FIFTH_MEMBER. Recurrent-state, draft-cache/head,
host, TP2 and graph rewrites retain their source, lifecycle and exposed-cost
prerequisites. C4 address work still lacks a proved cheaper lane map.

Operation224 supplies bounded F1/GEMM exactness and capture evidence at rows8/16
on both ranks, correcting the earlier hypothetical T8 fallback at count 64:
the device API recorded 32 and no T8 fallback. Its serving result remains
INCONCLUSIVE, with unequal generated work in all six cases and long C2 vetoes.
No new single-member contribution is inferred from the combined run.
Verification prefetch remains an explicitly retained fourth-bundle constituent
alongside cap32, F1 and gate/up hint removal; it is not an extra 25-ID lead.
Local F1 checks do not close its separate collective-fusion branch. GEMM keeps
selected-wrapper provenance and production qualification obligations; L01/G2
keep all-shape proof and the prospective allocation-contract requirement.
Shortlist and compact-cache qualification stay separate.

The 18-group count is a planning taxonomy: BW-01/BW-02 extend the same storage
and integer work, while BW-03 certified full-head screening is distinct from
D1 heuristic draft shortlisting. It is not a count of measured gains.

FINAL-DIAGNOSTIC-SPEC.md records the owner's requested post-list investigation.
It separates controlled-work mechanism tests from natural-stop quality/latency,
profiler-clock validity, power/thermal state, speculation and acceptance,
CPU/cgroup/NUMA, TP waiting versus communication, and cache/state events.
It is a design handoff, not an implemented packet or authority to run one.

Reproduce from this fresh root with isolated Python and source/ on sys.path:

```sh
python3.11 -I -B -S -c 'import pathlib,sys,unittest; p=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(p/"source")); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(p/"source"),pattern="test_*.py")); sys.exit(not r.wasSuccessful())' "$PWD"
python3.11 -I -B -S -c 'import pathlib,sys,runpy; p=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(p/"source")); runpy.run_path(str(p/"source/reconcile.py"),run_name="__main__")' "$PWD"
```

The second command reproduces RECONCILIATION.json on stdout. Do not redirect
onto the sealed retained file. The finite local receipt also reproduces the
v1 mode-confusion negative control: wrong byte 80 versus expected 1; both fixed
readers refuse. The unchanged native/public suites were not rerun for this
CPU-only reference. The original Opus-reviewed seal stays immutable.
