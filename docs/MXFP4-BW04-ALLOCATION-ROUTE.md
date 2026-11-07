# BW-04 allocation route and retained serving-work census

Status: reviewed CPU/source stage accepted with a required follow-up; parent follow-up tested. Compression remains HOLD_SOURCE_CHAIN_AND_WRITER_PROOF. No GPU
allocation, runtime import, device open, kernel launch, model start or serving
change occurred. This is not a ready GPU packet.

## Compression route

A source-level candidate now exists: a rank-local GEM buffer with the GFX12 DCC
flag, exported through a file descriptor, imported with HIP's opaque-FD API and
read through the mapped HIP pointer. The source path does not require a Vulkan
image. The earlier reversible image layout remains an optional CPU reference;
it is not the required route and supplies no compression or speed measurement.

The retained SDK records TheRock commit 535d29907838d00afdb1382b700a0b1223ca942f.
The upstream gitlink at that commit selects ROCm-systems
f92b29c0d7d76c2cc9e771e877e3c7f9734c425a. Source tracing covers HIP import and
pointer creation, ROCclr external-buffer mapping, ROCr interop registration,
the thunk's dma-buf import, and the reference kernel's BO import/PTE path.
These are revision-pinned URL source witnesses normalised by Exa. They are not Git-blob verification or a rebuilt-binary proof.

One finite read-only workstation inventory ran under native Python 3.14.4.
All six selected SDK file hashes match: HIP API/version headers, HSA extension
header, amdgpu UAPI header, libamdhip64 and libhsa-runtime64. Nine direct calls exercise the guard function before its audit hook is installed. They do not test interception by the installed hook. No GPU device was opened and no runtime was loaded. The files were read from the retained runtime-sitepackages tree; matching bytes do not establish which /usr/local path a live worker loads. The caller exited cleanly in 0.264 s,
with the child reaped and its process group absent.

The installed kernel reports 7.3.0-rc6-3-git. The source reference is upstream
v7.3-rc6; its family matches but the exact custom kernel source is unproven.
The module srcversion is retained as an observation, not a Git revision.

The selected UAPI header declares AMDGPU_GEM_CREATE_GFX12_DCC and describes
recompression on moves into VRAM. The reference TTM copy path conditionally
sets WRITE_COMPRESSED for a DCC destination in VRAM and derives settings from
tiling flags, including WRITE_COMPRESS_DISABLE. This identifies reference writer fields. The SDMA v7 callback translates WRITE_COMPRESS_DISABLE into its write-compression mode. Installed dispatch, a deterministic move trigger and active compression remain unproven. A failed blit can fall back to CPU memcpy, so successful migration cannot authenticate that writer.

The eight original source witnesses and six additional reference witnesses pass. The complete chain is not proven. ROUTE-RESULT.json lists six named R1 preconditions: conditional same-device BO reuse; creation, metadata and export; a bounded move trigger; installed SDMA dispatch; CPU-fallback detection; and the installed KFD-to-PTE mapping chain. The next required work is still concrete:
prove the installed kernel/import/mapping path, construct a bounded initial
write that reaches that writer, verify exact reads through the intended HIP
consumer, and measure active compression, physical traffic and full consumer
cost against a matched disabled-compression control. Migration, alignment,
padding, extra residency and cleanup all count. Keep allocations rank-local;
no cross-GPU sharing is qualified.

No BO was created or moved. Neither active compression nor compression on
ordinary HIP allocations has been established. No kernel setting, driver
installation or privileged page-table access is proposed.

Sources:
- [TheRock gitlink](https://api.github.com/repos/ROCm/TheRock/contents/rocm-systems?ref=535d29907838d00afdb1382b700a0b1223ca942f).
- [Revision-specific HIP import](https://github.com/ROCm/rocm-systems/blob/f92b29c0d7d76c2cc9e771e877e3c7f9734c425a/projects/clr/hipamd/src/hip_memory.cpp).
- [Reference compression writer](https://github.com/torvalds/linux/blob/v7.3-rc6/drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c).
Exa retrievals and twenty-one normalised source snapshots are retained. Literal
source checks are conservative witnesses and do not execute or qualify C/C++.

## What the existing operation224 counters add

The source/HTTP replay already authenticated these response counters. The new
reader verifies the sealed workload and copied summary hashes, exact 30-request
order, row/request identity, proposal and histogram accounting, and reported
ratios. It excludes the three warmup requests and keeps C2 request slots separate.

| Request slot | Control verify rounds | Bundle verify rounds |
| --- | ---: | ---: |
| Long C1, cold and warm | 125 | 130 |
| Long C2 a, cold and warm | 130 | 163 |
| Long C2 b, cold and warm | 95 | 97 |
| Short C1 | 307 | 299 |
| Short C2 a | 308 | 298 |
| Short C2 b | 308 | 298 |

Counts are identical across all three repetitions for each listed case.
Cold and warm remain separate cases in WORK-RESULT.json. These are per-request
server verification counters; C2 counts are not added or treated as the number
of physical GPU batch iterations.

Warm cache-hit counts are unchanged: 118784 for C1 and 59392 for both C2 requests.
The bundle's long C2 a request produced 998 rather than 804 total tokens and
required more verification rounds. Long C1 instead produced fewer tokens (737 to 721) but required more verification rounds (125 to 130), reducing tokens per verification round from 5.896 to 5.546. Those counters describe different reasoning token streams, so they do not isolate an acceptance-efficiency change. A control-only diagnostic cannot explain the control/candidate divergence. Short requests produced 2244 rather than 2300
tokens and required fewer rounds. This is an observed work difference that can
mask or exaggerate component savings. It does not establish its cause or
attribute the change to cap32, scheduling, clocks or memory pressure.

Operation224 stays INCONCLUSIVE. No latency normalisation or alternative
tokens/s rule replaces the original 2% work and 3% balanced gates. Numerical
association and reasoning divergence remain hypotheses for targeted diagnosis;
a hidden system fault has not been established.

## Validation and next action

Twenty focused tests pass with an import/exec/socket audit hook, covering source/SDK/gitlink and retained-retrieval drift, inventory-source binding, missing checked source hops, canonical temporary paths, kernel-family changes, incomplete/foreign HTTP rows, malformed speculation
counts and non-finite ratios. Both analyses reproduce their retained output
byte for byte. The required public preflight passed 43 tests in five suites.
No public or serving runtime code changed.

DIAGNOSTIC-PREPARATION.md defines separate preparation tracks and unapproved budget references. BW-04 has no complete writer/measurement design and stops before hardware implementation. Neither is registered or executable yet.
Do not spend a GPU operation until the route/writer proof, compression measurement and owner boundaries are ready. Use the retained work census to reduce the next serving
diagnostic's scope before collecting anything new.

Reproduce with Python 3.11 -I -B -S, add this root's source/ to sys.path, then load test_analysis with unittest and call route_audit.run(root) and work_census.run(root). Use a resolved canonical TMPDIR. The final finite receipt retains the exact command and installs the audit hook. The fixture also explicitly tests a symlinked temporary parent. Twenty-one normalised source snapshots are retained.
The native inventory is single-use; do not rerun run_read_only_inventory.py.


Opus accepted the immutable v1 CPU stage with R1 as a condition. REVIEW-RESOLUTIONS.md records the parent follow-up for R1 and N1–N6. The follow-up itself has not had a second independent review. Historical native inventory and public43 receipts are reused unchanged; no new native or GPU invocation ran.
