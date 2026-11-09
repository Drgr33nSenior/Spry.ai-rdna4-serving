# HRX upstream reassessment, 5 November 2026

Recommendation: refresh an isolated HRX research build after the current matched
profiling campaign. The upstream event fix justifies re-evaluation. This is a
source assessment, not approval to replace the active HIP runtime or run a new
GPU experiment.

Status, 9 November 2026: the HIP runtime and pins this assessment refers to
belonged to the SGLang 0.5.21 runtime that the 2026-11 workstation reinstall
wiped; they are historical. Current pins, including TheRock ROCm 10.3.0.dev20261108
(HIP 7.18), are in the workstation
[`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock).

Our earlier source build used HRX v0.1.0, commit
`d01d1c18b154189fd073777329052d23f6071f14`. Operation124 used the documented
`legacy-amdgpu` path. It passed exact model-free graph checks but showed no
GPU-side dispatch advantage on gfx1201. Host submission was cheaper, which
helps serving only when host work is exposed. The retained cleanup record says
the workstation build was archived and removed. This assessment did not check
the live host again. The SGLang packet of that time pinned the TheRock HIP
runtime.

The upstream snapshot examined is
`3c54fcbe3c71609d3c02f2b13daa3d7533fa525a`. This was a targeted review of the old
compatibility gaps and relevant changes, not an audit of the entire upstream diff.

- [PR2556](https://github.com/ROCm/hrx-system/pull/2556), merged 3 November,
  implements `hipEventRecordWithFlags`, including ordinary recording and explicit
  event nodes during capture. It fixes our reported
  [issue2486](https://github.com/ROCm/hrx-system/issues/2486). This removes the
  known source-level PyTorch event blocker; the selected binary and event paths
  still need testing.
- [PR2631](https://github.com/ROCm/hrx-system/pull/2631), merged 5 November,
  improves small fixed-argument recordings on the native AMD AQL path. Its fast
  path is bounded to small recordings; larger recordings retain device-side
  materialization. The upstream results are from MI300/MI210, not gfx1201. Our
  old legacy-path benchmark does not qualify this native path or its performance.
- [PR2660](https://github.com/ROCm/hrx-system/pull/2660), merged 5 November,
  moves native queue construction, shutdown and trim outside shared membership
  locks. It is relevant to concurrency and lifecycle work; it does not establish
  a steady-state SGLang throughput gain.
- [PR2498](https://github.com/ROCm/hrx-system/pull/2498) adds reusable command-buffer
  timestamps. The related Loom compiler
  [PR2504](https://github.com/ROCm/hrx-system/pull/2504) was still open when checked;
  do not treat that entire source-authored measurement path as available.

Two old RCCL compatibility risks remain in the pinned
[HIP source](https://github.com/ROCm/hrx-system/blob/3c54fcbe3c71609d3c02f2b13daa3d7533fa525a/libhrx/src/binding/hip/abi_stubs.c):
`hipMemGetHandleForAddressRange` still returns `hipErrorNotSupported`, and
`hipMemcpyBatchAsync` refuses nonzero attribute counts. Whether our single-node
TP2 route calls these functions remains unproven. Exporting a symbol alone does
not establish compatible execution.

For a separately requested follow-up, pin a fresh source/build beside the
selected stack. Check event semantics, the actual RCCL call path, graph replay
and exact outputs before serving. Name the native or legacy backend explicitly;
do not silently compare different runtime routes. Repeat the graph microbenchmark
only under a fresh finite approval, then consider serving if the results warrant
it. Use the current profile to decide whether dispatch/queue overhead is exposed
enough to justify that spend. Loom compiler fixes do not automatically change our
Triton kernels.

No source copy was updated, no dependency was installed or built, and no remote,
GPU, model, selection or Git mutation occurred. Exa supplied upstream discovery
and PR pages; the GitHub connector supplied pinned commits and source. Historical
HRX evidence remains unchanged.
