# OPT-01: host, runtime and communication audit

**CPU exit complete; independently reviewed by Opus 5.5.** No runtime settings changed, no GPU job ran and no performance gain is claimed. The work narrows the experiments worth paying for. The signoff preserves the sealed audit and carries its wording corrections separately.

The serving SDK's recorded source revision already contains HIP flat graph-packet dispatch. A backport is therefore not a useful candidate. Whether our graphs take that path, and whether queue submission is exposed, remain unmeasured. The source queue limit is four; this is not an observed queue count.

The workstation snapshot shows 24 physical cores, 48 logical CPUs, four L3 sharing groups and one NUMA node. Both GPUs report PCIe 5.0 ×16. The inspected SSH session and user ancestors show no CPU quota or recorded throttling. There was no serving process, so this does not establish serving-time placement or pressure. Active amd-pstate EPP uses `powersave` with `balance_performance`; that is not evidence of a fixed low clock.

Pinned SGLang automatic NUMA binding excludes ROCm. Rank affinity is disabled by default, and ROCr helpers can ignore the parent mask. No harmful placement has been demonstrated. Queue count, helper-aware placement and governor changes remain separate hypotheses rather than a combined tuning preset.

## Dispatch findings

The graph-source gap from OPT-00 is addressed at the CPU level. Scheduler construction captures the target and draft before readiness. The selected graph backend runs two warmups and one captured forward per batch size. Five CPU checks exercise the pinned batch selector and capture orchestration with inert callbacks, including an extra variant and a warmup failure. Physical route counts remain gated by the unchanged startup witness.

The draft route is more specific than the earlier lead label suggested:

- DFlash2 uses its selector graph path and the selected FP8 head proxy. The generic greedy sampler is a different branch.
- It already exchanges local top-16 scores and IDs across TP ranks, rather than full-vocabulary draft logits. The full local head projection still costs work.
- Compact draft-cache code exists, but its worker policy is opt-in. The frozen CLI does not set the window flag, whose default is None. The model's native 2048-token sliding attention window does not automatically enable that policy. This remains a CPU-first OPT-05/06 lead.
- The replicated embedding cache is conditional on DP attention, which this TP2 profile does not enable.

The current `no_buffer` validator requires overlap scheduling to stay disabled. The newer source includes HIP in its platform check for `extra_buffer`, subject to model checks; older blanket claims that HIP excludes it are not current evidence. A cache-strategy change would require lifecycle qualification.

Custom all-reduce stays disabled. PCIe-IPC has a separate gate: `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` defaults to false, and the cleared environment leaves it off. The custom-AR log line alone does not prove the collective route. Source routing identifies PyNccl during graph capture and the ordinary torch.distributed eager fallback, but does not measure the selected RCCL transport or exposed waits. The earlier FP8 prefill route is historical evidence, not a current MXFP4 measurement.

## Next work

One shared diagnostic design covers both ranks, C1/C2, submission gaps, CPU/cgroup behaviour, collective dependencies, draft/head/selector time and accepted-token work. The signoff adds an explicit environment allowlist, selector-tail witnesses on both ranks, window assignments and retention accounting. It is **not an executable packet**. Instrumentation, restoration, output caps and exact owner limits must be implemented and reviewed before approval is requested. Profiled durations will not be used as a serving-speed verdict.

Continue the inexpensive OPT-02 representation/GEMM screens. Keep metadata-copy work low priority and preserve all 21 original/Exa references. The [provenance](../provenance/MXFP4-OPT01-20261103.json) records the evidence, limits and review state.
