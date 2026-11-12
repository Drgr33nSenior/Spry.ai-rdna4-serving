# Agent briefing

Read this before changing anything in this repository. It says what the
repository is, what is live today, and the rules that protect the record.

## What this repository is

Source, tools and research for serving Qwen3.8-27B (MXFP4 weights) with SGLang
on two AMD Radeon AI PRO R9700 GPUs (gfx1201). It holds:

- the [`sglang-main` patch series](serving/patches/sglang-main/README.md) that
  makes SGLang run well on gfx1201;
- the vendored [BetterBench client](benchmarks/betterbench/native-v2/README.md)
  used by the throughput bench;
- GPU check tools in `tools/`, such as `check_fp8_kv_dequant.py` and
  `bench_int4_kv.py`;
- dated research records and sealed evidence from earlier runtimes.

This repository deploys nothing. The workstation repository,
[ArchLinuxThreadripper](https://github.com/Drgr33nSenior/ArchLinuxThreadripper),
builds the image, owns the serving profile and runs everything on K3s. The
[README](README.md) shows how the two fit together.

## What is live (12 November 2026)

| Item | State | Source |
| --- | --- | --- |
| Stack | SGLang `main` `fcb32746` (0.5.23.post957) plus this series, patchset `142d0ae7…`; PyTorch 2.16.0a0; ROCm 10.3.0-dev (2026-11-08), HIP 7.18 | Workstation [`stack.lock`](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/stack.lock) |
| Serving | TP2 on two R9700 at 235 W; MXFP4 target, DFlash2 draft; FP8 KV read as bf16 (patches 0026/0027) | [Serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json) |
| Qualified arms | Y (BF16 KV) on 2026-11-10; UQS and UQL (FP8 KV) on 2026-11-11 | The profile's `qualification` list |
| Modes | Shared mode runs arm UQS; large mode runs arm UQL | Workstation [modes guide](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/docs/MODES.md) |
| Parked | INT4 KV research (UQ3, K8V4), parked 2026-11-12 | [INT4 verify-kernel record](docs/UQ3-INT4-VERIFY-KERNEL-20261112.md) |
| Next | Bound the DFlash draft KV pool through upstream SGLang [#47058](https://github.com/sgl-project/sglang/issues/47058) | Owner decision, 2026-11-12 |

The owner's 2026-11-12 decision to follow upstream #47058 replaces the route in
the [11 November draft-pool plan](docs/DFLASH-BOUNDED-DRAFT-POOL-PLAN-20261111.md).
That plan's capacity arithmetic is still useful. Its UQ3 assumption is not,
because UQ3 is parked.

Everything built on the earlier SGLang 0.5.21 runtime ("op234" and older) is
history. The November reinstall wiped that runtime. It cannot be rebuilt from
this repository. See the [history index](docs/HISTORY-INDEX.md).

## The immutability rule

**Never edit a sealed or dated record. Put corrections in a new record.**

This covers:

- `provenance/`, `serving/measurements/` and `serving/proposals/`;
- every file whose name carries a date, operation or sprint, such as
  `docs/SPRINT*`, `docs/MXFP4-*`, `docs/FP8-*`, `docs/UQ3-*` and
  `docs/*-2026MMDD.md`;
- the historical logs `docs/CURRENT-SERVING-RESULTS.md`,
  `docs/CURRENT-COST-ANALYSIS.md` and `docs/AGENTS-LOG-20261112.md`;
- any private evidence root with a seal manifest.

To mark a record as superseded, add a line to
[docs/HISTORY-INDEX.md](docs/HISTORY-INDEX.md). Do not move or rename records,
because links and provenance files point at them. Tests must write only into
fresh fixtures and must refuse to write into sealed roots.

## Working rules

1. Preserve uncommitted work. Do not stage, commit, merge or push unless asked.
2. Run Python checks with `-I -B` (isolated, no bytecode), as the Makefile does.
3. After any change to `serving/patches/sglang-main/`, recompute the patchset
   hash and update `SGLANG_PATCHSET_SHA256` in the workstation `stack.lock`.
   [VERIFICATION](docs/VERIFICATION.md) gives the steps.
4. Keep every `SGLANG_GFX1201_*` switch off by default. A new patch is enabled
   only through a profile arm, after a bench Job and the quality menu.
5. Compare speed only at the same power cap and on the same image. Say
   "configured 235 W cap"; do not infer sustained draw from the limit.
6. Report a speed claim with its quality evidence. BetterBench has no
   answer-quality oracle, so pair it with the quality menu.
7. Treat instrumented or forced-length timings as diagnostic, not serving
   results.
8. Do not reopen a rejected idea unless new evidence is recorded in a new record.
9. A new model, runtime or shape needs fresh qualification. Do not copy old
   gains into its record.

## Where to read next

- [README](README.md): what the project is, with diagrams.
- [Documentation guide](docs/README.md): reading paths and the glossary.
- [Patch series](serving/patches/sglang-main/README.md): what each patch does
  and which arms use it.
- [History index](docs/HISTORY-INDEX.md): every older record, with
  "superseded by" pointers.
- [Agent log to 12 November 2026](docs/AGENTS-LOG-20261112.md): the previous
  AGENTS.md, unchanged. Its links are relative to the repository root.
- [Serving-validation skill](.agents/skills/aithreadripper-serving-validation/SKILL.md):
  the evidence workflow for the earlier runtime.
