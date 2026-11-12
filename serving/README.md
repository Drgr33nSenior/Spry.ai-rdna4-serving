# Serving source (historical 0.5.21 cleanup)

**This tree is not what serves today.** Apart from
[`patches/sglang-main/`](patches/sglang-main/README.md), everything under
`serving/` is the cleanup source for the SGLang 0.5.21 runtime. The November
2026 workstation reinstall wiped that runtime, so this source cannot be run as
documented.

What is live instead:

- The SGLang patch series in [`patches/sglang-main/`](patches/sglang-main/README.md).
- The workstation's
  [serving profile](https://github.com/Drgr33nSenior/ArchLinuxThreadripper/blob/main/infrastructure/ai-stack/serving/qwen3.8-27b-mxfp4-tp2.json)
  for flags, and its K3s bench and check Jobs for validation.
- The [project README](../README.md) explains how the two repositories fit
  together.

The rest of this page describes the cleanup source as it was on 2 November
2026. The [history index](../docs/HISTORY-INDEX.md) lists the related records.

## Result of the cleanup (operation 207)

Operation 207 completed the reviewed production-mode packet on two R9700 GPUs.
All 352 request and output token vectors matched B186; all 352 natural-answer
grades and long-answer coverage passed. Original host acceptance and
independent GPU/CPU release passed.

The fixed corpus has eight distinct prompt-message sequences in 176 cases per
start. The result establishes this bounded validation. It does not establish
general model quality, endurance or a deployed service.

See the [sign-off](measurements/native-qualification/CLEANUP-SIGNOFF.md),
[measurement report](measurements/native-qualification/GPU-207-MEASUREMENT.html)
and [machine-readable progress](CLEANUP-PROGRESS.json).

## Choose a task

| Task | Guide or source |
| --- | --- |
| Understand the repository and runtime boundaries | [Architecture](../docs/ARCHITECTURE.md) |
| Change a configured run | [W5 run configuration](config/runs/w5-v2/run.json), its menu, and the referenced host file |
| Assemble source for review | `build.py`, `propose_packet.py` and [caller projection](launcher/PROJECTION.md) |
| Find the current SGLang `main` patch series | [`sglang-main` series](patches/sglang-main/README.md) |
| Reproduce the historical 0.5.21 overlays | [Patch series](patches/README.md) |
| Understand a one-use native campaign | [Caller workflow](launcher/README.md) |
| Check or extend validation | [Testing procedure](measurements/native-qualification/TESTING.md) |
| Understand final-answer grading | [Quality guide](quality/README.md) |
| Trace an earlier decision or failure | [History](HISTORY.md), [proposal index](proposals/README.md) and the linked result records |

## Where the maintained source lives

| Path | Responsibility |
| --- | --- |
| `config/` | Run, request-menu and host configuration, with closed schemas |
| `build.py`, `propose_packet.py`, `prepare_gpu_run.py` | Deterministic source review and fresh packet preparation |
| `spry_serving/` | Worker, model, runner and memory-management code |
| `spry_evidence/` | HTTP receipt handling, observers and retained-result validation |
| `spry_common/` | Shared serialization, guards and limit derivation |
| `launcher/` | Authenticated one-use caller and power-observation source |
| `quality/` | Actual-input binding, isolated CPU grading and retained replay |
| `upstream/`, `patches/` | Exact 0.5.21 input identities, the historical 0.5.21 overlays and the current `sglang-main` series |
| `proposals/` | Versioned increments, source selections, diffs and verification records |
| `measurements/` | Published summaries and links to their retained evidence |
| `legacy/` | Retained source used by the base builder and compatibility checks |

A proposal selects exact source versions. Similar filenames in a retained
increment are not interchangeable: use its `PROPOSAL.json`, `SOURCE.json` and
recorded hashes. The base builder and the later proposal/packet preparers have
different responsibilities; a successful base build alone does not select the
final cleanup composition.

## Source review and native work

Source preparation runs on CPU. It authenticates a sealed donor, reads the
chosen configuration and source selection, and writes a fresh review tree.
Private corpus assets and accepted runtime/build evidence are separate inputs.

Inspect the preparation interfaces without creating a packet:

```sh
python3.14 -I -B -S serving/build.py --help
python3.14 -I -B -S serving/propose_packet.py --help
python3.14 -I -B -S serving/prepare_gpu_run.py --help
```

Use canonical absolute input paths and a new output directory for real
preparation. Preserve every failed attempt and sealed root. The later native
caller consumes an exact reviewed source, boot binding and one-use approval;
these help commands supply none of those inputs.

## Verify locally

From the repository root, use the configured interpreter:

```sh
make test-serving-cleanup PYTHON=python3.14
make test-validation-preflight PYTHON=python3.14
```

The public checks use the standard library and declared CPU substitutes. They
do not start a model or contact the workstation. Exact packet checks, native
interpreter checks and retained real-artifact replay are separate gates. The
[test procedure](measurements/native-qualification/TESTING.md) gives their order
and explains when a passing check must be repeated.

## Reading the evidence

Production mode omits graph and observer diagnostics. Instrumented mode retains
those diagnostics for its declared evidence packet. Read a result under its own
mode, source pins, profile and request menu.

The completed cleanup combines both production starts with retained first-start
GPU byte witnesses and the corrected CPU readback. Operation 206 remains a
failed partial run; it is not relabelled as accepted. The sign-off records the
absence of a second-start instrumented witness set and the limits of the
descriptive memory and timing comparison.

## Remaining gates

The declared cleanup scope is complete. No cleanup step or native allowance
remains. New experiments need their own reviewed plan and applicable approval.
Read older “remaining gates” in the [history](HISTORY.md) as statements about
those earlier increments.
