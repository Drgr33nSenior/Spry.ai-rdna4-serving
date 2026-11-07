# MXFP4 GEMM wave-hint screen

Status: Opus 5.5 accepts the retained CPU result, with amendments below. The original gate
rejected the candidate. No GPU test, numerical equivalence or performance gain
is established. The owner subsequently accepted the revised CPU contract, and
the [three-case follow-up passed](MXFP4-GEMM-WAVE-HINT-CPU-RESULTS.md). This
original experiment and its rejection remain unchanged.

The selected packed GEMM passes `waves_per_eu=1`. All ten retained M8/M16
baseline binaries use 109 vector registers, but their descriptors encode a
larger allocation. This follow-up changes only that launch hint to the compiler
default, `0`, for the five frozen projection families. The kernel source,
quantizer, arithmetic, layouts and tile geometry are unchanged. Other shapes
keep the selected hint. The host change is an unattached candidate.

One bounded native CPU invocation reproduced all five down-M16 baseline
artifacts and compiled the candidate. It stopped at the first candidate gate.
The remaining nine candidate cases did not compile.

| Down-M16 compiler observation | Selected hint 1 | Candidate hint 0 |
| --- | ---: | ---: |
| Used vector registers | 109 | 90 |
| Used scalar registers | 41 | 41 |
| Descriptor VGPR encoding | 30 | 11 |
| Next-free VGPR field | 241 | 90 |
| Compiler-reported occupancy | 5 | 16 |
| Dynamic shared memory | 4,096 B | 4,096 B |
| Spills / fixed private bytes | 0 / 0 | 0 / 0 |
| Machine-code size | 8,756 B | 9,296 B |
| Instructions in the static K-loop body | 779 | 895 |

The frozen contract required identical machine-code bytes, equal resource
metadata and descriptor changes confined to the VGPR field. The candidate
fails all three requirements. That verdict and its executable comparator remain
unchanged.

Follow-up analysis found byte-identical TTIR and TTGIR. LLVM IR differs only by
removal of the `"amdgpu-waves-per-eu"="1,1"` attribute. The machine code has
different scheduling and register assignment. Its complete loop opcode-count
delta is 116 additional `s_delay_alu` instructions; WMMA, global/LDS transfer and
barrier counts are unchanged. The second descriptor change is the instruction
prefetch-size field, consistent with the larger function. The review addendum
checks the actual ELF function sizes, independently of compiler comments:
8,756 and 9,296 bytes give prefetch fields 69 and 73.

The whole function adds 137 delay instructions: 116 in the loop and 21 outside.
It also removes the terminal nop and early register-release message. Those
changes explain the full 540-byte increase. The latter can affect residency
while stores drain, so it is a performance risk. A register-abstracted
whole-function comparison, including wait instructions and their operands,
otherwise matches. This does not prove dataflow or device numerical equality.

The occupancy field is a compiler limit, not measured active waves or a speedup
prediction. Opus's capacity model gives about 160 resident workgroups under
the baseline limit, assuming the observed WGP mode and 32 WGPs. Only gate/up's
272 programs exceed that estimate; down/out have 80, qkv 112 and qkvz_ba 129.
This is an inference, not observed placement. The model supplies no residency
upside for those four smaller families. Additional hardware limits, dependency
delays, instruction size and DRAM saturation can offset any gate/up benefit.
No request-level gain is forecast.

The separately accepted next stage requires exact compiler IR, except for the one
hint attribute, and non-increasing resource use. The scope is now only three
cases: down-M16 as a negative control, then gate/up M16 and M8. It would allow
only explicitly listed backend differences, with whole-function and K-loop
instruction/operand comparisons, a gfx1201 target check, removed-padding check
and ELF-derived prefetch check. This is a new acceptance contract, not a passing
reinterpretation of this experiment. Its single invocation has now completed;
no further compilation allowance remains. Any GPU component requires its own exact approval,
unchanged numerical gates, matched 235 W controls and complete release.

The CPU owner decision (`GEMM-WAVE-HINT-CPU-OWNER-DECISION-v1.md`, private evidence)
contains the exact contract, three-case scope and finite limits. Its sealed
"awaiting" status is historical; the follow-up retains the owner's acceptance.

Validation: 12 parser/dispatch/recipe tests and three readout tests pass. The
required public preflight passes 43 checks. The readout reproduces the retained
result and authenticates all ten compiler artifacts and the native receipt.
Opus independently reran the 15 focused tests. Ten new local addendum tests
also pass, including mutations of waits, arithmetic, target and prefetch fields.
The one native invocation took 8.25 seconds including transport, exited cleanly,
reaped its child and left no owned process group. Its CPU sandbox asserted no
GPU device nodes, GPU initialization or loaded compiled kernel. No model starts
or GPU operations ran.

The [provenance record](../provenance/MXFP4-GEMM-WAVE-HINT-20261104.json) binds
the evidence, amendments and original CPU owner decision, with a link to the
separately accepted execution. The previous v6 weight-layout rejection remains
unchanged. The [bandwidth leads](MXFP4-EFFECTIVE-BANDWIDTH-LEADS.md), original
21-lead register and three cumulative bundle members remain recorded. The
serving bundle stays on HOLD, and the selected deployment is unchanged.
