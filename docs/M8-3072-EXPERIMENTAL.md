# M8 3072 experimental selection follow-up

Status: all affected bounded screens passed; Astra approved experimental selection.
The 3072 stage-one source is selected for the next experimental session launch.

After the [Sprint 11 closeout](SPRINT11-PROJECTION-TUNING.md), the owner requested
promotion of the 3072 stage-one candidate, citing matching outputs and a
memory-channel jitter hypothesis. Assess it for experimental session use after
the affected bounded task and cache screens. The hardware explanation remains
unproven. Preserve the failed matched-work gates and the unresolved numerical
contract; do not relabel them as passed.

The selected candidate is `7c8fa3cb33169be35edd21553fa66f7d6d71f3799dc54767fad9fc53ce130d88`.
It adds stage one only for the exact untuned HIP/BF16 `(8,5120,3072)` shape.
It retains the selected M8 gate/up and prefill overrides and carries the same
metadata-only shape witness extension used in both benchmark arms. Keep the
current Torch/SDK stack, FP8 weights, BF16 attention caches, FP32 head/state,
snapshot, metadata reuse, DFlash8, pools, graphs and runtime flags.

## Retained benchmark meaning

All 12 output pairs matched across the three earlier 3072 comparisons. The
first forward comparison had one candidate request at 165 cycles versus 164;
the forward confirmation had one baseline request at 167 versus 164. Reverse
order matched at 164 cycles throughout. Identical emitted tokens can accompany
different speculative acceptance paths. The cause of this variation is unknown.

The candidate's measured 113.09–113.27 output tokens/s and roughly 1.7–1.8%
lower median latency remain descriptive combined TP2 results. The original
both-order matched-work performance requirement did not pass. This follow-up
does not rerun or discard those comparisons. Ordinary elapsed-time jitter does
not establish the cause of the algorithmic count differences.

## Affected bounded screens

Reuse the reviewed Sprint 10 candidate/candidate packets with the new source
in both starts. These compare task results against fixed oracles and retained
reference evidence; they are not contemporaneous quality A/B comparisons.

| Packet | Scope | Status |
| --- | --- | --- |
| Greedy | Four natural-stop tasks repeated across two starts; eight checks | 8/8 passed; all normal stops |
| Sampled | Eight natural-stop task checks with authenticated selector route | 8/8 passed; route verified |
| Lifecycle | 52 normal answers, two cancellations/recoveries, actual C2, reuse/branching/pressure and slot accounting | 52/52 answers passed; cancellation, actual C2 and slots passed |

Requests, grading, owner controls and limits remain unchanged. CPU preparation
passed 53 checks for each task packet and 65 for lifecycle. Readers retain the
existing decoding, source/native/HTTP and independent-release authentication.
The corrected sampled reader reports both repeated starts separately while
preserving the global all-request gate.

Astra reviewed the new experimental-use scope and found no technical blocker.
All six prepared folders matched the earlier sealed input inventory before the
fresh admission receipts and owner disposition were added. Fresh read-only
admission passed; source packets were then sealed. Earlier refusals and fixture
corrections remain in their evidence.

Greedy response authentication passed before the decoder refused a missing
tokenizer-reference metadata file in the prepared source. The reader now uses
the unchanged reference from its verified sealed donor, checks both donor and
local file hashes, and still checks the actual tokenizer bytes, closing marker
and full wire decoding in CPU isolation. The failed receipt remains. Refusal
tests cover local substitution and a wrong donor seal. No GPU rerun or native
evidence change was needed. The same metadata correction was applied to the
sampled reader before its readout.

All 68 normal answers passed across six starts and 70 requests, including two
cancellations. Both actual concurrent groups passed; each rank and start
recorded 41 batch-two replays and commits. All request slots released, with
eight free and 11 cached slots accounting for all 19 usable slots. Independent
release passed for all three campaigns. The three sources, native runs and
readers are sealed; all nine seals verified. The corrected greedy and sampled
reader suites passed 15 and 16 checks respectively.

Astra's final review found no blocking issue in the results, tokenizer-reference
correction, exact patch application or selection contract. The
[source patch](../patches/sglang-gfx1201-m8-3072-stages-serving.patch) applies on
top of the recorded Sprint 10 `cafc15b03fbb…` source and reproduces the tested
candidate exactly. Its SHA256 is
`515e4467908f1580e6a31b3a6c6dfe225482813e53ca3d645cc451b76fb98a0f`.
The [session selection](../provenance/CURRENT-SESSION-SELECTION.json) retains
that previous source and patch as `rollback_source_before_m8_3072`.

This experimental selection retains the failed performance-work gates,
unknown variation cause and unresolved numerical contract. Passing final
answers does not establish full-output/state or sampling-distribution
equivalence. Extended reliability remains deferred until tuning is exhausted.
Reopen diagnosis if concerning behavior recurs. No installed runtime,
persistent server or production default changed. See the
[evidence index](../provenance/M8-3072-EXPERIMENTAL-20261022.json).
