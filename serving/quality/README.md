# Configured quality source

These three modules come from the authenticated W5 CPU quality template. The
private request corpus and answer oracles remain outside this source directory.
[PROPOSAL.json](PROPOSAL.json) records the exact donor and proposed source hashes;
[SOURCE.patch](SOURCE.patch) records the changes.

`project_menu(private_rows, request_rows)` joins authenticated private oracle rows
to configured request ordinals, cases, body hashes and output caps. Both arms use
the same plan. Natural requests retain their answer oracle. Cap requests have the
`ungraded-cap-work` scope and no answer oracle or final-answer coverage target.
The caller must authenticate both inputs before calling this pure projection.
For aliases and repeated menus, pass `source_plan=` from the authenticated run
menu. The producer checks each frozen source ordinal and body pin before it
projects the resolved case, body hash and output cap. It does not change prompt
text, tokenizer vectors or answer oracles.

`build_input(arm_results, menu=projected_menu)` retains the complete token vector
and wire text from both recomputed serving readouts. Natural requests require a
normal stop. A cap request may stop naturally or end at its exact declared cap.
Both statuses remain ungraded for cap requests.

The decoder still requires the complete native decode to match every retained
wire text, including cap work. It does not split or grade cap answers. Answer
validation rejects answer hashes or coverage claims on those rows. An all-cap
packet reports zero natural rows and cannot claim answer-quality success. Row
counts follow the declared menu, including a doubled menu.

The 53 MiB input limit, 512 KiB wire limit, tokenizer identity, marker helper,
precision flags and qualification flags retain their original bounds. The
all-natural 352-row input, decoded answers and grade report remain byte identical
to the retained implementation in the CPU comparison.

The configured proposal assembles these sources into CPU packets for the natural
baseline, all-cap menu and doubled menu. `prepare_packet.py` derives the row count
from the validated input. `common.py` admits an even paired count up to 704 and
joins the declared count to the authenticated expected menu. The decoder work,
transport, memory, swap, process, input and report ceilings remain unchanged.
The materializer and common-policy changes are in
[MATERIALIZATION.patch](MATERIALIZATION.patch); their donor pins are in
[MATERIALIZER-SOURCE.json](MATERIALIZER-SOURCE.json).

`bind_future.py` writes a schema-2 binding with the actual paired count. Its
consumer also accepts retained schema-1 bindings for the original 352 rows. Both
paths reverify the four original serving/release seals and the complete CPU
input/package/plan/recipe joins. Synthetic materialization fixtures have no
actual-input binding and grant no execution authority.

Focused decoder tests use a CPU tokenizer stand-in. The packet checks exercise
the actual materializer without running a native tokenizer. They do not establish
native token identity or prove that a larger menu fits the unchanged native CPU
time budget. Existing sealed grading packets retain their original sources.
## Native CPU profile correction

The first cleanup baseline completed 352 responses and passed the complete
owner/native/HTTP/release replay. Native CPU decoding then refused the inherited
template's obsolete 75-second entry limit. Its stage producer also retained a
45-second child while the W5 reader expected 180 seconds, and the transport
reader still used an older 180-second total limit.

`tokenizer_entry.py`, `quality_stages.py` and `cpu_readout.py` now come from the
authenticated W5 CPU packet recorded in [CPU-PROFILE-SOURCE.json](CPU-PROFILE-SOURCE.json).
The three files are byte-identical to that previously validated source. The
renderer includes them with the existing 210-second work, 180-second decoder
and 320-second total limits. CPU quota, memory, namespace isolation, quality
rules and tokenizer source checks retain their existing W5 values.

The local replay adapters now derive the legacy active deadline from its actual
transport constants and register the quality contract before loading its
producer. The actual baseline's full replay and 352-row CPU input binding pass.
Five focused profile checks pass on macOS CPython 3.14.8 and native Linux
CPython 3.14.7, without native imports or GPU access. These checks do not grade
the new responses. The failed CPU attempt is retained and independently
released. The separately approved continuation subsequently decoded all 352
responses, passed all 352 natural-answer checks and required long-answer
coverage, and validated independent CPU release. The refreshed cleanup B
template had the same grading inventory. Its complete configured replay exposed
two local adapter errors: a missing authenticated spec hash and a typed artifact
plan converted to a dictionary before the artifact receiver consumed it.

The corrected reader passes the spec hash to the unchanged resource validator.
It retains the typed artifact plan for extraction and projects the dictionary
only for HTTP readout. Both complete B replay and legacy A replay pass. The
legacy CPU input remains byte-identical. Failed offline attempts remain sealed;
neither dispatched native CPU work. Native grading sources, quality rules and
resource limits remain unchanged. B's one approved CPU grade subsequently
decoded and passed all 352 natural answers and required long-answer coverage,
with independent CPU release verified. See the
[cleanup result](../measurements/gpu-run/CLEANUP-B-RESULT.json).
