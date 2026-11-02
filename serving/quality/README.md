# Final-answer quality and retained replay

This directory binds authenticated serving responses to an isolated CPU decoder
and grader. The private request corpus, tokenizer assets and answer oracles are
external inputs. For the completed 352-row result, see the
[cleanup sign-off](../measurements/native-qualification/CLEANUP-SIGNOFF.md).

## Follow the data

| Stage | Main source | Responsibility |
| --- | --- | --- |
| Project the menu | `final_quality_inputs.py` | Join authenticated oracle rows to configured ordinals, cases, body hashes and output caps |
| Recompute serving evidence | `retained_replay.py` | Recheck the actual source, HTTP, owner, resource and release records |
| Bind actual input | `bind_future.py`, `final_quality_inputs.py` | Retain complete output IDs and wire text from both validated arms |
| Prepare and grade | `prepare_packet.py`, `tokenizer_entry.py`, `quality_stages.py` | Materialize the declared input and run the approved isolated CPU stages |
| Verify completion | `cpu_readout.py` | Authenticate decoding, grades and independent CPU release |

`quality_contract.py` and `common.py` validate the menu, input and shared policy
used by these stages.

A complete serving operation and verified GPU release are prerequisites for a
conditionally approved CPU grade. Synthetic materialization fixtures have no
actual-input binding and no execution authority. The [caller workflow](../launcher/README.md)
shows where this continuation fits.

## Preserve answer meaning

- Natural rows require a normal stop and retain their authenticated answer
  oracle. Grade the final segment after the verified closing thinking marker.
- Cap rows may stop naturally or at the exact declared cap. They retain the
  complete decoded wire text but have no answer oracle or final-answer coverage
  target; their scope is `ungraded-cap-work`.
- An all-cap packet reports zero natural rows and cannot claim answer-quality
  success. Do not turn a truncated or reasoning-only response into a pass.
- Menu aliases and repeats use `source_plan=` from the authenticated menu.
  Check each frozen source ordinal and body pin; do not change prompt text,
  tokenizer vectors or oracles during projection.

The decoder must match the entire retained wire text for natural and cap work.
Counts follow the declared paired menu. The materializer and common policy
admit an even paired count up to 704; that capacity does not prove a larger
native menu fits the unchanged grading time budget.

The existing 53 MiB input limit, 512 KiB wire limit, tokenizer identity, marker
helper, precision flags and qualification flags remain in force. The W5 profile
retains 210-second work, 180-second decoder and 320-second total limits, with
its existing CPU, memory, process, namespace and transport bounds.

## Source and binding records

[PROPOSAL.json](PROPOSAL.json) and [SOURCE.patch](SOURCE.patch) retain the menu
projection's donor and proposed hashes. The materializer changes are in
[MATERIALIZATION.patch](MATERIALIZATION.patch) and
[MATERIALIZER-SOURCE.json](MATERIALIZER-SOURCE.json). The corrected CPU profile
is bound by [CPU-PROFILE-SOURCE.json](CPU-PROFILE-SOURCE.json).

The schema-2 actual-input binding records the declared paired count. The reader
also accepts retained schema-1 bindings for the original 352 rows. Both paths
reverify all four original serving/release seals and the complete input,
package, plan and recipe joins. Existing sealed packets keep their original
source and grading rules.

## Local checks

```sh
python3.14 -I -B -S serving/tests/test_quality_policy.py -v
python3.14 -I -B -S serving/tests/test_quality_materialization.py -v
python3.14 -I -B -S serving/tests/test_quality_cpu_profile.py -v
```

Run these from the repository root. Decoder fixtures use a CPU tokenizer
stand-in; they do not establish native token identity or run a native grade.

<details>
<summary>Historical W5 profile and replay corrections</summary>

These records explain the retained failed attempts and the later corrected
continuations. They do not authorize another invocation.

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

</details>
