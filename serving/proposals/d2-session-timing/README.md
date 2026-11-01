# D2 proposal: configured session timing

This isolated proposal extends [owner timing](../d2-owner-timing/README.md) to
the transport and controller. [D2.patch](D2.patch) changes those two protected
helpers. [PROPOSAL.json](PROPOSAL.json) pins the resulting nine-helper set,
including the preceding owner adapter. The active source builder still emits
the original protected sources. Complete D2 review and launch approval remain
separate gates.

## Behavior

The transport's immutable `SessionTiming` derives worker, arm, active and total
time from authenticated configuration primaries. The controller uses the same
class and constructs its owner plan from that session plan. Tests compare the
arithmetic with the canonical source builder's `spry_common.limits.derive`.
No package member or dependency is added; the archive remains at 80 members.

The baseline retains a 6,060-second active budget and a 6,360-second total
ceiling. Preparation, complete admission, locked admission, collection,
campaign margin and sealing retain their exact W5 values: 105, 180, 75, 330,
390 and 300 seconds. The 390-second margin remains an accounting remainder.
Worker and arm ceilings, minimum startup/cleanup/release reserves, signal
offsets, observer phases and release validators are preserved.

The transport checks that the manifest total equals the authenticated plan.
It creates a matching grant and calculates active/seal deadlines from the
original receipt. Before workload creation and controller start, it checks
the original preparation window and the complete remaining owner budget.
This replaces the stale 1,995-second check with a derived 5,385-second W5
requirement. Bootstrap failure uses the configured seal deadline once known;
before configuration is authenticated, it uses the original receipt plus the
protected maximum. Neither path restarts the clock.

The controller requires exact integer grant totals and reserves, verifies their
agreement with configuration, and passes the derived active deadline to the
owner. Its admission check and report use the verified plan. Sealing derives
its deadline from the same original receipt and revalidates the plan before
writing. Invalid timing, future receipts and expired sealing windows refuse
before seal publication.

The internal `new_grant` API now requires `timing`. The controller's
`seal_result` API takes `timing` instead of an independently supplied deadline.
Historical packets retain their original callers and helpers together.

## Verification and limits

The public suite checks baseline and shortened arithmetic, strict types,
ceilings, fixed reserves, configuration/spec binding, archive/grant agreement,
expired entry refusal and actual local seal publication. The human-entry test
uses synthetic identity and boot observations and proves refusal before
workload creation, staging or process launch. No privileged entry is executed
with real host observations.

Private checks authenticate all 313 donor members and import fresh copies with
native imports blocked. They combine the new transport and controller with
the preceding owner adapter. The archive/grant checks, actual owner success
and failure cases, and complete controller sealing use these full modules.
The controller argument check substitutes the prepared pre-launch result and
scope arguments, then stops at the campaign boundary before lease acquisition.
Native pre-launch still stops at the absent external runtime build proof.

The expected-response and retained-envelope replays use the same explicit
spec-hash fixture adaptation as the
[spec-binding proposal](../d2-spec-binding/FIXTURE-ADAPTER.patch). Controller
functions and spec hashes are not substituted during post-run replay.

```sh
python3 -I -B -S serving/tests/test_session_timing_proposal.py
python3 -I -B -S serving/tests/check_session_timing_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
make test
```

Logs are under `.test-build/serving-cleanup/session-timing/`.
[VERIFICATION.json](VERIFICATION.json) records 31 public tests, 17 private
checks and the passing full `make test` run, including 225 cleanup tests.
Both 176-row arms pass expected-response and retained-envelope replay, with
length stops refused. The minimum measured file-count margin remains 1.59
times, below the required two times.

Shorter plans are exercised at individual consumer boundaries. The complete
controller still requires the pinned W5 run, so whole-run mutations remain
unsupported. Next work is worker/client timing and its readout checks, then
artifact limits, stop policy, observed device/rank binding and fresh packet
identity. Instrumented health receipts, all-cap/doubled-menu evidence, native
validation and private publication remain open.
