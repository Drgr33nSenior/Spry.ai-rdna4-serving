# D2 proposal: configured owner timing

This isolated proposal makes the owner adapter consume timing from the verified
W5 configuration. It extends [spec binding](../d2-spec-binding/README.md).
[D2.patch](D2.patch) changes the controller and owner adapter;
[PROPOSAL.json](PROPOSAL.json) pins both sources and the composed protected set.
The source builder still emits the original nine protected helpers. This
proposal is not a launch packet, approval or completed D2 migration.

## Timing contract

The controller creates an immutable `CampaignTiming` after it authenticates the
run and reproduces the complete spec. The adapter requires this explicit plan,
copies and revalidates it before lease acquisition, and requires its worker
time to match the spec. Derived work and envelope values cannot be supplied
independently. The controller reports the plan's actual reserves.

- Worker time is at most 2,280 seconds.
- Owner cleanup is at least 150 seconds; worker plus cleanup is at most 2,430 seconds.
- Arm release is at least 60 seconds; the complete arm envelope is at most 2,490 seconds.
- Locked admission remains 75 seconds and final collection remains 330 seconds.
- TERM, KILL and reap deadlines remain 20, 25 and 30 seconds after the work deadline.

The adapter retains its remaining-budget checks before admission and each arm.
The lease, journal, release validators and independent release checks are
unchanged. The new required `timing` argument replaces the old envelope and
collection keyword arguments; the controller supplies it. Historical packets
continue to use their original controller and adapter together.

The supported run remains pinned to W5. A shorter plan can tighten this
adapter's deadlines, but a changed whole-run configuration still refuses.
Transport, controller admission/sealing, worker/client and artifact limits
must migrate before configuration mutations can be enabled end to end.

## Verification

The public tests check strict types, immutable inputs, original ceilings,
cleanup reserves, configuration authentication and the exact protected pins.
The private checker authenticates all 313 donor members and imports only fresh
copies with native imports blocked. It compares the original and proposed
complete owner adapters using the real lease, journal and release validators.
Child execution, platform observations, HTTP result records and clock readings
are CPU substitutes. These checks do not prove native execution.

The original and proposed adapters make identical decisions for successful
completion, a failed first arm, uncertain arm release, invalid locked admission
and invalid independent release. A plan returned by the actual configuration
verifier completes both synthetic arms. An adapter-only 2,180-second worker
plan shortens work and release deadlines by 100 seconds without changing the
cleanup reserves. Missing, malformed, corrupted or mismatched plans, absent
admission and insufficient remaining time refuse before creating a journal.

The complete package checks also cover archive inspection, staging, grant
authentication, spec reproduction, both response replays and refusal paths.
They reuse the explicit spec-hash API adaptation recorded in
[FIXTURE-ADAPTER.patch](../d2-spec-binding/FIXTURE-ADAPTER.patch). Post-run tests
do not replace controller functions or its spec hash.

```sh
python3 -I -B -S serving/tests/test_owner_timing_proposal.py
python3 -I -B -S serving/tests/check_owner_timing_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
make test
```

Logs are under `.test-build/serving-cleanup/owner-timing/`.
[VERIFICATION.json](VERIFICATION.json) records 16 public tests, 15 private
checks and a passing full `make test` run, including 194 cleanup tests.
Both 176-row arms pass expected-response and retained-envelope replay. The
minimum measured file-count margin remains 1.59 times, below the required two
times. Pre-launch reaches the retained external build-proof boundary;
native pre-launch remains unverified. Whole-run
configuration mutations, fresh packet identity, per-row stop policy, device
binding, instrumented health receipts, all-cap/doubled-menu evidence, native
validation and private publication remain open.
