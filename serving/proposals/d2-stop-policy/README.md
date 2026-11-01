# D2 proposal: declared stop policy

This increment extends [artifact limits](../d2-artifact-limits/README.md).
[D2.patch](D2.patch) binds protected reporting to the authenticated menu.
[SOURCE.patch](SOURCE.patch) records the canonical client and contract changes.
The builder consumes those canonical changes and retains the original nine
protected helpers. Complete D2 approval remains separate.

Each row accepts `natural` or `cap`. Historical rows without the field retain
natural-stop behavior. Natural rows require `NORMAL_STOP_COMPLETE` and `stop`.
A cap row can finish naturally. A length finish is accepted only when both the
reported completion count and retained response-ID count are exact integers
equal to the declared output cap. Such work remains ineligible for answer
grading. Unknown policies, type aliases, short or oversized counts refuse.

The client, reader and summary validator use the same completion rule. A
precomputed case index replaces the previous linear search on every completion
check. The protected report uses the authenticated menu; its compatibility API
can still read the retained settings menu. Request-body bytes are unchanged.

## Verification and limits

The CPU packet suite passes 20 checks. Both complete baseline response replays
retain their source/spec verification and natural length-stop refusals. A
consumer-boundary fixture declares row zero as cap work, sends an exact-cap
length response through the actual client, and completes readout and protected
reporting for both 176-row arms. Native observations remain CPU stand-ins.

That fixture does not pass the pinned whole-run configuration verifier. The
retained quality packet declares natural answers and correctly refuses the cap
row. The test asserts that refusal; it does not grade the work as an answer.
Quality-menu rendering and whole-run configuration support remain open.

```sh
python3 -I -B -S serving/tests/test_contract_cleanup.py -v
python3 -I -B -S serving/tests/check_stop_policy_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

[VERIFICATION.json](VERIFICATION.json) records exact source and log hashes.
Logs are under `.test-build/serving-cleanup/stop-policy/`. Earlier fixture
failures remain retained. No native or GPU run occurred.
