# Host observer-health import scope

Operation 206 completed the first 176 HTTP responses. Every raw request and
output token vector matched B186. Host readback then raised
`ModuleNotFoundError: No module named 'mtp_contract'` while validating observer
health. The original caller returned a failure receipt and independent GPU
release passed. No second start or CPU quality grade ran. Checkpoint 563 is
parked. The operation remains failed and partial.

The health validator performs a deferred contract import. Existing tests supplied
an ambient contract and missed the host import boundary. This proposal executes
the real validator inside the existing authenticated `_gdn_contract_scope`.
Only `serving_integration.py` changes from operation 206. The other 60 Python
files, ten protected entries, five tensor-storage sources and frozen requests
remain identical. Inference, tensor and kernel code do not change.

The nine new regression tests use real health validators with absent and hostile
ambient modules. They verify restoration and refusal of invalid health, counters
and source pins. They also reproduce the original missing-module error.

Astra reviewed the failure and exercised the complete retained
`read_http_artifacts` → physical resource validation → owner HTTP validation
chain. Six chains pass: operation 206's first arm and both production arms from
203, each with absent and hostile ambient modules. All 27 expected refusals pass.
The gate authenticates ten seals and compiles the whole candidate module before
substituting only its corrected health function into the sealed reader. This is
prospective CPU validation; it does not accept or grade operation 206.

See [verification](VERIFICATION.json), the
[gate summary](../../measurements/native-qualification/HOST-READOUT-CPU-GATE.json)
and the [testing procedure](../../measurements/native-qualification/TESTING.md).
The production packet `e2011102000000a3` combines the remaining bounded native
qualification and descriptive measurement. It has no launch authority yet.
Production omits the instrumented diagnostics, so this plan does not establish
second-start instrumented witness coverage. Cleanup sign-off remains pending.
