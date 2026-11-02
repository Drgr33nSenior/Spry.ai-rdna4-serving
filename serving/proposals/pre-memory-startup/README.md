# Pre-memory cleanup control

Operation 201 reproduced operation 196's first warmup failure in production:
the same 3,318 output token IDs contained two closing thinking markers. The
response guard refused both runs. The source of the divergence remains unknown.

This proposal prepares the earlier `pre-replay` cleanup stage, before shared
head allocation and packed parameter storage rebinding. It retains that stage's
computation sources, nine protected helpers and common helper. Its only worker
changes remove the retired `none` draft summary call and read the metadata policy
from the authenticated adapter module. The earlier startup failures and the
corrected producer-to-reader paths remain covered by tests.

The builder requires the exact original worker hash, each replacement exactly
once and the exact corrected hash. Existing proposals retain their original
bytes. `SOURCE.patch` records the builder and test-target additions;
`RUNTIME.patch` records the two changed worker functions. The source-history
test reverses this increment before checking the earlier proposal chain.

The selected next packet is baseline `c21e0b4ce8f4ed17`. All 176 frozen request
bodies and tokenizer-input records match passing B186. Six local campaign checks
pass. Thirteen native CPython 3.14.7 source-only tests pass, including both exact
archives, their own root transports, complete worker materialization, draft
preflight and startup proof producer/reader. GPU access, native serving imports,
network and subprocesses were denied inside that preflight. Native model and
tensor objects were explicitly substituted.

A successful run would narrow the regression to later increments; it would not
identify a single cause. This control has no new GPU qualification, tensor proof,
quality grade, memory result or performance result. The paired current candidate
is prepared for source comparison only. It has no execution authority.

See [verification](VERIFICATION.json) and
[operation 201](../../measurements/native-qualification/GPU-201-RESULT.json).
The baseline requires fresh exact one-use owner approval before registration.
