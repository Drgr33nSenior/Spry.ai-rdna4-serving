# W5 evidence capacity proposal

This proposal closes the measured W5 receipt and source-copy capacity gaps.
It preserves the existing owner ceilings, source archive member limit, decoded
archive limit, artifact limits and release validators. Canonical `build.py`
still emits the original nine protected helpers. Only an explicit proposal
renderer selects the changed transport and inherited protected sources.

The receipt writer compresses larger fields and groups logical control records
in one physical envelope. Each logical field remains write-once and bound to
its original name and bytes. The reader recomputes control timing and checks
every logical record. Version 1 request envelopes remain readable. Version 2
decompression has a declared byte bound and refuses excess output, incomplete
streams, concatenated streams and trailing data.

The proposed transport accepts a gzip source archive after checking its physical
hash. It bounds the decoded tar to the original 16 MiB ceiling before applying
the original member, type, path and hash checks. Raw tar remains supported.

All 37 composed packet checks passed on macOS CPython 3.14.8. The expected and
retained-envelope W5 cases use 230 client files, giving a minimum evidence
margin of 2.226. Composed source copies use 15,287,040 bytes, giving a margin
of 2.195 under the existing 32 MiB partition. These are CPU fixture results for
the pinned 176-request W5 menu. They do not establish all-cap or doubled-menu
capacity, native serving readiness, or an inference speedup.

See [verification](VERIFICATION.json) and the
[GPU preparation](../../measurements/gpu-run/README.md). Both proposed source
campaigns are sealed for review. Owner approval and native GPU validation remain
open; this proposal supplies no execution authority.
