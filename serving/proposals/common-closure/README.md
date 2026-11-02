# Protected shared helpers

This proposal routes 26 guards and 27 serializer functions through one common
implementation. It preserves error classes, exception arguments, message
prefixes and successful calls that do not evaluate a prefix. Canonical JSON
keeps its existing newline rules and raw-byte SHA-256 values.

The nine protected helper roles remain explicit. Their protected inventory now
also includes `spry_common_canon.py`. Each consumer authenticates the common
source before use, with bounded regular-file reads and a standard source loader.
Root helpers reach it directly; they do not import the unprotected serving
contract to obtain it. The loader ignores ambient modules and cached bytecode.
The controller and transport require the frozen common hash in addition to
self-consistent manifest, inventory and approval hashes.

The existing common payload is replaced, so the package retains 79 members
under the 80-member ceiling. Kernel and tensor sources, requests, settings,
pools, owner ceilings and launch arguments retain their values. Source pins
change with the new implementations. File-stream digests and serializers with
different contracts retain their semantic wrappers.

The remaining per-layer scale and P1 scalar reads precede dependent expansion
or warmup work. They retain their startup refusal order. The duplicated head
scan was removed in the earlier shared-memory increment; it reads one scalar.

See [PROPOSAL.json](PROPOSAL.json) and [VERIFICATION.json](VERIFICATION.json) for
source pins, observed checks and remaining gates. Source checks do not establish
real tensor behavior, GPU replay, model quality or measured memory recovery.
Fresh native qualification still needs its own approved execution envelope.
