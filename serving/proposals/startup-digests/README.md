# Reuse verified startup source digests

This proposal removes repeated file hashing from the native worker's source
checks. It builds on the [runtime binding proposal](../runtime-binding/README.md)
and changes only the worker source and its derived package bindings.

The worker reuses a digest only for a source path declared in its manifest,
within one worker invocation and process, while the path remains canonical and
its mount reports read-only. Each lookup checks the device, inode, mode, owner,
group, link count, size, modification time and change time. A changed identity
requires a fresh hash. A symlink or noncanonical path refuses. A first hash also
requires the file identity and read-only mount observation to remain stable
through the read.

Writable files, undeclared files, files outside the invocation and inherited
process state always use fresh hashes. If mount flags are unavailable, reuse
is disabled. A transition to a writable view discards the cached entry. The
worker clears the scope after success or any exception, including an exception
from native startup, serving or cleanup.

Every existing source-pin comparison, import-origin check and callable/code/
globals check remains. Model configuration metadata is outside the declared
source set and keeps its fresh reads. The cache retains digests and file
identities; it retains no source contents, tensors, buffers or streams.

## Source and package

`spry_serving/startup_worker.py` is the candidate. The prior worker remains
available for the runtime binding control. `RUNTIME.patch` records the exact
worker delta. No protected helper changes: `D2.patch` is empty and its recorded
hash authenticates that empty delta. The nine protected helper hashes and
their protected-set hash remain identical to the runtime binding proposal.

The package adds no members. Requests, settings, model and stack files, kernels
and all active runtime overlays remain byte identical to the control. This
proposal does not select a new runtime or authorize a model start.

## Verification and remaining work

The focused CPU tests run the actual worker functions against local files and
inert mount observations. They cover restored modification times after a byte
change, inode and link changes, symlink substitution, writable and unknown
mounts, process and invocation boundaries, and changes during hashing. The
actual manifest validator must still refuse a wrong pin after a cache hit.
The same suite also runs against the assembled packet's worker.

These fixtures establish source-check behavior. They do not establish live
mount behavior in a serving sandbox or native token equality and quality.
Fresh native qualification remains required. No startup or inference speedup
has been measured. Startup tensor scans and `.item()` synchronization, remaining
module closure, the stock runner hook, replay metadata and memory recovery
remain separate work.
