# Host-process benchmark ownership

Status: the first matched physical attempt stopped after its BF16-cache arm.
The inner benchmark completed, but scoped resource and release checks refused;
KV4 did not start. No matched speed comparison or verified release follows.
The owner-supplied
availability report and observed user-cgroup delegation do not qualify native
resource enforcement or current release. No deployment default is changed.
Runtime arithmetic, request bytes and weights are
unchanged. The entry now has an optional preweight identity and post-reap export
path; CPU tests do not qualify its native execution.

## Implemented boundary

`tools/host_process_owner.py` holds an existing private canonical lock on FD9.
It authenticates the inode and already-held open description, refuses any
existing unresolved record, and writes a durable version-2 host-process journal.
It never reuses Pod UID or CRI disposal records.

The supervisor starts a gated child. The parent persists launch intent and the
child's identity before releasing the gate. Identity includes boot, credentials,
PID/start ticks, process group, session and PID namespace. Live signaling uses
Linux pidfds with a second identity check; there is no numeric-PID fallback.
Tracked descendants can occupy different sessions. Polling and reaping happen
before release acceptance, including after nonzero exit, timeout or owner signal.
Authenticated TERM/KILL actions remain separate from native successful exit.

`tools/benchmark_owner_adapter.py` uses the same frozen command/source binding
for journal admission and supervision. It runs BF16 then KV4, stops after a
failed first arm, validates inner results with the existing readout contract,
and never reports serving qualification. Adapter completion is distinct from
physical campaign completion. The module has no remote/GPU-launching CLI.

Per-arm release requires exact checked process identities, observed private
workload namespace IDs with empty membership, current device/sysfs identities,
all-user holder visibility, zero active GPU allocation and healthy target/policy
observations. A bare Boolean or generic empty process list is insufficient.
Every observation binds the session, boot, arm, process and observer source.

Closeout has two stages. Closing FD9 after a launch retains the journal in
`pending-independent` state. After fresh independent observations pass, the core
opens the canonical lock again, authenticates its own journal, clears it while
holding the verification lock, then immediately releases that lock. Failure
retains refusal. A directory-fsync failure after unlink attempts to restore the
exact owned record without overwriting another file; restoration failure is an
explicit unresolved-record-loss error, never success.

## CPU verification

```sh
make test-host-owner PYTHON=/path/to/python3
```

The core suite covers real owned POSIX child completion, nonzero exit, timeout,
resource refusal, signal, gate failure and reaping. It covers lock contention,
inherited descriptor authentication, journal/source/inode substitution, PID reuse,
descendants, namespace-bound release and failed independent closeout. Procfs and
pidfd tests use explicit fixtures on the local non-Linux host.

The composition suite exercises the actual lease, ordered adapter and full
inner-result reader with inert process/platform fixtures. Failed independent
observations keep the durable journal and refuse another owner. Positive fixture
results are synthetic contract evidence, not physical Linux or GPU observations.

## Required before physical admission

Select an authenticated, bounded read-only platform observer. The retained full
FP8 checks use sudo for cluster metadata and all-user device holders. No selected
non-escalating replacement was found. The private smoke launcher uses ordinary
fuser and cannot establish full visibility or cluster-allocation absence.
Do not retrieve administrative credentials, invent a helper path, or turn a
synthetic record into live authority. An existing read-only platform interface
or an explicitly arranged owner-run collector must provide those observations.

The [owner-run metadata collector](OWNER-COLLECTION.md) supplies a bounded
availability snapshot for review. It does not implement per-arm process or
namespace release observation and is not an automatic owner callback.

Bind that observer, the tested adapter, both arm commands, final spec, runtime,
binary/build proof and complete 81-file model inventory into a fresh physical
package. Preserve the existing proof; do not copy it into shared composition
when missing. Current admission must establish both actual GPU UUID/BDF/render
bindings, resource headroom and no competing owner. Historical idle devices
are not a current slot.

The bwrap projection preserves read-only runtime/model mounts, private per-arm
output and caches, two render devices and native TP networking. Host networking
is shared; it is not Kubernetes deny-policy isolation. Sampled memory/storage
floors are not hard cgroup quotas. A future controller must disclose and enforce
its approved resource controls and bound the supplied observation/collection
callbacks; passing a deadline argument alone does not terminate a subprocess.

The proposed 2100-second envelope remains unexecuted. Local adapter readiness
does not fund it or waive any current admission, identity, cleanup or quality
gate. Rollback is non-adoption; installed sources, caches and defaults stay intact.

## Live-binding candidates

Three local preparation components address the remaining integration boundary:

- `benchmark_release_observer.py` binds fixed read-only platform queries to the
  owner session, exact journal, processes and private namespaces. Only an
  authorized human-run root context can execute its privileged observation.
- `benchmark_resource_gate.py` projects an existing systemd user scope and
  verifies actual CPU/memory/PID limits before delegating to bound bwrap bytes.
  Its CPU fixtures do not qualify native limit enforcement.
- `benchmark_physical_controller.py` composes authenticated inputs and the
  owner adapter through explicit observation/resource callbacks. It requires the
  OCI-derived base proof and rechecks the current base, complete private prefix,
  serving sources, native build and all 81 model files before model admission.
- `benchmark_host_transport.py` is an inactive human-root transport candidate.
  It delegates GPU work to UID/GID 1000 and permits only bound host observations
  over an inherited anonymous socket. It does not run sudo or accept commands
  from the workload. Only the human may start the generated privileged wrapper;
  the agent must not invoke it or its sudo commands.

The adapter preserves its 750/180-second default preparation contract. An
explicit new 795/330 pair retains the same 690-second work and 30-second owned
cleanup, followed by a full 75-second observation envelope. The corresponding
2400-second proposal contains 180 admission, two 795-second arms, 330 collection
and independent checks, and 300 sealing. It is not a funded physical ledger.
No request, model or quality count changes.

The optional bounded-scratch projection uses 3.5 GiB across private tmpfs mounts
and reserves 512 MiB for separately bounded durable results/logs. Mount sizes
alone do not enforce that durable allowance or establish native mount behavior.
The physical projection has no writable host result mount. The entry exports
eleven bounded raw files through the V2 artifact protocol after child reaping.
Each arm has two 80-MiB hard stdio caps. Missing, malformed or overflowing
exports cannot pass. The 467.25-MiB
session bound includes both copies of observer snapshots, logs and receipts,
protected source/archive copies, outer stdio, summaries and metadata. Fixed
member counts and bounded writes enforce those partitions. Native mount and
resource observations still belong to the conditional attempt. The old
projection remains the default; neither mode is an automatically runnable campaign.

The V2 export adds two binary rank-log tails (at most 2 MiB each), two progress
journals (64 KiB each), a 64-KiB worker inventory and 64-KiB diagnostic metadata.
Each log record identifies its full byte count and streamed SHA256, tail offset,
tail SHA256 and truncation or absence. Full log reads stop at the existing
128-MiB file limit. The encoded bundle limit is 40 MiB. The existing 80-MiB
stream and 512-MiB session ceilings remain in force. The new reader refuses V1
bundles; sealed historical packages retain their own original readers.

Rank progress is flushed before startup phases and batch work, and after each
completed batch. Collection retains valid prefix counts even when a native
abort prevents final result publication. Missing journals produce unknown
counts; torn records produce incomplete lower bounds. Both TP ranks execute
the same requests, so their counts must not be added together. Progress PIDs
must match the rank identities and observed host/private PID mapping before
successful acceptance. Worker inventory files alone provide no release proof.

After child reaping, the resource gate permits at most two seconds for final
counts and membership to settle inside the existing owner deadline. It retains
at most twelve bounded snapshots and requires two consecutive observations
with only the wrapper process/thread and `pids.current=1`. Changed identity,
limits, resource events or unreadable evidence refuse immediately. The controller
rechecks the observations, and independent scope/device/lock release remains
required. This local repair has CPU coverage; native settling is unverified.

## Verified runtime closure and human launch

The verifier derived the final tree from all eight authenticated OCI layers,
including whiteouts and hardlinks, without extraction. All 96,087 current base
entries matched: names, types, file bytes/executable bits and link targets.
Numeric archive ownership and timestamps are not part of that content claim;
the historical rootless extraction deliberately did not preserve archive owners.
The complete 44,881-entry installed-prefix inventory also matched.

The frozen verifier binds the derivation's source/report hashes and content
inventory. It rehashes the current base and prefix before each physical admission;
a retained pass is not current authority. The real final-package preflight,
including all 81 model files and serving sources, took 44.9 seconds within its
105-second source window. It acquired no canonical lease and imported no runtime.
File verification warms the filesystem cache; no cold-storage load claim follows.

After final local checks, the generated one-use human wrapper stages only pinned
sources in a root-private directory. Its root transport starts a fresh physical
receipt and delegates the continuous FD9 owner and GPU work to UID/GID 1000.
It repeats conflict-free admission under that lock, retains failures, verifies
release and seals the two evidence trees. The original one-use package is consumed, not
native-qualified. No retries, model warm-ups beyond the frozen schedule, image
rebuild, installation, download or cache purge are implied.

## Release-request and failure-evidence repair

The failed attempt recorded each TP worker before and after `setsid()`. The
same PID/start time then appeared twice with different group/session IDs. The
release observer correctly refused that duplicated request before querying
process, device or lock release. It did not establish a surviving process.

The controller now projects the full observation history to one release record
per PID. It refuses conflicting boot, start time, credentials or PID namespace.
It preserves the exact journal root record and retains the complete history in
the supervision result. Signaling and the observer's duplicate/substitution
checks are unchanged. Both release and independent requests use this projection.

The resource gate now retains bounded process/thread membership, usage and event
observations even when final validation fails. It still refuses unexpected
process/thread counts; no settling delay or tolerance was added. The earlier
failure did not retain these counts, so its cause remains unresolved. Current
cgroup absence is not a substitute for the failed historical release check.

The root transport copies bounded observer stdout/stderr and its supervision
receipt into its evidence tree before refusing a failed observer. It does not
publish a successful snapshot or advance the phase on failure. These repairs
require fresh package hashes and a new reviewed execution identity before any
future attempt. Do not patch or rerun the consumed package, delete the unresolved
record, or interpret CPU fixtures as physical release verification.
