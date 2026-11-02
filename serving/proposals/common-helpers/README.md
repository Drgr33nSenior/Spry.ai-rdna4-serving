# Shared JSON source and HTTP package cleanup

This proposal replaces the unused legacy graph-integration payload with the
existing canonical JSON/hash source. The physical archive stays at 80 members.
It builds on the [startup digest proposal](../startup-digests/README.md).

The installed `spry_common_canon.py` is byte identical to
`spry_common/canon.py`. The MTP contract, request/readout contract and receipt
codec use its functions. Bare JSON and JSON with a final newline remain separate
APIs. Raw-byte hashes remain distinct from hashes of encoded values. Strict JSON
parsing and domain-specific exceptions keep their existing implementations.

The MTP contract authenticates the shared source through a bounded, no-follow
file descriptor before executing it with a standard source loader. It checks
file identity across the read and ignores cached bytecode. Missing, changed,
linked or oversized helper sources refuse. The installed loader uses the
adjacent helper; the local authoring contract reads the canonical source.

## Removed closure

`graph_integration.py` contained the retired standalone context/graph experiment.
The current HTTP entry already refuses that experiment mode. Its lazy graph
readout hook and unused `cache_arm` helper are removed from the candidate entry.
The graph projection module retains `validate_projection` and `project_argv`,
which the HTTP controller and owner still use. Its mount projection body is
unchanged. The native graph backend, model runner and all active overlays are
byte identical to the control.

Historical controls retain their original files and interfaces. This proposal
does not remove every compatibility API or finish the whole package closure.

## Assembly and protected delta

`SOURCE.patch` adds one explicit source exchange to the proposer. It authenticates
both the replaced source and the canonical helper, refreshes the HTTP source
maps and replaces the corresponding projection-helper name. It rewrites no
Python or shell text during package assembly.

`RUNTIME.patch` records six source changes. `D2.patch` changes three controller
helper-name references and the fixed-spec remainder hash for the new helper
name. The remainder check, resource ceilings, owner/release rules and archive
limit remain. The other eight protected helpers are unchanged. These are source
proposals; publication does not approve native adoption or authorize a start.

## Verification and remaining work

Focused CPU checks compare previous JSON bytes and hashes, exercise all three
actual consumers, reject invalid helper sources and stale-bytecode substitution,
compare projection arguments and refuse retired launch modes. Consumed-packet
checks also compare every member with the startup-digest control and compare
native workload arguments after the expected source-pin updates.

The production and instrumented prelaunch paths, full producer/reader chain,
all-cap and doubled menus remain required. Native serving, token equality,
quality and live mount behavior remain unqualified for this proposal. No startup
or inference speedup is claimed. Remaining compatibility APIs, the stock runner
hook, replay metadata, startup tensor scans and memory recovery remain open.
