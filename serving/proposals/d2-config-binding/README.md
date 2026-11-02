# D2 proposal: configuration approval bindings

Status: **source review approved; not consumed by the builder**. The owner's
“OK, approved” response is recorded with the presented hashes in the
[approval record](../OWNER-APPROVAL-20261101.json). `PROPOSAL.json` and
`VERIFICATION.json` retain their original pre-approval status and bytes.
This is the
first D2 slice, based on the tested [D1 controller proposal](../d1-controller-inventory/README.md).
It adds authentication checks. It does not remove the existing spec pin or
make runtime consumers read configuration values.

[D2.patch](D2.patch) is the exact diff from D1 to the
[proposed controller](review/benchmark_physical_controller.py).
[PROPOSAL.json](PROPOSAL.json) records the source chain, patch hash and proposed
nine-file protected-set hash. The other eight helpers remain unchanged.

The proposed physical manifest adds `run_config`, a relative path of the form
`config/runs/<run-id>/run.json`. The physical grant must additionally name
`run_config_sha256`, `host_sha256` and `protected_set_sha256`. These are SHA256
hashes of canonical JSON bytes with a trailing newline, matching builder output.

After the existing parent, receipt, boot, package, source and ceiling checks,
the controller:

1. Binds the configuration root to the executed controller path.
2. Checks `PROTECTED-SHA256.json` against the approval, its exact nine members,
   the package manifest and each helper's file bytes.
3. Checks the approved run and host files, their cross-reference, the run ID and
   the menu named and hashed by the run. Host configuration cannot hold boot ID.
4. Reads only bounded, canonical, singly linked regular files. Reads use
   no-follow and nonblocking flags, strict JSON parsing, identity checks and
   the existing 105-second preparation deadline.
5. Reauthenticates inside `_run_bound_controller` before lease work. It adds file
   and parent-directory stamps to the existing before-launch rechecks.

The file-read ceilings are 64 KiB for the protected-set and host JSON, 256 KiB
for the run JSON, and 1 MiB for the menu and each protected helper. Runtime,
transport, session and owner ceilings remain unchanged.

## Verification

Fifteen focused checks and seven complete-module private checks pass on macOS
Python 3.11.6. [VERIFICATION.json](VERIFICATION.json) retains the private result
and its exact source pins.

```sh
python3 -I -B -S serving/tests/test_config_binding_proposal.py
python3 -I -B -S serving/tests/check_config_binding_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

The focused tests execute selected definitions from the proposal with explicit
CPU substitutes. They cover changed or missing approval pins, run/host/menu
references, helper-set membership, noncanonical/duplicate/nonfinite JSON,
links, FIFOs, byte limits, deadlines and changes during or after authentication.
The patch check verifies that the old grant checks, spec literal and existing
ceiling assignments remain intact.

The private tests authenticate the sealed donor before copying it. They import
the complete D1 and D2 controller modules while blocking native imports. They
check valid synthetic grants, nine unchanged grant-refusal decisions, the three
new pins and all nine changed-helper refusals. They also call the real bound
controller up to an explicit CPU boundary before lease work. The physical entry
still refuses missing retained-base evidence before the model path.

Pre-launch comparison reaches the same missing workstation build proof in both
versions, using the retained external-host substitutes. The first private test
attempt reused a controller modified by those substitutes; the corrected check
uses a fresh module for each case. Both attempts remain in the local logs.

Tests use temporary synthetic grants and substitute OS/RPC boundaries. No GPU,
root observer, lease or native model starts. The D1 post-run evidence result is
reused: its helper functions and the bound controller's post-run code are unchanged.

## Remaining D2 work

This gate authenticates file bytes; it does not prove that generated spec bytes
follow from those configuration values. The literal `SPEC_SHA256` and all old
runtime checks remain in place. Complete spec reproduction and semantic checks
before replacing that literal. The helper checks cover file bytes, not a new
proof of bytes loaded by every legacy module loader.

The [transport proposal](../d2-transport-binding/README.md) now produces the new
grant fields and authenticates configuration files during staging. Old grants
still refuse under this controller proposal. The complete configured archive
exceeds the unchanged 80-member ceiling, so transport integration still needs
package pruning. Migrate the runtime consumers, freeze any imported shared
helpers, and complete the capacity and native gates before complete D2 review.
Do not install this proposal independently.

The active builder continues to emit the original nine protected helpers. It
does not consume D1 or D2, generate an approval, or emit a launch manifest.
