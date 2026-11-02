# D2 proposal: configuration transport and grant production

Status: **isolated source proposal; not consumed by the builder**. This extends
the [configuration-binding controller](../d2-config-binding/README.md) with a
transport that accepts the same three independent approval hashes.
The [owner approval record](../OWNER-APPROVAL-20261101.json) covers the previously
presented controller proposal and continued implementation. This new transport
diff is a separate review artifact. No physical invocation is authorized.

[D2.patch](D2.patch) is the exact transport diff from the authenticated donor.
[PROPOSAL.json](PROPOSAL.json) pins the transport, its controller dependency and
the resulting nine-helper protected set. The controller source is unchanged.

## Behavior

The human entry requires `--run-config-sha256`, `--host-sha256` and
`--protected-set-sha256`. It checks these independently supplied hashes against
canonical configuration bytes before creating a workload directory. It checks
the exact nine helpers against both the approved protected set and the existing
package inventory. Run, host and menu references must agree with the controller's
configuration-binding rules.

The archive gains four authenticated files: `PROTECTED-SHA256.json`, the selected
run, its menu and its host configuration. These files are separate from the
existing flat `package_files` inventory. The approval hashes and run references
bind them. Unknown configuration files, duplicate entries, links, unsafe paths,
changed hashes and oversized inputs refuse.

Staging creates the configuration directories through held no-follow directory
handles and writes files exclusively. Configuration files use mode `0444` and
their directories use `0555` after staging. Existing or linked configuration
directories refuse. The grant copies the three verified owner hashes and keeps
the original receipt, parent, boot, source and package bindings.

Transport limits, root ownership checks, privilege separation, observer RPC,
process cleanup and sealing are unchanged. Total time remains 6,360 seconds
with a 300-second seal reserve. The archive remains limited to 16 MiB and at
most 80 members, with the existing lower per-mode limits.

## Verification

Twelve focused tests and nine private checks pass on macOS Python 3.11.6.
[VERIFICATION.json](VERIFICATION.json) records the private checks and source pins.

```sh
python3 -I -B -S serving/tests/test_transport_binding_proposal.py
python3 -I -B -S serving/tests/check_transport_binding_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

The focused tests exercise a small CPU archive through inspection, real local
staging and the controller's grant checks. They test byte and member limits,
bad or missing pins, changed helpers, invalid JSON, unsafe archive entries,
linked directories and CLI requirements. The human entry refuses a wrong pin
before staging or importing owner helpers, with OS identity checks substituted.

The private checks authenticate all 313 sealed donor members before copying.
They compare grants from the complete old and new transports against complete
controller modules, with native imports blocked. They preserve nine old grant
refusals, check three new pins and all nine helper substitutions, and exercise
the pre-lease and retained-base refusal boundaries. An AST comparison verifies
that the human entry changes only by passing the explicit configuration pins.

These are CPU fixtures with OS/RPC substitutes. No root observer, lease, model
or GPU starts. Native execution and full pre-launch completion are unverified.
The builder still emits the original nine protected helpers.

## Remaining integration

The unpruned retained archive contains **80 members**, including its manifest.
The four configuration files would make **84**. The private check confirms that
the old transport accepts the retained archive and the proposed transport
refuses the expanded archive. The ceiling is unchanged. Complete D1 package
pruning and its consumer checks before assembling a configured launch packet.
The successful small CPU fixture does not prove a complete package fits.

The subsequent [D1 package proposal](../d1-package-pruning/README.md) removes five
inactive members and the corresponding controller/transport requirements. Its
separate D2 composition contains 79 archive members including configuration.
The whole cleaned source bundle still needs further pruning before it fits.

Configuration-derived spec generation, runtime consumers, fresh packet identity,
capacity evidence and native gates remain open. The current manifest boot
binding also remains; the later boot-independent configuration migration is
not complete. This proposal must not be installed independently.
