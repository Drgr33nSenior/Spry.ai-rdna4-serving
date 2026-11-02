# Fresh caller and manifest proposals

Run `serving/propose_packet.py` to prepare review artifacts for a fresh session:

```sh
python3 -I -B -S serving/propose_packet.py \
  --donor /absolute/path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --output /absolute/path/to/new-proposal-review \
  --packet-id 0123456789abcdef
```

The output must be new and outside the donor and canonical source trees. The
packet ID must contain exactly 16 lowercase hexadecimal characters and must
differ from the donor identity. The renderer authenticates all 313 sealed donor
members before it uses their source or templates. It accepts only the exact nine
protected helpers named by the selected local boot-approval proposal.

The `package` directory contains the 75 review members, three configuration
files, the proposed protected hash set and `MANIFEST-PROPOSAL.json`. The manifest
has a fresh resource root and session. It has no boot identity. The `caller`
directory contains the pinned host configuration, power observer, connection
metadata and finite command helper. No private key or credential file is opened.

`LAUNCHER-BINDINGS-TEMPLATE.json` joins the run, host, proposed protected set,
manifest, fresh paths and derived caller duration. `APPROVAL-TEMPLATE.json` has
an empty boot identity, zero starts and no model-start authority. The source
archive stays independent of future boot observations.

The output is a review tree. It contains no `MANIFEST.json`,
`LAUNCHER-BINDINGS.json`, `RUN-AUTHORIZATION.json` or workstation launch script.
Caller activation still needs an approved source closure, a fresh one-use
approval and the remaining cleanup gates. The canonical source builder retains
its original protected-helper invariant.

Tests reproduce the complete output in two destinations and compare every byte.
They then project the manifest filename inside a temporary CPU fixture to
authenticate all 80 members with the real transport. The real controller
reproduces the configured spec and still refuses at the missing external runtime
proof. The staged caller reads its actual host file with SSH and command effects
mocked. These checks do not execute the caller or native workload.
