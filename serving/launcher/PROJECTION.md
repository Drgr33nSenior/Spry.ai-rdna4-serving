# Prepare caller source for review

This CPU-only step creates a fresh source review. It does not register or
invoke a workload. Use the [caller workflow](README.md) for the later native
boundary and the [serving guide](../README.md) for the current source layout.

## Inputs and outputs

| Input | Purpose |
| --- | --- |
| Authenticated sealed donor | Supplies the retained source, assets and templates |
| Explicit reviewed proposal | Selects exact runtime and protected-helper bytes |
| Run, menu and host configuration | Supplies the configured profile and bounds |
| Fresh 16-character lowercase hexadecimal packet ID | Derives session and workload identities |
| New canonical output path | Keeps the donor and previous evidence immutable |

Inspect the available arguments first:

```sh
python3.14 -I -B -S serving/propose_packet.py --help
python3.14 -I -B -S serving/prepare_gpu_run.py --help
```

`propose_packet.py` renders source, manifest and caller review artifacts.
`prepare_gpu_run.py` assembles candidate and baseline author/caller trees,
conditional grading templates and unfunded approval requests. Select a proposal
explicitly; a default or an old sealed packet is not a selection of the final
cleanup. Proposal pins are versioned and can refuse after later source changes.

Read the generated source review and approval request for actual membership,
identities and limits. Counts from older increments are historical. The source
archive stays independent of a later approval boot binding; the retained CPU
grading template also has its own reviewed boot requirement.

Before execution, complete the [test gates](../measurements/native-qualification/TESTING.md),
review the complete protected diff and retain the exact one-use approval.
Unfunded review artifacts grant zero model starts. Preparation does not open a
private key or contact the workstation.

<details>
<summary>Historical caller projection record</summary>

The following is the earlier projection-stage description. Its member counts,
missing-proof boundary and pending gates describe that source version.

### Earlier caller and manifest proposal

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

</details>
