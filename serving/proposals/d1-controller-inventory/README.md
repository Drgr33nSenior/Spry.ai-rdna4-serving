# D1 proposal: selected controller inventory

Status: **proposal only; not approved or consumed by the builder**. This is the
first D1 slice. It does not complete D1 or implement D2.

[D1.patch](D1.patch) deletes 47 lines from the sealed W5 v2 controller and adds
no lines. The [proposal copy](review/benchmark_physical_controller.py) contains
the resulting source. [PROPOSAL.json](PROPOSAL.json) records the donor, patch,
proposed source and both nine-file protected-set hashes. Protected-set hashes
use `review/` path keys and canonical JSON with a trailing newline, as the
source builder does.

The patch removes:

- Inventory branches for inactive norm, midpoint, raw-attention, verification
  A/B and GDN capture variants.
- The inactive draft-free and fixed-work inventory alternatives, plus obsolete
  capture-format insertions and removals that cancel out for W5.
- The unused `run_controller` CPU wrapper. The packet CLI calls
  `run_physical_controller`; the underlying `_run_bound_controller` remains.

The active verification-attention chain branch remains. The first draft
removed it, and the real inventory check refused that draft because W5 needs
`verify_decode_adapter.py`. The corrected patch preserves the selected source
and package lists. No package member is removed in this slice.

All remaining controller function bodies, top-level assignments, imports,
spec pins and owner ceilings are unchanged. The other eight protected helpers
remain at their donor hashes. The supported comparison scope is the retained
`sprint01-paired-serving` configuration; this proposal does not claim to preserve
retired configurations.

## Verification

Observed on macOS Python 3.11.6: five source checks, seven private packet checks
and the full `make test` suite pass. [VERIFICATION.json](VERIFICATION.json)
retains the exact private result, source pins, decisions and measured margins.
Both controllers select the same 48 source names and 79 package members.
The narrowest margin is 322 client files under the 512-file limit (1.59 times).
The design's two-times capacity requirement remains open.

Run the public source checks:

```sh
python3 -I -B -S serving/tests/test_protected_proposal.py
```

Run the private checks with the sealed preparation root and a fresh report path:

```sh
python3 -I -B -S serving/tests/check_protected_packet.py \
  --packet /path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --report /path/to/new-report.json -v
```

The private checker authenticates all 313 sealed members, then imports fresh
copies. It compares the actual controller inventory and pre-launch checks,
including bad spec/helper pins, extra source, raised time and launch flags.
It also runs the retained HTTP producer, exporter, receiver, readout and
controller acceptance for the real 176-row response replay and the retained
envelope, for both arms. Length stops and oversized artifact members must
refuse. Native imports are blocked.

Pre-launch verification stops at the same missing workstation build proof in
both versions. External root/tree checks, retained-runtime attestation and the
external draft tree use the retained CPU test substitutes. The draft package
binding is checked. This is not full native pre-launch verification.

Post-run evidence uses simulated live process observations and owner records.
The retained envelope is 1.25 times observed output counts, capped by each
request, with six text bytes per token. Its required margin is 1.5 times. It is
not the design's all-cap, doubled-menu or two-times-margin gate. These checks
isolate D1 against retained W5 sources; they do not run the entire cleaned source
bundle or account for its six new observer health receipts.

## Remaining work

Continue coupled retired-mode and package-consumer pruning before completing
D1. Implement D2 config authentication, ceilings and stop-policy consumers in
separate proposal copies. Complete the full controller and evidence-size gates
before requesting exact owner approval. The active builder must continue to
emit the original protected set until that approval applies.

No launch manifest, GPU run, commit or publication is produced by this proposal.
The current GitHub remote is public; the cleanup requires a private publication
destination.
