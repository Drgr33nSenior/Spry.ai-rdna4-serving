# Production parity import isolation

The model import chain previously loaded `serving_parity` in production. The
runtime and launcher also imported it at module startup, although the selected
production path does not collect parity evidence.

The authenticated model loader now snapshots the observation mode and loads
seven private modules in production. Instrumented mode retains all eight modules,
including parity. An ambient parity module refuses startup in either mode. A
changed observation mode during import also refuses and removes the new module
registrations. Checked source bytes and cached-bytecode exclusions remain in
force for every loaded helper.

The launcher imports parity inside its instrumented tokenizer callback. It still
delegates native tokenizer construction once and returns the original managers.
The retained HIP component reader loads its parity validator only when that
component is required. The ordinary W5 readiness path imports no parity code.

[SOURCE.patch](SOURCE.patch) changes the model, runtime and launcher. Their caller
APIs, response schema and export behavior remain unchanged. The exact protected
helper bytes and set remain identical to the preceding startup stage. Production
evidence mounts are still pending; this change removes unconditional imports.

The tests cover both observation modes, both authenticated arm import chains,
ambient-module rejection, mode changes during import, source substitution,
cached-bytecode rejection and rollback. Callback tests preserve instrumented
collection and the retained component-reader dispatch. Native imports and command
effects use explicit CPU substitutes.

The historical worker-source check now reverses the exact three-file import
patch before reversing the earlier artifact-binding patch. It verifies the
proposed and baseline hash at each step. The historical source pins remain
unchanged. The contract import check now requires seven production helpers,
retains the eight-helper instrumented check, and separately exercises the
authenticated host readout's explicit parity reader.

See [VERIFICATION.json](VERIFICATION.json) for observed test results. The retained
natural-contract comparison and all 11 overlay patches are checked again against
the current source. No native workload, protected-helper activation or publication
is part of this increment.
