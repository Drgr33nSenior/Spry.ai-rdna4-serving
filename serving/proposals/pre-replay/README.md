# Verify metadata before replay

The adapter now prepares causal KV indices and singleton split choices before
the forward graph runs. Eager initialization and the native outside-graph hook
prepare the stable buffers. The inherited in-graph hook preserves that state.
Capture refuses missing preparation. Eager stream or input-reference changes
refresh the metadata before the native forward call.

The target's pinned runner calls preparation on the capture stream before
capture and before each replay. It disables metadata glue graphs for DFlash.
The source inventory records the exact runner and stream-context hashes used
to verify that lifecycle. Width eight and overlap-off startup checks remain.

Buffer geometry, row order, singleton split selection, native cache writes and
both decode launch frontends remain. The source-only simulation compares batch
sizes one and two with the control, including changed prefixes and request
slots. Its captured forward work contains no metadata preparation.

The startup witness now declares `outside-forward-graph-v1`. The reader requires
zero metadata builds during capture, complete capture coverage, preparation
before forward and a matching count of eligible calls and metadata uses. Old
capture-preparation records refuse. Assemble the new worker and reader together;
historical packets retain their own readers. Production HTTP data is unchanged.

`RUNTIME.patch` records the adapter, worker and reader changes. `D2.patch` is
empty: all protected helpers, kernels, heads and active overlays retain their
control bytes. The package still has 78 members. `SOURCE.patch` records target
preflight selection and its optional bounded source-access inventory.

The target source archive excludes historical patch copies that the observed
selected tests do not open. Repository files and tests remain intact. This
keeps the original 16 MiB archive bound. If source-access observation saturates,
the report marks the inventory incomplete without changing test behavior.

`VERIFICATION.json` records public tests, target CPU checks and complete packet
transport. These checks establish source and fixture behavior. Native serving,
token equality and quality remain unqualified. Startup scan consolidation,
shared-head and packed-view memory recovery, remaining helper/marker consumers
and broader instrumented capacity qualification remain.
