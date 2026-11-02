# Runner post-load hook

This proposal moves the selected target and drafter installation code into
`spry_serving/post_load.py`. The runner calls it from `init_cuda_graphs`, after
weight loading and attachment of the target head to the drafter, before graph
capture. Importing the hook loads no model library and accesses no device.

The target installs its FP8 head, then packs all 48 GDN input projections. The
drafter installs its head proxy. Installation order, adapter calls and failure
propagation retain their previous behavior. Synchronous-load checks, model
replacement guards and retained deferred-capture callbacks remain in the
runner. Repeated installation refuses.

The worker authenticates the hook and registers its module before Scheduler
can load a model. It checks the module origin and the startup function's code
and globals. Missing, changed, ambient or rebound sources refuse. Both HTTP
bindings carry the hook and updated runner pins.

`RUNNER.patch` applies to the exact retained effective base named by the
existing upstream lock. That base already contains retained changes; it is
not an unmodified SGLang release. The new patch replaces the inline adapter
installation with one startup call and keeps the existing guards and callbacks.
The other active overlays and all kernels retain their control bytes.

`RUNTIME.patch` records the runner, contract and worker changes. `SOURCE.patch`
adds the explicit hook assembly and binding step. `D2.patch` adds the hook to
the controller's authenticated source inventory and pins the new runner and
hook together in archive admission. Self-consistent rollback and changed hook
archives refuse. The owner and the other six protected helpers retain their
parent bytes. The archive has 78 members under
the existing 80-member ceiling. The default builder remains unchanged.

The CPU checks exercise both startup roles and failure paths, exact patch
reproduction, source refusal, configured prelaunch and the complete packet
transport. `VERIFICATION.json` records observed results when verification is
complete. Publication authorizes no model start or protected-source adoption.

Replay metadata preparation, startup tensor validation and memory recovery
remain separate changes. Native token equality, quality and memory use remain
unqualified for this source proposal. Recovered memory must remain headroom.
