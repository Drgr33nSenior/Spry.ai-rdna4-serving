# Closed HTTP package

This source proposal completes the owner/controller migration to the admitted
W5 HTTP experiment. It builds on the
[shared canonical helper proposal](../common-helpers/README.md).

The candidate removes 13 disabled or forwarding feature APIs and the unused
`norm_source_for_arm` helper. All installed consumers migrate together. Calls
that validated settings still validate settings. The active precision, paired
serving and full-profile policy accessors remain.

The benchmark entry rejects retired modes and experiment fields before runtime
effects. It removes the standalone graph, capture and comparison entry points.
The owner readout retains its four HTTP parsing and acceptance functions with
unchanged bodies. The controller removes unreachable legacy diagnostics and
checks the HTTP domain before physical parsing. Independent resource release
checks remain required.

The package omits `graph_screen.py`, `context_tokenizer.py` and
`validation_comparison.py`. The retained runtime reference data is unchanged.
The archive contains 77 members under the existing 80-member ceiling. Request
assets, settings, kernels and active overlays retain their control bytes.
Native workload arguments differ only in the expected source pins.

`RUNTIME.patch` records the runtime changes. `D2.patch` records the exact
controller and owner changes. `SOURCE.patch` adds the explicit, authenticated
three-file retirement to the proposer. The default builder and historical
controls retain their previous source. Publication does not authorize adoption
of these protected changes or a model start.

`VERIFICATION.json` records the final public, target CPU and consumed-packet
checks. These checks cover source and transport behavior. Native serving,
token equality, answer quality and live memory use still require qualification.
No inference performance improvement is claimed.

The next implementation work is the stock runner with its post-load hook,
replay metadata preparation, startup tensor checks and memory recovery. The
recovered memory must remain headroom.
