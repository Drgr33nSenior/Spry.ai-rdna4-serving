# Verification scope

`make test-cpu PYTHON=/path/to/python3` runs the same declared 112-case selection
as the accepted source handoff. It imports only standard-library helpers and
individual source bodies. It must not import Torch or the SGLang package, load a
native library, query devices or execute a model. Native/tensor/device objects
are labelled CPU substitutes, not numerical GPU evidence.

The retained cases cover config/rank identity, refusal before allocation,
constructor-to-backend wiring, real strided caller layouts, cache ownership,
prefix protection, generations, scheduler/release boundaries and source/build
inventory checks. The legacy GPU executable is called only without its execution
switch to verify refusal. Tests use inert binary bytes and synthetic review
receipts; these never count as accepted build evidence.

The fixtures contain frozen earlier source and stock caller bodies so tests can
exercise actual interfaces and compare baseline traces. They are not another
supported serving version. Path bindings were relocated for a fresh clone;
`provenance/TEST-RELOCATION.json` records those changes. Existing assertions and
numerical tolerances were not relaxed. One historical native snapshot test still
uses its authenticated old native fixture; a separate test checks the four
selected current native hashes.

`make test-codec` builds and runs the two C++17 CPU suites with ASan/UBSan. These
exercise the codec, independent vectors, BF16 bits/RNE, FP32 WHT and canaries.
The predeclared WHT tolerance is `2e-6`; the dot tolerance is
`1e-5 + 2e-7*abs(reference)`; the same-code softmax tolerance is `1e-11`.
Those criteria are not silently applied to different BF16 arithmetic, and lossy
four-bit output is not required to equal BF16/FP8 output exactly. No HIP kernel
is compiled or executed by these CPU suites.

Runtime validation is separate: first bind a complete source/build identity,
then perform an authorised device/model smoke. Benchmark and task-quality gates
follow independently. The historical physical pass is described in `STATUS.md`;
it was not rerun for publication. The source subset here cannot substitute for
that complete installed composition.
