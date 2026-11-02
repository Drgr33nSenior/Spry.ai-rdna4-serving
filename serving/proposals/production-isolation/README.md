# Production evidence isolation

Production launch projections mask six private evidence helpers: the graph
observer, lifecycle probe, observer health, target-KV metadata, graph/KV join
and parity modules. Instrumented projections retain their authenticated source.
The host review tree keeps the original files for provenance and later analysis.

The proposed controller creates one exclusive, read-only, 84-byte Python file
outside the review directory. This file raises `ModuleNotFoundError`. Each
production sandbox binds it over the six helper filenames after the review
directory mount. The existing source validators check the actual masked bytes
against their declared hashes. Their validation code is unchanged.

The controller authenticates and stamps the generated file before launch. Durable
accounting includes the file in the existing small-metadata partition. A linked
or pre-existing destination refuses creation. The fresh bytecode prefix remains
required; cached bytecode from the original helpers cannot replace the masks.

The actual owner adapter produces both frozen production launch projections
with 246 arguments. The final projection retains the existing 256-argument
ceiling after bytecode-mask consolidation. No archive member, source tree or
resource allowance is added. The renderer refuses to pair these bindings with
an older protected proposal that lacks the corresponding mount support.

`SOURCE.patch` records the canonical contract, builder and proposal-renderer
changes. `D2.patch` records only the proposed owner-adapter and controller changes.
The canonical builder still emits the original nine protected helpers. The new
protected source remains a review proposal and grants no execution authority.

CPU checks cover both launch projections, actual source hashes, import refusal,
exclusive creation, source stamps, durable accounting and instrumented imports.
They do not execute Linux `bwrap`, the native serving stack or a GPU workload.
Native validation, complete protected review and the remaining cleanup gates
are still required. No inference performance improvement has been measured.
