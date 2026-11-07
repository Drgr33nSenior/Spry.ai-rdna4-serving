# OPT-00: baseline and production route evidence

**CPU exit complete; independently reviewed by Opus 5.5.** Sprint 00 is signed off for CPU implementation. No GPU operation or model start was used, and no performance gain is claimed. The [signoff record](../provenance/MXFP4-OPT00-20261103.json) binds the sealed evidence and review.

The baseline manifest freezes the adopted C4 composition and its source identities. The lead matrix retains all 21 references from the original list and Exa research in 16 shared work groups. It records consumed sources, dispatch, shapes, existing optimisations, known cost and missing evidence for every entry. Packing, FP8 heads, P3 normalization, copy fusion and aligned ring-fold are already selected; later work must establish an additional removable cost.

B0 has no new timing measurements. Future comparisons still need a fresh matched control. The proposed comparison conditions use the operation-212 control's cap-3 MAMBA experiment and equal observed 300 W caps. These differ from the selected pointer's historical 235 W/no-cap-3 settings. Neither setting is changed or globally adopted here.

## Production witness repair

Operation 212's writer was inside the instrumented startup path, while production readback accepted only readiness and binding records. The verification-prefetch experiment builder now installs a separate, source-authenticated startup handshake. It restores the native method before returning, performs only host metadata checks, and writes one bounded route record per rank before requests begin.

The records bind the arm, session, specification, source manifest, adapter, process identity and lifecycle. Separate counters require both captured BF16 geometries—8 and 16 rows—on both ranks. The candidate arm must capture the candidate route in each graph; the control must report zero candidate calls. Warmup counts cannot substitute for captured calls. The kernel digest field records the arm's expected source identity, not a measured executable digest.

The production readiness callback refuses missing or invalid records before constructing the request client. The unchanged collector retains the records, and the production reader checks them again from the bundle. Full instrumentation remains disabled. Kernel bytes, arithmetic, request menu, quality comparator, owner limits and release behavior are unchanged. The controller's derived specification digest changes with the package.

This repair is scoped to the verification-prefetch experiment builder. It does not retry operation 212, make its missing evidence complete retrospectively, or make prefetch selected. Future optimisation families need their own source and route contracts.

## Observed validation

| Check | Result |
| --- | --- |
| Focused local tests, Python 3.11 | 14 passed |
| Same focused tests on workstation Python 3.14.4 | 14 passed |
| Complete native source compilation | 70 modules passed |
| Public validation preflight | 43 checks passed |
| Remove production hook / remove production reader validation | Both mutations failed the tests as required |
| Selected, donor, research, triage and plan seals | Verified |
| Generated package | 81 members; approximately 21.5 MiB staged, below the inherited 32 MiB bound |

The tests execute the actual generated producer, hook, worker entry, readiness callback, bundle exporter and production reader. One reader fixture substitutes the unchanged native-identity and HTTP validators; it does not establish physical serving behavior. Native execution used the inherited finite, standard-library-only source-check workflow, with framework, device and subprocess access refused by its audit guard. No framework compilation was needed.

The new records are limited to 16 KiB each by readiness and readback. Existing 4 MiB raw and 6 MiB encoded native-bundle limits remain. Authenticated unchanged owner/caller receipts are reused. Physical graph counts and startup behavior still require a future exact approved packet.

## Review and next step

Opus 5.5 returned **AGREE**, verified all six cited seals and reran the 14 focused tests once. It found no blocking source or lifecycle defect. The review report SHA256 is `4808af787376cbec07d1ef2c85551c91e9dc38aeb5676dd44e46e5737843784f`.

Two limits carry forward. Before a future GPU packet, trace the pinned graph-capture source to confirm one synchronous capture per geometry before readiness; physical 16/16 counts remain unmeasured. The callback's plain `serving_integration` import resolves from the authenticated read-only review mount, but its actual import resolution and a separate per-module preload pin were not exercised by the fixture.

OPT-01 now audits host/TP2 behavior and the capture-path follow-up. OPT-02 can then screen exact weight packing. These start with retained source and CPU evidence. The sprint plan's GPU budgets remain proposals, not execution authority.

The private evidence root is `optimization-continuation-v1/opt-00-baseline-and-route-witness-v1`. Its principal records are `B0.json`, `LEAD-MATRIX.json`, `CONTRACTS.json`, `SOURCE-DELTA.json` and `VERIFICATION.json`.
