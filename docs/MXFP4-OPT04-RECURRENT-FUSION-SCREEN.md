# OPT-04: recurrent state and local fusion

Opus 5.5 signed off one CPU compiler survivor: combine gated RMSNorm with the following MXFP4 quantizer. **No GPU performance gain is established, and no GPU packet is ready.** The selected runtime remains unchanged.

The candidate removes a BF16 temporary while retaining the BF16 rounding in registers. It uses the original normalization arithmetic and the selected quantizer body. The recurrent state update, GEMM and TP2 collective are outside the change.

| CPU compiler observation | Standalone norm | Fused norm + quantizer |
| --- | ---: | ---: |
| Vector registers | 93 | 62 |
| Scalar registers | 26 | 31 |
| Spills / private memory | 0 / 0 | 0 / 0 |
| LDS allocation | 0 | 0 |

Both verification row counts, 8 and 16, pass the screen using byte-identical machine code within each arm: one binary per arm covers both launch descriptions. These are compiler resources, not speed measurements. The baseline column covers the norm alone; the baseline's separate quantizer was not compiled here. The fused kernel adds quantization work, so lower register use does not establish a latency gain.

At the selected geometry, the fusion removes 48 launches and 4.5 MiB or 9 MiB of logical intermediate write-plus-read traffic per rank per verification step. These source-derived counts are not measured HBM traffic or peak-memory savings. The retained historical profiles put the 48 gated norm kernels near 0.093 ms per sampled step. The exact cost of their 48 following quantizers remains unattributed; all 256 quantizers total about 0.316 ms. Neither figure is a new 235 W baseline or a saving forecast.

The selected ring-fold path already commits all 48 recurrent layers in one launch. Deferring that commit would require changes to the next verification read, raw-ring ownership, prefix publication, state copying and request exit. Skipping it is incorrect. R1 remains a structural design for OPT-06 rather than a local kernel candidate. The MI350 GDN work helped identify fusion boundaries, but its single-token gfx950 implementation does not directly cover this gfx1201 verification path.

SiLU/multiply-to-quantization, convolution/recurrence fusion and collective epilogues remain visible as separate deferred branches. They need their own arithmetic or lifecycle contracts. The existing NVFP4 activation fusion uses a different quantization method; the selected stack still uses the plain-tensor MXFP4 consumer and existing collective path.

Validation: 19 focused CPU tests, 43 public preflight checks, one bounded file-only native audit and one device-free compiler operation. All five retained attention calibration artifacts reproduced before the new compiles. The norm reduction layout and operation sequence match; explicit BF16 rounding survives compilation. The lifecycle fixtures cover acceptance, rollback, C2, publication, reuse and exit hazards, but are not live scheduler or GPU numerical tests.

Four additional signoff tests now read the address expressions from the candidate itself, detect a wrong head-index formula, execute the selected row-tiling helper with inert metadata, and verify binary equality. The future host gate must require one group, width128, 24 heads, exact token/head-row agreement, 2D Z, fresh buffers with the declared packed/scale strides, and the compiled four-row tiling. A different runtime count that changes tiling must refuse. NaN/subnormal device cases remain required.

Before GPU use, build and review the host integration and two-kernel baseline harness, bind the actual norm import and launch, compare packed bytes/scales and downstream outputs bitwise, test fallback and buffer ownership, and complete the owner lifecycle. This candidate joins the cumulative register as CPU-only and unmeasured; it is not yet eligible for a serving bundle. Cap32 and verification-prefetch retain their earlier verdicts.

The [provenance](../provenance/MXFP4-OPT04-20261103.json) binds the detailed evidence and review status. All 21 lead references remain tracked. OPT-05 and OPT-06 precede the final cumulative test selection.
