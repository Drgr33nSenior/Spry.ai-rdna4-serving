# OPT-02: MXFP4 representation and arithmetic

CPU screening is signed off after Opus 5.5 review. No inference gain was measured and no candidate is ready for a GPU packet. The selected runtime is unchanged.

| Branch | Evidence | Disposition |
| --- | --- | --- |
| G2 scale packing | Exact same-size scale transpose. Baseline compiler artifacts reproduced; VGPR 109→110, barrier pairs 4→6, LDS 4096 bytes in both arms. | Reject this variant before GPU use. A new explicit packed-weight layout needs its own mechanism and proof. |
| G3 lossless coding | 93 checkpoint tiles: 385152 raw bytes →357792 coded bytes, including tables, headers, offsets and alignment. Every CPU round trip exact. | Park GPU codec work. The 7.10% storage saving has no measured GPU decoding cost or serving benefit. |
| Integer reduction | A synthetic serial/group association difference survives the final BF16 cast, even with normal operands and a weight scale inside the sampled range. | Keep blocked pending selected WMMA and whole-K equivalence. This is not an observed GPU failure. |
| Other scheduling | Retained N32, scalar bitdecode and scale-prefetch failures remain closed. No new dependency or layout proof survives this screen. | No additional candidate. GEMM-specific stall evidence or an explicit layout is needed. |

The scale-transpose candidate removed two lane shuffles but introduced shared-memory stores, a load and two extra barriers. The compiler assigned different K-groups to different warps, forcing communication between warps. The baseline already divided rows between warps and needed only a shuffle within each warp. The candidate exceeded the frozen register gate on the first shape, so the remaining nine shapes were not compiled. Kernel arithmetic, packed weights, activation quantization, tile geometry and launch grid were unchanged. The production wrapper still requires the original scale strides.

The compression samples use fixed first/middle/last anchors in layers 0, 21, 42 and 63. They are neither random nor a whole-model survey. The header digest was computed and matched across both reads. Sample digests were recorded. File size equals the historical inventory; the full-file digest was not recomputed. Source-file statistics stayed stable during reading. No raw model tensors were retained. The ratio is a byte-count result, not a performance forecast.

The coding split matters: weights shrink from 362496 to 348920 bytes, only 3.75%, before alignment and offsets. Scales shrink from 22656 to 6531 bytes and provide 54.3% of the gross saving before those shared costs. Most scale tiles need two bits per value; one needs one bit and eight need three. Alignment and offsets add 2341 bytes, giving the stated combined total. This strengthens the decision to defer variable-length GPU weight decoding.

The integer example uses synthetic activation scale 110 and a synthetic incoming accumulator. Only weight scale 120 lies inside the observed sample range; the example does not reproduce serving activations or WMMA execution.

Verification: 14 focused CPU tests, 43 public preflight checks, two bounded native file-only audits and one device-free baseline/candidate compiler operation. All three native operations exited naturally and were reaped. Six prior evidence seals were reverified. No GPU launches or model starts occurred.

The [provenance record](../provenance/MXFP4-OPT02-20261103.json) binds the evidence, review and signoff corrections. The original sealed report is unchanged. The 21-reference lead matrix remains intact. Continue to OPT-03 attention work; G2 explicit-layout packing, G3 decoder cost and integer equivalence stay visible as deferred research.
