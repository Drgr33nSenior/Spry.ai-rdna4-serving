# FP8 weight-layout compile gate

Status: **rejected at the declared compile-resource gate, unselected**.
The first rank-0 packing case completed; candidate matmul and the second-card
case were **NOT RUN**. This is a feasibility result, not numerical or performance
qualification. [Provenance](../provenance/FP8-LAYOUT-COMPILE-GATE-20261024.json)

The candidate reversibly packs one real gate/up weight shard while retaining
the original K128 block scales and ordered dot/scale/add arithmetic. Exact
packing/inverse checks passed for 89,128,960 weight bytes without requantization
or scale modification. Packing took 10.857 ms in this case. Real activation
operands were not located; the admitted activation case was seeded.

| Compiler field | Selected | Packed candidate |
| --- | ---: | ---: |
| Registers | 114 | 256 |
| Reported spills | 0 | 81 |
| Shared memory, bytes | 16384 | 16384 |
| Explicit weight-LDS paths | 1 | 1 |
| Static K-loop instruction lines | 153 | 797 |
| Static buffer loads | 11 | 131 |
| Static DS loads / stores | 20 / 11 | 20 / 131 |
| Static waits | 26 | 109 |

Both builds used one stage and four warps. The packed build failed the declared
maximum of 128 registers and zero spills. The retained TTGIR also preserves a
weight-LDS path, so it does not establish the intended direct-register load
mechanism. WMMA count remains 16; K128 reduction, As-then-Bs scaling and
sequential FP32 accumulation are retained. Static instruction counts are not
executed traffic, utilisation or cycle measurements.

Original and packed TTGIR, AMDGPU assembly and HSACO were retained before handle
initialization. Handles initialized without a matmul. The gate stopped the
campaign at the first resource rejection; the toolchain was not classified as
unsupported. No generic stage/warp sweep or automatic retry followed.

Peak allocated device memory was 248,086,528 bytes, below the 256 MiB tensor
limit. Peak reserved memory, recorded separately, was 270,532,608 bytes. Host
peak was 1,004,466,176 bytes and final phase allocation was zero. Payload time
was 8.872 seconds and owner time 26.706 seconds. Independent release passed
with no cleanup errors or unresolved journal.

Astra independently recomputed the exact reader result, compiler analysis and
resource/release joins. Native evidence is under
`scratchpad/fp8-layout-v5-gpu-run.sipx4d_z`, seal
`5462d9fd2a8b6e061978a0d7908702be251535186f8e604718a860961390ab4f`
(175 members). The source seal is
`e903232a1ee29a2b090bdf0bb17833a31acf1159ff7fec719263a2709c1a4b4e`.
The generated code supports further source diagnosis, not a serving speedup or
all-layer memory-fit claim. Selected b101 remains unchanged.
