# Production cleanup measurement

Measure the cleanup with production instrumentation mode in both variants.
Use the retained W5 source as A and the proposed cleanup source as B. Keep the
model, native runtime, all eleven overlays, request bodies, thinking profile,
token caps, cache policy, graph settings, DFlash width, TP2 and 235 W caps equal.
This comparison uses W5 stock attention. It does not include the separately
adopted C4 attention change.

This is an intermediate cleanup checkpoint. Memory recovery and the remaining
refactors are not included. A result with no measurable gain is valid; production
mode was already enabled in the retained W5 baseline.

The preparation tool authenticates all 313 donor members. It stages the original
79 review files and a historical manifest, then uses the existing proposal
renderer for the cleanup candidate. It compares every retained file outside the
reviewed cleanup changes, all active overlays and both native execution
contracts. Settings and compressed request assets must be byte-identical.
The output is a source review with no execution authorization.

```sh
python3 -I -B -S serving/prepare_measurement.py \
  --donor /absolute/path/to/phase2-128k-w5p-ff396f8f-preparation-v1 \
  --output /absolute/path/to/new-measurement-review \
  --packet-id 0123456789abcdef \
  --hyperloom-root /absolute/path/to/ArchLinuxThreadripperAI
```

The output must be fresh and outside the donor and canonical source trees.
`baseline-source/HISTORICAL-MANIFEST.json` records provenance only. Never use
that historical manifest or its boot identity to launch another campaign.
The cleanup manifest and approval remain unfunded templates. Output packet IDs
are proposals; an executed ID cannot be reused.

Use the existing owner workflow to prepare separate A and B campaigns. The
supported W5 menu has 176 requests per start and two starts per campaign.
The initial A-then-B pilot therefore has four starts and 704 responses. Each
campaign retains its configured 6,360-second owner ceiling and 7,160-second
caller ceiling. These ceilings are allowances, not predicted durations. A
shorter menu needs the configuration-mutation and complete reader gates first.
Each campaign needs its own exact one-use approval. Stop and review after each
result; do not schedule an automatic sequence or retry.

Compare like owner/logical arms, request groups, tasks, concurrency and cache states. Exclude
warmup and non-timing groups. Report combined TP2 output throughput as total
output tokens divided by the sum of the corresponding HTTP group walls. Also
report request latency and the concurrent pair makespan. The HTTP interval
includes prefill and completion; it does not establish isolated decode speed,
time to first token or inter-token latency.

Require equal full output IDs, prompt IDs and speculative work before calling a
timing change a matched-work cleanup improvement. Retain every mismatch and all
groups. If generated work differs, report the observed latency and output sizes
without attributing a speedup to cleanup. Authenticate native quality and
independent release separately. A single A-then-B pilot is descriptive. Prepare
a separately approved B-then-A confirmation before attributing a repeatable gain.

After both campaigns pass native quality and independent release, use the local
comparison reader. Each input is a sealed execution directory containing its
`RESULT.json` and the CPU evidence seal. The reader verifies those seals and the
four original source, operation, workload and protected evidence seals. It
compares all 352 prompt and output vectors, speculative work, cache work,
sampling parameters and finish reasons. It reads both historical and compressed
receipt formats through the cleanup packet's authenticated codec.

```sh
python3 -I -B -S serving/compare_measurement.py \
  --baseline /absolute/path/to/baseline-cpu-execution \
  --baseline-seal BASELINE_FINAL_MANIFEST_SHA256 \
  --candidate /absolute/path/to/cleanup-cpu-execution \
  --candidate-seal CLEANUP_FINAL_MANIFEST_SHA256 \
  --sealer /absolute/path/to/baseline-caller/seal_evidence.py \
  --output /absolute/path/to/new-comparison
```

The output directory must be fresh and outside `serving` and every sealed input.
`COMPARISON.json` retains all groups and work differences. `comparison.html`
shows the four arm/concurrency populations. Rates use summed output tokens and
summed HTTP group walls. A work mismatch keeps the timing descriptive and
prevents matched-work attribution. The reader starts no model and grants no
execution authority.

Hyperloom can help with later offline reporting and trace analysis. Its reviewed
owner adapter admits four- or thirty-request schedules with a 5,400-second owner
budget. It is not qualified for this W5 packet. Preparation records the reviewed
integration source hashes when `--hyperloom-root` is supplied, and refuses if
those schedule guards change. Production-summary import remains unverified.
Keep profilers, automatic optimization and Hyperloom launch delegation outside
this timing run. Use a separately approved diagnostic run for traces if needed.

The initial review remains **not launchable**. The later
[GPU preparation](../gpu-run/README.md) supplies sealed campaign sources, caller
binding, native CPU source checks, grading templates and measured W5 capacity
margins. Both separately approved campaigns have now completed and passed native
quality and independent release. All 352 A/B prompt/output vectors and work
counters match. See the [pilot result](../gpu-run/CLEANUP-B-RESULT.json).
It establishes no repeatable performance improvement. The campaign allowances
are consumed, and the broader cleanup remains incomplete.
