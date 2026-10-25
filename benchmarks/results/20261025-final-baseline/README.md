# Final retained-baseline checkpoint, 25 October 2026

Final bounded QA and the final two-start BetterBench checkpoint passed. [Final tables and provenance](final-summary.html) report106 requests, independently reviewed native/owner joins and release. The historical BetterBench attempt 45 remains incomplete because its evidence exporter refused the native subtree. The corrected operation 46 exported all53first-start requests, then failed the strict native-counter join; its second start did not run. No new tuning candidate was selected and no new performance result is accepted from that attempt.

| Check | Result | Scope |
| --- | --- | --- |
| Fresh starts and requests | 2 starts, 16 requests | Unchanged b101/2K/BF16-KV/fixed-eight/low-thinking baseline |
| Native final-answer grading | 14/14 pass | Seven unique graded tasks repeated across starts |
| Isolated generated-code grading | 2/2 pass | Frozen Python task, explicit functional/format checks |
| Corresponding output/work | 8/8 token histories and verifier-count triples match | Frozen cases; no general numerical-equivalence claim |
| GPU and CPU release | Verified independently | Exact process, namespace, scope and GPU-holder checks |
| Final BetterBench | 2 starts,106 requests, strict checks/release PASS | Bounded mixed-workload checkpoint; no correctness oracle |
| Historical attempts 45/46 | INCOMPLETE, preserved | Export refusal then non-atomic counter join; not regraded |
| Endurance and production qualification | NOT RUN | Separate long-duration contract remains undefined |

[qa-summary.json](qa-summary.json) records the actual input/output/work comparisons and grades. QA host cgroup peaks were 26,047,483,904 and 21,946,019,840 bytes. These include the process tree and client; they are neither per-card VRAM nor a memory-leak measurement. No host OOM or memory-limit event was recorded.

[betterbench-incomplete.json](betterbench-incomplete.json) preserves the failed 561-second operation, exact package/source fingerprints and verified release. The original failure is not regraded as successful. [Consumed client source](../../betterbench/native-v2/README.md) remains available with explicit upstream patch and dependency identities. Corrected packing work is separate from the selected serving runtime.

The last comparable Sprint 39 service measurements remain C1 **119.95–120.18 output tok/s**, median request **8.50–8.52 seconds**, and C2 **172.75–172.98 aggregate tok/s**, median individual **11.76–11.87 seconds**. Rates cover both GPUs together and include prefill and HTTP completion. Different output paths and verifier counts prevent a matched-work kernel-speed comparison. These are historical retained measurements, not new BetterBench results.

See the [programme disposition matrix](../../../docs/REMAINING-FP8-OPTIMIZATIONS.md) for rejected, inconclusive, blocked and deferred candidates. The selected runtime is unchanged; publication does not deploy a server.

The [forensic first-start JSON](forensic-first-start.json) and [standalone table](forensic-first-start.html) report provisional mixed-workload C1 **81.3188 tok/s** and C2 **144.8048 aggregate tok/s**. These are not an accepted two-start checkpoint or a comparable regression test against Sprint 39. Request/output lengths and verifier work differ. See the [counter failure](betterbench-counter-incomplete.json); original failed statuses remain unchanged. Total owner charge is1,119seconds across both operations. All cache counters were unknown.

Final completed measurements (both GPUs together, including prefill/HTTP completion and instrumentation):

| Start | C1 aggregate tok/s | C1 median request seconds | C2 aggregate tok/s | C2 median individual seconds |
| --- | ---: | ---: | ---: | ---: |
| A | 81.4279 | 7.0087 | 134.8680 | 7.2174 |
| B | 87.2708 | 7.8549 | 133.1114 | 7.0678 |

C1 output totals are 4,450/4,618 and C2 totals 3,638/3,835 tokens per six-request group. Generated work differs; these natural-stop mixed tasks are not comparable with the earlier fixed-output Sprint 39 rates. All 106 cache counters are unknown. Measured prefill series exclude one warm-up at each depth; actual measured inputs were 1,520–1,577,5,903–5,990 and11,753–11,837 tokens. Client content TTFT was0.450–0.470,1.710–1.732 and3.599–3.626 seconds respectively. No uncached-prefill, pure-kernel or native per-token timing claim is made.

[Native JSON/HTML A](start-a.html) and [B](start-b.html) preserve the original output reports; [final-summary.json](final-summary.json) retains all 106 numeric rows, stream protocol/speculation fields, source pins and resource evidence. Host cgroup peaks were 22,198,013,952 and22,933,823,488 bytes; OOM/memory-limit events were zero. These are process-tree measurements, not per-card VRAM. Both historical failures remain; final operation charged 1,051 seconds and cumulative owner charge is 2,170 seconds. No new runtime candidate was promoted, no persistent service was deployed, and endurance remains NOT RUN.

Publication verification:43 repository source checks, three portable journal tests and the native source checks passed. Verbatim source patches/fixtures and native HTML retain whitespace-only lines flagged by Git. Native HTML is deliberately byte-identical to the reviewed output; its hashes are recorded rather than silently reformatted. This warning does not change the measured data.
