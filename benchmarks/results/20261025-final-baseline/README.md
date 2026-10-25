# Final retained-baseline checkpoint, 25 October 2026

Final bounded QA passed. BetterBench attempt 45 is incomplete because its evidence exporter refused the native subtree. The corrected operation46 exported all53first-start requests, then failed the strict native-counter join; its second start did not run. No new tuning candidate was selected and no new performance result is accepted from that attempt.

| Check | Result | Scope |
| --- | --- | --- |
| Fresh starts and requests | 2 starts, 16 requests | Unchanged b101/2K/BF16-KV/fixed-eight/low-thinking baseline |
| Native final-answer grading | 14/14 pass | Seven unique graded tasks repeated across starts |
| Isolated generated-code grading | 2/2 pass | Frozen Python task, explicit functional/format checks |
| Corresponding output/work | 8/8 token histories and verifier-count triples match | Frozen cases; no general numerical-equivalence claim |
| GPU and CPU release | Verified independently | Exact process, namespace, scope and GPU-holder checks |
| BetterBench | Incomplete; second start NOT RUN | Native export guard refusal; underlying client outcome not authenticated |
| Endurance and production qualification | NOT RUN | Separate long-duration contract remains undefined |

[qa-summary.json](qa-summary.json) records the actual input/output/work comparisons and grades. QA host cgroup peaks were 26,047,483,904 and 21,946,019,840 bytes. These include the process tree and client; they are neither per-card VRAM nor a memory-leak measurement. No host OOM or memory-limit event was recorded.

[betterbench-incomplete.json](betterbench-incomplete.json) preserves the failed 561-second operation, exact package/source fingerprints and verified release. The original failure is not regraded as successful. [Consumed client source](../../betterbench/native-v2/README.md) remains available with explicit upstream patch and dependency identities. Corrected packing work is separate from the selected serving runtime.

The last comparable Sprint 39 service measurements remain C1 **119.95–120.18 output tok/s**, median request **8.50–8.52 seconds**, and C2 **172.75–172.98 aggregate tok/s**, median individual **11.76–11.87 seconds**. Rates cover both GPUs together and include prefill and HTTP completion. Different output paths and verifier counts prevent a matched-work kernel-speed comparison. These are historical retained measurements, not new BetterBench results.

See the [programme disposition matrix](../../../docs/REMAINING-FP8-OPTIMIZATIONS.md) for rejected, inconclusive, blocked and deferred candidates. The selected runtime is unchanged; publication does not deploy a server.

The [forensic first-start JSON](forensic-first-start.json) and [standalone table](forensic-first-start.html) report provisional mixed-workload C1 **81.3188 tok/s** and C2 **144.8048 aggregate tok/s**. These are not an accepted two-start checkpoint or a comparable regression test against Sprint39. Request/output lengths and verifier work differ. See the [counter failure](betterbench-counter-incomplete.json); original failed statuses remain unchanged. Total owner charge is1,119seconds across both operations. All cache counters were unknown.
