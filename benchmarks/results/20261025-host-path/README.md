# Host observation A/B, 25 October 2026

The two-start, 106-request comparison passed strict evidence checks and independent release. See the [standalone table](summary.html), [full measurements](summary.json) and [interpretation](../../../docs/HOST-PATH-CLEANUP.md).

Control C1/C2 rates were 83.12/132.21 output tok/s; candidate rates were 77.85/135.25. Rates cover both GPUs, prefill and HTTP completion. Output and verifier work differed, so this is not a matched-work speed comparison. No useful performance winner was selected.

The native control/candidate JSON and HTML files are copied verbatim. Astra source and result reviews are included. Source tests are in [host-path](../../host-path/README.md). No deployment or endurance qualification follows.

Git whitespace checks flag ten whitespace-only lines in the verbatim native HTML and three blank context lines in the unified source patch. These are retained to preserve the reports and applicable patch bytes; no other staged whitespace findings were reported.
