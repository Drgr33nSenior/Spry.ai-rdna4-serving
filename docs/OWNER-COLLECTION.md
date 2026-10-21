# Owner-run platform metadata

`tools/collect_benchmark_platform.py` prepares an availability snapshot for the
FP8-weight BF16-KV versus KV4 benchmark. An authorized human runs it on
`threadripper-ai` (mapped `tst`), not through the agent. The collector never runs
sudo itself, launches a workload, acquires the canonical lock or changes its
journal. It writes only a fresh root-private evidence directory.

The script uses existing administrative access for four fixed read-only checks:
GPU allocations, node health, the retained deny-policy contract, and all-user
device holders. It does not query Secret objects or print/copy kubeconfig,
credentials, environment values, workload commands or full Pod JSON. Kubectl
outputs fixed field projections; its internal API authentication remains the
existing platform client's responsibility. The projection syntax follows the
[Kubernetes JSONPath reference](https://kubernetes.io/docs/reference/kubectl/jsonpath/).

## Before running

Review the collector and use the exact SHA256 supplied by the sealed handoff.
Copy it to a fresh private staging directory on the workstation using your
normal authenticated SSH route. Do not replace an existing staging directory.
Do not disable host-key checking, fetch a different script or install packages.

Use an isolated existing system interpreter with `-I -B -S`. Before executing
Python as root, copy the reviewed bytes into a new root-owned mode-0700
directory and verify their SHA256 there. A self-hash check cannot protect a
user-writable script that could be replaced before it starts. The handoff's
`RUN-ON-WORKSTATION.sh` performs this staging and verification for the human;
the agent must not execute it.

The collector accepts only a fresh output path shaped like
`/var/tmp/fp8-benchmark-metadata.<8-32 alphanumeric nonce>`. It refuses links,
existing output, wrong source bytes, non-root execution or a non-isolated
interpreter. Source staging and evidence directories are retained, not deleted.

Run in the native host SSH session, not a container or user namespace. Root
alone does not establish all-user holder visibility. The collector checks
full UID/GID maps, effective process-inspection and file-access capabilities,
and namespace agreement with visible PID 1. Missing or restricted visibility
refuses before platform commands. These are checked prerequisites, not
independent attestation that the session is in the initial host namespace.

## Result and limits

The intended collection envelope is 60 seconds including child cleanup and
report publication, excluding human transfer and authentication. A timeout,
permission diagnostic, malformed/missing response, active holder/allocation,
changed boot, wrong target, low headroom or unresolved owner cannot pass.
Uninterruptible kernel I/O cannot be proven absent by CPU tests; an incomplete
or externally killed collection is not release evidence.

The result is `REPORT.json`, root-owned mode 0600. The supplied human wrapper
prints this sanitized file after the run. Return that report for review; do not
send kubeconfig, private keys, tokens or unrelated logs. If collection fails,
retain the partial evidence and report the failure. Do not stop another workload
or clear an unresolved record to obtain a passing snapshot.

Current sysfs IDs and render/BDF associations are observations. Corresponding
HIP UUIDs are labelled retained provenance, not new GPU queries. No Torch,
ROCm library or model is loaded. Lock metadata is observed without lock
acquisition, so the snapshot does not reserve devices or prove the lock is free.

`OWNER_METADATA_COLLECTED` means this bounded snapshot passed its checks. It is
not a serving qualification, physical execution grant, durable device handover
or process/namespace release receipt. It cannot be fed directly into the host
owner's release validator. Current admission and release checks must still run
inside the future approved physical workflow. The live controller, callback
bindings and complete envelope remain separately reviewed work.

## Local tests

```sh
make test-platform-collector PYTHON=/path/to/python3
```

Tests substitute root identity, proc/sysfs and platform responses. Bounded local
child fixtures test command collection and failures; they do not contact the
cluster, invoke sudo or initialize a GPU. Existing benchmark/runtime code and
accepted caches stay unchanged. Rollback is non-adoption.
