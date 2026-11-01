# Canonical caller source

`power_observe.py` replaces the caller's copied device map and power value with
the canonical host configuration. The generated caller binding must supply
`host_config` and `host_sha256`. The file must be canonical, bounded, singly
linked JSON below `config/host/`. Unknown fields, incorrect pins, duplicate
render nodes, type aliases and caps above the fixed 235 W ceiling refuse before
remote work. The configured watts are converted to integer microwatts.

The remote program retains the existing boot, owner-journal, root-freshness,
render/sysfs and descriptor-close checks. It observes power caps and refuses a
mismatch. It changes no cap or runtime power-management setting. The native
render-node wake and existing root observer remain operations for the reviewed,
approved caller; this source file does not authorize their execution.

The [review renderer](PROJECTION.md) now stages the pinned configuration and
retained `connection.py` and `local_commands.py` dependencies with a fresh binding
template. The later [GPU preparation](../measurements/gpu-run/README.md) stages
`run_native.py` and `execute_once.py` with exact source, approval and boot checks.
Its one-use entry point retains failed transport results and attempts collection
and postchecks after launch failure. Registration and execution remain pending.
This directory is canonical source, not a registered launch packet.

```sh
python3 -I -B -S serving/tests/test_caller_power.py -v
```

All six tests replace SSH, the observer check and command execution with mocks.
They cover 235 W and 230 W configurations, strict validation, observed mismatch,
failure retention and the observation-only program. No remote connection,
privilege escalation, device wake or GPU operation is executed.
