"""One bounded compiler-driver workflow. Produces unreviewed facts only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import bf16_source_identity as source

FLAGS = source.FLAGS
TOOLS = ("hipcc", "clang", "nm", "readelf", "objdump", "objcopy", "bundler")
EXPORTS = frozenset({"uq_device", "uq_initialize", "uq_append", "uq_attention",
                     "uq_bf16_append_v2", "uq_bf16_attention_v2"})
V2_KERNELS = source.V2_KERNELS
DRIVER_SECONDS = 425
# Seven tool receipts, one compiler call, seven extraction/inspection calls.
COMMAND_WORK_SECONDS = 7 * 5 + 180 + 7 * 15
COMMAND_CLEANUP_SECONDS = 15 * 5
COLLECTION_SECONDS = 30
assert COMMAND_WORK_SECONDS + COMMAND_CLEANUP_SECONDS + COLLECTION_SECONDS == DRIVER_SECONDS


BuildError = source.ProtocolError


def utc():
    return datetime.now(timezone.utc).isoformat()


def inspect_outputs(symbols, metadata, disassembly):
    return source.inspect_outputs(symbols, metadata, disassembly)


def command_plan(root, attempt, tools, device_bundle="<enumerated-device-bundle>"):
    return source.commands(root, attempt, tools, device_bundle)


class BoundedRunner:
    """Own each tool process group; terminate/reap it on every exit path."""
    def __init__(self, attempt, deadline, env):
        self.attempt, self.env = Path(attempt), env
        self.stop = time.monotonic() + min(DRIVER_SECONDS, deadline - time.time())
        self.records, self.active = [], None

    def run(self, label, argv, work):
        if time.monotonic() + work + 5 > self.stop - COLLECTION_SECONDS:
            raise BuildError("complete command envelope does not fit")
        stdout, stderr = self.attempt / (label + ".stdout"), self.attempt / (label + ".stderr")
        record = {"argv": argv, "cwd": str(self.attempt), "started": utc(), "work_seconds": work,
                  "stdout": stdout.name, "stderr": stderr.name}
        self.records.append(record)
        start = time.monotonic()
        try:
            with stdout.open("xb") as out, stderr.open("xb") as err:
                self.active = subprocess.Popen(argv, cwd=self.attempt, env=self.env,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                record["pgid"] = self.active.pid
                try:
                    record["exit"] = self.active.wait(timeout=work)
                except subprocess.TimeoutExpired as error:
                    raise BuildError("tool work timeout") from error
                if record["exit"]:
                    raise BuildError("tool failed: " + label)
        finally:
            if self.active is not None:
                pid = self.active.pid
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(pid, sig)
                    except ProcessLookupError:
                        pass
                    if sig == signal.SIGTERM:
                        try:
                            self.active.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                self.active.wait(timeout=2)
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    record["group_absent"] = True
                else:
                    record["group_absent"] = False
                self.active = None
            else:
                record["group_absent"] = True
            record.update(finished=utc(), elapsed=time.monotonic() - start)
            (self.attempt / "commands.json").write_bytes(source.canonical(self.records) + b"\n")
            if not record["group_absent"]:
                raise BuildError("owned tool process group remains")
        if sum(p.stat().st_size for p in self.attempt.rglob("*") if p.is_file()) > 1024**3:
            raise BuildError("attempt artifact limit exceeded")
        return stdout.read_text(encoding="utf-8")


def build(root, stock, baseline, baseline_hash, attempt, tools, deadline, *,
          stack_receipt=None, stack_receipt_sha256=None):
    """Future native grant only. No library load, device discovery or execution."""
    root, attempt = Path(root), Path(attempt)
    if not root.is_absolute() or not attempt.is_absolute() or attempt.exists():
        raise ValueError("absolute fresh owned attempt required")
    if ".." in root.parts or ".." in attempt.parts:
        raise ValueError("noncanonical source/output path")
    root = source.canonical_directory(root, "candidate")
    stock = source.canonical_directory(stock, "stock")
    if (any(parent.samefile(root) for parent in (stock, *stock.parents))
            or any(parent.samefile(stock) for parent in root.parents)):
        raise ValueError("candidate and stock trees must be distinct")
    if any(p.is_symlink() for p in (attempt, *attempt.parents)) or not attempt.parent.is_dir():
        raise ValueError("unsafe attempt parent")
    if any(parent.samefile(root) for parent in attempt.parents):
        raise ValueError("outputs must be outside the candidate source tree")
    if any(parent.samefile(stock) for parent in attempt.parents):
        raise ValueError("outputs must be outside the authenticated stock tree")
    if attempt.parent.resolve(strict=True) != attempt.parent:
        raise ValueError("noncanonical output parent")
    if type(deadline) not in (int, float) or not 0 < deadline - time.time() <= DRIVER_SECONDS + 1:
        raise ValueError("future absolute driver deadline required")
    if deadline - time.time() < DRIVER_SECONDS - 1:
        raise ValueError("full 425-second driver envelope required")
    if set(tools) != set(TOOLS):
        raise ValueError("exact absolute toolchain paths required")
    stack_args = {"stack_receipt": stack_receipt, "stack_receipt_sha256": stack_receipt_sha256}
    identity, patch = source.inventory(root, stock, baseline, baseline_hash, **stack_args)
    inputs = source.build_inputs(root, identity)
    tool_specs = dict(tools)
    tool_identities = {k: source.tool_identity(v) for k, v in tools.items()}
    if "stack_receipt_sha256" in identity:
        source.verify_installed_stack(root, stack_receipt, stack_receipt_sha256, tool_identities)
    tools = {k: v["path"] for k, v in tool_identities.items()}
    if Path(tools["clang"]).name != "clang++":
        raise ValueError("unsupported compiler selection: clang++ required")
    attempt.mkdir(mode=0o700)
    temporary = attempt / "tmp"
    temporary.mkdir(mode=0o700)
    env = source.compiler_environment(identity, tools, temporary)
    plan = command_plan(root, attempt, tools)
    (attempt / "COMMAND-PLAN.json").write_bytes(source.canonical({"commands": plan, "environment": env,
        "deadline_epoch": deadline, "maximum_driver_seconds": DRIVER_SECONDS}) + b"\n")
    (attempt / "SOURCE-IDENTITY.json").write_bytes(source.canonical(identity) + b"\n")
    (attempt / "BUILD-INPUTS.json").write_bytes(source.canonical(inputs) + b"\n")
    (attempt / "kv-image-relative.patch").write_bytes(patch)
    runner = BoundedRunner(attempt, deadline, env)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise BuildError("driver termination requested")
    signal.signal(signal.SIGTERM, terminate)
    facts = {"schema": 2, "status": "BUILD_FAILED", "qualification": "unreviewed",
             "source_identity": identity, "build_inputs": inputs,
             "source_promotion": False, "launch_authorized": False,
             "gpu_execution": "not-run", "native_library_loaded": False}
    try:
        outputs = {}
        library_hash = None
        container_hash = None
        for label, argv, limit in plan:
            if label == "unbundle":
                unused, device = source.bundle_records(source.binary(attempt / "hip-fatbin.bin"))
                listed = outputs["enumerate"].splitlines()
                if len(listed) != len(set(listed)) or set(listed) != set(unused):
                    raise BuildError("bundle enumeration mismatch")
                argv = next(a for n, a, _ in command_plan(root, attempt, tools, device) if n == "unbundle")
            outputs[label] = runner.run(label, argv, limit)
            if label == "compile-library":
                library_hash = source.digest(attempt / "libuq_bf16_v2.so")
                receipt = outputs[label] + (attempt / "compile-library.stderr").read_text()
                compiler = source.compiler_binding(receipt, root, attempt, tool_identities, env)
            if label == "extract":
                container_hash = source.digest(attempt / "hip-fatbin.bin")
            if library_hash and source.digest(attempt / "libuq_bf16_v2.so") != library_hash:
                raise BuildError("library changed after compilation")
            if container_hash and source.digest(attempt / "hip-fatbin.bin") != container_hash:
                raise BuildError("container changed after extraction")
        for name in ("libuq_bf16_v2.so", "attention-gfx1201.hsaco"):
            if source.regular(attempt / name).stat().st_size == 0:
                raise BuildError("empty compiler output")
        inspection = inspect_outputs(outputs["exports"], outputs["architecture"], outputs["disassembly"])
        payload = source.payload_binding(attempt)
        if source.inventory(root, stock, baseline, baseline_hash, **stack_args)[0] != identity or source.build_inputs(root, identity) != inputs:
            raise BuildError("source or driver changed during build")
        if {k: source.tool_identity(v) for k, v in tool_specs.items()} != tool_identities:
            raise BuildError("tool changed during build")
        if "stack_receipt_sha256" in identity:
            source.verify_installed_stack(root, stack_receipt, stack_receipt_sha256, tool_identities)
        toolchain = {k: dict(v, version=outputs["version-" + k].strip()) for k, v in tool_identities.items()}
        if any(not item["version"] for item in toolchain.values()):
            raise BuildError("empty tool revision")
        artifacts = source.tree(attempt)
        facts.update(status="BUILD_INSPECTED_UNREVIEWED", toolchain=toolchain, inspection=inspection,
            compiler_revision="clang-sha256:" + tool_identities["clang"]["sha256"] + " hipcc-sha256:" + tool_identities["hipcc"]["sha256"],
            compiler_flags=list(FLAGS), artifacts=artifacts, commands=runner.records,
            compiler_binding=compiler, payload_binding=payload, environment=env,
            protocol=identity.get("installed_protocol", source.PROTOCOL))
        source.verify_protocol_facts(facts, attempt, root)
    except BaseException as error:
        facts.update(status="BUILD_FAILED", error=type(error).__name__ + ": " + str(error), commands=runner.records)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        facts["finished"] = utc()
        (attempt / "BUILD-FACTS.json").write_bytes(source.canonical(facts) + b"\n")
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "stock", "baseline", "baseline-sha256", "attempt", "tools", "deadline-epoch"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stack-receipt")
    parser.add_argument("--stack-receipt-sha256")
    args = parser.parse_args()
    build(Path(args.root), Path(args.stock), Path(args.baseline), args.baseline_sha256,
          Path(args.attempt), source.load_json(args.tools), float(args.deadline_epoch),
          stack_receipt=args.stack_receipt, stack_receipt_sha256=args.stack_receipt_sha256)


if __name__ == "__main__":
    main()
