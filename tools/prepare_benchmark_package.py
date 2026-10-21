"""Build only the reviewed benchmark package; never create a grant or launch.

The existing runtime proof is an input, not an output. The full cache-free
SGLang import tree and the four native sources have exact inventories. Other
composition documentation is outside those executable scopes and is not claimed
as an authenticated import source.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import io
import tarfile


def controller_module():
    path = Path(__file__).with_name("benchmark_physical_controller.py")
    spec = importlib.util.spec_from_file_location("spry_package_controller", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def mapping_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def runtime_inventory(facts, proof_sha256):
    """Derive only authenticated source identities; no filesystem writes."""
    c = controller_module()
    c.need(type(facts) is dict and type(facts.get("runtime_files")) is dict
           and len(facts["runtime_files"]) == 5182 and type(facts.get("native_sources")) is dict
           and set(facts["native_sources"]) == {"codec/codec.h", "runtime/attention.hip",
               "runtime/bf16_math.h", "runtime/uq.h"} and facts.get("binary_sha256") == c.BINARY_SHA256,
           "authenticated complete effective runtime/native inventory required")
    result = {"image-source/python/sglang/" + name: pin for name, pin in facts["runtime_files"].items()}
    result.update(facts["native_sources"])
    result.update({c.PROOF_NAME: proof_sha256, c.BINARY_NAME: c.BINARY_SHA256})
    for name, pin in result.items():
        c._relative(name)
        c.need(c._digest(pin) and "__pycache__" not in name.split("/") and not name.endswith(".pyc"),
               "cache-free canonical runtime member required")
    return result


def make_manifest(*, target_root, boot_id, session_sha256, retained_root, runtime_root,
                  model_root, package_files, resource_manifest_sha256, facts, proof_sha256,
                  retained_base=None):
    """Pure projection after a human parent copies the exact review files."""
    c = controller_module()
    c.require_retained_base_binding(retained_base)
    c.need(set(package_files) == c.LIVE_PACKAGE_NAMES and
           package_files["review/COMPOSITION-FACTS.json"] == c.FACTS_SHA256 and
           package_files["review/BENCHMARK-SPEC2.json"] == c.SPEC_SHA256 and
           package_files["review/MODEL-INVENTORY.json"] == c.MODEL_INVENTORY_SHA256,
           "fixed complete package inputs")
    return {"schema": 2, "scope": "matched-benchmark-physical-package-v2",
        "experiment_mode": "model-benchmark-v1", "target_root": str(target_root),
        "boot_id": boot_id, "session_sha256": session_sha256, "package_root": str(target_root),
        "package_files": package_files, "retained_root": str(retained_root),
        "runtime_root": str(runtime_root), "runtime_files": runtime_inventory(facts, proof_sha256),
        "runtime_scopes": list(c.RUNTIME_SCOPES), "retained_base": retained_base, "build_proof_file": c.PROOF_NAME,
        "native_binary_file": c.BINARY_NAME, "model_root": str(model_root),
        "model_inventory_file": "review/MODEL-INVENTORY.json", "spec_sha256": c.SPEC_SHA256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "launch_authorized": False, "proposed_complete_seconds": 2400}


def make_resource_manifest(*, target_root, session_sha256, retained_root, runtime_root,
                           model_root, package_files, bwrap_sha256):
    """Freeze both scope commands and the read-only, tmpfs-only workload argv."""
    c = controller_module()
    root = Path(target_root)
    value = {"schema": 1, "scope": c.resources.SCOPE, "root": str(root),
        "session_sha256": session_sha256, "benchmark_spec_sha256": c.SPEC_SHA256,
        "gate_path": str(root / "review/benchmark_resource_gate.py"),
        "gate_sha256": package_files["review/benchmark_resource_gate.py"],
        "entry_path": str(root / "review/model_benchmark.py"),
        "entry_sha256": package_files["review/model_benchmark.py"],
        "artifacts_sha256": package_files["review/benchmark_artifacts.py"],
        "bwrap_sha256": bwrap_sha256, "units": {}, "workload_argv": {}, "workload_argv_sha256": {}}
    for arm in ("bf16", "kv4"):
        argv = c.adapter.bwrap_argv(arm, retained_root=retained_root, composition=runtime_root,
            review=root / "review", model=model_root, output=root / "arms" / arm,
            entry=root / "review/model_benchmark.py", spec_sha256=c.SPEC_SHA256,
            bounded_scratch=True, physical_artifacts=True, session_sha256=session_sha256,
            artifacts_sha256=value["artifacts_sha256"])
        value["units"][arm] = "spry-fp8-benchmark-" + session_sha256[:16] + "-" + arm + ".scope"
        value["workload_argv"][arm] = list(argv)
        value["workload_argv_sha256"][arm] = c.resources.argv_sha256(list(argv))
    return c.resources.validate_manifest(value)


def write_exclusive(path, raw, mode=0o444):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short package write")
            view = view[count:]
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_review(destination, sources):
    """Copy fixed bytes into a fresh review directory; never copy runtime proof.

    sources maps exact review basenames to (path, independently supplied SHA256).
    Authentication and bounded held-file reads happen before exclusive writes.
    """
    c = controller_module()
    expected = {name.removeprefix("review/") for name in c.LIVE_PACKAGE_NAMES}
    c.need(type(sources) is dict and set(sources) == expected, "exact review source set required")
    target = Path(destination)
    c.need(target.is_absolute() and target.resolve() == target and target.name == "review",
           "canonical fresh review destination required")
    target.mkdir(mode=0o700, exist_ok=False)
    inventory = {}
    try:
        for name in sorted(sources):
            path, pin = sources[name]
            maximum = 16 << 20 if name == "COMPOSITION-FACTS.json" else 256 << 10 if name.endswith(".py") else 8 << 20
            raw = c.adapter.readout.bench.checked_file(path, pin, maximum=maximum)
            write_exclusive(target / name, raw)
            inventory["review/" + name] = pin
        os.chmod(target, 0o555)
        return inventory
    except BaseException:
        # Preserve partial package bytes and exclusive-write evidence.
        raise


def create_payload(output, *, sources, manifest):
    """Physical payload publication requires the missing retained-base proof."""
    c = controller_module()
    c.require_retained_base_binding(manifest.get('retained_base'))
    return _create_component_payload(output, sources=sources, manifest=manifest)


def _create_component_payload(output, *, sources, manifest):
    """CPU-only byte-packaging fixture; never a physical readiness statement."""
    c = controller_module()
    root = Path(output)
    c.need(root.is_absolute() and root.resolve() == root, "canonical fresh payload output required")
    root.mkdir(mode=0o700, exist_ok=False)
    inventory = copy_review(root / "review", sources)
    c.need(inventory == manifest["package_files"] and set(inventory) == c.LIVE_PACKAGE_NAMES,
           "payload manifest does not bind exact copied files")
    manifest_raw = canonical_json(manifest)
    write_exclusive(root / "MANIFEST.json", manifest_raw)
    archive = root / "payload.tar"
    with archive.open("xb") as stream, tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(["MANIFEST.json"] + list(inventory)):
            path = root / name
            raw = path.read_bytes()
            maximum = 16 << 20 if name in ("MANIFEST.json", "review/COMPOSITION-FACTS.json") else 256 << 10 if name.endswith(".py") else 8 << 20
            c.need(len(raw) <= maximum, "bounded package member")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.uid, info.gid, info.mtime = len(raw), 0o444, 0, 0, 0
            tar.addfile(info, io.BytesIO(raw))
    c.need(archive.stat().st_size <= 16 << 20, "payload archive bound")
    return {"archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "package_sha256": mapping_sha256(inventory),
        "source_manifest_sha256": mapping_sha256({name: pin for name, pin in inventory.items() if name.endswith(".py")}),
        "launch_authorized": False, "physical_ready": False, "qualification": False}


if __name__ == "__main__":
    raise SystemExit("NOT_RUN: use the reviewed human parent; this factory grants no execution")
