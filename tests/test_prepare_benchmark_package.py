"""CPU-only payload projection; inert assets cannot authorize native execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


factory = load("package_factory_tests", ROOT / "tools/prepare_benchmark_package.py")
controller = factory.controller_module()


class PackageTests(unittest.TestCase):
    def test_runtime_scope_is_complete_cache_free_and_never_emits_a_proof(self):
        facts = {"runtime_files": {f"inert-{i:04d}.py": "a" * 64 for i in range(5182)},
            "native_sources": {name: "b" * 64 for name in ("codec/codec.h", "runtime/attention.hip",
                "runtime/bf16_math.h", "runtime/uq.h")}, "binary_sha256": controller.BINARY_SHA256}
        with patch.object(factory, "controller_module", return_value=controller):
            mapping = factory.runtime_inventory(facts, "c" * 64)
            self.assertEqual(len(mapping), 5188)
            self.assertEqual(mapping[controller.PROOF_NAME], "c" * 64)
            bad = dict(facts, runtime_files=dict(facts["runtime_files"]))
            bad["runtime_files"].pop("inert-0000.py")
            with self.assertRaises(ValueError):
                factory.runtime_inventory(bad, "c" * 64)
            bad["runtime_files"]["__pycache__/stale.pyc"] = "a" * 64
            with self.assertRaises(ValueError):
                factory.runtime_inventory(bad, "c" * 64)

    def test_actual_payload_contains_only_exact_regular_bytes_and_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            sources = {}
            for member in sorted(controller.LIVE_PACKAGE_NAMES):
                name = member.removeprefix("review/")
                # Source bytes are real, assets are explicitly inert CPU data.
                data = (ROOT / "tools" / name).read_bytes() if name.endswith(".py") else b'{"inert_cpu_fixture":true}\n'
                path = inputs / name
                path.write_bytes(data)
                sources[name] = (path, hashlib.sha256(data).hexdigest())
            inventory = {"review/" + name: pin for name, (_, pin) in sources.items()}
            manifest = {"package_files": inventory, "launch_authorized": False}
            result = factory._create_component_payload(root / "payload", sources=sources, manifest=manifest)
            archive = Path(result["archive"]).read_bytes()
            self.assertEqual(hashlib.sha256(archive).hexdigest(), result["archive_sha256"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                self.assertEqual(set(tar.getnames()), set(inventory) | {"MANIFEST.json"})
                for member in tar:
                    self.assertTrue(member.isfile())
                    self.assertEqual(member.mode, 0o444)
                    raw = tar.extractfile(member).read()
                    if member.name == "MANIFEST.json":
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["manifest_sha256"])
                    else:
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), inventory[member.name])
            self.assertFalse(result["launch_authorized"])
            self.assertFalse(result["qualification"])
            self.assertFalse(result["physical_ready"])
            self.assertEqual(result["package_sha256"], factory.mapping_sha256(inventory))
            with self.assertRaises(FileExistsError):
                factory._create_component_payload(root / "payload", sources=sources, manifest=manifest)

    def test_copy_refuses_missing_sources_and_changed_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaises(ValueError):
                factory.copy_review(root / "review", {})
            path = root / "inert"
            path.write_bytes(b"declared CPU payload")
            sources = {name.removeprefix("review/"): (path, "0" * 64)
                       for name in controller.LIVE_PACKAGE_NAMES}
            with self.assertRaises(ValueError):
                factory.copy_review(root / "review", sources)
            self.assertTrue((root / "review").exists())

    def test_physical_payload_refuses_missing_retained_base_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "no-physical-payload"
            with self.assertRaisesRegex(ValueError, "NOT_READY.*base-rootfs"):
                factory.create_payload(target, sources={}, manifest={})
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
