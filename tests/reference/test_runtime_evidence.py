"""CPU regression tests for the actual runtime source/binary identity reader."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_runtime_contract import UPSTREAM, uq

FILE = UPSTREAM / "python/sglang/srt/layers/attention/ultraquant_evidence.py"
spec = importlib.util.spec_from_file_location("uq_evidence", FILE)
evidence = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_contract": uq}):
    spec.loader.exec_module(evidence)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.package = self.root / "python/sglang/srt/layers/attention"
        self.package.mkdir(parents=True)
        self.source = self.package / "ultraquant_evidence.py"
        self.source.write_bytes(b"synthetic source fixture\n")
        self.library = self.package / "libuq_hip.so"
        self.library.write_bytes(b"synthetic binary fixture, never loaded\n")
        self.link = self.root / "source-link"
        self.link.symlink_to(self.source.relative_to(self.root))
        self.files = {self.source.relative_to(self.root).as_posix(): evidence.digest(self.source)}
        self.links = {"source-link": str(self.link.readlink())}
        self.value = {
            "source_revision": uq.SOURCE_REVISION, "qualification": "unqualified",
            "runtime_files": self.files, "runtime_links": self.links, "experiment_files": {},
            "binary_sha256": evidence.digest(self.library),
            "source_tree_sha256": hashlib.sha256(evidence.canonical({
                "upstream": self.files, "upstream_links": self.links, "experiment": {}})).hexdigest(),
        }
        self.manifest = self.package / "ultraquant-build.json"
        self.write_manifest()
        context = patch.object(evidence, "__file__", str(self.source))
        context.start()
        self.addCleanup(context.stop)

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.value))

    def test_unchanged_build_is_accepted(self):
        self.assertEqual(evidence.build_identity(), self.value)

    def test_source_binary_addition_and_removal_invalidate(self):
        for path in (self.source, self.library):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(ValueError):
                evidence.build_identity()
            path.write_bytes(original)
        self.source.rename(self.source.with_suffix(".moved"))
        with self.assertRaises(ValueError):
            evidence.build_identity()

    def test_added_source_and_symlink_target_invalidate(self):
        added = self.root / "unexpected.py"
        added.write_bytes(b"extra source")
        with self.assertRaises(ValueError):
            evidence.build_identity()
        # Removing a synthetic fixture is part of this isolated test only.
        added.unlink()
        self.link.unlink()
        self.link.symlink_to("different-target")
        with self.assertRaises(ValueError):
            evidence.build_identity()

    def test_aggregate_and_qualification_are_checked(self):
        self.value["source_tree_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            evidence.build_identity()
        self.value["qualification"] = "qualified"
        self.write_manifest()
        with self.assertRaises(ValueError):
            evidence.build_identity()

    def test_descriptor_binds_configuration_and_is_never_qualified(self):
        self.value.update(patch_sha256="1" * 64, compiler_revision="synthetic-test-only")
        with patch.object(evidence.importlib.metadata, "version", return_value="fixture"):
            descriptor = evidence.descriptor(self.value, 128, 4, 16, "fixture")
        self.assertEqual(descriptor["qualification"], "unqualified")
        self.assertTrue(descriptor["restart_required"])
        self.assertEqual(descriptor["configuration_sha256"], hashlib.sha256(
            json.dumps(descriptor["configuration"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest())
        self.assertEqual(descriptor["binary_sha256"], self.value["binary_sha256"])


if __name__ == "__main__":
    unittest.main()
