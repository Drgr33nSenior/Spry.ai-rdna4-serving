"""Synthetic modern stack identities and static ELF data; no native execution."""
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_modern_stack import ROOT, OWNED, source, collector
from test_bf16_closure import SyntheticBuild, evidence, uq


class ModernIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OWNED / ".test-build/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.root = self.folder / "candidate"
        shutil.copytree(OWNED / "prototype", self.root)
        self.common = self.folder / "common"
        shutil.copytree(OWNED / "tests/fixtures/stock-v0521/python/sglang", self.common)
        self.contract = source.stack_contract(self.root)
        self.interpreter = self.folder / "INERT-interpreter"
        self.interpreter.write_bytes(b"INERT not executable")
        self.library = self.folder / "lib/INERT-library"
        self.library.parent.mkdir()
        self.library.write_bytes(b"INERT library, never loaded")
        self.tools = {}
        for role in source.TOOL_NAMES:
            path = self.folder / ("clang++" if role == "clang" else role)
            path.write_bytes(b"INERT tool role " + role.encode())
            self.tools[role] = {"path": str(path), "sha256": source.digest(path)}
        environment = source.compiler_environment({}, {"clang": self.tools["clang"]["path"]}, self.folder)
        keep = {"PATH", "ROCM_PATH", "HIP_CLANG_PATH", "HIP_PLATFORM", "HIP_COMPILER", "HIPCC_VERBOSE",
                "OFFLOAD_BUNDLER_IGNORE_ENV_VAR", "OFFLOAD_BUNDLER_COMPRESS"}
        self.receipt = {"schema": 1, "decision": "accept-common-stack-source-only",
            "owner_handoff": "INERT CPU fixture, NOT owner authority", "target": self.contract.TARGET,
            "image_manifest": "a"*64, "image_config": "b"*64, "installed_inventory_sha256": "c"*64,
            "components": {name: {"version": "10.1.0" if name == "sdk" else "INERT", "artifact_sha256": "d"*64}
                           for name in ("python", "sdk", "hip", "torch", "triton", "rccl", "kernel_packages", "aiter")},
            "device_targets": ["gfx1201"], "stock_files": source.tree(self.common),
            "common_files": source.tree(self.common), "repair_files": {},
            "tools": {role: dict(tool, interpreter=str(self.interpreter),
                       interpreter_sha256=source.digest(self.interpreter), filesystem_root=str(self.folder),
                       libraries={"lib/INERT-library": source.digest(self.library)}) for role, tool in self.tools.items()},
            "compiler_environment": {key: environment[key] for key in keep},
            "static_closure_complete": True, "dynamic_plugins_reviewed": True,
            "installed_protocol_reviewed": True, "protocol": {"label": "INERT syntax, not installed evidence"}}
        self.path = self.folder / "INERT-stack.json"
        self.baseline_path = self.folder / "INERT-baseline.json"
        self.write_receipt()
        self.baseline = {"schema": 2, "image_manifest": "a"*64, "image_config": "b"*64,
            "stack_receipt_sha256": self.receipt_hash, "stock_files": source.tree(self.common),
            "effective_files": source.tree(self.root / "image-source/python/sglang", source.EXCLUDED),
            "native_sources": {name: source.digest(self.root/name) for name in source.NATIVE}}
        self.baseline["added_files"] = sorted(set(self.baseline["effective_files"])-set(self.baseline["stock_files"]))
        self.write_baseline()

    def write_receipt(self):
        self.path.write_bytes(source.canonical(self.receipt))
        self.receipt_hash = source.digest(self.path)

    def write_baseline(self):
        self.baseline_path.write_bytes(source.canonical(self.baseline))
        self.baseline_hash = source.digest(self.baseline_path)

    def inventory(self):
        return source.inventory(self.root, self.common, self.baseline_path, self.baseline_hash,
                                stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)

    def test_positive_exact_modern_inventory_and_patch(self):
        value, delta = self.inventory()
        self.assertEqual(value["runtime_files"], self.baseline["effective_files"])
        self.assertEqual(value["stock_files"], self.baseline["stock_files"])
        self.assertEqual(value["source_sha256"], source.sha(source.canonical(value["runtime_files"])))
        self.assertEqual(value["kv_patch_sha256"], source.sha(delta))
        self.assertEqual(value["added_files"], self.baseline["added_files"])
        self.assertIn(b"+++ b/srt/layers/attention/ultraquant_stack_contract.py", delta)
        self.assertFalse(value["serving_composition_complete"])
        self.assertIsNone(value["image_git_revision"])

    def test_inventory_exact_missing_extra_changed_and_native(self):
        targets = [(self.root / "image-source/python/sglang/EXTRA.py", b"# INERT extra\n", "effective source"),
                   (self.root / "runtime/uq.h", b"# INERT changed\n", "native source")]
        for path, data, message in targets:
            original = path.read_bytes() if path.exists() else None
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, message): self.inventory()
            if original is None: path.unlink()
            else: path.write_bytes(original)
        path = self.root / "image-source/python/sglang/srt/layers/attention/ultraquant_stack_contract.py"
        original = path.read_bytes(); path.unlink()
        with self.assertRaises(ValueError): self.inventory()
        path.write_bytes(original)
        path = next(self.common.rglob("*.py")); original = path.read_bytes()
        path.write_bytes(original+b"\n# stale common bytes\n")
        with self.assertRaisesRegex(ValueError, "common FP8 baseline"): self.inventory()
        path.write_bytes(original)

    def test_baseline_image_schema_and_added_accounting_refuse(self):
        original = copy.deepcopy(self.baseline)
        for key, value, message in (("schema", 1, "wrong modern"), ("image_config", "0"*64, "wrong modern"),
                                     ("added_files", [], "counterpart accounting")):
            self.baseline = copy.deepcopy(original); self.baseline[key] = value; self.write_baseline()
            with self.assertRaisesRegex(ValueError, message): self.inventory()

    def test_installed_interpreter_library_and_roles_binding(self):
        accepted = source.verify_installed_stack(self.root, self.path, self.receipt_hash, self.tools)
        self.assertEqual(accepted["target"], self.contract.TARGET)
        with self.assertRaisesRegex(ValueError, "roles are incomplete"):
            source.verify_installed_stack(self.root, self.path, self.receipt_hash, {})
        for path, message in ((self.library, "compiler library changed"), (self.interpreter, "interpreter changed")):
            original = path.read_bytes(); path.write_bytes(original+b"stale")
            with self.assertRaisesRegex(ValueError, message):
                source.verify_installed_stack(self.root, self.path, self.receipt_hash, self.tools)
            path.write_bytes(original)
        changed = copy.deepcopy(self.tools); changed["clang"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "tool differs"):
            source.verify_installed_stack(self.root, self.path, self.receipt_hash, changed)

    def test_receipt_positive_and_independent_mutations(self):
        self.assertEqual(self.contract.read_receipt(self.path, self.receipt_hash), self.receipt)
        original = copy.deepcopy(self.receipt)
        mutations = [("target", {}, "incompatible"), ("image_manifest", "bad", "malformed stack digest"),
                     ("static_closure_complete", False, "unresolved static"),
                     ("dynamic_plugins_reviewed", False, "unresolved static"),
                     ("installed_protocol_reviewed", False, "syntax remains unreviewed"),
                     ("tools", {}, "complete tool role"), ("components", None, "malformed shared")]
        for key, value, message in mutations:
            self.receipt = copy.deepcopy(original); self.receipt[key] = value; self.write_receipt()
            with self.assertRaisesRegex(ValueError, message): self.contract.read_receipt(self.path, self.receipt_hash)

    def test_receipt_duplicate_paths_and_missing_argument_refuse(self):
        for path in ("../escape", "/absolute", "a//b", "a/./b", "a\\b", "a\0b"):
            with self.assertRaisesRegex(ValueError, "unsafe stack"):
                self.contract.checked_paths({path: "a"*64})
        for path in (None, "/a/../b", "/a//b", "relative"):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                self.contract.absolute_path(path)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.contract.read_receipt(None, "a"*64)
        self.path.write_bytes(b'{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, "duplicate stack"):
            self.contract.read_receipt(self.path, source.digest(self.path))

    def test_actual_runtime_gate_modern_positive_and_partial_refusal(self):
        fixture = SyntheticBuild(self.folder / "runtime-INERT")
        fixture.value.update(image_manifest="a"*64, image_config="b"*64, repair_files={},
            stack_receipt_sha256=self.receipt_hash, source_revision=self.contract.SOURCE_REVISION,
            serving_composition_complete=True)
        fixture.write()
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_stack_contract": self.contract}):
            values = []
            for rank in (0, 1):
                values.append(evidence.bf16_build_identity(fixture.package, uq.bf16_pool_policy(rank=rank, physical_slots=31),
                    stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash))
            self.assertNotEqual(values[0]["sha256"], values[1]["sha256"])
            with self.assertRaises(TypeError): values[0]["sha256"] = "0"*64
            fixture.value["serving_composition_complete"] = False; fixture.write()
            with self.assertRaisesRegex(ValueError, "complete installed serving"):
                evidence.bf16_build_identity(fixture.package, {}, stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)
            fixture.value["serving_composition_complete"] = True
            fixture.value["stack_receipt_sha256"] = "0"*64; fixture.write()
            with self.assertRaisesRegex(ValueError, "stale modern"):
                evidence.bf16_build_identity(fixture.package, {}, stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)

    def test_modern_producer_fake_tools_and_independent_review(self):
        import test_bf16_build_driver as fixture_module
        fake = fixture_module.BuildDriverTests()
        fake.setUp()
        common = fake.folder / "modern-common"
        shutil.copytree(fake.root / "image-source/python/sglang", common)
        self.receipt["common_files"] = source.tree(common)
        self.receipt["stock_files"] = source.tree(common)
        self.receipt["repair_files"] = dict(source.REPAIRS)
        self.receipt["protocol"] = source.PROTOCOL
        self.receipt["compiler_environment"]["HIP_CLANG_PATH"] = str(Path(fake.tools["clang"]).parent)
        self.receipt["compiler_environment"]["ROCM_PATH"] = str(fake.folder / "INERT-10.1-sdk")
        self.receipt["tools"] = {role: {"path": path, "sha256": source.digest(path),
            "interpreter": str(Path(sys.executable).resolve()), "interpreter_sha256": source.digest(Path(sys.executable).resolve()),
            "filesystem_root": str(self.folder), "libraries": {"lib/INERT-library": source.digest(self.library)}}
            for role, path in fake.tools.items()}
        self.write_receipt()
        baseline = dict(self.baseline, stack_receipt_sha256=self.receipt_hash,
            stock_files=source.tree(common), effective_files=source.tree(common), added_files=[])
        baseline_path = fake.folder / "INERT-modern-baseline.json"
        baseline_path.write_bytes(source.canonical(baseline))
        facts = fixture_module.driver.build(fake.root, common, baseline_path, source.digest(baseline_path),
            fake.attempt, fake.tools, time.time()+fixture_module.driver.DRIVER_SECONDS,
            stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)
        self.assertEqual(facts["status"], "BUILD_INSPECTED_UNREVIEWED")
        self.assertEqual(facts["source_identity"]["stack_receipt_sha256"], self.receipt_hash)
        self.assertEqual(facts["qualification"], "unreviewed")
        self.assertNotIn("review_receipt_sha256", facts)
        review_path = fake.folder / "INERT-independent-review.json"
        review_path.write_bytes(source.canonical({"schema": 1, "decision": "accept-build-only",
            "facts_sha256": source.digest(fake.attempt/"BUILD-FACTS.json"), "scope": source.SCOPE,
            "reviewer": "INERT CPU reviewer; NOT authority"}))
        proof = source.reviewed_proof(fake.attempt/"BUILD-FACTS.json", review_path, source.digest(review_path),
            fake.root, common, baseline_path, source.digest(baseline_path),
            stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)
        self.assertFalse(proof["serving_composition_complete"])
        proof["fixture_label"] = "INERT fake-tool proof, never serving evidence"
        package = fake.root / "image-source/python/sglang"
        proof_path = package / "srt/layers/attention/ultraquant-bf16-build.json"
        proof_path.write_bytes(source.canonical(proof))
        shutil.copyfile(fake.attempt/"libuq_bf16_v2.so", proof_path.with_name("libuq_bf16_v2.so"))
        with patch.dict(sys.modules, {"sglang.srt.layers.attention.ultraquant_stack_contract": self.contract}):
            with self.assertRaisesRegex(ValueError, "complete installed serving"):
                evidence.bf16_build_identity(package, {}, stack_receipt=self.path, stack_receipt_sha256=self.receipt_hash)

    def test_actual_v0521_gated_radix_hybrid_and_parent_writer(self):
        import test_strided_caller as caller
        import test_bf16_integration as integration
        case = caller.StridedCallerTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        original_method = caller.actual_method
        def actual_method(path, name, namespace):
            namespace["_use_aiter"] = False
            return original_method(path, name, namespace)
        modern = OWNED / "tests/fixtures/stock-v0521/python/sglang/srt"
        with patch.object(caller, "STOCK", modern), patch.object(integration, "STOCK", modern), patch.object(caller, "actual_method", actual_method):
            for rank in (0, 1):
                for scenario in ("initial", "continuation", "decode", "single"):
                    for physical in (3, 11, 59, 63):
                        pool, adapter, batch, dispatch = case.setup_batch(rank, scenario)
                        seen = case.caller(batch, dispatch, physical, rank)
                        seen["layer"].use_prefill_attention_wrapper = True
                        self.assertIsNone(dispatch.kv_index_translator)
                        self.assertEqual(adapter.data_type, "bfloat16")
                        before = case.native_count("uq_bf16_append_v2")
                        output = seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)
                        self.assertEqual(output.shape, (batch.out_cache_loc.numel(), 3072))
                        self.assertEqual(case.native_count("uq_bf16_append_v2"), before+1)
                        case.assert_order(seen, batch.out_cache_loc.numel(), physical in (3, 7, 59, 63))
                        with self.assertRaisesRegex(ValueError, "already appended"):
                            seen["radix"](seen["layer"], seen["q"], seen["k"], seen["v"], batch)


def inert_elf(needed=(), runpath=None, rpath=None, terminated=True):
    """Independent ELF fixture: one load segment and one dynamic segment."""
    strings = bytearray(b"\0")
    entries = []
    for tag, text in [(1, name) for name in needed] + [(tag, value) for tag, value in ((29, runpath), (15, rpath)) if value is not None]:
        entries.append((tag, len(strings))); strings.extend(text.encode()+b"\0")
    dynamic_size = (len(entries)+2+int(terminated))*16
    string_offset = 176 + dynamic_size
    entries += [(5, string_offset), (10, len(strings))]
    if terminated: entries.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *entry) for entry in entries)
    total = string_offset+len(strings)
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01"+b"\0"*10, 3, 62, 1, 0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 4, 0, 0, 0, total, total, 1)
    dyn = struct.pack("<IIQQQQQQ", 2, 4, 176, 176, 176, dynamic_size, dynamic_size, 8)
    return header+load+dyn+dynamic+strings


class StaticClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=OWNED / ".test-build/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_recursive_exact_inert_dependency_bytes_and_no_execution(self):
        tool = self.folder / "INERT-tool"; library = self.folder / "libINERT.so"
        tool.write_bytes(inert_elf([library.name], runpath="$ORIGIN")); library.write_bytes(inert_elf())
        value = collector.library_closure(tool, [])
        self.assertEqual(set(value["files"]), {str(tool), str(library)})
        self.assertEqual(value["files"][str(library)]["sha256"], source.digest(library))
        self.assertTrue(value["static_profile_complete"])
        self.assertFalse(value["dynamic_plugins_reviewed"])
        self.assertFalse(value["loader_cache_semantics_reviewed"])

    def test_missing_ambiguous_lookup_and_rpath_review_gaps(self):
        tool = self.folder / "INERT-tool"
        tool.write_bytes(inert_elf(["libINERT.so"]))
        self.assertFalse(collector.library_closure(tool, [str(self.folder)])["static_profile_complete"])
        folders = [self.folder / name for name in ("a", "b")]
        for directory in folders:
            directory.mkdir(); (directory / "libINERT.so").write_bytes(inert_elf())
        value = collector.library_closure(tool, [str(p) for p in folders])
        self.assertEqual(len(value["gaps"][0]["candidates"]), 2)
        tool.write_bytes(inert_elf(["libINERT.so"], rpath=str(folders[0])))
        self.assertIn("RPATH inheritance", collector.library_closure(tool, [])["gaps"][0]["reason"])
        with self.assertRaisesRegex(ValueError, "nonabsolute"):
            collector.library_closure(tool, ["relative"])

    def test_unterminated_dynamic_and_bad_encoding_refuse(self):
        path = self.folder / "INERT-tool"
        path.write_bytes(inert_elf(terminated=False))
        with self.assertRaisesRegex(ValueError, "unterminated"): collector.elf_metadata(path)
        path.write_bytes(inert_elf().replace(b"\x7fELF\x02\x01", b"\x7fELF\x02\x02", 1))
        with self.assertRaisesRegex(ValueError, "unsupported ELF"): collector.elf_metadata(path)
