"""Synthetic CPU parser checks; no compiler, native library, GPU or process launch."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "prototype/scripts"
EXPECTED_KERNELS = ("validate_write", "write", "validate_read", "read", "read_grouped")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load("cache_native_inspection_source", SCRIPTS / "bf16_source_identity.py")
with patch.dict(sys.modules, {"bf16_source_identity": source}):
    driver = load("cache_native_inspection_driver", SCRIPTS / "build_bf16_v2.py")

SYMBOLS = "\n".join("0000000000000000 T " + name for name in sorted(source.EXPORTS))
METADATA = "Machine: AMDGPU\namdhsa.target: amdgcn-amd-amdhsa--gfx1201\n"
BODIES = {name: f"\n  s_mov_b32 s0, {index}\n  s_endpgm\n"
          for index, name in enumerate(EXPECTED_KERNELS)}


def disassembly(blocks=None):
    blocks = list(BODIES.items()) if blocks is None else blocks
    legacy = ("0000000000000000 <legacy_attention()>:\n"
              "  v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
              "  s_endpgm\n")
    return legacy + "".join(f"{(index + 1) * 256:016x} <bf16_v2::{name}()>:" + body
                            for index, (name, body) in enumerate(blocks))


class CacheNativeInspectionTests(unittest.TestCase):
    def assert_refused(self, text, message, *, symbols=SYMBOLS, metadata=METADATA):
        for inspect in (source.inspect_outputs, driver.inspect_outputs):
            with self.subTest(inspector=inspect.__module__), self.assertRaisesRegex(source.ProtocolError, message):
                inspect(symbols, metadata, text)

    def test_exact_five_kernel_identities_and_wrapper_alias(self):
        self.assertEqual(source.V2_KERNELS, EXPECTED_KERNELS)
        self.assertIs(driver.V2_KERNELS, source.V2_KERNELS)
        self.assertIs(driver.BuildError, source.ProtocolError)
        expected = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in BODIES.items()}
        result = source.inspect_outputs(SYMBOLS, METADATA, disassembly())
        self.assertEqual(result["v2_kernel_blocks"], expected)
        self.assertEqual(driver.inspect_outputs(SYMBOLS, METADATA, disassembly()), result)
        self.assertEqual(result["gpu_execution"], "not-run")
        self.assertFalse(result["v2_wmma_acceleration"])

    def test_each_missing_kernel_is_refused(self):
        for missing in EXPECTED_KERNELS:
            with self.subTest(missing=missing):
                blocks = [(name, body) for name, body in BODIES.items() if name != missing]
                self.assert_refused(disassembly(blocks), "missing v2 device kernel disassembly")

    def test_each_duplicate_kernel_is_refused(self):
        for duplicate in EXPECTED_KERNELS:
            with self.subTest(duplicate=duplicate):
                blocks = list(BODIES.items()) + [(duplicate, BODIES[duplicate])]
                self.assert_refused(disassembly(blocks), "ambiguous v2 device symbol")

    def test_grouped_name_requires_exact_identity(self):
        for near_name in ("read_grouped_suffix", "read", "read_group"):
            with self.subTest(name=near_name):
                blocks = [(near_name if name == "read_grouped" else name, body)
                          for name, body in BODIES.items()]
                self.assert_refused(disassembly(blocks), "missing v2 device kernel|ambiguous v2 device symbol")

    def test_grouped_requires_decoded_instructions(self):
        blocks = [(name, "\n" if name == "read_grouped" else body) for name, body in BODIES.items()]
        self.assert_refused(disassembly(blocks), "v2 kernel has no decoded instructions")

    def test_grouped_matrix_instructions_are_refused(self):
        for instruction in ("v_wmma_f32_16x16x16_f16", "v_mfma_f32_16x16x16f16", "v_mfma_scale_f32_16x16x128_f8f6f4"):
            with self.subTest(instruction=instruction):
                blocks = [(name, "\n  " + instruction + "\n  s_endpgm\n" if name == "read_grouped" else body)
                          for name, body in BODIES.items()]
                self.assert_refused(disassembly(blocks), "unexpected matrix instruction in scalar v2 path")

    def test_architecture_exports_and_legacy_opcode_guards_are_preserved(self):
        text = disassembly()
        self.assert_refused(text, "wrong device architecture metadata",
                            metadata=METADATA.replace("gfx1201", "gfx1100"))
        self.assert_refused(text, "wrong device architecture metadata",
                            metadata=METADATA.replace("AMDGPU", "X86"))
        self.assert_refused(text, "mixed device architecture metadata", metadata=METADATA + "gfx1100\n")
        symbols = "\n".join(line for line in SYMBOLS.splitlines() if not line.endswith(" uq_bf16_attention_v2"))
        self.assert_refused(text, "missing required host exports", symbols=symbols)
        self.assert_refused(text.replace("v_wmma_f32_16x16x16_f16", "v_add_f32", 1),
                            "legacy gfx1201 instruction inspection failed")
        self.assert_refused(text.replace("<legacy_attention()>:", "<legacy_attention()>:\n  v_mfma_scale", 1),
                            "legacy gfx1201 instruction inspection failed")

    def test_wrapper_delegates_inspection_and_preserves_425_second_bound(self):
        marker = object()
        with patch.object(source, "inspect_outputs", return_value=marker) as inspect:
            self.assertIs(driver.inspect_outputs(SYMBOLS, METADATA, disassembly()), marker)
            inspect.assert_called_once_with(SYMBOLS, METADATA, disassembly())
        self.assertEqual(driver.DRIVER_SECONDS, 425)
        self.assertEqual((driver.COMMAND_WORK_SECONDS, driver.COMMAND_CLEANUP_SECONDS,
                          driver.COLLECTION_SECONDS), (320, 75, 30))
        tools = {name: "/cpu-fixture-tools/" + name for name in source.TOOL_NAMES}
        plan = source.commands("/cpu-fixture-source", "/cpu-fixture-attempt", tools)
        self.assertEqual(driver.command_plan("/cpu-fixture-source", "/cpu-fixture-attempt", tools), plan)
        self.assertEqual([seconds for _, _, seconds in plan], [5] * 7 + [180] + [15] * 7)
        self.assertEqual(sum(seconds for _, _, seconds in plan) +
                         driver.COMMAND_CLEANUP_SECONDS + driver.COLLECTION_SECONDS, 425)


if __name__ == "__main__":
    unittest.main()
