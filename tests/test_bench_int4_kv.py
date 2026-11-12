"""INT4 KV benchmark: the host-side codec the GPU kernel reads (CPU only, no SGLang)."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_int4_kv as b  # noqa: E402


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.x = (torch.randn(64, b.HKV, b.D) * 2).to(torch.bfloat16)

    def test_codes_scales_and_dequantized_values_agree_exactly(self):
        codes, scales, deq = b.quantize_i4(self.x)
        self.assertEqual((64, b.HKV, b.D // 2), tuple(codes.shape))
        self.assertEqual(torch.uint8, codes.dtype)
        self.assertEqual((64, b.HKV, b.D // b.G), tuple(scales.shape))
        self.assertEqual(b.FP8, scales.dtype)
        q = b.unpack_i4_reference(codes)
        self.assertGreaterEqual(int(q.min()), -7)
        self.assertLessEqual(int(q.max()), 7)
        rebuilt = (q.float().view(64, b.HKV, b.D // b.G, b.G) * scales.float()[..., None]).view(64, b.HKV, b.D)
        self.assertTrue(torch.equal(rebuilt, deq))
        self.assertTrue(torch.equal(deq.to(torch.bfloat16).float(), deq))  # exact in bf16

    def test_value_2i_is_the_low_nibble(self):
        x = torch.zeros(1, b.HKV, b.D)
        x[0, 0, :32] = torch.tensor([7.0, -7.0] * 16)  # one group, scale 1 after e4m3 rounding
        codes, scales, _ = b.quantize_i4(x.to(torch.bfloat16))
        self.assertEqual(1.0, float(scales[0, 0, 0].float()))
        self.assertEqual((7 + 8) | ((-7 + 8) << 4), int(codes[0, 0, 0]))
        self.assertEqual([7, -7, 7, -7], b.unpack_i4_reference(codes)[0, 0, :4].tolist())

    def test_quantization_error_is_bounded_by_half_a_step(self):
        _, scales, deq = b.quantize_i4(self.x)
        step = scales.float().repeat_interleave(b.G, dim=-1)
        self.assertTrue(bool(((deq - self.x.float()).abs() <= step / 2 + 1e-6).all()))


class HalfLayoutTests(unittest.TestCase):
    def test_split_half_codes_scales_and_values_agree_for_both_group_sizes(self):
        torch.manual_seed(5)
        x = (torch.randn(32, b.HKV, b.D) * 2).to(torch.bfloat16)
        for gph in (1, 4):
            codes, scales, deq = b.quantize_i4_half(x, gph)
            self.assertEqual((32, b.HKV, b.D // 2), tuple(codes.shape))
            self.assertEqual((32, b.HKV, 2 * gph), tuple(scales.shape))
            q = b.unpack_half_reference(codes).float()
            width = (b.D // 2) // gph
            rebuilt = (q.view(32, b.HKV, 2 * gph, width) * scales.float()[..., None]).view(32, b.HKV, b.D)
            self.assertTrue(torch.equal(rebuilt, deq), gph)
            self.assertTrue(torch.equal(deq.to(torch.bfloat16).float(), deq), gph)

    def test_byte_i_holds_dim_i_low_and_dim_half_plus_i_high(self):
        x = torch.zeros(1, b.HKV, b.D)
        x[0, 0, 0], x[0, 0, b.D // 2] = 7.0, -7.0
        codes, scales, _ = b.quantize_i4_half(x.to(torch.bfloat16), 1)
        self.assertEqual((7 + 8) | ((-7 + 8) << 4), int(codes[0, 0, 0]))
        self.assertEqual([1.0, 1.0], scales[0, 0].float().tolist())


class MagicUnpackTests(unittest.TestCase):
    def test_bf16_bits_0x43_then_a_nibble_are_128_plus_the_nibble(self):
        n = torch.arange(16, dtype=torch.int32)
        self.assertEqual((128 + n).float().tolist(), (0x4300 | n).to(torch.int16).view(torch.bfloat16).float().tolist())

    def test_per_token_and_per_64_scales_rebuild_the_dequantized_values(self):
        torch.manual_seed(7)
        x = (torch.randn(16, b.HKV, b.D) * 3).to(torch.bfloat16)
        for group in (64, b.D):
            words, scales, deq = b.magic_cache(x, group)
            self.assertEqual((16, b.HKV, b.D // 4), tuple(words.shape))
            self.assertEqual((16, b.HKV, b.D // group), tuple(scales.shape))
            q = b.unpack_i4_reference(words.view(torch.uint8)).float()
            rebuilt = (q.view(16, b.HKV, b.D // group, group) * scales.float()[..., None]).view(16, b.HKV, b.D)
            self.assertTrue(torch.equal(rebuilt, deq), group)


if __name__ == "__main__":
    unittest.main()
