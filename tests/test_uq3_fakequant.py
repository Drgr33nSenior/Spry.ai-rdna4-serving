"""UQ3 Phase 0a harness: codec invariants and both input paths, CPU only, no network."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import uq3_fakequant as u  # noqa: E402


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(64, 2, 256) * 3

    def test_int4_codes_and_bf16_exactness(self):
        g = self.x.reshape(64, 2, 8, 32)
        scale = u.e4m3_round(g.abs().amax(-1, keepdim=True) / 7).clamp_min(2 ** -9)
        deq = u.int4_groups(self.x, 32)
        codes = torch.round(deq.reshape(64, 2, 8, 32) / scale)
        self.assertLessEqual(float(codes.abs().max()), 7.0)
        # The fake-quant premise: every dequantized value is exact in BF16.
        self.assertTrue(bool((deq.bfloat16().float() == deq).all()))

    def test_zero_vectors_and_rotation(self):
        zero = torch.zeros(4, 2, 256)
        for name, (k_fmt, v_fmt) in u.SCHEMES.items():
            self.assertTrue(bool((k_fmt(zero) == 0).all() and (v_fmt(zero) == 0).all()), name)
        for kind in ("hadamard", "rht"):
            self.assertTrue(torch.allclose(u.unrotate(u.rotate(self.x, kind), kind), self.x, atol=1e-4))

    def test_mxfp4_values_are_on_the_e2m1_grid(self):
        g = self.x.reshape(64, 2, 8, 32)
        scale = torch.pow(2.0, torch.floor(torch.log2(g.abs().amax(-1, keepdim=True))) - 2)
        ratio = (u.mxfp4_groups(self.x).reshape(64, 2, 8, 32) / scale).abs()
        self.assertTrue(bool(torch.isin(ratio, u.E2M1).all()))


class HarnessTests(unittest.TestCase):
    def test_proxy_and_capture_paths_rank_formats(self):
        norms = {"layers": {str(layer): {"q_norm": [0.1] * 256, "k_norm": [(i % 7) * 0.3 - 0.5 for i in range(256)]}
                            for layer in (3, 63)}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "norms.json").write_text(json.dumps(norms))
            u.main(["--proxy", str(root / "norms.json"), "--tokens", "256", "--queries", "4",
                    "--out", str(root / "proxy.json")])
            proxy = json.loads((root / "proxy.json").read_text())["schemes"]
            means = {name: r["mean"]["out_sqnr"] for name, r in proxy.items()}
            self.assertGreater(means["Y (BF16)"], means["UQ2 (FP8)"])
            self.assertGreater(means["UQ2 (FP8)"], means["UQ3 = I4G32-RHT"])
            self.assertGreater(means["UQ3 = I4G32-RHT"], means["I4G32 no rotation"])
            q, k, v = u.proxy_layer(norms["layers"]["3"], "3", 128, 4, 2, 12, None)
            torch.save({3: {"q": q.bfloat16(), "k": k.bfloat16(), "v": v.bfloat16()}}, root / "capture.pt")
            u.main(["--capture", str(root / "capture.pt"), "--out", str(root / "capture.json")])
            capture = json.loads((root / "capture.json").read_text())
            self.assertEqual((128, ["3"]), (capture["tokens"], list(capture["schemes"]["Y (BF16)"]["per_layer"])))


if __name__ == "__main__":
    unittest.main()
