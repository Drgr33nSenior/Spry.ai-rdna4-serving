"""Score K/V INT4 scale-group mixes on a real Q/K/V capture (UQ3, 2026-11-12).

Source of the "K G64 / V per-token" and per-token rows in docs/UQ3-INT4-VERIFY-KERNEL-20261112.md.
Usage: python3 -I tools/uq3_group_mixes.py CAPTURE.pt   (CPU; same scoring as uq3_fakequant.py --capture)
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import uq3_fakequant as u  # noqa: E402


def main(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    layers = [{n: t.float() for n, t in v.items()} for v in data.values()]
    mixes = {"UQ2 (FP8)": u.SCHEMES["UQ2 (FP8)"], "INT4 G32 K+V (UQ3)": u.SCHEMES["UQ3 = I4G32-RHT"]}
    for kg in (64, 128, 256):
        for vg in (64, 256):
            mixes[f"K G{kg} / V G{vg}"] = (u.make_format("int4", "rht", kg), u.make_format("int4", "rht", vg))
    mixes["K8 (FP8) / V G256"] = (u.make_format("fp8"), u.make_format("int4", "rht", 256))
    for name, (kf, vf) in mixes.items():
        per = [u.score(t["q"], t["k"], t["v"], kf, vf) for t in layers]
        out = [r["out_sqnr"] for r in per]
        print(f"{name:22s} out dB {sum(out) / len(out):5.1f} (min {min(out):4.1f})  "
              f"attn KL {sum(r['attn_kl'] for r in per) / len(per):.2e}  "
              f"V dB {sum(r['v_sqnr'] for r in per) / len(per):5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
