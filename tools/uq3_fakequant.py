"""UQ3 Phase 0a fake-quant harness: score KV formats on attention, offline (CPU torch).

Every format is quantize-then-dequantize (the values a finished kernel would read), applied to
K and V of the 16 full-attention layers. Metrics per layer and KV head, against the BF16/FP32 reference:
  k_sqnr / v_sqnr     dB, signal over quantization noise
  logit_sigma         std of the QK logit error in nats (q.k / sqrt(d))
  attn_kl             mean KL(reference softmax || quantized softmax) per query row, nats
  out_sqnr            dB, attention output (softmax(QK) V) with quantized K and V

Inputs:
  --capture FILE.pt   real tensors: {layer: {"q": [Tq, Hq, 256], "k": [N, Hkv, 256], "v": [N, Hkv, 256]}}
                      captured after QK-norm and RoPE (what the KV cache stores)
  --proxy norms.json  no capture: K and Q are built from the checkpoint's real per-layer q_norm/k_norm
                      gains (GemmaRMSNorm, x_hat * (1 + w)) and real partial RoPE (64 of 256 dims,
                      theta 1e7); token content is synthetic (Student-t). V is synthetic (heavy-tailed
                      with log-normal channel scales). Proxy results rank formats; they are not model quality.
"""
import argparse
import json
import math
from pathlib import Path

import torch

HEAD_DIM, ROTARY, THETA = 256, 64, 1.0e7
FP8 = torch.float8_e4m3fn
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


H = hadamard(HEAD_DIM)
SIGNS = torch.where(torch.rand(HEAD_DIM, generator=torch.Generator().manual_seed(1729)) < 0.5, -1.0, 1.0)


def rotate(x, kind):
    if kind == "none":
        return x
    return (x * SIGNS if kind == "rht" else x) @ H


def unrotate(x, kind):
    if kind == "none":
        return x
    y = x @ H.T
    return y * SIGNS if kind == "rht" else y


def e4m3_round(x):
    return x.to(FP8).to(torch.float32)


def int4_groups(x, group, scale_fp8=True):
    """Signed 4-bit (-7..7) with one scale per `group` channels; scale stored as e4m3."""
    shape = x.shape
    g = x.reshape(*shape[:-1], shape[-1] // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 7.0
    if scale_fp8:
        scale = e4m3_round(scale).clamp_min(2.0 ** -9)
    q = torch.clamp(torch.round(g / scale), -7, 7)
    return (q * scale).reshape(shape)


def mxfp4_groups(x, group=32):
    """OCP MX FP4: E2M1 elements, one power-of-two (E8M0) scale per 32 values."""
    shape = x.shape
    g = x.reshape(*shape[:-1], shape[-1] // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2)
    a = (g.abs() / scale).clamp(max=6.0)
    idx = torch.argmin((a.unsqueeze(-1) - E2M1).abs(), -1)
    return (torch.sign(g) * E2M1[idx] * scale).reshape(shape)


def make_format(kind, rotation="none", group=32):
    def fmt(x):
        y = rotate(x, rotation)
        if kind == "fp8":
            y = e4m3_round(y)
        elif kind == "int4":
            y = int4_groups(y, group)
        elif kind == "mxfp4":
            y = mxfp4_groups(y, group)
        elif kind == "bf16":
            y = y.to(torch.bfloat16).to(torch.float32)
        return unrotate(y, rotation)
    return fmt


# name -> (K format, V format). UQ2 is the FP8 e4m3 cache serving today; Y is BF16.
SCHEMES = {
    "Y (BF16)": (make_format("bf16"), make_format("bf16")),
    "UQ2 (FP8)": (make_format("fp8"), make_format("fp8")),
    "I4G32 no rotation": (make_format("int4"), make_format("int4")),
    "I4G32 Hadamard": (make_format("int4", "hadamard"), make_format("int4", "hadamard")),
    "UQ3 = I4G32-RHT": (make_format("int4", "rht"), make_format("int4", "rht")),
    "UQ3, V unrotated": (make_format("int4", "rht"), make_format("int4")),
    "I4G16-RHT": (make_format("int4", "rht", 16), make_format("int4", "rht", 16)),
    "I4 per-vector-RHT": (make_format("int4", "rht", HEAD_DIM), make_format("int4", "rht", HEAD_DIM)),
    "MXFP4-RHT": (make_format("mxfp4", "rht"), make_format("mxfp4", "rht")),
    "K8V4 (K FP8, V I4G32-RHT)": (make_format("fp8"), make_format("int4", "rht")),
}


def rope(x, positions):
    """Partial RoPE on the first 64 dims (rotate-half), text positions (all MRoPE sections equal)."""
    half = ROTARY // 2
    freqs = THETA ** (-torch.arange(half, dtype=torch.float64) / half)
    ang = positions[:, None].double() * freqs[None]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    r, rest = x[..., :ROTARY], x[..., ROTARY:]
    a, b = r[..., :half], r[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos, rest], -1)


def rms_gemma(x, weight, eps=1e-6):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * (1.0 + weight)


def proxy_layer(norms, layer, tokens, queries, kv_heads, q_heads, generator):
    t = lambda *shape: torch.distributions.StudentT(5.0).sample(shape)  # noqa: E731
    torch.manual_seed(int(layer) * 7919 + 17)
    pos = torch.arange(tokens)
    k = rope(rms_gemma(t(tokens, kv_heads, HEAD_DIM), torch.tensor(norms["k_norm"])), pos)
    qpos = torch.arange(tokens - queries, tokens)
    q = rope(rms_gemma(t(queries, q_heads, HEAD_DIM), torch.tensor(norms["q_norm"])), qpos)
    channel = torch.exp(0.75 * torch.randn(kv_heads, HEAD_DIM))
    v = torch.distributions.StudentT(4.0).sample((tokens, kv_heads, HEAD_DIM)) * channel
    return q, k, v


def sqnr(ref, test):
    noise = (ref - test).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / noise.clamp_min(1e-30)))


def score(q, k, v, k_fmt, v_fmt):
    kq, vq = k_fmt(k), v_fmt(v)
    group = q.shape[1] // k.shape[1]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    sigmas, kls, outs = [], [], []
    for h in range(k.shape[1]):
        qh = q[:, h * group:(h + 1) * group].reshape(-1, HEAD_DIM)
        logits = qh @ k[:, h].T * scale
        logits_q = qh @ kq[:, h].T * scale
        sigmas.append(float((logits_q - logits).std()))
        p, lp = torch.softmax(logits, -1), torch.log_softmax(logits, -1)
        lq = torch.log_softmax(logits_q, -1)
        kls.append(float((p * (lp - lq)).sum(-1).mean()))
        outs.append(sqnr(p @ v[:, h], torch.softmax(logits_q, -1) @ vq[:, h]))
    return {"k_sqnr": sqnr(k, kq), "v_sqnr": sqnr(v, vq), "logit_sigma": sum(sigmas) / len(sigmas),
            "attn_kl": sum(kls) / len(kls), "out_sqnr": sum(outs) / len(outs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--proxy", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    if args.capture:
        data = torch.load(args.capture, map_location="cpu", weights_only=True)
        layers = {str(k): {n: t.float() for n, t in v.items()} for k, v in data.items()}
        kind = "capture"
    else:
        norms = json.loads(args.proxy.read_text())["layers"]
        layers = {}
        for layer, row in norms.items():
            q, k, v = proxy_layer(row, layer, args.tokens, args.queries, 4 // 2, 24 // 2, None)
            layers[layer] = {"q": q, "k": k, "v": v}
        kind = "proxy (real q_norm/k_norm gains and RoPE; synthetic token content and V)"
    keys = next(iter(layers.values()))["k"].shape[0]
    results = {"source": kind, "tokens": keys, "schemes": {}}
    for name, (k_fmt, v_fmt) in SCHEMES.items():
        per_layer = {layer: score(t["q"], t["k"], t["v"], k_fmt, v_fmt) for layer, t in layers.items()}
        metrics = list(next(iter(per_layer.values())))
        mean = {m: sum(r[m] for r in per_layer.values()) / len(per_layer) for m in metrics}
        worst_layer = min(per_layer, key=lambda layer: per_layer[layer]["out_sqnr"])
        results["schemes"][name] = {"mean": mean, "worst_layer": worst_layer,
                                    "worst_out_sqnr": per_layer[worst_layer]["out_sqnr"], "per_layer": per_layer}
    args.out.write_text(json.dumps(results, indent=1))
    print(f"source: {kind}; {len(layers)} layers; {keys} keys; KV heads per layer: {next(iter(layers.values()))['k'].shape[1]}")
    print(f"{'scheme':28s} {'K dB':>6s} {'V dB':>6s} {'logit σ':>8s} {'attn KL':>9s} {'out dB':>7s} {'worst layer (out dB)':>21s}")
    for name, r in results["schemes"].items():
        m = r["mean"]
        print(f"{name:28s} {m['k_sqnr']:6.1f} {m['v_sqnr']:6.1f} {m['logit_sigma']:8.4f} {m['attn_kl']:9.2e} "
              f"{m['out_sqnr']:7.1f} {r['worst_layer']:>13s} ({r['worst_out_sqnr']:.1f})")


if __name__ == "__main__":
    main()
