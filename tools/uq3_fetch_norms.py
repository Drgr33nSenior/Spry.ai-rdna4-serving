"""Range-fetch the q_norm/k_norm weights of the 16 full-attention layers from the pinned checkpoint.

Reads only the safetensors header and the norm tensors (a few KiB), never the 19.8 GB file.
Writes norms.json: {layer: {"q_norm": [...], "k_norm": [...]}, "dtype": ...}.
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4/resolve/"
       "5233554c5fa56afda40150556b95573c2d7d29c0/model.safetensors")
LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]


def fetch(start, end):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise SystemExit(f"short range read {start}-{end}: {len(data)} bytes")
    return data


def bf16_to_floats(raw):
    return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def main(out):
    (length,) = struct.unpack("<Q", fetch(0, 7))
    header = json.loads(fetch(8, 8 + length - 1))
    base = 8 + length
    result = {"source": URL, "layers": {}}
    for layer in LAYERS:
        row = {}
        for kind in ("q_norm", "k_norm"):
            names = [n for n in header if n.endswith(f"layers.{layer}.self_attn.{kind}.weight")]
            if len(names) != 1:
                raise SystemExit(f"layer {layer} {kind}: expected one tensor, found {names}")
            meta = header[names[0]]
            if meta["dtype"] not in ("BF16", "F32"):
                raise SystemExit(f"{names[0]}: unexpected dtype {meta['dtype']}")
            start, end = meta["data_offsets"]
            raw = fetch(base + start, base + end - 1)
            row[kind] = (bf16_to_floats(raw) if meta["dtype"] == "BF16"
                         else list(struct.unpack(f"<{len(raw) // 4}f", raw)))
            row[kind + "_tensor"] = names[0]
            row[kind + "_dtype"] = meta["dtype"]
        result["layers"][str(layer)] = row
    Path(out).write_text(json.dumps(result))
    first = result["layers"]["3"]
    print("layers:", len(result["layers"]), "| dims:", len(first["k_norm"]), "| example:", first["k_norm_tensor"],
          first["k_norm_dtype"])


if __name__ == "__main__":
    main(sys.argv[1])
