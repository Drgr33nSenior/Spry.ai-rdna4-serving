"""Separately authorized synthetic GPU test of actual SGLang writer/reader.

This is not a model benchmark. No GPU library is imported until --execute-gpu
is supplied explicitly. An inherited opt-in variable never activates this test.
"""
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/sglang-0b3bb0cbe31873994c9f989fddfe2f87ca839fdd/python"))


def rotate(vector):
    n = len(vector)
    # Dense parity-matrix oracle, independent of the production butterfly.
    return [sum(value * (-1 if (row & col).bit_count() % 2 else 1)
                for col, value in enumerate(vector)) / math.sqrt(n) for row in range(n)]

def encode(vector):
    levels = (0, .5, 1, 1.5, 2, 3, 4, 6)
    result = []
    for base in range(0, len(vector), 32):
        group = vector[base:base + 32]
        maximum = max(map(abs, group))
        exponent = max(-127, min(127, round(math.log2(.156 * maximum)))) if maximum else 0
        codes = []
        for value in group:
            index = min(range(8), key=lambda i: (abs(abs(value) / 2**exponent - levels[i]), i % 2, i))
            codes.append(index | (8 if value < 0 and index else 0))
        result.extend(codes[i] | codes[i + 1] << 4 for i in range(0, 32, 2))
        result.append(exponent + 127)
    return result

def decode(record, dim):
    levels = (0, .5, 1, 1.5, 2, 3, 4, 6)
    values = []
    for d in range(dim):
        base, c = (d // 32) * 17, d % 32
        nibble = (record[base + c // 2] >> (4 * (c % 2))) & 15
        values.append((-1 if nibble & 8 else 1) * levels[nibble & 7] * 2**(record[base + 16] - 127))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true", help="requires separate owner authorization")
    args = parser.parse_args()
    if not args.execute_gpu:
        parser.error("GPU execution NOT RUN: explicit --execute-gpu and owner authorization required")
    import torch
    from sglang.srt.layers.attention.ultraquant_backend import UltraQuantBackend, UltraQuantPool
    from sglang.srt.layers.radix_attention import AttentionType
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from test_runtime_contract import valid_args

    # Every supported dimension, GQA including MHA/MQA, and a partial 16-key tile.
    for dim in (32, 64, 128, 256):
        for kv_heads, q_heads in ((1, 1), (1, 4), (2, 4)):
            pool = UltraQuantPool(64, 1, torch.float16, kv_heads, dim, 1, "cuda:0", False)
            mapping = torch.zeros((2, 32), dtype=torch.int64, device="cuda:0")
            model = SimpleNamespace(hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                                    head_dim=dim, v_head_dim=dim, num_attention_heads=q_heads,
                                    get_num_kv_heads=lambda tp: kv_heads)
            server_args = valid_args()
            server_args.cuda_graph_config = SimpleNamespace(
                decode=SimpleNamespace(backend="disabled"), prefill=SimpleNamespace(backend="disabled"))
            runner = SimpleNamespace(server_args=server_args, model_config=model, dtype=torch.float16,
                                     token_to_kv_pool=pool, req_to_token_pool=SimpleNamespace(req_to_token=mapping),
                                     mambaish_config=None, use_mla_backend=False, is_hybrid_swa=False, dcp_size=1)
            backend = UltraQuantBackend(runner)
            layer = SimpleNamespace(layer_id=0, attn_type=AttentionType.DECODER, is_cross_attention=False,
                                    use_irope=False, sliding_window_size=-1, logit_cap=0, pos_encoding_mode="NONE",
                                    xai_temperature_len=-1, tp_q_head_num=q_heads, tp_k_head_num=kv_heads,
                                    tp_v_head_num=kv_heads, qk_head_dim=dim, v_head_dim=dim,
                                    k_scale=None, v_scale=None, scaling=1 / math.sqrt(dim))
            expected_k, expected_v = {}, {}

            def run(sequence_rows, prefixes, locations, mode):
                extends = [len(seq) - prefix for seq, prefix in zip(sequence_rows, prefixes)]
                rows = sum(extends)
                for r, sequence in enumerate(sequence_rows):
                    mapping[r, :len(sequence)] = torch.tensor(sequence, dtype=torch.int64, device="cuda:0")
                # Exactly representable inputs avoid an accidental CPU/GPU
                # transcendental threshold difference in the byte comparison.
                q = ((torch.arange(rows * q_heads * dim, device="cuda:0") % 13 - 6) / 32).half().reshape(rows, q_heads, dim)
                k = ((torch.arange(rows * kv_heads * dim, device="cuda:0") % 11 - 5) / 16).half().reshape(rows, kv_heads, dim)
                v = ((torch.arange(rows * kv_heads * dim, device="cuda:0") % 7 - 3) / 8).half().reshape(rows, kv_heads, dim)
                starts, cursor = [], 0
                for n in extends:
                    starts.append(cursor)
                    cursor += n
                batch = SimpleNamespace(forward_mode=mode, spec_info=None,
                    req_pool_indices=torch.arange(len(sequence_rows), device="cuda:0"),
                    seq_lens=torch.tensor(list(map(len, sequence_rows)), device="cuda:0"),
                    extend_prefix_lens=torch.tensor(prefixes, device="cuda:0"),
                    extend_seq_lens=torch.tensor(extends, device="cuda:0"),
                    extend_start_loc=torch.tensor(starts, device="cuda:0"),
                    out_cache_loc=torch.tensor(locations, dtype=torch.int64, device="cuda:0"),
                    input_ids=torch.zeros(rows, dtype=torch.int64, device="cuda:0"))
                backend.init_forward_metadata(batch)
                output = backend.forward(q, k, v, layer, batch).reshape(rows, q_heads, dim).cpu().double()
                for slot, kr, vr in zip(locations, k.cpu().tolist(), v.cpu().tolist()):
                    expected_k[slot] = [encode(rotate(head)) for head in kr]
                    expected_v[slot] = [encode(head) for head in vr]
                    assert pool.get_key_buffer(0)[slot].cpu().tolist() == expected_k[slot], "GPU packed K != independent oracle"
                    assert pool.get_value_buffer(0)[slot].cpu().tolist() == expected_v[slot], "GPU packed V != independent oracle"
                query = q.cpu().tolist()
                row = 0
                for sequence, prefix, extend in zip(sequence_rows, prefixes, extends):
                    for step in range(extend):
                        visible = sequence[:prefix + step + 1]
                        for h in range(q_heads):
                            kh = h // (q_heads // kv_heads)
                            qr = torch.tensor(rotate(query[row][h]), dtype=torch.float16).double().tolist()
                            keys = [decode(expected_k[slot][kh], dim) for slot in visible]
                            values = [decode(expected_v[slot][kh], dim) for slot in visible]
                            scores = [sum(a * b for a, b in zip(qr, key)) * layer.scaling for key in keys]
                            maximum = max(scores)
                            weights = [math.exp(score - maximum) for score in scores]
                            total = sum(weights)
                            expected = torch.tensor([sum(w * value[d] for w, value in zip(weights, values)) / total
                                                     for d in range(dim)], dtype=torch.float64)
                            torch.testing.assert_close(output[row, h], expected, atol=8e-4, rtol=3e-3)
                        row += 1
                return output

            prefix = list(range(1, 18))
            run([prefix], [0], prefix, ForwardMode.EXTEND)
            run([prefix + [23]], [17], [23], ForwardMode.DECODE)
            run([prefix + [23, 31, 27]], [18], [31, 27], ForwardMode.EXTEND)
            # A new request shares a retained prefix. Old suffix slots can be
            # reused only after release by the runtime allocator; here 31 is
            # explicitly absent from every reused prefix.
            run([prefix + [23, 45], prefix + [31]], [18, 17], [45, 31], ForwardMode.EXTEND)
            assert pool.get_kv_size_bytes() == (65 * kv_heads * dim // 32 * 17,) * 2
            # A malformed block must fail, then poison this session.
            pool.get_key_buffer(0)[1, 0, 0] = 8
            try:
                run([prefix + [41]], [17], [41], ForwardMode.DECODE)
                raise AssertionError("corrupt cache was accepted")
            except RuntimeError:
                assert pool.native.failed
    print("GPU codec and actual SGLang pool/backend synthetic tests passed; model qualification NOT RUN")


if __name__ == "__main__":
    main()
