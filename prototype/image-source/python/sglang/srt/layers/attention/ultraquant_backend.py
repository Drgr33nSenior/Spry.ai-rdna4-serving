"""Local experimental gfx1201 backend. Requires the separately built HIP library.

No JIT build, library-path option, decompressed shadow cache, fallback, or GPU
execution during import. This backend is not upstream or qualified functionality.
"""
import ctypes
from pathlib import Path

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.ultraquant_contract import (
    MODE, causal_rows, cell_bytes, validate_args, validate_model,
)
from sglang.srt.layers.attention.ultraquant_evidence import build_identity, descriptor, emit_receipt
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc


class Native:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.version.hip or torch.cuda.device_count() != 1:
            raise ValueError("UltraQuant requires exactly one visible ROCm gfx1201 GPU")
        if torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0] != "gfx1201":
            raise ValueError("UltraQuant requires gfx1201")
        # Fixed package-local build artifact; no user-provided executable path.
        path = Path(__file__).with_name("libuq_hip.so")
        if path.is_symlink() or not path.is_file():
            raise ValueError("the experimental HIP library has not been built")
        self.build = build_identity()
        self.lib = ctypes.CDLL(str(path))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.uq_device.argtypes = []
        self.lib.uq_initialize.argtypes = [ptr, ptr, integer, integer, integer, ptr]
        self.lib.uq_append.argtypes = [ptr, ptr, integer, integer, integer, ptr, ptr, ptr, integer, ptr, ptr]
        self.lib.uq_attention.argtypes = [ptr, ptr, ptr, integer, integer, integer, integer,
                                         ptr, integer, ptr, ptr, integer, ctypes.c_float, ptr, ptr, ptr]
        for name in ("uq_device", "uq_initialize", "uq_append", "uq_attention"):
            getattr(self.lib, name).restype = integer
        self.failed = False
        self.status = torch.empty(1, dtype=torch.int32, device=self.device)
        self.call("uq_device")

    def stream(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    def call(self, name, *args):
        if self.failed:
            raise RuntimeError("UltraQuant session failed; stop and restart without reusing its cache")
        with torch.cuda.device(self.device):
            status = getattr(self.lib, name)(*args)
        if status:
            self.failed = True
            raise RuntimeError(f"UltraQuant {name} refused or failed ({status}); restart required")

    def tensor(self, tensor, dtype, shape):
        if tensor.device != self.status.device or tensor.dtype != dtype or list(tensor.shape) != list(shape):
            raise ValueError("unsupported UltraQuant tensor device, dtype or shape")
        if not tensor.is_contiguous():
            raise ValueError("UltraQuant native tensors must be contiguous")
        return tensor.data_ptr()


class UltraQuantPool(KVCache):
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device,
                 enable_memory_saver, v_head_dim=None, start_layer=None, end_layer=None,
                 enable_alt_stream=False, enable_kv_cache_copy=False):
        if page_size != 1 or dtype != torch.float16 or enable_memory_saver or enable_kv_cache_copy:
            raise ValueError("unsupported UltraQuant pool configuration")
        if v_head_dim not in (None, head_dim) or not 1 <= size < 1048576:
            raise ValueError("unsupported UltraQuant capacity/dimension")
        cell_bytes(layer_num, head_num, head_dim)
        super().__init__(size, page_size, torch.uint8, layer_num, device, False, start_layer, end_layer)
        self.head_num, self.head_dim, self.v_head_dim = head_num, head_dim, head_dim
        self.native = Native(device)
        shape = (size + 1, head_num, head_dim // 32 * 17)
        self.k_buffer = [torch.empty(shape, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.v_buffer = [torch.empty(shape, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        for k, v in zip(self.k_buffer, self.v_buffer):
            self.native.call("uq_initialize", k.data_ptr(), v.data_ptr(), size + 1,
                             head_num, head_dim, self.native.stream())
        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        return (sum(x.numel() for x in self.k_buffer), sum(x.numel() for x in self.v_buffer))

    def _layer(self, layer_id):
        index = layer_id - self.start_layer
        if not 0 <= index < self.layer_num:
            raise ValueError("UltraQuant layer out of bounds")
        return index

    def get_key_buffer(self, layer_id):
        return self.k_buffer[self._layer(layer_id)]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[self._layer(layer_id)]

    def get_kv_buffer(self, layer_id):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        if args or kwargs:
            raise ValueError("UltraQuant refuses additional cache scales or layouts")
        if isinstance(loc, KVWriteLoc):
            if loc.swa_loc is not None or loc.full_loc is not None:
                raise ValueError("UltraQuant refuses translated cache locations")
            loc = loc.loc
        n = loc.numel()
        k, v = self.get_kv_buffer(layer.layer_id)
        self.native.call("uq_append", k.data_ptr(), v.data_ptr(), self.size + 1,
                         self.head_num, self.head_dim,
                         self.native.tensor(cache_k, torch.float16, (n, self.head_num, self.head_dim)),
                         self.native.tensor(cache_v, torch.float16, (n, self.head_num, self.head_dim)),
                         self.native.tensor(loc, torch.int64, (n,)), n,
                         self.native.status.data_ptr(), self.native.stream())

    def move_kv_cache(self, *args, **kwargs):
        raise ValueError("UltraQuant does not support cache copying/offload")


class UltraQuantBackend(AttentionBackend):
    def __init__(self, runner):
        super().__init__()
        validate_args(runner.server_args)
        validate_model(runner.model_config)
        if runner.dtype != torch.float16 or not isinstance(runner.token_to_kv_pool, UltraQuantPool):
            raise ValueError("UltraQuant requires its compressed pool and FP16 computation")
        if runner.mambaish_config or runner.use_mla_backend or runner.is_hybrid_swa or runner.dcp_size != 1:
            raise ValueError("UltraQuant refuses hybrid/latent/parallel attention")
        for phase in (runner.server_args.cuda_graph_config.decode, runner.server_args.cuda_graph_config.prefill):
            if phase.backend != "disabled":
                raise ValueError("UltraQuant does not support graph capture")
        self.pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.query_heads = runner.model_config.num_attention_heads
        self.metadata = None
        self.receipt_emitted = False

    def init_forward_metadata(self, batch):
        mode = batch.forward_mode
        if mode.is_idle():
            self.metadata = None
            return
        if mode.name not in ("DECODE", "EXTEND") or batch.spec_info is not None:
            raise ValueError("UltraQuant supports ordinary prefill and decode only")
        reqs = batch.req_pool_indices.cpu().tolist()
        totals = batch.seq_lens.cpu().tolist()
        if len(reqs) != len(totals) or len(set(reqs)) != len(reqs):
            raise ValueError("invalid UltraQuant request rows")
        if any(type(r) is not int or not 0 <= r < self.req_to_token.shape[0] for r in reqs):
            raise ValueError("UltraQuant request index out of bounds")
        if any(type(n) is not int or not 0 < n <= min(131072, self.req_to_token.shape[1]) for n in totals):
            raise ValueError("UltraQuant sequence length out of bounds")
        if mode.is_decode():
            prefixes, extends = [n - 1 for n in totals], [1] * len(totals)
        else:
            prefixes = batch.extend_prefix_lens.cpu().tolist()
            extends = batch.extend_seq_lens.cpu().tolist()
            starts, cursor = [], 0
            for n in extends:
                starts.append(cursor)
                cursor += n
            if batch.extend_start_loc.cpu().tolist() != starts:
                raise ValueError("UltraQuant extend row order mismatch")
        sequences = [self.req_to_token[r, :n].cpu().tolist() for r, n in zip(reqs, totals)]
        locations = batch.out_cache_loc.to(dtype=torch.int64).contiguous()
        indices, starts, lengths = causal_rows(sequences, prefixes, extends,
                                               locations.cpu().tolist(), self.pool.size + 1)
        if len(starts) != batch.input_ids.numel():
            raise ValueError("UltraQuant query/token count mismatch")
        device = self.pool.native.device
        self.metadata = (locations, torch.tensor(indices, dtype=torch.int64, device=device),
                         torch.tensor(starts, dtype=torch.int32, device=device),
                         torch.tensor(lengths, dtype=torch.int32, device=device))

    def _forward(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        if kwargs or not save_kv_cache or k is None or v is None or self.metadata is None:
            raise ValueError("UltraQuant requires an ordinary cache append/read")
        if (layer.attn_type != AttentionType.DECODER or layer.is_cross_attention or layer.use_irope or
                layer.sliding_window_size != -1 or layer.logit_cap != 0 or
                layer.pos_encoding_mode != "NONE" or layer.xai_temperature_len != -1 or
                layer.tp_q_head_num != self.query_heads or layer.tp_k_head_num != self.pool.head_num or
                layer.tp_v_head_num != self.pool.head_num or layer.qk_head_dim != self.pool.head_dim or
                layer.v_head_dim != self.pool.head_dim or layer.k_scale is not None or layer.v_scale is not None):
            raise ValueError("unsupported UltraQuant attention layer")
        loc, indices, starts, lengths = self.metadata
        rows, dim = loc.numel(), self.pool.head_dim
        q = q.reshape(rows, self.query_heads, dim).contiguous()
        k = k.reshape(rows, self.pool.head_num, dim).contiguous()
        v = v.reshape(rows, self.pool.head_num, dim).contiguous()
        native = self.pool.native
        native.tensor(q, torch.float16, (rows, self.query_heads, dim))
        self.pool.set_kv_buffer(layer, loc, k, v)
        packed_k, packed_v = self.pool.get_kv_buffer(layer.layer_id)
        output = torch.empty_like(q)
        native.call("uq_attention", q.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
                    self.pool.size + 1, self.pool.head_num, self.query_heads, dim,
                    indices.data_ptr(), indices.numel(), starts.data_ptr(), lengths.data_ptr(), rows,
                    layer.scaling, output.data_ptr(), native.status.data_ptr(), native.stream())
        if not self.receipt_emitted:
            # Both writer and compressed reader completed on the GPU. This is
            # execution evidence only, not a numerical or model-quality result.
            self.receipt_emitted = emit_receipt(descriptor(native.build, dim, self.pool.head_num,
                                                           self.query_heads, torch.version.hip))
        return output.reshape(rows, self.query_heads * dim)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def support_triton(self):
        return False

class BF16CacheDescriptor(ctypes.Structure):
    _fields_ = [("k", ctypes.c_void_p), ("v", ctypes.c_void_p),
                ("bytes_per_buffer", ctypes.c_uint64), ("identity", ctypes.c_uint64 * 4),
                ("slots", ctypes.c_uint32), ("protected_layer", ctypes.c_uint32)]


class BF16Native(Native):
    """Separate source-only BF16 ABI adapter; no server registration."""
    def __init__(self, device, identity):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.version.hip:
            raise ValueError("BF16 candidate requires reviewed ROCm execution")
        if torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0] != "gfx1201":
            raise ValueError("BF16 candidate requires gfx1201")
        path = Path(__file__).with_name("libuq_bf16_v2.so")
        self.lib = ctypes.CDLL(str(path))
        self.identity = (ctypes.c_uint64 * 4).from_buffer_copy(bytes.fromhex(identity["sha256"]))
        self.failed = False
        ptr, integer, wide = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.lib.uq_bf16_append_v2.argtypes = [ptr, ptr, ptr, ptr, wide, ptr, integer, ptr, ptr]
        self.lib.uq_bf16_attention_v2.argtypes = [
            ptr, ptr, ptr, wide, ptr, integer, ptr, ptr, integer, ctypes.c_float, ptr, wide, ptr, ptr]
        self.lib.uq_bf16_append_v2.restype = integer
        self.lib.uq_bf16_attention_v2.restype = integer
        self.status = torch.empty(1, dtype=torch.int32, device=device)

    def descriptor(self, k, v, slots, protected):
        return BF16CacheDescriptor(k.data_ptr(), v.data_ptr(), k.numel() * k.element_size(),
                                   self.identity, slots, int(protected))


class QwenBF16MixedPool(KVCache):
    """Full-attention subpool only. GDN allocation and lifecycle stay in parent."""
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device,
                 enable_memory_saver, v_head_dim=None, start_layer=None, end_layer=None,
                 enable_alt_stream=False, enable_kv_cache_copy=False, *, policy,
                 quant_method=None, post_capture_active=False, stack_receipt=None,
                 stack_receipt_sha256=None):
        from sglang.srt.layers.attention.ultraquant_contract import bf16_pool_policy
        from sglang.srt.layers.attention.ultraquant_evidence import bf16_build_identity, BF16CacheOwner
        expected = bf16_pool_policy(rank=policy["rank"], physical_slots=size)
        if dict(policy) != dict(expected) or (page_size, dtype, head_num, head_dim, layer_num) != (
            1, torch.bfloat16, 2, 256, 16
        ) or any((enable_memory_saver, enable_alt_stream, enable_kv_cache_copy,
                  post_capture_active)) or quant_method is not None or (
            v_head_dim not in (None, 256) or start_layer not in (None, 0) or end_layer not in (None, 16)
        ):
            raise ValueError("unsupported mixed BF16 pool configuration")
        # Proof is checked BEFORE native loading or device allocation.
        if stack_receipt is None and stack_receipt_sha256 is None:
            identity = bf16_build_identity(Path(__file__).parents[3], expected)
        else:
            identity = bf16_build_identity(Path(__file__).parents[3], expected,
                stack_receipt=stack_receipt, stack_receipt_sha256=stack_receipt_sha256)
        self.identity = identity
        self.owner = BF16CacheOwner(identity)
        self.policy = expected
        super().__init__(size, page_size, torch.bfloat16, layer_num, device, False, start_layer, end_layer)
        self.native = BF16Native(device, identity)
        self.head_num, self.head_dim, self.v_head_dim = 2, 256, 256
        self.k_buffer, self.v_buffer = [], []
        self.immutable = frozenset()
        for index in range(16):
            protected = index in policy["protected"]
            shape = (size + 1, 2, 256 if protected else 136)
            # BF16 NaN / packed invalid exponent marks unwritten records.
            dt = torch.bfloat16 if protected else torch.uint8
            fill = float("nan") if protected else 255
            self.k_buffer.append(torch.full(shape, fill, dtype=dt, device=device))
            self.v_buffer.append(torch.full(shape, fill, dtype=dt, device=device))

    def get_kv_size_bytes(self):
        return tuple(sum(x.numel() * x.element_size() for x in group)
                     for group in (self.k_buffer, self.v_buffer))

    def get_key_buffer(self, layer_id):
        from sglang.srt.layers.attention.ultraquant_contract import qwen_layer_route
        route = qwen_layer_route(compact_id=layer_id)
        if not route["protected"]:
            raise ValueError("packed KV4 cannot be exposed to an ordinary BF16 reader")
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        self.get_key_buffer(layer_id)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_contiguous_buf_infos(self):
        raise ValueError("mixed pool offload/cache-copy is unsupported")

    def maybe_get_custom_mem_pool(self):
        return None

    def register_layer_transfer_counter(self, counter):
        if counter is not None:
            raise ValueError("mixed pool layer transfer is unsupported")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=1.0, v_scale=1.0,
                      layer_id_override=None, dcp_kv_mask=None):
        from sglang.srt.layers.attention.ultraquant_contract import qwen_writer_metadata, bf16_write_slots
        route = qwen_writer_metadata(layer, loc, cache_k, cache_v, k_scale, v_scale,
                                    layer_id_override=layer_id_override, dcp_kv_mask=dcp_kv_mask)
        self.owner.check_process()
        n = loc.numel()
        location_pointer = self.native.tensor(loc, torch.int64, (n,))
        kp = self.native.tensor(cache_k, torch.bfloat16, (n, 2, 256))
        vp = self.native.tensor(cache_v, torch.bfloat16, (n, 2, 256))
        bf16_write_slots(loc.cpu().tolist(), self.size + 1, self.immutable)
        self.owner.check_writable(loc.cpu().tolist())
        index = route["compact_id"]
        descriptor = self.native.descriptor(self.k_buffer[index], self.v_buffer[index],
                                            self.size + 1, route["protected"])
        self.native.call("uq_bf16_append_v2", ctypes.byref(descriptor), self.native.identity,
                         kp, vp, n * 1024, location_pointer, n, self.native.status.data_ptr(),
                         self.native.stream())

    def move_kv_cache(self, *args, **kwargs):
        raise ValueError("mixed pool cache copy/offload is unsupported")


class QwenBF16SourceAdapter(AttentionBackend):
    """Unregistered full-attention adapter. Caller supplies post-native-MRoPE Q/K."""
    def __init__(self, pool, req_to_token_pool=None, token_to_kv_pool=None, max_context_len=131072):
        if not isinstance(pool, QwenBF16MixedPool):
            raise ValueError("mixed full-attention pool required")
        self.pool = pool
        if type(max_context_len) is not int or not 1 <= max_context_len <= 131072:
            raise ValueError("unsupported BF16 context bound")
        self.token_to_kv_pool = token_to_kv_pool if token_to_kv_pool is not None else pool
        # The pinned 0.5.21 hybrid wrapper reads these full-backend attributes.
        self.data_type = torch.bfloat16
        self.kv_index_translator = None
        self.req_to_token_pool = req_to_token_pool
        self.max_context_len = max_context_len
        self.needs_cpu_seq_lens = True
        self.metadata = None
        self._device_metadata = None
        self._batch = None
        self._write_locations = None
        self._written_layers = set()
        self._slot_snapshot = None
        self._map_snapshot = None
        self.pool.owner.register_adapter(self)

    def invalidate(self):
        """Discard prepared metadata without releasing slots or recurrent state."""
        self.metadata = None
        self._device_metadata = None
        self._batch = None
        self._write_locations = None
        self._written_layers.clear()
        self._slot_snapshot = None
        self._map_snapshot = None

    def _eager_mode(self, batch):
        if batch.forward_mode.name not in ("EXTEND", "DECODE") or batch.spec_info is not None:
            raise ValueError("BF16 supports ordinary eager extend/decode only")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("BF16 does not support graph capture/replay")
        return batch.forward_mode.name

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self.invalidate()
        if in_capture:
            raise ValueError("BF16 does not support graph capture/replay")
        mode = self._eager_mode(forward_batch)
        if self.req_to_token_pool is None:
            raise ValueError("BF16 requires the request-to-token map")
        table = self.req_to_token_pool.req_to_token
        reqs = forward_batch.req_pool_indices.cpu().tolist()
        totals = forward_batch.seq_lens.cpu().tolist()
        if len(table.shape) != 2 or not reqs or len(reqs) != len(totals) or len(set(reqs)) != len(reqs):
            raise ValueError("invalid BF16 request-map metadata")
        if any(type(req) is not int or not 0 <= req < table.shape[0] for req in reqs) or any(
            type(total) is not int or not 1 <= total <= min(self.max_context_len, table.shape[1])
            for total in totals
        ):
            raise ValueError("BF16 request-map bounds exceeded")
        if mode == "DECODE":
            prefixes, extends = [total - 1 for total in totals], [1] * len(totals)
        else:
            prefixes = forward_batch.extend_prefix_lens_cpu
            extends = forward_batch.extend_seq_lens_cpu
            if prefixes is None or extends is None:
                raise ValueError("BF16 requires actual extend prefix/length metadata")
        if not (len(prefixes) == len(extends) == len(totals)) or any(
            type(prefix) is not int or type(extend) is not int or prefix + extend != total
            for prefix, extend, total in zip(prefixes, extends, totals)
        ):
            raise ValueError("inconsistent BF16 extend lengths")
        loc = forward_batch.out_cache_loc
        self.pool.native.tensor(loc, torch.int64, (sum(extends),))
        locations = loc.cpu().tolist()
        sequences = [table[req, :total].cpu().tolist() for req, total in zip(reqs, totals)]
        # Batch prefixes supplement the owner's cache-wide prefix protection.
        rows = self.prepare(sequences, prefixes, extends, locations,
                            self.pool.owner.prefix_key(None), None)
        self._batch = forward_batch
        self._write_locations = tuple(locations)
        self._map_snapshot = (tuple(reqs), tuple(totals), tuple(tuple(row) for row in sequences))
        return rows

    def init_forward_metadata_in_graph(self, forward_batch):
        # Eager callers also use this callback. No host copies or allocations.
        try:
            self._eager_mode(forward_batch)
            if self._batch is not forward_batch or self.metadata is None:
                raise ValueError("BF16 eager metadata is missing or stale")
        except Exception:
            self.invalidate()
            raise

    def init_cuda_graph_state(self, *args, **kwargs):
        self.invalidate()
        raise ValueError("BF16 does not support graph capture/replay")

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.init_cuda_graph_state()

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        self.init_cuda_graph_state()

    def init_forward_metadata_for_breakable_cuda_graph_capture(self, *args, **kwargs):
        self.init_cuda_graph_state()

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, *args, **kwargs):
        self.init_cuda_graph_state()

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self._eager_mode(forward_batch)
        return super().forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if self._eager_mode(forward_batch) != "EXTEND":
            raise ValueError("BF16 extend called with another mode")
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if self._eager_mode(forward_batch) != "DECODE":
            raise ValueError("BF16 decode called with another mode")
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def _forward(self, q, k, v, layer, batch, save_kv_cache, kwargs):
        import math
        from sglang.srt.layers.attention.ultraquant_contract import qwen_writer_metadata
        if self._batch is not batch or self.metadata is None:
            raise ValueError("BF16 eager metadata is missing or stale")
        self.pool.owner.validate_snapshot(self._slot_snapshot)
        reqs, totals, sequences = self._map_snapshot
        if tuple(batch.req_pool_indices.cpu().tolist()) != reqs or tuple(batch.seq_lens.cpu().tolist()) != totals or tuple(
            tuple(self.req_to_token_pool.req_to_token[req, :total].cpu().tolist())
            for req, total in zip(reqs, totals)
        ) != sequences:
            self.invalidate()
            raise ValueError("BF16 request map changed after preparation")
        if type(save_kv_cache) is not bool or any(name not in ("k_scale", "v_scale", "dcp_kv_mask") for name in kwargs):
            raise ValueError("unsupported BF16 forward arguments")
        if tuple(getattr(layer, name, None) for name in (
            "tp_q_head_num", "tp_k_head_num", "tp_v_head_num", "qk_head_dim", "v_head_dim"
        )) != (12, 2, 2, 256, 256):
            raise ValueError("unsupported BF16 layer geometry")
        scale = layer.scaling
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("unsupported BF16 attention scale")
        ks = kwargs.get("k_scale", getattr(layer, "k_scale", None))
        vs = kwargs.get("v_scale", getattr(layer, "v_scale", None))
        scales = [ks, vs]
        for index, name in enumerate(("k_scale", "v_scale")):
            value = scales[index]
            if name not in kwargs and value is not None and type(value) not in (int, float):
                unit = getattr(layer, name + "_float", None)
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != ()
                        or value.dtype != torch.float32 or value.device != self.pool.native.status.device
                        or type(unit) not in (int, float) or not math.isfinite(unit) or unit != 1):
                    raise ValueError("unsupported implicit BF16 KV scale metadata")
                scalar = value.item()
                if type(scalar) not in (int, float) or not math.isfinite(scalar) or scalar != 1:
                    raise ValueError("BF16 requires unit implicit KV scales")
                scales[index] = 1.0
        ks, vs = scales
        mask = kwargs.get("dcp_kv_mask")
        qwen_writer_metadata(layer, None, k, v, ks, vs, dcp_kv_mask=mask)
        n = len(self.metadata[1])
        native = self.pool.native
        if not 1 <= n <= 4096:
            raise ValueError("BF16 input row bound exceeded")
        if (k is None) != (v is None) or save_kv_cache and k is None:
            raise ValueError("BF16 requires paired K/V for cache writes")
        inputs = ((q, 12),) if k is None else ((q, 12), (k, 2), (v, 2))
        # Validate every view before allocating a temporary or invoking native
        # code. A split from the gated QKV projection has a non-dense row stride.
        for tensor, heads in inputs:
            if tensor.device != native.status.device or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) not in (
                (n, heads * 256), (n, heads, 256)
            ):
                raise ValueError("unsupported BF16 input device, dtype or shape")
        loc = batch.out_cache_loc
        native.tensor(loc, torch.int64, (n,))
        if tuple(loc.cpu().tolist()) != self._write_locations:
            raise ValueError("BF16 write locations changed after preparation")
        if save_kv_cache and layer.layer_id in self._written_layers:
            raise ValueError("BF16 layer already appended for this batch")
        def shaped(tensor, heads):
            # At most 4096*(12+2+2)*256*2 = 32 MiB for Q/K/V, within
            # the existing 2 GiB workspace policy. No cast or host transfer.
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tuple(tensor.shape) == (n, heads * 256):
                tensor = tensor.view(n, heads, 256)
            native.tensor(tensor, torch.bfloat16, (n, heads, 256))
            return tensor
        q = shaped(q, 12)
        if k is not None:
            k, v = shaped(k, 2), shaped(v, 2)
        if save_kv_cache:
            # Preserve parent physical-to-compact translation and its writer ABI.
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, ks, vs, dcp_kv_mask=mask)
            self._written_layers.add(layer.layer_id)
        return self.attention(q, layer.layer_id, scale).view(n, 12 * 256)

    def prepare(self, sequences, prefixes, extends, locations, prefix_key, extra_key):
        # A refused new batch must not leave a previous batch executable. Keep
        # existing prefix protection until another preparation succeeds.
        self.invalidate()
        self.pool.owner.validate_prefix(prefix_key, extra_key)
        rows = causal_rows(sequences, prefixes, extends, locations, self.pool.size + 1)
        snapshot = self.pool.owner.snapshot(rows[0])
        device_metadata = tuple(
            torch.tensor(values, dtype=dtype, device=self.pool.native.status.device)
            for values, dtype in zip(rows, (torch.int64, torch.int32, torch.int32))
        )
        # Commit metadata only after all validation; no recurrent-state emulation.
        immutable = frozenset(slot for seq, prefix in zip(sequences, prefixes) for slot in seq[:prefix])
        self.pool.immutable = immutable
        self.metadata = tuple(tuple(row) for row in rows)
        self._device_metadata = device_metadata
        self._slot_snapshot = snapshot
        return self.metadata

    def attention(self, q, physical_layer, scale):
        import math
        from sglang.srt.layers.attention.ultraquant_contract import qwen_layer_route
        if self.metadata is None:
            raise ValueError("causal metadata required")
        self.pool.owner.check_process()
        self.pool.owner.validate_snapshot(self._slot_snapshot)
        if self._device_metadata is None:
            raise ValueError("causal metadata required")
        route = qwen_layer_route(physical_id=physical_layer)
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("unsupported BF16 attention scale")
        indices, starts, lengths = self.metadata
        n = len(starts)
        native = self.pool.native
        qp = native.tensor(q, torch.bfloat16, (n, 12, 256))
        device = q.device
        ids, begin, size = self._device_metadata
        out = torch.empty((n, 12, 256), dtype=torch.bfloat16, device=device)
        i = route["compact_id"]
        descriptor = native.descriptor(self.pool.k_buffer[i], self.pool.v_buffer[i],
                                       self.pool.size + 1, route["protected"])
        native.call("uq_bf16_attention_v2", ctypes.byref(descriptor), native.identity, qp, n * 6144,
                    ids.data_ptr(), len(indices), begin.data_ptr(), size.data_ptr(), n, scale,
                    out.data_ptr(), n * 6144, native.status.data_ptr(), native.stream())
        return out
