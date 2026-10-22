"""Private, bounded DFlash verify A/B; installed only in admitted TP workers.

Keep native cache writes and dispatch. Replace only the final attention call for
one causal eight-token request. All other geometries retain native attention.
The native split selector runs on each row as a singleton decode request.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
from types import CodeType

CURRENT = None


def need(value, message):
    if not value:
        raise ValueError(message)


def source_module(source):
    path = Path(source['path'])
    need(path.is_file() and not path.is_symlink() and path.resolve() == path and
         hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], 'adapter source pin')
    module = importlib.import_module(source['module'])
    need(Path(module.__file__).resolve() == path, 'adapter module origin')
    return module


def checked_method(cls, name, source):
    raw = Path(source['path']).read_bytes()
    code = compile(raw, source['path'], 'exec', dont_inherit=True)
    for part in (cls.__name__, name):
        matches = [x for x in code.co_consts if type(x) is CodeType and x.co_name == part]
        need(len(matches) == 1, 'adapter native method source')
        code = matches[0]
    method = cls.__dict__.get(name)
    need(inspect.isfunction(method) and method.__code__ == code, 'adapter native method changed')
    return method


def eligible(backend, batch, args, kwargs):
    q, k, v, out, kb, vb, qo, ptr, indices, mask, causal, mi, width, ks, vs, scale = args
    return (batch.forward_mode.is_target_verify() and batch.batch_size == 1 and
        tuple(q.shape) == (8, 12, 256) and tuple(k.shape) == tuple(v.shape) == (8, 2, 256) and
        tuple(out.shape) == (8, 12, 256) and tuple(kb.shape[-2:]) == tuple(vb.shape[-2:]) == (2, 256) and
        str(q.dtype) == str(k.dtype) == str(v.dtype) == str(out.dtype) == str(kb.dtype) == str(vb.dtype) == 'torch.bfloat16' and
        causal is True and mask is None and width == 8 and ptr.numel() == qo.numel() == 2 and
        ks == vs == 1.0 and scale == 0.0625 and backend.page_size == 1 and
        backend.dcp_size == 1 and backend.use_mla is False and
        backend.enable_deterministic is False and backend.static_kv_splits is False and
        backend.split_tile_size is None and backend.max_kv_splits == 16 and
        backend.device_core_count > 0 and backend.num_head == 12 and backend.num_kv_head == 2 and
        backend.kv_index_translator.reads_are_translated is False and
        8 <= backend.max_context_len <= 32768 and
        kwargs.get('logit_cap', 0) == 0 and kwargs.get('sliding_window_size', -1) == -1 and
        kwargs.get('page_size', 1) == 1 and kwargs.get('xai_temperature_len') in (None, -1, 0) and
        all(kwargs.get(key) is None for key in ('sinks', 'window_kv_offsets', 'score_mod', 'aux_tensors')))


class Adapter:
    def __init__(self, manifest, worker, cls, torch):
        self.manifest, self.worker, self.cls, self.torch = manifest, worker, cls, torch
        self.enabled = manifest['arm'] == 'mtp'
        self.original = checked_method(cls, 'forward_extend', manifest['sources']['triton_backend'])
        checked_method(cls, 'get_num_kv_splits', manifest['sources']['triton_backend'])
        self.buffers = {}
        self.host_calls = self.eligible_host_calls = self.capture_host_calls = 0
        self.layers = set()
        self.last = None
        adapter = self
        original = self.original

        def wrapper(backend, q, k, v, layer, forward_batch, *args, **kwargs):
            if not forward_batch.forward_mode.is_target_verify():
                return original(backend, q, k, v, layer, forward_batch, *args, **kwargs)
            adapter.host_calls += 1
            native = backend.extend_attention_fwd
            had_own = 'extend_attention_fwd' in backend.__dict__

            def attention(*values, **options):
                if not eligible(backend, forward_batch, values, options):
                    return native(*values, **options)
                adapter.eligible_host_calls += 1
                adapter.layers.add(layer.layer_id)
                if torch.cuda.is_current_stream_capturing():
                    adapter.capture_host_calls += 1
                if not adapter.enabled:
                    return native(*values, **options)
                return adapter.decode(backend, forward_batch, values)

            backend.extend_attention_fwd = attention
            try:
                return original(backend, q, k, v, layer, forward_batch, *args, **kwargs)
            finally:
                need(backend.extend_attention_fwd is attention, 'adapter attention hook replaced')
                if had_own:
                    backend.extend_attention_fwd = native
                else:
                    del backend.extend_attention_fwd

        self.wrapper = wrapper
        cls.forward_extend = wrapper

    def decode(self, backend, batch, args):
        torch = self.torch
        q, k, v, out, kb, vb, qo, ptr, indices, mask, causal, mi, width, ks, vs, scale = args
        key = id(backend)
        if key not in self.buffers:
            native_decode = importlib.import_module(self.manifest['sources']['decode_attention_fwd']['module'])
            need(inspect.unwrap(backend.decode_attention_fwd) is native_decode.decode_attention_fwd,
                 'adapter decode callable changed')
            device = q.device
            empty = lambda shape, dtype: torch.empty(shape, device=device, dtype=dtype)
            self.buffers[key] = dict(
                offsets=torch.arange(1, 9, device=device, dtype=torch.int32),
                lengths=empty(8, torch.int32), indptr=torch.zeros(9, device=device, dtype=torch.int32),
                indices=empty(8*backend.max_context_len, torch.int64),
                requests=empty(8, batch.req_pool_indices.dtype), splits=empty(8, torch.int32),
                partial=empty((8, 12, 16, 256), torch.float32), lse=empty((8, 12, 16), torch.float32))
        b = self.buffers[key]
        torch.add(ptr[1]-ptr[0], b['offsets'], out=b['lengths'])
        torch.cumsum(b['lengths'], dim=0, out=b['indptr'][1:])
        b['requests'].copy_(batch.req_pool_indices[:1].expand(8))
        backend.kv_index_translator.fill_packed_read_stream(
            req_pool_indices=b['requests'], seq_lens=b['lengths'], indptr=b['indptr'],
            total_tokens=8*backend.max_context_len, out=b['indices'])
        for row in range(8):
            backend.get_num_kv_splits(b['splits'][row:row+1], b['lengths'][row:row+1])
        backend.decode_attention_fwd(q, kb, vb, out, b['indptr'], b['indices'],
            b['partial'], b['lse'], b['splits'], 16, scale, ks, vs, enable_lean=False, page_size=1)
        self.last = (backend, b, ptr, indices, batch.out_cache_loc)

    def snapshot(self, prefix):
        need(not self.torch.cuda.is_current_stream_capturing(), 'adapter snapshot during capture')
        result = dict(enabled=self.enabled, source=self.manifest['sources']['verify_decode_adapter'],
            rank=self.worker['rank'], host_calls=self.host_calls, eligible_host_calls=self.eligible_host_calls,
            capture_host_calls=self.capture_host_calls, layers=sorted(self.layers),
            policy='native singleton per causal row', qualification=False)
        if not self.enabled:
            return result
        need(self.last is not None, 'adapter never dispatched')
        backend, b, ptr, indices, suffix = self.last
        lengths = b['lengths'].cpu().tolist()
        indptr = b['indptr'].cpu().tolist()
        splits = b['splits'].cpu().tolist()
        native_ptr = ptr.cpu().tolist()
        need(native_ptr[1]-native_ptr[0] == prefix and lengths == list(range(prefix+1, prefix+9)),
             'adapter snapshot prefix differs from selected verify block')
        matches = []
        for row in range(8):
            actual = b['indices'][indptr[row]:indptr[row+1]]
            matches.append(self.torch.equal(actual[:prefix], indices[native_ptr[0]:native_ptr[1]]) and
                           self.torch.equal(actual[prefix:], suffix[:row+1]))
        result.update(lengths=lengths, indptr=indptr, splits=splits,
            mapped_causal_rows_match=all(matches), device_core_count=backend.device_core_count,
            retained_buffer_bytes=sum(x.numel()*x.element_size() for x in b.values()),
            metadata_cpu_copy_bytes=(8+9+8+len(native_ptr))*4,
            mapping_comparisons=16, snapshot_instrumented=True)
        need(result['mapped_causal_rows_match'], 'adapter causal KV map differs')
        return result

    def restore(self):
        need(self.cls.forward_extend is self.wrapper or self.cls.forward_extend is self.original,
             'adapter forward hook replaced')
        self.cls.forward_extend = self.original


def install(manifest, worker):
    global CURRENT
    need(CURRENT is None and manifest['arm'] in ('control', 'mtp'), 'adapter duplicate install/arm')
    sources = manifest['sources']
    modules = {role: source_module(sources[role]) for role in
        ('triton_backend', 'decode_attention_fwd', 'extend_attention_fwd',
         'verify_metadata', 'verify_kv_indices', 'verify_kv_translator', 'verify_common')}
    backend = modules['triton_backend']
    need(backend.get_num_kv_splits_triton is modules['verify_metadata'].get_num_kv_splits_triton and
         modules['verify_kv_translator'].create_flashinfer_kv_indices_triton is
         modules['verify_kv_indices'].create_flashinfer_kv_indices_triton, 'adapter native dependency identity')
    import torch
    CURRENT = Adapter(manifest, worker, backend.TritonAttnBackend, torch)
    return CURRENT


def snapshot(prefix):
    need(CURRENT is not None, 'adapter absent')
    return CURRENT.snapshot(prefix)
