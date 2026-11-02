"""Source-bound J01 logit and optional graph-input/weight capture.

Native arithmetic, RNG, and collectives stay unchanged. Optional input capture
delegates existing graph callbacks/target forwards and copies bounded rows to
CPU; no extra model forward is performed. Instrumented timings are not native.
"""
from array import array
import base64
import hashlib
import inspect
from importlib.machinery import SourceFileLoader
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
from types import CodeType


def _load_observer_helper(name, expected_sha256):
    """Execute authenticated source bytes, without consulting cached bytecode."""
    import importlib.util
    path = Path(__file__).with_name(name + '.py')
    error = name.replace('_', ' ') + ' source binding'
    if (path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1
            or path.stat().st_size > 64 << 10):
        raise ValueError(error)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(error)
    loader = SourceFileLoader('_spry_' + name, str(path))
    loader.get_code = lambda fullname: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_health = _load_observer_helper('observer_health',
    '63404904c25e874bcecb58d14860925df9766deb68c487515e7af1fad50db12a')

MODE = 'j01-greedy-index6-v1'
CASE = 'prompt-j01-only'
ORDINAL, INDEX, VOCAB = 8, 6, 248320
HISTORICAL = [58, 4754, 306, 3147, 41, 15]
HISTORY_SHA256 = 'd65403d629ae47c7ae2c6f9eff668c6c288bd539aad90160801ed799a88c0275'
SCORES = (763, 50303)
MAX_BYTES, MAX_NS = 65536, 15_000_000_000
IDENTITY = ('arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')
WORKER = ('rank', 'pid', 'start_ticks', 'boot_id')
HEAD_ROLES = frozenset(('head_logits', 'head_embedding', 'head_unquant', 'head_linear',
                        'head_features', 'head_context', 'head_model'))
INPUT_MODE = 'j01-hidden-weights-v1'
INPUT_ROLES = HEAD_ROLES | {'head_decode_runner', 'head_model_base', 'backend', 'model_runner'}
INPUT_FILE_BYTES = 65536
INPUT_WIDTH = 5120


import functools

def _load_spry_common(caller, pin, error):
    """Authenticate common source before any shared guard can execute."""
    import hashlib
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import os
    from pathlib import Path
    import stat

    def check(value, message):
        if not value:
            raise error('shared common source: ' + message)

    origin = Path(caller)
    path = origin.with_name('spry_common_canon.py')
    if origin.name.startswith('common_') and origin.parent.name in ('spry_serving', 'spry_evidence'):
        path = origin.resolve().parents[1] / 'spry_common/selected.py'
    elif (origin.parent.name == 'review' and origin.parent.parent.name == 'common-closure'
          and origin.parent.parent.parent.name == 'proposals'):
        path = origin.resolve().parents[3] / 'spry_common/selected.py'
    check(path.is_absolute() and path.resolve(strict=True) == path, 'canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 4096, 'bounded regular file required')
        raw = os.read(descriptor, 4097)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        check(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(descriptor))
              == stamp(path.lstat()), 'file changed during read')
    finally:
        os.close(descriptor)
    check(hashlib.sha256(raw).hexdigest() == pin, 'digest differs')
    # Use the checked bytes, ignoring ambient modules and cached bytecode.
    loader = SourceFileLoader('_spry_common', str(path))
    loader.get_code = lambda name: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader('_spry_common', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_SPRY_COMMON_SHA256 = 'c2cb7702d6ae482b0f9d70b6b838af3d0b9eb312140ff09d001df0880a9e6bed'
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, ValueError)


def need(value, reason):
    return _SPRY_COMMON.need(value, reason, error=ValueError, prefix='J01 probe: ')


canonical = _SPRY_COMMON.canonical_line


sha = _SPRY_COMMON.digest


def ids(value, length=None):
    return type(value) is list and (length is None or len(value) == length) and all(type(x) is int and 0 <= x < VOCAB for x in value)


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def head_enabled(binding):
    roles = HEAD_ROLES & binding['sources'].keys()
    need(not roles or roles == HEAD_ROLES, 'incomplete head source roles')
    return bool(roles)


def head_module(source):
    path = Path(source['path'])
    module = sys.modules.get(source['module'])
    need(module is not None and path.resolve(strict=True) == path and not path.is_symlink() and
         sha(path.read_bytes()) == source['sha256'] and module.__file__ == str(path) and
         module.__spec__.origin == str(path) and type(module.__spec__.loader) is SourceFileLoader,
         'loaded head module source identity')
    return module


class GatherStartup:
    """Observe FP32 native gather geometry at startup; no tensor operations."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.source = manifest['sources']['head_gather']
        self.module = head_module(self.source)
        self.cls = self.module.MultimemAllGatherer
        self.original = self.cls.__dict__.get('__call__')
        code = SourceFileLoader(self.source['module'], self.source['path']).source_to_code(
            Path(self.source['path']).read_bytes(), self.source['path'])
        for name in ('MultimemAllGatherer', '__call__'):
            matches = [x for x in code.co_consts if type(x) is CodeType and x.co_name == name]
            need(len(matches) == 1, 'gather native method source')
            code = matches[0]
        need(inspect.isfunction(self.original) and self.original.__code__ == code and
             self.original.__globals__ is self.module.__dict__, 'gather native method substituted')
        self.records, self.calls, self.restored = {}, 0, False
        original = self.original
        def wrapper(gatherer, x):
            try:
                timer=getattr(self, 'phase_timing', None)
                ticket=timer.capture_gather_begin(x) if timer is not None else None
                result = original(gatherer, x)
                if timer is not None:timer.events.capture_end(ticket)
                self.observe(gatherer, x, result)
                return result
            except BaseException:
                self.restore()
                raise
        self.wrapper = wrapper
        self.cls.__call__ = wrapper

    def observe(self, gatherer, x, result):
        self.calls += 1
        need(self.calls <= 4096, 'gather startup call bound')
        if str(x.dtype) != 'torch.float32':
            return
        need(len(x.shape) == len(result.shape) == 2, 'gather matrix geometry')
        capture = bool(sys.modules['torch'].cuda.is_current_stream_capturing())
        shape, output = tuple(x.shape), tuple(result.shape)
        key = (shape, output, str(result.dtype), capture)
        if key not in self.records:
            need(len(self.records) < 32, 'gather metadata record bound')
            self.records[key] = dict(pid=os.getpid(), arm=self.manifest['arm'],
                source_sha256=self.source['sha256'], input_shape=list(shape),
                output_shape=list(output), dtype=str(x.dtype), output_dtype=str(result.dtype),
                local_tensor_bytes=x.numel()*x.element_size(),
                result_tensor_bytes=result.numel()*result.element_size(),
                capture=capture, native_fp32_fallback=gatherer._state is None,
                calls=0, extra_tensor_operations=0, extra_model_forwards=0,
                qualification=False)
        row = self.records[key]
        need(row['native_fp32_fallback'] is (gatherer._state is None), 'gather route changed')
        row['calls'] += 1

    def restore(self):
        current = self.cls.__dict__.get('__call__')
        need(current is self.wrapper or current is self.original, 'gather observer ownership')
        if current is self.wrapper:
            self.cls.__call__ = self.original
        self.restored = True

    def finish(self):
        self.restore()
        need(self.records, 'FP32 gather never observed')
        for row in self.records.values():
            print('SPRY_HEAD_GATHER ' + json.dumps(dict(row, hooks_restored=self.restored),
                  sort_keys=True, separators=(',', ':')), flush=True)


class HeadStartup:
    """Observe unchanged native head calls during startup; restore before traffic.

    Only tensor metadata is read. No tensor math, D2H copies, synchronizations,
    warmup calls, allocator changes, or result substitutions are added.
    """
    def __init__(self, manifest, input_capture=None, *, phase_timing=None):
        need(head_enabled(manifest), 'head startup mode required')
        self.sources = {role: manifest['sources'][role] for role in sorted(HEAD_ROLES)}
        self.module = head_module(self.sources['head_logits'])
        self.cls = self.module.LogitsProcessor
        self.original = self.cls.__dict__.get('_compute_lm_head')
        source = self.sources['head_logits']
        code = SourceFileLoader(source['module'], source['path']).source_to_code(
            Path(source['path']).read_bytes(), source['path'])
        for name in ('LogitsProcessor', '_compute_lm_head'):
            matches = [value for value in code.co_consts if type(value) is CodeType and value.co_name == name]
            need(len(matches) == 1, 'native head body missing')
            code = matches[0]
        need(inspect.isfunction(self.original) and self.original.__code__ == code and
             self.original.__globals__ is self.module.__dict__, 'native head body substituted')
        self.entries, self.calls, self.restored = {}, 0, False
        self.d3b_entries = {}  # D3b amendment: replaced target verification calls
        self.input_capture = input_capture
        self.phase_timing = phase_timing
        self.gather_startup = GatherStartup(manifest)
        self.gather_startup.phase_timing = phase_timing
        original = self.original
        def wrapper(processor, hidden_states, lm_head, embedding_bias=None):
            try:
                timer=self.phase_timing
                ticket=timer.capture_head_begin(processor, hidden_states, lm_head) if timer is not None else None
                result = original(processor, hidden_states, lm_head, embedding_bias)
                if timer is not None:timer.events.capture_end(ticket)
            except BaseException:
                self.restore()
                raise
            # Record actual inputs and the result before the native FP32-buffer copy.
            try:
                self.observe(processor, hidden_states, lm_head, result)
                if self.input_capture is not None:
                    self.input_capture.observe(processor, hidden_states, lm_head, embedding_bias)
            except BaseException:
                self.restore()
                raise
            return result
        wrapper.d3b_observe = self.observe_d3b
        self.wrapper = wrapper
        self.cls._compute_lm_head = wrapper

    def observe_d3b(self, processor, hidden, head, result):
        """D3b amendment: the target FP8 head reports its replaced call; metadata only, no substitution."""
        need(not self.restored, 'D3b head report after startup restore')
        self.observe(processor, hidden, head, result, d3b=True)

    def observe(self, processor, hidden, head, result, d3b=False):
        self.calls += 1
        need(self.calls <= 4096, 'startup head callback bound')
        key = (id(processor), id(head))
        quant = getattr(head, 'quant_method', None)
        apply_quant = self.module.should_apply_lm_head_quant_method(head, quant)
        lora = hasattr(head, 'set_lora') and hasattr(head, 'apply_lora')
        use_fp32 = processor.use_fp32_lm_head
        weight = head.weight
        same_half = hidden.is_cuda and hidden.dtype == weight.dtype and str(hidden.dtype) in (
            'torch.float16', 'torch.bfloat16')
        branch = ('lora' if lora else 'quantized' if apply_quant else
                  'dense_mm_out_dtype' if use_fp32 and same_half else
                  'dense_explicit_fp32' if use_fp32 else 'other_native_head')
        if d3b:
            need(branch == 'dense_mm_out_dtype' and len(hidden.shape) == 2 and int(hidden.shape[0]) in D3B_ROWS,
                 'D3b head call outside the dense FP32 verification rows')
            branch = 'd3b_fp8_fp32_logits'
        context = head_module(self.sources['head_context'])
        feature = context.get_exec().features.enable_fp32_lm_head
        record = dict(use_fp32_lm_head=use_fp32, global_fp32_feature=feature,
            branch=branch, input_dtype=str(hidden.dtype), weight_dtype=str(weight.dtype),
            output_dtype=str(result.dtype), input_shape=list(hidden.shape),
            weight_shape=list(weight.shape), output_shape=list(result.shape),
            processor_class=type(processor).__name__, head_class=type(head).__name__,
            quant_method_class=None if quant is None else type(quant).__name__,
            calls=1, graph_capture_calls=int(sys.modules['torch'].cuda.is_current_stream_capturing()))
        entries = self.d3b_entries if d3b else self.entries
        if d3b:
            record['rows'] = [int(hidden.shape[0])]
        if key in entries:
            old = entries[key][2]
            record['calls'] += old['calls']
            record['graph_capture_calls'] += old['graph_capture_calls']
            if d3b:
                record['rows'] = sorted(set(old['rows']) | set(record['rows']))
        else:
            need(len(entries) < 8, 'startup head instance bound')
        entries[key] = (processor, head, record)

    def restore(self):
        current = self.cls.__dict__.get('_compute_lm_head')
        need(current is self.wrapper or current is self.original, 'owned startup head callback replaced')
        if current is self.wrapper:
            self.cls._compute_lm_head = self.original
        self.restored = True
        self.gather_startup.restore()

    def finish(self, scheduler):
        self.restore()
        model = scheduler.tp_worker.model_runner.model
        processor, head = model.logits_processor, model.lm_head
        key = (id(processor), id(head))
        d3b = d3b_target_head_enabled()
        need(d3b or not self.d3b_entries, 'D3b head report without the D3b setting')
        if d3b:
            need(set(self.d3b_entries) == {key}, 'D3b target head not observed during startup')
        else:
            need(key in self.entries, 'target head not observed during native startup')
        need(type(processor) is self.cls, 'target processor class differs')
        embedding = head_module(self.sources['head_embedding'])
        need(type(head).__module__ == self.sources['head_embedding']['module'] and
             getattr(embedding, type(head).__name__, None) is type(head), 'target head class source differs')
        head_module(self.sources['head_model'])
        need(type(model).__module__ == self.sources['head_model']['module'], 'target model class source differs')
        record = dict(schema=1, mode=D3B_HEAD_MODE if d3b else 'native-fp32-lm-head-startup-v1',
            sources=self.sources, target=self.entries[key][2] if key in self.entries else None,
            other_processors=[value[2] for other, value in self.entries.items() if other != key],
            startup_compute_calls=self.calls, restored=self.restored,
            before_external_generations=True, extra_model_forwards=0,
            extra_tensor_operations=0, qualification=False)
        if d3b:
            record['d3b_target'] = self.d3b_entries[key][2]
        validate_head_startup(record, scheduler_manifest_sources=self.sources, d3b=d3b)
        self.gather_startup.finish()
        if self.phase_timing is not None:self.phase_timing.finish_capture()
        if self.input_capture is not None:
            self.input_capture.finish(scheduler)
        return record


D3B_ROWS = (8, 16)
D3B_SETTING = 'fp8-fp32-logits-verify-rows-v1'
D3B_HEAD_MODE = 'd3b-fp8-fp32-logits-head-startup-v1'


def d3b_target_head_enabled():
    import mtp_contract
    value = mtp_contract.settings().get('d3b_target_head')
    need(value in (None, D3B_SETTING), 'unknown D3b target head setting')
    return value is not None


def _validate_d3b_head_startup(record, *, scheduler_manifest_sources):
    """D3b amendment (owner-approved 2026-10-29): verification rows run the FP8 head with FP32 logits."""
    need(type(record) is dict and record.get('mode') == D3B_HEAD_MODE and type(record.get('d3b_target')) is dict,
         'D3b startup head record mode')
    native = dict(record, mode='native-fp32-lm-head-startup-v1')
    row = dict(native.pop('d3b_target'))
    rows = row.pop('rows', None)
    need(row.get('branch') == 'd3b_fp8_fp32_logits' and type(rows) is list and rows and rows == sorted(set(rows)) and
         set(rows) <= set(D3B_ROWS) and type(row.get('input_shape')) is list and row['input_shape'][:1] in ([8], [16]) and
         row['input_shape'][0] in rows, 'D3b target head rows/branch')
    # Every other property of the replaced call must satisfy the unchanged native FP32 rule.
    validate_head_startup(dict(native, target=dict(row, branch='dense_mm_out_dtype')),
                          scheduler_manifest_sources=scheduler_manifest_sources)
    if native['target'] is not None:
        validate_head_startup(native, scheduler_manifest_sources=scheduler_manifest_sources)
    return record


def validate_head_startup(record, *, scheduler_manifest_sources, d3b=False):
    if d3b:
        return _validate_d3b_head_startup(record, scheduler_manifest_sources=scheduler_manifest_sources)
    fields = {'schema', 'mode', 'sources', 'target', 'other_processors', 'startup_compute_calls',
              'restored', 'before_external_generations', 'extra_model_forwards', 'extra_tensor_operations', 'qualification'}
    need(type(record) is dict and set(record) == fields and record['schema'] == 1 and
         record['mode'] == 'native-fp32-lm-head-startup-v1' and record['qualification'] is False and
         record['sources'] == {role: scheduler_manifest_sources[role] for role in sorted(HEAD_ROLES)} and
         record['restored'] is True and record['before_external_generations'] is True and
         record['extra_model_forwards'] == record['extra_tensor_operations'] == 0 and
         type(record['startup_compute_calls']) is int and 1 <= record['startup_compute_calls'] <= 4096 and
         type(record['other_processors']) is list and len(record['other_processors']) <= 7,
         'startup head record identity/schema')
    row = record['target']
    need(type(row) is dict and set(row) == {'use_fp32_lm_head', 'global_fp32_feature', 'branch',
         'input_dtype', 'weight_dtype', 'output_dtype', 'input_shape', 'weight_shape', 'output_shape',
         'processor_class', 'head_class', 'quant_method_class', 'calls', 'graph_capture_calls'} and
         row['use_fp32_lm_head'] is row['global_fp32_feature'] is True and
         row['branch'] == 'dense_mm_out_dtype' and row['input_dtype'] == row['weight_dtype'] == 'torch.bfloat16' and
         row['output_dtype'] == 'torch.float32' and row['processor_class'] == 'LogitsProcessor' and
         row['head_class'] == 'ParallelLMHead' and row['quant_method_class'] in (None, 'UnquantizedEmbeddingMethod', 'UnquantizedLinearMethod') and
         type(row['calls']) is int and 1 <= row['calls'] <= record['startup_compute_calls'] and
         type(row['graph_capture_calls']) is int and 0 <= row['graph_capture_calls'] <= row['calls'],
         'native FP32 target head branch/dtype not witnessed')
    for field in ('input_shape', 'weight_shape', 'output_shape'):
        need(type(row[field]) is list and len(row[field]) == 2 and
             all(type(value) is int and value > 0 for value in row[field]), 'native head matrix geometry')
    need(row['input_shape'][1] == row['weight_shape'][1] and
         row['output_shape'] == [row['input_shape'][0], row['weight_shape'][0]] and
         len(canonical(record)) < 16384, 'native head output geometry/record bound')
    return record


def draft_kv_startup(scheduler, manifest, worker):
    """Read resolved cache and buffer metadata; perform no device operations."""
    need(set(('native_kv_dtype', 'native_spec_fields')) <= manifest['sources'].keys(),
         'native draft KV configuration sources missing')
    import mtp_contract
    profile = mtp_contract.cache_profile(manifest['arm'])
    fp8 = profile['target'] == 'fp8_e4m3'
    runners = _target_kv_metadata.startup(scheduler, target_fp8=fp8,
        target_cache_dtype='bf16' if profile['target'] == 'bf16' else None)
    record = dict(schema=1, **{key: manifest[key] for key in IDENTITY},
        **{key: worker[key] for key in WORKER}, worker_identity_sha256=worker['identity_sha256'],
        helper_source=manifest['sources']['j01_probe'],
        configuration_sources={key: manifest['sources'][key] for key in
                               ('native_kv_dtype', 'native_spec_fields')},
        status=('FP8_TARGET_BF16_DRAFT_CACHE_CONFIRMED' if fp8 else
                'BF16_TARGET_AND_DRAFT_CACHE_CONFIRMED'), runners=runners,
        scope='startup metadata before external requests',
        extra_tensor_operations=0, extra_model_forwards=0, cpu_tensor_copies=0,
        qualification=False)
    if mtp_contract.dense_mxfp4_enabled():
        record['cache_profile'] = profile
    need(len(canonical(record)) < 16384, 'draft KV startup record budget')
    need(_FP8_TARGET_WITNESS_STATE is not None, 'target KV witness state absent')
    path = Path(manifest['evidence_dir']) / ('TARGET-KV-STARTUP-rank-'+str(worker['rank'])+'.json')
    with path.open('xb') as stream:stream.write(canonical(record))
    _FP8_TARGET_WITNESS_STATE['stage'] = 'external'
    return record


class LifecycleSlots:
    def __init__(self,scheduler,manifest,worker):
        self.manifest=manifest;self.worker=worker;self.saved=[];self.seq=0
        self.event_limit = 8192
        self.observer_failed = False
        self.saturated = False
        self.failure_type = None
        self.closed = False
        self.pool=scheduler.tp_worker.model_runner.req_to_token_pool
        self.allocator=self.pool.mamba_allocator;self.cache=scheduler.tree_cache
        self.active={};self.pending={};self.scalar_reads=0
        self.universe=set(range(1,20))
        assert self.allocator.size==19 and self.pool.mamba_pool.size==19
        self.check_source(self.pool.alloc,manifest['sources']['memory_pool'])
        self.check_source(self.pool.free,manifest['sources']['memory_pool'])
        self.check_source(self.allocator._do_alloc,manifest['sources']['mamba_allocator'])
        self.check_source(self.allocator.free,manifest['sources']['mamba_allocator'])
        self.check_source(self.cache.match_prefix,manifest['sources']['cache_tree'])
        self.check_source(self.cache.mamba_evictable_size,manifest['sources']['cache_tree'])
        self.check_source(self.cache.mamba_protected_size,manifest['sources']['cache_tree'])
        try:
            self.wrap(self.allocator,'_do_alloc',self.on_index_alloc)
            self.wrap(self.allocator,'free',self.on_index_free)
            self.wrap(self.pool,'alloc',self.on_request_alloc)
            self.wrap(self.pool,'free',self.on_request_free)
            self.wrap(self.cache,'match_prefix',self.on_cache_match)
            self.observe(self.emit, 'startup', stable=True)
        except BaseException:
            self.restore();raise

    def fail(self, error=None, *, saturated=False):
        """Disable only observation. Never include exception text in the log."""
        if self.observer_failed:
            return
        self.observer_failed = True
        self.saturated = saturated
        self.failure_type = 'event_limit' if saturated else type(error).__name__
        try:
            print('SPRY_LIFECYCLE_OBSERVER_FAILED ' + canonical(dict(
                rank=self.worker['rank'], observer_failed=True,
                saturated=self.saturated, failure_type=self.failure_type)).decode().strip(), flush=True)
        except Exception:
            # The final health receipt is authoritative, even if logging fails.
            pass

    def observe(self, callback, *args, **kwargs):
        """Contain evidence callbacks only; serving calls stay outside this boundary."""
        if self.observer_failed or self.closed:
            return None
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            self.fail(error)
            return None

    def check_source(self,method,pin):
        fn=inspect.unwrap(method)
        path=Path(inspect.getsourcefile(fn)).resolve()
        assert str(path)==pin['path'] and sha(path.read_bytes())==pin['sha256']
        assert method.__self__ in (self.pool,self.allocator,self.cache)

    def indices(self,tensor,maximum=19):
        if tensor is None:return None
        assert not sys.modules['torch'].cuda.is_current_stream_capturing()
        assert tensor.numel()<=maximum and str(tensor.dtype)=='torch.int64'
        # Deliberate, bounded integer-index device-to-host observation. This is
        # disclosed independently from the metadata-only startup cache witness.
        values=tensor.tolist();self.scalar_reads+=tensor.numel()
        if type(values) is int:values=[values]
        assert type(values) is list and all(type(x) is int and x in self.universe for x in values)
        return values

    def scalar(self,tensor):
        values=self.indices(tensor,1)
        return None if values is None else values[0]

    def request(self,req):
        ids=req.origin_input_ids
        assert (type(ids) is list or (type(ids) is array and ids.typecode=='q')) and ids and len(ids)<=131072 and all(type(x) is int for x in ids)
        ids=list(ids)
        row=dict(rid=req.rid,input_ids_sha256=sha(canonical(ids)),cache_salt=req.cache_salt,
                 request_slot=req.kv.req_pool_idx,mamba_slot=self.scalar(req.kv.mamba_pool_idx),
                 cow_source_slot=self.scalar(getattr(req.kv,'mamba_cow_src_index',None)))
        assert type(row['rid']) is str and 0<len(row['rid'])<=256
        assert row['request_slot'] is None or type(row['request_slot']) is int
        return row

    def emit(self,event,*,stable=False,**fields):
        if self.seq >= self.event_limit:
            self.fail(saturated=True)
            return
        free=self.indices(self.allocator.free_slots)
        assert len(set(free))==len(free)
        reqfree=list(self.pool.free_slots)
        assert all(type(x) is int for x in reqfree) and len(set(reqfree))==len(reqfree)
        active=[dict(rid=k,**v) for k,v in sorted(self.active.items())]
        pending=[dict(rid=k,**v) for k,v in sorted(self.pending.items())]
        if stable:
            held=[x['mamba_slot'] for x in active+pending if x['mamba_slot'] is not None]
            assert len(set(held))==len(held) and not(set(held)&set(free))
        record={k:self.manifest[k] for k in ['mode','arm','session_sha256','spec_sha256','source_manifest_sha256']}
        record.update(schema=1,rank=self.worker['rank'],pid=self.worker['pid'],
            worker_identity_sha256=self.worker['identity_sha256'],sequence=self.seq,
            observed_monotonic=time.monotonic(),event=event,stable=stable,free_mamba_slots=free,
            free_request_slots=reqfree,active_requests=active,pending_requests=pending,
            mamba_evictable=self.cache.mamba_evictable_size(),mamba_protected=self.cache.mamba_protected_size(),
            integer_indices_read_total=self.scalar_reads,numerical_tensor_values_read=0,
            extra_model_forwards=0,performance_measurement=False,qualification=False,**fields)
        raw=canonical(record);assert len(raw)<16384
        print('SPRY_LIFECYCLE_SLOTS '+raw.decode().strip(),flush=True);self.seq+=1

    def wrap(self,obj,name,callback):
        assert name not in obj.__dict__
        original=getattr(obj,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):return callback(original,*args,**kwargs)
        setattr(obj,name,wrapped);self.saved.append((obj,name,wrapped))

    def on_index_alloc(self,original,*args,**kwargs):
        result=original(*args,**kwargs)
        self.observe(self.record_index_alloc, result)
        return result

    def record_index_alloc(self, result):
        self.emit('index_alloc',indices=self.indices(result))

    def on_index_free(self,original,values,*args,**kwargs):
        indices=self.observe(self.indices, values)
        result=original(values,*args,**kwargs)
        self.observe(self.emit, 'index_free', indices=indices)
        return result

    def on_request_alloc(self,original,reqs,*args,**kwargs):
        result=original(reqs,*args,**kwargs)
        self.observe(self.record_request_alloc, reqs, result)
        return result

    def record_request_alloc(self, reqs, result):
        records=[]
        if result is not None:
            for req in reqs:
                row=self.request(req);records.append(row);self.pending.pop(req.rid,None)
                self.active[req.rid]={k:row[k] for k in ['request_slot','mamba_slot']}
        self.emit('request_alloc',stable=True,requests=records)

    def on_request_free(self,original,req,*args,**kwargs):
        before=self.observe(self.request, req)
        result=original(req,*args,**kwargs)
        self.observe(self.record_request_free, req, before)
        return result

    def record_request_free(self, req, before):
        self.active.pop(req.rid,None);self.pending.pop(req.rid,None)
        after=self.request(req)
        self.emit('request_free',stable=True,before=before,after=after)

    def on_cache_match(self,original,params,*args,**kwargs):
        result=original(params,*args,**kwargs)
        self.observe(self.record_cache_match, params, result)
        return result

    def record_cache_match(self, params, result):
        req=params.req
        if req is not None:
            row=self.request(req)
            if row['mamba_slot'] is not None and req.rid not in self.active:
                self.pending[req.rid]={k:row[k] for k in ['request_slot','mamba_slot']}
            self.emit('cache_match',stable=True,request=row,matched_tokens=len(result.device_indices),
                      mamba_branching_seqlen=result.mamba_branching_seqlen)

    def restore(self):
        if self.closed:
            return
        restored = True
        for obj,name,wrapped in reversed(self.saved):
            try:
                if obj.__dict__.get(name) is not wrapped:
                    raise ValueError('lifecycle wrapper replaced')
                delattr(obj,name)
            except Exception as error:
                restored = False
                self.fail(error)
        self.saved=[]
        self.closed = True
        # Publish only after restore. Missing, partial or failed receipts cannot
        # pass post-run acceptance. No status writes occur in the forward path.
        try:
            record = {key: self.manifest[key] for key in ('mode',) + IDENTITY}
            record.update({key: self.worker[key] for key in WORKER})
            record.update(schema=1, observer='lifecycle_slots',
                worker_identity_sha256=self.worker['identity_sha256'],
                helper_source=self.manifest['sources']['j01_probe'],
                event_limit=self.event_limit, events_emitted=self.seq,
                observer_failed=self.observer_failed, saturated=self.saturated,
                failure_type=self.failure_type, restored=restored, complete=True)
            path = Path(self.manifest['evidence_dir']) / f'LIFECYCLE-HEALTH-rank-{self.worker["rank"]}.json'
            pending = path.with_suffix('.pending')
            # Publish without overwriting an earlier receipt. On any failure the
            # pending file remains, so exact native membership rejects the run.
            with pending.open('xb') as stream:
                raw = canonical(record)
                if stream.write(raw) != len(raw):
                    raise OSError('short lifecycle health write')
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, path)
            pending.unlink()
        except Exception as error:
            self.fail(error)


def validate_lifecycle_health(record, *, binding, worker):
    """Accept only a complete, healthy, source/worker-bound lifecycle receipt."""
    fields = {'schema', 'observer', 'mode', *IDENTITY, *WORKER,
              'worker_identity_sha256', 'helper_source', 'event_limit',
              'events_emitted', 'observer_failed', 'saturated', 'failure_type',
              'restored', 'complete'}
    need(type(record) is dict and set(record) == fields, 'lifecycle health schema')
    need(type(record['schema']) is int and record['schema'] == 1 and
         record['observer'] == 'lifecycle_slots', 'lifecycle health version')
    need(all(type(record[key]) is type(binding[key]) and record[key] == binding[key]
             for key in ('mode',) + IDENTITY), 'lifecycle health run binding')
    need(all(type(record[key]) is type(worker[key]) and record[key] == worker[key]
             for key in WORKER) and
         record['worker_identity_sha256'] == worker['identity_sha256'] and
         record['helper_source'] == binding['sources']['j01_probe'],
         'lifecycle health worker/source binding')
    need(record['complete'] is True and record['restored'] is True and
         record['observer_failed'] is False and record['saturated'] is False and
         record['failure_type'] is None, 'lifecycle observer incomplete or failed')
    need(type(record['event_limit']) is int and record['event_limit'] == 8192 and
         type(record['events_emitted']) is int and
         0 < record['events_emitted'] <= record['event_limit'],
         'lifecycle health event bound')
    return record


def install_lifecycle_slots(scheduler,manifest,worker):
    return LifecycleSlots(scheduler,manifest,worker)



_target_kv_metadata = _load_observer_helper('target_kv_metadata',
    '8c8ce7ad11e927494fd7fc3c78b14353ae40b6c413fd0ed2a9a945dcc033f74b')

# Target-KV witness; its adjacent metadata validator is pinned above.
_FP8_TARGET_WITNESS_STATE = None

def _target_kv_active_request_group(manifest, worker, state, observed):
    """Only admit an open menu interval; unjournaled warmup consumes no slot.

    The existing offline reader still requires a matching completed after record.
    This cursor reads at most 53 existing journals, each at most two 8192-byte
    records. It creates no files and performs no tensor or device operation.
    """
    import stat
    need(type(observed) in (int, float) and math.isfinite(observed), 'route observation clock')
    while state['request_cursor'] < 53:
        ordinal = state['request_cursor']
        path = Path(manifest['evidence_dir']) / ('request-%02d.jsonl' % ordinal)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return False
        try:
            before_stat = os.fstat(descriptor)
            need(stat.S_ISREG(before_stat.st_mode) and before_stat.st_nlink == 1 and
                 before_stat.st_uid == os.getuid() and 0 < before_stat.st_size <= 16384,
                 'bounded regular request journal')
            raw = os.read(descriptor, 16385)
            after_stat = os.fstat(descriptor)
            fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            need(all(getattr(before_stat, key) == getattr(after_stat, key) for key in fields) and
                 len(raw) == before_stat.st_size, 'stable request journal read')
        finally:
            os.close(descriptor)
        lines = raw.splitlines(keepends=True)
        need(1 <= len(lines) <= 2 and all(line.endswith(b'\n') and len(line) <= 8192 for line in lines),
             'complete bounded request journal records')
        rows = [json.loads(line) for line in lines]
        need(all(canonical(row) == line for row, line in zip(rows, lines, strict=True)),
             'canonical request journal records')
        before = rows[0]
        group = before['group_ordinals']
        need(type(group) is list and 1 <= len(group) <= 2 and
             all(type(value) is int and 0 <= value < 53 for value in group) and
             group == list(range(ordinal, ordinal + len(group))) and
             type(before['group_id']) is str and 0 < len(before['group_id']) <= 128,
             'ordered bounded request group')
        for index, row in enumerate(rows):
            need(row['schema'] == 1 and row['mode'] == manifest['mode'] and row['arm'] == manifest['arm'] and
                 type(row['ordinal']) is int and row['ordinal'] == ordinal and
                 row['phase'] == ('before' if index == 0 else 'after') and
                 row['group_id'] == before['group_id'] and row['group_ordinals'] == group and
                 row['body_sha256'] == before['body_sha256'] and row['case'] == before['case'] and
                 row['instrumented_latency'] is True and row['qualification'] is False,
                 'request journal scope and phase')
            tick = row['observed_monotonic']
            need(type(tick) in (int, float) and math.isfinite(tick) and tick <= observed,
                 'request journal precedes route observation')
            snapshot = row['snapshot']
            ranks = snapshot['ranks']
            need(snapshot['arm'] == manifest['arm'] and len(ranks) == 2 and
                 [item['rank'] for item in ranks] == [0, 1] and
                 ranks[worker['rank']]['pid'] == worker['pid'] and
                 ranks[worker['rank']]['captures'] == 2,
                 'request journal current worker and captured graph')
        if len(rows) == 1:
            return True
        need(before['observed_monotonic'] < rows[1]['observed_monotonic'], 'closed request group chronology')
        state['request_cursor'] = group[-1] + 1
    return False

def install_target_kv_witness(manifest, worker):
    global _FP8_TARGET_WITNESS_STATE
    need(_FP8_TARGET_WITNESS_STATE is None, 'duplicate target KV witness')
    source = manifest['sources']['triton_backend']
    module = __import__('importlib').import_module(source['module'])
    path = Path(module.__file__).resolve()
    need(str(path) == source['path'] and sha(path.read_bytes()) == source['sha256'], 'target KV backend source binding')
    need(getattr(module, '_FP8_TARGET_KV_OBSERVER', None) is None, 'target KV observer already installed')
    health = _health.Health('target_kv', worker['rank'])
    state = dict(stage='startup', seen=set(), emitted_bytes=0, request_cursor=0, records=[])
    _FP8_TARGET_WITNESS_STATE = state
    def observe(backend, q, kb, vb, layer, route_name, capture):
        return health.call(collect, backend, q, kb, vb, layer, route_name, capture)

    def collect(backend, q, kb, vb, layer, route_name, capture):
        key = (state['stage'], route_name, capture, q.shape[0] if capture else 0, layer.layer_id)
        if key in state['seen']:
            return
        if state['stage'] == 'external' and not _target_kv_active_request_group(manifest, worker, state, time.monotonic()):
            return
        if len(state['seen']) >= 128:
            raise _health.Saturated('target KV route record cap')
        evidence = _target_kv_metadata.route(backend, q, kb, vb, layer, route_name=route_name, capture=capture)
        record = dict(schema=1, **{key: manifest[key] for key in IDENTITY},
            **{key: worker[key] for key in WORKER}, worker_identity_sha256=worker['identity_sha256'],
            helper_source=manifest['sources']['j01_probe'], backend_source=source,
            stage=state['stage'], sequence=len(state['seen']), observed_monotonic=time.monotonic(),
            evidence=evidence, scope='host call metadata; captured route is not an external replay observation',
            qualification=False)
        raw = canonical(record)
        if len(raw) > 8192 or state['emitted_bytes'] + len(raw) > 1 << 20:
            raise _health.Saturated('target KV witness bytes')
        state['records'].append(raw)
        state['seen'].add(key)
        state['emitted_bytes'] += len(raw)
    module._FP8_TARGET_KV_OBSERVER = observe
    def restore():
        global _FP8_TARGET_WITNESS_STATE
        if health.closed:
            return
        try:
            need(module._FP8_TARGET_KV_OBSERVER is observe, 'target KV observer replaced')
            module._FP8_TARGET_KV_OBSERVER = None
            if state['records']:
                path = Path(manifest['evidence_dir']) / f'TARGET-KV-ROUTES-rank-{worker["rank"]}.jsonl'
                with path.open('xb') as stream:
                    for raw in state['records']:
                        if stream.write(raw) != len(raw):
                            raise OSError('short target KV route write')
                    stream.flush()
                    os.fsync(stream.fileno())
                for raw in state['records']:
                    print('SPRY_TARGET_KV_ROUTE ' + raw.decode().strip(), flush=True)
        except Exception as error:
            health.fail(error)
        finally:
            _FP8_TARGET_WITNESS_STATE = None
            health.finish(manifest, worker, manifest['sources']['j01_probe'],
                          dict(routes=len(state['seen']), bytes=state['emitted_bytes']))
    return restore


def validate_target_kv_health(record, *, binding, worker):
    return _health.validate(record, binding=binding, worker=worker, observer='target_kv',
        helper_source=binding['sources']['j01_probe'], limits=dict(routes=128, bytes=1 << 20))
