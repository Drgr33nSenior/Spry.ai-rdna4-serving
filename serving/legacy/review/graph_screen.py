"""Exploratory native graph observations; no runtime imports or launch entry point."""

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


MODE = 'native-model-context-retrieval-32k-v1'
ARMS = {'eager': 'disabled', 'graph': 'full'}
REPORT_BYTES = 256 << 10
CELLS = (('ctx16384-seed1729-depth50', 1, 16352), ('ctx32768-seed1729-depth10', 1, 32736), ('ctx32768-seed1729-depth50', 1, 32736), ('ctx32768-seed1729-depth90', 1, 32736), ('ctx32768-seed2718-depth10', 1, 32736), ('ctx32768-seed2718-depth50', 1, 32735), ('ctx32768-seed2718-depth90', 1, 32736))
NATIVE = 'test-results/uq-27b-gpu-ready.FPkX5N/source/sglang/python/sglang/'
RECOVERED = 'scratchpad/fp8-recovered-debug.XVyDRmJJ/review/'
SOURCE_PINS = {
    NATIVE + 'benchmark/one_batch.py': '451dec50e63144d72e5c7d73a4c2bbeafafeaa426d6304bdfc608e64a5425d8e',
    NATIVE + 'srt/model_executor/cuda_graph_config.py': 'a7e16d34042149bf488d62dcb0240a17a7ddefa025b6bf5c9cfabc7bcab00b91',
    NATIVE + 'srt/arg_groups/cuda_graph_hook.py': 'dee58c77ffd0cb28107f23ef6f7921db187bf3dbcef8ee8d803c9546952b57dc',
    NATIVE + 'srt/server_args.py': '1dc6d150bf7c4ef5c2316c2ce21ca28ea557ac26fcdb150d21ac48fff1a784e6',
    NATIVE + 'srt/model_executor/runner_backend/full_cuda_graph_backend.py': 'fcfca3c5d286a77594742fd05791fe5ad4c7fd46327ab7f126b2cdcf697b685d',
    RECOVERED + 'model_benchmark.py': '3d714cbbc7fc5c3c2b9f59081b605f49b8a43eb2b2815d72f9583af28508c674',
    RECOVERED + 'host_process_owner.py': 'b200f9db9c0a6f9b61e0fd903487803ea5d0812ec530725f48ad224c3cd1356f',
    RECOVERED + 'benchmark_host_transport.py': '0a30d8e215f925e8b9e98caa626125b8b2b0fe3741c97d65321952aee3f0877a',
    RECOVERED + 'benchmark_owner_adapter.py': '22f3c37b2d2c13df140148de659f1e1c552c58c4f932916ebc1e5422374bcf0e',
    RECOVERED + 'benchmark_resource_gate.py': '418bf2d020547246ffc7a88571599588590b48bfe328219102d1f9a8c176b15f',
    RECOVERED + 'benchmark_physical_controller.py': '1e9876974dcff8b35535f2587333bf6ec993e7be0390fea20fe8ecaa0773b045',
}
_FORWARD = None
_CAPTURE = None


def need(value, message):
    if not value:
        raise ValueError('graph potential: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def source_inputs(repo_root):
    """Authenticate retained sources, without importing native modules."""
    root, bodies = Path(repo_root), {}
    for name, pin in SOURCE_PINS.items():
        path = root / name
        before = path.lstat()
        need(path.is_file() and not path.is_symlink() and before.st_nlink == 1 and
             before.st_size <= 2 << 20, 'source type/size: ' + name)
        raw = path.read_bytes()
        after = path.lstat()
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        need(stamp(before) == stamp(after) and hashlib.sha256(raw).hexdigest() == pin,
             'source pin/change: ' + name)
        bodies[name] = raw
    return bodies


def projected_server_args(baseline, arm):
    """Only decode backend differs between arms; both use native BF16 KV."""
    need(arm in ARMS and type(baseline) is dict, 'explicit arm')
    required = dict(attention_backend='triton', prefill_attention_backend='triton',
        decode_attention_backend='triton', quantization='fp8', fp8_gemm_runner_backend='triton',
        dtype='bfloat16', kv_cache_dtype='auto', tp_size=2, pp_size=1, dp_size=1,
        max_running_requests=2, page_size=1, context_length=32768, max_total_tokens=32768,
        mem_fraction_static=0.8, disable_overlap_schedule=True, disable_custom_all_reduce=True,
        mamba_radix_cache_strategy='no_buffer', speculative_algorithm=None,
        cuda_graph_backend_decode='disabled', cuda_graph_backend_prefill='disabled')
    need(all(type(baseline.get(k)) is type(v) and baseline.get(k) == v for k, v in required.items()),
         'native baseline settings')
    for key in ('cuda_graph_config', 'cuda_graph_bs_decode', 'cuda_graph_max_bs_decode'):
        need(baseline.get(key) is None, 'preexisting graph override: ' + key)
    for key in ('disable_cuda_graph', 'disable_decode_cuda_graph', 'disable_prefill_cuda_graph'):
        need(baseline.get(key) in (None, False), 'legacy graph override: ' + key)
    result = copy.deepcopy(baseline)
    result.update(cuda_graph_backend_decode=ARMS[arm], cuda_graph_backend_prefill='disabled',
                  cuda_graph_bs_decode=[1, 2], cuda_graph_max_bs_decode=2)
    return result


def validate_resolved(arm, get_exec):
    need(arm in ARMS, 'resolved arm')
    config = get_exec().graph.cuda_graph_config
    need(config.decode.backend == ARMS[arm] and config.prefill.backend == 'disabled' and
         type(config.decode.bs) is list and config.decode.bs == [1, 2] and
         all(type(value) is int for value in config.decode.bs) and
         type(config.decode.max_bs) is int and config.decode.max_bs == 2, 'resolved graph configuration')
    return dict(decode_backend=config.decode.backend, decode_bs=list(config.decode.bs),
                decode_max_bs=2, prefill_backend='disabled')


def expected_context(index):
    need(type(index) is int and 0 <= index < 224, 'fixed context forward schedule bound')
    ordinal, step = divmod(index, 32)
    cell, batch, _ = CELLS[ordinal]
    return dict(ordinal=ordinal, cell=cell, sample=0, step=step, batch=batch, decode=step > 0)


def _backend(backend, shape_key):
    cls = type(backend)
    need(cls.__name__ == 'FullCudaGraphBackend' and cls.__module__ ==
         'sglang.srt.model_executor.runner_backend.full_cuda_graph_backend', 'native backend class')
    size = getattr(shape_key, 'size', None)
    need(type(size) is int and size in (1, 2), 'native capture/replay shape')
    return size


class Observer:
    def __init__(self, arm, rank, resolved, session_sha256, spec_sha256):
        need(arm in ARMS and type(rank) is int and rank in (0, 1), 'arm/rank')
        for value in (session_sha256, spec_sha256):
            need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                 'session/spec hash')
        need(resolved == dict(decode_backend=ARMS[arm], decode_bs=[1, 2], decode_max_bs=2,
                             prefill_backend='disabled'), 'validated settings')
        self.arm, self.rank = arm, rank
        self.session_sha256, self.spec_sha256 = session_sha256, spec_sha256
        self.resolved = copy.deepcopy(resolved)
        self.records, self.capture_records, self.batches = [], [], []
        self.failure = None

    @contextmanager
    def initializing_graphs(self):
        global _CAPTURE
        need(_CAPTURE is None and _FORWARD is None and not self.records, 'graph initialization scope')
        _CAPTURE = self
        try:
            yield
        except BaseException as exc:
            self.failure = dict(type=type(exc).__name__, message=str(exc)[:1024])
            raise
        finally:
            _CAPTURE = None

    def capture_forward(self, backend, shape_key, phase, warmup_step=None):
        need(_CAPTURE is self and self.arm == 'graph' and len(self.capture_records) < 6, 'capture scope/count')
        need((phase == 'warmup' and type(warmup_step) is int and warmup_step in (0, 1)) or
             (phase == 'capture' and warmup_step is None), 'capture phase')
        self.capture_records.append(dict(size=_backend(backend, shape_key), phase=phase, warmup_step=warmup_step))

    @contextmanager
    def forward(self, fb):
        global _FORWARD
        need(_FORWARD is None and _CAPTURE is None and self.failure is None, 'forward scope')
        row = expected_context(len(self.records))
        need(type(fb.batch_size) is int and fb.batch_size == row['batch'] and
             bool(fb.forward_mode.is_decode()) is row['decode'], 'actual forward schedule')
        row.update(replays=[], result_observed=False, can_run_graph=None)
        self.records.append(row)
        _FORWARD = (self, row)
        try:
            yield
            need(row['result_observed'], 'missing native result')
        except BaseException as exc:
            self.failure = dict(type=type(exc).__name__, message=str(exc)[:1024])
            raise
        finally:
            _FORWARD = None

    def backend_replayed(self, backend, shape_key, static_forward_batch):
        need(_FORWARD is not None and _FORWARD[0] is self, 'replay outside current forward')
        row = _FORWARD[1]
        size = _backend(backend, shape_key)
        need(self.arm == 'graph' and row['decode'] and not row['replays'] and
             size == row['batch'] and static_forward_batch.forward_mode.is_decode(), 'actual replay count/phase')
        row['replays'].append(size)

    def result(self, output):
        need(_FORWARD is not None and _FORWARD[0] is self, 'result scope')
        row = _FORWARD[1]
        expected = self.arm == 'graph' and row['decode']
        flag = getattr(output, 'can_run_graph', None)
        need(type(flag) is bool and flag is expected and not row['result_observed'] and
             row['replays'] == ([row['batch']] if expected else []), 'native flag/replay join')
        row.update(result_observed=True, can_run_graph=flag)
        return output

    def complete_batch(self, ordinal, generated_ids, generated_logprobs, step_seconds, prompt_sha256):
        need(type(ordinal) is int and ordinal == len(self.batches) and 0 <= ordinal < 7 and
             len(self.records) == (ordinal + 1) * 32, 'batch schedule/forward join')
        _, count, _ = CELLS[ordinal]
        row = dict(ordinal=ordinal, generated_ids=copy.deepcopy(generated_ids),
            generated_logprobs=copy.deepcopy(generated_logprobs), step_seconds=list(step_seconds),
            prompt_sha256=list(prompt_sha256))
        _validate_batch(row, ordinal, count)
        self.batches.append(row)

    def report(self, require_complete=True):
        result = dict(schema=1, mode=MODE, arm=self.arm, rank=self.rank,
            session_sha256=self.session_sha256, spec_sha256=self.spec_sha256,
            resolved=copy.deepcopy(self.resolved), source_pins=dict(SOURCE_PINS),
            records=copy.deepcopy(self.records), capture_forwards=copy.deepcopy(self.capture_records),
            batches=copy.deepcopy(self.batches), failure=copy.deepcopy(self.failure),
            complete=self.failure is None and len(self.records) == 224 and len(self.batches) == 7,
            qualification=False, source_promotion=False, intermediate_boundaries_observed=False,
            timing_scope='host-synchronized native forward+sample; not HTTP/serving throughput')
        validate_report(result, require_complete=require_complete)
        return result


def _validate_batch(row, ordinal, count):
    need(type(row) is dict and set(row) == {'ordinal', 'generated_ids', 'generated_logprobs',
        'step_seconds', 'prompt_sha256'} and type(row['ordinal']) is int and row['ordinal'] == ordinal,
        'batch fields/order')
    need(type(row['prompt_sha256']) is list and len(row['prompt_sha256']) == count and
         all(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
             for value in row['prompt_sha256']), 'prompt identities')
    need(type(row['generated_ids']) is list and type(row['generated_logprobs']) is list and
         len(row['generated_ids']) == len(row['generated_logprobs']) == count, 'response count')
    for ids, scores in zip(row['generated_ids'], row['generated_logprobs']):
        need(type(ids) is list and type(scores) is list and len(ids) == len(scores) == 32 and
             all(type(value) is int and 0 <= value < 248320 for value in ids) and
             all(type(value) in (int, float) and math.isfinite(value) for value in scores), 'finite output shape/range')
    need(type(row['step_seconds']) is list and len(row['step_seconds']) == 32 and
         all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in row['step_seconds']),
         'finite positive host timings')


def validate_report(report, require_complete=True):
    fields = {'schema', 'mode', 'arm', 'rank', 'session_sha256', 'spec_sha256', 'resolved', 'source_pins',
              'records', 'capture_forwards', 'batches', 'failure', 'complete', 'qualification',
              'source_promotion', 'intermediate_boundaries_observed', 'timing_scope'}
    need(type(report) is dict and set(report) == fields and type(report.get('schema')) is int and
         report.get('schema') == 1 and report.get('mode') == MODE and
         report.get('arm') in ARMS and type(report.get('rank')) is int and report['rank'] in (0, 1) and
         report.get('qualification') is False and report.get('source_promotion') is False and
         report.get('intermediate_boundaries_observed') is False and report.get('source_pins') == SOURCE_PINS,
         'exploratory source/mode/claim binding')
    for field in ('session_sha256', 'spec_sha256'):
        value = report.get(field)
        need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), field)
    config = report.get('resolved')
    need(type(config) is dict and config == dict(decode_backend=ARMS[report['arm']], decode_bs=[1, 2],
        decode_max_bs=2, prefill_backend='disabled') and type(config['decode_bs']) is list and
        all(type(value) is int for value in config['decode_bs']) and type(config['decode_max_bs']) is int,
        'report resolved configuration')
    need(report['failure'] is None or (type(report['failure']) is dict and
         set(report['failure']) == {'type', 'message'} and type(report['failure']['type']) is str and
         type(report['failure']['message']) is str and len(report['failure']['message']) <= 1024),
         'bounded partial failure')
    rows, batches, captures = report.get('records'), report.get('batches'), report.get('capture_forwards')
    need(type(rows) is list and len(rows) <= 224 and type(batches) is list and len(batches) <= 7 and
         type(captures) is list and len(captures) <= 6, 'bounded report lists')
    for index, row in enumerate(rows):
        context = expected_context(index)
        expected = report['arm'] == 'graph' and context['decode']
        need(type(row) is dict and set(row) == set(context) | {'replays', 'result_observed', 'can_run_graph'} and
             all(type(row[k]) is type(v) and row[k] == v for k, v in context.items()) and
             type(row['replays']) is list and all(type(value) is int for value in row['replays']), 'row schedule')
        if row['result_observed'] is True:
            need(row['can_run_graph'] is expected and row['replays'] == ([context['batch']] if expected else []),
                 'row native graph coverage')
        else:
            need(row['result_observed'] is False and index == len(rows) - 1 and report.get('complete') is False,
                 'only last partial forward may lack result')
    for ordinal, row in enumerate(batches):
        _validate_batch(row, ordinal, CELLS[ordinal][1])
        need(len(rows) >= (ordinal + 1) * 32 and all(r['result_observed'] for r in rows[ordinal*32:(ordinal+1)*32]),
             'complete batch actual forward coverage')
    expected_captures = [dict(size=size, phase=phase, warmup_step=step)
        for size in (1, 2) for phase, step in (('warmup', 0), ('warmup', 1), ('capture', None))]
    for row in captures:
        need(type(row) is dict and any(set(row) == set(expected) and
             all(type(row[key]) is type(value) and row[key] == value for key, value in expected.items())
             for expected in expected_captures), 'capture phase/shape')
    need(len({canonical(row) for row in captures}) == len(captures), 'duplicate capture observation')
    need(report['arm'] == 'graph' or not captures, 'eager captured graph')
    computed = report.get('failure') is None and len(rows) == 224 and len(batches) == 7
    need(type(report.get('complete')) is bool and report['complete'] is computed, 'truthful completeness')
    if computed:
        need(all(row['result_observed'] for row in rows) and
             (report['arm'] == 'eager' or sorted(map(canonical, captures)) == sorted(map(canonical, expected_captures))),
             'complete capture/replay coverage')
    need(not require_complete or computed, 'incomplete exploratory observations')
    need(len(canonical(report)) <= REPORT_BYTES, 'report byte cap')
    return report


def compare_reports(eager, graph):
    """Finite output drift is retained as data, never converted to equivalence."""
    for value in (eager, graph):
        validate_report(value)
    need(eager['arm'] == 'eager' and graph['arm'] == 'graph' and
         all(eager[key] == graph[key] for key in ('rank', 'session_sha256', 'spec_sha256', 'source_pins')),
         'paired source/session/rank binding')
    comparisons, cells = [], []
    for before, after in zip(eager['batches'], graph['batches']):
        need(before['prompt_sha256'] == after['prompt_sha256'], 'identical prompt conditioning')
        for request, (left_ids, right_ids, left_scores, right_scores) in enumerate(zip(
                before['generated_ids'], after['generated_ids'], before['generated_logprobs'], after['generated_logprobs'])):
            aligned = next((i for i, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]), 32)
            gaps = [abs(left_scores[i] - right_scores[i]) for i in range(aligned)]
            comparisons.append(dict(ordinal=before['ordinal'], request=request, ids_equal=aligned == 32,
                first_different_token=None if aligned == 32 else aligned, aligned_score_count=aligned,
                aligned_scores_equal=all(value == 0 for value in gaps), max_aligned_score_gap=max(gaps, default=None)))
    for cell_index, (cell, _, _) in enumerate(CELLS):
        medians = []
        for report in (eager, graph):
            durations = [seconds for row in report['batches'][cell_index:cell_index+1]
                         for seconds in row['step_seconds'][1:]]
            medians.append(statistics.median(durations))
        cells.append(dict(cell=cell, measured_decode_windows=31, eager_median_seconds=medians[0],
                          graph_median_seconds=medians[1], eager_over_graph=medians[0]/medians[1]))
    return dict(mode=MODE, status='EXPLORATORY_OBSERVATIONS_ONLY', qualification=False,
        source_promotion=False, rank=eager['rank'], comparisons=comparisons, cells=cells,
        exact_output_equivalence=all(row['ids_equal'] and row['aligned_scores_equal'] for row in comparisons),
        drift_policy='Retain every finite difference; no quality, qualification or observer-free speedup claim.')


def observe_step(runner, batch, forward_batch, torch, dist, group, observer, timed_forward, checked_observation):
    """Materialize final IDs/scores before a later replay overwrites static outputs."""
    need(getattr(runner, '_fp8_decode_graph_observer', None) is observer, 'step observer belongs to runner')
    logits, sampled, seconds, fb = timed_forward(runner, batch, forward_batch, torch)
    tokens, scores = checked_observation(logits, sampled, fb, fb.batch_size, 248320, torch, dist, group)
    return list(tokens), list(scores), seconds, fb
