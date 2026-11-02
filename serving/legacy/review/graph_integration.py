"""Explicit exploratory worker/readout integration; no native imports at module load."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path


MODE = 'native-model-context-retrieval-32k-v1'
ARMS = ('bf16', 'kv4')  # Unchanged owner keys/order, never cache formats.
ARM_MAP_SHA256 = '53387c2d9776ec21f3d6aeff48767f75a4abbf159ebb348b83a00871432a33c1'
SCREEN_PIN = '9b5a1431b3d35770b1103a40212353eaea534d50a4b84d5f8be16aac7d56c8be'
CONFIG_PIN = '74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc'
STOCK_PIN = '3d9d603b39f0aa297711bf5abca379933dcbccc7ec060a282404af39f7d40384'


def need(value, message):
    if not value:
        raise ValueError('graph integration: ' + message)


def load_screen():
    path = Path(__file__).with_name('graph_screen.py')
    raw = path.read_bytes()
    need(not path.is_symlink() and len(raw) < 256 << 10 and hashlib.sha256(raw).hexdigest() == SCREEN_PIN,
         'exact adjacent tested graph helper')
    spec = importlib.util.spec_from_file_location('private_graph_potential_helper', path)
    spec.loader.get_code = lambda fullname: spec.loader.source_to_code(raw, str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


screen = load_screen()


def logical_arms(spec):
    mapping = spec.get('graph_arm_map')
    need(type(mapping) is dict and set(mapping) == set(ARMS) and
         all(type(value) is str for value in mapping.values()) and
         set(mapping.values()) == {'eager', 'graph'} and
         hashlib.sha256(screen.canonical(mapping)).hexdigest() == ARM_MAP_SHA256,
         'exact frozen reverse owner-to-logical arm map')
    return dict(mapping)


def logical_arm(spec, arm):
    need(arm in ARMS, 'outer arm')
    return logical_arms(spec)[arm]


def validate_spec(spec):
    need(type(spec) is dict and spec.get('experiment_mode') == MODE and spec.get('schema') == 1 and
         spec.get('scope') == 'reviewed-private-tp2-model-benchmark-v1' and spec.get('qualification') is False and
         spec.get('source_promotion') is False and spec.get('gpu_ids') == [0, 1], 'explicit exploratory mode')
    need(not any(key in spec for key in ('validation_comparison', 'callback_profile', 'stall_trace', 'cache_profile')),
         'qualification/capture contract not admitted in exploratory mode')
    need(spec.get('context_length') == spec.get('physical_slots') == 32768 and spec.get('outputs') == 32 and
         spec.get('warmup_batches') == 0 and spec.get('measured_batches') == 1 and
         type(spec.get('worker_seconds')) is int and 60 <= spec['worker_seconds'] <= 1200, 'fixed bounded workload')
    need(spec.get('cells') == [dict(name=name, concurrency=count, input_tokens=length)
                              for name, count, length in screen.CELLS], 'fixed five cells')
    for arm in screen.ARMS:
        screen.projected_server_args(spec['server_args'], arm)
    prompts, pins = spec.get('frozen_prompt_ids'), spec.get('frozen_prompt_sha256')
    need(type(prompts) is dict and type(pins) is dict and set(prompts) == set(pins) ==
         {cell[0] for cell in screen.CELLS}, 'frozen prompt inventory')
    for cell, count, length in screen.CELLS:
        need(type(prompts[cell]) is list and type(pins[cell]) is list and len(prompts[cell]) == len(pins[cell]) == count,
             'prompt count')
        for ids, pin in zip(prompts[cell], pins[cell]):
            need(type(ids) is list and len(ids) == length and
                 all(type(token) is int and 0 <= token < 248320 for token in ids) and
                 hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == pin,
                 'prompt token identity')
    logical_arms(spec)
    reader = spec.get('attention_reader')
    need(type(reader) is dict and reader.get('mode') == 'native-bf16-kv-cache-v1' and
         reader.get('compact_reader') is False and reader.get('cache_dtype') == 'torch.bfloat16' and
         reader.get('pool', {}).get('class') == 'MHATokenToKVPool' and
         reader.get('backend', {}).get('class') == 'TritonAttnBackend', 'both arms require native BF16 storage')
    runtime = spec.get('graph_runtime_sources')
    need(type(runtime) is dict and set(runtime) == {'backend', 'model_runner', 'pool', 'attention'} and
         all(type(row) is dict and set(row) == {'path', 'sha256'} and type(row['path']) is str and
             row['path'].startswith('/mnt/composition/image-source/python/sglang/') and
             type(row['sha256']) is str and len(row['sha256']) == 64 for row in runtime.values()),
         'explicit private native source identities')
    need(spec.get('graph_screen_sha256') == SCREEN_PIN and spec.get('graph_report_bytes') == screen.REPORT_BYTES,
         'tested graph helper/report bound')
    context_validate_spec(spec)
    return spec


def preflight(namespace, spec, arm=None, capture_observations=None):
    validate_spec(spec)
    namespace['_GRAPH_OUTER'].validate_projection(spec['graph_projection'])
    need(arm in (None, *ARMS), 'outer arm')
    checked = namespace['checked_file']
    config = checked(spec['config_path'], CONFIG_PIN)
    checked(spec['stack_receipt'], spec['stack_receipt_sha256'])
    proof = json.loads(checked(Path(namespace['__file__']).with_name('STOCK-BUILD-PROOF.json'), STOCK_PIN))
    need(proof.get('schema') == 2 and proof.get('serving_composition_complete') is True and
         proof.get('qualification') == 'build-reviewed-not-serving-qualified' and
         proof.get('stack_receipt_sha256') == spec['stack_receipt_sha256'], 'original closed build provenance')
    for row in spec['graph_runtime_sources'].values():
        checked(row['path'], row['sha256'])
    context_source_preflight(namespace, spec)
    return config


def attach(namespace, runner, spec, arm, rank, session, spec_hash, get_exec):
    validate_spec(spec)
    logical = logical_arm(spec, arm)
    resolved = screen.validate_resolved(logical, get_exec)
    observer = screen.Observer(logical, rank, resolved, session, spec_hash)
    need(not hasattr(runner, '_fp8_decode_graph_observer'), 'observer already attached')
    runner._fp8_decode_graph_observer = observer
    return namespace['_GRAPH_OUTER'].bind_backend(namespace, spec, observer)


def validate_cache_identity(value, spec, arm, rank, session, spec_hash, pid):
    need(type(value) is dict and value.get('schema') == 1 and value.get('mode') == MODE and
         value.get('arm') == arm and value.get('rank') == rank and value.get('pid') == pid and
         value.get('session_sha256') == session and value.get('spec_sha256') == spec_hash and
         value.get('reader') == spec['attention_reader'], 'cache/native worker binding')
    expected = dict(before_requests=True, after_weights=True, uq_bf16_request=None, compact_owner_present=False,
        pool_class='MHATokenToKVPool', pool_module='sglang.srt.mem_cache.memory_pool',
        key_dtype='torch.bfloat16', value_dtype='torch.bfloat16', cache_dtype='torch.bfloat16',
        store_dtype='torch.bfloat16', buffer_count=16, backend_class='TritonAttnBackend',
        backend_module='sglang.srt.layers.attention.triton_backend', backend_pool_is_runner_pool=True,
        compact_reader_selected=False)
    need(all(type(value.get(key)) is type(item) and value.get(key) == item for key, item in expected.items()),
         'actual native BF16 cache class/ownership')
    for key in ('k_buffers', 'v_buffers'):
        rows = value.get(key)
        need(type(rows) is list and len(rows) == 16 and all(type(row) is dict and
             row.get('dtype') == 'torch.bfloat16' and type(row.get('shape')) is list and
             len(row['shape']) == 3 and row['shape'][1:] == [2, 256] and
             type(row['shape'][0]) is int and row['shape'][0] >= 32768 for row in rows), 'native KV buffer geometry')
    return value


def _read(namespace, path, maximum):
    raw = namespace['read_bounded_file'](path, maximum)
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite graph JSON')))


def read_arm(namespace, spec, directory, inner, require_complete=True):
    validate_spec(spec)
    arm, session, spec_hash = inner['arm'], inner['session_sha256'], inner['spec_sha256']
    need(arm in ARMS and inner.get('experiment_mode') == MODE and inner.get('qualification') is False,
         'inner exploratory identity')
    rows = inner.get('ranks')
    need(type(rows) is list and len(rows) == 2, 'exact two native rank records')
    reports = []
    for rank, row in enumerate(rows):
        need(type(row) is dict and row.get('rank') == rank and row.get('arm') == arm, 'rank order/arm')
        need(_read(namespace, Path(directory)/f'rank-{rank}.json', 8 << 20) == row, 'durable rank/inner record identity')
        identity = _read(namespace, Path(directory)/f'rank-{rank}-identity.json', 64 << 10)
        namespace['validate_rank_identity'](identity, rank=rank, arm=arm, session_sha256=session, spec_sha256=spec_hash)
        need(row.get('native_identity') == identity, 'rank/native identity join')
        cache = _read(namespace, Path(directory)/f'rank-{rank}-native-cache.json', 8192)
        validate_cache_identity(cache, spec, arm, rank, session, spec_hash, identity['process']['pid'])
        need(row.get('native_cache_identity') == cache, 'rank/durable cache identity join')
        report = _read(namespace, Path(directory)/f'rank-{rank}-graph-potential.json', screen.REPORT_BYTES)
        screen.validate_report(report, require_complete=require_complete)
        need(report['arm'] == logical_arm(spec, arm) and report['rank'] == rank and report['session_sha256'] == session and
             report['spec_sha256'] == spec_hash and row.get('graph_potential') == report,
             'rank/durable graph identity join')
        need(len(report['batches']) == len(row.get('batches', [])), 'graph/rank batch count')
        for actual, observed in zip(row['batches'], report['batches']):
            need(observed['prompt_sha256'] == spec['frozen_prompt_sha256'][actual['cell']] and
                 all(observed[key] == actual[key] for key in ('ordinal', 'generated_ids', 'generated_logprobs', 'step_seconds')),
                 'exact final output/timing/frozen-prompt join')
        reports.append(report)
    context_read_rows(rows)
    namespace['_GRAPH_OUTER'].validate_memory(spec, rows, logical_arm(spec, arm))
    samples = namespace['summarize'](rows, arm)
    return dict(complete=True, qualification=False, source_promotion=False,
                reports=reports, samples=samples, interpretation='Exploratory final-output observations; no intermediate qualification')


def publish(namespace, observer, result, output, rank):
    if observer is None:
        return
    result['graph_backend_identity'] = observer.native_backend_identity
    if result['failure'] is not None or result['cleanup_errors']:
        failure = result['failure'] or dict(type='CleanupError', message=str(result['cleanup_errors']))
        observer.failure = dict(type=str(failure['type']), message=str(failure['message'])[:1024])
    report = observer.report(require_complete=result['failure'] is None and not result['cleanup_errors'])
    result['graph_potential'] = report
    namespace['write_result'](Path(output)/f'rank-{rank}-graph-potential.json', report)


def analyze(namespace, spec, spec_hash, left, right, directories):
    need(type(directories) is dict and set(directories) == set(ARMS), 'durable graph directories required')
    arms = {arm: read_arm(namespace, spec, directories[arm], value) for arm, value in (('bf16', left), ('kv4', right))}
    for arm, value in (('bf16', left), ('kv4', right)):
        need(value.get('status') == 'MODEL_BENCHMARK_COMPLETE' and value.get('spec_sha256') == spec_hash and
             value.get('parent_error') is None and value.get('cleanup_errors') == [] and
             value.get('exits') == [0, 0] and value.get('groups_observed_absent') == [True, True] and
             value.get('timed_out') is False and value.get('samples') == arms[arm]['samples'],
             'original inner completion/cleanup required')
    owner_by_logical = {logical: owner for owner, logical in logical_arms(spec).items()}
    return dict(mode=MODE, status='EXPLORATORY_CONTEXT_RETRIEVAL_COMPLETE', qualification=False,
        owner_arm_map=logical_arms(spec), owner_arm_order=list(ARMS), logical_execution_order=[logical_arm(spec, arm) for arm in ARMS],
        source_promotion=False, intermediate_boundaries_observed=False,
        ranks=[screen.compare_reports(arms[owner_by_logical['eager']]['reports'][rank], arms[owner_by_logical['graph']]['reports'][rank]) for rank in (0, 1)],
        timing_scope='host-synchronized forward+sample, not serving throughput',
        retrieval={arm: context_grade_summary(value['ranks']) for arm, value in (('bf16', left), ('kv4', right))},
        native_admission=False, independent_release_required=True)


"""Fixed exploratory context contract; standard-library checks only.

This source is not a native entry point. The candidate builder will bind the
actual native tokenizer report before copying these checks into its private
integration module. No candidate is ready while that binding is absent.
"""

import copy
import hashlib
import json
import math


MODE = 'native-model-context-retrieval-32k-v1'
# The preserved 16K control is deliberately first.  The remaining six rows
# are the only new 32K work: two deterministic seeds at three key depths.
CASE_PLAN = ((16352, 1729, 50),
             (32736, 1729, 10), (32736, 1729, 50), (32736, 1729, 90),
             (32736, 2718, 10), (32736, 2718, 50), (32736, 2718, 90))
OUTPUTS = 32
CASES = 7
WORKLOAD_FORWARDS = CASES * OUTPUTS
WORKLOAD_REPLAYS = CASES * (OUTPUTS - 1)
CONTEXT_LENGTH = 32768
VOCAB_SIZE = 248320
PREFILL_POLICY = dict(before_free_min_bytes=2 << 30,
                      after_free_min_bytes=1 << 30,
                      hard_transient_peak_quota=False)
DECODE_KWARGS = dict(skip_special_tokens=False,
                     clean_up_tokenization_spaces=False)


def require(value, message):
    if not value:
        raise ValueError('context retrieval: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode() + b'\n'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_digest(value):
    return (type(value) is str and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def validate_native_prompts(report):
    """Validate the fixed native input data, not a synthetic tokenizer fixture."""
    require(type(report) is dict and report.get('schema') == 1 and
            type(report.get('schema')) is int and
            report.get('status') == 'NATIVE_TOKENIZER_PROMPTS_FROZEN' and
            report.get('native_tokenization_performed') is True and
            report.get('gpu_execution_performed') is False and
            report.get('model_loaded') is False and
            report.get('qualification') is False and
            report.get('source_promotion') is False and
            report.get('case_count') == CASES and
            type(report.get('case_count')) is int,
            'complete native tokenizer provenance required')
    identity = report.get('tokenizer_identity')
    require(type(identity) is dict and
            identity.get('factory') == 'transformers.AutoTokenizer.from_pretrained' and
            identity.get('factory_kwargs') == dict(local_files_only=True,
                trust_remote_code=False, use_fast=True), 'native tokenizer factory')
    termination = identity.get('native_termination')
    require(type(termination) is dict and
            termination.get('generation_stop_ids') == [248046, 248044] and
            termination.get('assistant_end_of_turn_ids') == [248046] and
            termination.get('assistant_end_of_turn_text') == '<|im_end|>',
            'authenticated generation stops and assistant end of turn')
    rows = report.get('prompts')
    require(type(rows) is list and len(rows) == CASES,
            'exact seven native prompt records')
    expected = CASE_PLAN
    names, answers = set(), set()
    for row, (target, seed, depth) in zip(rows, expected):
        require(type(row) is dict and
                (row.get('target_input_tokens'), row.get('seed'),
                 row.get('requested_depth_percent')) == (target, seed, depth) and
                all(type(row.get(key)) is int for key in
                    ('target_input_tokens', 'seed', 'requested_depth_percent')),
                'fixed context/seed/depth order')
        name = f'ctx{target + OUTPUTS}-seed{seed}-depth{depth}'
        require(row.get('case') == name and name not in names,
                'exact unique native case identity')
        names.add(name)
        ids = row.get('input_ids')
        require(type(ids) is list and target - 16 <= len(ids) <= target and
                all(type(token) is int and 0 <= token < VOCAB_SIZE for token in ids) and
                row.get('input_tokens') == len(ids) and
                type(row.get('input_tokens')) is int and
                row.get('input_ids_sha256') == digest(canonical(ids)),
                'native full-chat IDs and occupied input band')
        key = row.get('expected_answer')
        require(type(key) is str and 0 < len(key.encode()) <= 256 and
                key not in answers and row.get('expected_answer_sha256') ==
                digest(key.encode()) and row.get('expected_answer_occurrences') == 1 and
                type(row.get('expected_answer_occurrences')) is int,
                'unique literal answer identity')
        answers.add(key)
        for field in ('body', 'rendered_text', 'roundtrip_text'):
            text = row.get(field)
            require(type(text) is str and len(text.encode()) <= 256 << 10 and
                    text.count(key) == 1, 'bounded complete prompt text: ' + field)
        require(row['rendered_text'] == row['roundtrip_text'] and
                row.get('body_sha256') == digest(row['body'].encode()) and
                row.get('rendered_sha256') == digest(row['rendered_text'].encode()) and
                row.get('roundtrip_sha256') == digest(row['roundtrip_text'].encode()),
                'native full-chat render/decode roundtrip hashes')
        require(row.get('messages') == [dict(role='user', content=row['body'])],
                'complete original user message')
        span = row.get('key_token_span')
        require(type(span) is list and len(span) == 2 and
                all(type(value) is int for value in span) and
                0 <= span[0] < span[1] <= len(ids) and
                row.get('key_depth_denominator') ==
                'complete template-produced input token count' and
                type(row.get('actual_key_depth_percent')) in (int, float) and
                math.isfinite(row['actual_key_depth_percent']) and
                row['actual_key_depth_percent'] == 100 * span[0] / len(ids) and
                abs(row['actual_key_depth_percent'] - depth) <= 2,
                'observed final-chat key-token depth')
        require(row.get('reserved_output_tokens') == OUTPUTS and
                len(ids) + OUTPUTS <= CONTEXT_LENGTH and
                row.get('occupancy_band') == [target - 16, target],
                'unchunked occupied context/output reservation')
        if (target, seed, depth) == CASE_PLAN[0]:
            control = report['contract'].get('preserved_control')
            require(type(control) is dict and
                    row.get('case') == control.get('case') and
                    row.get('body_sha256') == control.get('body_sha256') and
                    row.get('input_ids_sha256') == control.get('input_ids_sha256'),
                    'unchanged reobserved 16K control body and IDs')
        options = row.get('apply_chat_template_kwargs')
        require(type(options) is dict and options.get('add_generation_prompt') is True and
                options.get('continue_final_message') is False and
                options.get('enable_thinking') is False and
                options.get('return_dict') is False,
                'explicit native chat options without keyword fallback')
        answer = row.get('expected_answer_output')
        require(type(answer) is dict and
                answer.get('native_end_of_turn_ids') == [248046] and
                answer.get('output_budget_tokens') == OUTPUTS,
                'source-bound standalone answer/EOT feasibility')
        encoded = answer.get('expected_answer_ids')
        require(type(encoded) is list and 0 < len(encoded) < OUTPUTS and
                all(type(token) is int and 0 <= token < VOCAB_SIZE for token in encoded) and
                answer.get('expected_answer_ids_sha256') == digest(canonical(encoded)) and
                answer.get('expected_answer_token_count') == len(encoded) and
                answer.get('answer_plus_one_end_of_turn_tokens') == len(encoded) + 1,
                'literal answer plus one assistant EOT fits the output budget')
    return report


def schedule(report):
    validate_native_prompts(report)
    return [dict(cell=row['case'], concurrency=1, input_tokens=len(row['input_ids']),
                 sample=0, measured=True, request_ids=[row['case'] + '-0-0'])
            for row in report['prompts']]


def decode_output(tokenizer, ids, case, termination, identity_sha256):
    """Use the actual source-bound tokenizer; retain all 32 generated IDs."""
    require(type(ids) is list and len(ids) == OUTPUTS and
            all(type(token) is int and 0 <= token < VOCAB_SIZE for token in ids),
            'exact finite output token vector')
    require(valid_digest(identity_sha256), 'native decoding identity binding')
    stops = termination['generation_stop_ids']
    index = next((offset for offset, token in enumerate(ids) if token in stops), None)
    boundary = OUTPUTS if index is None else index
    stop = None if index is None else ids[index]
    full = tokenizer.decode(ids, **DECODE_KWARGS)
    prefix = tokenizer.decode(ids[:boundary], **DECODE_KWARGS)
    tail = tokenizer.decode([] if index is None else ids[index + 1:], **DECODE_KWARGS)
    key = case['expected_answer']
    observed = dict(schema=1, mode=MODE, case=case['case'],
        native_tokenizer_identity_sha256=identity_sha256,
        raw_ids=copy.deepcopy(ids), raw_text=full, before_first_stop_text=prefix,
        first_stop_index=index, first_stop_id=stop, after_first_stop_ids=(
            [] if index is None else copy.deepcopy(ids[index + 1:])),
        after_first_stop_text=tail,
        assistant_end_of_turn_observed=stop in termination['assistant_end_of_turn_ids'],
        literal_key_occurrences_before_stop=prefix.count(key),
        retrieval_exactly_once=prefix.count(key) == 1,
        strict_answer_format=(prefix == key and
            stop in termination['assistant_end_of_turn_ids']),
        decode_kwargs=dict(DECODE_KWARGS), qualification=False,
        interpretation='Known-answer retrieval data, not broad quality qualification')
    validate_output(observed, ids, case, termination, identity_sha256)
    return observed


def validate_output(value, ids, case, termination, identity_sha256):
    fields = {'schema', 'mode', 'case', 'native_tokenizer_identity_sha256',
        'raw_ids', 'raw_text', 'before_first_stop_text', 'first_stop_index',
        'first_stop_id', 'after_first_stop_ids', 'after_first_stop_text',
        'assistant_end_of_turn_observed', 'literal_key_occurrences_before_stop',
        'retrieval_exactly_once', 'strict_answer_format', 'decode_kwargs',
        'qualification', 'interpretation'}
    require(type(value) is dict and set(value) == fields and
            type(value['schema']) is int and value['schema'] == 1 and
            value['mode'] == MODE and value['case'] == case['case'] and
            value['qualification'] is False and value['decode_kwargs'] == DECODE_KWARGS and
            all(type(value['decode_kwargs'].get(key)) is bool for key in DECODE_KWARGS) and
            value['interpretation'] == 'Known-answer retrieval data, not broad quality qualification' and
            value['native_tokenizer_identity_sha256'] == identity_sha256 and
            valid_digest(identity_sha256), 'native decoded-output binding/claims')
    require(type(ids) is list and len(ids) == OUTPUTS and
            all(type(token) is int and 0 <= token < VOCAB_SIZE for token in ids) and
            value['raw_ids'] == ids and all(type(token) is int for token in value['raw_ids']),
            'complete raw output IDs retained')
    index = next((offset for offset, token in enumerate(ids)
                  if token in termination['generation_stop_ids']), None)
    stop = None if index is None else ids[index]
    require(value['first_stop_index'] == index and
            (value['first_stop_index'] is None or type(value['first_stop_index']) is int) and
            value['first_stop_id'] == stop and
            (value['first_stop_id'] is None or type(value['first_stop_id']) is int) and
            value['after_first_stop_ids'] == ([] if index is None else ids[index + 1:]) and
            all(type(token) is int for token in value['after_first_stop_ids']),
            'first authenticated generation stop and untrimmed tail')
    for key in ('raw_text', 'before_first_stop_text', 'after_first_stop_text'):
        require(type(value[key]) is str and len(value[key].encode()) <= 64 << 10,
                'bounded native decoded text: ' + key)
    if index is None:
        require(value['before_first_stop_text'] == value['raw_text'] and
                value['after_first_stop_text'] == '', 'no invented termination')
    prefix, key = value['before_first_stop_text'], case['expected_answer']
    eot = stop in termination['assistant_end_of_turn_ids']
    require(type(value['assistant_end_of_turn_observed']) is bool and
            value['assistant_end_of_turn_observed'] is eot and
            type(value['literal_key_occurrences_before_stop']) is int and
            value['literal_key_occurrences_before_stop'] == prefix.count(key) and
            type(value['retrieval_exactly_once']) is bool and
            value['retrieval_exactly_once'] is (prefix.count(key) == 1) and
            type(value['strict_answer_format']) is bool and
            value['strict_answer_format'] is (prefix == key and eot),
            'literal retrieval and strict format recomputed separately')
    return value


def validate_snapshot(value, rank):
    fields = {'rank', 'device', 'free_bytes', 'total_bytes', 'allocated_bytes',
              'reserved_bytes', 'lifetime_peak_allocated_bytes',
              'lifetime_peak_reserved_bytes'}
    require(type(value) is dict and set(value) == fields and
            all(type(number) is int and number >= 0 for number in value.values()) and
            value['rank'] == value['device'] == rank and value['total_bytes'] > 0 and
            value['free_bytes'] <= value['total_bytes'] and
            value['allocated_bytes'] <= value['reserved_bytes'] <= value['total_bytes'] and
            value['lifetime_peak_allocated_bytes'] >= value['allocated_bytes'] and
            value['lifetime_peak_reserved_bytes'] >= value['reserved_bytes'] and
            value['lifetime_peak_allocated_bytes'] <=
                value['lifetime_peak_reserved_bytes'] <= value['total_bytes'],
            'actual bounded memory snapshot')
    return value


def validate_prefill_memory(records):
    """Recompute both ranks' continuation guards, never a transient peak bound."""
    require(type(records) is list and len(records) == 2,
            'both rank context memory records')
    for rank, rows in enumerate(records):
        require(type(rows) is list and len(rows) == CASES,
                'one before/after memory record per native unchunked prefill')
        for ordinal, row in enumerate(rows):
            require(type(row) is dict and set(row) == {'schema', 'mode', 'rank',
                    'ordinal', 'policy', 'before', 'after', 'peers_before',
                    'peers_after', 'complete', 'error'} and
                    type(row['schema']) is int and row['schema'] == 1 and
                    row['mode'] == MODE and type(row['rank']) is int and
                    row['rank'] == rank and type(row['ordinal']) is int and
                    row['ordinal'] == ordinal and row['policy'] == PREFILL_POLICY and
                    all(type(row['policy'].get(key)) is type(number)
                        for key, number in PREFILL_POLICY.items()) and
                    row['complete'] is True and row['error'] is None,
                    'complete bound prefill continuation record')
            for phase, threshold in (('before', PREFILL_POLICY['before_free_min_bytes']),
                                     ('after', PREFILL_POLICY['after_free_min_bytes'])):
                snapshot = validate_snapshot(row[phase], rank)
                require(snapshot['free_bytes'] >= threshold,
                        phase + '-prefill local headroom refused')
                peers = row['peers_' + phase]
                require(type(peers) is list and len(peers) == 2,
                        'exact memory TP consensus')
                for peer_rank, peer in enumerate(peers):
                    require(type(peer) is dict and set(peer) == {'rank', 'snapshot',
                            'error', 'allowed'} and type(peer['rank']) is int and
                            peer['rank'] == peer_rank and peer['error'] is None and
                            peer['allowed'] is True,
                            'successful peer memory observation')
                    actual = validate_snapshot(peer['snapshot'], peer_rank)
                    require(actual['free_bytes'] >= threshold,
                            'peer prefill threshold recomputation')
                require(peers[rank]['snapshot'] == snapshot,
                        'local snapshot equals its TP consensus record')
            require(row['before']['total_bytes'] == row['after']['total_bytes'],
                    'same prefill device memory total')
            require(all(row['after'][key] >= row['before'][key] for key in
                        ('lifetime_peak_allocated_bytes', 'lifetime_peak_reserved_bytes')),
                    'monotone native lifetime peaks across prefill')
    for ordinal in range(CASES):
        require(all(records[0][ordinal]['peers_' + phase] ==
                    records[1][ordinal]['peers_' + phase] for phase in ('before', 'after')),
                'both ranks retain the exact same memory consensus')
    return records


CONTEXT_BINDING = {'schema': 1, 'mode': 'native-model-context-retrieval-32k-v1', 'native_report_sha256': '72a61c9d97b07f023d7ca524ede4de82770d7a22a0dcd27285f19c5a0d2348a7', 'native_report_bytes': 5757066, 'compressed_report_sha256': 'cef78631478b938dec73af5382e54bf2470393cfce3f0eb6beabe753b0306f21', 'compressed_report_bytes': 577638, 'report_encoding': 'zlib-rfc1950-single-stream', 'tokenizer_helper_sha256': 'e0a5a7a1ea41f253ed0fbc08f0bfd83aa5d8037f0ac9f36bac875b5917d43160', 'tokenizer_contract_sha256': 'e0c8b26bd821bcc42719e55fbad658f7eecdae0053c0b8e05f48540cb3acdb88', 'native_cpu_receipt_sha256': '29a60455f14aa2c931d7b59ec7df447c8e5baaab85d03e907a99e4a9e9da9d99', 'control_case_sha256': 'bf47dcc43d36709311422af34f98f9a49ce51628240c2cc22ffb27b82d7ca201', 'control_case_bytes': 45475, 'facts_sha256': 'd414db807ccec0f7c285be99cd040c86e0230521b03996f64f9bc4f9392d3072', 'tokenizer_factory_before_model_imports': True, 'prefill': 'direct-native-unchunked'}

"""Context-only integration body appended to the authenticated graph helper.

The builder supplies CONTEXT_BINDING and copies context_contract.py before this
body. This file does not import a model, tokenizer, Torch, or a native runtime.
"""

import os
import stat
import sys
import zlib
from importlib.machinery import SourceFileLoader


_CONTEXT_REPORT = None
_CONTEXT_CONTRACT = None
_CONTEXT_FACTS = None
IDENTITY_FIELDS = ('factory', 'factory_kwargs', 'class_name', 'class_module',
                   'transformers_version', 'tokenizers_version', 'template_sha256',
                   'reasoning_policy', 'native_termination', 'bos_token_id',
                   'eos_token_id', 'apply_chat_template_signature')
TOKENIZER_ENV = {'USE_TORCH': '0', 'USE_TF': '0', 'USE_FLAX': '0',
                 'TOKENIZERS_PARALLELISM': 'false', 'RAYON_NUM_THREADS': '2',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def context_asset(name, pin, maximum):
    import mtp_contract
    if name in mtp_contract.PACKED_REQUEST_NAMES:
        return mtp_contract.logical_asset(name, pin, maximum)
    if name == 'COMPOSITION-FACTS.json':
        data = mtp_contract.composition_facts_bytes()
        require(len(data) <= maximum and digest(data) == pin, 'original full facts identity')
        return data
    path = Path(__file__).with_name(name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 <= before.st_size <= maximum, 'bounded context asset')
        data = b''
        while len(data) <= maximum:
            block = os.read(fd, min(1 << 20, maximum + 1 - len(data)))
            if not block:
                break
            data += block
        after = os.fstat(fd)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        require(stamp(before) == stamp(after) == stamp(path.lstat()) and
                len(data) <= maximum and digest(data) == pin,
                'unchanged authenticated context asset: ' + name)
        return data
    finally:
        os.close(fd)


def context_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate context JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite context JSON')))


def context_inputs():
    global _CONTEXT_REPORT, _CONTEXT_CONTRACT, _CONTEXT_FACTS
    if _CONTEXT_REPORT is None:
        report = context_json(context_decompress_report(context_asset('CONTEXT-PROMPTS.zlib',
            CONTEXT_BINDING['compressed_report_sha256'], 8 << 20)))
        contract = context_json(context_asset('TOKENIZER-CONTRACT.json',
            CONTEXT_BINDING['tokenizer_contract_sha256'], 256 << 10))
        require(report['contract'] == contract, 'native report/exact tokenizer contract')
        validate_native_prompts(report)
        facts = context_json(context_asset('COMPOSITION-FACTS.json',
            CONTEXT_BINDING['facts_sha256'], 16 << 20))
        _CONTEXT_REPORT, _CONTEXT_CONTRACT, _CONTEXT_FACTS = report, contract, facts
    return _CONTEXT_REPORT, _CONTEXT_CONTRACT, _CONTEXT_FACTS


def context_decompress_report(data, binding=None):
    """Bound expansion before parsing; authenticate both stored and full raw bytes."""
    binding = CONTEXT_BINDING if binding is None else binding
    require(type(data) is bytes and type(binding.get('native_report_bytes')) is int and
            0 < binding['native_report_bytes'] <= 8 << 20 and
            type(binding.get('compressed_report_bytes')) is int and
            0 < binding['compressed_report_bytes'] <= 8 << 20 and
            binding.get('report_encoding') == 'zlib-rfc1950-single-stream' and
            len(data) == binding['compressed_report_bytes'] and
            digest(data) == binding['compressed_report_sha256'],
            'exact bounded compressed native report')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data, binding['native_report_bytes'] + 1)
    require(len(raw) == binding['native_report_bytes'] and decoder.eof is True and
            decoder.unused_data == decoder.unconsumed_tail == b'' and
            digest(raw) == binding['native_report_sha256'],
            'complete single native report stream/raw size/hash')
    return raw


def context_validate_spec(spec):
    report, _, _ = context_inputs()
    require(spec.get('context_retrieval') == CONTEXT_BINDING and
            spec.get('prefill_policy') == PREFILL_POLICY and
            all(type(spec['prefill_policy'].get(key)) is type(value)
                for key, value in PREFILL_POLICY.items()), 'frozen context source/prefill binding')
    counts = spec.get('context_counts')
    expected = dict(requests=CASES, batches=CASES, workload_forwards=WORKLOAD_FORWARDS,
                    decode_replays_per_graph_rank=WORKLOAD_REPLAYS,
                    startup_records_per_graph_rank=6, workload_warmup_batches=0)
    require(type(counts) is dict and counts == expected and
            all(type(counts[key]) is int for key in expected), 'exact finite context schedule')
    require(all(type(spec.get(key)) is int for key in
                ('schema', 'context_length', 'physical_slots', 'outputs',
                 'warmup_batches', 'measured_batches')) and
            spec['context_length'] == spec['physical_slots'] == CONTEXT_LENGTH and
            spec['warmup_batches'] == 0 and spec['measured_batches'] == 1,
            'native unchunked context dimensions/types')
    args = spec['server_args']
    require(type(args.get('chunked_prefill_size')) is int and
            args['chunked_prefill_size'] == -1 and
            args['max_running_requests'] == 2 and type(args['max_running_requests']) is int,
            'intentional unchunked prefill and native graph request capacity')
    require(list(spec['frozen_prompt_ids']) == [row['case'] for row in report['prompts']] or
            set(spec['frozen_prompt_ids']) == {row['case'] for row in report['prompts']},
            'seven frozen input names')
    for row in report['prompts']:
        require(spec['frozen_prompt_ids'][row['case']] == [row['input_ids']] and
                spec['frozen_prompt_sha256'][row['case']] ==
                [digest(json.dumps(row['input_ids'], separators=(',', ':')).encode())],
                'exact native CPU full-chat input IDs')


def context_source_preflight(namespace, spec):
    context_validate_spec(spec)
    for name, key, maximum in (('context_tokenizer.py', 'tokenizer_helper_sha256', 256 << 10),
            ('TOKENIZER-CONTRACT.json', 'tokenizer_contract_sha256', 256 << 10),
            ('CONTEXT-PROMPTS.zlib', 'compressed_report_sha256', 8 << 20),
            ('CONTROL-CASE.zlib', 'control_case_sha256', 256 << 10)):
        if name in ('TOKENIZER-CONTRACT.json', 'CONTEXT-PROMPTS.zlib', 'CONTROL-CASE.zlib'):
            context_asset(name, CONTEXT_BINDING[key], maximum)
        else:
            namespace['checked_file'](Path(__file__).with_name(name), CONTEXT_BINDING[key], maximum)


def context_validate_tokenizer_identity(identity):
    report, contract, facts = context_inputs()
    expected = report['tokenizer_identity']
    require(type(identity) is dict and
            all(identity.get(key) == expected[key] for key in IDENTITY_FIELDS) and
            type(identity.get('factory_kwargs')) is dict and
            all(type(identity['factory_kwargs'].get(key)) is bool for key in
                ('local_files_only', 'trust_remote_code', 'use_fast')) and
            identity.get('reasoning_policy', {}).get('enable_thinking') is False and
            identity.get('reasoning_policy', {}).get('declared_template_variable') is True and
            identity.get('reasoning_policy', {}).get('probe_renderings_differ') is True and
            identity.get('still_forbidden_namespaces') == ['jax', 'sglang', 'tensorflow'] and
            identity.get('forbidden_modules_observed_absent_before_import_and_after_load') is True and
            identity.get('transient_imports_attested') is False,
            'actual native tokenizer factory/class/template/API identity')
    before, after = (identity.get('native_torch_dependency_' + phase)
                     for phase in ('before', 'after'))
    require(type(before) is dict and before.get('native_torch_loaded') is False and
            before.get('native_triton_loaded') is False and type(after) is dict and
            after.get('native_torch_loaded') is True and after.get('native_triton_loaded') is True and
            after.get('authenticated_modules') ==
                expected['native_torch_dependency_after']['authenticated_modules'] and
            after.get('version_source') == expected['native_torch_dependency_after']['version_source'] and
            after.get('declared_version') == expected['native_torch_dependency_after']['declared_version'] and
            after.get('tensor_device_model_apis_called_by_helper') is False and
            after.get('triton_jit_compile_apis_called_by_helper') is False and
            after.get('transient_imports_attested') is False,
            'source-bound allowed native CPU dependencies before model imports')
    loaded = identity.get('loaded_sources')
    require(type(loaded) is dict and expected['class_module'] in loaded and
            loaded[expected['class_module']] == expected['loaded_sources'][expected['class_module']],
            'native resolved tokenizer class source')
    prefix = contract['installed_prefix'].rstrip('/') + '/'
    for name, row in loaded.items():
        require(type(name) is str and name.split('.')[0] in
                ('transformers', 'tokenizers', 'jinja2', 'huggingface_hub') and
                type(row) is dict and set(row) == {'path', 'sha256', 'bytes'} and
                type(row['path']) is str and type(row['bytes']) is int and
                0 < row['bytes'] <= 64 << 20, 'loaded native tokenizer source metadata')
        path = row['path']
        if path.startswith(prefix):
            pin = facts['installed_prefix_files'].get(path[len(prefix):])
        else:
            allowed = contract['base_runtime_sources'].get(path)
            require(name.split('.')[0] == 'jinja2' and type(allowed) is dict and
                    row['bytes'] == allowed['bytes'], 'exact selected base Jinja2 source')
            pin = allowed['sha256']
        require(valid_digest(pin) and row['sha256'] == pin, 'actual loaded tokenizer source inventory')
    return identity


def context_prepare_tokenizer(namespace, spec):
    """Run the consumed native factory before importing SGLang/model/Torch here."""
    context_source_preflight(namespace, spec)
    report, contract, _ = context_inputs()
    helper = namespace['authenticated_adjacent']('context_tokenizer.py',
                                                CONTEXT_BINDING['tokenizer_helper_sha256'])
    prior = {key: os.environ.get(key) for key in TOKENIZER_ENV}
    try:
        os.environ.update(TOKENIZER_ENV)
        tokenizer, identity = helper.native_tokenizer(copy.deepcopy(contract), Path('/mnt/models'))
        context_validate_tokenizer_identity(identity)
        for expected in report['prompts']:
            actual = helper.observe(tokenizer, expected['body'], expected['expected_answer'])
            require(all(actual[key] == expected[key] for key in actual),
                    'native worker re-observes exact frozen full-chat input')
            require(helper.expected_answer_budget(tokenizer, expected['expected_answer']) ==
                    expected['expected_answer_output'], 'native worker answer/EOT contract')
        return tokenizer, identity
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def context_class_source(namespace, value):
    _, _, facts = context_inputs()
    cls = type(value)
    module = sys.modules.get(cls.__module__)
    origin = getattr(getattr(module, '__spec__', None), 'origin', None)
    loader = getattr(getattr(module, '__spec__', None), 'loader', None)
    require(module is not None and module.__file__ == origin and
            isinstance(loader, SourceFileLoader) and loader.path == origin and
            getattr(module, cls.__name__, None) is cls and
            type(origin) is str and origin.startswith('/mnt/composition/image-source/python/sglang/'),
            'actual native pool class/module/loader')
    relative = origin.removeprefix('/mnt/composition/image-source/python/sglang/')
    pin = facts['runtime_files'].get(relative)
    require(valid_digest(pin), 'pool class source in effective runtime inventory')
    namespace['checked_file'](origin, pin)
    return dict(class_name=cls.__name__, module=cls.__module__, path=origin, sha256=pin)


def context_observe_pools(namespace, runner):
    def tensor(value):
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                    element_size=value.element_size(), numel=value.numel())
    req = runner.req_to_token_pool
    mamba = req.mamba_pool
    cache = mamba.mamba_cache
    result = dict(schema=1, mode=MODE, before_graph_initialization=True,
        request_pool=context_class_source(namespace, req),
        mamba_pool=context_class_source(namespace, mamba),
        request_to_token=tensor(req.req_to_token),
        mamba_declared_slots=mamba.size, mamba_layer_map=dict(req.mamba_map),
        temporal=tensor(cache.temporal), conv=[tensor(value) for value in cache.conv],
        optional_state_present={name: getattr(cache, name, None) is not None for name in
            ('replayssm_d', 'replayssm_k', 'replayssm_g', 'replayssm_rawv',
             'replayssm_rawk', 'replayssm_beta', 'intermediate_ssm')})
    context_validate_pools(result)
    return result


def context_validate_pools(value):
    _, _, facts = context_inputs()
    require(type(value) is dict and set(value) == {'schema', 'mode',
            'before_graph_initialization', 'request_pool', 'mamba_pool',
            'request_to_token', 'mamba_declared_slots', 'mamba_layer_map',
            'temporal', 'conv', 'optional_state_present'} and value.get('schema') == 1 and
            type(value.get('schema')) is int and value.get('mode') == MODE and
            value.get('before_graph_initialization') is True, 'actual native pool observation')
    for name in ('request_pool', 'mamba_pool'):
        source = value.get(name)
        require(type(source) is dict and set(source) == {'class_name', 'module', 'path', 'sha256'} and
                all(type(item) is str for item in source.values()) and
                source['path'].startswith('/mnt/composition/image-source/python/sglang/') and
                facts['runtime_files'].get(source['path'].removeprefix(
                    '/mnt/composition/image-source/python/sglang/')) == source['sha256'],
                'native actual pool source binding')
    def tensor(row):
        require(type(row) is dict and set(row) == {'shape', 'stride', 'dtype', 'element_size', 'numel'} and
                type(row['shape']) is list and type(row['stride']) is list and
                len(row['shape']) == len(row['stride']) and row['shape'] and
                all(type(number) is int and number > 0 for number in row['shape']) and
                all(type(number) is int and number >= 0 for number in row['stride']) and
                type(row['element_size']) is int and row['element_size'] in (2, 4, 8) and
                type(row['numel']) is int and row['numel'] == math.prod(row['shape']),
                'observed pool tensor dimensions/strides')
        return row
    request = tensor(value['request_to_token'])
    require(len(request['shape']) == 2 and request['shape'][0] >= 3 and
            request['shape'][1] >= CONTEXT_LENGTH and request['dtype'] in
            ('torch.int32', 'torch.int64'), 'native request/context capacity at least two/16K')
    require(type(value['mamba_layer_map']) is dict and
            all((type(key) is int or (type(key) is str and key.isdecimal() and
                 str(int(key)) == key)) and type(index) is int
                for key, index in value['mamba_layer_map'].items()) and
            type(value['mamba_declared_slots']) is int and value['mamba_declared_slots'] >= 2 and
            {int(key): item for key, item in value['mamba_layer_map'].items()} ==
                {layer: index for index, layer in enumerate(i for i in range(64) if i % 4 != 3)},
            'actual 48-linear-layer mapping and finite state capacity')
    temporal = tensor(value['temporal'])
    require(len(temporal['shape']) == 5 and temporal['shape'][0] == 48 and
            temporal['shape'][1] >= value['mamba_declared_slots'] and
            temporal['shape'][2:] == [24, 128, 128] and
            temporal['dtype'] == 'torch.float32' and temporal['element_size'] == 4,
            'observed native FP32 GDN state geometry')
    require(type(value['conv']) is list and len(value['conv']) == 1,
            'native single GDN convolution-state field')
    conv = tensor(value['conv'][0])
    require(conv['shape'] == [48, temporal['shape'][1], 5120, 3] and
            conv['dtype'] == 'torch.bfloat16' and conv['element_size'] == 2,
            'observed native BF16 GDN convolution-state geometry')
    require(type(value['optional_state_present']) is dict and
            set(value['optional_state_present']) == {'replayssm_d', 'replayssm_k', 'replayssm_g',
                'replayssm_rawv', 'replayssm_rawk', 'replayssm_beta', 'intermediate_ssm'} and
            all(item is False for item in value['optional_state_present'].values()),
            'no speculative/replay-state protocol added')
    return value


def context_memory_before(namespace, torch, rank, dist, group, result, ordinal):
    records = result.setdefault('context_prefill_memory', [])
    require(type(ordinal) is int and ordinal == len(records) and ordinal < CASES,
            'next unchunked prefill memory record')
    row = dict(schema=1, mode=MODE, rank=rank, ordinal=ordinal, policy=dict(PREFILL_POLICY),
               before=None, after=None, peers_before=None, peers_after=None, complete=False, error=None)
    records.append(row)
    try:
        snapshot = namespace['_GRAPH_OUTER'].snapshot(torch, rank)
        validate_snapshot(snapshot, rank)
        local = dict(rank=rank, snapshot=snapshot, error=None,
                     allowed=snapshot['free_bytes'] >= PREFILL_POLICY['before_free_min_bytes'])
        row['before'] = snapshot
    except BaseException as error:
        local = dict(rank=rank, snapshot=None, error=(type(error).__name__ + ': ' + str(error))[:512], allowed=False)
    row['peers_before'] = namespace['_GRAPH_OUTER'].gather(dist, group, local)
    if not context_peer_headroom(row['peers_before'], PREFILL_POLICY['before_free_min_bytes']):
        row['error'] = 'before-prefill headroom refused by at least one TP rank'
        require(False, row['error'])


def context_memory_after(namespace, torch, rank, dist, group, result, ordinal):
    row = result['context_prefill_memory'][ordinal]
    try:
        snapshot = namespace['_GRAPH_OUTER'].snapshot(torch, rank)
        validate_snapshot(snapshot, rank)
        local = dict(rank=rank, snapshot=snapshot, error=None,
                     allowed=snapshot['free_bytes'] >= PREFILL_POLICY['after_free_min_bytes'])
        row['after'] = snapshot
    except BaseException as error:
        local = dict(rank=rank, snapshot=None, error=(type(error).__name__ + ': ' + str(error))[:512], allowed=False)
    row['peers_after'] = namespace['_GRAPH_OUTER'].gather(dist, group, local)
    row['complete'] = context_peer_headroom(row['peers_after'], PREFILL_POLICY['after_free_min_bytes'])
    if not row['complete']:
        row['error'] = 'after-prefill continuation refused by at least one TP rank'
        require(False, row['error'])


def context_peer_headroom(peers, threshold):
    require(type(peers) is list and len(peers) == 2, 'two observed memory peers')
    allowed = True
    for rank, peer in enumerate(peers):
        require(type(peer) is dict and set(peer) == {'rank', 'snapshot', 'error', 'allowed'} and
                type(peer['rank']) is int and peer['rank'] == rank and
                type(peer['allowed']) is bool, 'typed memory peer record')
        if peer['error'] is not None:
            require(type(peer['error']) is str and len(peer['error']) <= 512 and
                    peer['snapshot'] is None and peer['allowed'] is False,
                    'bounded unsuccessful memory observation')
            allowed = False
        else:
            snapshot = validate_snapshot(peer['snapshot'], rank)
            actual = snapshot['free_bytes'] >= threshold
            require(peer['allowed'] is actual, 'truthful live peer headroom result')
            allowed = allowed and actual
    return allowed


def context_finish_memory(namespace, torch, rank, dist, group, result):
    # Normal completion only: never enter this collective while unwinding a
    # prior native exception. A local snapshot failure is still sent to peers.
    value = dict(snapshot=None, peers=None, complete=False, error=None,
        scope='Torch lifetime peaks through all native workload forwards; driver transient peak unknown',
        hard_transient_peak_quota=False)
    result['context_full_workload_memory'] = value
    try:
        snapshot = namespace['_GRAPH_OUTER'].snapshot(torch, rank)
        validate_snapshot(snapshot, rank)
        local = dict(rank=rank, snapshot=snapshot, error=None,
                     allowed=snapshot['free_bytes'] >= PREFILL_POLICY['after_free_min_bytes'])
        value['snapshot'] = snapshot
    except BaseException as error:
        local = dict(rank=rank, snapshot=None,
                     error=(type(error).__name__ + ': ' + str(error))[:512], allowed=False)
    value['peers'] = namespace['_GRAPH_OUTER'].gather(dist, group, local)
    value['complete'] = context_peer_headroom(value['peers'], PREFILL_POLICY['after_free_min_bytes'])
    if not value['complete']:
        value['error'] = 'final workload memory observation/headroom refused by at least one TP rank'
        require(False, value['error'])
    return value


def context_read_rows(rows):
    report, _, _ = context_inputs()
    termination = report['tokenizer_identity']['native_termination']
    require(type(rows) is list and len(rows) == 2, 'exact two context rank records')
    for rank, row in enumerate(rows):
        require(type(row.get('batches')) is list and len(row['batches']) == CASES,
                'exact seven complete context batches')
        identity = context_validate_tokenizer_identity(row.get('context_tokenizer_identity'))
        identity_pin = digest(canonical(identity))
        context_validate_pools(row.get('context_pool_identity'))
        baseline = row.get('capacity_before')
        require(type(baseline) is list and len(baseline) == 3 and
                all(type(number) is int for number in baseline) and
                baseline[0] == CONTEXT_LENGTH and baseline[1] == 2 and baseline[2] >= 2 and
                row.get('capacity_after') == baseline, 'cold native pool capacity restored')
        for ordinal, (batch, case) in enumerate(zip(row['batches'], report['prompts'])):
            require(batch['ordinal'] == ordinal and batch['cell'] == case['case'] and
                    batch['concurrency'] == 1 and batch['input_tokens'] == len(case['input_ids']) and
                    batch['sample'] == 0 and batch['measured'] is True and
                    batch['capacity_before'] == batch['capacity_after'] == baseline,
                    'fixed cold singleton native case and pool restoration')
            validate_output(batch.get('context_output'), batch['generated_ids'][0], case,
                            termination, identity_pin)
        final = row.get('context_full_workload_memory')
        require(type(final) is dict and set(final) == {'snapshot', 'peers', 'complete', 'scope', 'error',
                'hard_transient_peak_quota'} and final['complete'] is True and
                final['error'] is None and
                final['hard_transient_peak_quota'] is False and
                final['scope'] == 'Torch lifetime peaks through all native workload forwards; driver transient peak unknown',
                'actual whole-workload Torch lifetime peak observation')
        validate_snapshot(final['snapshot'], rank)
        require(all(final['snapshot'][key] >= max(memory['after'][key]
                    for memory in row['context_prefill_memory']) for key in
                    ('lifetime_peak_allocated_bytes', 'lifetime_peak_reserved_bytes')),
                'final lifetime peaks include every native prefill')
        require(type(final['peers']) is list and len(final['peers']) == 2 and
                final['peers'][rank]['snapshot'] == final['snapshot'], 'final memory TP join')
        for peer_rank, peer in enumerate(final['peers']):
            require(type(peer) is dict and set(peer) == {'rank', 'snapshot', 'error', 'allowed'} and
                    type(peer['rank']) is int and peer['rank'] == peer_rank and peer['error'] is None and
                    peer['allowed'] is True, 'complete final memory consensus')
            actual = validate_snapshot(peer['snapshot'], peer_rank)
            require(actual['free_bytes'] >= PREFILL_POLICY['after_free_min_bytes'],
                    'final peer headroom recomputation')
    validate_prefill_memory([row['context_prefill_memory'] for row in rows])
    require(rows[0]['context_full_workload_memory']['peers'] ==
            rows[1]['context_full_workload_memory']['peers'], 'identical final memory consensus')
    require([batch['context_output'] for batch in rows[0]['batches']] ==
            [batch['context_output'] for batch in rows[1]['batches']],
            'TP decoded response and native decoding identity agreement')


def context_grade_summary(rows):
    require(type(rows) is list and len(rows) == 2 and
            all(type(row.get('batches')) is list and len(row['batches']) == CASES for row in rows),
            'exact eighteen-case grading denominator on both ranks')
    values = [batch['context_output'] for batch in rows[0]['batches']]
    return dict(requests=CASES, retrieval_exactly_once=sum(value['retrieval_exactly_once'] for value in values),
        strict_answer_format=sum(value['strict_answer_format'] for value in values),
        assistant_end_of_turn_observed=sum(value['assistant_end_of_turn_observed'] for value in values),
        cases=copy.deepcopy(values), qualification=False,
        interpretation='Exact literal retrieval and strict format are separate; no broad reasoning-quality qualification',
        offline_decode_limit='Native source-bound tokenizer produced text; offline reader recomputes grading, not tokenizer decoding')
