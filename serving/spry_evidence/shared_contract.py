"""Pure normal-stop request/readout contract; no transport, launch or code execution."""
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from functools import lru_cache
import mtp_contract


MODE = 'native-http-fp8-sprint01-v1'
MODEL = 'fp8-serving-screen'
# Exact proposed request asset, within the existing asset bounds.
def _kv_rows():
    import zlib
    config=mtp_contract.raw_configuration()
    inventory=None if config is None else config['logical_package_inventory']['request_container']['members']['CLIENT-BINDING.zlib']
    raw = mtp_contract.logical_asset('CLIENT-BINDING.zlib',
        'ae0145b4b0d59c4c03fe090bb80852c79a740d597806d643f2f1b51af80b3280' if inventory is None else inventory['sha256'], 2 << 20)
    if config is not None:return mtp_contract._configured_client_rows(raw,len(config['source_ordinals']))
    decoder = zlib.decompressobj()
    expanded = decoder.decompress(raw, (16<<20)+1)
    if len(expanded)>16<<20 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('complete composed client asset')
    if config is None and hashlib.sha256(expanded).hexdigest() != 'd09b4e4978b6a3ba99b009968164cc891ffe8b6a3a569ed312e531a3f6bf7c4f':
        raise ValueError('composed client decoded pin')
    rows = json.loads(expanded)['rows']
    count=176 if config is None else len(config['source_ordinals'])
    if len(rows)!=count or [r['ordinal'] for r in rows]!=list(range(count)):
        raise ValueError('composed request ordinal coverage')
    return rows

_KV_ROWS = _kv_rows()
_STOP_POLICIES = {row['case']: row.get('stop_policy', 'natural')
                  for row in mtp_contract.settings()['requests']}
_COMPLETION_PLANS = {row['case']: dict(stop_policy=_STOP_POLICIES[row['case']],
    maximum_output_tokens=row['maximum_output_tokens']) for row in _KV_ROWS}
CASES = tuple(row['case'] for row in _KV_ROWS)
CAPS = tuple(row['maximum_output_tokens'] for row in _KV_ROWS)
SPECULATION_FLOAT_KEYS = (
    'spec_accept_rate', 'spec_accept_length', 'spec_cap_length',
    'spec_block_accept_length',
)
SPECULATION_COUNT_KEYS = (
    'spec_num_correct_drafts', 'spec_num_proposed_drafts', 'spec_verify_ct',
)
SPECULATION_HISTOGRAM_KEYS = (
    'spec_correct_drafts_histogram', 'spec_cap_lens_histogram',
)
SPECULATION_KEYS = frozenset(
    SPECULATION_FLOAT_KEYS + SPECULATION_COUNT_KEYS + SPECULATION_HISTOGRAM_KEYS
)
COMBINED_PHASE = 'sprint01-combined-forward'
SAMPLING_PHASE = 'sprint01-sampling-forward'
HIP_SAMPLING_PHASE = 'sprint01-hip-sampling-forward'
CONFIRMATION_PHASE = 'sprint01-hip-confirmation-reverse'
GDN_PAIR_PHASE = mtp_contract.GDN_PAIR_PHASE
HIP_PHASE_ARMS = {HIP_SAMPLING_PHASE: 'mtp', CONFIRMATION_PHASE: 'control',
                  GDN_PAIR_PHASE: ('control', 'mtp')}
SAMPLING_PHASES = frozenset((SAMPLING_PHASE, *mtp_contract.PRECISION_REFERENCE_PHASES, *HIP_PHASE_ARMS))
MAX_STREAM_BYTES = 512 << 10
MAX_STREAM_CHUNKS = 4096


canonical = mtp_contract.json_line


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


PROMPT_FLUSH_ENDPOINT = '/flush_cache?timeout=0'
PROMPT_FLUSH_ACK = (b'Cache flushed.\nPlease check backend logs for more details. '
                    b'(When there are running or waiting requests, the operation will not be performed.)\n')


def prompt_flush_contract():
    _need(mtp_contract.prompt_minimization_enabled(), 'prompt flush outside minimization phase')
    return dict(schema=1, before_ordinals=[1, 2], method='POST', endpoint=PROMPT_FLUSH_ENDPOINT,
        request_bytes=0, success_http_status=200,
        acknowledgment_scope='native TP leader acknowledgment; no per-rank completion acknowledgments are exported',
        tp_reset_inference='authenticated native broadcast, serial dispatch and state-synchronous TP group contract',
        cache_hit_count_observed=False)


def screen_policy():
    return mtp_contract.settings().get('screen_policy')


def response_completed(row):
    plan = _COMPLETION_PLANS.get(row.get('case'))
    return plan is not None and mtp_contract.completion_allowed(row, plan)


def sampling_contract():
    return None  # Exact per-request parameters are bound; no uniform composed sampling claim.


def validate_sampling_parameters(body, ordinal):
    if mtp_contract.settings().get('quality_screen') == 'sprint7-low-thinking-publisher-sampled-four-v1':
        _need(type(body.get('temperature')) is float and body['temperature']==1.0 and
              type(body.get('top_p')) is float and body['top_p']==0.95 and
              type(body.get('top_k')) is int and body['top_k']==20 and
              type(body.get('min_p')) is float and body['min_p']==0.0 and
              type(body.get('presence_penalty')) is float and body['presence_penalty']==0.0 and
              type(body.get('repetition_penalty')) is float and body['repetition_penalty']==1.0 and
              'seed' not in body and body.get('stream') is False and type(body.get('n')) is int and body['n']==1 and
              body.get('return_token_ids') is True and body.get('return_meta_info') is True,
              'frozen publisher sampling recipe differs')
        return
    reference = mtp_contract.precision_reference_enabled()
    greedy = mtp_contract.greedy_quality_draft_ab() or reference
    seed_valid = ('seed' not in body if reference or mtp_contract.settings()['phase'] in HIP_PHASE_ARMS else
                  type(body.get('seed')) is int and body['seed'] == 1729 + ordinal)
    _need(type(body.get('temperature')) is float and body['temperature'] == (0.0 if greedy else 0.7) and
          type(body.get('top_p')) is float and body['top_p'] == (1.0 if greedy else 0.9) and
          type(body.get('top_k')) is int and body['top_k'] == (1 if greedy else 40) and
          seed_valid and
          body.get('stream') is False and type(body.get('n')) is int and body['n'] == 1 and
          body.get('return_token_ids') is True and body.get('return_meta_info') is True,
          'frozen native sampling parameters differ')


def hip_sampling_arm(phase, arm):
    if phase == GDN_PAIR_PHASE and mtp_contract.greedy_quality_draft_ab():
        return arm in ('control', 'mtp') if mtp_contract.verify_attention_ab_enabled() else arm == 'mtp'
    if phase == mtp_contract.LIFECYCLE_PHASE:
        return arm in ('control', 'mtp')
    selected = HIP_PHASE_ARMS.get(phase)
    return arm in selected if type(selected) is tuple else arm == selected


def validate_thinking_quality_request(body):
    """Reject accidental reuse of the old nonthinking request profile."""
    _need(body.get('chat_template_kwargs') == dict(enable_thinking=True) and
          body['chat_template_kwargs']['enable_thinking'] is True and
          body.get('reasoning_effort') == 'low' and body.get('ignore_eos') is False and
          type(body.get('max_completion_tokens')) is int and body['max_completion_tokens'] == 4096,
          'selected thinking quality request profile differs')


_FROZEN_REQUESTS = {}


def requests():
    # W5 v2: validate the frozen set once per authenticated settings/request-container pair. It was rebuilt and revalidated for
    # every response (O(N) per response, about 4.4 s of client CPU each at 176 rows; found by the post-run acceptance harness).
    # A changed settings file or container is revalidated in full; callers always receive a private copy.
    key = (hashlib.sha256(Path(mtp_contract.__file__).with_name('FINAL-SETTINGS.json').read_bytes()).hexdigest(),
           hashlib.sha256(Path(mtp_contract.__file__).with_name('FINAL-REQUESTS.zlib').read_bytes()).hexdigest())
    if key not in _FROZEN_REQUESTS:
        _FROZEN_REQUESTS.clear()
        _FROZEN_REQUESTS[key] = paired_requests()
    return copy.deepcopy(_FROZEN_REQUESTS[key])


def _validate_s35_schedule(rows):
    _need([x['case'] for x in rows] == ['warmup','cold-a','warm-a','branch-a-warm','isolated-b-cold','branch-b-warm'], 'S35 request order')
    bodies=[r['body'] for r in rows]
    _need(bodies[1] == bodies[2] and bodies[4] == bodies[5], 'S35 repeated bodies differ')
    b=dict(bodies[4]); b['cache_salt']=bodies[3]['cache_salt']
    _need(b==bodies[3] and bodies[1]['cache_salt']==bodies[3]['cache_salt'] and bodies[4]['cache_salt']!=bodies[3]['cache_salt'], 'S35 branch salt binding')
    ids=[expected_case(r['case'])['input_ids'] for r in rows]
    _need(ids[1]==ids[2] and ids[3]==ids[4]==ids[5], 'S35 repeat token ids differ')
    lcp=next((i for i,(a,b) in enumerate(zip(ids[1],ids[3])) if a!=b),min(len(ids[1]),len(ids[3])))
    _need(lcp==25890 and lcp < min(len(ids[1]),len(ids[3])), 'S35 token prefix binding differs')


def s35_cache_policy(case):
    ids=expected_case(case)['input_ids']
    if case in ('warmup','cold-a','isolated-b-cold'): low,high=0,0
    elif case == 'branch-a-warm': low,high=1,25890
    else:
        _need(case in ('warm-a','branch-b-warm'), 'unknown S35 case')
        low,high=1,len(ids)-1
    return dict(minimum_cached_tokens=low,maximum_cached_tokens=high,intent=case)


def expected_case(name, assistant_text=None):
    _need(assistant_text is None, 'no composed generated-history variant')
    rows = [row for row in _KV_ROWS if row['case']==name]
    _need(len(rows)==1, 'exact composed expected case')
    if mtp_contract.raw_configuration() is not None:
        return mtp_contract.strict_json(mtp_contract.asset_bytes(rows[0]['expected_path']))
    return copy.deepcopy(rows[0]['expected'])


def history_branch(case, branch_index):
    """Materialize one source-bound native-preflight branch, without sending it."""
    template = next(item for item in requests() if item['case'] == case['case'])
    _need(type(branch_index) is int and 'generated_history' in template and
          0 <= branch_index < len(template['generated_history']['branches']), 'invalid history branch')
    actual = copy.deepcopy(template)
    actual['body']['messages'][2]['content'] = template['generated_history']['branches'][branch_index]['assistant_text']
    actual['history_branch_index'] = branch_index
    return actual


def expected_for_case(case):
    if 'history_branch_index' in case:
        template = history_branch(case, case['history_branch_index'])
        _need(case['body'] == template['body'], 'history body differs from selected branch')
        return expected_case(case['case'], template['body']['messages'][2]['content'])
    return expected_case(case['case'])


class HistoryRefusal(ValueError):
    """A normal response did not select an exact frozen feedback branch."""


def materialize_history(case, previous_case, previous_raw):
    """Bind the literal prior wire answer; never trim, replace, or retry it."""
    _need(mtp_contract.settings()['phase'] in mtp_contract.HISTORY_PHASES and
          case.get('generated_history', {}).get('previous_case') == previous_case and
          type(previous_raw) is bytes and 0 < len(previous_raw) <= 512 << 10,
          'generated history parent binding differs')
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'duplicate generated history JSON key')
            result[key] = value
        return result
    response = json.loads(previous_raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite generated history JSON')))
    choices = response.get('choices')
    if (screen_policy() == 'exploratory-stop-or-length-v1' and type(choices) is list and
            len(choices) == 1 and type(choices[0]) is dict and choices[0].get('finish_reason') == 'length'):
        raise HistoryRefusal('capped generated answer cannot supply a completed history turn')
    _need(type(choices) is list and len(choices) == 1 and choices[0].get('finish_reason') == 'stop' and
          choices[0].get('message', {}).get('role') == 'assistant' and
          type(choices[0]['message'].get('content')) is str, 'generated history parent is not normal assistant text')
    text = choices[0]['message']['content']
    indices = [index for index, branch in enumerate(case['generated_history']['branches'])
               if branch['assistant_text'] == text]
    if len(indices) != 1:
        raise HistoryRefusal('generated answer is outside the frozen exact-text branches')
    actual = history_branch(case, indices[0])
    actual['history_binding'] = dict(schema=1, previous_case=previous_case,
        parent_response_sha256=hashlib.sha256(previous_raw).hexdigest(),
        assistant_text_sha256=hashlib.sha256(text.encode()).hexdigest(), branch_index=indices[0],
        template_request_sha256=digest(case['body']), request_sha256=digest(actual['body']),
        expected_sha256=case['generated_history']['branches'][indices[0]]['expected_sha256'])
    return actual


def history_refusal(case, previous_raw):
    return dict(schema=1, case=case['case'], previous_case=case['generated_history']['previous_case'],
        parent_response_sha256=hashlib.sha256(previous_raw).hexdigest(),
        status='GENERATED_HISTORY_REFUSAL', request_submitted=False, qualification=False)


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _count(value):
    return type(value) is int and value >= 0


def _ids(value, *, maximum=131072):
    return (type(value) is list and 0 < len(value) <= maximum and
            all(type(token) is int and token >= 0 for token in value))


def inspect_response(case, response, client_elapsed_seconds):
    """Preserve a refusal as data; do not invent EOS inclusion or server timing."""
    expected = {row['case']: row for row in requests()}
    known = expected.get(case['case'])
    if 'history_branch_index' in case:
        known = history_branch(case, case['history_branch_index'])
        if 'history_binding' in case:
            known['history_binding'] = case['history_binding']
    _need(known is not None and canonical(case) == canonical(known),
          'unfrozen request body')
    result = dict(schema=1, mode=MODE, case=case['case'], request_sha256=digest(case['body']),
        raw_response_sha256=None, client_elapsed_seconds=None,
        status='UNREADABLE_RESPONSE', metric_eligible=False, qualification=False,
        code_execution_performed=False, code_correctness_graded=False,
        numerical_equivalence_established=False, normal_stop_is_not_strict_eos_proof=True,
        ttft_measured=False, decode_latency_measured=False)
    try:
        _need(type(client_elapsed_seconds) in (int, float) and math.isfinite(client_elapsed_seconds)
              and client_elapsed_seconds > 0, 'positive finite client duration required')
        result['client_elapsed_seconds'] = client_elapsed_seconds
        _need(type(response) is dict and len(canonical(response)) <= 2 << 20, 'bounded response object required')
        result['raw_response_sha256'] = digest(response)
        if 'error' in response:
            result.update(status='SERVER_ERROR', error=copy.deepcopy(response['error']))
            return result
        choices = response.get('choices')
        _need(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict, 'one choice required')
        choice = choices[0]
        _need(type(choice.get('index')) is int and choice['index'] == 0, 'choice index required')
        message = choice.get('message')
        _need(type(message) is dict and message.get('role') == 'assistant' and
              type(message.get('content')) is str, 'assistant text required')
        meta, usage = choice.get('meta_info'), response.get('usage')
        _need(type(meta) is dict and type(usage) is dict, 'terminal metadata and usage required')
        if case['_kv_block'] == 'paired':
            _need(response.get('id') == meta.get('id') == case['body']['rid'], 'paired wire RID differs')
        _need(_ids(choice.get('prompt_token_ids')) and _ids(choice.get('response_token_ids')),
              'retained prompt and response token vectors required')
        finish = choice.get('finish_reason')
        _need(type(finish) is str and type(meta.get('finish_reason')) is dict and
              meta['finish_reason'].get('type') == finish, 'finish representations differ')
        for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'cached_tokens'):
            _need(_count(meta.get(key)), 'missing or invalid terminal count: ' + key)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            _need(_count(usage.get(key)), 'missing or invalid usage: ' + key)
        _need(type(meta.get('e2e_latency')) in (int, float) and
              math.isfinite(meta['e2e_latency']) and meta['e2e_latency'] >= 0,
              'finite terminal e2e_latency metadata required')
        _need(0 < meta['completion_tokens'] <= case['body']['max_completion_tokens'], 'completion cap differs')
        if mtp_contract.settings().get('quality_screen') == 'sprint35-long-prefix-reuse-v1':
            policy = s35_cache_policy(case['case'])
            result.update(cached_tokens=meta['cached_tokens'], cache_policy=copy.deepcopy(policy))
            _need(policy['minimum_cached_tokens'] <= meta['cached_tokens'] <= policy['maximum_cached_tokens'], 'S35 cache hit outside frozen bounds')
        elif case['_kv_block'] == 'lifecycle':
            policy = expected_case(case['case'])['cache_policy']
            result.update(cached_tokens=meta['cached_tokens'], cache_policy=copy.deepcopy(policy))
            _need(policy['minimum_cached_tokens'] <= meta['cached_tokens'] <= policy['maximum_cached_tokens'],
                  'observed native cached_tokens violates frozen ' + policy['intent'] + ' bounds')
        else:
            _need(meta['cached_tokens'] == 0, 'final verification requires zero cache hits')
        _need(type(meta.get('num_retractions')) is int and meta['num_retractions'] == 0,
              'final verification requires observed zero retractions')
        _need(meta['prompt_tokens'] == usage['prompt_tokens'] == len(choice['prompt_token_ids']) and
              meta['completion_tokens'] == usage['completion_tokens'] == len(choice['response_token_ids']) and
              usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
              'terminal token counts differ')
        frozen = expected_for_case(case)
        if frozen is not None:
            _need(choice['prompt_token_ids'] == frozen['input_ids'], 'frozen prompt IDs differ')
        result.update(finish_reason=finish, matched_stop=choice.get('matched_stop'),
            terminal_meta_info=copy.deepcopy(meta), usage=copy.deepcopy(usage),
            prompt_token_ids_sha256=digest(choice['prompt_token_ids']), prompt_token_count=len(choice['prompt_token_ids']),
            response_token_ids=list(choice['response_token_ids']),
            response_id_count=len(choice['response_token_ids']),
            reported_completion_tokens=meta['completion_tokens'],
            id_count_equals_reported_count=len(choice['response_token_ids']) == meta['completion_tokens'],
            id_eos_inclusion='UNPROVEN', text=message['content'],
            server_timing_interpretation='raw terminal metadata retained; units/scope not asserted',
            frozen_history_exact_answer=(message['content'] == 'SPRUCE-7|32') if case['case'] == 'frozen-two-turn' else None)
        sampled = mtp_contract.settings()['phase'] in SAMPLING_PHASES
        if sampled:
            result.update(sampled_request=not (mtp_contract.greedy_quality_draft_ab() or mtp_contract.precision_reference_enabled()),
                sampling_parameters={key: case['body'][key] for key in ('temperature', 'top_p', 'top_k', 'seed')
                                     if key in case['body']},
                sampling_distribution_verified=False,
                expected_answer_matches=None,
                answer_grade_status='PENDING_POST_RELEASE_NATIVE_DECODE')
        if 'history_binding' in case:
            result['generated_history'] = copy.deepcopy(case['history_binding'])
        if finish == 'length' and case['_kv_block'] == 'paired' and screen_policy() == 'exploratory-stop-or-length-v1':
            _need(meta['completion_tokens'] == case['body']['max_completion_tokens'],
                  'length finish does not match frozen output cap')
            if message.get('tool_calls') or message.get('reasoning_content'):
                result['status'] = 'UNREQUESTED_TOOL_OR_REASONING_REFUSAL'
            else:
                result.update(status='OUTPUT_CAP_REACHED', output_cap_reached=True,
                    screen_policy=screen_policy(), metric_eligible=False,
                    answer_grade_status='UNGRADED_CAP_WORK', native_full_wire_decode_required=True)
        elif finish != 'stop':
            result['status'] = 'STOP_LENGTH_REFUSAL' if finish == 'length' else 'STOP_REFUSAL'
        elif message.get('tool_calls') or message.get('reasoning_content'):
            result['status'] = 'UNREQUESTED_TOOL_OR_REASONING_REFUSAL'
        else:
            segmentation = split_thinking_ids(choice['response_token_ids'])
            result.update(status='NORMAL_STOP_COMPLETE', metric_eligible=True,
                thinking_segments=segmentation, answer_grade_status='PENDING_POST_RELEASE_NATIVE_DECODE',
                native_full_wire_decode_required=True, natural_answer_graded=False)
    except (ValueError, KeyError, TypeError) as error:
        result.update(status='STRUCTURAL_REFUSAL', error=type(error).__name__ + ': ' + str(error)[:512])
    return result


def inspect_speculation(response):
    """Validate the exact native SGLang speculative-response extension.

    ``spec_verify_ct`` is target verification-pass count.  The native producer
    derives proposed drafts separately, and accepted drafts exclude a target
    bonus token.  Neither relationship is a completion-token count.
    """
    _need(type(response) is dict and type(response.get('sglext')) is dict,
          'MTP response extension required')
    details = response['sglext'].get('spec_tokens_details')
    _need(type(details) is dict, 'MTP speculation details required')
    _need(frozenset(details) == SPECULATION_KEYS, 'unexpected native MTP telemetry schema')
    for key in SPECULATION_COUNT_KEYS:
        _need(type(details.get(key)) is int and details[key] >= 0,
              'invalid MTP speculation count: ' + key)
    # The native OpenAI formatter defaults independently absent metrics to
    # zero.  Retain those source-produced values; do not invent a relation
    # between proposal, verification, and accepted-draft counters here.
    for key in SPECULATION_FLOAT_KEYS:
        value = details[key]
        _need(type(value) is float and math.isfinite(value) and value >= 0,
              'invalid MTP speculation metric: ' + key)
    for key in SPECULATION_HISTOGRAM_KEYS:
        value = details[key]
        _need(type(value) is list and len(value) <= 256 and
              all(type(item) is int and item >= 0 for item in value),
              'invalid MTP speculation histogram: ' + key)
    return copy.deepcopy(details)


def stream_events(raw, *, final):
    """Parse complete native SSE frames and retain their exact ending offsets."""
    _need(type(raw) is bytes and len(raw) <= MAX_STREAM_BYTES, 'bounded SSE bytes required')
    events, offset, done = [], 0, False
    for match in re.finditer(br'\r?\n\r?\n', raw):
        frame = raw[offset:match.start()]
        offset = match.end()
        _need(frame and not done and len(events) < MAX_STREAM_CHUNKS, 'empty, post-DONE or excessive SSE frame')
        # This client never requests named/framed token-ID events.
        _need(frame.startswith(b'data: ') and b'\n' not in frame and b'\r' not in frame,
              'unexpected native SSE fields')
        data = frame[6:].decode('utf-8')
        if data == '[DONE]':
            value, done = None, True
        else:
            def pairs(items):
                value = {}
                for key, item in items:
                    _need(key not in value, 'duplicate SSE JSON key')
                    value[key] = item
                return value
            value = json.loads(data, object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite SSE JSON')))
            _need(type(value) is dict and 'error' not in value, 'native SSE error or non-object')
            pending = [(value, 0)]
            while pending:
                item, depth = pending.pop()
                _need(depth <= 64, 'SSE JSON nesting bound')
                if isinstance(item, dict):
                    pending.extend((child, depth + 1) for child in item.values())
                elif isinstance(item, list):
                    pending.extend((child, depth + 1) for child in item)
                elif isinstance(item, float):
                    _need(math.isfinite(item), 'nonfinite SSE number')
        events.append(dict(value=value, end=offset))
    _need(not final or (offset == len(raw) and done), 'incomplete SSE frame or missing DONE')
    _need(not done or offset == len(raw), 'data after SSE DONE')
    return events


def generated_stream_event(value):
    choices = value.get('choices') if type(value) is dict else None
    return (type(choices) is list and len(choices) == 1 and type(choices[0]) is dict and
            type(choices[0].get('delta')) is dict and
            type(choices[0]['delta'].get('content')) is str and bool(choices[0]['delta']['content']))


def inspect_stream_response(case, raw, client_elapsed_seconds):
    """Read native stream fields without inventing absent non-stream metadata."""
    expected = {row['case']: row for row in requests()}
    _need(case['case'] in expected and canonical(case) == canonical(expected[case['case']]) and
          mtp_contract.settings()['phase'] == COMBINED_PHASE and case['body'].get('stream') is True,
          'unfrozen combined stream request')
    result = dict(schema=1, mode=MODE, case=case['case'], request_sha256=digest(case['body']),
        raw_response_sha256=hashlib.sha256(raw).hexdigest(), raw_http_response_sha256=hashlib.sha256(raw).hexdigest(),
        client_elapsed_seconds=client_elapsed_seconds, status='STRUCTURAL_REFUSAL', metric_eligible=False,
        qualification=False, source_promotion=False, code_execution_performed=False,
        code_correctness_graded=False, numerical_equivalence_established=False,
        normal_stop_is_not_strict_eos_proof=True, ttft_measured=False, decode_latency_measured=False,
        transport='native-sse', terminal_meta_info=None, cached_tokens=None, num_retractions=None,
        server_e2e_latency=None, metadata_availability='native SSE omits terminal meta_info and retraction count')
    try:
        _need(type(client_elapsed_seconds) in (int, float) and math.isfinite(client_elapsed_seconds) and
              client_elapsed_seconds > 0, 'positive finite client duration required')
        events = stream_events(raw, final=True)
        _need(events and events[-1]['value'] is None, 'terminal SSE DONE required')
        identity, text, terminal, extension, usage, role = None, [], None, None, None, False
        generated = 0
        for event in events[:-1]:
            value = event['value']
            _need(type(value) is dict and set(value) <= {'id', 'object', 'created', 'model', 'choices', 'usage', 'sglext'},
                  'unexpected native stream fields')
            _need(type(value.get('id')) is str and value['id'] and value.get('model') == case['body']['model'] and
                  value.get('object') == 'chat.completion.chunk' and _count(value.get('created')),
                  'native SSE identity differs')
            if identity is None:
                identity = value['id']
            _need(value['id'] == identity, 'SSE response ID changed')
            choices = value.get('choices')
            _need(type(choices) is list and len(choices) <= 1, 'SSE choices differ')
            if choices:
                _need(terminal is None and extension is None and usage is None, 'SSE choice after terminal')
                choice = choices[0]
                _need(type(choice) is dict and set(choice) <= {'index', 'delta', 'logprobs', 'finish_reason', 'matched_stop'} and
                      type(choice.get('index')) is int and choice['index'] == 0 and choice.get('logprobs') is None,
                      'unexpected stream choice')
                delta = choice.get('delta')
                _need(type(delta) is dict and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'} and
                      delta.get('role') in (None, 'assistant') and delta.get('reasoning_content') in (None, '') and
                      delta.get('tool_calls') in (None, []) and
                      (delta.get('content') is None or type(delta['content']) is str), 'tool/reasoning/invalid SSE delta')
                if delta.get('role') is not None:
                    _need(not role and not text and delta.get('content') in (None, ''), 'duplicate or late assistant role')
                    role = True
                content, finish = delta.get('content'), choice.get('finish_reason')
                _need(role, 'SSE content before assistant role')
                _need(content or finish is not None or delta.get('role') == 'assistant',
                      'unrequested empty SSE choice')
                if content:
                    _need(finish is None, 'content in native finish frame')
                    text.append(content)
                    generated += 1
                if finish is not None:
                    _need(type(finish) is str and not content, 'invalid native finish frame')
                    terminal = dict(finish_reason=finish, matched_stop=choice.get('matched_stop'))
                _need(value.get('sglext') is None and value.get('usage') is None, 'premature stream extension or usage')
            else:
                _need(terminal is not None, 'SSE extension or usage before terminal')
                current_extension, current_usage = value.get('sglext'), value.get('usage')
                _need((current_extension is not None) != (current_usage is not None), 'empty or mixed terminal SSE frame')
                if current_extension is not None:
                    _need(extension is None and usage is None and type(current_extension) is dict and
                          set(current_extension) == {'input_ids', 'output_ids', 'spec_tokens_details'},
                          'duplicate, reordered or unexpected native SSE extension')
                    extension = current_extension
                else:
                    _need(extension is not None and usage is None and type(current_usage) is dict,
                          'duplicate or reordered native SSE usage')
                    usage = current_usage
        _need(role and generated > 0 and terminal is not None and extension is not None and usage is not None,
              'missing stream content, terminal, IDs or usage')
        input_ids, output_ids = extension['input_ids'], extension['output_ids']
        _need(_ids(input_ids) and type(output_ids) is list and len(output_ids) == 1 and
              _ids(output_ids[0], maximum=case['body']['max_completion_tokens']), 'invalid native stream IDs')
        frozen = expected_case(case['case'])
        _need(input_ids == frozen['input_ids'], 'frozen stream prompt IDs differ')
        _need(set(usage) <= {'prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'prompt_tokens_details'} and
              all(_count(usage.get(key)) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')) and
              usage['prompt_tokens'] == len(input_ids) and usage['completion_tokens'] == len(output_ids[0]) and
              usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'] and
              _count(usage.get('reasoning_tokens', 0)) and usage.get('reasoning_tokens', 0) == 0,
              'stream usage/count disagreement')
        cached = usage.get('prompt_tokens_details')
        if cached is not None:
            _need(type(cached) is dict and set(cached) <= {'cached_tokens', 'image_tokens', 'audio_tokens', 'video_tokens'} and
                  _count(cached.get('cached_tokens')) and cached['cached_tokens'] == 0 and
                  all(cached.get(key) in (None, 0) for key in ('image_tokens', 'audio_tokens', 'video_tokens')),
                  'stream usage cache or modality counts differ')
            result['cached_tokens'] = cached['cached_tokens']
        complete_text = ''.join(text)
        result.update(**terminal, usage=copy.deepcopy(usage), prompt_token_ids=list(input_ids),
            response_token_ids=list(output_ids[0]), response_id_count=len(output_ids[0]),
            reported_completion_tokens=usage['completion_tokens'], id_count_equals_reported_count=True,
            id_eos_inclusion='UNPROVEN', text=complete_text, response_id=identity,
            generated_content_chunks=generated, stream_done=True,
            spec_tokens_details=inspect_speculation(dict(sglext=extension)))
        if terminal['finish_reason'] != 'stop':
            result['status'] = 'STOP_LENGTH_REFUSAL' if terminal['finish_reason'] == 'length' else 'STOP_REFUSAL'
        elif 'expected_answer' in frozen and complete_text.strip() != frozen['expected_answer']:
            result['status'] = 'KNOWN_KEY_REFUSAL'
        else:
            result.update(status='NORMAL_STOP_COMPLETE', metric_eligible=True)
    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as error:
        result.update(status='STRUCTURAL_REFUSAL', error=type(error).__name__ + ': ' + str(error)[:512])
    return result


def inspect_stream_observation(case, raw, observation, *, ordinal, prepared, sent, headers, received):
    """Recompute receive identities and the first complete nonempty delta receipt."""
    fields = {'schema', 'ordinal', 'case', 'chunks', 'stream_eof', 'failure',
              'first_generated_monotonic', 'generated_prefix_bytes', 'generated_prefix_sha256'}
    _need(type(observation) is dict and set(observation) == fields and observation['schema'] == 1 and
          observation['ordinal'] == ordinal and observation['case'] == case['case'] and
          observation['stream_eof'] is True and observation['failure'] is None, 'stream observation binding')
    times = [prepared, sent, headers, received]
    _need(all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in times) and
          times == sorted(times), 'stream request chronology')
    chunks = observation['chunks']
    _need(type(chunks) is list and 0 < len(chunks) <= MAX_STREAM_CHUNKS, 'stream receive count bound')
    offset, previous_time, first = 0, headers, None
    generated_ends = [event['end'] for event in stream_events(raw, final=True)
                      if generated_stream_event(event['value'])]
    _need(generated_ends, 'no generated stream event')
    for chunk in chunks:
        _need(type(chunk) is dict and set(chunk) == {'end', 'sha256', 'received_monotonic'} and
              type(chunk['end']) is int and offset < chunk['end'] <= len(raw) and
              chunk['sha256'] == hashlib.sha256(raw[offset:chunk['end']]).hexdigest(), 'stream receive byte identity')
        tick = chunk['received_monotonic']
        _need(type(tick) in (int, float) and math.isfinite(tick) and previous_time <= tick <= received,
              'stream receive chronology')
        if first is None and generated_ends[0] <= chunk['end']:
            first = chunk
        offset, previous_time = chunk['end'], tick
    _need(offset == len(raw) and first is not None and
          observation['first_generated_monotonic'] == first['received_monotonic'] and
          observation['generated_prefix_bytes'] == first['end'] and
          observation['generated_prefix_sha256'] == hashlib.sha256(raw[:first['end']]).hexdigest(),
          'first generated stream prefix binding')
    _need(first['received_monotonic'] > prepared, 'positive TTFT required')
    return dict(ttft_measured=True, client_ttft_seconds=first['received_monotonic'] - prepared,
                ttft_origin='client prepared_monotonic before request transmission',
                ttft_endpoint='receipt containing first complete nonempty native content delta',
                stream_observation_validated=True, receive_chunk_count=len(chunks))


def compare(left, right):
    _need(left.get('case') == right.get('case') and left.get('mode') == right.get('mode') == MODE,
          'paired case/mode differs')
    result = dict(schema=1, mode=MODE, case=left['case'], qualification=False,
        numerical_equivalence_established=False, comparison_eligible=False,
        client_seconds=dict(control=left.get('client_elapsed_seconds'), mtp=right.get('client_elapsed_seconds')),
        completion_tokens=dict(control=left.get('reported_completion_tokens'), mtp=right.get('reported_completion_tokens')),
        interpretation=('same accepted candidate repeated; client request-to-terminal; stream TTFT reported separately; no speedup claim'
            if mtp_contract.settings()['phase'] == COMBINED_PHASE else
            'non-stream client request-to-terminal latency, not decode TPOT or TTFT'))
    if not (left.get('metric_eligible') is right.get('metric_eligible') is True):
        result['reason'] = 'one or both responses did not complete normally'
    elif left['request_sha256'] != right['request_sha256'] or left['prompt_token_ids'] != right['prompt_token_ids']:
        result['reason'] = 'different request bytes or actual prompt IDs'
    else:
        result.update(
            full_response_ids_equal=left['response_token_ids'] == right['response_token_ids'],
            generated_id_difference_count=(sum(left_id != right_id for left_id, right_id in
                zip(left['response_token_ids'], right['response_token_ids'])) +
                abs(len(left['response_token_ids']) - len(right['response_token_ids']))),
            equal_output_counts_do_not_establish_quality_or_numerical_equivalence=True)
        if mtp_contract.settings()['phase'] == SAMPLING_PHASE:
            result.update(sampled_request=True, sampling_distribution_verified=False,
                seed_reproducibility_observed=result['full_response_ids_equal'] and left['text'] == right['text'],
                interpretation='same requested seed repeated across two native sampling starts; output equality is descriptive',
                reason='sampled outputs retained without a cross-start exactness or speedup gate')
            return result
        if mtp_contract.settings()['phase'] in HIP_PHASE_ARMS:
            greedy = mtp_contract.greedy_quality_draft_ab()
            result.update(sampled_request=not greedy, sampling_distribution_verified=False,
                request_seed_present=False,
                interpretation=('greedy target-only versus greedy DFlash; quality and output differences are data'
                    if greedy else 'native sampling versus HIP DFlash sampling; stochastic output differences are data'),
                reason='outputs retained without a cross-arm exactness or paired-output speedup gate')
            return result
        if left['reported_completion_tokens'] != right['reported_completion_tokens']:
            result['reason'] = 'different actual completion counts; retain raw latency without a paired speed claim'
        elif not result['full_response_ids_equal'] or left['text'] != right['text']:
            result['reason'] = 'different generated IDs or text; no paired speed claim'
        else:
            result.update(comparison_eligible=True,
                control_over_mtp_client_latency=left['client_elapsed_seconds'] / right['client_elapsed_seconds'])
    return result


def _validate_closure_schedule(cases, settings):
    """Validate the independent frozen lifecycle screen, not the old five rows."""
    boundary = tuple('boundary-%d-%s' % (offset, suffix)
                     for offset in (4095, 4096, 4097)
                     for suffix in ('cold-a', 'branch-b', 'replay-a', 'isolated-b'))
    names = boundary + ('victim-cold-a', 'victim-warm-a') + tuple(
        'pressure-%02d' % index for index in range(6)) + (
        'victim-replay-a', 'victim-recovery-a', 'cancel-victim', 'cancel-recovery-a',
        'concurrent-a', 'concurrent-b', 'final-replay-a')
    _need(settings['mode'] == MODE and tuple(row['case'] for row in cases) == names,
          'closure mode/order/count differs')
    frozen = [expected_case(name) for name in names]
    bodies = [case['body'] for case in cases]
    vectors = [row['input_ids'] for row in frozen]
    groups = []
    for start, offset in zip((0, 4, 8), (4095, 4096, 4097)):
        a, b, replay, isolated = vectors[start:start + 4]
        _need(a == replay and b == isolated and a != b, 'closure boundary native prompt pairing')
        lcp = 0
        for left, right in zip(a, b):
            if left != right:
                break
            lcp += 1
        _need(lcp == offset and min(len(a), len(b)) > offset,
              'closure native checkpoint-boundary offset differs')
        _need(bodies[start] == bodies[start + 2], 'closure boundary replay body differs')
        strip = lambda body: {key: value for key, value in body.items() if key != 'cache_salt'}
        _need(strip(bodies[start + 1]) == strip(bodies[start + 3]), 'closure isolated branch changed settings')
        shared, isolated_salt = bodies[start]['cache_salt'], bodies[start + 3]['cache_salt']
        _need(shared == bodies[start + 1]['cache_salt'] and shared != isolated_salt,
              'closure boundary namespace binding')
        groups.extend((shared, isolated_salt))
    victim = bodies[12]['cache_salt']
    groups.append(victim)
    for ordinal in (12, 13, 20, 21, 22, 23, 24, 26):
        _need(vectors[ordinal] == vectors[4] and bodies[ordinal]['messages'] == bodies[4]['messages'] and
              bodies[ordinal]['cache_salt'] == victim, 'closure victim/recovery prompt or namespace changed')
    _need(vectors[25] == vectors[5] and bodies[25]['messages'] == bodies[5]['messages'] and
          bodies[25]['cache_salt'] == victim, 'closure concurrent branch binding')
    for ordinal in range(14, 20):
        _need(8128 <= len(vectors[ordinal]) <= 8192, 'closure pressure occupied band differs')
        groups.append(bodies[ordinal]['cache_salt'])
    _need(len(groups) == len(set(groups)) and all(type(value) is str and value for value in groups),
          'closure independent pressure/boundary namespaces differ')
    _need(sum(len(vectors[index]) for index in range(14, 20)) > mtp_contract.pool_token_capacity() and
          mtp_contract.pool_token_capacity() in (32768, 32800),
          'closure pressure does not exceed the fixed pool')
    # Only two requests can be active. Do not mistake deliberate accumulated
    # cache pressure for a larger live request allocation or increase capacity.
    _need(len(vectors[24]) + len(vectors[25]) + 2 * 128 + 32 < 32768,
          'closure simultaneous prompt/output envelope exceeds pool')
    for ordinal, (case, row, expected, ids) in enumerate(zip(cases, settings['requests'], frozen, vectors)):
        body, info = case['body'], row.get('stress_info')
        _need(type(info) is dict and info == expected.get('stress_info') and
              info.get('terminal_expectation') == ('cancelled' if ordinal == 22 else 'normal'),
              'closure lifecycle scenario metadata differs')
        _need(expected.get('input_tokens') == len(ids) == row['input_tokens'] and
              expected.get('input_ids_sha256') == digest(ids) and
              expected.get('messages_sha256') == digest(body['messages']),
              'closure native ID/message digest differs')
        cancel = ordinal == 22
        _need(body.get('max_completion_tokens') == row['maximum_output_tokens'] == (512 if cancel else 128) and
              body.get('ignore_eos') is cancel and body.get('stream') is cancel and
              body.get('temperature') == 0 and body.get('seed') == 1729 and body.get('n') == 1,
              'closure frozen sampling/stopping differs')
        if cancel:
            _need(body.get('rid') == 'fp8-closure-64Q8v0X1-cancel', 'closure cancel request ID differs')
            _need(body.get('return_token_ids') is False and body.get('return_meta_info') is False and
                  body.get('return_prompt_token_ids', False) is False and
                  body.get('return_input_ids_in_sglext') is True and
                  body.get('return_output_ids_in_sglext') is True,
                  'closure native streaming response flags differ')
        else:
            _need('rid' not in body, 'unrequested closure request ID')
            _need(body.get('return_token_ids') is True and body.get('return_meta_info') is True,
                  'closure normal response evidence flags differ')
        policy = row.get('cache_policy')
        _need(type(policy) is dict and policy == expected.get('cache_policy') and
              type(policy.get('minimum_cached_tokens')) is int and
              type(policy.get('maximum_cached_tokens')) is int and
              0 <= policy['minimum_cached_tokens'] <= policy['maximum_cached_tokens'] < len(ids) and
              policy.get('prompt_pair_sha256') == digest({key: value for key, value in body.items() if key != 'cache_salt'}),
              'closure frozen cache policy differs')
        cold = ordinal in (0, 3, 4, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19)
        if cold:
            _need(policy['minimum_cached_tokens'] == policy['maximum_cached_tokens'] == 0,
                  'closure cold namespaces must require zero hits')
        if ordinal in (1, 5, 9, 25):
            maximum = (4095, 4096, 4097)[ordinal // 4] if ordinal < 12 else 4096
            _need(policy['maximum_cached_tokens'] <= maximum, 'closure branch exceeds actual LCP')
        if ordinal in (2, 6, 10, 13, 21, 23, 24, 26):
            _need(policy['minimum_cached_tokens'] >= 1, 'closure replay requires observed reuse')
        branch = ordinal in (1, 3, 5, 7, 9, 11, 25)
        answer = 'SPRUCE-7|POST /jobs|X-Queue-Key|' + ('64|192' if branch else '48|144')
        _need(expected.get('expected_answer') == answer and
              expected.get('task_grade') == dict(kind='exact_text', expected=answer),
              'closure task oracle differs')


def compare_closure(cold, warm):
    """Compare only frozen paired identities; do not infer internal state equality."""
    _need(mtp_contract.settings()['phase'] in mtp_contract.CLOSURE_PHASES and
          cold.get('mode') == warm.get('mode') == MODE, 'explicit closure comparison mode required')
    cases = {row['case']: row for row in requests()}
    _need(cold.get('case') in cases and warm.get('case') in cases, 'unknown closure comparison case')
    first, second = expected_case(cold['case']), expected_case(warm['case'])
    _need(second['stress_info'].get('comparison_case') == cold['case'] and
          first['cache_policy']['intent'] == 'cold' and
          first['stress_info']['terminal_expectation'] == second['stress_info']['terminal_expectation'] == 'normal',
          'undeclared closure normal-response pair')
    for result, frozen in ((cold, first), (warm, second)):
        _need(result.get('request_sha256') == digest(cases[result['case']]['body']) and
              result.get('prompt_token_ids') == frozen['input_ids'], 'closure pair frozen request/IDs differ')
    strip = lambda body: {key: value for key, value in body.items() if key != 'cache_salt'}
    same = (first['cache_policy']['prompt_pair_sha256'] == second['cache_policy']['prompt_pair_sha256'] and
            cold['prompt_token_ids'] == warm['prompt_token_ids'] and
            strip(cases[cold['case']]['body']) == strip(cases[warm['case']]['body']))
    value = _compare_outputs(cold, warm, same)
    value.update(comparison_scope='frozen-closure-output-parity', cold_case=cold['case'], warm_case=warm['case'],
                 cached_tokens=dict(cold=cold.get('cached_tokens'), warm=warm.get('cached_tokens')),
                 performance_claim=False)
    value.pop('control_over_mtp_client_latency', None)
    return value


def _compare_outputs(left, right, same_request):
    result = dict(schema=1, mode=MODE, case=left['case'], qualification=False,
        numerical_equivalence_established=False, comparison_eligible=False,
        client_seconds=dict(control=left.get('client_elapsed_seconds'), mtp=right.get('client_elapsed_seconds')),
        completion_tokens=dict(control=left.get('reported_completion_tokens'), mtp=right.get('reported_completion_tokens')),
        interpretation='non-stream client request-to-terminal latency, not decode TPOT or TTFT')
    if not (left.get('metric_eligible') is right.get('metric_eligible') is True):
        result['reason'] = 'one or both responses did not complete normally'
    elif not same_request:
        result['reason'] = 'different request bytes or actual prompt IDs'
    else:
        result.update(
            full_response_ids_equal=left['response_token_ids'] == right['response_token_ids'],
            generated_id_difference_count=(sum(left_id != right_id for left_id, right_id in
                zip(left['response_token_ids'], right['response_token_ids'])) +
                abs(len(left['response_token_ids']) - len(right['response_token_ids']))),
            equal_output_counts_do_not_establish_quality_or_numerical_equivalence=True)
        if left['reported_completion_tokens'] != right['reported_completion_tokens']:
            result['reason'] = 'different actual completion counts; retain raw latency without a paired speed claim'
        elif not result['full_response_ids_equal'] or left['text'] != right['text']:
            result['reason'] = 'different generated IDs or text; no paired speed claim'
        else:
            result.update(comparison_eligible=True,
                control_over_mtp_client_latency=left['client_elapsed_seconds'] / right['client_elapsed_seconds'])
    return result


def paired_requests():
    settings=mtp_contract.settings()
    if 'configuration' in settings:return configured_requests(settings)
    _need(len(_KV_ROWS)==len(settings['requests'])==176,'frozen W5 v2 requests')
    b1={171: {'body_sha256': '6b990ba41dc6a4a0a071f89193e4444da50313ffe920786f244bb6f124c73144', 'cache_salt': 'b1-warmup-00', 'rid': 'b1-00-warmup-c1'}, 172: {'body_sha256': '4082594c2166d892cdb07cc9fd3b3250f2232fea9358f88045abe65112dc9ceb', 'cache_salt': 'b1-warmup-01', 'rid': 'b1-01-warmup-c2-a'}, 173: {'body_sha256': '9af35fa8c0a52f441ab3d8add8297c2457687015e59d4297f801779804743af2', 'cache_salt': 'b1-warmup-02', 'rid': 'b1-02-warmup-c2-b'}, 174: {'body_sha256': 'da015df9b9e103faeefa7a76dba157348d888277c37d43a7aafaec7924325105', 'cache_salt': 'b1-pair-a', 'rid': 'b1-08-pair-c2-a'}, 175: {'body_sha256': 'd3471f9454f633673a6f667e05342d3d49358b745833a752dca3c8b4f9aadbfa', 'cache_salt': 'b1-pair-b', 'rid': 'b1-09-pair-c2-b'}}
    result=[];identities={};salts=set();rids=set()
    for plan,stored in zip(settings['requests'],_KV_ROWS,strict=True):
        body=json.loads(mtp_contract.asset_bytes(plan['path']))
        expected=json.loads(mtp_contract.asset_bytes(plan['expected']))
        ordinal=plan['ordinal']
        _need(stored['body']==body and stored['body_sha256']==digest(body) and
              stored['expected']==expected and stored['ordinal']==ordinal and stored['block']=='paired',
              'bound request/container join')
        validate_sampling_parameters(body,ordinal)
        ids=expected['input_ids']
        if ordinal in b1:
            row=b1[ordinal]
            _need(digest(body)==row['body_sha256'] and body['cache_salt']==row['cache_salt'] and body['rid']==row['rid'] and
                  body['ignore_eos'] is False and body['chat_template_kwargs']==dict(enable_thinking=True) and
                  body['reasoning_effort']=='low' and body['max_completion_tokens']==plan['maximum_output_tokens']==4096 and
                  body['return_spec_tokens_details'] is True and body['model']==MODEL and
                  body['cache_salt'] not in salts and body['rid'] not in rids,
                  'verbatim B1 v3 (operation 169) request')
            _need(_ids(ids) and len(ids)==plan['input_tokens']==expected['input_tokens'] and
                  digest(ids)==expected['input_ids_sha256'] and digest(body['messages'])==expected['messages_sha256'] and
                  expected['case']==plan['case'] and expected['context_length']==131040 and expected['internal_reserve_tokens']==16 and
                  expected['cache_policy']==dict(intent='cold',minimum_cached_tokens=0,maximum_cached_tokens=0) and
                  expected['completion_policy']=='natural-stop' and expected['task_grade']['kind']=='exact_json',
                  'frozen native IDs, cold cache and natural-stop oracle')
        else:
            _need(body['ignore_eos'] is False and body['chat_template_kwargs']==dict(enable_thinking=True) and
                  body['reasoning_effort']==('xhigh' if plan['task']=='sre-log-report' else 'low') and body['max_completion_tokens']==plan['maximum_output_tokens'] and
                  body['return_spec_tokens_details'] is True and body['model']==MODEL,
                  'natural-stop thinking/sampling/cap contract')
            _need(body['cache_salt']==f'nstop-cold-{ordinal:02d}' and
                  body['rid']==f"nstop-{ordinal:02d}-{plan['case']}" and
                  body['cache_salt'] not in salts and body['rid'] not in rids,
                  'unique cold request salt and RID')
            _need(_ids(ids) and len(ids)==plan['input_tokens']==expected['input_tokens'] and
                  digest(ids)==expected['input_ids_sha256'] and digest(body['messages'])==expected['messages_sha256'] and
                  expected['case']==plan['case'] and expected['context_length']==32768 and
                  expected['internal_reserve_tokens']==16 and
                  expected['cache_policy']==dict(intent='cold',minimum_cached_tokens=0,maximum_cached_tokens=0),
                  'frozen native IDs and cold cache binding')
            _need(expected['task_grade']['kind'] in ('exact_json','exact_text'),'exact non-executing answer oracle')
            task=plan['task']+('-'+plan['case'][-1] if plan['task']=='retrieval' else '')
            identity=({k:v for k,v in body.items() if k not in ('rid','cache_salt')},ids,expected['task_grade'])
            if task in identities:_need(identity==identities[task],'C1/C2/warmup task drift')
            identities[task]=identity
        salts.add(body['cache_salt']);rids.add(body['rid'])
        result.append(dict(case=plan['case'],code_execution_authorized=False,body=body,
                           _kv_block='paired',_kv_ordinal=ordinal))
    _need(len(salts)==len(rids)==176,'complete natural-stop corpus')
    return result


def paired_group_counts(delta, counts, concurrency):
    """Join true unpadded target graph size with per-request verifier telemetry."""
    _need(set(delta)=={'replays','completed_commits','bs1_replays','bs2_replays'} and
        all(type(v) is int and v>=0 for v in delta.values()), 'paired graph counter fields/types')
    _need(concurrency in (1,2) and len(counts) in (1,2) and
        all(type(v) is int and v>0 for v in counts), 'paired verifier count fields')
    _need(delta['completed_commits']==delta['replays']==delta['bs1_replays']+delta['bs2_replays']>0,
        'paired commit/replay accounting')
    _need((delta['bs2_replays']==0 if concurrency==1 else delta['bs2_replays']>0),
        'paired actual native concurrency differs')
    _need(all(delta['bs2_replays']<=v<=delta['replays'] for v in counts) and
        sum(counts)==delta['bs1_replays']+2*delta['bs2_replays'], 'paired weighted verifier participation')
    return dict(delta,request_verifier_counts=list(counts),concurrency=concurrency)


def paired_wire_group(group, timing, responses):
    """Derive combined service throughput from authenticated local receipt times."""
    _need(group in mtp_contract.paired_group_specs() and len(timing)==len(responses)==len(group['ordinals']),
        'paired wire group membership')
    for clock,response in zip(timing,responses,strict=True):
        _need(set(clock)=={'prepared','sent','received'} and
            all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in clock.values()) and
            clock['prepared']<=clock['sent']<clock['received'] and response_completed(response),
            'paired completion/timing contract')
    if group['concurrency']==2:
        _need(max(t['sent'] for t in timing)<min(t['received'] for t in timing), 'paired requests did not overlap on HTTP')
    elif len(timing)==2:
        _need(timing[0]['received']<=timing[1]['prepared'], 'C1 requests overlap')
    start=min(t['prepared'] for t in timing);end=max(t['received'] for t in timing)
    tokens=sum(r['reported_completion_tokens'] for r in responses)
    _need(type(tokens) is int and tokens>0 and end>start, 'paired positive work/duration')
    measured=group['kind']=='timing'
    _need(all(response_completed(r) for r in responses), 'all measured work must satisfy its stop policy')
    return dict(group_id=group['id'],ordinals=list(group['ordinals']),concurrency=group['concurrency'],
        kind=group['kind'],measured=measured,excluded_warmup=group['kind']=='warmup',
        prepared_monotonic=start,received_monotonic=end,makespan_seconds=end-start,
        completion_tokens=tokens,combined_tp2_tokens_per_second=tokens/(end-start) if measured else None,
        per_request_latency_seconds=[t['received']-t['prepared'] for t in timing],
        timing_scope='instrumented HTTP service throughput; combined TP2',native_batch_size_proven=False,
        qualification=False)

def validate_kv_summary(summary, paired_raw=None):
    expected_count=len(mtp_contract.settings()['requests'])
    _need(type(summary) is dict and summary.get('schema')==1 and summary.get('mode')==MODE and
          summary.get('composition')=='configured-paired-v1' and
          summary.get('frozen_case_count')==summary.get('executed_case_count')==expected_count and
          summary.get('stopped_early') is False and summary.get('error') is None and
          summary.get('qualification') is False and summary.get('precision_accepted') is False,
          'complete natural-stop diagnostic summary required')
    blocks=summary.get('blocks')
    _need(type(blocks) is list and len(blocks)==1 and blocks[0]['name']=='paired' and
          blocks[0]['global_offset']==0,'one paired block')
    if mtp_contract.instrumentation_enabled():
        value=blocks[0]['summary']
    else:
        _need(set(blocks[0])=={'name','global_offset','summary_sha256','summary_bytes'} and type(paired_raw) is bytes and
              len(paired_raw)==blocks[0]['summary_bytes'] and hashlib.sha256(paired_raw).hexdigest()==blocks[0]['summary_sha256'],
              'referenced paired summary identity')
        value=json.loads(paired_raw)
        _need(type(value) is dict and canonical(value)==paired_raw,'canonical referenced paired summary')
    rows=value.get('rows');counts=value.get('counts')
    _need(value.get('frozen_case_count')==value.get('executed_case_count')==expected_count and
          type(rows) is list and len(rows)==expected_count and value.get('paired_error') is None and
          counts.get('valid_complete_count')==expected_count and
          counts.get('normal_complete_count')==sum(row.get('contract',{}).get('status')=='NORMAL_STOP_COMPLETE' for row in rows) and
          all(response_completed(row.get('contract',{})) for row in rows),
          'every request must satisfy its source-bound stop policy')
    return summary


def summary_contract(row):
    """W5 v2 production: a summary row names the response IDs and text by digest; the raw response receipt holds them."""
    value=dict(row)
    if type(value.get('response_token_ids')) is list:
        value['response_token_ids_sha256']=digest(value.pop('response_token_ids'))
    if type(value.get('text')) is str:
        raw=value.pop('text').encode()
        value.update(text_sha256=hashlib.sha256(raw).hexdigest(),text_utf8_bytes=len(raw))
    if type(value.get('thinking_segments')) is dict:
        segments=dict(value['thinking_segments'])
        for key in ('final_ids','final_segment_ids'):
            if type(segments.get(key)) is list:
                ids=segments.pop(key);segments.update({key+'_sha256':digest(ids),key+'_count':len(ids)})
        value['thinking_segments']=segments
    return value


def split_thinking_ids(ids):
    if not (type(ids) is list and ids and all(type(x) is int and x>=0 for x in ids)):
        raise ValueError('nonempty integer output vector required')
    if ids.count(248069)!=1:
        raise ValueError('exactly one closing thinking marker required')
    index=ids.index(248069)
    tail=ids[index+1:]
    if any(x in (248068,248069) for x in tail):
        raise ValueError('thinking marker after final boundary')
    final_ids=[x for x in tail if x not in (248046,248044)]
    if not final_ids:
        raise ValueError('final answer vector is empty')
    return dict(reasoning_token_count=index,closing_marker_index=index,final_ids=final_ids,
                final_segment_ids=tail,final_answer_token_count=len(final_ids),output_tokens=len(ids))


def configured_requests(settings):
    # settings() has just reauthenticated both source files. Cache validation
    # by its complete value, including the exact logical-asset source pins.
    return copy.deepcopy(_configured_requests(mtp_contract.json_line(settings)))


@lru_cache(maxsize=1)
def _configured_requests(settings_raw):
    """Validate the generated menu against complete pinned bodies and native IDs."""
    settings=mtp_contract.strict_json(settings_raw)
    plans=settings['requests'];_need(len(plans)==len(_KV_ROWS),'complete configured client binding')
    rows=[];salts=set();rids=set()
    for plan,stored in zip(plans,_KV_ROWS,strict=True):
        body=mtp_contract.strict_json(mtp_contract.asset_bytes(plan['path']))
        expected=mtp_contract.strict_json(mtp_contract.asset_bytes(plan['expected']))
        _need(stored['ordinal']==plan['ordinal'] and stored['case']==plan['case']
              and stored['expected_path']==plan['expected'] and stored['expected_sha256']==digest(expected)==plan['expected_sha256']
              and stored['body_sha256']==digest(body)==plan['sha256'],
              'configured complete asset/body join')
        validate_sampling_parameters(body,plan['ordinal'])
        _need(body['ignore_eos'] is False and body['chat_template_kwargs']==dict(enable_thinking=True)
              and body['reasoning_effort'] in ('low','xhigh') and body['max_completion_tokens']==plan['maximum_output_tokens']
              and body['return_spec_tokens_details'] is True and body['model']==MODEL,'configured thinking/output policy')
        ids=expected['input_ids']
        _need(_ids(ids) and len(ids)==plan['input_tokens']==expected['input_tokens']
              and digest(ids)==expected['input_ids_sha256'] and digest(body['messages'])==expected['messages_sha256']
              and expected['case']==plan['case'] and expected['context_length']==plan['context_envelope_tokens']
              and expected['internal_reserve_tokens']==plan['internal_reserve_tokens']==16,'configured native prompt IDs and context')
        _need(type(body['cache_salt']) is str and type(body['rid']) is str
              and body['cache_salt'] not in salts and body['rid'] not in rids
              and expected['cache_policy']==dict(intent='cold',minimum_cached_tokens=0,maximum_cached_tokens=0),
              'configured unique cold salt/RID')
        salts.add(body['cache_salt']);rids.add(body['rid'])
        rows.append(dict(case=plan['case'],code_execution_authorized=False,body=body,
                         _kv_block='paired',_kv_ordinal=plan['ordinal']))
    return rows
