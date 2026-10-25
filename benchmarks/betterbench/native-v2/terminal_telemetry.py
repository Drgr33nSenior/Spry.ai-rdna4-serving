"""Validate selective native terminal records, not a full raw SSE transcript."""
import hashlib
import json
import math
import re

COUNT_KEYS = ('spec_verify_ct', 'spec_num_proposed_drafts', 'spec_num_correct_drafts')
FLOAT_KEYS = ('spec_accept_rate', 'spec_accept_length', 'spec_cap_length', 'spec_block_accept_length')
HISTOGRAM_KEYS = ('spec_correct_drafts_histogram', 'spec_cap_lens_histogram')


def _bb_need(value, message):
    if not value:
        raise ValueError('native terminal protocol: ' + message)


def _bb_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _bb_need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite terminal JSON')))


def _bb_frame(raw, pin):
    _bb_need(type(raw) is str and 0 < len(raw.encode('utf-8')) <= 16384, 'bounded terminal frame')
    _bb_need(type(pin) is str and hashlib.sha256(raw.encode('utf-8')).hexdigest() == pin, 'actual frame hash')
    return _bb_json(raw)


def validate_native_terminal(result, *, model):
    _bb_need(type(result) is dict and result.get('ok') is True, 'complete measured result required')
    response_id = result.get('native_response_id')
    _bb_need(type(response_id) is str and 0 < len(response_id) <= 128, 'actual response identity')
    protocol, observation = result.get('native_protocol'), result.get('native_speculation')
    fields = {'schema', 'data_frames', 'metadata_frames', 'usage_frames', 'finish_markers', 'done_markers',
              'metadata_index', 'usage_index', 'finish_index', 'done_index', 'finish_frame', 'finish_frame_sha256',
              'usage_frame', 'usage_frame_sha256', 'done_frame', 'done_frame_sha256', 'complete'}
    _bb_need(type(protocol) is dict and set(protocol) == fields and type(protocol['schema']) is int and protocol['schema'] == 1
             and protocol['complete'] is True, 'source-authenticated terminal protocol record')
    for key in ('metadata_frames', 'usage_frames', 'finish_markers', 'done_markers'):
        _bb_need(type(protocol[key]) is int and protocol[key] == 1, 'exactly one terminal marker ' + key)
    positions = [protocol[key] for key in ('finish_index', 'metadata_index', 'usage_index', 'done_index')]
    _bb_need(all(type(value) is int and value > 0 for value in positions)
             and all(a < b for a, b in zip(positions, positions[1:]))
             and type(protocol['data_frames']) is int and positions[-1] == protocol['data_frames'],
             'source-observed ordered terminal marker positions')
    _bb_need(type(observation) is dict and set(observation) == {'response_id', 'raw_frame', 'raw_frame_sha256', 'details'}
             and observation['response_id'] == response_id, 'actual speculation observation identity')
    metadata = _bb_frame(observation['raw_frame'], observation['raw_frame_sha256'])
    finish = _bb_frame(protocol['finish_frame'], protocol['finish_frame_sha256'])
    usage = _bb_frame(protocol['usage_frame'], protocol['usage_frame_sha256'])
    for value in (metadata, finish, usage):
        _bb_need(type(value) is dict and value.get('id') == response_id and value.get('model') == model,
                 'same actual model/response identity across terminal frames')
    _bb_need(metadata.get('choices') == [] and type(metadata.get('sglext')) is dict
             and json.dumps(metadata['sglext'].get('spec_tokens_details'),sort_keys=True,allow_nan=False)==json.dumps(observation['details'],sort_keys=True,allow_nan=False), 'native sglext details/raw join')
    details = observation['details']
    _bb_need(type(details) is dict and set(details) == set(COUNT_KEYS + FLOAT_KEYS + HISTOGRAM_KEYS),
             'pinned native SpecTokensDetails schema')
    for key in COUNT_KEYS:
        _bb_need(type(details[key]) is int and details[key] >= 0, 'source counter ' + key)
    for key in FLOAT_KEYS:
        _bb_need(type(details[key]) is float and math.isfinite(details[key]) and details[key] >= 0,
                 'source float ' + key)
    for key in HISTOGRAM_KEYS:
        _bb_need(type(details[key]) is list and len(details[key]) <= 256
                 and all(type(value) is int and value >= 0 for value in details[key]), 'source histogram ' + key)
    choices = finish.get('choices')
    _bb_need(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
             and type(choices[0].get('index')) is int and choices[0]['index'] == 0
             and type(result.get('finish_reason')) is str and result['finish_reason']
             and choices[0].get('finish_reason') == result['finish_reason'], 'actual singular completion marker')
    native_usage = usage.get('usage')
    _bb_need(usage.get('choices') == [] and type(native_usage) is dict, 'separate native usage marker')
    for key in ('prompt_tokens', 'completion_tokens'):
        _bb_need(type(native_usage.get(key)) is int and native_usage[key] == result[key] and native_usage[key] > 0,
                 'actual usage/result join ' + key)
    _bb_need(type(native_usage.get('total_tokens')) is int
             and native_usage['total_tokens'] == result['prompt_tokens'] + result['completion_tokens'], 'actual total usage')
    reported_cache = (native_usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    _bb_need(reported_cache == result['cached_tokens'], 'actual reported cache count or explicit unknown')
    _bb_need(protocol['done_frame'] == '[DONE]'
             and hashlib.sha256(protocol['done_frame'].encode()).hexdigest() == protocol['done_frame_sha256'],
             'actual terminal marker identity/hash')
    return dict(response_id=response_id, counts={key: details[key] for key in COUNT_KEYS},
        metadata_frame_sha256=observation['raw_frame_sha256'], finish_frame_sha256=protocol['finish_frame_sha256'],
        usage_frame_sha256=protocol['usage_frame_sha256'], done_frame_sha256=protocol['done_frame_sha256'],
        evidence_scope='Pinned native client protocol-state record and selective terminal frames; full raw content SSE not retained')


def aggregate_native_terminal(rows):
    _bb_need(type(rows) is list and rows, 'actual completed telemetry rows')
    ids = [row['response_id'] for row in rows]
    _bb_need(len(set(ids)) == len(ids), 'unique native response identities')
    return dict(response_count=len(rows), **{key: sum(row['counts'][key] for row in rows) for key in COUNT_KEYS})


def native_worker_identities(workers, expected):
    _bb_need(type(workers) is list and len(workers) == 2 and [row.get('rank') for row in workers] == [0, 1],
             'both already-authenticated native rank identities')
    result = []
    for row in workers:
        _bb_need(all(row.get(key) == value for key, value in expected.items()), 'actual worker/source/request binding')
        _bb_need(type(row.get('pid')) is int and row['pid'] > 0 and type(row.get('start_ticks')) is int and row['start_ticks'] > 0
                 and type(row.get('boot_id')) is str and row['boot_id']
                 and type(row.get('identity_sha256')) is str and re.fullmatch('[0-9a-f]{64}', row['identity_sha256']),
                 'actual worker lifetime identity')
        result.append({key: row[key] for key in ('rank', 'pid', 'start_ticks', 'boot_id', 'identity_sha256')})
    return result
