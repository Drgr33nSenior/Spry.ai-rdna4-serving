"""Project recomputed serving wire rows into the separate CPU decoder input.

This pure projection does not authenticate physical release or grant execution.
The CPU packet binder must authenticate both complete serving owner readouts
and independent releases before admitting this input.
"""
import copy
from quality_contract import menu_rows

MODE = 'normal-stop-paired-native-quality-352-v1'


def need(value, reason):
    if not value:
        raise ValueError('final quality input: ' + reason)


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def project_menu(private_rows, request_rows, *, source_plan=None):
    """Join already authenticated oracle and run rows without retaining cap oracles."""
    need(type(private_rows) is list and type(request_rows) is list and request_rows,
         'complete matching oracle/run menus')
    if source_plan is not None:
        need(type(source_plan) is list and len(source_plan)==len(request_rows),'complete authenticated source plan')
        projected=[]
        for ordinal,(source,request) in enumerate(zip(source_plan,request_rows,strict=True)):
            index=source.get('source_ordinal',ordinal)
            need(type(index) is int and 0<=index<len(private_rows),'declared oracle source ordinal')
            private=private_rows[index]
            need(type(private) is dict and type(private['ordinal']) is int and private['ordinal']==index
                 and private['request_body_sha256']==source['sha256']
                 and type(source['ordinal']) is type(request['ordinal']) is int and source['ordinal']==request['ordinal']==ordinal
                 and source['case']==request['case'] and source['stop_policy']==request['stop_policy']
                 and type(request['maximum_output_tokens']) is type(source['maximum_output_tokens']) is type(private['maximum_output_tokens']) is int
                 and 0<request['maximum_output_tokens']==source['maximum_output_tokens']<=private['maximum_output_tokens']
                 and digest(request['sha256']),'authenticated source oracle and resolved request join')
            projected.append(dict(copy.deepcopy(private),ordinal=ordinal,case=request['case'],
                request_body_sha256=request['sha256'],maximum_output_tokens=request['maximum_output_tokens']))
        private_rows=projected
    need(len(private_rows)==len(request_rows),'complete matching oracle/run menus')
    rows=[]
    for ordinal,(private,request) in enumerate(zip(private_rows,request_rows,strict=True)):
        need(type(private) is dict and type(request) is dict and
             type(private.get('ordinal')) is type(request.get('ordinal')) is int and
             private['ordinal']==request['ordinal']==ordinal and private.get('case')==request.get('case') and
             private.get('request_body_sha256')==request.get('sha256') and
             type(private.get('maximum_output_tokens')) is type(request.get('maximum_output_tokens')) is int and
             private['maximum_output_tokens']==request['maximum_output_tokens'], 'exact oracle/run identity and cap')
        policy=request.get('stop_policy')
        need(policy in ('natural','cap'), 'explicit configured stop policy')
        row=copy.deepcopy(private)
        row['scope']='natural-oracle' if policy=='natural' else 'ungraded-cap-work'
        if policy=='cap': row.update(expected_answer=None,grade_kind=None,minimum_final_tokens=0)
        rows.append(row)
    projected=[dict(copy.deepcopy(row),arm=arm) for arm in ('bf16','kv4') for row in rows]
    menu_rows(projected)
    return projected


def build_input(arm_results, *, menu):
    need(type(arm_results) is dict and set(arm_results) == {'bf16', 'kv4'}, 'exact two owner arms')
    menu_rows(menu)
    count=len(menu)//2
    rows, cancellations = [], []
    for arm, logical in (('bf16', 'control'), ('kv4', 'mtp')):
        result = arm_results[arm]
        need(result['wire_complete'] is True and result['arm'] == logical and
             result['qualification'] is result['precision_accepted'] is False and
             result['answer_grade_status'] == 'PENDING_POST_RELEASE_NATIVE_DECODE' and
             type(result['rows']) is list and len(result['rows']) == count,
             'complete recomputed unqualified serving arm')
        plans = [row for row in menu if row['arm'] == arm]
        for plan, actual in zip(plans, result['rows']):
            ordinal = plan['ordinal']
            need(type(actual['ordinal']) is int and actual['ordinal'] == ordinal and
                 actual.get('case', plan['case']) == plan['case'] and
                 actual['request_body_sha256'] == plan['request_body_sha256'] and
                 digest(actual['response_raw_sha256']), 'exact source ordinal/body/raw response join')
            natural = plan['scope'] == 'natural-oracle'
            terminal=(actual['status'],actual['finish_reason'])
            need((terminal==('NORMAL_STOP_COMPLETE','stop') or
                  not natural and terminal==('OUTPUT_CAP_REACHED','length')) and
                 digest(actual['response_canonical_sha256']) and
                 actual['response_canonical_sha256'] == actual['raw_response_sha256'],
                 'terminal and independent canonical-response join')
            ids = actual['response_token_ids']
            need(type(ids) is list and 0 < len(ids) <= plan['maximum_output_tokens'] and
                 all(type(token) is int and 0 <= token < 248320 for token in ids) and
                 type(actual['text']) is str, 'bounded complete raw token vector and text')
            need(actual['finish_reason']!='length' or len(ids)==plan['maximum_output_tokens'],
                 'length-stopped cap-work requires the exact declared token count')
            rows.append(dict(arm=arm, ordinal=ordinal, case=plan['case'], scope=plan['scope'],
                output_token_ids=list(ids), wire_text=actual['text'], finish_reason=actual['finish_reason'],
                expected_answer=plan['expected_answer'], request_body_sha256=actual['request_body_sha256'],
                response_raw_sha256=actual['response_raw_sha256'],
                response_canonical_sha256=actual['response_canonical_sha256'],
                grade_kind=plan['grade_kind'],minimum_final_tokens=plan['minimum_final_tokens']))
    need(len(rows) == len(menu) and len(cancellations) == 0, 'all declared responses; no cancellation cases')
    return dict(schema=1, mode=MODE, rows=rows, cancellations=cancellations,
                qualification=False, precision_accepted=False)
