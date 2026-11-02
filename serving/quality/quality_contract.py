"""Full-wire native decodes, natural-answer checks and explicit ungraded cap work."""
import hashlib
import json

MODE = 'normal-stop-paired-native-quality-352-v1'
LEGACY_INPUT_LIMIT = 1 << 20
INPUT_LIMIT = 53 << 20  # Explicit proposed ordinary input boundary; no execution authority.
REPORT_LIMIT = 4 << 20
WIRE_LIMIT = 512 << 10
TOKENIZER = dict(bytes=12809320, sha256='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3')
HELPER = dict(bytes=872, sha256='231408f571eab7cc90ea500cc9edb525d5f95e5b127fadb628b1cb7916999c29')
ROW_FIELDS = {'arm','ordinal','case','scope','output_token_ids','wire_text','finish_reason','expected_answer',
              'request_body_sha256','response_raw_sha256','response_canonical_sha256','grade_kind','minimum_final_tokens'}
CANCEL_FIELDS = {'arm','ordinal','case','scope','status','request_body_sha256','response_raw_sha256','completed_answer_claimed'}
IDENTITY_FIELDS = ('arm','ordinal','case','scope','request_body_sha256','response_raw_sha256','response_canonical_sha256','finish_reason')


def need(value, reason):
    if not value: raise ValueError('KV final quality: '+reason)


def encoded(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def id_hash(ids): return sha((json.dumps(ids,separators=(',',':'))+'\n').encode())
def digest(value): return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def strict(raw):
    def pairs(rows):
        out={}
        for key,value in rows:
            need(key not in out, 'duplicate JSON key')
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def menu_rows(expected_menu):
    need(type(expected_menu) is list and len(expected_menu)>0 and len(expected_menu)%2==0,
         'nonempty paired request menu')
    count=len(expected_menu)//2
    identities=set()
    for row,(arm,ordinal) in zip(expected_menu,((arm,n) for arm in ('bf16','kv4') for n in range(count)),strict=True):
        need(type(row) is dict and type(row.get('ordinal')) is int and row['ordinal']==ordinal and
             row.get('arm')==arm and row.get('scope') in ('natural-oracle','ungraded-cap-work'),
             'frozen menu arm/order/scope')
        need(type(row.get('case')) is str and 0<len(row['case'])<=128 and
             type(row.get('maximum_output_tokens')) is int and 0<row['maximum_output_tokens']<=8192 and
             digest(row.get('request_body_sha256')),'frozen request identity/cap')
        identity=(arm,row['case'])
        need(identity not in identities,'unique per-arm request case')
        identities.add(identity)
        if row['scope']=='ungraded-cap-work':
            need(row.get('expected_answer') is None and row.get('grade_kind') is None and
                 type(row.get('minimum_final_tokens')) is int and row['minimum_final_tokens']==0,
                 'cap-work has no answer oracle or final-answer coverage claim')
            continue
        need(type(row.get('expected_answer')) is str and 0<len(row['expected_answer'].encode())<=16384 and
             row.get('grade_kind') in ('exact_text','exact_json') and
             type(row.get('minimum_final_tokens')) is int and row['minimum_final_tokens'] in (0,512),
             'exact final-answer oracle and separate long-answer coverage')
        if row['grade_kind']=='exact_json':
            need(encoded(strict(row['expected_answer'])).decode().strip()==row['expected_answer'],
                 'canonical exact JSON oracle')
    left,right=expected_menu[:count],expected_menu[count:]
    need(all({k:v for k,v in a.items() if k!='arm'}=={k:v for k,v in b.items() if k!='arm'}
             for a,b in zip(left,right,strict=True)), 'both arms share one request/oracle plan')
    return expected_menu,[]


def validate_input(value, expected_menu):
    need(type(value) is dict and set(value)=={'schema','mode','rows','cancellations','qualification','precision_accepted'} and
         type(value['schema']) is int and value['schema']==1 and value['mode']==MODE and
         value['qualification'] is value['precision_accepted'] is False, 'exact unqualified input envelope')
    need(len(encoded(value))<=INPUT_LIMIT, 'proposed53MiB input cap; preserve complete input and refuse, never truncate')
    expected,cancel=menu_rows(expected_menu)
    need(type(value['rows']) is list and len(value['rows'])==len(expected), 'all declared responses required')
    for row,frozen in zip(value['rows'],expected):
        need(type(row) is dict and set(row)==ROW_FIELDS, 'exact normal response row schema')
        for key in ('arm','ordinal','case','scope','expected_answer','request_body_sha256','grade_kind','minimum_final_tokens'):
            need(type(row[key]) is type(frozen[key]) and row[key]==frozen[key], 'frozen row/request/oracle '+key)
        ids=row['output_token_ids']
        need(type(ids) is list and 0<len(ids)<=frozen['maximum_output_tokens'] and
             all(type(x) is int and 0<=x<248320 for x in ids), 'bounded complete output token vector')
        need(type(row['wire_text']) is str and 0<len(row['wire_text'].encode())<=WIRE_LIMIT, 'complete bounded wire text')
        need(len(encoded(dict(output_token_ids=ids,wire_text=row['wire_text'])))<=(512<<10)+64, 'projection of one canonical512KiB serving response')
        need(all(digest(row[key]) for key in ('request_body_sha256','response_raw_sha256','response_canonical_sha256')),
             'distinct rawHTTP and canonicalJSON response identity fields')
        need(type(row['finish_reason']) is str and row['finish_reason'] in ('stop','length'), 'terminal response type')
        if row['scope']=='ungraded-cap-work':
            need(row['finish_reason']=='stop' or len(ids)==frozen['maximum_output_tokens'],
                 'length-stopped cap-work contains the exact declared token count')
        else:
            need(row['finish_reason']=='stop', 'natural oracle requires a normal stop')
    need(type(value['cancellations']) is list and len(value['cancellations'])==0, 'no cancellation rows in this menu')
    for row,frozen in zip(value['cancellations'],cancel):
        need(type(row) is dict and set(row)==CANCEL_FIELDS and row['status']=='CANCELLED_COMPLETE' and
             row['completed_answer_claimed'] is False and digest(row['response_raw_sha256']), 'state-only cancellation schema')
        for key in ('arm','ordinal','case','scope','request_body_sha256'):
            need(type(row[key]) is type(frozen[key]) and row[key]==frozen[key], 'frozen cancellation identity '+key)
    return value


def decode_rows(value, expected_menu, tokenizer):
    from thinking_segments import split_thinking_ids
    validate_input(value,expected_menu)
    need(tokenizer.token_to_id('</think>')==248069 and tokenizer.token_to_id('<think>')==248068, 'pinned native marker identities')
    answers=[]
    for row in value['rows']:
        ids=row['output_token_ids']
        need(tokenizer.decode(ids,skip_special_tokens=True)==row['wire_text'], 'full native decode equals whole retained wire text')
        out={key:row[key] for key in IDENTITY_FIELDS}
        out.update(output_ids_sha256=id_hash(ids),wire_text_sha256=sha(row['wire_text'].encode()),
                   wire_decode_matches=True,output_tokens=len(ids),final_answer_sha256=None,final_answer_utf8_bytes=None,
                   final_answer_token_count=None,reasoning_token_count=None,closing_marker_index=None,error=None,
                   grade_value_sha256=None,grade_value_utf8_bytes=None)
        if row['scope']=='ungraded-cap-work':
            out['final_status']='UNGRADED_CAP_WORK'
        else:
            try:
                need(row['finish_reason']=='stop', 'natural response did not stop normally')
                need(row['wire_text'].count('</think>')==1, 'exactly one closing-think text marker')
                split=split_thinking_ids(ids)
                text=tokenizer.decode(split['final_segment_ids'],skip_special_tokens=True).strip()
                need(text, 'nonempty native final text')
                grade_text=encoded(strict(text)).decode().strip() if row['grade_kind']=='exact_json' else text
                out.update(grade_value_sha256=sha(grade_text.encode()),grade_value_utf8_bytes=len(grade_text.encode()))
                out.update({key:split[key] for key in ('final_answer_token_count','reasoning_token_count','closing_marker_index')})
                out.update(final_status='FINAL_SEGMENT_DECODED',final_answer_sha256=sha(text.encode()),final_answer_utf8_bytes=len(text.encode()))
            except ValueError as error:
                out.update(final_status='NATURAL_RESPONSE_REFUSAL',error=str(error)[:256])
        answers.append(out)
    return answers


def validate_answers(value, expected_menu, answers):
    from thinking_segments import split_thinking_ids
    validate_input(value,expected_menu)
    need(type(answers) is list and len(answers)==len(value['rows']), 'all decoded responses retained')
    fields=set(IDENTITY_FIELDS)|{'output_ids_sha256','wire_text_sha256','wire_decode_matches','output_tokens',
        'final_status','final_answer_sha256','final_answer_utf8_bytes','final_answer_token_count','reasoning_token_count','closing_marker_index','error','grade_value_sha256','grade_value_utf8_bytes'}
    for row,answer in zip(value['rows'],answers):
        need(type(answer) is dict and set(answer)==fields and all(type(answer[k]) is type(row[k]) and answer[k]==row[k] for k in IDENTITY_FIELDS), 'decoded source/response identity joins')
        need(answer['wire_decode_matches'] is True and answer['output_ids_sha256']==id_hash(row['output_token_ids']) and
             answer['wire_text_sha256']==sha(row['wire_text'].encode()) and type(answer['output_tokens']) is int and
             answer['output_tokens']==len(row['output_token_ids']), 'full token-vector/wire hashes and count')
        if row['scope']=='ungraded-cap-work':
            need(answer['final_status']=='UNGRADED_CAP_WORK' and all(answer[k] is None for k in
                 ('final_answer_sha256','final_answer_utf8_bytes','final_answer_token_count','reasoning_token_count','closing_marker_index','error','grade_value_sha256','grade_value_utf8_bytes')), 'cap-work never graded or segmented')
        elif answer['final_status']=='FINAL_SEGMENT_DECODED':
            need(row['finish_reason']=='stop' and row['wire_text'].count('</think>')==1, 'normal natural stop and single text marker')
            split=split_thinking_ids(row['output_token_ids'])
            need(all(type(answer[k]) is int and answer[k]==split[k] for k in ('closing_marker_index','reasoning_token_count','final_answer_token_count')) and
                 digest(answer['final_answer_sha256']) and type(answer['final_answer_utf8_bytes']) is int and
                 0<answer['final_answer_utf8_bytes']<=WIRE_LIMIT and answer['error'] is None and
                 digest(answer['grade_value_sha256']) and type(answer['grade_value_utf8_bytes']) is int and
                 0<answer['grade_value_utf8_bytes']<=WIRE_LIMIT, 'native final segment counts/text/grade normalization')
        else:
            need(answer['final_status']=='NATURAL_RESPONSE_REFUSAL' and all(answer[k] is None for k in
                 ('final_answer_sha256','final_answer_utf8_bytes','final_answer_token_count','reasoning_token_count','closing_marker_index','grade_value_sha256','grade_value_utf8_bytes')) and
                 type(answer['error']) is str and 0<len(answer['error'])<=256, 'natural refusal cannot pass')
    return answers


def grade_answers(value,expected_menu,answers,python_results):
    validate_answers(value,expected_menu,answers)
    need(type(python_results) is list and python_results==[],'no code execution in this JSON/text menu')
    grades=[]
    for row,answer in zip(value['rows'],answers,strict=True):
        if row['scope']=='ungraded-cap-work':
            grades.append(dict(arm=row['arm'],ordinal=row['ordinal'],case=row['case'],scope=row['scope'],
                status='UNGRADED_CAP_WORK',quality_passed=None,minimum_final_tokens=0,
                final_answer_token_count=None,long_answer_coverage_passed=None))
            continue
        passed=(answer['final_status']=='FINAL_SEGMENT_DECODED' and
                answer['grade_value_sha256']==sha(row['expected_answer'].encode()) and
                answer['grade_value_utf8_bytes']==len(row['expected_answer'].encode()))
        coverage=(answer['final_answer_token_count'] is not None and
                  answer['final_answer_token_count']>=row['minimum_final_tokens'])
        grades.append(dict(arm=row['arm'],ordinal=row['ordinal'],case=row['case'],scope=row['scope'],
            status='QUALITY_PASS' if passed else 'QUALITY_REFUSAL',quality_passed=passed,
            minimum_final_tokens=row['minimum_final_tokens'],final_answer_token_count=answer['final_answer_token_count'],
            long_answer_coverage_passed=coverage))
    natural=[row for row in grades if row['scope']=='natural-oracle']
    return dict(grades=grades,decoded_rows=len(grades),natural_rows=len(natural),
        ungraded_cap_work_rows=len(grades)-len(natural),state_only_cancellations=0,
        quality_pass=bool(natural) and all(row['quality_passed'] for row in natural),
        long_answer_coverage_pass=bool(natural) and all(row['long_answer_coverage_passed'] for row in natural),
        precision_accepted=False,selected=False,qualification=False)
