"""Recompute the bounded native-HTTP screen from exclusive wire evidence.

This reader does not launch, contact, or clean up a server. It treats the
client summary as a cross-check, not as evidence of completed requests.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import serving_client
import serving_contract as contract


MAX_JSON_BYTES = 64 << 10
MAX_RAW_BYTES = 2 << 20


def _need(value, message):
    if not value:
        raise ValueError("serving readout: " + message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _read(path, maximum=MAX_RAW_BYTES):
    """Read a regular, non-linked member, rejecting any concurrent replacement."""
    path = Path(path)
    if serving_client._kv_receipts.is_receipt(path) and not (
            path.parent.name=='controls' and path.exists()):
        return serving_client._kv_receipts.read(path,maximum)
    _need(path.is_absolute() and path.resolve(strict=True) == path, "canonical evidence path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
              "bounded regular evidence member")
        remaining, pieces = before.st_size, []
        while remaining:
            piece = os.read(descriptor, remaining)
            _need(piece, "short regular-file read")
            pieces.append(piece)
            remaining -= len(piece)
        after, named = os.fstat(descriptor), path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        _need(all(getattr(before, field) == getattr(after, field) == getattr(named, field)
                  for field in fields), "evidence changed during read")
        return b"".join(pieces)
    finally:
        os.close(descriptor)


def _object(path, maximum=MAX_JSON_BYTES):
    value = _json(_read(path, maximum))
    _need(type(value) is dict, "JSON evidence member must be an object")
    return value


def _time(value, label):
    _need(type(value) in (int, float) and math.isfinite(value) and value >= 0, label + " is not finite")


def _utc(value, label):
    _need(type(value) is str and value, label + " missing")


def _names(directory):
    return serving_client._kv_receipts.logical_names(directory)


def _files(prefix):
    return {
        "request_raw": "requests/%s.raw" % prefix,
        "intent": "requests/%s.intent.json" % prefix,
        "entered": "requests/%s.send-entered.json" % prefix,
        "returned": "requests/%s.send-returned.json" % prefix,
        "headers": "responses/%s.headers.json" % prefix,
        "raw": "responses/%s.raw" % prefix,
        "partial": "responses/%s.partial.raw" % prefix,
        "record": "responses/%s.record.json" % prefix,
        "failure": "responses/%s.failure.json" % prefix,
        "stream": "responses/%s.stream.json" % prefix,
        "history": "requests/%s.history.json" % prefix,
        "preparation_failure": "requests/%s.prepare-failure.json" % prefix,
    }


def _check_intent(directory, files, case, ordinal):
    raw = _read(directory / files["request_raw"], 1 << 20)
    _need(raw == contract.canonical(case["body"]), "frozen request bytes differ")
    intent = _object(directory / files["intent"])
    _need(set(intent) == {"schema", "ordinal", "case", "body_sha256", "prepared_monotonic", "prepared_utc",
                          "intent_is_not_submission_proof"}, "intent schema differs")
    _need(intent["schema"] == 1 and intent["ordinal"] == ordinal and type(intent["ordinal"]) is int and
          intent["case"] == case["case"] and intent["body_sha256"] == _sha(raw) and
          intent["intent_is_not_submission_proof"] is True, "intent ordinal/body binding")
    _time(intent["prepared_monotonic"], "intent time")
    _utc(intent["prepared_utc"], "intent UTC")
    return raw


def _check_send(directory, files, case, ordinal, raw, present):
    _need("entered" in present, "request intent lacks send attempt")
    entered = _object(directory / files["entered"])
    _need(set(entered) == {"schema", "ordinal", "case", "body_sha256", "entered_monotonic", "entered_utc",
                           "send_call_entered", "submission_unproven_until_return"}, "send-entered schema differs")
    _need(entered["schema"] == 1 and entered["ordinal"] == ordinal and entered["case"] == case["case"] and
          entered["body_sha256"] == _sha(raw) and entered["send_call_entered"] is True and
          entered["submission_unproven_until_return"] is True, "send-entered binding")
    _time(entered["entered_monotonic"], "send-entered time")
    _utc(entered["entered_utc"], "send-entered UTC")
    if "returned" not in present:
        return False
    returned = _object(directory / files["returned"])
    _need(set(returned) == {"schema", "ordinal", "case", "returned_monotonic", "returned_utc",
                            "send_call_returned", "request_body_sha256"}, "send-returned schema differs")
    _need(returned["schema"] == 1 and returned["ordinal"] == ordinal and returned["case"] == case["case"] and
          returned["request_body_sha256"] == _sha(raw) and returned["send_call_returned"] is True,
          "send-returned binding")
    _time(returned["returned_monotonic"], "send-returned time")
    _utc(returned["returned_utc"], "send-returned UTC")
    return True


def _check_headers(directory, files, case, ordinal):
    headers = _object(directory / files["headers"])
    _need(set(headers) == {"schema", "ordinal", "case", "status", "reason", "headers_observed_monotonic",
                           "headers_observed_utc", "response_observed"}, "response-header schema differs")
    _need(headers["schema"] == 1 and headers["ordinal"] == ordinal and headers["case"] == case["case"] and
          type(headers["status"]) is int and type(headers["reason"]) is str and
          headers["response_observed"] is True, "response-header binding")
    _time(headers["headers_observed_monotonic"], "response-header time")
    _utc(headers["headers_observed_utc"], "response-header UTC")
    return headers


def _check_body(directory, files, case, ordinal, present, headers, expected_arm):
    _need(('raw' in present) != ('partial' in present), 'missing or duplicate raw response')
    raw_key = 'partial' if 'partial' in present else 'raw'
    body = _read(directory / files[raw_key])
    record = _object(directory / files['record'])
    _need(set(record) == {'schema', 'ordinal', 'case', 'status', 'reason', 'response_sha256', 'response_bytes', 'received_monotonic', 'received_utc', 'response_observed', 'client_elapsed_seconds', 'response_truncated', 'raw_body_path'}, 'response record schema differs')
    _need(record['schema'] == 1 and record['ordinal'] == ordinal and (record['case'] == case['case']) and (record['status'] == headers['status']) and (record['reason'] == headers['reason']) and (record['response_sha256'] == _sha(body)) and (record['response_bytes'] == len(body)) and (record['response_observed'] is True) and (record['raw_body_path'] == files[raw_key]), 'response body binding')
    _time(record['received_monotonic'], 'response time')
    _utc(record['received_utc'], 'response UTC')
    _need(type(record['client_elapsed_seconds']) in (int, float) and math.isfinite(record['client_elapsed_seconds']) and (record['client_elapsed_seconds'] > 0), 'positive finite client elapsed required')
    _need(record['response_truncated'] is (raw_key == 'partial'), 'response truncation binding')
    stream = case['body'].get('stream') is True
    _need(('stream' in present) is stream, 'missing or unrequested stream receive evidence')
    observation = _object(directory / files['stream'], 1 << 20) if stream else None
    contract.mtp_contract.settings()
    if raw_key == 'partial':
        return dict(status='PARTIAL_RESPONSE_REFUSAL', metric_eligible=False, qualification=False)
    try:
        if stream:
            intent = _object(directory / files['intent'])
            entered = _object(directory / files['entered'])
            returned = _object(directory / files['returned'])
            times = [intent['prepared_monotonic'], entered['entered_monotonic'], returned['returned_monotonic'], headers['headers_observed_monotonic'], record['received_monotonic']]
            _need(times == sorted(times) and math.isclose(record['client_elapsed_seconds'], times[-1] - times[0], rel_tol=0, abs_tol=1e-07), 'stream request chronology differs')
            result = contract.inspect_stream_response(case, body, record['client_elapsed_seconds'])
            if result['status'] == 'NORMAL_STOP_COMPLETE':
                result.update(contract.inspect_stream_observation(case, body, observation, ordinal=ordinal, prepared=times[0], sent=times[2], headers=times[3], received=times[4]))
        else:
            decoded = _json(body)
            result = contract.inspect_response(case, decoded, record['client_elapsed_seconds'])
            if contract.mtp_contract.contract_for_arm(expected_arm)['enabled']:
                result['spec_tokens_details'] = contract.inspect_speculation(decoded)
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        result = dict(status='STRUCTURAL_REFUSAL', metric_eligible=False, qualification=False)
    if headers['status'] != 200:
        result.update(status='HTTP_STATUS_REFUSAL', metric_eligible=False)
    return result


def read_client(directory, *, expected, native_gate_raw):
    directory=Path(directory)
    _need(directory.is_absolute() and directory.resolve(strict=True)==directory, 'canonical composed client root')
    physical=list(directory.rglob('*'))
    _need(len(physical)<=528 and all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in physical),
          'unchanged composed client physical member bound')
    _need(sum(p.stat().st_size for p in physical if p.is_file())<=24<<20, 'unchanged client tree bytes')
    _need({p.name for p in directory.iterdir()}=={'paired','readiness','RUN-SUMMARY.json'},
          'exact composed top-level members')
    summary=_object(directory/'RUN-SUMMARY.json',8<<20)
    production=not contract.mtp_contract.instrumentation_enabled()
    contract.validate_kv_summary(summary,_read(directory/'paired'/'RUN-SUMMARY.json',8<<20) if production else None)
    blocks=[];rows=[];details=[]
    expected_count=len(contract.mtp_contract.settings()['requests'])
    for name,offset,count,reader in (('paired',0,expected_count,_read_paired_client),):
        for receipt in ('accepted-gate.json','live-workers-before-first-generation.json'):
            _need(_read(directory/name/'readiness'/receipt)==_read(directory/'readiness'/receipt),
                  'one-start readiness identity across blocks')
        block=reader(directory/name,expected=expected,native_gate_raw=native_gate_raw)
        _need(block['wire_complete'] is True and len(block['rows'])==count,'complete recomputed block')
        _need(_sha(_read(directory/name/'RUN-SUMMARY.json',8<<20))==summary['blocks'][len(blocks)]['summary_sha256'] if production else
              _object(directory/name/'RUN-SUMMARY.json',8<<20)==summary['blocks'][len(blocks)]['summary'],
              'parent/block summary byte semantics')
        blocks.append(dict(name=name,global_offset=offset,wire=block))
        for local,row in enumerate(block['rows']):
            plan = contract._KV_ROWS[offset + local]
            files = _files('%02d-%s' % (local, plan['case']))
            response_raw = _read(directory / name / files['raw'])
            raw_sha = _sha(response_raw)
            joined = dict(row,ordinal=offset+local,block=name,local_ordinal=local,
                request_body_sha256=plan['body_sha256'], response_raw_sha256=raw_sha)
            if offset + local != -1:  # no cancellation ordinal in the natural-stop menu
                canonical_sha = _sha(contract.canonical(_json(response_raw)))
                _need(canonical_sha == row['raw_response_sha256'], 'canonical response identity differs')
                joined['response_canonical_sha256'] = canonical_sha
            rows.append(joined)
            if offset+local!=-1:details.append(row['spec_tokens_details'])
    _need(len(rows)==expected_count and len(details)==expected_count,'exact response/cancel accounting')
    speculation=dict(response_count=expected_count,**{key:sum(row[key] for row in details) for key in
        ('spec_verify_ct','spec_num_proposed_drafts','spec_num_correct_drafts')})
    _need(speculation['spec_verify_ct']>0,'actual native DFlash verification telemetry')
    return dict(schema=1,mode=contract.MODE,arm=expected['arm'],binding=dict(expected),
        status='HTTP_WIRE_COMPLETE',wire_complete=True,rows=rows,blocks=blocks,
        paired_groups=blocks[0]['wire']['paired_groups'],
        speculation=speculation,qualification=False,source_promotion=False,
        code_correctness_graded=False,precision_accepted=False,independent_release_required=True,
        answer_grade_status='PENDING_POST_RELEASE_NATIVE_DECODE', native_full_wire_decode_required=True)

def _read_context_client(directory, *, expected, native_gate_raw):
    """Return recomputed wire status only; it is not release or quality evidence."""
    directory = Path(directory)
    _need(directory.is_absolute() and directory.resolve(strict=True) == directory and directory.is_dir(), 'canonical client directory')
    _need(type(native_gate_raw) is bytes and 0 < len(native_gate_raw) <= MAX_JSON_BYTES, 'bounded raw readiness record required')
    gate = _json(native_gate_raw)
    _need(type(gate) is dict, 'readiness record must be an object')
    workers, gate_sha = (serving_client.validate_gate(gate, expected), _sha(native_gate_raw))
    names = _names(directory)
    root = {'readiness/accepted-gate.json', 'readiness/live-workers-before-first-generation.json', 'RUN-SUMMARY.json'}
    _need(root <= names, 'missing readiness or terminal evidence')
    accepted = _object(directory / 'readiness/accepted-gate.json')
    _need(set(accepted) == {'schema', 'gate_sha256', 'expected', 'mode', 'accepted_utc', 'workers'} and accepted['schema'] == 1 and (accepted['gate_sha256'] == gate_sha) and (accepted['expected'] == expected) and (accepted['mode'] == contract.MODE) and (accepted['workers'] == workers), 'accepted readiness binding')
    _utc(accepted['accepted_utc'], 'accepted readiness UTC')
    live = _object(directory / 'readiness/live-workers-before-first-generation.json')
    _need(set(live) == {'schema', 'gate_sha256', 'observed_utc', 'workers'} and live['schema'] == 1 and (live['gate_sha256'] == gate_sha) and (type(live['workers']) is list) and (len(live['workers']) == 2), 'pre-request worker evidence')
    _utc(live['observed_utc'], 'live-worker UTC')
    for actual, worker in zip(live['workers'], workers):
        _need(actual == dict(pid=worker['pid'], start_ticks=worker['start_ticks'], boot_id=worker['boot_id'], alive=True, rank=worker['rank'], identity_sha256=worker['identity_sha256']), 'pre-request worker identity differs')
    counts = dict(intent_count=0, send_attempt_entered_count=0, send_call_returned_count=0, response_observed_count=0, response_body_observed_count=0, normal_complete_count=0)
    if contract.screen_policy() is not None:
        counts['valid_complete_count'] = 0
    allowed, rows, stopped = (set(root), [], False)
    frozen = [c for c in contract.requests() if c['_kv_block'] == 'context']
    for ordinal, case in enumerate(frozen):
        prefix, files = ('%02d-%s' % (ordinal, case['case']), _files('%02d-%s' % (ordinal, case['case'])))
        present = {key for key, path in files.items() if path in names}
        if stopped:
            _need(not present, 'post-stop request evidence')
            continue
        if not present:
            stopped = True
            continue
        allowed.update((files[key] for key in present))
        if 'generated_history' in case:
            _need(ordinal > 0 and len(rows) == ordinal and contract.response_completed(rows[-1]), 'valid generated history parent required')
            previous = frozen[ordinal - 1]
            previous_files = _files('%02d-%s' % (ordinal - 1, previous['case']))
            previous_raw = _read(directory / previous_files['raw'])
            try:
                case = contract.materialize_history(case, previous['case'], previous_raw)
            except contract.HistoryRefusal:
                _need(present == {'preparation_failure'} and _object(directory / files['preparation_failure']) == contract.history_refusal(case, previous_raw), 'unsupported generated history must refuse without submission')
                rows.append(contract.history_refusal(case, previous_raw))
                stopped = True
                continue
            _need('history' in present and 'preparation_failure' not in present and (_object(directory / files['history']) == case['history_binding']), 'generated history wire binding differs')
            present.remove('history')
        else:
            _need(not {'history', 'preparation_failure'} & present, 'unrequested generated history evidence')
        _need({'request_raw', 'intent'} <= present, 'missing or reordered request evidence')
        raw = _check_intent(directory, files, case, ordinal)
        counts['intent_count'] += 1
        returned = _check_send(directory, files, case, ordinal, raw, present)
        counts['send_attempt_entered_count'] += 1
        counts['send_call_returned_count'] += int(returned)
        if not returned:
            _need(present == {'request_raw', 'intent', 'entered', 'failure'}, 'incomplete pre-response failure')
            failure = _object(directory / files['failure'])
            _need(failure.get('ordinal') == ordinal and failure.get('case') == case['case'] and (failure.get('response_observed') is False), 'pre-response failure binding')
            rows.append(dict(status='PARTIAL_FAILURE', metric_eligible=False, qualification=False))
            stopped = True
            continue
        if 'headers' not in present:
            _need(present == {'request_raw', 'intent', 'entered', 'returned', 'failure'}, 'missing response-header failure')
            failure = _object(directory / files['failure'])
            _need(failure.get('ordinal') == ordinal and failure.get('case') == case['case'] and (failure.get('response_observed') is False), 'header failure binding')
            rows.append(dict(status='PARTIAL_FAILURE', metric_eligible=False, qualification=False))
            stopped = True
            continue
        headers = _check_headers(directory, files, case, ordinal)
        counts['response_observed_count'] += 1
        if 'record' not in present:
            _need(present == {'request_raw', 'intent', 'entered', 'returned', 'headers', 'failure'}, 'missing raw response failure')
            failure = _object(directory / files['failure'])
            _need(failure.get('ordinal') == ordinal and failure.get('case') == case['case'] and (failure.get('response_observed') is False), 'body failure binding')
            rows.append(dict(status='PARTIAL_FAILURE', metric_eligible=False, qualification=False))
            stopped = True
            continue
        result = _check_body(directory, files, case, ordinal, present, headers, expected['arm'])
        counts['response_body_observed_count'] += 1
        if result['status'] == 'PARTIAL_RESPONSE_REFUSAL':
            _need('failure' in present, 'truncated response lacks failure record')
            stopped = True
        else:
            _need('failure' not in present, 'complete response has contradictory failure')
            if result['status'] == 'NORMAL_STOP_COMPLETE':
                counts['normal_complete_count'] += 1
            if 'valid_complete_count' in counts and contract.response_completed(result):
                counts['valid_complete_count'] += 1
            if not contract.response_completed(result):
                stopped = True
        rows.append(result)
    contract.mtp_contract.settings()
    _need(names == allowed, 'extra, duplicate, or post-stop client evidence member')
    summary = _object(directory / 'RUN-SUMMARY.json', 4 << 20)
    _need(summary.get('schema') == 1 and summary.get('mode') == contract.MODE and (summary.get('qualification') is False) and (summary.get('frozen_case_count') == len(frozen)) and (summary.get('executed_case_count') == len(rows)) and (summary.get('counts') == counts) and (summary.get('stopped_early') is (len(rows) != len(frozen))), 'summary disagrees with durable evidence')
    count_key = 'valid_complete_count' if contract.screen_policy() is not None else 'normal_complete_count'
    complete = counts[count_key] == len(frozen) and len(rows) == len(frozen) and (not stopped)
    _need(summary.get('screen_policy') == contract.screen_policy(), 'summary screen policy differs')
    contract.mtp_contract.settings()
    sampling = contract.sampling_contract()
    if sampling is not None:
        _need(summary.get('sampling') == sampling, 'sampled summary contract differs')
    speculation = None
    if contract.mtp_contract.contract_for_arm(expected['arm'])['enabled'] and complete:
        details = [row.get('spec_tokens_details') for row in rows]
        _need(all((type(row) is dict for row in details)), 'MTP details missing from completed response')
        speculation = dict(response_count=len(details), spec_verify_ct=sum((row['spec_verify_ct'] for row in details)), spec_num_proposed_drafts=sum((row['spec_num_proposed_drafts'] for row in details)), spec_num_correct_drafts=sum((row['spec_num_correct_drafts'] for row in details)))
        _need(speculation['spec_verify_ct'] > 0, 'MTP produced no verification telemetry')
    _wire_result = dict(schema=1, mode=contract.MODE, arm=expected['arm'], binding=dict(expected), counts=counts, rows=rows, wire_complete=complete, qualification=False, source_promotion=False, code_correctness_graded=False, speculation=speculation)
    contract.mtp_contract.settings()
    _wire_result.update(**{'screen_policy': contract.screen_policy()} if contract.screen_policy() is not None else {}, **{'sampling': sampling} if sampling is not None else {}, status='HTTP_WIRE_COMPLETE' if complete else 'HTTP_WIRE_INCOMPLETE', independent_release_required=True)
    return _wire_result


def accept_with_owner(wire, campaign, *, owner_arm):
    """Attach only an already validated two-arm owner result; grant no authority."""
    _need(owner_arm in ("bf16", "kv4"), "known legacy owner arm")
    _need(wire.get("mode") == contract.MODE and wire.get("wire_complete") is True and
          wire.get("qualification") is False and wire.get("arm") == contract.mtp_contract.owner_arm_map()[owner_arm],
          "complete wire and exact MTP mapping required")
    if contract.mtp_contract.contract_for_arm(wire['arm'])['enabled']:
        _need(type(wire.get("speculation")) is dict and wire["speculation"].get("spec_verify_ct", 0) > 0,
              "MTP wire lacks aggregate verification telemetry")
    _need(campaign.get("scope") == "matched-host-owner-adapter-v1" and
          campaign.get("status") == "HOST_OWNER_ADAPTER_COMPLETE" and campaign.get("adapter_complete") is True and
          campaign.get("independent_release_verified") is True and campaign.get("unresolved") is False and
          campaign.get("failure") is None and campaign.get("cleanup_errors") == [],
          "validated finite-owner completion required")
    arms = campaign.get("arms")
    _need(type(arms) is list and {row.get("arm") for row in arms if type(row) is dict} == {"bf16", "kv4"} and
          len(arms) == 2 and all(row.get("release_verified") is True and row.get("inner_complete") is True
                              for row in arms), "both owner arms must be independently complete")
    _need(type(wire.get('binding')) is dict and
          campaign.get('experiment_mode') == contract.MODE and
          type(campaign.get('http_bindings')) is dict and
          campaign['http_bindings'].get(owner_arm) == wire['binding'],
          'same HTTP session/spec/source and owner arm required')
    return dict(wire, status="SERVING_SCREEN_COMPLETE", qualification=False,
                independent_release_required=False,
                release_evidence_scope="existing validated two-arm finite-owner campaign",
                independent_authority_claimed=False)


def _check_closure_body(directory, files, case, ordinal, present, headers, expected_arm):
    _need(("raw" in present) != ("partial" in present), "missing or duplicate raw response")
    raw_key = "partial" if "partial" in present else "raw"
    body = _read(directory / files[raw_key])
    record = _object(directory / files["record"])
    _need(set(record) == {"schema", "ordinal", "case", "status", "reason", "response_sha256", "response_bytes",
                           "received_monotonic", "received_utc", "response_observed", "client_elapsed_seconds",
                           "response_truncated", "raw_body_path"}, "response record schema differs")
    _need(record["schema"] == 1 and record["ordinal"] == ordinal and record["case"] == case["case"] and
          record["status"] == headers["status"] and record["reason"] == headers["reason"] and
          record["response_sha256"] == _sha(body) and record["response_bytes"] == len(body) and
          record["response_observed"] is True and record["raw_body_path"] == files[raw_key],
          "response body binding")
    _time(record["received_monotonic"], "response time")
    _utc(record["received_utc"], "response UTC")
    _need(type(record["client_elapsed_seconds"]) in (int, float) and
          math.isfinite(record["client_elapsed_seconds"]) and record["client_elapsed_seconds"] > 0,
          "positive finite client elapsed required")
    _need(record["response_truncated"] is (raw_key == "partial"), "response truncation binding")
    if serving_client.closure_mode() or contract.mtp_contract.paired_serving_enabled():
        intent = _object(directory / files["intent"])
        entered = _object(directory / files["entered"])
        returned = _object(directory / files["returned"])
        times = [intent["prepared_monotonic"], entered["entered_monotonic"],
                 returned["returned_monotonic"], headers["headers_observed_monotonic"], record["received_monotonic"]]
        _need(times == sorted(times) and
              math.isclose(record["client_elapsed_seconds"], times[-1] - times[0], rel_tol=0, abs_tol=1e-7),
              "closure request chronology differs")
    if raw_key == "partial":
        return dict(status="PARTIAL_RESPONSE_REFUSAL", metric_eligible=False, qualification=False)
    try:
        if serving_client.closure_mode() and ordinal == 22:
            abort = _control_bundle(directory, "abort", "POST", "/abort_request",
                contract.canonical(dict(rid=serving_client.CANCEL_RID, abort_all=False)))
            abort["stream"] = _object(directory / "controls/cancel-stream.json")
            result = serving_client.inspect_cancel_evidence(case, body, abort)
            _need(headers["headers_observed_monotonic"] <= abort["stream"]["first_generated_monotonic"] <=
                  abort["stream"]["drained_monotonic"] <= record["received_monotonic"], "cancel observation chronology")
        else:
            decoded = _json(body)
            result = contract.inspect_response(case, decoded, record["client_elapsed_seconds"])
            if contract.mtp_contract.contract_for_arm(expected_arm)['enabled']:
                result["spec_tokens_details"] = contract.inspect_speculation(decoded)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, json.JSONDecodeError):
        result = dict(status="STRUCTURAL_REFUSAL", metric_eligible=False, qualification=False)
    if headers["status"] != 200:
        result.update(status="HTTP_STATUS_REFUSAL", metric_eligible=False)
    return result


def _control_members(name):
    return {"controls/" + name + suffix for suffix in (".request.raw", ".raw", ".record.json")}


def _control_bundle(directory, name, method, endpoint, body):
    request = _read(directory / ("controls/" + name + ".request.raw"), 4096)
    raw = _read(directory / ("controls/" + name + ".raw"), serving_client.MAX_METRICS_BYTES)
    record = _object(directory / ("controls/" + name + ".record.json"))
    fields = {"schema", "name", "method", "endpoint", "request_sha256", "request_bytes", "started_monotonic",
              "returned_monotonic", "headers_monotonic", "status", "reason", "finished_monotonic",
              "response_sha256", "response_bytes", "response_truncated", "failure"}
    _need(set(record) == fields and record["schema"] == 1 and record["name"] == name and
          record["method"] == method and record["endpoint"] == endpoint and request == body and
          record["request_sha256"] == _sha(request) and record["request_bytes"] == len(request) and
          record["response_sha256"] == _sha(raw) and record["response_bytes"] == len(raw), "control raw/source-action binding")
    times = [record[key] for key in ("started_monotonic", "returned_monotonic", "headers_monotonic", "finished_monotonic")]
    for value in times:
        if value is not None:
            _time(value, "control time")
    if record["failure"] is None:
        _need(None not in times and times == sorted(times), "control chronology differs")
    else:
        _need(type(record["failure"]) is str and record["failure"], "control failure differs")
    _need(type(record["response_truncated"]) is bool, "control truncation flag")
    return dict(request_raw=request, response_raw=raw, record=record)


def _prompt_controls(directory, names, rows, frozen):
    """Require exact immediate flushes between completed and subsequent requests."""
    allowed = set()
    for ordinal in (1, 2):
        name = f'prompt-flush-before-{ordinal:02d}'
        members = _control_members(name)
        present = names & members
        if len(rows) > ordinal:
            _need(present == members, 'missing prompt flush control evidence')
        if not present:
            continue
        _need(present == members and len(rows) >= ordinal and
              rows[ordinal - 1]['status'] == 'NORMAL_STOP_COMPLETE',
              'partial or premature prompt flush control evidence')
        bundle = _control_bundle(directory, name, 'POST', contract.PROMPT_FLUSH_ENDPOINT, b'')
        serving_client.inspect_prompt_flush(bundle, ordinal)
        previous = '%02d-%s' % (ordinal - 1, frozen[ordinal - 1]['case'])
        previous_wire = _object(directory / ('responses/%s.record.json' % previous))
        action = bundle['record']
        _need(previous_wire['received_monotonic'] <= action['started_monotonic'],
              'prompt flush preceded previous response completion')
        if len(rows) > ordinal:
            following = '%02d-%s' % (ordinal, frozen[ordinal]['case'])
            intent = _object(directory / ('requests/%s.intent.json' % following))
            entered = _object(directory / ('requests/%s.send-entered.json' % following))
            _need(action['finished_monotonic'] <= intent['prepared_monotonic'] <= entered['entered_monotonic'],
                  'prompt request preceded successful flush completion')
        allowed.update(members)
    return allowed


def _closure_controls(directory, names, rows, counts, arm):
    """Recompute control evidence; reported client success is not an input."""
    bundles = {}
    for name, method, endpoint, body in (
        ("metrics-before-pressure", "GET", "/metrics", b""),
        ("metrics-after-pressure", "GET", "/metrics", b""),
        ("abort", "POST", "/abort_request", contract.canonical(dict(rid=serving_client.CANCEL_RID, abort_all=False))),
    ):
        members = _control_members(name)
        if names & members:
            _need(members <= names, "partial control member set")
            bundles[name] = _control_bundle(directory, name, method, endpoint, body)
    pressure = None
    frozen = [c for c in contract.requests() if c['_kv_block']=='lifecycle']
    def timing(ordinal, suffix, key):
        prefix = "%02d-%s" % (ordinal, frozen[ordinal]["case"])
        return _object(directory / (suffix % prefix))[key]
    if len(rows) > 14:
        _need("metrics-before-pressure" in bundles, "missing pre-pressure metrics")
    if len(rows) > 19 and rows[19]["status"] == "NORMAL_STOP_COMPLETE":
        _need("metrics-after-pressure" in bundles, "missing post-pressure metrics")
    if "metrics-after-pressure" in bundles:
        _need("metrics-before-pressure" in bundles and len(rows) > 19, "post-pressure metrics without pressure history")
        before, after = (bundles[name] for name in ("metrics-before-pressure", "metrics-after-pressure"))
        for bundle in (before, after):
            _need(bundle["record"]["status"] == 200 and bundle["record"]["failure"] is None and
                  bundle["record"]["response_truncated"] is False, "incomplete metrics response")
        pressure = serving_client.pressure_observation(before['response_raw'], after['response_raw'], arm)
        _need(timing(13, "responses/%s.record.json", "received_monotonic") <= before["record"]["started_monotonic"] and
              before["record"]["finished_monotonic"] <= timing(14, "requests/%s.send-entered.json", "entered_monotonic") and
              timing(19, "responses/%s.record.json", "received_monotonic") <= after["record"]["started_monotonic"],
              "pressure metrics do not bracket requests 14..19")
        if len(rows) > 20:
            _need(after["record"]["finished_monotonic"] <= timing(20, "requests/%s.send-entered.json", "entered_monotonic"),
                  "victim replay preceded pressure observation")
    cancelled = counts["aborted_complete_count"] == 1
    if "controls/cancel-stream.json" in names:
        stream = _object(directory / "controls/cancel-stream.json")
        _need(len(rows) > 22 and stream.get("ordinal") == 22 and stream.get("case") == frozen[22]["case"] and
              stream.get("rid") == serving_client.CANCEL_RID, "cancel stream history binding")
        if len(rows) > 23:
            _need(cancelled and stream["drained_monotonic"] <= timing(23, "requests/%s.send-entered.json", "entered_monotonic"),
                  "cancel recovery preceded verified stream drain")
    if "abort" in bundles:
        _need("controls/cancel-stream.json" in names, "abort lacks stream evidence")
    concurrent = None
    if "controls/concurrent.json" in names:
        group = _object(directory / "controls/concurrent.json")
        _need(set(group) == {"schema", "ordinals", "joined", "failures", "native_batch_size_proven"} and
              group["schema"] == 1 and group["ordinals"] == [24, 25] and group["joined"] is True and
              type(group["failures"]) is dict and group["native_batch_size_proven"] is False,
              "concurrent owned-thread closure differs")
        if len(rows) > 25 and all(row["status"] == "NORMAL_STOP_COMPLETE" for row in rows[24:26]):
            sent = [timing(i, "requests/%s.send-returned.json", "returned_monotonic") for i in (24, 25)]
            ended = [timing(i, "responses/%s.record.json", "received_monotonic") for i in (24, 25)]
            _need(not group["failures"] and max(sent) < min(ended), "concurrent HTTP intervals do not overlap")
            if len(rows) > 26:
                _need(max(ended) <= timing(26, "requests/%s.send-entered.json", "entered_monotonic"),
                      "final replay preceded concurrent completion")
            concurrent = dict(ordinals=[24, 25], sent_returned_monotonic=sent,
                received_monotonic=ended, overlap_seconds=min(ended)-max(sent),
                http_overlap_observed=True, native_batch_size_proven=False)
    if "controls/lifecycle.failure.json" in names:
        failure = _object(directory / "controls/lifecycle.failure.json")
        _need(set(failure) == {"schema", "failure"} and failure["schema"] == 1 and
              type(failure["failure"]) is str and failure["failure"], "lifecycle failure record")
    all_rows = len(rows) == 27
    if all_rows:
        _need(pressure is not None and cancelled and concurrent is not None,
              "closure lacks pressure measurement, native abort or HTTP overlap evidence")
    return dict(pressure=pressure, cancellation_proven=cancelled, concurrency=concurrent,
                complete=all_rows and pressure is not None and cancelled and concurrent is not None and
                         "controls/lifecycle.failure.json" not in names,
                native_batch_size_proven=False)


def _read_closure_client(directory, *, expected, native_gate_raw):
    """Return recomputed wire status only; it is not release or quality evidence."""
    directory = Path(directory)
    _need(directory.is_absolute() and directory.resolve(strict=True) == directory and directory.is_dir(),
          "canonical client directory")
    _need(type(native_gate_raw) is bytes and 0 < len(native_gate_raw) <= MAX_JSON_BYTES,
          "bounded raw readiness record required")
    gate = _json(native_gate_raw)
    _need(type(gate) is dict, "readiness record must be an object")
    workers, gate_sha = serving_client.validate_gate(gate, expected), _sha(native_gate_raw)
    names = _names(directory)
    root = {"readiness/accepted-gate.json", "readiness/live-workers-before-first-generation.json", "RUN-SUMMARY.json"}
    _need(root <= names, "missing readiness or terminal evidence")
    accepted = _object(directory / "readiness/accepted-gate.json")
    _need(set(accepted) == {"schema", "gate_sha256", "expected", "mode", "accepted_utc", "workers"} and
          accepted["schema"] == 1 and accepted["gate_sha256"] == gate_sha and accepted["expected"] == expected and
          accepted["mode"] == contract.MODE and accepted["workers"] == workers, "accepted readiness binding")
    _utc(accepted["accepted_utc"], "accepted readiness UTC")
    live = _object(directory / "readiness/live-workers-before-first-generation.json")
    _need(set(live) == {"schema", "gate_sha256", "observed_utc", "workers"} and live["schema"] == 1 and
          live["gate_sha256"] == gate_sha and type(live["workers"]) is list and len(live["workers"]) == 2,
          "pre-request worker evidence")
    _utc(live["observed_utc"], "live-worker UTC")
    for actual, worker in zip(live["workers"], workers):
        _need(actual == dict(pid=worker["pid"], start_ticks=worker["start_ticks"], boot_id=worker["boot_id"],
                             alive=True, rank=worker["rank"], identity_sha256=worker["identity_sha256"]),
              "pre-request worker identity differs")

    counts = dict(intent_count=0, send_attempt_entered_count=0, send_call_returned_count=0,
                  response_observed_count=0, response_body_observed_count=0, normal_complete_count=0)
    closure = serving_client.closure_mode()
    if closure:
        counts["aborted_complete_count"] = 0
    allowed, rows, stopped = set(root), [], False
    concurrent_entered = False
    if closure:
        for name in ("metrics-before-pressure", "metrics-after-pressure", "abort"):
            allowed.update(names & _control_members(name))
        allowed.update(names & {"controls/cancel-stream.json", "controls/concurrent.json", "controls/lifecycle.failure.json"})
    frozen=[c for c in contract.requests() if c['_kv_block']=='lifecycle']
    for ordinal, case in enumerate(frozen):
        prefix, files = "%02d-%s" % (ordinal, case["case"]), _files("%02d-%s" % (ordinal, case["case"]))
        present = {key for key, path in files.items() if path in names}
        _need(not ({'stream', 'history', 'preparation_failure'} & present),
              'undeclared closure stream/history/preparation evidence')
        if stopped and not (closure and ordinal == 25 and concurrent_entered):
            _need(not present, "post-stop request evidence")
            continue
        if not present:
            stopped = True
            continue
        if closure and ordinal == 24:
            concurrent_entered = True
        allowed.update(files[key] for key in present)
        _need({"request_raw", "intent"} <= present, "missing or reordered request evidence")
        raw = _check_intent(directory, files, case, ordinal)
        counts["intent_count"] += 1
        returned = _check_send(directory, files, case, ordinal, raw, present)
        counts["send_attempt_entered_count"] += 1
        counts["send_call_returned_count"] += int(returned)
        if not returned:
            _need(present == {"request_raw", "intent", "entered", "failure"}, "incomplete pre-response failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "pre-response failure binding")
            rows.append(dict(status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            continue
        if "headers" not in present:
            _need(present == {"request_raw", "intent", "entered", "returned", "failure"},
                  "missing response-header failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "header failure binding")
            rows.append(dict(status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            continue
        headers = _check_headers(directory, files, case, ordinal)
        counts["response_observed_count"] += 1
        if "record" not in present:
            _need(present == {"request_raw", "intent", "entered", "returned", "headers", "failure"},
                  "missing raw response failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "body failure binding")
            rows.append(dict(status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            continue
        result = _check_closure_body(directory, files, case, ordinal, present, headers, expected["arm"])
        counts["response_body_observed_count"] += 1
        if result["status"] == "PARTIAL_RESPONSE_REFUSAL":
            _need("failure" in present, "truncated response lacks failure record")
            stopped = True
        else:
            _need("failure" not in present, "complete response has contradictory failure")
            if result["status"] == "NORMAL_STOP_COMPLETE":
                counts["normal_complete_count"] += 1
            elif closure and ordinal == 22 and result["status"] == "CANCELLED_COMPLETE":
                counts["aborted_complete_count"] += 1
            else:
                stopped = True
        rows.append(result)
    _need(names == allowed, "extra, duplicate, or post-stop client evidence member")
    summary = _object(directory / "RUN-SUMMARY.json", 4 << 20)
    _need(summary.get("schema") == 1 and summary.get("mode") == contract.MODE and
          summary.get("qualification") is False and summary.get("frozen_case_count") == len(frozen) and
          summary.get("executed_case_count") == len(rows) and summary.get("counts") == counts and
          summary.get("stopped_early") is (len(rows) != len(frozen)), "summary disagrees with durable evidence")
    complete = counts["normal_complete_count"] == len(frozen) and len(rows) == len(frozen) and not stopped
    lifecycle = None
    if closure:
        lifecycle = _closure_controls(directory, names, rows, counts, expected['arm'])
        complete = (counts["normal_complete_count"] == 26 and counts["aborted_complete_count"] == 1 and
                    len(rows) == 27 and not stopped and lifecycle["complete"])
    speculation = None
    if contract.mtp_contract.contract_for_arm(expected['arm'])['enabled'] and complete:
        details = [row.get("spec_tokens_details") for row in rows if row["status"] == "NORMAL_STOP_COMPLETE"]
        _need(all(type(row) is dict for row in details), "MTP details missing from completed response")
        speculation = dict(
            response_count=len(details),
            spec_verify_ct=sum(row["spec_verify_ct"] for row in details),
            spec_num_proposed_drafts=sum(row["spec_num_proposed_drafts"] for row in details),
            spec_num_correct_drafts=sum(row["spec_num_correct_drafts"] for row in details),
        )
        _need(speculation["spec_verify_ct"] > 0, "MTP produced no verification telemetry")
    result = dict(schema=1, mode=contract.MODE, arm=expected["arm"], binding=dict(expected), counts=counts, rows=rows,
                wire_complete=complete, qualification=False, source_promotion=False,
                code_correctness_graded=False,
                speculation=speculation,
                status="HTTP_WIRE_COMPLETE" if complete else "HTTP_WIRE_INCOMPLETE",
                independent_release_required=True)
    if closure:
        result["lifecycle"] = lifecycle
    return result


def _read_paired_client(directory, *, expected, native_gate_raw):
    """Return recomputed wire status only; it is not release or quality evidence."""
    directory = Path(directory)
    _need(directory.is_absolute() and directory.resolve(strict=True) == directory and directory.is_dir(),
          "canonical client directory")
    _need(type(native_gate_raw) is bytes and 0 < len(native_gate_raw) <= MAX_JSON_BYTES,
          "bounded raw readiness record required")
    gate = _json(native_gate_raw)
    _need(type(gate) is dict, "readiness record must be an object")
    workers, gate_sha = serving_client.validate_gate(gate, expected), _sha(native_gate_raw)
    names = _names(directory)
    root = {"readiness/accepted-gate.json", "readiness/live-workers-before-first-generation.json", "RUN-SUMMARY.json"}
    _need(root <= names, "missing readiness or terminal evidence")
    accepted = _object(directory / "readiness/accepted-gate.json")
    _need(set(accepted) == {"schema", "gate_sha256", "expected", "mode", "accepted_utc", "workers"} and
          accepted["schema"] == 1 and accepted["gate_sha256"] == gate_sha and accepted["expected"] == expected and
          accepted["mode"] == contract.MODE and accepted["workers"] == workers, "accepted readiness binding")
    _utc(accepted["accepted_utc"], "accepted readiness UTC")
    live = _object(directory / "readiness/live-workers-before-first-generation.json")
    _need(set(live) == {"schema", "gate_sha256", "observed_utc", "workers"} and live["schema"] == 1 and
          live["gate_sha256"] == gate_sha and type(live["workers"]) is list and len(live["workers"]) == 2,
          "pre-request worker evidence")
    _utc(live["observed_utc"], "live-worker UTC")
    for actual, worker in zip(live["workers"], workers):
        _need(actual == dict(pid=worker["pid"], start_ticks=worker["start_ticks"], boot_id=worker["boot_id"],
                             alive=True, rank=worker["rank"], identity_sha256=worker["identity_sha256"]),
              "pre-request worker identity differs")

    counts = dict(intent_count=0, send_attempt_entered_count=0, send_call_returned_count=0,
                  response_observed_count=0, response_body_observed_count=0, normal_complete_count=0)
    counts.update(aborted_complete_count=0,valid_complete_count=0)
    allowed, rows, stopped = set(root), [], False
    failed_group = None
    groups=contract.mtp_contract.paired_groups(expected['arm'])
    frozen=[c for c in contract.requests() if c['_kv_block']=='paired']
    allowed.update(names & {'controls/paired.failure.json'})
    for group in groups:
        allowed.update(names & {'controls/group-'+group['id']+'.json','controls/concurrent-'+group['id']+'.json'})
    for ordinal in [i for group in groups for i in group['ordinals']]:
        case=frozen[ordinal]
        current_group=contract.mtp_contract.paired_group_for(ordinal)
        prefix, files = "%02d-%s" % (ordinal, case["case"]), _files("%02d-%s" % (ordinal, case["case"]))
        present = {key for key, path in files.items() if path in names}
        _need(not ({'stream', 'history', 'preparation_failure'} & present),
              'undeclared closure stream/history/preparation evidence')
        if stopped and not (current_group['concurrency']==2 and current_group['id']==failed_group):
            _need(not present, "post-stop request evidence")
            continue
        if not present:
            stopped = True
            if failed_group is None: failed_group=current_group['id']
            continue
        allowed.update(files[key] for key in present)
        _need({"request_raw", "intent"} <= present, "missing or reordered request evidence")
        raw = _check_intent(directory, files, case, ordinal)
        counts["intent_count"] += 1
        returned = _check_send(directory, files, case, ordinal, raw, present)
        counts["send_attempt_entered_count"] += 1
        counts["send_call_returned_count"] += int(returned)
        if not returned:
            _need(present == {"request_raw", "intent", "entered", "failure"}, "incomplete pre-response failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "pre-response failure binding")
            rows.append(dict(case=case["case"],status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            if failed_group is None: failed_group=current_group['id']
            continue
        if "headers" not in present:
            _need(present == {"request_raw", "intent", "entered", "returned", "failure"},
                  "missing response-header failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "header failure binding")
            rows.append(dict(case=case["case"],status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            if failed_group is None: failed_group=current_group['id']
            continue
        headers = _check_headers(directory, files, case, ordinal)
        counts["response_observed_count"] += 1
        if "record" not in present:
            _need(present == {"request_raw", "intent", "entered", "returned", "headers", "failure"},
                  "missing raw response failure")
            failure = _object(directory / files["failure"])
            _need(failure.get("ordinal") == ordinal and failure.get("case") == case["case"] and
                  failure.get("response_observed") is False, "body failure binding")
            rows.append(dict(case=case["case"],status="PARTIAL_FAILURE", metric_eligible=False, qualification=False))
            stopped = True
            if failed_group is None: failed_group=current_group['id']
            continue
        result = _check_closure_body(directory, files, case, ordinal, present, headers, expected["arm"])
        counts["response_body_observed_count"] += 1
        if result["status"] == "PARTIAL_RESPONSE_REFUSAL":
            _need("failure" in present, "truncated response lacks failure record")
            stopped = True
            if failed_group is None: failed_group=current_group['id']
        else:
            _need("failure" not in present, "complete response has contradictory failure")
            if result["status"] == "NORMAL_STOP_COMPLETE":
                counts["normal_complete_count"] += 1
            if contract.response_completed(result):
                counts['valid_complete_count'] += 1
            else:
                stopped = True
                if failed_group is None: failed_group=current_group['id']
        result.setdefault("case",case["case"])
        rows.append(result)
    row_order={r['case']:i for i,r in enumerate(frozen)}
    rows.sort(key=lambda r:row_order.get(r.get('case'), -1))
    _need(names == allowed, "extra, duplicate, or post-stop client evidence member")
    summary = _object(directory / "RUN-SUMMARY.json", 8 << 20)
    _need(summary.get("schema") == 1 and summary.get("mode") == contract.MODE and
          summary.get("qualification") is False and summary.get("frozen_case_count") == len(frozen) and
          summary.get("executed_case_count") == len(rows) and summary.get("counts") == counts and
          summary.get("stopped_early") is (len(rows) != len(frozen)), "summary disagrees with durable evidence")
    complete = counts["valid_complete_count"] == len(frozen) and len(rows) == len(frozen) and not stopped
    if complete and not contract.mtp_contract.instrumentation_enabled():
        _need(len(summary.get("rows", [])) == len(rows), "production summary row count")
        for local, (held, recomputed) in enumerate(zip(summary["rows"], rows)):
            held = dict(held.get("contract"), case=held.get("case"))
            raw = _read(directory / _files("%02d-%s" % (local, recomputed["case"]))["raw"])
            _need(held.pop("raw_http_response_sha256", None) == _sha(raw) and held == contract.summary_contract(recomputed),
                  "production summary rows differ from the recomputed wire")
    speculation = None
    if contract.mtp_contract.contract_for_arm(expected['arm'])['enabled'] and complete:
        details = [row.get("spec_tokens_details") for row in rows if contract.response_completed(row)]
        _need(all(type(row) is dict for row in details), "MTP details missing from completed response")
        speculation = dict(
            response_count=len(details),
            spec_verify_ct=sum(row["spec_verify_ct"] for row in details),
            spec_num_proposed_drafts=sum(row["spec_num_proposed_drafts"] for row in details),
            spec_num_correct_drafts=sum(row["spec_num_correct_drafts"] for row in details),
        )
        _need(speculation["spec_verify_ct"] > 0, "MTP produced no verification telemetry")
    result = dict(schema=1, mode=contract.MODE, arm=expected["arm"], binding=dict(expected), counts=counts, rows=rows,
                wire_complete=complete, qualification=False, source_promotion=False,
                code_correctness_graded=False,
                speculation=speculation,
                status="HTTP_WIRE_COMPLETE" if complete else "HTTP_WIRE_INCOMPLETE",
                independent_release_required=True)
    result['paired_groups'] = _paired_controls(directory,names,rows,groups,summary,complete)
    return result


def _paired_controls(directory,names,rows,groups,summary,complete):
    results=[];stopped=False;previous=None
    by_case={row.get('case'):row for row in rows}
    frozen=contract.requests()
    for group in groups:
        path='controls/group-'+group['id']+'.json'
        concurrent='controls/concurrent-'+group['id']+'.json'
        if concurrent in names:
            _need(group['concurrency']==2,'unexpected C1 concurrency evidence')
            record=_object(directory/concurrent)
            _need(set(record)=={'schema','ordinals','joined','failures','native_batch_size_proven'} and
                record['schema']==1 and record['ordinals']==group['ordinals'] and
                type(record['joined']) is bool and type(record['failures']) is dict and
                record['native_batch_size_proven'] is False,'paired owned thread record')
            if path in names:
                _need(record['joined'] is True and record['failures']=={},'paired thread cleanup failed')
        if path not in names:
            stopped=True
            continue
        _need(not stopped,'paired group evidence after stop')
        if group['concurrency']==2:_need(concurrent in names,'paired owned threads evidence absent')
        timing=[];responses=[]
        for ordinal in group['ordinals']:
            case=frozen[ordinal];prefix=f"{ordinal:02d}-{case['case']}"
            intent=_object(directory/f'requests/{prefix}.intent.json')
            sent=_object(directory/f'requests/{prefix}.send-returned.json')
            received=_object(directory/f'responses/{prefix}.record.json')
            timing.append(dict(prepared=intent['prepared_monotonic'],sent=sent['returned_monotonic'],received=received['received_monotonic']))
            _need(case['case'] in by_case,'group response absent')
            responses.append(by_case[case['case']])
        result=contract.paired_wire_group(group,timing,responses)
        _need(_object(directory/path)==result,'paired durable group timing summary differs')
        if previous is not None:
            _need(previous['received_monotonic']<=result['prepared_monotonic'],'paired frozen execution order differs')
        results.append(result);previous=result
    failure='controls/paired.failure.json'
    if failure in names:
        record=_object(directory/failure)
        _need(set(record)=={'schema','failure'} and record['schema']==1 and type(record['failure']) is str and record['failure'],
            'paired failure record invalid')
        _need(summary.get('paired_error')==record['failure'],'paired failure summary differs')
    else:
        _need(summary.get('paired_error') is None,'paired unretained failure')
    _need(summary.get('groups')==results,'paired client/reader groups differ')
    if complete:
        _need(len(results)==len(groups) and failure not in names,'paired completed wire lacks complete groups')
    return results
