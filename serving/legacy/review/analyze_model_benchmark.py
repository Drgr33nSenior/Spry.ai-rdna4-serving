"""Offline descriptive comparison of two complete inner-model timing records.

This tool neither starts work nor authenticates protected-host cleanup. Input
measurements must come from the separately reviewed physical owner workflow.
"""
import argparse
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import statistics


_ENTRY = Path(__file__).resolve().with_name("model_benchmark.py")
_MODULE_SPEC = importlib.util.spec_from_file_location("spry_benchmark_contract", _ENTRY)
bench = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(bench)
MAXIMUM_BYTES = 8 << 20


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON member")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def read_json(path, expected_sha256=None):
    raw = bench.read_bounded_file(Path(path), MAXIMUM_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("spec digest mismatch")
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value, digest


def checked_arm(result, arm, spec_sha256, profile_contract=None, comparison_contract=None):
    """Recompute the complete native-record contract, without native imports."""
    required = {"schema", "scope", "arm", "status", "spec_sha256", "timed_out", "exits",
                "groups_observed_absent", "parent_error", "cleanup_errors", "rank_parse_errors",
                "qualification", "serving_quality_qualified", "http_throughput_measured",
                "independent_release_required", "ranks", "samples"}
    if (not required.issubset(result) or result.get("schema") != 1 or result.get("scope") != bench.SCOPE
            or result.get("arm") != arm
            or result.get("status") != "MODEL_BENCHMARK_COMPLETE"
            or result.get("spec_sha256") != spec_sha256
            or result.get("timed_out") is not False
            or result.get("exits") != [0, 0]
            or any(type(value) is not int for value in result["exits"])
            or result.get("groups_observed_absent") != [True, True]
            or any(value is not True for value in result["groups_observed_absent"])
            or result.get("parent_error") is not None
            or result.get("cleanup_errors") != []
            or result.get("rank_parse_errors") != []
            or result.get("qualification") is not False
            or result.get("serving_quality_qualified") is not False
            or result.get("http_throughput_measured") is not False
            or result.get("independent_release_required") is not True):
        raise ValueError("incomplete or unbound inner-model arm")
    ranks = result.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError("exactly two rank records required")
    for rank in ranks:
        if (not isinstance(rank, dict) or rank.get("qualification") is not False
                or rank.get("serving_quality_qualified") is not False
                or type(rank.get("rank")) is not int):
            raise ValueError("rank qualification contract")
        for batch in rank.get("batches", []):
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in batch.get("step_seconds", [])):
                raise ValueError("positive finite per-rank step timing required")
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   for row in batch.get("generated_logprobs", []) for value in row):
                raise ValueError("finite numeric logprobs required")
    if comparison_contract is not None:
        bench.comparison().validate_contract(comparison_contract)
        if (result.get("experiment_mode") != comparison_contract['mode'] or
                result.get("validation_comparison") != comparison_contract):
            raise ValueError("comparison result mode/contract")
    elif "validation_comparison" in result or bench.comparison_mode(result):
        raise ValueError("comparison result cannot use legacy acceptance")
    samples = bench.summarize(ranks, arm, profile_contract, spec_sha256, comparison_contract)
    if result.get("samples") != samples:
        raise ValueError("stored summary differs from rank measurements")
    for sample in samples:
        for key in ("prefill_rank_max_seconds", "decode_31_seconds",
                    "forward_sample_total_seconds", "batch_wall_rank_max_seconds"):
            value = sample[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite timing required")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in sample["decode_rank_max_step_seconds"]):
            raise ValueError("positive finite decode steps required")
    return samples


def checked_http_arm(result, arm, spec_sha256):
    """Check the narrow owner-facing result from the raw HTTP evidence reader.

    Full wire/gate/source validation happens in ``serving_integration``.  This
    function deliberately does not reinterpret HTTP latency as model quality.
    """
    contract_spec = importlib.util.spec_from_file_location('final_readout_contract', Path(__file__).with_name('mtp_contract.py'))
    final_contract = importlib.util.module_from_spec(contract_spec)
    contract_spec.loader.exec_module(final_contract)
    logical = final_contract.owner_arm_map()
    if (not isinstance(result, dict) or result.get("schema") != 1 or
            result.get("mode") != "native-http-fp8-sprint01-v1" or
            result.get("status") != "HTTP_ARM_COMPLETE" or result.get("owner_arm") != arm or
            result.get("logical_arm") != logical[arm] or result.get("qualification") is not False or
            result.get("binding", {}).get("spec_sha256") != spec_sha256 or
            result.get("wire", {}).get("status") != "HTTP_WIRE_COMPLETE" or
            result.get("wire", {}).get("wire_complete") is not True):
        raise ValueError("incomplete or unbound HTTP arm")
    if final_contract.contract_for_owner_arm(arm)['enabled']:
        telemetry = result["wire"].get("speculation")
        if not (isinstance(telemetry, dict) and type(telemetry.get("spec_verify_ct")) is int and
                telemetry["spec_verify_ct"] > 0):
            raise ValueError("MTP arm lacks verified speculation telemetry")
    return result


def _statistics(values):
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def compare_outputs(left, right):
    """Only equal generated tokens with equal preceding histories are comparable."""
    observations = []
    for ordinal, (a, b) in enumerate(zip(left["ranks"][0]["batches"],
                                           right["ranks"][0]["batches"])):
        for request, request_id in enumerate(a["request_ids"]):
            x, y = a["generated_ids"][request], b["generated_ids"][request]
            first = next((index for index, pair in enumerate(zip(x, y))
                          if pair[0] != pair[1]), None)
            common = len(x) if first is None else first
            gaps = []
            for rank in (0, 1):
                p = left["ranks"][rank]["batches"][ordinal]["generated_logprobs"][request]
                q = right["ranks"][rank]["batches"][ordinal]["generated_logprobs"][request]
                gaps.extend({"rank": rank, "generated_position": index,
                             "absolute_gap": abs(p[index] - q[index])}
                            for index in range(common))
            observations.append({"cell": a["cell"], "sample": a["sample"],
                "measured": a["measured"], "request_id": request_id,
                "bf16_generated_ids": x, "kv4_generated_ids": y,
                "generated_ids_equal": first is None,
                "first_different_generated_token": first,
                "matching_generated_common_prefix_tokens": common,
                "largest_common_prefix_logprob_gap": (
                    max(gaps, key=lambda item: item["absolute_gap"]) if gaps else None)})
    return observations


def capture_arm(spec, result, directory):
    """Read both real capture files and join to native workers and responses."""
    if not bench.capture_mode(spec):
        raise ValueError('capture reader requires explicit capture mode')
    module = bench.capture_module(spec)
    expected_sources = {name: {key: row[key] for key in ('path', 'sha256', 'bytes')}
        for name, row in spec['validation_comparison']['capture']['sources'].items()}
    decoded = []
    for rank, row in enumerate(result['ranks']):
        if spec['experiment_mode'] == 'native-decode-layer-capture-v1':
            operation = module.read_operation(Path(directory) / f'rank-{rank}-operation.json',
                bench.capture_binding(spec, row['native_identity']))
            if operation['phase'] != 'capture_closed' or operation['complete'] is not True:
                raise ValueError('capture operation journal incomplete')
        if bench.native_capture_mode(spec):
            identity = row.get('native_cache_identity')
            native = row['native_identity']
            bench.comparison().validate_native_cache_identity(identity, result['arm'], rank,
                native['session_sha256'], native['spec_sha256'], native['process']['pid'],
                spec['validation_comparison'])
            raw_identity = bench.read_bounded_file(Path(directory) / f'rank-{rank}-native-cache.json', 8192)
            expected_identity = json.dumps(identity, sort_keys=True, allow_nan=False).encode() + b'\n'
            if raw_identity != expected_identity:
                raise ValueError('durable native cache identity differs from rank observation')
        report = row.get('capture')
        if (not isinstance(report, dict) or report.get('records_complete') is not True
                or report.get('qualification') is not False or report.get('file') != f'rank-{rank}-capture.bin'
                or not bench.valid_digest(report.get('sha256'))):
            raise ValueError('capture rank report incomplete')
        workspace = report.get('workspace')
        if (type(workspace) is not dict or workspace.get('cpu_tensor_cap_bytes') != 16 << 20
                or workspace.get('device_cap_bytes') != 8 << 20
                or type(workspace.get('largest_cpu_tensor_copy_bytes')) is not int
                or not 0 <= workspace['largest_cpu_tensor_copy_bytes'] <= 16 << 20
                or type(workspace.get('maximum_observed_device_current_growth_bytes')) is not int
                or not 0 <= workspace['maximum_observed_device_current_growth_bytes'] <= 8 << 20
                or (spec['experiment_mode'] == 'native-async-prefill-layer-capture-v1'
                    and workspace.get('deferred_device_buffer_bytes') != 2703360)
                or workspace.get('transient_peak_measured') is not False):
            raise ValueError('capture workspace observations missing or outside configured bounds')
        value = module.read_file(Path(directory) / report['file'], bench.capture_binding(spec, row['native_identity']),
            row['batches'], expected_sha256=report['sha256'],
            expected_prompts=spec['frozen_prompt_ids'])
        preflight = value['metadata'].get('pre_import')
        masks = {'/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/srt/' + name: 1 << 20
            for name in ('model_executor', 'layers', 'layers/attention', 'models')}
        loaders = {name: dict(origin=source['path'], loader='SourceFileLoader') for name, source in expected_sources.items()}
        expected_preflight = dict(sources=expected_sources, masks=masks, loaders=loaders,
            pycache_prefix='/mnt/capture-bytecode', bytecode_disabled=True)
        if preflight != expected_preflight or row.get('capture_preflight') != expected_preflight:
            raise ValueError('durable native pre-import source/mask/loader observations missing or substituted')
        if (value['metadata'].get('sources') != expected_sources
                or value['callback_elapsed_ns'] != report.get('callback_elapsed_ns')
                or type(report.get('bytes')) is not int
                or report['bytes'] != (Path(directory) / report['file']).stat().st_size):
            raise ValueError('capture source/elapsed observations differ')
        if spec['experiment_mode'] == 'native-prefill-row-audit-v1':
            value['prefill_row_audit'] = prefill_row_audit(spec, row, value, directory)
        decoded.append(value)
    return decoded


def prefill_row_audit(spec, rank_record, capture, directory):
    """Join complete-row hashes to the independently validated last-row capture."""
    helper = bench.row_audit_module(spec)
    rank = rank_record['rank']
    report = rank_record['capture'].get('prefill_row_audit')
    if (type(report) is not dict or set(report) != {'file', 'sha256', 'bytes', 'records_complete',
            'record_count', 'qualification'} or report['file'] != f'rank-{rank}-prefill-rows.jsonl'
            or report['records_complete'] is not True or report['qualification'] is not False
            or type(report['record_count']) is not int
            or report['record_count'] != len(helper.ROW_AUDIT_RECORDS)
            or type(report['bytes']) is not int
            or not 0 < report['bytes'] <= helper.FILE_CAP
            or not bench.valid_digest(report['sha256'])):
        raise ValueError('prefill row audit complete rank report')
    prompts = [hashlib.sha256(bench.capture_module(spec).canonical(ids)).hexdigest()
        for ids in spec['frozen_prompt_ids']['C2-1024']]
    contract = spec['validation_comparison']['capture']['prefill_row_audit']
    if contract['prompt_sha256'] != prompts:
        raise ValueError('prefill row audit frozen prompts')
    expected = {}
    for ordinal, boundary, tensor in helper.ROW_AUDIT_RECORDS:
        field = 'v' if tensor == 'value' else 'r'
        rows = [capture['layer_rows'][ordinal, row, 0, boundary] for row in (0, 1)]
        values = [row[field] for row in rows]
        if any(value is None for value in values) or values[0][:2] != values[1][:2]:
            raise ValueError('prefill row audit binary dtype/layout')
        expected[ordinal, boundary, tensor] = dict(dtype=values[0][0],
            sha256=[value[2] for value in values], monotonic_ns=[row['n'] for row in rows])
    path = Path(directory) / report['file']
    observed = helper.read_file(path, bench.capture_binding(spec, rank_record['native_identity']),
        {ordinal: prompts for ordinal in range(24, 30)}, report['sha256'], expected)
    if path.stat().st_size != report['bytes']:
        raise ValueError('prefill row audit retained size differs')
    for item in observed['records']:
        layers = [capture['layer_rows'][item['ordinal'], row, 0, item['boundary']] for row in (0, 1)]
        completion = item['completion']
        if any(layer['o'] != completion['kind'] for layer in layers):
            raise ValueError('prefill row audit completion/binary binding')
        if completion['kind'] == 'CarriedSum':
            expected_group = dict(qualname='sglang.srt.distributed.parallel_state.GroupCoordinator',
                ranks=[0, 1], world_size=2, rank=rank, rank_in_group=rank)
            if (completion['group'] != expected_group or completion['branch'] != 'group-all-reduce'
                    or completion['redistribution'] != 'absent'):
                raise ValueError('prefill row audit exact native TP completion')
        elif completion['kind'] != 'none':
            raise ValueError('prefill row audit unsupported completion')
    return observed


def compare_prefill_rows(captures):
    """Describe every same-boundary repeat without overriding numerical refusal."""
    comparisons = []
    for rank in (0, 1):
        indexed = {arm: {(row['ordinal'], row['boundary'], row['tensor']): row
            for row in captures[arm][rank]['prefill_row_audit']['records']} for arm in ('bf16', 'kv4')}
        samples = [(arm, sample, 24 + sample) for arm in ('bf16', 'kv4') for sample in range(6)]
        for left, right in combinations(samples, 2):
            for boundary, names in ((0, ('value',)), (16, ('value', 'residual')), (32, ('value', 'residual'))):
                for tensor in names:
                    a, b = (indexed[arm][ordinal, boundary, tensor] for arm, _, ordinal in (left, right))
                    context_equal = (a['dtype'], a['shape'], a['prompt_sha256'], a['completion']) == (
                        b['dtype'], b['shape'], b['prompt_sha256'], b['completion'])
                    comparisons.append(dict(rank=rank, left=list(left), right=list(right),
                        relation='within_start' if left[0] == right[0] else 'cross_start',
                        boundary=boundary, tensor=tensor, context_equal=context_equal,
                        complete_matrix_equal=a['raw_sha256'] == b['raw_sha256'],
                        last_rows_equal=a['last_row_sha256'] == b['last_row_sha256'],
                        qualification=False))
    if len(comparisons) != 660:
        raise ValueError('prefill row audit pair coverage')
    return dict(schema=1, mode='native-prefill-row-audit-v1', records_complete=True,
        comparisons=comparisons, qualification=False, cause_identified=False,
        interpretation='Full row hashes do not establish equality of unmeasured KV/recurrent contents or future collective execution.')


def native_repeatability(left, right, captures):
    """Compare every repeat pair after complete arm and capture validation."""
    mode = left.get('experiment_mode')
    if not bench.comparison().native_mode(mode):
        raise ValueError('native repeatability requires explicit native capture mode')
    records = {'bf16': left, 'kv4': right}
    if type(captures) is not dict or set(captures) != set(records):
        raise ValueError('native repeatability requires both complete captures')
    expected_keys = {(cell_index * 6 + sample, request, step)
        for cell_index, (_, count, _) in enumerate(bench.CELLS)
        for sample in range(6) for request in range(count) for step in range(32)}
    for arm, value in records.items():
        if (value.get('experiment_mode') != mode or type(value.get('ranks')) is not list
                or len(value['ranks']) != 2 or type(captures[arm]) is not list or len(captures[arm]) != 2):
            raise ValueError('native repeatability mode/two-rank structure')
        for rank, row in enumerate(value['ranks']):
            capture = captures[arm][rank]
            if (row.get('rank') != rank or type(row.get('batches')) is not list
                    or len(row['batches']) != 30 or capture.get('mode') != mode
                    or capture.get('records_complete') is not True
                    or capture.get('binding', {}).get('arm') != arm
                    or capture.get('binding', {}).get('rank') != rank
                    or set(capture.get('steps', {})) != expected_keys):
                raise ValueError('native repeatability complete schedule/capture membership')
            for ordinal, batch in enumerate(row['batches']):
                cell, count, _ = bench.CELLS[ordinal // 6]
                if (batch.get('cell') != cell or batch.get('sample') != ordinal % 6
                        or len(batch.get('request_ids', [])) != count
                        or len(batch.get('generated_ids', [])) != count
                        or len(batch.get('generated_logprobs', [])) != count
                        or any(len(values) != 32 for values in
                            batch['generated_ids'] + batch['generated_logprobs'])):
                    raise ValueError('native repeatability fixed response shape')
    boundary_order = ['hidden', 'model_logits', 'sampler_input', 'post_sample_logits', 'normalized_scores']
    pairs = []
    pair_counts = dict(within_start=0, cross_start=0)
    request_pairs = step_pairs = 0
    for rank in (0, 1):
        for cell_index, (cell, count, _) in enumerate(bench.CELLS):
            samples = [(arm, sample, cell_index * 6 + sample)
                for arm in ('bf16', 'kv4') for sample in range(6)]
            for a, b in combinations(samples, 2):
                relation = 'within_start' if a[0] == b[0] else 'cross_start'
                pair_counts[relation] += 1
                batches = [records[arm]['ranks'][rank]['batches'][ordinal] for arm, _, ordinal in (a, b)]
                requests = []
                for request in range(count):
                    steps = []
                    passed = True
                    for step in range(32):
                        x, y = [captures[arm][rank]['steps'][(ordinal, request, step)]
                            for arm, _, ordinal in (a, b)]
                        aligned = x['c'] == y['c']
                        same_token = batches[0]['generated_ids'][request][step] == batches[1]['generated_ids'][request][step]
                        comparable = aligned and same_token
                        score_equal = x['s'] == y['s'] if comparable else None
                        score_gap = abs(batches[0]['generated_logprobs'][request][step] -
                            batches[1]['generated_logprobs'][request][step]) if comparable else None
                        mask = sum(1 << index for index, values in enumerate(zip(x['h'], y['h']))
                            if values[0] == values[1]) if aligned else None
                        passed = passed and aligned and same_token and score_equal is True and mask == 31
                        steps.append([step, aligned, same_token, comparable, score_equal, score_gap, mask])
                        step_pairs += 1
                    requests.append(dict(request=request,
                        left_request_id=batches[0]['request_ids'][request],
                        right_request_id=batches[1]['request_ids'][request], passed=passed, steps=steps))
                    request_pairs += 1
                pairs.append(dict(relation=relation, rank=rank, cell=cell,
                    left=list(a), right=list(b), passed=all(row['passed'] for row in requests), requests=requests))
    if pair_counts != {'within_start': 300, 'cross_start': 360} or request_pairs != 924 or step_pairs != 29568:
        raise ValueError('native repeatability pair coverage')
    return dict(schema=1, mode=mode, passed=all(pair['passed'] for pair in pairs),
        pair_counts=pair_counts, request_pairs=request_pairs, step_pairs=step_pairs,
        sample_columns=['arm', 'sample', 'ordinal'],
        step_columns=['step', 'conditioning_equal', 'generated_ids_equal', 'score_comparable',
            'selected_logprob_equal', 'selected_logprob_gap', 'boundary_equal_mask'],
        boundary_order=boundary_order, boundary_equal_mask_all=31, pairs=pairs,
        score_rule='Compare scores only when conditioning histories and generated token identity match.',
        raw_vectors='Every generated ID and score vector remains in the bound per-rank batch records.',
        tolerance=None, qualification=False, cause_identified=False, independent_cleanup_required=True)


def analyze(spec, spec_sha256, left, right, capture_directories=None):
    if spec.get("experiment_mode") == "native-model-context-retrieval-32k-v1":
        checked_arm(left, "bf16", spec_sha256)
        checked_arm(right, "kv4", spec_sha256)
        return bench._GRAPH.analyze(bench.__dict__, spec, spec_sha256, left, right, capture_directories)
    bench.validate_spec(spec)
    if capture_directories is None:
        raise ValueError('callback profile requires artifact directories')
    for arm, record in (('bf16', left), ('kv4', right)):
        bench.read_callback_profile_evidence(spec, capture_directories[arm], record)
        bench.read_stall_trace_evidence(spec, capture_directories[arm], record)
    profile = spec.get("cache_profile")
    comparison = spec.get("validation_comparison")
    samples = {"bf16": checked_arm(left, "bf16", spec_sha256, profile, comparison),
               "kv4": checked_arm(right, "kv4", spec_sha256, profile, comparison)}
    cells = []
    metrics = ("prefill_rank_max_seconds", "decode_31_seconds", "forward_sample_total_seconds")
    for name, concurrency, length in bench.CELLS:
        arms = {}
        for arm in ("bf16", "kv4"):
            rows = [dict(row, decode_tokens_per_second=31 * concurrency / row["decode_31_seconds"])
                    for row in samples[arm] if row["cell"] == name]
            arms[arm] = {"warmup": rows[0], "measured_samples": rows[1:],
                "statistics": {key: _statistics([row[key] for row in rows[1:]])
                               for key in (*metrics, "decode_tokens_per_second")}}
        faster = {key: 100 * (arms["bf16"]["statistics"][key]["median"]
                              / arms["kv4"]["statistics"][key]["median"] - 1)
                  for key in metrics}
        cells.append({"cell": name, "concurrency": concurrency, "input_tokens": length,
                      "arms": arms, "kv4_faster_percent_by_latency_median": faster})
    result = {"schema": 1, "status": "OFFLINE_COMPARISON_COMPLETE",
        "scope": "inner-model-timing-only", "spec_sha256": spec_sha256,
        "qualification": False, "serving_quality_qualified": False,
        "physical_campaign_complete": False, "independent_release_verified": False,
        "accepted_live_cleanup_authority": False, "http_throughput_measured": False,
        "ttft_measured": False, "end_to_end_throughput_measured": False,
        "overall_winner": None, "cells": cells, "output_comparisons": compare_outputs(left, right),
        "interpretation": ["One fixed-order two-arm inner-model screen, not independent-start evidence.",
            "Warmups are retained but excluded from the five-sample statistics.",
            "Positive latency-ratio percent means KV4 faster in that cell and metric only.",
            "Logprob comparisons stop before the first different generated token.",
            "Worker/process-group records do not establish protected-host or device release.",
            "No quality, HTTP, native performance or general serving winner is established."]}
    if profile is not None:
        result["cache_attribution"] = cache_attribution(right)
    if comparison is not None:
        module = bench.comparison()
        result.update(experiment_mode=comparison['mode'], arm_labels=comparison["arm_labels"],
            exact_output_gate=module.exact_outputs(left, right),
            telemetry_available=all(row["batch_telemetry"]["sensors_available"]
                for arm in (left, right) for row in arm["ranks"]),
            source_promotion=False)
        for cell in cells:
            cell["candidate_faster_percent_by_latency_median"] = cell.pop("kv4_faster_percent_by_latency_median")
        if module.native_mode(comparison['mode']):
            result["interpretation"][0] = "Two fresh starts use the same native BF16 KV cache and current FP8 weights."
            result["interpretation"][2] = "Latency ratios describe instrumented starts only and do not establish native performance."
            result["interpretation"].append("Internal bf16/kv4 ownership keys mean native BF16 KV start 1/start 2; BF16 cache does not mean BF16 weights.")
            result["interpretation"].append("Exact agreement is non-reproduction; it does not establish a cure or attribute earlier failures to V4.")
        else:
            result["reference_cache_attribution"] = cache_attribution(left)
            result["interpretation"][2] = "Positive latency-ratio percent means candidate faster than V4 compact in that cell only."
            result["interpretation"].append("Internal bf16/kv4 ownership keys mean compact reference/candidate in this explicit mode; neither arm is ordinary BF16 KV.")
        result["interpretation"].append("Boundary sensor reads are outside timed forwards but can perturb TP arrival and thermal history. They are not DRAM bandwidth counters.")
    if bench.capture_mode(spec):
        if type(capture_directories) is not dict or set(capture_directories) != {'bf16', 'kv4'}:
            raise ValueError('both raw capture directories required')
        raw = {arm: capture_arm(spec, record, capture_directories[arm]) for arm, record in (('bf16', left), ('kv4', right))}
        module = bench.capture_module(spec)
        result['capture'] = dict(records_complete=True,
            comparisons=[module.compare(raw['bf16'][rank], raw['kv4'][rank]) for rank in (0, 1)],
            qualification=False, independent_cleanup_required=True, timing_is_instrumented=True)
        if bench.comparison().native_mode(comparison['mode']):
            result['native_repeatability_gate'] = native_repeatability(left, right, raw)
        if spec['experiment_mode'] == 'native-prefill-row-audit-v1':
            result['prefill_row_audit'] = compare_prefill_rows(raw)
    return result


def cache_attribution(right):
    """Use the same slower rank as each model timing, without adding rank times."""
    module = bench.authenticated_adjacent("cache_profile.py", bench.CACHE_PROFILE_SHA256)
    rows = []
    for ordinal, batch in enumerate(right["ranks"][0]["batches"]):
        phases = {"prefill": {}, "decode_31": {}}
        for step in range(32):
            rank = max((0, 1), key=lambda r: right["ranks"][r]["batches"][ordinal]["step_seconds"][step])
            record = right["ranks"][rank]["cache_profile"]["steps"][ordinal * 32 + step]
            parts = {"metadata": record["metadata_ns"], "other_model": record["total_ns"] -
                     record["metadata_ns"] - sum(v[1] for v in record["layers"])}
            for values in record["layers"]:
                physical, full, append, attention, native_append, native_read, host, validation, reader = values[:9]
                prefix = "protected_" if physical in module.PROTECTED else "compact_"
                costs = {"adapter_checks_and_views": full - append - attention,
                         "append_host": append - native_append, "append_native": native_append,
                         "attention_host": attention - native_read,
                         "attention_native_overhead": native_read - host - validation - reader,
                         "native_host_checks": host, "native_validation_and_wait": validation,
                         "native_reader_and_wait": reader}
                for name, value in costs.items():
                    parts[prefix + name] = parts.get(prefix + name, 0) + value
            if sum(parts.values()) != record["total_ns"] or any(v < 0 for v in parts.values()):
                raise ValueError("cache profile accounting partition")
            dest = phases["prefill" if step == 0 else "decode_31"]
            for name, value in parts.items():
                dest[name] = dest.get(name, 0) + value
        rows.append(dict(ordinal=ordinal, cell=batch["cell"], sample=batch["sample"],
                         measured=batch["measured"], partition_ns=phases))
    cells = []
    for cell, _, _ in bench.CELLS:
        selected = [r for r in rows if r["cell"] == cell and r["measured"]]
        cells.append({"cell": cell, "statistics_seconds": {
            phase: {name: _statistics([r["partition_ns"][phase][name] / 1e9 for r in selected])
                    for name in selected[0]["partition_ns"][phase]}
            for phase in ("prefill", "decode_31")}})
    return {"schema": 1, "qualification": False, "cells": cells, "samples": rows,
            "units": "inclusive host wait attribution partition, seconds; raw integers are nanoseconds",
            "rank_rule": "slower measured model rank per step; both raw rank profiles retained",
            "caveats": ["A synchronization can charge earlier queued work to the observed call.",
                "Native reader time combines all reader kernels and its existing stream wait.",
                "Append native time combines validation, packing/write and existing waits.",
                "Getter and Python observer overhead is included; this is instrumented timing.",
                "Phase medians need not sum to the median of a complete window."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--kv4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec, digest = read_json(args.spec, args.expected_spec_sha256)
    left, left_digest = read_json(args.bf16)
    right, right_digest = read_json(args.kv4)
    directories = {'bf16': Path(args.bf16).parent, 'kv4': Path(args.kv4).parent} if bench.capture_mode(spec) else None
    report = analyze(spec, digest, left, right, directories)
    report["input_result_sha256"] = {"bf16": left_digest, "kv4": right_digest}
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # This identifies this offline parser contract, not the physically executed file.
    report["analysis_contract_source_sha256"] = hashlib.sha256(_ENTRY.read_bytes()).hexdigest()
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("absolute output under existing owned parent required")
    bench.write_result(output, report)


if __name__ == "__main__":
    main()
