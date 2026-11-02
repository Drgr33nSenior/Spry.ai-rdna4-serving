"""Compare two sealed, graded W5 campaigns; never launch or grant execution."""
import argparse
import html
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build as builder
import prepare_measurement
from launcher.run_native import checked_module,held,strict
from spry_common.canon import canonical_line as canonical,digest,need

SEALER_SHA256='082835dabf0bf1e96d9bba6f08918914b01a1306a5a2cd939a0c9309ad334e18'
ARMS={'bf16':'control','kv4':'mtp'}
WORK_FIELDS=('case','body_sha256','prompt_ids','output_ids','speculative_work','cache_work','sampling_parameters','finish_reason')


def read(path,maximum=8<<20):return strict(held(path,maximum=maximum))
def positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0
def ids(value):return type(value) is list and bool(value) and all(type(x) is int and 0<=x<248320 for x in value)


def receipt_codec(package,manifest):
    """Load the codec with its authenticated contract, preserving ambient modules."""
    missing=object();names=('mtp_contract','request_receipts')
    previous={name:sys.modules.get(name,missing) for name in names}
    try:
        for name in names:
            module=checked_module(package/'review',name,manifest['package_files']['review/'+name+'.py'])
        return module
    finally:
        for name,value in previous.items():
            if value is missing:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def load_side(root,pin,sealer):
    root=Path(root);execution_seal=sealer.verify(root,pin);result=read(root/'RESULT.json')
    need(result['decoded_rows']==result['natural_quality_passes']==result['total_responses']==352
         and result['natural_quality_failures']==0 and result['independent_gpu_release'] is result['independent_cpu_release'] is True,
         'complete graded and released W5 result required')
    cpu=Path(result['cpu_evidence_seal']['root'])
    need(sealer.verify(cpu,result['cpu_evidence_seal']['manifest_sha256'])==result['cpu_evidence_seal'],'CPU evidence seal differs')
    summary=read(cpu/'RESULT-SUMMARY.json')
    need(digest(held(cpu/'RESULT-SUMMARY.json'))==result['quality_summary_sha256'] and
         summary['status']=='NATIVE_CPU_FINAL_QUALITY_AND_RELEASE_VALIDATED' and
         summary['quality_pass'] is summary['long_answer_coverage_pass'] is summary['independent_release_verified'] is True
         and summary['natural_rows']==summary['decoded_rows']==len(summary['grades'])==352
         and all(row['quality_passed'] is True for row in summary['grades']), 'native quality/release result differs')
    proof=read(cpu/'ACTUAL-INPUT-BINDING.json')
    need((proof['status']=='ACTUAL_352_INPUT_BOUND_BY_RECOMPUTED_OWNER_RESOURCE_RELEASE_HTTP' or
          proof.get('schema')==2 and proof['status']=='ACTUAL_PAIRED_INPUT_BOUND_BY_RECOMPUTED_OWNER_RESOURCE_RELEASE_HTTP'
          and proof.get('rows')==352),'complete W5 input binding required')
    seals=proof['replay_provenance']['seals']
    need(set(seals)=={'operation','workload','protected','source_author'},'four original evidence seals required')
    for seal in seals.values():need(sealer.verify(Path(seal['root']),seal['manifest_sha256'])==seal,'original evidence changed')
    input_raw=held(cpu/'source/QUALITY-CPU-INPUT.json',proof['input_sha256'],53<<20);quality=strict(input_raw)
    need(len(input_raw)==proof['input_bytes'] and proof['input_sha256']==result['actual_input_sha256']
         and len(quality['rows'])==352 and quality['cancellations']==[],'complete bound quality input required')
    rows={(row['arm'],row['ordinal']):row for row in quality['rows']}
    need(len(rows)==352,'unique quality rows required')
    package=Path(seals['source_author']['root'])/'package-v2'
    need(str(package)==proof['request']['package_root'],'authored package mapping differs')
    workload=Path(seals['workload']['root']);summaries={}
    for arm in ARMS:
        tree=workload/'results'/(arm+'-http-client-tree')/'paired';value=read(tree/'RUN-SUMMARY.json')
        need(value['executed_case_count']==value['frozen_case_count']==len(value['rows'])==176
             and value['stopped_early'] is False and value['paired_error'] is None,'complete paired HTTP summary required')
        need([row['ordinal'] for row in value['rows']]==list(range(176)),'exact HTTP ordinal order required')
        summaries[arm]=dict(tree=tree,value=value)
    return dict(root=root,result=result,cpu=cpu,proof=proof,seals=seals,package=package,
                rows=rows,summaries=summaries,execution_seal=execution_seal)


def request_work(side,arm,row,codec):
    quality=side['rows'][(arm,row['ordinal'])];contract=row['contract'];tree=side['summaries'][arm]['tree']
    relative=row['wire']['raw_body_path'];builder.relative_name(relative)
    raw=codec.read(tree/relative,2<<20)
    need(digest(raw)==row['wire']['response_sha256']==quality['response_raw_sha256'],'raw response identity differs')
    body=strict(raw);need(type(body.get('choices')) is list and len(body['choices'])==1,'one complete response choice required')
    choice=body['choices'][0];prompt=choice['prompt_token_ids'];output=choice['response_token_ids'];meta=choice['meta_info']
    need(ids(prompt) and ids(output) and output==quality['output_token_ids'] and
         len(prompt)==contract['prompt_token_count'] and len(output)==contract['response_id_count']==contract['reported_completion_tokens']
         and choice['message']['content']==quality['wire_text'],'actual prompt/output vectors differ from validated readout')
    need(row['case']==quality['case'] and row['body_sha256']==quality['request_body_sha256']
         and choice['finish_reason']==quality['finish_reason'],'source case/body/terminal join differs')
    need(type(meta['cached_tokens']) is int and meta['cached_tokens']>=0 and type(meta['num_retractions']) is int
         and meta['num_retractions']==0,'actual cache/retraction work required')
    return dict(case=row['case'],body_sha256=row['body_sha256'],prompt_ids=prompt,output_ids=output,
        speculative_work=contract['spec_tokens_details'],cache_work=dict(cached_tokens=meta['cached_tokens'],
            cached_tokens_details=meta.get('cached_tokens_details'),num_retractions=meta['num_retractions']),
        sampling_parameters=contract['sampling_parameters'],finish_reason=choice['finish_reason'])


def work_differences(arm,ordinal,left,right):
    differences=[]
    for field in WORK_FIELDS:
        if canonical(left[field])!=canonical(right[field]):
            row=dict(arm=arm,ordinal=ordinal,case=left['case'],field=field,
                     baseline_sha256=digest(canonical(left[field])),cleanup_sha256=digest(canonical(right[field])))
            if field in ('prompt_ids','output_ids'):
                row.update(baseline_count=len(left[field]),cleanup_count=len(right[field]))
            else:row.update(baseline=left[field],cleanup=right[field])
            differences.append(row)
    return differences


def paired_groups(arm,left,right,codec):
    a,b=left['summaries'][arm],right['summaries'][arm]
    ga,gb=a['value']['groups'],b['value']['groups']
    need(len(ga)==len(gb) and len({row['group_id'] for row in ga})==len(ga),'exact unique group inventory required')
    groups=[]
    for x,y in zip(ga,gb,strict=True):
        fields=('group_id','kind','concurrency','excluded_warmup','measured','ordinals')
        need(all(canonical(x[key])==canonical(y[key]) for key in fields),'group order, task or concurrency differs')
        need(type(x['measured']) is type(x['excluded_warmup']) is bool and type(x['concurrency']) is int
             and x['concurrency'] in (1,2),'typed group measurement policy required')
        need(type(x['ordinals']) is list and x['ordinals'] and all(type(n) is int and 0<=n<176 for n in x['ordinals']),
             'source-bound group ordinals required')
        pair={}
        for label,side,source,g in (('baseline',left,a,x),('cleanup',right,b,y)):
            relative='controls/group-'+g['group_id']+'.json';builder.relative_name(relative);control=source['tree']/relative
            raw=held(control,maximum=64<<10) if control.exists() else codec.read(control,64<<10)
            need(strict(raw)==g,'group summary differs from its authenticated logical control record')
            count=sum(len(side['rows'][(arm,n)]['output_token_ids']) for n in g['ordinals'])
            latencies=g['per_request_latency_seconds'];wall=g['makespan_seconds']
            need(type(g['completion_tokens']) is int and g['completion_tokens']==count and positive(wall)
                 and math.isclose(g['received_monotonic']-g['prepared_monotonic'],wall,rel_tol=0,abs_tol=1e-9),
                 'actual group token count or HTTP wall differs')
            need(type(latencies) is list and len(latencies)==len(g['ordinals']) and all(positive(value) for value in latencies)
                 and latencies==[source['value']['rows'][n]['wire']['client_elapsed_seconds'] for n in g['ordinals']],
                 'group/request latency mapping differs')
            pair[label]=dict(completion_tokens=count,wall_seconds=wall,per_request_latency_seconds=latencies,
                combined_tp2_tokens_per_second=count/wall)
        groups.append(dict(arm=arm,logical_arm=ARMS[arm],group_id=x['group_id'],kind=x['kind'],concurrency=x['concurrency'],
            ordinals=x['ordinals'],timing_eligible=x['kind']=='timing' and x['measured'] and not x['excluded_warmup'],**pair))
    return groups


def aggregate(groups):
    result=[]
    for arm in ARMS:
        for concurrency in (1,2):
            selected=[g for g in groups if g['arm']==arm and g['concurrency']==concurrency and g['timing_eligible']]
            need(selected,'all four timing populations required');values={}
            for label in ('baseline','cleanup'):
                tokens=0;wall=0;latencies=[];makespans=[]
                for group in selected:
                    row=group[label]
                    need(type(row['completion_tokens']) is int and row['completion_tokens']>0 and positive(row['wall_seconds'])
                         and all(positive(x) for x in row['per_request_latency_seconds']),'finite measured timing population required')
                    tokens+=row['completion_tokens'];wall+=row['wall_seconds'];makespans.append(row['wall_seconds'])
                    latencies.extend(row['per_request_latency_seconds'])
                values[label]=dict(completion_tokens=tokens,http_wall_seconds=wall,requests=len(latencies),
                    combined_tp2_tokens_per_second=tokens/wall,mean_request_latency_seconds=statistics.mean(latencies),
                    median_request_latency_seconds=statistics.median(latencies),median_group_makespan_seconds=statistics.median(makespans))
            result.append(dict(arm=arm,logical_arm=ARMS[arm],concurrency=concurrency,groups=len(selected),**values,
                observed_wall_change_percent=100*(values['cleanup']['http_wall_seconds']/values['baseline']['http_wall_seconds']-1),
                observed_throughput_change_percent=100*(values['cleanup']['combined_tp2_tokens_per_second']/values['baseline']['combined_tp2_tokens_per_second']-1)))
    return result


def report_html(report):
    cells=report['timing_populations'];maximum=max(row[label]['combined_tp2_tokens_per_second'] for row in cells for label in ('baseline','cleanup'))
    svg=['<svg viewBox="0 0 900 390" role="img" aria-label="Observed combined TP2 output tokens per second">']
    for tick in range(5):
        value=maximum*tick/4;y=320-270*tick/4
        svg.append(f'<path d="M70 {y:.2f}H870" stroke="var(--border,#ccc)"/><text x="60" y="{y+5:.2f}" text-anchor="end">{value:.1f}</text>')
    for index,row in enumerate(cells):
        x=110+index*190
        for offset,label,color in ((0,'baseline','var(--chart-1,#3274a1)'),(60,'cleanup','var(--chart-2,#df8f44)')):
            value=row[label]['combined_tp2_tokens_per_second'];height=270*value/maximum
            svg.append(f'<rect x="{x+offset}" y="{320-height:.2f}" width="50" height="{height:.2f}" fill="{color}"><title>{label}: {value:.4f} tokens/s</title></rect>')
        svg.append(f'<text x="{x+55}" y="350" text-anchor="middle">{row["logical_arm"]} C{row["concurrency"]}</text>')
    svg.append('</svg>')
    title='Matched-work descriptive pilot' if report['matched_work_all_rows'] else 'Work differs: descriptive timing only'
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in (g['logical_arm'],g['group_id'],g['timing_eligible'],
        round(g['baseline']['wall_seconds'],6),round(g['cleanup']['wall_seconds'],6),g['baseline']['completion_tokens'],g['cleanup']['completion_tokens']))+'</tr>' for g in report['groups'])
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Production cleanup pilot</title><style>body{font:16px var(--font-sans,system-ui);color:var(--foreground,#222);margin:0}svg{width:100%;height:330px}svg text{font-size:14px;fill:currentColor}td,th{text-align:left;padding:6px;border-bottom:1px solid var(--border,#ccc)}table{border-collapse:collapse}details{overflow-x:auto;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px var(--font-mono,monospace)}.legend{display:inline-block;width:12px;height:12px;margin-right:5px}.baseline{background:var(--chart-1,#3274a1)}.cleanup{background:var(--chart-2,#df8f44)}</style><h2>'+title+'</h2><p>Production mode, stock W5 attention. Output tokens per second combine both GPUs and include prefill and completion. C1/C2 refer to HTTP concurrency; native batch size is unproven.</p><p><span class="legend baseline"></span>Baseline A &nbsp; <span class="legend cleanup"></span>Cleanup B</p>'+''.join(svg)+'<p>Warmup and non-timing groups are excluded from the chart and pooled metrics. All groups and mismatches are retained below. Reverse-order confirmation has not run; this pilot establishes no causal speedup.</p><details><summary>All group observations</summary><table><tr><th>Arm</th><th>Group</th><th>Timing</th><th>A wall (s)</th><th>B wall (s)</th><th>A tokens</th><th>B tokens</th></tr>'+rows+'</table></details><details><summary>Work mismatches ('+str(len(report['work_differences']))+')</summary><pre>'+html.escape(json.dumps(report['work_differences'],indent=2))+'</pre></details></html>'


def compare(baseline,baseline_pin,candidate,candidate_pin,sealer_path,output,*,paired_cleanup=False):
    need(sys.flags.optimize==0,'unoptimized original seal verifier required')
    need(type(paired_cleanup) is bool,'explicit boolean paired cleanup selection required')
    output=Path(output)
    need(output.is_absolute() and output.resolve()==output and output.parent.is_dir() and not output.exists(),'fresh canonical comparison output required')
    need(not output.is_relative_to(HERE),'comparison output must be outside the canonical source tree')
    sealer_path=Path(sealer_path);need(sealer_path.name=='seal_evidence.py','original evidence sealer required')
    sealer=checked_module(sealer_path.parent,'seal_evidence',SEALER_SHA256)
    a,b=load_side(baseline,baseline_pin,sealer),load_side(candidate,candidate_pin,sealer)
    for side in (a,b):
        for seal in (side['execution_seal'],dict(root=str(side['cpu'])),*side['seals'].values()):
            need(not output.is_relative_to(Path(seal['root'])),'comparison must be outside every sealed input')
    metadata=read(HERE/'SOURCE.json');run=read(b['package'].parent/'config/runs/w5-v2/run.json')
    names=set(metadata['source_files'])|set(metadata['source_extractions'])|{Path(name).name for name in metadata['new_sources']}
    comparison=prepare_measurement.compare_inputs({p.name:held(p) for p in (a['package']/'review').iterdir()},
        {p.name:held(p) for p in (b['package']/'review').iterdir()},run=run,canonical_sources=names,
        protected_sources=metadata['protected_source_files'],paired_cleanup=paired_cleanup)
    manifest=read(b['package']/'MANIFEST.json');codec=receipt_codec(b['package'],manifest)
    differences=[];groups=[]
    for arm in ARMS:
        for left,right in zip(a['summaries'][arm]['value']['rows'],b['summaries'][arm]['value']['rows'],strict=True):
            differences.extend(work_differences(arm,left['ordinal'],request_work(a,arm,left,codec),request_work(b,arm,right,codec)))
        groups.extend(paired_groups(arm,a,b,codec))
    mismatched={(row['arm'],row['ordinal']) for row in differences}
    compared_fields={row['field'] for row in differences}
    comparison['output_token_identity_verified']='output_ids' not in compared_fields
    report=dict(schema=1,status='MATCHED_WORK_DESCRIPTIVE_PILOT' if not differences else 'WORK_MISMATCH_DESCRIPTIVE_ONLY',
        source_comparison=comparison,rows_compared=352,matched_rows=352-len(mismatched),mismatched_rows=len(mismatched),
        matched_work_all_rows=not differences,work_differences=differences,groups=groups,timing_populations=aggregate(groups),
        baseline_operation=a['result']['operation'],cleanup_operation=b['result']['operation'],
        evidence=dict(baseline=a['execution_seal'],cleanup=b['execution_seal']),
        analyzer_sha256=digest(held(Path(__file__).resolve(),maximum=128<<10)),
        request_receipt_codec_sha256=manifest['package_files']['review/request_receipts.py'],evidence_sealer_sha256=SEALER_SHA256,
        raw_output_and_prompt_vectors_compared=True,speculative_and_cache_work_compared=True,
        prompt_token_identity_verified='prompt_ids' not in compared_fields,
        output_token_identity_verified='output_ids' not in compared_fields,
        speculative_work_equal='speculative_work' not in compared_fields,cache_work_equal='cache_work' not in compared_fields,
        native_batch_size_proven=False,concurrency_scope='HTTP client concurrency',
        reverse_order_confirmation_run=False,causal_speedup_established=False,qualification=False,
        metric_scope='Combined TP2 output tokens divided by summed corresponding HTTP group walls; includes prefill and completion; no isolated decode or TTFT claim.')
    for side in (a,b):
        need(sealer.verify(side['root'],side['execution_seal']['manifest_sha256'])==side['execution_seal'],'closed execution changed during comparison')
    output.mkdir(mode=0o700)
    with (output/'COMPARISON.json').open('xb') as stream:stream.write(canonical(report))
    with (output/'comparison.html').open('x') as stream:stream.write(report_html(report))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','sealer','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('baseline-seal','candidate-seal'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--paired-cleanup',action='store_true')
    args=parser.parse_args();result=compare(args.baseline,args.baseline_seal,args.candidate,args.candidate_seal,
        args.sealer,args.output,paired_cleanup=args.paired_cleanup)
    print(json.dumps({key:result[key] for key in ('status','rows_compared','matched_rows','mismatched_rows','matched_work_all_rows')},sort_keys=True))
