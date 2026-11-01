"""Exact composed request/group joins after the existing worker/source validators."""
import hashlib
import json
import math


def need(value,reason):
    if not value:raise ValueError(reason)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def planned_groups(rows,paired_groups):
    need(len(rows)==176 and [r['_kv_ordinal'] for r in rows]==list(range(176)),'frozen176case order')
    result=[dict(group,id=group['id'],ordinals=list(group['ordinals'])) for group in paired_groups]
    need(sorted(i for g in result for i in g['ordinals'])==list(range(176)),'group partition')
    return result


def validate(observations,rows,wire,groups,workers,*,paired_group_counts):
    """No host-duration or transport claim. Input worker/source checks are mandatory upstream."""
    need(len(workers)==2 and [w['rank'] for w in workers]==[0,1] and
         wire['wire_complete'] is True and len(wire['rows'])==176,'complete two-rank wire/worker input')
    need(set(observations)=={f'request-{i:02d}-{phase}.json' for i in range(176) for phase in ('before','after')},
         'exact composed observation membership')
    previous=None;result=[]
    for group in groups:
        indices=group['ordinals']
        before=[observations[f'request-{i:02d}-before.json'] for i in indices]
        after=[observations[f'request-{i:02d}-after.json'] for i in indices]
        for phase,records in (('before',before),('after',after)):
            for ordinal,record in zip(indices,records,strict=True):
                need(record['phase']==phase and record['ordinal']==ordinal and record['case']==rows[ordinal]['case'] and
                     record['arm']==wire['arm'] and record['body_sha256']==hashlib.sha256(canonical(rows[ordinal]['body'])).hexdigest(),
                     'request/body/arm observation join')
                need(record['group_id']==group['id'] and record['group_ordinals']==indices and
                     record['snapshot']==records[0]['snapshot'] and
                     record['observed_monotonic']==records[0]['observed_monotonic'],'shared exact group snapshot')
        left,right=before[0],after[0]
        need(type(left['observed_monotonic'])in(int,float) and type(right['observed_monotonic'])in(int,float) and
             math.isfinite(left['observed_monotonic']) and math.isfinite(right['observed_monotonic']) and
             left['observed_monotonic']<right['observed_monotonic'],'bounded observation chronology')
        if previous is not None:need(previous['observed_monotonic']<=left['observed_monotonic'],'group chronology')
        ranks=[]
        for rank,(a,b) in enumerate(zip(left['snapshot']['ranks'],right['snapshot']['ranks'],strict=True)):
            need(a['rank']==b['rank']==rank and a['pid']==b['pid']==workers[rank]['pid'] and
                 a['captures']==b['captures']==2 and
                 all(a[k]==b[k] for k in ('backend_identity','gdn_sha256')),'live worker/captured graph identity')
            if previous is not None:
                old=previous['snapshot']['ranks'][rank]
                need(all(old[k]==a[k] for k in ('replays','completed_commits','bs1_replays','bs2_replays','captures')),
                     'unattributed graph work between groups')
            delta={k:b[k]-a[k] for k in ('replays','completed_commits','bs1_replays','bs2_replays')}
            need(all(type(v)is int and v>=0 for v in delta.values()),'nonnegative graph deltas')
            if indices==[48]:
                need(wire['rows'][48]['status']=='CANCELLED_COMPLETE' and
                     delta['replays']==delta['completed_commits']==delta['bs1_replays']>0 and delta['bs2_replays']==0,
                     'cancel terminal and singleton committed graph join')
                joined=dict(delta,response_verifier_count=None,scope='cancellation terminal; no fabricated response verifier count')
            else:
                counts=[wire['rows'][i]['spec_tokens_details']['spec_verify_ct'] for i in indices]
                joined=paired_group_counts(delta,counts,group['concurrency'])
            ranks.append(dict(rank=rank,**joined))
        need(len(ranks)==2,'exact rank count')
        result.append(dict(group_id=group['id'],ordinals=indices,concurrency=group['concurrency'],ranks=ranks,
            before_monotonic=left['observed_monotonic'],after_monotonic=right['observed_monotonic']))
        previous=right
    return result


def route_groups(records,joined,*,target_fp8,validate_bound_routes,manifest,workers):
    if not target_fp8:
        need(records==[],'BF16 control cannot claim FP8 route evidence')
        import mtp_contract
        return dict(scope='explicit BF16 target cache profile' if mtp_contract.dense_mxfp4_enabled() else 'BF16 control', fp8_route_records=0)
    base=validate_bound_routes(records,manifest,{worker['rank']:worker for worker in workers})
    coverage={0:set(),1:set()};mapped=[]
    for row in records:
        rank=row['rank'];event=row['evidence']
        if row['stage']=='startup' and event['capture']:
            coverage[rank].add(event['q']['shape'][0])
        elif row['stage']=='external':
            matches=[g for g in joined if g['before_monotonic']<=row['observed_monotonic']<=g['after_monotonic']]
            need(len(matches)==1,'external route belongs to exactly one request group')
            mapped.append(dict(rank=rank,sequence=row['sequence'],group_id=matches[0]['group_id'],
                               ordinals=matches[0]['ordinals'],route=event['route']))
    need(all({8,16}<=values for values in coverage.values()),'both captured DFlash B1/B2 query-row geometries on both ranks')
    return dict(host_metadata=base,request_group_join_complete=True,native_graph_counter_join_complete=True,
        captured_query_rows={str(k):sorted(v) for k,v in coverage.items()},
        external_group_membership=mapped,scope='immutable source/captured metadata plus native replay counters; no device transport timing',
        serving_precision_accepted=False)


def startup_record(record,*,manifest,worker,validate_startup):
    for key in ('arm','session_sha256','spec_sha256','source_manifest_sha256'):
        need(record[key]==manifest[key],'startup source/session identity')
    need(record['schema']==1 and record['worker_identity_sha256']==worker['identity_sha256'] and
         all(record[k]==worker[k] for k in ('rank','pid','start_ticks','boot_id')),
         'startup worker lifecycle identity')
    need(record['helper_source']==manifest['sources']['j01_probe'] and record['configuration_sources']==
         {k:manifest['sources'][k] for k in ('native_kv_dtype','native_spec_fields')},
         'startup producer/configuration source identity')
    import mtp_contract
    profile = mtp_contract.cache_profile(manifest['arm'])
    fp8 = profile['target'] == 'fp8_e4m3'
    if mtp_contract.dense_mxfp4_enabled():
        need(record.get('cache_profile') == profile, 'explicit startup cache profile binding')
    else:
        need('cache_profile' not in record, 'unexpected legacy cache profile')
    need(record['status']==('FP8_TARGET_BF16_DRAFT_CACHE_CONFIRMED' if fp8 else
        'BF16_TARGET_AND_DRAFT_CACHE_CONFIRMED') and record['scope']=='startup metadata before external requests' and
        record['qualification'] is False and
        record['extra_tensor_operations']==record['extra_model_forwards']==record['cpu_tensor_copies']==0,
        'startup diagnostic-only metadata scope')
    return validate_startup(record['runners'],target_fp8=fp8,
        target_cache_dtype='bf16' if profile['target'] == 'bf16' else None)
