"""Future reader contract for global counters during the fixed six-request C2 phase."""
def _bb_validate_c2_counter_group(observations, wire):
    ordinals = tuple(range(47, 53))
    api = {row['ordinal']: row for row in wire['speculation_requests']}
    _need(set(api) == set(range(53)), 'all53 per-request native API counters')
    events = []
    for ordinal in ordinals:
        before, after = [_observation_json(observations, f'request-{ordinal:02d}-{phase}.json')
                         for phase in ('before', 'after')]
        _need(type(before.get('observed_monotonic')) in (int, float) and
              type(after.get('observed_monotonic')) in (int, float) and
              math.isfinite(before['observed_monotonic']) and math.isfinite(after['observed_monotonic']) and
              before['observed_monotonic'] < after['observed_monotonic'], 'ordered C2 request intervals')
        events.extend((before, after))
        _need(type(api[ordinal]['counts'].get('spec_verify_ct')) is int and
              api[ordinal]['counts']['spec_verify_ct'] > 0, 'actual per-request C2 verification')
    events.sort(key=lambda row: row['observed_monotonic'])
    _need(len({row['observed_monotonic'] for row in events}) == 12 and
          events[0]['phase'] == 'before' and events[-1]['phase'] == 'after', 'unique C2 chronology')
    live = set()
    for row in events:
        ordinal = row['ordinal']
        if row['phase'] == 'before':
            _need(ordinal not in live, 'duplicate C2 request admission')
            live.add(ordinal)
        else:
            _need(ordinal in live, 'C2 completion without admission')
            live.remove(ordinal)
        _need(len(live) <= 2, 'C2 concurrency exceeds two')
    _need(not live, 'all C2 requests completed')
    anchor = _observation_json(observations, 'request-46-after.json')
    _need(anchor['observed_monotonic'] < events[0]['observed_monotonic'], 'C1 finishes before C2')
    fields = ('replays', 'completed_commits', 'bs1_replays', 'bs2_replays')
    ranks = []
    for rank in (0, 1):
        start, end = events[0]['snapshot']['ranks'][rank], events[-1]['snapshot']['ranks'][rank]
        previous = anchor['snapshot']['ranks'][rank]
        _need(all(start[key] == previous[key] for key in fields) and
              start['replays'] == start['completed_commits'], 'quiescent common C1/C2 start')
        for event in events:
            current = event['snapshot']['ranks'][rank]
            _need(all(type(current.get(key)) is int and previous[key] <= current[key] <= end[key]
                      for key in fields) and current['captures'] == 2 and
                  current['replays'] == current['bs1_replays'] + current['bs2_replays'],
                  'monotonic bounded C2 global counters and batch decomposition')
            previous = current
        captures, replays, bs1, bs2 = struct.unpack('<QQQQ', observations[f'graph-counter-rank-{rank}.bin'])
        completed = struct.unpack('<Q', observations[f'replay-commit-counter-rank-{rank}.bin'])[0]
        _need(captures == 2 and [end[key] for key in fields] == [replays, completed, bs1, bs2] and
              replays == completed, 'quiescent C2 end equals final complete counters')
        delta = {key: end[key] - start[key] for key in fields}
        _need(delta['replays'] == delta['completed_commits'] > 0 and delta['bs2_replays'] > 0,
              'exact complete C2 group replay/commit total')
        delta['weighted_request_verifications'] = delta['bs1_replays'] + 2 * delta['bs2_replays']
        ranks.append(delta)
    _need(ranks[0] == ranks[1], 'both-rank C2 group totals match')
    actual_api = sum(api[ordinal]['counts']['spec_verify_ct'] for ordinal in ordinals)
    _need(ranks[0]['weighted_request_verifications'] == actual_api,
          'batch-weighted C2 graph work equals six actual per-request API verifications')
    return dict(mode='fixed-betterbench-c2-global-counter-group-v1', ordinals=list(ordinals),
                ranks=ranks, actual_API_verify_total=actual_api,
                per_request_global_deltas_are_not_individual_work=True)
