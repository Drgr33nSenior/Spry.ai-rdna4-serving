"""Validate phase joins after the campaign, worker and host summary authenticate.

This check does not authenticate raw logs, source trees, host summaries, HTTP,
device identity or resource release. The existing campaign reader must do that
first and supply its authenticated identity and matching host row. Additional
target parts require a separately authenticated capture/group witness.
"""
import math
import re


def validate_phase_pair(pending, collected, host_row, *, identity, bindings, target_parts=()):
    def need(value, reason):
        if not value:
            raise ValueError('phase readout: ' + reason)

    need(type(target_parts) is tuple and len(target_parts) <= 768 and
         all(type(p) is str and re.fullmatch(r'(mlp|gdn|attention|projection|comm)_[0-9]{3}', p)
             for p in target_parts) and len(set(target_parts)) == len(target_parts), 'target part schema')

    common = {'ordinal', 'case', 'cycle', 'rid', 'request_sha256', 'host_request_prefix_tokens',
              'selected', 'replays_before', 'commits_before', 'event_generation'}
    need(set(pending) == set(identity) | common, 'pending fields')
    for field, expected in identity.items():
        need(type(pending[field]) is type(expected) and pending[field] == expected, 'worker/source identity')
    need(type(pending['ordinal']) is int and 0 <= pending['ordinal'] < len(bindings), 'request ordinal')
    binding = bindings[pending['ordinal']]
    need(pending['case'] == binding['case'] and pending['request_sha256'] == binding['request_sha256'], 'frozen request')
    need(type(pending['cycle']) is int and pending['cycle'] in binding['cycles'] and pending['selected'] is True,
         'frozen cycle schedule')
    need(type(pending['rid']) is str and 0 < len(pending['rid']) <= 256, 'request ID')
    need(type(pending['host_request_prefix_tokens']) is int and
         binding['input_tokens'] <= pending['host_request_prefix_tokens'] <= 16384, 'prefix range')
    need(all(type(pending[k]) is int and 0 <= pending[k] < 36864 for k in ('replays_before', 'commits_before')) and
         type(pending['event_generation']) is int and 1 <= pending['event_generation'] <= 36864,
         'native counter bounds')
    status = collected.get('status')
    need(status in ('COMPLETE', 'NOT_READY', 'FINAL_UNCOLLECTED'), 'sample disposition')
    extra = {'status', 'graph_generations', 'extra_synchronizations', 'extra_model_forwards', 'qualification'}
    need(set(collected) == set(pending) | extra | ({'device_intervals_ms'} if status == 'COMPLETE' else set()),
         'collected fields')
    need(all(type(collected[k]) is type(v) and collected[k] == v for k, v in pending.items()), 'pending/current generation join')
    need(all(type(collected[k]) is int and collected[k] == 0 for k in ('extra_synchronizations', 'extra_model_forwards')) and
         collected['qualification'] is False, 'instrumentation contract')
    generations = collected['graph_generations']
    need(type(generations) is dict and set(generations) == {'draft', 'target'} and
         all(type(n) is int and 1 <= n <= 36864 for n in generations.values()) and
         generations['target'] == pending['replays_before'] + 1, 'graph generations')
    for field in ('ordinal', 'case', 'cycle', 'rid', 'request_sha256', 'host_request_prefix_tokens'):
        need(type(host_row.get(field)) is type(pending[field]) and host_row[field] == pending[field], 'host/request/cycle join')
    ns = ('observer_before_native_ns', 'native_accept_host_ns', 'observer_after_native_ns', 'host_total_ns')
    need(host_row.get('status') == 'RETURNED' and all(type(host_row.get(k)) is int and
         0 <= host_row[k] <= 1800 * 10**9 for k in ns) and sum(host_row[k] for k in ns[:3]) == host_row[ns[3]],
         'host elapsed partition')
    if status == 'COMPLETE':
        parts = {'cycle', 'accept', 'target_probs', 'mamba_commit', 'draft_append', 'next_input',
                 'draft.1.graph', 'draft.1.candidate_total', 'draft.1.projection', 'draft.1.local_topk',
                 'draft.1.selector_walk', 'target.1.graph', 'target.1.head', 'target.1.gather'}
        parts.update('target.1.' + name for name in target_parts)
        values = collected['device_intervals_ms']
        need(type(values) is dict and set(values) == parts and all(type(n) in (int, float) and
             math.isfinite(n) and 0 <= n <= 1800000 for n in values.values()), 'device interval schema/bounds')
        need(values['target.1.head'] + values['target.1.gather'] <= values['target.1.graph'] + .1 and
             values['draft.1.projection'] + values['draft.1.local_topk'] <= values['draft.1.candidate_total'] + .1 and
             values['draft.1.candidate_total'] + values['draft.1.selector_walk'] <= values['draft.1.graph'] + .1 and
             values['target_probs'] <= values['accept'] + .1, 'nested intervals')
        family_total = sum(values['target.1.' + name] for name in target_parts
                           if name.startswith(('mlp_', 'gdn_', 'attention_')))
        need(family_total + values['target.1.head'] + values['target.1.gather'] <=
             values['target.1.graph'] + .1, 'target family containment')
        for family in ('projection_', 'comm_'):
            need(sum(values['target.1.' + name] for name in target_parts if name.startswith(family)) <=
                 values['target.1.graph'] + .1, 'target nested group containment')
    return dict(status=status, case=pending['case'], cycle=pending['cycle'],
                host_and_device_joined=True, whole_request_bottleneck_established=False, qualification=False)
