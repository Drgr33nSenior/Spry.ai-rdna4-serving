"""Bounded host attribution for the frozen four-request sampling packet.

This recorder performs no tensor operations. Native-call elapsed time can include
device waits; it is not device execution time or CPU consumption. A packet must
authenticate the route, frozen bindings and returned records separately.
"""
import hashlib
import json
import re
import time


class SamplingHostTiming:
    """Retain the same 128 scheduled cycle rows as the planned device profiler."""

    CALL_LIMIT = 36864

    def __init__(self, bindings, *, clock=None):
        self.clock = time.monotonic_ns if clock is None else clock
        self.bindings = json.loads(json.dumps(bindings, allow_nan=False))
        self._need(type(self.bindings) is list and len(self.bindings) == 4, 'four frozen requests required')
        self.by_key, self.rids, self.attempts = {}, {}, {}
        self.records, self.pending, self.route_identity = [], None, None
        self.successes = dict.fromkeys(range(4), 0)
        for ordinal, row in enumerate(self.bindings):
            self._need(set(row) == {'case', 'request_sha256', 'input_ids_sha256', 'cache_salt',
                                   'input_tokens', 'reference_verifier_cycles', 'cycles'}, 'binding fields')
            self._need(type(row['case']) is str and 0 < len(row['case']) <= 256, 'case bound')
            for field in ('request_sha256', 'input_ids_sha256'):
                self._need(type(row[field]) is str and re.fullmatch('[0-9a-f]{64}', row[field]), 'binding SHA256')
            salt = row['cache_salt']
            self._need(salt is None or type(salt) is str and len(salt) <= 256, 'cache salt bound')
            self._need(type(row['input_tokens']) is int and 0 < row['input_tokens'] <= 16384, 'input bound')
            count = row['reference_verifier_cycles']
            self._need(type(count) is int and 41 <= count <= self.CALL_LIMIT, 'reference cycle bound')
            expected = [9 + j * (count - 10) // 31 for j in range(32)]
            self._need(type(row['cycles']) is list and all(type(n) is int for n in row['cycles']) and
                       row['cycles'] == expected, 'frozen stratified schedule')
            key = (row['input_ids_sha256'], salt)
            self._need(key not in self.by_key and row['case'] not in
                       [b['case'] for b in self.bindings[:ordinal]], 'duplicate request binding')
            self.by_key[key] = ordinal
            self.attempts[ordinal] = 0

    @staticmethod
    def _need(value, reason):
        if not value:
            raise ValueError('sampling host timing: ' + reason)

    def bind_route(self, identity):
        self._need(self.route_identity is None, 'route already bound')
        raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self._need(len(raw) <= 8192, 'route identity byte bound')
        self.route_identity = json.loads(raw)
        self.identity_sha256 = hashlib.sha256(raw).hexdigest()

    def before_native(self, entered_ns, request, prefix_tokens):
        self._need(self.route_identity is not None and self.pending is None, 'route or active call ownership')
        key = (request['input_ids_sha256'], request['cache_salt'])
        ordinal = self.by_key.get(key)
        if ordinal is None:
            return None  # Readiness traffic never becomes an external-request sample.
        row = self.bindings[ordinal]
        self._need(type(prefix_tokens) is int and row['input_tokens'] <= prefix_tokens <= 16384, 'host prefix bound')
        rid = request['rid']
        self._need(type(rid) is str and 0 < len(rid) <= 256, 'request ID bound')
        self._need(ordinal not in self.rids or self.rids[ordinal] == rid, 'frozen request replayed with another ID')
        self._need(all(other == ordinal or value != rid for other, value in self.rids.items()), 'request ID reused')
        self._need(sum(self.attempts.values()) < self.CALL_LIMIT, 'finite native-call observations')
        self.rids[ordinal] = rid
        self.attempts[ordinal] += 1
        cycle = self.attempts[ordinal]
        self.pending = dict(ordinal=ordinal, cycle=cycle, rid=rid, host_request_prefix_tokens=prefix_tokens,
                            entered_ns=entered_ns, selected=cycle in row['cycles'])
        self.pending['native_enter_ns'] = self.clock()
        return self.pending

    def finish(self, ticket, native_return_ns, *, native_exception=False):
        if ticket is None:
            return
        self._need(ticket is self.pending, 'stale or duplicate call ticket')
        finished_ns = self.clock()
        stamps = [ticket['entered_ns'], ticket['native_enter_ns'], native_return_ns, finished_ns]
        self._need(all(type(n) is int and n >= 0 for n in stamps) and stamps == sorted(stamps) and
                   finished_ns - stamps[0] <= 1800 * 10**9, 'monotonic interval bound')
        ordinal = ticket['ordinal']
        if not native_exception:
            self.successes[ordinal] += 1
        if ticket['selected']:
            self._need(len(self.records) < 128, 'retained row bound')
            binding = self.bindings[ordinal]
            self.records.append(dict(case=binding['case'], ordinal=ordinal, cycle=ticket['cycle'], rid=ticket['rid'],
                request_sha256=binding['request_sha256'], route_identity_sha256=self.identity_sha256,
                host_request_prefix_tokens=ticket['host_request_prefix_tokens'],
                observer_before_native_ns=stamps[1] - stamps[0], native_accept_host_ns=stamps[2] - stamps[1],
                observer_after_native_ns=stamps[3] - stamps[2], host_total_ns=stamps[3] - stamps[0],
                status='NATIVE_EXCEPTION' if native_exception else 'RETURNED',
                scope='HOST_ELAPSED_NOT_GPU_TIME', includes_timing_bookkeeping=True,
                excludes_record_assembly=True, qualification=False))
        self.pending = None

    def snapshot(self):
        self._need(self.pending is None, 'unfinished native call')
        coverage = []
        for ordinal, binding in enumerate(self.bindings):
            rows = [r for r in self.records if r['ordinal'] == ordinal]
            sampled = [r['cycle'] for r in rows]
            prefix = [r['host_request_prefix_tokens'] for r in rows]
            coverage.append(dict(case=binding['case'], attempted_cycles=self.attempts[ordinal],
                successful_calls=self.successes[ordinal], sampled_cycles=sampled,
                missing_scheduled_cycles=[n for n in binding['cycles'] if n not in sampled],
                sampled_prefix_range=None if not prefix else [min(prefix), max(prefix)],
                uncovered_tail_cycles=max(0, self.attempts[ordinal] - binding['cycles'][-1])))
        # Return a detached snapshot; the packet reader must still authenticate it.
        return json.loads(json.dumps(dict(schema=1, route_identity=self.route_identity,
            route_identity_sha256=getattr(self, 'identity_sha256', None), records=self.records,
            coverage=coverage, whole_request_bottleneck_established=False, device_time_measured=False,
            qualification=False), allow_nan=False))
