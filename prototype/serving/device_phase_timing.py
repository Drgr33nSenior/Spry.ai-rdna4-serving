"""Event ownership for one bounded C1 attribution packet; no runtime imports."""
import json
import math


class DevicePhaseEvents:
    GRAPH_PARTS = {'draft': ('candidate_total', 'projection', 'local_topk', 'selector_walk'),
                   'target': ('head', 'gather')}
    EAGER_PARTS = ('cycle', 'accept', 'target_probs', 'mamba_commit', 'draft_append', 'next_input')

    def __init__(self, cuda, *, emit=None):
        self.cuda, self.emit = cuda, emit
        self.pairs, self.capture_counts, self.generations = {}, {}, {}
        self.active = self.pending = self.capture_key = None
        self.records, self.sequence = [], 0
        self.device = None
        self._need(not cuda.is_current_stream_capturing(), 'initialize outside capture')

    @staticmethod
    def _need(value, reason):
        if not value:
            raise ValueError('device phase timing: ' + reason)

    def _allocate(self, name):
        self._need(name not in self.pairs and not self.cuda.is_current_stream_capturing(), 'event allocation lifetime')
        self._need(self.device is not None and self.cuda.current_device() == self.device, 'event device changed')
        pair = tuple(self.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        for event in pair:
            event.record()
        self.pairs[name] = pair

    def prepare_graph(self, role, size):
        self._need(role in self.GRAPH_PARTS and type(size) is int and size in (1, 2), 'graph role/bucket')
        key = (role, size)
        self._need(key not in self.generations, 'unexpected recapture')
        if self.device is None:
            self.device = self.cuda.current_device()
        self.generations[key] = 0
        for part in ('graph', *self.GRAPH_PARTS[role]):
            name = (role, size, part)
            self._allocate(name)
            self.capture_counts[name] = 0

    def arm_capture(self, role, size):
        self._need((role, size) in self.generations and self.cuda.is_current_stream_capturing(), 'capture ownership')
        self.capture_key = (role, size)

    def capture_begin(self, part):
        if not self.cuda.is_current_stream_capturing():
            return None
        self._need(self.capture_key is not None, 'unowned capture')
        name = (*self.capture_key, part)
        self._need(part in self.GRAPH_PARTS[self.capture_key[0]] and self.capture_counts[name] == 0,
                   'unexpected or duplicate captured span')
        self.capture_counts[name] = 1
        self.pairs[name][0].record()
        return name

    def capture_end(self, ticket):
        if ticket is not None:
            self._need(ticket[:2] == self.capture_key and self.capture_counts[ticket] == 1, 'capture span ticket')
            self.pairs[ticket][1].record()
            self.capture_counts[ticket] = 2

    def finish_capture(self):
        self._need(set(self.generations) == {(r, n) for r in self.GRAPH_PARTS for n in (1, 2)}, 'both roles/buckets required')
        self._need(all(self.capture_counts[(r, n, p)] == 2 for r, n in self.generations
                       for p in self.GRAPH_PARTS[r]), 'captured span coverage')
        self.capture_key = None
        # The scheduler selects the rank's GPU after this recorder is constructed.
        # Bind eager events here, after native capture has selected that device.
        for name in self.EAGER_PARTS:
            self._allocate(name)

    def start_cycle(self, context):
        self._need(self.active is None, 'overlapping host cycles')
        self._need(self.device is not None and self.cuda.current_device() == self.device, 'event device changed')
        self.collect()  # Includes unsampled/readiness calls before their event reuse.
        self.sequence += 1
        self._need(self.sequence <= 36864, 'native-call observation bound')
        if context is not None:
            self.active = dict(context=json.loads(json.dumps(context, allow_nan=False)),
                               generation=self.sequence, started=set(), completed=set(), graphs={})
            self.begin('cycle')

    def begin(self, name):
        if self.active is None or not self.active['context']['selected']:
            return None
        self._need(name in self.pairs and name not in self.active['started'], 'span ownership or duplicate')
        self.active['started'].add(name)
        self.pairs[name][0].record()
        return (self.sequence, name)

    def end(self, ticket):
        if ticket is not None:
            generation, name = ticket
            self._need(self.active is not None and generation == self.active['generation'] and
                       name in self.active['started'] and name not in self.active['completed'], 'stale span ticket')
            self.pairs[name][1].record()
            self.active['completed'].add(name)

    def begin_graph(self, role, size):
        self._need(self.pending is None and (role, size) in self.generations, 'graph reuse before collection')
        key = (role, size)
        self.generations[key] += 1
        if self.active is not None:
            self._need(role not in self.active['graphs'] and size == 1, 'external graph geometry/count')
            self.active['graphs'][role] = (key, self.generations[key])
        return self.begin((role, size, 'graph'))

    def end_graph(self, ticket):
        self.end(ticket)  # Fresh outer completion event follows this actual replay.
        if ticket is not None:
            _, name = ticket
            role, size, _ = name
            for part in self.GRAPH_PARTS[role]:
                key = (role, size, part)
                self._need(self.capture_counts[key] == 2, 'graph inner span never captured')
                self.active['started'].add(key)
                self.active['completed'].add(key)

    def finish_cycle(self, joined):
        if self.active is None:
            return
        self._need(joined is True and set(self.active['graphs']) == {'draft', 'target'}, 'request/replay/commit join')
        self.end((self.sequence, 'cycle') if self.active['context']['selected'] else None)
        if self.active['context']['selected']:
            required = set(self.EAGER_PARTS)
            required.update((role, 1, part) for role in self.GRAPH_PARTS
                            for part in ('graph', *self.GRAPH_PARTS[role]))
            self._need(self.active['completed'] == required and self.active['started'] == required,
                       'incomplete phase coverage')
            self.pending = self.active
            if self.emit:
                self.emit('pending', dict(self.pending['context'], event_generation=self.sequence))
        self.active = None

    def collect(self, *, final=False):
        if self.pending is None:
            return
        self._need(self.cuda.current_device() == self.device, 'event device changed')
        row = self.pending
        for key, generation in row['graphs'].values():
            self._need(self.generations[key] == generation, 'overwritten event generation')
        names = sorted(row['completed'], key=str)
        outer = [self.pairs['cycle'][1], *(self.pairs[(role, 1, 'graph')][1] for role in ('draft', 'target'))]
        ready = all(event.query() for event in outer)
        if ready:
            ready = all(event.query() for name in names for event in self.pairs[name])
        record = dict(row['context'], event_generation=row['generation'],
                      status='COMPLETE' if ready else 'FINAL_UNCOLLECTED' if final else 'NOT_READY',
                      graph_generations={role: generation for role, (_, generation) in row['graphs'].items()},
                      extra_synchronizations=0, extra_model_forwards=0, qualification=False)
        if ready:
            values = {'.'.join(map(str, name)) if isinstance(name, tuple) else name:
                      float(self.pairs[name][0].elapsed_time(self.pairs[name][1])) for name in names}
            self._need(all(math.isfinite(v) and 0 <= v <= 1800000 for v in values.values()), 'finite interval bound')
            self._need(values['target.1.head'] + values['target.1.gather'] <= values['target.1.graph'] + 0.1 and
                       values['draft.1.projection'] + values['draft.1.local_topk'] <= values['draft.1.candidate_total'] + 0.1 and
                       values['draft.1.candidate_total'] + values['draft.1.selector_walk'] <= values['draft.1.graph'] + 0.1 and
                       values['target_probs'] <= values['accept'] + 0.1, 'nested interval containment')
            record['device_intervals_ms'] = values
        self._need(len(self.records) < 128, 'retained cycle row bound')
        self.records.append(record)
        self.pending = None
        if self.emit:
            self.emit('collected', record)

    def snapshot(self):
        self._need(self.active is None, 'unfinished host cycle')
        self.collect(final=True)
        return json.loads(json.dumps(dict(records=self.records, observed_calls=self.sequence,
                                         whole_request_bottleneck_established=False, qualification=False)))
