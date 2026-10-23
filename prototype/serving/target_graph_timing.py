"""Bounded target-family event hooks for the selected, source-pinned C1 model.

The assembled diagnostic supplies its existing authenticated source checkers.
This module imports no model runtime. Intervals can include communication/waits;
projection, communication and model-family intervals are nested, not additive.
"""


class TargetGraphHooks:
    COMM_SOURCE = dict(
        module='sglang.srt.distributed.parallel_state',
        path='/mnt/composition/image-source/python/sglang/srt/distributed/parallel_state.py',
        sha256='8c2cdbbe98ae59894f1ff682a038cb21581421d2f889aa36604cd51542395da8')
    SAMPLE_OFFSETS = (8, 24)
    PART_LIMIT = 768

    def __init__(self, timer, source_module, check_function):
        self.timer, self.source_module, self.check_function = timer, source_module, check_function
        self.discovery = self.capture = None
        self.comm_index = 0
        self.trace, self.expected_traces, self.active = [], {}, []
        self.parts, self.seen, self.objects = {}, {}, set()
        self.installed = False
        self.selection = {i: [row['cycles'][j] for j in self.SAMPLE_OFFSETS]
                          for i, row in enumerate(timer.bindings)}
        self.need(len(self.selection) == 4 and all(len(set(v)) == 2 for v in self.selection.values()),
                  'eight frozen selected cycles')
        events = timer.events
        events.GRAPH_PARTS = dict(events.GRAPH_PARTS)
        original_prepare, original_capture = timer.prepare_backend, timer.capture_wrapper
        original_finish, original_cycle = timer.finish_capture, events.start_cycle

        def prepare(role, backend, shape):
            result = original_prepare(role, backend, shape)
            if role == 'target' and not self.installed:
                self.install(backend._cuda_graph_runner.model_runner.model)
            return result

        def capture(backend, shape, phase, warmup_step=None):
            self.finish_transcript()
            target = backend._cuda_graph_runner.model_runner.is_draft_worker is False
            self.discovery = shape.size if target and phase == 'warmup' and warmup_step == 0 else None
            self.capture = shape.size if target and phase == 'capture' else None
            self.comm_index = 0
            return original_capture(backend, shape, phase, warmup_step)

        def start_cycle(context):
            if context is not None:
                context = dict(context, selected=context['cycle'] in self.selection[context['ordinal']])
            return original_cycle(context)

        def finish():
            self.finish_transcript()
            self.check_coverage()
            rows = [r for r in timer.saved if r[4] and id(r[0]) in self.objects]
            timer.restore_rows(rows)
            timer.saved = [r for r in timer.saved if not (r[4] and id(r[0]) in self.objects)]
            self.discovery = self.capture = None
            result = original_finish()
            timer.emit('target_groups', dict(parts=list(self.parts.values()), selected_cycles=self.selection,
                capture_hooks_restored=True, ordered_calls_verified=True, instrumented_nodes_execute_on_all_replays=True,
                extra_model_forwards=0, extra_synchronizations=0, qualification=False))
            return result

        timer.prepare_backend = prepare
        timer.capture_wrapper = capture
        timer.observer.capture_forward = capture
        timer.finish_capture = finish
        events.start_cycle = start_cycle

    @staticmethod
    def need(value, reason):
        if not value:
            raise ValueError('target graph timing: ' + reason)

    def wrap(self, obj, method_name, source, family, index, object_path, expected_qualname):
        self.need(method_name not in obj.__dict__, 'foreign instance hook')
        original = getattr(obj, method_name)
        function = getattr(original, '__func__', None)
        self.need(function is not None, 'native bound method required')
        self.check_function(function, source, expected_qualname)
        self.need(type(obj).__module__ == source['module'], 'native object module')
        self.objects.add(id(obj))

        def wrapped(*args, **kwargs):
            ticket = None
            target_capture = (self.timer.cuda.is_current_stream_capturing() and
                              self.timer.events.capture_key[0] == 'target')
            if self.discovery is not None or target_capture:
                ordinal = index
                if family == 'comm':
                    ordinal = self.comm_index
                    self.comm_index += 1
                part = family + '_' + str(ordinal).zfill(3)
                descriptor = dict(part=part, family=family, object_path=object_path,
                                  method=method_name, qualname=function.__qualname__, source=source)
                entry = (part, tuple(self.active))
                expected = self.expected_traces.get(self.capture)
                if self.discovery is not None:
                    expected = self.expected_traces.get(self.discovery,
                        next(iter(self.expected_traces.values()), None))
                if expected is not None:
                    self.need(len(self.trace) < len(expected) and entry == expected[len(self.trace)],
                              'ordered target call transcript')
                self.trace.append(entry)
                if self.discovery is not None:
                    self.discover(part, descriptor, self.discovery)
                elif target_capture:
                    self.need(self.capture in (1, 2) and self.parts.get(part) == descriptor,
                              'capture call identity/order')
                    ticket = self.timer.events.capture_begin(part)
                self.active.append(part)
            tracked = self.discovery is not None or target_capture
            try:
                result = original(*args, **kwargs)
                self.timer.events.capture_end(ticket)
                return result
            finally:
                if tracked:
                    self.active.pop()

        self.timer.bind(obj, method_name, wrapped, instance=True)

    def finish_transcript(self):
        size = self.discovery if self.discovery is not None else self.capture
        if size is not None:
            self.need(not self.active and self.trace, 'complete target call transcript')
            if self.discovery is not None and size not in self.expected_traces:
                other = next(iter(self.expected_traces.values()), self.trace)
                self.need(self.trace == other, 'ordered bucket discovery transcript')
                self.expected_traces[size] = list(self.trace)
            else:
                self.need(self.trace == self.expected_traces[size], 'complete ordered capture transcript')
        self.trace = []

    def discover(self, part, descriptor, size):
        events = self.timer.events
        self.need(size in (1, 2) and not self.timer.cuda.is_current_stream_capturing(),
                  'discovery outside capture')
        self.need((size, part) not in self.seen, 'repeated module invocation')
        self.seen[(size, part)] = True
        if part not in self.parts:
            self.need(len(self.parts) < self.PART_LIMIT, 'finite target part count')
            self.parts[part] = descriptor
            events.GRAPH_PARTS['target'] = (*events.GRAPH_PARTS['target'], part)
        self.need(self.parts[part] == descriptor, 'bucket call identity/order')
        key = ('target', size, part)
        if key not in events.pairs:
            events._allocate(key)
            events.capture_counts[key] = 0

    def install(self, model):
        self.need(not self.installed, 'duplicate model installation')
        sources = self.timer.manifest['sources']
        model_module = self.source_module(sources['head_model'])
        mlp_module = self.source_module(sources['gdn_mlp'])
        linear_module = self.source_module(sources['head_linear'])
        comm_module = self.source_module(self.COMM_SOURCE)
        classes = dict(mlp=mlp_module.Qwen2MoeMLP, gdn=model_module.Qwen3_5GatedDeltaNet,
                       attention=model_module.Qwen3_5AttentionDecoderLayer)
        counts = dict.fromkeys(classes, 0)
        projections = 0
        objects = list(model.named_modules())
        self.need(len(objects) <= 4096 and len({name for name, _ in objects}) == len(objects),
                  'bounded unique target modules')
        for name, obj in objects:
            for family, cls in classes.items():
                if type(obj) is cls:
                    source = sources['gdn_mlp' if family == 'mlp' else 'head_model']
                    method = 'self_attention' if family == 'attention' else 'forward'
                    qualname = {'mlp': 'Qwen2MoeMLP.forward',
                                'gdn': 'Qwen3_5GatedDeltaNet.forward',
                                'attention': 'Qwen3_5AttentionDecoderLayer.self_attention'}[family]
                    self.wrap(obj, method, source, family, counts[family], name, qualname)
                    counts[family] += 1
            if isinstance(obj, linear_module.LinearBase) and type(obj).__module__ == sources['head_linear']['module']:
                self.need(projections < 512, 'projection object bound')
                declaring = {'ReplicatedLinear': 'ReplicatedLinear',
                             'ColumnParallelLinear': 'ColumnParallelLinear',
                             'MergedColumnParallelLinear': 'ColumnParallelLinear',
                             'QKVParallelLinear': 'ColumnParallelLinear',
                             'RowParallelLinear': 'RowParallelLinear',
                             'MergedColumnParallelRepeatedLinear': 'MergedColumnParallelRepeatedLinear'}
                cls_name = type(obj).__name__
                self.need(cls_name in declaring and type(obj) is getattr(linear_module, cls_name),
                          'known native projection class')
                self.wrap(obj, 'forward', sources['head_linear'], 'projection', projections, name,
                          declaring[cls_name] + '.forward')
                projections += 1
        self.need(counts == dict(mlp=64, gdn=48, attention=16), 'selected dense target layer counts')
        parallel = model_module.get_parallel()
        groups = []
        for name in ('tp_group', 'attn_tp_group'):
            group = getattr(parallel, name)
            self.need(type(group) is comm_module.GroupCoordinator and group.world_size == 2,
                      'selected TP2 group')
            if not any(group is old for old in groups):
                self.wrap(group, 'all_reduce', self.COMM_SOURCE, 'comm', 0, name, 'GroupCoordinator.all_reduce')
                groups.append(group)
        self.installed = True

    def check_coverage(self):
        self.need(self.installed, 'target model not installed')
        counts = {family: sum(row['family'] == family for row in self.parts.values())
                  for family in ('mlp', 'gdn', 'attention', 'projection', 'comm')}
        self.need(counts['mlp'] == 64 and counts['gdn'] == 48 and counts['attention'] == 16 and
                  128 <= counts['projection'] <= 512 and 64 <= counts['comm'] <= 256,
                  'native target family coverage')
        self.need(set(self.expected_traces) == {1, 2}, 'both ordered bucket transcripts')
        for size in (1, 2):
            self.need({part for bucket, part in self.seen if bucket == size} == set(self.parts),
                      'both graph bucket discovery coverage')
