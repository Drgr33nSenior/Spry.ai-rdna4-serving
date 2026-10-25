"""Authenticate the native HTTP export before the finite owner accepts it.

This is deliberately an archive reader.  It does not import a serving runtime,
start a process, or assert that an HTTP screen qualifies the model.
"""
import base64
from contextlib import contextmanager
import hashlib
import hashlib as _bbhash,importlib.util as _bbimport
from pathlib import Path as _BBPath
_bbpath=_BBPath(__file__).with_name('j01_probe.py')
assert not _bbpath.is_symlink() and _bbpath.stat().st_nlink==1 and _bbpath.stat().st_size<=256<<10
assert _bbhash.sha256(_bbpath.read_bytes()).hexdigest()=='3b83ccb75ebf834a97022f28188a0398ca8267a711fbfb5545b2ef3bbeb15899'
_bbspec=_bbimport.spec_from_file_location('_source_bound_betterbench_helper',_bbpath)
_bbmodule=_bbimport.module_from_spec(_bbspec);_bbspec.loader.exec_module(_bbmodule)
betterbench_owner=_bbmodule.betterbench_owner
del _bbhash,_bbimport,_BBPath,_bbpath,_bbspec,_bbmodule
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
MODE = 'native-http-fp8-sprint01-v1'
MAX_BUNDLE_BYTES = 36 << 20
MAX_MEMBER_BYTES = 4 << 20
MAX_MEMBERS = 256
_MEMBER_FIELDS = {'base64', 'bytes', 'sha256'}
_GDN_SOURCE_ROLES = {'gdn_backend', 'gdn_triton', 'gdn_packed', 'gdn_verify', 'gdn_prefill', 'hybrid_backend', 'memory_pool'}

def _need(value, message):
    if not value:
        raise ValueError('HTTP integration: ' + message)

def _digest(value):
    return type(value) is str and len(value) == 64 and all((char in '0123456789abcdef' for char in value))

def _json(raw):

    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def _read(path, maximum=MAX_BUNDLE_BYTES):
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path, 'canonical evidence path')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and (0 < before.st_size <= maximum), 'bounded regular evidence member')
        parts, remaining = ([], before.st_size)
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _need(block, 'short evidence read')
            parts.append(block)
            remaining -= len(block)
        raw = b''.join(parts)
        after, named = (os.fstat(descriptor), path.lstat())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')
        _need(all((getattr(before, field) == getattr(after, field) == getattr(named, field) for field in fields)), 'evidence changed during read')
        return raw
    finally:
        os.close(descriptor)

def _relative(name):
    _need(type(name) is str and name and ('\x00' not in name), 'relative member name required')
    path = PurePosixPath(name)
    _need(not path.is_absolute() and str(path) == name and all((part not in ('', '.', '..', '.idea', 'keeper-env') for part in path.parts)), 'unsafe client member name')
    return path

def _decode_member(row):
    _need(type(row) is dict and set(row) == _MEMBER_FIELDS and (type(row['bytes']) is int) and (0 <= row['bytes'] <= MAX_MEMBER_BYTES) and _digest(row['sha256']) and (type(row['base64']) is str), 'client member schema')
    try:
        raw = base64.b64decode(row['base64'], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('HTTP integration: invalid client member base64') from error
    _need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'client member identity')
    return raw

def _write_exclusive(path, raw):
    path.parent.mkdir(mode=448, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            _need(count > 0, 'short client tree write')
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _private_reader(*, include_parity=False):
    """Load source-bound stdlib helpers without an ambient sys.path."""
    root = Path(__file__).resolve().parent
    names = ('mtp_contract', 'serving_contract', 'serving_client', 'serving_reader')
    if include_parity:
        names += ('serving_parity',)
    originals = {name: sys.modules.get(name) for name in names}
    try:
        loaded = {}
        for name in names:
            path = root / (name + '.py')
            _need(path.is_file() and (not path.is_symlink()), 'private helper source missing')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        result = (loaded['serving_contract'], loaded['serving_reader'])
        return result + (loaded['serving_parity'],) if include_parity else result
    finally:
        for name, previous in originals.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

def _mtp_contract():
    """Load the adjacent pure contract rather than trusting the ambient path."""
    path = Path(__file__).resolve().with_name('mtp_contract.py')
    _need(path.is_file() and (not path.is_symlink()), 'MTP contract source missing')
    spec = importlib.util.spec_from_file_location('spry_mtp_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _unpack_client(raw, destination):
    value = _json(raw)
    _need(type(value) is dict and set(value) == {'schema', 'mode', 'files'} and (value['schema'] == 1) and (value['mode'] == MODE) and (type(value['files']) is dict) and (1 <= len(value['files']) <= MAX_MEMBERS), 'HTTP client bundle schema')
    destination = Path(destination)
    _need(destination.is_absolute() and (not destination.exists()), 'fresh client extraction root required')
    destination.mkdir(mode=448)
    names = set()
    for name, row in sorted(value['files'].items()):
        path = _relative(name)
        _need(name not in names, 'duplicate client member')
        names.add(name)
        _write_exclusive(destination / path, _decode_member(row))
    return (destination, hashlib.sha256(raw).hexdigest())

def _native(raw, expected):
    value = _json(raw)
    _need(type(value) is dict and set(value) == {'schema', 'mode', 'gate', 'workers', 'observations'} and (value['schema'] == 1) and (value['mode'] == MODE) and (type(value['workers']) is list) and (len(value['workers']) == 2) and (type(value['observations']) is dict) and (1 <= len(value['observations']) <= MAX_MEMBERS), 'native HTTP bundle schema')
    gate_raw = _decode_member(value['gate'])
    contract, reader = _private_reader()
    gate = _json(gate_raw)
    workers = reader.serving_client.validate_gate(gate, expected)
    _need([row.get('rank') for row in value['workers']] == [0, 1], 'native worker order')
    for full, compact in zip(value['workers'], workers):
        _need(type(full) is dict and full.get('schema') == 1 and (full.get('mode') == MODE) and (full.get('arm') == expected['arm']) and (full.get('rank') == compact['rank']) and (full.get('pid') == compact['pid']) and (full.get('start_ticks') == compact['start_ticks']) and (full.get('boot_id') == compact['boot_id']) and (full.get('session_sha256') == expected['session_sha256']) and (full.get('spec_sha256') == expected['spec_sha256']) and (full.get('source_manifest_sha256') == expected['source_manifest_sha256']) and (type(full.get('sources')) is dict) and _digest(full.get('identity_sha256')) and (hashlib.sha256(json.dumps(full['sources'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == expected['source_manifest_sha256']) and (hashlib.sha256(contract.canonical({key: item for key, item in full.items() if key != 'identity_sha256'})).hexdigest() == compact['identity_sha256']), 'full native worker identity differs')
    observations = {}
    for name, row in sorted(value['observations'].items()):
        _relative(name)
        observations[name] = _decode_member(row)
    return (gate_raw, workers, value['workers'], observations, hashlib.sha256(raw).hexdigest(), reader)

def _observation_json(observations, name):
    _need(name in observations, 'missing native observation: ' + name)
    return _json(observations[name])

def _gdn_observer(sources):
    """Authenticate the adjacent pure validator against the native source map."""
    _need(type(sources) is dict, 'GDN source map differs')
    if not _GDN_SOURCE_ROLES - {'gdn_backend', 'memory_pool'} & set(sources):
        return None
    _need(_GDN_SOURCE_ROLES <= set(sources), 'complete GDN source roles required')
    source = sources.get('graph_observer')
    path = Path(__file__).resolve().with_name('serving_graph_observer.py')
    _need(type(source) is dict and set(source) == {'module', 'path', 'sha256'} and (source['module'] == 'serving_graph_observer') and (type(source['path']) is str) and PurePosixPath(source['path']).is_absolute() and (PurePosixPath(source['path']).name == path.name) and _digest(source['sha256']), 'GDN graph observer source binding differs')
    raw = _read(path, MAX_MEMBER_BYTES)
    _need(hashlib.sha256(raw).hexdigest() == source['sha256'], 'GDN graph observer source changed')
    spec = importlib.util.spec_from_file_location('spry_gdn_graph_observer', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    _need(callable(getattr(module, 'validate_gdn_record', None)) and callable(getattr(module, 'canonical', None)), 'GDN graph observer validator missing')
    return module

@contextmanager
def _gdn_contract_scope(sources):
    """Bind only the authenticated adjacent contract for the observer's lazy import."""
    path = Path(__file__).resolve().with_name('mtp_contract.py')
    source = sources.get('mtp_contract')
    _need(type(source) is dict and set(source) == {'path', 'sha256'} and (type(source['path']) is str) and PurePosixPath(source['path']).is_absolute() and (str(PurePosixPath(source['path'])) == source['path']) and ('..' not in PurePosixPath(source['path']).parts) and (PurePosixPath(source['path']).name == path.name) and _digest(source['sha256']), 'GDN contract source binding differs')
    raw = _read(path, MAX_MEMBER_BYTES)
    _need(hashlib.sha256(raw).hexdigest() == source['sha256'], 'GDN contract source changed')
    spec = importlib.util.spec_from_file_location('mtp_contract', path)
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get('mtp_contract', missing)
    sys.modules['mtp_contract'] = module
    try:
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        yield module
    finally:
        if previous is missing:
            sys.modules.pop('mtp_contract', None)
        else:
            sys.modules['mtp_contract'] = previous

def _gdn_capture(capture, *, observer, sources, speculation, arm, rank, pid):
    with _gdn_contract_scope(sources):
        observer.validate_gdn_record(capture.get('gdn'), arm, rank, pid, sources=sources)
    _need(capture['gdn']['speculation'] == {'algorithm': speculation['speculative_algorithm'], 'draft_tokens': speculation['speculative_num_draft_tokens']}, 'GDN live speculation differs from binding')
    identity, backend = (capture.get('backend_identity'), sources.get('backend'))
    _need(type(identity) is dict and type(backend) is dict and (type(backend.get('module')) is str) and (type(backend.get('path')) is str) and _digest(backend.get('sha256')) and all((identity.get(key) == backend.get(key) for key in ('module', 'path', 'sha256'))) and (identity.get('class_name') == 'FullCudaGraphBackend') and (identity.get('model_runner_role') == 'target'), 'GDN graph backend source identity differs')
    fold = _mtp_contract().replay_serving_enabled(arm)
    expected_commit = dict(sources['replay_fold' if fold else 'replay_scatter'], name='commit_gdn_replayssm_fold_after_verify' if fold else 'scatter_mamba_states_after_mtp_verify')
    _need(capture.get('commit_source') == expected_commit, 'final observed completed native commit source differs')
    return hashlib.sha256(observer.canonical(capture['gdn'])).hexdigest()

def _probe_helper(sources):
    source = sources.get('j01_probe')
    path = Path(__file__).resolve().with_name('j01_probe.py')
    _need(type(source) is dict and set(source) == {'module', 'path', 'sha256'} and (source['module'] == 'j01_probe') and (source['path'] == '/mnt/review/j01_probe.py') and (hashlib.sha256(_read(path, MAX_MEMBER_BYTES)).hexdigest() == source['sha256']), 'J01 helper source binding')
    spec = importlib.util.spec_from_file_location('j01_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _final_norm_helper(sources, role='final_norm_probe'):
    _need(role in ('final_norm_probe', 'midpoint_capture'), 'unknown boundary helper')
    source = sources.get(role)
    path = Path(__file__).resolve().with_name(role + '.py')
    _need(type(source) is dict and set(source) == {'module', 'path', 'sha256'} and (source['module'] == role) and (source['path'] == '/mnt/review/' + role + '.py') and (hashlib.sha256(_read(path, MAX_MEMBER_BYTES)).hexdigest() == source['sha256']), 'final-norm helper source binding')
    spec = importlib.util.spec_from_file_location(role, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('j01_probe')
    sys.modules['j01_probe'] = _probe_helper(sources)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop('j01_probe', None)
        else:
            sys.modules['j01_probe'] = previous
    return module

def _validate_final_norm(observations, binding, workers, states):
    if not _mtp_contract().final_norm_capture_enabled():
        _need(not any((name.startswith('final-norm-') for name in observations)), 'unbound final-norm artifact')
        return None
    _need({name for name in observations if name.startswith('final-norm-')} == {f'final-norm-rank-{rank}.json' for rank in (0, 1)}, 'final-norm rank membership differs')
    helper = _final_norm_helper(binding['sources'])
    contexts = []
    for rank, (worker, state) in enumerate(zip(workers, states)):
        head = _observation_json(observations, f'head-input-rank-{rank}.json')
        record = _observation_json(observations, f'final-norm-rank-{rank}.json')
        helper.validate_record(record, binding=binding, worker=worker, state=state, head_record=head)
        contexts.append(record['context_sha256'])
    _need(len(contexts) == 2 and contexts[0] == contexts[1], 'final-norm rank context differs')
    return dict(ranks=[0, 1], captured=True, context_sha256=contexts[0])

def _validate_midpoint(observations, binding, workers, states):
    names = {name for name in observations if name.startswith('midpoint-')}
    if not _mtp_contract().midpoint_capture_enabled():
        _need(not names, 'unbound midpoint artifacts')
        return None
    _need(names == {f'midpoint-rank-{rank}.json' for rank in (0, 1)}, 'midpoint rank membership differs')
    module = _final_norm_helper(binding['sources'], 'midpoint_capture')
    contexts, boundaries = ([], [])
    with _gdn_contract_scope(binding['sources']):
        for rank, (worker, state) in enumerate(zip(workers, states)):
            head = _observation_json(observations, f'head-input-rank-{rank}.json')
            record = _observation_json(observations, f'midpoint-rank-{rank}.json')
            if _mtp_contract().settings()['midpoint_capture'] == 'j01-block3-prepared-attention-v1':
                name = f'prepared-pending-rank-{rank}.json'
                packet = _observation_json(observations, name)
                raw = observations[name]
                reference = record['pending_reference']
                _need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'], 'prepared pending packet raw reference differs')
                module.validate_record(record, binding=binding, worker=worker, state=state, head_record=head, pending_packet=packet)
            elif _mtp_contract().norm_capture_enabled():
                name = f'norm-operands-rank-{rank}.bin'
                module.validate_record(record, binding=binding, worker=worker, state=state, head_record=head, artifacts={name: observations[name]})
            elif _mtp_contract().gdn_capture_enabled():
                names = [f'gdn-operands-rank-{rank}.bin']
                if binding['arm'] == 'mtp' and rank == 0:
                    names.append('gdn-state-initial-rank-0.bin')
                module.validate_record(record, binding=binding, worker=worker, state=state, head_record=head, artifacts={name: observations[name] for name in names})
            else:
                module.validate_record(record, binding=binding, worker=worker, state=state, head_record=head)
            contexts.append(record['context_sha256'])
            boundaries.append(record['boundary'])
    _need(len(contexts) == 2 and contexts[0] == contexts[1] and (boundaries[0] == boundaries[1]), 'midpoint rank context or boundary differs')
    return dict(ranks=[0, 1], captured=True, context_sha256=contexts[0], boundary=boundaries[0])

def validate_probe_wire(observations, workers, wire):
    contract = _mtp_contract()
    if not getattr(contract, 'j01_probe_enabled', lambda: False)():
        return None
    binding = _observation_json(observations, 'SERVING-BINDING.json')
    helper = _probe_helper(binding['sources'])
    states = [_observation_json(observations, f'graph-captures-rank-{rank}.json').get('token_probe') for rank in (0, 1)]
    result = helper.validate_wire(states, binding=binding, workers=workers, contract=contract, wire=wire)
    if contract.head_input_capture_enabled():
        _need(type(workers) is list and len(workers) == 2 and all((type(worker) is dict and worker.get('rank') == rank for rank, worker in enumerate(workers))) and ({name for name in observations if name.startswith('head-input-')} == {f'head-input-rank-{rank}.json' for rank in (0, 1)}), 'head-input rank membership differs')
        _need(result['complete'] is True and len(result['ranks']) == 2 and all((all(row['wire_join'].values()) for row in result['ranks'])), 'head-input probe capture/wire join incomplete')
        for rank, (state, worker) in enumerate(zip(states, workers)):
            helper.validate_head_input(_observation_json(observations, f'head-input-rank-{rank}.json'), binding=binding, worker=worker, state=state, complete=True)
        records = [state['record'] for state in states]
        _need(all((row['rid'] == records[0]['rid'] and row['context_sha256'] == records[0]['context_sha256'] for row in records)), 'head-input rank request/context differs')
        result['head_inputs'] = dict(ranks=[0, 1], context_sha256=records[0]['context_sha256'], captured=True)
        norm = _validate_final_norm(observations, binding, workers, states)
        if norm is not None:
            result['final_norm'] = norm
        midpoint = _validate_midpoint(observations, binding, workers, states)
        if midpoint is not None:
            result['midpoint'] = midpoint
    return result

def validate_prefill_wire(observations, workers, wire):
    """Require the explicit S36 observer to join the actual authenticated wire."""
    contract = _mtp_contract()
    if contract.settings().get('quality_screen') != 'sprint36-chunk-prefill-v1':
        return None
    _need(not contract.j01_probe_enabled(), 'prefill/J01 mode collision')
    _need(type(workers) is list and len(workers) == 2 and
          [worker.get('rank') for worker in workers] == [0, 1], 'prefill worker rank coverage')
    binding = _observation_json(observations, 'SERVING-BINDING.json')
    helper = _probe_helper(binding['sources'])
    _need(callable(getattr(helper, 'validate_prefill_observation', None)), 'prefill validator missing')
    for rank, worker in enumerate(workers):
        capture = _observation_json(observations, f'graph-captures-rank-{rank}.json')
        helper.validate_prefill_observation(capture.get('token_probe'), binding=binding,
            worker=worker, contract=contract, responses=wire['rows'])

def _validate_precision_observations(observations, binding, workers, observer):
    """Validate both loaded-weight witnesses under the authenticated contract."""
    _need(observer is not None and callable(getattr(observer, 'validate_precision_weights', None)), 'precision weight source-bound validator missing')
    helper = _probe_helper(binding['sources'])
    with _gdn_contract_scope(binding['sources']):
        for rank, worker in enumerate(workers):
            precision = _observation_json(observations, f'PRECISION-WEIGHTS-rank-{rank}.json')
            observer.validate_precision_weights(precision, binding=binding, worker=worker)
            helper.validate_head_startup(precision['head_startup'], scheduler_manifest_sources=binding['sources'])

def _validate_native_observations(observations, *, expected, gate, full_workers):
    """Require the finite post-warmup, HIP, and graph evidence from the native path."""
    required = {'NATIVE-READINESS-CALLBACK.json', 'SERVING-BINDING.json', 'WORKER-MANIFEST.json', 'MTP-PREFLIGHT.json', 'NATIVE-CHAT-PARITY.json', 'NATIVE-CHAT-PARITY.ready'}
    required.update({f'worker-rank-{rank}-preweight.json' for rank in (0, 1)})
    required.update({f'MTP-PREFLIGHT-rank-{rank}.json' for rank in (0, 1)})
    required.update({f'graph-captures-rank-{rank}.json' for rank in (0, 1)})
    required.update({f'graph-counter-rank-{rank}.bin' for rank in (0, 1)})
    required.update({f'replay-commit-counter-rank-{rank}.bin' for rank in (0, 1)})
    required.update({f'request-{ordinal:02d}-{phase}.json' for ordinal in range(_mtp_contract().phase_config()[2]) for phase in ('before', 'after')})
    phase = _mtp_contract().settings()['phase']
    screen_contract, _ = _private_reader()
    hip_components_required = screen_contract.hip_sampling_arm(phase, expected['arm'])
    if hip_components_required:
        required.update({f'HIP-SAMPLING-COMPONENT-rank-{rank}.json' for rank in (0, 1)})
    allocator_required = _mtp_contract().settings().get('allocator_snapshot') is True
    if allocator_required:
        required.update({f'GPU-ALLOCATOR-rank-{rank}.json' for rank in (0, 1)})
    if _mtp_contract().precision_reference_enabled():
        required.update({f'PRECISION-WEIGHTS-rank-{rank}.json' for rank in (0, 1)})
    head_input_required = _mtp_contract().head_input_capture_enabled()
    if head_input_required:
        required.update({f'head-input-rank-{rank}.json' for rank in (0, 1)})
    if _mtp_contract().final_norm_capture_enabled():
        required.update({f'final-norm-rank-{rank}.json' for rank in (0, 1)})
    if _mtp_contract().midpoint_capture_enabled():
        required.update({f'midpoint-rank-{rank}.json' for rank in (0, 1)})
        if _mtp_contract().settings()['midpoint_capture'] == 'j01-block3-prepared-attention-v1':
            required.update({f'prepared-pending-rank-{rank}.json' for rank in (0, 1)})
        if _mtp_contract().norm_capture_enabled():
            required.update({f'norm-operands-rank-{rank}.bin' for rank in (0, 1)})
        if _mtp_contract().gdn_capture_enabled():
            required.update({f'gdn-operands-rank-{rank}.bin' for rank in (0, 1)})
            if expected['arm'] == 'mtp':
                required.add('gdn-state-initial-rank-0.bin')
    if _mtp_contract().raw_attention_capture_enabled():
        required.update({f'attention-launch-rank-{rank}.json' for rank in (0, 1)})
    if _mtp_contract().full_enabled_benchmark():
        required.update({f'FULL-ENABLED-rank-{rank}.json' for rank in (0, 1)})
    _need(set(observations) == required, 'native observation membership differs')
    _need(_observation_json(observations, 'NATIVE-READINESS-CALLBACK.json') == gate, 'native callback readiness differs')
    binding = _observation_json(observations, 'SERVING-BINDING.json')
    worker_manifest = _observation_json(observations, 'WORKER-MANIFEST.json')
    _need(type(binding) is dict and type(worker_manifest) is dict and (binding.get('mode') == worker_manifest.get('mode') == MODE) and (binding.get('arm') == worker_manifest.get('arm') == expected['arm']) and (binding.get('session_sha256') == worker_manifest.get('session_sha256') == expected['session_sha256']) and (binding.get('spec_sha256') == worker_manifest.get('spec_sha256') == expected['spec_sha256']) and (binding.get('source_manifest_sha256') == worker_manifest.get('source_manifest_sha256') == expected['source_manifest_sha256']) and (binding.get('sources') == worker_manifest.get('sources')) and (hashlib.sha256(json.dumps(binding.get('sources'), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == expected['source_manifest_sha256']), 'native source/binding observation differs')
    gdn_observer = _gdn_observer(binding['sources'])
    if allocator_required:
        _need(gdn_observer is not None and callable(getattr(gdn_observer, 'validate_allocator_record', None)), 'allocator source-bound validator missing')
        for rank, worker in enumerate(full_workers):
            gdn_observer.validate_allocator_record(_observation_json(observations, f'GPU-ALLOCATOR-rank-{rank}.json'), binding=binding, worker=worker, preweight=_observation_json(observations, f'worker-rank-{rank}-preweight.json'), expected_devices=worker_manifest['expected_devices'])
    if _mtp_contract().precision_reference_enabled():
        _validate_precision_observations(observations, binding, full_workers, gdn_observer)
    if _mtp_contract().full_enabled_benchmark():
        for rank, worker in enumerate(full_workers):
            validate_full_enabled_record(_observation_json(observations, f'FULL-ENABLED-rank-{rank}.json'), binding, worker)
    gdn_captures = {}
    _mtp_contract().validate_recurrent_capacity_cli(binding.get('canonical_server_cli'))
    contract = binding.get('resolved_contract')
    _need(contract.get('enable_linear_replayssm') is False and contract.get('enable_linear_replayssm_spec') is _mtp_contract().replay_serving_enabled(expected['arm']), 'ReplaySSM resolved arm flags differ')
    cli = binding.get('canonical_server_cli', [])
    _need('--enable-linear-replayssm' not in cli and cli.count('--enable-linear-replayssm-spec') == int(_mtp_contract().replay_serving_enabled(expected['arm'])), 'ReplaySSM exact launch flag differs')
    _need(type(contract) is dict and contract.get('tp_size') == 2 and (contract.get('context_length') == _mtp_contract().phase_config()[0]) and (contract.get('chunked_prefill_size') == _mtp_contract().prefill_chunk_size(binding['arm'])) and (contract.get('max_running_requests') == 2) and (contract.get('cuda_graph_bs_decode') == [1, 2]) and (contract.get('cuda_graph_max_bs_decode') == 2), 'resolved native serving contract differs')
    if _mtp_contract().precision_reference_enabled():
        _mtp_contract().validate_precision_loading_policy({key: contract.get(key) for key in _mtp_contract().PRECISION_LOADING_FIELDS})
        _mtp_contract().validate_precision_loading_cli(binding.get('canonical_server_cli'))
        _need(contract.get('model_config_sha256') == _mtp_contract().target_config_sha256() and contract.get('mem_fraction_static') == _mtp_contract().memory_fraction_static(), 'precision reference target config binding differs')
    if _mtp_contract().serving_seed_required():
        cli = binding.get('canonical_server_cli', [])
        _need(type(contract.get('random_seed')) is int and contract['random_seed'] == 12345 and (cli.count('--random-seed') == 1) and (cli[cli.index('--random-seed') + 1:cli.index('--random-seed') + 2] == ['12345']), 'native chain diagnostic seed binding differs')
    head_roles = {'head_logits', 'head_embedding', 'head_unquant', 'head_linear', 'head_features', 'head_context', 'head_model'}
    if 'enable_fp32_lm_head' in contract:
        _need(contract['enable_fp32_lm_head'] is True and head_roles <= binding['sources'].keys() and (binding['canonical_server_cli'].count('--enable-fp32-lm-head') == 1), 'native FP32 head binding incomplete')
    else:
        _need(not head_roles & binding['sources'].keys() and '--enable-fp32-lm-head' not in binding.get('canonical_server_cli', []), 'unbound native FP32 head mode')
    hip_components = None
    if hip_components_required:
        _, _, parity_validator = _private_reader(include_parity=True)
        hip_components = parity_validator.validate_hip_sampling_components([_observation_json(observations, f'HIP-SAMPLING-COMPONENT-rank-{rank}.json') for rank in (0, 1)], binding=binding, workers=full_workers, phase=phase)
    mtp_contract = _mtp_contract()
    pool = mtp_contract.memory_pool_config(expected['arm'])
    if pool is not None:
        mtp_contract.validate_memory_pool_config({key: contract.get(key) for key in pool}, expected['arm'])
    if mtp_contract.cache_config() is not None:
        mtp_contract.validate_cache_config(contract.get('prefix_cache'))
    parity = _observation_json(observations, 'NATIVE-CHAT-PARITY.json')
    _need(observations['NATIVE-CHAT-PARITY.ready'].decode('ascii') == hashlib.sha256(observations['NATIVE-CHAT-PARITY.json']).hexdigest(), 'native parity publication hash')
    _need(parity.get('passed') is True and parity.get('native_manager') is True and (parity.get('before_scheduler_delegation') is True) and (parity.get('qualification') is False) and (parity.get('mode') == MODE) and all((parity.get(key) == expected[key] for key in ('arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256'))), 'native preweight parity identity')
    parity_contract, _ = _private_reader()
    frozen = parity_contract.requests()
    _need(type(parity.get('rows')) is list and len(parity['rows']) == len(frozen), 'native parity count')
    for ordinal, (row, case) in enumerate(zip(parity['rows'], frozen)):
        _need(row.get('ordinal') == ordinal and row.get('case') == case['case'] and (row.get('request_sha256') == parity_contract.digest(case['body'])) and (row.get('actual_ids') == parity_contract.expected_case(case['case'])['input_ids']) and (row.get('parity') is True) and (row.get('reasoning_disabled') is True) and (row.get('error') is None), 'native preweight parity row')
        variants = row.get('generated_history_branches', [])
        branches = case.get('generated_history', {}).get('branches', [])
        _need(type(variants) is list and len(variants) == len(branches), 'native generated history branch count')
        for index, (observed, branch) in enumerate(zip(variants, branches)):
            selected = parity_contract.history_branch(case, index)
            _need(type(observed) is dict and observed.get('ordinal') == ordinal and (observed.get('case') == case['case']) and (observed.get('branch_index') == index) and (type(observed.get('branch_index')) is int) and (observed.get('assistant_text_sha256') == hashlib.sha256(branch['assistant_text'].encode()).hexdigest()) and (observed.get('expected_sha256') == branch['expected_sha256']) and (observed.get('request_sha256') == parity_contract.digest(selected['body'])) and (observed.get('actual_ids') == parity_contract.expected_for_case(selected)['input_ids']) and (observed.get('actual_ids_sha256') == parity_contract.digest(observed['actual_ids'])) and (observed.get('parity') is True) and (observed.get('reasoning_disabled') is True) and (observed.get('error') is None), 'native generated history branch identity or parity')
    expected_mtp = mtp_contract.contract_for_arm(expected['arm'])
    _need(binding.get('speculation') == expected_mtp, 'native serving speculation binding differs')
    preflight = _observation_json(observations, 'MTP-PREFLIGHT.json')
    for rank, worker in enumerate(full_workers):
        preweight = _observation_json(observations, f'worker-rank-{rank}-preweight.json')
        _need(type(preweight) is dict and preweight.get('schema') == 1 and (preweight.get('mode') == MODE) and (preweight.get('arm') == expected['arm']) and (preweight.get('rank') == rank) and (preweight.get('session_sha256') == expected['session_sha256']) and (preweight.get('spec_sha256') == expected['spec_sha256']) and (preweight.get('before_scheduler') is True) and (preweight.get('qualification') is False) and (type(preweight.get('devices')) is list) and (len(preweight['devices']) == 2) and all((type(row) is dict and set(row) == {'index', 'bdf', 'uuid', 'architecture', 'vram_bytes'} for row in preweight['devices'])) and (type(preweight.get('render_bindings')) is list) and (len(preweight['render_bindings']) == 2) and all((type(row) is dict and set(row) == {'bdf', 'render', 'sysfs_bdf', 'uuid'} for row in preweight['render_bindings'])) and (type(preweight.get('hip')) is dict) and (preweight.get('hip', {}).get('no_load_handle') is True) and (type(preweight.get('identity_sha256')) is str), 'native HIP preweight observation differs')
        mtp = _observation_json(observations, f'MTP-PREFLIGHT-rank-{rank}.json')
        expected_fields = {'schema', 'mode', 'arm', 'rank', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'contract', 'environment', 'resolved_speculative', 'draft_model_config', 'model_config_source', 'before_scheduler', 'qualification'}
        if mtp_contract.cache_config() is not None:
            expected_fields.add('resolved_prefix_cache')
        if mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES:
            expected_fields.add('resolved_observability')
        if pool is not None:
            expected_fields.add('resolved_memory_pool')
        _need(type(mtp) is dict and set(mtp) == expected_fields and (mtp['schema'] == 1) and (mtp['mode'] == MODE) and (mtp['arm'] == expected['arm']) and (mtp['rank'] == rank) and (mtp['session_sha256'] == expected['session_sha256']) and (mtp['spec_sha256'] == expected['spec_sha256']) and (mtp['source_manifest_sha256'] == expected['source_manifest_sha256']) and (mtp_contract.validate_contract(mtp['contract']) == expected_mtp) and (mtp['environment'] == mtp_contract.expected_environment(expected['arm'])) and (mtp['before_scheduler'] is True) and (mtp['qualification'] is False) and (type(mtp['model_config_source']) is dict) and (set(mtp['model_config_source']) == {'module', 'path', 'sha256'}) and (mtp['model_config_source'].get('module') == 'sglang.srt.configs.model_config') and (type(mtp['model_config_source'].get('path')) is str) and _digest(mtp['model_config_source'].get('sha256')), 'MTP native preflight differs')
        if mtp_contract.cache_config() is not None:
            mtp_contract.validate_cache_config(mtp['resolved_prefix_cache'])
            _need(mtp['resolved_prefix_cache'] == contract['prefix_cache'], 'native rank prefix-cache binding differs')
        if pool is not None:
            mtp_contract.validate_memory_pool_config(mtp['resolved_memory_pool'], expected['arm'])
            _need(mtp['resolved_memory_pool'] == {key: contract[key] for key in pool}, 'native rank memory-pool binding differs')
        if mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES:
            flags = mtp['resolved_observability']
            _need(type(flags) is dict and set(flags) == {'enable_metrics', 'decode_log_interval'} and (flags['enable_metrics'] is True) and (type(flags['decode_log_interval']) is int) and (flags['decode_log_interval'] == mtp_contract.expected_decode_log_interval(expected['arm'])), 'closure native rank observability differs')
        resolved_fields = {'speculative_algorithm', 'enable_multi_layer_eagle', 'speculative_draft_model_path', 'speculative_draft_model_revision', 'speculative_num_steps', 'speculative_eagle_topk', 'speculative_num_draft_tokens', 'speculative_dflash_block_size', 'quantization', 'speculative_draft_model_quantization', '_speculative_draft_quantization_explicitly_set', 'speculative_use_rejection_sampling', 'speculative_accept_threshold_single', 'speculative_accept_threshold_acc'}
        resolved = mtp['resolved_speculative']
        explicit_draft_quantization = resolved.get('_speculative_draft_quantization_explicitly_set')
        _need(type(resolved) is dict and set(resolved) == resolved_fields and (resolved['quantization'] == mtp_contract.target_quantization()) and (type(explicit_draft_quantization) is bool or (not expected_mtp['enabled'] and explicit_draft_quantization is None)) and (type(resolved['speculative_use_rejection_sampling']) is bool) and (type(resolved['speculative_accept_threshold_single']) in (int, float)) and (type(resolved['speculative_accept_threshold_acc']) in (int, float)), 'MTP resolved speculative provenance differs')
        _need(all((resolved[field] == expected_mtp[field] for field in ('speculative_algorithm', 'enable_multi_layer_eagle', 'speculative_draft_model_path', 'speculative_draft_model_revision', 'speculative_num_steps', 'speculative_eagle_topk', 'speculative_num_draft_tokens', 'speculative_dflash_block_size'))), 'MTP effective contract differs')
        if not expected_mtp['enabled']:
            _need(resolved['speculative_draft_model_quantization'] in (None, 'fp8') and mtp['draft_model_config'] == {'is_draft_model': False, 'architecture': None, 'num_nextn_predict_layers': None, 'text_num_nextn_predict_layers': None}, 'control arm enabled MTP')
        elif expected_mtp['speculative_kind'] == 'dflash2':
            _need(resolved['speculative_draft_model_quantization'] is None and explicit_draft_quantization is True and (mtp['draft_model_config'] == mtp_contract.expected_draft_summary('dflash2')), 'DFlash2 effective native preflight differs')
        else:
            _need(resolved['speculative_draft_model_quantization'] == 'fp8' and type(mtp['draft_model_config']) is dict and (mtp['draft_model_config'].get('architecture') == 'Qwen3_5ForCausalLMMTP') and (mtp['draft_model_config'].get('num_nextn_predict_layers') == 1) and (mtp['draft_model_config'].get('text_num_nextn_predict_layers') == 1) and (mtp['draft_model_config'].get('is_draft_model') is True) and (mtp['draft_model_config'].get('quantization') == 'fp8') and (mtp['draft_model_config'].get('is_draft_quantization_explicit') is False), 'MTP effective native preflight differs')
        if rank == 0:
            _need(preflight == mtp, 'canonical MTP preflight differs from rank zero')
        capture = _observation_json(observations, f'graph-captures-rank-{rank}.json')
        if getattr(mtp_contract, 'j01_probe_enabled', lambda: False)():
            helper = _probe_helper(binding['sources'])
            helper.validate(capture.get('token_probe'), binding=binding, worker=worker, contract=mtp_contract, complete=head_input_required)
            if head_input_required:
                helper.validate_head_input(_observation_json(observations, f'head-input-rank-{rank}.json'), binding=binding, worker=worker, state=capture['token_probe'], complete=True)
        elif mtp_contract.settings().get('quality_screen') == 'sprint36-chunk-prefill-v1':
            helper = _probe_helper(binding['sources'])
            prefill = capture.get('token_probe')
            _need(type(prefill) is dict and prefill.get('mode') == helper.PREFILL_MODE,
                  'explicit prefill observation missing or wrong mode')
        else:
            _need('token_probe' not in capture, 'unexpected J01 probe record')
        counter = observations[f'graph-counter-rank-{rank}.bin']
        closure = mtp_contract.settings()['phase'] in mtp_contract.BATCH_PHASES
        _need(type(capture) is dict and capture.get('schema') == 1 and (capture.get('rank') == rank) and (capture.get('arm') == expected['arm']) and (capture.get('native_graph_api_observations') is True) and (capture.get('qualification') is False) and (capture.get('gpu_time_measured') is False) and (capture.get('pid') == worker['pid']) and (type(capture.get('events')) is list) and (len(counter) == (32 if closure else 16)), 'native graph startup observation differs')
        if closure:
            _need(capture.get('phase') == mtp_contract.settings()['phase'] and capture.get('counter_layout') == '<QQQQ captures,total,bs1,bs2' and (capture.get('replay_batch_binding') == 'original-forward-batch.batch_size=runner.raw_bs=shape.size'), 'lifecycle graph request-count binding differs')
            captures, replays, singleton_replays, batch2_replays = struct.unpack('<QQQQ', counter)
            _need(singleton_replays > 0 and batch2_replays > 0 and (replays == singleton_replays + batch2_replays), 'closure lacks actual target batch-two graph replay')
        else:
            captures, replays = struct.unpack('<QQ', counter)
        wanted = {(size, phase, step) for size in (1, 2) for phase, step in (('warmup', 0), ('warmup', 1), ('capture', None))}
        actual = {(row.get('size'), row.get('phase'), row.get('warmup_step')) for row in capture['events'] if type(row) is dict}
        _need(len(capture['events']) == 6 and actual == wanted and (captures == 2) and isinstance(capture.get('backend_identity'), dict), 'target graph capture buckets incomplete')
        if gdn_observer is not None:
            _need(worker.get('sources') == binding['sources'], 'GDN worker sources differ from binding')
            gdn_captures[rank] = {'pid': worker['pid'], 'backend_identity': capture['backend_identity'], 'gdn_sha256': _gdn_capture(capture, observer=gdn_observer, sources=binding['sources'], speculation=binding['speculation'], arm=expected['arm'], rank=rank, pid=worker['pid'])}
    for ordinal in range(_mtp_contract().phase_config()[2]):
        before = _observation_json(observations, f'request-{ordinal:02d}-before.json')
        after = _observation_json(observations, f'request-{ordinal:02d}-after.json')
        _need(type(before) is dict and type(after) is dict and (before.get('phase') == 'before') and (after.get('phase') == 'after') and (before.get('ordinal') == after.get('ordinal') == ordinal) and (before.get('arm') == after.get('arm') == expected['arm']) and (before.get('body_sha256') == after.get('body_sha256')) and (before.get('instrumented_latency') is True) and (after.get('instrumented_latency') is True) and (before.get('qualification') is False) and (after.get('qualification') is False), 'per-request native graph observation differs')
        before_rows, after_rows = (before.get('snapshot', {}).get('ranks'), after.get('snapshot', {}).get('ranks'))
        _need(type(before_rows) is list and type(after_rows) is list and ([row.get('rank') for row in before_rows if type(row) is dict] == [0, 1]) and ([row.get('rank') for row in after_rows if type(row) is dict] == [0, 1]), 'per-request graph rank observation differs')
        if gdn_observer is not None:
            for rows in (before_rows, after_rows):
                for rank, row in enumerate(rows):
                    _need(type(row.get('rank')) is int and all((row.get(key) == item for key, item in gdn_captures[rank].items())), 'per-request GDN startup identity differs')
        deltas = [after_rows[index].get('replays', -1) - before_rows[index].get('replays', -1) for index in (0, 1)]
        _need(all((delta > 0 for delta in deltas)), 'per-request target graph replay did not advance')
        for rank, delta in enumerate(deltas):
            left, right = (before_rows[rank].get('completed_commits'), after_rows[rank].get('completed_commits'))
            _need(type(left) is int and type(right) is int and (0 <= left < right <= mtp_contract.commit_observation_limit()) and (right - left == delta or (_mtp_contract().betterbench_checkpoint_enabled() and 47 <= ordinal < 53)), 'per-request completed commit/replay count differs')
            final_counter = observations[f'replay-commit-counter-rank-{rank}.bin']
            _need(len(final_counter) == 8 and right <= struct.unpack('<Q', final_counter)[0] <= mtp_contract.commit_observation_limit(), 'completed commit final counter binding')
    if _mtp_contract().settings()['phase'] in _mtp_contract().CLOSURE_PHASES:
        first = _observation_json(observations, 'request-24-before.json')
        last = _observation_json(observations, 'request-25-after.json')
        for left, right in zip(first['snapshot']['ranks'], last['snapshot']['ranks']):
            _need(right.get('bs2_replays', -1) > left.get('bs2_replays', -1) >= 0, 'closure concurrent group lacks native batch-two replay')
    states = [_observation_json(observations, f'graph-captures-rank-{rank}.json').get('token_probe') for rank in (0, 1)]
    _validate_final_norm(observations, binding, full_workers, states)
    _validate_midpoint(observations, binding, full_workers, states)
    return hip_components

def _joined(workers, joins, supervision):
    _need(type(joins) is list and joins and (supervision is not None), 'live worker joins unavailable')
    observed = getattr(supervision, 'observed_processes', ())
    for worker in workers:
        matches = [row for row in joins if type(row) is dict and type(row.get('host_process')) is dict and (type(row.get('nspid')) is list) and row['nspid'] and (row['nspid'][-1] == worker['pid']) and all((row['host_process'].get(key) == worker[key] for key in ('boot_id', 'start_ticks')))]
        _need(matches and any((item.start_ticks == worker['start_ticks'] and item.boot_id == worker['boot_id'] for row in matches for item in observed)), 'native HTTP worker lacks observed live-process join')

def validate_raw_attention(directory, *, observations, workers):
    """Called only after native, live-worker, accepted-head and wire gates pass."""
    binding = _observation_json(observations, 'SERVING-BINDING.json')
    helper_root = Path(__file__).resolve().parent
    originals = {name: sys.modules.get(name) for name in ('attention_capture', 'raw_attention')}
    try:
        loaded = {}
        for name in originals:
            path = helper_root / (name + '.py')
            source = binding['sources'].get(name)
            _need(type(source) is dict and set(source) == {'module', 'path', 'sha256'} and (source['module'] == name) and (source['path'] in ('/mnt/review/' + path.name, str(path))) and (source['sha256'] == hashlib.sha256(_read(path, 1 << 20)).hexdigest()), 'raw reader source binding')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        _need([row.get('rank') for row in workers] == [0, 1], 'raw reader rank membership')
        results = []
        for worker in workers:
            rank = worker['rank']
            head = _observation_json(observations, f'head-input-rank-{rank}.json')
            mapping = head['mapping']
            replay = dict(forward_pass_id=mapping['native_forward_pass_id'], epoch=mapping['forward_epoch'], replay_sequence=mapping['replay_sequence'], context_sha256=head['context_sha256'], phase='after_replay_before_accept')
            launch = loaded['attention_capture'].validate_launch_record(_observation_json(observations, f'attention-launch-rank-{rank}.json'), binding=binding, worker=worker, packet_sha256=hashlib.sha256(observations[f'prepared-pending-rank-{rank}.json']).hexdigest(), replay=replay)
            loaded['attention_capture'].validate_launch_geometry(launch, _observation_json(observations, f'prepared-pending-rank-{rank}.json'))
            results.append(loaded['raw_attention'].validate_capture(Path(directory) / f'raw-rank-{rank}', packet_raw=observations[f'prepared-pending-rank-{rank}.json'], binding=binding, worker=worker, replay=replay))
        return dict(mode=loaded['raw_attention'].MODE, ranks=results, qualification=False)
    finally:
        for name, previous in originals.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

def validate_prompt_cache_chronology(observations, client_root, wire):
    """Join authenticated flush completion, native snapshot and next wire intent."""
    if not _mtp_contract().prompt_minimization_enabled():
        return
    for ordinal, row in enumerate(wire['rows']):
        if ordinal == 0 or row.get('status') != 'NORMAL_STOP_COMPLETE':
            continue
        prefix = '%02d-%s' % (ordinal, row['case'])
        control = _json(_read(Path(client_root) / ('controls/prompt-flush-before-%02d.record.json' % ordinal)))
        intent = _json(_read(Path(client_root) / ('requests/%s.intent.json' % prefix)))
        observed = _observation_json(observations, 'request-%02d-before.json' % ordinal)
        request_raw = _read(Path(client_root) / ('requests/%s.raw' % prefix), 256 << 10)
        body_raw = json.dumps(_json(request_raw), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        _need(observed.get('case') == row['case'] and observed.get('body_sha256') == hashlib.sha256(body_raw).hexdigest() and (intent.get('body_sha256') == hashlib.sha256(request_raw).hexdigest()), 'prompt flush/native observer/request identity differs')
        times = [control.get('finished_monotonic'), observed.get('observed_monotonic'), intent.get('prepared_monotonic')]
        _need(all((type(tick) in (int, float) and math.isfinite(tick) and (tick >= 0) for tick in times)) and times == sorted(times), 'prompt flush/native observer/intent chronology differs')

def read_http_artifacts(directory, *, extraction_root, expected, owner_arm, joins, supervision):
    """Recompute one complete HTTP arm from exactly three exported raw members."""
    _need(owner_arm in ('bf16', 'kv4') and type(expected) is dict and (set(expected) == {'arm', 'source_manifest_sha256', 'session_sha256', 'spec_sha256'}), 'HTTP expected binding')
    logical = _mtp_contract().owner_arm_map()[owner_arm]
    _need(expected['arm'] == logical, 'owner/logical arm binding')
    directory = Path(directory)
    client_raw = _read(directory / 'http-client-bundle.json')
    native_raw = _read(directory / 'native-http-bundle.json')
    result_raw = _read(directory / 'result.json')
    client_root, client_sha = _unpack_client(client_raw, extraction_root)
    gate_raw, workers, full_workers, observations, native_sha, reader = _native(native_raw, expected)
    _joined(workers, joins, supervision)
    observation_storage = None
    if _mtp_contract().betterbench_checkpoint_enabled():
        observations, observation_storage = _bb_expand_observations(observations)
    hip_components = _validate_native_observations(observations, expected=expected, gate=_json(gate_raw), full_workers=full_workers)
    if _mtp_contract().betterbench_checkpoint_enabled():
        wire=betterbench_owner.read_owner_artifacts(client_root,observations,expected=expected,workers=full_workers)
        _need(not({'schema','mode','arm','binding','status'}&set(wire)),'child wire cannot assert parent identity envelope')
        wire=dict(wire,schema=1,mode=MODE,arm=logical,binding=dict(expected),status='HTTP_WIRE_COMPLETE')
        wire['c2_global_counter_join'] = _bb_validate_c2_counter_group(observations, wire)
        probe=raw_attention=None
        _need(not any(directory.glob('raw-rank-*')), 'unbound raw attention files')
    else:
        wire = reader.read_client(client_root, expected=expected, native_gate_raw=gate_raw)
        validate_prompt_cache_chronology(observations, client_root, wire)
        validate_prefill_wire(observations, full_workers, wire)
        probe = validate_probe_wire(observations, full_workers, wire)
        if _mtp_contract().paired_serving_enabled():
            wire['paired_native_groups'] = validate_paired_graph_groups(observations, wire)
        raw_attention = None
        if _mtp_contract().raw_attention_capture_enabled():
            raw_attention = validate_raw_attention(directory, observations=observations, workers=full_workers)
        else:
            _need(not any(directory.glob('raw-rank-*')), 'unbound raw attention files')
        for ordinal, row in enumerate(wire['rows']):
            if reader.contract.mtp_contract.settings()['phase'] in reader.contract.mtp_contract.CLOSURE_PHASES and ordinal == 22 and (row.get('status') == 'CANCELLED_COMPLETE'):
                request_raw = _read(client_root / f"requests/{ordinal:02d}-{row['case']}.raw", 256 << 10)
                _need(hashlib.sha256(request_raw).hexdigest() == reader.contract.digest(reader.contract.requests()[ordinal]['body']), 'validated cancel request bytes changed')
                body_raw = json.dumps(_json(request_raw), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
                for phase in ('before', 'after'):
                    _need(_observation_json(observations, f'request-{ordinal:02d}-{phase}.json').get('body_sha256') == hashlib.sha256(body_raw).hexdigest(), 'native cancel observation differs from actual wire body')
            if reader.contract.response_completed(row):
                request_raw = _read(client_root / f"requests/{ordinal:02d}-{row['case']}.raw", 256 << 10)
                _need(hashlib.sha256(request_raw).hexdigest() == row.get('request_sha256'), 'validated wire request bytes changed')
                body_raw = json.dumps(_json(request_raw), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
                body_sha256 = hashlib.sha256(body_raw).hexdigest()
                for phase in ('before', 'after'):
                    _need(_observation_json(observations, f'request-{ordinal:02d}-{phase}.json').get('body_sha256') == body_sha256, 'native request observation differs from actual wire body')
    result = _json(result_raw)
    required = {'schema', 'mode', 'status', 'arm', 'logical_arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'client_summary', 'native_gate', 'parent_error', 'qualification'}
    _need(type(result) is dict and set(result) == required and (result['schema'] == 1) and (result['mode'] == MODE) and (result['arm'] == owner_arm) and (result['logical_arm'] == logical) and (result['session_sha256'] == expected['session_sha256']) and (result['spec_sha256'] == expected['spec_sha256']) and (result['source_manifest_sha256'] == expected['source_manifest_sha256']) and (result['qualification'] is False), 'HTTP result binding')
    _need(result['native_gate'] == _json(gate_raw) and result['client_summary'] == _json(_read(client_root / 'RUN-SUMMARY.json', 4 << 20)), 'HTTP result/raw evidence disagreement')
    complete = wire['wire_complete'] is True and result['status'] == 'HTTP_ARM_COMPLETE' and (result['parent_error'] is None)
    if not complete:
        _need(result['status'] == 'HTTP_ARM_FAILED', 'HTTP incomplete result status')
    return dict(schema=1, mode=MODE, owner_arm=owner_arm, logical_arm=logical, binding=dict(expected), wire=wire, status='HTTP_ARM_COMPLETE' if complete else 'HTTP_ARM_FAILED', qualification=False, client_bundle_sha256=client_sha, native_bundle_sha256=native_sha, full_workers=full_workers, native_observation_storage=observation_storage, native_observations={name: hashlib.sha256(raw).hexdigest() for name, raw in observations.items()}, result_sha256=hashlib.sha256(result_raw).hexdigest(), **{'j01_probe': probe} if probe is not None else {}, **{'raw_attention': raw_attention} if raw_attention is not None else {}, **{'hip_sampling_components': hip_components} if hip_components is not None else {})

def accept_with_owner(value, campaign, *, owner_arm):
    """Attach the already-validated owner result; this creates no authority."""
    _need(type(value) is dict and value.get('status') == 'HTTP_ARM_COMPLETE', 'incomplete HTTP arm')
    _, reader = _private_reader()
    return reader.accept_with_owner(value['wire'], campaign, owner_arm=owner_arm)

def validate_full_enabled_record(record, binding, worker):
    fields = {'mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'schema', 'rank', 'pid', 'worker_identity_sha256', 'enabled', 'block_size', 'before_external_generations', 'extra_tensor_operations', 'extra_model_forwards', 'qualification', 'adapter_source', 'eligible_host_calls', 'capture_host_calls', 'layers', 'buffers'}
    fields.update({'metadata_build_host_calls', 'metadata_reuse_host_calls', 'metadata_capture_build_host_calls'})
    fields.add('indices_elements')
    _need(type(record) is dict and set(record) == fields and (record['schema'] == 1), 'full-enabled witness schema')
    for key in ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256'):
        _need(record[key] == binding[key], 'full-enabled witness source/session binding')
    _need(record['rank'] == worker['rank'] and record['pid'] == worker['pid'] and (record['worker_identity_sha256'] == worker['identity_sha256']), 'full-enabled witness worker identity')
    enabled = _mtp_contract().contract_for_arm(binding['arm'])['enabled']
    _need(record['enabled'] is enabled and record['before_external_generations'] is True and (record['qualification'] is False) and (record['extra_tensor_operations'] == record['extra_model_forwards'] == 0), 'full-enabled witness scope')
    for key in ('eligible_host_calls', 'capture_host_calls', 'buffers'):
        _need(type(record[key]) is int and record[key] >= 0, 'full-enabled witness count')
    if enabled:
        _need(record['block_size'] == 8 and record['adapter_source'] == binding['sources']['verify_decode_adapter'] and (0 < record['capture_host_calls'] <= record['eligible_host_calls']) and (record['buffers'] > 0) and (type(record['layers']) is list) and (len(record['layers']) == len(set(record['layers'])) == 16), 'split-KV adapter absent or not captured')
    else:
        _need(record['block_size'] is record['adapter_source'] is None and record['layers'] == [] and (record['buffers'] == record['capture_host_calls'] == record['eligible_host_calls'] == 0), 'ordinary baseline unexpectedly enabled')
    counts = [record[key] for key in ('metadata_build_host_calls', 'metadata_reuse_host_calls', 'metadata_capture_build_host_calls')]
    _need(all((type(n) is int and n >= 0 for n in counts)), 'metadata reuse witness counts')
    if enabled:
        _need(type(record['indices_elements']) is list and len(record['indices_elements']) == record['buffers'] and binding.get('resolved_contract', {}).get('context_length') in (32768, 65536) and all(type(n) is int and n == 8 * binding['resolved_contract']['context_length'] for n in record['indices_elements']), '64K split-KV indices geometry differs')
        _need(record['metadata_capture_build_host_calls'] == 1 and record['capture_host_calls'] == 16 and (record['metadata_build_host_calls'] >= 1) and (record['metadata_reuse_host_calls'] >= 15) and (record['metadata_build_host_calls'] + record['metadata_reuse_host_calls'] == record['eligible_host_calls']), 'metadata not reused across captured attention layers')
    else:
        _need(counts == [0, 0, 0], 'ordinary reference reported metadata reuse')
    return record


def validate_paired_graph_groups(observations, wire):
    """Native worker/graph identities are authenticated by _validate_native first."""
    _, reader = _private_reader()
    contract=reader.contract
    _need(contract.mtp_contract.paired_serving_enabled(), 'paired native phase')
    groups=contract.mtp_contract.paired_groups(wire['arm'])
    _need(wire.get('wire_complete') is True and len(wire.get('paired_groups',[]))==len(groups),
        'paired complete wire groups required')
    results=[];previous_after=None
    for group,measured in zip(groups,wire['paired_groups'],strict=True):
        _need(measured['group_id']==group['id'] and measured['ordinals']==group['ordinals'], 'paired group order')
        before=[_observation_json(observations,f'request-{i:02d}-before.json') for i in group['ordinals']]
        after=[_observation_json(observations,f'request-{i:02d}-after.json') for i in group['ordinals']]
        for records,phase in ((before,'before'),(after,'after')):
            for ordinal,row in zip(group['ordinals'],records,strict=True):
                _need(row['ordinal']==ordinal and row['phase']==phase and row['group_id']==group['id'] and
                    row['group_ordinals']==group['ordinals'] and row['arm']==wire['arm'] and
                    row['snapshot']==records[0]['snapshot'] and row['observed_monotonic']==records[0]['observed_monotonic'],
                    'paired shared group snapshot differs')
        first,last=before[0],after[0]
        _need(first['observed_monotonic']<=measured['prepared_monotonic']<measured['received_monotonic']<=last['observed_monotonic'],
            'paired native/wire group chronology')
        if previous_after is not None:
            _need(previous_after['observed_monotonic']<=first['observed_monotonic'], 'paired native group chronology')
        ranks=[]
        for rank,(left,right) in enumerate(zip(first['snapshot']['ranks'],last['snapshot']['ranks'],strict=True)):
            _need(left['rank']==right['rank']==rank and left['captures']==right['captures']==2,
                'paired graph recapture or rank differs')
            _need(all(left[k]==right[k] for k in ('pid','backend_identity','gdn_sha256')), 'paired graph identity drift')
            if previous_after is not None:
                prior=previous_after['snapshot']['ranks'][rank]
                _need(all(prior[k]==left[k] for k in ('replays','completed_commits','bs1_replays','bs2_replays','captures')),
                    'unattributed graph work between paired groups')
            delta={key:right[key]-left[key] for key in ('replays','completed_commits','bs1_replays','bs2_replays')}
            counts=[wire['rows'][i]['spec_tokens_details']['spec_verify_ct'] for i in group['ordinals']]
            ranks.append(dict(rank=rank,**contract.paired_group_counts(delta,counts,group['concurrency'])))
        _need(len(ranks)==2, 'paired graph rank count')
        results.append(dict(group_id=group['id'],ordinals=group['ordinals'],concurrency=group['concurrency'],ranks=ranks))
        previous_after=last
    return results


"""Expand authenticated journal bytes before unchanged logical witness checks."""
def _bb_expand_observations(observations):
    result = dict(observations)
    storage = {}
    for ordinal in range(53):
        name = f'request-{ordinal:02d}.jsonl'
        raw = result.pop(name, None)
        _need(type(raw) is bytes and 0 < len(raw) <= 16384, 'bounded complete BetterBench journal')
        lines = raw.splitlines(keepends=True)
        _need(len(lines) == 2, 'exact two BetterBench observation records')
        logical = {}
        for phase, line in zip(('before', 'after'), lines):
            _need(0 < len(line) <= 8192 and line.endswith(b'\n'), 'bounded terminated BetterBench observation')
            row = _json(line)
            canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            _need(canonical + b'\n' == line and type(row) is dict and row.get('phase') == phase
                  and type(row.get('ordinal')) is int and row['ordinal'] == ordinal,
                  'canonical ordered BetterBench observation')
            old_name = f'request-{ordinal:02d}-{phase}.json'
            _need(old_name not in result, 'mixed legacy/journal BetterBench observations')
            result[old_name] = canonical
            logical[old_name] = hashlib.sha256(canonical).hexdigest()
        storage[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), logical_sha256=logical)
    _need(not any(name.startswith('request-') and name.endswith('.jsonl') for name in result),
          'unexpected BetterBench journal ordinal')
    return result, dict(mode='betterbench-two-record-observation-journal-v1', journals=storage,
                        logical_records=106, lossless=True, protected_export_limits_changed=False)


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
