"""Private W5 HTTP entry and retained controller/readout compatibility helpers.

The outer owner must authenticate the runtime, model, spec and two devices.
Native execution requires the reviewed entry flag and physical artifact identity.
Retired model timing and independent-reference modes cannot launch here.
"""
import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
SCOPE = 'reviewed-private-tp2-model-benchmark-v1'
CELLS = [('ctx16384-seed1729-depth50', 1, 16352), ('ctx32768-seed1729-depth10', 1, 32736), ('ctx32768-seed1729-depth50', 1, 32736), ('ctx32768-seed1729-depth90', 1, 32736), ('ctx32768-seed2718-depth10', 1, 32736), ('ctx32768-seed2718-depth50', 1, 32735), ('ctx32768-seed2718-depth90', 1, 32736)]
GPU_IDENTITY_SHA256 = '7324842c4f779749d561dc586168bd923e84e6900d17e3d22c140647bcd133df'
CACHE_PROFILE_SHA256 = '256250bc43c4dfd7146f14f60b03bfa09f7d3648dd5f5e9c3298e4edb34851e5'
CACHE_PROFILE = {'schema': 1, 'mode': 'v4-host-attribution-v1', 'native_phase_abi_bytes': 64, 'extra_device_bytes': 0, 'extra_synchronizations': 0, 'reader_events': False}
COMPARISON_SHA256 = '607c5a44c3192329ec7566d0deadeafe5dcfcd7447fa306caa037aa20a7f5d28'
TELEMETRY_SHA256 = '0619773ea3ba5730381ef9dee0424d4488e69e8b33943e12b00b8d64a5720c74'
PHYSICAL_DEVICES = {'0000:c3:00.0': '39373864-6465-3038-6163-303837666533', '0000:23:00.0': '34663466-6561-3362-3832-346261366439'}
INDEPENDENT_MODE = 'independent-bf16-reference-v1'

def read_bounded_file(path, maximum):
    """Read one regular no-follow, singly linked file without size/identity races."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('absolute reviewed input required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError('input type/size')
        data = b''
        while len(data) <= maximum:
            block = os.read(fd, min(1048576, maximum + 1 - len(data)))
            if not block:
                break
            data += block
        after = os.fstat(fd)
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if stamp(before) != stamp(after) or stamp(after) != stamp(os.stat(path, follow_symlinks=False)):
            raise ValueError('input changed')
        if len(data) > maximum:
            raise ValueError('input size')
        return data
    finally:
        os.close(fd)

def checked_file(path, expected, maximum=8 << 20):
    data = read_bounded_file(path, maximum)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('input digest')
    return data

def _verified_source_spec(name, path, raw):
    """Bind a loader to the caller's checked bytes, excluding cached bytecode."""
    loader = SourceFileLoader(name, str(path))
    loader.get_code = lambda fullname: loader.source_to_code(raw, str(path))
    return importlib.util.spec_from_loader(name, loader)

def authenticated_adjacent(filename, expected):
    path = Path(__file__).resolve().with_name(filename)
    raw = checked_file(path, expected, 256 << 10)
    spec = _verified_source_spec('spry_' + path.stem, path, raw)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def http_runtime_module(spec, arm):
    """Source-load the private HTTP chain in isolated Python mode.

    ``python -I`` intentionally ignores ambient import roots.  The HTTP
    template is already package-pinned, so this loader reads that one template
    from its fixed `/mnt/review` location and executes only its exact private
    dependencies in dependency order.  Native SGLang modules are not imported
    here.
    """
    http = spec['http_serving']
    bootstrap = http['bindings']['control']
    if bootstrap.get('path') != 'review/HTTP-BINDING-control.json' or not valid_digest(bootstrap.get('sha256')):
        raise ValueError('HTTP mapping bootstrap template pin')
    bootstrap = json.loads(checked_file(Path('/mnt/review/HTTP-BINDING-control.json'), bootstrap['sha256'], 256 << 10))
    source = bootstrap['binding_template']['sources']['mtp_contract']
    if source.get('path') != str(Path(__file__).resolve().with_name('mtp_contract.py')) or not valid_digest(source.get('sha256')):
        raise ValueError('HTTP mapping bootstrap contract pin')
    mapping_module = authenticated_adjacent('mtp_contract.py', source['sha256'])
    mapping = mapping_module.owner_arm_map()
    logical = mapping.get(arm)
    if logical is None:
        raise ValueError('HTTP owner arm')
    row = http['bindings'][logical]
    relative = 'review/HTTP-BINDING-' + logical + '.json'
    if type(row) is not dict or set(row) != {'path', 'sha256'} or row['path'] != relative or (not valid_digest(row['sha256'])):
        raise ValueError('HTTP binding template pin')
    outer = mapping_module.expand_http_template(json.loads(checked_file(Path('/mnt') / relative, row['sha256'], 256 << 10)))
    required_outer = {'schema', 'mode', 'arm', 'source_manifest_sha256', 'port', 'binding_template', 'worker_template', 'speculation'}
    if not (type(outer) is dict and set(outer) == required_outer and (outer['schema'] == 1) and (outer['mode'] == 'native-http-fp8-sprint01-v1') and (outer['arm'] == logical) and (outer['port'] == {'control': 18087, 'mtp': 18088}[logical])):
        raise ValueError('HTTP binding template schema')
    sources = outer['binding_template'].get('sources')
    if not (type(sources) is dict and sources == outer['worker_template'].get('sources') and (outer['binding_template'].get('speculation') == outer['speculation'])):
        raise ValueError('HTTP private source map')
    helper_root = Path(__file__).resolve().parent
    required = ('mtp_contract', 'serving_contract', 'serving_client', 'serving_parity', 'worker_entry', 'serving_launcher', 'serving_entry', 'serving_runtime')
    instrumented = mapping_module.instrumentation_enabled()
    if type(instrumented) is not bool:
        raise ValueError('HTTP observation mode must be boolean')
    if not instrumented:
        required = tuple(name for name in required if name != 'serving_parity')
    if any((name in sys.modules for name in (*required, 'serving_parity'))):
        raise ValueError('HTTP private helper already imported')
    loaded = []
    try:
        for name in required:
            source = sources.get(name)
            path = helper_root / (name + '.py')
            if not (type(source) is dict and {'path', 'sha256'} <= set(source) and (source['path'] == str(path)) and valid_digest(source['sha256'])):
                raise ValueError('HTTP private helper pin: ' + name)
            raw = checked_file(path, source['sha256'], 2 << 20)
            private_spec = _verified_source_spec(name, path, raw)
            if private_spec is None or not isinstance(private_spec.loader, SourceFileLoader):
                raise ValueError('HTTP private helper loader: ' + name)
            module = importlib.util.module_from_spec(private_spec)
            sys.modules[name] = module
            loaded.append(name)
            private_spec.loader.exec_module(module)
        if sys.modules['mtp_contract'].validate_contract(outer['speculation']) != sys.modules['mtp_contract'].contract_for_arm(logical):
            raise ValueError('HTTP MTP speculation contract')
        if sys.modules['mtp_contract'].owner_arm_map() != mapping:
            raise ValueError('HTTP selected runtime owner mapping differs')
        if sys.modules['mtp_contract'].instrumentation_enabled() is not instrumented:
            raise ValueError('HTTP observation mode changed during private imports')
        return sys.modules['serving_runtime']
    except BaseException:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        raise

def valid_digest(value):
    return type(value) is str and len(value) == 64 and all((c in '0123456789abcdef' for c in value))

def independent_runtime(spec):
    """Authenticate the separate runner before any native model imports."""
    validate_spec(spec)
    row = spec['independent_reference']
    raw = checked_file(Path(__file__).resolve().with_name('INDEPENDENT-REFERENCE.json'), row['sha256'])

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate independent manifest key')
            result[key] = value
        return result
    manifest = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite independent manifest')))
    runner = manifest.get('runner')
    if type(runner) is not dict or set(runner) != {'path', 'sha256', 'bytes'} or runner['path'] != '/mnt/review/independent_runner.py' or (not valid_digest(runner['sha256'])) or (type(runner['bytes']) is not int) or (not 0 < runner['bytes'] <= 256 << 10):
        raise ValueError('independent runner source binding')
    if len(checked_file(Path(__file__).resolve().with_name('independent_runner.py'), runner['sha256'], 256 << 10)) != runner['bytes']:
        raise ValueError('independent runner source size')
    module = authenticated_adjacent('independent_runner.py', runner['sha256'])
    module.validate_manifest(manifest)
    candidate = manifest.get('torch_candidate')
    if candidate is not None and (row.get('torch_candidate_root') != candidate['root'] or row.get('torch_candidate_metadata') != candidate['metadata_directory']):
        raise ValueError('candidate Torch mount/manifest binding')
    if candidate is None and set(row) != {'path', 'sha256'}:
        raise ValueError('unexpected candidate Torch projection')
    sdk = candidate.get('sdk') if candidate is not None else None
    if (sdk is None) != ('runtime_overlay_root' not in row) or (sdk is not None and row['runtime_overlay_root'] != sdk['root']):
        raise ValueError('source SDK mount/manifest binding')
    return (module, manifest)

def checked_independent_result(spec, value, *, arm, session_sha256, spec_sha256):
    module, manifest = independent_runtime(spec)
    module.validate_result(value, manifest, arm=arm, session_sha256=session_sha256, spec_sha256=spec_sha256)
    return value

class _LazyAdjacent:
    """Load graph-shell projection and compatibility readers only on demand."""

    def __init__(self, filename, pin):
        self.filename, self.pin, self.module = (filename, pin, None)

    def __getattr__(self, name):
        if self.module is None:
            self.module = authenticated_adjacent(self.filename, self.pin)
        return getattr(self.module, name)

def stall_module(spec):
    config = spec.get('stall_trace')
    helper = authenticated_adjacent('stall_trace.py', config['helper_sha256'])
    helper.validate_config(config)
    return helper

def read_stall_trace_evidence(spec, directory, inner, require_complete=True):
    return stall_module(spec).read_evidence(spec['stall_trace'], Path(directory), inner, require_complete=require_complete)

def callback_module(spec, register=False):
    config = spec.get('callback_profile')
    if config is None:
        return None
    helper = authenticated_adjacent('callback_profile.py', config['helper_sha256'])
    helper.validate_config(config)
    if config['capture_helper_sha256'] != spec['validation_comparison']['capture']['helper_sha256']:
        raise ValueError('callback counter source differs from capture helper')
    if register:
        if helper.ALIAS in sys.modules:
            raise ValueError('callback helper alias already installed')
        sys.modules[helper.ALIAS] = helper
    return helper

def read_callback_profile_evidence(spec, directory, inner, require_complete=True):
    helper = callback_module(spec)
    reports = []
    for rank in (0, 1):
        raw = read_bounded_file(Path(directory) / f'rank-{rank}-callback-profile.json', 256 << 10)

        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError('duplicate callback profile JSON key')
                value[key] = item
            return value
        report = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite callback profile JSON')))
        rows = inner.get('ranks', [])
        row = rows[rank] if len(rows) == 2 and type(rows[rank]) is dict else {}
        helper.validate_report(report, spec['callback_profile'], inner['arm'], rank, inner['session_sha256'], inner['spec_sha256'], batches=row.get('batches'), capture_report=row.get('capture'), require_complete=require_complete)
        if require_complete and row.get('callback_profile') != report:
            raise ValueError('rank callback profile differs from durable report')
        if require_complete and row.get('callback_profile_identity') != dict(alias=helper.ALIAS, path='/mnt/review/callback_profile.py', helper_sha256=spec['callback_profile']['helper_sha256'], arm=inner['arm'], rank=rank, configured=True, instrumentation_only=True):
            raise ValueError('callback configured helper identity differs')
        reports.append(report)
    return reports

def comparison():
    return authenticated_adjacent('validation_comparison.py', COMPARISON_SHA256)

def comparison_mode(spec):
    return spec.get('experiment_mode') in ('v4-repeatability-control-v1', 'v4-first-divergence-capture-v1', 'native-repeatability-capture-v1', 'native-decode-layer-capture-v1', 'native-prefill-row-audit-v1', 'native-async-prefill-layer-capture-v1')

def capture_mode(spec):
    return False

def native_capture_mode(spec):
    return spec.get('experiment_mode') == 'native-model-context-retrieval-32k-v1'

def capture_module(spec):
    comparison().validate_spec(spec)
    return authenticated_adjacent('divergence_capture.py', spec['validation_comparison']['capture']['helper_sha256'])

def row_audit_module(spec):
    comparison().validate_spec(spec)
    if spec.get('experiment_mode') != 'native-prefill-row-audit-v1':
        raise ValueError('prefill row audit mode required')
    return authenticated_adjacent('prefill_row_audit.py', spec['validation_comparison']['capture']['prefill_row_audit']['helper_sha256'])

def capture_binding(spec, native):
    return dict(schema=1, mode=spec['experiment_mode'], arm=native['arm'], rank=native['rank'], session_sha256=native['session_sha256'], spec_sha256=native['spec_sha256'], worker=native['process'], source_contract_sha256=hashlib.sha256(comparison().canonical(spec['validation_comparison']['capture']).encode()).hexdigest())


PROGRESS_LIMIT = 64 << 10
PROGRESS_PHASES = ('rank_started', 'preflight_complete', 'torch_import', 'native_identity_complete', 'serving_imports', 'arguments', 'model_config', 'parallel_runtime', 'layer_runtime', 'model_runner', 'memory_pool', 'attention_backends', 'graphs', 'cache', 'ready', 'batch_intent', 'batch_complete', 'cleanup', 'rank_finished')

def progress_readout(raw, *, rank, arm, session_sha256, spec_sha256):
    """Retain validated prefix counts; absence or a torn tail never means zero work."""
    summary = dict(rank=rank, valid=False, counts_complete=False, finished=False, intent_batches=None, completed_batches=None, intent_requests=None, completed_requests=None, last_phase=None, pid=None, error=None, counts_semantics='validated-prefix-lower-bounds', qualification=False)
    if raw is None:
        summary['error'] = 'progress unavailable'
        return summary
    intents = completed = 0
    plans = list(request_plan({}))
    previous_time = -1.0
    ready = finished = False
    startup = [phase for phase in PROGRESS_PHASES[:PROGRESS_PHASES.index('ready') + 1] if phase != 'native_identity_complete' or session_sha256 is not None]
    startup_index = 0
    cleaning = False
    try:
        if not raw or len(raw) > PROGRESS_LIMIT:
            raise ValueError('progress empty/oversized')
        for sequence, line in enumerate(raw.splitlines(keepends=True)):
            if not line.endswith(b'\n'):
                raise ValueError('partial progress line')

            def pairs(items):
                value = {}
                for key, item in items:
                    if key in value:
                        raise ValueError('duplicate progress key')
                    value[key] = item
                return value
            row = json.loads(line, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite progress')))
            expected = dict(schema=1, rank=rank, arm=arm, session_sha256=session_sha256, spec_sha256=spec_sha256, sequence=sequence)
            if type(row) is not dict or set(row) != set(expected) | {'pid', 'phase', 'utc_unix', 'monotonic', 'details'} or any((type(row.get(key)) is not type(value) or row[key] != value for key, value in expected.items())) or (type(row['pid']) is not int) or (row['pid'] <= 0) or (sequence and row['pid'] != summary['pid']) or (row['phase'] not in PROGRESS_PHASES) or (type(row['details']) is not dict) or any((type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0 for key in ('utc_unix', 'monotonic'))) or (row['monotonic'] < previous_time) or finished:
                raise ValueError('progress identity/order/type')
            phase, details = (row['phase'], row['details'])
            if (sequence == 0) != (phase == 'rank_started'):
                raise ValueError('progress start order')
            if cleaning and phase != 'rank_finished':
                raise ValueError('progress after cleanup')
            if phase in startup:
                if startup_index >= len(startup) or phase != startup[startup_index] or intents:
                    raise ValueError('progress startup order')
                startup_index += 1
            elif phase == 'native_identity_complete':
                raise ValueError('unexpected native identity phase')
            if phase == 'cleanup':
                cleaning = True
            if phase == 'ready':
                if ready or intents:
                    raise ValueError('duplicate/late readiness')
                ready = True
            if phase in ('batch_intent', 'batch_complete'):
                ordinal = intents if phase == 'batch_intent' else completed
                if not ready or ordinal >= len(plans) or json.dumps(details, sort_keys=True) != json.dumps(dict(plans[ordinal], ordinal=ordinal), sort_keys=True) or (phase == 'batch_intent' and intents != completed) or (phase == 'batch_complete' and intents != completed + 1):
                    raise ValueError('progress batch binding/order')
                if phase == 'batch_intent':
                    intents += 1
                else:
                    completed += 1
            elif details and phase != 'rank_finished':
                raise ValueError('unexpected phase details')
            if phase == 'rank_finished':
                if not cleaning or set(details) != {'success'} or type(details['success']) is not bool:
                    raise ValueError('progress finish record')
                if details['success'] and (intents != 7 or completed != 7):
                    raise ValueError('premature successful finish')
                finished = True
                summary['finished'] = details['success']
            previous_time = row['monotonic']
            summary.update(last_phase=phase, pid=row['pid'], intent_batches=intents, completed_batches=completed, intent_requests=sum((p['concurrency'] for p in plans[:intents])), completed_requests=sum((p['concurrency'] for p in plans[:completed])))
        summary.update(valid=True, counts_complete=finished)
    except (ValueError, TypeError, KeyError) as error:
        summary['error'] = type(error).__name__ + ': ' + str(error)[:256]
    return summary

def read_progress(output, *, rank, arm, session_sha256, spec_sha256):
    try:
        raw = read_bounded_file(output / f'rank-{rank}-progress.jsonl', PROGRESS_LIMIT)
    except FileNotFoundError:
        raw = None
    except (OSError, ValueError) as error:
        result = progress_readout(None, rank=rank, arm=arm, session_sha256=session_sha256, spec_sha256=spec_sha256)
        result['error'] = type(error).__name__ + ': ' + str(error)[:256]
        return result
    return progress_readout(raw, rank=rank, arm=arm, session_sha256=session_sha256, spec_sha256=spec_sha256)

def validate_native_process(record):
    if type(record) is not dict or set(record) != {'pid', 'start_ticks', 'uid', 'gid', 'boot_id', 'pid_namespace'} or any((type(record[key]) is not int or record[key] <= 0 for key in ('pid', 'start_ticks'))) or (type(record['uid']) is not int) or (type(record['gid']) is not int) or ((record['uid'], record['gid']) != (1000, 1000)) or (type(record['boot_id']) is not str) or (re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', record['boot_id']) is None) or (type(record['pid_namespace']) is not str) or (re.fullmatch('pid:\\[[1-9][0-9]*\\]', record['pid_namespace']) is None):
        raise ValueError('native worker process binding')

def validate_physical_details(helper, devices, proof):
    if type(devices) is not list or len(devices) != 2 or any((type(row) is not dict or set(row) != {'index', 'bdf', 'uuid', 'architecture', 'vram_bytes'} for row in devices)) or (type(proof) is not dict) or (set(proof) != {'library', 'sha256', 'bytes', 'loaded_inode', 'loaded_device', 'no_load_handle', 'symbol', 'header_sha256'}):
        raise ValueError('exact native device/proof observations required')
    helper.validate_devices(devices, PHYSICAL_DEVICES)
    helper.validate_proof(proof)

def validate_rank_identity(record, *, rank, arm, session_sha256, spec_sha256):
    expected = {'schema': 1, 'scope': 'benchmark-native-preweight-identity-v1', 'rank': rank, 'gpu_id': rank, 'arm': arm, 'session_sha256': session_sha256, 'spec_sha256': spec_sha256, 'helper_sha256': GPU_IDENTITY_SHA256, 'before_model_config': True, 'before_model_runner': True, 'qualification': False}
    if type(record) is not dict or set(record) != set(expected) | {'devices', 'hip', 'process'} or any((type(record.get(key)) is not type(value) or record.get(key) != value for key, value in expected.items())):
        raise ValueError('native identity record binding')
    validate_native_process(record['process'])
    helper = authenticated_adjacent('benchmark_gpu_identity.py', GPU_IDENTITY_SHA256)
    validate_physical_details(helper, record['devices'], record['hip'])
    return record

def write_result(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b'\n'
    if len(raw) > 8 << 20:
        raise ValueError('result bound')
    with open(path, 'xb') as stream:
        stream.write(raw)

def validate_spec(spec):
    if type(spec) is dict and spec.get('experiment_mode') == 'native-http-fp8-sprint01-v1':
        http = spec.get('http_serving')
        fields = {'schema', 'mode', 'bindings', 'client_seconds', 'expected_devices'}
        if type(http) is dict and 'artifact_limits' in http:
            primary_limits = http['artifact_limits']
            if not (type(primary_limits) is dict and len(primary_limits) == 16
                    and all(type(value) is int and 0 < value < 2**63 for value in primary_limits.values())):
                raise ValueError('bounded HTTP artifact primaries required')
            fields.add('artifact_limits')
        worker = spec.get('worker_seconds')
        client = http.get('client_seconds') if type(http) is dict else None
        if not (type(worker) is int and type(client) is int
                and 1 <= client <= 2040 and client + 240 <= worker <= 2280):
            raise ValueError('bounded HTTP worker/client timing required')
        if type(http) is dict and 'diagnostic_environment' in http:
            path = Path(__file__).resolve().with_name('mtp_contract.py')
            module_spec = importlib.util.spec_from_file_location('frozen_http_phase_contract', path)
            phase_contract = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(phase_contract)
            if phase_contract.settings()['phase'] not in phase_contract.FAIL_FAST_PHASES and (not phase_contract.precision_reference_enabled()) or http['diagnostic_environment'] != {arm: phase_contract.expected_environment(arm) for arm in ('control', 'mtp')}:
                raise ValueError('native diagnostic environment binding')
            fields.add('diagnostic_environment')
        if not (type(http) is dict and set(http) == fields and (http['schema'] == 1) and (http['mode'] == 'native-http-fp8-sprint01-v1') and (type(http['client_seconds']) is int) and (1 <= http['client_seconds'] <= 2040) and (type(http['bindings']) is dict) and (set(http['bindings']) == {'control', 'mtp'}) and all((type(row) is dict and set(row) == {'path', 'sha256'} and (type(row['path']) is str) and (row['path'] == 'review/HTTP-BINDING-' + arm + '.json') and valid_digest(row['sha256']) for arm, row in http['bindings'].items())) and (type(http['expected_devices']) is dict) and (len(http['expected_devices']) == 2) and all((type(bdf) is str and type(row) is dict and (set(row) == {'uuid', 'render'}) and (type(row['uuid']) is str) and (type(row['render']) is str) for bdf, row in http['expected_devices'].items()))):
            raise ValueError('native HTTP serving specification')
        return spec
    raise ValueError('only the W5 native HTTP serving specification is supported')

def request_plan(spec):
    """Exactly seven cold singleton cases per arm, with no workload warmups."""
    for cell, concurrency, length in CELLS:
        for sample in range(1):
            yield {'cell': cell, 'concurrency': concurrency, 'input_tokens': length, 'sample': sample, 'measured': True, 'request_ids': [f'{cell}-{sample}-{i}' for i in range(concurrency)]}

def validate_cache_profiles(records, arm, contract, spec_sha256, comparison_contract=None):
    if contract is None:
        if any((row.get('cache_profile') is not None for row in records)):
            raise ValueError('unexpected cache profile without spec binding')
        return
    if json.dumps(contract, sort_keys=True) != json.dumps(CACHE_PROFILE, sort_keys=True):
        raise ValueError('cache profile contract differs')
    if comparison_contract is not None:
        comparison().validate_contract(comparison_contract)
    if arm == 'bf16' and comparison_contract is None:
        if any((row.get('cache_profile') is not None for row in records)):
            raise ValueError('baseline must use ordinary unprofiled BF16 backend')
        return
    module = authenticated_adjacent('cache_profile.py', CACHE_PROFILE_SHA256)
    for row in records:
        native = row['native_identity']
        if native['spec_sha256'] != spec_sha256 or native['rank'] != row['rank'] or native['arm'] != arm:
            raise ValueError('cache profile/native identity binding')
        module.validate_report(row.get('cache_profile'), row['rank'], row['batches'], native['session_sha256'], spec_sha256, native['process']['pid'])

def summarize(records, expected_arm, profile_contract=None, profile_spec_sha256=None, comparison_contract=None):
    if expected_arm not in ('bf16', 'kv4'):
        raise ValueError('explicit expected arm required')
    if len(records) != 2 or any((row is None or row['failure'] is not None or row['cleanup_errors'] or (not row['owner_cleanup']) or (row['intent_batches'] != 7) or (row['completed_batches'] != 7) or (row['intent_requests'] != 7) or (row['completed_requests'] != 7) or (row['arm'] != expected_arm) or (len(row['batches']) != 7) for row in records)):
        raise ValueError('incomplete arm')
    if [record['rank'] for record in records] != [0, 1]:
        raise ValueError('exact ordered rank identities required')
    validate_cache_profiles(records, expected_arm, profile_contract, profile_spec_sha256, comparison_contract)
    if comparison_contract is not None:
        module = comparison()
        module.validate_contract(comparison_contract)
        telemetry = authenticated_adjacent('batch_telemetry.py', TELEMETRY_SHA256)
        for row in records:
            native = row['native_identity']
            if module.native_mode(comparison_contract['mode']):
                if row.get('actual_pool') != 'MHATokenToKVPool':
                    raise ValueError('native comparison requires ordinary BF16 pool')
                module.validate_native_cache_identity(row.get('native_cache_identity'), expected_arm, row['rank'], native['session_sha256'], profile_spec_sha256, native['process']['pid'], comparison_contract)
            elif row.get('actual_pool') != 'QwenBF16MixedPool':
                raise ValueError('both comparison arms require the compact hybrid pool')
            module.validate_identity(row.get('validation_identity'), expected_arm, row['rank'], native['session_sha256'], profile_spec_sha256, native['process']['pid'], comparison_contract)
            telemetry.validate_report(row.get('batch_telemetry'), native, expected_arm, row['rank'])
    samples = []
    for index, plan in enumerate(request_plan({})):
        a, b = (record['batches'][index] for record in records)
        if any((row[key] != plan[key] for row in (a, b) for key in plan)) or not a['complete'] or (not b['complete']) or (a['generated_ids'] != b['generated_ids']) or any((len(row['step_seconds']) != 32 for row in (a, b))):
            raise ValueError('rank schedule/output mismatch')
        for row in (a, b):
            if not all((type(x) in (int, float) and math.isfinite(x) and (x >= 0) for x in row['step_seconds'])) or not math.isfinite(row['batch_wall_seconds']) or row['batch_wall_seconds'] < 0 or (len(row['generated_ids']) != plan['concurrency']) or (len(row['generated_logprobs']) != plan['concurrency']) or any((len(ids) != 32 or any((type(x) is not int or x < 0 for x in ids)) for ids in row['generated_ids'])) or any((len(scores) != 32 or not all((math.isfinite(x) for x in scores)) for scores in row['generated_logprobs'])):
                raise ValueError('malformed rank measurements')
        steps = [max(x, y) for x, y in zip(a['step_seconds'], b['step_seconds'])]
        if not all((math.isfinite(x) and x >= 0 for x in steps)):
            raise ValueError('invalid timing')
        samples.append({**plan, 'prefill_rank_max_seconds': steps[0], 'decode_rank_max_step_seconds': steps[1:], 'decode_31_seconds': sum(steps[1:]), 'forward_sample_total_seconds': sum(steps), 'batch_wall_rank_max_seconds': max(a['batch_wall_seconds'], b['batch_wall_seconds'])})
    return samples
def prepare_serving_scratch_directories():
    """Create only the three reviewed directories in the already bounded tmpfs."""
    expected = {'HOME': '/mnt/output/home', 'TMPDIR': '/mnt/output/tmp', 'XDG_CACHE_HOME': '/mnt/output/cache'}
    if any((os.environ.get(key) != value for key, value in expected.items())):
        raise ValueError('reviewed serving scratch environment differs')
    parent = Path('/mnt/output')
    if parent.resolve() != parent or not parent.is_dir():
        raise ValueError('canonical bounded output tmpfs required')
    for value in expected.values():
        path = Path(value)
        path.mkdir(mode=493)
        if path.resolve() != path or path.stat().st_uid != os.getuid():
            raise ValueError('owned serving scratch directory required')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed-benchmark', action='store_true')
    parser.add_argument('--execute-serving-screen', action='store_true')
    parser.add_argument('--spec')
    parser.add_argument('--spec-sha256')
    parser.add_argument('--output')
    parser.add_argument('--arm', choices=('bf16', 'kv4'))
    parser.add_argument('--physical-artifacts', action='store_true')
    parser.add_argument('--session-sha256')
    parser.add_argument('--artifacts-sha256')
    args = parser.parse_args()
    if not args.execute_reviewed_benchmark:
        raise SystemExit('NOT_RUN: --execute-reviewed-benchmark is required')
    if args.arm is None:
        raise ValueError('explicit arm required')
    if args.physical_artifacts:
        if not valid_digest(args.session_sha256) or not valid_digest(args.artifacts_sha256):
            raise ValueError('physical mode requires independently pinned session/artifact identities')
    elif args.session_sha256 is not None or args.artifacts_sha256 is not None:
        raise ValueError('physical artifact flags must be supplied together')
    spec = json.loads(checked_file(args.spec, args.spec_sha256))
    validate_spec(spec)
    worker_seconds = spec['worker_seconds']
    artifact_values = spec['http_serving'].get('artifact_limits')
    artifact_values = None if artifact_values is None else dict(artifact_values)
    if args.execute_serving_screen:
        raise ValueError('HTTP screen uses the canonical reviewed-benchmark entry flag')
    if not args.physical_artifacts:
        raise ValueError('native HTTP screen requires physical artifact identity')
    artifacts = authenticated_adjacent('benchmark_artifacts.py', args.artifacts_sha256)
    artifact_options = {}
    if artifact_values is not None:
        factory = getattr(artifacts, 'HTTPArtifactLimits', None)
        if factory is None:
            raise ValueError('configured artifact helper required')
        plan = factory.from_limits(artifact_values)
        if set(artifact_values) != set(plan.__dataclass_fields__):
            raise ValueError('exact HTTP artifact primaries required')
        artifact_options['limits'] = plan
    prepare_serving_scratch_directories()
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError('fresh absolute owned output required')
    output.mkdir(mode=448)
    serving_runtime = http_runtime_module(spec, args.arm)
    binding = {'session_sha256': args.session_sha256, 'arm': args.arm, 'spec_sha256': args.spec_sha256, 'experiment_mode': 'native-http-fp8-sprint01-v1'}
    raw_transport = serving_runtime.mtp_contract.raw_attention_capture_enabled()
    if raw_transport and artifact_options:
        raise ValueError('configured artifact limits do not support raw-frame transport')
    if raw_transport:
        binding['artifact_transport'] = artifacts.RAW_TRANSPORT
    result = None
    try:
        result = serving_runtime.execute(args, spec, output,
            deadline=time.monotonic() + worker_seconds, artifacts=artifacts)
    finally:
        if (output / 'result.json').is_file():
            artifacts.pack_http_artifacts(output, binding, **artifact_options)
            if raw_transport:
                sys.stdout.flush()
                artifacts.export_raw_frames(output, binding=binding, sink=sys.stdout.buffer)
            else:
                staged = artifacts.stage_artifacts(output, binding, **artifact_options)
                artifacts.export_bundle(staged, binding=binding, sink=sys.stdout.buffer, **artifact_options)
    raise SystemExit(0 if type(result) is dict and result.get('status') == 'HTTP_ARM_COMPLETE' else 1)

_GRAPH_OUTER = _LazyAdjacent('graph_outer.py', 'f797dad9ec5742d2056df358a33e42f062775f52a510a7bd2600d80fd725ae70')

if __name__ == '__main__':
    main()
