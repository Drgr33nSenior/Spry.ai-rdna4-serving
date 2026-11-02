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
GPU_IDENTITY_SHA256 = '7324842c4f779749d561dc586168bd923e84e6900d17e3d22c140647bcd133df'
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



class _LazyAdjacent:
    """Load graph-shell projection and compatibility readers only on demand."""

    def __init__(self, filename, pin):
        self.filename, self.pin, self.module = (filename, pin, None)

    def __getattr__(self, name):
        if self.module is None:
            self.module = authenticated_adjacent(self.filename, self.pin)
        return getattr(self.module, name)













PROGRESS_LIMIT = 64 << 10
PROGRESS_PHASES = ('rank_started', 'preflight_complete', 'torch_import', 'native_identity_complete', 'serving_imports', 'arguments', 'model_config', 'parallel_runtime', 'layer_runtime', 'model_runner', 'memory_pool', 'attention_backends', 'graphs', 'cache', 'ready', 'batch_intent', 'batch_complete', 'cleanup', 'rank_finished')



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


def validate_spec(spec):
    if type(spec) is not dict or spec.get('experiment_mode') != 'native-http-fp8-sprint01-v1':
        raise ValueError('only the W5 native HTTP serving specification is supported')
    if any(spec.get(key) is not None for key in ('independent_reference','validation_comparison','cache_profile','callback_profile','stall_trace')):
        raise ValueError('retired experiment fields are not admitted in the HTTP specification')

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
    serving_runtime.mtp_contract.settings()
    raw_transport = False
    result = None
    try:
        result = serving_runtime.execute(args, spec, output, deadline=time.monotonic() + worker_seconds, artifacts=artifacts)
    finally:
        if (output / 'result.json').is_file():
            artifacts.pack_http_artifacts(output, binding, **artifact_options)
            staged = artifacts.stage_artifacts(output, binding, **artifact_options)
            artifacts.export_bundle(staged, binding=binding, sink=sys.stdout.buffer, **artifact_options)
    raise SystemExit(0 if type(result) is dict and result.get('status') == 'HTTP_ARM_COMPLETE' else 1)

_GRAPH_OUTER = _LazyAdjacent('graph_outer.py', '9902d9103a0696edeba14fcfed5ce5c9047252c4223a2814f49f69583e288d74')

if __name__ == '__main__':
    main()


HTTP_REFERENCE = {'schema': 1, 'mode': 'ordered-score-v4-model-v1', 'entry': 'uq_bf16_attention_scores_v3', 'workspace_limit_bytes': 268435456, 'binary_sha256': '1ca7efb0cf9220adb16a56e30e4cd2931b5d49edbe3277fc15d68e629500cd7b', 'device_sha256': '48d2c77e69df1e8b352fbafbf2f1ff832d8cc6717b1a1ffc47970a4f59ee64df', 'source_sha256': '661441715312cba400f04bc0168818092d031fc6799affbcbe78fa0e8fa2557b', 'reader_events': False}
