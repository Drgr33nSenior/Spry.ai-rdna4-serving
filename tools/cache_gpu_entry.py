"""Explicit synthetic cache ABI experiment; no model or serving runtime import."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


MODE = 'native-cache-regression-v1'
COUNTERBALANCED_MODE = 'native-cache-counterbalanced-v2'
PHASE_PROFILE_MODE = 'native-cache-phase-profile-v1'
BALANCED_MODES = (COUNTERBALANCED_MODE, PHASE_PROFILE_MODE)
PROFILE_SOURCE_SHA256 = '2206e70d233cacd86bea286be36c04a42d504e5807fe05985b2b2fa3319db88d'
UNPROFILED_CANDIDATE_SHA256 = '5eea8858d524d3d532d80ea8da582d6955e2cfb855cbf444c34491e820a7db3b'
TIMING_PROTOCOL = 'warm-synchronized-output-v2'
PREFIX = 'SPRY_CACHE_V1 '
DEVICES = {'0000:c3:00.0': '39373864-6465-3038-6163-303837666533',
           '0000:23:00.0': '34663466-6561-3362-3832-346261366439'}
SPEC_FIELDS = {'schema', 'experiment_mode', 'session_sha256', 'native_files', 'devices',
               'timing_repeats', 'native_processes', 'model_starts', 'arm_order',
               'entry_sha256', 'reader_sha256', 'owner_sha256', 'artifacts_sha256'}
COUNTERBALANCED_FIELDS = {'timing_protocol', 'timing_warmups', 'variant_order'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read(path, maximum, expected=None):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'canonical source/output required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        import stat
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
             'bounded regular singly linked file required')
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(fd, min(1 << 20, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')
        stamp = lambda st: tuple(getattr(st, key) for key in fields)
        need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
             'file changed during read')
        need(expected is None or sha(raw) == expected, 'source/output SHA256 differs')
        return bytes(raw)
    finally:
        os.close(fd)


def adjacent(filename, expected):
    path = Path(__file__).resolve().with_name(filename)
    raw = read(path, 256 << 10, expected)
    name = 'spry_cache_' + filename.replace('.', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_spec(value):
    need(type(value) is dict, 'exact native cache contract')
    balanced = value.get('experiment_mode') in BALANCED_MODES
    profiling = value.get('experiment_mode') == PHASE_PROFILE_MODE
    need(set(value) == SPEC_FIELDS | (COUNTERBALANCED_FIELDS if balanced else set()) | ({'phase_profile'} if profiling else set()) and value['schema'] == 1
         and value['experiment_mode'] in (MODE, *BALANCED_MODES) and value['arm_order'] == ['bf16', 'kv4']
         and type(value['timing_repeats']) is int and value['timing_repeats'] == 5
         and type(value['native_processes']) is int and value['native_processes'] == (16 if balanced else 8)
         and type(value['model_starts']) is int and value['model_starts'] == 0, 'exact native cache contract')
    if balanced:
        need(value['timing_protocol'] == TIMING_PROTOCOL and type(value['timing_warmups']) is int
             and value['timing_warmups'] == 2 and value['variant_order'] ==
             ['baseline', 'candidate', 'candidate', 'baseline'], 'exact counterbalanced timing contract')
    for key in ('session_sha256', 'entry_sha256', 'reader_sha256', 'owner_sha256', 'artifacts_sha256'):
        need(digest(value[key]), 'source/session SHA256 required')
    need(type(value['native_files']) is dict and set(value['native_files']) ==
         {'fixture', 'baseline.so', 'candidate.so'} and all(digest(pin) for pin in value['native_files'].values()),
         'exact old/candidate/fixture inventory')
    if profiling:
        profile = value['phase_profile']
        fixed = {'schema': 1, 'variant': 'candidate', 'boundary': 'host-wall-three-phase-v1',
                 'source_sha256': PROFILE_SOURCE_SHA256,
                 'unprofiled_candidate_sha256': UNPROFILED_CANDIDATE_SHA256}
        identities = {'projected_source_sha256', 'projector_sha256', 'profile_header_sha256'}
        need(type(profile) is dict and set(profile) == set(fixed) | identities and
             all(type(profile[key]) is type(item) and profile[key] == item for key, item in fixed.items()) and
             all(digest(profile[key]) for key in identities) and
             profile['projected_source_sha256'] != PROFILE_SOURCE_SHA256 and
             value['native_files']['candidate.so'] != UNPROFILED_CANDIDATE_SHA256,
             'explicit candidate-only phase projection binding required')
    devices = value['devices']
    need(type(devices) is list and len(devices) == 2 and all(type(row) is dict and
         set(row) == {'device', 'bdf', 'uuid'} and type(row['device']) is int for row in devices)
         and [row['device'] for row in devices] == [0, 1]
         and {row['bdf']: row['uuid'] for row in devices} == DEVICES, 'two native device bindings')
    return value


def native_argv(spec, arm, device, variant, output):
    validate_spec(spec)
    need(arm in spec['arm_order'] and device in spec['devices'] and variant in ('baseline', 'candidate'),
         'native invocation binding')
    argv = ['/mnt/native/fixture', '--execute-gpu', '--library', '/mnt/native/' + variant + '.so',
            '--output', str(output), '--device', str(device['device']), '--expected-bdf', device['bdf'],
            '--expected-uuid', device['uuid'],
            '--cache-mode', arm, '--timing-repeats', str(spec['timing_repeats'])]
    if spec['experiment_mode'] in BALANCED_MODES:
        argv.extend(('--timing-protocol', spec['timing_protocol'], '--timing-warmups', str(spec['timing_warmups'])))
    if profiled_variant(spec, variant):
        argv.append('--phase-profile')
    return argv


def profiled_variant(spec, variant):
    need(variant in ('baseline', 'candidate'), 'profile variant binding')
    return spec['experiment_mode'] == PHASE_PROFILE_MODE and variant == 'candidate'


def reader_options(spec, variant=None):
    options = {'expected_timing_repeats': spec['timing_repeats']}
    if spec['experiment_mode'] in BALANCED_MODES:
        options.update(expected_timing_protocol=spec['timing_protocol'], expected_timing_warmups=spec['timing_warmups'])
    if variant is not None:
        options['expected_phase_profile'] = profiled_variant(spec, variant)
    return options


def validate_native(value, spec, arm, device, variant=None):
    need(type(value) is dict and value.get('cache_mode') == arm and value.get('device') == device['device']
         and value.get('bdf') == device['bdf'] and value.get('uuid') == device['uuid']
         and value.get('visible_devices') == 2 and value.get('architecture', '').split(':')[0] == 'gfx1201',
         'fresh native allocation identity differs')
    need(value.get('timing_repeats') == spec['timing_repeats'], 'timing repetition binding')
    if spec['experiment_mode'] in BALANCED_MODES:
        need(value.get('timing_protocol') == spec['timing_protocol'] and
             type(value.get('timing_warmups')) is int and value['timing_warmups'] == spec['timing_warmups'],
             'explicit timing protocol binding')
    profiling = spec['experiment_mode'] == PHASE_PROFILE_MODE
    need(not profiling or variant in ('baseline', 'candidate'), 'profile result variant required')
    phase = profiling and profiled_variant(spec, variant)
    need(value.get('phase_profile', False) is phase, 'native profile selection differs')
    timings = value.get('timings')
    expected = [(length, i) for length in (128, 1024, 3966) for i in range(spec['timing_repeats'])]
    need(type(timings) is list and len(timings) == len(expected), 'complete bounded timing records')
    for row, (length, i) in zip(timings, expected):
        need(type(row) is dict and set(row) == ({'protected', 'length', 'repetition', 'synchronized_call_ms'} | ({'phase_wall'} if phase else set()))
             and row['protected'] is (arm == 'bf16') and row['length'] == length and row['repetition'] == i
             and type(row['synchronized_call_ms']) in (int, float) and math.isfinite(row['synchronized_call_ms'])
             and 0 < row['synchronized_call_ms'] < 300000, 'malformed native timing')
    return value


def run(spec, spec_sha256, arm, *, clock=time.monotonic):
    validate_spec(spec)
    need(arm in spec['arm_order'] and sha(canonical(spec)) == spec_sha256, 'spec/mode binding')
    need((os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (1000, 1000, 1000, 1000), 'UID/GID1000 required')
    need(not any(name in os.environ for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
         'CUDA_VISIBLE_DEVICES', 'HSA_OVERRIDE_GFX_VERSION')), 'device/architecture override refused')
    read(Path(__file__).resolve(), 256 << 10, spec['entry_sha256'])
    owner = adjacent('host_process_owner.py', spec['owner_sha256'])
    artifacts = adjacent('benchmark_artifacts.py', spec['artifacts_sha256'])
    reader = adjacent('cache_gpu_fixture.py', spec['reader_sha256'])
    for name, pin in spec['native_files'].items():
        read('/mnt/native/' + name, 16 << 20, pin)
    output = Path('/mnt/output/cache')
    output.mkdir(mode=0o700)
    balanced = spec['experiment_mode'] in BALANCED_MODES
    variants = spec['variant_order'] if balanced else ('baseline', 'candidate')
    result = {'schema': 1, 'experiment_mode': spec['experiment_mode'], 'arm': arm, 'spec_sha256': spec_sha256,
        'session_sha256': spec['session_sha256'], 'qualification': False, 'model_starts': 0,
        'status': 'INCOMPLETE', 'processes': [], 'comparisons': [], 'timings_usable': False}
    try:
        for device in spec['devices']:
            reference = None
            reference_path = None
            for ordinal, variant in enumerate(variants):
                name = str(device['device']) + '-' + (str(ordinal) + '-' if balanced else '') + variant
                path = output / (name + '.bin')
                # Fresh isolated Python process verifies sources, then execs the
                # native executable once. HIP is never loaded in this collector.
                argv = [sys.executable, '-I', '-B', '-S', str(Path(__file__).resolve()), '--native-child',
                        '--spec-sha256', spec_sha256, '--arm', arm, '--device', str(device['device']),
                        '--variant', variant, '--output', str(path)]
                if balanced:
                    argv.extend(('--ordinal', str(ordinal)))
                cap = artifacts.CapturePipes(output / (name + '-capture'), stream_limit=64 << 10)
                row = {'device': device['device'], 'variant': variant, 'argv': argv,
                       'intent_monotonic': clock(), 'exit_code': None}
                if balanced:
                    row['ordinal'] = ordinal
                result['processes'].append(row)
                started = clock()
                supervised = None
                try:
                    supervised = owner.supervise(argv, script_path=str(Path(__file__).resolve()),
                        script_sha256=spec['entry_sha256'], deadlines=owner.Deadlines(
                            started + 95, started + 100, started + 103, started + 105),
                        on_child=lambda identity: row.update(process=asdict(identity)),
                        stdout=cap.stdout, stderr=cap.stderr, resource_check=cap.check)
                    row['supervision'] = asdict(supervised)
                    row['exit_code'] = supervised.exit_code
                finally:
                    try:
                        row['capture'] = cap.finish()
                    except BaseException as error:
                        row['capture'] = {'capture_complete': False, 'failure': type(error).__name__ + ': ' + str(error)[:512]}
                need(supervised is not None, 'native supervisor returned no owned-process result')
                raw = read(output / (name + '-capture/stdout.raw'), 64 << 10)
                row['stdout'] = raw.decode('utf-8', errors='replace')
                row['stderr'] = read(output / (name + '-capture/stderr.raw'), 64 << 10).decode('utf-8', errors='replace')
                need(supervised.process_cleanup_observed and supervised.exit_code == 0 and
                     supervised.failure is None and supervised.owner_signal is None and
                     row['capture']['capture_complete'] is True, 'native process or cleanup failed')
                summary = decode(raw)
                clean = validate_native(summary, spec, arm, device, variant)
                binary = read(path, 32 << 20)
                row.update(native=summary, output_sha256=sha(binary), output_bytes=len(binary))
                need(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= 64 << 20,
                     'native per-arm output bound')
                if reference is None:
                    reference, reference_path = (binary, clean), path
                else:
                    options = reader_options(spec)
                    if spec['experiment_mode'] == PHASE_PROFILE_MODE:
                        options.update(expected_old_phase_profile=False,
                                       expected_candidate_phase_profile=profiled_variant(spec, variant))
                    comparison = reader.compare(reference[0], binary, reference[1], clean, **options)
                    if balanced:
                        comparison.update(reference_ordinal=0, compared_ordinal=ordinal)
                    result['comparisons'].append(comparison)
                    # Remove only a successfully compared private tmpfs buffer.
                    if balanced:
                        path.unlink()
            # Keep the first occurrence until every later occurrence matches it.
            reference_path.unlink()
            if not balanced:
                path.unlink()
        result['status'] = 'NATIVE_CACHE_ARM_COMPLETE'
        result['timings_usable'] = True
    except BaseException as error:
        result['failure'] = type(error).__name__ + ': ' + str(error)[:1024]
    print(PREFIX + canonical(result).decode().rstrip(), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-native-cache', action='store_true')
    parser.add_argument('--native-child', action='store_true')
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--arm', required=True, choices=('bf16', 'kv4'))
    parser.add_argument('--device', type=int)
    parser.add_argument('--variant', choices=('baseline', 'candidate'))
    parser.add_argument('--output')
    parser.add_argument('--ordinal', type=int)
    args = parser.parse_args(argv)
    need(args.execute_native_cache != args.native_child, 'exact native execution route required')
    spec = validate_spec(decode(read('/mnt/review/NATIVE-SPEC.json', 64 << 10, args.spec_sha256)))
    if args.native_child:
        balanced = spec['experiment_mode'] in BALANCED_MODES
        need(args.device in (0, 1) and args.variant is not None, 'native child device/variant binding')
        need((balanced and args.ordinal in range(4) and spec['variant_order'][args.ordinal] == args.variant)
             or (not balanced and args.ordinal is None), 'native child ordinal binding')
        name = str(args.device) + '-' + (str(args.ordinal) + '-' if balanced else '') + args.variant
        need(args.output == '/mnt/output/cache/' + name + '.bin', 'native child output binding')
        for name, pin in spec['native_files'].items():
            read('/mnt/native/' + name, 16 << 20, pin)
        argv = native_argv(spec, args.arm, spec['devices'][args.device], args.variant, args.output)
        os.execv(argv[0], argv)
    return 0 if run(spec, args.spec_sha256, args.arm)['status'] == 'NATIVE_CACHE_ARM_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
