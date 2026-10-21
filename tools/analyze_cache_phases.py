"""Candidate host-wall phases only; archive authentication is a prerequisite.

The baseline is uninstrumented. Do not use these records for an uninstrumented
speed comparison, isolated decoder/kernel timing, or model throughput.
"""
import importlib.util
from pathlib import Path
import statistics

PHASES = ('host_checks_ns', 'validation_status_ns', 'reader_sync_ns')


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(result, spec, *, controller=None, reader=None):
    root = Path(__file__).resolve().parents[1]
    if controller is None:
        controller = _module(root / 'tools/cache_gpu_controller.py', 'phase_readout_controller')
    if reader is None:
        reader = _module(root / 'tests/test_cache_gpu_fixture.py', 'phase_readout_fixture')
    need, entry = controller.need, controller.entry
    entry.validate_spec(spec)
    need(spec['experiment_mode'] == entry.PHASE_PROFILE_MODE, 'explicit phase-profile records required')
    need(type(result) is dict and result.get('experiment_mode') == spec['experiment_mode'] and
         result.get('status') == 'NATIVE_CACHE_REGRESSION_COMPLETE' and result.get('qualification') is False and
         type(result.get('model_starts')) is int and result['model_starts'] == 0 and
         result.get('timings_usable') is True and result.get('cleanup_errors') == [] and
         result.get('unresolved') is False and result.get('independent_release_verified') is True,
         'complete observed campaign required')
    arms = result.get('arms')
    need(type(arms) is list and [arm.get('arm') for arm in arms] == spec['arm_order'], 'exact arm order')
    cells = []
    for arm in arms:
        need(arm.get('complete') is True and arm.get('release_verified') is True,
             'complete released arm required')
        native = controller.validate_result(arm['native_result'], spec, arm['arm'], reader=reader)
        for device in spec['devices']:
            for length in (128, 1024, 3966):
                occurrences = []
                for process in native['processes']:
                    if process['device'] != device['device'] or process['variant'] != 'candidate':
                        continue
                    samples = [dict(row) for row in process['native']['timings'] if row['length'] == length]
                    phases = [row['phase_wall'] for row in samples]
                    # Keep raw nanoseconds and per-sample shares. A ratio of
                    # independently computed medians need not add to 100%.
                    shares = [{key: 100 * row[key] / row['total_ns'] for key in PHASES} for row in phases]
                    occurrences.append({'ordinal': process['ordinal'], 'samples': samples,
                        'phase_share_of_profiled_call_percent': shares,
                        'median_phase_ms': {key: statistics.median(row[key] for row in phases) / 1e6
                                            for key in (*PHASES, 'total_ns')},
                        'median_outer_call_ms': statistics.median(row['synchronized_call_ms'] for row in samples)})
                cells.append({'cache_mode': arm['arm'], 'bdf': device['bdf'], 'uuid': device['uuid'],
                              'length': length, 'candidate_occurrences': occurrences})
    return {'schema': 1, 'status': 'CANDIDATE_HOST_PHASE_READOUT', 'qualification': False,
        'model_starts': 0, 'native_processes': 16, 'whole_call_samples': 240, 'profiled_samples': 120,
        'timing_scope': 'candidate-host-wall-existing-synchronized-phases',
        'phase_projection': dict(spec['phase_profile']), 'cells': cells,
        'archive_authentication': 'required separately before consumption',
        'limitations': [
            'Host clocks include launch, scheduling and synchronization costs; these are not isolated GPU durations.',
            'Validation includes status reset, validation kernel, stream completion and the existing four-byte status copy.',
            'Reader time includes query transform, fused decode, attention and completion; decode is not separately measured.',
            'Only the candidate is instrumented. Do not infer a clean speed ratio against the uninstrumented baseline.',
            'Instrumentation can perturb host scheduling and kernel layout; native effects remain to be measured.',
            'Synthetic cache ABI timings do not establish full-model throughput, BF16 fidelity or model qualification.']}
