"""Descriptive counterbalanced cache readout, after separate archive authentication.

This module consumes retained records only. It does not authenticate a remote
seal, grant a launch, load a native library, or promote a model implementation.
"""
import importlib.util
from pathlib import Path
import statistics


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(result, spec, *, controller=None, reader=None):
    root = Path(__file__).resolve().parents[1]
    if controller is None:
        controller = _module(root / 'tools/cache_gpu_controller.py', 'cache_readout_controller')
    if reader is None:
        reader = _module(root / 'tests/test_cache_gpu_fixture.py', 'cache_readout_fixture')
    need, entry = controller.need, controller.entry
    entry.validate_spec(spec)
    need(spec['experiment_mode'] == entry.COUNTERBALANCED_MODE, 'counterbalanced records required')
    need(type(result) is dict and result.get('experiment_mode') == spec['experiment_mode'] and
         result.get('status') == 'NATIVE_CACHE_REGRESSION_COMPLETE' and
         result.get('qualification') is False and type(result.get('model_starts')) is int and
         result['model_starts'] == 0 and result.get('timings_usable') is True and
         result.get('cleanup_errors') == [] and result.get('unresolved') is False and
         result.get('independent_release_verified') is True, 'complete observed campaign required')
    arms = result.get('arms')
    need(type(arms) is list and [arm.get('arm') for arm in arms] == spec['arm_order'], 'exact arm order')
    cells = []
    for arm in arms:
        need(arm.get('complete') is True and arm.get('release_verified') is True,
             'complete released arm required')
        native = controller.validate_result(arm['native_result'], spec, arm['arm'], reader=reader)
        for device in spec['devices']:
            processes = [row for row in native['processes'] if row['device'] == device['device']]
            for length in (128, 1024, 3966):
                occurrences = []
                for row in processes:
                    samples = [cell['synchronized_call_ms'] for cell in row['native']['timings']
                               if cell['length'] == length]
                    occurrences.append({'ordinal': row['ordinal'], 'variant': row['variant'],
                        'samples_ms': samples, 'median_ms': statistics.median(samples),
                        'range_ms': [min(samples), max(samples)]})
                medians = [row['median_ms'] for row in occurrences]
                # Keep both adjacent order directions. Pooling alone can hide
                # an order interaction; neither ratio is a causal correction.
                pairs = []
                for order, baseline, candidate in (('AB', 0, 1), ('BA', 3, 2)):
                    pairs.append({'order': order, 'baseline_ordinal': baseline,
                        'candidate_ordinal': candidate,
                        'speed_ratio': medians[baseline] / medians[candidate],
                        'latency_reduction_percent': 100 * (1 - medians[candidate] / medians[baseline])})
                cells.append({'cache_mode': arm['arm'], 'bdf': device['bdf'], 'uuid': device['uuid'],
                    'length': length, 'occurrences': occurrences, 'adjacent_pairs': pairs,
                    'within_variant_later_over_earlier': {
                        'baseline': medians[3] / medians[0], 'candidate': medians[2] / medians[1]}})
    return {'schema': 1, 'status': 'COUNTERBALANCED_DESCRIPTIVE_READOUT',
        'qualification': False, 'model_starts': 0, 'native_processes': 16, 'timing_samples': 240,
        'timing_scope': 'synchronized-whole-attention-API-call', 'cells': cells,
        'archive_authentication': 'required separately before consumption',
        'limitations': ['ABBA is not randomized and cannot exclude nonlinear thermal/history effects.',
                       'Two process occurrences per variant are not ten independent experiments.',
                       'Whole-call time is not reader-kernel time, memory bandwidth, or model throughput.',
                       'No BF16-control subtraction, statistical significance, or causal attribution.']}
