"""Pure, source-bound speculative contract for the native HTTP screen.

This module deliberately has no SGLang, Torch, or environment imports.  It
gives the package builder, launcher, and readers one canonical description of
the two logical arms; native code still resolves and verifies the fields.
"""

from __future__ import annotations

from typing import Any, Mapping
import hashlib
import base64
import json
import re
import zlib
from pathlib import Path
from functools import lru_cache


MODE = "native-http-fp8-sprint01-v1"
LEGACY_MODE = "native-http-final-verification-v1"
MODEL_PATH = "/mnt/models"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
DRAFT_GRAPH_ENV = "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH"
OWNER_ARMS = {"bf16": "control", "kv4": "mtp"}
LOGICAL_ARMS = frozenset(OWNER_ARMS.values())
ORIGINAL = '6f7c1ca7e17c9d7b686255f6c4e9fa9830e8ef4f6f898c15c26f0b834722c3da'
CANDIDATE = 'fdb3bc828ae7bcf090022035cbe5cf15fe4f747e8a12299e94b290091f8c03b2'
TAIL = 'b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069'
GEMM_TARGET = '/mnt/composition/image-source/python/sglang/kernels/ops/gemm/fp8_kernel.py'
GEMM_MASK = '/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/kernels/ops/gemm'
GDN_PAIR_PHASE = 'sprint01-gdn-aligned-pair'
LIFECYCLE_PHASE = 'sprint01-gdn-aligned-lifecycle'
PRECISION_REFERENCE_PHASE = 'sprint01-bf16-reference-pair'
PROMPT_MINIMIZATION_PHASE = 'sprint01-prompt-minimization'
PAIRED_PHASE = 'sprint01-paired-serving'
PRECISION_REFERENCE_PHASES = frozenset({PRECISION_REFERENCE_PHASE, PROMPT_MINIMIZATION_PHASE, PAIRED_PHASE})
PROMPT_MINIMIZATION_CASES = ('quality16k-audit-threshold', 'prompt-ledger-no-filler', 'prompt-j01-only')
CLOSURE_PHASES = frozenset({LIFECYCLE_PHASE})
BATCH_PHASES = CLOSURE_PHASES | {PAIRED_PHASE}
CACHE_PHASES = BATCH_PHASES
HISTORY_PHASES = frozenset({'sprint01-hip-confirmation-reverse', GDN_PAIR_PHASE})
GDN_STOCK = '7cad2a1b690b15c8d13f15cf1e458a7fba85f2bca9e47327a9804a1e7ba92943'
GDN_ALIGNED = '0ae8788fc8194505fb6a0ee5e320456f5fb9ebf45d4b83a4d28e2e196ec15b87'
LIFECYCLE_WORKER_SHA256 = '8702620c2d6c7101d0682ec21d3dff57f636446620eb9250ba084014a41c810f'
GDN_TARGET = '/mnt/composition/image-source/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py'
GDN_MASK = '/mnt/capture-bytecode' + str(Path(GDN_TARGET).parent)
NORM_STOCK = 'cb87d85a89fdd06c5ab2f362e9f06b11dc32a6b5519440805bde038172e4c82b'
NORM_CANDIDATE = 'a3e74974dda5a209a97d491f197fe124351e930a15ac828e2de60c53bb55dbe5'
NORM_VARIANT = 'gemma-native-eight-singleton-variance-v1'
NORM_TARGET = '/mnt/composition/image-source/python/sglang/srt/layers/layernorm.py'
NORM_MASK = '/mnt/capture-bytecode' + str(Path(NORM_TARGET).parent)
PHASES = {
    PAIRED_PHASE: (32768, 1740, 19, ('dflash2', 'dflash2'), ('tail', 'tail')),
    PROMPT_MINIMIZATION_PHASE: (16384, 420, 3, ('none', 'none'), ('candidate', 'candidate')),
    PRECISION_REFERENCE_PHASE: (65536, 1740, 7, ('dflash2', 'dflash2'), ('tail', 'tail')),
    LIFECYCLE_PHASE: (32768, 420, 27, ('dflash2', 'dflash2'), ('candidate', 'candidate')),
    GDN_PAIR_PHASE: (32768, 420, 12, ('dflash2', 'dflash2'), ('candidate', 'candidate')),
    'main-forward': (32768, 300, 27, ('none', 'none'), ('original', 'candidate')),
    'main-reverse': (32768, 300, 27, ('none', 'none'), ('candidate', 'original')),
    'mtp-forward': (4096, 120, 3, ('none', 'mtp'), ('candidate', 'candidate')),
    'mtp-reverse': (4096, 120, 3, ('mtp', 'none'), ('candidate', 'candidate')),
    'dflash-forward': (4096, 180, 3, ('none', 'dflash2'), ('candidate', 'candidate')),
    'dflash-reverse': (4096, 180, 3, ('dflash2', 'none'), ('candidate', 'candidate')),
    'mtp-dflash-forward': (4096, 180, 3, ('mtp', 'dflash2'), ('candidate', 'candidate')),
    'mtp-dflash-reverse': (4096, 180, 3, ('dflash2', 'mtp'), ('candidate', 'candidate')),
    'dflash-context-forward': (32800, 300, 8, ('none', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-forward': (32768, 420, 12, ('none', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-reverse': (32768, 420, 12, ('dflash2', 'none'), ('candidate', 'candidate')),
    'sprint01-localize': (32768, 420, 9, ('none', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-width-forward': (32768, 420, 12, ('dflash2', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-width-reverse': (32768, 420, 12, ('dflash2', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-tail-forward': (32768, 420, 14, ('dflash2', 'dflash2'), ('candidate', 'tail')),
    'sprint01-tail-reverse': (32768, 420, 14, ('dflash2', 'dflash2'), ('tail', 'candidate')),
    'sprint01-combined-forward': (32768, 420, 16, ('dflash2', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-sampling-forward': (32768, 420, 12, ('none', 'none'), ('candidate', 'candidate')),
    'sprint01-hip-sampling-forward': (32768, 420, 12, ('none', 'dflash2'), ('candidate', 'candidate')),
    'sprint01-hip-confirmation-reverse': (32768, 420, 12, ('dflash2', 'none'), ('candidate', 'candidate')),
}
WIDTH_PHASES = {'sprint01-width-forward': (8, 4), 'sprint01-width-reverse': (4, 8)}
TAIL_PHASES = frozenset({'sprint01-tail-forward', 'sprint01-tail-reverse'})
FAIL_FAST_PHASES = frozenset({'sprint01-localize', 'sprint01-combined-forward', 'sprint01-sampling-forward', 'sprint01-hip-sampling-forward', 'sprint01-hip-confirmation-reverse', GDN_PAIR_PHASE, LIFECYCLE_PHASE, *WIDTH_PHASES, *TAIL_PHASES})
_LEGACY_PHASES = frozenset({'main-forward', 'main-reverse', 'mtp-forward', 'mtp-reverse'})
_DFLASH_PHASES = frozenset(PHASES) - _LEGACY_PHASES


def settings():
    path = Path(__file__).with_name('FINAL-SETTINGS.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size < 32768,
             'final settings missing or oversized')
    value = json.loads(path.read_bytes())
    fields = {'schema', 'mode', 'phase', 'requests'}
    allowed = (fields, fields | {'screen_policy'}, fields | {'screen_policy', 'execution_order'})
    if value.get('phase') == GDN_PAIR_PHASE:
        allowed += tuple(row | {'aligned_gdn_both_arms'} for row in allowed)
        allowed += tuple(row | {'greedy_quality_draft_ab'} for row in allowed
                         if 'aligned_gdn_both_arms' in row)
        allowed += tuple(row | {'j01_probe'} for row in allowed if 'greedy_quality_draft_ab' in row)
        allowed += tuple(row | {'verify_attention_ab'} for row in allowed if 'j01_probe' in row)
        allowed += tuple(row | {'head_input_capture'} for row in allowed if 'j01_probe' in row)
        allowed += tuple(row | {'final_norm_capture'} for row in allowed if 'head_input_capture' in row)
        allowed += tuple(row | {'midpoint_capture'} for row in allowed if 'head_input_capture' in row
                         and 'final_norm_capture' not in row)
        allowed += tuple(row | {'raw_attention_capture'} for row in allowed if 'midpoint_capture' in row)
        allowed += tuple(row | {'norm_candidate'} for row in allowed if 'midpoint_capture' in row)
    if value.get('phase') in PRECISION_REFERENCE_PHASES:
        allowed = (fields | {'execution_order', 'precision_reference'}, fields | {'execution_order', 'precision_reference', 'serving_stack'}, fields | {'execution_order', 'precision_reference', 'serving_stack', 'performance_benchmark', 'screen_policy', 'full_enabled_benchmark'}, fields | {'execution_order', 'precision_reference', 'serving_stack', 'quality_screen', 'screen_policy', 'full_enabled_benchmark'})
    if value.get('phase') == PAIRED_PHASE:
        allowed = (fields | {'execution_order','precision_reference','serving_stack','quality_screen','screen_policy','full_enabled_benchmark','paired_serving'},)
    if value.get('phase') == LIFECYCLE_PHASE:
        allowed = (fields | {'execution_order'},
                   fields | {'execution_order', 'diagnostic_kernel_serialization'})
    if 'phase_timing' in value:
        _require(value.get('phase') == PRECISION_REFERENCE_PHASE and
                 value.get('quality_screen') == 'sprint7-low-thinking-normal-stop-four-v1' and
                 type(value['phase_timing']) is str and value['phase_timing'] == 'sprint8-c1-phase-timing-v1',
                 'phase timing profile differs')
        allowed += tuple(row | {'phase_timing'} for row in allowed if 'quality_screen' in row)
    if 'target_timing' in value:
        _require(value.get('phase_timing') == 'sprint8-c1-phase-timing-v1' and
                 type(value['target_timing']) is str and value['target_timing'] == 'sprint9-native-target-families-v1',
                 'target timing profile differs')
        allowed += tuple(row | {'target_timing'} for row in allowed if 'phase_timing' in row)
    _require(type(value) is dict and
             (set(value) in allowed or set(value) - {'allocator_snapshot'} in allowed), 'final settings fields')
    if 'serving_stack' in value:
        _require(value['serving_stack']=='torch214-sdk102-compat-v1' and value['phase'] in (PRECISION_REFERENCE_PHASE, PAIRED_PHASE), 'serving stack variant')
    if 'precision_reference' in value:
        _require(value['phase'] in PRECISION_REFERENCE_PHASES and
                 type(value['precision_reference']) is str and value['precision_reference'] in ('fp8', 'bf16') and
                 (value['phase'] != PROMPT_MINIMIZATION_PHASE or value['precision_reference'] == 'bf16'),
                 'precision reference flag/phase differs')
    if 'diagnostic_kernel_serialization' in value:
        _require(value['phase'] == LIFECYCLE_PHASE and
                 type(value['diagnostic_kernel_serialization']) is str and
                 value['diagnostic_kernel_serialization'] == '3',
                 'kernel serialization diagnostic binding differs')
    if 'verify_attention_ab' in value:
        _require(value['verify_attention_ab'] is True and value.get('j01_probe') is True and
                 not any(key in value for key in ('head_input_capture', 'final_norm_capture',
                     'midpoint_capture', 'raw_attention_capture')), 'verify attention A/B binding differs')
    if 'raw_attention_capture' in value:
        _require(value['raw_attention_capture'] == 'j01-full-kv-reference-v1' and
                 value.get('midpoint_capture') == 'j01-block3-prepared-attention-v1',
                 'raw attention requires prepared-attention capture')
    if 'aligned_gdn_both_arms' in value:
        _require(value['phase'] == GDN_PAIR_PHASE and value['aligned_gdn_both_arms'] is True,
                 'both-aligned GDN quality binding differs')
    if 'greedy_quality_draft_ab' in value:
        _require(value['phase'] == GDN_PAIR_PHASE and value['greedy_quality_draft_ab'] is True and
                 value.get('aligned_gdn_both_arms') is True,
                 'greedy quality draft A/B binding differs')
    if 'j01_probe' in value:
        _require(value['j01_probe'] is True and value.get('greedy_quality_draft_ab') is True and
                 value['phase'] == GDN_PAIR_PHASE, 'J01 probe binding differs')
    if 'head_input_capture' in value:
        _require(type(value['head_input_capture']) is str and
                 value['head_input_capture'] == 'j01-hidden-weights-v1' and
                 value.get('j01_probe') is True and value.get('greedy_quality_draft_ab') is True and
                 value['phase'] == GDN_PAIR_PHASE, 'J01 head-input capture binding differs')
    if 'allocator_snapshot' in value:
        _require(value['allocator_snapshot'] is True and value['phase'] in (GDN_PAIR_PHASE, LIFECYCLE_PHASE, *PRECISION_REFERENCE_PHASES),
                 'allocator snapshot flag/phase differs')
    if 'final_norm_capture' in value:
        _require(type(value['final_norm_capture']) is str and
                 value['final_norm_capture'] == 'j01-final-norm-v1' and
                 value.get('head_input_capture') == 'j01-hidden-weights-v1' and
                 value.get('j01_probe') is True and value['phase'] == GDN_PAIR_PHASE,
                 'J01 final-norm capture binding differs')
    if 'midpoint_capture' in value:
        _require(type(value['midpoint_capture']) is str and
                 value['midpoint_capture'] in ('j01-quarter-native-output-v1','j01-early-native-outputs-v1','j01-block3-native-entry-v1','j01-block3-post-attention-v1','j01-block3-qkv-projection-v1','j01-block3-prepared-attention-v1','j01-block11-chain-v1','j01-block8-gdn-v1','j01-layer8-norm-v1') and
                 'final_norm_capture' not in value and
                 value.get('head_input_capture') == 'j01-hidden-weights-v1' and
                 value.get('j01_probe') is True and value.get('greedy_quality_draft_ab') is True and
                 value['phase'] == GDN_PAIR_PHASE, 'J01 midpoint capture binding differs')
    if 'norm_candidate' in value:
        _require(type(value['norm_candidate']) is str and value['norm_candidate'] == NORM_VARIANT and
                 value.get('midpoint_capture') == 'j01-layer8-norm-v1' and
                 value['phase'] == GDN_PAIR_PHASE, 'norm candidate scope or variant differs')
    if 'execution_order' in value:
        _require(value['phase'] in (GDN_PAIR_PHASE, LIFECYCLE_PHASE, *PRECISION_REFERENCE_PHASES) and value['execution_order'] in
                 (['control', 'mtp'], ['mtp', 'control']), 'frozen execution order differs')
    if 'screen_policy' in value:
        _require(value['screen_policy'] == 'exploratory-stop-or-length-v1' and
                 (value['phase'] in HISTORY_PHASES or value.get('performance_benchmark',{}).get('mode')=='m2047-fixed1024-c1-v1' or value.get('quality_screen') in ('sprint7-low-thinking-normal-stop-four-v1', 'sprint33-sre64k-normal-stop-six-v1', 'sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1')), 'exploratory screen policy/phase differs')
    _require(value['schema'] == 1 and
             ((value['mode'] == LEGACY_MODE and value['phase'] in _LEGACY_PHASES) or
              (value['mode'] == MODE and value['phase'] in _DFLASH_PHASES)), 'final phase binding')
    quality = value.get('quality_screen') in ('sprint7-low-thinking-normal-stop-four-v1', 'sprint33-sre64k-normal-stop-six-v1', 'sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1')
    performance = value.get('performance_benchmark') == {'mode':'m2047-fixed1024-c1-v1','stack_by_arm':{'control':'current','mtp':'current'},'warmup_requests':1,'measured_requests':3}
    _require(value.get('full_enabled_benchmark') == 'dflash8-splitkv-aligned-gdn-v1' and
             value.get('screen_policy') == 'exploratory-stop-or-length-v1' and
             ((quality and 'performance_benchmark' not in value) or
              (performance and 'quality_screen' not in value)), 'full-enabled screen protocol')
    if quality and value['quality_screen'] == 'sprint7-low-thinking-normal-stop-four-v1':
        _require(value['phase'] == PRECISION_REFERENCE_PHASE and value['precision_reference'] == 'fp8' and
                 tuple((row['case'], row['maximum_output_tokens']) for row in value['requests']) ==
                 (('coding-running-totals', 4096), ('coding-unique-last', 4096),
                  ('quality16k-audit-threshold', 4096), ('heldout16k-audit', 4096)),
                 'quality frozen cases/caps differ')
    if value.get('quality_screen') == 'sprint33-sre64k-normal-stop-six-v1':
        _require(value['phase'] == PRECISION_REFERENCE_PHASE and value['precision_reference'] == 'fp8' and
                 tuple((row['case'], row['maximum_output_tokens']) for row in value['requests']) ==
                 tuple((name, 4096) for name in ('sre-warmup', 'incident-10', 'incident-50', 'incident-90', 'runbook-10', 'runbook-50', 'runbook-90')),
                 'S32 frozen cases/caps differ')
    if value.get('quality_screen') == 'sprint35-long-prefix-reuse-v1':
        _require(value['phase'] == PRECISION_REFERENCE_PHASE and value['precision_reference'] == 'fp8' and tuple((row['case'], row['maximum_output_tokens']) for row in value['requests']) == tuple((name, 4096) for name in ('warmup','cold-a','warm-a','branch-a-warm','isolated-b-cold','branch-b-warm')), 'S35 frozen cases/caps differ')
    if value.get('quality_screen') == 'sprint36-chunk-prefill-v1':
        _require(value['phase'] == PRECISION_REFERENCE_PHASE and value['precision_reference'] == 'fp8' and tuple((row['case'], row['maximum_output_tokens']) for row in value['requests']) == (('warmup',4096),('cold-a',4096),('isolated-b-cold',4096)), 'S36 frozen cases/caps differ')
    if value.get('phase') == PAIRED_PHASE:
        _require(value.get('quality_screen') == 'sprint39-paired-serving-v1' and value.get('precision_reference') == 'fp8', 'S39 phase/precision differs')
        validate_paired_settings(value)
    rows = value['requests']
    assets = _assets()
    _require(type(rows) is list and len(rows) == (3 if value.get('quality_screen') == 'sprint36-chunk-prefill-v1' else (6 if value.get('quality_screen') == 'sprint35-long-prefix-reuse-v1' else PHASES[value['phase']][2])) and
             [row['ordinal'] for row in rows] == list(range(len(rows))), 'final request order/count')
    _require(len({row['case'] for row in rows}) == len(rows), 'duplicate final request')
    if value['phase'] == PROMPT_MINIMIZATION_PHASE:
        _require(tuple(row['case'] for row in rows) == PROMPT_MINIMIZATION_CASES and
                 all(type(row.get('maximum_output_tokens')) is int and row['maximum_output_tokens'] == 1024
                     for row in rows) and rows[0]['sha256'] ==
                 '97611cd1d5d3fae7a56eb2d5be21853334267c3da9417a68d6a3da377c0f0394',
                 'prompt minimization frozen order/cap/original binding differs')
    inventory, history_count = set(), 0
    for row in rows:
        bound_assets = [(row, 'path', 'sha256'), (row, 'expected', 'expected_sha256')]
        if 'generated_history' in row:
            history_count += 1
            history = row['generated_history']
            _require(value['phase'] in HISTORY_PHASES and
                     type(history) is dict and set(history) == {'previous_case', 'branches'} and
                     row['ordinal'] > 0 and history['previous_case'] == rows[row['ordinal'] - 1]['case'] and
                     'generated_history' not in rows[row['ordinal'] - 1], 'generated history parent/phase differs')
            branches = history['branches']
            _require(type(branches) is list and len(branches) == 3 and
                     all(type(branch) is dict and set(branch) == {'assistant_text', 'expected', 'expected_sha256'} and
                         type(branch['assistant_text']) is str and 0 < len(branch['assistant_text'].encode()) <= 256
                         for branch in branches) and
                     len({branch['assistant_text'] for branch in branches}) == len(branches),
                     'generated history branch schema differs')
            bound_assets.extend((branch, 'expected', 'expected_sha256') for branch in branches)
        for owner, field, pin in bound_assets:
            if field not in owner:
                _require(field == 'expected' and row['ordinal'] < 3, 'missing frozen vector')
                continue
            relative = owner[field]
            _require(type(relative) is str and relative == Path(relative).name and relative.startswith('FINAL-REQUEST-'),
                     'request asset outside private inventory')
            _require(relative in assets and len(assets[relative]) <= 262144, 'request asset missing or oversized')
            _require(hashlib.sha256(assets[relative]).hexdigest() == owner[pin], 'request asset hash mismatch')
            inventory.add(relative)
    _require(history_count == (2 if value['phase'] in HISTORY_PHASES else 0),
             'generated history case count differs')
    _require(set(assets) == inventory, 'exact compressed asset inventory')
    return value


def _assets():
    path = Path(__file__).with_name('FINAL-REQUESTS.zlib')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 << 20, 'bounded frozen assets')
    return _decoded_assets(path.read_bytes())


@lru_cache(maxsize=1)
def _decoded_assets(compressed):
    # Cache by exact compressed bytes, never by path or an unchecked timestamp.
    # Every settings call still reads the bound file and verifies member hashes.
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, (8 << 20) + 1)
    _require(len(raw) <= 8 << 20 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
             'frozen asset expanded size or trailing content')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate frozen asset')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs)
    _require(type(value) is dict and len(value) in (2, 6, 8, 12, 14, 16, 18, 24, 28, 30, 32, 38, 54), 'frozen asset count')
    return {key: base64.b64decode(item, validate=True) for key, item in value.items()}


def asset_bytes(name):
    return _assets()[name]


def phase_config():
    if betterbench_checkpoint_enabled():return (32768,1740,53,('dflash2','dflash2'),('tail','tail'))
    value = settings()
    result = PHASES[value['phase']]
    if value.get('quality_screen') in ('sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1'):
        result = (32768, result[1], 3 if value.get('quality_screen') == 'sprint36-chunk-prefill-v1' else 6, *result[3:])
    if value.get('verify_attention_ab') is True:
        return result
    if value.get('greedy_quality_draft_ab') is True:
        return result[:3] + (('none', 'dflash2'),) + result[4:]
    return result


def prompt_minimization_enabled():
    return settings()['phase'] == PROMPT_MINIMIZATION_PHASE


def precision_reference_enabled():
    return settings()['phase'] in PRECISION_REFERENCE_PHASES


def precision_profile():
    """Exact source profile for the two ordinary-decode precision starts."""
    value = settings()
    _require(value['phase'] in PRECISION_REFERENCE_PHASES, 'precision reference phase required')
    path = Path(__file__).with_name('PRECISION-REFERENCE.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size < 32768,
             'precision reference profile missing or oversized')
    profile = json.loads(path.read_bytes())
    fields = {'schema', 'precision', 'repository', 'revision', 'model_root',
              'inventory_sha256', 'config_sha256', 'files', 'bytes', 'quantization',
              'memory_fraction_static', 'context_length', 'max_total_tokens', 'seed',
              'model_loader_extra_config', 'weight_loader_drop_cache_after_load',
              'weight_loader_disable_mmap', 'weight_loader_prefetch_checkpoints'}
    _require(type(profile) is dict and set(profile) == fields and
             type(profile['schema']) is int and profile['schema'] == 1 and
             profile['precision'] == value['precision_reference'], 'precision profile schema/selection differs')
    _require(type(profile['repository']) is str and
             re.fullmatch('[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', profile['repository']) and
             type(profile['revision']) is str and re.fullmatch('[0-9a-f]{40}', profile['revision']),
             'precision profile immutable model identity differs')
    root = profile['model_root']
    _require(type(root) is str and root.startswith('/home/developer/') and
             str(Path(root)) == root and '..' not in Path(root).parts and
             4 <= len(Path(root).parts) <= 10 and len(root) <= 4096,
             'precision profile model root differs')
    _require(all(type(profile[key]) is str and re.fullmatch('[0-9a-f]{64}', profile[key])
                 for key in ('inventory_sha256', 'config_sha256')) and
             all(type(profile[key]) is int and profile[key] > 0 for key in ('files', 'bytes')),
             'precision profile inventory binding differs')
    expected_quantization = 'fp8' if profile['precision'] == 'fp8' else None
    _require(profile['quantization'] == expected_quantization and
             type(profile['memory_fraction_static']) is float and profile['memory_fraction_static'] == .92 and
             all(type(profile[key]) is int and profile[key] == expected for key, expected in
                 (('context_length', 32768 if value.get('quality_screen') in ('sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1') else 65536), ('max_total_tokens', 32800 if value.get('quality_screen') in ('sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1') else 65568), ('seed', 12345))),
             'precision profile runtime binding differs')
    loading = {key: profile[key] for key in PRECISION_LOADING_FIELDS}
    _require(loading == PRECISION_LOADING_POLICY and
             type(loading['model_loader_extra_config']) is dict and
             set(loading['model_loader_extra_config']) == {'enable_multithread_load'} and
             loading['model_loader_extra_config']['enable_multithread_load'] is False and
             all(type(loading[key]) is bool for key in PRECISION_LOADING_FIELDS
                 if key != 'model_loader_extra_config'), 'precision profile loading policy differs')
    return profile


PRECISION_LOADING_POLICY = dict(model_loader_extra_config={'enable_multithread_load': False},
    weight_loader_drop_cache_after_load=True, weight_loader_disable_mmap=False,
    weight_loader_prefetch_checkpoints=False)
PRECISION_LOADING_FIELDS = frozenset(PRECISION_LOADING_POLICY)


def precision_loading_policy():
    profile = precision_profile()
    return {key: profile[key] for key in PRECISION_LOADING_FIELDS}


def validate_precision_loading_policy(value, *, encoded_extra_config=False):
    expected = precision_loading_policy()
    _require(type(value) is dict and set(value) == PRECISION_LOADING_FIELDS,
             'precision loading policy fields differ')
    extra = value['model_loader_extra_config']
    if encoded_extra_config:
        _require(type(extra) is str and extra == json.dumps(expected['model_loader_extra_config'],
                 sort_keys=True, separators=(',', ':')), 'precision native loader extra config differs')
    else:
        _require(type(extra) is dict and set(extra) == {'enable_multithread_load'} and
                 extra['enable_multithread_load'] is False, 'precision loader extra config differs')
    _require(all(type(value[key]) is bool and value[key] is expected[key]
                 for key in PRECISION_LOADING_FIELDS if key != 'model_loader_extra_config'),
             'precision loading flags differ')
    return expected


def validate_precision_loading_cli(cli):
    _require(type(cli) is list and all(type(item) is str for item in cli), 'precision loading CLI fields')
    expected = json.dumps(precision_loading_policy()['model_loader_extra_config'],
                          sort_keys=True, separators=(',', ':'))
    _require(cli.count('--model-loader-extra-config') == 1 and
             cli[cli.index('--model-loader-extra-config') + 1:cli.index('--model-loader-extra-config') + 2] == [expected] and
             cli.count('--weight-loader-drop-cache-after-load') == 1 and
             '--weight-loader-disable-mmap' not in cli and
             '--weight-loader-prefetch-checkpoints' not in cli,
             'precision loading CLI differs')
    return precision_loading_policy()


def precision_loading_sources(sources):
    """Stock source identities used by the observed native loading policy."""
    prefix = '/mnt/composition/image-source/python/sglang/'
    expected = {
        'model_loader': ('srt/model_loader/loader.py',
            'ef2c09ef3abcbf65390347cda178443f59e6345b45ea55348a851ae06f59edac'),
        'loader_weight_utils': ('srt/model_loader/weight_utils.py',
            '4d2449d1e2a1c1be97c057fd86ad3a7a6e53a89e88ea9ec9b420c8868749760b'),
        'loader_model_fields': ('srt/arg_groups/fields/model.py',
            'ca29dfdd7406129c2b327675f7270d4e3e8512770055e6b37e1c1db17f52be41'),
        'loader_load_config': ('srt/configs/load_config.py',
            '9255a24fe8a1553360190e7a6e5baeb94caa2a4e6d2170eee37391e91f6844cc'),
        'head_context': ('srt/runtime_context.py',
            '50f1e87490d0b5c2bb1ddf820d65719cd1c9b876d265e902934692cd3719b10e'),
        'model_config': ('srt/configs/model_config.py',
            'be3a0168bdd3b4a6b1f7778420b83f95c315535812b44e9422e1b48dd856a11e'),
    }
    result = {}
    for role, (relative, digest) in expected.items():
        wanted = dict(path=prefix + relative, module='sglang.' + relative[:-3].replace('/', '.'), sha256=digest)
        _require(sources.get(role) == wanted, 'precision loading source differs: ' + role)
        result[role] = wanted
    return result


def target_quantization():
    return precision_profile()['quantization'] if precision_reference_enabled() else 'fp8'


def target_model_revision():
    return precision_profile()['revision'] if precision_reference_enabled() else MODEL_REVISION


def target_config_sha256():
    return precision_profile()['config_sha256'] if precision_reference_enabled() else None


def memory_fraction_static():
    return precision_profile()['memory_fraction_static'] if precision_reference_enabled() else .80


def serving_seed_required():
    return precision_reference_enabled() or verify_attention_chain_enabled()


def verify_attention_ab_enabled():
    return settings().get('verify_attention_ab') is True


def verify_attention_chain_enabled():
    return full_enabled_benchmark() or settings().get('midpoint_capture') in ('j01-block11-chain-v1','j01-block8-gdn-v1','j01-layer8-norm-v1')


def gdn_capture_enabled():
    return settings().get('midpoint_capture') == 'j01-block8-gdn-v1'


def norm_capture_enabled():
    return settings().get('midpoint_capture') == 'j01-layer8-norm-v1'


def norm_candidate_enabled():
    value = settings()
    return value.get('midpoint_capture') == 'j01-layer8-norm-v1' and value.get('norm_candidate') == NORM_VARIANT


def verify_attention_observed(arm):
    return verify_attention_ab_enabled() or (verify_attention_chain_enabled() and arm in LOGICAL_ARMS)


def greedy_quality_draft_ab():
    return settings().get('greedy_quality_draft_ab') is True


def j01_probe_enabled():
    return settings().get('j01_probe') is True


def head_input_capture_enabled():
    return settings().get('head_input_capture') == 'j01-hidden-weights-v1'


def final_norm_capture_enabled():
    return settings().get('final_norm_capture') == 'j01-final-norm-v1'


def midpoint_capture_enabled():
    return settings().get('midpoint_capture') in ('j01-quarter-native-output-v1','j01-early-native-outputs-v1','j01-block3-native-entry-v1','j01-block3-post-attention-v1','j01-block3-qkv-projection-v1','j01-block3-prepared-attention-v1','j01-block11-chain-v1','j01-block8-gdn-v1','j01-layer8-norm-v1')


def raw_attention_capture_enabled():
    return settings().get('raw_attention_capture') == 'j01-full-kv-reference-v1'


def cache_config():
    if settings()['phase'] not in CACHE_PHASES:
        return None
    return dict(disable_radix_cache=False, page_size=1,
                mamba_radix_cache_strategy='no_buffer', disable_overlap_schedule=True,
                enable_int8_mamba_checkpoint=False, uses_mamba_radix_cache=True)


def validate_cache_config(value):
    expected = cache_config()
    _require(expected is not None and type(value) is dict and set(value) == set(expected) and
             all(type(value[key]) is type(wanted) and value[key] == wanted
                 for key, wanted in expected.items()), 'native prefix-cache configuration differs')
    return dict(value)


def memory_pool_config(arm):
    if precision_reference_enabled():
        _require(arm in LOGICAL_ARMS, 'unknown memory-pool arm')
        profile = precision_profile()
        return dict(context_length=profile['context_length'], max_running_requests=2,
                    max_total_tokens=profile['max_total_tokens'], kv_cache_dtype='auto',
                    max_mamba_cache_size=19)
    if settings()['phase'] != LIFECYCLE_PHASE:
        return None
    _require(arm in LOGICAL_ARMS, 'unknown memory-pool arm')
    return dict(context_length=32768, max_running_requests=2,
                max_total_tokens=32800, kv_cache_dtype='auto')


def validate_recurrent_capacity_cli(cli):
    _require(type(cli) is list and cli.count('--max-mamba-cache-size') == 1 and
             cli[cli.index('--max-mamba-cache-size')+1:cli.index('--max-mamba-cache-size')+2] == ['19'],
             'Sprint7 exact recurrent capacity CLI differs')


def validate_memory_pool_config(value, arm):
    expected = memory_pool_config(arm)
    _require(expected is not None and type(value) is dict and set(value) == set(expected) and
             all(type(value[key]) is type(wanted) and value[key] == wanted
                 for key, wanted in expected.items()), 'native memory-pool configuration differs')
    return dict(value)


def expected_recurrent_slots():
    if betterbench_checkpoint_enabled():return 19
    _require(settings()['phase'] in BATCH_PHASES, 'lifecycle recurrent geometry phase')
    return 19 if paired_serving_enabled() else 12


def pool_token_capacity(arm=None):
    context = phase_config()[0]
    if precision_reference_enabled():
        _require(arm is None or arm in LOGICAL_ARMS, 'unknown memory-pool arm')
        return precision_profile()['max_total_tokens']
    if settings()['phase'] == LIFECYCLE_PHASE:
        _require(arm is None or arm in LOGICAL_ARMS, 'unknown memory-pool arm')
        return 32800
    if settings()['phase'] in ('dflash-context-forward', 'sprint01-forward', 'sprint01-reverse', 'sprint01-localize', 'sprint01-combined-forward', 'sprint01-sampling-forward', 'sprint01-hip-sampling-forward', 'sprint01-hip-confirmation-reverse', GDN_PAIR_PHASE, *WIDTH_PHASES, *TAIL_PHASES):
        # Native speculative allocation reserves twice block8 per request,
        # even with overlap disabled. Fund that lookahead for both configured
        # request slots equally in both arms. The context phase also keeps
        # its frozen <=32768-token requests 32 below the server context cap,
        # avoiding the native context+12 request-row/full-boundary mismatch.
        # Neither allowance increases the configured memory fraction.
        return context + 2 * 2 * 8
    return context


def gemm_for_arm(arm):
    _require(arm in LOGICAL_ARMS, 'unknown source arm')
    kind = ('tail' if settings().get('quality_screen') in ('sprint33-sre64k-normal-stop-six-v1', 'sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1') else ('reference' if arm == 'control' else 'tail')) if full_enabled_benchmark() else phase_config()[4][('control', 'mtp').index(arm)]
    pins = {'original': ORIGINAL, 'candidate': CANDIDATE, 'reference': 'b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069', 'tail': TAIL}
    return {'file': 'GEMM-' + kind + '.py', 'sha256': pins[kind],
            'target': GEMM_TARGET, 'bytecode_mask': GEMM_MASK}


def replay_serving_enabled(arm):
    _require(arm in LOGICAL_ARMS and full_enabled_benchmark(), 'replay serving arm/phase')
    return True


def replay_serving_sandbox_argv(argv, package_files):
    """Preserve exact native mounts while deferring only reviewed empty dirs."""
    result = list(argv)
    if 'review/replay_memory_pool_candidate.py' not in package_files:
        return result
    _require(full_enabled_benchmark(), 'replay directory projection phase')
    _require(any(result[i] == '--tmpfs' and result[i+1].startswith('/mnt/capture-bytecode/')
                 for i in range(len(result)-1)), 'nested bytecode tmpfs creates parent')
    for directory in ('/mnt/output/home', '/mnt/output/tmp', '/mnt/output/cache',
                      '/mnt/capture-bytecode'):
        pair = ['--dir', directory]
        positions = [i for i in range(len(result)-1) if result[i:i+2] == pair]
        _require(len(positions) == 1, 'exact deferred sandbox directory')
        del result[positions[0]:positions[0]+2]
    return result


def replay_pool_overlay():
    path = Path(__file__).with_name('replay_memory_pool_candidate.py')
    pin = 'cdbbac7dca001f3f3d61f7cea6df83e3f6ecf933634cad83e2b40a9cd6a596ab'
    _require(path.is_file() and not path.is_symlink() and
             hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'replay pool source differs')
    target = '/mnt/composition/image-source/python/sglang/srt/mem_cache/memory_pool.py'
    return dict(key='memory_pool', file=path.name, target=target,
        module='sglang.srt.mem_cache.memory_pool',
        original_sha256='47e13954eb8b06836a5279fedd1f79f03dbd40245819ae152af3a02f3887c1ac',
        sha256=pin, bytecode_mask='/mnt/capture-bytecode'+str(Path(target).parent))


def replay_configurator_overlay():
    path = Path(__file__).with_name('replay_kv_cache_configurator_candidate.py')
    pin = 'cf7341e9f31593488f10fc19d1e95ab469817c56f8cb022b5cadf5734e83ea04'
    _require(path.is_file() and not path.is_symlink() and
             hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'replay configurator source differs')
    target = '/mnt/composition/image-source/python/sglang/srt/mem_cache/kv_cache_configurator.py'
    return dict(key='kv_cache_configurator', file=path.name, target=target,
        module='sglang.srt.mem_cache.kv_cache_configurator',
        original_sha256='afaac626b5ad14440521666310379c84dcc109e3d9f48ef09c9a0fd470cc0d52',
        sha256=pin, bytecode_mask='/mnt/capture-bytecode'+str(Path(target).parent))


def replay_hybrid_overlay():
    path = Path(__file__).with_name('capture_hybrid_linear_attn_backend.py')
    pin = '4106e9b9981b0f18e415114d8fbc6512fab25c5841d1a1a06f3e8ca2cb4e2804'
    _require(path.is_file() and not path.is_symlink() and
             hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'replay commit source differs')
    target = '/mnt/composition/image-source/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py'
    return dict(key='hybrid_backend', file=path.name, target=target,
        module='sglang.srt.layers.attention.hybrid_linear_attn_backend',
        original_sha256='8339a92ed5e4fd91f306f81f48d29c60f93a9e0e2ae4974cb6916f9f9a46f858',
        sha256=pin, bytecode_mask='/mnt/capture-bytecode'+str(Path(target).parent.parent))


def package_assets():
    value = settings()
    names = {'review/FINAL-SETTINGS.json', 'review/FINAL-REQUESTS.zlib', 'review/GEMM-original.py', 'review/GEMM-candidate.py'}
    if precision_reference_enabled():
        precision_profile()
        names.add('review/PRECISION-REFERENCE.json')
        if value.get('serving_stack') is not None:names.add('review/SERVING-STACK.json')
    if value['mode'] == MODE and (not precision_reference_enabled() or full_enabled_benchmark()):
        draft_model_binding()
        names.add('review/DRAFT-MODEL.json')
    if settings()['phase'].startswith('sprint01-'):
        # Both sprint arms use the candidate GEMM. Keep the authenticated
        # original as local provenance without shipping an unused live asset.
        names.remove('review/GEMM-original.py')
        names.add('review/ALLOCATION-OVERLAY.json')
        names.update('review/' + row['file'] for row in runtime_overlays())
    if full_enabled_benchmark() or settings()['phase'] in TAIL_PHASES:
        names.add('review/GEMM-tail.py')
    if full_enabled_benchmark() or settings()['phase'] in (GDN_PAIR_PHASE, LIFECYCLE_PHASE):
        names.add('review/gdn_verify_candidate.py')
    if norm_candidate_enabled():
        names.add('review/layernorm_candidate.py')
    names.update({'review/replay_kv_cache_configurator_candidate.py', 'review/replay_memory_pool_candidate.py',
                  'review/capture_hybrid_linear_attn_backend.py'})
    expected = ['GEMM-tail.py', 'GEMM-tail.py'] if value.get('quality_screen') in ('sprint33-sre64k-normal-stop-six-v1', 'sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1') else ['GEMM-reference.py', 'GEMM-tail.py']
    _require(full_enabled_benchmark() and [gemm_for_arm(arm)['file'] for arm in ('control', 'mtp')] == expected, 'bounded replay packet requires bound GEMMs')
    if expected[0] != expected[1]:
        names.add('review/GEMM-reference.py')
    names.remove('review/GEMM-candidate.py')
    return names


def runtime_overlays(arm=None):
    """Exact read-only allocation repairs, separately bound from stock sources."""
    if not settings()['phase'].startswith('sprint01-'):
        return []
    path = Path(__file__).with_name('ALLOCATION-OVERLAY.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16384,
             'allocation overlay manifest missing or oversized')
    value = json.loads(path.read_bytes())
    _require(type(value) is dict and set(value) == {'schema', 'files'} and value['schema'] == 1,
             'allocation overlay manifest schema')
    rows = value['files']
    prefix = '/mnt/composition/image-source/python/sglang/'
    expected = {
        'allocation_sizing': ('srt/mem_cache/allocation_sizing.py',
            'e5dc3befbe93b2908a79fb8f279a643d2c4aeb098ec46f16b740560c04070de1'),
        'dflash_info_v2': ('srt/speculative/dflash_info_v2.py',
            'c66fd2a9629c5f0f36ad4eca34f2910dc8e8af509fd8c6b2fa3fd2722f18004f'),
    }
    if settings()['phase'] in FAIL_FAST_PHASES:
        expected['dflash_worker'] = ('srt/speculative/dflash_worker_v2.py',
            '99a44f45747a2cfc69423193cf8d2745c9a4ac320e3c200e37fe46395208f91f')
    _require(type(rows) is list and len(rows) == len(expected) and
             {row.get('key') for row in rows if type(row) is dict} == set(expected),
             'exact allocation overlay members')
    for row in rows:
        _require(set(row) == {'key', 'file', 'target', 'module', 'original_sha256',
                             'sha256', 'bytecode_mask'}, 'allocation overlay fields')
        relative, original = expected[row['key']]
        _require(row['file'] == row['key'] + '_candidate.py' and
                 row['target'] == prefix + relative and row['original_sha256'] == original and
                 row['module'] == 'sglang.' + relative[:-3].replace('/', '.') and
                 row['bytecode_mask'] == '/mnt/capture-bytecode' + str(Path(prefix + relative).parent),
                 'allocation overlay target or stock provenance')
        candidate = Path(__file__).with_name(row['file'])
        _require(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']) and
                 candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == row['sha256'],
                 'allocation candidate bytes differ')
        if settings()['phase'] in ('sprint01-hip-sampling-forward', 'sprint01-hip-confirmation-reverse') and row['key'] == 'dflash_worker':
            _require(row['sha256'] == '5c802372752a12db893f914d6bc9eb6b680bda03d8d65f86af5a2c9fe92d7933',
                     'HIP sampled worker candidate differs')
        if settings()['phase'] == GDN_PAIR_PHASE and row['key'] == 'dflash_worker':
            _require(row['sha256'] == '8702620c2d6c7101d0682ec21d3dff57f636446620eb9250ba084014a41c810f',
                     'GDN pair worker phase-binding candidate differs')
        if settings()['phase'] == LIFECYCLE_PHASE and row['key'] == 'dflash_worker':
            _require(row['sha256'] == LIFECYCLE_WORKER_SHA256,
                     'lifecycle worker phase-binding candidate differs')
    if ((full_enabled_benchmark() and arm in LOGICAL_ARMS) or (settings()['phase'] == GDN_PAIR_PHASE and
         (arm == 'mtp' or (arm in LOGICAL_ARMS and settings().get('aligned_gdn_both_arms') is True))) or
            (settings()['phase'] == LIFECYCLE_PHASE and arm in LOGICAL_ARMS)):
        candidate = Path(__file__).with_name('gdn_verify_candidate.py')
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == GDN_ALIGNED,
                 'exact aligned GDN candidate differs')
        rows = rows + [dict(key='gdn_verify', file=candidate.name, target=GDN_TARGET,
            module='sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent',
            original_sha256=GDN_STOCK, sha256=GDN_ALIGNED, bytecode_mask=GDN_MASK)]
    if norm_candidate_enabled() and arm == 'mtp':
        candidate = Path(__file__).with_name('layernorm_candidate.py')
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == NORM_CANDIDATE,
                 'exact norm candidate differs')
        rows = rows + [dict(key='midpoint_layernorm', file=candidate.name, target=NORM_TARGET,
            module='sglang.srt.layers.layernorm', original_sha256=NORM_STOCK,
            sha256=NORM_CANDIDATE, bytecode_mask=NORM_MASK)]
    # Both isolated arms receive the same guarded pool source. Only the candidate
    # enables the explicit spec flag; the control retains snapshot behavior.
    if arm in LOGICAL_ARMS:
        rows = rows + [replay_pool_overlay(), replay_hybrid_overlay(), replay_configurator_overlay()]
    return rows


def bytecode_mask_count(arm=None):
    return len(bytecode_mask_paths(arm))


def bytecode_mask_paths(arm=None):
    """Ordered effective roots; overlays can share an existing graph mask."""
    prefix = '/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/srt/'
    return list(dict.fromkeys(
        [prefix + name for name in graph_bytecode_mask_dirs()] + [GEMM_MASK] +
        [row['bytecode_mask'] for row in runtime_overlays(arm)]))


def graph_bytecode_mask_dirs():
    # Empty read-only parent masks already cover both child bytecode paths.
    # Keep the historical nested mount projection for all earlier phases.
    if settings()['phase'] in (GDN_PAIR_PHASE, LIFECYCLE_PHASE, *PRECISION_REFERENCE_PHASES):
        return ('model_executor', 'layers', 'models')
    return ('model_executor', 'model_executor/runner_backend', 'layers', 'layers/attention', 'models')


def gdn_source_for_arm(arm):
    _require(arm in LOGICAL_ARMS and (full_enabled_benchmark() or settings()['phase'] in (GDN_PAIR_PHASE, LIFECYCLE_PHASE)), 'GDN pair arm/phase')
    aligned = (full_enabled_benchmark() or arm == 'mtp' or settings()['phase'] == LIFECYCLE_PHASE or
               settings().get('aligned_gdn_both_arms') is True)
    return dict(path=GDN_TARGET, module='sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent',
        sha256=GDN_ALIGNED if aligned else GDN_STOCK, stock_sha256=GDN_STOCK,
        alignment='beta-bf16' if aligned else 'stock')


def norm_source_for_arm(arm):
    _require(arm in LOGICAL_ARMS and norm_candidate_enabled(), 'norm candidate arm or scope')
    candidate = arm == 'mtp'
    return dict(path=NORM_TARGET, module='sglang.srt.layers.layernorm',
        sha256=NORM_CANDIDATE if candidate else NORM_STOCK, stock_sha256=NORM_STOCK,
        variant=NORM_VARIANT if candidate else 'stock')


def expected_environment(arm):
    _require(arm in LOGICAL_ARMS, 'diagnostic environment arm')
    env = {DRAFT_GRAPH_ENV: '1'}
    if settings()['phase'] in FAIL_FAST_PHASES and contract_for_arm(arm)['enabled']:
        env.update(FP8_DFLASH_FAIL_FAST='1')
    if settings().get('diagnostic_kernel_serialization') == '3':
        env['AMD_SERIALIZE_KERNEL'] = '3'
    return env


def draft_model_binding():
    """Read the bounded draft inventory that is sealed as a package asset."""
    path = Path(__file__).with_name('DRAFT-MODEL.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size < 32768,
             'DFlash2 draft inventory missing or oversized')
    value = json.loads(path.read_bytes())
    _require(type(value) is dict and value.get('schema') == 1 and
             value.get('mount_path') == '/mnt/draft-model' and
             value.get('repository') == 'incoai/Qwen3.8-27B-DFlash2' and
             type(value.get('revision')) is str and re.fullmatch('[0-9a-f]{40}', value['revision']) is not None and
             value.get('host_path') == '/home/developer/fp8-dflash2-draft.' + value['revision'][:12] and
             value.get('runtime_quantization') is None, 'DFlash2 draft binding differs')
    config = value.get('config')
    _require(type(config) is dict and config.get('architectures') == ['DFlash2DraftModel'] and
             config.get('num_hidden_layers') == 5 and config.get('hidden_size') == 5120 and
             config.get('num_key_value_heads') == 8 and config.get('sliding_window') == 2048 and
             config.get('num_target_layers') == 64 and
             config.get('layer_types') == ['sliding_attention'] * 5 and
             config.get('dtype') == 'bfloat16' and type(config.get('dflash_config')) is dict and
             config['dflash_config'].get('block_size') == 8 and
             config['dflash_config'].get('target_layer_ids') == [5, 19, 33, 47, 61],
             'DFlash2 draft architecture differs')
    files = value.get('files')
    _require(type(files) is list and len(files) == 2 and
             {row.get('name') for row in files if type(row) is dict} == {'config.json', 'model.safetensors'} and
             all(type(row.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None and
                 type(row.get('size')) is int and row['size'] > 0 for row in files),
             'DFlash2 draft inventory differs')
    return value


def expected_draft_summary(kind):
    """The exact metadata record shared by the preweight writer and reader."""
    _require(kind in ('none', 'mtp', 'dflash2'), 'unknown speculative kind')
    if kind == 'none':
        return dict(is_draft_model=False, architecture=None, num_nextn_predict_layers=None,
                    text_num_nextn_predict_layers=None)
    if kind == 'mtp':
        return dict(is_draft_model=True, architecture='Qwen3_5ForCausalLMMTP',
                    num_nextn_predict_layers=1, text_num_nextn_predict_layers=1,
                    quantization='fp8', is_draft_quantization_explicit=False)
    config = draft_model_binding()['config']
    fields = ('num_hidden_layers', 'hidden_size', 'num_key_value_heads', 'sliding_window',
              'layer_types', 'num_target_layers', 'dflash_config')
    return dict(is_draft_model=True, architecture='DFlash2DraftModel',
                num_nextn_predict_layers=None, text_num_nextn_predict_layers=None,
                quantization=None, is_draft_quantization_explicit=True, dtype='torch.bfloat16',
                **{name: config[name] for name in fields})

_FIELDS = frozenset({
    "schema", "logical_arm", "enabled", "speculative_algorithm",
    "speculative_draft_model_path", "speculative_draft_model_revision",
    "speculative_num_steps", "speculative_eagle_topk",
    "speculative_num_draft_tokens", "enable_multi_layer_eagle",
    "draft_extend_cuda_graph_disabled", "target_cuda_graph_backend_decode",
    "target_cuda_graph_backend_prefill", "target_cuda_graph_capture_bs",
    "target_cuda_graph_max_bs",
    "speculative_kind", "speculative_draft_model_quantization",
    "speculative_draft_quantization_explicit", "speculative_dflash_block_size",
    "draft_graph_policy", "draft_model_config_sha256", "draft_model_config",
})


class MtpContractError(ValueError):
    """A binding does not describe the reviewed speculative experiment."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise MtpContractError(message)


def contract_for_arm(logical_arm: str) -> dict[str, Any]:
    """Return the explicit speculative kind for one of the fixed logical arms."""
    _require(logical_arm in LOGICAL_ARMS, "unknown logical MTP arm")
    kind = phase_config()[3][('control', 'mtp').index(logical_arm)]
    enabled = kind != 'none'
    dflash = kind == 'dflash2'
    draft = draft_model_binding() if dflash else None
    config_file = next(row for row in draft['files'] if row['name'] == 'config.json') if dflash else None
    width = WIDTH_PHASES.get(settings()['phase'], (8, 8))[('control', 'mtp').index(logical_arm)]
    return {
        "schema": 1,
        "logical_arm": logical_arm,
        "enabled": enabled,
        "speculative_kind": kind,
        "speculative_algorithm": "DFLASH" if dflash else ("EAGLE" if enabled else None),
        "speculative_draft_model_path": draft['mount_path'] if dflash else (MODEL_PATH if enabled else None),
        "speculative_draft_model_revision": draft['revision'] if dflash else (MODEL_REVISION if enabled else None),
        "speculative_num_steps": 1 if enabled else None,
        "speculative_eagle_topk": 1 if enabled else None,
        "speculative_num_draft_tokens": width if dflash else (2 if enabled else None),
        # Explicit CLI unquant resolves to None before ModelConfig sees it.
        "speculative_draft_model_quantization": None if dflash else 'fp8',
        "speculative_draft_quantization_explicit": dflash,
        "speculative_dflash_block_size": width if dflash else None,
        "draft_graph_policy": 'native' if dflash else ('eager' if enabled else 'none'),
        "draft_model_config_sha256": config_file['sha256'] if dflash else None,
        "draft_model_config": draft['config'] if dflash else None,
        # One-step ordinary EAGLE uses the native MTP model rewrite but not
        # the MultiLayerEagle worker. DFLASH keeps its native draft decode
        # graph policy; the shared environment flag only disables draft extend.
        "enable_multi_layer_eagle": False,
        "draft_extend_cuda_graph_disabled": True,
        "target_cuda_graph_backend_decode": "full",
        "target_cuda_graph_backend_prefill": "disabled",
        "target_cuda_graph_capture_bs": [1, 2],
        "target_cuda_graph_max_bs": 2,
    }


def contract_for_owner_arm(owner_arm: str) -> dict[str, Any]:
    """Map the fixed benchmark owner names to the logical experiment arms."""
    _require(owner_arm in OWNER_ARMS, "unknown owner arm")
    return contract_for_arm(owner_arm_map()[owner_arm])


def owner_arm_map() -> dict[str, str]:
    """Bind the fixed owner's start order to the frozen logical arm order."""
    order = settings().get('execution_order', ['control', 'mtp'])
    return dict(zip(OWNER_ARMS, order))


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``value`` is byte-for-byte equivalent in structure."""
    _require(type(value) is dict and set(value) == _FIELDS, "MTP contract fields differ")
    logical_arm = value.get("logical_arm")
    _require(type(logical_arm) is str and logical_arm in LOGICAL_ARMS, "MTP logical arm differs")
    expected = contract_for_arm(logical_arm)
    _require(value == expected, "MTP contract values differ")
    return dict(expected)


def serving_stack(arm=None):
    value=settings()
    _require(value.get('serving_stack')=='torch214-sdk102-compat-v1','explicit serving stack required')
    path=Path(__file__).with_name('SERVING-STACK.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size<=2<<20,'bounded serving stack')
    stack=json.loads(path.read_bytes())
    _require(set(stack)=={'schema','torch_candidate','compatibility','environment','worker_seconds','client_seconds','reasoning_effort','runtime_path','runtime_root','execution_modes','shared_memory_bytes','scratch_transfer_bytes','toolchain','toolchain_environment'} and stack['schema']==1,'serving stack fields')
    candidate=stack['torch_candidate'];compat=stack['compatibility']
    _require(candidate['version']=='2.14.0+git2e66fe81.rocm10.2.0' and candidate['sdk']['version']=='10.2.0.dev20261020+git535d2990','current stack identity')
    _require(re.fullmatch(r'/home/developer/fp8-serving-torch214\.[0-9a-f]{16}/sitepackages',compat['sitepackages']) and compat['sitepackages']==compat['root']+'/sitepackages','private compatibility prefix')
    _require(stack['environment']=={'TORCH_BLAS_PREFER_HIPBLASLT':'1','ROCBLAS_USE_HIPBLASLT':'-1'} and stack['worker_seconds']==1800 and stack['client_seconds']==1740 and stack['reasoning_effort']=='low','low thinking resource/settings contract')
    _require(stack['runtime_path']=='/usr/local/lib/python3.14/dist-packages/_rocm_sdk_core/bin:/mnt/work/build-venv/bin:/usr/local/bin:/usr/bin:/bin','current SDK executable PATH')
    _require(re.fullmatch(r'/home/developer/fp8-serving-runtime-torch214\.[0-9a-f]{16}/sitepackages',stack['runtime_root']) and type(stack['execution_modes']) is dict and stack['execution_modes'].get('_rocm_sdk_core/bin/rocminfo')==0o555,'executable SDK clone binding')
    _require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and 'bin' in Path(n).parts[:-1] and mode==0o555 for n,mode in stack['execution_modes'].items()),'executable tool modes only')
    _require(stack['shared_memory_bytes']==512<<20 and stack['scratch_transfer_bytes']==256<<20,'bounded RCCL scratch redistribution')
    tc=stack['toolchain']
    _require(set(tc)=={'root','manifest','sdk_version'} and re.fullmatch(r'/home/developer/fp8-serving-devel-torch214\.[0-9a-f]{16}/devel',tc['root']) and tc['sdk_version']==candidate['sdk']['version'],'selected read-only development view')
    _require(set(tc['manifest'])=={'path','sha256','bytes'} and tc['manifest']['path']==str(Path(tc['root']).parent/'DEVELOPMENT-MANIFEST.json') and re.fullmatch('[0-9a-f]{64}',tc['manifest']['sha256']) and 0<tc['manifest']['bytes']<=16<<20,'development manifest binding')
    _require(stack['toolchain_environment']=={'ROCM_HOME':'/mnt/rocm-devel','ROCM_PATH':'/mnt/rocm-devel','HIP_CLANG_PATH':'/mnt/rocm-devel/lib/llvm/bin'},'JIT toolchain environment')
    mapping=value.get('performance_benchmark',{}).get('stack_by_arm',{})
    if value.get('quality_screen') in ('sprint7-low-thinking-normal-stop-four-v1', 'sprint33-sre64k-normal-stop-six-v1', 'sprint35-long-prefix-reuse-v1', 'sprint36-chunk-prefill-v1', 'sprint39-paired-serving-v1'):
        mapping = {'control':'current', 'mtp':'current'}
    _require(arm is None or arm in mapping,'performance stack arm')
    return dict(stack,is_baseline=arm is not None and mapping[arm]=='baseline')


def prefill_chunk_size(arm):
    _require(arm in LOGICAL_ARMS, 'unknown chunk arm')
    return 4096 if settings().get('quality_screen') == 'sprint36-chunk-prefill-v1' and arm == 'mtp' else 2048


def full_enabled_benchmark():
    return settings().get('full_enabled_benchmark')=='dflash8-splitkv-aligned-gdn-v1'


def commit_observation_limit():
    """Frozen four-response cap plus startup/readiness allowance."""
    if betterbench_checkpoint_enabled():return 44*4096+9*16+4096
    value=settings()
    if value.get('phase') == PAIRED_PHASE:
        caps = sum(row['maximum_output_tokens'] for row in value['requests'])
        _require(caps == 19 * 1024, 'S39 output cap sum differs')
        return caps + 4096
    if value.get('quality_screen') == 'sprint36-chunk-prefill-v1':
        caps = sum(row['maximum_output_tokens'] for row in value['requests'])
        _require(caps == 3 * 4096, 'S36 output cap sum differs')
        return caps + 4096
    if value.get('quality_screen') == 'sprint35-long-prefix-reuse-v1':
        caps = sum(row['maximum_output_tokens'] for row in value['requests'])
        _require(caps == 6 * 4096, 'S35 output cap sum differs')
        return caps + 4096
    if value.get('quality_screen') == 'sprint33-sre64k-normal-stop-six-v1':
        caps = sum(row['maximum_output_tokens'] for row in value['requests'])
        _require(caps == 7 * 4096, 'S32 output cap sum differs')
        return caps + 4096
    _require(value.get('performance_benchmark',{}).get('mode')=='m2047-fixed1024-c1-v1', 'fixed-work counter scope differs')
    caps=sum(row['maximum_output_tokens'] for row in value['requests'])
    _require(caps==4096, 'fixed-work output cap sum differs')
    return caps+4096


def paired_serving_enabled():
    if betterbench_checkpoint_enabled():return False
    return settings()['phase'] == PAIRED_PHASE


def paired_group_specs():
    """The bounded nineteen-request schedule, independent of mutable input."""
    groups = [dict(id='warmup-c1',concurrency=1,kind='warmup',ordinals=[0]),
              dict(id='warmup-c2',concurrency=2,kind='warmup',ordinals=[1,2])]
    for concurrency, start in ((1,3),(2,9)):
        for number in range(3):
            groups.append(dict(id=f'timing-c{concurrency}-{number+1}',concurrency=concurrency,
                kind='timing',ordinals=[start+2*number,start+2*number+1]))
    return groups + [dict(id='quality-c1',concurrency=1,kind='quality',ordinals=[15,16]),
                     dict(id='quality-c2',concurrency=2,kind='quality',ordinals=[17,18])]


def validate_paired_settings(value):
    protocol = value.get('paired_serving')
    groups = paired_group_specs()
    orders = {'control':[g['id'] for g in groups],
              'mtp':[g['id'] for g in groups[:2]+groups[5:8]+groups[2:5]+[groups[9],groups[8]]]}
    expected = dict(mode='sprint39-c1-c2-pairs-v1',groups=groups,order=orders,
        reserve_per_request=16,context_length=32768,token_pool_capacity=32800,requests_per_start=19,
        measured_fixed_requests_per_start=12,natural_quality_requests_per_start=4,warmup_requests_per_start=3,
        throughput_scope='instrumented HTTP service throughput; combined TP2',
        group_seconds='max(received_monotonic)-min(prepared_monotonic)',historical_uninstrumented_comparison=False)
    _require(type(protocol) is dict and json.dumps(protocol, sort_keys=True) == json.dumps(expected, sort_keys=True), 'S39 frozen group/order/measurement contract differs')
    rows=value['requests']
    _require(type(rows) is list and len(rows)==19 and value['execution_order']==['control','mtp'], 'S39 count/order')
    for group in groups:
        for index,ordinal in enumerate(group['ordinals']):
            row=rows[ordinal]
            name=group['id']+('-'+chr(97+index) if len(group['ordinals'])==2 else '')
            _require(type(row['ordinal']) is int and row['ordinal']==ordinal and row['case']==name and row['group']==group['id'] and
                row['category']==group['kind'] and row['input_tokens']==(5461 if group['kind']=='quality' else 2047) and
                row['maximum_output_tokens']==1024 and row['internal_reserve_tokens']==16 and
                row['context_envelope_tokens']==32768, 'S39 frozen request shape/case differs')
        envelope=sum(rows[i]['input_tokens']+rows[i]['maximum_output_tokens']+16 for i in group['ordinals'])
        _require(envelope <= protocol['token_pool_capacity'], 'S39 paired token capacity exceeded')


def paired_groups(arm):
    value=settings()
    _require(value['phase']==PAIRED_PHASE and arm in LOGICAL_ARMS, 'S39 group phase/arm')
    groups={g['id']:g for g in value['paired_serving']['groups']}
    return [groups[name] for name in value['paired_serving']['order'][arm]]


def paired_group_for(ordinal):
    _require(type(ordinal) is int and 0<=ordinal<19, 'S39 request ordinal')
    return next(group for group in paired_groups('control') if ordinal in group['ordinals'])


def betterbench_checkpoint_enabled():
    return True  # Explicit selected-baseline adapter; no operation authority.


def host_path_cleanup_enabled(arm):
    """Select the host-only candidate within the exact BetterBench checkpoint."""
    _require(arm in LOGICAL_ARMS, 'unknown host-path cleanup arm')
    return betterbench_checkpoint_enabled() and arm == 'mtp'


def expected_decode_log_interval(arm):
    return 100 if host_path_cleanup_enabled(arm) else 1
