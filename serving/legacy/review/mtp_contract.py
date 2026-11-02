"""Pure, source-bound contract for the selected W5 paired-serving phase.

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
MODEL_PATH = "/mnt/models"
DRAFT_GRAPH_ENV = "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH"
OWNER_ARMS = {"bf16": "control", "kv4": "mtp"}
LOGICAL_ARMS = frozenset(OWNER_ARMS.values())
PRODUCTION_EVIDENCE_FILES = ('serving_graph_observer.py', 'j01_probe.py',
    'observer_health.py', 'target_kv_metadata.py', 'kv_graph_join.py', 'serving_parity.py')
PRODUCTION_EVIDENCE_MASK_NAME = 'PRODUCTION-EVIDENCE-MASK.py'
PRODUCTION_EVIDENCE_MASK_BYTES = b'raise ModuleNotFoundError("private evidence helpers are unavailable in production")\n'
TAIL = 'b101e71751ce995b23b327dd3ee87662eb0efd3ef7dc2d35387d1acc58d8e069'
GEMM_TARGET = '/mnt/composition/image-source/python/sglang/kernels/ops/gemm/fp8_kernel.py'
GEMM_MASK = '/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/kernels/ops/gemm'
# Comparison markers retained for consumers awaiting migration; never selectable.
GDN_PAIR_PHASE = 'sprint01-gdn-aligned-pair'
LIFECYCLE_PHASE = 'sprint01-gdn-aligned-lifecycle'
PAIRED_PHASE = 'sprint01-paired-serving'
PRECISION_REFERENCE_PHASES = frozenset({PAIRED_PHASE})
CLOSURE_PHASES = frozenset()
BATCH_PHASES = frozenset({PAIRED_PHASE})
CACHE_PHASES = BATCH_PHASES
HISTORY_PHASES = frozenset()
GDN_STOCK = '7cad2a1b690b15c8d13f15cf1e458a7fba85f2bca9e47327a9804a1e7ba92943'
GDN_ALIGNED = '0ae8788fc8194505fb6a0ee5e320456f5fb9ebf45d4b83a4d28e2e196ec15b87'
GDN_TARGET = '/mnt/composition/image-source/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py'
GDN_MASK = '/mnt/capture-bytecode' + str(Path(GDN_TARGET).parent)
NORM_STOCK = 'cb87d85a89fdd06c5ab2f362e9f06b11dc32a6b5519440805bde038172e4c82b'
NORM_TARGET = '/mnt/composition/image-source/python/sglang/srt/layers/layernorm.py'
NORM_MASK = '/mnt/capture-bytecode' + str(Path(NORM_TARGET).parent)
NORM_P3_CANDIDATE = 'f1855c99386c39802b07dd1848b0d9197994f67181cd1265dbf85b02da513b42'
NORM_P3_VARIANT = 'gemma-rmsnorm-triton-fp32-native-order-v1'
D3B_TARGET_RUNNER = '5e2325bb08930cdffa55cc9d944cb1dc589b662c42de821eb391c82c8bdf37de'
D3B_NATIVE_RUNNER = 'd8ce6f38279913d11b4a7e910a7fd1432e952a0bdcf33ebe726b6f49af0ba6ba'
D3B_RUNNER_TARGET = '/mnt/composition/image-source/python/sglang/srt/model_executor/model_runner.py'
D3B_TARGET_FILES = ('review/d3b_target_head_adapter.py', 'review/d3b_head_launches.py', 'review/d3b_head_fp8_ops.py', 'review/d3b_head_fp8_kernels.py')
PHASES = {PAIRED_PHASE: (131040, 2040, 176, ('dflash2', 'dflash2'), ('tail', 'tail'))}
TAIL_PHASES = frozenset()
FAIL_FAST_PHASES = frozenset()


def settings():
    path = Path(__file__).with_name('FINAL-SETTINGS.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size < 262144,
             'final settings missing or oversized')
    value = json.loads(path.read_bytes())
    _require(type(value) is dict and value.get('phase') == PAIRED_PHASE,
             'only the selected paired-serving phase is supported')
    fields = {'schema', 'mode', 'phase', 'requests', 'execution_order',
              'precision_reference', 'serving_stack', 'quality_screen',
              'screen_policy', 'full_enabled_benchmark', 'paired_serving'}
    allowed = (fields, fields | {'p1_prefill'}, fields | {'p1_prefill', 'd3b_target_head'},
               fields | {'p1_prefill', 'd3b_target_head', 'gdn_layout_both', 'd3b_draft_head_arm'})
    _require(set(value) - {'instrumentation'} in allowed or
             set(value) - {'allocator_snapshot', 'instrumentation'} in allowed,
             'final settings fields')
    _require(value['schema'] == 1 and value['mode'] == MODE, 'final phase binding')
    if 'instrumentation' in value:
        _require(value['instrumentation'] in ('on', 'production'), 'instrumentation mode differs')
    _require(value['serving_stack'] == 'torch214-sdk102-compat-v1', 'serving stack variant')
    _require(type(value['precision_reference']) is str and
             value['precision_reference'] in ('fp8', 'mxfp4'), 'S39 phase/precision differs')
    if 'allocator_snapshot' in value:
        _require(value['allocator_snapshot'] is True, 'allocator snapshot flag/phase differs')
    _require(value['execution_order'] in (['control', 'mtp'], ['mtp', 'control']),
             'frozen execution order differs')
    if 'p1_prefill' in value:
        _require(value['precision_reference'] == 'mxfp4' and
                 value['p1_prefill'] == 'fp8-g128-b101-m1024-v1', 'P1 prefill scope differs')
    if 'd3b_target_head' in value:
        _require(value['precision_reference'] == 'mxfp4' and
                 value.get('p1_prefill') == 'fp8-g128-b101-m1024-v1' and
                 value['d3b_target_head'] == 'fp8-fp32-logits-verify-rows-v1',
                 'D3b target head scope differs')
    if 'gdn_layout_both' in value:
        _require(value['precision_reference'] == 'mxfp4' and
                 value.get('p1_prefill') == 'fp8-g128-b101-m1024-v1' and
                 value.get('d3b_target_head') == 'fp8-fp32-logits-verify-rows-v1' and
                 value['gdn_layout_both'] is True and
                 value.get('d3b_draft_head_arm') in ('control', 'mtp', 'both'),
                 'current drafter-head composition scope differs')
    _require(value['full_enabled_benchmark'] == 'dflash8-splitkv-aligned-gdn-v1' and
             value['screen_policy'] == 'exploratory-stop-or-length-v1' and
             value['quality_screen'] == 'normal-stop-fp8-mxfp4-v1',
             'full-enabled screen protocol')
    validate_paired_settings(value)
    rows = value['requests']
    assets = _assets()
    _require(type(rows) is list and len(rows) == PHASES[PAIRED_PHASE][2] and
             [row['ordinal'] for row in rows] == list(range(len(rows))), 'final request order/count')
    _require(len({row['case'] for row in rows}) == len(rows), 'duplicate final request')
    inventory = set()
    for row in rows:
        _require('generated_history' not in row, 'generated history parent/phase differs')
        for field, pin in (('path', 'sha256'), ('expected', 'expected_sha256')):
            if field not in row:
                _require(field == 'expected' and row['ordinal'] < 3, 'missing frozen vector')
                continue
            relative = row[field]
            _require(type(relative) is str and relative == Path(relative).name and
                     relative.startswith('FINAL-REQUEST-'), 'request asset outside private inventory')
            _require(relative in assets and len(assets[relative]) <= 1 << 20,
                     'request asset missing or oversized')
            _require(hashlib.sha256(assets[relative]).hexdigest() == row[pin], 'request asset hash mismatch')
            inventory.add(relative)
    _require(set(assets) == inventory, 'exact compressed asset inventory')
    return value


def _assets():
    return _decoded_assets(logical_asset('FINAL-REQUESTS.zlib',
        REQUEST_INHERITED['sha256'], 2 << 20))


@lru_cache(maxsize=1)
def _decoded_assets(compressed):
    # Cache by exact compressed bytes, never by path or an unchecked timestamp.
    # Every settings call still reads the bound file and verifies member hashes.
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, (16 << 20) + 1)
    _require(len(raw) <= 16 << 20 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
             'frozen asset expanded size or trailing content')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate frozen asset')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs)
    _require(type(value) is dict and len(value) == 352, 'frozen asset count')
    return {key: base64.b64decode(item, validate=True) for key, item in value.items()}


def asset_bytes(name):
    return _assets()[name]


def phase_config():
    return PHASES[settings()['phase']]


def prompt_minimization_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def precision_reference_enabled():
    return settings()['phase'] in PRECISION_REFERENCE_PHASES


def precision_profile():
    """One immutable checkpoint profile shared by both ordinary serving starts."""
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
    if value['precision_reference'] == 'mxfp4':
        fields.add('cache_profile')
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
    expected_quantization = {'fp8': 'fp8', 'mxfp4': 'quark'}[profile['precision']]
    if profile['precision'] == 'mxfp4':
        _require(profile['repository'] == 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4' and
                 profile['revision'] == '5233554c5fa56afda40150556b95573c2d7d29c0' and
                 profile['config_sha256'] == '4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55' and
                 profile['cache_profile'] == {'target': 'bf16', 'draft': 'bf16'},
                 'published serialized MXFP4 identity/cache profile differs')
    _require(profile['quantization'] == expected_quantization and
             type(profile['memory_fraction_static']) is float and profile['memory_fraction_static'] == .92 and
             all(type(profile[key]) is int and profile[key] == expected for key, expected in
                 (('context_length', 131040), ('max_total_tokens', 262144), ('seed', 12345))),
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


MXFP4_PRIVATE_SOURCES = {'mxfp4_kernels': ('kernels.py',
                   'mxfp4_component.kernels',
                   '5571b10654697561455081455b6cd28d43314702f56c80160ef2e33e535b3693'),
 'mxfp4_ops': ('ops.py',
               'mxfp4_component.ops',
               '66d25968516c6c9509923ad363a291d57d192918f1e7121f96481202391c3776'),
 'mxfp4_package': ('__init__.py',
                   'mxfp4_component',
                   '39cd59ab506e69df4cb77664d67392e6c510174be5eeb61c172e291b47c59539'),
 'mxfp4_prefill_reference': ('prefill_reference.py',
                             'mxfp4_component.prefill_reference',
                             'cd5841650fc59da7f1537e82ffebf16fb07ca0db7d191cdfec59540575bbd887'),
 'mxfp4_scheme': ('isolated_scheme.py',
                  'mxfp4_component.isolated_scheme',
                  'f396186d36fa83812d7d3d4ec538e47588d3513a06e9996539e1481bebe8c185'),
 'mxfp4_selection': ('private_selection.py',
                     'mxfp4_component.private_selection',
                     'e0120a933d0848de1fd82135274deac4ca850af82715112a2a1bfa94107c52ed')}
MXFP4_NATIVE_SOURCES = {
    'quark_config': (
        'srt/layers/quantization/quark/quark.py',
        '1632fadb10f1ea5f2e77e905c654e0b61e3d7c563198389011488722dbdeeef3',
    ),
    'quark_mxfp4_scheme': (
        'srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py',
        '52f81e669c769f6fc1d7ae8195f7e203c783952e8ce00879eafb491603266436',
    ),
    'quark_parameter': (
        'srt/layers/parameter.py',
        '0e5b09e820b9fd3ad2d5d36107f55861e50c8226dea56fb27d0edb09fbe32330',
    ),
}


def mxfp4_sources(sources):
    """Exact private adapter and native loader classes consumed by this branch."""
    result = {}
    for role, (filename, module, digest) in MXFP4_PRIVATE_SOURCES.items():
        wanted = dict(path='/mnt/review/' + filename, module=module, sha256=digest)
        _require(sources.get(role) == wanted, 'MXFP4 private source differs: ' + role)
        result[role] = wanted
    for role, (relative, digest) in MXFP4_NATIVE_SOURCES.items():
        wanted = dict(path='/mnt/composition/image-source/python/sglang/' + relative,
                      module='sglang.' + relative[:-3].replace('/', '.'), sha256=digest)
        _require(sources.get(role) == wanted, 'MXFP4 native source differs: ' + role)
        result[role] = wanted
    loading = precision_loading_sources(sources)
    result.update({role: loading[role] for role in ('model_loader', 'loader_weight_utils')})
    return result


def dense_mxfp4_enabled():
    return settings().get('precision_reference') == 'mxfp4'


def cache_profile(arm):
    _require(arm in LOGICAL_ARMS, 'unknown cache-profile arm')
    if dense_mxfp4_enabled():
        _require(precision_profile()['cache_profile']==dict(target='bf16',draft='bf16'),
                 'MXFP4 cache profile changed')
    return dict(target='bf16',draft='bf16')


def mxfp4_linear_layout(rank):
    """Loaded projection of the pinned dense loader's complete TP2 plan.

    The preparation test compares every path/shape to loader_layout.all_plans.
    The serialized checkpoint has 496 weights; fusion yields 304 per rank.
    """
    _require(type(rank) is int and rank in (0, 1), 'dense TP2 rank')
    result = []
    for layer in range(64):
        rows = [('mlp.gate_up_proj', 17408, 5120, 'MergedColumnParallelLinear'),
                ('mlp.down_proj', 5120, 8704, 'RowParallelLinear')]
        if layer % 4 != 3:
            rows += [('linear_attn.in_proj_qkvz', 8192, 5120, 'MergedColumnParallelLinear'),
                     ('linear_attn.in_proj_ba', 48, 5120, 'MergedColumnParallelLinear'),
                     ('linear_attn.out_proj', 5120, 3072, 'RowParallelLinear')]
        else:
            rows += [('qkv_proj', 7168, 5120, 'QKVParallelLinear'),
                     ('o_proj', 5120, 3072, 'RowParallelLinear')]
        for suffix, n, k, cls in rows:
            result.append(dict(path=f'model.layers.{layer}.{suffix}', class_name=cls,
                               logical_shape=[n, k], weight_shape=[n, k // 2],
                               scale_shape=[n, k // 32]))
    return result


def target_quantization():
    return precision_profile()['quantization']


def target_model_revision():
    return precision_profile()['revision']


def target_config_sha256():
    return precision_profile()['config_sha256']


def memory_fraction_static():
    return precision_profile()['memory_fraction_static']


def serving_seed_required():
    return precision_reference_enabled()


def verify_attention_ab_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def verify_attention_chain_enabled():
    return full_enabled_benchmark()


def gdn_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def norm_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def norm_candidate_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def verify_attention_observed(arm):
    return verify_attention_ab_enabled() or (verify_attention_chain_enabled() and arm in LOGICAL_ARMS)


def greedy_quality_draft_ab():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def j01_probe_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def head_input_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def final_norm_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def midpoint_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def raw_attention_capture_enabled():
    # Retained API for consumers awaiting their separate source migration.
    settings()
    return False


def cache_config():
    settings()
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
    _require(arm in LOGICAL_ARMS, 'unknown memory-pool arm')
    profile = precision_profile()
    return dict(context_length=profile['context_length'], max_running_requests=2,
                max_total_tokens=profile['max_total_tokens'], kv_cache_dtype=cache_profile(arm)['target'],
                max_mamba_cache_size=19)


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
    _require(settings()['phase'] in BATCH_PHASES, 'lifecycle recurrent geometry phase')
    return 19


def pool_token_capacity(arm=None):
    _require(arm is None or arm in LOGICAL_ARMS, 'unknown memory-pool arm')
    return precision_profile()['max_total_tokens']


def gemm_for_arm(arm):
    _require(arm in LOGICAL_ARMS, 'unknown source arm')
    settings()
    return dict(file='GEMM-tail.py', sha256=TAIL, target=GEMM_TARGET, bytecode_mask=GEMM_MASK)


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
    paths = [result[i+1] for i in range(len(result)-1)
             if result[i]=='--tmpfs' and result[i+1].startswith('/mnt/capture-bytecode/')]
    return _consolidate_bytecode_argv(result, paths)


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
    precision_profile()
    draft_model_binding()
    names = {'review/FINAL-SETTINGS.json', 'review/FINAL-REQUESTS.zlib',
             'review/PRECISION-REFERENCE.json', 'review/SERVING-STACK.json',
             'review/DRAFT-MODEL.json', 'review/ALLOCATION-OVERLAY.json',
             'review/GEMM-tail.py', 'review/gdn_verify_candidate.py',
             'review/replay_kv_cache_configurator_candidate.py',
             'review/replay_memory_pool_candidate.py', 'review/capture_hybrid_linear_attn_backend.py'}
    names.update('review/' + row['file'] for row in runtime_overlays())
    if dense_mxfp4_enabled():
        names.update('review/' + row[0] for row in MXFP4_PRIVATE_SOURCES.values())
        names.add('review/layernorm_p3_candidate.py')
        if value.get('gdn_layout_both') is True:
            names.update(('review/qwen3_5_candidate.py', 'review/d3b_head_adapter.py', 'review/d3b_combined_model_runner.py'))
        if value.get('p1_prefill') is not None:
            names.update(('review/p1_prefill.py', 'review/p1_fp8_kernels.py'))
        if value.get('d3b_target_head') is not None:
            names.update(D3B_TARGET_FILES)
    names.difference_update('review/' + name for name in PACKED_REQUEST_NAMES if name != 'FINAL-REQUESTS.zlib')
    return names


def runtime_overlays(arm=None):
    """Exact read-only allocation repairs, separately bound from stock sources."""
    settings()
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
    expected.update({'triton_backend': ('srt/layers/attention/triton_backend.py', '9924224f0dd4150fe1fc9ef7de6a556c49857608792479379179a973436762c1'), 'decode_attention_fwd': ('kernels/ops/attention/decode_attention.py', '34ca0066e2716e1ffad048ec7cbb3a25af13eeb61e99034a663fed6277bf691a'), 'extend_attention_fwd': ('kernels/ops/attention/extend_attention.py', 'f061599bb2c9243d50b565281c47de077ccc90135b9b1e5320a07425bedac112')})
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
    if arm in LOGICAL_ARMS:
        candidate = Path(__file__).with_name('gdn_verify_candidate.py')
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == GDN_ALIGNED,
                 'exact aligned GDN candidate differs')
        rows = rows + [dict(key='gdn_verify', file=candidate.name, target=GDN_TARGET,
            module='sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent',
            original_sha256=GDN_STOCK, sha256=GDN_ALIGNED, bytecode_mask=GDN_MASK)]
    if dense_mxfp4_enabled() and arm in LOGICAL_ARMS:
        # Screen 6: the P3 fused Gemma RMSNorm replaces the stock module on the dense
        # MXFP4 arms. Kernel body verbatim from the sealed P3 component (operation 87).
        candidate = Path(__file__).with_name('layernorm_p3_candidate.py')
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == NORM_P3_CANDIDATE,
                 'exact P3 norm candidate differs')
        rows = rows + [dict(key='dense_layernorm', file=candidate.name, target=NORM_TARGET,
            module='sglang.srt.layers.layernorm', original_sha256=NORM_STOCK,
            sha256=NORM_P3_CANDIDATE, bytecode_mask=NORM_MASK)]
    if dense_mxfp4_enabled() and arm in LOGICAL_ARMS and settings().get('d3b_target_head') is not None:
        # Preserve the current capture runner; install the target head before graph capture.
        # 'both': the candidate quality menu runs the FP8 drafter head in both starts.
        draft_head = settings().get('d3b_draft_head_arm') in (arm, 'both')
        runner_name = 'd3b_combined_model_runner.py' if draft_head else 'capture_model_runner.py'
        runner_pin = 'da81d4c393cf5910753c48036bedbe593e851ffe0985997f4eca02d79737df76' if draft_head else D3B_TARGET_RUNNER
        candidate = Path(__file__).with_name(runner_name)
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == runner_pin,
                 'exact D3b target runner overlay differs')
        rows = rows + [dict(key='model_runner', file=candidate.name, target=D3B_RUNNER_TARGET,
            module='sglang.srt.model_executor.model_runner', original_sha256=D3B_NATIVE_RUNNER,
            sha256=runner_pin, bytecode_mask='/mnt/capture-bytecode' + str(Path(D3B_RUNNER_TARGET).parent))]
    if dense_mxfp4_enabled() and arm in LOGICAL_ARMS and settings().get('gdn_layout_both') is True:
        # Unchanged component115 / serving117 copy-only guard.
        candidate = Path(__file__).with_name('qwen3_5_candidate.py')
        _require(candidate.is_file() and not candidate.is_symlink() and
                 hashlib.sha256(candidate.read_bytes()).hexdigest() == '2a76ddea9a380dd8e8fd71a8919fc4aadae702ec5c5f35bd052c614ed07a78e8',
                 'exact GDN layout model candidate differs')
        target = '/mnt/composition/image-source/python/sglang/srt/models/qwen3_5.py'
        rows = rows + [dict(key='head_model', file=candidate.name, target=target,
            module='sglang.srt.models.qwen3_5', original_sha256='9ccff88d991fc492daff984e73b5f88fb4daac62d19a2c19c6c686c9bbe5499c',
            sha256='2a76ddea9a380dd8e8fd71a8919fc4aadae702ec5c5f35bd052c614ed07a78e8',
            bytecode_mask='/mnt/capture-bytecode' + str(Path(target).parent))]
    # Both isolated arms receive the same guarded pool source. Only the candidate
    # enables the explicit spec flag; the control retains snapshot behavior.
    if arm in LOGICAL_ARMS:
        rows = rows + [replay_pool_overlay(), replay_hybrid_overlay(), replay_configurator_overlay()]
    if arm in LOGICAL_ARMS:
        rows = [row for row in rows if row['key'] not in ('triton_backend', 'decode_attention_fwd', 'extend_attention_fwd')]
    return rows


def bytecode_mask_count(arm=None):
    return len(bytecode_mask_paths(arm))


def bytecode_mask_paths(arm=None):
    """Ordered effective roots; overlays can share an existing graph mask."""
    prefix = '/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/srt/'
    return _effective_bytecode_masks(list(dict.fromkeys(
        [prefix + name for name in graph_bytecode_mask_dirs()] + [GEMM_MASK] +
        [row['bytecode_mask'] for row in runtime_overlays(arm)])))


def graph_bytecode_mask_dirs():
    settings()
    return ('model_executor', 'layers', 'models')


def gdn_source_for_arm(arm):
    _require(arm in LOGICAL_ARMS and full_enabled_benchmark(), 'GDN pair arm/phase')
    return dict(path=GDN_TARGET, module='sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent',
                sha256=GDN_ALIGNED, stock_sha256=GDN_STOCK, alignment='beta-bf16')


def norm_source_for_arm(arm):
    _require(arm in LOGICAL_ARMS and norm_candidate_enabled(), 'norm candidate arm or scope')


def expected_environment(arm):
    _require(arm in LOGICAL_ARMS, 'diagnostic environment arm')
    settings()
    return {DRAFT_GRAPH_ENV: '1'}


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
    _require(kind == 'dflash2', 'only the selected DFlash2 speculative kind is supported')
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
    """Return the selected DFlash8 contract for either logical arm."""
    _require(logical_arm in LOGICAL_ARMS, "unknown logical MTP arm")
    phase_config()
    draft = draft_model_binding()
    config_file = next(row for row in draft['files'] if row['name'] == 'config.json')
    return {
        "schema": 1,
        "logical_arm": logical_arm,
        "enabled": True,
        "speculative_kind": 'dflash2',
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": draft['mount_path'],
        "speculative_draft_model_revision": draft['revision'],
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 8,
        # Explicit CLI unquant resolves to None before ModelConfig sees it.
        "speculative_draft_model_quantization": None,
        "speculative_draft_quantization_explicit": True,
        "speculative_dflash_block_size": 8,
        "draft_graph_policy": 'native',
        "draft_model_config_sha256": config_file['sha256'],
        "draft_model_config": draft['config'],
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
    _require(settings()['serving_stack']=='torch214-sdk102-compat-v1','explicit serving stack required')
    path=Path(__file__).with_name('SERVING-STACK.json')
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size<=2<<20,'bounded serving stack')
    stack=json.loads(path.read_bytes())
    _require(set(stack)=={'schema','torch_candidate','compatibility','environment','worker_seconds','client_seconds','reasoning_effort','runtime_path','runtime_root','execution_modes','shared_memory_bytes','scratch_transfer_bytes','toolchain','toolchain_environment'} and stack['schema']==1,'serving stack fields')
    candidate=stack['torch_candidate'];compat=stack['compatibility']
    _require(candidate['version']=='2.14.0+git2e66fe81.rocm10.2.0' and candidate['sdk']['version']=='10.2.0.dev20261020+git535d2990','current stack identity')
    _require(re.fullmatch(r'/home/developer/fp8-serving-torch214\.[0-9a-f]{16}/sitepackages',compat['sitepackages']) and compat['sitepackages']==compat['root']+'/sitepackages','private compatibility prefix')
    _require(stack['environment']=={'TORCH_BLAS_PREFER_HIPBLASLT':'1','ROCBLAS_USE_HIPBLASLT':'-1'} and stack['worker_seconds']==2280 and stack['client_seconds']==2040 and stack['reasoning_effort']=='low','low thinking resource/settings contract')
    _require(stack['runtime_path']=='/usr/local/lib/python3.14/dist-packages/_rocm_sdk_core/bin:/mnt/work/build-venv/bin:/usr/local/bin:/usr/bin:/bin','current SDK executable PATH')
    _require(re.fullmatch(r'/home/developer/fp8-serving-runtime-torch214\.[0-9a-f]{16}/sitepackages',stack['runtime_root']) and type(stack['execution_modes']) is dict and stack['execution_modes'].get('_rocm_sdk_core/bin/rocminfo')==0o555,'executable SDK clone binding')
    _require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and 'bin' in Path(n).parts[:-1] and mode==0o555 for n,mode in stack['execution_modes'].items()),'executable tool modes only')
    _require(stack['shared_memory_bytes']==512<<20 and stack['scratch_transfer_bytes']==256<<20,'bounded RCCL scratch redistribution')
    tc=stack['toolchain']
    _require(set(tc)=={'root','manifest','sdk_version'} and re.fullmatch(r'/home/developer/fp8-serving-devel-torch214\.[0-9a-f]{16}/devel',tc['root']) and tc['sdk_version']==candidate['sdk']['version'],'selected read-only development view')
    _require(set(tc['manifest'])=={'path','sha256','bytes'} and tc['manifest']['path']==str(Path(tc['root']).parent/'DEVELOPMENT-MANIFEST.json') and re.fullmatch('[0-9a-f]{64}',tc['manifest']['sha256']) and 0<tc['manifest']['bytes']<=16<<20,'development manifest binding')
    _require(stack['toolchain_environment']=={'ROCM_HOME':'/mnt/rocm-devel','ROCM_PATH':'/mnt/rocm-devel','HIP_CLANG_PATH':'/mnt/rocm-devel/lib/llvm/bin'},'JIT toolchain environment')
    _require(arm is None or arm in LOGICAL_ARMS, 'performance stack arm')
    return dict(stack, is_baseline=False)


def prefill_chunk_size(arm):
    _require(arm in LOGICAL_ARMS, 'unknown chunk arm')
    settings()
    return 2048


def full_enabled_benchmark():
    return settings().get('full_enabled_benchmark')=='dflash8-splitkv-aligned-gdn-v1'


def commit_observation_limit():
    """Twice the frozen output allowance plus startup/readiness headroom."""
    caps = sum(row['maximum_output_tokens'] for row in settings()['requests'])
    _require(caps == 794624, 'natural-stop output cap sum differs')
    return 2 * caps + 4096


def paired_serving_enabled():
    return settings()['phase'] == PAIRED_PHASE


def paired_group_specs():
    return [{'concurrency': 1, 'id': 'warmup-c1', 'kind': 'warmup', 'ordinals': [0], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2', 'kind': 'warmup', 'ordinals': [1, 2], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1', 'kind': 'timing', 'ordinals': [3, 4], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2', 'kind': 'timing', 'ordinals': [5, 6], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3', 'kind': 'timing', 'ordinals': [7, 8], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1', 'kind': 'timing', 'ordinals': [9, 10], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2', 'kind': 'timing', 'ordinals': [11, 12], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3', 'kind': 'timing', 'ordinals': [13, 14], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1', 'kind': 'quality', 'ordinals': [15, 16], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2', 'kind': 'quality', 'ordinals': [17, 18], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r1', 'kind': 'warmup', 'ordinals': [19], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r1', 'kind': 'warmup', 'ordinals': [20, 21], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r1', 'kind': 'timing', 'ordinals': [22, 23], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r1', 'kind': 'timing', 'ordinals': [24, 25], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r1', 'kind': 'timing', 'ordinals': [26, 27], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r1', 'kind': 'timing', 'ordinals': [28, 29], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r1', 'kind': 'timing', 'ordinals': [30, 31], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r1', 'kind': 'timing', 'ordinals': [32, 33], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r1', 'kind': 'quality', 'ordinals': [34, 35], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r1', 'kind': 'quality', 'ordinals': [36, 37], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r2', 'kind': 'warmup', 'ordinals': [38], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r2', 'kind': 'warmup', 'ordinals': [39, 40], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r2', 'kind': 'timing', 'ordinals': [41, 42], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r2', 'kind': 'timing', 'ordinals': [43, 44], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r2', 'kind': 'timing', 'ordinals': [45, 46], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r2', 'kind': 'timing', 'ordinals': [47, 48], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r2', 'kind': 'timing', 'ordinals': [49, 50], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r2', 'kind': 'timing', 'ordinals': [51, 52], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r2', 'kind': 'quality', 'ordinals': [53, 54], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r2', 'kind': 'quality', 'ordinals': [55, 56], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r3', 'kind': 'warmup', 'ordinals': [57], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r3', 'kind': 'warmup', 'ordinals': [58, 59], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r3', 'kind': 'timing', 'ordinals': [60, 61], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r3', 'kind': 'timing', 'ordinals': [62, 63], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r3', 'kind': 'timing', 'ordinals': [64, 65], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r3', 'kind': 'timing', 'ordinals': [66, 67], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r3', 'kind': 'timing', 'ordinals': [68, 69], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r3', 'kind': 'timing', 'ordinals': [70, 71], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r3', 'kind': 'quality', 'ordinals': [72, 73], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r3', 'kind': 'quality', 'ordinals': [74, 75], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r4', 'kind': 'warmup', 'ordinals': [76], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r4', 'kind': 'warmup', 'ordinals': [77, 78], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r4', 'kind': 'timing', 'ordinals': [79, 80], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r4', 'kind': 'timing', 'ordinals': [81, 82], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r4', 'kind': 'timing', 'ordinals': [83, 84], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r4', 'kind': 'timing', 'ordinals': [85, 86], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r4', 'kind': 'timing', 'ordinals': [87, 88], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r4', 'kind': 'timing', 'ordinals': [89, 90], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r4', 'kind': 'quality', 'ordinals': [91, 92], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r4', 'kind': 'quality', 'ordinals': [93, 94], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r5', 'kind': 'warmup', 'ordinals': [95], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r5', 'kind': 'warmup', 'ordinals': [96, 97], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r5', 'kind': 'timing', 'ordinals': [98, 99], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r5', 'kind': 'timing', 'ordinals': [100, 101], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r5', 'kind': 'timing', 'ordinals': [102, 103], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r5', 'kind': 'timing', 'ordinals': [104, 105], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r5', 'kind': 'timing', 'ordinals': [106, 107], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r5', 'kind': 'timing', 'ordinals': [108, 109], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r5', 'kind': 'quality', 'ordinals': [110, 111], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r5', 'kind': 'quality', 'ordinals': [112, 113], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r6', 'kind': 'warmup', 'ordinals': [114], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r6', 'kind': 'warmup', 'ordinals': [115, 116], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r6', 'kind': 'timing', 'ordinals': [117, 118], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r6', 'kind': 'timing', 'ordinals': [119, 120], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r6', 'kind': 'timing', 'ordinals': [121, 122], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r6', 'kind': 'timing', 'ordinals': [123, 124], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r6', 'kind': 'timing', 'ordinals': [125, 126], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r6', 'kind': 'timing', 'ordinals': [127, 128], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r6', 'kind': 'quality', 'ordinals': [129, 130], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r6', 'kind': 'quality', 'ordinals': [131, 132], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r7', 'kind': 'warmup', 'ordinals': [133], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r7', 'kind': 'warmup', 'ordinals': [134, 135], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r7', 'kind': 'timing', 'ordinals': [136, 137], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r7', 'kind': 'timing', 'ordinals': [138, 139], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r7', 'kind': 'timing', 'ordinals': [140, 141], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r7', 'kind': 'timing', 'ordinals': [142, 143], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r7', 'kind': 'timing', 'ordinals': [144, 145], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r7', 'kind': 'timing', 'ordinals': [146, 147], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r7', 'kind': 'quality', 'ordinals': [148, 149], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r7', 'kind': 'quality', 'ordinals': [150, 151], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r8', 'kind': 'warmup', 'ordinals': [152], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r8', 'kind': 'warmup', 'ordinals': [153, 154], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r8', 'kind': 'timing', 'ordinals': [155, 156], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r8', 'kind': 'timing', 'ordinals': [157, 158], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r8', 'kind': 'timing', 'ordinals': [159, 160], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r8', 'kind': 'timing', 'ordinals': [161, 162], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r8', 'kind': 'timing', 'ordinals': [163, 164], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r8', 'kind': 'timing', 'ordinals': [165, 166], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r8', 'kind': 'quality', 'ordinals': [167, 168], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r8', 'kind': 'quality', 'ordinals': [169, 170], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'b1-warmup-c1', 'kind': 'warmup', 'ordinals': [171], 'task': 'warmup'}, {'concurrency': 2, 'id': 'b1-warmup-c2', 'kind': 'warmup', 'ordinals': [172, 173], 'task': 'warmup'}, {'concurrency': 2, 'id': 'b1-pair-c2', 'kind': 'timing', 'ordinals': [174, 175], 'task': 'pair-retrieval'}]


def validate_paired_settings(value):
    expected = {'context_length': 131040, 'group_seconds': 'max(received_monotonic)-min(prepared_monotonic)', 'groups': [{'concurrency': 1, 'id': 'warmup-c1', 'kind': 'warmup', 'ordinals': [0], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2', 'kind': 'warmup', 'ordinals': [1, 2], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1', 'kind': 'timing', 'ordinals': [3, 4], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2', 'kind': 'timing', 'ordinals': [5, 6], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3', 'kind': 'timing', 'ordinals': [7, 8], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1', 'kind': 'timing', 'ordinals': [9, 10], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2', 'kind': 'timing', 'ordinals': [11, 12], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3', 'kind': 'timing', 'ordinals': [13, 14], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1', 'kind': 'quality', 'ordinals': [15, 16], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2', 'kind': 'quality', 'ordinals': [17, 18], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r1', 'kind': 'warmup', 'ordinals': [19], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r1', 'kind': 'warmup', 'ordinals': [20, 21], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r1', 'kind': 'timing', 'ordinals': [22, 23], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r1', 'kind': 'timing', 'ordinals': [24, 25], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r1', 'kind': 'timing', 'ordinals': [26, 27], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r1', 'kind': 'timing', 'ordinals': [28, 29], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r1', 'kind': 'timing', 'ordinals': [30, 31], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r1', 'kind': 'timing', 'ordinals': [32, 33], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r1', 'kind': 'quality', 'ordinals': [34, 35], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r1', 'kind': 'quality', 'ordinals': [36, 37], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r2', 'kind': 'warmup', 'ordinals': [38], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r2', 'kind': 'warmup', 'ordinals': [39, 40], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r2', 'kind': 'timing', 'ordinals': [41, 42], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r2', 'kind': 'timing', 'ordinals': [43, 44], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r2', 'kind': 'timing', 'ordinals': [45, 46], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r2', 'kind': 'timing', 'ordinals': [47, 48], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r2', 'kind': 'timing', 'ordinals': [49, 50], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r2', 'kind': 'timing', 'ordinals': [51, 52], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r2', 'kind': 'quality', 'ordinals': [53, 54], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r2', 'kind': 'quality', 'ordinals': [55, 56], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r3', 'kind': 'warmup', 'ordinals': [57], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r3', 'kind': 'warmup', 'ordinals': [58, 59], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r3', 'kind': 'timing', 'ordinals': [60, 61], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r3', 'kind': 'timing', 'ordinals': [62, 63], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r3', 'kind': 'timing', 'ordinals': [64, 65], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r3', 'kind': 'timing', 'ordinals': [66, 67], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r3', 'kind': 'timing', 'ordinals': [68, 69], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r3', 'kind': 'timing', 'ordinals': [70, 71], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r3', 'kind': 'quality', 'ordinals': [72, 73], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r3', 'kind': 'quality', 'ordinals': [74, 75], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r4', 'kind': 'warmup', 'ordinals': [76], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r4', 'kind': 'warmup', 'ordinals': [77, 78], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r4', 'kind': 'timing', 'ordinals': [79, 80], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r4', 'kind': 'timing', 'ordinals': [81, 82], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r4', 'kind': 'timing', 'ordinals': [83, 84], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r4', 'kind': 'timing', 'ordinals': [85, 86], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r4', 'kind': 'timing', 'ordinals': [87, 88], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r4', 'kind': 'timing', 'ordinals': [89, 90], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r4', 'kind': 'quality', 'ordinals': [91, 92], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r4', 'kind': 'quality', 'ordinals': [93, 94], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r5', 'kind': 'warmup', 'ordinals': [95], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r5', 'kind': 'warmup', 'ordinals': [96, 97], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r5', 'kind': 'timing', 'ordinals': [98, 99], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r5', 'kind': 'timing', 'ordinals': [100, 101], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r5', 'kind': 'timing', 'ordinals': [102, 103], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r5', 'kind': 'timing', 'ordinals': [104, 105], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r5', 'kind': 'timing', 'ordinals': [106, 107], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r5', 'kind': 'timing', 'ordinals': [108, 109], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r5', 'kind': 'quality', 'ordinals': [110, 111], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r5', 'kind': 'quality', 'ordinals': [112, 113], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r6', 'kind': 'warmup', 'ordinals': [114], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r6', 'kind': 'warmup', 'ordinals': [115, 116], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r6', 'kind': 'timing', 'ordinals': [117, 118], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r6', 'kind': 'timing', 'ordinals': [119, 120], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r6', 'kind': 'timing', 'ordinals': [121, 122], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r6', 'kind': 'timing', 'ordinals': [123, 124], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r6', 'kind': 'timing', 'ordinals': [125, 126], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r6', 'kind': 'timing', 'ordinals': [127, 128], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r6', 'kind': 'quality', 'ordinals': [129, 130], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r6', 'kind': 'quality', 'ordinals': [131, 132], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r7', 'kind': 'warmup', 'ordinals': [133], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r7', 'kind': 'warmup', 'ordinals': [134, 135], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r7', 'kind': 'timing', 'ordinals': [136, 137], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r7', 'kind': 'timing', 'ordinals': [138, 139], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r7', 'kind': 'timing', 'ordinals': [140, 141], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r7', 'kind': 'timing', 'ordinals': [142, 143], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r7', 'kind': 'timing', 'ordinals': [144, 145], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r7', 'kind': 'timing', 'ordinals': [146, 147], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r7', 'kind': 'quality', 'ordinals': [148, 149], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r7', 'kind': 'quality', 'ordinals': [150, 151], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'warmup-c1-r8', 'kind': 'warmup', 'ordinals': [152], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'warmup-c2-r8', 'kind': 'warmup', 'ordinals': [153, 154], 'task': 'service-config'}, {'concurrency': 1, 'id': 'timing-c1-1-r8', 'kind': 'timing', 'ordinals': [155, 156], 'task': 'ledger'}, {'concurrency': 1, 'id': 'timing-c1-2-r8', 'kind': 'timing', 'ordinals': [157, 158], 'task': 'sre-log-report'}, {'concurrency': 1, 'id': 'timing-c1-3-r8', 'kind': 'timing', 'ordinals': [159, 160], 'task': 'service-config'}, {'concurrency': 2, 'id': 'timing-c2-1-r8', 'kind': 'timing', 'ordinals': [161, 162], 'task': 'ledger'}, {'concurrency': 2, 'id': 'timing-c2-2-r8', 'kind': 'timing', 'ordinals': [163, 164], 'task': 'sre-log-report'}, {'concurrency': 2, 'id': 'timing-c2-3-r8', 'kind': 'timing', 'ordinals': [165, 166], 'task': 'service-config'}, {'concurrency': 1, 'id': 'quality-c1-r8', 'kind': 'quality', 'ordinals': [167, 168], 'task': 'retrieval'}, {'concurrency': 2, 'id': 'quality-c2-r8', 'kind': 'quality', 'ordinals': [169, 170], 'task': 'retrieval'}, {'concurrency': 1, 'id': 'b1-warmup-c1', 'kind': 'warmup', 'ordinals': [171], 'task': 'warmup'}, {'concurrency': 2, 'id': 'b1-warmup-c2', 'kind': 'warmup', 'ordinals': [172, 173], 'task': 'warmup'}, {'concurrency': 2, 'id': 'b1-pair-c2', 'kind': 'timing', 'ordinals': [174, 175], 'task': 'pair-retrieval'}], 'historical_uninstrumented_comparison': False, 'measured_natural_requests_per_start': 110, 'mode': 'normal-stop-fp8-mxfp4-pairs-v1', 'natural_quality_requests_per_start': 36, 'order': {'control': ['warmup-c1', 'warmup-c2', 'timing-c1-1', 'timing-c1-2', 'timing-c1-3', 'timing-c2-1', 'timing-c2-2', 'timing-c2-3', 'quality-c1', 'quality-c2', 'warmup-c1-r1', 'warmup-c2-r1', 'timing-c1-1-r1', 'timing-c1-2-r1', 'timing-c1-3-r1', 'timing-c2-1-r1', 'timing-c2-2-r1', 'timing-c2-3-r1', 'quality-c1-r1', 'quality-c2-r1', 'warmup-c1-r2', 'warmup-c2-r2', 'timing-c1-1-r2', 'timing-c1-2-r2', 'timing-c1-3-r2', 'timing-c2-1-r2', 'timing-c2-2-r2', 'timing-c2-3-r2', 'quality-c1-r2', 'quality-c2-r2', 'warmup-c1-r3', 'warmup-c2-r3', 'timing-c1-1-r3', 'timing-c1-2-r3', 'timing-c1-3-r3', 'timing-c2-1-r3', 'timing-c2-2-r3', 'timing-c2-3-r3', 'quality-c1-r3', 'quality-c2-r3', 'warmup-c1-r4', 'warmup-c2-r4', 'timing-c1-1-r4', 'timing-c1-2-r4', 'timing-c1-3-r4', 'timing-c2-1-r4', 'timing-c2-2-r4', 'timing-c2-3-r4', 'quality-c1-r4', 'quality-c2-r4', 'warmup-c1-r5', 'warmup-c2-r5', 'timing-c1-1-r5', 'timing-c1-2-r5', 'timing-c1-3-r5', 'timing-c2-1-r5', 'timing-c2-2-r5', 'timing-c2-3-r5', 'quality-c1-r5', 'quality-c2-r5', 'warmup-c1-r6', 'warmup-c2-r6', 'timing-c1-1-r6', 'timing-c1-2-r6', 'timing-c1-3-r6', 'timing-c2-1-r6', 'timing-c2-2-r6', 'timing-c2-3-r6', 'quality-c1-r6', 'quality-c2-r6', 'warmup-c1-r7', 'warmup-c2-r7', 'timing-c1-1-r7', 'timing-c1-2-r7', 'timing-c1-3-r7', 'timing-c2-1-r7', 'timing-c2-2-r7', 'timing-c2-3-r7', 'quality-c1-r7', 'quality-c2-r7', 'warmup-c1-r8', 'warmup-c2-r8', 'timing-c1-1-r8', 'timing-c1-2-r8', 'timing-c1-3-r8', 'timing-c2-1-r8', 'timing-c2-2-r8', 'timing-c2-3-r8', 'quality-c1-r8', 'quality-c2-r8', 'b1-warmup-c1', 'b1-warmup-c2', 'b1-pair-c2'], 'mtp': ['warmup-c1', 'warmup-c2', 'timing-c1-1', 'timing-c1-2', 'timing-c1-3', 'timing-c2-1', 'timing-c2-2', 'timing-c2-3', 'quality-c1', 'quality-c2', 'warmup-c1-r1', 'warmup-c2-r1', 'timing-c1-1-r1', 'timing-c1-2-r1', 'timing-c1-3-r1', 'timing-c2-1-r1', 'timing-c2-2-r1', 'timing-c2-3-r1', 'quality-c1-r1', 'quality-c2-r1', 'warmup-c1-r2', 'warmup-c2-r2', 'timing-c1-1-r2', 'timing-c1-2-r2', 'timing-c1-3-r2', 'timing-c2-1-r2', 'timing-c2-2-r2', 'timing-c2-3-r2', 'quality-c1-r2', 'quality-c2-r2', 'warmup-c1-r3', 'warmup-c2-r3', 'timing-c1-1-r3', 'timing-c1-2-r3', 'timing-c1-3-r3', 'timing-c2-1-r3', 'timing-c2-2-r3', 'timing-c2-3-r3', 'quality-c1-r3', 'quality-c2-r3', 'warmup-c1-r4', 'warmup-c2-r4', 'timing-c1-1-r4', 'timing-c1-2-r4', 'timing-c1-3-r4', 'timing-c2-1-r4', 'timing-c2-2-r4', 'timing-c2-3-r4', 'quality-c1-r4', 'quality-c2-r4', 'warmup-c1-r5', 'warmup-c2-r5', 'timing-c1-1-r5', 'timing-c1-2-r5', 'timing-c1-3-r5', 'timing-c2-1-r5', 'timing-c2-2-r5', 'timing-c2-3-r5', 'quality-c1-r5', 'quality-c2-r5', 'warmup-c1-r6', 'warmup-c2-r6', 'timing-c1-1-r6', 'timing-c1-2-r6', 'timing-c1-3-r6', 'timing-c2-1-r6', 'timing-c2-2-r6', 'timing-c2-3-r6', 'quality-c1-r6', 'quality-c2-r6', 'warmup-c1-r7', 'warmup-c2-r7', 'timing-c1-1-r7', 'timing-c1-2-r7', 'timing-c1-3-r7', 'timing-c2-1-r7', 'timing-c2-2-r7', 'timing-c2-3-r7', 'quality-c1-r7', 'quality-c2-r7', 'warmup-c1-r8', 'warmup-c2-r8', 'timing-c1-1-r8', 'timing-c1-2-r8', 'timing-c1-3-r8', 'timing-c2-1-r8', 'timing-c2-2-r8', 'timing-c2-3-r8', 'quality-c1-r8', 'quality-c2-r8', 'b1-warmup-c1', 'b1-warmup-c2', 'b1-pair-c2']}, 'requests_per_start': 176, 'reserve_per_request': 16, 'throughput_scope': 'instrumented HTTP service throughput; combined TP2', 'token_pool_capacity': 262144, 'warmup_requests_per_start': 30}
    _require(type(value.get('paired_serving')) is dict and value['paired_serving']==expected,
             'frozen W5 v2 group/order/measurement contract')
    rows=value['requests']
    _require(type(rows) is list and len(rows)==176 and value['execution_order']==['control','mtp'],
             'W5 v2 count/order')
    b1={171: ('b1-warmup-c1', 2040), 172: ('b1-warmup-c2-a', 2040), 173: ('b1-warmup-c2-b', 2040), 174: ('b1-pair-c2-a', 59993), 175: ('b1-pair-c2-b', 59980)}
    shapes={'ledger':(2047,4096),'retrieval':(5461,1024),'sre-log-report':(1453,8192),'service-config':(724,4096)}
    for group in expected['groups']:
        for index,ordinal in enumerate(group['ordinals']):
            row=rows[ordinal]
            _require(row.get('stop_policy', 'natural') in ('natural', 'cap'),
                     'explicit natural or cap stop policy required')
            if ordinal in b1:
                ok=(row['case']==b1[ordinal][0] and (row['input_tokens'],row['maximum_output_tokens'])==(b1[ordinal][1],4096) and
                    row['context_envelope_tokens']==131040)
            else:
                name=group['id']+('-'+chr(97+index) if len(group['ordinals'])==2 else '')
                ok=(row['case']==name and (row['input_tokens'],row['maximum_output_tokens'])==shapes[group['task']] and
                    row['context_envelope_tokens']==32768)
            _require(ok and row['ordinal']==ordinal and type(row['ordinal']) is int and
                     row['group']==group['id'] and row['category']==group['kind'] and row['task']==group['task'] and
                     row['internal_reserve_tokens']==16,
                     'frozen W5 v2 request shape/case')
        sizes=[rows[i]['input_tokens']+rows[i]['maximum_output_tokens']+16 for i in group['ordinals']]
        envelope,pool=(131040,262144) if group['ordinals'][0] in b1 else (32768,32800)
        _require(max(sizes)<=envelope and (sum(sizes) if group['concurrency']==2 else max(sizes))<=pool,
                 'W5 v2 concurrent token capacity')


def paired_groups(arm):
    value=settings()
    _require(value['phase']==PAIRED_PHASE and arm in LOGICAL_ARMS, 'S39 group phase/arm')
    groups={g['id']:g for g in value['paired_serving']['groups']}
    return [groups[name] for name in value['paired_serving']['order']['control']]


def paired_group_for(ordinal):
    _require(type(ordinal) is int and 0<=ordinal<176, 'S39 request ordinal')
    return next(group for group in paired_groups('control') if ordinal in group['ordinals'])




def host_path_cleanup_enabled(arm):
    """Keep the selected H control host path in both arms."""
    _require(arm in LOGICAL_ARMS, 'unknown host-path cleanup arm')
    return False  # Both arms preserve the selected H control host path.


def expected_decode_log_interval(arm):
    return 100 if host_path_cleanup_enabled(arm) else 1

def kv_composition_enabled():
    return True


def instrumentation_enabled():
    """Production mode serves the same path with no probe, observer, parity gate or evidence writer (W5 v2)."""
    return settings().get('instrumentation', 'production') == 'on'


def completion_allowed(response, request):
    """Apply a source-bound row policy without grading cap work as an answer."""
    if type(response) is not dict or type(request) is not dict:
        return False
    policy = request.get('stop_policy', 'natural')
    if policy not in ('natural', 'cap'):
        return False
    if response.get('status') == 'NORMAL_STOP_COMPLETE' and response.get('finish_reason') == 'stop':
        return True
    maximum = request.get('maximum_output_tokens')
    return (policy == 'cap' and type(maximum) is int and maximum > 0
            and response.get('status') == 'OUTPUT_CAP_REACHED'
            and response.get('finish_reason') == 'length'
            and response.get('metric_eligible') is False
            and all(type(response.get(key)) is int and response[key] == maximum
                    for key in ('reported_completion_tokens', 'response_id_count')))


def bind_observed_rank(devices, renders, expected, indices, rank, gpu_id):
    """Join observed HIP indices to BDF/UUID/render identity, never key order."""
    _require(type(indices) is list and len(indices) == 2
             and all(type(index) is int for index in indices) and set(indices) == {0, 1}
             and type(rank) is int and rank in (0, 1)
             and type(gpu_id) is int and gpu_id == indices[rank], 'native rank/index binding differs')
    _require(type(expected) is dict and len(expected) == 2
             and all(type(bdf) is str and re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf)
                     and type(item) is dict and set(item) == {'uuid', 'render'}
                     and type(item['uuid']) is str and item['uuid']
                     and type(item['render']) is str and re.fullmatch(r'/dev/dri/renderD[0-9]+', item['render'])
                     for bdf, item in expected.items()), 'configured device identities differ')
    _require(type(devices) is list and len(devices) == 2
             and all(type(row) is dict and type(row.get('index')) is int
                     and type(row.get('bdf')) is str for row in devices)
             and {row['index'] for row in devices} == {0, 1}
             and len({row['bdf'] for row in devices}) == 2
             and {row['bdf']: row.get('uuid') for row in devices}
                 == {bdf: item['uuid'] for bdf, item in expected.items()}, 'observed HIP device set differs')
    _require(type(renders) is list and len(renders) == 2
             and all(type(row) is dict and type(row.get('bdf')) is str for row in renders)
             and {row['bdf'] for row in renders} == set(expected)
             and all(row.get('sysfs_bdf') == row['bdf'] and row.get('render') == expected[row['bdf']]['render']
                     and row.get('uuid') == expected[row['bdf']]['uuid'] for row in renders),
             'observed render/sysfs device set differs')
    selected = next(row for row in devices if row['index'] == gpu_id)
    return dict(rank=rank, hip_index=gpu_id, bdf=selected['bdf'], uuid=selected['uuid'],
                render=expected[selected['bdf']]['render'])

def kv_request_groups():
    return [dict(group) for group in paired_groups('control')]


REQUEST_LOGICAL = {'CLIENT-BINDING.zlib': {'sha256': 'ae0145b4b0d59c4c03fe090bb80852c79a740d597806d643f2f1b51af80b3280', 'bytes': 259034}, 'CONTEXT-PROMPTS.zlib': {'sha256': 'cef78631478b938dec73af5382e54bf2470393cfce3f0eb6beabe753b0306f21', 'bytes': 577638}, 'CONTROL-CASE.zlib': {'sha256': 'bf47dcc43d36709311422af34f98f9a49ce51628240c2cc22ffb27b82d7ca201', 'bytes': 45475}, 'FINAL-REQUESTS.zlib': {'sha256': '9154ce89d7d692b4305e5fe35c78e532b50fa28d07973d5811d17778005b24fd', 'bytes': 400483}, 'TOKENIZER-CONTRACT.json': {'sha256': 'e0c8b26bd821bcc42719e55fbad658f7eecdae0053c0b8e05f48540cb3acdb88', 'bytes': 10284}}
REQUEST_STORED = {'sha256': '3990e581b9675f2ee893d3acb7596b4eb129f6128a3b4c87e830f7f3ca23e3aa', 'bytes': 1246622}
REQUEST_EXPANDED = {'sha256': '1e9c42958b942b1161195cde4c4631910920fff58fde697897ca8ff3c919eb3d', 'bytes': 1724022}
REQUEST_INHERITED = {'sha256': '9154ce89d7d692b4305e5fe35c78e532b50fa28d07973d5811d17778005b24fd', 'bytes': 400483}
PACKED_REQUEST_NAMES = ('CLIENT-BINDING.zlib', 'CONTEXT-PROMPTS.zlib', 'CONTROL-CASE.zlib', 'TOKENIZER-CONTRACT.json', 'FINAL-REQUESTS.zlib')
FACTS_STORED = {'sha256': '376dcbd9eb70a39684d99e75e270083e45c66436eac51517eecce52642d77021', 'bytes': 3004352}
FACTS_LOGICAL = {'sha256': 'd414db807ccec0f7c285be99cd040c86e0230521b03996f64f9bc4f9392d3072', 'bytes': 9605821}
"""Inserted into the ordinary fixed serving contract; data assets only."""
import os
import stat


def _asset_stable_read(name, pin, size):
    path = Path(__file__).with_name(name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                 before.st_size == size, 'exact ordinary stored asset')
        data = bytearray()
        while len(data) <= size:
            block = os.read(fd, min(1 << 20, size + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        _require(stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()) and
                 len(data) == size and hashlib.sha256(data).hexdigest() == pin,
                 'unchanged complete ordinary stored asset')
        return bytes(data)
    finally:
        os.close(fd)


def _asset_inflate(raw, bound):
    _require(type(raw) is bytes and type(bound) is int and 0 < bound <= 16 << 20,
             'fixed decoded asset limit')
    decoder = zlib.decompressobj()
    value = decoder.decompress(raw, bound + 1)
    _require(len(value) <= bound and decoder.eof and
             not decoder.unused_data and not decoder.unconsumed_tail,
             'complete bounded single asset stream')
    return value


@lru_cache(maxsize=1)
def _request_asset_map(raw):
    _require(len(raw) == REQUEST_STORED['bytes'] and
             hashlib.sha256(raw).hexdigest() == REQUEST_STORED['sha256'],
             'fixed request container identity')
    expanded = _asset_inflate(raw, REQUEST_EXPANDED['bytes'])
    _require(len(expanded) == REQUEST_EXPANDED['bytes'] and
             hashlib.sha256(expanded).hexdigest() == REQUEST_EXPANDED['sha256'],
             'fixed complete request container decoded identity')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate request asset')
            value[key] = item
        return value
    value = json.loads(expanded, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite request asset')))
    _require(type(value) is dict and set(value) == set(REQUEST_LOGICAL),
             'exact decoded request asset membership')
    result = {}
    for name, row in REQUEST_LOGICAL.items():
        _require(type(value[name]) is str, 'base64 request asset string')
        decoded = base64.b64decode(value[name], validate=True)
        _require(len(decoded) == row['bytes'] and
                 hashlib.sha256(decoded).hexdigest() == row['sha256'],
                 'exact decoded request asset: ' + name)
        result[name] = decoded
    return result


def _request_assets():
    # Reauthenticate the held stored file on every call. The cache is keyed by
    # complete authenticated bytes, never a path or unchecked timestamp.
    return _request_asset_map(_asset_stable_read('FINAL-REQUESTS.zlib',
        REQUEST_STORED['sha256'], REQUEST_STORED['bytes']))


def logical_asset(name, pin, maximum):
    _require(type(name) is str and name in PACKED_REQUEST_NAMES and
             type(maximum) is int and maximum >= 0, 'fixed logical request asset')
    data = _request_assets()[name]
    _require(len(data) <= maximum and hashlib.sha256(data).hexdigest() == pin,
             'original logical request asset size/hash')
    return data


@lru_cache(maxsize=1)
def _decoded_composition_facts(raw):
    _require(len(raw) == FACTS_STORED['bytes'] and
             hashlib.sha256(raw).hexdigest() == FACTS_STORED['sha256'],
             'fixed stored composition facts identity')
    data = _asset_inflate(raw, FACTS_LOGICAL['bytes'])
    _require(len(data) == FACTS_LOGICAL['bytes'] and
             hashlib.sha256(data).hexdigest() == FACTS_LOGICAL['sha256'],
             'original full composition facts identity')
    return data


def composition_facts_bytes():
    return _decoded_composition_facts(_asset_stable_read('COMPOSITION-FACTS.zlib',
        FACTS_STORED['sha256'], FACTS_STORED['bytes']))


def logical_package_inventory():
    return {'request_container': dict(stored_name='review/FINAL-REQUESTS.zlib',
                stored=dict(REQUEST_STORED), expanded=dict(REQUEST_EXPANDED),
                members={name: dict(row) for name, row in REQUEST_LOGICAL.items()},
                inherited_container=dict(REQUEST_INHERITED)),
            'composition_facts': dict(stored_name='review/COMPOSITION-FACTS.zlib',
                stored=dict(FACTS_STORED), logical_name='review/COMPOSITION-FACTS.json',
                logical=dict(FACTS_LOGICAL)), 'executable_members_packed': 0}


def validate_logical_package_inventory(value):
    _require(type(value) is dict and
             json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) ==
             json.dumps(logical_package_inventory(), sort_keys=True, separators=(',', ':'), allow_nan=False),
             'explicit exact typed logical package inventory')
    _request_assets()
    composition_facts_bytes()


"""Consolidate only empty bytecode tmpfs mounts; preserve source mounts."""
BYTECODE_ROOT = '/mnt/capture-bytecode/mnt/composition/image-source/python/sglang'
BYTECODE_PARENTS = (BYTECODE_ROOT,)


def _effective_bytecode_masks(paths):
    _require(type(paths) is list and paths and len(set(paths)) == len(paths),
             'unique original bytecode masks')
    for path in paths:
        _require(type(path) is str and str(Path(path)) == path and
                 '..' not in Path(path).parts and
                 sum(path == parent or path.startswith(parent + '/')
                     for parent in BYTECODE_PARENTS) == 1,
                 'bytecode parent contains every original descendant')
    return list(BYTECODE_PARENTS)


def _consolidate_bytecode_argv(argv, paths):
    result = list(argv)
    parents = _effective_bytecode_masks(paths)
    actual = [result[i + 1] for i in range(len(result) - 1)
              if result[i] == '--tmpfs' and result[i + 1].startswith('/mnt/capture-bytecode/')]
    _require(set(actual) == set(paths) and len(actual) == len(paths),
             'exact complete original bytecode tmpfs set')
    original_sources = [result[i:i + 3] for i in range(len(result) - 2)
                        if result[i] in ('--ro-bind', '--bind', '--ro-bind-try', '--dev-bind')]
    first = None
    for path in paths:
        for group in (['--size', str(1 << 20), '--tmpfs', path], ['--remount-ro', path]):
            matches = [i for i in range(len(result) - len(group) + 1)
                       if result[i:i + len(group)] == group]
            _require(len(matches) == 1, 'exact sized read-only original bytecode mask')
            pos = matches[0]
            first = pos if first is None else min(first, pos)
            del result[pos:pos + len(group)]
    mounts = []
    for parent in parents:
        mounts += ['--size', str(len(paths) << 20), '--tmpfs', parent, '--remount-ro', parent]
    result[first:first] = mounts
    _require([result[i:i + 3] for i in range(len(result) - 2)
              if result[i] in ('--ro-bind', '--bind', '--ro-bind-try', '--dev-bind')] == original_sources,
             'all actual source/device bindings unchanged')
    _require(all(path.startswith('/mnt/capture-bytecode/') for path in parents),
             'no actual source path masked')
    return result


def expand_http_template(value):
    """Expand one exact MXFP4 source-map reference before existing validation."""
    settings()
    _require(type(value) is dict and type(value.get('binding_template')) is dict and
             type(value.get('worker_template')) is dict, 'MXFP4 HTTP template objects')
    sources = value['binding_template'].get('sources')
    reference = value['worker_template'].get('sources')
    _require(type(sources) is dict and type(reference) is dict and
             set(reference) == {'same_as', 'source_manifest_sha256'} and
             reference['same_as'] == 'binding_template.sources', 'MXFP4 HTTP source reference')
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    _require(reference['source_manifest_sha256'] == value.get('source_manifest_sha256') ==
             value['binding_template'].get('source_manifest_sha256') ==
             value['worker_template'].get('source_manifest_sha256') == digest,
             'MXFP4 HTTP source reference digest')
    # Detach both records: later worker materialization cannot mutate the entry.
    result = json.loads(json.dumps(value, allow_nan=False))
    result['worker_template']['sources'] = json.loads(json.dumps(sources, allow_nan=False))
    return result
