"""Check frozen chat IDs using the live native tokenizer before scheduler entry."""
import copy
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import time
import serving_contract


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError('native chat parity: ' + message)


def publish(path, raw):
    require(len(raw) <= 4 << 20, 'report size')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def validate_hip_sampling_components(records, *, binding, workers, phase):
    """Validate native startup results, independently of model task acceptance."""
    require((phase in serving_contract.HIP_PHASE_ARMS or phase == serving_contract.mtp_contract.LIFECYCLE_PHASE) and
            serving_contract.hip_sampling_arm(phase, binding.get('arm')),
            'HIP component trial phase')
    require(type(records) is list and len(records) == 2 and type(workers) is list and len(workers) == 2,
            'HIP component rank count')
    require([row.get('rank') for row in records if type(row) is dict] == [0, 1] and
            [row.get('rank') for row in workers if type(row) is dict] == [0, 1],
            'HIP component rank order or duplicate')
    source = binding['sources']['dflash_worker']
    expected_source = {key: source[key] for key in ('module', 'path', 'sha256')}
    require(expected_source['module'] == 'sglang.srt.speculative.dflash_worker_v2' and
            type(expected_source['sha256']) is str and len(expected_source['sha256']) == 64,
            'HIP component source')
    fields = {'schema', 'mode', 'phase', 'arm', 'rank', 'pid', 'start_ticks', 'boot_id',
              'session_sha256', 'spec_sha256', 'source_manifest_sha256', 'worker_identity_sha256',
              'worker_source', 'component', 'native', 'prewarm', 'qualification'}
    for rank, (row, worker) in enumerate(zip(records, workers)):
        require(set(row) == fields and type(row['schema']) is int and row['schema'] == 1 and
                row['phase'] == phase and row['qualification'] is False,
                'HIP component schema')
        require(all(row[key] == binding[key] for key in
                    ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')) and
                all(type(row[key]) is int and row[key] == worker[key] for key in ('rank', 'pid', 'start_ticks')) and
                row['pid'] > 0 and row['start_ticks'] > 0 and row['boot_id'] == worker['boot_id'] and
                row['worker_identity_sha256'] == worker['identity_sha256'] and
                row['worker_source'] == expected_source,
                'HIP component live worker or source identity')
        expected_component = {
            'schema': 1, 'mode': 'hip-native-target-sample-prefix-v1', 'rank': rank,
            'shape': [2, 8, 32], 'one_hot_cases': [
                {'accept_len': [0, 3], 'bonus': [21, 22]},
                {'accept_len': [7, 1], 'bonus': [23, 24]}],
            'passed': True, 'distribution_statistical_test': False,
            'model_forward_count': 0, 'seed_reset': False,
        }
        require(serving_contract.canonical(row['component']) == serving_contract.canonical(expected_component),
                'HIP native component result')
        expected_native = {'sampling_backend': 'pytorch', 'sampling_phase_enabled': True,
                           'tp_sync_site14_enabled': True, 'simulate_acc_len': -1,
                           'acceptance_thresholds': [1.0, 1.0], 'graph_pool_borrowing': False,
                           'proposal_mode': 'selector-greedy-target-sample-v1',
                           'selector_present': True, 'selector_sampling_enabled': False,
                           'lilicorr_present': False}
        require(type(row['native']) is dict and set(row['native']) == set(expected_native) and
                row['native'] == expected_native and
                row['native']['sampling_phase_enabled'] is True and
                row['native']['tp_sync_site14_enabled'] is True and
                row['native']['graph_pool_borrowing'] is False and
                row['native']['selector_present'] is True and
                row['native']['selector_sampling_enabled'] is False and
                row['native']['lilicorr_present'] is False and
                type(row['native']['simulate_acc_len']) in (int, float),
                'HIP native sampling or synchronization mode')
        memory = row['prewarm']
        require(type(memory) is dict and set(memory) ==
                {'shape', 'sampling_input_bytes', 'peak_bytes', 'sampling_headroom_bytes'} and
                memory['shape'] == [2, 8, 248320] and
                all(type(item) is int for item in memory['shape']) and
                all(type(memory[key]) is int and memory[key] > 0 for key in
                    ('sampling_input_bytes', 'peak_bytes', 'sampling_headroom_bytes')) and
                memory['sampling_input_bytes'] == 2 * 8 * 248320 * 4 and
                memory['sampling_headroom_bytes'] == memory['peak_bytes'] + memory['sampling_input_bytes'],
                'HIP native memory rehearsal')
    require(len({(row['pid'], row['start_ticks'], row['boot_id']) for row in records}) == 2,
            'HIP component duplicate worker')
    return records


def collect(manager, template, manifest):
    """Use the native constructed managers, not a fabricated tokenizer context."""
    for role in ('serving_chat', 'protocol', 'template_manager', 'tokenizer_manager', 'engine'):
        row = manifest['sources'][role]
        module = importlib.import_module(row['module'])
        require(module.__file__ == module.__spec__.origin == row['path'] and
                type(module.__spec__.loader) is SourceFileLoader and
                sha(Path(row['path']).read_bytes()) == row['sha256'], 'loaded origin: ' + role)
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager
    require(type(manager) is TokenizerManager and type(template) is TemplateManager,
            'native manager constructors required')
    handler = OpenAIServingChat(manager, template)
    def render(ordinal, case):
        row = dict(ordinal=ordinal, case=case['case'], request_sha256=serving_contract.digest(case['body']),
                   actual_ids_count=None, parity=False, reasoning_disabled=False, error=None)
        try:
            require(time.monotonic() < manifest['preweight']['deadline_monotonic'], 'preweight deadline')
            request = ChatCompletionRequest(**copy.deepcopy(case['body']))
            result = handler._process_messages(request, manager.model_config.is_multimodal)
            require(type(result.prompt_ids) is list and all(type(x) is int for x in result.prompt_ids),
                    'native prompt vector')
            actual = result.prompt_ids  # compared in-process below; the report keeps digest and length only
            row['actual_ids_count'] = len(actual)
            expected = serving_contract.expected_for_case(case)
            require(expected is not None, 'missing frozen expected vector')
            row['parity'] = actual == expected['input_ids']
            row['actual_ids_sha256'] = serving_contract.digest(actual)
            row['reasoning_disabled'] = handler._get_reasoning_from_request(request) is False
            require(row['parity'] and row['reasoning_disabled'], 'IDs or reasoning differ')
        except Exception as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)[:512]
        return row
    rows = []
    for ordinal, case in enumerate(serving_contract.requests()):
        row = render(ordinal, case)
        if 'generated_history' in case:
            row['generated_history_branches'] = []
            for index, branch in enumerate(case['generated_history']['branches']):
                observed = render(ordinal, serving_contract.history_branch(case, index))
                observed.update(branch_index=index, assistant_text_sha256=sha(branch['assistant_text'].encode()),
                                expected_sha256=branch['expected_sha256'])
                row['generated_history_branches'].append(observed)
        rows.append(row)
    checked_rows = [item for row in rows for item in [row, *row.get('generated_history_branches', [])]]
    value = dict(schema=1, mode=manifest['mode'], arm=manifest['arm'],
                 session_sha256=manifest['session_sha256'], spec_sha256=manifest['spec_sha256'],
                 source_manifest_sha256=manifest['source_manifest_sha256'], rows=rows,
                 native_manager=True, before_scheduler_delegation=True, qualification=False,
                 passed=all(row['error'] is None and row['parity'] and row['reasoning_disabled'] for row in checked_rows))
    raw = serving_contract.canonical(value)
    directory = Path(manifest['evidence_dir'])
    publish(directory/'NATIVE-CHAT-PARITY.json', raw)
    require(value['passed'], 'frozen native chat preflight refused; full report retained')
    publish(directory/'NATIVE-CHAT-PARITY.ready', sha(raw).encode('ascii'))
    return value


def wait_before_weights(manifest):
    """Both spawned schedulers wait for one fully published parent record."""
    root = Path(manifest['evidence_dir'])
    marker = root/'NATIVE-CHAT-PARITY.ready'
    end = manifest['preweight']['deadline_monotonic']
    while not marker.exists():
        require(time.monotonic() < end, 'no native parity before worker deadline')
        time.sleep(min(.05, max(0, end-time.monotonic())))
    require(not marker.is_symlink() and marker.stat().st_size == 64, 'ready marker')
    report = root/'NATIVE-CHAT-PARITY.json'
    require(report.is_file() and not report.is_symlink() and report.stat().st_size <= 4 << 20, 'parity record')
    raw = report.read_bytes()
    require(sha(raw) == marker.read_text(), 'published parity hash')
    value = json.loads(raw)
    require(value.get('passed') is True and all(value.get(key) == manifest[key] for key in
            ('mode', 'arm', 'session_sha256', 'spec_sha256', 'source_manifest_sha256')), 'parity identity')
    return value
