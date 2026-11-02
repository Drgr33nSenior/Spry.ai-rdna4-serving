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
    require(False, 'HIP component trial phase')


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
