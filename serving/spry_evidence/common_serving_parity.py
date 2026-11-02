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


def _load_spry_common(caller, pin, error):
    """Authenticate common source before any shared guard can execute."""
    import hashlib
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import os
    from pathlib import Path
    import stat

    def check(value, message):
        if not value:
            raise error('shared common source: ' + message)

    origin = Path(caller)
    path = origin.with_name('spry_common_canon.py')
    if origin.name.startswith('common_') and origin.parent.name in ('spry_serving', 'spry_evidence'):
        path = origin.resolve().parents[1] / 'spry_common/selected.py'
    elif (origin.parent.name == 'review' and origin.parent.parent.name == 'common-closure'
          and origin.parent.parent.parent.name == 'proposals'):
        path = origin.resolve().parents[3] / 'spry_common/selected.py'
    check(path.is_absolute() and path.resolve(strict=True) == path, 'canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 4096, 'bounded regular file required')
        raw = os.read(descriptor, 4097)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        check(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(descriptor))
              == stamp(path.lstat()), 'file changed during read')
    finally:
        os.close(descriptor)
    check(hashlib.sha256(raw).hexdigest() == pin, 'digest differs')
    # Use the checked bytes, ignoring ambient modules and cached bytecode.
    loader = SourceFileLoader('_spry_common', str(path))
    loader.get_code = lambda name: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader('_spry_common', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_SPRY_COMMON_SHA256 = 'c2cb7702d6ae482b0f9d70b6b838af3d0b9eb312140ff09d001df0880a9e6bed'
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, ValueError)


sha = _SPRY_COMMON.digest


def require(value, message):
    return _SPRY_COMMON.need(value, message, error=ValueError, prefix='native chat parity: ')


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
