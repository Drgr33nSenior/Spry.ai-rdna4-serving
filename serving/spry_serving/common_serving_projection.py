"""Exact inner-entry substitution over the unchanged reviewed owner projection."""
from pathlib import PurePosixPath
import re

MODE = 'native-http-fp8-sprint01-v1'
ARM_MAP = {'bf16': 'graph', 'kv4': 'eager'}
BACKEND_SHA256 = '0875f6cf8368f19fc43df5759fddc59c49044389d1273eb6df2541a6b5d62c7d'
ENTRY = '/mnt/review/serving_entry.py'


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


def need(value, reason):
    return _SPRY_COMMON.need(value, reason, error=ValueError, prefix='HTTP projection: ')


def project(original_argv, *, owner_arm, binding_sha256):
    """No process/resource/lock operation; preserve every pre-entry mount/limit."""
    need(owner_arm in ARM_MAP and type(binding_sha256) is str and
         re.fullmatch('[0-9a-f]{64}', binding_sha256), 'exact arm and binding digest')
    argv = list(original_argv)
    need(argv[0] == '/usr/bin/bwrap' and '--bind' not in argv and
         all(flag in argv for flag in ('--die-with-parent', '--new-session', '--unshare-all', '--share-net')),
         'original read-only namespace projection')
    python = argv.index('/mnt/work/build-venv/bin/python3')
    old = argv.index('/mnt/review/model_benchmark.py', python)
    need(argv[python:old] == ['/mnt/work/build-venv/bin/python3', '-I', '-B', '-X',
                            'pycache_prefix=/mnt/capture-bytecode'], 'original isolated bytecode policy')
    need(argv[old+1:old+4] == ['--execute-reviewed-benchmark', '--spec', '/mnt/review/BENCHMARK-SPEC2.json'] and
         argv.count('--physical-artifacts') == 1 and
         argv[argv.index('--arm')+1] == owner_arm, 'original exact direct-worker command')
    backend = '/mnt/composition/image-source/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py'
    need(argv.count(backend) == 1 and argv[argv.index(backend)-2] == '--ro-bind', 'one read-only graph overlay')
    before = argv[:old]
    result = before + [ENTRY, '--execute-serving-screen', '--binding',
        '/mnt/review/SERVING-' + ARM_MAP[owner_arm] + '.json', '--binding-sha256', binding_sha256]
    need(result[:old] == before and '--bind' not in result and len(result) <= 256,
         'no namespace, mounts, devices, environment or scratch change')
    need(all('..' not in PurePosixPath(value).parts for value in result if value.startswith('/')),
         'canonical projected paths')
    return tuple(result)


def controller_plan():
    """Explicit missing live dispatch; this is not an execution authority."""
    return dict(schema=1, mode=MODE, status='NON_EXECUTABLE_COMPONENT_PACKAGE',
        launch_authorized=False, qualification=False,
        source_promotion=False, owner_arm_map=ARM_MAP,
        unchanged=['canonical FD9 lease', 'host process supervisor', 'root release observer',
                   'cgroup limits and independent release', 'complete remaining envelopes'],
        pending=['resource gate exact HTTP argv branch', 'bounded HTTP artifact export/receive branch',
                 'physical controller HTTP source/worker/result joins including observed graph behavior',
                 'controller-facing composed acceptance and negative admission checks'],
        entry_requires_private_client_callback=True,
        backend_sha256_requires_effective_source_join=BACKEND_SHA256,
        projected_argv_is_proposal_not_executable=True)
