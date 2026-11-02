"""Exact inner-entry substitution over the unchanged reviewed owner projection."""
from pathlib import PurePosixPath
import re

MODE = 'native-http-fp8-sprint01-v1'
ARM_MAP = {'bf16': 'graph', 'kv4': 'eager'}
BACKEND_SHA256 = '0875f6cf8368f19fc43df5759fddc59c49044389d1273eb6df2541a6b5d62c7d'
ENTRY = '/mnt/review/serving_entry.py'


def need(value, reason):
    if not value:
        raise ValueError('HTTP projection: ' + reason)


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
