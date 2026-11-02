"""Explicit graph-only outer projection and observed memory continuation policy."""

from pathlib import Path

MODE = 'native-model-context-retrieval-32k-v1'

PREFIX = '/mnt/capture-bytecode'

MASK_DIRS = ('model_executor', 'model_executor/runner_backend', 'layers', 'layers/attention', 'models')

BACKEND_MODULE = 'sglang.srt.model_executor.runner_backend.full_cuda_graph_backend'

BACKEND_PATH = '/mnt/composition/image-source/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py'

MEMORY_POLICY = dict(before_free_min_bytes=2 << 30, after_free_min_bytes=1 << 30,
                     growth_max_bytes=1 << 30, hard_transient_peak_quota=False)

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


def need(value, message):
    return _SPRY_COMMON.need(value, message, error=ValueError, prefix='graph outer: ')

def validate_projection(contract):
    need(type(contract) is dict and set(contract) == {'schema','mode','overlay','helpers','masks','mask_bytes_each','memory'} and
         type(contract['schema']) is int and contract['schema'] == 1 and contract['mode'] == MODE and
         contract['memory'] == MEMORY_POLICY and type(contract['masks']) is int and contract['masks'] == 5 and
         all(type(contract['memory'][key]) is type(value) for key,value in MEMORY_POLICY.items()) and
         type(contract['mask_bytes_each']) is int and contract['mask_bytes_each'] == 1 << 20,
         'exact exploratory projection/memory/mask contract')
    row=contract['overlay']
    need(type(row) is dict and set(row) == {'file','path','sha256','original_sha256'} and
         row['file'] == 'graph_full_cuda_graph_backend.py' and row['path'] == BACKEND_PATH and
         row['original_sha256'] == 'fcfca3c5d286a77594742fd05791fe5ad4c7fd46327ab7f126b2cdcf697b685d', 'one native backend overlay')
    need(type(contract['helpers']) is dict and set(contract['helpers']) ==
         {'spry_common_canon.py','graph_outer.py'}, 'private helper inventory')
    for pin in [row['sha256'],*contract['helpers'].values()]:
        need(type(pin) is str and len(pin)==64 and all(c in '0123456789abcdef' for c in pin), 'source pin')
    return contract

def project_argv(argv, review, contract):
    validate_projection(contract)
    result=list(argv)
    python=result.index('/mnt/work/build-venv/bin/python3')
    need(result[python:python+4] == ['/mnt/work/build-venv/bin/python3','-I','-B','/mnt/review/model_benchmark.py'],
         'original exact native entry')
    mounts=['--ro-bind',str(Path(review)/contract['overlay']['file']),BACKEND_PATH,'--dir',PREFIX]
    for directory in MASK_DIRS:
        mounts += ['--size',str(1<<20),'--tmpfs',PREFIX+'/mnt/composition/image-source/python/sglang/srt/'+directory]
    for directory in reversed(MASK_DIRS):
        mounts += ['--remount-ro',PREFIX+'/mnt/composition/image-source/python/sglang/srt/'+directory]
    mounts += ['--setenv','PYTHONPYCACHEPREFIX',PREFIX]
    result[python:python]=mounts
    python=result.index('/mnt/work/build-venv/bin/python3')
    result[python+3:python+3]=['-X','pycache_prefix='+PREFIX]
    output=next(i for i in range(len(result)-1) if result[i:i+2]==['--tmpfs','/mnt/output'])
    need(result[output-2:output] == ['--size',str(2816*1024**2)], 'unchanged total scratch allocation')
    result[output-1]=str(2811*1024**2)
    need(len(result)<=256 and '--bind' not in result and result.count(BACKEND_PATH)==1,
         'bounded read-only physical projection')
    return tuple(result)
