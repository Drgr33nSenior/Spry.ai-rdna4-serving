"""Fresh CPU tokenizer contract and namespace projection; stdlib only."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import stat
import sys
ROOT=Path(__file__).resolve().parent
BASE='/home/developer/kv4-serving-integration-20261014.M4uQHI-base2'
SDK='/usr/local/lib/python3.14/dist-packages/_rocm_sdk_core'
PYTHON='/mnt/work/build-venv/bin/python3'
MODE='normal-stop-paired-native-quality-352-v1'
FILE_LIMIT=16<<20
OUTPUT_LIMIT=32<<20
MEMBER_LIMIT=128
REQUIRED_SOURCES=set(('CPU-TOKENIZER-INPUT.json', 'EXPECTED-QUALITY.json', 'QUALITY-CPU-INPUT.json', 'SERVING-CHAT-SOURCE.py', 'STACK.json', 'TOKENIZER-BASE-INVENTORY.json', 'TOKENIZER-CONFIG.json', 'TOKENIZER-PREFIX-INVENTORY.zlib', 'common.py', 'compat_bootstrap.py', 'cpu_resources.py', 'decode_original.py', 'quality_contract.py', 'quality_stages.py', 'quality_witness.py', 'review/benchmark_artifacts.py', 'review/benchmark_gpu_identity.py', 'review/benchmark_host_transport.py', 'review/benchmark_owner_adapter.py', 'review/benchmark_physical_controller.py', 'review/benchmark_release_observer.py', 'review/benchmark_resource_gate.py', 'review/collect_benchmark_platform.py', 'review/host_process_owner.py', 'stack_guard.py', 'thinking_segments.py', 'tokenizer_core.py', 'tokenizer_entry.py', 'tokenizer_witness.py'))

def load(name):
    key = 'smoke_private_' + name
    if key in sys.modules:
        return sys.modules[key]
    path = ROOT/'review'/(name+'.py')
    if not path.exists():
        path = ROOT/(name+'.py')
    spec = importlib.util.spec_from_file_location(key,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

def sha(data):
    return hashlib.sha256(data).hexdigest()

def raw(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()

def relative_name(name):
    assert type(name) is str and name and len(name) <= 4096 and '\x00' not in name and '\\' not in name
    path = PurePosixPath(name)
    assert not path.is_absolute() and str(path) == name and all(part not in ('', '.', '..') for part in name.split('/'))
    return path

def stamp(info):
    return tuple(getattr(info, key) for key in ('st_dev','st_ino','st_mode','st_nlink',
        'st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns'))

def checked_file(path, pin, *, maximum, single_link=True):
    """Hash one canonical held regular file; installed hardlinks are explicit."""
    path = Path(path)
    assert type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin)
    assert path.is_absolute() and path.resolve() == path and not path.is_symlink()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        assert stat.S_ISREG(before.st_mode) and before.st_nlink >= 1
        assert not single_link or before.st_nlink == 1
        assert before.st_size <= maximum
        digest = hashlib.sha256()
        count = 0
        while True:
            block = os.read(descriptor, min(1 << 20, maximum + 1 - count))
            if not block:
                break
            count += len(block)
            assert count <= maximum
            digest.update(block)
        assert count == before.st_size and stamp(before) == stamp(os.fstat(descriptor)) == stamp(path.lstat())
        assert digest.hexdigest() == pin, 'source/runtime file identity changed'
        return dict(sha256=pin, bytes=count, device=before.st_dev, inode=before.st_ino,
                    nlink=before.st_nlink)
    finally:
        os.close(descriptor)

def validate_package(path,pin):
    path = Path(path)
    checked_file(path, pin, maximum=1 << 20)
    data = path.read_bytes()
    assert sha(data) == pin
    value = validate_contract(json.loads(data))
    for name,digest in value['sources'].items():
        checked_file(ROOT/name, digest, maximum=(53 << 20) if name=='QUALITY-CPU-INPUT.json' else (2 << 20))
    expected=json.loads((ROOT/'EXPECTED-QUALITY.json').read_bytes())
    assert type(expected) is list and len(expected)==value['final_quality']['rows'], 'declared quality menu row count'
    return value

def runtime_identities(package, base):
    base = Path(base)
    assert base.resolve() == base and base.is_dir()
    return {name: dict(checked_file(base/name, pin, maximum=512 << 20, single_link=False),path=str(base/name))
            for name, pin in package['runtime_files'].items()}

def validate_contract(value):
    assert type(value)is dict and type(value['schema'])is int and value['schema']==1
    assert value['mode']==MODE and value['qualification']is False and value['allow_gpu']is False and value['model_starts']==0 and type(value['model_starts'])is int
    rows=value['final_quality']['rows']
    assert type(rows) is int and 0<rows<=704 and rows%2==0
    assert value['payload']=='tokenizer_entry.py'and value['final_quality']==dict(decoder_runs=1,rows=rows,grader_runs=0,input_bytes=55574528,report_bytes=4194304)
    assert value['worker_seconds']==210 and value['source_seconds']==60 and value['total_seconds']==320
    assert value['cpu_properties']==['CPUQuota=200%','MemoryMax=4294967296','MemorySwapMax=0','TasksMax=128']
    assert type(value['sources'])is dict and REQUIRED_SOURCES==set(value['sources'])and len(value['sources'])<=64
    for manifest in (value['sources'],value['runtime_files']):
        assert type(manifest)is dict and 1<=len(manifest)<=256
        for name,pin in manifest.items():relative_name(name);assert type(pin)is str and re.fullmatch('[0-9a-f]{64}',pin)
    assert re.fullmatch(r'/home/developer/fp8-kv-benchmark\.[a-f0-9]{16}',value['remote_root'])
    return value


def bootstrap_bwrap(package,deadline,*,admission_file,admission_sha256,stack_proof_file,cgroup_directory):
    """Explicit CPU namespace: no GPU node/sys mounts, exactly seven asset files."""
    validate_contract(package)
    assert type(deadline)in(int,float)and math.isfinite(deadline)and deadline>0
    checked_file(Path(admission_file),admission_sha256,maximum=1<<20)
    admission=json.loads(Path(admission_file).read_bytes())
    checked_file(Path(stack_proof_file),admission['stack_proof_sha256'],maximum=1<<20)
    assert str(cgroup_directory)=='/sys/fs/cgroup'+admission['host_cgroup_path']and Path(cgroup_directory).resolve()==Path(cgroup_directory)
    return namespace_argv(package,deadline,admission=admission,admission_file=admission_file,admission_sha256=admission_sha256,stack_proof_file=stack_proof_file,cgroup_directory=cgroup_directory)

def namespace_argv(package,deadline,*,admission,admission_file,admission_sha256,stack_proof_file,cgroup_directory):
    """Exact source-derived namespace argv, also joined by the retained reader."""
    root=package['remote_root'];stack=json.loads((ROOT/'STACK.json').read_bytes());frozen=json.loads((ROOT/'CPU-TOKENIZER-INPUT.json').read_bytes())
    argv=['/usr/bin/bwrap','--die-with-parent','--new-session','--unshare-all','--uid','1000','--gid','1000',
      '--ro-bind',BASE+'/base-rootfs','/','--tmpfs','/mnt','--ro-bind',root,'/mnt/review','--ro-bind',BASE+'/build-output3/build-venv','/mnt/work/build-venv',
      '--bind',root+'/output','/mnt/output','--proc','/proc','--dev','/dev','--size',str(256<<20),'--tmpfs','/tmp','--size',str(64<<20),'--tmpfs','/dev/shm',
      '--ro-bind',stack['runtime_root'],'/usr/local/lib/python3.14/dist-packages','--ro-bind',stack['toolchain']['root'],'/mnt/rocm-devel',
      '--ro-bind',stack['compatibility']['sitepackages'],'/mnt/compat',
      '--ro-bind',str(admission_file),'/mnt/admission.json','--ro-bind',str(stack_proof_file),'/mnt/stack-proof.json',
      '--ro-bind',str(cgroup_directory),'/mnt/cpu-cgroup','--ro-bind',str(Path(admission_file).parent/'TOKENIZER-RUNTIME-INVENTORY.zlib'),'/mnt/tokenizer-runtime-inventory.zlib',
      '--dir','/mnt/tokenizer-model','--clearenv']
    assets={**frozen['model_files'],'config.json':frozen['model_config'],'generation_config.json':frozen['model_generation']}
    assert len(assets)==7
    for name in sorted(assets):argv+=['--ro-bind',frozen['model_root']+'/'+name,'/mnt/tokenizer-model/'+name]
    env=dict(PATH=stack['runtime_path'],LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',HOME='/tmp',TMPDIR='/tmp',XDG_CACHE_HOME='/tmp/cache',
      TRITON_CACHE_DIR='/tmp/cache/triton',TORCH_EXTENSIONS_DIR='/tmp/cache/torch',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',HIPCC_JOBS='2',TRITON_COMPILE_THREADS='1',
      ROCM_HOME=SDK,ROCM_PATH=SDK,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='2',USE_TF='0',USE_FLAX='0',USE_TORCH='1')
    env.update(stack['toolchain_environment']);env.update(stack['environment'])
    for key,value in env.items():argv+=['--setenv',key,value]
    pin=sha((ROOT/'PACKAGE.json').read_bytes())
    argv += [PYTHON,'-I','-B','-S','/mnt/review/tokenizer_entry.py','--deadline-monotonic',str(deadline),'--package-sha256',pin,'--admission-sha256',admission_sha256]
    return argv
