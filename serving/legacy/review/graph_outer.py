"""Explicit graph-only outer projection and observed memory continuation policy."""

from contextlib import contextmanager
import hashlib
from importlib.machinery import SourceFileLoader
import math
from pathlib import Path
import sys


MODE = 'native-model-context-retrieval-32k-v1'
PREFIX = '/mnt/capture-bytecode'
MASK_DIRS = ('model_executor', 'model_executor/runner_backend', 'layers', 'layers/attention', 'models')
BACKEND_MODULE = 'sglang.srt.model_executor.runner_backend.full_cuda_graph_backend'
BACKEND_PATH = '/mnt/composition/image-source/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py'
MEMORY_POLICY = dict(before_free_min_bytes=2 << 30, after_free_min_bytes=1 << 30,
                     growth_max_bytes=1 << 30, hard_transient_peak_quota=False)


def need(value, message):
    if not value: raise ValueError('graph outer: '+message)


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
         {'graph_screen.py','graph_integration.py','graph_outer.py','context_tokenizer.py'}, 'private helper inventory')
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


def backend_identity(namespace, spec, backend):
    cls=type(backend)
    module=sys.modules.get(BACKEND_MODULE)
    origin=getattr(getattr(module,'__spec__',None),'origin',None)
    loader=getattr(getattr(module,'__spec__',None),'loader',None)
    need(cls.__name__=='FullCudaGraphBackend' and cls.__module__==BACKEND_MODULE and
         module is not None and module.__file__==origin==BACKEND_PATH and
         getattr(module,'FullCudaGraphBackend',None) is cls and isinstance(loader,SourceFileLoader) and
         loader.path==BACKEND_PATH and str(Path(BACKEND_PATH).resolve())==BACKEND_PATH,
         'actual native backend module/class/loader origin')
    row=validate_projection(spec['graph_projection'])['overlay']
    raw=namespace['checked_file'](BACKEND_PATH,row['sha256'])
    return dict(module=BACKEND_MODULE,path=BACKEND_PATH,sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),class_name=cls.__name__,source_checked_before_capture=True)


def bind_backend(namespace, spec, observer):
    original_capture,original_replay=observer.capture_forward,observer.backend_replayed
    observer.native_backend_identity=None
    backend_type=[None]
    def capture(backend,shape_key,phase,warmup_step=None):
        if backend_type[0] is None:
            need(phase=='warmup' and warmup_step==0,'source identity before first native warmup')
            observer.native_backend_identity=backend_identity(namespace,spec,backend)
            backend_type[0]=type(backend)
        need(type(backend) is backend_type[0],'same authenticated native backend type')
        return original_capture(backend,shape_key,phase,warmup_step)
    def replay(backend,shape_key,static_forward_batch):
        need(backend_type[0] is not None and type(backend) is backend_type[0] and
             getattr(sys.modules.get(BACKEND_MODULE),'FullCudaGraphBackend',None) is backend_type[0],
             'replay uses authenticated loaded backend without reload')
        return original_replay(backend,shape_key,static_forward_batch)
    observer.capture_forward,observer.backend_replayed=capture,replay
    return observer


def snapshot(torch,rank):
    torch.cuda.synchronize(rank)
    free,total=torch.cuda.mem_get_info(rank)
    result=dict(rank=rank,device=rank,free_bytes=int(free),total_bytes=int(total),
        allocated_bytes=int(torch.cuda.memory_allocated(rank)),reserved_bytes=int(torch.cuda.memory_reserved(rank)),
        lifetime_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(rank)),
        lifetime_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(rank)))
    need(all(type(value) is int and value>=0 for value in result.values()) and
         0<result['total_bytes'] and result['free_bytes']<=result['total_bytes'] and
         result['allocated_bytes']<=result['reserved_bytes'],'actual memory byte counters')
    return result


def before_ok(row): return row['free_bytes']>=MEMORY_POLICY['before_free_min_bytes']


def after_ok(before,after):
    return (before['rank']==after['rank'] and before['total_bytes']==after['total_bytes'] and
        after['free_bytes']>=MEMORY_POLICY['after_free_min_bytes'] and
        max(after['allocated_bytes']-before['allocated_bytes'],after['reserved_bytes']-before['reserved_bytes'],
            before['free_bytes']-after['free_bytes'])<=MEMORY_POLICY['growth_max_bytes'])


def gather(dist,group,value):
    peers=[None,None]
    dist.all_gather_object(peers,value,group=group)
    need(all(type(row) is dict and type(row.get('rank')) is int and row.get('rank')==rank for rank,row in enumerate(peers)), 'memory TP rank agreement')
    return peers


@contextmanager
def memory_scope(torch,rank,arm,dist,group,result):
    result.update(schema=1,mode=MODE,rank=rank,arm=arm,policy=dict(MEMORY_POLICY),complete=False,
        before=None,after=None,peers_before=None,peers_after=None,error=None,
        scope='Synchronized headroom/continuation snapshots and lifetime peaks; not a hard transient peak quota')
    before=None
    try:
        before=snapshot(torch,rank)
        local=dict(rank=rank,snapshot=before,error=None,allowed=before_ok(before))
    except BaseException as error:
        local=dict(rank=rank,snapshot=None,error=(type(error).__name__+': '+str(error))[:512],allowed=False)
    result['before']=before
    result['peers_before']=gather(dist,group,local)
    if not all(row['allowed'] is True for row in result['peers_before']):
        result['error']='before-capture headroom refused by at least one TP rank'
        need(False,result['error'])
    active_error=False
    try:
        yield result
    except BaseException as error:
        active_error=True
        result['error']=(type(error).__name__+': '+str(error))[:512]
        raise
    finally:
        after=None
        try:
            after=snapshot(torch,rank)
            local=dict(rank=rank,snapshot=after,error=None,allowed=after_ok(before,after))
        except BaseException as error:
            local=dict(rank=rank,snapshot=None,error=(type(error).__name__+': '+str(error))[:512],allowed=False)
        result['after']=after
        # Do not add a collective while a native exception is already unwinding.
        if not active_error:
            result['peers_after']=gather(dist,group,local)
            result['complete']=all(row['allowed'] is True for row in result['peers_after'])
            if not result['complete']:
                result['error']='after-capture continuation refused by at least one TP rank'
                need(False,result['error'])


def validate_memory(spec,rows,logical_arm):
    need(logical_arm in ('eager','graph'),'explicit validated logical arm')
    validate_projection(spec['graph_projection'])
    def valid_snapshot(value,rank):
        keys={'rank','device','free_bytes','total_bytes','allocated_bytes','reserved_bytes',
              'lifetime_peak_allocated_bytes','lifetime_peak_reserved_bytes'}
        return (type(value) is dict and set(value)==keys and all(type(item) is int and item>=0 for item in value.values()) and
                value['rank']==value['device']==rank and 0<value['total_bytes'] and value['free_bytes']<=value['total_bytes'] and
                value['allocated_bytes']<=value['reserved_bytes']<=value['total_bytes'] and
                value['lifetime_peak_allocated_bytes']>=value['allocated_bytes'] and
                value['lifetime_peak_reserved_bytes']>=value['reserved_bytes'])
    for rank,row in enumerate(rows):
        record=row.get('graph_memory')
        need(type(record) is dict and type(record.get('schema')) is int and record['schema']==1 and
             record.get('mode')==MODE and type(record.get('rank')) is int and record.get('rank')==rank and
             record.get('arm')==row['arm'] and record.get('policy')==MEMORY_POLICY and record.get('complete') is True and
             record.get('error') is None,'complete bound graph memory continuation evidence')
        before,after=record.get('before'),record.get('after')
        need(valid_snapshot(before,rank) and valid_snapshot(after,rank) and before_ok(before) and after_ok(before,after),
             'memory thresholds')
        for phase in ('before','after'):
            peers=record.get('peers_'+phase)
            need(type(peers) is list and len(peers)==2 and peers[rank].get('snapshot')==record[phase] and
                 all(type(item.get('rank')) is int and item.get('rank')==index and item.get('allowed') is True and item.get('error') is None
                     and valid_snapshot(item.get('snapshot'),index) for index,item in enumerate(peers)), 'memory TP peer continuation join')
        identity=row.get('graph_backend_identity')
        if logical_arm=='graph':
            overlay=spec['graph_projection']['overlay']
            need(type(identity) is dict and identity.get('module')==BACKEND_MODULE and identity.get('path')==BACKEND_PATH and
                 identity.get('sha256')==overlay['sha256'] and identity.get('source_checked_before_capture') is True,
                 'loaded private backend source identity required')
        else: need(identity is None,'eager must not claim native graph capture')
    for phase in ('before','after'):
        need(rows[0]['graph_memory']['peers_'+phase]==rows[1]['graph_memory']['peers_'+phase],
             'both TP ranks retain the same memory decision')
    for rank in (0,1):
        before=rows[0]['graph_memory']['peers_before'][rank]['snapshot']
        after=rows[0]['graph_memory']['peers_after'][rank]['snapshot']
        need(before_ok(before) and after_ok(before,after),'all peer headroom thresholds')
