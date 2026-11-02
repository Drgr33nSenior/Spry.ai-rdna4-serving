"""Run the complete startup proof producer and real reader with metadata substitutes."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_draft_preflight


def source(path,module):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),module=module,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@contextmanager
def loaded(review):
    with check_draft_preflight.loaded(review) as (contract,worker):
        review=Path(review).resolve(strict=True)
        adapter_source=source(review/'verify_decode_adapter.py','verify_decode_adapter')
        adapter_module=worker._source_module(adapter_source)
        reader=worker._source_module(source(review/'serving_integration.py','serving_integration'))
        yield contract,worker,adapter_module,adapter_source,reader


def run_case(contract,worker,adapter_module,adapter_source,reader,*,arm='control',rank=0,
             mutate_adapter=None,module_policy=None,remove_module_policy=False,expected_error=None,
             reader_error=None):
    """Keep the producer, private module authentication, writer and full-enabled reader real."""
    context=contract.precision_profile()['context_length']
    adapter=object.__new__(adapter_module.Adapter)
    adapter.enabled=True;adapter.eligible_host_calls=40;adapter.capture_host_calls=32
    adapter.metadata_build_host_calls=2;adapter.metadata_reuse_host_calls=40;adapter.metadata_capture_build_host_calls=0
    adapter.layers=set(range(16))
    adapter.buffers={n:dict(indices=SimpleNamespace(numel=lambda n=n:n*(context+8))) for n in (8,16)}
    assert not hasattr(adapter,'METADATA_PREPARATION_POLICY')
    if mutate_adapter:mutate_adapter(adapter)
    linear_source=dict(module='sglang.srt.layers.linear',path='/mnt/composition/linear.py',sha256='a'*64)
    method_source=dict(module='sglang.srt.layers.quantization.quark',path='/mnt/composition/quark.py',sha256='b'*64)
    linear_class=type('RowParallelLinear',(),{'__module__':linear_source['module']})
    method_class=type('QuarkLinearMethod',(),{'__module__':method_source['module']})
    linear=linear_class();linear.quant_method=method_class()
    linear.weight=SimpleNamespace(dtype='torch.uint8',shape=[17408,4352],device='cuda:0',is_cuda=True,is_contiguous=lambda:True)
    pointer=lambda n:SimpleNamespace(data_ptr=lambda:n)
    target=SimpleNamespace(b8=pointer(101),scales=pointer(202))
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=linear))]),
        lm_head=SimpleNamespace(_spry_shared_draft_binding=dict(fp8_pointer=101,scales_pointer=202)),
        logits_processor=SimpleNamespace(_compute_lm_head=target))
    runner=SimpleNamespace(model=model,model_config=SimpleNamespace(quantization=contract.target_quantization()),
        _spry_memory_recovery=dict(cpu_metadata_fixture=True))
    scheduler=SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
    native={linear_source['module']:SimpleNamespace(RowParallelLinear=linear_class),
            method_source['module']:SimpleNamespace(QuarkLinearMethod=method_class)}
    observer=SimpleNamespace(validate_precision_weights=Mock())
    identity=dict(rank=rank,pid=123+rank,start_ticks=456,boot_id='fixture-boot',identity_sha256='c'*64)
    with tempfile.TemporaryDirectory(prefix='startup-proof-') as temporary:
        root=Path(temporary).resolve()
        manifest=dict(mode=contract.MODE,arm=arm,session_sha256='d'*64,spec_sha256='e'*64,
            source_manifest_sha256='f'*64,evidence_dir=str(root),resolved_contract=dict(context_length=context),
            sources=dict(head_linear=linear_source,quark_config=method_source,verify_decode_adapter=adapter_source))
        original_source=worker._source_module
        with patch.object(adapter_module,'CURRENT',adapter), \
                patch.object(worker,'_native_module',side_effect=lambda row:native[row['module']]), \
                patch.object(worker,'_precision_loading_policy',return_value=dict(cpu_metadata_fixture=True)), \
                patch.object(worker,'_mxfp4_loaded_weights',return_value=dict(cpu_metadata_fixture=True)), \
                patch.object(worker,'_source_module',wraps=original_source) as module_loader:
            original_policy=adapter_module.METADATA_PREPARATION_POLICY
            try:
                if remove_module_policy:del adapter_module.METADATA_PREPARATION_POLICY
                elif module_policy is not None:adapter_module.METADATA_PREPARATION_POLICY=module_policy
                try:record=worker._precision_weights(scheduler,manifest,identity,observer,contract,{})
                except Exception as error:
                    assert expected_error is not None and expected_error in str(error),(type(error).__name__,str(error),expected_error)
                    assert (root/f'PRECISION-WEIGHTS-rank-{rank}.json').is_file()
                    assert not (root/f'FULL-ENABLED-rank-{rank}.json').exists()
                    return dict(arm=arm,rank=rank,expected_error=expected_error,precision_witness_retained=True,
                        incomplete_full_enabled_witness_absent=True,model_construction=False,gpu_work=False)
            finally:adapter_module.METADATA_PREPARATION_POLICY=original_policy
            assert expected_error is None,('expected refusal',expected_error)
            observer.validate_precision_weights.assert_called_once()
            module_loader.assert_called_once_with(adapter_source)
            proof=json.loads((root/f'FULL-ENABLED-rank-{rank}.json').read_bytes())
            assert json.loads((root/f'PRECISION-WEIGHTS-rank-{rank}.json').read_bytes())==record
            assert len(list(root.iterdir()))==2
            try:reader.validate_full_enabled_record(proof,manifest,identity)
            except Exception as error:
                assert reader_error is not None and reader_error in str(error),(type(error).__name__,str(error),reader_error)
                return dict(arm=arm,rank=rank,expected_reader_error=reader_error,reader_refused=True,
                    model_construction=False,gpu_work=False)
            assert reader_error is None,('expected reader refusal',reader_error)
            assert proof['metadata_preparation_policy']==original_policy=='outside-forward-graph-v1'
            return dict(arm=arm,rank=rank,complete_producer=True,real_full_enabled_reader_passed=True,
                authenticated_adapter_module_used=True,metadata_policy_read_from_module=True,
                native_linear_and_weight_validation_substituted=True,metadata_buffers_and_adapter_lifecycle_substituted=True,
                retained_files=2,model_construction=False,gpu_work=False)


def check(review,*,expected_error=None):
    reports=[]
    with loaded(review) as values:
        for arm in ('control','mtp'):
            for rank in (0,1):reports.append(run_case(*values,arm=arm,rank=rank,expected_error=expected_error))
    return dict(passed=True,scope='complete actual startup proof producer and full-enabled reader with metadata substitutes',
        cases=reports,native_serving_or_quality_proven=False)


if __name__=='__main__':
    print(json.dumps(check(sys.argv[1],expected_error=sys.argv[2] if len(sys.argv)>2 else None),sort_keys=True))
