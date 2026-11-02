"""Execute the actual selected preflight with metadata-only native substitutes."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch


@contextmanager
def loaded(review):
    previous=dict(sys.modules)
    try:
        values=[]
        for name in ('mtp_contract','worker_entry'):
            spec=importlib.util.spec_from_file_location(name,Path(review)/(name+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);values.append(module)
        yield values
    finally:
        for name in tuple(sys.modules):
            if name not in previous:sys.modules.pop(name,None)
        sys.modules.update(previous)


def run_case(worker,contract,*,arm='control',rank=0,mutate_draft=None,mutate_config=None,
             wrong_file_digest=False,expected_error=None):
    """Keep contract, equality checks and JSON writer real; substitute native metadata."""
    selected=contract.validate_contract(contract.contract_for_arm(arm))
    draft_config=copy.deepcopy(contract.draft_model_binding()['config'])
    fields=dict(selected)
    fields.update(contract.memory_pool_config(arm));fields.update(contract.cache_config())
    fields.update(contract.precision_loading_policy())
    fields['model_loader_extra_config']=json.dumps(fields['model_loader_extra_config'],sort_keys=True,separators=(',',':'))
    fields.update(quantization=contract.target_quantization(),
        _speculative_draft_quantization_explicitly_set=selected['speculative_draft_quantization_explicit'],
        speculative_use_rejection_sampling=False,speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,enable_metrics=True,
        decode_log_interval=contract.expected_decode_log_interval(arm))
    cfg=SimpleNamespace(**fields)
    if mutate_config:mutate_config(cfg)
    draft=SimpleNamespace(is_draft_model=True,hf_config=SimpleNamespace(**draft_config),
        hf_text_config=SimpleNamespace(),quantization=None,is_draft_quantization_explicit=True,dtype='torch.bfloat16')
    if mutate_draft:mutate_draft(draft)
    constructor=Mock(return_value=draft);resolver=Mock(return_value=cfg)
    native=SimpleNamespace(resolving_view=resolver,ModelConfig=SimpleNamespace(from_server_args=constructor))
    with tempfile.TemporaryDirectory(prefix='draft-preflight-') as temporary:
        root=Path(temporary).resolve();config_path=root/'config.json';config_path.write_text(json.dumps(draft_config))
        evidence=root/'evidence';evidence.mkdir()
        model_source=dict(module='sglang.srt.configs.model_config',path='/mnt/composition/model_config.py',sha256='a'*64)
        contract_path=Path(contract.__file__);contract_source=dict(path=str(contract_path),sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest())
        manifest=dict(mode=contract.MODE,arm=arm,session_sha256='b'*64,spec_sha256='c'*64,
            source_manifest_sha256='d'*64,evidence_dir=str(evidence),sources=dict(mtp_contract=contract_source,model_config=model_source))
        def source(row):
            assert row==dict(contract_source,module='mtp_contract')
            assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==row['sha256']
            return contract
        original_safe=worker._safe_regular;original_digest=worker._file_digest
        def safe(path):
            assert Path(path)==Path('/mnt/draft-model/config.json')
            return original_safe(config_path)
        def digest(path):
            if Path(path)==config_path:return '0'*64 if wrong_file_digest else selected['draft_model_config_sha256']
            return original_digest(path)
        with patch.dict(os.environ,contract.expected_environment(arm),clear=True), \
                patch.object(worker,'_source_module',side_effect=source), \
                patch.object(worker,'_native_module',return_value=native), \
                patch.object(worker,'_validate_precision_config'), \
                patch.object(worker,'_safe_regular',side_effect=safe), \
                patch.object(worker,'_file_digest',side_effect=digest), \
                patch.object(contract,'expected_draft_summary',wraps=contract.expected_draft_summary) as summary:
            try:value=worker._mtp_preflight(manifest,rank,cfg)
            except Exception as error:
                assert expected_error is not None and expected_error in str(error),(type(error).__name__,str(error),expected_error)
                assert list(evidence.iterdir())==[]
                return dict(arm=arm,rank=rank,expected_error=expected_error,refused_before_evidence=True,
                    native_metadata_constructor_calls=constructor.call_count,model_construction=False,gpu_work=False)
            assert expected_error is None,('expected refusal',expected_error)
            assert summary.call_args_list and all(call.args==('dflash2',) for call in summary.call_args_list)
            constructor.assert_called_once_with(cfg,model_path=selected['speculative_draft_model_path'],
                model_revision=selected['speculative_draft_model_revision'],is_draft_model=True)
            assert value['draft_model_config']==contract.expected_draft_summary('dflash2')
            assert value['before_scheduler'] is True and value['qualification'] is False
            expected={f'MTP-PREFLIGHT-rank-{rank}.json'}|({'MTP-PREFLIGHT.json'} if rank==0 else set())
            assert {p.name for p in evidence.iterdir()}==expected
            for p in evidence.iterdir():assert json.loads(p.read_bytes())==value
            return dict(arm=arm,rank=rank,complete_preflight=True,native_metadata_constructor_calls=1,
                exact_observed_draft_summary=True,retained_files=len(expected),
                native_resolver_and_model_config_substituted=True,draft_file_digest_substituted=True,
                stock_precision_source_validation_substituted=True,model_construction=False,gpu_work=False)


def check(review,*,expected_error=None):
    reports=[]
    with loaded(review) as (contract,worker):
        for arm in ('control','mtp'):
            for rank in (0,1):reports.append(run_case(worker,contract,arm=arm,rank=rank,expected_error=expected_error))
    return dict(passed=True,scope='complete actual selected draft preflight with real contract and writer',
        cases=reports,native_serving_or_quality_proven=False)


if __name__=='__main__':
    print(json.dumps(check(sys.argv[1],expected_error=sys.argv[2] if len(sys.argv)>2 else None),sort_keys=True))
