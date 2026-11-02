"""Run the consumed template reader and complete materializer with filesystem seams."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import build as builder


@contextmanager
def modules():
    previous=dict(sys.modules);paths=list(sys.path)
    try:yield
    finally:
        for name in tuple(sys.modules):
            if name not in previous:sys.modules.pop(name,None)
        sys.modules.update(previous);sys.path[:]=paths


def check(review,*,expected_error=None):
    review=Path(review).resolve(strict=True);reports=[]
    for arm in ('control','mtp'):
        with modules(),tempfile.TemporaryDirectory(prefix='worker-materialization-') as temporary:
            root=Path(temporary).resolve();view=root/'review';view.mkdir()
            files={p.name:p.read_bytes() for p in review.iterdir() if p.is_file()}
            policy=builder.production_evidence_policy(files)
            for name,raw in files.items():(view/name).write_bytes(raw)
            if policy is not None:
                for name in policy['files']:(view/name).write_bytes(policy['raw'])
            sys.path.insert(0,str(view));path=view/'serving_runtime.py'
            spec=importlib.util.spec_from_file_location('serving_runtime',path)
            runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
            name='HTTP-BINDING-'+arm+'.json';raw=(review/name).read_bytes();authored=json.loads(raw)
            sources=authored['binding_template']['sources']
            original_path=runtime._canonical_regular;original_digest=runtime._file_digest
            original_fresh=runtime._fresh_directory;original_write=runtime._write_exclusive
            stock={row['path']:row['sha256'] for row in sources.values() if not row['path'].startswith('/mnt/review/')}
            logical_output=Path(authored['binding_template']['evidence_dir']).parent
            actual_output=root/'output';actual_output.mkdir()
            masks={}
            for value in authored['binding_template']['bytecode_masks']:
                target=root/'mask-view'/Path(value).relative_to('/');target.mkdir(parents=True,exist_ok=True);masks[value]=target
            hashed=[];written=[]
            def physical_output(path):
                path=Path(path);assert path.is_relative_to(logical_output)
                return actual_output/path.relative_to(logical_output)
            def canonical(value):
                value=str(value)
                if value.startswith('/mnt/review/'):return original_path(view/Path(value).name)
                if Path(value).is_relative_to(logical_output):return original_path(physical_output(value))
                assert value in stock,('unsubstituted stock source',value)
                return Path(value)
            def digest(path):
                if str(path) in stock:return stock[str(path)]
                if Path(path).parent==view:hashed.append(Path(path).name)
                return original_digest(path)
            def fresh(path):
                original_fresh(physical_output(path));return Path(path)
            def write(path,value):
                target=physical_output(path);pin=original_write(target,value);written.append(target);return pin
            with patch.object(runtime,'_canonical_regular',side_effect=canonical), \
                    patch.object(runtime,'_file_digest',side_effect=digest), \
                    patch.object(runtime,'_fresh_directory',side_effect=fresh), \
                    patch.object(runtime,'_write_exclusive',side_effect=write), \
                    patch.object(runtime.serving_entry,'_canonical_directory',side_effect=lambda value:masks[value]):
                template,pin=runtime._read_template(dict(path='review/'+name,sha256=hashlib.sha256(raw).hexdigest()),arm=arm)
                try:
                    material=runtime._materialize(template,template_sha256=pin,session_sha256='b'*64,
                        spec_sha256='c'*64,output=logical_output,deadline=10**12,
                        expected_devices=template['worker_template']['expected_devices'])
                except runtime.ServingRuntimeError as error:
                    assert expected_error is not None and str(error)==expected_error,(arm,str(error),expected_error)
                    assert written==[]
                    material=None
                else:
                    assert expected_error is None,('expected refusal',expected_error)
                    assert len(written)==2 and {p.name for p in written}=={'WORKER-MANIFEST.json','SERVING-BINDING.json'}
                    worker=json.loads(physical_output(material['worker_manifest_path']).read_bytes())
                    binding=json.loads(physical_output(material['entry_binding_path']).read_bytes())
                    assert worker['sources']==binding['sources']==sources
                    assert worker['session_sha256']==binding['session_sha256']=='b'*64
                    assert worker['spec_sha256']==binding['spec_sha256']=='c'*64
                    assert worker['preweight']['deadline_monotonic']==10**12
                    for helper in ('gpu_identity','graph_observer'):assert worker['preweight'][helper]==sources[helper]
                    assert hashlib.sha256(physical_output(material['worker_manifest_path']).read_bytes()).hexdigest()==binding['worker_manifest_sha256']
                    entry=runtime.serving_entry
                    with patch.object(entry,'_canonical_regular',side_effect=canonical),patch.object(entry,'_file_digest',side_effect=digest):
                        entry._validate_worker_cross_binding(binding)
            private_names={Path(row['path']).name for row in sources.values() if row['path'].startswith('/mnt/review/')}
            assert private_names<=set(hashed)
            reports.append(dict(arm=arm,template_reader_complete=True,complete_materialization=material is not None,
                expected_error=expected_error,private_sources_checked=len(private_names),actual_private_file_hashing=True,
                stock_image_sources_substituted=True,output_filesystem_redirected=True,empty_bytecode_mounts_substituted=True,
                production_evidence_mask_applied=policy is not None,worker_cross_binding_complete=material is not None,
                generated_manifest_files=len(written),native_entry_admission=False,model_construction=False,gpu_work=False))
    return dict(passed=True,arms=reports,scope='actual template reader, complete materializer and written worker cross-binding',
        native_serving_or_quality_proven=False)


if __name__=='__main__':
    print(json.dumps(check(sys.argv[1],expected_error=sys.argv[2] if len(sys.argv)>2 else None),sort_keys=True))
