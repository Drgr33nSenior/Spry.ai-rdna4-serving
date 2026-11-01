"""Configured caller authority/source joins; every external effect is substituted."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load
from test_import_cleanup import stale_bytecode
from test_config_binding_proposal import canonical,sha

caller=load('configured_caller_tests',HERE.parent/'launcher/run_native.py')


class ConfiguredCaller(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='configured-caller-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.source=self.root/'author';self.work=self.root/'caller'
        self.source.mkdir();self.work.mkdir()
        upload=self.source/'owner-upload';(upload/'review').mkdir(parents=True)
        for name in caller.HELPERS:(upload/'review'/name).write_bytes(b'# pinned inert helper\n')
        (upload/'payload.tar').write_bytes(b'CPU fixture archive');(upload/'RUN-ON-WORKSTATION.sh').write_bytes(b'# CPU fixture script\n')
        for name in ('run_native.py','connection.py','local_commands.py','seal_evidence.py'):
            (self.work/name).write_bytes((HERE.parent/'launcher/run_native.py').read_bytes() if name=='run_native.py' else b'# inert pinned dependency\n')
        nonce='1234567890abcdef'
        (self.source/'package-v2').mkdir()
        manifest=canonical(dict(schema=3))
        (self.source/'package-v2/MANIFEST.json').write_bytes(manifest)
        self.template=dict(nonce=nonce,run_config_sha256='a'*64,host_sha256='b'*64,protected_set_sha256='c'*64,
            manifest_sha256=sha(manifest),archive_sha256=sha((upload/'payload.tar').read_bytes()),
            target_root='/home/developer/fp8-kv-benchmark.'+nonce,protected_root='/var/tmp/fp8-benchmark-source.'+nonce,
            remote_upload_root='/home/developer/fp8-benchmark-upload.'+nonce,total_seconds=6360,caller_seconds=7160,
            upload_inventory={p.relative_to(upload).as_posix():sha(p.read_bytes()) for p in upload.rglob('*') if p.is_file()})
        boot='12345678-1234-1234-1234-123456789abc'
        (self.source/'LAUNCHER-BINDINGS-TEMPLATE.json').write_bytes(canonical(self.template))
        (self.work/'LAUNCHER-BINDINGS.json').write_bytes(canonical(dict(self.template,boot_id=boot)))
        self.auth=dict(status='REGISTERED_CPU_FIXTURE',owner_invocations_allowed=1,model_starts_allowed=2,
            no_automatic_retry=True,deployment_authorized=False,promotion_authorized=False,boot_id=boot,
            source_root=str(self.source),source_seal=dict(root=str(self.source),manifest_sha256='e'*64,members=1),
            owner_seconds=6360,caller_seconds=7160,binding_template_sha256=sha((self.source/'LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes()),
            binding_sha256=sha((self.work/'LAUNCHER-BINDINGS.json').read_bytes()),
            caller_inventory={p.name:sha(p.read_bytes()) for p in self.work.glob('*.py')},
            **{k:self.template[k] for k in ('nonce','run_config_sha256','host_sha256','protected_set_sha256','archive_sha256','manifest_sha256')})
        self.write_auth()

    def write_auth(self):
        raw=canonical(self.auth);(self.work/'RUN-AUTHORIZATION.json').write_bytes(raw);self.auth_pin=sha(raw)

    def validate(self):return caller.validate_context(self.work,self.auth_pin)

    def test_exact_context_validates_without_imports_or_network(self):
        with patch.object(caller,'checked_module') as module:
            auth,binding,source=self.validate()
        module.assert_not_called();self.assertEqual(source,self.source);self.assertEqual(auth,self.auth)
        self.assertEqual(binding['boot_id'],self.auth['boot_id']);self.assertNotIn('boot_id',self.template)

    def test_unfunded_start_allowance_and_boolean_counts_refuse(self):
        original=copy.deepcopy(self.auth)
        for field,value in (('status','UNFUNDED'),('model_starts_allowed',0),('owner_invocations_allowed',True),('no_automatic_retry',False)):
            self.auth=dict(original,**{field:value});self.write_auth()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'authority'):self.validate()

    def test_reapproved_config_and_duration_changes_cannot_replace_source_binding(self):
        original=copy.deepcopy(self.auth)
        for field,value in (('host_sha256','f'*64),('run_config_sha256','f'*64),('protected_set_sha256','f'*64),('owner_seconds',6359),('boot_id','f'*36)):
            self.auth=dict(original,**{field:value});self.write_auth()
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate()

    def test_unreviewed_extra_caller_source_refuses(self):
        (self.work/'extra.py').write_bytes(b'raise AssertionError("must not execute")')
        with self.assertRaisesRegex(ValueError,'membership'):self.validate()

    def test_changed_upload_source_refuses(self):
        (self.source/'owner-upload/payload.tar').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):self.validate()

    def test_checked_loader_ignores_unchecked_cached_bytecode(self):
        path=self.work/'connection.py';path.write_bytes(b'VALUE = "trusted"\n');stale_bytecode(path)
        with patch.dict(sys.modules):
            self.assertEqual(caller.checked_module(self.work,'connection',sha(path.read_bytes())).VALUE,'trusted')

    def test_failed_source_seal_cannot_reach_connection(self):
        sealer=SimpleNamespace(verify=Mock(return_value={'wrong':'seal'}))
        with patch.object(caller,'ROOT',self.work),patch.object(caller,'checked_module',return_value=sealer) as load_module:
            with self.assertRaisesRegex(ValueError,'sealed author'):caller.main(['upload','--authorization-sha256',self.auth_pin])
        self.assertEqual(load_module.call_count,1)

    def effects(self):
        receipt=dict(exit_code=0,elapsed_seconds=1,child_reaped=True,owned_group_absent=True)
        commands=SimpleNamespace(run=Mock(return_value=receipt))
        modules=dict(seal_evidence=SimpleNamespace(verify=Mock(return_value=self.auth['source_seal'])),
                     connection=SimpleNamespace(check=Mock(),SSH=['CPU_FIXTURE_SSH']),local_commands=commands)
        return commands,lambda root,name,pin:modules[name]

    def test_upload_uses_exact_reviewed_inventory_once(self):
        commands,loader=self.effects()
        with patch.object(caller,'ROOT',self.work),patch.object(caller,'checked_module',side_effect=loader):
            caller.main(['upload','--authorization-sha256',self.auth_pin])
        commands.run.assert_called_once();self.assertEqual(commands.run.call_args.args[2],60)
        with tarfile.open(fileobj=io.BytesIO(commands.run.call_args.kwargs['stdin']),mode='r:') as archive:
            self.assertEqual({m.name for m in archive},set(self.template['upload_inventory']))

    def test_launch_requires_completed_upload_before_external_effect(self):
        commands,loader=self.effects()
        with patch.object(caller,'ROOT',self.work),patch.object(caller,'checked_module',side_effect=loader):
            with self.assertRaises(FileNotFoundError):caller.main(['launch','--authorization-sha256',self.auth_pin])
        commands.run.assert_not_called()

    def test_failed_transport_returns_failure_to_parent(self):
        commands,loader=self.effects();commands.run.return_value['exit_code']=1
        with patch.object(caller,'ROOT',self.work),patch.object(caller,'checked_module',side_effect=loader):
            self.assertEqual(caller.main(['upload','--authorization-sha256',self.auth_pin]),1)

    def test_launch_marker_prevents_a_second_dispatch(self):
        commands,loader=self.effects();raw=b'CPU fixture uploaded archive'
        (self.work/'upload.tar').write_bytes(raw)
        (self.work/'upload-once.json').write_bytes(canonical(dict(exit_code=0,child_reaped=True,
            owned_group_absent=True,stdin_sha256=sha(raw))))
        with patch.object(caller,'ROOT',self.work),patch.object(caller,'checked_module',side_effect=loader):
            self.assertEqual(caller.main(['launch','--authorization-sha256',self.auth_pin]),0)
            with self.assertRaises(FileExistsError):caller.main(['launch','--authorization-sha256',self.auth_pin])
        commands.run.assert_called_once()


if __name__=='__main__':unittest.main()
