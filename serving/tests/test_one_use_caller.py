"""CPU-only authority and failure retention checks for the one-use caller."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load
from test_config_binding_proposal import canonical,sha

driver=load('one_use_driver_tests',HERE.parent/'launcher/execute_once.py')
caller=load('one_use_caller_tests',HERE.parent/'launcher/run_native.py')


class OneUseSequence(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='one-use-cpu-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.names=[];self.fail={};self.physical=True;self.stopped=0
        self.grant=dict(caller_seconds=7160,owner_seconds=6360,global_gpu_attempt_index=1,nonce='a'*16)

    def runner(self,name,argv,seconds):
        self.names.append(name)
        failure=self.fail.get(name,0)
        if isinstance(failure,BaseException):raise failure
        if name=='collect' and failure==0:
            path=self.root/'collected/protected';path.mkdir(parents=True)
            (path/('fp8-benchmark-source.'+self.grant['nonce']+'.closeout.json')).write_bytes(canonical(dict(physical_success=self.physical)))
        if name=='closeout':self.observer=argv[0]
        return dict(name=name,exit_code=failure)

    def sampler(self):
        def stop():self.stopped+=1;return dict(name='host-sample',exit_code=0)
        return stop

    def run_once(self,**kwargs):
        return driver.run(self.grant,self.root,self.runner,self.sampler,authorization_sha256='f'*64,**kwargs)

    def test_success_retains_full_sequence_and_never_qualifies(self):
        report=self.run_once()
        self.assertEqual(report['status'],'COMPLETE');self.assertEqual(self.stopped,1)
        self.assertEqual(self.names,['power-pre','upload','launch','collect','closeout','power-post','host-faults'])
        self.assertFalse(report['automatic_retry']);self.assertFalse(report['qualification'])
        self.assertEqual(self.observer,'observe_closeout.py')
        with self.assertRaises(FileExistsError):self.run_once()

    def test_precheck_failure_stops_before_upload(self):
        self.fail['power-pre']=1
        self.assertEqual(self.run_once()['status'],'STOPPED_BEFORE_LAUNCH_PRE_CHECK')
        self.assertEqual(self.names,['power-pre']);self.assertEqual(self.stopped,0)

    def test_upload_failure_stops_before_launch(self):
        self.fail['upload']=1
        self.assertEqual(self.run_once()['status'],'STOPPED_BEFORE_LAUNCH_UPLOAD')
        self.assertNotIn('launch',self.names)

    def test_short_caller_reserve_cannot_truncate_owner(self):
        values=iter([0,300,300])
        self.assertEqual(self.run_once(clock=lambda:next(values))['status'],'STOPPED_BEFORE_LAUNCH_CALLER_RESERVE')
        self.assertNotIn('launch',self.names)

    def test_failed_launch_still_collects_and_observes_successful_physical_release(self):
        self.fail['launch']=1
        self.assertEqual(self.run_once()['status'],'INCOMPLETE_RETAINED')
        self.assertEqual(self.names.count('launch'),1);self.assertEqual(self.stopped,1)
        self.assertEqual(self.observer,'observe_closeout.py');self.assertEqual(self.names[-1],'host-faults')

    def test_launch_exception_still_stops_sampler_and_collects(self):
        self.fail['launch']=RuntimeError('CPU fixture failure')
        report=self.run_once();self.assertEqual(report['status'],'INCOMPLETE_RETAINED')
        self.assertIn('collect',self.names);self.assertEqual(self.stopped,1)
        self.assertIn('RuntimeError',next(row for row in report['steps'] if row['name']=='launch')['failure'])

    def test_collection_failure_still_attempts_closeout_and_postchecks(self):
        self.fail['collect']=1
        self.assertEqual(self.run_once()['status'],'INCOMPLETE_RETAINED')
        self.assertEqual(self.observer,'observe_closeout.py');self.assertEqual(self.names[-2:],['power-post','host-faults'])

    def test_failed_physical_closeout_uses_failure_observer(self):
        self.physical=False
        self.assertEqual(self.run_once()['status'],'INCOMPLETE_RETAINED')
        self.assertEqual(self.observer,'observe_failed_closeout.py')

    def test_sampler_start_failure_retains_postchecks_without_launch(self):
        def failed_sampler():raise OSError('CPU fixture sampler')
        report=driver.run(self.grant,self.root,self.runner,failed_sampler,authorization_sha256='f'*64)
        self.assertEqual(report['status'],'INCOMPLETE_RETAINED');self.assertNotIn('launch',self.names)
        self.assertIn('collect',self.names);self.assertEqual(self.names[-1],'host-faults')


class ApprovalJoins(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='one-use-approval-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.source=self.root/'author';self.source.mkdir()
        (self.source/'PROTECTED-COMPLETE.patch').write_bytes(b'CPU fixture protected delta\n')
        self.request={key:'a'*64 for key in ('run_config_sha256','host_sha256','protected_set_sha256',
            'manifest_sha256','archive_sha256','binding_template_sha256')}
        self.request.update(nonce='a'*16,caller_inventory={'execute_once.py':'b'*64},observed_boot_id='CPU_FIXTURE_BOOT',
            requested_owner_seconds=6360,requested_caller_seconds=7160,requested_model_starts=2,
            requested_generations=352,requested_cpu_grading_seconds=320,
            protected_patch_sha256=sha((self.source/'PROTECTED-COMPLETE.patch').read_bytes()))
        (self.source/'CALLER-SOURCES.json').write_bytes(canonical(self.request['caller_inventory']))
        raw=canonical(self.request);(self.root/'APPROVAL-REQUEST.json').write_bytes(raw)
        seal=dict(root=str(self.source),manifest_sha256='c'*64,members=1)
        approval=dict(owner_answer='CPU fixture only; external effects denied',approval_request_sha256=sha(raw),source_seal=seal)
        raw=canonical(approval);path=self.root/'OWNER-APPROVAL.json';path.write_bytes(raw)
        record=dict(path=str(path),sha256=sha(raw))
        self.grant={k:self.request[k] for k in ('nonce','run_config_sha256','host_sha256','protected_set_sha256',
            'manifest_sha256','archive_sha256','binding_template_sha256','caller_inventory')}
        self.grant.update(registered=True,proposal=False,gpu_allowance=True,global_gpu_attempt_index=1,
            approval_record=record,registration_authority=record,owner_answer=approval['owner_answer'],source_seal=seal,
            approval_request_sha256=approval['approval_request_sha256'],owner_seconds=6360,caller_seconds=7160,
            model_starts_allowed=2,generations=352,cpu_grading_seconds=320,boot_id='CPU_FIXTURE_BOOT')
        self.register()

    def register(self):
        self.grant.pop('registration_sha256',None)
        raw=canonical(self.grant);(self.root/'REGISTRATION.json').write_bytes(raw)
        self.grant['registration_sha256']=sha(raw)

    def validate(self):return driver.validate_approval(caller,self.root,self.grant,self.source)

    def test_exact_registered_request_and_source_seal_join(self):self.assertEqual(self.validate(),self.request)

    def test_rehashed_registration_cannot_change_reviewed_source_or_work(self):
        original=copy.deepcopy(self.grant)
        for key,value in (('archive_sha256','d'*64),('nonce','d'*16),('caller_seconds',7161),
                          ('generations',353),('boot_id','DIFFERENT_BOOT'),('cpu_grading_seconds',321)):
            self.grant=dict(original,**{key:value});self.register()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'approved'):self.validate()

    def test_rehashed_registration_cannot_change_approved_author_seal(self):
        self.grant['source_seal']=dict(root=str(self.source),manifest_sha256='e'*64,members=1);self.register()
        with self.assertRaisesRegex(ValueError,'approval record'):self.validate()

    def test_changed_protected_delta_refuses(self):
        (self.source/'PROTECTED-COMPLETE.patch').write_bytes(b'unreviewed CPU delta\n')
        with self.assertRaisesRegex(ValueError,'hash differs'):self.validate()


if __name__=='__main__':unittest.main()
