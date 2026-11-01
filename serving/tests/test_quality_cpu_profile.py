"""Exercise actual W5 CPU deadline consumers without native imports or execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

QUALITY=Path(__file__).resolve().parents[1]/'quality'


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,QUALITY/filename)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


q=load('quality_contract','quality_contract.py')
entry=load('cpu_profile_tokenizer_entry','tokenizer_entry.py')
stages=load('cpu_profile_stages','quality_stages.py')


class CPUProfile(unittest.TestCase):
    def test_whole_work_deadline_reaches_namespace_guard_without_native_imports(self):
        before=set(sys.modules)
        with patch.object(entry.time,'monotonic',return_value=1000):
            for deadline in (1076,1210):
                with self.subTest(deadline=deadline),self.assertRaisesRegex(ValueError,'owned CPU tokenizer namespace'):
                    entry.run(deadline,'a'*64,'b'*64)
        self.assertFalse({'torch','triton','sglang'} & (set(sys.modules)-before))

    def test_expired_nonfinite_and_excess_work_deadlines_refuse_before_namespace(self):
        with patch.object(entry.time,'monotonic',return_value=1000):
            for deadline in (1000,1210.01,float('inf'),float('nan'),True,'1210'):
                with self.subTest(deadline=deadline),self.assertRaisesRegex(ValueError,'shared210second'):
                    entry.run(deadline,'a'*64,'b'*64)

    def test_actual_stage_producer_funds_one_180_second_child_in_210_seconds(self):
        common=SimpleNamespace(bootstrap_bwrap=Mock(return_value=['CPU_FIXTURE_NAMESPACE']))
        host=SimpleNamespace(authenticated_common=Mock(return_value=(common,{})),
            own_scope=Mock(return_value=dict(host_cgroup_path='/cpu-fixture')),BOOT='CPU_FIXTURE_BOOT',
            held=Mock(return_value=b'CPU fixture source'),sha=lambda raw:hashlib.sha256(raw).hexdigest(),
            put=Mock(return_value='a'*64),run_finite=Mock(side_effect=RuntimeError('stop before external execution')))
        plan=dict(source_root='/cpu-fixture/source',run_root='/cpu-fixture/run',package_sha256='b'*64,
                  unit='CPU_FIXTURE_UNIT',sources={'QUALITY-CPU-INPUT.json':'c'*64})
        with patch.object(stages.time,'monotonic',return_value=1000):
            with self.assertRaisesRegex(RuntimeError,'stop before external execution'):stages.run(plan,host)
        self.assertEqual(host.run_finite.call_args.args[3],180)
        self.assertEqual(host.run_finite.call_args.kwargs['term'],1)
        self.assertEqual(host.run_finite.call_args.kwargs['reap'],1)
        self.assertEqual(common.bootstrap_bwrap.call_args.args[1],1210)
        self.assertEqual(host.put.call_args_list[0].args[1]['work_deadline_monotonic'],1210)

    def test_stage_consumer_refuses_the_old_45_second_receipt(self):
        record=dict(argv=['CPU_FIXTURE_NAMESPACE'],seconds=180,term_seconds=1,reap_seconds=1,
            complete=True,observed_processes=[dict(pid=123,start_ticks=456)],signals=[],failures=[],pid=123,pgid=123,
            exit_code=0,child_reaped=True,owned_group_absent=True,elapsed_seconds=10)
        row=dict(label='quality-decode',started_monotonic=10,finished_monotonic=20,supervision=record)
        stages.check_stage(row,argv=record['argv'],seconds=180,deadline=210)
        record['seconds']=45
        with self.assertRaisesRegex(ValueError,'source-bound finite child'):stages.check_stage(row,argv=record['argv'],seconds=180,deadline=210)

    def test_readout_uses_the_fixed_320_second_transport_envelope(self):
        host=SimpleNamespace(need=q.need,strict_json=json.loads)
        with patch.dict(sys.modules,host_cpu=host):readout=load('cpu_profile_readout','cpu_readout.py')
        prepared=dict(argv=['CPU_FIXTURE_SSH'],recipe_sha256='a'*64)
        record=dict(argv=prepared['argv'],cwd=str(readout.ROOT),stdin_sha256=prepared['recipe_sha256'],
            exit_code=0,child_reaped=True,owned_group_absent=True,work_seconds=310,term_seconds=5,reap_seconds=5,
            output_limit_bytes=16<<20,pid=123,pgid=123,elapsed_seconds=185)
        record.update(stdout=dict(sha256='b'*64,bytes=0),stderr=dict(sha256='c'*64,bytes=0))
        def held(path,*args,**kwargs):
            if path.name=='cpu-import-once.json':return json.dumps(record).encode()
            raise RuntimeError('past transport deadline check; stop before artifact reads')
        host.held=held
        for elapsed in (185,320):
            record['elapsed_seconds']=elapsed
            with self.subTest(elapsed=elapsed),self.assertRaisesRegex(RuntimeError,'past transport deadline check'):
                readout.transport(Path('/cpu-fixture'),{},b'',prepared,None)
        for elapsed in (0,320.01,float('inf'),float('nan')):
            record['elapsed_seconds']=elapsed
            with self.subTest(elapsed=elapsed),self.assertRaisesRegex(ValueError,'local320 transport envelope'):
                readout.transport(Path('/cpu-fixture'),{},b'',prepared,None)


if __name__=='__main__':unittest.main()
