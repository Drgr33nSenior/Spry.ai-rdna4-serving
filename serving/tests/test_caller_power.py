"""Exercise caller configuration with all remote execution replaced by a mock."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_contract_cleanup import load


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()


class CallerPower(unittest.TestCase):
    def setUp(self):
        self.check=Mock();self.run=Mock()
        with patch.dict(sys.modules,connection=SimpleNamespace(SSH=['fake-ssh'],check=self.check),
                        local_commands=SimpleNamespace(run=self.run)):
            self.module=load('cleanup_power_observer',HERE.parent/'launcher/power_observe.py')
        temporary=tempfile.TemporaryDirectory(prefix='caller-power-test-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.module.ROOT=self.root
        self.host=json.loads((HERE.parent/'config/host/r9700-pair.json').read_bytes())
        self.binding=dict(target_root='/home/developer/fp8-kv-benchmark.0123456789abcdef',
            protected_root='/home/developer/spry-protected.0123456789abcdef',boot_id='synthetic-boot',
            host_config='config/host/r9700-pair.json')
        (self.root/'config/host').mkdir(parents=True)
        self.write_host()

    def write_host(self):
        raw=canonical(self.host)
        (self.root/self.binding['host_config']).write_bytes(raw)
        self.binding['host_sha256']=hashlib.sha256(raw).hexdigest()
        (self.root/'LAUNCHER-BINDINGS.json').write_bytes(canonical(self.binding))

    def record(self,cap=None):
        return dict(boot_id='synthetic-boot',owner_journal_absent=True,descriptors_closed=True,
            power_microwatts={bdf:(self.host['power_cap_watts'] if cap is None else cap)*1000000
                             for bdf in self.host['devices']},
            campaign_roots_present={self.binding[key]:False for key in ('target_root','protected_root')})

    def fake_run(self,name,argv,seconds,**kwargs):
        (self.root/(name+'.stdout')).write_bytes(canonical(self.record()))
        return dict(exit_code=0,child_reaped=True,owned_group_absent=True)

    def test_baseline_and_tighter_cap_flow_through_actual_main(self):
        for cap in (235,230):
            self.host['power_cap_watts']=cap;self.write_host()
            self.run.side_effect=self.fake_run
            output=io.StringIO()
            with contextlib.redirect_stdout(output):self.module.main(['pre'])
            self.assertEqual(json.loads(output.getvalue())['power_microwatts'],self.record()['power_microwatts'])
            call=self.run.call_args
            self.assertEqual(call.args[1],['fake-ssh','/usr/bin/sudo -n /usr/bin/python3 -I -B -S -'])
            self.assertEqual(call.args[2],15)
            self.assertIn(b'NODES = ',call.kwargs['stdin'])
            ast.parse(call.kwargs['stdin'])
        self.assertEqual(self.run.call_count,2)

    def test_cap_increase_type_alias_and_bad_device_map_refuse_before_remote_work(self):
        original=copy.deepcopy(self.host)
        for mutation in ('increase','boolean','float','duplicate_render','boot_field','rank_boolean'):
            self.host=copy.deepcopy(original)
            if mutation=='increase':self.host['power_cap_watts']=236
            elif mutation=='boolean':self.host['power_cap_watts']=True
            elif mutation=='float':self.host['power_cap_watts']=235.0
            elif mutation=='duplicate_render':
                values=list(self.host['devices'].values());values[0]['render']=values[1]['render']
            elif mutation=='boot_field':self.host['boot_id']='forbidden'
            else:self.host['rank_indices']=[False,1]
            self.write_host()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.module.main(['pre'])
        self.check.assert_not_called();self.run.assert_not_called()

    def test_wrong_pin_link_or_path_refuses(self):
        self.binding['host_sha256']='0'*64
        with self.assertRaises(ValueError):self.module.read_host(self.binding)
        self.binding['host_config']='../outside.json'
        with self.assertRaises(ValueError):self.module.read_host(self.binding)
        self.binding['host_config']='config/host/linked.json'
        (self.root/self.binding['host_config']).symlink_to(self.root/'config/host/r9700-pair.json')
        with self.assertRaises(ValueError):self.module.read_host(self.binding)

    def test_noncanonical_or_duplicate_key_json_refuses(self):
        for raw in (json.dumps(self.host,indent=2).encode(),b'{"schema":1,"schema":1}'):
            self.binding['host_sha256']=hashlib.sha256(raw).hexdigest()
            (self.root/self.binding['host_config']).write_bytes(raw)
            with self.assertRaises(ValueError):self.module.read_host(self.binding)

    def test_observed_wrong_cap_is_retained_and_refused(self):
        self.host['power_cap_watts']=230;self.write_host()
        def wrong(name,*args,**kwargs):
            (self.root/(name+'.stdout')).write_bytes(canonical(self.record(cap=235)))
            return dict(exit_code=0,child_reaped=True,owned_group_absent=True)
        self.run.side_effect=wrong
        with self.assertRaisesRegex(AssertionError,'authenticated host configuration'):self.module.main(['pre'])
        self.assertTrue((self.root/'power-observe-pre.stdout').is_file())

    def test_remote_program_only_observes_power_settings(self):
        code=self.module.observation_code(self.binding,self.host,'post')
        tree=ast.parse(code)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(node.func,ast.Attribute) and
            node.func.attr in ('write_text','write_bytes','system','popen') for node in calls))
        self.assertIn('power1_cap',code)
        self.assertNotIn('235000000',code)
        self.assertEqual(self.run.call_count,0)


if __name__=='__main__':unittest.main()
