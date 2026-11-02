"""Configured profile, menu and approval projections without private assets or native imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spry_common.canon import canonical_line,digest
spec=importlib.util.spec_from_file_location('spry_configuration_contract',ROOT/'spry_serving/contract.py')
contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract)


class ConfigurationProjection(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config/runs/w5-v2/run.json').read_bytes())
        self.menu=json.loads((ROOT/'config/runs/w5-v2/menu.json').read_bytes())
        self.host=json.loads((ROOT/self.config['host']['file']).read_bytes())

    def test_projection_binds_the_actual_run_host_menu_and_time(self):
        value=contract.configuration_projection(self.config,self.host,self.menu,None)
        self.assertEqual(value['run_config_sha256'],digest(canonical_line(self.config)))
        self.assertEqual(value['host_sha256'],digest(canonical_line(self.host)))
        self.assertEqual(value['menu_sha256'],digest(canonical_line(self.menu)))
        self.assertEqual((value['client_seconds'],value['worker_seconds']),(2040,2280))
        self.assertEqual(value['source_ordinals'],list(range(176)))

    def test_supported_scalar_profiles_preserve_menu_and_pool_feasibility(self):
        for name,value in (('context_length',65536),('max_total_tokens',131072),('chunked_prefill_size',4096),
                           ('mem_fraction_static',0.91),('max_mamba_cache_size',18)):
            profile=copy.deepcopy(self.config['profile']);profile[name]=value
            with self.subTest(name=name):contract.validate_profile(profile);contract.validate_menu(self.menu,profile)

    def test_boolean_numeric_aliases_overlap_and_profile_ceiling_increases_refuse(self):
        for name,value in (('context_length',True),('max_running_requests',3),('max_mamba_cache_size',20),
                           ('mem_fraction_static',1.0),('mem_fraction_static',float('nan')),
                           ('chunked_prefill_size',3000),('disable_overlap_schedule',False)):
            profile=copy.deepcopy(self.config['profile']);profile[name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):contract.validate_profile(profile)

    def test_context_and_concurrent_pool_overflows_refuse(self):
        for name,value in (('context_length',32768),('max_total_tokens',128000)):
            profile=copy.deepcopy(self.config['profile']);profile[name]=value
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'overflow'):
                contract.validate_menu(self.menu,profile)

    def test_duplicate_missing_or_out_of_range_group_members_refuse(self):
        for mutate in (lambda m:m['groups'].pop(),lambda m:m['groups'][0]['ordinals'].append(0),
                       lambda m:m['groups'][0]['ordinals'].__setitem__(0,176)):
            menu=copy.deepcopy(self.menu);mutate(menu)
            with self.assertRaises(ValueError):contract.validate_menu(menu,self.config['profile'])

    def test_cap_work_is_explicit_and_unknown_policies_refuse(self):
        menu=copy.deepcopy(self.menu)
        for row in menu['rows']:row['stop_policy']='cap'
        contract.validate_menu(menu,self.config['profile'])
        menu['rows'][0]['stop_policy']='unknown'
        with self.assertRaisesRegex(ValueError,'policy'):contract.validate_menu(menu,self.config['profile'])

    def test_host_pin_drift_and_power_ceiling_increases_refuse(self):
        for watts in (230,236,True):
            host=copy.deepcopy(self.host);host['power_cap_watts']=watts
            with self.subTest(watts=watts),self.assertRaises(ValueError):
                contract.configuration_projection(self.config,host,self.menu,None)
        host=copy.deepcopy(self.host);host['power_cap_watts']=230
        run=copy.deepcopy(self.config);run['host']['sha256']=digest(canonical_line(host))
        self.assertEqual(contract.configuration_projection(run,host,self.menu,None)['host_sha256'],run['host']['sha256'])

    def test_duplicate_and_nonfinite_json_refuse(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):contract.strict_json(raw)

    def corpus(self):
        body=dict(messages=[dict(role='user',content='frozen prompt')],rid='original-rid',cache_salt='original-salt',
                  max_completion_tokens=16)
        expected=dict(case='original',input_tokens=2,input_ids=[10,11],internal_reserve_tokens=16,
            context_length=128,messages_sha256=digest(canonical_line(body['messages'])),
            input_ids_sha256=digest(canonical_line([10,11])),completion_policy='natural-stop')
        assets={'FINAL-REQUEST-000-original.json':canonical_line(body),
                'FINAL-REQUEST-000-original.expected.json':canonical_line(expected)}
        rows=[]
        for n in range(2):
            rows.append(dict(ordinal=n,source_ordinal=0,case='alias-'+str(n),path='FINAL-REQUEST-000-original.json',
                expected='FINAL-REQUEST-000-original.expected.json',sha256=digest(canonical_line(body)),
                expected_sha256=digest(canonical_line(expected)),input_tokens=2,internal_reserve_tokens=16,
                maximum_output_tokens=8,stop_policy='natural' if n==0 else 'cap',context_envelope_tokens=128))
        return assets,dict(rows=rows)

    def test_compact_aliases_preserve_prompt_ids_and_use_unique_cold_identities(self):
        assets,menu=self.corpus();original=copy.deepcopy(assets)
        plans,rows=contract.resolve_configured_requests(assets,self.config,menu)
        self.assertEqual(assets,original)
        self.assertTrue(all('body' not in row and 'input_ids' not in row for row in rows))
        mapped=contract._ConfiguredAssets(rows,assets)
        bodies=[json.loads(mapped[row['path']]) for row in plans]
        expected=[json.loads(mapped[row['expected']]) for row in plans]
        self.assertEqual([value['messages'] for value in bodies],[[dict(role='user',content='frozen prompt')]]*2)
        self.assertEqual([value['input_ids'] for value in expected],[[10,11]]*2)
        self.assertEqual(len({value['rid'] for value in bodies}),2)
        self.assertEqual(len({value['cache_salt'] for value in bodies}),2)
        self.assertEqual(expected[1]['completion_policy'],'ungraded-cap-work')
        self.assertEqual(mapped._cached.cache_info().maxsize,16)

    def test_compact_source_tampering_and_payload_updates_refuse(self):
        assets,menu=self.corpus();_,rows=contract.resolve_configured_requests(assets,self.config,menu)
        for kind in ('source','messages','ids','resolved-pin','duplicate'):
            changed=copy.deepcopy(rows);base=copy.deepcopy(assets)
            if kind=='source':base[changed[0]['source']['path']]=b'{}\n'
            if kind=='messages':changed[0]['body_updates']['messages']=[]
            if kind=='ids':changed[0]['expected_updates']['input_ids']=[99]
            if kind=='resolved-pin':changed[0]['body_updates']['rid']='unbound-rid'
            if kind=='duplicate':changed[1]['path']=changed[0]['path']
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                mapped=contract._ConfiguredAssets(changed,base)
                for name in mapped:mapped[name]

    def test_compact_binding_schema_is_closed_and_rejects_old_full_payload_schema(self):
        for schema in (1,True,3):
            raw=zlib.compress(canonical_line(dict(schema=schema,mode='normal-stop-fp8-mxfp4-workload-v1',rows=[{}])))
            with self.subTest(schema=schema),self.assertRaises(ValueError):contract._configured_client_rows(raw,1)

    def test_production_log_interval_changes_without_selecting_a_host_path(self):
        for mode,interval in (('production',100),('on',1)):
            with self.subTest(mode=mode),patch.object(contract,'raw_configuration',return_value={}),\
                    patch.object(contract,'settings',return_value=dict(instrumentation=mode)):
                self.assertEqual(contract.expected_decode_log_interval('control'),interval)
                self.assertEqual(contract.expected_decode_log_interval('mtp'),interval)
                self.assertFalse(contract.host_path_cleanup_enabled('control'))
        with self.assertRaises(ValueError):contract.expected_decode_log_interval('unknown')


if __name__=='__main__':unittest.main()
