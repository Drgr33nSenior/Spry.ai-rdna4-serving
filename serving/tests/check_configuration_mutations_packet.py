"""Exercise configured packages through the retained real CPU controller boundary."""
import argparse
import ast
import __future__
from contextlib import contextmanager
import copy
import difflib
from functools import partial
import gzip
import hashlib
import importlib.util
import io
import json
import random
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
import build
import propose_packet
import prepare_gpu_run
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest,parse_json

PROPOSAL=HERE.parent/'proposals/configuration-mutations'
TEST_SOURCE_SHA256=digest(Path(__file__).read_bytes())


def repeat_menu(menu):
    result=copy.deepcopy(menu);count=len(menu['rows'])
    for original in menu['rows']:
        row=dict(original,source_ordinal=original['ordinal'],ordinal=original['ordinal']+count,
                 case=original['case']+'-copy2',group=original['group']+'-copy2')
        result['rows'].append(row)
    result['groups'] += [dict(group,id=group['id']+'-copy2',ordinals=[n+count for n in group['ordinals']]) for group in menu['groups']]
    return result


class ConfigurationPacket(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.metadata=parse_json((HERE.parent/'SOURCE.json').read_bytes())
        cls.authenticated=build.authenticated_packet(PACKET.resolve(),cls.metadata)
        cls.temp=tempfile.TemporaryDirectory(prefix='spry-configuration-');cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name).resolve();fixture=cls.root/'fixture';fixture.mkdir()
        for name,raw in cls.authenticated.items():
            path=fixture/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        cls.binding=load('configured_native_boundary_fixture',fixture/'test_controller_binding.py')
        original=cls.authenticated['w5_evidence.py'].decode();adapted=original.replace('controller.SPEC_SHA256','controller.mapping_sha256(spec)')
        replacements={
            'from pathlib import Path\n':'from pathlib import Path\nfrom dataclasses import asdict\n',
            'controller.http_expected_binding(root, spec, owner_arm=arm, session_sha256=session)':
                'controller.http_expected_binding(root, spec, owner_arm=arm, session_sha256=session, spec_sha256=controller.mapping_sha256(spec))',
            "        spec = json.loads((root / 'review/BENCHMARK-SPEC2.json').read_bytes())\n":
                "        spec = json.loads((root / 'review/BENCHMARK-SPEC2.json').read_bytes())\n        artifact_plan = artifacts.HTTPArtifactLimits.from_limits(spec['http_serving']['artifact_limits'])\n",
            'artifacts.pack_http_artifacts(output, art_binding)':'artifacts.pack_http_artifacts(output, art_binding, limits=artifact_plan)',
            'artifacts.stage_artifacts(output, art_binding)':'artifacts.stage_artifacts(output, art_binding, limits=artifact_plan)',
            'artifacts.export_bundle(export, art_binding, sink)':'artifacts.export_bundle(export, art_binding, sink, limits=artifact_plan)',
            'artifacts.receive_bundle(stdout, stderr, destination, art_binding)':'artifacts.receive_bundle(stdout, stderr, destination, art_binding, limits=artifact_plan)',
            'joins=joins, supervision=supervision)':'joins=joins, supervision=supervision, limits=asdict(artifact_plan))',
            "            refused['rows'][-1].update(status='OUTPUT_CAP_REACHED', finish_reason='length')":
                "            refused['rows'][-1].update(status='OUTPUT_CAP_REACHED', finish_reason='length', metric_eligible=False, reported_completion_tokens=0, response_id_count=0)",
            "            stderr = filler(int(W1_REFERENCE['stderr']['bytes'] * scale * 2))":
                "            stderr = filler(diagnostic_stderr_envelope(controller.mtp_contract, contract.requests()))",
            "            stdout = filler(int(W1_REFERENCE['stdout']['non_bundle_bytes'] * scale * 2)) + resource + line":
                "            stdout = filler(4 << 20) + resource + line",
            "            receipts = sorted(client.rglob('receipts/*.json'))":
                "            receipts = sorted(client.rglob('receipts/*'))",
            "                for name, row in json.loads(path.read_bytes())['fields'].items():":
                "                for name, row in client_module._kv_receipts.decode(path.read_bytes())[0]['fields'].items():",
            "            choice['response_token_ids'] = [STRESS_TOKEN_ID] * (cap // 2) + [THINK_END] + [STRESS_TOKEN_ID] * (cap - cap // 2 - 2) + [EOS]":
                "            choice['response_token_ids'] = stress_ids(cap, ordinal)",
            "            choice['message']['content'] = 'x' * (text_bytes * cap)":
                "            choice['message']['content'] = stress_text(text_bytes * cap, ordinal)",
        }
        for before,after in replacements.items():
            if adapted.count(before)!=1:raise AssertionError('retained producer interface changed: '+before)
            adapted=adapted.replace(before,after)
        (fixture/'w5_configuration_evidence.py').write_text(adapted)
        cls.postrun=load('configured_full_evidence_producer',fixture/'w5_configuration_evidence.py')
        cls.postrun.diagnostic_stderr_envelope=cls.diagnostic_stderr_envelope
        cls.postrun.stress_ids=lambda count,ordinal:random.Random(ordinal).choices(range(100000,200000),k=count//2)+[248069]+random.Random(ordinal+352).choices(range(100000,200000),k=count-count//2-2)+[248046]
        cls.postrun.stress_text=lambda count,ordinal:''.join(random.Random(ordinal).choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_',k=count))
        cls.evidence['producer_adapter_sha256']=digest(adapted.encode())
        cls.evidence['stdout_fixture']=dict(non_bundle_bytes=4<<20,retained_B_maximum_non_bundle_bytes=30341,
            scope='Fixed startup envelope; variable complete artifact line is measured separately')
        cls.evidence['producer_adapter_patch']=''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile='retained/w5_evidence.py',tofile='fixture/w5_configuration_evidence.py'))
        cls.run_config=parse_json((HERE.parent/'config/runs/w5-v2/run.json').read_bytes())
        cls.menu=parse_json((HERE.parent/'config/runs/w5-v2/menu.json').read_bytes())
        cls.host=parse_json((HERE.parent/cls.run_config['host']['file']).read_bytes())

    @staticmethod
    def diagnostic_stderr_envelope(contract,requests):
        # The pinned metrics reporter returns before logging unless decode
        # forward count is divisible by this interval. Each productive decode
        # consumes at least one output token. Allow 512 bytes per line (the
        # retained maximum was 238), both ranks, plus 4MiB startup/prefill logs.
        # This is a disclosed fixture envelope, not a runtime logging guarantee.
        interval=contract.expected_decode_log_interval('control')
        tokens=sum(row['body']['max_completion_tokens'] for row in requests)
        return (4<<20)+2*((tokens+interval-1)//interval)*512

    def package(self,name,*,run=None,menu=None,host=None):
        source=self.root/(name+'-config');shutil.copytree(HERE.parent/'config',source/'config')
        run=copy.deepcopy(self.run_config if run is None else run);menu=copy.deepcopy(self.menu if menu is None else menu)
        host=copy.deepcopy(self.host if host is None else host)
        run['host']['sha256']=digest(canonical(host));run['menu']['sha256']=digest(canonical(menu))
        run_dir=source/'config/runs/w5-v2'
        for path,value in ((run_dir/'run.json',run),(run_dir/'menu.json',menu),(source/run['host']['file'],host)):
            path.write_bytes(canonical(value))
        output=self.root/(name+'-review')
        report=propose_packet.build(PACKET.resolve(),output,packet_id='e201110200000010',proposal=PROPOSAL.resolve(),run_dir=run_dir)
        package=output/'package';(package/'MANIFEST.json').write_bytes((package/'MANIFEST-PROPOSAL.json').read_bytes())
        binding=parse_json((output/'caller/LAUNCHER-BINDINGS-TEMPLATE.json').read_bytes())
        pins={key:binding[key] for key in ('run_config_sha256','host_sha256','protected_set_sha256')}
        return package,pins,report

    @contextmanager
    def controller(self,package):
        with isolated_modules(),self.binding.controller_on_mirror(package) as values:
            controller,path,pin,manifest,root=values
            for source in (package/'config').rglob('*'):
                if source.is_file():
                    target=root/source.relative_to(package);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
            (root/'PROTECTED-SHA256.json').write_bytes((package/'PROTECTED-SHA256.json').read_bytes())
            yield values

    def verify(self,name,**kwargs):
        package,pins,report=self.package(name,**kwargs)
        with self.controller(package) as (controller,path,pin,manifest,root):
            value=parse_json(path.read_bytes());verified=controller.verify_configured_spec(pins,value,deadline=time.monotonic()+105)
            self.assertEqual(verified['spec_sha256'],manifest['spec_sha256'])
            actual_contract,_=controller.http_integration._private_reader()
            self.assertEqual(len(actual_contract.requests()),len(verified['menu']['rows']))
            first=actual_contract.requests();rid=first[0]['body']['rid']
            first[0]['body']['rid']='caller mutation'
            self.assertEqual(actual_contract.requests()[0]['body']['rid'],rid)
            self.assertEqual(actual_contract._configured_requests.cache_info().maxsize,1)
            for arm in ('control','mtp'):
                binding=controller.mtp_contract.expand_http_template(parse_json((root/('review/HTTP-BINDING-'+arm+'.json')).read_bytes()))
                cli=binding['binding_template']['canonical_server_cli']
                selected=controller.mtp_contract.configured_profile()
                for key,item in selected.items():
                    flag='--'+key.replace('_','-')
                    self.assertEqual(cli.count(flag),1)
                    if type(item) is not bool:self.assertEqual(cli[cli.index(flag)+1],str(item))
                self.assertEqual(cli[cli.index('--decode-log-interval')+1],str(controller.mtp_contract.expected_decode_log_interval(arm)))
            validate=partial(controller.validate_manifest,configuration_pins=dict(pins,boot_id='ff396f8f-e5c3-4ae5-86c1-b986bc483185'))
            with patch.object(controller,'validate_manifest',validate):error,frames=self.binding.run_prelaunch(controller,path,pin,manifest,root)
            runtime_present=Path(value['runtime_root']).exists()
            if runtime_present:
                self.assertIsNone(error,None if error is None else (type(error).__name__,str(error)))
            else:
                self.assertIsInstance(error,FileNotFoundError,(type(error).__name__,str(error)))
                self.assertTrue(any('proof, _ = adapter.readout.read_json(runtime / proof_name' in (frame.line or '') for frame in frames),str(error))
            row=dict(rows=len(verified['menu']['rows']),spec_sha256=verified['spec_sha256'],settings_sha256=value['package_files']['review/FINAL-SETTINGS.json'],
                total_seconds=verified['session_timing'].total_seconds,context=verified['spec']['context_length'],external_build_proof_required=True,
                protected_set_sha256=pins['protected_set_sha256'],native_prelaunch_complete=runtime_present,native_execution=False)
        self.evidence[name]=row
        return package,pins,report

    def test_baseline_and_scalar_mutations_reach_the_same_real_prelaunch_boundary(self):
        self.verify('baseline')
        for field,value in (('context_length',65536),('max_total_tokens',131072),('chunked_prefill_size',4096),
                             ('mem_fraction_static',0.91),('max_mamba_cache_size',18)):
            run=copy.deepcopy(self.run_config);run['profile'][field]=value
            with self.subTest(field=field):self.verify(field,run=run)
        run=copy.deepcopy(self.run_config);run['limits']['client_seconds']-=10;self.verify('shorter-time',run=run)
        host=copy.deepcopy(self.host);host['power_cap_watts']=230;self.verify('power-230',host=host)
        run=copy.deepcopy(self.run_config);run['instrumentation']='on';self.verify('instrumented',run=run)

    def test_all_cap_and_doubled_menus_reach_the_real_prelaunch_boundary(self):
        menu=copy.deepcopy(self.menu)
        for row in menu['rows']:row['stop_policy']='cap'
        self.verify('all-cap',menu=menu);self.verify('doubled',menu=repeat_menu(self.menu))

    def test_owner_ceiling_invalid_policy_and_unaccounted_groups_refuse_before_output(self):
        for kind in ('ceiling','overlap','policy','group'):
            run=copy.deepcopy(self.run_config);menu=copy.deepcopy(self.menu)
            if kind=='ceiling':run['limits']['client_seconds']+=1
            if kind=='overlap':run['profile']['disable_overlap_schedule']=False
            if kind=='policy':menu['rows'][0]['stop_policy']='unknown'
            if kind=='group':menu['groups'].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.package('invalid-'+kind,run=run,menu=menu)
            self.assertFalse((self.root/('invalid-'+kind+'-review')).exists())

    def test_pinned_runtime_decode_logging_obeys_selected_interval(self):
        raw=METRICS_REPORTER.read_bytes()
        self.assertEqual(digest(raw),'2e943a26a04170300f7055b48b6f7d205625934354db1ffb9798926e5f18dd50')
        method=next(node for node in ast.walk(ast.parse(raw)) if isinstance(node,ast.FunctionDef) and node.name=='report_decode_stats')
        class WindowReached(Exception):pass
        def periodic():raise WindowReached()
        namespace=dict(time=SimpleNamespace(perf_counter=periodic))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(METRICS_REPORTER),'exec',
                     flags=__future__.annotations.compiler_flag),namespace)
        batch=SimpleNamespace();observed={}
        for interval in (1,100):
            reporter=SimpleNamespace(current_scheduler_metrics_enabled=False,is_stats_logging_rank=True,
                                     decode_log_interval=interval)
            windows=0
            for step in range(1,251):
                reporter.forward_ct_decode=step
                try:namespace['report_decode_stats'](reporter,True,batch,1)
                except WindowReached:windows+=1
            self.assertEqual(windows,250//interval)
            observed[str(interval)]=windows
        self.evidence['diagnostic_logger']=dict(source_sha256=digest(raw),calls=250,periodic_windows=observed,
            stderr_fixture=dict(per_rank_line_bytes=512,ranks=2,startup_prefill_bytes=4<<20,
                scope='CPU capacity envelope; no universal runtime log-size guarantee'))

    def evidence_case(self,name,shape,*,menu=None,run=None,host=None):
        package,pins,_=self.package(name,menu=menu,run=run,host=host);resolved=[];quality=[]
        source_plan=parse_json((package/'config/runs/w5-v2/menu.json').read_bytes())['rows']
        @contextmanager
        def mirror(package):
            with self.controller(package) as (controller,path,_,manifest,root):
                value=parse_json(path.read_bytes());checked=controller.verify_configured_spec(pins,value,deadline=time.monotonic()+105)
                resolved[:]=checked['menu']['rows']
                yield controller,value,root
                controller.verify_unchanged(checked)
        original_responder=self.postrun.responder;original_real=self.postrun.real_responses
        def responder(contract,shape,arm):
            requests=contract.requests();config=contract.mtp_contract.settings()['configuration']
            sources=config['source_ordinals']
            facade=SimpleNamespace(requests=lambda:[dict(row,_kv_ordinal=sources[n]) for n,row in enumerate(requests)],
                                   expected_case=contract.expected_case)
            original_assets=contract.mtp_contract._decoded_assets(contract.mtp_contract.logical_asset('FINAL-REQUESTS.zlib',contract.mtp_contract.REQUEST_INHERITED['sha256'],2<<20))
            def real(owner):
                w1,b1=original_real(owner);b1=dict(b1)
                for n,row in enumerate(requests):
                    source_body=parse_json(original_assets[source_plan[n]['path']]);source_rid=source_body['rid']
                    if source_rid in b1:
                        value=copy.deepcopy(b1[source_rid]);value['id']=value['choices'][0]['meta_info']['id']=row['body']['rid']
                        b1[row['body']['rid']]=value
                return w1,b1
            with patch.object(self.postrun,'real_responses',side_effect=real):return original_responder(facade,shape,arm)
        def quality_size(wires):
            q=load('quality_contract',HERE.parent/'quality/quality_contract.py')
            with patch.dict(sys.modules,quality_contract=q):
                producer=load('configured_quality_input_producer',HERE.parent/'quality/final_quality_inputs.py')
                private=parse_json(self.authenticated['quality-template-w5/producer/inputs/QUALITY-MENU.json'])['rows']
                planned=producer.project_menu(private,resolved,source_plan=source_plan)
                value=producer.build_input(wires,menu=planned);q.validate_input(value,planned)
                template=self.root/(name+'-quality-template')
                assembled=prepare_gpu_run.quality_template(self.authenticated,template,resolved,source_plan=source_plan)
                materializer=prepare_gpu_run.caller.checked_module(template,'prepare_packet',digest((template/'prepare_packet.py').read_bytes()))
                cpu=self.root/(name+'-cpu-fixture')
                proof=materializer.materialize(q.encoded(value),cpu,projection_inventory_sha256=assembled['projection_inventory_sha256'],fixture=True)
                packet=parse_json((cpu/'source/PACKAGE.json').read_bytes())
                self.assertEqual(proof['rows'],len(planned));self.assertEqual(packet['final_quality']['rows'],len(planned))
                self.assertEqual(packet['worker_seconds'],210);self.assertEqual(packet['total_seconds'],320)
                self.assertFalse((cpu/'ACTUAL-INPUT-BINDING.json').exists())
            quality.append(dict(rows=len(value['rows']),natural_rows=sum(row['scope']=='natural-oracle' for row in value['rows']),
                                cap_rows=sum(row['scope']=='ungraded-cap-work' for row in value['rows']),native_decode=False,
                                actual_cpu_packet_materialized=True,fixture=True,native_execution_authorized=False))
            return len(q.encoded(value))
        with isolated_modules(),patch.object(self.postrun,'mirror',mirror),patch.object(self.postrun,'responder',side_effect=responder),patch.object(self.postrun,'quality_input_bytes',side_effect=quality_size):
            measured,references,refusals=self.postrun.run(package,shape)
        margins=self.postrun.table(measured)
        self.evidence[name]=dict(shape=shape,measurements=measured,margins=margins,references=references,refusals=refusals,
                                 quality=quality,validators_substituted=False,native_execution=False)
        self.assertTrue(all(row['passes'] and row['margin']>=2 for row in margins),
                        [row for row in margins if not row['passes'] or row['margin']<2])
        self.assertEqual(len(quality),1)
        return margins

    def test_full_expected_baseline_export_receive_readout_and_quality_projection(self):
        self.evidence_case('full-expected','expected')

    def test_full_all_cap_and_doubled_menu_capacity(self):
        menu=copy.deepcopy(self.menu)
        for row in menu['rows']:row['stop_policy']='cap'
        self.evidence_case('full-all-cap','worst',menu=menu)
        self.evidence_case('full-doubled','envelope',menu=repeat_menu(self.menu))

    def test_full_changed_profile_export_receive_readout_and_quality_projection(self):
        run=copy.deepcopy(self.run_config)
        run['profile'].update(context_length=65536,max_total_tokens=131072,chunked_prefill_size=4096,
                              mem_fraction_static=0.91,max_mamba_cache_size=18)
        run['limits']['client_seconds']=2030
        host=copy.deepcopy(self.host);host['power_cap_watts']=230
        self.evidence_case('full-composed-profile','expected',run=run,host=host)

    def test_new_source_review_is_deterministic(self):
        left,_,report_a=self.package('deterministic-a')
        right,_,report_b=self.package('deterministic-b')
        files=lambda root:{p.relative_to(root).as_posix():digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
        self.assertEqual(files(left),files(right))
        self.assertEqual(report_a,report_b)
        self.evidence['determinism']=dict(package_members=len(files(left)),identical_file_bytes=True,
                                           content_sha256=report_a['content_sha256'])

    def test_actual_cpu_packet_materializer_accepts_full_cap_and_doubled_menus(self):
        from test_quality_policy import arm_results
        cap=copy.deepcopy(self.menu)
        for row in cap['rows']:row['stop_policy']='cap'
        for name,menu in (('quality-baseline',self.menu),('quality-all-cap',cap),('quality-doubled',repeat_menu(self.menu))):
            with self.subTest(name=name):
                package,pins,_=self.package(name,menu=menu)
                with self.controller(package) as (controller,path,_,_,root):
                    checked=controller.verify_configured_spec(pins,parse_json(path.read_bytes()),deadline=time.monotonic()+105)
                    template=self.root/(name+'-template')
                    assembled=prepare_gpu_run.quality_template(self.authenticated,template,checked['menu']['rows'],source_plan=menu['rows'])
                    q=prepare_gpu_run.caller.checked_module(template/'source','quality_contract',digest((template/'source/quality_contract.py').read_bytes()))
                    producer=prepare_gpu_run.caller.checked_module(template/'producer','final_quality_inputs',digest((template/'producer/final_quality_inputs.py').read_bytes()))
                    expected=q.strict((template/'source/EXPECTED-QUALITY.json').read_bytes())
                    value=producer.build_input(arm_results(expected),menu=expected)
                    materializer=prepare_gpu_run.caller.checked_module(template,'prepare_packet',digest((template/'prepare_packet.py').read_bytes()))
                    cpu=self.root/(name+'-cpu')
                    proof=materializer.materialize(q.encoded(value),cpu,projection_inventory_sha256=assembled['projection_inventory_sha256'],fixture=True)
                    contract=parse_json((cpu/'source/PACKAGE.json').read_bytes())
                    self.assertEqual(contract['final_quality']['rows'],len(expected));self.assertEqual(proof['rows'],len(expected))
                    self.assertEqual(contract['cpu_properties'],['CPUQuota=200%','MemoryMax=4294967296','MemorySwapMax=0','TasksMax=128'])
                    self.assertFalse(proof['execution_authorized']);self.assertFalse((cpu/'ACTUAL-INPUT-BINDING.json').exists())
                    self.evidence[name]=dict(rows=len(expected),actual_materialization=True,synthetic_fixture=True,
                        native_decode=False,execution_authorized=False,projection_inventory_sha256=assembled['projection_inventory_sha256'],
                        sources={item:digest((HERE.parent/item).read_bytes()) for item in ('prepare_gpu_run.py','quality/prepare_packet.py','quality/common.py','quality/bind_future.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    options,remaining=parser.parse_known_args();PACKET=options.packet;METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        with options.report.open('xb') as out:out.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=TEST_SOURCE_SHA256,cases=ConfigurationPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
