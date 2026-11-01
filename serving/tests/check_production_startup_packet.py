"""Check actual preweight guards against authenticated W5 templates and overlays."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_packet_projection_packet as projection_checks
import check_cleaned_inventory_packet as inventory_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical,sha
from test_contract_cleanup import load

PROPOSAL=HERE.parent/'proposals/production-startup'


class ProductionStartupPacket(projection_checks.PacketProjectionPacket):
    evidence={}

    @classmethod
    def setUpClass(cls):
        with patch.object(projection_checks.boot_checks,'PROPOSAL',PROPOSAL):super().setUpClass()

    def test_exact_diff_and_unmodified_release_validators(self):
        record=json.loads((PROPOSAL/'PROPOSAL.json').read_bytes())
        before=json.loads((HERE.parent/'proposals/d2-approval-boot/PROPOSAL.json').read_bytes())
        self.assertEqual(record['protected_files'],before['protected_files'])
        self.assertEqual(record['protected_set_sha256'],before['protected_set_sha256'])
        for name,pin in record['protected_files'].items():self.assertEqual(sha(self.members['review/'+name]),pin)
        for name,row in record['source_changes'].items():
            self.assertEqual(sha((HERE.parent/name).read_bytes()),row['proposed_sha256'])
        self.assertEqual(sha((PROPOSAL/'SOURCE.patch').read_bytes()),record['source_patch_sha256'])
        self.assertEqual(record['source_changes']['legacy/review/worker_entry.py']['baseline_sha256'],
            json.loads((HERE.parent/'proposals/d2-host-binding/PROPOSAL.json').read_bytes())['source_changes']['legacy/review/worker_entry.py']['proposed_sha256'])
        self.evidence.update(protected_helpers_unchanged_from_boot_stage=True,
            exact_source_pins_verified=True,release_validators_unchanged=True,owner_ceilings_changed=False)

    def test_authenticated_templates_pass_startup_checks_for_both_arms_and_ranks(self):
        decisions=[]
        with self.controller() as (controller,_,_,_,root):
            worker=load('cleanup_actual_startup_worker',root/'review/worker_entry.py')
            contract=controller.mtp_contract
            safe=worker._safe_regular
            for arm in ('control','mtp'):
                template=contract.expand_http_template(json.loads((root/f'review/HTTP-BINDING-{arm}.json').read_bytes()))
                manifest=template['worker_template']
                manifest.update(session_sha256='a'*64,spec_sha256='b'*64)
                manifest['preweight']['deadline_monotonic']=200
                selected=contract.contract_for_arm(arm)
                expected={name:selected[name] for name in ('speculative_algorithm','speculative_num_steps',
                    'speculative_eagle_topk','speculative_num_draft_tokens','speculative_dflash_block_size','enable_multi_layer_eagle')}
                expected.update(tp_size=2,pp_size=1,disable_overlap_schedule=True,
                    cuda_graph_backend_decode='full',cuda_graph_backend_prefill='disabled',
                    cuda_graph_bs_decode=[1,2],cuda_graph_max_bs_decode=2)
                cfg=SimpleNamespace(**copy.deepcopy(expected))
                events=[]
                resolver=Mock(side_effect=lambda value:events.append('configuration') or cfg)
                overlays=contract.runtime_overlays(arm)
                paths={row['target']:root/'review'/row['file'] for row in overlays}
                expected_devices=manifest['expected_devices']
                devices=[dict(index=index,bdf=bdf,uuid=expected_devices[bdf]['uuid'],architecture='gfx1201',vram_bytes=24<<30)
                    for index,bdf in enumerate(('0000:c3:00.0','0000:23:00.0'))]
                renders=[dict(bdf=bdf,sysfs_bdf=bdf,**row) for bdf,row in expected_devices.items()]
                physical=Mock(side_effect=lambda *args:(events.append('identity') or devices,{}))
                helper=SimpleNamespace(physical_devices=physical)
                with patch.object(worker,'_native_module',return_value=SimpleNamespace(resolving_view=resolver)), \
                        patch.object(worker,'_safe_regular',side_effect=lambda path:safe(paths.get(str(path),path))), \
                        patch.object(worker,'_source_module',return_value=helper) as sources, \
                        patch.object(worker,'_render_device_bindings',return_value=renders), \
                        patch.object(worker.time,'monotonic',return_value=100), \
                        patch.object(worker,'_atomic_json') as writes,patch.dict(sys.modules,torch=SimpleNamespace()):
                    for rank in (0,1):
                        result=worker._preweight_before_scheduler(manifest,rank,object(),gpu_id=rank,
                            contract=contract,instrumented=False)
                        self.assertEqual(result['rank_binding']['rank'],rank)
                        decisions.append(dict(arm=arm,rank=rank,validated_overlays=len(overlays)))
                    writes.assert_not_called();self.assertEqual(sources.call_count,2)
                    self.assertEqual(events,['configuration','identity','configuration','identity'])
                    for field,value in (('speculative_num_draft_tokens',4),('disable_overlap_schedule',False),
                                        ('cuda_graph_backend_decode','disabled'),('pp_size',True)):
                        original=getattr(cfg,field);setattr(cfg,field,value)
                        sources.reset_mock();writes.reset_mock();physical.reset_mock()
                        with self.subTest(arm=arm,field=field),self.assertRaises(ValueError):
                            worker._preweight_before_scheduler(manifest,0,object(),gpu_id=0,
                                contract=contract,instrumented=False)
                        sources.assert_not_called();writes.assert_not_called();physical.assert_not_called()
                        setattr(cfg,field,original)
        self.evidence['actual_production_preweight_guards']=dict(cases=decisions,negative_flag_cases=8,
            actual_contract_and_overlay_bytes=True,configuration_before_identity=True,
            no_observer_import_or_extra_receipt=True,
            substitutes=['native resolved-config constructor','HIP/device query','render/sysfs observation','clock'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--report',type=Path)
    options,remaining=parser.parse_known_args();inventory_checks.PACKET=options.packet
    with isolated_modules():result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    if options.report:
        report=dict(ProductionStartupPacket.evidence,tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
