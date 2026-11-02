"""Descriptive measurement arithmetic and strict complete-work comparison."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
import compare_measurement as comparison
import test_measurement_preparation as preparation_tests


def groups():
    values=[]
    for arm in comparison.ARMS:
        for concurrency in (1,2):
            for number,wall in enumerate((1,4)):
                row=dict(completion_tokens=100,wall_seconds=wall,per_request_latency_seconds=[wall/2,wall/2])
                values.append(dict(arm=arm,logical_arm=comparison.ARMS[arm],concurrency=concurrency,
                    group_id='timing-c%d-%d'%(concurrency,number),timing_eligible=True,
                    baseline=copy.deepcopy(row),cleanup=copy.deepcopy(row)))
    return values


def response_fixture():
    body=dict(choices=[dict(prompt_token_ids=[10,11],response_token_ids=[20,21],
        message=dict(content='complete answer'),finish_reason='stop',
        meta_info=dict(cached_tokens=1,cached_tokens_details={'prefix':1},num_retractions=0))])
    raw=comparison.canonical(body);pin=comparison.digest(raw)
    quality=dict(case='timing-c1-1-a',request_body_sha256='a'*64,response_raw_sha256=pin,
        output_token_ids=[20,21],wire_text='complete answer',finish_reason='stop')
    row=dict(ordinal=0,case=quality['case'],body_sha256=quality['request_body_sha256'],
        wire=dict(raw_body_path='responses/0-timing-c1-1-a.raw',response_sha256=pin),
        contract=dict(prompt_token_count=2,response_id_count=2,reported_completion_tokens=2,
            spec_tokens_details={'spec_verify_ct':2},sampling_parameters={'temperature':0}))
    side=dict(rows={('bf16',0):quality},summaries={'bf16':dict(tree=Path('/retained/paired'))})
    return body,raw,side,row


class MeasurementComparison(unittest.TestCase):
    def final_pair(self,root):
        fixture=preparation_tests.MeasurementPreparation();fixture.setUp()
        before=copy.deepcopy(fixture.after)
        for arm in comparison.ARMS.values():
            name='HTTP-BINDING-'+arm+'.json';value=comparison.strict(before[name])
            value['binding_template']['sources']['mxfp4_scheme']['module']='mxfp4_component.isolated_scheme'
            before[name]=comparison.canonical(value)
        for name in ('kernels.py','GEMM-tail.py','ops.py','p1_fp8_kernels.py',
                     'd3b_head_fp8_kernels.py','d3b_head_fp8_ops.py','d3b_head_launches.py'):
            before[name]=b'fixed kernel source'
        after=dict(before,**{'spry_fp8_head.py':b'shared head source'})
        sides=[]
        for label,files in (('baseline',before),('cleanup',after)):
            package=root/label/'package-v2';review=package/'review';review.mkdir(parents=True)
            for name,raw in files.items():(review/name).write_bytes(raw)
            run=package.parent/'config/runs/w5-v2';run.mkdir(parents=True)
            (run/'run.json').write_bytes(comparison.canonical(fixture.run))
            (package/'MANIFEST.json').write_bytes(comparison.canonical(dict(package_files={'review/request_receipts.py':'a'*64})))
            execution=root/(label+'-execution');execution.mkdir()
            sides.append(dict(root=execution,cpu=root/(label+'-cpu'),package=package,
                seals={},execution_seal=dict(root=str(execution),manifest_sha256='b'*64,members=1),
                result=dict(operation=1 if label=='baseline' else 2),
                summaries={arm:dict(value=dict(rows=[dict(ordinal=n) for n in range(176)])) for arm in comparison.ARMS}))
        return sides

    def run_final_pair(self,root,output,*,paired_cleanup=False):
        a,b=self.final_pair(root)
        metadata=dict(source_files={'worker_entry.py':{}},source_extractions={},new_sources={},protected_source_files={})
        read=comparison.read
        def configured_read(path,*args,**kwargs):
            return metadata if path==comparison.HERE/'SOURCE.json' else read(path,*args,**kwargs)
        sealer=SimpleNamespace(verify=lambda path,pin:next(side['execution_seal'] for side in (a,b) if side['root']==path))
        work=dict(case='same frozen request',body_sha256='a'*64,prompt_ids=[10],output_ids=[20],
            speculative_work={},cache_work={},sampling_parameters={'temperature':0},finish_reason='stop')
        with patch.object(comparison,'load_side',side_effect=[a,b]), \
             patch.object(comparison,'checked_module',side_effect=[sealer,SimpleNamespace()]), \
             patch.object(comparison,'read',side_effect=configured_read), \
             patch.object(comparison,'request_work',return_value=work) as requests, \
             patch.object(comparison,'paired_groups',side_effect=lambda arm,*args:[g for g in groups() if g['arm']==arm]):
            report=comparison.compare(a['root'],'b'*64,b['root'],'b'*64,root/'seal_evidence.py',
                output,paired_cleanup=paired_cleanup)
        self.assertEqual(requests.call_count,704)
        return report

    def test_final_pair_requires_explicit_membership_selection_and_preserves_all_work_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();output=root/'comparison'
            with self.assertRaisesRegex(ValueError,'membership'):
                self.run_final_pair(root,output)
            self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();report=self.run_final_pair(root,root/'comparison',paired_cleanup=True)
            self.assertTrue(report['matched_work_all_rows']);self.assertEqual(report['rows_compared'],352)
            self.assertEqual(report['source_comparison']['comparison_scope'],'verified pre-memory source versus final cleanup')
            self.assertTrue(report['source_comparison']['native_execution_contracts_equal'])
            self.assertFalse(report['causal_speedup_established'])
            self.assertTrue((root/'comparison/COMPARISON.json').is_file())

    def test_paired_selection_refuses_non_boolean_before_source_or_output_access(self):
        for value in (1,None,'true'):
            with self.subTest(value=value),patch.object(comparison,'checked_module') as loader:
                with self.assertRaisesRegex(ValueError,'boolean paired cleanup'):
                    comparison.compare(None,None,None,None,None,None,paired_cleanup=value)
            loader.assert_not_called()

    def test_optimized_python_refuses_before_any_source_or_output_access(self):
        with patch.object(comparison.sys,'flags',SimpleNamespace(optimize=1)),patch.object(comparison,'checked_module') as loader:
            with self.assertRaisesRegex(ValueError,'unoptimized original seal verifier'):
                comparison.compare(None,None,None,None,None,None)
        loader.assert_not_called()

    def test_generated_output_inside_canonical_sources_refuses_before_loading_evidence(self):
        with patch.object(comparison,'checked_module') as loader:
            with self.assertRaisesRegex(ValueError,'outside the canonical source tree'):
                comparison.compare(None,None,None,None,None,comparison.HERE/'comparison-unit-output-not-created')
        loader.assert_not_called()

    def test_pooled_rate_uses_sum_tokens_over_sum_walls(self):
        for row in comparison.aggregate(groups()):
            self.assertEqual(row['groups'],2);self.assertEqual(row['baseline']['completion_tokens'],200)
            self.assertEqual(row['baseline']['http_wall_seconds'],5)
            self.assertEqual(row['baseline']['combined_tp2_tokens_per_second'],40)
            self.assertEqual(row['baseline']['requests'],4)
            self.assertEqual(row['observed_wall_change_percent'],0)

    def test_warmup_and_quality_observations_remain_outside_primary_totals(self):
        values=groups();excluded=copy.deepcopy(values[0]);excluded['timing_eligible']=False
        excluded['baseline']['completion_tokens']=excluded['cleanup']['completion_tokens']=999999
        values.append(excluded)
        self.assertEqual(comparison.aggregate(values)[0]['baseline']['completion_tokens'],200)
        self.assertEqual(len(values),9)

    def test_missing_timing_population_refuses(self):
        with self.assertRaisesRegex(ValueError,'all four timing populations'):
            comparison.aggregate([row for row in groups() if row['arm']=='bf16'])

    def test_nonfinite_walls_and_boolean_token_counts_refuse(self):
        for field,value in (('wall_seconds',float('nan')),('wall_seconds',float('inf')),('wall_seconds',0),('completion_tokens',True)):
            values=groups();values[0]['baseline'][field]=value
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'finite measured timing'):
                comparison.aggregate(values)

    def test_smaller_group_wall_has_negative_observed_wall_change(self):
        values=groups()
        for row in values:row['cleanup']['wall_seconds']*=0.5
        for row in comparison.aggregate(values):
            self.assertEqual(row['observed_wall_change_percent'],-50)
            self.assertEqual(row['observed_throughput_change_percent'],100)

    def test_full_vector_and_each_work_field_difference_are_retained(self):
        work=dict(case='timing-c1-1-a',body_sha256='a'*64,prompt_ids=[10,11],output_ids=[20,21],
            speculative_work={'spec_verify_ct':2},cache_work={'cached_tokens':0},
            sampling_parameters={'temperature':0},finish_reason='stop')
        replacements=dict(case='different',body_sha256='b'*64,prompt_ids=[10,12],output_ids=[20,22],
            speculative_work={'spec_verify_ct':3},cache_work={'cached_tokens':1},
            sampling_parameters={'temperature':1},finish_reason='length')
        self.assertEqual(comparison.work_differences('bf16',0,work,work),[])
        for field,replacement in replacements.items():
            other=copy.deepcopy(work);other[field]=replacement
            with self.subTest(field=field):
                rows=comparison.work_differences('bf16',0,work,other)
                self.assertEqual(len(rows),1);self.assertEqual(rows[0]['field'],field)
                self.assertNotEqual(rows[0]['baseline_sha256'],rows[0]['cleanup_sha256'])

    def test_token_ids_refuse_boolean_or_empty_vectors(self):
        self.assertTrue(comparison.ids([0,248319]))
        for value in ([],[True],[248320],[-1],(1,2)):
            with self.subTest(value=value):self.assertFalse(comparison.ids(value))

    def test_request_work_reads_full_vectors_and_actual_cache_metadata(self):
        _,raw,side,row=response_fixture()
        work=comparison.request_work(side,'bf16',row,SimpleNamespace(read=lambda path,limit:raw))
        self.assertEqual(work['prompt_ids'],[10,11]);self.assertEqual(work['output_ids'],[20,21])
        self.assertEqual(work['cache_work'],dict(cached_tokens=1,cached_tokens_details={'prefix':1},num_retractions=0))

    def test_request_work_refuses_raw_identity_and_validated_output_disagreement(self):
        _,raw,side,row=response_fixture();codec=SimpleNamespace(read=lambda path,limit:raw)
        row['wire']['response_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'raw response identity'):
            comparison.request_work(side,'bf16',row,codec)
        _,raw,side,row=response_fixture();side['rows'][('bf16',0)]['output_token_ids']=[20,22]
        with self.assertRaisesRegex(ValueError,'actual prompt/output vectors'):
            comparison.request_work(side,'bf16',row,SimpleNamespace(read=lambda path,limit:raw))

    def test_request_work_refuses_retractions_and_unbounded_prompt_ids(self):
        for field,value,reason in (('num_retractions',1,'actual cache/retraction work'),
                                   ('num_retractions',False,'actual cache/retraction work'),
                                   ('prompt_token_ids',[10,248320],'actual prompt/output vectors')):
            body,_,side,row=response_fixture()
            if field=='prompt_token_ids':body['choices'][0][field]=value
            else:body['choices'][0]['meta_info'][field]=value
            raw=comparison.canonical(body);pin=comparison.digest(raw)
            row['wire']['response_sha256']=side['rows'][('bf16',0)]['response_raw_sha256']=pin
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,reason):
                comparison.request_work(side,'bf16',row,SimpleNamespace(read=lambda path,limit:raw))

    def test_html_labels_work_mismatch_and_keeps_all_groups(self):
        values=groups()
        report=dict(timing_populations=comparison.aggregate(values),matched_work_all_rows=False,groups=values,
            work_differences=[dict(case='<unsafe>',field='output_ids')])
        page=comparison.report_html(report)
        self.assertIn('Work differs: descriptive timing only',page)
        self.assertIn('this pilot establishes no causal speedup',page)
        self.assertIn('Output tokens per second combine both GPUs',page)
        self.assertIn('C1/C2 refer to HTTP concurrency; native batch size is unproven',page)
        self.assertIn('&lt;unsafe&gt;',page);self.assertNotIn('<unsafe>',page)
        self.assertEqual(page.count('<rect '),8)


if __name__=='__main__':unittest.main()
