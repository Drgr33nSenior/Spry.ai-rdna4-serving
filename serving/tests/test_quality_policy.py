"""CPU grading semantics; tokenizer stand-in cannot demonstrate native token identity."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load

SOURCE = HERE.parent / 'quality'


class Tokenizer:
    def token_to_id(self, token): return {'<think>':248068,'</think>':248069}[token]
    def decode(self, ids, skip_special_tokens=True):
        return ''.join({10:'thinking',12:'answer',248068:'<think>',248069:'</think>'}[token] for token in ids)


def menus(count=176, cap_ordinals=()):
    private, run = [], []
    for ordinal in range(count):
        pin = f'{ordinal:064x}'
        private.append(dict(ordinal=ordinal,case=f'case-{ordinal}',request_body_sha256=pin,
            maximum_output_tokens=1024,scope='natural-oracle',expected_answer='answer',
            grade_kind='exact_text',minimum_final_tokens=0))
        run.append(dict(ordinal=ordinal,case=f'case-{ordinal}',sha256=pin,maximum_output_tokens=1024,
                        stop_policy='cap' if ordinal in cap_ordinals else 'natural'))
    return private,run


def arm_results(menu, *, stop_caps=False):
    result={}
    for arm,logical in (('bf16','control'),('kv4','mtp')):
        rows=[]
        for plan in (row for row in menu if row['arm']==arm):
            capped=plan['scope']=='ungraded-cap-work' and not stop_caps
            ids=[12]*plan['maximum_output_tokens'] if capped else [10,248069,12]
            rows.append(dict(ordinal=plan['ordinal'],case=plan['case'],
                request_body_sha256=plan['request_body_sha256'],response_raw_sha256='a'*64,
                response_canonical_sha256='b'*64,raw_response_sha256='b'*64,
                status='OUTPUT_CAP_REACHED' if capped else 'NORMAL_STOP_COMPLETE',
                finish_reason='length' if capped else 'stop',response_token_ids=ids,
                text=Tokenizer().decode(ids)))
        result[arm]=dict(wire_complete=True,arm=logical,qualification=False,precision_accepted=False,
            answer_grade_status='PENDING_POST_RELEASE_NATIVE_DECODE',rows=rows)
    return result


class QualityPolicy(unittest.TestCase):
    def setUp(self):
        self.quality=load('cleanup_quality_policy',SOURCE/'quality_contract.py')
        self.thinking=load('cleanup_quality_thinking',SOURCE/'thinking_segments.py')
        self.patcher=patch.dict(sys.modules,quality_contract=self.quality,thinking_segments=self.thinking)
        self.patcher.start();self.addCleanup(self.patcher.stop)
        self.producer=load('cleanup_quality_producer',SOURCE/'final_quality_inputs.py')

    def fixture(self,count=176,caps=(),stop_caps=False):
        private,run=menus(count,caps)
        menu=self.producer.project_menu(private,run)
        wires=arm_results(menu,stop_caps=stop_caps)
        value=self.producer.build_input(wires,menu=menu)
        return menu,wires,value

    def grade(self,menu,value):
        answers=self.quality.decode_rows(value,menu,Tokenizer())
        return answers,self.quality.grade_answers(value,menu,answers,[])

    def test_all_natural_baseline_keeps_the_352_row_result_schema_and_grades(self):
        menu,_,value=self.fixture()
        answers,report=self.grade(menu,value)
        self.assertEqual((report['decoded_rows'],report['natural_rows'],report['ungraded_cap_work_rows']),(352,352,0))
        self.assertTrue(report['quality_pass']);self.assertTrue(report['long_answer_coverage_pass'])
        self.assertFalse(report['qualification']);self.assertEqual(len(answers),352)

    def test_one_declared_cap_in_each_arm_is_ungraded_with_no_answer_hash_or_coverage_claim(self):
        menu,_,value=self.fixture(caps={0})
        answers,report=self.grade(menu,value)
        self.assertEqual((report['natural_rows'],report['ungraded_cap_work_rows']),(350,2))
        for index in (0,176):
            self.assertEqual(answers[index]['final_status'],'UNGRADED_CAP_WORK')
            self.assertIsNone(answers[index]['grade_value_sha256'])
            self.assertEqual(report['grades'][index]['status'],'UNGRADED_CAP_WORK')
            self.assertIsNone(report['grades'][index]['quality_passed'])
            self.assertIsNone(report['grades'][index]['long_answer_coverage_passed'])
        self.assertTrue(report['quality_pass'])

    def test_a_naturally_stopped_cap_request_still_has_no_answer_grade(self):
        menu,_,value=self.fixture(caps={0},stop_caps=True)
        answers,report=self.grade(menu,value)
        self.assertEqual(value['rows'][0]['finish_reason'],'stop')
        self.assertEqual(answers[0]['final_status'],'UNGRADED_CAP_WORK')
        self.assertIsNone(report['grades'][0]['quality_passed'])

    def test_all_cap_and_doubled_menus_count_actual_rows_and_cannot_claim_quality_success(self):
        for count in (176,352):
            with self.subTest(count=count):
                menu,_,value=self.fixture(count=count,caps=set(range(count)))
                answers,report=self.grade(menu,value)
                self.assertEqual((report['decoded_rows'],report['natural_rows'],report['ungraded_cap_work_rows']),
                                 (2*count,0,2*count))
                self.assertEqual(len(answers),2*count)
                self.assertFalse(report['quality_pass']);self.assertFalse(report['long_answer_coverage_pass'])

    def test_doubled_natural_menu_preserves_every_row_and_both_arm_orders(self):
        menu,_,value=self.fixture(count=352)
        _,report=self.grade(menu,value)
        self.assertEqual(report['decoded_rows'],704);self.assertTrue(report['quality_pass'])
        self.assertEqual((value['rows'][351]['arm'],value['rows'][352]['arm']),('bf16','kv4'))

    def test_length_stop_requires_declared_cap_and_exact_output_length_in_both_consumers(self):
        for capped in (False,True):
            menu,wires,value=self.fixture(caps={0} if capped else ())
            actual=wires['bf16']['rows'][0]
            actual.update(status='OUTPUT_CAP_REACHED',finish_reason='length')
            actual['response_token_ids']=actual['response_token_ids'][:-1]
            with self.subTest(capped=capped),self.assertRaises(ValueError):
                self.producer.build_input(wires,menu=menu)
            value['rows'][0]['finish_reason']='length'
            value['rows'][0]['output_token_ids']=value['rows'][0]['output_token_ids'][:-1]
            with self.assertRaises(ValueError): self.quality.validate_input(value,menu)

    def test_caps_still_require_the_full_wire_decode_to_match(self):
        menu,_,value=self.fixture(caps={0})
        value['rows'][0]['wire_text']='wrong'
        with self.assertRaisesRegex(ValueError,'full native decode'): self.grade(menu,value)

    def test_cap_work_cannot_smuggle_an_answer_hash_into_decoded_output(self):
        menu,_,value=self.fixture(caps={0})
        answers,_=self.grade(menu,value)
        answers[0]['grade_value_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'cap-work never graded'):
            self.quality.grade_answers(value,menu,answers,[])

    def test_projection_refuses_wrong_body_case_cap_order_and_implicit_policy(self):
        private,run=menus()
        for key,replacement in (('sha256','f'*64),('case','wrong'),('maximum_output_tokens',1023),
                                ('ordinal',True),('stop_policy',None)):
            changed=copy.deepcopy(run);changed[0][key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError): self.producer.project_menu(private,changed)

    def test_projection_removes_cap_oracles_without_mutating_private_input(self):
        private,run=menus(cap_ordinals={0});before=copy.deepcopy(private)
        projected=self.producer.project_menu(private,run)
        self.assertEqual(private,before)
        self.assertIsNone(projected[0]['expected_answer']);self.assertIsNone(projected[0]['grade_kind'])
        self.assertEqual(projected[1]['expected_answer'],'answer')

    def test_menu_arm_drift_duplicate_cases_and_cap_oracles_refuse(self):
        menu,_,_=self.fixture(caps={0})
        for mutate in (lambda m:m[176].update(request_body_sha256='f'*64),
                       lambda m:m[1].update(case=m[0]['case']),
                       lambda m:m[0].update(expected_answer='must not grade')):
            changed=copy.deepcopy(menu);mutate(changed)
            with self.assertRaises(ValueError): self.quality.menu_rows(changed)

    def test_resolved_doubled_source_plan_keeps_oracles_and_removes_cap_oracles(self):
        private,source=menus(count=2);before=copy.deepcopy(private)
        plan=copy.deepcopy(source)+[dict(row,source_ordinal=row['ordinal'],ordinal=row['ordinal']+2,
            case=row['case']+'-copy2',stop_policy='cap') for row in source]
        resolved=[dict(row,sha256=f'{n+100:064x}') for n,row in enumerate(plan)]
        projected=self.producer.project_menu(private,resolved,source_plan=plan)
        self.assertEqual(len(projected),8);self.assertEqual(private,before)
        self.assertEqual(projected[0]['expected_answer'],'answer')
        self.assertEqual(projected[2]['scope'],'ungraded-cap-work');self.assertIsNone(projected[2]['expected_answer'])
        self.assertEqual(projected[2]['request_body_sha256'],resolved[2]['sha256'])

    def test_resolved_source_plan_refuses_oracle_body_drift_or_cap_growth(self):
        private,plan=menus(count=2);resolved=copy.deepcopy(plan)
        for mutate in (lambda p,r:p[0].update(sha256='f'*64),lambda p,r:r[0].update(stop_policy='cap'),
                       lambda p,r:r[0].update(ordinal=True),lambda p,r:r[0].update(maximum_output_tokens=1025)):
            source=copy.deepcopy(plan);actual=copy.deepcopy(resolved);mutate(source,actual)
            with self.assertRaises(ValueError):self.producer.project_menu(private,actual,source_plan=source)


if __name__=='__main__': unittest.main()
