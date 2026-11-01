"""Compare quality projections against sealed sources and actual CPU serving wire."""
import argparse
import copy
import difflib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_durable_limits_packet as durable_checks
from check_protected_packet import isolated_modules
from test_config_binding_proposal import canonical, sha
from test_contract_cleanup import load
from test_quality_policy import menus, arm_results, Tokenizer

SOURCE = HERE.parent / 'quality'


class QualityPolicyPacket(durable_checks.DurableLimitsPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        baseline = cls.root / 'quality-baseline'; baseline.mkdir()
        for name in ('quality_contract.py', 'thinking_segments.py', 'final_quality_inputs.py'):
            branch = 'producer' if name == 'final_quality_inputs.py' else 'source'
            (baseline / name).write_bytes(cls.files['quality-template-w5/' + branch + '/' + name])
        cls.old_quality = load('cleanup_retained_quality', baseline / 'quality_contract.py')
        cls.old_producer = load('cleanup_retained_quality_producer', baseline / 'final_quality_inputs.py')
        cls.quality = load('cleanup_current_quality', SOURCE / 'quality_contract.py')
        cls.thinking = load('cleanup_quality_marker_split', SOURCE / 'thinking_segments.py')
        with patch.dict(sys.modules, quality_contract=cls.quality):
            cls.producer = load('cleanup_configured_quality_producer', SOURCE / 'final_quality_inputs.py')
        cls.evidence['quality_native_tokenizer_executed'] = False

    def project_wire(self, wires, *, cap=False):
        private = json.loads(self.files['quality-template-w5/producer/inputs/QUALITY-MENU.json'])['rows']
        run = json.loads(self.members['config/runs/w5-v2/menu.json'])['rows']
        if cap:
            run[0]['stop_policy'] = 'cap'
        menu = self.producer.project_menu(private, run)
        value = self.producer.build_input(wires, menu=menu)
        self.quality.validate_input(value, menu)
        if cap:
            self.assertEqual(value['rows'][0]['scope'], 'ungraded-cap-work')
            self.assertIsNone(value['rows'][0]['expected_answer'])
            self.assertEqual(sum(row['scope'] == 'ungraded-cap-work' for row in value['rows']), 2)
        else:
            frozen = [dict(copy.deepcopy(row), arm=arm) for arm in ('bf16', 'kv4') for row in private]
            self.assertEqual(canonical(menu), canonical(frozen))
            with patch.object(self.old_producer, 'expected_menu', return_value=frozen):
                retained = self.old_producer.build_input(wires)
            self.old_quality.validate_input(retained, frozen)
            self.assertEqual(self.quality.encoded(value), self.old_quality.encoded(retained))
        return len(self.quality.encoded(value))

    def check_postrun(self, shape):
        original = self.postrun.quality_input_bytes
        sizes = []
        def verified(wires):
            sizes.append(self.project_wire(wires))
            return original(wires)
        with patch.object(self.postrun, 'quality_input_bytes', side_effect=verified):
            super().check_postrun(shape)
        self.evidence['quality_projection_' + shape] = dict(input_bytes=sizes,
            baseline_input_byte_identical=True, private_oracles_kept_outside_source=True,
            full_wire_decode_executed=False)

    def test_declared_cap_row_completes_actual_client_reader_summary_and_protected_report(self):
        original = self.postrun.quality_input_bytes
        sizes = []
        def validated_cap(wires):
            sizes.append(self.project_wire(wires, cap=True))
            return original(wires)  # Retained natural-only packet must still refuse.
        with patch.object(self.postrun, 'quality_input_bytes', side_effect=validated_cap):
            super().test_declared_cap_row_completes_actual_client_reader_summary_and_protected_report()
        self.evidence['declared_cap_quality_projection'] = dict(input_bytes=sizes,
            both_complete_wires_retained=True, cap_oracles_removed=True, grade_claimed=False)

    def test_all_natural_352_row_input_decode_and_grade_match_the_retained_implementation(self):
        private, run = menus()
        menu = self.producer.project_menu(private, run)
        wires = arm_results(menu)
        value = self.producer.build_input(wires, menu=menu)
        with patch.object(self.old_producer, 'expected_menu', return_value=menu):
            retained = self.old_producer.build_input(wires)
        self.assertEqual(self.quality.encoded(value), self.old_quality.encoded(retained))
        with patch.dict(sys.modules, thinking_segments=self.thinking):
            answers = self.quality.decode_rows(value, menu, Tokenizer())
            old_answers = self.old_quality.decode_rows(retained, menu, Tokenizer())
            grades = self.quality.grade_answers(value, menu, answers, [])
            old_grades = self.old_quality.grade_answers(retained, menu, old_answers, [])
        self.assertEqual(self.quality.encoded(answers), self.old_quality.encoded(old_answers))
        self.assertEqual(self.quality.encoded(grades), self.old_quality.encoded(old_grades))
        self.evidence['quality_natural_baseline'] = dict(rows=352,
            input_answers_and_grades_byte_identical=True, tokenizer='CPU stand-in')

    def test_quality_source_pins_and_exact_patch(self):
        record = json.loads((SOURCE / 'PROPOSAL.json').read_bytes())
        expected = ''
        for name, row in record['sources'].items():
            before = self.files[row['source_member']]
            after = (SOURCE / name).read_bytes()
            self.assertEqual(sha(before), row['baseline_sha256'])
            self.assertEqual(sha(after), row['proposed_sha256'])
            expected += ''.join(difflib.unified_diff(before.decode().splitlines(True),
                after.decode().splitlines(True), fromfile='a/' + name, tofile='b/' + name))
        self.assertEqual(expected, (SOURCE / 'SOURCE.patch').read_text())
        self.assertEqual(sha((SOURCE / 'SOURCE.patch').read_bytes()), record['source_patch_sha256'])
        self.evidence.update(quality_patch_verified=True, quality_source_consumed_by_builder=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    durable_checks.binding_checks.host_checks.stop_checks.artifact_checks.worker_checks.session_checks.inventory_checks.PACKET = options.packet
    with isolated_modules():
        result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report:
        report = dict(QualityPolicyPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream: stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
