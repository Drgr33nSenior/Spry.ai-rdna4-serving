"""Refuse production performance comparisons with changed work or native inputs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prepare_measurement as measurement


class MeasurementPreparation(unittest.TestCase):
    def setUp(self):
        fixed = ('FINAL-SETTINGS.json', 'FINAL-REQUESTS.zlib', 'MODEL-INVENTORY.json',
                 'DRAFT-MODEL.json', 'PRECISION-REFERENCE.json', 'SERVING-STACK.json',
                 'COMMON-STACK-RECEIPT.json', 'COMPOSITION-FACTS.zlib', 'native-overlay.py', 'retained.py')
        self.before = {name: b'unchanged bytes' for name in fixed}
        self.before['FINAL-SETTINGS.json'] = measurement.canonical(dict(instrumentation='production'))
        self.before.update({name: b'retired' for name in measurement.source_builder.RETIRED_W5_MEMBERS})
        self.before['worker_entry.py'] = b'old checked implementation'
        template = dict(source_manifest_sha256='a'*64,
            binding_template=dict(source_manifest_sha256='a'*64,
                canonical_server_cli=['--speculative-num-draft-tokens', '8'],
                sources=dict(native=dict(path='/mnt/composition/native.py', sha256='b'*64),
                             mxfp4_scheme=dict(path='/mnt/review/isolated_scheme.py', sha256='d'*64),
                             worker=dict(path='/mnt/review/worker_entry.py', sha256='c'*64))),
            worker_template=dict(source_manifest_sha256='a'*64,
                sources=dict(same_as='binding_template.sources', source_manifest_sha256='a'*64),
                arm_config=dict(disable_cuda_graph=False),
                preweight=dict(graph_observer=dict(path='/mnt/review/serving_graph_observer.py', sha256='d'*64))))
        for arm in ('control','mtp'):
            self.before['HTTP-BINDING-'+arm+'.json'] = measurement.canonical(template)
        self.after = {name: raw for name,raw in self.before.items()
                      if name not in measurement.source_builder.RETIRED_W5_MEMBERS}
        self.after.update({name+'.py': b'new helper' for name in measurement.source_builder.EXTRACTED_HELPERS})
        self.after['worker_entry.py'] = b'new checked implementation'
        self.run = dict(instrumentation='production', overlays=['native-overlay.py'])

    def compare(self):
        return measurement.compare_inputs(self.before, self.after, run=self.run,
            canonical_sources={'worker_entry.py'}, protected_sources={})

    def mutate_binding(self, mutate):
        name = 'HTTP-BINDING-mtp.json'
        value = measurement.parse_json(self.after[name]); mutate(value)
        self.after[name] = measurement.canonical(value)

    def test_checked_cleanup_changes_preserve_execution_contract_without_claiming_token_identity(self):
        self.mutate_binding(lambda v: v['binding_template']['sources']['worker'].update(sha256='e'*64))
        self.mutate_binding(lambda v: v['worker_template']['preweight']['graph_observer'].update(sha256='f'*64))
        before = copy.deepcopy(self.after)
        result = self.compare()
        self.assertEqual(self.after, before)
        self.assertTrue(result['native_execution_contracts_equal'])
        self.assertFalse(result['output_token_identity_verified'])
        self.assertFalse(result['c4_included'])

    def test_both_modes_must_be_explicit_production(self):
        for mode in ('on', None, True):
            self.run['instrumentation'] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'production mode'): self.compare()

    def test_changed_requests_model_stack_or_attention_refuse(self):
        for name in ('FINAL-REQUESTS.zlib','FINAL-SETTINGS.json','MODEL-INVENTORY.json',
                     'SERVING-STACK.json','native-overlay.py','retained.py'):
            old = self.after[name]; self.after[name] = old+b'changed'
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'measurement input differs'): self.compare()
            self.after[name] = old

    def test_changed_draft_width_refuses_even_when_private_pins_are_allowed(self):
        self.mutate_binding(lambda v: v['binding_template']['canonical_server_cli'].__setitem__(1, '4'))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_changed_native_source_pin_refuses(self):
        self.mutate_binding(lambda v: v['binding_template']['sources']['native'].update(sha256='f'*64))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_changed_private_numeric_source_pin_refuses(self):
        self.mutate_binding(lambda v: v['binding_template']['sources']['mxfp4_scheme'].update(sha256='f'*64))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_private_source_cannot_move_under_an_allowed_pin_change(self):
        self.mutate_binding(lambda v: v['binding_template']['sources']['worker'].update(path='/mnt/review/other.py'))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_changed_graph_policy_refuses(self):
        self.mutate_binding(lambda v: v['worker_template']['arm_config'].update(disable_cuda_graph=True))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_numeric_zero_cannot_replace_a_false_execution_flag(self):
        self.mutate_binding(lambda v: v['worker_template']['arm_config'].update(disable_cuda_graph=0))
        with self.assertRaisesRegex(ValueError, 'native execution contract differs'): self.compare()

    def test_unreviewed_additions_and_missing_files_refuse(self):
        self.after['extra.py'] = b'extra'
        with self.assertRaisesRegex(ValueError, 'membership'): self.compare()
        self.after.pop('extra.py');self.after.pop('retained.py')
        with self.assertRaisesRegex(ValueError, 'membership'): self.compare()

    def test_existing_output_refuses_before_authenticating_or_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); marker = root/'keep'; marker.write_bytes(b'user data')
            with patch.object(measurement.source_builder, 'authenticated_packet') as authenticate:
                with self.assertRaisesRegex(ValueError, 'fresh canonical'):
                    measurement.prepare(root/'donor', root, packet_id='a'*16)
            authenticate.assert_not_called();self.assertEqual(marker.read_bytes(), b'user data')


if __name__ == '__main__': unittest.main()
