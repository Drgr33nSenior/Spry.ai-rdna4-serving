"""W5 configuration structure, arithmetic and mutation refusal without native code."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_config
from spry_common.canon import canonical_bare, canonical_line, digest, parse_json
from spry_common.limits import derive


class RunConfiguration(unittest.TestCase):
    def setUp(self):
        self.run = parse_json((HERE.parent / 'config/runs/w5-v2/run.json').read_bytes())
        self.menu = parse_json((HERE.parent / 'config/runs/w5-v2/menu.json').read_bytes())
        self.host = parse_json((HERE.parent / 'config/host/r9700-pair.json').read_bytes())
        self.schemas = {name: parse_json((HERE.parent / 'config/schema' / (name + '.schema.json')).read_bytes())
                        for name in ('run-config', 'host', 'menu')}

    def test_real_run_host_and_menu_match_their_closed_schemas_and_pins(self):
        for name, value in (('run-config', self.run), ('menu', self.menu), ('host', self.host)):
            run_config.validate_schema(value, self.schemas[name])
        self.assertEqual(digest(canonical_line(self.menu)), self.run['menu']['sha256'])
        self.assertEqual(digest(canonical_line(self.host)), self.run['host']['sha256'])
        self.assertEqual(len(self.menu['rows']), 176)
        self.assertEqual({row['stop_policy'] for row in self.menu['rows']}, {'natural'})
        self.assertNotIn('boot_id', self.host)

    def test_explicit_json_encodings_preserve_newline_and_reject_nonfinite_values(self):
        self.assertEqual(canonical_bare({'b': 2, 'a': 1}), b'{"a":1,"b":2}')
        self.assertEqual(canonical_line({'b': 2, 'a': 1}), b'{"a":1,"b":2}\n')
        self.assertEqual(parse_json(b'{ "a": 1 }'), {'a': 1})
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                parse_json(raw)
        with self.assertRaises(ValueError):
            canonical_line(float('nan'))

    def test_unknown_missing_and_derived_fields_are_rejected(self):
        for key in self.run:
            value = copy.deepcopy(self.run)
            del value[key]
            with self.assertRaisesRegex(ValueError, 'missing or unknown'):
                run_config.validate_schema(value, self.schemas['run-config'])
        for scope in ('root', 'profile', 'limits'):
            value = copy.deepcopy(self.run)
            (value if scope == 'root' else value[scope])['worker_seconds'] = 2280
            with self.assertRaises(ValueError):
                run_config.validate_schema(value, self.schemas['run-config'])

    def test_wrong_numeric_types_and_bad_paths_are_rejected(self):
        for bad in (True, '2040', 2040.0, 0, -1):
            value = copy.deepcopy(self.run)
            value['limits']['client_seconds'] = bad
            with self.assertRaises(ValueError):
                run_config.validate_schema(value, self.schemas['run-config'])
        for bad in ('../../outside', '/absolute', 'config/host/../outside.json'):
            value = copy.deepcopy(self.run)
            value['host']['file'] = bad
            with self.assertRaises(ValueError):
                run_config.validate_schema(value, self.schemas['run-config'])

    def test_unknown_stop_policy_and_boot_id_are_rejected(self):
        self.menu['rows'][0]['stop_policy'] = 'anything'
        with self.assertRaises(ValueError):
            run_config.validate_schema(self.menu, self.schemas['menu'])
        self.host['boot_id'] = 'not-a-configuration-fact'
        with self.assertRaises(ValueError):
            run_config.validate_schema(self.host, self.schemas['host'])

    def test_schema_keywords_cannot_be_silently_ignored(self):
        schema = copy.deepcopy(self.schemas['host'])
        schema['unimplementedKeyword'] = True
        with self.assertRaisesRegex(ValueError, 'unsupported schema keyword'):
            run_config.validate_schema(self.host, schema)

    def test_retained_time_and_durable_limits_are_derived(self):
        value = derive(self.run['limits'])
        self.assertEqual(value, dict(worker_seconds=2280, arm_work_seconds=2430, arm_envelope_seconds=2490,
            active_seconds=6060, total_seconds=6360, caller_seconds=7160,
            http_payload_bytes=47 << 20, http_session_max_bytes=int(496.5 * (1 << 20))))

    def test_primary_changes_flow_through_time_derivations(self):
        baseline = derive(self.run['limits'])
        value = copy.deepcopy(self.run['limits'])
        value['client_seconds'] -= 10
        changed = derive(value)
        for key in ('worker_seconds', 'arm_work_seconds', 'arm_envelope_seconds'):
            self.assertEqual(changed[key], baseline[key] - 10)
        for key in ('active_seconds', 'total_seconds', 'caller_seconds'):
            self.assertEqual(changed[key], baseline[key] - 20)

    def test_derived_overrides_invalid_integers_and_session_overflow_refuse(self):
        for key, bad in (('worker_seconds', 2280), ('client_seconds', True),
                         ('native_tree_files', -1), ('session_bytes', 400 << 20)):
            value = copy.deepcopy(self.run['limits'])
            value[key] = bad
            with self.assertRaises(ValueError):
                derive(value)

    def test_protected_literal_reader_does_not_evaluate_code(self):
        self.assertEqual(run_config.literal_assignment(b'CAP = {"file": 8 << 20}\n', 'CAP'), {'file': 8 << 20})
        for raw in (b'CAP = dangerous_call()\n', b'CAP = 1\nCAP = 2\n', b'CAP = other\n'):
            with self.assertRaises(ValueError):
                run_config.literal_assignment(raw, 'CAP')

    def test_changed_config_references_and_baseline_refuse_before_donor_access(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-config-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        shutil.copytree(HERE.parent / 'config', root / 'config')
        run_dir = root / 'config/runs/w5-v2'
        metadata = parse_json((HERE.parent / 'SOURCE.json').read_bytes())
        for kind in ('profile', 'limits', 'host', 'menu', 'policy', 'instrumentation'):
            run, menu, host = copy.deepcopy(self.run), copy.deepcopy(self.menu), copy.deepcopy(self.host)
            if kind == 'profile':run['profile']['context_length'] = 65536
            if kind == 'limits':run['limits']['client_seconds'] = 2030
            if kind == 'instrumentation':run['instrumentation'] = 'on'
            if kind == 'host':host['power_cap_watts'] = 230
            if kind == 'menu':menu['rows'] = menu['rows'] * 2
            if kind == 'policy':menu['rows'][0]['stop_policy'] = 'cap'
            run['host']['sha256'] = digest(canonical_line(host))
            run['menu']['sha256'] = digest(canonical_line(menu))
            for path, value in ((run_dir / 'run.json', run), (run_dir / 'menu.json', menu),
                                (root / run['host']['file'], host)):
                path.write_bytes(canonical_line(value))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'supported W5 baseline'):
                run_config.load(run_dir, root, metadata, {}, lambda path, maximum: path.read_bytes())


if __name__ == '__main__':
    unittest.main()
