"""Compare actual transport grants with authenticated complete controller copies."""
import argparse
import ast
import copy
import difflib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_config_binding_packet as binding_checks
from test_contract_cleanup import load
from test_transport_binding_proposal import PROPOSAL, TRANSPORT, canonical, sha


class TransportBindingPacket(binding_checks.ConfigBindingPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transport_record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.before = (cls.packages['baseline'] / TRANSPORT).read_bytes()
        cls.after = (PROPOSAL / TRANSPORT).read_bytes()
        if (sha(cls.before) != cls.transport_record['baseline_sha256']
                or sha(cls.after) != cls.transport_record['proposed_sha256']):
            raise AssertionError('transport proposal does not join authenticated donor')
        (cls.packages['proposed'] / TRANSPORT).write_bytes(cls.after)
        path = cls.packages['proposed'] / 'MANIFEST.json'
        manifest = json.loads(path.read_bytes())
        manifest['package_files'][TRANSPORT] = sha(cls.after)
        path.write_bytes(canonical(manifest))
        cls.transports = {label: load('cleanup_transport_' + label, package / TRANSPORT)
                          for label, package in cls.packages.items()}
        cls.evidence.update(proposed_transport_sha256=sha(cls.after),
            transport_patch_sha256=cls.transport_record['patch_sha256'],
            inherited_check_source_sha256=sha((HERE / 'check_config_binding_packet.py').read_bytes()),
            transport_grant_fields_implemented=True, configuration_staging_implemented=True,
            transport_producer_migrated=False, complete_configured_archive_admitted=False,
            controller_source_review_approved=True, transport_source_review_approved=False)

    def grant(self, controller, manifest, digest, root, label):
        expected, parent = super().grant(controller, manifest, digest, root, label)
        args = dict(received_utc=expected['received_utc'], received_monotonic=expected['received_monotonic'], parent=parent)
        if label == 'proposed':
            args['configuration_pins'] = {key: expected[key] for key in self.transports[label].CONFIGURATION_PIN_FIELDS}
        actual = self.transports[label].new_grant(manifest, digest, **args)
        self.assertEqual(actual, expected, 'producer changed an existing grant field')
        return actual, parent

    def test_exact_transport_patch_and_unchanged_owner_control_flow(self):
        patch_bytes = ''.join(difflib.unified_diff(self.before.decode().splitlines(True),
            self.after.decode().splitlines(True), fromfile='a/' + TRANSPORT, tofile='b/' + TRANSPORT)).encode()
        self.assertEqual((PROPOSAL / 'D2.patch').read_bytes(), patch_bytes)
        old, new = ast.parse(self.before), ast.parse(self.after)
        baseline = {node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        proposed = {node.name: node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        changed = [name for name, node in baseline.items() if ast.dump(node) != ast.dump(proposed[name])]
        self.assertEqual(changed, self.transport_record['changed_existing_definitions'])
        original_assignments = lambda tree: [ast.dump(node) for node in tree.body if isinstance(node,
            (ast.Assign, ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id in {'CONFIGURATION_PIN_FIELDS', 'PROTECTED_SOURCE_NAMES'}
                for target in node.targets))]
        self.assertEqual(original_assignments(old), original_assignments(new))
        restored = copy.deepcopy(proposed['human_main'])
        restored.body = [node for node in restored.body if not (isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == 'configuration_pins' for target in node.targets))]
        removed = 0
        for node in ast.walk(restored):
            if isinstance(node, ast.Call):
                removed += sum(keyword.arg == 'configuration_pins' for keyword in node.keywords)
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'configuration_pins']
        self.assertEqual(removed, 2)
        self.assertEqual(ast.dump(restored), ast.dump(baseline['human_main']))
        self.evidence['unchanged_owner_control_flow'] = 'PASS_EXCEPT_EXPLICIT_CONFIGURATION_PINS'

    def test_real_retained_archive_fits_original_limit_but_configured_extension_refuses(self):
        package = self.packages['baseline']
        manifest = json.loads((package / 'MANIFEST.json').read_bytes())
        members = {name: (package / name).read_bytes() for name in manifest['package_files']}
        members['MANIFEST.json'] = (package / 'MANIFEST.json').read_bytes()
        self.assertEqual(len(members), 80)
        def archive(name):
            destination = self.root / name
            with tarfile.open(destination, 'w') as output:
                for path, raw in members.items():
                    item = tarfile.TarInfo(path)
                    item.size = len(raw)
                    output.addfile(item, io.BytesIO(raw))
            return destination
        path = archive('baseline-archive.tar')
        self.assertEqual(self.transports['baseline'].inspect_archive(path, sha(path.read_bytes()),
            sha(members['MANIFEST.json']))[0], manifest)
        members[TRANSPORT] = self.after
        members['review/benchmark_physical_controller.py'] = (
            self.packages['proposed'] / 'review/benchmark_physical_controller.py').read_bytes()
        manifest['package_files'].update({name: sha(members[name]) for name in
            (TRANSPORT, 'review/benchmark_physical_controller.py')})
        manifest['run_config'] = 'config/runs/w5-v2/run.json'
        for name in ('config/runs/w5-v2/run.json', 'config/runs/w5-v2/menu.json', 'config/host/r9700-pair.json'):
            members[name] = canonical(json.loads((HERE.parent / name).read_bytes()))
        members['PROTECTED-SHA256.json'] = canonical({'review/' + name: pin for name, pin in
            self.transport_record['proposed_protected_files'].items()})
        members['MANIFEST.json'] = canonical(manifest)
        pins = dict(run_config_sha256=sha(members[manifest['run_config']]),
            host_sha256=sha(members['config/host/r9700-pair.json']),
            protected_set_sha256=sha(members['PROTECTED-SHA256.json']))
        self.assertEqual(len(members), 84)
        path = archive('configured-archive.tar')
        with self.assertRaisesRegex(ValueError, 'archive expanded size/member limit'):
            self.transports['proposed'].inspect_archive(path, sha(path.read_bytes()),
                sha(members['MANIFEST.json']), configuration_pins=pins)
        self.evidence['archive_capacity'] = dict(retained_members=80, configured_members=84,
            maximum_members=80, original_archive_accepted=True, configured_archive_refused=True,
            ceiling_changed=False, package_pruning_required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    binding_checks.PACKET = options.packet
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(TransportBindingPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
