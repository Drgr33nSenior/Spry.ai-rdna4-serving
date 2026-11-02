"""Test the D2 approval boundary through complete authenticated CPU controller copies."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
from test_source_builder import builder
from test_config_binding_proposal import CONTROLLER, D1, PROPOSAL, canonical, sha
from check_protected_packet import isolated_modules


class ConfigBindingPacket(unittest.TestCase):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        files = builder.authenticated_packet(PACKET.resolve(strict=True), cls.metadata)
        cls.record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.d1 = json.loads((D1 / 'PROPOSAL.json').read_bytes())
        if (cls.record['source_seal_sha256'] != cls.metadata['source_seal_sha256']
                or cls.d1['baseline_sha256'] != cls.metadata['protected_source_files'][Path(CONTROLLER).name]
                or sha((D1 / CONTROLLER).read_bytes()) != cls.record['baseline_sha256']
                or sha((PROPOSAL / CONTROLLER).read_bytes()) != cls.record['proposed_sha256']):
            raise AssertionError('proposal source chain differs')
        cls.temporary = tempfile.TemporaryDirectory(prefix='cleanup-d2-packet-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        helper = cls.root / 'test_controller_binding.py'
        helper.write_bytes(files['test_controller_binding.py'])
        cls.binding = load('cleanup_d2_retained_boundary', helper)
        cls.packages = {}
        for label, proposal in [('baseline', D1), ('proposed', PROPOSAL)]:
            package = cls.root / label
            for name, raw in files.items():
                if name.startswith('author/package-v2/'):
                    path = package / name.removeprefix('author/package-v2/')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
            (package / CONTROLLER).write_bytes((proposal / CONTROLLER).read_bytes())
            manifest = json.loads((package / 'MANIFEST.json').read_bytes())
            manifest['package_files'][CONTROLLER] = sha((package / CONTROLLER).read_bytes())
            if label == 'proposed':
                manifest['run_config'] = 'config/runs/w5-v2/run.json'
            (package / 'MANIFEST.json').write_bytes(canonical(manifest))
            cls.packages[label] = package
        cls.evidence.update(source_seal_sha256=cls.metadata['source_seal_sha256'],
            baseline_controller_sha256=cls.record['baseline_sha256'],
            proposed_controller_sha256=cls.record['proposed_sha256'],
            retained_prelaunch_fixture_sha256=sha(files['test_controller_binding.py']),
            native_runtime_tested=False, launch_authorized=False, owner_approved=False,
            transport_producer_migrated=False, runtime_config_consumers_migrated=False,
            full_prelaunch_completed=False)

    @contextmanager
    def controller(self, label):
        with isolated_modules(), self.binding.controller_on_mirror(self.packages[label]) as value:
            controller, path, digest, manifest, root = value
            if label == 'proposed':
                for name in ('config/runs/w5-v2/run.json', 'config/runs/w5-v2/menu.json',
                             'config/host/r9700-pair.json'):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(canonical(json.loads((HERE.parent / name).read_bytes())))
                pins = {'review/' + name: sha((root / 'review' / name).read_bytes())
                        for name in controller.PROTECTED_SOURCE_NAMES}
                (root / 'PROTECTED-SHA256.json').write_bytes(canonical(pins))
            yield controller, path, digest, manifest, root

    def grant(self, controller, manifest, digest, root, label):
        parent = dict(pid=100, start_ticks=200, uid=0, boot_id=manifest['boot_id'])
        value = dict(schema=2, scope='one-matched-fp8-kv-benchmark', model_start_authorized=True,
            maximum_starts=2, arm_order=['bf16', 'kv4'], cumulative_ceiling_seconds=6360,
            fixed_reserve_seconds=300, received_utc='2026-01-01T00:00:00Z', received_monotonic=100,
            boot_id=manifest['boot_id'], manifest_sha256=digest,
            package_sha256=controller.mapping_sha256(manifest['package_files']),
            source_manifest_sha256=controller.mapping_sha256({name: pin for name, pin in
                manifest['package_files'].items() if name.endswith('.py')}), transport_parent=parent)
        if label == 'proposed':
            value.update(run_config_sha256=sha((root / manifest['run_config']).read_bytes()),
                         host_sha256=sha((root / 'config/host/r9700-pair.json').read_bytes()),
                         protected_set_sha256=sha((root / 'PROTECTED-SHA256.json').read_bytes()))
        return value, parent

    def validate(self, controller, grant, manifest, digest, parent):
        return controller.validate_physical_grant(grant, manifest, digest, parent, clock=lambda: 101,
            utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1))

    def test_actual_grant_accepts_exact_binding_for_both_controller_versions(self):
        for label in self.packages:
            with self.controller(label) as (controller, path, digest, _, root):
                manifest = json.loads(path.read_bytes())
                grant, parent = self.grant(controller, manifest, digest, root, label)
                self.assertEqual(self.validate(controller, grant, manifest, digest, parent), grant)
        self.evidence['valid_grants'] = 'PASS_D1_AND_D2_WITH_THEIR_EXACT_FIELDS'

    def test_existing_grant_refusals_are_unchanged(self):
        mutations = dict(maximum_starts=True, arm_order=['kv4', 'bf16'], cumulative_ceiling_seconds=6361,
            fixed_reserve_seconds=299, package_sha256='0' * 64, source_manifest_sha256='0' * 64,
            received_monotonic=-1000, boot_id='other')
        results = {}
        for label in self.packages:
            with self.controller(label) as (controller, path, digest, _, root):
                manifest = json.loads(path.read_bytes())
                grant, parent = self.grant(controller, manifest, digest, root, label)
                results[label] = {}
                for key, bad in mutations.items():
                    with self.subTest(label=label, key=key), self.assertRaises(ValueError) as caught:
                        self.validate(controller, dict(grant, **{key: bad}), manifest, digest, parent)
                    results[label][key] = str(caught.exception)
                with self.assertRaises(ValueError) as caught:
                    self.validate(controller, grant, manifest, digest, dict(parent, uid=1000))
                results[label]['parent_uid'] = str(caught.exception)
        self.assertEqual(results['baseline'], results['proposed'])
        self.evidence['preserved_grant_refusals'] = results['proposed']

    def test_actual_grant_refuses_wrong_new_pins_and_each_changed_protected_helper(self):
        with self.controller('proposed') as (controller, path, digest, _, root):
            manifest = json.loads(path.read_bytes())
            grant, parent = self.grant(controller, manifest, digest, root, 'proposed')
            for key in ('run_config_sha256', 'host_sha256', 'protected_set_sha256'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.validate(controller, dict(grant, **{key: '0' * 64}), manifest, digest, parent)
            for name in controller.PROTECTED_SOURCE_NAMES:
                source = root / 'review' / name
                before = source.read_bytes()
                source.write_bytes(before + b'\n# changed after import\n')
                try:
                    with self.subTest(helper=name), self.assertRaisesRegex(ValueError, 'digest mismatch'):
                        self.validate(controller, grant, manifest, digest, parent)
                finally:
                    source.write_bytes(before)
        self.evidence['new_pin_refusals'] = 3
        self.evidence['changed_protected_helper_refusals'] = 9

    def test_actual_prelaunch_reaches_the_same_boundary_and_refuses_wrong_spec(self):
        results, refusals = [], []
        for label in self.packages:
            with self.controller(label) as (controller, path, digest, manifest, root):
                error, frames = self.binding.run_prelaunch(controller, path, digest, manifest, root)
                self.assertIsInstance(error, (FileNotFoundError, OSError, ValueError))
                last = [frame for frame in frames if frame.name == 'validate_manifest'][-1]
                self.assertIn('proof, _ = adapter.readout.read_json(runtime / proof_name', last.line)
                results.append((type(error).__name__, last.line))
            # The retained helper replaces OS-facing methods on its controller.
            # Give the negative case a fresh module rather than reusing those substitutes.
            with self.controller(label) as (controller, path, _, manifest, root):
                value = json.loads(path.read_bytes())
                value['spec_sha256'] = '0' * 64
                raw = canonical(value)
                path.write_bytes(raw)
                error, _ = self.binding.run_prelaunch(controller, path, sha(raw), manifest, root)
                self.assertIsInstance(error, ValueError)
                refusals.append(str(error))
        self.assertEqual(*results)
        self.assertEqual(*refusals)
        self.evidence['prelaunch_boundary'] = dict(exception=results[0][0], statement=results[0][1])
        self.evidence['wrong_spec_refusal'] = refusals[0]

    def test_bound_controller_reauthenticates_before_lease_and_retains_stamps(self):
        with self.controller('proposed') as (controller, path, digest, _, root):
            manifest = json.loads(path.read_bytes())
            grant, _ = self.grant(controller, manifest, digest, root, 'proposed')
            for valid in (True, False):
                prepared = dict(manifest=manifest, stamps={})
                lease_access = []
                class StopBeforeLease:
                    @property
                    def root(self):
                        lease_access.append(True)
                        raise RuntimeError('CPU boundary before lease work')
                effects = {key: Mock(side_effect=AssertionError('unexpected native/owner effect'))
                    for key in ('observer', 'read_inner_result', 'read_resource_receipt', 'resource_check', 'supervisor')}
                current = dict(grant)
                if not valid:
                    current['run_config_sha256'] = '0' * 64
                with patch.object(controller, 'validate_manifest', return_value=prepared):
                    result = controller._run_bound_controller(manifest_path=path, manifest_sha256=digest,
                        grant=current, physical_context=current, lease=StopBeforeLease(),
                        streams={arm: (None, None) for arm in ('bf16', 'kv4')}, clock=lambda: 101, **effects)
                self.assertEqual(result['status'], 'INCOMPLETE')
                if valid:
                    self.assertIn('CPU boundary before lease work', result['failure'])
                    self.assertIn(str(root / manifest['run_config']), prepared['stamps'])
                    self.assertEqual(lease_access, [True])
                else:
                    self.assertIn('digest mismatch', result['failure'])
                    self.assertFalse(lease_access)
                for effect in effects.values():
                    effect.assert_not_called()
        self.evidence['bound_controller_recheck'] = 'PASS_VALID_AND_INVALID_BINDINGS_BEFORE_LEASE'

    def test_physical_entry_preserves_retained_base_refusal_before_model_path(self):
        with self.controller('proposed') as (controller, path, _, _, root):
            manifest = json.loads(path.read_bytes())
            manifest['retained_base'] = None
            path.write_bytes(canonical(manifest))
            digest = sha(path.read_bytes())
            grant, parent = self.grant(controller, manifest, digest, root, 'proposed')
            # Synthetic CPU-only receipt: keep it fresh for the real entry's clock.
            grant.update(received_monotonic=controller.time.monotonic(),
                         received_utc=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
            grant_path = root / 'INERT-CPU-GRANT.json'
            grant_path.write_bytes(canonical(grant))
            rpc = Mock()
            rpc.authenticate_parent.return_value = parent
            with ExitStack() as stack:
                for name in ('getuid', 'geteuid', 'getgid', 'getegid'):
                    stack.enter_context(patch.object(controller.os, name, return_value=1000))
                after = stack.enter_context(patch.object(controller, '_run_physical_after_retained_closure',
                    side_effect=AssertionError('CPU test must not reach physical work')))
                with self.assertRaisesRegex(ValueError, 'NOT_READY.*base-rootfs'):
                    controller.run_physical_controller(manifest_path=path, manifest_sha256=digest,
                        grant_path=grant_path, grant_sha256=sha(grant_path.read_bytes()), rpc=rpc)
                after.assert_not_called()
        self.evidence['physical_entry_retained_base_gate'] = 'REFUSED_BEFORE_PHYSICAL_WORK'

    def test_source_builder_still_excludes_the_unapproved_proposal(self):
        output = self.root / 'source-review'
        builder.build(PACKET.resolve(strict=True), output, run_dir=HERE.parent / 'config/runs/w5-v2')
        for name, pin in self.metadata['protected_source_files'].items():
            self.assertEqual(sha((output / 'review' / name).read_bytes()), pin)
        self.assertFalse((output / 'MANIFEST.json').exists())
        self.evidence['builder_protected_set_unchanged'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    PACKET = options.packet
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(ConfigBindingPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
