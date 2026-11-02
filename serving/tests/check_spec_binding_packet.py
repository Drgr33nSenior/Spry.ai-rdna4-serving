"""Verify the configured spec through complete protected controller CPU copies."""
import argparse
from contextlib import contextmanager
import difflib
from functools import partial
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_cleaned_inventory_packet as inventory_checks
import check_protected_packet as protected_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_spec_binding_proposal import PROPOSAL, CONTROLLER, canonical, sha


class SpecBindingPacket(inventory_checks.CleanedInventoryPacket):
    evidence = {}
    test_real_prelaunch_keeps_frozen_spec_refusal = None
    test_oversized_artifact_member_refuses_before_output = protected_checks.ProtectedPacket.test_oversized_artifact_member_refuses_before_output

    @classmethod
    def setUpClass(cls):
        # Reuse the authenticated source assembly, unchanged transport and
        # exact 80-member archive fixture with this new controller proposal.
        with patch.object(inventory_checks, 'PROPOSAL', PROPOSAL):
            super().setUpClass()
        cls.packages = {'baseline': cls.root / 'fixture/author/package-v2', 'proposed': cls.package}
        original = cls.files['w5_evidence.py'].decode()
        call = 'controller.http_expected_binding(root, spec, owner_arm=arm, session_sha256=session)'
        if original.count('controller.SPEC_SHA256') != 5 or original.count(call) != 1:
            raise AssertionError('retained fixture API call sites changed')
        # The retained data producer used the retired controller constant. Its
        # adapted calls now hash the verified spec and pass the explicit API
        # argument. Controller source/attributes are never patched in post-run.
        adapted = original.replace('controller.SPEC_SHA256', 'controller.mapping_sha256(spec)')
        adapted = adapted.replace(call, call[:-1] + ', spec_sha256=controller.mapping_sha256(spec))')
        adapter_patch = ''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
            fromfile='a/w5_evidence.py', tofile='b/w5_evidence.py')).encode()
        path = Path(cls.postrun.__file__).with_name('w5_configured_evidence.py')
        path.write_text(adapted)
        cls.postrun = load('cleanup_spec_postrun', path)
        cls.evidence.update(fixture_adapter_sha256=sha(adapted.encode()),
            fixture_adapter_patch=adapter_patch.decode(), fixture_adapter_patch_sha256=sha(adapter_patch),
            inherited_checker_sha256=sha((HERE / 'check_cleaned_inventory_packet.py').read_bytes()),
            oversized_member_checker_sha256=sha((HERE / 'check_protected_packet.py').read_bytes()),
            test_only_spec_pin_substitution=False, supported_config_mutations=False)

    def prelaunch(self, controller, path, digest, manifest, root):
        # The retained pre-launch fixture predates the new independent pin
        # argument. Supply it while keeping its external-host substitutes.
        validate = partial(controller.validate_manifest, configuration_pins=self.pins)
        with patch.object(controller, 'validate_manifest', validate):
            return self.binding.run_prelaunch(controller, path, digest, manifest, root)

    def test_real_prelaunch_matches_retained_workstation_boundary(self):
        # Override the older inventory test: verified configured sources should
        # now reach the same missing external proof as the original W5 packet.
        results = []
        baseline = self.root / 'fixture/author/package-v2'
        with isolated_modules(), self.binding.controller_on_mirror(baseline) as value:
            controller, path, digest, manifest, root = value
            error, frames = self.binding.run_prelaunch(controller, path, digest, manifest, root)
            self.assertIsInstance(error, FileNotFoundError)
            line = [frame for frame in frames if frame.name == 'validate_manifest'][-1].line
            results.append((type(error).__name__, line))
        with self.controller() as (controller, path, digest, manifest, root):
            self.assertFalse(hasattr(controller, 'SPEC_SHA256'))
            verified = controller.verify_configured_spec(self.pins, json.loads(path.read_bytes()),
                                                         deadline=time.monotonic() + 105)
            self.assertEqual(canonical(verified['spec']), self.members['review/BENCHMARK-SPEC2.json'])
            error, frames = self.prelaunch(controller, path, digest, manifest, root)
            self.assertIsInstance(error, FileNotFoundError)
            line = [frame for frame in frames if frame.name == 'validate_manifest'][-1].line
            self.assertIn('proof, _ = adapter.readout.read_json(runtime / proof_name', line)
            results.append((type(error).__name__, line))
        self.assertEqual(*results)
        self.evidence['prelaunch'] = dict(retained_and_configured_boundary_identical=True,
            exception=results[0][0], source_statement=results[0][1],
            configuration_spec_reproduction=True, full_native_prelaunch_completed=False,
            substitutes=['external root/tree observations', 'retained runtime attestation', 'external draft model tree'])

    def test_repinned_wrong_spec_refuses_in_actual_prelaunch_and_before_lease(self):
        for mutation in ('fixed_template', 'configured_value', 'unknown_field'):
            with self.subTest(mutation=mutation), self.controller() as (controller, path, _, _, root):
                manifest = json.loads(path.read_bytes())
                spec_path = root / 'review/BENCHMARK-SPEC2.json'
                spec = json.loads(spec_path.read_bytes())
                if mutation == 'fixed_template':
                    spec['qualification'] = True
                elif mutation == 'configured_value':
                    spec['worker_seconds'] -= 1
                else:
                    spec['unexpected'] = True
                spec_path.write_bytes(canonical(spec))
                manifest['spec_sha256'] = sha(spec_path.read_bytes())
                manifest['package_files']['review/BENCHMARK-SPEC2.json'] = manifest['spec_sha256']
                path.write_bytes(canonical(manifest))
                error, frames = self.prelaunch(controller, path, sha(path.read_bytes()), manifest, root)
                self.assertIsInstance(error, ValueError)
                self.assertIn(str(error), ('fixed spec template differs', 'configuration-generated spec differs'))
                self.assertFalse(any('proof, _ = ' in (frame.line or '') for frame in frames))
                grant = dict(self.pins, schema=2, received_monotonic=time.monotonic())
                lease, effect = Mock(), Mock()
                report = controller._run_bound_controller(manifest_path=str(path), manifest_sha256=sha(path.read_bytes()),
                    grant=grant, physical_context=grant, lease=lease, observer=effect, read_inner_result=effect,
                    read_resource_receipt=effect, resource_check=effect, streams={}, supervisor=effect)
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertIn(str(error), report['failure'])
                self.assertEqual(lease.mock_calls, [])
                effect.assert_not_called()
        self.evidence['wrong_spec_refusals'] = dict(repinned_cases=3, before_external_proof=True, before_lease=True)

    def test_wrong_approval_and_reapproved_unsupported_config_refuse(self):
        for key in self.pins:
            with self.controller() as (controller, path, digest, manifest, root):
                validate = partial(controller.validate_manifest, configuration_pins=dict(self.pins, **{key: '0' * 64}))
                with patch.object(controller, 'validate_manifest', validate):
                    error, _ = self.binding.run_prelaunch(controller, path, digest, manifest, root)
                self.assertIsInstance(error, ValueError)
        with self.controller() as (controller, path, _, _, root):
            actual = json.loads(path.read_bytes())
            run_path = root / actual['run_config']
            run = json.loads(run_path.read_bytes())
            run['limits']['client_seconds'] -= 1
            run_path.write_bytes(canonical(run))
            pins = dict(self.pins, run_config_sha256=sha(run_path.read_bytes()))
            with self.assertRaisesRegex(ValueError, 'unsupported run configuration'):
                controller.verify_configured_spec(pins, actual, deadline=time.monotonic() + 105)
        self.evidence['configuration_refusals'] = dict(wrong_approval_pins=3, reapproved_unsupported_run=True)

    def check_postrun(self, shape):
        original_mirror = self.postrun.mirror

        @contextmanager
        def verified_mirror(package):
            with original_mirror(package) as (controller, manifest, root):
                for name in self.config_names:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(self.members[name])
                mirrored = dict(manifest, package_root=str(root))
                verified = controller.verify_configured_spec(self.pins, mirrored, deadline=time.monotonic() + 105)
                self.assertEqual(verified['spec_sha256'], manifest['spec_sha256'])
                self.assertFalse(hasattr(controller, 'SPEC_SHA256'))
                yield controller, manifest, root
                controller.verify_unchanged(verified)

        with isolated_modules(), patch.object(self.postrun, 'mirror', verified_mirror):
            measured, references, refusals = self.postrun.run(self.package, shape)
        margins = self.postrun.table(measured)
        self.assertEqual([row['row'] for row in margins if not row['passes']], [])
        decisions = {}
        for arm in ('bf16', 'kv4'):
            keys = ('status', 'wire_complete', 'request_count', 'response_count',
                    'cancellation_count', 'qualification', 'owner_arm')
            decisions[arm] = {key: references[arm][key] for key in keys}
            self.assertEqual(tuple(decisions[arm].values()),
                ('SERVING_SCREEN_COMPLETE', True, 176, 176, 0, False, arm))
            self.assertEqual(references[arm]['binding']['spec_sha256'], sha(self.members['review/BENCHMARK-SPEC2.json']))
            self.assertIn('natural-stop response menu', refusals[arm])
        self.evidence['postrun_' + shape] = dict(decisions=decisions, length_stop_refusals=refusals,
            margins=margins, test_only_spec_pin_substitution=False, configuration_verified_before_and_after=True,
            instrumentation='production', native_and_process_observations='retained CPU fixture substitutes')
        print(shape + ': configured spec verified; both 176-row arms accepted; length stops refused', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    inventory_checks.PACKET = options.packet
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(SpecBindingPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
