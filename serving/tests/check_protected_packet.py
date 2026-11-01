"""Compare the isolated D1 controller with sealed W5 helpers on fresh CPU copies."""
import argparse
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
from test_protected_proposal import PROPOSAL, restore_baseline, sha
from test_source_builder import builder


class NoNativeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'triton', 'sglang'):
            raise AssertionError('CPU check attempted a native import: ' + fullname)
        return None


@contextmanager
def isolated_modules():
    with patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)), \
            patch.object(sys, 'meta_path', [NoNativeImports(), *sys.meta_path]):
        if {'torch', 'triton', 'sglang'} & sys.modules.keys():
            raise AssertionError('native modules already loaded')
        yield


class ProtectedPacket(unittest.TestCase):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        cls.files = builder.authenticated_packet(PACKET.resolve(strict=True), cls.metadata)
        cls.proposal = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.raw = (PROPOSAL / cls.proposal['changed_file']).read_bytes()
        diff = (PROPOSAL / 'D1.patch').read_bytes()
        name = 'author/package-v2/' + cls.proposal['changed_file']
        if (restore_baseline(cls.raw, diff) != cls.files[name]
                or sha(cls.raw) != cls.proposal['proposed_sha256']
                or sha(diff) != cls.proposal['patch_sha256']):
            raise AssertionError('proposal does not match the authenticated baseline')
        cls.temporary = tempfile.TemporaryDirectory(prefix='cleanup-protected-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        fixture = cls.root / 'fixture'
        # Import only authenticated bytes from a new tree, including the retained
        # evidence producer's quality and response fixtures. Never import in place.
        for name, raw in cls.files.items():
            path = fixture / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        cls.binding = load('cleanup_retained_controller_binding', fixture / 'test_controller_binding.py')
        cls.postrun = load('cleanup_retained_postrun', fixture / 'w5_evidence.py')
        baseline = fixture / 'author/package-v2'
        proposed = cls.root / 'proposed'
        for name, raw in cls.files.items():
            if name.startswith('author/package-v2/'):
                path = proposed / name.removeprefix('author/package-v2/')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        (proposed / cls.proposal['changed_file']).write_bytes(cls.raw)
        manifest = json.loads((proposed / 'MANIFEST.json').read_bytes())
        manifest['package_files'][cls.proposal['changed_file']] = sha(cls.raw)
        (proposed / 'MANIFEST.json').write_bytes(builder.canonical(manifest))
        cls.packages = {'baseline': baseline, 'proposed': proposed}
        cls.evidence.update(proposal_sha256=sha(cls.raw), patch_sha256=sha(diff),
            source_seal_sha256=cls.metadata['source_seal_sha256'],
            retained_fixture_pins={name: sha(cls.files[name]) for name in
                ('test_controller_binding.py', 'w5_evidence.py')},
            native_runtime_tested=False, full_prelaunch_completed=False,
            launch_authorized=False, owner_approved=False)

    @contextmanager
    def controller(self, package):
        with isolated_modules(), self.binding.controller_on_mirror(package) as value:
            yield value

    def test_selected_inventory_and_limits_are_identical(self):
        inventories = []
        for package in self.packages.values():
            with self.controller(package) as (controller, _, _, manifest, _):
                mode = manifest['experiment_mode']
                self.assertEqual(controller.mtp_contract.settings()['phase'],
                                 self.proposal['selected_phase'])
                self.assertEqual(controller.live_package_names(mode), set(manifest['package_files']))
                inventories.append(dict(sources=list(controller.live_source_names(mode)),
                    package=sorted(controller.live_package_names(mode)),
                    spec=controller.SPEC_SHA256, model=controller.MODEL_INVENTORY_SHA256,
                    config=controller.CONFIG_SHA256))
        self.assertEqual(*inventories)
        self.evidence['inventory'] = dict(source_count=len(inventories[0]['sources']),
            package_count=len(inventories[0]['package']), identical=True)

    def test_actual_prelaunch_reaches_the_same_workstation_boundary(self):
        results = []
        for package in self.packages.values():
            with self.controller(package) as (controller, path, digest, manifest, root):
                error, frames = self.binding.run_prelaunch(controller, path, digest, manifest, root)
                self.assertIsNotNone(error, 'this CPU check expects no workstation runtime tree')
                last = [frame for frame in frames if frame.name == 'validate_manifest'][-1]
                self.assertIn('proof, _ = adapter.readout.read_json(runtime / proof_name', last.line)
                self.assertIsInstance(error, (FileNotFoundError, OSError, ValueError))
                results.append((type(error).__name__, last.line))
        self.assertEqual(*results)
        self.evidence['prelaunch_boundary'] = dict(exception=results[0][0],
            source_statement=results[0][1], identical=True,
            stubs=['external root resolution and trees', 'retained runtime attestation',
                   'external draft model tree (package half checked)'])

    def test_prelaunch_refusals_match(self):
        mutations = {
            'spec_pin': lambda value: value.update(spec_sha256='0' * 64),
            'time_ceiling': lambda value: value.update(proposed_complete_seconds=6361),
            'launch_flag': lambda value: value.update(launch_authorized=True),
            'helper_pin': lambda value: value['package_files'].update(
                {'review/benchmark_artifacts.py': '0' * 64}),
            'extra_source': lambda value: value['package_files'].update(
                {'review/unexpected.py': '0' * 64}),
        }
        decisions = {}
        for name, mutate in mutations.items():
            results = []
            for package in self.packages.values():
                with self.subTest(case=name, package=package.name), self.controller(package) as (
                        controller, path, _, manifest, root):
                    value = json.loads(path.read_bytes())
                    mutate(value)
                    raw = builder.canonical(value)
                    path.write_bytes(raw)
                    error, frames = self.binding.run_prelaunch(controller, path, sha(raw), manifest, root)
                    self.assertIsInstance(error, ValueError)
                    self.assertFalse(any('proof, _ = ' in (frame.line or '') for frame in frames))
                    results.append((type(error).__name__, str(error)))
            self.assertEqual(*results, name)
            decisions[name] = list(results[0])
        self.evidence['prelaunch_refusals'] = decisions

    def check_postrun(self, shape):
        results = {}
        for label, package in self.packages.items():
            with isolated_modules():
                measured, references, refusals = self.postrun.run(package, shape)
            rows = self.postrun.table(measured)
            self.assertEqual([row['row'] for row in rows if not row['passes']], [])
            decisions = {}
            for arm in ('bf16', 'kv4'):
                keys = ('status', 'wire_complete', 'request_count', 'response_count',
                        'cancellation_count', 'qualification', 'owner_arm')
                decisions[arm] = {key: references[arm][key] for key in keys}
                self.assertEqual(tuple(decisions[arm].values()),
                    ('SERVING_SCREEN_COMPLETE', True, 176, 176, 0, False, arm))
                self.assertIn('natural-stop response menu', refusals[arm])
            results[label] = dict(decisions=decisions, length_stop_refusals=refusals,
                                 margins=rows)
        self.assertEqual(results['baseline']['decisions'], results['proposed']['decisions'])
        self.assertEqual(results['baseline']['length_stop_refusals'],
                         results['proposed']['length_stop_refusals'])
        self.evidence['postrun_' + shape] = results
        print(shape + ': both controllers accepted 176 rows per arm; length stops refused', flush=True)

    def test_expected_response_replay_through_actual_postrun(self):
        self.check_postrun('expected')

    def test_retained_envelope_through_actual_postrun(self):
        self.check_postrun('envelope')

    def test_oversized_artifact_member_refuses_before_output(self):
        decisions = []
        for label, package in self.packages.items():
            with isolated_modules():
                artifacts = load('cleanup_artifacts_' + label, package / 'review/benchmark_artifacts.py')
                binding = dict(session_sha256='a' * 64, spec_sha256='b' * 64,
                               arm='bf16', experiment_mode=artifacts.HTTP_MODE)
                export = self.root / ('empty-' + label)
                export.mkdir(mode=0o700)
                sink = io.BytesIO()
                value = artifacts.export_bundle(export, binding, sink)
                value = copy.deepcopy(value)
                value['members']['result.json'] = dict(exists=True, bytes=(8 << 20) + 1,
                                                       sha256='0' * 64, base64='')
                raw = artifacts.PREFIX + artifacts._canonical(value) + b'\n'
                destination = self.root / ('refused-' + label)
                with self.assertRaisesRegex(ValueError, 'artifact shape/size/hash') as caught:
                    artifacts.receive_bundle(raw, b'', destination, binding)
                self.assertFalse(destination.exists())
                decisions.append(str(caught.exception))
        self.assertEqual(*decisions)
        self.evidence['oversized_member_refusal'] = decisions[0]

    def test_source_builder_still_emits_the_original_protected_set(self):
        output = self.root / 'source-review'
        builder.build(PACKET.resolve(strict=True), output,
                      run_dir=HERE.parent / 'config/runs/w5-v2')
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
        report = dict(ProtectedPacket.evidence, tests=result.testsRun,
                      passed=result.wasSuccessful(), test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(builder.canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
