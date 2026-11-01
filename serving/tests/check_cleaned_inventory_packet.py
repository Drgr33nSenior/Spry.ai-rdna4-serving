"""Check the complete cleaned W5 inventory on authenticated, fresh CPU copies."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_protected_packet import isolated_modules
from test_cleaned_inventory_proposal import BASELINE, PROPOSAL, CONTROLLER, TRANSPORT, canonical, sha
from test_contract_cleanup import load
from test_source_builder import builder, tree_bytes


class CleanedInventoryPacket(unittest.TestCase):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        cls.files = builder.authenticated_packet(PACKET.resolve(strict=True), cls.metadata)
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-inventory-')
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name).resolve()
        fixture = cls.root / 'fixture'
        for name, raw in cls.files.items():
            path = fixture / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        cls.binding = load('cleanup_inventory_binding', fixture / 'test_controller_binding.py')
        cls.postrun = load('cleanup_inventory_postrun', fixture / 'w5_evidence.py')
        cls.source = cls.root / 'source'
        cls.report = builder.build(PACKET.resolve(strict=True), cls.source,
                                   run_dir=HERE.parent / 'config/runs/w5-v2')
        cls.record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        cls.package = cls.root / 'integrated'
        shutil.copytree(cls.source, cls.package)
        for name, path in ((CONTROLLER, PROPOSAL / CONTROLLER),
                           (TRANSPORT, BASELINE / 'integration' / TRANSPORT)):
            raw = path.read_bytes()
            if sha(raw) != cls.record['protected_files'][Path(name).name]:
                raise AssertionError('proposed protected source pin differs')
            (cls.package / name).write_bytes(raw)
        protected = {'review/' + name: pin for name, pin in cls.record['protected_files'].items()}
        (cls.package / 'PROTECTED-SHA256.json').write_bytes(canonical(protected))
        manifest = json.loads(cls.files['author/package-v2/MANIFEST.json'])
        manifest['package_files'] = {'review/' + p.name: sha(p.read_bytes())
                                    for p in sorted((cls.package / 'review').iterdir())}
        manifest.update(run_config='config/runs/w5-v2/run.json',
            spec_sha256=manifest['package_files']['review/BENCHMARK-SPEC2.json'],
            resource_manifest_sha256=manifest['package_files']['review/RESOURCE-MANIFEST.json'])
        (cls.package / 'MANIFEST.json').write_bytes(canonical(manifest))
        cls.config_names = {'PROTECTED-SHA256.json', manifest['run_config'],
                            'config/runs/w5-v2/menu.json', 'config/host/r9700-pair.json'}
        cls.members = {name: (cls.package / name).read_bytes() for name in
                       set(manifest['package_files']) | cls.config_names | {'MANIFEST.json'}}
        cls.pins = dict(run_config_sha256=sha(cls.members[manifest['run_config']]),
            host_sha256=sha(cls.members['config/host/r9700-pair.json']),
            protected_set_sha256=sha(cls.members['PROTECTED-SHA256.json']))
        cls.evidence.update(source_seal_sha256=cls.metadata['source_seal_sha256'],
            proposal_sha256=sha((PROPOSAL / 'PROPOSAL.json').read_bytes()),
            controller_sha256=cls.record['proposed_sha256'], transport_sha256=sha(cls.members[TRANSPORT]),
            source_bundle_content_sha256=cls.report['content_sha256'],
            package_inventory=manifest['package_files'], protected_set_sha256=cls.pins['protected_set_sha256'],
            omitted_donor_members=cls.report['omitted_donor_members'],
            fixture_sha256={name: sha(cls.files[name]) for name in ('test_controller_binding.py', 'w5_evidence.py')},
            native_runtime_tested=False, full_prelaunch_completed=False,
            launch_authorized=False, owner_approved=False)

    @contextmanager
    def controller(self, package=None):
        with isolated_modules(), self.binding.controller_on_mirror(package or self.package) as value:
            controller, path, digest, manifest, root = value
            for name in self.config_names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.members[name])
            yield controller, path, digest, manifest, root

    def archive(self, name, members):
        path = self.root / name
        with tarfile.open(path, 'x') as output:
            for filename, raw in sorted(members.items()):
                item = tarfile.TarInfo(filename)
                item.size = len(raw)
                output.addfile(item, io.BytesIO(raw))
        return path

    def test_exact_inventory_preserves_protected_sources_and_active_overlays(self):
        for name, pin in self.metadata['protected_source_files'].items():
            self.assertEqual(sha((self.source / 'review' / name).read_bytes()), pin)
        self.assertFalse((self.source / 'MANIFEST.json').exists())
        with self.controller() as (controller, _, _, manifest, root):
            self.assertEqual(controller.live_package_names(manifest['experiment_mode']), set(manifest['package_files']))
            self.assertEqual(len(manifest['package_files']), 75)
            self.assertEqual(len(self.members), 80)
            self.assertFalse({'review/' + name for name in builder.RETIRED_W5_MEMBERS} & set(self.members))
            sources = controller.live_source_names(manifest['experiment_mode'])
            self.assertEqual(len(sources), len(set(sources)))
            self.assertEqual(len(sources), 46)
            for name in self.record['added_source_names']:
                self.assertIn(name, sources)
            for arm in ('control', 'mtp'):
                for row in controller.mtp_contract.runtime_overlays(arm):
                    self.assertNotIn(row['file'], builder.RETIRED_W5_MEMBERS)
                    raw = (root / 'review' / row['file']).read_bytes()
                    self.assertEqual(sha(raw), row['sha256'])
                    self.assertEqual(raw, self.files['author/package-v2/review/' + row['file']])
        self.evidence['inventory'] = dict(review_members=75, configured_archive_members=80,
            source_names=46, original_builder_protected_set=True, active_overlay_bytes_unchanged=True)

    def test_archive_stages_and_grant_authenticates_at_the_unchanged_80_member_ceiling(self):
        transport = load('cleanup_inventory_transport', self.package / TRANSPORT)
        path = self.archive('configured.tar', self.members)
        manifest, payload = transport.inspect_archive(path, sha(path.read_bytes()),
            sha(self.members['MANIFEST.json']), configuration_pins=self.pins)
        self.assertEqual(payload, self.members)
        self.assertEqual(transport._archive_member_limit(manifest, payload), 80)
        staged = self.root / 'staged'
        staged.mkdir(); (staged / 'review').mkdir()
        fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            transport.stage_members(fd, payload)
        finally:
            os.close(fd)
        self.addCleanup(lambda: [p.chmod(0o755) for p in (staged / 'config').rglob('*') if p.is_dir()])
        self.addCleanup((staged / 'config').chmod, 0o755)
        self.assertEqual(tree_bytes(staged), payload)
        with self.controller(staged) as (controller, mirror_path, digest, _, _):
            value = json.loads(mirror_path.read_bytes())
            parent = dict(pid=100, uid=0, start_ticks=200, boot_id=value['boot_id'])
            grant = transport.new_grant(value, digest, received_utc='2026-01-01T00:00:00Z',
                received_monotonic=100, parent=parent, configuration_pins=self.pins)
            self.assertEqual(controller.validate_physical_grant(grant, value, digest, parent, clock=lambda: 101,
                utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1)), grant)
        path = self.archive('overflow.tar', dict(self.members, **{'review/extra.py': b'# inert\n'}))
        with self.assertRaisesRegex(ValueError, 'archive expanded size/member limit'):
            transport.inspect_archive(path, sha(path.read_bytes()), sha(self.members['MANIFEST.json']),
                                      configuration_pins=self.pins)
        self.evidence['archive'] = dict(members=80, ceiling=80, inspected=True, staged=True,
            synthetic_grant_accepted=True, member_81_refused=True, owner_ceilings_changed=False)

    def test_real_prelaunch_keeps_frozen_spec_refusal(self):
        with self.controller() as (controller, path, digest, manifest, root):
            self.assertNotEqual(controller.SPEC_SHA256, manifest['spec_sha256'])
            error, _ = self.binding.run_prelaunch(controller, path, digest, manifest, root)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(str(error), 'frozen benchmark spec identity')
        self.evidence['prelaunch'] = dict(accepted=False, refusal=str(error),
            inventory_and_executed_source_checks_passed=True, config_spec_reproduction_pending=True)

    def test_missing_or_changed_extracted_source_refuses(self):
        for name in self.record['added_source_names']:
            members = dict(self.members)
            members['review/' + name] = b'raise AssertionError("untrusted code executed")\n'
            path = self.archive('changed-' + name + '.tar', members)
            transport = load('cleanup_inventory_bad_source', self.package / TRANSPORT)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'archive exact source inventory differs'):
                transport.inspect_archive(path, sha(path.read_bytes()), sha(self.members['MANIFEST.json']),
                                          configuration_pins=self.pins)
            del members['review/' + name]
            path = self.archive('missing-' + name + '.tar', members)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'archive exact source inventory differs'):
                transport.inspect_archive(path, sha(path.read_bytes()), sha(self.members['MANIFEST.json']),
                                          configuration_pins=self.pins)
        self.evidence['extracted_source_refusals'] = dict(changed=4, missing=4, before_staging=True)

    def check_postrun(self, shape):
        original_mirror = self.postrun.mirror

        @contextmanager
        def postrun_mirror(package):
            with original_mirror(package) as (controller, manifest, root):
                # Test seam only: isolate post-run consumers from the unfinished
                # config-to-spec verifier. The real pre-launch refusal is tested
                # above with unmodified source and constants. Never persist this.
                pin = sha((root / 'review/BENCHMARK-SPEC2.json').read_bytes())
                self.assertEqual(pin, manifest['spec_sha256'])
                self.assertNotEqual(controller.SPEC_SHA256, pin)
                with patch.object(controller, 'SPEC_SHA256', pin):
                    yield controller, manifest, root

        with isolated_modules(), patch.object(self.postrun, 'mirror', postrun_mirror):
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
            self.assertEqual(references[arm]['binding']['spec_sha256'],
                             sha(self.members['review/BENCHMARK-SPEC2.json']))
            self.assertIn('natural-stop response menu', refusals[arm])
        self.evidence['postrun_' + shape] = dict(decisions=decisions, length_stop_refusals=refusals,
            margins=margins, test_only_spec_pin_substitution=True, production_instrumentation=True,
            native_and_process_observations='retained CPU fixture substitutes')
        print(shape + ': cleaned producer/export/receive/readout accepted 176 rows per arm; length stops refused', flush=True)

    def test_expected_response_replay_through_cleaned_postrun(self):
        self.check_postrun('expected')

    def test_retained_envelope_through_cleaned_postrun(self):
        self.check_postrun('envelope')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    PACKET = options.packet
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(CleanedInventoryPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
