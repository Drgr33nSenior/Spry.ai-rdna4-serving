"""Exercise the pruned W5 package on authenticated CPU copies, including post-run replay."""
import argparse
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_protected_packet as retained_checks
from check_protected_packet import isolated_modules
from test_contract_cleanup import load
from test_package_pruning_proposal import PROPOSAL, CONTROLLER, TRANSPORT, canonical, sha


class PackagePruningPacket(retained_checks.ProtectedPacket):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        for label, package in cls.packages.items():
            manifest = json.loads((package / 'MANIFEST.json').read_bytes())
            changes = {CONTROLLER: (HERE.parent / 'proposals/d1-controller-inventory' / CONTROLLER).read_bytes()}
            if label == 'proposed':
                changes = {name: (PROPOSAL / name).read_bytes() for name in cls.record['changes']}
                for name in cls.record['removed_package_files']:
                    (package / name).unlink()  # Only this test's freshly copied temporary package.
                    del manifest['package_files'][name]
            for name, raw in changes.items():
                (package / name).write_bytes(raw)
                manifest['package_files'][name] = sha(raw)
            (package / 'MANIFEST.json').write_bytes(canonical(manifest))
        cls.integrated = cls.root / 'integrated'
        shutil.copytree(cls.packages['proposed'], cls.integrated)
        manifest = json.loads((cls.integrated / 'MANIFEST.json').read_bytes())
        for name, change in cls.record['changes'].items():
            raw = (PROPOSAL / 'integration' / name).read_bytes()
            if sha(raw) != change['integrated_sha256']:
                raise AssertionError('composed proposal hash differs')
            (cls.integrated / name).write_bytes(raw)
            manifest['package_files'][name] = sha(raw)
        manifest['run_config'] = 'config/runs/w5-v2/run.json'
        (cls.integrated / 'MANIFEST.json').write_bytes(canonical(manifest))
        for name in ('config/runs/w5-v2/run.json', 'config/runs/w5-v2/menu.json', 'config/host/r9700-pair.json'):
            path = cls.integrated / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical(json.loads((HERE.parent / name).read_bytes())))
        (cls.integrated / 'PROTECTED-SHA256.json').write_bytes(canonical(
            {'review/' + name: pin for name, pin in cls.record['integrated_protected_files'].items()}))
        cls.evidence.update(proposal_sha256=sha((PROPOSAL / 'PROPOSAL.json').read_bytes()),
            patch_sha256=cls.record['D1_patch_sha256'],D2_patch_sha256=cls.record['D2_patch_sha256'],
            changed_source_pins=cls.record['changes'],removed_package_files=cls.record['removed_package_files'],
            inherited_check_source_sha256=sha((HERE / 'check_protected_packet.py').read_bytes()),
            whole_cleaned_source_archive_ready=False,owner_ceilings_changed=False)

    def test_selected_inventory_and_limits_are_identical(self):
        inventories = {}
        for label, package in self.packages.items():
            with self.controller(package) as (controller, _, _, manifest, _):
                self.assertEqual(controller.mtp_contract.settings()['d3b_draft_head_arm'], 'both')
                members = controller.live_package_names(manifest['experiment_mode'])
                self.assertEqual(members, set(manifest['package_files']))
                inventories[label] = members
                self.assertEqual(controller.SPEC_SHA256, manifest['spec_sha256'])
                if label == 'proposed':
                    self.assertEqual(manifest['package_files'], json.loads((PROPOSAL / 'PACKAGE-INVENTORY.json').read_bytes()))
        self.assertEqual(inventories['baseline'] - inventories['proposed'], set(self.record['removed_package_files']))
        self.assertEqual(len(inventories['baseline']), 79)
        self.assertEqual(len(inventories['proposed']), 74)
        self.evidence['inventory'] = dict(baseline=79, pruned=74, removed=self.record['removed_package_files'])

    def test_selected_runtime_overlays_and_both_sandbox_argv_are_identical(self):
        decisions = {}
        for label, package in self.packages.items():
            with self.controller(package) as (controller, _, _, manifest, root):
                spec = json.loads((root / 'review/BENCHMARK-SPEC2.json').read_bytes())
                target = Path(manifest['target_root'])
                decisions[label] = {}
                for arm, logical in (('bf16', 'control'), ('kv4', 'mtp')):
                    overlays = controller.mtp_contract.runtime_overlays(logical)
                    self.assertFalse({row['file'] for row in overlays} & {Path(n).name for n in self.record['removed_package_files']})
                    for row in overlays:
                        self.assertEqual(sha((root / 'review' / row['file']).read_bytes()), row['sha256'])
                    runner = [row for row in overlays if row['module'] == 'sglang.srt.model_executor.model_runner']
                    self.assertEqual([row['file'] for row in runner], ['d3b_combined_model_runner.py'])
                    argv = controller.adapter.bwrap_argv(arm, retained_root=Path(manifest['retained_root']),
                        composition=Path(manifest['runtime_root']), review=target / 'review', model=manifest['model_root'],
                        output=target / 'arms' / arm, entry=target / 'review/model_benchmark.py',
                        spec_sha256=manifest['spec_sha256'], bounded_scratch=True, physical_artifacts=True,
                        session_sha256=manifest['session_sha256'],
                        artifacts_sha256=manifest['package_files']['review/benchmark_artifacts.py'],
                        graph_contract=spec['graph_projection'], http_contract=spec['http_serving'])
                    argv = controller.mtp_contract.replay_serving_sandbox_argv(argv, manifest['package_files'])
                    self.assertEqual(list(argv), json.loads((root / 'review/RESOURCE-MANIFEST.json').read_bytes())['workload_argv'][arm])
                    decisions[label][arm] = dict(argv=list(argv), overlays=overlays)
        self.assertEqual(decisions['baseline'], decisions['proposed'])
        self.evidence['runtime_projection'] = dict(both_arms_identical=True, argv_comparisons=2,
            selected_runner='d3b_combined_model_runner.py', removed_sources_mounted=False)

    def test_all_remaining_members_keep_their_original_bytes_except_the_two_proposals(self):
        before = self.packages['baseline'] / 'review'
        after = self.packages['proposed'] / 'review'
        for path in after.iterdir():
            name = 'review/' + path.name
            if name not in self.record['changes']:
                self.assertEqual(path.read_bytes(), (before / path.name).read_bytes(), name)
        for name in ('capture_hybrid_linear_attn_backend.py', 'd3b_combined_model_runner.py', 'verify_decode_adapter.py'):
            self.assertTrue((after / name).is_file(), name)
        self.evidence['unchanged_remaining_members'] = len(list(after.iterdir())) - 2

    def test_pruned_configured_archive_stages_and_produces_an_accepted_controller_grant(self):
        package = self.integrated
        transport = load('cleanup_pruned_transport', package / TRANSPORT)
        members = {p.relative_to(package).as_posix(): p.read_bytes() for p in package.rglob('*') if p.is_file()}
        # Author package copies may carry non-payload bookkeeping; use the exact manifest inventory.
        manifest = json.loads(members['MANIFEST.json'])
        config_names = {'PROTECTED-SHA256.json', manifest['run_config'], 'config/runs/w5-v2/menu.json', 'config/host/r9700-pair.json'}
        members = {name: members[name] for name in set(manifest['package_files']) | {'MANIFEST.json'} | config_names}
        self.assertEqual(len(members), 79)
        pins = dict(run_config_sha256=sha(members[manifest['run_config']]),
            host_sha256=sha(members['config/host/r9700-pair.json']),
            protected_set_sha256=sha(members['PROTECTED-SHA256.json']))
        def archive(name, values):
            path = self.root / name
            with tarfile.open(path, 'w') as output:
                for filename, raw in values.items():
                    item = tarfile.TarInfo(filename)
                    item.size = len(raw)
                    output.addfile(item, io.BytesIO(raw))
            return path
        path = archive('pruned-configured.tar', members)
        checked, payload = transport.inspect_archive(path, sha(path.read_bytes()), sha(members['MANIFEST.json']),
            configuration_pins=pins)
        self.assertEqual(checked, manifest)
        staged = self.root / 'staged'
        staged.mkdir(); (staged / 'review').mkdir()
        fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            transport.stage_members(fd, payload)
        finally:
            os.close(fd)
        self.addCleanup(lambda: [p.chmod(0o755) for p in (staged / 'config').rglob('*') if p.is_dir()])
        self.addCleanup((staged / 'config').chmod, 0o755)
        self.assertEqual({p.relative_to(staged).as_posix(): p.read_bytes() for p in staged.rglob('*') if p.is_file()}, payload)
        with self.controller(staged) as (controller, mirror_path, digest, _, mirror):
            for name in config_names:
                target = mirror / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(members[name])
            value = json.loads(mirror_path.read_bytes())
            parent = dict(pid=100, uid=0, start_ticks=200, boot_id=value['boot_id'])
            grant = transport.new_grant(value, digest, received_utc='2026-01-01T00:00:00Z',
                received_monotonic=100, parent=parent, configuration_pins=pins)
            self.assertEqual(controller.validate_physical_grant(grant, value, digest, parent, clock=lambda: 101,
                utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1)), grant)
            error, frames = self.binding.run_prelaunch(controller, mirror_path, digest, value, mirror)
            self.assertIsInstance(error, FileNotFoundError)
            self.assertIn('proof, _ = adapter.readout.read_json(runtime / proof_name',
                [frame for frame in frames if frame.name == 'validate_manifest'][-1].line)
        overflow = dict(members, **{'review/extra-one.py': b'# inert\n', 'review/extra-two.py': b'# inert\n'})
        path = archive('pruned-overflow.tar', overflow)
        with self.assertRaisesRegex(ValueError, 'archive expanded size/member limit'):
            transport.inspect_archive(path, sha(path.read_bytes()), sha(members['MANIFEST.json']), configuration_pins=pins)
        self.evidence['configured_archive'] = dict(members=79, maximum=80, inspected=True, staged=True,
            actual_grant_accepted=True, same_prelaunch_boundary=True, overflow_refused=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    options, remaining = parser.parse_known_args()
    retained_checks.PACKET = options.packet
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if options.report is not None:
        report = dict(PackagePruningPacket.evidence, tests=result.testsRun, passed=result.wasSuccessful(),
                      test_source_sha256=sha(Path(__file__).read_bytes()))
        with options.report.open('xb') as stream:
            stream.write(canonical(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
