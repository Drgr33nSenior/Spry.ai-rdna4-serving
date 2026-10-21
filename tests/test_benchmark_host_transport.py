"""CPU transport fixtures: no root, host observer, model, SSH or GPU execution."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transport_under_test', ROOT / 'tools/benchmark_host_transport.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.manifest = {'schema': 2, 'scope': 'matched-benchmark-physical-package-v2',
            'target_root': '/home/developer/fp8-kv-benchmark.fixture123',
            'package_root': '/home/developer/fp8-kv-benchmark.fixture123',
            'session_sha256': 'd' * 64, 'boot_id': 'd819d208-9753-464e-b20a-a78b2fff9234',
            'launch_authorized': False, 'proposed_complete_seconds': 2400,
            'package_files': {'review/inert.py': transport.sha(b'raise RuntimeError("CPU fixture")\n')}}
        self.raw = transport.canonical(self.manifest)
        self.pin = transport.sha(self.raw)
        self.parent = {'pid': 12, 'uid': 0, 'start_ticks': 999, 'boot_id': self.manifest['boot_id']}

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, changes=None, duplicate=False, link=False):
        members = {'MANIFEST.json': self.raw, 'review/inert.py': b'raise RuntimeError("CPU fixture")\n'}
        members.update(changes or {})
        destination = self.base / 'payload.tar'
        with tarfile.open(destination, 'w') as archive:
            for name, raw in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
            if duplicate:
                info = tarfile.TarInfo('MANIFEST.json')
                archive.addfile(info, io.BytesIO())
            if link:
                info = tarfile.TarInfo('review/link.py')
                info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
                archive.addfile(info)
        return destination, transport.sha(destination.read_bytes())

    def test_archive_exact_members_and_hashes_without_extraction(self):
        path, pin = self.archive()
        value, members = transport.inspect_archive(path, pin, self.pin)
        self.assertEqual(value, self.manifest)
        self.assertEqual(set(members), {'MANIFEST.json', 'review/inert.py'})
        self.assertFalse((self.base / 'review').exists())

    def test_archive_rejects_bad_hash_extra_traversal_links_and_duplicates(self):
        for change, duplicate, link in (({'review/extra.py': b'x'}, False, False),
            ({'../escape': b'x'}, False, False), ({'review/inert.py': b'changed'}, False, False),
            ({}, True, False), ({}, False, True)):
            with self.subTest(change=change, duplicate=duplicate, link=link):
                path, pin = self.archive(change, duplicate, link)
                with self.assertRaises(ValueError):
                    transport.inspect_archive(path, pin, self.pin)
        path, pin = self.archive()
        for archive_pin, manifest_pin in (('a' * 64, self.pin), (pin, 'a' * 64)):
            with self.assertRaises(ValueError):
                transport.inspect_archive(path, archive_pin, manifest_pin)

    def test_grant_binds_original_receipt_parent_and_exact_inventory(self):
        grant = transport.new_grant(self.manifest, self.pin, received_utc='2026-10-15T00:00:00Z',
            received_monotonic=10.25, parent=self.parent)
        self.assertEqual(grant['received_monotonic'], 10.25)
        self.assertEqual(grant['maximum_starts'], 2)
        self.assertEqual(grant['arm_order'], ['bf16', 'kv4'])
        self.assertEqual(grant['cumulative_ceiling_seconds'], 2400)
        self.assertEqual(grant['fixed_reserve_seconds'], 300)
        self.assertEqual(grant['source_manifest_sha256'], transport.sha(transport.canonical(self.manifest['package_files'])))
        with self.assertRaises(ValueError):
            transport.new_grant(self.manifest, self.pin, received_utc='fixture',
                received_monotonic=10, parent={**self.parent, 'uid': 1000})

    def test_counterbalanced_mode_binds_fixed_ceiling_count_and_archive(self):
        for mode, seconds, processes in (('native-cache-regression-v1', 2400, 8),
                                        ('native-cache-counterbalanced-v2', 3000, 16)):
            value = dict(self.manifest, schema=3, scope='native-cache-regression-package-v1',
                         experiment_mode=mode, proposed_complete_seconds=seconds)
            raw = transport.canonical(value)
            path, pin = self.archive({'MANIFEST.json': raw})
            checked, _ = transport.inspect_archive(path, pin, transport.sha(raw))
            grant = transport.new_grant(checked, transport.sha(raw), received_utc='fixture',
                                        received_monotonic=10, parent=self.parent)
            self.assertEqual(grant['cumulative_ceiling_seconds'], seconds)
            self.assertEqual(grant['maximum_native_processes'], processes)
            self.assertEqual(grant['maximum_starts'], 0)
            self.assertEqual(grant['fixed_reserve_seconds'], 300)
            self.assertFalse(grant['model_start_authorized'])
            self.assertEqual(grant['experiment_mode'], mode)
            # No freely supplied duration, mode swap, or silent legacy expansion.
            for change in ({'proposed_complete_seconds': seconds + 1},
                           {'proposed_complete_seconds': True},
                           {'experiment_mode': 'unknown'}, {'scope': 'other'}):
                bad = dict(value, **change)
                bad_raw = transport.canonical(bad)
                path, pin = self.archive({'MANIFEST.json': bad_raw})
                with self.assertRaises(ValueError):
                    transport.inspect_archive(path, pin, transport.sha(bad_raw))
                with self.assertRaises(ValueError):
                    transport.new_grant(bad, transport.sha(bad_raw), received_utc='fixture',
                                        received_monotonic=10, parent=self.parent)

    def request(self, mode, arm=None):
        end = time.monotonic() + 1
        return {'schema': 1, 'operation': 'observe', 'expected_observer_sha256': 'c' * 64,
            'deadline_monotonic': end, 'request': {'root': self.manifest['target_root'],
                'session_sha256': self.manifest['session_sha256'], 'boot_id': self.manifest['boot_id'],
                'mode': mode, 'arm': arm, 'deadline_monotonic': end}}

    def test_bootstrap_refusal_uses_only_original_receipt_and_verified_ceiling(self):
        # Effect boundaries are CPU substitutes: no root, transport, or launch.
        # Archive validation and the actual filesystem sealer have separate tests.
        for ceiling in (2400, 3000):
            stage = self.base / ('refusal-' + str(ceiling))
            stage.mkdir()
            observed = []
            def fail(args):
                # A v2 ceiling is set by human_main only after inspect_archive
                # and transport-source binding; emulate that specific boundary.
                if ceiling == 3000:
                    args.verified_total_seconds = ceiling
                observed.append(args.original_receipt[0])
                raise ValueError('declared post-authentication refusal')
            with patch.object(transport, 'human_main', side_effect=fail), \
                 patch.object(transport, 'root_source', return_value=b'CPU source boundary'), \
                 patch.object(transport, 'protected_stage', return_value=stage), \
                 patch.object(transport, 'seal', return_value={'fixture': True}) as seal, \
                 patch('builtins.print'):
                code = transport.main(['--human-root-benchmark', '--archive', '/fixture/payload.tar',
                    '--archive-sha256', 'a' * 64, '--manifest-sha256', 'b' * 64,
                    '--expected-self-sha256', 'c' * 64])
            self.assertEqual(code, 1)
            self.assertEqual(seal.call_args.kwargs['deadline'], observed[0] + ceiling)
            failure = json.loads((stage / 'BOOTSTRAP-FAILURE.json').read_text())
            self.assertFalse(failure['retry_authorized'])
            self.assertFalse(failure['release_authority'])

    def test_phase_sequence_is_one_admission_two_ordered_releases_one_independent(self):
        sequence = transport.PhaseSequence(self.manifest, 'c' * 64, time.monotonic() + 30)
        for mode, arm in (('locked-admission', None), ('release', 'bf16'), ('release', 'kv4'), ('independent', None)):
            request = sequence.check(self.request(mode, arm))
            sequence.finish(request)
        self.assertTrue(sequence.closed)
        with self.assertRaises(ValueError):
            sequence.check(self.request('independent'))

    def test_phase_rejects_command_rpc_substitution_reorder_and_deadline_reset(self):
        for change in ({'operation': 'exec'}, {'expected_observer_sha256': 'a' * 64}, {'extra': True}):
            sequence = transport.PhaseSequence(self.manifest, 'c' * 64, time.monotonic() + 30)
            with self.assertRaises(ValueError):
                sequence.check({**self.request('locked-admission'), **change})
        sequence = transport.PhaseSequence(self.manifest, 'c' * 64, time.monotonic() + 30)
        with self.assertRaises(ValueError):
            sequence.check(self.request('release', 'kv4'))
        packet = self.request('locked-admission')
        packet['request']['session_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            sequence.check(packet)
        packet = self.request('locked-admission')
        sequence.active_end = time.monotonic() - 1
        with self.assertRaises(ValueError):
            sequence.check(packet)

    def test_real_anonymous_packet_roundtrip_and_duplicate_json_rejection(self):
        # macOS has the SOCK_SEQPACKET constant but not AF_UNIX implementation.
        # A real datagram pair exercises packet serialization only; Linux peer
        # authentication and the physical SOCK_SEQPACKET route remain unrun.
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            end = time.monotonic() + 1
            transport.send(left, {'fixture': True}, end)
            self.assertEqual(transport.receive(right, end), {'fixture': True})
            left.send(b'{"same":1,"same":2}')
            with self.assertRaises(ValueError):
                transport.receive(right, end)
            with self.assertRaises(ValueError):
                transport.send(left, {'oversized': 'a' * transport.PACKET_LIMIT}, end)
        finally:
            left.close()
            right.close()

    def test_rpc_parent_authenticates_pid_start_uid_boot_and_manifest(self):
        client = object.__new__(transport.RpcClient)
        client.channel = object()
        client.grant = {'transport_parent': self.parent, 'boot_id': self.manifest['boot_id'],
                        'manifest_sha256': self.pin}
        client.manifest_sha256 = self.pin
        with patch.object(transport, 'peer_identity', return_value={'pid': 12, 'uid': 0, 'gid': 0}), \
             patch.object(transport, 'process_start', return_value=999):
            self.assertEqual(client.authenticate_parent(), self.parent)
            client.manifest_sha256 = 'a' * 64
            with self.assertRaises(ValueError):
                client.authenticate_parent()
        client.manifest_sha256 = self.pin
        for identity in ({'pid': 13, 'uid': 0, 'gid': 0}, {'pid': 12, 'uid': 1000, 'gid': 1000}):
            with patch.object(transport, 'peer_identity', return_value=identity), \
                 patch.object(transport, 'process_start', return_value=999):
                with self.assertRaises(ValueError):
                    client.authenticate_parent()

    def test_resource_absence_checks_only_session_bound_units(self):
        with patch.object(Path, 'lstat', side_effect=FileNotFoundError):
            result = transport.resource_absence(['bf16', 'kv4'], 'd' * 64)
        self.assertEqual([row['arm'] for row in result['scopes']], ['bf16', 'kv4'])
        self.assertTrue(all(row['absent'] for row in result['scopes']))
        self.assertTrue(all(row['cgroup_path'].startswith(transport.USER_CGROUP + 'spry-fp8-benchmark-')
                            for row in result['scopes']))
        with patch.object(Path, 'lstat', return_value=object()):
            with self.assertRaises(ValueError):
                transport.resource_absence(['bf16'], 'd' * 64)
        with patch.object(Path, 'lstat', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                transport.resource_absence(['bf16'], 'd' * 64)

    def test_observer_failure_retains_bounded_output_before_refusing_snapshot(self):
        modules = {}
        for name in ('host_process_owner', 'benchmark_artifacts'):
            spec = importlib.util.spec_from_file_location('transport_fixture_' + name,
                ROOT / 'tools' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        core, artifacts = modules['host_process_owner'], modules['benchmark_artifacts']
        for index, (code, cleanup_errors) in enumerate(((1, ()), (0, ('uncertain cleanup',)), (0, ()))):
            with self.subTest(code=code, cleanup_errors=cleanup_errors):
                evidence = self.base / ('evidence-' + str(index))
                evidence.mkdir()
                folder = self.base / ('observer-' + str(index))
                result = core.SupervisionResult(None, (), code, 'child exited nonzero' if code else None,
                    cleanup_errors, (), True, None)
                def supervise(argv, **kwargs):
                    kwargs['stderr'].write(b'declared observer error context\n')
                    kwargs['stdout'].write(b'declared bounded observer output\n')
                    (folder / 'SNAPSHOT.json').write_bytes(transport.canonical({'status': 'OBSERVATION_COMPLETE'}))
                    return result
                def local_path(path):
                    # Only the root-private directory creation is substituted.
                    # Actual bounded capture pipes, exclusive writes and reads run.
                    return folder if str(path).startswith('/var/tmp/fp8-benchmark-observer.') else Path(path)
                request = {'mode': 'release', 'arm': 'bf16', 'deadline_monotonic': time.monotonic() + 75}
                with patch.object(transport, 'Path', side_effect=local_path):
                    options = dict(core=SimpleNamespace(Deadlines=core.Deadlines, supervise=supervise),
                        artifacts=artifacts, evidence_root=evidence)
                    if code or cleanup_errors:
                        with self.assertRaisesRegex(ValueError, 'root observer failed'):
                            transport.run_observer(request, ROOT / 'tools', 'a' * 64, **options)
                        self.assertFalse((evidence / 'release-bf16.json').exists())
                    else:
                        self.assertEqual(transport.run_observer(request, ROOT / 'tools', 'a' * 64, **options),
                                         {'status': 'OBSERVATION_COMPLETE'})
                        self.assertTrue((evidence / 'release-bf16.json').exists())
                self.assertEqual((evidence / 'release-bf16-stderr.raw').read_bytes(),
                                 b'declared observer error context\n')
                self.assertEqual((evidence / 'release-bf16-stdout.raw').read_bytes(),
                                 b'declared bounded observer output\n')
                receipt = transport.decode((evidence / 'release-bf16-owner.json').read_bytes())
                self.assertEqual(receipt['supervision']['exit_code'], code)
                self.assertEqual(receipt['supervision']['cleanup_errors'], list(cleanup_errors))
                self.assertTrue(receipt['capture']['reader_thread_absent'])

    def test_seal_checks_actual_files_links_hashes_and_root_timestamp(self):
        root = self.base / 'evidence'
        root.mkdir()
        (root / 'file').write_bytes(b'bounded fixture')
        result = transport.seal(root, deadline=time.monotonic() + 3)
        self.assertEqual(result['member_count'], 1)
        self.assertEqual(result['manifest_sha256'], transport.sha((root / 'FINAL-ARTIFACTS.sha256').read_bytes()))
        other = self.base / 'linked-evidence'
        other.mkdir()
        (other / 'link').symlink_to(root / 'file')
        with self.assertRaises(ValueError):
            transport.seal(other, deadline=time.monotonic() + 3)

    def test_nonroot_cannot_invoke_human_entry(self):
        with patch.object(os, 'getuid', return_value=1000):
            with self.assertRaises(ValueError):
                transport.human_main(object())

    def test_evidence_size_refusal_happens_before_creation(self):
        path = self.base / 'bounded-control'
        with self.assertRaises(ValueError):
            transport.exclusive(path, b'x' * 129, maximum=128)
        self.assertFalse(path.exists())
        transport.exclusive(path, b'x' * 128, maximum=128)
        self.assertEqual(path.stat().st_size, 128)

    def test_held_directory_write_refuses_symlink_parent_and_preserves_bytes(self):
        folder = self.base / 'held'
        folder.mkdir()
        (folder / 'real').mkdir()
        (folder / 'linked').symlink_to(folder / 'real')
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            transport.exclusive_beneath(fd, 'real/record.json', b'exact')
            self.assertEqual((folder / 'real/record.json').read_bytes(), b'exact')
            with self.assertRaises(OSError):
                transport.exclusive_beneath(fd, 'linked/forbidden', b'never')
            with self.assertRaises(ValueError):
                transport.exclusive_beneath(fd, '../escape', b'never')
            with self.assertRaises(FileExistsError):
                transport.exclusive_beneath(fd, 'real/record.json', b'replacement')
        finally:
            os.close(fd)


if __name__ == '__main__':
    unittest.main(verbosity=2)
