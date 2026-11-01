"""Session timing, receipt windows and configured sealing without native work."""
import ast
from contextlib import redirect_stdout
from dataclasses import asdict, FrozenInstanceError, replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_spec_binding_proposal as spec_tests
import test_owner_timing_proposal as owner_tests
import test_transport_binding_proposal as transport_tests
from test_contract_cleanup import load
from test_config_binding_proposal import canonical, sha
from test_owner_timing_proposal import derive

PROPOSAL = HERE.parent / 'proposals/d2-session-timing'
CONTROLLER = 'review/benchmark_physical_controller.py'
TRANSPORT = 'review/benchmark_host_transport.py'
transport = load('cleanup_session_transport', PROPOSAL / TRANSPORT)
RUN = json.loads((HERE.parent / 'config/runs/w5-v2/run.json').read_bytes())


def baseline_timing():
    return transport.SessionTiming.from_limits(RUN['limits'])


class SessionBounds(unittest.TestCase):
    def test_arithmetic_agrees_with_canonical_builder_for_baseline_and_shorter_work(self):
        for client in (2040, 1940, 1):
            limits = dict(RUN['limits'], client_seconds=client)
            expected = derive(limits)
            timing = transport.SessionTiming.from_limits(limits)
            for key in ('worker_seconds', 'arm_work_seconds', 'arm_envelope_seconds', 'active_seconds', 'total_seconds'):
                self.assertEqual(getattr(timing, key), expected[key])
            self.assertEqual(timing.deadlines(100), (100 + expected['active_seconds'], 100 + expected['total_seconds']))
        self.assertEqual(baseline_timing().owner_required_seconds, 5385)

    def test_invalid_primaries_derived_inputs_and_modified_plans_refuse(self):
        base = asdict(baseline_timing())
        for key in base:
            for value in (True, 0, -1, 1.0, None, 2**63):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    transport.SessionTiming(**dict(base, **{key: value}))
        for key, value in (('client_seconds', 2041), ('worker_startup_seconds', 239),
                           ('owner_cleanup_seconds', 149), ('arm_release_seconds', 59),
                           ('owner_cleanup_seconds', 151), ('arm_release_seconds', 61)):
            with self.assertRaises(ValueError):
                transport.SessionTiming(**dict(base, **{key: value}))
        for key in ('admission_seconds', 'source_preparation_seconds', 'locked_admission_seconds',
                    'collection_reserve_seconds', 'campaign_margin_seconds', 'seal_reserve_seconds'):
            for offset in (-1, 1):
                with self.assertRaises(ValueError):
                    transport.SessionTiming(**dict(base, **{key: base[key] + offset}))
        for key in ('active_seconds', 'total_seconds', 'worker_seconds', 'caller_seconds'):
            with self.assertRaises(ValueError):
                transport.SessionTiming.from_limits(dict(RUN['limits'], **{key: 1}))
        with self.assertRaises(FrozenInstanceError):
            baseline_timing().client_seconds = 1

    def test_original_receipt_window_never_restarts_after_preparation(self):
        timing = baseline_timing()
        for now in (100, 204.999):
            timing.require_start_window(100, clock=lambda: now)
            self.assertEqual(timing.deadlines(100), (6160, 6460))
        for now in (99, 205, 700, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                timing.require_start_window(100, clock=lambda: now)
        for received in (True, None, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                timing.deadlines(received)


class ConfiguredSession(spec_tests.SpecBinding):
    def setUp(self):
        with patch.object(spec_tests, 'PROPOSAL', PROPOSAL):
            super().setUp()
        self.controller.adapter.CampaignTiming = owner_tests.timing_module().CampaignTiming
        self.controller.transport = transport
        self.manifest['proposed_complete_seconds'] = 6360

    def test_session_and_owner_values_are_bound_to_the_same_verified_inputs(self):
        value = self.verify()
        self.assertEqual(value['session_timing'], baseline_timing())
        for key in asdict(value['owner_timing']):
            self.assertEqual(getattr(value['owner_timing'], key), getattr(value['session_timing'], key))
        for bad in (True, 6360.0, 6359, 6361):
            self.manifest['proposed_complete_seconds'] = bad
            with self.assertRaisesRegex(ValueError, 'manifest/session timing differs'):
                self.verify()


class TransportSession(transport_tests.TransportBinding):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(transport_tests, 'transport', transport).start()
        patch.object(transport_tests, 'PROPOSAL', PROPOSAL).start()
        super().setUp()

    def grant(self, pins=None, parent=None):
        return transport.new_grant(self.manifest, self.manifest_pin, received_utc='2026-01-01T00:00:00Z',
            received_monotonic=100, parent=self.parent if parent is None else parent,
            configuration_pins=self.pins if pins is None else pins, timing=baseline_timing())

    def test_archive_staging_and_grant_join_the_controller_approval(self):
        # Complete controller agreement is checked by the private packet suite.
        # The inherited public fixture contains inert adjacent protected files.
        manifest, members = self.inspect()
        self.assertEqual(members, self.members)
        root = self.stage()
        self.assertEqual({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}, members)
        grant = self.grant()
        self.assertEqual((grant['cumulative_ceiling_seconds'], grant['fixed_reserve_seconds']), (6360, 300))
        self.assertEqual(grant['received_monotonic'], 100)

    def test_shorter_transport_plan_binds_archive_and_grant_but_stale_total_refuses(self):
        run = json.loads(self.members[self.run_name])
        run['limits']['client_seconds'] = 1940
        self.members[self.run_name] = canonical(run)
        self.pins['run_config_sha256'] = sha(self.members[self.run_name])
        with self.assertRaisesRegex(ValueError, 'physical package contract'):
            self.inspect()
        self.manifest['proposed_complete_seconds'] = 6160
        self.refresh_manifest()
        self.inspect()
        timing = transport.SessionTiming.from_limits(run['limits'])
        grant = transport.new_grant(self.manifest, self.manifest_pin, received_utc='2026-01-01T00:00:00Z',
            received_monotonic=100, parent=self.parent, configuration_pins=self.pins, timing=timing)
        self.assertEqual((grant['cumulative_ceiling_seconds'], grant['fixed_reserve_seconds']), (6160, 300))
        self.assertEqual(timing.deadlines(grant['received_monotonic']), (5960, 6260))

    def test_grant_refuses_a_corrupted_plan_or_unbound_manifest(self):
        for kind in ('corrupted', 'manifest', 'type'):
            timing = baseline_timing()
            if kind == 'corrupted':
                object.__setattr__(timing, 'seal_reserve_seconds', 299)
            elif kind == 'manifest':
                self.manifest['proposed_complete_seconds'] = 6361
            else:
                timing = asdict(timing)
            with self.assertRaises(ValueError):
                transport.new_grant(self.manifest, self.manifest_pin, received_utc='2026-01-01T00:00:00Z',
                    received_monotonic=100, parent=self.parent, configuration_pins=self.pins, timing=timing)

    def test_bootstrap_failure_preserves_configured_deadline_and_original_receipt(self):
        args = ['--human-root-benchmark', '--archive', '/fixture/payload.tar', '--archive-sha256', 'a'*64,
                '--manifest-sha256', 'b'*64, '--expected-self-sha256', 'c'*64]
        args += [item for key, value in self.pins.items() for item in ('--'+key.replace('_','-'), value)]
        for configured in (False, True):
            def failed(options):
                if configured:
                    options.seal_deadline = options.original_receipt[0] + 6160
                raise ValueError('synthetic bootstrap failure')
            with patch.object(transport, 'human_main', failed), patch.object(transport, 'root_source'), \
                    patch.object(transport, 'protected_stage', return_value=self.root), \
                    patch.object(transport, 'exclusive'), patch.object(transport, 'seal', return_value={}) as seal, \
                    patch.object(transport.time, 'monotonic', return_value=100), redirect_stdout(io.StringIO()):
                self.assertEqual(transport.main(args), 1)
                self.assertEqual(seal.call_args.kwargs['deadline'], 6260 if configured else 6460)

    def test_expired_receipt_in_actual_human_entry_refuses_before_workload_creation(self):
        stage = self.root / 'root-stage'
        review = stage / 'review'
        review.mkdir(parents=True)
        for name in ('benchmark_host_transport.py', 'host_process_owner.py', 'benchmark_release_observer.py',
                     'collect_benchmark_platform.py', 'benchmark_artifacts.py'):
            (review / name).write_bytes(self.members['review/' + name])
        archive, pin = self.archive()
        payload = stage / 'payload.tar'
        payload.write_bytes(archive.read_bytes())
        args = SimpleNamespace(original_receipt=(100, '2026-01-01T00:00:00Z'), archive=str(payload),
            archive_sha256=pin, manifest_sha256=self.manifest_pin,
            expected_self_sha256=sha(self.members[TRANSPORT]), **self.pins)
        read_text = Path.read_text
        def read_boot(path, *args, **kwargs):
            return self.manifest['boot_id'] if str(path) == '/proc/sys/kernel/random/boot_id' else read_text(path, *args, **kwargs)
        with patch.object(transport, '__file__', str(review / 'benchmark_host_transport.py')), \
                patch.multiple(transport.os, getuid=lambda: 0, geteuid=lambda: 0, getgid=lambda: 0, getegid=lambda: 0), \
                patch.object(transport, 'protected_stage'), \
                patch.object(transport, 'root_source', side_effect=lambda path, pin: path.read_bytes()), \
                patch.object(Path, 'read_text', read_boot), patch.object(Path, 'mkdir') as mkdir, \
                patch.object(transport.time, 'monotonic', return_value=205), \
                patch.object(transport, 'stage_members') as stage_members, \
                patch.object(transport.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'receipt window expired'):
                transport.human_main(args)
            for effect in (mkdir, stage_members, popen):
                effect.assert_not_called()
        self.assertEqual(args.seal_deadline, 6460)


class ControllerSeal(unittest.TestCase):
    def setUp(self):
        with patch.object(spec_tests, 'PROPOSAL', PROPOSAL):
            self.controller = spec_tests.isolated_controller(Path('/tmp/fixture/controller.py'))
        self.controller.transport = transport
        tree = ast.parse((PROPOSAL / CONTROLLER).read_bytes())
        names = {'_relative', '_hash', '_tree', '_publish', 'seal_result'}
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(PROPOSAL / CONTROLLER), 'exec'), self.controller.__dict__)
        tmp = tempfile.TemporaryDirectory(prefix='cleanup-session-seal-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / 'result'
        self.root.mkdir()
        (self.root / 'evidence.json').write_bytes(b'{}\n')

    def test_actual_seal_uses_configured_total_and_preserves_exclusive_publication(self):
        timing = replace(baseline_timing(), client_seconds=1940)
        result = self.controller.seal_result(self.root, timing=timing, received_monotonic=100,
            received_utc='2026-01-01T00:00:00Z', clock=lambda: 6259.5)
        self.assertEqual(result['charged_seconds'], 6160)
        self.assertFalse(result['qualification'])
        self.assertTrue((self.root / 'FINAL-ARTIFACTS.sha256').is_file())
        with self.assertRaisesRegex(ValueError, 'cannot be amended'):
            self.controller.seal_result(self.root, timing=timing, received_monotonic=100,
                received_utc='2026-01-01T00:00:00Z', clock=lambda: 6259.5)

    def test_expired_or_future_receipt_and_bad_timing_refuse_before_writes(self):
        for now in (99, 6260, 6360):
            with self.assertRaises(ValueError):
                self.controller.seal_result(self.root, timing=replace(baseline_timing(), client_seconds=1940),
                    received_monotonic=100, received_utc='2026-01-01T00:00:00Z', clock=lambda: now)
        invalid = baseline_timing()
        object.__setattr__(invalid, 'campaign_margin_seconds', 1)
        with self.assertRaises(ValueError):
            self.controller.seal_result(self.root, timing=invalid, received_monotonic=100,
                received_utc='2026-01-01T00:00:00Z', clock=lambda: 101)
        self.assertEqual([p.name for p in self.root.iterdir()], ['evidence.json'])
        self.assertFalse(self.root.with_name('result.closeout.json').exists())


class ProposalIntegrity(unittest.TestCase):
    def test_exact_pins_and_unchanged_control_paths(self):
        record = json.loads((PROPOSAL / 'PROPOSAL.json').read_bytes())
        self.assertEqual(sha((PROPOSAL/'D2.patch').read_bytes()),record['patch_sha256'])
        for name, row in record['changes'].items():
            self.assertEqual(sha((PROPOSAL/name).read_bytes()),row['proposed_sha256'])
        self.assertEqual(sha(canonical({'review/'+k:v for k,v in record['protected_files'].items()})),
                         record['protected_set_sha256'])
        baseline = HERE.parent/'proposals/d1-package-pruning/integration'/TRANSPORT
        old, new = ast.parse(baseline.read_bytes()), ast.parse((PROPOSAL/TRANSPORT).read_bytes())
        changed = {'SessionTiming','inspect_archive','new_grant','human_main','main'}
        stable = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                               and n.name not in changed]
        self.assertEqual(stable(old),stable(new))
        human = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='human_main')
        self.assertFalse(any(isinstance(n,ast.Constant) and n.value==1995 for n in ast.walk(human)))
        calls = [n for n in ast.walk(human) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                 and n.func.attr=='require_start_window']
        self.assertEqual(len(calls),2)
        for key in ('owner_approved','launch_authorized','owner_ceilings_changed','consumed_by_builder','entire_D2_complete'):
            self.assertFalse(record[key])


if __name__ == '__main__':
    unittest.main()
