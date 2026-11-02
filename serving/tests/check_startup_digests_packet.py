"""Exercise the startup digest worker through the actual configured packet."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_runtime_binding_packet as runtime
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical, digest

configured = runtime.configured
PROPOSAL = HERE.parent / 'proposals/startup-digests'
configured.PROPOSAL = PROPOSAL


class StartupDigestPacket(runtime.RuntimeBindingPacket):
    evidence = {}

    def test_only_worker_and_derived_source_bindings_change_from_runtime_control(self):
        candidate, _, _ = self.package('startup-delta')
        parent = self.root / 'runtime-control'
        configured.propose_packet.build(configured.PACKET.resolve(), parent,
            packet_id='e201110200000010', proposal=(HERE.parent/'proposals/runtime-binding').resolve())
        baseline = parent / 'package'
        (baseline/'MANIFEST.json').write_bytes((baseline/'MANIFEST-PROPOSAL.json').read_bytes())
        data = lambda root: {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before, after = data(baseline), data(candidate)
        self.assertEqual(set(before), set(after))
        changed = {name for name in before if before[name] != after[name]}
        expected = {'review/worker_entry.py', 'review/BENCHMARK-SPEC2.json',
                    'review/HTTP-BINDING-control.json', 'review/HTTP-BINDING-mtp.json',
                    'review/RESOURCE-MANIFEST.json', 'MANIFEST.json', 'MANIFEST-PROPOSAL.json'}
        self.assertEqual(changed, expected)
        self.evidence['startup_payload_delta'] = dict(changed=sorted(changed),
            unchanged_files=len(before)-len(changed), protected_helpers_byte_identical=True,
            requests_settings_model_stack_kernels_and_overlays_byte_identical=True, native_execution=False)

    def test_packaged_worker_passes_all_source_digest_regressions(self):
        package, _, _ = self.package('startup-worker-tests')
        unit = load('packaged_startup_digest_tests', HERE/'test_startup_digests.py')
        unit.WORKER = package / 'review/worker_entry.py'
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(unit.StartupDigests))
        self.assertTrue(result.wasSuccessful(), output.getvalue())
        self.assertFalse(result.skipped)
        self.evidence['packaged_worker_regressions'] = dict(tests=result.testsRun, passed=True,
            source_sha256=digest(unit.WORKER.read_bytes()), native_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--metrics-reporter', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    options, remaining = parser.parse_known_args()
    configured.PACKET = options.packet; configured.METRICS_REPORTER = options.metrics_reporter
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    with options.report.open('xb') as output:
        output.write(canonical(dict(tests=result.testsRun, passed=result.wasSuccessful(),
            test_source_sha256=digest(Path(__file__).read_bytes()), cases=StartupDigestPacket.evidence,
            native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
