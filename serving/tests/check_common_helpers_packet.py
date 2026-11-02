"""Shared helper source exchange through real package, imports and readout."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_runtime_binding_packet as runtime
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical, digest

configured = runtime.configured
PROPOSAL = HERE.parent/'proposals/common-helpers'
configured.PROPOSAL = PROPOSAL


class CommonHelperPacket(configured.ConfigurationPacket):
    evidence = {}
    archive = runtime.RuntimeBindingPacket.archive
    test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins = runtime.RuntimeBindingPacket.test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins

    def pair(self, name):
        candidate, _, _ = self.package(name)
        output = self.root/(name+'-control')
        configured.propose_packet.build(configured.PACKET.resolve(), output, packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/startup-digests').resolve())
        parent = output/'package'
        (parent/'MANIFEST.json').write_bytes((parent/'MANIFEST-PROPOSAL.json').read_bytes())
        return parent, candidate

    def test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings(self):
        parent, candidate = self.pair('common-delta')
        data = lambda root: {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before, after = data(parent), data(candidate)
        self.assertEqual(set(before)-set(after), {'review/graph_integration.py'})
        self.assertEqual(set(after)-set(before), {'review/spry_common_canon.py'})
        changed = {name for name in set(before)&set(after) if before[name]!=after[name]}
        expected = {'review/mtp_contract.py','review/serving_contract.py','review/request_receipts.py',
            'review/serving_client.py','review/model_benchmark.py','review/graph_outer.py',
            'review/benchmark_physical_controller.py','review/BENCHMARK-SPEC2.json',
            'review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json',
            'review/RESOURCE-MANIFEST.json','PROTECTED-SHA256.json','MANIFEST.json','MANIFEST-PROPOSAL.json'}
        self.assertEqual(changed, expected)
        self.assertEqual(after['review/spry_common_canon.py'], (HERE.parent/'spry_common/canon.py').read_bytes())
        self.evidence['common_payload_delta'] = dict(changed=sorted(changed), removed=['review/graph_integration.py'],
            added=['review/spry_common_canon.py'], package_members=len(after), native_execution=False,
            requests_settings_model_stack_kernels_and_active_overlays_byte_identical=True)

    def test_all_three_actual_consumers_share_exact_json_functions_and_bytes(self):
        package, _, _ = self.package('common-consumers')
        review = package/'review'
        contract = load('common_packet_contract', review/'mtp_contract.py')
        with patch.dict(sys.modules, mtp_contract=contract):
            serving = load('common_packet_serving_contract', review/'serving_contract.py')
            receipts = load('common_packet_receipts', review/'request_receipts.py')
        self.assertIs(serving.canonical, contract.json_line)
        self.assertIs(receipts.canonical, contract.json_line)
        self.assertIs(receipts.sha, contract.digest_bytes)
        self.assertEqual(contract._SHARED_CANON.__file__, str(review/'spry_common_canon.py'))
        values = [None, True, 3, 1.25, 'é\n雪', {'z':[1,None], 'a':True}]
        for value in values:
            previous = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
            self.assertEqual(contract.json_line(value), previous)
            self.assertEqual(serving.canonical(value), previous)
            self.assertEqual(receipts.canonical(value), previous)
        self.assertEqual(len(serving.requests()), 176)
        self.evidence['shared_consumers'] = dict(consumers=3, callable_identity_shared=True,
            exact_previous_json_bytes=True, request_rows=176, helper_sha256=digest((review/'spry_common_canon.py').read_bytes()),
            native_execution=False)

    def test_packaged_common_source_change_or_removal_refuses_before_code_exec(self):
        package, _, _ = self.package('common-refusals')
        review = package/'review';helper=review/'spry_common_canon.py';raw=helper.read_bytes()
        helper.write_bytes(b'raise AssertionError("changed source executed")\n')
        with self.assertRaisesRegex(ValueError,'shared canonical helper source changed'):
            load('changed_common_packet_contract',review/'mtp_contract.py')
        helper.unlink()
        with self.assertRaises(FileNotFoundError): load('missing_common_packet_contract',review/'mtp_contract.py')
        helper.write_bytes(raw)
        self.evidence['shared_source_refusals'] = dict(changed=True, missing=True, before_execution=True)

    def test_native_workload_argv_keeps_every_mount_resource_and_model_argument(self):
        parent, candidate = self.pair('common-argv')
        before, after = [json.loads((p/'review/RESOURCE-MANIFEST.json').read_bytes()) for p in (parent,candidate)]
        old_files = {p.relative_to(parent).as_posix():p.read_bytes() for p in parent.rglob('*') if p.is_file()}
        pins = {digest(raw):digest((candidate/name).read_bytes()) for name,raw in old_files.items()
                if (candidate/name).is_file()}
        for arm, argv in before['workload_argv'].items():
            self.assertEqual([pins.get(item,item) for item in argv], after['workload_argv'][arm])
        for field in ('units','bwrap_sha256','root','session_sha256','experiment_mode'):
            self.assertEqual(before[field], after[field])
        self.evidence['native_argv_parity'] = dict(all_mount_resource_model_and_sampling_arguments_preserved=True,
            only_expected_source_pins_changed=True, native_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as output:
        output.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=digest(Path(__file__).read_bytes()),cases=CommonHelperPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
