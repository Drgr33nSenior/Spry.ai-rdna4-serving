"""Run source-bound W5 checks on native CPython without GPU or network access."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',required=True,type=Path)
    parser.add_argument('--donor',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--configuration-proposal',action='store_true')
    parser.add_argument('--metrics-reporter',type=Path)
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,14) or platform.machine()!='x86_64' or sys.platform!='linux':
        raise ValueError('native Linux x86_64 CPython 3.14 required')
    repository=args.repository.resolve(strict=True);donor=args.donor.resolve(strict=True)
    if args.configuration_proposal and args.metrics_reporter is None:
        raise ValueError('configured source preflight requires the pinned metrics reporter')
    denied=[]
    def guard(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            path=os.fsdecode(values[0])
            if path=='/dev/kfd' or path.startswith('/dev/dri/'):
                denied.append(event);raise PermissionError('CPU preflight denies GPU device access')
        if event=='import' and values[0].split('.')[0] in ('torch','triton','sglang','numpy','cupy'):
            denied.append(event);raise PermissionError('CPU preflight denies native serving imports')
        if event=='socket.connect':
            denied.append(event);raise PermissionError('CPU preflight denies network connections')
        if event=='subprocess.Popen' and Path(str(values[0])).name!='patch':
            denied.append(event);raise PermissionError('CPU preflight permits only the source patch tool')
    sys.addaudithook(guard)
    scripts=('tests/test_validation_preflight.py','tests/test_verify_metadata_reuse.py',
        'tests/test_context64_public_patch.py','tests/test_prepare_benchmark_package.py',
        'tests/test_platform_collector.py','serving/tests/test_evidence_capacity.py',
        'serving/tests/test_startup_configuration.py','serving/tests/test_host_binding.py',
        'serving/tests/test_source_fingerprints.py','serving/tests/test_configured_caller.py',
        'serving/tests/test_gpu_run_packaging.py','serving/tests/test_one_use_caller.py')
    if args.configuration_proposal:
        scripts+=('serving/tests/test_configuration_projection.py','serving/tests/test_chunked_receipts.py',
                  'serving/tests/test_quality_materialization.py','serving/tests/test_quality_policy.py')
    private='serving/tests/check_configuration_mutations_packet.py' if args.configuration_proposal else 'serving/tests/check_evidence_capacity_packet.py'
    cases=[];passed=True
    for number,relative in enumerate((*scripts,private)):
        path=repository/relative;previous=dict(sys.modules);old_path=list(sys.path)
        sys.path.insert(0,str(path.parent));sys.path.insert(0,str(repository/'serving'))
        try:
            spec=importlib.util.spec_from_file_location('native_cpu_preflight_'+str(number),path)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            if relative.endswith('check_configuration_mutations_packet.py'):
                module.PACKET=donor;module.METRICS_REPORTER=args.metrics_reporter
                names=['ConfigurationPacket.'+name for name in (
                    'test_baseline_and_scalar_mutations_reach_the_same_real_prelaunch_boundary',
                    'test_all_cap_and_doubled_menus_reach_the_real_prelaunch_boundary',
                    'test_owner_ceiling_invalid_policy_and_unaccounted_groups_refuse_before_output',
                    'test_new_source_review_is_deterministic','test_pinned_runtime_decode_logging_obeys_selected_interval')]
                suite=unittest.defaultTestLoader.loadTestsFromNames(names,module)
            elif relative.endswith('check_evidence_capacity_packet.py'):
                module.inventory_checks.PACKET=donor
                names=['EvidenceCapacityPacket.'+name for name in (
                    'test_authenticated_templates_pass_startup_checks_for_both_arms_and_ranks',
                    'test_actual_owner_mount_projection_matches_both_frozen_production_arms',
                    'test_workload_and_root_seals_charge_the_actual_archive_sources_and_both_closeouts',
                    'test_exact_diff_and_unmodified_release_validators')]
                suite=unittest.defaultTestLoader.loadTestsFromNames(names,module)
            else:suite=unittest.defaultTestLoader.loadTestsFromModule(module)
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            good=result.wasSuccessful() and not result.skipped
            cases.append(dict(source=relative,tests=result.testsRun,passed=good,skipped=len(result.skipped)))
            if not good:passed=False;break
        finally:
            sys.path[:]=old_path
            for name in list(sys.modules):
                if name not in previous:sys.modules.pop(name,None)
            sys.modules.update(previous)
    report=dict(status='PASS_NATIVE_CPU_SOURCE_BOUNDARY' if passed else 'FAILED_NATIVE_CPU_SOURCE_BOUNDARY',
        python=sys.version,architecture=platform.machine(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cases=cases,tests=sum(case['tests'] for case in cases),passed=passed,
        gpu_work=False,native_quality_grading=False,model_starts=0,network_denied=True,
        native_imports_denied=True,gpu_devices_denied=True,denied_events=denied,
        worker_profile='MXFP4 W5 startup API; separate frozen FP8 regression corpus')
    report['configuration_proposal']=args.configuration_proposal
    with args.report.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
