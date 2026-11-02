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
    proposal_group=parser.add_mutually_exclusive_group()
    proposal_group.add_argument('--configuration-proposal',action='store_true')
    proposal_group.add_argument('--runtime-binding-proposal',action='store_true')
    proposal_group.add_argument('--startup-digests-proposal',action='store_true')
    proposal_group.add_argument('--common-helpers-proposal',action='store_true')
    proposal_group.add_argument('--closed-http-proposal',action='store_true')
    proposal_group.add_argument('--post-load-proposal',action='store_true')
    proposal_group.add_argument('--source-references-proposal',action='store_true')
    proposal_group.add_argument('--pre-replay-proposal',action='store_true')
    proposal_group.add_argument('--shared-memory-proposal',action='store_true')
    proposal_group.add_argument('--phase-closure-proposal',action='store_true')
    proposal_group.add_argument('--common-closure-proposal',action='store_true')
    parser.add_argument('--metrics-reporter',type=Path)
    parser.add_argument('--source-access-inventory',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,14) or platform.machine()!='x86_64' or sys.platform!='linux':
        raise ValueError('native Linux x86_64 CPython 3.14 required')
    repository=args.repository.resolve(strict=True);donor=args.donor.resolve(strict=True)
    phase=args.phase_closure_proposal or args.common_closure_proposal
    memory=args.shared_memory_proposal or phase
    replay=args.pre_replay_proposal or memory
    source_references=args.source_references_proposal or replay
    post_load=args.post_load_proposal or source_references
    closed=args.closed_http_proposal or post_load
    common=args.common_helpers_proposal or closed
    startup=args.startup_digests_proposal or common
    runtime=args.runtime_binding_proposal or startup
    configured=args.configuration_proposal or runtime
    if configured and args.metrics_reporter is None:
        raise ValueError('configured source preflight requires the pinned metrics reporter')
    denied=[]
    repository_prefix=str(repository)+'/'
    opened_sources=set()
    source_inventory_complete=True
    def guard(event,values):
        nonlocal source_inventory_complete
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            path=os.fsdecode(values[0])
            if path=='/dev/kfd' or path.startswith('/dev/dri/'):
                denied.append(event);raise PermissionError('CPU preflight denies GPU device access')
            if args.source_access_inventory and path.startswith(repository_prefix):
                relative=path[len(repository_prefix):]
                if relative in opened_sources or len(opened_sources)<8192:
                    opened_sources.add(relative)
                else:
                    source_inventory_complete=False
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
    if configured:
        scripts+=('serving/tests/test_configuration_projection.py','serving/tests/test_chunked_receipts.py',
                  'serving/tests/test_quality_materialization.py','serving/tests/test_quality_policy.py')
    if runtime:scripts+=('serving/tests/test_runtime_binding.py',)
    if startup:scripts+=('serving/tests/test_startup_digests.py',)
    if common:scripts+=('serving/tests/test_common_helpers.py',)
    if closed:scripts+=('serving/tests/test_closed_http.py',)
    if post_load:scripts+=('serving/tests/test_post_load.py',)
    if source_references:scripts+=('serving/tests/test_protected_source_references.py',)
    if replay:scripts+=('serving/tests/test_pre_replay_metadata.py',)
    if memory:scripts+=('serving/tests/test_shared_memory.py',)
    if phase:scripts+=('serving/tests/test_phase_closure.py',)
    if args.common_closure_proposal:scripts+=('serving/tests/test_common_closure.py',)
    private=('serving/tests/check_common_closure_packet.py' if args.common_closure_proposal else
             'serving/tests/check_phase_closure_packet.py' if phase else
             'serving/tests/check_shared_memory_packet.py' if memory else
             'serving/tests/check_pre_replay_packet.py' if replay else
             'serving/tests/check_source_references_packet.py' if source_references else
             'serving/tests/check_post_load_packet.py' if post_load else
             'serving/tests/check_closed_http_packet.py' if closed else
             'serving/tests/check_common_helpers_packet.py' if common else
             'serving/tests/check_startup_digests_packet.py' if startup else
             'serving/tests/check_runtime_binding_packet.py' if runtime else
             'serving/tests/check_configuration_mutations_packet.py' if configured else 'serving/tests/check_evidence_capacity_packet.py')
    cases=[];passed=True
    for number,relative in enumerate((*scripts,private)):
        path=repository/relative;previous=dict(sys.modules);old_path=list(sys.path)
        sys.path.insert(0,str(path.parent));sys.path.insert(0,str(repository/'serving'))
        try:
            spec=importlib.util.spec_from_file_location('native_cpu_preflight_'+str(number),path)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            if relative.endswith(('check_configuration_mutations_packet.py','check_runtime_binding_packet.py',
                                  'check_startup_digests_packet.py','check_common_helpers_packet.py','check_closed_http_packet.py','check_post_load_packet.py','check_source_references_packet.py','check_pre_replay_packet.py','check_shared_memory_packet.py','check_phase_closure_packet.py','check_common_closure_packet.py')):
                context=module.configured if runtime else module
                context.PACKET=donor;context.METRICS_REPORTER=args.metrics_reporter
                prefix=('CommonClosurePacket.' if args.common_closure_proposal else
                        'PhaseClosurePacket.' if phase else
                        'SharedMemoryPacket.' if memory else
                        'PreReplayPacket.' if replay else
                        'SourceReferencePacket.' if source_references else
                        'PostLoadPacket.' if post_load else
                        'ClosedHttpPacket.' if closed else
                        'CommonHelperPacket.' if common else
                        'StartupDigestPacket.' if startup else
                        'RuntimeBindingPacket.' if runtime else 'ConfigurationPacket.')
                names=[prefix+name for name in (
                    'test_baseline_and_scalar_mutations_reach_the_same_real_prelaunch_boundary',
                    'test_all_cap_and_doubled_menus_reach_the_real_prelaunch_boundary',
                    'test_owner_ceiling_invalid_policy_and_unaccounted_groups_refuse_before_output',
                    'test_new_source_review_is_deterministic','test_pinned_runtime_decode_logging_obeys_selected_interval')]
                if runtime:
                    names += [prefix+'test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins']
                    names += [prefix+('test_runtime_payload_changes_only_shared_helpers_and_derived_source_bindings'
                        if common else 'test_runtime_payload_changes_only_reviewed_code_and_derived_source_bindings')]
                if args.startup_digests_proposal:
                    names += [prefix+name for name in (
                        'test_only_worker_and_derived_source_bindings_change_from_runtime_control',
                        'test_packaged_worker_passes_all_source_digest_regressions')]
                if common:
                    names += [prefix+name for name in (
                        'test_all_three_actual_consumers_share_exact_json_functions_and_bytes',
                        'test_packaged_common_source_change_or_removal_refuses_before_code_exec',
                        'test_native_workload_argv_keeps_every_mount_resource_and_model_argument')]
                if post_load:
                    names += [prefix+'test_packaged_hook_is_source_bound_in_both_workers_and_refuses_changed_bytes']
                if replay:
                    names += [prefix+'test_packaged_adapter_prepares_before_capture_and_strict_witness_refuses_old_counts']
                if memory:
                    names += [prefix+'test_packaged_memory_reader_refuses_incomplete_and_old_witnesses']
                if phase:
                    names += [prefix+'test_packaged_selected_contract_and_consumers_have_no_legacy_phase_dependency']
                if args.common_closure_proposal:
                    names += [prefix+name for name in (
                        'test_packaged_guard_semantics_and_frozen_json_pins_match_the_parent',
                        'test_self_consistent_substituted_common_inventory_refuses')]
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
    report['runtime_binding_proposal']=args.runtime_binding_proposal
    report['startup_digests_proposal']=args.startup_digests_proposal
    report['common_helpers_proposal']=args.common_helpers_proposal
    report['closed_http_proposal']=args.closed_http_proposal
    report['post_load_proposal']=args.post_load_proposal
    report['source_references_proposal']=args.source_references_proposal
    report['pre_replay_proposal']=args.pre_replay_proposal
    if args.source_access_inventory:
        report['repository_files_opened']=sorted(opened_sources)
        report['source_access_inventory_complete']=source_inventory_complete
    with args.report.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
