"""Render fresh manifest and caller proposals without creating launchable packets."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import functools

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as source_builder
import run_config
from launcher.run_native import checked_module
from spry_common.canon import canonical_bare, canonical_line as canonical, digest, need, parse_json


def install_shared_common(files, record):
    """Exchange the unused legacy graph integration for one pinned common source."""
    row = record['shared_common_source']
    legacy = type(row) is dict and set(row) == {'file', 'sha256', 'replaces', 'replaced_sha256'}
    protected = type(row) is dict and set(row) == {'file', 'sha256', 'replaces', 'replaced_sha256', 'protected'}
    need((legacy and row['file'] == 'spry_common/canon.py'
          or protected and row['file'] == 'spry_common/selected.py' and row['protected'] is True)
         and row['replaces'] == 'graph_integration.py',
         'exact shared common source exchange required')
    raw = source_builder.read_regular(HERE / row['file'], 4096)
    need(digest(raw) == row['sha256'] and digest(files[row['replaces']]) == row['replaced_sha256'],
         'shared common source or replaced graph integration differs')
    name = 'spry_common_canon.py'
    need(name not in files, 'common source role already exists')
    del files[row['replaces']]
    files[name] = raw
    spec = parse_json(files['BENCHMARK-SPEC2.json'])
    helpers = spec['graph_projection']['helpers']
    need(set(helpers) == {'graph_screen.py', 'graph_integration.py', 'graph_outer.py', 'context_tokenizer.py'},
         'exact inherited projection helper inventory required')
    del helpers[row['replaces']]
    helpers[name] = digest(raw)
    for arm in ('control', 'mtp'):
        binding_name = 'HTTP-BINDING-' + arm + '.json'
        value = parse_json(files[binding_name])
        sources = value['binding_template']['sources']
        need('spry_common_canon' not in sources, 'common manifest source already exists')
        sources['spry_common_canon'] = dict(module='spry_common_canon', path='/mnt/review/' + name,
                                            sha256=digest(raw))
        pin = digest(canonical_bare(sources))
        value['source_manifest_sha256'] = pin
        value['binding_template']['source_manifest_sha256'] = pin
        value['worker_template']['source_manifest_sha256'] = pin
        value['worker_template']['sources']['source_manifest_sha256'] = pin
        files[binding_name] = canonical(value)
        spec['http_serving']['bindings'][arm]['sha256'] = digest(files[binding_name])
    files['BENCHMARK-SPEC2.json'] = canonical(spec)


def protected_inventory(selected, record):
    """Bind the nine helper roles and any common code they execute."""
    result = {'review/' + name: digest(raw) for name, raw in selected.items()}
    common = record.get('shared_common_source', {})
    need(type(common) is dict, 'shared common source descriptor required')
    if 'protected' in common:
        need(common.get('protected') is True and common.get('file') == 'spry_common/selected.py',
             'exact protected common source role required')
        raw = source_builder.read_regular(HERE / common['file'], 4096)
        need(digest(raw) == common['sha256'], 'protected common source identity differs')
        result['review/spry_common_canon.py'] = digest(raw)
    return result


def retire_http_experiment_sources(files, record):
    """Remove only the three source-pinned experiment payloads outside HTTP."""
    retired = record['retired_http_sources']
    need(type(retired) is dict and set(retired) == {
        'graph_screen.py', 'context_tokenizer.py', 'validation_comparison.py'},
        'exact retired HTTP experiment source set required')
    need(all(name in files and digest(files[name]) == pin for name, pin in retired.items()),
         'retired HTTP source bytes differ')
    spec = parse_json(files['BENCHMARK-SPEC2.json'])
    need(spec.get('experiment_mode') == source_builder.MODE and
         not any(spec.get(key) is not None for key in (
             'independent_reference', 'validation_comparison', 'cache_profile', 'callback_profile', 'stall_trace')),
         'only the closed HTTP specification admits source retirement')
    helpers = spec['graph_projection']['helpers']
    need(set(helpers) == {'context_tokenizer.py', 'graph_screen.py', 'graph_outer.py', 'spry_common_canon.py'},
         'exact inherited HTTP projection helper set required')
    for name in ('context_tokenizer.py', 'graph_screen.py'):
        del helpers[name]
    for name in retired:
        del files[name]
    files['BENCHMARK-SPEC2.json'] = canonical(spec)


def protected_sources(proposal, donor, metadata):
    """Select only exact proposed helper bytes; never alter the canonical builder."""
    proposal = Path(proposal)
    need(proposal.is_absolute() and proposal.resolve(strict=True) == proposal and
         proposal.is_relative_to(HERE / 'proposals'), 'canonical local proposal required')
    record_raw = source_builder.read_regular(proposal / 'PROPOSAL.json', 64 << 10)
    record = parse_json(record_raw)
    expected = record.get('protected_files')
    need(type(expected) is dict and set(expected) == set(metadata['protected_source_files'])
         and record.get('owner_ceilings_changed') is False and record.get('launch_authorized') is False
         and record.get('boot_independent_manifest_schema') == 3,
         'exact nine-helper boot-independent source proposal required')
    need(digest(source_builder.read_regular(proposal / 'D2.patch', 1 << 20)) == record['patch_sha256'],
         'exact protected proposal patch required')
    references = record.get('protected_source_references', {})
    need(type(references) is dict and set(references) <= set(expected)
         and all(type(value) is str and value and not Path(value).is_absolute()
                 for value in references.values()), 'relative protected source references required')
    selected = {}
    for name, pin in expected.items():
        need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'protected helper digest required')
        path = proposal / 'review' / name
        if path.exists() or path.is_symlink():
            need(name not in references, 'protected source has both a copy and a reference')
            raw = source_builder.read_regular(path, 512 << 10)
        elif name in references:
            referenced = Path(os.path.abspath(proposal / references[name]))
            need(referenced.is_relative_to(HERE / 'proposals') and referenced.name == name,
                 'protected reference must name the local proposal source role')
            raw = source_builder.read_regular(referenced, 512 << 10)
        elif digest(donor[name]) == pin:
            raw = donor[name]
        else:
            need(name == 'benchmark_owner_adapter.py', 'missing proposed protected helper: ' + name)
            owner = (proposal / record['owner_adapter_proposal']).resolve(strict=True)
            need(owner.is_relative_to(HERE / 'proposals'), 'local owner proposal required')
            raw = source_builder.read_regular(owner, 512 << 10)
        need(digest(raw) == pin, 'proposed helper source differs: ' + name)
        selected[name] = raw
    protected = protected_inventory(selected, record)
    need(digest(canonical(protected)) == record['protected_set_sha256'], 'proposed protected set differs')
    return selected, record, digest(record_raw)


def install_post_load_source(files, record, metadata):
    """Add the pinned startup hook and bind it to both native workers."""
    need('retired_http_sources' in record, 'post-load hook requires the closed HTTP package')
    row = record['post_load_source']
    need(type(row) is dict and set(row) == {'file', 'sha256'}
         and row['file'] in ('spry_serving/post_load.py', 'spry_serving/memory_post_load.py'),
         'exact post-load source role required')
    raw = source_builder.read_regular(HERE / row['file'], 8192)
    need(digest(raw) == row['sha256'] and 'spry_post_load.py' not in files,
         'fresh authenticated post-load source required')
    files['spry_post_load.py'] = raw
    shared = record.get('shared_fp8_head_source')
    need((shared is not None) == (row['file'] == 'spry_serving/memory_post_load.py'),
         'memory post-load hook and shared head source must be selected together')
    if shared is not None:
        need(type(shared) is dict and set(shared) == {'file', 'sha256'}
             and shared['file'] == 'spry_serving/fp8_head.py', 'exact shared FP8 head source role required')
        helper = source_builder.read_regular(HERE / shared['file'], 16384)
        need(digest(helper) == shared['sha256'] and 'spry_fp8_head.py' not in files,
             'fresh authenticated shared FP8 head source required')
        files['spry_fp8_head.py'] = helper
    spec = parse_json(files['BENCHMARK-SPEC2.json'])
    for arm in ('control', 'mtp'):
        name = 'HTTP-BINDING-' + arm + '.json'
        value = parse_json(files[name])
        sources = value['binding_template']['sources']
        alias = value['worker_template']['sources']
        need('post_load' not in sources and set(alias) == {'same_as', 'source_manifest_sha256'}
             and alias['same_as'] == 'binding_template.sources', 'post-load source alias differs')
        runner = sources['model_runner']
        need(runner['path'] == '/mnt/composition/image-source/python/sglang/srt/model_executor/model_runner.py'
             and runner['sha256'] == metadata['source_files']['d3b_combined_model_runner.py'],
             'inherited runner source binding differs')
        runner['sha256'] = digest(files['d3b_combined_model_runner.py'])
        sources['post_load'] = dict(path='/mnt/review/spry_post_load.py', module='spry_post_load',
                                    sha256=row['sha256'])
        if shared is not None:
            need('shared_fp8_head' not in sources, 'shared FP8 head source already bound')
            sources['shared_fp8_head'] = dict(path='/mnt/review/spry_fp8_head.py', module='spry_fp8_head',
                                              sha256=shared['sha256'])
        pin = digest(canonical_bare(sources))
        value['source_manifest_sha256'] = pin
        for template in ('binding_template', 'worker_template'):
            value[template]['source_manifest_sha256'] = pin
        alias['source_manifest_sha256'] = pin
        files[name] = canonical(value)
        need(len(files[name]) < 65536, 'post-load HTTP template exceeds retained byte limit')
        spec['http_serving']['bindings'][arm]['sha256'] = digest(files[name])
    files['BENCHMARK-SPEC2.json'] = canonical(spec)


def build(donor_root, output, *, packet_id, proposal, run_dir=HERE / 'config/runs/w5-v2'):
    """Authenticate all inputs before publishing an inert, deterministic review tree."""
    output, donor_root, run_dir = Path(output), Path(donor_root), Path(run_dir)
    need(output.is_absolute() and output.resolve() == output and output.parent.is_dir() and not output.exists(),
         'fresh canonical proposal output required')
    need(not output.is_relative_to(HERE) and not output.is_relative_to(donor_root),
         'proposal output must be outside canonical source and donor trees')
    metadata = parse_json(source_builder.read_regular(HERE / 'SOURCE.json', 1 << 20))
    authenticated = source_builder.authenticated_packet(donor_root, metadata)
    donor = {name.removeprefix('author/package-v2/review/'): raw for name, raw in authenticated.items()
             if name.startswith('author/package-v2/review/')}
    seed = parse_json(authenticated['author/package-v2/MANIFEST.json'])
    selected, record, proposal_pin = protected_sources(proposal, donor, metadata)
    overrides={}
    for name,row in record.get('source_files',{}).items():
        need(name in source_builder.canonical_sources(HERE,metadata) and type(row) is dict
             and set(row)=={'file','sha256'},'exact existing non-protected source role required')
        path=(HERE/row['file']).resolve(strict=True)
        need(path.is_relative_to(HERE) and not path.is_relative_to(HERE/'legacy'),'canonical proposed source location required')
        raw=source_builder.read_regular(path,2<<20)
        need(digest(raw)==row['sha256'],'proposed non-protected source identity differs')
        overrides[name]=raw
    projector=None
    if record.get('configuration_mutations_supported') is True:
        need('mtp_contract.py' in overrides,'configured runtime contract source required')
        contract_path=HERE/record['source_files']['mtp_contract.py']['file']
        contract=checked_module(contract_path.parent,contract_path.stem,digest(overrides['mtp_contract.py']))
        projector=functools.partial(run_config.project_configuration,contract=contract)
    with tempfile.TemporaryDirectory(prefix='spry-packet-proposal-') as temporary:
        baseline = Path(temporary).resolve() / 'source'
        original = source_builder.build(donor_root, baseline, run_dir=run_dir,
            packet_id=packet_id, configured_artifacts=True,source_overrides=overrides,
            configuration_projector=projector)
        need(original['protected_sources_unchanged'] is True, 'canonical protected builder invariant required')
        files = {path.name: source_builder.read_regular(path) for path in (baseline / 'review').iterdir()}
        files.update(selected)
        if 'shared_common_source' in record:
            install_shared_common(files, record)
        if 'retired_http_sources' in record:
            retire_http_experiment_sources(files, record)
        if 'post_load_source' in record:
            install_post_load_source(files, record, metadata)
        policy=source_builder.production_evidence_policy(files)
        need(policy is None or record.get('production_evidence_mount_isolation') is True,
             'production evidence view requires matching protected consumers')
        # Rebuild source pins using the authenticated sandbox, then apply the
        # same fresh session identity. No Python or shell text is rewritten.
        source_builder.refresh_http_source_bindings(files)
        source_builder.render_resource_manifest(files, donor, seed)
        identity = source_builder.render_session_identity(files, original['configuration'], packet_id)
        runs = list((baseline / 'config/runs').glob('*/run.json'))
        need(len(runs) == 1, 'one authenticated staged run required')
        run_name = runs[0].relative_to(baseline).as_posix()
        run = parse_json(source_builder.read_regular(baseline / run_name))
        derived = parse_json(source_builder.read_regular(baseline / 'DERIVED-LIMITS.json'))
        package_files = {'review/' + name: digest(raw) for name, raw in sorted(files.items())}
        manifest = dict(seed, schema=3, scope='matched-benchmark-physical-package-v3',
            target_root=identity['target_root'], package_root=identity['target_root'],
            session_sha256=identity['session_sha256'], package_files=package_files,
            spec_sha256=package_files['review/BENCHMARK-SPEC2.json'],
            resource_manifest_sha256=package_files['review/RESOURCE-MANIFEST.json'],
            run_config=run_name, proposed_complete_seconds=derived['total_seconds'], launch_authorized=False)
        if projector is not None:
            manifest['logical_package_inventory']=original['configuration']['projection']['logical_inventory']
        manifest.pop('boot_id')
        package = dict(('package/review/' + name, raw) for name, raw in files.items())
        for path in (baseline / 'config').rglob('*'):
            if path.is_file():
                package['package/' + path.relative_to(baseline).as_posix()] = source_builder.read_regular(path)
        protected = canonical(protected_inventory(selected, record))
        package['package/PROTECTED-SHA256.json'] = protected
        package['package/MANIFEST-PROPOSAL.json'] = canonical(manifest)
        member_count = 77 if 'retired_http_sources' in record else 80
        if 'post_load_source' in record:
            member_count += 1
        if 'shared_fp8_head_source' in record:
            member_count += 1
        need(len(package) == member_count, 'exact proposed HTTP package membership required')
        # These dependencies are code/public connection metadata only. Their
        # private key path is retained; this renderer never opens that path.
        caller = {'caller/' + name: authenticated['launcher/caller-preview/' + name]
                  for name in ('connection.py', 'local_commands.py')}
        power = source_builder.read_regular(HERE / 'launcher/power_observe.py', 64 << 10)
        need(digest(power) == metadata['caller_power_source']['source_sha256'], 'canonical caller power source differs')
        caller['caller/power_observe.py'] = power
        caller['caller/' + run['host']['file']] = source_builder.read_regular(baseline / run['host']['file'])
        binding = dict(schema=2, scope='configured-caller-source-proposal-v1', nonce=packet_id,
            target_root=identity['target_root'], protected_root='/var/tmp/fp8-benchmark-source.' + packet_id,
            remote_upload_root='/home/developer/fp8-benchmark-upload.' + packet_id,
            host_config=run['host']['file'], host_sha256=digest(caller['caller/' + run['host']['file']]),
            run_config_sha256=digest(source_builder.read_regular(baseline / run_name)),
            protected_set_sha256=digest(protected), manifest_sha256=digest(canonical(manifest)),
            caller_seconds=derived['caller_seconds'], total_seconds=derived['total_seconds'],
            approval_boot_identity_required=True, launch_authorized=False, uploaded=False, executed=False)
        caller['caller/LAUNCHER-BINDINGS-TEMPLATE.json'] = canonical(binding)
        approval = dict(schema=2, scope='UNFUNDED_CONFIGURED_APPROVAL_TEMPLATE', boot_id=None,
            run_config_sha256=binding['run_config_sha256'], host_sha256=binding['host_sha256'],
            protected_set_sha256=binding['protected_set_sha256'], packet_id=packet_id,
            model_start_authorized=False, maximum_starts=0, retry_authorized=False, qualification=False)
        members = {**package, **caller, 'APPROVAL-TEMPLATE.json': canonical(approval)}
    pins = {name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in sorted(members.items())}
    report = dict(schema=1, status='PROPOSAL_REVIEW_ONLY', launch_authorized=False,
        renderer_sha256=digest(source_builder.read_regular(Path(__file__).resolve(), 128 << 10)),
        source_builder_sha256=original['inputs']['builder_sha256'],
        donor_seal_sha256=metadata['source_seal_sha256'], protected_proposal_sha256=proposal_pin,
        protected_set_sha256=record['protected_set_sha256'], session_identity=identity,
        source_builder_protected_sources_unchanged=True, proposed_package_members=len(package),
        caller_activation_complete=False, caller_dependencies_staged=True,
        production_evidence_view=None if policy is None else dict(files=list(policy['files']),
            mask_sha256=policy['sha256'],generated_on_native_execution=True),
        boot_identity_only_in_approval_template=True, files=pins,
        content_sha256=digest(canonical(pins)),
        remaining_gates=['complete_protected_diff_review', 'caller_approval_and_execution_binding',
            'native_grading_assembly', 'configuration_mutations', 'two_times_evidence_margins',
            'production_isolation', 'stock_runner_and_memory_cleanup', 'native_token_identity'])
    members['SOURCE-REVIEW.json'] = canonical(report)
    output.mkdir(mode=0o700)
    for name, raw in sorted(members.items()):
        path = output / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream: stream.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet-id', required=True)
    parser.add_argument('--proposal', type=Path, default=HERE / 'proposals/production-isolation')
    parser.add_argument('--run-dir', type=Path, default=HERE / 'config/runs/w5-v2')
    options = parser.parse_args()
    report = build(options.donor, options.output, packet_id=options.packet_id,
                   proposal=options.proposal, run_dir=options.run_dir)
    print(json.dumps(dict(status=report['status'], content_sha256=report['content_sha256']), sort_keys=True))


if __name__ == '__main__': main()
