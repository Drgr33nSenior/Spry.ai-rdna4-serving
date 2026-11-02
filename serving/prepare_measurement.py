"""Prepare a production cleanup comparison for review, without launch authority."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as source_builder
import propose_packet
from spry_common.canon import canonical_line as canonical, digest, need, parse_json
from spry_common.limits import derive


def execution_contract(raw, *, paired_cleanup=False):
    """Keep native sources and every execution setting; exclude private code pins."""
    value = copy.deepcopy(parse_json(raw))
    value.pop('source_manifest_sha256')
    binding, worker = value['binding_template'], value['worker_template']
    binding.pop('source_manifest_sha256')
    sources = {}
    for name, row in binding['sources'].items():
        if name in source_builder.EXTRACTED_HELPERS:
            continue
        if paired_cleanup and name == 'shared_fp8_head':
            need(row['path'] == '/mnt/review/spry_fp8_head.py' and row['module'] == 'spry_fp8_head',
                 'exact shared head comparison source required')
            continue
        private = row['path'].startswith('/mnt/review/')
        fixed = name.startswith(('mxfp4_', 'asset-')) and name != 'asset-ALLOCATION-OVERLAY.json'
        if paired_cleanup and name == 'mxfp4_scheme':
            need(row['path'] == '/mnt/review/isolated_scheme.py'
                 and row['module'] == 'mxfp4_component.isolated_scheme',
                 'exact packed scheme comparison source required')
            fixed = False
        if private and not fixed:
            row.pop('sha256')
        sources[name] = row
    binding['sources'] = sources
    worker.pop('source_manifest_sha256')
    worker['sources'].pop('source_manifest_sha256')
    worker['preweight']['graph_observer'].pop('sha256')
    return value


def compare_inputs(before, after, *, run, canonical_sources, protected_sources, paired_cleanup=False):
    """Refuse a mode, request, native overlay, model, stack or execution confound."""
    settings = parse_json(before['FINAL-SETTINGS.json'])
    need(settings.get('instrumentation') == run.get('instrumentation') == 'production',
         'both measurement variants must use explicit production mode')
    removed = set() if paired_cleanup else set(source_builder.RETIRED_W5_MEMBERS)
    added = {'spry_fp8_head.py'} if paired_cleanup else {name + '.py' for name in source_builder.EXTRACTED_HELPERS}
    need(set(before) - set(after) == removed and set(after) - set(before) == added,
         'exact reviewed cleanup membership changes required')
    allowed = set(canonical_sources) | set(protected_sources) | {
        'ALLOCATION-OVERLAY.json', 'BENCHMARK-SPEC2.json', 'RESOURCE-MANIFEST.json',
        'HTTP-BINDING-control.json', 'HTTP-BINDING-mtp.json'}
    if paired_cleanup:
        allowed.update(('spry_common_canon.py', 'spry_post_load.py'))
    fixed = (set(before) & set(after)) - allowed
    fixed.update(run['overlays'])
    fixed.update(('FINAL-SETTINGS.json', 'FINAL-REQUESTS.zlib', 'MODEL-INVENTORY.json',
                  'DRAFT-MODEL.json', 'PRECISION-REFERENCE.json', 'SERVING-STACK.json',
                  'COMMON-STACK-RECEIPT.json', 'COMPOSITION-FACTS.zlib'))
    if paired_cleanup:
        fixed.update(('kernels.py', 'GEMM-tail.py', 'ops.py', 'p1_fp8_kernels.py',
                      'd3b_head_fp8_kernels.py', 'd3b_head_fp8_ops.py', 'd3b_head_launches.py'))
    for name in sorted(fixed):
        need(name in before and name in after and before[name] == after[name],
             'measurement input differs: ' + name)
    for arm in ('control', 'mtp'):
        name = 'HTTP-BINDING-' + arm + '.json'
        need(canonical(execution_contract(before[name],paired_cleanup=paired_cleanup)) ==
             canonical(execution_contract(after[name],paired_cleanup=paired_cleanup)),
             'native execution contract differs: ' + arm)
    return dict(instrumentation='production', attention='retained-W5-stock',
                c4_included=False, fixed_files={name: digest(before[name]) for name in sorted(fixed)},
                native_execution_contracts_equal=True, output_token_identity_verified=False,
                comparison_scope='verified pre-memory source versus final cleanup' if paired_cleanup else 'retained donor versus cleanup')


def prepare(donor, output, *, packet_id, hyperloom_root=None):
    output, donor = Path(output), Path(donor)
    need(output.is_absolute() and output.resolve() == output and output.parent.is_dir()
         and not output.exists() and not output.is_symlink(), 'fresh canonical measurement output required')
    need(not output.is_relative_to(HERE) and not output.is_relative_to(donor),
         'measurement output must be outside source and donor trees')
    metadata = parse_json(source_builder.read_regular(HERE / 'SOURCE.json', 1 << 20))
    authenticated = source_builder.authenticated_packet(donor, metadata)
    seed = parse_json(authenticated['author/package-v2/MANIFEST.json'])
    before = {}
    for relative, pin in seed['package_files'].items():
        source_builder.relative_name(relative)
        need(Path(relative).parent == Path('review'), 'flat authenticated baseline required')
        raw = authenticated['author/package-v2/' + relative]
        need(digest(raw) == pin, 'baseline package pin differs')
        before[Path(relative).name] = raw
    with tempfile.TemporaryDirectory(prefix='spry-measurement-review-') as temporary:
        candidate = Path(temporary).resolve() / 'candidate'
        proposal = HERE / 'proposals/production-isolation'
        report = propose_packet.build(donor, candidate, packet_id=packet_id, proposal=proposal)
        review = candidate / 'package/review'
        after = {path.name: source_builder.read_regular(path) for path in review.iterdir()}
        source_report = parse_json(source_builder.read_regular(candidate / 'SOURCE-REVIEW.json'))
        run_raw = source_builder.read_regular(candidate / 'package/config/runs/w5-v2/run.json')
        run = parse_json(run_raw)
        canonical_names = (set(metadata['source_files']) | set(metadata['source_extractions'])
                           | {Path(name).name for name in metadata['new_sources']})
        comparison = compare_inputs(before, after, run=run, canonical_sources=canonical_names,
                                    protected_sources=metadata['protected_source_files'])
        menu = parse_json(source_builder.read_regular(candidate / 'package/config/runs/w5-v2/menu.json'))
        limits = derive(run['limits'])
        baseline_protected = {'review/' + name: digest(before[name]) for name in metadata['protected_source_files']}
        need(all(digest(before[name]) == pin for name,pin in metadata['protected_source_files'].items()),
             'original baseline protected set required')
        hyperloom = dict(role='optional offline reporting after native readout',
            measured_path_enabled=False, profiler_enabled=False, upstream_optimizer_enabled=False,
            owner_adapter_compatible=None, production_summary_import_verified=False)
        if hyperloom_root is not None:
            hyperloom_root = Path(hyperloom_root)
            paths = ('docs/HYPERLOOM.md', 'infrastructure/hyperloom/versions.lock',
                     'lib/workstation/hyperloom_owner.py', 'lib/workstation/hyperloom_sglang.py')
            evidence = {name: source_builder.read_regular(hyperloom_root / name, 128 << 10) for name in paths}
            owner = evidence['lib/workstation/hyperloom_owner.py'].decode()
            need('len(rows) in (4, 30)' in owner and 'owner_seconds == 5400' in owner,
                 'Hyperloom owner policy changed; review compatibility again')
            hyperloom.update(owner_adapter_compatible=False,
                owner_adapter_reason='Reviewed owner adapter admits only 4/30 rows and 5400 owner seconds',
                reviewed_sources={name: digest(raw) for name,raw in evidence.items()})
        slots = [('baseline', 'A'), ('cleanup', 'B')]
        plan = dict(schema=1, status='SOURCE_REVIEW_ONLY_NOT_LAUNCHABLE',
            purpose='Measure cleanup regression or improvement with matched production work',
            preparer_sha256=digest(source_builder.read_regular(Path(__file__).resolve(), 128 << 10)),
            donor_seal_sha256=metadata['source_seal_sha256'], donor_members=len(authenticated),
            baseline_manifest_sha256=digest(authenticated['author/package-v2/MANIFEST.json']),
            baseline_protected_set_sha256=digest(canonical(baseline_protected)),
            candidate_content_sha256=report['content_sha256'],
            candidate_review_sha256=digest(source_builder.read_regular(candidate / 'SOURCE-REVIEW.json')),
            candidate_protected_set_sha256=source_report['protected_set_sha256'],
            run_config_sha256=digest(run_raw),
            host_sha256=digest(source_builder.read_regular(candidate / 'package' / run['host']['file'])),
            matched_inputs=comparison,
            primary_metric='combined TP2 output tokens / HTTP group wall seconds',
            compare_groups='Match request group, task, concurrency, cache state and frozen ordinal',
            exclude='warmup and non-timing groups',
            work_gate=['full output token IDs equal', 'prompt IDs equal', 'speculative work equal',
                       'independent native quality and release checks pass'],
            pilot=[dict(variant=name, label=label, model_starts=2, requests_per_start=len(menu['rows']))
                   for name,label in slots],
            per_campaign_owner_seconds=limits['total_seconds'],
            per_campaign_caller_seconds=limits['caller_seconds'],
            pilot_model_starts=4, confirmation_order=['B', 'A'],
            confirmation='Separate review and fresh approvals after pilot; no automatic continuation',
            interpretation='A then B is a pilot; do not attribute a gain without reverse-order confirmation',
            hyperloom=hyperloom,
            launch_authorized=False, model_start_authorized=False, maximum_starts=0,
            baseline_historical_manifest_is_not_launch_input=True,
            remaining_gates=['exact complete protected approval', 'fresh baseline and candidate caller binding',
                'native CPU prelaunch and quality-packet integration', 'required evidence capacity margins',
                'fresh one-use approval for each campaign', 'native token identity and physical release'])
        output.mkdir(mode=0o700)
        baseline = output / 'baseline-source/review'; baseline.mkdir(parents=True, mode=0o700)
        for name, raw in sorted(before.items()):
            with (baseline / name).open('xb') as stream: stream.write(raw)
        with (baseline.parent / 'HISTORICAL-MANIFEST.json').open('xb') as stream:
            stream.write(authenticated['author/package-v2/MANIFEST.json'])
        shutil.copytree(candidate, output / 'candidate-review')
        with (output / 'MEASUREMENT-REVIEW.json').open('xb') as stream:
            stream.write(source_builder.canonical(plan))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--packet-id', required=True)
    parser.add_argument('--hyperloom-root', type=Path)
    options = parser.parse_args()
    plan = prepare(options.donor, options.output, packet_id=options.packet_id,
                   hyperloom_root=options.hyperloom_root)
    print(json.dumps(dict(status=plan['status'], matched_inputs=True, launch_authorized=False), sort_keys=True))


if __name__ == '__main__': main()
