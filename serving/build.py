"""Assemble a deterministic W5 source review bundle; never emit a launch manifest."""
import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spry_common.canon import canonical_bare, canonical_line as canonical, digest, need, parse_json
import run_config
MODE = 'native-http-fp8-sprint01-v1'
EXTRACTED_HELPERS = ('observer_health', 'target_kv_metadata', 'kv_graph_join', 'request_receipts')
HTTP_IMPORTS = ('mtp_contract', 'serving_contract', 'serving_client', 'serving_parity',
                'worker_entry', 'serving_launcher', 'serving_entry', 'serving_runtime')
RETIRED_W5_MEMBERS = (
    'CAPTURE-BUILD-PROOF.json', 'STOCK-BUILD-PROOF.json', 'batch_telemetry.py',
    'cache_profile.py', 'capture_model_runner.py', 'divergence_capture.py',
    'prefill_row_audit.py', 'prepare_benchmark_package.py')


def relative_name(name):
    need(type(name) is str and name and '\\' not in name
         and not any(char.isspace() for char in name), 'unsafe member name')
    path = PurePosixPath(name)
    need(not path.is_absolute() and str(path) == name and '..' not in path.parts
         and name != '.', 'unsafe member path')
    return name


def read_regular(path, maximum=64 << 20):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'canonical input path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and 0 <= before.st_size <= maximum, 'bounded singly linked regular input required')
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            need(chunk, 'short input read')
            chunks.append(chunk)
            remaining -= len(chunk)
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                               value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        need(stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), 'input changed during read')
        return b''.join(chunks)
    finally:
        os.close(fd)


def authenticated_packet(root, metadata):
    """Verify the complete sealed tree without importing any donor code."""
    root = Path(root)
    need(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(),
         'canonical donor directory required')
    seal = root / 'FINAL-ARTIFACTS.sha256'
    raw = read_regular(seal, 1 << 20)
    need(digest(raw) == metadata['source_seal_sha256'], 'donor seal pin differs')
    expected = {}
    for line in raw.decode('ascii').splitlines():
        pin, name = line.split('  ', 1)
        relative_name(name)
        need(re.fullmatch('[0-9a-f]{64}', pin) and name not in expected
             and name != seal.name, 'invalid seal member')
        expected[name] = pin
    need(len(expected) == metadata['source_seal_members'], 'donor seal membership differs')
    seal_time = seal.stat().st_mtime_ns
    files = {}
    stamps = {}
    for path in (root, *sorted(root.rglob('*'))):
        info = path.lstat()
        stamps[path] = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)
        need(path.resolve(strict=True) == path and info.st_mtime_ns <= seal_time,
             'donor link or post-seal modification')
        if stat.S_ISDIR(info.st_mode):
            continue
        need(stat.S_ISREG(info.st_mode), 'nonregular donor member')
        if path == seal:
            continue
        name = path.relative_to(root).as_posix()
        need(name in expected, 'unsealed donor member: ' + name)
        content = read_regular(path)
        need(digest(content) == expected[name], 'donor member pin differs: ' + name)
        files[name] = content
    need(set(files) == set(expected), 'missing donor member')
    for path, before in stamps.items():
        info = path.lstat()
        need(before == (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns),
             'donor changed during authentication')
    return files


def canonical_sources(source_root, metadata):
    names = set(metadata['source_files']) | set(metadata['source_extractions'])
    for name in metadata['new_sources']:
        need(PurePosixPath(name).parent == PurePosixPath('legacy/review'), 'unexpected new source location')
        names.add(PurePosixPath(name).name)
    directory = source_root / 'legacy/review'
    need({path.name for path in directory.iterdir()} == names, 'canonical source membership differs')
    result = {}
    for name in sorted(names):
        need(relative_name(name) == PurePosixPath(name).name and name.endswith('.py'),
             'flat Python source required')
        result[name] = read_regular(directory / name, 2 << 20)
    return result


def production_evidence_policy(files):
    """Read literal policy from canonical source without executing that source."""
    if parse_json(files['FINAL-SETTINGS.json']).get('instrumentation', 'production') != 'production':
        return None
    wanted = {'PRODUCTION_EVIDENCE_FILES', 'PRODUCTION_EVIDENCE_MASK_NAME', 'PRODUCTION_EVIDENCE_MASK_BYTES'}
    values = {}
    for node in ast.parse(files['mtp_contract.py']).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                need(name not in values, 'unique production evidence policy required')
                values[name] = ast.literal_eval(node.value)
    need(set(values) == wanted, 'complete literal production evidence policy required')
    names, mask, raw = (values[name] for name in (
        'PRODUCTION_EVIDENCE_FILES', 'PRODUCTION_EVIDENCE_MASK_NAME', 'PRODUCTION_EVIDENCE_MASK_BYTES'))
    need(type(names) is tuple and len(names) == len(set(names)) == 6 and
         all(type(name) is str and re.fullmatch('[a-z][a-z0-9_]*\\.py', name) for name in names),
         'six explicit production evidence files required')
    need(type(mask) is str and relative_name(mask) == mask and PurePosixPath(mask).name == mask
         and mask not in names and type(raw) is bytes and 0 < len(raw) <= 256,
         'bounded production import-refusal source required')
    return dict(files=names, name=mask, raw=raw, sha256=digest(raw))


def render_data(files):
    """Refresh the existing JSON dependency order; do not rewrite Python source."""
    overlay = parse_json(files['ALLOCATION-OVERLAY.json'])
    for row in overlay['files']:
        row['sha256'] = digest(files[relative_name(row['file'])])
    files['ALLOCATION-OVERLAY.json'] = canonical(overlay)
    spec = parse_json(files['BENCHMARK-SPEC2.json'])
    need(spec['experiment_mode'] == MODE, 'W5 HTTP source build required')
    policy = production_evidence_policy(files)
    for arm in ('control', 'mtp'):
        name = 'HTTP-BINDING-' + arm + '.json'
        value = parse_json(files[name])
        need(value['mode'] == MODE and value['arm'] == arm, 'HTTP template selection differs')
        sources = value['binding_template']['sources']
        alias = value['worker_template']['sources']
        need(set(alias) == {'same_as', 'source_manifest_sha256'}
             and alias['same_as'] == 'binding_template.sources', 'worker source alias differs')
        for helper in EXTRACTED_HELPERS:
            need(helper not in sources, 'extracted helper role already present')
            sources[helper] = dict(path='/mnt/review/' + helper + '.py', module=helper,
                                   sha256=digest(files[helper + '.py']))
        for row in sources.values():
            if row['path'].startswith('/mnt/review/'):
                filename = relative_name(row['path'].removeprefix('/mnt/review/'))
                need(filename in files, 'private source missing: ' + filename)
                row['sha256'] = digest(files[filename])
                if policy is not None and filename in policy['files']:
                    row['sha256'] = policy['sha256']
        pin = digest(canonical_bare(sources))
        value['source_manifest_sha256'] = pin
        for template in ('binding_template', 'worker_template'):
            value[template]['source_manifest_sha256'] = pin
        alias['source_manifest_sha256'] = pin
        value['worker_template']['preweight']['graph_observer'] = dict(sources['graph_observer'])
        files[name] = canonical(value)
        need(len(files[name]) < 65536, 'HTTP template exceeds retained byte limit')
        spec['http_serving']['bindings'][arm]['sha256'] = digest(files[name])
    for name in spec['graph_projection']['helpers']:
        spec['graph_projection']['helpers'][name] = digest(files[relative_name(name)])
    files['BENCHMARK-SPEC2.json'] = canonical(spec)


def render_resource_manifest(files, donor, manifest):
    """Refresh W5 source bindings in the authenticated, fixed sandbox template."""
    value = parse_json(donor['RESOURCE-MANIFEST.json'])
    fields = {'schema', 'scope', 'root', 'session_sha256', 'benchmark_spec_sha256',
              'gate_path', 'gate_sha256', 'entry_path', 'entry_sha256', 'bwrap_sha256',
              'units', 'workload_argv', 'workload_argv_sha256', 'artifacts_sha256', 'experiment_mode'}
    need(type(value) is dict and set(value) == fields and type(value['schema']) is int
         and value['schema'] == 1 and value['scope'] == 'benchmark-resource-gate-v1'
         and value['experiment_mode'] == MODE, 'exact W5 resource template required')
    root, session = value['root'], value['session_sha256']
    need(type(root) is str and re.fullmatch(r'/home/developer/fp8-kv-benchmark\.[A-Za-z0-9]{8,64}', root)
         and root == manifest['target_root'] and session == manifest['session_sha256']
         and type(session) is str and re.fullmatch('[0-9a-f]{64}', session), 'resource root/session binding differs')
    need(value['entry_path'] == root + '/review/model_benchmark.py'
         and value['gate_path'] == root + '/review/benchmark_resource_gate.py'
         and type(value['bwrap_sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['bwrap_sha256']),
         'resource source path/bwrap binding differs')
    references = {'benchmark_spec_sha256': 'BENCHMARK-SPEC2.json',
                  'entry_sha256': 'model_benchmark.py', 'gate_sha256': 'benchmark_resource_gate.py',
                  'artifacts_sha256': 'benchmark_artifacts.py'}
    for key, name in references.items():
        need(value[key] == digest(donor[name]), 'resource donor source binding differs: ' + key)
    for key in ('units', 'workload_argv', 'workload_argv_sha256'):
        need(type(value[key]) is dict and set(value[key]) == {'bf16', 'kv4'}, 'resource exact two-arm mapping')
    for arm in ('bf16', 'kv4'):
        need(value['units'][arm] == 'spry-fp8-benchmark-' + session[:16] + '-' + arm + '.scope',
             'resource unit/session/arm binding differs')
        argv = value['workload_argv'][arm]
        need(type(argv) is list and 20 <= len(argv) <= 256
             and all(type(item) is str and item and len(item) <= 8192 and '\x00' not in item for item in argv)
             and argv[0] == '/usr/bin/bwrap'
             and digest(canonical_bare(argv)) == value['workload_argv_sha256'][arm],
             'resource bounded argv/hash differs')
        tail = ['/mnt/work/build-venv/bin/python3', '-I', '-B', '-X',
                'pycache_prefix=/mnt/capture-bytecode', '/mnt/review/model_benchmark.py',
                '--execute-reviewed-benchmark', '--spec', '/mnt/review/BENCHMARK-SPEC2.json',
                '--spec-sha256', value['benchmark_spec_sha256'], '--output', '/mnt/output/benchmark',
                '--arm', arm, '--physical-artifacts', '--session-sha256', session,
                '--artifacts-sha256', value['artifacts_sha256']]
        need(argv[-len(tail):] == tail and all(argv.count(flag) == 1 for flag in
             ('--execute-reviewed-benchmark', '--spec', '--spec-sha256', '--output', '--arm',
              '--physical-artifacts', '--session-sha256', '--artifacts-sha256')),
             'resource exact W5 invocation differs')
        mount = ['--ro-bind', value['entry_path'], '/mnt/review/model_benchmark.py']
        need(sum(argv[index:index + 3] == mount for index in range(len(argv) - 2)) == 1,
             'resource exact entry mount differs')
        # Only source pins change. Preserve every mount, isolation option,
        # environment variable, session identifier and bounded scratch size.
        argv[argv.index('--spec-sha256') + 1] = digest(files['BENCHMARK-SPEC2.json'])
        argv[argv.index('--artifacts-sha256') + 1] = digest(files['benchmark_artifacts.py'])
        value['workload_argv_sha256'][arm] = digest(canonical_bare(argv))
    for key, name in references.items():
        value[key] = digest(files[name])
    policy = production_evidence_policy(files)
    if policy is not None:
        source = root + '/' + policy['name']
        for arm, argv in value['workload_argv'].items():
            position = argv.index('/mnt/work/build-venv/bin/python3')
            masks = [item for name in policy['files']
                     for item in ('--ro-bind', source, '/mnt/review/' + name)]
            targets=['/mnt/review/'+name for name in policy['files']]
            if any(target in argv for target in targets):
                need(argv[position-len(masks):position] == masks
                     and all(argv.count(target)==1 for target in targets),
                     'one complete final production evidence mask set required')
            else:
                argv[position:position] = masks
            need(len(argv) <= 256, 'production evidence masks exceed existing argv bound')
            value['workload_argv_sha256'][arm] = digest(canonical_bare(argv))
    files['RESOURCE-MANIFEST.json'] = canonical(value)


def reproduce_overlays(files, upstream_root, source_root, overlay_order):
    """Apply pinned patches to copies of the retained effective base sources."""
    upstream_root = Path(upstream_root)
    need(upstream_root.is_absolute() and upstream_root.resolve(strict=True) == upstream_root
         and upstream_root.is_dir(), 'canonical upstream source directory required')
    lock_raw = read_regular(source_root / 'upstream/sglang-0.5.21.lock', 256 << 10)
    lock = parse_json(lock_raw)
    need(type(lock) is dict and set(lock) == {'schema', 'version', 'scope', 'complete', 'rows'}
         and type(lock['schema']) is int and lock['schema'] == 1 and lock['complete'] is True
         and lock['scope'] == 'active W5 runtime and graph/GEMM overlays'
         and type(lock['rows']) is list and len(lock['rows']) == 11, 'complete W5 overlay lock required')
    need(type(overlay_order) is list and [row.get('overlay') for row in lock['rows']
         if type(row) is dict] == overlay_order, 'patch series/config overlay order differs')
    names, sources, targets, patches = set(), set(), set(), set()
    rendered, pins = {}, {}
    with tempfile.TemporaryDirectory(prefix='spry-patch-build-') as temporary:
        scratch = Path(temporary).resolve()
        for row in lock['rows']:
            need(type(row) is dict and set(row) == {'source', 'target', 'original_sha256',
                 'overlay', 'overlay_sha256', 'patch', 'patch_sha256'}, 'exact overlay lock row required')
            source, target, patch_name = (relative_name(row[key]) for key in ('source', 'target', 'patch'))
            name = row['overlay']
            need(type(name) is str and relative_name(name) == PurePosixPath(name).name
                 and name in files and name not in names and str(source).startswith('python/sglang/')
                 and str(target) == 'image-source/' + str(source)
                 and source not in sources and target not in targets and patch_name not in patches
                 and str(patch_name).startswith('patches/sglang-0.5.21/'), 'overlay lock path or uniqueness differs')
            need(all(type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key]) for key in
                     ('original_sha256', 'overlay_sha256', 'patch_sha256')), 'overlay lock digest required')
            need(digest(files[name]) == row['overlay_sha256'], 'overlay output differs from authenticated package')
            original = read_regular(upstream_root / source, 2 << 20)
            patch_raw = read_regular(source_root / patch_name, 2 << 20)
            need(digest(original) == row['original_sha256'] and digest(patch_raw) == row['patch_sha256'],
                 'upstream or patch pin differs')
            headers = patch_raw.splitlines()
            need(len(headers) >= 3 and headers[0] == ('--- a/' + str(target)).encode()
                 and headers[1] == ('+++ b/' + str(target)).encode()
                 and not any(line.startswith((b'--- ', b'+++ ')) for line in headers[2:])
                 and all(line[:1] in (b'@', b' ', b'-', b'+', b'\\') for line in headers[2:]),
                 'exact single-target patch required')
            destination = scratch / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(original)
            result = subprocess.run(['/usr/bin/patch', '--batch', '--fuzz=0', '-p1'], cwd=scratch,
                input=patch_raw, capture_output=True, timeout=30)
            need(result.returncode == 0, 'overlay patch did not apply exactly: ' + name)
            candidate = read_regular(destination, 2 << 20)
            need(digest(candidate) == row['overlay_sha256'], 'patched overlay bytes differ: ' + name)
            rendered[name] = candidate
            pins[name] = {key: row[key] for key in ('original_sha256', 'overlay_sha256', 'patch_sha256')}
            names.add(name); sources.add(source); targets.add(target); patches.add(patch_name)
    files.update(rendered)
    return dict(lock_sha256=digest(lock_raw), overlays=pins, exact_output_bytes=True,
                effective_base='pinned retained image sources; not an unmodified upstream release')


def render_session_identity(files, configuration, packet_id):
    """Bind a fresh review session without granting launch authority."""
    need(type(packet_id) is str and re.fullmatch('[0-9a-f]{16}', packet_id),
         'packet ID must be exactly 16 lowercase hexadecimal characters')
    need(type(configuration) is dict and type(configuration.get('pins')) is dict
         and re.fullmatch('[0-9a-f]{64}', configuration['pins'].get('run', '')),
         'fresh identity requires authenticated run configuration')
    value = parse_json(files['RESOURCE-MANIFEST.json'])
    previous_root, previous_session = value['root'], value['session_sha256']
    root = '/home/developer/fp8-kv-benchmark.' + packet_id
    need(root != previous_root, 'historical packet identity cannot be reused')
    identity = dict(schema=1, scope='spry-source-review-session-v1', packet_id=packet_id,
                    run_config_sha256=configuration['pins']['run'])
    session = digest(canonical(identity))
    need(session != previous_session, 'historical session identity cannot be reused')
    for arm, argv in value['workload_argv'].items():
        for index, item in enumerate(argv):
            if previous_root in item:
                need(item == previous_root or item.startswith(previous_root + '/'),
                     'resource path must have a complete root boundary')
                relative = PurePosixPath(item).relative_to(previous_root)
                need('..' not in relative.parts, 'resource path cannot traverse its root')
                argv[index] = str(PurePosixPath(root) / relative)
        argv[argv.index('--session-sha256') + 1] = session
        value['units'][arm] = 'spry-fp8-benchmark-' + session[:16] + '-' + arm + '.scope'
        value['workload_argv_sha256'][arm] = digest(canonical_bare(argv))
    value.update(root=root, session_sha256=session,
                 entry_path=root + '/review/model_benchmark.py',
                 gate_path=root + '/review/benchmark_resource_gate.py')
    files['RESOURCE-MANIFEST.json'] = canonical(value)
    return dict(identity, target_root=root, session_sha256=session, launch_authorized=False)


def source_inventory(files, donor, protected):
    """Record static imports and references to old pins, including embedded text."""
    changed = {name: (digest(raw), digest(files[name])) for name, raw in donor.items()
               if name in files and raw != files[name]}
    old_pins = {}
    for name, (old, new) in changed.items():
        old_pins.setdefault(old, []).append((name, new))
    imports, references = {}, []
    for name, raw in sorted(files.items()):
        if not name.endswith('.py'):
            continue
        tree = ast.parse(raw, filename=name)
        private, external, standard = set(), set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                modules = []
            for module in modules:
                first = module.split('.')[0]
                if first + '.py' in files:
                    private.add(first + '.py')
                elif first in ('sglang', 'torch', 'triton', 'mxfp4_component', 'numpy'):
                    external.add(module)
                else:
                    standard.add(module)
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                value = node.value.decode('utf8', errors='replace') if isinstance(node.value, bytes) else node.value
                for pin in sorted(set(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', value))):
                    for target, replacement in old_pins.get(pin, []):
                        references.append(dict(source=name, line=node.lineno, target=target,
                            old_sha256=pin, current_sha256=replacement, protected=name in protected))
        imports[name] = dict(private=sorted(private), native_or_component=sorted(external),
                             other=sorted(standard))
    return dict(static_imports=imports, references_to_changed_members=references,
                dynamic_import_closure_verified=False)


def build(packet, output, *, source_root=HERE, run_dir=None, packet_id=None, upstream_root=None,
          configured_artifacts=False, source_overrides=None, configuration_projector=None):
    source_root = Path(source_root)
    need(source_root.is_absolute() and source_root.resolve(strict=True) == source_root,
         'canonical source root required')
    output = Path(output)
    need(output.is_absolute() and output.resolve() == output
         and output.parent.is_dir() and not output.exists(), 'fresh canonical output required')
    packet = Path(packet)
    need(not output.is_relative_to(packet) and not output.is_relative_to(source_root),
         'output must be outside donor and canonical source trees')
    metadata_raw = read_regular(source_root / 'SOURCE.json', 1 << 20)
    metadata = parse_json(metadata_raw)
    authenticated = authenticated_packet(packet, metadata)
    manifest_raw = authenticated['author/package-v2/MANIFEST.json']
    manifest = parse_json(manifest_raw)
    donor = {}
    for name, pin in manifest['package_files'].items():
        relative_name(name)
        need(PurePosixPath(name).parent == PurePosixPath('review'), 'flat review package required')
        raw = authenticated['author/package-v2/' + name]
        need(digest(raw) == pin, 'package manifest pin differs: ' + name)
        donor[PurePosixPath(name).name] = raw
    protected = metadata['protected_source_files']
    need(len(protected) == 9, 'exact nine protected source pins required')
    for name, pin in protected.items():
        need(digest(donor[name]) == pin, 'protected donor pin differs: ' + name)
    for name, pin in metadata['source_files'].items():
        need(digest(donor[name]) == pin, 'canonical source donor pin differs: ' + name)
    edited = canonical_sources(source_root, metadata)
    if source_overrides is not None:
        need(type(source_overrides) is dict and set(source_overrides) <= set(edited)
             and all(type(raw) is bytes and 0 < len(raw) <= 2 << 20 for raw in source_overrides.values()),
             'explicit existing non-protected source overrides required')
        edited.update(source_overrides)
    need(not set(edited) & set(protected), 'protected canonical override refused')
    files = {**donor, **edited}
    settings = parse_json(files['FINAL-SETTINGS.json'])
    selected_spec = parse_json(files['BENCHMARK-SPEC2.json'])
    need(settings.get('phase') == 'sprint01-paired-serving'
         and settings.get('d3b_draft_head_arm') == 'both'
         and selected_spec.get('experiment_mode') == MODE
         and not any(selected_spec.get(key) is not None for key in
                     ('validation_comparison', 'cache_profile')),
         'package pruning requires the retained W5 HTTP selection')
    need(not set(RETIRED_W5_MEMBERS) & (set(protected) | set(edited)),
         'cannot omit protected or canonical source')
    omitted = {name: digest(files.pop(name)) for name in RETIRED_W5_MEMBERS}
    need(all(name + '.py' in files for name in (*HTTP_IMPORTS, 'serving_reader')),
         'HTTP/readout source missing')
    need(type(configured_artifacts) is bool and (not configured_artifacts or run_dir is not None),
         'configured artifact projection requires a run directory')
    configuration = run_config.load(Path(run_dir), source_root, metadata, files, read_regular,
        configured_artifacts=configured_artifacts, projector=configuration_projector) if run_dir is not None else None
    if upstream_root is not None:
        need(configuration is not None, 'patch reproduction requires authenticated run configuration')
    overlay_series = reproduce_overlays(files, upstream_root, source_root,
        configuration['run']['overlays']) if upstream_root is not None else None
    render_data(files)
    render_resource_manifest(files, donor, manifest)
    identity = render_session_identity(files, configuration, packet_id) if packet_id is not None else None
    for name, pin in protected.items():
        need(digest(files[name]) == pin, 'protected source changed')
    inventory = source_inventory(files, donor, protected)
    payload = {'review/' + name: raw for name, raw in files.items()}
    payload['PROTECTED-SHA256.json'] = canonical({'review/' + name: pin for name, pin in protected.items()})
    if configuration is not None:
        run = configuration['run']
        prefix = 'config/runs/' + run['run_id'] + '/'
        payload[prefix + 'run.json'] = canonical(run)
        payload[prefix + 'menu.json'] = canonical(configuration['menu'])
        payload[run['host']['file']] = canonical(configuration['host'])
        payload['DERIVED-LIMITS.json'] = canonical(configuration['derived_limits'])
    pins = {name: dict(sha256=digest(raw), bytes=len(raw)) for name, raw in sorted(payload.items())}
    report = dict(schema=1, status='SOURCE_REVIEW_ONLY', launch_authorized=False,
        inputs=dict(source_metadata_sha256=digest(metadata_raw), donor_seal_sha256=metadata['source_seal_sha256'],
                    donor_manifest_sha256=digest(manifest_raw), builder_sha256=digest(read_regular(Path(__file__).resolve())),
                    implementation_sha256={name: digest(read_regular(HERE / name)) for name in
                        ('run_config.py', 'spry_common/__init__.py', 'spry_common/canon.py', 'spry_common/limits.py')}),
        files=pins, content_sha256=digest(canonical_bare(pins)),
        canonical_sources={name: digest(raw) for name, raw in sorted(edited.items())},
        omitted_donor_members=omitted,
        generated_review_files=['ALLOCATION-OVERLAY.json', 'BENCHMARK-SPEC2.json',
                                'HTTP-BINDING-control.json', 'HTTP-BINDING-mtp.json', 'RESOURCE-MANIFEST.json'],
        resource_manifest=dict(rendering='RETAINED_W5_SANDBOX_TEMPLATE_WITH_REFRESHED_SOURCE_BINDINGS',
            new_packet_identity_generated=False, runtime_config_mutations_supported=False),
        protected_sources_unchanged=True, dependency_inventory=inventory,
        remaining_gates=['source_literal_pin_review', 'runtime_config_and_fresh_packet_identity',
                         'package_membership_and_caps', 'protected_controller_config_migration',
                         'native_cpu_evidence', 'token_identity'])
    if identity is not None:
        report['session_identity'] = identity
        report['resource_manifest'].update(
            rendering='CONFIGURED_SESSION_WITH_RETAINED_SANDBOX_TEMPLATE', new_packet_identity_generated=True)
        report['remaining_gates'].remove('runtime_config_and_fresh_packet_identity')
        report['remaining_gates'].extend(('runtime_config_consumers', 'fresh_launch_manifest_and_caller'))
    if overlay_series is not None:
        report['overlay_series'] = overlay_series
    if configured_artifacts:
        report['configured_artifacts'] = dict(spec_projection=True, protected_consumer_proposal_required=True)
    # No donor code executes during the build. All inputs and renders are checked
    # before the first output write; failures retain their incomplete output.
    output.mkdir(mode=0o700)
    (output / 'review').mkdir(mode=0o700)
    if configuration is not None:
        (output / prefix).mkdir(mode=0o700, parents=True)
        (output / run['host']['file']).parent.mkdir(mode=0o700, parents=True)
        report['configuration'] = {key: value for key, value in configuration.items() if key not in ('run', 'host', 'menu')}
        report['generated_review_files'].append('FINAL-SETTINGS.json')
    payload['SOURCE-BUILD.json'] = canonical(report)
    for name, raw in payload.items():
        fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, nargs='?', help='optional absolute W5 configuration directory')
    parser.add_argument('--donor', type=Path, required=True, help='authenticated W5 v2 preparation root')
    parser.add_argument('--output', type=Path, required=True, help='new absolute source-review directory')
    parser.add_argument('--packet-id', help='fresh 16-hex review identity; requires a run directory')
    parser.add_argument('--upstream-root', type=Path, help='pinned effective base sources for patch reproduction')
    parser.add_argument('--configured-artifacts', action='store_true', help='project artifact limits for the protected consumer proposal')
    args = parser.parse_args()
    report = build(args.donor, args.output, run_dir=args.run_dir, packet_id=args.packet_id,
                   upstream_root=args.upstream_root, configured_artifacts=args.configured_artifacts)
    print(json.dumps(dict(status=report['status'], content_sha256=report['content_sha256'],
                         files=len(report['files']), launch_authorized=False), sort_keys=True))


if __name__ == '__main__':
    main()
