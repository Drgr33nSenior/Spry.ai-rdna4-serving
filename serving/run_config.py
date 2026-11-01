"""Validate and project the pinned W5 configuration into the source review build."""
import ast
import copy
import re
from pathlib import Path

from spry_common.canon import canonical_line, digest, need, parse_json
from spry_common.limits import derive, HTTP_ARTIFACT_KEYS

HERE = Path(__file__).resolve().parent


def validate_schema(value, schema, path='$'):
    """Validate only the closed keyword subset used by the three local schemas."""
    allowed = {'$schema', 'title', 'description', 'type', 'properties', 'required', 'additionalProperties',
               'items', 'minItems', 'maxItems', 'minimum', 'maximum', 'pattern', 'enum', 'const'}
    need(not set(schema) - allowed, 'unsupported schema keyword at ' + path)
    kinds = {'object': (dict,), 'array': (list,), 'string': (str,), 'integer': (int,),
             'number': (int, float), 'boolean': (bool,)}
    need(schema.get('type') in kinds, 'explicit schema type required at ' + path)
    need(type(value) in kinds[schema['type']], 'wrong type at ' + path)
    if 'const' in schema:
        need(canonical_line(value) == canonical_line(schema['const']), 'wrong fixed value at ' + path)
    if 'enum' in schema:
        need(any(canonical_line(value) == canonical_line(item) for item in schema['enum']), 'invalid choice at ' + path)
    if type(value) is dict:
        need(schema.get('additionalProperties') is False, 'closed object schema required at ' + path)
        properties = schema['properties']
        need(set(schema['required']) <= set(value) <= set(properties), 'missing or unknown field at ' + path)
        for key, item in value.items():
            validate_schema(item, properties[key], path + '.' + key)
    elif type(value) is list:
        need(schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 2**63), 'array size at ' + path)
        for index, item in enumerate(value):
            validate_schema(item, schema['items'], path + '[' + str(index) + ']')
    elif type(value) is str and 'pattern' in schema:
        need(re.fullmatch(schema['pattern'], value) is not None, 'invalid string at ' + path)
    elif type(value) in (int, float):
        need(schema.get('minimum', -2**63) <= value <= schema.get('maximum', 2**63), 'numeric range at ' + path)


def literal_assignment(raw, name):
    """Read named literal ceilings from authenticated code without executing it."""
    nodes = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    need(len(nodes) == 1, 'single protected assignment required: ' + name)
    def literal(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, str):
            return node.value
        if isinstance(node, ast.Dict):
            return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp):
            left, right = literal(node.left), literal(node.right)
            need(type(left) is int and type(right) is int, 'integer ceiling expression required')
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.Add):
                return left + right
        raise ValueError('nonliteral protected ceiling: ' + name)
    return literal(nodes[0])


def load(run_dir, source_root, metadata, files, read, *, configured_artifacts=False, projector=None):
    run_dir = Path(run_dir)
    need(run_dir.is_absolute() and run_dir.resolve(strict=True) == run_dir, 'canonical run directory required')
    run = parse_json(read(run_dir / 'run.json', 256 << 10))
    schema_raw = {name: read(HERE / 'config/schema' / (name + '.schema.json'), 256 << 10)
                  for name in ('run-config', 'host', 'menu')}
    validate_schema(run, parse_json(schema_raw['run-config']))
    configuration_root=run_dir.parent.parent.parent if projector is not None else source_root
    host_path = configuration_root / run['host']['file']
    need(host_path.is_relative_to(configuration_root / 'config/host'), 'host config location differs')
    host = parse_json(read(host_path, 64 << 10))
    menu = parse_json(read(run_dir / run['menu']['file'], 1 << 20))
    validate_schema(host, parse_json(schema_raw['host']))
    validate_schema(menu, parse_json(schema_raw['menu']))
    pins = {name: digest(canonical_line(value)) for name, value in (('run', run), ('host', host), ('menu', menu))}
    need(pins['host'] == run['host']['sha256'] and pins['menu'] == run['menu']['sha256'], 'config reference pin differs')
    baseline = metadata['run_configuration']['baseline_sha256']
    if projector is not None:
        need(configured_artifacts and callable(projector), 'configured consumers required for mutation projection')
        limits = derive(run['limits'])
        ceiling_run=parse_json(read(HERE/'config/runs/w5-v2/run.json',256<<10))
        need(digest(canonical_line(ceiling_run))==baseline['run'],'authenticated primary ceiling baseline required')
        need(all(value<=ceiling_run['limits'][name] for name,value in run['limits'].items()),
             'configuration may only tighten the existing primary ceilings')
        projected = projector(files, run, host, menu, run_dir=run_dir, read=read)
        return dict(run=run, host=host, menu=menu, pins=pins, derived_limits=limits,
                    schema_sha256={name: digest(raw) for name, raw in schema_raw.items()},
                    evidence_capacity_verified=False, protected_migration_applied=False,
                    configuration_mutations_supported=True, projection=projected)
    need(pins == baseline, 'configuration differs from the supported W5 baseline; migration required')
    limits = derive(run['limits'])
    protected = files['benchmark_host_transport.py']
    need(limits['total_seconds'] <= literal_assignment(protected, 'TOTAL_SECONDS'), 'protected total ceiling exceeded')
    artifacts = files['benchmark_artifacts.py']
    member_limits = literal_assignment(artifacts, 'HTTP_MEMBER_LIMITS')
    for key, name in (('result_json_bytes', 'result.json'), ('client_bundle_bytes', 'http-client-bundle.json'),
                      ('native_bundle_bytes', 'native-http-bundle.json')):
        need(run['limits'][key] <= member_limits[name], 'protected artifact ceiling exceeded')
    need(run['limits']['http_stream_bytes'] <= literal_assignment(artifacts, 'HTTP_STREAM_LIMIT'),
         'protected stream ceiling exceeded')
    adapter = files['benchmark_owner_adapter.py']
    for key, name in (('worker_seconds', 'INNER_WORK'), ('arm_work_seconds', 'ARM_WORK'), ('arm_envelope_seconds', 'ARM_ENVELOPE')):
        need(limits[key] == literal_assignment(adapter, name), 'legacy owner timing differs: ' + key)
    for ref in (*run['base']['assets'], *run['model']['assets']):
        need(ref['file'] in files and digest(files[ref['file']]) == ref['sha256'], 'pinned base/model asset differs')
    need(run['base']['sglang'] == parse_json(files['COMMON-STACK-RECEIPT.json'])['target']['sglang'], 'SGLang version differs')
    need(run['model']['cache_profile'] == parse_json(files['PRECISION-REFERENCE.json'])['cache_profile'], 'cache precision differs')
    for arm in ('control', 'mtp'):
        template = parse_json(files['HTTP-BINDING-' + arm + '.json'])
        need(template['worker_template']['expected_ranks'] == host['rank_indices'], 'rank order differs')
        need(template['speculation']['speculative_dflash_block_size'] == run['model']['draft_width']
             == template['speculation']['speculative_num_draft_tokens'], 'DFlash width differs')
    settings = parse_json(files['FINAL-SETTINGS.json'])
    requests = [{key: value for key, value in row.items() if key != 'stop_policy'} for row in menu['rows']]
    need(requests == settings['requests'] and menu['groups'] == settings['paired_serving']['groups'], 'frozen menu differs')
    need(all(row['stop_policy'] == 'natural' for row in menu['rows']), 'cap policy requires protected migration')
    profile = run['profile']
    need([row['ordinal'] for row in menu['rows']] == list(range(len(menu['rows']))), 'menu ordinal order')
    for row in menu['rows']:
        size = row['input_tokens'] + row['maximum_output_tokens'] + row['internal_reserve_tokens']
        need(size <= row['context_envelope_tokens'] <= profile['context_length'], 'request context overflow')
    for group in menu['groups']:
        rows = [menu['rows'][ordinal] for ordinal in group['ordinals']]
        sizes = [row['input_tokens'] + row['maximum_output_tokens'] + row['internal_reserve_tokens'] for row in rows]
        need((sum(sizes) if group['concurrency'] == 2 else max(sizes)) <= profile['max_total_tokens'], 'token pool overflow')
    spec = parse_json(files['BENCHMARK-SPEC2.json'])
    need(host['devices'] == spec['http_serving']['expected_devices'], 'host device binding differs')
    need(limits['worker_seconds'] == spec['worker_seconds'] and run['limits']['client_seconds'] == spec['http_serving']['client_seconds'], 'retained timing differs')
    for key, value in profile.items():
        need(spec['server_args'][key] == value, 'retained profile differs: ' + key)
    resource = parse_json(files['RESOURCE-MANIFEST.json'])
    for argv in resource['workload_argv'].values():
        overlays = [Path(argv[index + 1]).name for index, arg in enumerate(argv[:-2])
                    if arg == '--ro-bind' and '/review/' in argv[index + 1]
                    and argv[index + 2].startswith('/mnt/composition/') and argv[index + 1].endswith('.py')]
        need(overlays == run['overlays'], 'retained overlay order differs')
    settings['requests'] = requests
    settings['paired_serving']['groups'] = menu['groups']
    settings['instrumentation'] = run['instrumentation']
    rendered = canonical_line(settings)
    need(rendered == files['FINAL-SETTINGS.json'], 'W5 settings reproduction differs')
    need(2 * len(rendered) <= run['limits']['settings_bytes'], 'settings lack twofold byte margin')
    files['FINAL-SETTINGS.json'] = rendered
    spec['worker_seconds'] = limits['worker_seconds']
    spec['http_serving']['client_seconds'] = run['limits']['client_seconds']
    spec['http_serving']['expected_devices'] = copy.deepcopy(host['devices'])
    if configured_artifacts:
        spec['http_serving']['artifact_limits'] = {key: run['limits'][key] for key in HTTP_ARTIFACT_KEYS}
    spec['server_args'].update(profile)
    files['BENCHMARK-SPEC2.json'] = canonical_line(spec)
    return dict(run=run, host=host, menu=menu, pins=pins, derived_limits=limits,
                schema_sha256={name: digest(raw) for name, raw in schema_raw.items()},
                evidence_capacity_verified=False, protected_migration_applied=False)


def project_configuration(files,run,host,menu,*,run_dir,read,contract):
    """Render the configured consumer proposal without changing protected code."""
    import base64
    import zlib
    def inflate(raw):
        decoder=zlib.decompressobj();value=decoder.decompress(raw,(16<<20)+1)
        need(len(value)<=16<<20 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
             'complete bounded configuration asset')
        return parse_json(value)
    limits=derive(run['limits'])
    need(limits['total_seconds']<=6360 and limits['caller_seconds']<=7160
         and limits['worker_seconds']<=2280 and limits['arm_work_seconds']<=2430
         and limits['arm_envelope_seconds']<=2490,'configured owner timing ceiling')
    for ref in (*run['base']['assets'],*run['model']['assets']):
        need(ref['file'] in files and digest(files[ref['file']])==ref['sha256'],'pinned configuration input asset differs')
    need(run['base']['sglang']==parse_json(files['COMMON-STACK-RECEIPT.json'])['target']['sglang'],
         'configured SGLang identity differs')
    original_overlay=parse_json(files['RESOURCE-MANIFEST.json'])['workload_argv']['bf16']
    overlays=[Path(original_overlay[i+1]).name for i,item in enumerate(original_overlay[:-2]) if item=='--ro-bind'
              and '/review/' in original_overlay[i+1] and original_overlay[i+2].startswith('/mnt/composition/')
              and original_overlay[i+1].endswith('.py')]
    need(run['overlays']==overlays,'configured overlay selection requires a separate source proposal')
    outer={name:base64.b64decode(value,validate=True) for name,value in inflate(files['FINAL-REQUESTS.zlib']).items()}
    inherited=outer['FINAL-REQUESTS.zlib']
    need(digest(inherited)==contract.REQUEST_INHERITED['sha256'] and len(inherited)==contract.REQUEST_INHERITED['bytes'],
         'immutable configured tokenizer source')
    base_assets={name:base64.b64decode(value,validate=True) for name,value in inflate(inherited).items()}
    rows,client=contract.resolve_configured_requests(base_assets,run,menu)
    old_client=inflate(outer['CLIENT-BINDING.zlib'])
    client_raw=canonical_line(dict(old_client,schema=2,rows=client));need(len(client_raw)<=16<<20,'configured client decoded bound')
    outer['CLIENT-BINDING.zlib']=zlib.compress(client_raw,9)
    outer_raw=canonical_line({name:base64.b64encode(raw).decode() for name,raw in outer.items()})
    stored=zlib.compress(outer_raw,9);need(len(stored)<=2<<20 and len(outer_raw)<=16<<20,'configured request container bound')
    inventory=dict(request_container=dict(stored_name='review/FINAL-REQUESTS.zlib',stored=dict(bytes=len(stored),sha256=digest(stored)),
        expanded=dict(bytes=len(outer_raw),sha256=digest(outer_raw)),
        members={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in outer.items()},
        inherited_container=dict(contract.REQUEST_INHERITED)),composition_facts=dict(stored_name='review/COMPOSITION-FACTS.zlib',
        stored=dict(contract.FACTS_STORED),logical_name='review/COMPOSITION-FACTS.json',logical=dict(contract.FACTS_LOGICAL)),
        executable_members_packed=0)
    settings=contract.project_settings(parse_json(files['FINAL-SETTINGS.json']),run,host,menu,rows,inventory)
    contract.validate_configured_settings(dict(settings,requests=rows))
    rendered=canonical_line(settings);need(2*len(rendered)<=run['limits']['settings_bytes'],'configured settings require twofold margin')
    files['FINAL-SETTINGS.json']=rendered;files['FINAL-REQUESTS.zlib']=stored
    precision=parse_json(files['PRECISION-REFERENCE.json'])
    precision.update(context_length=run['profile']['context_length'],max_total_tokens=run['profile']['max_total_tokens'],
                     memory_fraction_static=run['profile']['mem_fraction_static'])
    files['PRECISION-REFERENCE.json']=canonical_line(precision)
    stack=parse_json(files['SERVING-STACK.json']);stack.update(worker_seconds=limits['worker_seconds'],client_seconds=run['limits']['client_seconds'])
    files['SERVING-STACK.json']=canonical_line(stack)
    for arm in ('control','mtp'):
        name='HTTP-BINDING-'+arm+'.json';binding=parse_json(files[name]);resolved=binding['binding_template']['resolved_contract']
        binding['worker_template']['expected_devices']=copy.deepcopy(host['devices'])
        for key,value in run['profile'].items():
            if key=='disable_overlap_schedule':resolved['prefix_cache'][key]=value
            else:need(key in resolved,'configured resolved field missing');resolved[key]=value
        cli=binding['binding_template']['canonical_server_cli']
        for key,value in dict(run['profile'],decode_log_interval=1 if run['instrumentation']=='on' else 100).items():
            flag='--'+key.replace('_','-')
            need(cli.count(flag)==1,'one configured CLI flag required: '+flag)
            if type(value) is not bool:cli[cli.index(flag)+1]=str(value)
        files[name]=canonical_line(binding)
    spec=parse_json(files['BENCHMARK-SPEC2.json']);spec.update(worker_seconds=limits['worker_seconds'],
        context_length=run['profile']['context_length'],physical_slots=run['profile']['context_length'],gpu_ids=host['rank_indices'])
    spec['server_args'].update(run['profile']);spec['http_serving'].update(client_seconds=run['limits']['client_seconds'],
        expected_devices=copy.deepcopy(host['devices']),artifact_limits={key:run['limits'][key] for key in HTTP_ARTIFACT_KEYS})
    files['BENCHMARK-SPEC2.json']=canonical_line(spec)
    return dict(schema=1,rows=len(rows),source_prompt_ids_unchanged=True,logical_inventory=inventory,
                settings_bytes=len(rendered),configuration_sha256=settings['configuration']['run_config_sha256'])
