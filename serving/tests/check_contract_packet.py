"""Compare the cleaned contract with an authenticated private packet on fresh copies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import SOURCE, canonical, load
from test_entrypoint_cleanup import exercise_main
from test_source_builder import builder, tree_bytes


class ContractPacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        root = PACKET.resolve(strict=True)
        seal = root / 'FINAL-ARTIFACTS.sha256'
        if hashlib.sha256(seal.read_bytes()).hexdigest() != metadata['source_seal_sha256']:
            raise AssertionError('retained packet seal pin differs')
        members = dict((name, digest) for digest, name in
                       (line.split('  ', 1) for line in seal.read_text().splitlines()))
        cls.temporary = tempfile.TemporaryDirectory(prefix='cleanup-contract-packet-')
        cls.addClassCleanup(cls.temporary.cleanup)
        scratch = Path(cls.temporary.name).resolve()
        verifier_name = 'quality-template-w5/seal_evidence.py'
        verifier_raw = (root / verifier_name).read_bytes()
        if hashlib.sha256(verifier_raw).hexdigest() != members[verifier_name]:
            raise AssertionError('retained seal verifier source differs')
        verifier_path = scratch / 'seal_evidence.py'
        verifier_path.write_bytes(verifier_raw)
        verifier = load('cleanup_contract_seal', verifier_path)
        verified = verifier.verify(root, metadata['source_seal_sha256'])
        if verified['members'] != metadata['source_seal_members']:
            raise AssertionError('retained membership differs')
        donor = root / 'author/package-v2/review'
        cls.manifest = json.loads((donor.parent / 'MANIFEST.json').read_bytes())
        if hashlib.sha256((donor / 'mtp_contract.py').read_bytes()).hexdigest() != metadata['source_files']['mtp_contract.py']:
            raise AssertionError('retained contract source differs')
        cls.roots = [scratch / name for name in ('baseline', 'cleaned')]
        for destination in cls.roots:
            shutil.copytree(donor, destination, copy_function=shutil.copyfile)
            # The sealed donor is read-only. Only these disposable copies are mutable.
            destination.chmod(0o700)
            for directory in destination.rglob('*'):
                if directory.is_dir():
                    directory.chmod(0o700)
        shutil.copyfile(SOURCE / 'mtp_contract.py', cls.roots[1] / 'mtp_contract.py')
        for name in ('serving_client.py', 'request_receipts.py', 'model_benchmark.py',
                     'serving_contract.py', 'serving_parity.py', 'serving_launcher.py',
                     'serving_entry.py', 'serving_reader.py'):
            shutil.copyfile(SOURCE / name, cls.roots[1] / name)
        cls.modules = [load('cleanup_contract_' + path.name, path / 'mtp_contract.py') for path in cls.roots]
        cls.settings = (donor / 'FINAL-SETTINGS.json').read_bytes()
        cls.comparisons = 0
        cls.bundles = [scratch / name for name in ('build-one', 'build-two')]
        cls.build_reports = [builder.build(root, destination) for destination in cls.bundles]
        cls.configured = [scratch / name for name in ('configured-one', 'configured-two')]
        cls.configured_reports = [builder.build(root, destination,
            run_dir=HERE.parent / 'config/runs/w5-v2') for destination in cls.configured]

    @classmethod
    def tearDownClass(cls):
        print(f'{cls.comparisons} byte-identical contract output comparisons')

    def tearDown(self):
        for root in self.roots:
            (root / 'FINAL-SETTINGS.json').write_bytes(self.settings)

    def same(self, name, *args):
        values = [getattr(module, name)(*args) for module in self.modules]
        self.assertIs(type(values[0]), type(values[1]), name)
        raw = [canonical(value) for value in values]
        self.assertEqual(raw[0], raw[1], name)
        type(self).comparisons += 1
        return values[1]

    def test_selected_profile_and_both_arm_outputs_match(self):
        for name in ('settings', 'phase_config', 'precision_profile', 'precision_loading_policy',
                     'prompt_minimization_enabled', 'precision_reference_enabled', 'dense_mxfp4_enabled',
                     'target_quantization', 'target_model_revision', 'target_config_sha256',
                     'memory_fraction_static', 'serving_seed_required', 'verify_attention_ab_enabled',
                     'verify_attention_chain_enabled', 'gdn_capture_enabled', 'norm_capture_enabled',
                     'norm_candidate_enabled', 'greedy_quality_draft_ab', 'j01_probe_enabled',
                     'head_input_capture_enabled', 'final_norm_capture_enabled', 'midpoint_capture_enabled',
                     'raw_attention_capture_enabled', 'cache_config', 'expected_recurrent_slots',
                     'pool_token_capacity', 'package_assets', 'runtime_overlays', 'bytecode_mask_count',
                     'bytecode_mask_paths', 'graph_bytecode_mask_dirs', 'draft_model_binding',
                     'owner_arm_map', 'serving_stack', 'full_enabled_benchmark', 'commit_observation_limit',
                     'paired_serving_enabled', 'paired_group_specs', 'kv_composition_enabled',
                     'instrumentation_enabled', 'kv_request_groups', 'logical_package_inventory'):
            with self.subTest(function=name):
                self.same(name)
        for arm in ('control', 'mtp'):
            for name in ('contract_for_arm', 'memory_pool_config', 'cache_profile', 'pool_token_capacity',
                         'gemm_for_arm', 'replay_serving_enabled', 'runtime_overlays', 'bytecode_mask_count',
                         'bytecode_mask_paths', 'gdn_source_for_arm', 'expected_environment', 'serving_stack',
                         'prefill_chunk_size', 'paired_groups', 'host_path_cleanup_enabled',
                         'expected_decode_log_interval', 'verify_attention_observed'):
                with self.subTest(function=name, arm=arm):
                    self.same(name, arm)
        for arm in ('bf16', 'kv4'):
            self.same('contract_for_owner_arm', arm)
        self.same('expected_draft_summary', 'dflash2')
        for ordinal in (0, 48, 170, 171, 175):
            self.same('paired_group_for', ordinal)

    def test_supported_settings_variants_keep_acceptance_and_output(self):
        original = json.loads(self.settings)
        variants = []
        for mode in ('on', 'production', None):
            value = copy.deepcopy(original)
            if mode is None:
                value.pop('instrumentation')
            else:
                value['instrumentation'] = mode
            variants.append(value)
        value = copy.deepcopy(original)
        value.pop('allocator_snapshot')
        variants.append(value)
        for arm in ('control', 'mtp'):
            value = copy.deepcopy(original)
            value['d3b_draft_head_arm'] = arm
            variants.append(value)
        value = copy.deepcopy(original)
        for keys in [('gdn_layout_both', 'd3b_draft_head_arm'), ('d3b_target_head',), ('p1_prefill',)]:
            for key in keys:
                value.pop(key)
            variants.append(copy.deepcopy(value))
        value['precision_reference'] = 'fp8'
        variants.append(value)
        for index, value in enumerate(variants):
            for root in self.roots:
                (root / 'FINAL-SETTINGS.json').write_bytes(canonical(value))
            with self.subTest(variant=index):
                self.same('settings')
                self.same('phase_config')
                self.same('contract_for_arm', 'control')
                self.same('contract_for_arm', 'mtp')
                if 'instrumentation' not in value:
                    # The owner explicitly selected production as the default.
                    # Preserve this intentional change as its own assertion;
                    # explicit modes and every other output still compare.
                    self.assertIs(self.modules[0].instrumentation_enabled(),True)
                    self.assertIs(self.modules[1].instrumentation_enabled(),False)
                else:
                    self.same('instrumentation_enabled')

    def test_invalid_settings_keep_refusal(self):
        original = json.loads(self.settings)
        variants = []
        for key, bad in [('schema', 2), ('mode', 'native-http-final-verification-v1'),
                         ('precision_reference', 'bf16'), ('allocator_snapshot', False),
                         ('instrumentation', 'off'), ('serving_stack', 'unselected'),
                         ('execution_order', ['mtp', 'control']), ('quality_screen', 'sprint36-chunk-prefill-v1'),
                         ('p1_prefill', 'changed'), ('d3b_target_head', 'changed'),
                         ('gdn_layout_both', False), ('d3b_draft_head_arm', 'unknown'),
                         ('screen_policy', 'changed'), ('full_enabled_benchmark', 'changed')]:
            value = copy.deepcopy(original)
            value[key] = bad
            variants.append((key, value))
        for key in ('phase_timing', 'target_timing', 'performance_benchmark', 'j01_probe',
                    'midpoint_capture', 'diagnostic_kernel_serialization'):
            value = copy.deepcopy(original)
            value[key] = True
            variants.append((key, value))
        for key, bad in [('maximum_output_tokens', 1), ('input_tokens', 1), ('internal_reserve_tokens', 0),
                         ('ordinal', True), ('context_envelope_tokens', 1), ('sha256', '0' * 64),
                         ('path', '../FINAL-REQUEST-00.json'), ('generated_history', {})]:
            value = copy.deepcopy(original)
            value['requests'][0][key] = bad
            variants.append(('request.' + key, value))
        for name, value in variants:
            for root, module in zip(self.roots, self.modules):
                (root / 'FINAL-SETTINGS.json').write_bytes(canonical(value))
                with self.subTest(field=name, source=root.name), self.assertRaises(module.MtpContractError):
                    module.settings()

    def test_all_24_inactive_phases_refuse_before_asset_read(self):
        before, after = self.modules
        old = set(before.PHASES) - {after.PAIRED_PHASE}
        self.assertEqual(len(old), 24)
        for phase in old:
            value = json.loads(self.settings)
            value['phase'] = phase
            (self.roots[1] / 'FINAL-SETTINGS.json').write_bytes(canonical(value))
            with self.subTest(phase=phase), patch.object(after, '_assets') as assets, \
                 self.assertRaisesRegex(after.MtpContractError, 'selected paired-serving phase'):
                after.settings()
            assets.assert_not_called()

    def test_frozen_assets_and_source_tampering_still_refuse(self):
        for filename, function in [('FINAL-REQUESTS.zlib', 'settings'),
                                   ('allocation_sizing_candidate.py', 'runtime_overlays')]:
            for root, module in zip(self.roots, self.modules):
                path = root / filename
                original = path.read_bytes()
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                try:
                    with self.subTest(file=filename, source=root.name), self.assertRaises(module.MtpContractError):
                        getattr(module, function)()
                finally:
                    path.write_bytes(original)

    def test_generated_sandbox_arguments_and_template_expansion_match(self):
        before, after = self.modules
        paths = ['/mnt/capture-bytecode/mnt/composition/image-source/python/sglang/srt/' + name
                 for name in before.graph_bytecode_mask_dirs()]
        argv = ['bwrap', '--ro-bind', '/source', '/mnt/review', '--dev-bind', '/dev/kfd', '/dev/kfd']
        for directory in ('/mnt/output/home', '/mnt/output/tmp', '/mnt/output/cache', '/mnt/capture-bytecode'):
            argv.extend(['--dir', directory])
        for path in paths:
            argv.extend(['--size', str(1 << 20), '--tmpfs', path, '--remount-ro', path])
        files = {'review/replay_memory_pool_candidate.py': 'unused'}
        self.assertEqual(before.replay_serving_sandbox_argv(argv, files),
                         after.replay_serving_sandbox_argv(argv, files))
        sources = {'source': {'sha256': '1' * 64}}
        digest = hashlib.sha256(canonical(sources)).hexdigest()
        value = dict(source_manifest_sha256=digest,
            binding_template=dict(sources=sources, source_manifest_sha256=digest),
            worker_template=dict(source_manifest_sha256=digest,
                sources=dict(same_as='binding_template.sources', source_manifest_sha256=digest)))
        self.same('expand_http_template', value)

    def test_actual_client_contract_and_reader_preserve_all_frozen_request_receipts(self):
        results = []
        for root, module in zip(self.roots, self.modules):
            with patch.dict(sys.modules, mtp_contract=module):
                contract = load('serving_contract', root / 'serving_contract.py')
                with patch.dict(sys.modules, serving_contract=contract):
                    client = load('serving_client', root / 'serving_client.py')
                    with patch.dict(sys.modules, serving_client=client):
                        reader = load('serving_reader', root / 'serving_reader.py')
                self.assertIs(reader.serving_client, client)
                self.assertIs(reader.contract, contract)
                requests = contract.requests()
                self.assertEqual(len(requests), 176)
                block = root / 'receipt-comparison' / 'paired'
                bodies = []
                for ordinal, row in enumerate(requests):
                    name = f'{ordinal:02d}-' + row['case'] + '.raw'
                    path = block / 'requests' / name
                    client._write_json(path, row['body'])
                    raw = reader.serving_client._kv_receipts.read(path)
                    self.assertEqual(raw, contract.canonical(row['body']))
                    bodies.append(raw)
                files = {p.relative_to(block).as_posix(): p.read_bytes()
                         for p in block.rglob('*') if p.is_file()}
                results.append((bodies, files))
        self.assertEqual(results[0], results[1])

    def test_real_w5_spec_entry_dispatch_and_export_match_sealed_runner(self):
        models = [load('cleanup_packet_entry_' + root.name, root / 'model_benchmark.py')
                  for root in self.roots]
        spec = json.loads((self.roots[0] / 'BENCHMARK-SPEC2.json').read_bytes())
        for arm in ('bf16', 'kv4'):
            for outcome in ('complete', 'failed', 'error_with_result', 'error_without_result', 'interrupt'):
                results = []
                for root, module in zip(self.roots, models):
                    self.assertEqual(module.validate_spec(spec), spec)
                    results.append(exercise_main(module, root / ('entry-' + arm + '-' + outcome),
                        spec=spec, arm=arm, outcome=outcome))
                self.assertEqual(results[0], results[1], (arm, outcome))
                expected = ('SystemExit', 0 if outcome == 'complete' else 1)
                if outcome.startswith('error'):
                    expected = ('RuntimeError', 'injected runtime failure')
                elif outcome == 'interrupt':
                    expected = ('KeyboardInterrupt', 'injected interrupt')
                self.assertEqual(results[1]['termination'], expected)

    def test_unchanged_protected_adapter_preserves_w5_projection_and_proof_reader(self):
        results = []
        spec = json.loads((self.roots[0] / 'BENCHMARK-SPEC2.json').read_bytes())
        for root in self.roots:
            adapter = load('cleanup_packet_adapter_' + root.name, root / 'benchmark_owner_adapter.py')
            self.assertEqual(adapter.readout.bench.validate_spec(spec), spec)
            projection = adapter.readout.bench._GRAPH_OUTER.validate_projection(spec['graph_projection'])
            argv = {}
            for arm in ('bf16', 'kv4'):
                argv[arm] = adapter.bwrap_argv(arm, retained_root=Path('/retained'),
                    composition=Path('/composition'), review=Path('/packet/review'), model=Path('/model'),
                    output=Path('/output') / arm, entry=Path('/packet/review/model_benchmark.py'),
                    spec_sha256='a' * 64, bounded_scratch=True, physical_artifacts=True,
                    session_sha256='b' * 64, artifacts_sha256='c' * 64,
                    graph_contract=spec['graph_projection'], http_contract=spec['http_serving'])
            results.append((projection, argv, adapter.readout.bench.comparison().REFERENCE))
        self.assertEqual(results[0], results[1])

    def test_real_source_builds_are_identical_and_preserve_protected_sources_and_menu(self):
        self.assertEqual(self.build_reports[0], self.build_reports[1])
        self.assertEqual(tree_bytes(self.bundles[0]), tree_bytes(self.bundles[1]))
        report = self.build_reports[0]
        self.assertEqual(report['status'], 'SOURCE_REVIEW_ONLY')
        self.assertFalse(report['launch_authorized'])
        self.assertFalse((self.bundles[0] / 'MANIFEST.json').exists())
        protected = json.loads((self.bundles[0] / 'PROTECTED-SHA256.json').read_bytes())
        for name, pin in protected.items():
            raw = (self.bundles[0] / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin)
            self.assertEqual(raw, (self.roots[0] / Path(name).name).read_bytes())
        for name in ('FINAL-SETTINGS.json', 'FINAL-REQUESTS.zlib', 'PRECISION-REFERENCE.json', 'DRAFT-MODEL.json'):
            self.assertEqual((self.bundles[0] / 'review' / name).read_bytes(), (self.roots[0] / name).read_bytes())
        references = report['dependency_inventory']['references_to_changed_members']
        self.assertTrue(any(row['protected'] and row['source'] == 'benchmark_physical_controller.py'
                            and row['target'] == 'BENCHMARK-SPEC2.json' for row in references))

    def test_configured_build_is_deterministic_and_preserves_every_review_byte(self):
        self.assertEqual(self.configured_reports[0], self.configured_reports[1])
        self.assertEqual(tree_bytes(self.configured[0]), tree_bytes(self.configured[1]))
        self.assertEqual(tree_bytes(self.configured[0] / 'review'), tree_bytes(self.bundles[0] / 'review'))
        root = self.configured[0]
        run_dir = root / 'config/runs/w5-v2'
        run = json.loads((run_dir / 'run.json').read_bytes())
        self.assertTrue((root / run['host']['file']).is_file())
        self.assertTrue((run_dir / run['menu']['file']).is_file())
        metadata = json.loads((HERE.parent / 'SOURCE.json').read_bytes())
        config = self.configured_reports[0]['configuration']
        self.assertEqual(config['pins'], metadata['run_configuration']['baseline_sha256'])
        self.assertFalse(config['evidence_capacity_verified'])
        self.assertFalse(config['protected_migration_applied'])
        artifacts = load('cleanup_config_protected_artifacts', root / 'review/benchmark_artifacts.py')
        self.assertEqual(config['derived_limits']['http_session_max_bytes'], artifacts.HTTP_SESSION_MAX_BYTES)
        self.assertEqual(config['derived_limits']['http_payload_bytes'], sum(artifacts.HTTP_MEMBER_LIMITS.values()))

    def test_generated_resource_manifest_passes_actual_gate_and_owner_projection(self):
        root = self.configured[0]
        review = root / 'review'
        manifest = self.manifest
        spec_raw = (review / 'BENCHMARK-SPEC2.json').read_bytes()
        spec = json.loads(spec_raw)
        resource_raw = (review / 'RESOURCE-MANIFEST.json').read_bytes()
        package_files = {name: row['sha256'] for name, row in self.configured_reports[0]['files'].items()
                         if name.startswith('review/')}

        class NoNativeImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('torch', 'triton', 'sglang'):
                    raise AssertionError('resource check attempted native import: ' + fullname)
                return None

        with patch.dict(sys.modules), patch.object(sys, 'meta_path', [NoNativeImports(), *sys.meta_path]):
            adapter = load('cleanup_resource_adapter', review / 'benchmark_owner_adapter.py')
            gate = load('cleanup_resource_gate', review / 'benchmark_resource_gate.py')
            contract = load('cleanup_resource_contract', review / 'mtp_contract.py')
            value = gate.read_manifest(review / 'RESOURCE-MANIFEST.json', hashlib.sha256(resource_raw).hexdigest())
            self.assertEqual(adapter.readout.bench.validate_spec(spec), spec)
            for key, name in (('benchmark_spec_sha256', 'BENCHMARK-SPEC2.json'),
                              ('entry_sha256', 'model_benchmark.py'), ('gate_sha256', 'benchmark_resource_gate.py'),
                              ('artifacts_sha256', 'benchmark_artifacts.py')):
                self.assertEqual(value[key], package_files['review/' + name])
            target = Path(manifest['target_root'])
            for arm in ('bf16', 'kv4'):
                argv = adapter.bwrap_argv(arm, retained_root=Path(manifest['retained_root']),
                    composition=Path(manifest['runtime_root']), review=target / 'review',
                    model=manifest['model_root'], output=target / 'arms' / arm,
                    entry=target / 'review/model_benchmark.py', spec_sha256=hashlib.sha256(spec_raw).hexdigest(),
                    bounded_scratch=True, physical_artifacts=True, session_sha256=manifest['session_sha256'],
                    artifacts_sha256=value['artifacts_sha256'], graph_contract=spec['graph_projection'],
                    http_contract=spec['http_serving'])
                expected = contract.replay_serving_sandbox_argv(argv, package_files)
                self.assertEqual(value['workload_argv'][arm], list(expected), arm)
                self.assertEqual(value['workload_argv_sha256'][arm], gate.argv_sha256(list(expected)))
            donor = json.loads((self.roots[0] / 'RESOURCE-MANIFEST.json').read_bytes())
            self.assertNotEqual(value['benchmark_spec_sha256'], donor['benchmark_spec_sha256'])
            self.assertNotEqual(value['entry_sha256'], donor['entry_sha256'])
            for mutation in ('stale_argv_hash', 'changed_arm', 'changed_session'):
                bad = copy.deepcopy(value)
                if mutation == 'stale_argv_hash':
                    bad['workload_argv_sha256']['bf16'] = donor['workload_argv_sha256']['bf16']
                else:
                    argv = bad['workload_argv']['bf16']
                    flag, replacement = ('--arm', 'kv4') if mutation == 'changed_arm' else ('--session-sha256', '0' * 64)
                    argv[argv.index(flag) + 1] = replacement
                    bad['workload_argv_sha256']['bf16'] = gate.argv_sha256(argv)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    gate.validate_manifest(bad)

    def test_builder_refuses_changed_run_before_creating_output(self):
        root = self.bundles[0].parent / 'changed-configuration-source'
        (root / 'legacy').mkdir(parents=True)
        shutil.copytree(SOURCE, root / 'legacy/review')
        shutil.copytree(HERE.parent / 'config', root / 'config')
        shutil.copyfile(HERE.parent / 'SOURCE.json', root / 'SOURCE.json')
        run_dir = root / 'config/runs/w5-v2'
        path = run_dir / 'run.json'
        value = json.loads(path.read_bytes())
        value['profile']['context_length'] = 65536
        path.write_bytes(builder.canonical(value))
        output = root.parent / 'refused-config-output'
        with self.assertRaisesRegex(ValueError, 'supported W5 baseline'):
            builder.build(PACKET.resolve(), output, source_root=root, run_dir=run_dir)
        self.assertFalse(output.exists())

    def test_real_source_build_refreshes_overlay_and_private_pins(self):
        review = self.bundles[0] / 'review'
        module = load('cleanup_built_contract', review / 'mtp_contract.py')
        spec = json.loads((review / 'BENCHMARK-SPEC2.json').read_bytes())
        for arm in ('control', 'mtp'):
            row = spec['http_serving']['bindings'][arm]
            raw = (self.bundles[0] / row['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
            outer = module.expand_http_template(json.loads(raw))
            sources = outer['binding_template']['sources']
            self.assertEqual(sources, outer['worker_template']['sources'])
            for name in builder.EXTRACTED_HELPERS:
                self.assertIn(name, sources)
            for source in sources.values():
                if source['path'].startswith('/mnt/review/'):
                    self.assertEqual(source['sha256'], hashlib.sha256((review / Path(source['path']).name).read_bytes()).hexdigest())
            module.mxfp4_sources(sources)
            overlays = module.runtime_overlays(arm)
            self.assertEqual(overlays, self.modules[0].runtime_overlays(arm))
        retained_overlay = json.loads((review / 'ALLOCATION-OVERLAY.json').read_bytes())
        triton = next(row for row in retained_overlay['files'] if row['key'] == 'triton_backend')
        self.assertEqual(triton['sha256'], hashlib.sha256((review / triton['file']).read_bytes()).hexdigest())
        self.assertEqual(module.settings(), self.modules[0].settings())

    def test_built_actual_http_and_readout_import_chains_need_no_native_modules(self):
        review = self.bundles[0] / 'review'
        model = load('cleanup_built_entry', review / 'model_benchmark.py')
        spec = json.loads((review / 'BENCHMARK-SPEC2.json').read_bytes())
        projection = self.bundles[0].parent / 'cpu-template-projection'
        projection.mkdir()
        # The source bytes stay unchanged. Only virtual /mnt/review paths in
        # temporary test templates are projected onto the local CPU directory.
        for arm in ('control', 'mtp'):
            name = f'HTTP-BINDING-{arm}.json'
            value = json.loads((review / name).read_bytes())
            sources = value['binding_template']['sources']
            for row in sources.values():
                if row['path'].startswith('/mnt/review/'):
                    row['path'] = str(review / Path(row['path']).name)
            pin = hashlib.sha256(canonical(sources)).hexdigest()
            value['source_manifest_sha256'] = pin
            for key in ('binding_template', 'worker_template'):
                value[key]['source_manifest_sha256'] = pin
            value['worker_template']['sources']['source_manifest_sha256'] = pin
            value['worker_template']['preweight']['graph_observer'] = dict(sources['graph_observer'])
            raw = builder.canonical(value)
            (projection / name).write_bytes(raw)
            spec['http_serving']['bindings'][arm]['sha256'] = hashlib.sha256(raw).hexdigest()
        checked = model.checked_file
        def local_template(path, expected, maximum=8 << 20):
            if str(path) in ('/mnt/review/HTTP-BINDING-control.json', '/mnt/review/HTTP-BINDING-mtp.json'):
                path = projection / Path(path).name
            return checked(path, expected, maximum)
        class NoNativeImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('torch', 'triton', 'sglang'):
                    raise AssertionError('CPU import boundary reached native module: ' + fullname)
                return None
        for arm in ('bf16', 'kv4'):
            with patch.dict(sys.modules), patch.object(sys, 'meta_path', [NoNativeImports(), *sys.meta_path]), \
                 patch.object(model, 'checked_file', side_effect=local_template):
                for name in builder.HTTP_IMPORTS:
                    sys.modules.pop(name, None)
                self.assertFalse({'torch', 'triton', 'sglang'} & sys.modules.keys())
                runtime = model.http_runtime_module(spec, arm)
                self.assertEqual(Path(runtime.__file__), review / 'serving_runtime.py')
                imported=tuple(name for name in builder.HTTP_IMPORTS
                    if name!='serving_parity' or runtime.mtp_contract.instrumentation_enabled())
                if not runtime.mtp_contract.instrumentation_enabled():
                    self.assertNotIn('serving_parity',sys.modules)
                for name in imported:
                    self.assertEqual(Path(sys.modules[name].__file__), review / (name + '.py'))
                self.assertEqual(len(sys.modules['serving_contract'].CASES), 176)
                integration = load('cleanup_built_integration', review / 'serving_integration.py')
                registered = {name: sys.modules[name] for name in imported}
                contract, reader, parity = integration._private_reader(include_parity=True)
                self.assertEqual(len(contract.CASES), 176)
                self.assertEqual(Path(reader.__file__), review / 'serving_reader.py')
                self.assertEqual(Path(parity.__file__), review / 'serving_parity.py')
                self.assertTrue(all(sys.modules[name] is module for name, module in registered.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    options, remaining = parser.parse_known_args()
    PACKET = options.packet
    unittest.main(argv=[sys.argv[0], *remaining])
