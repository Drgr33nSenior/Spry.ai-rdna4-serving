"""Deterministic source assembly and refusal boundaries using sealed CPU fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cleanup_builder', HERE.parent / 'build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


class SourceBuild(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-build-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.packet = self.root / 'packet'
        self.source = self.root / 'source'
        self.review = self.packet / 'author/package-v2/review'
        self.review.mkdir(parents=True)
        (self.source / 'legacy/review').mkdir(parents=True)
        protected = json.loads((HERE.parent / 'SOURCE.json').read_bytes())['protected_source_files']
        names = set(protected) | {name + '.py' for name in builder.HTTP_IMPORTS} | {
            'serving_reader.py', 'serving_graph_observer.py', 'model_benchmark.py', 'triton_backend_candidate.py'}
        self.files = {name: ('NAME = ' + repr(name) + '\n').encode() for name in names}
        self.files.update({name: b'# retired fixture\n' for name in builder.RETIRED_W5_MEMBERS})
        self.files['FINAL-SETTINGS.json'] = builder.canonical(dict(
            phase='sprint01-paired-serving', d3b_draft_head_arm='both'))
        policy=builder.production_evidence_policy(dict(self.files,**{
            'mtp_contract.py':(HERE.parent/'legacy/review/mtp_contract.py').read_bytes()}))
        declarations=dict(PRODUCTION_EVIDENCE_FILES=policy['files'],
            PRODUCTION_EVIDENCE_MASK_NAME=policy['name'],PRODUCTION_EVIDENCE_MASK_BYTES=policy['raw'])
        self.files['mtp_contract.py'] += ''.join(name+' = '+repr(value)+'\n'
            for name,value in declarations.items()).encode()
        sources = {name[:-3]: dict(path='/mnt/review/' + name, sha256=builder.digest(self.files[name]))
                   for name in names if name[:-3] in builder.HTTP_IMPORTS}
        sources['graph_observer'] = dict(path='/mnt/review/serving_graph_observer.py',
            module='serving_graph_observer', sha256=builder.digest(self.files['serving_graph_observer.py']))
        sources['gpu_identity'] = dict(path='/mnt/review/benchmark_gpu_identity.py',
            module='benchmark_gpu_identity', sha256=builder.digest(self.files['benchmark_gpu_identity.py']))
        sources['stock'] = dict(path='/mnt/composition/stock.py', sha256='e' * 64)
        source_pin = builder.digest(builder.canonical(sources)[:-1])
        for arm in ('control', 'mtp'):
            self.files[f'HTTP-BINDING-{arm}.json'] = builder.canonical(dict(schema=1, arm=arm,
                mode=builder.MODE, source_manifest_sha256=source_pin,
                binding_template=dict(sources=sources, source_manifest_sha256=source_pin),
                worker_template=dict(sources=dict(same_as='binding_template.sources',
                    source_manifest_sha256=source_pin), source_manifest_sha256=source_pin,
                    preweight={name:copy.deepcopy(sources[name]) for name in ('gpu_identity','graph_observer')})))
        self.files['ALLOCATION-OVERLAY.json'] = builder.canonical(dict(schema=1, files=[dict(
            file='triton_backend_candidate.py', original_sha256='f' * 64,
            sha256=builder.digest(self.files['triton_backend_candidate.py']))]))
        self.files['BENCHMARK-SPEC2.json'] = builder.canonical(dict(experiment_mode=builder.MODE,
            http_serving=dict(bindings={arm: dict(path=f'review/HTTP-BINDING-{arm}.json',
                sha256=builder.digest(self.files[f'HTTP-BINDING-{arm}.json'])) for arm in ('control', 'mtp')}),
            graph_projection=dict(helpers={'model_benchmark.py': builder.digest(self.files['model_benchmark.py'])})))
        self.files['benchmark_physical_controller.py'] += (
            'SPEC_SHA256 = ' + repr(builder.digest(self.files['BENCHMARK-SPEC2.json'])) + '\n').encode()
        target = '/home/developer/fp8-kv-benchmark.1234567890abcdef'
        session = 'a' * 64
        resource = dict(schema=1, scope='benchmark-resource-gate-v1', root=target,
            experiment_mode=builder.MODE, session_sha256=session,
            benchmark_spec_sha256=builder.digest(self.files['BENCHMARK-SPEC2.json']),
            gate_path=target + '/review/benchmark_resource_gate.py',
            gate_sha256=builder.digest(self.files['benchmark_resource_gate.py']),
            entry_path=target + '/review/model_benchmark.py',
            entry_sha256=builder.digest(self.files['model_benchmark.py']), bwrap_sha256='b' * 64,
            artifacts_sha256=builder.digest(self.files['benchmark_artifacts.py']),
            units={}, workload_argv={}, workload_argv_sha256={})
        for arm in ('bf16', 'kv4'):
            argv = ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', resource['entry_path'],
                '/mnt/review/model_benchmark.py', '/mnt/work/build-venv/bin/python3', '-I', '-B', '-X',
                'pycache_prefix=/mnt/capture-bytecode', '/mnt/review/model_benchmark.py',
                '--execute-reviewed-benchmark', '--spec', '/mnt/review/BENCHMARK-SPEC2.json',
                '--spec-sha256', resource['benchmark_spec_sha256'], '--output', '/mnt/output/benchmark',
                '--arm', arm, '--physical-artifacts', '--session-sha256', session,
                '--artifacts-sha256', resource['artifacts_sha256']]
            resource['workload_argv'][arm] = argv
            resource['workload_argv_sha256'][arm] = builder.digest(builder.canonical_bare(argv))
            resource['units'][arm] = 'spry-fp8-benchmark-' + session[:16] + '-' + arm + '.scope'
        self.files['RESOURCE-MANIFEST.json'] = builder.canonical(resource)
        for name, raw in self.files.items():
            (self.review / name).write_bytes(raw)
        manifest = dict(target_root=target, session_sha256=session,
                        package_files={'review/' + name: builder.digest(raw) for name, raw in self.files.items()})
        (self.review.parent / 'MANIFEST.json').write_bytes(builder.canonical(manifest))
        (self.packet / 'other-evidence.txt').write_text('retained, not shipped\n')
        self.metadata = dict(source_files={name: builder.digest(self.files[name])
                                          for name in ('serving_runtime.py', 'model_benchmark.py')},
            source_extractions={name + '.py': {} for name in builder.EXTRACTED_HELPERS if name != 'observer_health'},
            new_sources=['legacy/review/observer_health.py'],
            protected_source_files={name: builder.digest(self.files[name]) for name in protected})
        for name in (*builder.EXTRACTED_HELPERS, 'serving_runtime', 'model_benchmark'):
            (self.source / 'legacy/review' / (name + '.py')).write_text('VALUE = "canonical"\n')
        self.seal()

    def seal(self):
        rows = {name: builder.digest(raw) for name, raw in tree_bytes(self.packet).items()
                if name != 'FINAL-ARTIFACTS.sha256'}
        raw = ''.join(pin + '  ' + name + '\n' for name, pin in sorted(rows.items())).encode()
        (self.packet / 'FINAL-ARTIFACTS.sha256').write_bytes(raw)
        self.metadata.update(source_seal_sha256=builder.digest(raw), source_seal_members=len(rows))
        self.save_metadata()

    def save_metadata(self):
        (self.source / 'SOURCE.json').write_bytes(builder.canonical(self.metadata))

    def build(self, name='output'):
        return builder.build(self.packet, self.root / name, source_root=self.source)

    def test_two_builds_are_byte_identical_and_keep_protected_sources(self):
        before = tree_bytes(self.packet)
        left, right = self.build('one'), self.build('two')
        self.assertEqual(left, right)
        self.assertEqual(tree_bytes(self.root / 'one'), tree_bytes(self.root / 'two'))
        self.assertEqual(tree_bytes(self.packet), before)
        self.assertFalse((self.root / 'one/MANIFEST.json').exists())
        self.assertFalse(left['launch_authorized'])
        self.assertEqual(left['status'], 'SOURCE_REVIEW_ONLY')
        for name in self.metadata['protected_source_files']:
            self.assertEqual((self.root / 'one/review' / name).read_bytes(), self.files[name])
        self.assertTrue(any(row['protected'] and row['target'] == 'BENCHMARK-SPEC2.json'
                            for row in left['dependency_inventory']['references_to_changed_members']))

    def test_every_refreshed_private_pin_and_worker_alias_matches_output_bytes(self):
        report = self.build()
        output = self.root / 'output'
        spec = builder.parse_json((output / 'review/BENCHMARK-SPEC2.json').read_bytes())
        for arm in ('control', 'mtp'):
            path = output / spec['http_serving']['bindings'][arm]['path']
            self.assertEqual(builder.digest(path.read_bytes()), spec['http_serving']['bindings'][arm]['sha256'])
            value = builder.parse_json(path.read_bytes())
            sources = value['binding_template']['sources']
            pin = builder.digest(builder.canonical(sources)[:-1])
            self.assertEqual(pin, value['source_manifest_sha256'])
            self.assertEqual(pin, value['worker_template']['sources']['source_manifest_sha256'])
            for helper in builder.EXTRACTED_HELPERS:
                self.assertIn(helper, sources)
            for row in sources.values():
                if row['path'].startswith('/mnt/review/'):
                    name=Path(row['path']).name
                    policy=builder.production_evidence_policy(self.files)
                    raw = policy['raw'] if name in policy['files'] else (output/'review'/name).read_bytes()
                    self.assertEqual(row['sha256'], builder.digest(raw))
            self.assertEqual(sources['stock']['sha256'], 'e' * 64)
        overlay = builder.parse_json((output / 'review/ALLOCATION-OVERLAY.json').read_bytes())
        self.assertEqual(overlay['files'][0]['original_sha256'], 'f' * 64)
        for name, row in report['files'].items():
            self.assertEqual(row, dict(sha256=builder.digest((output / name).read_bytes()),
                                      bytes=(output / name).stat().st_size))

    def test_retired_members_are_authenticated_but_not_shipped(self):
        report = self.build()
        self.assertEqual(report['omitted_donor_members'],
                         {name: builder.digest(self.files[name]) for name in builder.RETIRED_W5_MEMBERS})
        for name in builder.RETIRED_W5_MEMBERS:
            self.assertTrue((self.review / name).is_file())
            self.assertFalse((self.root / 'output/review' / name).exists())
        # Even omitted source bytes remain subject to the complete donor seal.
        path = self.review / 'prepare_benchmark_package.py'
        before = path.stat()
        path.write_bytes(b'# changed authoring source\n')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'donor member pin differs'):
            self.build('changed')
        self.assertFalse((self.root / 'changed').exists())

    def test_pruning_refuses_other_runner_or_legacy_profile_before_output(self):
        settings = builder.parse_json(self.files['FINAL-SETTINGS.json'])
        spec = builder.parse_json(self.files['BENCHMARK-SPEC2.json'])
        cases = [('FINAL-SETTINGS.json', dict(settings, d3b_draft_head_arm='mtp')),
                 ('FINAL-SETTINGS.json', dict(settings, phase='legacy')),
                 ('BENCHMARK-SPEC2.json', dict(spec, cache_profile={})),
                 ('BENCHMARK-SPEC2.json', dict(spec, validation_comparison={}))]
        for name, value in cases:
            for restore in ('FINAL-SETTINGS.json', 'BENCHMARK-SPEC2.json'):
                (self.review / restore).write_bytes(self.files[restore])
            (self.review / name).write_bytes(builder.canonical(value))
            path = self.review.parent / 'MANIFEST.json'
            manifest = builder.parse_json(path.read_bytes())
            for changed in ('FINAL-SETTINGS.json', 'BENCHMARK-SPEC2.json'):
                manifest['package_files']['review/' + changed] = builder.digest((self.review / changed).read_bytes())
            path.write_bytes(builder.canonical(manifest))
            self.seal()
            with self.subTest(member=name, value=value), self.assertRaisesRegex(ValueError, 'package pruning requires'):
                self.build()
            self.assertFalse((self.root / 'output').exists())

    def test_source_change_changes_content_hash_without_changing_protected_bytes(self):
        before = self.build('before')
        (self.source / 'legacy/review/serving_runtime.py').write_text('VALUE = "changed"\n')
        after = self.build('after')
        self.assertNotEqual(before['content_sha256'], after['content_sha256'])
        self.assertEqual(before['files']['review/benchmark_physical_controller.py'],
                         after['files']['review/benchmark_physical_controller.py'])

    def test_resource_source_pins_and_argument_hashes_match_rendered_bytes(self):
        report = self.build()
        root = self.root / 'output/review'
        value = builder.parse_json((root / 'RESOURCE-MANIFEST.json').read_bytes())
        donor = builder.parse_json(self.files['RESOURCE-MANIFEST.json'])
        self.assertNotEqual(value['entry_sha256'], donor['entry_sha256'])
        self.assertNotEqual(value['benchmark_spec_sha256'], donor['benchmark_spec_sha256'])
        for key, name in (('benchmark_spec_sha256', 'BENCHMARK-SPEC2.json'),
                          ('entry_sha256', 'model_benchmark.py'), ('gate_sha256', 'benchmark_resource_gate.py'),
                          ('artifacts_sha256', 'benchmark_artifacts.py')):
            self.assertEqual(value[key], builder.digest((root / name).read_bytes()))
        for arm, argv in value['workload_argv'].items():
            expected = list(donor['workload_argv'][arm])
            for flag, key in (('--spec-sha256', 'benchmark_spec_sha256'),
                              ('--artifacts-sha256', 'artifacts_sha256')):
                expected[expected.index(flag) + 1] = value[key]
            policy=builder.production_evidence_policy(self.files)
            position=expected.index('/mnt/work/build-venv/bin/python3')
            expected[position:position]=[item for name in policy['files']
                for item in ('--ro-bind',donor['root']+'/'+policy['name'],'/mnt/review/'+name)]
            self.assertEqual(argv, expected)
            self.assertEqual(value['workload_argv_sha256'][arm], builder.digest(builder.canonical_bare(argv)))
        for key in ('root', 'session_sha256', 'units', 'bwrap_sha256', 'entry_path', 'gate_path'):
            self.assertEqual(value[key], donor[key])
        self.assertIn('RESOURCE-MANIFEST.json', report['generated_review_files'])
        self.assertFalse(report['resource_manifest']['new_packet_identity_generated'])

    def test_inconsistent_resource_bindings_refuse_before_output(self):
        original = builder.parse_json(self.files['RESOURCE-MANIFEST.json'])
        mutations = {
            'schema_bool': lambda value: value.update(schema=True),
            'extra_field': lambda value: value.update(unexpected=True),
            'session': lambda value: value.update(session_sha256='c' * 64),
            'root': lambda value: value.update(root=original['root'] + 'other'),
            'unit': lambda value: value['units'].update(bf16='other.scope'),
            'entry_path': lambda value: value.update(entry_path=original['entry_path'] + '.other'),
            'entry_pin': lambda value: value.update(entry_sha256='0' * 64),
            'gate_pin': lambda value: value.update(gate_sha256='0' * 64),
            'spec_pin': lambda value: value.update(benchmark_spec_sha256='0' * 64),
            'artifacts_pin': lambda value: value.update(artifacts_sha256='0' * 64),
            'missing_arm': lambda value: value['workload_argv'].pop('kv4'),
        }
        for name, mutate in mutations.items():
            value = copy.deepcopy(original)
            mutate(value)
            path = self.review / 'RESOURCE-MANIFEST.json'
            path.write_bytes(builder.canonical(value))
            manifest_path = self.review.parent / 'MANIFEST.json'
            manifest = builder.parse_json(manifest_path.read_bytes())
            manifest['package_files']['review/RESOURCE-MANIFEST.json'] = builder.digest(path.read_bytes())
            manifest_path.write_bytes(builder.canonical(manifest))
            self.seal()
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.build()
            self.assertFalse((self.root / 'output').exists())

    def test_refreshing_a_masked_resource_manifest_keeps_exactly_one_mask_set(self):
        self.build()
        files={p.name:p.read_bytes() for p in (self.root/'output/review').iterdir()}
        original=dict(files)
        value=builder.parse_json(files['RESOURCE-MANIFEST.json'])
        manifest=dict(target_root=value['root'],session_sha256=value['session_sha256'])
        builder.render_resource_manifest(files,original,manifest)
        self.assertEqual(files['RESOURCE-MANIFEST.json'],original['RESOURCE-MANIFEST.json'])
        policy=builder.production_evidence_policy(files)
        for argv in builder.parse_json(files['RESOURCE-MANIFEST.json'])['workload_argv'].values():
            for name in policy['files']:self.assertEqual(argv.count('/mnt/review/'+name),1)

    def test_malformed_resource_arguments_refuse_even_with_updated_hash(self):
        original = builder.parse_json(self.files['RESOURCE-MANIFEST.json'])
        for mutation in ('duplicate_flag', 'changed_session', 'changed_arm', 'changed_output',
                         'missing_entry_mount', 'changed_argv_hash'):
            value = copy.deepcopy(original)
            argv = value['workload_argv']['bf16']
            if mutation == 'duplicate_flag':
                argv[1:1] = ['--spec-sha256', value['benchmark_spec_sha256']]
            elif mutation == 'missing_entry_mount':
                argv[argv.index('--ro-bind') + 1] += '.other'
            elif mutation != 'changed_argv_hash':
                flag, replacement = {'changed_session': ('--session-sha256', '0' * 64),
                    'changed_arm': ('--arm', 'kv4'), 'changed_output': ('--output', '/mnt/results/benchmark')}[mutation]
                argv[argv.index(flag) + 1] = replacement
            value['workload_argv_sha256']['bf16'] = (builder.digest(builder.canonical_bare(argv))
                if mutation != 'changed_argv_hash' else '0' * 64)
            path = self.review / 'RESOURCE-MANIFEST.json'
            path.write_bytes(builder.canonical(value))
            manifest_path = self.review.parent / 'MANIFEST.json'
            manifest = builder.parse_json(manifest_path.read_bytes())
            manifest['package_files']['review/RESOURCE-MANIFEST.json'] = builder.digest(path.read_bytes())
            manifest_path.write_bytes(builder.canonical(manifest))
            self.seal()
            with self.subTest(case=mutation), self.assertRaises(ValueError):
                self.build()
            self.assertFalse((self.root / 'output').exists())

    def test_corrupt_seal_refuses_before_output(self):
        (self.packet / 'FINAL-ARTIFACTS.sha256').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'seal pin'):
            self.build()
        self.assertFalse((self.root / 'output').exists())

    def test_changed_member_is_rejected_even_with_original_mtime(self):
        path = self.review / 'serving_runtime.py'
        stamp = path.stat()
        path.write_bytes(b'changed')
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'member pin'):
            self.build()
        self.assertFalse((self.root / 'output').exists())

    def test_missing_and_extra_donor_members_refuse(self):
        path = self.packet / 'other-evidence.txt'
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            self.build()
        path.write_bytes(raw)
        self.seal()
        (self.packet / 'unexpected').write_bytes(b'extra')
        with self.assertRaises(ValueError):
            self.build()

    def test_canonical_links_and_missing_sources_refuse(self):
        path = self.source / 'legacy/review/serving_runtime.py'
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.build()
        path.symlink_to(self.review / 'serving_runtime.py')
        with self.assertRaisesRegex(ValueError, 'canonical input'):
            self.build()
        path.unlink()
        path.write_bytes(raw)
        os.link(path, self.root / 'alias.py')
        with self.assertRaisesRegex(ValueError, 'singly linked'):
            self.build()

    def test_protected_override_is_refused(self):
        name = 'benchmark_artifacts.py'
        (self.source / 'legacy/review' / name).write_bytes(self.files[name])
        self.metadata['source_files'][name] = builder.digest(self.files[name])
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, 'protected canonical override'):
            self.build()
        self.assertFalse((self.root / 'output').exists())

    def test_existing_output_and_source_or_donor_destinations_refuse(self):
        self.build()
        before = tree_bytes(self.root / 'output')
        with self.assertRaisesRegex(ValueError, 'fresh canonical output'):
            self.build()
        self.assertEqual(tree_bytes(self.root / 'output'), before)
        for target in (self.source / 'new', self.packet / 'new'):
            with self.assertRaisesRegex(ValueError, 'outside donor'):
                builder.build(self.packet, target, source_root=self.source)

    def test_invalid_python_fails_before_output(self):
        (self.source / 'legacy/review/serving_runtime.py').write_text('def broken(\n')
        with self.assertRaises(SyntaxError):
            self.build()
        self.assertFalse((self.root / 'output').exists())

    def test_duplicate_json_nonfinite_and_unsafe_paths_refuse(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                builder.parse_json(raw)
        for name in ('../escape', '/absolute', 'a/../escape', 'a//b', './a', 'a\nb'):
            with self.assertRaises(ValueError):
                builder.relative_name(name)

    def test_nonregular_input_refuses_without_blocking(self):
        path = self.root / 'pipe'
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'regular input'):
            builder.read_regular(path)


if __name__ == '__main__':
    unittest.main()
