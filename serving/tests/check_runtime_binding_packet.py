"""Authenticate the runtime-binding proposal through the actual CPU packet chain."""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_configuration_mutations_packet as configured
from test_contract_cleanup import load
from spry_common.canon import canonical_line as canonical,digest

PROPOSAL=HERE.parent/'proposals/runtime-binding'
configured.PROPOSAL=PROPOSAL


class RuntimeBindingPacket(configured.ConfigurationPacket):
    evidence={}

    def archive(self,name,files):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for key,raw in sorted(files.items()):
                info=tarfile.TarInfo(key);info.size=len(raw);info.mode=0o444;archive.addfile(info,io.BytesIO(raw))
        path=self.root/(name+'.tar.gz');path.write_bytes(gzip.compress(stream.getvalue(),mtime=0));return path

    def test_actual_transport_binds_new_ops_and_rejects_rollback_with_self_consistent_package_pins(self):
        package,pins,_=self.package('runtime-archive')
        files={p.relative_to(package).as_posix():p.read_bytes() for p in package.rglob('*')
               if p.is_file() and p.name!='MANIFEST-PROPOSAL.json'}
        transport=load('runtime_binding_transport',package/'review/benchmark_host_transport.py')
        archive=self.archive('runtime-source',files)
        transport.inspect_archive(archive,digest(archive.read_bytes()),digest(files['MANIFEST.json']),configuration_pins=pins)
        for variant,raw in (('changed-bytes',b'# changed operator\n'),
                             ('old-operator',self.authenticated['author/package-v2/review/ops.py'])):
            bad=dict(files);bad['review/ops.py']=raw
            manifest=json.loads(files['MANIFEST.json']);manifest['package_files']['review/ops.py']=digest(raw)
            bad['MANIFEST.json']=canonical(manifest);archive=self.archive(variant,bad)
            with self.subTest(variant=variant),self.assertRaisesRegex(ValueError,'exact MXFP4 archive source/profile extension'):
                transport.inspect_archive(archive,digest(archive.read_bytes()),digest(bad['MANIFEST.json']),configuration_pins=pins)
        self.evidence['runtime_source_archive']=dict(positive=True,rollback_refused=True,changed_bytes_refused=True,
            archive_members=len(files),owner_ceilings_changed=False,native_execution=False)

    def test_runtime_payload_changes_only_reviewed_code_and_derived_source_bindings(self):
        candidate,_,_=self.package('runtime-delta')
        parent=self.root/'configuration-control'
        configured.propose_packet.build(configured.PACKET.resolve(),parent,packet_id='e201110200000010',
            proposal=(HERE.parent/'proposals/configuration-mutations').resolve())
        baseline=parent/'package';(baseline/'MANIFEST.json').write_bytes((baseline/'MANIFEST-PROPOSAL.json').read_bytes())
        data=lambda root:{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        before,after=data(baseline),data(candidate);self.assertEqual(set(before),set(after))
        changed={name for name in before if before[name]!=after[name]}
        expected={'review/ops.py','review/isolated_scheme.py','review/mtp_contract.py','review/worker_entry.py','review/benchmark_host_transport.py',
            'review/HTTP-BINDING-control.json','review/HTTP-BINDING-mtp.json','review/BENCHMARK-SPEC2.json',
            'review/RESOURCE-MANIFEST.json','PROTECTED-SHA256.json','MANIFEST.json','MANIFEST-PROPOSAL.json'}
        self.assertEqual(changed,expected)
        self.evidence['runtime_payload_delta']=dict(changed=sorted(changed),unchanged_files=len(before)-len(changed),
            request_settings_model_stack_and_overlays_byte_identical=True,native_execution=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--metrics-reporter',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    options,remaining=parser.parse_known_args();configured.PACKET=options.packet;configured.METRICS_REPORTER=options.metrics_reporter
    result=unittest.main(argv=[sys.argv[0],*remaining],exit=False).result
    with options.report.open('xb') as output:
        output.write(canonical(dict(tests=result.testsRun,passed=result.wasSuccessful(),
            test_source_sha256=digest(Path(__file__).read_bytes()),cases=RuntimeBindingPacket.evidence,native_execution=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
