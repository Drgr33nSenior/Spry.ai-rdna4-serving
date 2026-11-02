"""Bounded lossless receipt compression and authenticated compressed source input."""
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_import_cleanup as import_tests
import test_transport_binding_proposal as transport_tests
from test_config_binding_proposal import sha

PROPOSAL=HERE.parent/'proposals/evidence-capacity'
spec=importlib.util.spec_from_file_location('capacity_transport',PROPOSAL/'review/benchmark_host_transport.py')
transport=importlib.util.module_from_spec(spec);sys.modules[spec.name]=transport;spec.loader.exec_module(transport)


class ReceiptCapacity(import_tests.ReceiptModules):
    def test_all_group_records_share_one_physical_file_and_preserve_logical_bytes(self):
        for number in range(96):
            path=self.block/f'controls/group-measured-{number}.json'
            raw=self.receipts.canonical(dict(ordinal=number,wall_seconds=number+1))
            self.receipts.write(path,raw)
            self.assertEqual(self.receipts.read(path),raw)
        files=[p for p in self.block.rglob('*') if p.is_file()]
        self.assertEqual(len(files),1)
        self.assertEqual(len(self.receipts.logical_names(self.block)),96)
        with self.assertRaisesRegex(ValueError,'write-once'):
            self.receipts.write(self.block/'controls/group-measured-0.json',b'replacement')

    def test_compressible_and_incompressible_fields_round_trip(self):
        for name,raw in (('raw',b'answer '*30000),('record.json',b'\x00\x03\x01')):
            path=self.block/('responses/00-sample.'+name)
            envelope=self.receipts.write(path,raw)
            self.assertEqual(self.receipts.read(path),raw)
        self.assertLess(envelope.stat().st_size,5000)
        value=json.loads(envelope.read_bytes())
        self.assertEqual({row['codec'] for row in value['fields'].values()},{'base64','zlib+base64'})

    def test_truncated_trailing_and_oversized_compressed_fields_refuse(self):
        path=self.block/'responses/00-sample.raw'
        envelope=self.receipts.write(path,b'x'*10000)
        original=json.loads(envelope.read_bytes());field='responses/00-sample.raw'
        packed=base64.b64decode(original['fields'][field]['base64'])
        for data in (packed[:-1],packed+b'extra',zlib.compress(b'x'*10001)):
            value=json.loads(self.receipts.canonical(original))
            value['fields'][field]['base64']=base64.b64encode(data).decode()
            with self.subTest(size=len(data)),self.assertRaises((ValueError,zlib.error)):
                self.receipts.decode(self.receipts.canonical(value))

    def test_control_container_cannot_own_request_fields(self):
        envelope=self.receipts.write(self.block/'controls/group-one.json',b'{}')
        value=json.loads(envelope.read_bytes())
        value['fields']['requests/00-sample.raw']=value['fields'].pop('controls/group-one.json')
        with self.assertRaisesRegex(ValueError,'ownership'):
            self.receipts.decode(self.receipts.canonical(value))

    def test_legacy_request_envelope_remains_readable(self):
        raw=b'original wire bytes'
        value=dict(schema=1,kind='native-http-request-receipts-v1',fields={
            'responses/00-sample.raw':dict(bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode())})
        self.assertEqual(self.receipts.decode(self.receipts.canonical(value))[1]['responses/00-sample.raw'],raw)


class SourceArchiveCapacity(unittest.TestCase):
    def setUp(self):
        self.fixture=transport_tests.TransportBinding()
        with patch.object(transport_tests,'transport',transport),patch.object(transport_tests,'PROPOSAL',PROPOSAL):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_gzip_archive_retains_every_original_authenticated_member(self):
        path,pin=self.fixture.archive()
        expected=transport.inspect_archive(path,pin,self.fixture.manifest_pin,configuration_pins=self.fixture.pins)
        packed=path.with_name('compressed.tar');packed.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
        self.assertEqual(transport.inspect_archive(packed,sha(packed.read_bytes()),self.fixture.manifest_pin,
            configuration_pins=self.fixture.pins),expected)

    def test_compressed_bomb_truncation_and_trailing_bytes_refuse(self):
        path,_=self.fixture.archive();packed=gzip.compress(path.read_bytes(),mtime=0)
        for number,raw in enumerate((packed[:-1],packed+b'extra',gzip.compress(b'x'*(transport.ARCHIVE_LIMIT+1),mtime=0))):
            target=path.with_name(f'invalid-{number}.tar');target.write_bytes(raw)
            with self.subTest(number=number),self.assertRaisesRegex(ValueError,'bounded complete gzip'):
                transport.inspect_archive(target,sha(raw),self.fixture.manifest_pin,configuration_pins=self.fixture.pins)

    def test_compression_cannot_hide_a_link(self):
        import tarfile
        path,_=self.fixture.archive(special=('review/linked.py',tarfile.SYMTYPE))
        raw=gzip.compress(path.read_bytes(),mtime=0);target=path.with_name('linked.tar');target.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'unsafe, linked'):
            transport.inspect_archive(target,sha(raw),self.fixture.manifest_pin,configuration_pins=self.fixture.pins)


if __name__=='__main__':unittest.main()
