"""The actual client preserves bounded write-once fields in three-request chunks."""
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_evidence_capacity as capacity
from test_config_binding_proposal import sha
import unittest
import base64
import zlib


class ChunkedReceiptCapacity(capacity.ReceiptCapacity):
    def copy(self,*names):
        sources={'serving_client.py':'client.py','request_receipts.py':'receipts.py'}
        for name in names:shutil.copyfile(HERE.parent/'spry_evidence'/sources[name],self.root/name)

    def test_actual_client_write_read_and_inventory_preserve_bytes(self):
        fields={'requests/00-one.raw':b'one','responses/00-one.record.json':b'first',
                'requests/01-two.raw':b'two','responses/01-two.record.json':b'second'}
        for name,raw in fields.items():self.module._write_exclusive(self.block/name,raw)
        self.assertEqual(self.receipts.logical_names(self.block),set(fields))
        self.assertEqual(len(list((self.block/'receipts').glob('*.json.zlib'))),1)
        for name,raw in fields.items():self.assertEqual(self.receipts.read(self.block/name),raw)
        value,_=self.receipts.decode((self.block/'receipts/chunk-000.json.zlib').read_bytes());self.assertEqual(value['schema'],3)

    def test_chunk_cannot_own_a_different_ordinal(self):
        bundle=self.receipts.write(self.block/'responses/00-one.raw',b'one')
        value,_=self.receipts.decode(bundle.read_bytes());value['fields']['responses/03-other.raw']=value['fields'].pop('responses/00-one.raw')
        bundle.write_bytes(self.receipts.canonical(value))
        with self.assertRaisesRegex(ValueError,'ownership'):self.receipts.logical_names(self.block)

    def test_legacy_v2_file_remains_readable_and_mixed_layout_refuses(self):
        raw=b'old wire';path=self.block/'responses/00-one.raw';bundle=self.block/'receipts/00-one.json';bundle.parent.mkdir(parents=True)
        value=dict(schema=2,kind='native-http-request-receipts-v2',fields={'responses/00-one.raw':dict(bytes=len(raw),
            sha256=sha(raw),codec='base64',base64='b2xkIHdpcmU=')})
        bundle.write_bytes(self.receipts.canonical(value));self.assertEqual(self.receipts.read(path),raw)
        with self.assertRaisesRegex(ValueError,'cannot append'):self.receipts.write(path,b'new')
        (bundle.parent/'chunk-000.json.zlib').write_bytes(self.receipts.canonical(value))
        with self.assertRaisesRegex(ValueError,'mixed legacy/chunk'):self.receipts.read(path)

    def test_compressible_and_incompressible_fields_round_trip(self):
        for name,raw in (('raw',b'answer '*30000),('record.json',b'\x00\x03\x01')):
            path=self.block/('responses/00-sample.'+name)
            envelope=self.receipts.write(path,raw)
            self.assertEqual(self.receipts.read(path),raw)
        self.assertLess(envelope.stat().st_size,5000)
        value,_=self.receipts.decode(envelope.read_bytes())
        self.assertEqual({row['codec'] for row in value['fields'].values()},{'base64','zlib+base64'})

    def test_truncated_trailing_and_oversized_compressed_fields_refuse(self):
        path=self.block/'responses/00-sample.raw';envelope=self.receipts.write(path,b'x'*10000)
        original,_=self.receipts.decode(envelope.read_bytes());field='responses/00-sample.raw'
        packed=base64.b64decode(original['fields'][field]['base64'])
        for data in (packed[:-1],packed+b'extra',zlib.compress(b'x'*10001)):
            value=json.loads(self.receipts.canonical(original));value['fields'][field]['base64']=base64.b64encode(data).decode()
            with self.subTest(size=len(data)),self.assertRaises((ValueError,zlib.error)):
                self.receipts.decode(self.receipts.canonical(value))

    def test_outer_compression_truncation_trailing_data_and_bomb_refuse(self):
        path=self.receipts.write(self.block/'responses/00-sample.raw',b'original')
        raw=path.read_bytes();self.assertTrue(raw.startswith(self.receipts.CHUNK_MAGIC))
        for changed in (raw[:-1],raw+b'extra',self.receipts.CHUNK_MAGIC+zlib.compress(b'x'*(self.receipts.MEMBER_LIMIT+1))):
            with self.subTest(size=len(changed)),self.assertRaises((ValueError,zlib.error)):self.receipts.decode(changed)

    def test_receipt_digest_and_canonical_encoding_remain_required(self):
        path=self.block/'requests/00-sample.raw';envelope=self.module._write_exclusive(path,b'original')
        value,_=self.receipts.decode(envelope.read_bytes())
        with self.assertRaisesRegex(ValueError,'canonical receipt'):self.receipts.decode(json.dumps(value).encode())
        value['fields']['requests/00-sample.raw']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'bytes/hash'):self.receipts.decode(self.receipts.canonical(value))


if __name__=='__main__':unittest.main()
