"""Fresh bounded GPU source packaging; no registration or native execution."""
import gzip
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import prepare_gpu_run as prepare


class GPUPackaging(unittest.TestCase):
    def members(self):return {'review/file-'+str(n)+'.py':b'# inert CPU fixture\n' for n in range(80)}

    def test_raw_and_compressed_archives_have_identical_members(self):
        members=self.members();raw=prepare.archive_bytes(members,compressed=False)
        compressed=prepare.archive_bytes(members,compressed=True)
        self.assertEqual(gzip.decompress(compressed),raw)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
            self.assertEqual({item.name:archive.extractfile(item).read() for item in archive},members)

    def test_compressed_archive_is_deterministic(self):
        self.assertEqual(prepare.archive_bytes(self.members(),compressed=True),prepare.archive_bytes(self.members(),compressed=True))

    def test_extra_member_and_parent_traversal_refuse(self):
        members=self.members();members['extra.py']=b'extra'
        with self.assertRaisesRegex(ValueError,'member ceiling'):prepare.archive_bytes(members,compressed=True)
        members=self.members();members['../outside.py']=members.pop(next(iter(members)))
        with self.assertRaisesRegex(ValueError,'unsafe'):prepare.archive_bytes(members,compressed=True)

    def test_expanded_archive_bound_applies_before_compression(self):
        members=self.members();members[next(iter(members))]=b'x'*(16<<20)
        with self.assertRaisesRegex(ValueError,'archive size'):prepare.archive_bytes(members,compressed=True)

    def test_existing_destination_is_preserved_before_donor_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'keep').write_bytes(b'user data')
            with patch.object(prepare.builder,'authenticated_packet') as authenticate:
                with self.assertRaisesRegex(ValueError,'fresh canonical'):
                    prepare.prepare(root/'donor',root,packet_id='a'*16,observed_boot_id='12345678-1234-1234-1234-123456789abc')
            authenticate.assert_not_called();self.assertEqual((root/'keep').read_bytes(),b'user data')


if __name__=='__main__':unittest.main()
