"""Hash-pinned local protected-source references without copied review modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import propose_packet

class ProtectedSourceReferences(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='protected-source-refs-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.proposal=self.root/'proposals/new';self.parent=self.root/'proposals/parent'
        self.proposal.mkdir(parents=True);(self.parent/'review').mkdir(parents=True)
        self.sha=lambda raw:hashlib.sha256(raw).hexdigest()
        self.names=('benchmark_artifacts.py','benchmark_gpu_identity.py','benchmark_host_transport.py',
            'benchmark_owner_adapter.py','benchmark_physical_controller.py','benchmark_release_observer.py',
            'benchmark_resource_gate.py','collect_benchmark_platform.py','host_process_owner.py')
        self.donor={name:('# donor '+name+'\n').encode() for name in self.names}
        self.selected=dict(self.donor);self.role='benchmark_physical_controller.py';self.raw=b'# unchanged parent source\n'
        self.selected[self.role]=self.raw;self.source=self.parent/'review'/self.role;self.source.write_bytes(self.raw)
        pins={name:self.sha(raw) for name,raw in self.selected.items()}
        protected=propose_packet.canonical({'review/'+name:pin for name,pin in pins.items()})
        self.record=dict(protected_files=pins,owner_ceilings_changed=False,launch_authorized=False,
            boot_independent_manifest_schema=3,patch_sha256=self.sha(b''),protected_set_sha256=self.sha(protected),
            protected_source_references={self.role:'../parent/review/'+self.role})
        (self.proposal/'D2.patch').write_bytes(b'');self.metadata=dict(protected_source_files={name:self.sha(raw) for name,raw in self.donor.items()})
        self.context=patch.object(propose_packet,'HERE',self.root);self.context.start();self.addCleanup(self.context.stop)
    def select(self):
        (self.proposal/'PROPOSAL.json').write_text(json.dumps(self.record))
        return propose_packet.protected_sources(self.proposal,self.donor,self.metadata)[0]
    def test_reference_returns_exact_parent_bytes_without_a_new_copy(self):
        self.assertEqual(self.select(),self.selected);self.assertFalse((self.proposal/'review').exists())
    def test_changed_missing_and_linked_source_refuse(self):
        self.source.write_bytes(b'# substituted source\n')
        with self.assertRaisesRegex(ValueError,'proposed helper source differs'):self.select()
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):self.select()
        original=self.parent/'original.py';original.write_bytes(self.raw);self.source.symlink_to(original)
        with self.assertRaisesRegex(ValueError,'canonical input path'):self.select()
    def test_absolute_escape_and_different_role_refuse(self):
        for value in (str(self.source),'../../outside/'+self.role,'../parent/review/other.py'):
            with self.subTest(reference=value):
                self.record['protected_source_references']={self.role:value}
                with self.assertRaisesRegex(ValueError,'relative protected|local proposal source role'):self.select()
    def test_unknown_reference_role_and_invalid_shape_refuse(self):
        for value in (None,[],{self.role:None},{'unknown.py':'../parent/review/unknown.py'}):
            with self.subTest(references=value):
                self.record['protected_source_references']=value
                with self.assertRaisesRegex(ValueError,'relative protected source references'):self.select()
    def test_a_copy_and_reference_cannot_silently_compete(self):
        (self.proposal/'review').mkdir();(self.proposal/'review'/self.role).write_bytes(self.raw)
        with self.assertRaisesRegex(ValueError,'both a copy and a reference'):self.select()
        del self.record['protected_source_references'];self.assertEqual(self.select(),self.selected)
    def test_references_preserve_protected_set_and_ceiling_checks(self):
        self.record['protected_set_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'protected set differs'):self.select()
        self.record['owner_ceilings_changed']=True
        with self.assertRaisesRegex(ValueError,'exact nine-helper'):self.select()

if __name__=='__main__':unittest.main()
