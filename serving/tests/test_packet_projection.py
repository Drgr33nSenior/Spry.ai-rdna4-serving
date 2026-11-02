"""Proposal-only source selection refuses pin drift, links and unsafe destinations."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_contract_cleanup import load
from test_config_binding_proposal import canonical, sha

renderer = load('cleanup_packet_projection', HERE.parent / 'propose_packet.py')


class PacketProjection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cleanup-projection-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.proposal = self.root / 'proposals/current'; (self.proposal / 'review').mkdir(parents=True)
        self.owner = self.root / 'proposals/owner/review/benchmark_owner_adapter.py'
        self.owner.parent.mkdir(parents=True); self.owner.write_bytes(b'# exact owner source\n')
        names = ('benchmark_host_transport.py', 'benchmark_physical_controller.py', 'benchmark_artifacts.py',
            'benchmark_owner_adapter.py', 'benchmark_resource_gate.py', 'benchmark_gpu_identity.py',
            'benchmark_release_observer.py', 'collect_benchmark_platform.py', 'host_process_owner.py')
        self.donor = {name: ('# original ' + name + '\n').encode() for name in names}
        self.selected = dict(self.donor)
        for name in names[:3]:
            self.selected[name] = ('# proposed ' + name + '\n').encode()
            (self.proposal / 'review' / name).write_bytes(self.selected[name])
        self.selected[self.owner.name] = self.owner.read_bytes()
        self.metadata = dict(protected_source_files={name: sha(raw) for name, raw in self.donor.items()})
        (self.proposal / 'D2.patch').write_bytes(b'# exact CPU fixture patch\n')
        self.record = dict(protected_files={name: sha(raw) for name, raw in self.selected.items()},
            protected_set_sha256=sha(canonical({'review/' + name: sha(raw) for name, raw in self.selected.items()})),
            owner_ceilings_changed=False, launch_authorized=False, boot_independent_manifest_schema=3,
            owner_adapter_proposal='../owner/review/benchmark_owner_adapter.py',
            patch_sha256=sha((self.proposal / 'D2.patch').read_bytes()))
        self.write_record()
        active = patch.object(renderer, 'HERE', self.root); active.start(); self.addCleanup(active.stop)

    def write_record(self): (self.proposal / 'PROPOSAL.json').write_bytes(canonical(self.record))

    def select(self): return renderer.protected_sources(self.proposal, self.donor, self.metadata)

    def test_only_exact_nine_sources_are_selected_from_current_owner_and_donor_sources(self):
        selected, record, pin = self.select()
        self.assertEqual(selected, self.selected); self.assertEqual(record, self.record)
        self.assertEqual(pin, sha((self.proposal / 'PROPOSAL.json').read_bytes()))

    def test_changed_patch_or_source_cannot_be_presented_under_the_reviewed_pins(self):
        source = self.proposal / 'review/benchmark_artifacts.py'
        source.write_bytes(b'# unreviewed replacement\n')
        with self.assertRaisesRegex(ValueError, 'helper source differs'): self.select()
        source.write_bytes(self.selected[source.name])
        (self.proposal / 'D2.patch').write_bytes(b'# changed patch\n')
        with self.assertRaisesRegex(ValueError, 'proposal patch required'): self.select()

    def test_unapproved_ceiling_shape_extra_helpers_and_execution_claims_refuse(self):
        original = copy.deepcopy(self.record)
        for key, value in (('owner_ceilings_changed', True), ('launch_authorized', True),
                           ('boot_independent_manifest_schema', 2)):
            self.record = dict(original, **{key: value}); self.write_record()
            with self.subTest(key=key), self.assertRaises(ValueError): self.select()
        self.record = copy.deepcopy(original); self.record['protected_files']['extra.py'] = 'a'*64; self.write_record()
        with self.assertRaises(ValueError): self.select()

    def test_linked_proposed_helper_is_rejected_even_when_bytes_match(self):
        source = self.proposal / 'review/benchmark_artifacts.py'
        outside = self.root / 'outside.py'; outside.write_bytes(source.read_bytes())
        source.unlink(); source.symlink_to(outside)
        with self.assertRaises(ValueError): self.select()

    def test_unsafe_destinations_refuse_before_reading_or_importing_donor(self):
        donor = self.root / 'donor'; donor.mkdir()
        for destination in (self.root / 'output', donor / 'output'):
            with self.subTest(destination=destination), patch.object(renderer.source_builder, 'authenticated_packet') as authenticate:
                with self.assertRaisesRegex(ValueError, 'outside canonical source and donor'):
                    renderer.build(donor, destination, packet_id='a'*16, proposal=self.proposal)
                authenticate.assert_not_called()


if __name__ == '__main__': unittest.main()
