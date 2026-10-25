"""Portable journal representation tests; no model, network or GPU evidence."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from journal_codec import write_observation,expand_observations
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def row(ordinal,phase):
    return dict(schema=1,mode='native-http-fp8-sprint01-v1',arm='control',ordinal=ordinal,phase=phase,
        case='betterbench-'+str(ordinal),body_sha256=hashlib.sha256(str(ordinal).encode()).hexdigest(),
        snapshot=dict(ranks=[dict(rank=rank,replays=2*ordinal+(phase=='after'))for rank in(0,1)]))
class JournalTests(unittest.TestCase):
    def test_all53_pairs_preserve106_canonical_records_and_physical_hashes(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve()
            for ordinal in range(53):
                write_observation(root,row(ordinal,'before'))
                path=root/f'request-{ordinal:02d}.jsonl';before=path.read_bytes();inode=path.stat().st_ino
                self.assertEqual(len(before.splitlines()),1)
                self.assertEqual(write_observation(root,row(ordinal,'after')),row(ordinal,'before'))
                self.assertEqual(inode,path.stat().st_ino);self.assertTrue(path.read_bytes().startswith(before))
            observed={p.name:p.read_bytes()for p in root.iterdir()};self.assertEqual(len(observed),53)
            expanded,storage=expand_observations(observed);self.assertEqual(len(expanded),106)
            for ordinal in range(53):
                for phase in('before','after'):
                    name=f'request-{ordinal:02d}-{phase}.json';self.assertEqual(expanded[name],canonical(row(ordinal,phase)))
            self.assertTrue(storage['lossless']);self.assertFalse(storage['protected_export_limits_changed'])
    def test_missing_duplicate_reordered_mixed_and_truncated_records_refuse(self):
        good={f'request-{i:02d}.jsonl':b''.join(canonical(row(i,p))+b'\n'for p in('before','after'))for i in range(53)}
        for kind in('missing','duplicate','order','mixed','truncated','wrong-ordinal'):
            bad=dict(good);name='request-00.jsonl';lines=bad[name].splitlines(keepends=True)
            if kind=='missing':bad.pop(name)
            if kind=='duplicate':bad[name]+=lines[1]
            if kind=='order':bad[name]=b''.join(reversed(lines))
            if kind=='mixed':bad['request-00-before.json']=lines[0][:-1]
            if kind=='truncated':bad[name]=bad[name][:-1]
            if kind=='wrong-ordinal':bad[name]=canonical(row(1,'before'))+b'\n'+lines[1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):expand_observations(bad)
    def test_duplicate_append_oversized_and_symlink_writes_refuse(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();write_observation(root,row(0,'before'));write_observation(root,row(0,'after'))
            saved=(root/'request-00.jsonl').read_bytes()
            with self.assertRaises(ValueError):write_observation(root,row(0,'after'))
            self.assertEqual(saved,(root/'request-00.jsonl').read_bytes())
            with self.assertRaises(ValueError):write_observation(root,dict(row(1,'before'),padding='x'*8192))
            self.assertFalse((root/'request-01.jsonl').exists())
            target=root/'target';target.write_bytes(b'preserved');os.symlink(target,root/'request-02.jsonl')
            with self.assertRaises(OSError):write_observation(root,row(2,'after'))
            self.assertEqual(target.read_bytes(),b'preserved')
if __name__=='__main__':unittest.main(verbosity=2)
