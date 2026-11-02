"""Exercise real health validators with an authenticated CPU contract fixture.

The fixture is a pure metadata contract, not native runtime evidence. The
separate retained-arm gate exercises the complete reader/resource/owner chain.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Hostile(ModuleType):
    def __getattr__(self,name):raise AssertionError('ambient contract used: '+name)

@contextmanager
def ambient(hostile):
    absent=object();before=sys.modules.pop('mtp_contract',absent)
    before_path=list(sys.path)
    # Legacy fixture search paths must not supply an ambient file import.
    sys.path[:]=[path for path in before_path if not(Path(path or Path.cwd())/'mtp_contract.py').is_file()]
    trap=Hostile('mtp_contract')if hostile else None
    if hostile:sys.modules['mtp_contract']=trap
    try:yield trap
    finally:
        sys.path[:]=before_path
        if before is absent:sys.modules.pop('mtp_contract',None)
        else:sys.modules['mtp_contract']=before


class HostReadoutScope(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='host-readout-scope-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name).resolve()
        for name,source in {
            'spry_common_canon.py':'spry_common/selected.py',
            'serving_integration.py':'spry_evidence/common_host_scope_integration.py',
            'retained_integration.py':'spry_evidence/common_serving_integration.py',
            'serving_graph_observer.py':'spry_evidence/common_serving_graph_observer.py',
            'j01_probe.py':'spry_evidence/common_j01_probe.py',
            'kv_graph_join.py':'spry_evidence/common_kv_graph_join.py',
            'target_kv_metadata.py':'spry_evidence/common_target_kv_metadata.py',
            'observer_health.py':'spry_evidence/common_observer_health.py',
        }.items():(self.root/name).write_bytes((ROOT/source).read_bytes())
        self.contract=self.root/'mtp_contract.py'
        self.contract.write_text('def commit_observation_limit():return 8\n')
        self.reader=load('host_scope_candidate',self.root/'serving_integration.py')
        self.retained=load('host_scope_retained',self.root/'retained_integration.py')
        source=lambda name:dict(module=name[:-3],path='/mnt/review/'+name,sha256=digest(self.root/name))
        sources=dict(graph_observer=source('serving_graph_observer.py'),j01_probe=source('j01_probe.py'),
            mtp_contract=dict(path='/mnt/review/mtp_contract.py',sha256=digest(self.contract)))
        self.binding=dict(mode=self.reader.MODE,arm='mtp',evidence_dir=str(self.root),session_sha256='1'*64,spec_sha256='2'*64,
            source_manifest_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest(),sources=sources)
        self.graph=self.reader._gdn_observer(sources,require_gdn=False)
        self.workers=[];self.observations={}
        for rank in (0,1):
            worker=dict(rank=rank,pid=100+rank,start_ticks=500+rank,boot_id='cpu-fixture-boot',identity_sha256=str(3+rank)*64)
            self.workers.append(worker)
            for name,counters,role in [('graph',dict(captures=2,replays=2,commits=2,bs1_replays=1,bs2_replays=1),'graph_observer'),('target_kv',dict(routes=0,bytes=0),'j01_probe')]:
                health=self.graph._health.Health(name,rank);health.finish(self.binding,worker,sources[role],counters)
                filename=self.graph._health.filename(name,rank);self.observations[filename]=(self.root/filename).read_bytes()
            self.observations[f'graph-counter-rank-{rank}.bin']=struct.pack('<QQQQ',2,2,1,1)
            self.observations[f'replay-commit-counter-rank-{rank}.bin']=struct.pack('<Q',2)

    def validate(self):
        graph=self.reader._gdn_observer(self.binding['sources'],require_gdn=False)
        self.reader._validate_observer_health(self.observations,binding=self.binding,workers=self.workers,graph=graph)

    def test_retained_reader_reproduces_missing_ambient_contract(self):
        with ambient(False),self.assertRaises(ModuleNotFoundError)as error:
            self.retained._validate_observer_health(self.observations,binding=self.binding,workers=self.workers,graph=self.graph)
        self.assertEqual(error.exception.name,'mtp_contract')

    def test_actual_validators_pass_with_absent_and_hostile_ambient_contracts(self):
        for hostile in (False,True):
            with self.subTest(hostile=hostile),ambient(hostile)as trap:
                self.validate();self.assertIs(sys.modules.get('mtp_contract'),trap)

    def test_failed_health_refuses_and_restores_both_ambient_states(self):
        for observer in ('graph','target_kv'):
            name=f'OBSERVER-HEALTH-{observer}-rank-0.json';original=self.observations[name]
            row=json.loads(original);row['observer_failed']=True;self.observations[name]=self.graph.canonical(row)
            for hostile in (False,True):
                with self.subTest(observer=observer,hostile=hostile),ambient(hostile)as trap:
                    with self.assertRaisesRegex(ValueError,'incomplete or failed'):self.validate()
                    self.assertIs(sys.modules.get('mtp_contract'),trap)
            self.observations[name]=original

    def test_missing_health_receipts_refuse(self):
        for name in [name for name in self.observations if name.endswith('.json')]:
            original=self.observations.pop(name)
            with self.subTest(name=name),ambient(True)as trap:
                with self.assertRaisesRegex(ValueError,'missing or oversized'):self.validate()
                self.assertIs(sys.modules.get('mtp_contract'),trap)
            self.observations[name]=original

    def test_wrong_contract_digest_refuses_without_using_ambient_module(self):
        self.binding['sources']['mtp_contract']['sha256']='f'*64
        for hostile in (False,True):
            with ambient(hostile)as trap:
                with self.assertRaisesRegex(ValueError,'contract source changed'):self.validate()
                self.assertIs(sys.modules.get('mtp_contract'),trap)

    def test_contract_loading_exception_restores_ambient_state(self):
        self.contract.write_text('raise RuntimeError("contract fixture load failed")\n')
        self.binding['sources']['mtp_contract']['sha256']=digest(self.contract)
        for hostile in (False,True):
            with ambient(hostile)as trap:
                with self.assertRaisesRegex(RuntimeError,'contract fixture load failed'):self.validate()
                self.assertIs(sys.modules.get('mtp_contract'),trap)

    def test_wrong_graph_and_probe_digests_refuse(self):
        for role in ('graph_observer','j01_probe'):
            original=self.binding['sources'][role]['sha256'];self.binding['sources'][role]['sha256']='f'*64
            with self.subTest(role=role),ambient(True)as trap:
                with self.assertRaisesRegex(ValueError,'source'):self.validate()
                self.assertIs(sys.modules.get('mtp_contract'),trap)
            self.binding['sources'][role]['sha256']=original

    def test_graph_and_commit_counter_joins_still_refuse(self):
        for name,raw in [('graph-counter-rank-0.bin',struct.pack('<QQQQ',2,3,1,2)),('replay-commit-counter-rank-0.bin',struct.pack('<Q',3))]:
            original=self.observations[name];self.observations[name]=raw
            with self.subTest(name=name),ambient(True)as trap:
                with self.assertRaisesRegex(ValueError,'counter.*differ'):self.validate()
                self.assertIs(sys.modules.get('mtp_contract'),trap)
            self.observations[name]=original

    def test_target_kv_journal_join_still_refuses(self):
        self.observations['TARGET-KV-ROUTES-rank-0.jsonl']=b'{}\n'
        with ambient(True)as trap:
            with self.assertRaisesRegex(ValueError,'journal differs'):self.validate()
            self.assertIs(sys.modules.get('mtp_contract'),trap)


if __name__=='__main__':unittest.main()
