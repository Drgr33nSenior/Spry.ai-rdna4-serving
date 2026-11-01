"""Declared CPU row counts preserve the existing resource and namespace policy."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('configured_cpu_common',ROOT/'quality/common.py')
common=importlib.util.module_from_spec(spec);sys.modules[spec.name]=common;spec.loader.exec_module(common)


def package(rows):
    return dict(schema=1,mode=common.MODE,qualification=False,allow_gpu=False,model_starts=0,
        payload='tokenizer_entry.py',final_quality=dict(decoder_runs=1,rows=rows,grader_runs=0,
            input_bytes=53<<20,report_bytes=4<<20),worker_seconds=210,source_seconds=60,total_seconds=320,
        cpu_properties=['CPUQuota=200%','MemoryMax=4294967296','MemorySwapMax=0','TasksMax=128'],
        sources={name:'a'*64 for name in common.REQUIRED_SOURCES},runtime_files={'runtime.py':'b'*64},
        remote_root='/home/developer/fp8-kv-benchmark.'+'c'*16)


class QualityMaterialization(unittest.TestCase):
    def test_bounded_paired_counts_preserve_the_resource_policy(self):
        for count in (2,352,704):
            value=package(count)
            with self.subTest(count=count):self.assertIs(common.validate_contract(value),value)

    def test_boolean_odd_empty_or_excess_counts_refuse(self):
        for count in (True,False,0,-2,1,353,706,352.0,'352'):
            with self.subTest(count=count),self.assertRaises(AssertionError):common.validate_contract(package(count))

    def test_count_support_does_not_expand_native_authority_or_resources(self):
        for key,value in (('allow_gpu',True),('model_starts',1),('worker_seconds',211),('source_seconds',61),
                          ('total_seconds',321),('qualification',True)):
            changed=package(704);changed[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):common.validate_contract(changed)
        for key,value in (('decoder_runs',2),('grader_runs',1),('input_bytes',(53<<20)+1),('report_bytes',(4<<20)+1)):
            changed=package(704);changed['final_quality'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):common.validate_contract(changed)


if __name__=='__main__':unittest.main()
