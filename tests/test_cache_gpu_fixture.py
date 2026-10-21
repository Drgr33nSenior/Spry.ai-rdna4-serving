"""Offline reader and source-bound syntax checks for the native cache fixture.

No HIP/SGLang/Torch import or native library load occurs. Syntax compilation uses
explicit declaration-only HIP stand-ins. A separate CPU-only harness exercises
the actual measurement code with host-memory stand-ins, never a GPU kernel.
CLI: this_file.py --compare OLD.bin NEW.bin OLD.json NEW.json
The physical owner separately binds the executable, libraries, device and exits.
"""
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "prototype/tests/bf16_cache_gpu_test.cpp"
MAX_BYTES = 32 * 1024 * 1024
TIMING_V2 = "warm-synchronized-output-v2"
EXPECTED = [
    ("decoder", 8192 * 8),
    ("protected/append-k", 8192 * 1024),
    ("protected/append-v", 8192 * 1024),
    ("protected/mixed-lengths", 9 * 6144),
    ("protected/after-refusals", 9 * 6144),
    ("compact/append-k", 8192 * 272),
    ("compact/append-v", 8192 * 272),
    ("compact/mixed-lengths", 9 * 6144),
    ("compact/after-refusals", 9 * 6144),
    ("compact/boundary-scales", 6144),
]


def _timing_contract(timing_protocol, timing_repeats, timing_warmups=None):
    if type(timing_protocol) is not str or timing_protocol not in {"legacy-v1", TIMING_V2}:
        raise ValueError("unknown timing protocol")
    if type(timing_repeats) is not int or not 0 <= timing_repeats <= 5:
        raise ValueError("requested timing bound")
    if timing_protocol == TIMING_V2 and timing_repeats != 5:
        raise ValueError("v2 timing repeat binding")
    if timing_warmups is not None and (type(timing_warmups) is not int
            or timing_warmups != (2 if timing_protocol == TIMING_V2 else 0)):
        raise ValueError("timing protocol/warmup binding")


def expected_records(mode, timing_protocol="legacy-v1", timing_repeats=0):
    if type(mode) is not str or mode not in {"both", "bf16", "kv4"}:
        raise ValueError("cache mode")
    _timing_contract(timing_protocol, timing_repeats)
    result = [EXPECTED[0]]
    prefixes = ("protected", "compact") if mode == "both" else ("protected" if mode == "bf16" else "compact",)
    for prefix in prefixes:
        result.extend((name, size) for name, size in EXPECTED if name.startswith(prefix + "/"))
        if timing_protocol == TIMING_V2:
            for length in (128, 1024, 3966):
                tags = ["reference", "warmup-0", "warmup-1"] + [f"repetition-{r}" for r in range(timing_repeats)]
                result.extend((f"{prefix}/timing/{length}/{tag}", 6144) for tag in tags)
    return result


def records(data, mode="both", timing_protocol="legacy-v1", timing_repeats=0):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES or data[:8] != b"UQBFV2G1":
        raise ValueError("fixture header/size")
    offset, result = 8, []
    for expected_name, expected_size in expected_records(mode, timing_protocol, timing_repeats):
        if offset + 12 > len(data):
            raise ValueError("missing record")
        size, payload = struct.unpack_from("<IQ", data, offset)
        offset += 12
        if not 0 < size <= 128 or payload != expected_size or offset + size + payload > len(data):
            raise ValueError("record dimensions")
        name = data[offset:offset + size].decode("ascii")
        if name != expected_name:
            raise ValueError("record order/identity")
        offset += size
        result.append((name, memoryview(data)[offset:offset + payload]))
        offset += payload
    if offset != len(data):
        raise ValueError("extra record/bytes")
    if timing_protocol == TIMING_V2:
        references = {}
        for name, payload in result:
            if "/timing/" not in name:
                continue
            context, kind = name.rsplit("/", 1)
            if kind == "reference":
                references[context] = payload
            elif payload != references[context]:
                raise ValueError("timing output differs from reference: " + name)
            for (value,) in struct.iter_unpack("<H", payload):
                if value & 0x7f80 == 0x7f80:
                    raise ValueError("nonfinite timing output: " + name)
    return result


def summary(value, *, expected_timing_repeats=0, expected_timing_protocol="legacy-v1", expected_timing_warmups=0,
            expected_phase_profile=False):
    _timing_contract(expected_timing_protocol, expected_timing_repeats, expected_timing_warmups)
    if (type(expected_phase_profile) is not bool
            or expected_phase_profile and expected_timing_protocol != TIMING_V2):
        raise ValueError("phase profile requires explicit v2 timing")
    mode = value.get("cache_mode") if isinstance(value, dict) else None
    members = expected_records(mode, expected_timing_protocol, expected_timing_repeats)
    extra_rows = (48 if mode == "both" else 24) if expected_timing_protocol == TIMING_V2 else 0
    exact = {"schema": 1, "status": "NATIVE_FIXTURE_PASSED", "qualification": False,
             "decoder_comparisons": 8192, "attention_rows": {"both": 37, "bf16": 18, "kv4": 19}[mode] + extra_rows,
             "refusals": {"both": 86, "bf16": 35, "kv4": 51}[mode], "records": len(members),
             "device_buffer_free_errors": 0}
    fields = set(exact) | {"cache_mode", "bdf", "uuid", "visible_devices", "architecture", "device",
                          "output_bytes", "peak_explicit_device_buffer_bytes", "timing_repeats", "timings"}
    if expected_timing_protocol == TIMING_V2:
        fields |= {"timing_protocol", "timing_warmups"}
    if expected_phase_profile:
        exact["phase_profile"] = True
        fields.add("phase_profile")
    if type(value) is not dict or set(value) != fields:
        raise ValueError("unexpected native fixture summary fields")
    if not isinstance(value, dict) or any(type(value.get(k)) is not type(v) or value.get(k) != v
                                          for k, v in exact.items()):
        raise ValueError("incomplete native fixture summary")
    protocol = value.get("timing_protocol", "legacy-v1")
    warmups = value.get("timing_warmups", 0)
    if (type(protocol) is not str or protocol != expected_timing_protocol
            or type(warmups) is not int or warmups != expected_timing_warmups):
        raise ValueError("unexpected timing protocol/warmups")
    if value.get("bdf") not in {"0000:c3:00.0", "0000:23:00.0"}:
        raise ValueError("unknown device")
    pairs = {"0000:c3:00.0": "39373864-6465-3038-6163-303837666533",
             "0000:23:00.0": "34663466-6561-3362-3832-346261366439"}
    if value.get("uuid") != pairs[value["bdf"]]:
        raise ValueError("device UUID binding")
    if type(value.get("visible_devices")) is not int or value["visible_devices"] != 2:
        raise ValueError("device allocation count")
    arch = value.get("architecture")
    if not isinstance(arch, str) or not (arch == "gfx1201" or arch.startswith("gfx1201:")):
        raise ValueError("device architecture")
    if type(value.get("device")) is not int or not 0 <= value["device"] < 2:
        raise ValueError("device ordinal")
    if type(value.get("output_bytes")) is not int or not 0 < value["output_bytes"] <= MAX_BYTES:
        raise ValueError("output size")
    peak = value.get("peak_explicit_device_buffer_bytes")
    if type(peak) is not int or not 0 < peak <= 64 * 1024 * 1024:
        raise ValueError("explicit device buffer bound")
    if type(value.get("timing_repeats")) is not int or value["timing_repeats"] != expected_timing_repeats:
        raise ValueError("unexpected timing count")
    modes = [True, False] if mode == "both" else [mode == "bf16"]
    cells = [(protected, length, repetition) for protected in modes
             for length in (128, 1024, 3966) for repetition in range(expected_timing_repeats)]
    timings = value.get("timings")
    if type(timings) is not list or len(timings) != len(cells):
        raise ValueError("timing cell inventory")
    previous_sequence = None
    for row, cell in zip(timings, cells):
        row_fields = {"protected", "length", "repetition", "synchronized_call_ms"}
        if expected_phase_profile:
            row_fields.add("phase_wall")
        if (type(row) is not dict or set(row) != row_fields
                or type(row["protected"]) is not bool or type(row["length"]) is not int
                or type(row["repetition"]) is not int
                or (row["protected"], row["length"], row["repetition"]) != cell
                or type(row["synchronized_call_ms"]) not in (int, float)
                or not math.isfinite(row["synchronized_call_ms"]) or row["synchronized_call_ms"] <= 0):
            raise ValueError("malformed or reordered timing cell")
        if expected_phase_profile:
            phase = row["phase_wall"]
            phase_fields = {"schema", "sequence", "host_checks_ns", "validation_status_ns", "reader_sync_ns", "total_ns"}
            if (type(phase) is not dict or set(phase) != phase_fields
                    or any(type(phase[key]) is not int or not 0 <= phase[key] < 2**64 for key in phase_fields)
                    or phase["schema"] != 1 or phase["sequence"] == 0 or phase["total_ns"] == 0
                    or sum(phase[key] for key in ("host_checks_ns", "validation_status_ns", "reader_sync_ns")) != phase["total_ns"]
                    or phase["total_ns"] > math.nextafter(row["synchronized_call_ms"] * 1_000_000, math.inf)):
                raise ValueError("malformed phase wall timing")
            if previous_sequence is not None and (phase["sequence"] <= previous_sequence
                    or row["repetition"] > 0 and phase["sequence"] != previous_sequence + 1):
                raise ValueError("stale or skipped phase wall sequence")
            previous_sequence = phase["sequence"]
    return value


def compare(old, candidate, old_summary, candidate_summary, *, expected_timing_repeats=0,
            expected_timing_protocol="legacy-v1", expected_timing_warmups=0,
            expected_candidate_phase_profile=False, expected_old_phase_profile=False):
    options = dict(expected_timing_repeats=expected_timing_repeats,
                   expected_timing_protocol=expected_timing_protocol, expected_timing_warmups=expected_timing_warmups)
    old_summary = summary(old_summary, expected_phase_profile=expected_old_phase_profile, **options)
    candidate_summary = summary(candidate_summary, expected_phase_profile=expected_candidate_phase_profile, **options)
    if old_summary["cache_mode"] != candidate_summary["cache_mode"]:
        raise ValueError("different cache modes")
    left = records(old, old_summary["cache_mode"], expected_timing_protocol, expected_timing_repeats)
    right = records(candidate, candidate_summary["cache_mode"], expected_timing_protocol, expected_timing_repeats)
    if (old_summary["bdf"], old_summary["device"]) != (candidate_summary["bdf"], candidate_summary["device"]):
        raise ValueError("different physical device")
    if old_summary["output_bytes"] != len(old) or candidate_summary["output_bytes"] != len(candidate):
        raise ValueError("summary/output size mismatch")
    differences = []
    for (name, a), (_, b) in zip(left, right):
        if a != b:
            first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            differences.append({"record": name, "first_differing_byte": first})
    if differences:
        raise ValueError("native bit mismatch: " + json.dumps(differences))
    return {"schema": 1, "status": "NATIVE_OLD_CANDIDATE_BITS_IDENTICAL", "qualification": False,
            "records": len(left), "compared_bytes": sum(len(v) for _, v in left),
            "bdf": old_summary["bdf"], "scope": "synthetic-cache-ABI-only"}


HIP_DECLARATIONS = r"""
#pragma once
#include <cstddef>
#define __host__
#define __device__
#define __global__
struct dim3 { unsigned x=0,y=0,z=0; dim3(unsigned v=0):x(v){} };
extern dim3 blockIdx,blockDim,threadIdx;
using hipError_t=int; constexpr int hipSuccess=0;
enum hipMemcpyKind {hipMemcpyHostToDevice,hipMemcpyDeviceToHost};
struct hipDeviceProp_t {char gcnArchName[256];};
struct hipUUID {char bytes[16];};
const char* hipGetErrorString(hipError_t);
hipError_t hipMalloc(void**,size_t); hipError_t hipFree(void*);
hipError_t hipMemcpy(void*,const void*,size_t,hipMemcpyKind);
hipError_t hipMemset(void*,int,size_t);
hipError_t hipGetDeviceCount(int*); hipError_t hipSetDevice(int);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*,int);
hipError_t hipDeviceGetPCIBusId(char*,int,int);
hipError_t hipDeviceGetUuid(hipUUID*,int);
hipError_t hipGetLastError(); hipError_t hipDeviceSynchronize();
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) kernel(__VA_ARGS__)
"""


CPU_TIMING_HARNESS = r"""
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
void* fixture_dlopen(const char*,int);
void* fixture_dlsym(void*,const char*);
#define dlopen fixture_dlopen
#define dlsym fixture_dlsym
#define main fixture_main
#include "SOURCE_PATH"
#undef main
#undef dlopen
#undef dlsym
dim3 blockIdx,blockDim,threadIdx;
std::string events,fault;
int attention_calls=0,device_probes=0,phase_lookups=0;
uq_cache_phase_profile_v1 phase_snapshot{};
const char* hipGetErrorString(hipError_t) {return "CPU stand-in";}
hipError_t hipMalloc(void** p,size_t n) {*p=std::malloc(n);return *p?0:1;}
hipError_t hipFree(void* p) {std::free(p);return 0;}
hipError_t hipMemcpy(void* dst,const void* src,size_t n,hipMemcpyKind kind) {
  events+=kind==hipMemcpyHostToDevice?'u':'d';std::memcpy(dst,src,n);return 0;
}
hipError_t hipMemset(void* p,int value,size_t n) {events+='r';std::memset(p,value,n);return 0;}
hipError_t hipDeviceSynchronize() {events+='s';return 0;}
hipError_t hipGetDeviceCount(int* n) {++device_probes;*n=0;return 0;}
hipError_t hipSetDevice(int) {return 1;}
hipError_t hipGetDeviceProperties(hipDeviceProp_t*,int) {return 1;}
hipError_t hipDeviceGetPCIBusId(char*,int,int) {return 1;}
hipError_t hipDeviceGetUuid(hipUUID*,int) {return 1;}
hipError_t hipGetLastError() {return 0;}
int fake_attention(const uq_bf16_cache_v2*,const uint64_t*,const uint16_t*,uint64_t,
  const int64_t*,int,const int32_t*,const int32_t* lengths,int rows,float,
  uint16_t* output,uint64_t bytes,int* status,void*) {
  events+='a';++attention_calls;
  phase_snapshot={1,uint64_t(attention_calls),1,0,1,1,1,3};
  require(rows==1&&bytes==6144,"CPU measurement shape");
  std::fill(output,output+bytes/2,uint16_t(0x3f00+lengths[0]%16));*status=0;
  if((fault=="warmup"&&attention_calls==2)||(fault=="timed"&&attention_calls==4))output[3071]^=1;
  if(attention_calls==4) {
    if(fault=="nonfinite")output[3071]=0x7f80;
    if(fault=="left-canary")output[-1]=0;
    if(fault=="right-canary")output[3072]=0;
    if(fault=="status")*status=1;
    if(fault=="status-canary")status[-1]=0;
  }
  return 0;
}
int fake_append(const uq_bf16_cache_v2*,const uint64_t*,const uint16_t*,const uint16_t*,
  uint64_t,const int64_t*,int,int*,void*) {return 0;}
int fake_phase_last(uq_cache_phase_profile_v1* value,uint64_t bytes) {
  events+='p';require(bytes==sizeof(*value),"phase ABI size");*value=phase_snapshot;
  if(fault=="phase-pre-error" || (fault=="phase-error"&&attention_calls==4))return 1;
  if(attention_calls==4) {
    if(fault=="phase-stale")--value->sequence;
    if(fault=="phase-skipped")++value->sequence;
    if(fault=="phase-schema")value->schema=2;
    if(fault=="phase-incomplete")value->complete=0;
    if(fault=="phase-result")value->api_result=1;
    if(fault=="phase-zero-sequence")value->sequence=0;
    if(fault=="phase-zero-total")value->total_ns=0;
    if(fault=="phase-sum")value->host_checks_ns=2;
    if(fault=="phase-overflow")value->host_checks_ns=UINT64_MAX;
    if(fault=="phase-enclosing") {value->reader_sync_ns=UINT64_MAX-2;value->total_ns=UINT64_MAX;}
  }
  return 0;
}
void* fixture_dlopen(const char*,int) {return reinterpret_cast<void*>(1);}
void* fixture_dlsym(void*,const char* name) {
  if(std::strcmp(name,"uq_bf16_attention_v2")==0)return reinterpret_cast<void*>(&fake_attention);
  if(std::strcmp(name,"uq_bf16_append_v2")==0)return reinterpret_cast<void*>(&fake_append);
  if(std::strcmp(name,"uq_cache_phase_profile_last_v1")==0) {
    ++phase_lookups;return fault=="phase-missing"?nullptr:reinterpret_cast<void*>(&fake_phase_last);
  }
  return nullptr;
}
int main(int argc,char** argv) {
  try {
    if(argc>1&&(std::string(argv[1])=="parse"||std::string(argv[1])=="parse-valid")) {
      const int result=fixture_main(argc-1,argv+1);
      require(device_probes==(std::string(argv[1])=="parse-valid"?1:0),
              "unexpected arguments reached HIP device discovery");return result;
    }
    require(argc==3,"CPU harness arguments");fault=argv[2];
    const bool phase_profile=fault.rfind("phase-",0)==0;
    require(fixture_dlsym(nullptr,"unknown_export")==nullptr,"unknown symbol must fail");
    Library lib("/cpu-stand-in",phase_profile);Output out(argv[1]);
    require(phase_lookups==(phase_profile?1:0),"unrequested phase getter lookup");
    for(bool protect:{true,false}) {
      Fixture fixture(lib,out,protect);events.clear();
      for(int length:{128,1024,3966})fixture.synchronized_timings(length,5,2);
      std::string expected;
      for(int context=0;context<3;++context) {
        expected+="uu";for(int call=0;call<8;++call)expected+=phase_profile&&call>=3?"urspaspdd":"ursasdd";
      }
      require(events==expected,"measurement operation order changed");
      require(fixture.accepted==24&&fixture.timing.size()==15,"actual measurement counts");
      require(fixture.phase_timing.size()==(phase_profile?15u:0u),"phase measurement counts");
      for(size_t i=0;i<fixture.phase_timing.size();++i) {
        const auto& phase=fixture.phase_timing[i];
        require(phase.sequence==uint64_t(protect?0:24)+4+(i/5)*8+i%5,"retained phase sequence");
        require(phase.host_checks_ns==1&&phase.validation_status_ns==1&&phase.reader_sync_ns==1&&phase.total_ns==3,
                "retained phase durations");
      }
    }
    require(attention_calls==48&&out.records==48,"complete output inventory");
    require(free_errors==0&&device_bytes==0,"CPU stand-in release");out.finish();
    std::cout<<"CPU_ONLY_COMPOSED_MEASUREMENT_PASSED\n";return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
"""


class CacheGPUFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = bytearray(b"UQBFV2G1")
        for name, size in EXPECTED:
            encoded = name.encode("ascii")
            data.extend(struct.pack("<IQ", len(encoded), size))
            data.extend(encoded)
            data.extend(bytes(size))
        cls.data = bytes(data)
        cls.observation = {"schema": 1, "status": "NATIVE_FIXTURE_PASSED", "qualification": False, "cache_mode": "both",
            "decoder_comparisons": 8192, "attention_rows": 37, "refusals": 86,
            "records": 10, "bdf": "0000:c3:00.0", "device": 0,
            "uuid": "39373864-6465-3038-6163-303837666533", "visible_devices": 2, "architecture": "gfx1201",
            "output_bytes": len(cls.data), "timing_repeats": 0, "timings": [],
            "peak_explicit_device_buffer_bytes": 32 * 1024 * 1024, "device_buffer_free_errors": 0}

    def test_explicit_synthetic_complete_and_bit_difference(self):
        result = compare(self.data, self.data, self.observation, self.observation)
        self.assertFalse(result["qualification"])
        bad = self.data[:-1] + b"\x01"
        with self.assertRaisesRegex(ValueError, "native bit mismatch"):
            compare(self.data, bad, self.observation, self.observation)

    def test_missing_extra_reordered_malformed_record(self):
        for bad in (self.data[:-1], self.data + b"x", b"bad" + self.data[3:],
                    self.data[:8] + struct.pack("<IQ", 129, 1) + self.data[20:],
                    self.data.replace(b"protected/append-k", b"protected/append-v", 1)):
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                records(bad)

    def test_separate_cache_modes_keep_real_arm_counts(self):
        for mode, rows, refusals in (("bf16", 18, 35), ("kv4", 19, 51)):
            selected = expected_records(mode)
            data = bytearray(b"UQBFV2G1")
            for name, size in selected:
                encoded = name.encode("ascii")
                data.extend(struct.pack("<IQ", len(encoded), size) + encoded + bytes(size))
            observation = dict(self.observation, cache_mode=mode, attention_rows=rows,
                               refusals=refusals, records=len(selected), output_bytes=len(data))
            result = compare(bytes(data), bytes(data), observation, observation)
            self.assertEqual(result["records"], len(selected))
            self.assertFalse(result["qualification"])

    def test_summary_refuses_missing_incomplete_identity_and_timing(self):
        for key in self.observation:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                summary({k: v for k, v in self.observation.items() if k != key})
        for key, value in (("qualification", True), ("refusals", 85), ("attention_rows", 36),
                           ("device", True), ("bdf", "0000:00:00.0"), ("timing_repeats", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary(dict(self.observation, **{key: value}))
        with self.assertRaisesRegex(ValueError, "different physical"):
            compare(self.data, self.data, self.observation, dict(self.observation, bdf="0000:23:00.0",
                    uuid="34663466-6561-3362-3832-346261366439"))

    def test_explicit_timing_inventory_and_uuid_refusals(self):
        rows = [{"protected": p, "length": length, "repetition": repeat, "synchronized_call_ms": 1.25}
                for p in (True, False) for length in (128, 1024, 3966) for repeat in range(5)]
        observation = dict(self.observation, timing_repeats=5, timings=rows)
        self.assertFalse(compare(self.data, self.data, observation, observation,
                                 expected_timing_repeats=5)["qualification"])
        for bad in (rows[:-1], rows + [rows[-1]], [rows[1], rows[0]] + rows[2:],
                    [dict(rows[0], synchronized_call_ms=float("nan"))] + rows[1:],
                    [dict(rows[0], synchronized_call_ms=True)] + rows[1:],
                    [dict(rows[0], protected=1)] + rows[1:]):
            with self.assertRaises(ValueError):
                summary(dict(observation, timings=bad), expected_timing_repeats=5)
        for key, value in (("uuid", "0"*36), ("visible_devices", 1), ("architecture", "gfx1100")):
            with self.assertRaises(ValueError):
                summary(dict(self.observation, **{key: value}))
        source = SOURCE.read_text()
        self.assertLess(source.index('actual_uuid==uuid'), source.index('Library lib(library,phase_profile)'))

    @staticmethod
    def make_data(members, changes=None):
        data = bytearray(b"UQBFV2G1")
        for name, size in members:
            encoded = name.encode("ascii")
            data.extend(struct.pack("<IQ", len(encoded), size) + encoded)
            data.extend((changes or {}).get(name, bytes(size)))
        return bytes(data)

    def v2_observation(self, mode, data):
        modes = (True, False) if mode == "both" else (mode == "bf16",)
        return dict(self.observation, cache_mode=mode, timing_protocol=TIMING_V2, timing_warmups=2,
                    timing_repeats=5, attention_rows={"both": 85, "bf16": 42, "kv4": 43}[mode],
                    refusals={"both": 86, "bf16": 35, "kv4": 51}[mode], output_bytes=len(data),
                    records=len(expected_records(mode, TIMING_V2, 5)),
                    timings=[{"protected": p, "length": n, "repetition": r, "synchronized_call_ms": 1.25}
                             for p in modes for n in (128, 1024, 3966) for r in range(5)])

    def test_v2_complete_outputs_real_counts_and_full_comparison(self):
        options = dict(expected_timing_repeats=5, expected_timing_protocol=TIMING_V2, expected_timing_warmups=2)
        for mode in ("both", "bf16", "kv4"):
            members = expected_records(mode, TIMING_V2, 5)
            data = self.make_data(members)
            observation = self.v2_observation(mode, data)
            result = compare(data, data, observation, observation, **options)
            self.assertFalse(result["qualification"])
            self.assertEqual(result["records"], len(members))
            self.assertEqual(len({name for name, _ in members}), len(members))
            self.assertLess(len(data), MAX_BYTES)
            # Every retained output differs in the same final value: internally
            # stable candidate outputs still must fail the baseline comparison.
            changed = {name: bytes(size - 2) + b"\x01\x00" for name, size in members if "/timing/" in name}
            candidate = self.make_data(members, changed)
            with self.assertRaisesRegex(ValueError, "native bit mismatch"):
                compare(data, candidate, observation, observation, **options)
            with self.assertRaises(ValueError):
                compare(data, data, observation, observation)

    def test_v2_missing_reordered_changed_or_nonfinite_outputs_refused(self):
        members = expected_records("kv4", TIMING_V2, 5)
        first_timed = next(i for i, (name, _) in enumerate(members) if name.endswith("repetition-0"))
        malformed = [members[:first_timed] + members[first_timed + 1:],
                     members[:first_timed] + [members[first_timed + 1], members[first_timed]] + members[first_timed + 2:]]
        for inventory in malformed:
            with self.assertRaises(ValueError):
                records(self.make_data(inventory), "kv4", TIMING_V2, 5)
        for suffix in ("warmup-0", "repetition-4"):
            name = next(name for name, _ in members if name.endswith(suffix))
            with self.assertRaisesRegex(ValueError, "differs from reference"):
                records(self.make_data(members, {name: bytes(6142) + b"\x01\x00"}), "kv4", TIMING_V2, 5)
        nonfinite = {name: bytes(size - 2) + b"\x80\x7f" for name, size in members if "/timing/" in name}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            records(self.make_data(members, nonfinite), "kv4", TIMING_V2, 5)

    def test_v2_protocol_and_warmup_binding_refusals(self):
        data = self.make_data(expected_records("kv4", TIMING_V2, 5))
        observation = self.v2_observation("kv4", data)
        options = dict(expected_timing_repeats=5, expected_timing_protocol=TIMING_V2, expected_timing_warmups=2)
        for key, value in (("timing_protocol", "legacy-v1"), ("timing_protocol", True),
                           ("timing_warmups", 0), ("timing_warmups", True), ("timing_warmups", "2")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summary(dict(observation, **{key: value}), **options)
        for missing in ("timing_protocol", "timing_warmups"):
            with self.assertRaises(ValueError):
                summary({k: v for k, v in observation.items() if k != missing}, **options)
        for replacement in ({"expected_timing_protocol": "unknown"}, {"expected_timing_warmups": 1},
                            {"expected_timing_warmups": True}, {"expected_timing_repeats": 4}):
            with self.assertRaises(ValueError):
                summary(observation, **(options | replacement))
        with self.assertRaises(ValueError):
            summary(dict(self.observation, timing_protocol=TIMING_V2, timing_warmups=2))

    def phase_observation(self, mode, data):
        observation = self.v2_observation(mode, data)
        observation["phase_profile"] = True
        for index, row in enumerate(observation["timings"]):
            row["phase_wall"] = {"schema": 1, "sequence": 4 + index + 3 * (index // 5),
                                 "host_checks_ns": 100, "validation_status_ns": 400,
                                 "reader_sync_ns": 500, "total_ns": 1000}
        return observation

    def test_candidate_phase_profile_preserves_complete_output_comparison(self):
        options = dict(expected_timing_repeats=5, expected_timing_protocol=TIMING_V2, expected_timing_warmups=2)
        for mode in ("both", "bf16", "kv4"):
            members = expected_records(mode, TIMING_V2, 5)
            data = self.make_data(members)
            baseline, candidate = self.v2_observation(mode, data), self.phase_observation(mode, data)
            result = compare(data, data, baseline, candidate, expected_candidate_phase_profile=True, **options)
            self.assertFalse(result["qualification"])
            self.assertEqual(result["records"], len(members))
            compare(data, data, candidate, candidate, expected_old_phase_profile=True,
                    expected_candidate_phase_profile=True, **options)
            with self.assertRaises(ValueError):
                compare(data, data, baseline, candidate, **options)
            with self.assertRaises(ValueError):
                compare(data, data, candidate, candidate, expected_candidate_phase_profile=True, **options)
            changed = {name: bytes(size - 2) + b"\x01\x00" for name, size in members if "/timing/" in name}
            with self.assertRaisesRegex(ValueError, "native bit mismatch"):
                compare(data, self.make_data(members, changed), baseline, candidate,
                        expected_candidate_phase_profile=True, **options)

    def test_phase_profile_exact_summary_and_row_contract(self):
        data = self.make_data(expected_records("kv4", TIMING_V2, 5))
        observation = self.phase_observation("kv4", data)
        options = dict(expected_timing_repeats=5, expected_timing_protocol=TIMING_V2,
                       expected_timing_warmups=2, expected_phase_profile=True)
        summary(observation, **options)
        for key in observation:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                summary({k: v for k, v in observation.items() if k != key}, **options)
        for value in (False, 1, "true", None):
            with self.subTest(phase_profile=value), self.assertRaises(ValueError):
                summary(dict(observation, phase_profile=value), **options)
        for value in (True, False, 1):
            with self.subTest(legacy_flag=value), self.assertRaises(ValueError):
                summary(dict(self.observation, phase_profile=value))
        for expected in (1, "true", None):
            with self.assertRaises(ValueError):
                summary(observation, **(options | {"expected_phase_profile": expected}))
        with self.assertRaises(ValueError):
            summary(self.observation, expected_phase_profile=True)
        for target, arguments in ((self.observation, {}), (observation, options)):
            with self.assertRaises(ValueError):
                summary(dict(target, unexpected=True), **arguments)
        first = observation["timings"][0]
        for malformed in ({k: v for k, v in first.items() if k != "phase_wall"}, dict(first, extra=True)):
            with self.assertRaises(ValueError):
                summary(dict(observation, timings=[malformed] + observation["timings"][1:]), **options)
        ordinary = self.v2_observation("kv4", data)
        ordinary["timings"][0]["phase_wall"] = first["phase_wall"]
        with self.assertRaises(ValueError):
            summary(ordinary, **(options | {"expected_phase_profile": False}))

    def test_phase_profile_integer_bounds_sums_and_fresh_sequences(self):
        data = self.make_data(expected_records("kv4", TIMING_V2, 5))
        observation = self.phase_observation("kv4", data)
        options = dict(expected_timing_repeats=5, expected_timing_protocol=TIMING_V2,
                       expected_timing_warmups=2, expected_phase_profile=True)
        first = observation["timings"][0]
        valid = first["phase_wall"]
        malformed = [{k: v for k, v in valid.items() if k != key} for key in valid]
        malformed += [dict(valid, extra=1), [], None]
        for key in valid:
            malformed += [dict(valid, **{key: value}) for value in (True, False, -1, 1.5, "1", 2**64)]
        malformed += [dict(valid, schema=2), dict(valid, sequence=0), dict(valid, host_checks_ns=101),
                      dict(valid, host_checks_ns=0, validation_status_ns=0, reader_sync_ns=0, total_ns=0),
                      dict(valid, host_checks_ns=0, validation_status_ns=0, reader_sync_ns=1_250_001, total_ns=1_250_001)]
        for phase in malformed:
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                summary(dict(observation, timings=[dict(first, phase_wall=phase)] + observation["timings"][1:]), **options)
        # Zero-duration component phases and exact enclosing duration are valid.
        boundary = dict(valid, host_checks_ns=0, validation_status_ns=0, reader_sync_ns=1_250_000, total_ns=1_250_000)
        summary(dict(observation, timings=[dict(first, phase_wall=boundary)] + observation["timings"][1:]), **options)
        for index, sequence in ((1, 4), (1, 6), (5, 8)):
            rows = [dict(row, phase_wall=dict(row["phase_wall"])) for row in observation["timings"]]
            rows[index]["phase_wall"]["sequence"] = sequence
            with self.assertRaisesRegex(ValueError, "sequence"):
                summary(dict(observation, timings=rows), **options)

    def test_v2_source_timer_excludes_preparation_and_validation(self):
        source = SOURCE.read_text().split("void synchronized_timings(", 1)[1].split("void refusal(", 1)[0]
        measured = source.split("for(unsigned r=0;r<repeats;++r)", 1)[1]
        preparation, interval = measured.split("const auto begin=", 1)
        interval, validation = interval.split("require(result==0", 1)
        self.assertLess(preparation.index("reset_status()"), preparation.index("reset_output()"))
        self.assertLess(preparation.index("reset_output()"), preparation.index("hipDeviceSynchronize()"))
        self.assertLess(preparation.index("hipDeviceSynchronize()"), preparation.index("lib.phase()"))
        self.assertLess(interval.index("attention(1)"), interval.index("hipDeviceSynchronize()"))
        self.assertLess(interval.index("hipDeviceSynchronize()"), interval.index("const double ms="))
        for operation in ("reset_status", "reset_output", "retain_output", "check_status", "out.record", ".get()", "require(", "lib.phase()"):
            self.assertNotIn(operation, interval)
        self.assertIn("lib.phase()", validation)
        self.assertIn("retain_output", validation)
        self.assertIn("==reference", validation)

    def test_actual_cpp_cpu_composed_timing_and_cli_refusals(self):
        compiler = shutil.which("clang++")
        self.assertIsNotNone(compiler, "installed clang++ required; do not install for this fixture")
        with tempfile.TemporaryDirectory(prefix="cache-cpu-measurement-") as tmp:
            root = Path(tmp)
            (root / "hip").mkdir()
            (root / "hip/hip_runtime.h").write_text(HIP_DECLARATIONS)
            harness = root / "cpu_measurement.cpp"
            harness.write_text(CPU_TIMING_HARNESS.replace("SOURCE_PATH", str(SOURCE)))
            executable = root / "cpu_measurement"
            command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-D__HIPCC__",
                       "-I", str(root), str(harness), "-o", str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            path = root / "complete.bin"
            run = subprocess.run([str(executable), str(path), "none"], capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("CPU_ONLY_COMPOSED_MEASUREMENT_PASSED", run.stdout)
            phase_path = root / "phase-complete.bin"
            run = subprocess.run([str(executable), str(phase_path), "phase-valid"],
                                 capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("CPU_ONLY_COMPOSED_MEASUREMENT_PASSED", run.stdout)
            self.assertEqual(phase_path.read_bytes(), path.read_bytes())
            # Parse every actually retained timing vector from the CPU harness.
            data, offset = path.read_bytes(), 8
            timing_members = [(name, size) for name, size in expected_records("both", TIMING_V2, 5) if "/timing/" in name]
            for name, size in timing_members:
                name_size, payload_size = struct.unpack_from("<IQ", data, offset)
                offset += 12
                self.assertEqual(data[offset:offset + name_size].decode("ascii"), name)
                self.assertEqual(payload_size, size)
                offset += name_size
                length = int(name.split("/")[2])
                self.assertEqual(data[offset:offset + size], struct.pack("<H", 0x3f00 + length % 16) * 3072)
                offset += size
            self.assertEqual(offset, len(data))
            for fault, message in (("warmup", "warmup output changed"), ("timed", "timed output changed"),
                                   ("nonfinite", "nonfinite"), ("left-canary", "left output canary"),
                                   ("right-canary", "right output canary"), ("status", "status mismatch"),
                                   ("status-canary", "status canary"), ("phase-missing", "missing phase profile export"),
                                   ("phase-pre-error", "getter failed"), ("phase-error", "getter failed"),
                                   ("phase-stale", "phase profile sequence"), ("phase-skipped", "phase profile sequence"),
                                   ("phase-schema", "invalid phase profile snapshot"),
                                   ("phase-incomplete", "invalid phase profile snapshot"),
                                   ("phase-result", "invalid phase profile snapshot"),
                                   ("phase-zero-sequence", "invalid phase profile snapshot"),
                                   ("phase-zero-total", "invalid phase profile durations"),
                                   ("phase-sum", "invalid phase profile durations"),
                                   ("phase-overflow", "invalid phase profile durations"),
                                   ("phase-enclosing", "exceeds enclosing call")):
                result = subprocess.run([str(executable), str(root / (fault + ".bin")), fault],
                                        capture_output=True, text=True, timeout=20, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
            base = [str(executable), "parse", "--execute-gpu", "--library", "/unused", "--output", "/unused",
                    "--expected-bdf", "unused", "--expected-uuid", "unused", "--device", "0"]
            valid_v2 = ["--timing-protocol", TIMING_V2, "--timing-warmups", "2", "--timing-repeats", "5"]
            refused = [["--timing-protocol", "unknown"], ["--timing-warmups", "2"],
                       ["--timing-protocol", TIMING_V2], valid_v2 + ["--timing-protocol", TIMING_V2],
                       valid_v2 + ["--timing-warmups", "2"],
                       ["--timing-repeats", "0"] + valid_v2, ["--phase-profile"],
                       ["--phase-profile", "--timing-repeats", "5"],
                       valid_v2 + ["--phase-profile", "--phase-profile"],
                       valid_v2 + ["--phase-profile", "true"]]
            refused.extend(["--timing-protocol", TIMING_V2, "--timing-repeats", "5", "--timing-warmups", value]
                           for value in ("0", "1", "3", "-2", "+2", " 2", "2.0", "2junk"))
            refused.append(["--timing-protocol", TIMING_V2, "--timing-warmups", "2", "--timing-repeats", "4"])
            for arguments in refused:
                result = subprocess.run(base + arguments, capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertNotIn("reached HIP", result.stderr)
            for flags in ([], ["--phase-profile"]):
                valid_base = [base[0], "parse-valid"] + base[2:]
                result = subprocess.run(valid_base + valid_v2 + flags, capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("exact two-device allocation required", result.stderr)
                self.assertNotIn("reached HIP", result.stderr)

    def test_actual_cpp_syntax_with_declared_hip_boundary(self):
        compiler = shutil.which("clang++")
        self.assertIsNotNone(compiler, "installed clang++ required; do not install for this fixture")
        with tempfile.TemporaryDirectory(prefix="cache-gpu-syntax-") as tmp:
            root = Path(tmp)
            (root / "hip").mkdir()
            (root / "hip/hip_runtime.h").write_text(HIP_DECLARATIONS)
            command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                       "-D__HIPCC__", "-I", str(root), str(SOURCE)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--compare":
        paths = list(map(Path, sys.argv[2:]))
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError("nonregular or oversized fixture input")
        print(json.dumps(compare(paths[0].read_bytes(), paths[1].read_bytes(),
                                 json.loads(paths[2].read_text()), json.loads(paths[3].read_text())), sort_keys=True))
    else:
        unittest.main()
