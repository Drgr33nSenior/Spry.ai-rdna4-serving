"""CPU-only source-bound projection checks; no HIP compiler, library, or GPU.

The syntax check supplies declaration-only HIP headers. The execution check
extracts the actual original/projected host attention bodies, actual host
predicates, uq_device, finish, checked_status, and the projected helper. Its
declared substitutions are HIP functions/launches that record operations and
a deterministic clock replacing only the helper's steady_clock type. Device
kernels are never executed; this cannot establish native or GPU readiness.
"""
from pathlib import Path
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype/scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import profile_cache_source as projection
from test_grouped_cache_reader import (
    FLAGS, HIP_HALF_DECLARATIONS, HIP_RUNTIME_DECLARATIONS, block,
)

SOURCE = ROOT / "prototype/runtime/attention.hip"
HEADER = ROOT / "prototype/runtime/cache_phase_profile.h"

HOST_STUBS = r'''
#include "uq.h"
#include "cache_phase_profile.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct cpu_clock {
    using time_point = std::chrono::steady_clock::time_point;
    static constexpr bool is_steady = true;
    static thread_local int64_t ticks;
    static time_point now() {
        ticks += 10;
        return time_point(std::chrono::nanoseconds(ticks));
    }
};
thread_local int64_t cpu_clock::ticks = 0;

struct dim3 {
    unsigned x,y,z;
    dim3(unsigned a=1,unsigned b=1,unsigned c=1) : x(a),y(b),z(c) {}
};
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipFailure };
enum hipMemcpyKind { hipMemcpyDeviceToHost };
struct hipDeviceProp_t { char gcnArchName[256]; };
enum class Fault { none, device, properties, architecture, reset, validate_launch,
    validate_sync, copy, refused, read_launch, read_sync };
thread_local Fault fault = Fault::none;
thread_local std::vector<std::string> operations;
thread_local std::string last_kernel;
hipError_t hipGetDevice(int* device) {
    operations.push_back("device"); *device=0;
    return fault==Fault::device ? hipFailure : hipSuccess;
}
hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties,int) {
    operations.push_back("properties");
    std::strcpy(properties->gcnArchName,fault==Fault::architecture ? "gfx1100" : "gfx1201");
    return fault==Fault::properties ? hipFailure : hipSuccess;
}
hipError_t hipMemsetAsync(void*,int value,std::size_t bytes,hipStream_t) {
    assert(value==0 && bytes==4); operations.push_back("reset");
    return fault==Fault::reset ? hipFailure : hipSuccess;
}
void trace_launch(const char* name,dim3 grid,dim3 threads,int shared,hipStream_t) {
    assert(grid.x==1 && grid.z==1 && threads.y==1 && threads.z==1 && shared==0);
    last_kernel=name;
    if(last_kernel=="validate_read") assert(grid.y==1 && threads.x==256);
    else if(last_kernel=="read") assert(grid.y==12 && threads.x==32);
    else { assert(last_kernel=="read_grouped"); assert(grid.y==2 && threads.x==192); }
    operations.push_back(last_kernel);
}
#define hipLaunchKernelGGL(kernel,grid,threads,shared,stream,...) \
    trace_launch(#kernel,grid,threads,shared,stream)
hipError_t hipGetLastError() {
    operations.push_back("last_error");
    return ((fault==Fault::validate_launch && last_kernel=="validate_read") ||
            (fault==Fault::read_launch && last_kernel!="validate_read")) ? hipFailure : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
    operations.push_back("sync");
    return ((fault==Fault::validate_sync && last_kernel=="validate_read") ||
            (fault==Fault::read_sync && last_kernel!="validate_read")) ? hipFailure : hipSuccess;
}
hipError_t hipMemcpy(void* destination,const void*,std::size_t bytes,hipMemcpyKind kind) {
    assert(bytes==4 && kind==hipMemcpyDeviceToHost); operations.push_back("copy");
    *static_cast<int*>(destination)=fault==Fault::refused ? 1 : 0;
    return fault==Fault::copy ? hipFailure : hipSuccess;
}
'''

HOST_CASES = r'''
using Attention = decltype(&uq_bf16_attention_v2);
int invoke(Attention call,bool protect,int invalid) {
    alignas(8) std::array<uint8_t,4096> k{},v{};
    alignas(8) std::array<uint16_t,3072> q{},out{};
    int64_t indices[2]={1,2}; int32_t starts[1]={0},lengths[1]={2}; int status=0;
    uint64_t identity[4]={1,2,3,4};
    uq_bf16_cache_v2 cache{k.data(),v.data(),uint64_t(4*2*(protect ? 512 : 136)),{1,2,3,4},4,unsigned(protect)};
    const uq_bf16_cache_v2* c=&cache; const uint16_t* query=q.data();
    uint16_t* output=out.data(); const int64_t* locations=indices;
    int* state=&status; int queries=1,count=2; uint64_t qbytes=6144,outbytes=6144;
    float scale=0.0625f;
    switch(invalid) {
    case 1: c=nullptr;break;
    case 2: identity[0]=7;break;
    case 3: cache.bytes_per_buffer++;break;
    case 4: queries=0;break;
    case 5: queries=4097;break;
    case 6: count=0;break;
    case 7: count=1048577;break;
    case 8: qbytes--;break;
    case 9: outbytes--;break;
    case 10: scale=NAN;break;
    case 11: scale=0;break;
    case 12: scale=2;break;
    case 13: query=nullptr;break;
    case 14: query=reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(q.data())+1);break;
    case 15: output=reinterpret_cast<uint16_t*>(k.data());break;
    case 16: state=reinterpret_cast<int*>(out.data());break;
    case 17: locations=reinterpret_cast<int64_t*>(k.data());break;
    case 18: output=q.data();break;
    case 19: state=reinterpret_cast<int*>(q.data());break;
    case 20: cache.protected_layer=2;break;
    }
    return call(c,identity,query,qbytes,locations,count,starts,lengths,queries,scale,output,outbytes,state,nullptr);
}

void run_case(bool protect,Fault error,int invalid,uint64_t& sequence) {
    fault=error; operations.clear();last_kernel.clear();
    const int expected=invoke(baseline_attention,protect,invalid);
    const auto baseline_operations=operations;
    operations.clear();last_kernel.clear();cpu_clock::ticks=0;
    const int actual=invoke(uq_bf16_attention_v2,protect,invalid);
    assert(actual==expected && operations==baseline_operations);
    uq_cache_phase_profile_v1 record{};
    const auto calls=operations;
    const auto ticks=cpu_clock::ticks;
    assert(uq_cache_phase_profile_last_v1(&record,sizeof(record))==0);
    assert(operations==calls && cpu_clock::ticks==ticks);
    assert(record.schema==1 && record.sequence==++sequence && record.api_result==actual);
    assert(record.complete==uint64_t(actual==0));
    assert(record.host_checks_ns+record.validation_status_ns+record.reader_sync_ns==record.total_ns);
    assert(record.host_checks_ns==10);
    if(invalid || error==Fault::device || error==Fault::properties || error==Fault::architecture) {
        assert(record.validation_status_ns==0 && record.reader_sync_ns==0 && record.total_ns==10);
    } else if(error==Fault::reset || error==Fault::validate_launch || error==Fault::validate_sync ||
              error==Fault::copy || error==Fault::refused) {
        assert(record.validation_status_ns==10 && record.reader_sync_ns==0 && record.total_ns==20);
    } else {
        assert(record.validation_status_ns==10 && record.reader_sync_ns==10 && record.total_ns==30);
    }
    if(actual==0) {
        const std::vector<std::string> exact={"device","properties","reset","validate_read",
            "last_error","sync","copy",protect ? "read" : "read_grouped","last_error","sync"};
        assert(operations==exact);
    }
    uq_cache_phase_profile_v1 same{};
    assert(uq_cache_phase_profile_last_v1(&same,sizeof(same))==0);
    assert(std::memcmp(&record,&same,sizeof(record))==0);
    same.schema=99;
    assert(uq_cache_phase_profile_last_v1(nullptr,sizeof(same))==1);
    assert(uq_cache_phase_profile_last_v1(&same,sizeof(same)-1)==1);
    assert(uq_cache_phase_profile_last_v1(&same,sizeof(same)+1)==1);
    alignas(8) unsigned char unaligned[sizeof(same)+1]{};
    assert(uq_cache_phase_profile_last_v1(reinterpret_cast<uq_cache_phase_profile_v1*>(unaligned+1),sizeof(same))==1);
    assert(same.schema==99 && operations==calls && cpu_clock::ticks==ticks);
}

int main() {
    uq_cache_phase_profile_v1 record{};
    assert(uq_cache_phase_profile_last_v1(&record,sizeof(record))==1);
    uint64_t sequence=0;
    for(bool protect : {false,true}) {
        for(Fault error : {Fault::none,Fault::device,Fault::properties,Fault::architecture,Fault::reset,
                Fault::validate_launch,Fault::validate_sync,Fault::copy,Fault::refused,Fault::read_launch,Fault::read_sync})
            run_case(protect,error,0,sequence);
        for(int invalid=1;invalid<=20;++invalid) run_case(protect,Fault::none,invalid,sequence);
        // An invalid call must replace, rather than expose, the preceding success.
        run_case(protect,Fault::none,0,sequence);
        run_case(protect,Fault::none,1,sequence);
    }
    std::thread other([] {
        uq_cache_phase_profile_v1 fresh{};
        assert(uq_cache_phase_profile_last_v1(&fresh,sizeof(fresh))==1);
        uint64_t independent=0;
        run_case(false,Fault::none,0,independent);
        assert(independent==1);
    });
    other.join();
    assert(uq_cache_phase_profile_last_v1(&record,sizeof(record))==0 && record.sequence==sequence);
    // Missing either boundary cannot produce a complete record, even on result 0.
    for(int missing=0;missing<2;++missing) {
        uq_cache_phase_detail::Scope incomplete;
        if(missing==0) incomplete.validation_done(); else incomplete.host_done();
        assert(incomplete.finish(0)==0);
        assert(uq_cache_phase_profile_last_v1(&record,sizeof(record))==0);
        assert(record.sequence==++sequence && record.api_result==0 && record.complete==0);
    }
    uq_cache_phase_detail::sequence=UINT64_MAX;
    uq_cache_phase_detail::Scope exhausted;
    exhausted.host_done();exhausted.validation_done();assert(exhausted.finish(0)==0);
    assert(uq_cache_phase_profile_last_v1(&record,sizeof(record))==1);
}
'''


def host_execution_source(source, candidate):
    original = source.decode("utf-8")
    candidate = candidate.decode("utf-8")
    predicates = original[original.index("namespace bf16_v2 {"):
                          original.index("__global__ void validate_write(")] + "}\n"
    functions = [block(original, signature)[0] for signature in (
        "int finish(hipStream_t stream)", "int checked_status(int* status, hipStream_t stream)",
        'extern "C" int uq_device()')]
    baseline = block(original, projection.SIGNATURE.decode())[0].replace(
        "uq_bf16_attention_v2", "baseline_attention", 1)
    projected = block(candidate, projection.SIGNATURE.decode())[0]
    helper = projection.HELPER.decode().replace("std::chrono::steady_clock", "cpu_clock")
    return "\n".join([HOST_STUBS, helper, predicates, *functions, baseline, projected, HOST_CASES])


class CachePhaseProjectionTests(unittest.TestCase):
    def test_pinned_reversible_deterministic_projection(self):
        source = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), projection.PINNED_SOURCE_SHA256)
        candidate = projection.project_source(source)
        self.assertEqual(projection.project_source(source), candidate)
        self.assertEqual(projection.recover_source(candidate), source)
        # Every original source byte, including device arithmetic, is recoverable.
        self.assertIn(b"using Clock = std::chrono::steady_clock;", candidate)
        self.assertNotIn(b"#define", projection.HELPER)
        self.assertNotIn(b"hip", projection.HELPER)
        self.assertEqual(candidate.count(projection.HOST_BOUNDARY), 1)
        self.assertEqual(candidate.count(projection.VALIDATION_BOUNDARY), 1)
        original_host = block(source.decode(), projection.SIGNATURE.decode())[0]
        projected_host = block(candidate.decode(), projection.SIGNATURE.decode())[0]
        for insertion in (projection.WRAPPER_OPEN, projection.WRAPPER_CLOSE,
                          projection.HOST_BOUNDARY, projection.VALIDATION_BOUNDARY):
            projected_host = projected_host.replace(insertion.decode(), "", 1)
        self.assertEqual(projected_host, original_host)

    def test_drift_reinstrumentation_and_candidate_mutations_refused(self):
        source = SOURCE.read_bytes()
        candidate = projection.project_source(source)
        for changed in (source+b"\n", source.replace(b"dim3(192)", b"dim3(160)"), candidate):
            with self.subTest(digest=hashlib.sha256(changed).hexdigest()):
                with self.assertRaises(ValueError):
                    projection.project_source(changed)
        for changed in (candidate+b"\n", candidate.replace(projection.HOST_BOUNDARY, b""),
                        candidate.replace(b"last.complete =", b"last.complete +="),
                        candidate.replace(b"dim3(192)", b"dim3(160)"),
                        candidate.replace(projection.VALIDATION_BOUNDARY, b"")):
            with self.assertRaises(ValueError):
                projection.recover_source(changed)

    def test_private_file_creation_refuses_overwrite_and_symlinks(self):
        with tempfile.TemporaryDirectory(prefix="phase-project-") as directory:
            folder = Path(directory)
            source, output = folder / "attention.hip", folder / "candidate.hip"
            source.write_bytes(SOURCE.read_bytes())
            digest = projection.project_file(source, output)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(source.read_bytes(), SOURCE.read_bytes())
            with self.assertRaises(FileExistsError):
                projection.project_file(source, output)
            with self.assertRaisesRegex(ValueError, "overwrite source"):
                projection.project_file(source, source)
            link = folder / "link.hip"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                projection.project_file(link, folder / "unused.hip")
            with self.assertRaisesRegex(ValueError, "overwrite source"):
                projection.project_file(source, link)
            self.assertFalse((folder / "unused.hip").exists())

    def test_entire_projected_runtime_host_syntax_only(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        with tempfile.TemporaryDirectory(prefix="phase-syntax-") as directory:
            folder = Path(directory)
            (folder / "hip").mkdir()
            (folder / "hip/hip_runtime.h").write_text(HIP_RUNTIME_DECLARATIONS)
            (folder / "hip/hip_fp16.h").write_text(HIP_HALF_DECLARATIONS)
            candidate = folder / "attention.hip"
            candidate.write_bytes(projection.project_source(SOURCE.read_bytes()))
            result = subprocess.run(compiler + ["-std=c++17", "-Werror", "-x", "c++", "-fsyntax-only",
                "-I", str(folder), "-I", str(SOURCE.parent), str(candidate)],
                cwd=ROOT, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_header_is_c_compatible_with_exact_field_layout(self):
        compiler = shlex.split(os.environ.get("CC", "cc"))
        fields = ("schema", "sequence", "complete", "api_result", "host_checks_ns",
                  "validation_status_ns", "reader_sync_ns", "total_ns")
        source = '#include "cache_phase_profile.h"\n#include <stddef.h>\n'
        source += '_Static_assert(sizeof(uq_cache_phase_profile_v1)==64,"ABI size");\n'
        for index, field in enumerate(fields):
            source += f'_Static_assert(offsetof(uq_cache_phase_profile_v1,{field})=={index*8},"{field}");\n'
        result = subprocess.run(compiler + ["-std=c11", "-Werror", "-x", "c", "-fsyntax-only",
            "-I", str(HEADER.parent), "-"], input=source, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_projected_host_body_matches_original_calls_and_records(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        source = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory(prefix="phase-host-") as directory:
            folder = Path(directory)
            harness, binary = folder / "host.cpp", folder / "host-test"
            harness.write_text(host_execution_source(source, projection.project_source(source)))
            result = subprocess.run(compiler + FLAGS + ["-pthread", "-I", str(SOURCE.parent),
                str(harness), "-o", str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
