"""Source-bound, single-host-thread arithmetic checks; no HIP/GPU execution.

The explicit phase splitter accepts only the reader's current barrier/loop
shape. It compiles each actual phase body, then schedules all logical lanes
between barriers. This does not model GPU memory visibility or qualification.
"""
from pathlib import Path
import hashlib
import os
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "prototype/runtime/attention.hip"
BASELINE = ROOT / "tests/fixtures/cache-decoder-validator/attention.hip"
BASELINE_SHA256 = "a11c985bb5495e0445034a4aabe5b89fe1ed22fbd6995d76f9aaf4cda823a84b"
HARNESS = ROOT / "prototype/tests/grouped_cache_reader_test.cpp"
FLAGS = ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
         "-ffp-contract=off", "-fno-fast-math"]
BARRIER = "__syncthreads();"
TOKEN_LOOP = "for(int t=0;t<lengths[row];++t)"
PHASES = ("query", "rotate", "decode", "product", "score", "accumulate", "output")
HIP_RUNTIME_DECLARATIONS = r"""
#pragma once
#include <cmath>
#include <cstddef>
#define __host__
#define __device__
#define __global__
#define __shared__
using std::isfinite;
struct dim3 { unsigned x,y,z; dim3(unsigned=1,unsigned=1,unsigned=1); };
extern dim3 blockIdx,threadIdx,blockDim;
void __syncthreads();
int atomicExch(int*,int);
using hipStream_t=void*;
enum hipError_t { hipSuccess };
enum hipMemcpyKind { hipMemcpyDeviceToHost };
struct hipDeviceProp_t { char gcnArchName[256]; };
hipError_t hipGetLastError();
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipMemcpy(void*,const void*,std::size_t,hipMemcpyKind);
hipError_t hipGetDevice(int*);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*,int);
hipError_t hipMemsetAsync(void*,int,std::size_t,hipStream_t);
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) \
    ((void)(grid),(void)(block),(void)(shared),(void)(stream),kernel(__VA_ARGS__))
"""
HIP_HALF_DECLARATIONS = """
#pragma once
using half=_Float16;
float __half2float(half);
half __float2half_rn(float);
"""


def compact(text):
    return re.sub(r"\s+", "", text)


def block(text, signature):
    """One known function/block; source contains no braces in string literals."""
    if text.count(signature) != 1:
        raise ValueError("expected one " + signature)
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("unclosed " + signature)
    return text[start:end], text[opening + 1:end - 1]


def split_once(text, delimiter):
    if text.count(delimiter) != 1:
        raise ValueError("expected one boundary: " + delimiter)
    return text.split(delimiter, 1)


def preamble(text, grouped):
    binding, rest = text.split("__shared__", 1)
    expected = ("int row=blockIdx.x,kvhead=blockIdx.y,group=threadIdx.x/32,lane=threadIdx.x%32;"
                "int head=kvhead*6+group;" if grouped else
                "int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;")
    if compact(binding) != compact(expected):
        raise ValueError("reader head/lane binding changed")
    shared, tail = split_once("__shared__" + rest, "float acc[8]={},maximum=-INFINITY;")
    expected_shared = ("float rotated[6][256],products[6][256],decoded_k[256],decoded_v[256];"
                       "float alpha[6],weight[6],denominator[6];" if grouped else
                       "float rotated[256],products[256],alpha,weight,denominator;")
    shared = shared.replace("__shared__", "")
    if compact(shared) != compact(expected_shared):
        raise ValueError("reader shared storage changed")
    return binding, shared, tail


def grouped_phases(body):
    prefix, remaining = split_once(body, "// phase: query")
    binding, shared, tail = preamble(prefix, True)
    if tail.strip() or body.count(BARRIER) != 6:
        raise ValueError("grouped barrier count/preamble")
    phases = {}
    for name, following in zip(PHASES, PHASES[1:]):
        phase, remaining = split_once(remaining, "// phase: " + following)
        phases[name] = phase
    phases["output"] = remaining
    for name in PHASES[:-1]:
        phase, after = split_once(phases[name], BARRIER)
        expected_after = TOKEN_LOOP + " {" if name == "rotate" else "}" if name == "accumulate" else ""
        if compact(after) != compact(expected_after):
            raise ValueError("grouped barrier/loop order: " + name)
        phases[name] = phase
    return binding, shared, phases


def original_phases(body):
    binding, shared, remaining = preamble(body, False)
    parts = remaining.split(BARRIER)
    if len(parts) != 5:
        raise ValueError("original barrier shape")
    query, rotate, product, score, tail = parts
    loop_prefix, product = split_once(product, TOKEN_LOOP + " {")
    if loop_prefix.strip():
        raise ValueError("original loop prefix")
    record, product = split_once(product, "for(int i=0;i<8;++i)")
    if compact(record) != "size_trecord=size_t(indices[starts[row]+t])*2+head/6;":
        raise ValueError("original record binding")
    # Two identical, unnested channel loops remain: accumulation then output.
    if tail.count("for(int i=0;i<8;++i)") != 2:
        raise ValueError("original accumulation/output shape")
    first = tail.index("for(int i=0;i<8;++i)")
    opening = tail.index("{", first)
    close = tail.index("}", opening)
    accumulation = tail[first:close + 1]
    rest = tail[tail.index(accumulation) + len(accumulation):].lstrip()
    if not rest.startswith("}"):
        raise ValueError("original token-loop close")
    phases = {"query": query, "rotate": rotate,
              "product": record + "for(int i=0;i<8;++i)" + product,
              "score": score, "accumulate": record + accumulation,
              "output": rest[1:]}
    return binding, shared, phases


def assert_dispatch(source):
    _, host = block(source, 'extern "C" int uq_bf16_attention_v2(')
    expected = """
        hipLaunchKernelGGL(validate_read,dim3(queries),dim3(256),0,stream,*c,q,indices,count,starts,lengths,status);
        int checked=checked_status(status,stream);if(checked) return checked;
        if(c->protected_layer) {
            hipLaunchKernelGGL(read,dim3(queries,12),dim3(32),0,stream,*c,q,indices,starts,lengths,scale,out);
        } else {
            hipLaunchKernelGGL(read_grouped,dim3(queries,2),dim3(192),0,stream,*c,q,indices,starts,lengths,scale,out);
        }
        return finish(stream);
    """
    if compact(expected) not in compact(host):
        raise ValueError("compact-only dispatch, launch geometry, or status ordering changed")


def extracted(source):
    original, original_body = block(source, "__global__ void read(")
    baseline_bytes = BASELINE.read_bytes()
    if hashlib.sha256(baseline_bytes).hexdigest() != BASELINE_SHA256:
        raise ValueError("immutable decoder-validator baseline changed")
    baseline, _ = block(baseline_bytes.decode("utf-8"), "__global__ void read(")
    if original != baseline:
        raise ValueError("protected/original read changed")
    assert_dispatch(source)
    _, grouped = block(source, "__global__ void read_grouped(")
    load, _ = block(source, "__device__ float load(")
    parts = ["namespace runtime_source {\n#define __device__\n", load,
             "\n#undef __device__\n}\n"]
    for name, lanes, values in (("Original", 32, original_phases(original_body)),
                                ("Grouped", 192, grouped_phases(grouped))):
        binding, shared, phases = values
        binding = binding.replace("int ", "[[maybe_unused]] int ")
        parts.append(f"struct {name} : Common {{\n{shared}\nLane states[{lanes}]{{}};\n")
        for phase, code in phases.items():
            method = "emit_output" if phase == "output" else phase
            parts.append(f"void {method}(int logical_lane, [[maybe_unused]] int t=0) {{\n"
                         "threadIdx.x=logical_lane;\n" + binding +
                         "\n[[maybe_unused]] float (&acc)[8]=states[logical_lane].acc;\n"
                         "[[maybe_unused]] float& maximum=states[logical_lane].maximum;\n" +
                         code + "\n}\n")
        parts.append("};\n")
    return "".join(parts)


class GroupedReaderTests(unittest.TestCase):
    def test_entire_runtime_host_syntax_only(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        with tempfile.TemporaryDirectory(prefix="grouped-syntax-") as directory:
            folder = Path(directory)
            (folder / "hip").mkdir()
            (folder / "hip/hip_runtime.h").write_text(HIP_RUNTIME_DECLARATIONS)
            (folder / "hip/hip_fp16.h").write_text(HIP_HALF_DECLARATIONS)
            result = subprocess.run(compiler + ["-std=c++17", "-Werror", "-x", "c++", "-fsyntax-only",
                                                "-I", str(folder), str(RUNTIME)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shape_dispatch_and_negative_mutations(self):
        source = RUNTIME.read_text()
        extracted(source)
        mutations = [
            source.replace("// phase: query", "// removed phase: query", 1),
            source.replace("int head=kvhead*6+group;", "int head=group;", 1),
            source.replace("dim3(192)", "dim3(160)", 1),
            source.replace("if(c->protected_layer) {\n        hipLaunchKernelGGL(read,", "if(!c->protected_layer) {\n        hipLaunchKernelGGL(read,", 1),
        ]
        start = source.index("__global__ void read_grouped(")
        for index in [m.start() for m in re.finditer(re.escape(BARRIER), source[start:])]:
            mutations.append(source[:start + index] + source[start + index:].replace(BARRIER, "", 1))
        for changed in mutations:
            with self.subTest(mutation=mutations.index(changed)), self.assertRaises(ValueError):
                extracted(changed)

    def test_exact_phase_arithmetic_and_mutation_detection(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        with tempfile.TemporaryDirectory(prefix="grouped-reader-") as directory:
            folder = Path(directory)
            for name, source in (("current", RUNTIME.read_text()),
                                 ("wrong-product", RUNTIME.read_text().replace(
                                     "uq_bf16::mul(rotated[group][j],decoded_k[j])",
                                     "uq_bf16::mul(rotated[group][j],-decoded_k[j])", 1))):
                include = folder / (name + ".inc")
                include.write_text(extracted(source))
                executable = folder / name
                command = compiler + FLAGS + [f'-DUQ_GROUPED_PHASES_FILE="{include}"',
                                               str(HARNESS), "-o", str(executable)]
                compiled = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                result = subprocess.run([str(executable)], cwd=ROOT, capture_output=True, text=True, timeout=60)
                if name == "current":
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    print(result.stdout, end="")
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("FP32 mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
