"""Compile the candidate and inspect gfx1201 machine code; never run a GPU test."""
import json
import os
from pathlib import Path
import subprocess

from source_identity import ROOT, UPSTREAM, canonical, digest, source_identity


def main():
    compiler = Path("/opt/rocm/bin/hipcc")
    clang = Path("/opt/rocm/llvm/bin/clang++")
    objdump = Path("/opt/rocm/llvm/bin/llvm-objdump")
    for tool in (compiler, clang, objdump):
        if not tool.is_file():
            raise SystemExit(f"NOT RUN: prepared tool missing: {tool}")
    env = {"PATH": "/opt/rocm/bin:/usr/bin:/bin", "TMPDIR": "/tmp"}
    identity = source_identity()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    package = UPSTREAM / "python/sglang/srt/layers/attention"
    common = [str(compiler), "--offload-arch=gfx1201", "-std=c++17", "-O2",
              "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-fno-fast-math",
              str(ROOT / "runtime/attention.hip")]
    commands = [common + ["-shared", "-fPIC", "-o", str(package / "libuq_hip.so")],
                common + ["--genco", "-o", str(build / "attention.hsaco")]]
    for command in commands:
        subprocess.run(command, env=env, check=True)
    code = subprocess.check_output([str(objdump), "--disassemble", str(build / "attention.hsaco")], env=env)
    (build / "attention.isa.txt").write_bytes(code)
    if b"v_wmma_f32_16x16x16_f16" not in code or b"v_mfma_scale" in code:
        raise SystemExit("gfx1201 WMMA instruction inspection failed")
    version = subprocess.check_output([str(clang), "--version"], env=env).decode().strip()
    identity.update({
        "binary_sha256": digest((package / "libuq_hip.so").read_bytes()),
        "compiler_revision": "clang-sha256:" + digest(clang.resolve().read_bytes()) +
                             " hipcc-sha256:" + digest(compiler.resolve().read_bytes()),
        "compiler_version": version,
        "compiler_sha256": digest(clang.resolve().read_bytes()),
        "hipcc_sha256": digest(compiler.resolve().read_bytes()),
        "commands": commands, "isa_sha256": digest(code),
        "device_code_sha256": digest((build / "attention.hsaco").read_bytes()),
        "gpu_execution": "not-run", "qualification": "unqualified",
    })
    (package / "ultraquant-build.json").write_bytes(canonical(identity) + b"\n")
    print("gfx1201 compilation and WMMA inspection passed; GPU execution NOT RUN")


if __name__ == "__main__":
    main()
