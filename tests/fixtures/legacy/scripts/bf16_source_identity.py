"""Stdlib inventory for the partial image-derived v2 native candidate."""
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import shlex
import struct

IMAGE = "51f63a2d201ca944a43b8f1a18ac7da73ed9f6743c45bb82bbc40b44e6cf18d2"
IMAGE_CONFIG = "8474934b5049db5e3dbd9ee220e41b9d1a397a25ca0daac8a0b38850f2478495"
REPAIRS = {
    "srt/configs/model_config.py": "dadbc11c0b020f6a9b7f2ed71809f06d2ba4a9191f1df4600a8f29ced153534a",
    "srt/layers/quantization/__init__.py": "0b909ac91199253300f029dc544173067ea10208de69e5789f58f7a7500ec0b2",
}
NATIVE = ("runtime/uq.h", "runtime/attention.hip", "runtime/bf16_math.h", "codec/codec.h")
DRIVERS = ("scripts/bf16_source_identity.py", "scripts/build_bf16_v2.py")
EXCLUDED = frozenset({"srt/layers/attention/libuq_bf16_v2.so", "srt/layers/attention/ultraquant-bf16-build.json"})
SCOPE = "partial-image-candidate-standalone-native-test-only"
FLAGS = ["--offload-arch=gfx1201", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-fno-fast-math"]
PROTOCOL = {
    "name": "single-tu-uncompressed-hip-fatbin-v1",
    "hipcc_source_revision": "fd8e1efcd3dcc39b021c3bae4600a1a980ab27ef",
    "hipcc_source_sha256": "0c66b4724518ea864ffa828b90b3bb05ddaca4a52b4362746ced488bed42adda",
    "llvm_source_revision": "llvmorg-20.1.8",
    "hip_utility_sha256": "2c3d3aa8fe155408311012e1eca692d24f98031d360d41a2d608deceba26928f",
    "bundler_source_sha256": "c920cb0cea105df2eda685f58e014cf9c4d74b788e913551822e896cdf1cabb4",
    "image_tool_mapping": "unverified-until-independent-image-bound-review",
}
HOST_BUNDLE = "host-x86_64-unknown-linux-gnu"
MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
MAX_BINARY = 64 * 1024**2


class ProtocolError(ValueError, RuntimeError):
    pass


def tool_identity(spec):
    """Links require explicit path, target, link-chain and byte declarations."""
    path = Path(spec if isinstance(spec, str) else spec["path"])
    if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[/A-Za-z0-9_.+\-]+", str(path)):
        raise ValueError("unsupported tool path")
    links = {str(p): str(p.readlink()) for p in (path, *path.parents) if p.is_symlink()}
    target = path.resolve(strict=True)
    value = {"path": str(path), "canonical": str(target), "links": links, "sha256": digest(target)}
    if isinstance(spec, str):
        if links:
            raise ValueError("unadmitted tool link")
    elif spec != value:
        raise ValueError("tool path/target/bytes admission mismatch")
    return value


def binary(path):
    path = regular(path)
    if not 0 < path.stat().st_size <= MAX_BINARY:
        raise ProtocolError("invalid bounded binary size")
    return path.read_bytes()


def bundle_records(container):
    """Only the documented uncompressed two-entry HIP single-TU container."""
    if not container.startswith(MAGIC) or len(container) < len(MAGIC) + 8:
        raise ProtocolError("unsupported or missing bundle format")
    pos = len(MAGIC)
    count, = struct.unpack_from("<Q", container, pos)
    pos += 8
    if count != 2:
        raise ProtocolError("missing or extra bundle entries")
    entries = {}
    for _ in range(count):
        if pos + 24 > len(container):
            raise ProtocolError("truncated bundle header")
        offset, size, length = struct.unpack_from("<QQQ", container, pos)
        pos += 24
        if not 0 < length <= 256 or pos + length > len(container):
            raise ProtocolError("invalid bundle ID length")
        try:
            name = container[pos:pos+length].decode("ascii")
        except UnicodeDecodeError as error:
            raise ProtocolError("invalid bundle ID") from error
        pos += length
        if name in entries:
            raise ProtocolError("duplicate bundle ID")
        entries[name] = {"offset": offset, "size": size}
    devices = [n for n in entries if re.fullmatch(r"hip(?:v4)?-amdgcn-amd-amdhsa--gfx1201", n)]
    if len(devices) != 1 or set(entries) != {HOST_BUNDLE, devices[0]}:
        raise ProtocolError("wrong or ambiguous bundle target set")
    if entries[HOST_BUNDLE]["size"] != 0 or entries[devices[0]]["size"] == 0:
        raise ProtocolError("unexpected host or empty device payload")
    end = pos
    for item in sorted(entries.values(), key=lambda x: (x["offset"], x["size"])):
        offset, size = item["offset"], item["size"]
        if offset < end or offset + size > len(container) or any(container[end:offset]):
            raise ProtocolError("invalid bundle range or padding")
        end = offset + size
    if any(container[end:]):
        raise ProtocolError("extra trailing bundle data")
    return entries, devices[0]


def payload_binding(attempt):
    attempt = Path(attempt)
    library = binary(attempt / "libuq_bf16_v2.so")
    container = binary(attempt / "hip-fatbin.bin")
    section_text = regular(attempt / "sections.stdout").read_text()
    section_lines = [line for line in section_text.splitlines() if ".hip_fatbin" in line]
    pattern = r"\s*\[\s*\d+\]\s+\.hip_fatbin\s+PROGBITS\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+.*"
    if len(section_lines) != 1 or not (match := re.fullmatch(pattern, section_lines[0])):
        raise ProtocolError("missing or ambiguous library fatbin section")
    offset, size = (int(x, 16) for x in match.groups())
    if size != len(container) or offset + size > len(library) or library[offset:offset+size] != container:
        raise ProtocolError("extraction is not the library section bytes")
    entries, device = bundle_records(container)
    listed = regular(attempt / "enumerate.stdout").read_text().splitlines()
    if len(listed) != len(set(listed)) or set(listed) != set(entries):
        raise ProtocolError("bundle enumeration mismatch")
    item = entries[device]
    payload = binary(attempt / "attention-gfx1201.hsaco")
    if payload != container[item["offset"]:item["offset"]+item["size"]]:
        raise ProtocolError("substituted device payload")
    if regular(attempt / "host-empty.bin").read_bytes() != b"":
        raise ProtocolError("nonempty dummy host payload")
    return {"library_sha256": sha(library), "container_sha256": sha(container),
            "device_sha256": sha(payload), "section_offset": offset, "section_size": size,
            "bundle_records": entries, "device_bundle": device,
            "sections_sha256": digest(attempt / "sections.stdout"),
            "enumeration_sha256": digest(attempt / "enumerate.stdout")}


def compiler_binding(text, root, attempt, tools, environment):
    claims = [line for line in text.splitlines() if "hipcc-cmd" in line]
    if len(claims) != 1 or not claims[0].startswith("hipcc-cmd: "):
        raise ProtocolError("missing or ambiguous delegated compiler receipt")
    raw = claims[0][len("hipcc-cmd: "):]
    if any(c in raw for c in "\n\r;$`|&<>"):
        raise ProtocolError("unsupported delegated command grammar")
    try:
        argv = shlex.split(raw)
    except ValueError as error:
        raise ProtocolError("malformed delegated command") from error
    selected = tools["clang"]["path"]
    expected_env = {"HIP_CLANG_PATH": str(Path(selected).parent), "HIP_PLATFORM": "amd",
                    "HIP_COMPILER": "clang", "HIPCC_VERBOSE": "1",
                    "OFFLOAD_BUNDLER_IGNORE_ENV_VAR": "1", "OFFLOAD_BUNDLER_COMPRESS": "0"}
    allowed_env = set(expected_env) | {"PATH", "TMPDIR", "ROCM_PATH", "PYTHONDONTWRITEBYTECODE",
        "XDG_CACHE_HOME", "LC_ALL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "MAX_JOBS", "HIPCC_JOBS"}
    if set(environment) != allowed_env:
        raise ProtocolError("unsupported compiler environment")
    if (Path(selected).name != "clang++" or any(environment.get(k) != v for k, v in expected_env.items())
            or not argv or argv[0] != selected):
        raise ProtocolError("unrelated or unsupported compiler selection")
    source_path = str(Path(root) / "runtime/attention.hip")
    output = str(Path(attempt) / "libuq_bf16_v2.so")
    required = FLAGS + ["-shared", "-fPIC", source_path, "-o", output]
    if any(argv.count(x) != 1 for x in required) or argv[argv.index("-o")+1] != output:
        raise ProtocolError("delegated flags/source/output mismatch")
    # Only source-supported HIPCC additions. Unknown semantics fail closed.
    plain = set(required) | {"--hip-link", "--rtlib=compiler-rt", "-unwindlib=libgcc", "--driver-mode=g++"}
    pairs = {"-x": {"hip"}, "-Xclang": {"-fallow-half-arguments-and-returns"},
             "-mllvm": {"-amdgpu-early-inline-all=true", "-amdgpu-function-calls=false"}}
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in pairs:
            if i+1 >= len(argv) or argv[i+1] not in pairs[arg]:
                raise ProtocolError("unsupported delegated compiler option")
            i += 2
            continue
        if arg not in plain and not re.fullmatch(
                r"(?:-I|-L|--hip-device-lib-path=|--hip-path=)/[A-Za-z0-9_./+\-]+|"
                r"-D__HIP_(?:HCC_COMPAT_MODE__|PLATFORM_AMD__)=1", arg):
            raise ProtocolError("unsupported or conflicting delegated compiler option")
        i += 1
    return {"argv": argv, "selection_environment": expected_env,
            "compiler": tools["clang"], "hipcc_wrapper": tools["hipcc"],
            "receipt_sha256": sha(text.encode())}


TOOL_NAMES = ("hipcc", "clang", "nm", "readelf", "objdump", "objcopy", "bundler")
EXPORTS = frozenset({"uq_device", "uq_initialize", "uq_append", "uq_attention",
                     "uq_bf16_append_v2", "uq_bf16_attention_v2"})
V2_KERNELS = ("validate_write", "write", "validate_read", "read")


def commands(root, attempt, tools, device="<enumerated-device-bundle>"):
    root, attempt = Path(root), Path(attempt)
    lib = str(attempt / "libuq_bf16_v2.so")
    bundle = str(attempt / "hip-fatbin.bin")
    payload = str(attempt / "attention-gfx1201.hsaco")
    # HIPCC processes default target discovery even for --version. Pin its target.
    result = [("version-"+k, [str(tools[k]), *([FLAGS[0]] if k == "hipcc" else []), "--version"], 5)
              for k in TOOL_NAMES]
    result += [
        ("compile-library", [str(tools["hipcc"]), *FLAGS, str(root / "runtime/attention.hip"),
                             "-shared", "-fPIC", "-o", lib], 180),
        ("sections", [str(tools["readelf"]), "--section-headers", "--wide", lib], 15),
        ("extract", [str(tools["objcopy"]), "--dump-section=.hip_fatbin="+bundle,
                     lib, str(attempt / "objcopy-host.so")], 15),
        ("enumerate", [str(tools["bundler"]), "--type=o", "--input="+bundle, "--list"], 15),
        ("unbundle", [str(tools["bundler"]), "--type=o", "--input="+bundle,
                      "--targets="+HOST_BUNDLE+","+device,
                      "--output="+str(attempt / "host-empty.bin"),
                      "--output="+payload, "--unbundle"], 15),
        ("exports", [str(tools["nm"]), "--dynamic", "--defined-only", lib], 15),
        ("architecture", [str(tools["readelf"]), "--file-header", "--notes", payload], 15),
        ("disassembly", [str(tools["objdump"]), "--disassemble", "--demangle", payload], 15)]
    return result


def inspect_outputs(symbols, metadata, disassembly):
    exported = {parts[-1] for line in symbols.splitlines()
                if len(parts := line.split()) >= 3 and parts[-2] in ("T", "W")}
    if not EXPORTS <= exported:
        raise ProtocolError("missing required host exports")
    if "gfx1201" not in metadata or not re.search(r"AMDGPU|AMD GPU", metadata):
        raise ProtocolError("wrong device architecture metadata")
    if set(re.findall(r"gfx[0-9a-f]+", metadata)) != {"gfx1201"}:
        raise ProtocolError("mixed device architecture metadata")
    blocks = re.split(r"(?m)^\s*[0-9a-fA-F]+ <", disassembly)
    selected = {}
    for block in blocks[1:]:
        label, _, body = block.partition(">:")
        for name in V2_KERNELS:
            if "bf16_v2::" + name + "(" in label:
                if name in selected:
                    raise ProtocolError("ambiguous v2 device symbol")
                if not re.search(r"\bs_endpgm\b", body):
                    raise ProtocolError("v2 kernel has no decoded instructions")
                if re.search(r"\bv_(?:wmma|mfma)", body):
                    raise ProtocolError("unexpected matrix instruction in scalar v2 path")
                selected[name] = sha(body.encode())
    if set(selected) != set(V2_KERNELS):
        raise ProtocolError("missing v2 device kernel disassembly")
    if "v_wmma_f32_16x16x16_f16" not in disassembly or "v_mfma_scale" in disassembly:
        raise ProtocolError("legacy gfx1201 instruction inspection failed")
    return {"exports": sorted(EXPORTS), "target": "gfx1201", "v2_kernel_blocks": selected,
            "v2_path": "scalar-reference", "v1_wmma_observed": True,
            "v2_wmma_acceleration": False, "gpu_execution": "not-run"}


def verify_protocol_facts(facts, attempt, root):
    attempt = Path(attempt)
    protocol = facts.get("source_identity", {}).get("installed_protocol", PROTOCOL)
    if facts.get("protocol") != protocol or facts.get("compiler_flags") != FLAGS:
        raise ProtocolError("missing or changed protocol identity")
    toolchain = facts.get("toolchain", {})
    if set(toolchain) != set(TOOL_NAMES):
        raise ProtocolError("missing toolchain roles")
    tools = {}
    for name, value in toolchain.items():
        spec = {k: v for k, v in value.items() if k != "version"}
        tools[name] = tool_identity(spec)
        version = regular(attempt / ("version-"+name+".stdout")).read_text().strip()
        if not version or value.get("version") != version:
            raise ProtocolError("stale tool version receipt")
    expected_revision = "clang-sha256:"+tools["clang"]["sha256"]+" hipcc-sha256:"+tools["hipcc"]["sha256"]
    if facts.get("compiler_revision") != expected_revision:
        raise ProtocolError("stale compiler revision")
    roles = {"libuq_bf16_v2.so", "hip-fatbin.bin", "attention-gfx1201.hsaco", "host-empty.bin",
             "objcopy-host.so", "COMMAND-PLAN.json", "SOURCE-IDENTITY.json", "BUILD-INPUTS.json",
             "kv-image-relative.patch", "commands.json"}
    for label, unused, limit in commands(root, attempt, {k: v["path"] for k, v in tools.items()}):
        roles.update({label+".stdout", label+".stderr"})
    artifacts = checked_map(facts.get("artifacts"))
    if not roles <= set(artifacts):
        raise ProtocolError("missing mandatory artifact role")
    if tree(attempt, {"BUILD-FACTS.json"}) != artifacts:
        raise ProtocolError("artifact membership or bytes mismatch")
    raw = (attempt / "compile-library.stdout").read_text() + (attempt / "compile-library.stderr").read_text()
    compiler = compiler_binding(raw, root, attempt, tools, facts.get("environment", {}))
    payload = payload_binding(attempt)
    if facts.get("compiler_binding") != compiler or facts.get("payload_binding") != payload:
        raise ProtocolError("stale compiler or payload binding")
    expected = commands(root, attempt, {k: v["path"] for k, v in tools.items()}, payload["device_bundle"])
    records = facts.get("commands")
    if type(records) is not list or len(records) != len(expected) or load_json(attempt / "commands.json") != records:
        raise ProtocolError("missing command receipts")
    for record, (label, argv, work) in zip(records, expected):
        if (record.get("argv") != argv or record.get("cwd") != str(attempt)
                or record.get("stdout") != label+".stdout" or record.get("stderr") != label+".stderr"
                or record.get("exit") != 0 or record.get("group_absent") is not True
                or record.get("work_seconds") != work):
            raise ProtocolError("stale or conflicting command receipt")
    plan = load_json(attempt / "COMMAND-PLAN.json")
    if (plan.get("commands") != [list(x) for x in commands(root, attempt, {k: v["path"] for k, v in tools.items()})]
            or plan.get("environment") != facts.get("environment")):
        raise ProtocolError("command plan identity mismatch")
    if load_json(attempt / "SOURCE-IDENTITY.json") != facts["source_identity"] or load_json(attempt / "BUILD-INPUTS.json") != facts["build_inputs"]:
        raise ProtocolError("input receipt mismatch")
    if digest(attempt / "kv-image-relative.patch") != facts["source_identity"]["kv_patch_sha256"]:
        raise ProtocolError("patch receipt mismatch")
    inspection = inspect_outputs(*(regular(attempt / (name+".stdout")).read_text()
                                  for name in ("exports", "architecture", "disassembly")))
    if facts.get("inspection") != inspection:
        raise ProtocolError("inspection receipt mismatch")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_value(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA256")
    return value


def relative(value):
    if type(value) is not str or not value or "\\" in value or "\0" in value:
        raise ValueError("unsafe source path")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError("unsafe source path")
    return value


def regular(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("source/output link is forbidden")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("regular source/output required")
    return path


def canonical_directory(path, label):
    """Admit private inputs only by absolute, non-link directory paths."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("noncanonical " + label + " path")
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(label + " link is forbidden")
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError("canonical " + label + " directory required")
    return path


def digest(path):
    value = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(regular(path).read_bytes(), object_pairs_hook=pairs)


def tree(folder, excluded=frozenset()):
    folder = Path(folder).absolute()
    if not folder.is_dir() or any(p.is_symlink() for p in (folder, *folder.parents)):
        raise ValueError("regular source directory required")
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("source/output link is forbidden")
        if path.is_dir():
            continue
        name = relative(path.relative_to(folder).as_posix())
        regular(path)
        if name not in excluded:
            result[name] = digest(path)
    return result


def checked_map(value):
    if type(value) is not dict or not value:
        raise ValueError("missing source inventory")
    return {relative(k): digest_value(v) for k, v in value.items()}


def inventory(root, stock, baseline_path, baseline_sha256, *, stack_receipt=None, stack_receipt_sha256=None):
    """Verify externally bound stock/effective membership before deriving a patch."""
    if stack_receipt is not None or stack_receipt_sha256 is not None:
        return modern_inventory(root, stock, baseline_path, baseline_sha256,
                                stack_receipt, stack_receipt_sha256)
    if (Path(root) / "TARGET-REQUIREMENTS.json").exists():
        raise ValueError("modern target requires the externally bound common stack receipt")
    root, stock = Path(root).absolute(), Path(stock).absolute()
    if digest(baseline_path) != digest_value(baseline_sha256):
        raise ValueError("baseline receipt changed")
    baseline = load_json(baseline_path)
    if (baseline.get("schema"), baseline.get("image_manifest"), baseline.get("image_config"),
            baseline.get("repair_files")) != (2, IMAGE, IMAGE_CONFIG, REPAIRS):
        raise ValueError("wrong baseline schema/image/repairs")
    files = tree(root / "image-source/python/sglang", EXCLUDED)
    original = tree(stock)
    if files != checked_map(baseline.get("effective_files")):
        raise ValueError("effective source membership or bytes changed")
    if original != checked_map(baseline.get("stock_files")):
        raise ValueError("stock source membership or bytes changed")
    added = baseline.get("added_files")
    if type(added) is not list or len(set(added)) != len(added):
        raise ValueError("duplicate or absent added source declaration")
    if any(relative(p) not in files for p in added) or set(files) - set(original) != set(added):
        raise ValueError("stock counterpart accounting mismatch")
    if set(original) - set(files) or any(files.get(p) != h for p, h in REPAIRS.items()):
        raise ValueError("repair bytes or selected membership mismatch")
    native = {p: digest(root / p) for p in NATIVE}
    if native != checked_map(baseline.get("native_sources")):
        raise ValueError("native source bytes changed")
    patch, deltas = [], {}
    package = root / "image-source/python/sglang"
    for name in sorted(files):
        effective = (package / name).read_text(encoding="utf-8")
        before = (stock / name).read_text(encoding="utf-8") if name in original else ""
        repaired = effective if name in REPAIRS else before
        # The FP8 repairs have their own fixed identity and are excluded from KV deltas.
        patch.extend(difflib.unified_diff(repaired.splitlines(keepends=True), effective.splitlines(keepends=True),
            fromfile="a/" + name if name in original else "/dev/null", tofile="b/" + name))
        deltas[name] = {"stock": original.get(name), "repaired": REPAIRS.get(name, original.get(name)),
                        "effective": files[name], "added": name in added}
    patch_bytes = "".join(patch).encode()
    value = {"schema": 2, "scope": SCOPE, "image_manifest": IMAGE, "image_config": IMAGE_CONFIG,
        "image_git_revision": None, "repair_files": REPAIRS, "runtime_files": files,
        "source_sha256": sha(canonical(files)), "native_sources": native,
        "kv_patch_sha256": sha(patch_bytes), "baseline_receipt_sha256": baseline_sha256,
        "stock_files": original, "source_deltas": deltas, "added_files": sorted(added),
        "qualification": "source-only-unreviewed", "serving_composition_complete": False}
    return value, patch_bytes


def build_inputs(root, identity):
    protocol = identity.get("installed_protocol", PROTOCOL)
    return {"schema": 2, "source_identity_sha256": sha(canonical(identity)),
            "driver_files": {p: digest(Path(root) / p) for p in DRIVERS},
            "protocol": protocol, "protocol_sha256": sha(canonical(protocol)),
            "scope": SCOPE, "source_promotion": False, "launch_authorized": False}


def stack_contract(root):
    """Load one pure metadata module directly; never import SGLang/Torch."""
    import importlib.util
    path = Path(root) / "image-source/python/sglang/srt/layers/attention/ultraquant_stack_contract.py"
    if not path.is_file():
        raise ValueError("missing modern stack contract source")
    regular(path)
    spec = importlib.util.spec_from_file_location("uq_shared_stack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modern_inventory(root, stock, baseline_path, baseline_sha256, receipt_path, receipt_sha256):
    root, stock = Path(root).absolute(), Path(stock).absolute()
    receipt = stack_contract(root).read_receipt(receipt_path, receipt_sha256)
    if digest(baseline_path) != digest_value(baseline_sha256):
        raise ValueError("baseline receipt changed")
    baseline = load_json(baseline_path)
    if (baseline.get("schema"), baseline.get("image_manifest"), baseline.get("image_config"),
        baseline.get("stack_receipt_sha256")) != (
        2, receipt["image_manifest"], receipt["image_config"], receipt_sha256
    ):
        raise ValueError("wrong modern baseline image/stack identity")
    files = tree(root / "image-source/python/sglang", EXCLUDED)
    common = tree(stock)
    if common != checked_map(receipt["common_files"]) or common != checked_map(baseline.get("stock_files")):
        raise ValueError("common FP8 baseline membership or bytes changed")
    if files != checked_map(baseline.get("effective_files")):
        raise ValueError("effective source membership or bytes changed")
    added = baseline.get("added_files")
    if type(added) is not list or len(added) != len(set(added)) or set(files)-set(common) != set(added) or set(common)-set(files):
        raise ValueError("modern stock counterpart accounting mismatch")
    native = {p: digest(root / p) for p in NATIVE}
    if native != checked_map(baseline.get("native_sources")):
        raise ValueError("native source bytes changed")
    patch = []
    for name in sorted(files):
        previous = (stock / name).read_text() if name in common else ""
        effective = (root / "image-source/python/sglang" / name).read_text()
        patch.extend(difflib.unified_diff(previous.splitlines(True), effective.splitlines(True),
            fromfile="a/"+name if name in common else "/dev/null", tofile="b/"+name))
    data = "".join(patch).encode()
    result = {"schema": 2, "scope": SCOPE, "image_manifest": receipt["image_manifest"],
        "image_config": receipt["image_config"], "image_git_revision": None,
        "source_revision": receipt["target"]["source_revision"], "stack_receipt_sha256": receipt_sha256,
        "repair_files": receipt["repair_files"], "runtime_files": files, "source_sha256": sha(canonical(files)),
        "native_sources": native, "stock_files": common, "added_files": sorted(added),
        "kv_patch_sha256": sha(data), "baseline_receipt_sha256": baseline_sha256,
        "installed_protocol": receipt["protocol"], "compiler_environment": receipt["compiler_environment"],
        "qualification": "source-only-unreviewed", "serving_composition_complete": False}
    return result, data


def compiler_environment(identity, tools, temporary):
    base = {"PATH": "/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/bin:/bin", "ROCM_PATH": "/opt/rocm",
        "HIP_CLANG_PATH": str(Path(tools["clang"]).parent), "HIP_PLATFORM": "amd", "HIP_COMPILER": "clang",
        "HIPCC_VERBOSE": "1", "OFFLOAD_BUNDLER_IGNORE_ENV_VAR": "1", "OFFLOAD_BUNDLER_COMPRESS": "0"}
    if "stack_receipt_sha256" in identity:
        admitted = identity["compiler_environment"]
        if set(admitted) != set(base) or any(type(v) is not str or not v for v in admitted.values()):
            raise ValueError("missing exact modern compiler environment")
        if any(admitted[k] != base[k] for k in base if k not in ("PATH", "ROCM_PATH")):
            raise ValueError("unsupported modern compiler selection/environment")
        if not Path(admitted["ROCM_PATH"]).is_absolute():
            raise ValueError("nonabsolute SDK root")
        base = dict(admitted)
    base.update(TMPDIR=str(temporary), XDG_CACHE_HOME=str(temporary), PYTHONDONTWRITEBYTECODE="1", LC_ALL="C")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS", "HIPCC_JOBS"):
        base[name] = "2"
    return base


def reviewed_proof(facts_path, review_path, expected_review_sha256, root, stock, baseline, baseline_hash, *,
                   stack_receipt=None, stack_receipt_sha256=None):
    """Assemble data only after an externally authenticated independent review.

    This function never installs a proof or loads a binary. The build producer
    does not call it. The owner must supply the independent receipt digest.
    """
    facts_path, review_path = Path(facts_path), Path(review_path)
    if digest(review_path) != digest_value(expected_review_sha256):
        raise ValueError("independent review receipt digest mismatch")
    facts, review = load_json(facts_path), load_json(review_path)
    if (review.get("schema"), review.get("decision"), review.get("facts_sha256"), review.get("scope")) != (
            1, "accept-build-only", digest(facts_path), SCOPE) or not review.get("reviewer"):
        raise ValueError("missing or stale independent review")
    if facts.get("schema") != 2 or facts.get("status") != "BUILD_INSPECTED_UNREVIEWED" or facts.get("qualification") != "unreviewed":
        raise ValueError("build facts are not inspectable unreviewed evidence")
    identity, unused = inventory(root, stock, baseline, baseline_hash,
        stack_receipt=stack_receipt, stack_receipt_sha256=stack_receipt_sha256)
    if facts.get("source_identity") != identity or facts.get("build_inputs") != build_inputs(root, identity):
        raise ValueError("stale source or driver facts")
    for name, expected in checked_map(facts.get("artifacts")).items():
        if digest(facts_path.parent / name) != expected:
            raise ValueError("stale build artifact")
    verify_protocol_facts(facts, facts_path.parent, root)
    tools = facts.get("toolchain")
    if type(tools) is not dict or not tools or any(not v.get("version", "").strip() for v in tools.values()):
        raise ValueError("missing toolchain facts")
    value = {k: identity[k] for k in ("schema", "image_manifest", "image_config", "repair_files",
        "runtime_files", "source_sha256", "native_sources", "kv_patch_sha256")}
    value.update(qualification="build-reviewed-not-serving-qualified",
        binary_sha256=facts["artifacts"]["libuq_bf16_v2.so"],
        compiler_revision=facts["compiler_revision"], compiler_flags=facts["compiler_flags"],
        review_receipt_sha256=expected_review_sha256, build_facts_sha256=digest(facts_path),
        protocol_sha256=sha(canonical(identity.get("installed_protocol", PROTOCOL))), payload_binding=facts["payload_binding"],
        compiler_binding=facts["compiler_binding"],
        scope=SCOPE, serving_composition_complete=False)
    if "stack_receipt_sha256" in identity:
        value.update(stack_receipt_sha256=identity["stack_receipt_sha256"],
                     source_revision=identity["source_revision"])
    return value


def verify_installed_stack(root, receipt_path, receipt_sha256, tools):
    receipt = stack_contract(root).read_receipt(receipt_path, receipt_sha256)
    if type(tools) is not dict or set(tools) != set(TOOL_NAMES):
        raise ValueError("installed tool roles are incomplete")
    for role, measured in tools.items():
        admitted = receipt["tools"][role]
        if measured["path"] != admitted["path"] or measured["sha256"] != admitted["sha256"]:
            raise ValueError("installed tool differs from common stack")
        if digest(Path(admitted["interpreter"])) != admitted["interpreter_sha256"]:
            raise ValueError("installed ELF interpreter changed")
        filesystem = Path(admitted.get("filesystem_root", "/"))
        if not filesystem.is_absolute():
            raise ValueError("nonabsolute library filesystem root")
        for relative_name, expected in admitted["libraries"].items():
            if digest(filesystem / relative_name) != expected:
                raise ValueError("installed compiler library changed")
    return receipt
