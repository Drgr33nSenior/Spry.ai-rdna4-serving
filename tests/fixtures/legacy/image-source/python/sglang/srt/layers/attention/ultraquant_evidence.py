"""Stable candidate identity and local process receipt; no qualification claim."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from sglang.srt.layers.attention.ultraquant_contract import MODE, SOURCE_REVISION


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("experimental runtime source/binary must be a regular file")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def build_identity():
    package = Path(__file__).parent
    manifest = package / "ultraquant-build.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("experimental runtime requires a bounded local build manifest")
    result = json.loads(manifest.read_bytes())
    if result["source_revision"] != SOURCE_REVISION or result["qualification"] != "unqualified":
        raise ValueError("experimental runtime build identity mismatch")
    if digest(package / "libuq_hip.so") != result["binary_sha256"]:
        raise ValueError("experimental runtime binary changed since compilation")
    source_root = Path(__file__).parents[5]
    observed = set()
    links = {}
    for path in source_root.rglob("*"):
        if path.is_symlink():
            links[path.relative_to(source_root).as_posix()] = str(path.readlink())
        elif path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name not in (
            "libuq_hip.so", "ultraquant-build.json"
        ):
            observed.add(path.relative_to(source_root).as_posix())
    if observed != set(result["runtime_files"]) or links != result["runtime_links"]:
        raise ValueError("experimental runtime source file set changed since compilation")
    source_digest = hashlib.sha256(canonical({"upstream": result["runtime_files"],
                                            "upstream_links": result["runtime_links"],
                                            "experiment": result["experiment_files"]})).hexdigest()
    if source_digest != result["source_tree_sha256"]:
        raise ValueError("experimental runtime aggregate source identity mismatch")
    for relative, expected in result["runtime_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(source_root / path) != expected:
            raise ValueError("experimental runtime source changed since compilation")
    return result


def descriptor(build, dim, kv_heads, query_heads, hip_version):
    configuration = {
        "gpu_arch": "gfx1201", "tensor_parallel": 1, "compute_dtype": "float16",
        "attention": "causal-full", "head_dimension": dim, "kv_heads": kv_heads,
        "query_heads": query_heads, "page_size": 1, "rotation": "post-rope-qk",
    }
    dependencies = {"python": sys.version.split()[0], "hip": hip_version}
    for name in ("sglang", "torch", "triton", "pytorch-triton-rocm", "sglang-kernel", "transformers"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "schema": 1, "runtime": "sglang", "source_repository": "sgl-project/sglang",
        "source_revision": SOURCE_REVISION, "source_tree_sha256": build["source_tree_sha256"],
        "patch_sha256": build["patch_sha256"], "binary_sha256": build["binary_sha256"],
        "compiler_revision": build["compiler_revision"],
        "dependencies_sha256": hashlib.sha256(canonical(dependencies)).hexdigest(),
        "codec": MODE, "configuration_sha256": hashlib.sha256(canonical(configuration)).hexdigest(),
        "configuration": configuration, "requested_mode": MODE, "observed_mode": "compressed-gpu",
        "qualification": "unqualified", "restart_required": True,
    }


def process(pid):
    # comm may contain spaces or closing parentheses; fields after its last ')'
    # begin at stat field3. starttime is field22, index19 in this suffix.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def emit_receipt(value):
    pid = os.getpid()
    parent, start = process(pid)
    launcher, launcher_start = None, None
    current = pid
    for _ in range(64):
        arguments = Path(f"/proc/{current}/cmdline").read_bytes().split(b"\0")
        parent, ticks = process(current)
        if b"sglang.launch_server" in arguments:
            launcher, launcher_start = current, ticks
            break
        if parent <= 0 or parent == current:
            break
        current = parent
    if launcher is None:
        # Standalone synthetic backend tests are not serving observations.
        return False
    root = Path("/tmp/sglang-experimental-kv")
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("experimental runtime receipt directory is not private")
    receipt = {"schema": 1, "pid": pid, "start_ticks": start, "launcher_pid": launcher,
               "launcher_start_ticks": launcher_start, "descriptor": value}
    fd, temporary = tempfile.mkstemp(prefix=f".{pid}-", dir=root)
    # Failed outputs are retained; there is no cleanup of prior evidence.
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / f"{pid}.json")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return True

# Separate image-derived v2 identity. Never reuse preparation SOURCE_REVISION.
BF16_IMAGE = "51f63a2d201ca944a43b8f1a18ac7da73ed9f6743c45bb82bbc40b44e6cf18d2"
BF16_IMAGE_CONFIG = "8474934b5049db5e3dbd9ee220e41b9d1a397a25ca0daac8a0b38850f2478495"
BF16_REPAIRS = {
    "srt/configs/model_config.py": "dadbc11c0b020f6a9b7f2ed71809f06d2ba4a9191f1df4600a8f29ced153534a",
    "srt/layers/quantization/__init__.py": "0b909ac91199253300f029dc544173067ea10208de69e5789f58f7a7500ec0b2",
}


def bf16_build_identity(package, policy, *, stack_receipt=None, stack_receipt_sha256=None):
    """Verify installed immutable build proof; no such proof is supplied here."""
    from types import MappingProxyType
    package = Path(package)
    proof = package / "srt/layers/attention/ultraquant-bf16-build.json"
    if proof.is_symlink() or not proof.is_file() or proof.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("missing reviewed BF16 build proof")
    build = json.loads(proof.read_bytes())
    image, config, repairs = BF16_IMAGE, BF16_IMAGE_CONFIG, BF16_REPAIRS
    if stack_receipt is not None or stack_receipt_sha256 is not None:
        from sglang.srt.layers.attention.ultraquant_stack_contract import read_receipt, SOURCE_REVISION
        common = read_receipt(stack_receipt, stack_receipt_sha256)
        image, config, repairs = common["image_manifest"], common["image_config"], common["repair_files"]
        if build.get("stack_receipt_sha256") != stack_receipt_sha256 or build.get("source_revision") != SOURCE_REVISION:
            raise ValueError("stale modern stack/source identity")
        if build.get("serving_composition_complete") is not True:
            raise ValueError("modern runtime requires complete installed serving composition")
    elif (package.parents[2] / "TARGET-REQUIREMENTS.json").exists():
        raise ValueError("modern target requires the externally bound shared stack")
    if (build.get("schema"), build.get("image_manifest"), build.get("image_config"),
        build.get("qualification"), build.get("repair_files")) != (
            2, image, config, "build-reviewed-not-serving-qualified", repairs):
        raise ValueError("stale image/repair/build identity")
    for name in ("kv_patch_sha256", "source_sha256", "binary_sha256", "review_receipt_sha256"):
        value = build.get(name)
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("missing build digest")
    if type(build.get("compiler_revision")) is not str or not build["compiler_revision"].strip() or (
        type(build.get("compiler_flags")) is not list or not build["compiler_flags"] or any(
            type(flag) is not str or not flag.strip() for flag in build["compiler_flags"]
        )
    ):
        raise ValueError("missing compiler/flags proof")
    files = build.get("runtime_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("missing effective source inventory")
    excluded = {"srt/layers/attention/libuq_bf16_v2.so", "srt/layers/attention/ultraquant-bf16-build.json"}
    actual = set()
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError("runtime identity requires separately resolved regular source")
        if path.is_file() and path.relative_to(package).as_posix() not in excluded:
            actual.add(path.relative_to(package).as_posix())
    if actual != set(files):
        raise ValueError("effective source membership changed")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(package / path) != expected:
            raise ValueError("effective source hash changed")
    if hashlib.sha256(canonical(files)).hexdigest() != build["source_sha256"]:
        raise ValueError("effective source aggregate changed")
    native_files = build.get("native_sources")
    required_native = {"runtime/uq.h", "runtime/attention.hip", "runtime/bf16_math.h", "codec/codec.h"}
    if not isinstance(native_files, dict) or set(native_files) != required_native:
        raise ValueError("missing native source inventory")
    native_root = package.parents[2]
    if any(digest(native_root / name) != expected for name, expected in native_files.items()):
        raise ValueError("native source changed since build")
    if any(files.get(path) != value for path, value in repairs.items()):
        raise ValueError("repair bytes changed")
    if digest(package / "srt/layers/attention/libuq_bf16_v2.so") != build["binary_sha256"]:
        raise ValueError("missing or stale BF16 binary")
    value = {"build": build, "policy": dict(policy), "model_config":
        "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc",
        "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"}
    # Canonical immutable bytes avoid shallow nested-dict mutation.
    encoded = canonical(value)
    return MappingProxyType({"canonical": encoded, "sha256": hashlib.sha256(encoded).hexdigest()})


class BF16CacheOwner:
    """Local process/cache capability. Prefix tags retain caller/LoRA extra_key."""
    def __init__(self, identity):
        import weakref
        self._identity = bytes(identity["canonical"])
        self._pid = os.getpid()
        self._nonce = os.urandom(32)
        self._capacity = None
        self._allocator = None
        self._generations = {}
        self._pending = set()
        self._next_generation = 0
        self._cache = None
        self._adapters = weakref.WeakSet()
        self._release_permission = frozenset()
        self._flushing = False

    def prefix_key(self, extra_key):
        self.check_process()
        return (self._identity, self._pid, self._nonce, extra_key)

    def check_process(self):
        if os.getpid() != self._pid:
            raise ValueError("cross-process BF16 cache reuse is forbidden")

    def validate_prefix(self, observed, extra_key):
        if observed != self.prefix_key(extra_key):
            raise ValueError("stale, mixed, cross-rank or foreign cache prefix")

    def namespace(self, tag):
        capability = self.prefix_key(None)[:-1]
        if isinstance(tag, BF16Namespace):
            if tag.capability != capability:
                raise ValueError("foreign BF16 cache namespace")
            return tag
        return BF16Namespace(capability, tag, _bf16_tag_identity(tag))

    def bind_allocator(self, allocator, capacity):
        self.check_process()
        if type(capacity) is not int or not 1 <= capacity < 1048576 or allocator.page_size != 1:
            raise ValueError("BF16 owner requires the bounded page-one allocator")
        if self._allocator is not None and self._allocator is not allocator:
            raise ValueError("BF16 owner already has an allocator")
        self._allocator, self._capacity = allocator, capacity

    def bind_cache(self, cache):
        import weakref
        self.check_process()
        existing = self._cache() if self._cache is not None else None
        if self._cache is not None and existing is None:
            raise ValueError("BF16 prefix cache owner expired")
        if existing is not None and existing is not cache:
            raise ValueError("BF16 owner already has a prefix cache")
        if cache.page_size != 1 or self._capacity is None:
            raise ValueError("BF16 prefix cache requires page one")
        self._cache = weakref.ref(cache)

    def register_adapter(self, adapter):
        self.check_process()
        if len(self._adapters) >= 4:
            raise ValueError("too many BF16 full-attention components")
        self._adapters.add(adapter)

    def invalidate(self):
        self.check_process()
        for adapter in tuple(self._adapters):
            adapter.invalidate()

    def _slots(self, values):
        self.check_process()
        slots = tuple(values)
        if self._capacity is None or len(slots) > self._capacity or len(set(slots)) != len(slots) or any(
            type(slot) is not int or not 1 <= slot <= self._capacity for slot in slots
        ):
            raise ValueError("invalid or unbound BF16 physical slots")
        return slots

    def allocate(self, values):
        slots = self._slots(values)
        if self._next_generation + len(slots) >= 2**64:
            raise ValueError("BF16 slot generation counter exhausted")
        if any(slot in self._generations or slot in self._pending for slot in slots):
            raise ValueError("BF16 slot is already owned")
        if slots:
            self.invalidate()
        for slot in slots:
            self._next_generation += 1
            self._generations[slot] = self._next_generation

    def snapshot(self, values):
        slots = self._slots(tuple(dict.fromkeys(values)))
        if any(slot not in self._generations for slot in slots):
            raise ValueError("BF16 slot generation is not live")
        return tuple((slot, self._generations[slot]) for slot in slots)

    def validate_snapshot(self, snapshot):
        self.check_process()
        if self._cache is not None and self._cache() is None:
            raise ValueError("BF16 prefix cache owner expired")
        if snapshot is None or any(self._generations.get(slot) != generation for slot, generation in snapshot):
            raise ValueError("stale BF16 slot generation")

    def cached_slots(self):
        self.check_process()
        cache = self._cache() if self._cache is not None else None
        if self._cache is not None and cache is None:
            raise ValueError("BF16 prefix cache owner expired")
        if cache is None or not hasattr(cache, "root_node"):
            return frozenset()
        slots, seen, stack = set(), set(), list(cache.root_node.children.values())
        while stack:
            node = stack.pop()
            if id(node) in seen or len(seen) >= 2 * self._capacity:
                raise ValueError("invalid BF16 prefix ownership tree")
            seen.add(id(node))
            if node.value is not None:
                slots.update(node.value.cpu().tolist())
            stack.extend(node.children.values())
        self.snapshot(slots)
        return frozenset(slots)

    def check_writable(self, values):
        slots = self._slots(values)
        self.snapshot(slots)
        if self.cached_slots().intersection(slots):
            raise ValueError("BF16 cache-owned prefix is immutable")

    def retire(self, values, deferred=False):
        slots = self._slots(values)
        if not slots:
            return
        if self._flushing:
            if not set(slots) <= self._pending:
                raise ValueError("invalid BF16 grouped free commit")
            self._pending.difference_update(slots)
            return
        self.snapshot(slots)
        if (self.cached_slots().intersection(slots) - self._release_permission):
            raise ValueError("BF16 cached slots require actual cache release")
        self.invalidate()
        for slot in slots:
            del self._generations[slot]
        if deferred:
            self._pending.update(slots)

    def release_cached(self, cache, values):
        from contextlib import contextmanager
        slots = self._slots(values)
        if self._cache is None or self._cache() is not cache or not set(slots) <= self.cached_slots():
            raise ValueError("foreign BF16 prefix ownership release")
        @contextmanager
        def permission():
            if self._release_permission:
                raise ValueError("nested BF16 prefix release")
            self._release_permission = frozenset(slots)
            try:
                yield
            finally:
                self._release_permission = frozenset()
        return permission()

    def refuse_reset(self):
        self.check_process()
        if self._generations or self._pending or self.cached_slots():
            raise ValueError("BF16 allocator/cache reset with owned slots is unsupported")
        self.invalidate()


from typing import NamedTuple


class BF16Namespace(NamedTuple):
    capability: tuple
    original_tag: object
    typed_tag: tuple


def _bf16_tag_identity(tag, depth=0, budget=None):
    budget = [4096] if budget is None else budget
    budget[0] -= 1 + (len(tag) if type(tag) in (str, bytes) else 0)
    if budget[0] < 0:
        raise ValueError("BF16 cache tag exceeds the aggregate bound")
    if depth > 16:
        raise ValueError("BF16 cache tag exceeds the depth bound")
    if type(tag) in (str, bytes, int, bool, type(None)):
        if isinstance(tag, (str, bytes)) and len(tag) > 4096:
            raise ValueError("BF16 cache tag exceeds the size bound")
        return (type(tag).__name__, tag)
    if type(tag) is tuple and len(tag) <= 64:
        return ("tuple", tuple(_bf16_tag_identity(value, depth+1, budget) for value in tag))
    raise ValueError("unsupported mutable or ambiguous BF16 cache tag")
