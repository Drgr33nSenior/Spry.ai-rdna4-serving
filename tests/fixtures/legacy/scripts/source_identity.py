"""Generate reproducible local source identity and a patch from the pinned archive."""
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
ARCHIVE_SHA256 = "8ee50ca05e5d866074f0666fbd9697f9a42564b49874b74b4bd0df78e19e7084"
UPSTREAM = ROOT / "upstream" / ("sglang-" + REVISION)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def source_identity():
    archive = ROOT / "upstream" / ("sglang-" + REVISION + ".tar.gz")
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("pinned upstream archive checksum mismatch")
    original, links = {}, {}
    with tarfile.open(archive) as source:
        for entry in source:
            if entry.isfile():
                relative = Path(*Path(entry.name).parts[1:]).as_posix()
                if ".." in Path(relative).parts:
                    raise ValueError("unsafe archive member")
                original[relative] = source.extractfile(entry).read()
            elif entry.issym():
                links[Path(*Path(entry.name).parts[1:]).as_posix()] = entry.linkname
    changed, patch, runtime_files = {}, [], {}
    # Cover actual upstream source content, including unchanged files. Build
    # products and bytecode are excluded, and never become another source input.
    names = set(original)
    for path in UPSTREAM.rglob("*"):
        if path.is_symlink():
            relative = path.relative_to(UPSTREAM).as_posix()
            if links.get(relative) != str(path.readlink()):
                raise ValueError("extracted source symlink differs from the pinned archive")
            continue
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name not in (
            "libuq_hip.so", "ultraquant-build.json"
        ):
            names.add(path.relative_to(UPSTREAM).as_posix())
    for relative, target in links.items():
        path = UPSTREAM / relative
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError("pinned source symlink missing or changed")
    for name in sorted(names):
        path = UPSTREAM / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or nonregular source file: {name}")
        data = path.read_bytes()
        runtime_files[name] = digest(data)
        if data != original.get(name):
            changed[name] = digest(data)
            patch.extend(difflib.unified_diff(original.get(name, b"").decode().splitlines(True),
                                              data.decode().splitlines(True),
                                              fromfile="a/" + name, tofile="b/" + name))
    local_files = {}
    for directory in ("codec", "runtime", "scripts", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in experiment source")
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".hip"):
                local_files[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    local_files["Makefile"] = digest((ROOT / "Makefile").read_bytes())
    patch_bytes = "".join(patch).encode()
    identity = {
        "schema": 1, "source_repository": "sgl-project/sglang", "source_revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256, "patch_sha256": digest(patch_bytes),
        "source_tree_sha256": digest(canonical({"upstream": runtime_files, "upstream_links": links,
                                               "experiment": local_files})),
        "changed_upstream_files": changed, "experiment_files": local_files,
        "runtime_files": runtime_files,
        "runtime_links": links,
    }
    output = ROOT / "patches"
    output.mkdir(exist_ok=True)
    (output / "sglang-ultraquant.patch").write_bytes(patch_bytes)
    (output / "source.json").write_bytes(canonical(identity) + b"\n")
    return identity


if __name__ == "__main__":
    result = source_identity()
    print(json.dumps({name: result[name] for name in (
        "source_revision", "archive_sha256", "source_tree_sha256", "patch_sha256")}, indent=2))
