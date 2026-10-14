"""Bounded image-filesystem metadata collector. Never execute image tools."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

MAX_FILE = 1024**3
MAX_TABLE = 8 * 1024**2


def file_identity(path):
    path = Path(path)
    target = path.resolve(strict=True)
    if not target.is_file() or target.stat().st_size > MAX_FILE:
        raise ValueError("missing or oversized metadata input")
    links = {str(p): str(p.readlink()) for p in (path, *path.parents) if p.is_symlink()}
    checksum = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            checksum.update(chunk)
    return {"path": str(path), "canonical": str(target), "links": links,
            "bytes": target.stat().st_size, "sha256": checksum.hexdigest()}


def elf_metadata(path):
    """ELF64 little-endian host programs only; bounded static table reads."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("unsupported ELF encoding")
        values = struct.unpack("<16sHHIQQQIHHHHHH", header)
        if values[2] != 62 or values[9] != 56 or values[10] > 4096:
            raise ValueError("unsupported host ELF/program headers")
        segments = []
        for index in range(values[10]):
            stream.seek(values[5] + index * 56)
            raw = stream.read(56)
            if len(raw) != 56:
                raise ValueError("truncated ELF program header")
            segment = struct.unpack("<IIQQQQQQ", raw)
            if segment[5] > MAX_FILE or segment[2] + segment[5] > path.stat().st_size:
                raise ValueError("ELF segment out of bounds")
            segments.append(segment)
        if any(sum(s[0] == kind for s in segments) > 1 for kind in (2, 3)):
            raise ValueError("ambiguous ELF interpreter/dynamic segment")
        interpreter, dynamic = None, {}
        for segment in segments:
            if segment[0] == 3:
                if segment[5] > 4096:
                    raise ValueError("oversized ELF interpreter")
                stream.seek(segment[2]); raw = stream.read(segment[5])
                if not raw.endswith(b"\0") or b"\0" in raw[:-1] or not raw.startswith(b"/"):
                    raise ValueError("malformed ELF interpreter")
                interpreter = raw[:-1].decode("utf-8")
            if segment[0] == 2:
                if segment[5] > MAX_TABLE or segment[5] % 16:
                    raise ValueError("invalid ELF dynamic table")
                stream.seek(segment[2])
                terminated = False
                for tag, value in struct.iter_unpack("<qQ", stream.read(segment[5])):
                    if tag == 0:
                        terminated = True
                        break
                    dynamic.setdefault(tag, []).append(value)
                if not terminated:
                    raise ValueError("unterminated ELF dynamic table")
        if any(len(dynamic.get(tag, [])) > 1 for tag in (15, 29)):
            raise ValueError("ambiguous ELF lookup table")
        strings = b""
        if 5 in dynamic:
            if len(dynamic[5]) != 1 or len(dynamic.get(10, [])) != 1 or dynamic[10][0] > MAX_TABLE:
                raise ValueError("ambiguous/oversized ELF string table")
            address, size = dynamic[5][0], dynamic[10][0]
            candidates = [s for s in segments if s[0] == 1 and s[3] <= address and address + size <= s[3] + s[5]]
            if len(candidates) != 1:
                raise ValueError("unmapped ELF string table")
            segment = candidates[0]
            stream.seek(segment[2] + address - segment[3]); strings = stream.read(size)
        def names(tag):
            result = []
            for offset in dynamic.get(tag, []):
                if offset >= len(strings) or b"\0" not in strings[offset:]:
                    raise ValueError("ELF string out of bounds")
                result.append(strings[offset:].split(b"\0", 1)[0].decode("utf-8"))
            return result
        return {"interpreter": interpreter, "needed": names(1), "rpath": names(15), "runpath": names(29)}


def library_closure(tool, search_directories):
    """Explicit lookup profile only; ambiguity/cache/plugins remain review gaps."""
    if any(not Path(directory).is_absolute() for directory in search_directories):
        raise ValueError("nonabsolute explicit library lookup")
    pending = [Path(tool).resolve(strict=True)]
    files, gaps = {}, []
    while pending:
        path = pending.pop()
        if str(path) in files:
            continue
        if len(files) >= 256:
            raise ValueError("library closure exceeds bound")
        info = file_identity(path)
        metadata = elf_metadata(path)
        files[str(path)] = dict(info, elf=metadata)
        if metadata["interpreter"]:
            interpreter = Path(metadata["interpreter"])
            if interpreter.is_file():
                pending.append(interpreter.resolve())
            else:
                gaps.append({"interpreter": str(interpreter), "reason": "missing"})
        local = metadata["runpath"] or metadata["rpath"]
        if metadata["rpath"] and not metadata["runpath"] and metadata["needed"]:
            gaps.append({"path": str(path), "reason": "transitive RPATH inheritance requires review"})
        directories = []
        for value in local:
            for directory in value.split(":"):
                directory = directory.replace("${ORIGIN}", str(path.parent)).replace("$ORIGIN", str(path.parent))
                if "$" in directory or not Path(directory).is_absolute():
                    gaps.append({"lookup": directory, "reason": "unsupported expansion"})
                else:
                    directories.append(Path(directory))
        for name in metadata["needed"]:
            if "/" in name:
                gaps.append({"dependency": name, "reason": "path dependency requires review"}); continue
            matches = {str((directory/name).resolve()) for directory in directories + [Path(p) for p in search_directories]
                       if (directory/name).is_file()}
            if len(matches) == 1:
                pending.append(Path(matches.pop()))
            else:
                gaps.append({"dependency": name, "reason": "missing or ambiguous", "candidates": sorted(matches)})
    return {"files": files, "gaps": gaps, "static_profile_complete": not gaps,
            "dynamic_plugins_reviewed": False, "loader_cache_semantics_reviewed": False,
            "scope": "explicit lookup profile, not a complete glibc loader emulation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--lookup", required=True)
    args = parser.parse_args()
    tools = json.loads(Path(args.tools).read_bytes())
    lookup = json.loads(Path(args.lookup).read_bytes())
    result = {"schema": 1, "qualification": "unreviewed-metadata", "tools": {},
              "native_tools_executed": False, "sdk_expansion_executed": False}
    for name, path in tools.items():
        result["tools"][name] = library_closure(path, lookup)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
