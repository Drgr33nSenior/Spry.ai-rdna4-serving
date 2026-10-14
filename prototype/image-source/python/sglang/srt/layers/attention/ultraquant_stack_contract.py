"""Shared FP8-owner stack receipt for KV4 preparation; stdlib only."""
import hashlib
import json
from pathlib import Path

SOURCE_REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"
TARGET = {"sglang": "0.5.21", "rocm": "10.1.0", "source_revision": SOURCE_REVISION}
ROLES = {"hipcc", "clang", "nm", "readelf", "objdump", "objcopy", "bundler"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def hex_digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("missing or malformed stack digest")
    return value


def checked_paths(files):
    if type(files) is not dict or not files:
        raise ValueError("missing stack source/library inventory")
    for name, value in files.items():
        if type(name) is not str or "\\" in name or "\0" in name:
            raise ValueError("unsafe stack source/library path")
        path = Path(name)
        if type(name) is not str or path.is_absolute() or not path.parts or any(
            part in (".", "..") for part in path.parts
        ) or path.as_posix() != name:
            raise ValueError("unsafe stack source/library path")
        hex_digest(value)
    return files


def unique_object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate stack receipt key")
        value[name] = item
    return value


def absolute_path(value):
    if type(value) is not str or "\\" in value or "\0" in value:
        raise ValueError("unresolved tool/interpreter path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("unresolved tool/interpreter path")
    return path


def read_receipt(path, expected_sha256):
    """Expected digest is an external owner input, never emitted by a producer."""
    if path is None or expected_sha256 is None:
        raise ValueError("shared stack receipt and digest must be supplied together")
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("missing bounded shared stack receipt")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != hex_digest(expected_sha256):
        raise ValueError("shared stack receipt digest changed")
    value = json.loads(raw, object_pairs_hook=unique_object)
    if type(value) is not dict or type(value.get("components")) is not dict:
        raise ValueError("malformed shared stack receipt")
    if (value.get("schema"), value.get("decision"), value.get("target")) != (
        1, "accept-common-stack-source-only", TARGET
    ) or not value.get("owner_handoff"):
        raise ValueError("unreviewed or incompatible shared stack")
    for name in ("image_manifest", "image_config", "installed_inventory_sha256"):
        hex_digest(value.get(name))
    for name in ("python", "sdk", "hip", "torch", "triton", "rccl", "kernel_packages", "aiter"):
        component = value.get("components", {}).get(name)
        if type(component) is not dict or type(component.get("version")) is not str or not component["version"]:
            raise ValueError("missing matched component identity: " + name)
        hex_digest(component.get("artifact_sha256"))
    if value["components"]["sdk"]["version"] != "10.1.0" or "gfx1201" not in value.get("device_targets", []):
        raise ValueError("wrong SDK or missing explicit gfx1201 dependency support")
    checked_paths(value.get("stock_files"))
    checked_paths(value.get("common_files"))
    repairs = value.get("repair_files")
    if type(repairs) is not dict or any(value["common_files"].get(p) != h for p, h in repairs.items()):
        raise ValueError("common repair membership changed")
    checked_paths(repairs) if repairs else None
    tools = value.get("tools")
    if type(tools) is not dict or set(tools) != ROLES:
        raise ValueError("missing complete tool role mapping")
    for tool in tools.values():
        if type(tool) is not dict:
            raise ValueError("unresolved tool/interpreter path")
        absolute_path(tool.get("path"))
        absolute_path(tool.get("interpreter"))
        absolute_path(tool.get("filesystem_root", "/"))
        hex_digest(tool.get("sha256"))
        hex_digest(tool.get("interpreter_sha256"))
        checked_paths(tool.get("libraries"))
    env = value.get("compiler_environment")
    if type(env) is not dict or any(type(k) is not str or type(v) is not str or not v for k, v in env.items()):
        raise ValueError("missing reviewed lookup environment")
    if value.get("static_closure_complete") is not True or value.get("dynamic_plugins_reviewed") is not True:
        raise ValueError("unresolved static or dynamic compiler dependencies")
    if value.get("installed_protocol_reviewed") is not True or type(value.get("protocol")) is not dict:
        raise ValueError("installed command syntax remains unreviewed")
    return value
