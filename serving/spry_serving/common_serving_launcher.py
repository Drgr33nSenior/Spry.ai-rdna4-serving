"""Minimal binding of the native HTTP launch seams to a worker manifest."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import worker_entry


class LaunchBindingError(ValueError):
    """The local HTTP launch binding is not safe to invoke."""


def _load_spry_common(caller, pin, error):
    """Authenticate common source before any shared guard can execute."""
    import hashlib
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import os
    from pathlib import Path
    import stat

    def check(value, message):
        if not value:
            raise error('shared common source: ' + message)

    origin = Path(caller)
    path = origin.with_name('spry_common_canon.py')
    if origin.name.startswith('common_') and origin.parent.name in ('spry_serving', 'spry_evidence'):
        path = origin.resolve().parents[1] / 'spry_common/selected.py'
    elif (origin.parent.name == 'review' and origin.parent.parent.name == 'common-closure'
          and origin.parent.parent.parent.name == 'proposals'):
        path = origin.resolve().parents[3] / 'spry_common/selected.py'
    check(path.is_absolute() and path.resolve(strict=True) == path, 'canonical path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 < before.st_size <= 4096, 'bounded regular file required')
        raw = os.read(descriptor, 4097)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                             row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        check(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(descriptor))
              == stamp(path.lstat()), 'file changed during read')
    finally:
        os.close(descriptor)
    check(hashlib.sha256(raw).hexdigest() == pin, 'digest differs')
    # Use the checked bytes, ignoring ambient modules and cached bytecode.
    loader = SourceFileLoader('_spry_common', str(path))
    loader.get_code = lambda name: loader.source_to_code(raw, str(path))
    spec = importlib.util.spec_from_loader('_spry_common', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_SPRY_COMMON_SHA256 = 'c2cb7702d6ae482b0f9d70b6b838af3d0b9eb312140ff09d001df0880a9e6bed'
_SPRY_COMMON = _load_spry_common(__file__, _SPRY_COMMON_SHA256, LaunchBindingError)


_canonical = _SPRY_COMMON.canonical_bare


def _digest(value: Any) -> str:
    return _SPRY_COMMON.digest(_canonical(value))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    return _SPRY_COMMON.need(condition, message, error=LaunchBindingError, prefix='')


def _read_checked_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), "worker manifest path invalid")
    _require(_file_digest(path) == expected_sha256, "worker manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(type(manifest) is dict, "worker manifest must be an object")
    # The worker validates all detailed fields.  Validate the outer gate before
    # exposing a spawned callable to native Engine.
    _require(manifest.get("schema") == 1 and manifest.get("mode") == "native-http-fp8-sprint01-v1", "worker manifest mode differs")
    _require(manifest.get("arm") in ("control", "mtp"), "worker manifest arm differs")
    _require(manifest.get("expected_ranks") == [0, 1], "worker ranks differ")
    _require(type(manifest.get("evidence_dir")) is str, "worker evidence directory missing")
    worker_entry._validate_manifest(manifest)
    launcher = manifest["sources"]["serving_launcher"]
    _require(
        Path(launcher["path"]) == Path(__file__).resolve()
        and launcher["sha256"] == _file_digest(Path(__file__).resolve()),
        "launcher helper origin differs",
    )
    return manifest


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(_canonical(value))
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)


def _read_worker(path: Path, manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "worker identity file missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    _require(type(record) is dict and record.get("identity_sha256") == worker_entry.identity_digest({key: value for key, value in record.items() if key != "identity_sha256"}), "worker identity digest differs")
    expected = {
        "schema": 1,
        "mode": manifest["mode"],
        "arm": manifest["arm"],
        "rank": rank,
        "session_sha256": manifest["session_sha256"],
        "spec_sha256": manifest["spec_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "arm_config": manifest["arm_config"],
        "sources": manifest["sources"],
    }
    _require(all(record.get(key) == value for key, value in expected.items()), "worker identity binding differs")
    _require(type(record.get("pid")) is int and record["pid"] > 0 and type(record.get("start_ticks")) is int and record["start_ticks"] > 0, "worker process fields invalid")
    _require(type(record.get("boot_id")) is str and record["boot_id"], "worker boot identity missing")
    return record


def make_post_warmup_callback(
    manifest: dict[str, Any], on_ready: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[], None]:
    """Return the one native post-warmup callback.  It has no model side effects."""
    evidence = Path(manifest["evidence_dir"])

    def record_ready() -> None:
        workers = [_read_worker(evidence / f"worker-rank-{rank}.json", manifest, rank) for rank in manifest["expected_ranks"]]
        value = {
            "schema": 1,
            "mode": manifest["mode"],
            "status": "NATIVE_POSTWARMUP_READY",
            "arm": manifest["arm"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "session_sha256": manifest["session_sha256"],
            "spec_sha256": manifest["spec_sha256"],
            "post_warmup_native_ready": True,
            "workers": [{key: worker[key] for key in ("rank", "pid", "start_ticks", "boot_id", "identity_sha256")} for worker in workers],
        }
        _atomic_json(evidence / "NATIVE-READINESS.json", value)
        if on_ready is not None:
            on_ready(value)

    return record_ready


@contextmanager
def _worker_manifest_environment(path: Path, expected_sha256: str) -> Iterator[None]:
    keys = ("FP8_HTTP_WORKER_MANIFEST", "FP8_HTTP_WORKER_MANIFEST_SHA256")
    _require(all(key not in os.environ for key in keys), "worker manifest environment already set")
    os.environ[keys[0]] = str(path)
    os.environ[keys[1]] = expected_sha256
    try:
        yield
    finally:
        for key in keys:
            os.environ.pop(key, None)


def launch_native_http(
    server_args: Any,
    *,
    native_launch_server: Callable[..., Any],
    worker_manifest: Path,
    worker_manifest_sha256: str,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Call native HTTP ``launch_server`` while preserving every other default seam."""
    manifest = _read_checked_manifest(worker_manifest, worker_manifest_sha256)
    callback = make_post_warmup_callback(manifest, on_ready)
    from sglang.srt.entrypoints.engine import init_tokenizer_manager

    contract = worker_entry._source_module(dict(manifest['sources']['mtp_contract'], module='mtp_contract'))

    def checked_tokenizer(*args, **kwargs):
        managers = init_tokenizer_manager(*args, **kwargs)
        if contract.instrumentation_enabled():  # production mode: the client still checks every response's prompt IDs
            import serving_parity
            serving_parity.collect(*managers, manifest)
        return managers

    with _worker_manifest_environment(worker_manifest, worker_manifest_sha256):
        return native_launch_server(
            server_args,
            init_tokenizer_manager_func=checked_tokenizer,
            run_scheduler_process_func=worker_entry.run_scheduler_process,
            launch_callback=callback,
        )
