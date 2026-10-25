"""Selected-profile adapter for the exact vendored BetterBench runner."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def exclusive(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def verify_upstream():
    manifest = json.loads((ROOT / "UPSTREAM.json").read_bytes())
    root = ROOT / "upstream"
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("upstream membership differs; use -B and preserve changed evidence")
    for name, row in manifest["files"].items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("upstream source differs: " + name)
    return digest(ROOT / "UPSTREAM.json")


def profile():
    value = json.loads((ROOT / "profile.json").read_bytes())
    cfg = value["config"]
    if (value["schema"] != 1 or value["python"] != "3.14.4" or
            value["request_body"] != {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"} or
            value["context_tokens"] != 32768 or value["output_tokens"] != 4096 or
            cfg["concurrency_levels"] != [1, 2] or cfg["prefill_depths"] != [2000, 8000, 16000] or
            cfg["max_model_len"] != value["context_tokens"] or
            cfg["prefill_max_tokens"] + max(cfg["prefill_depths"]) + cfg["prefill_ctx_margin"] > 32768 or
            request_count(value) != value["max_requests"] or value["max_requests"] != 53 or
            value["campaign_seconds"] != 2400 or value["request_timeout_seconds"] != 120 or
            value["max_artifact_bytes"] != 32 << 20):
        raise ValueError("selected benchmark bounds/profile differ")
    return value


def request_count(value, categories=8):
    c = value["config"]
    return (categories * (c["warmup"] + c["runs_per_category"]) +
            len(c["prefill_depths"]) * (c["prefill_warmup"] + c["prefill_runs"]) +
            len(c["concurrency_levels"]) * c["concurrency_requests"])


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            parsed.path != "/v1" or parsed.port is None or not 1024 <= parsed.port <= 65535):
        raise ValueError("explicit loopback HTTP /v1 endpoint required")
    return value


def numpy_inventory(numpy):
    roots = [Path(numpy.__file__).resolve().parent]
    libraries = roots[0].parent / "numpy.libs"
    if libraries.exists():
        roots.append(libraries)
    files = {}
    total = 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc" or not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            if path.is_symlink() or size > 128 << 20 or total > 1 << 30 or len(files) >= 10000:
                raise ValueError("NumPy inventory type/size bound differs")
            files[str(path)] = {"bytes": size, "sha256": digest(path)}
    return files


def native_preflight():
    """CPU-only native dependency observation; never import Torch or SGLang."""
    if (platform.python_version() != "3.14.4" or platform.system() != "Linux" or
            platform.machine() != "x86_64"):
        raise ValueError("native Linux x86_64 Python 3.14.4 required")
    import numpy
    version = tuple(int(part) for part in numpy.__version__.split(".")[:2])
    if version < (1, 24):
        raise ValueError("upstream requires NumPy >=1.24")
    executable = Path(sys.executable).resolve()
    if "torch" in sys.modules or "sglang" in sys.modules:
        raise ValueError("native dependency probe imported a model runtime")
    return {"schema": 1, "python": platform.python_version(), "architecture": platform.machine(),
        "sys_executable": sys.executable, "executable": str(executable), "executable_sha256": digest(executable),
        "numpy_version": numpy.__version__, "numpy_path": str(Path(numpy.__file__).resolve()),
        "numpy_files": numpy_inventory(numpy),
        "torch_imported": "torch" in sys.modules, "sglang_imported": "sglang" in sys.modules,
        "gpu_probe_performed": False, "model_load_performed": False}


def load_upstream():
    verify_upstream()
    sys.path.insert(0, str(ROOT / "upstream"))
    return importlib.import_module("betterbench.runner")


class SelectedRequests:
    """Bind every upstream request path and preserve partial request results."""
    def __init__(self, runner, value, output, *, deadline=None):
        self.runner, self.value, self.output = runner, value, Path(output)
        self.deadline = deadline if deadline is not None else time.monotonic() + value["campaign_seconds"]
        self.count = self.active = self.peak_active = 0
        self.original = runner.stream_chat

    def record(self, row):
        raw = canonical(row)
        journal = self.output / "requests.jsonl"
        current = journal.stat().st_size if journal.exists() else 0
        if current + len(raw) > self.value["max_artifact_bytes"] // 2:
            raise ValueError("partial request evidence cap reached")
        with journal.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    async def stream_chat(self, url, model, messages, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.count >= self.value["max_requests"] or self.active >= 2:
            raise ValueError("request/deadline/concurrency ceiling reached")
        endpoint(url)
        if kwargs.get("api_key") is not None or kwargs.get("extra_body"):
            raise ValueError("unexpected authentication or body override")
        from betterbench.client import _build_payload
        kwargs["extra_body"] = self.value["request_body"]
        if kwargs.get("category") != "prefill":
            kwargs["max_tokens"] = self.value["output_tokens"]
        kwargs["timeout"] = min(remaining, self.value["request_timeout_seconds"])
        body = _build_payload(model, messages, max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"], top_p=kwargs.get("top_p", .95),
            top_k=kwargs.get("top_k", 20), seed=kwargs.get("seed"), extra_body=kwargs["extra_body"])
        ordinal = self.count
        self.count += 1
        self.record({"event": "before", "ordinal": ordinal, "body": body,
                     "body_sha256": hashlib.sha256(canonical(body)).hexdigest()})
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            result = await self.original(url, model, messages, **kwargs)
            self.record({"event": "after", "ordinal": ordinal, "result": result.as_dict()})
            if not result.ok:
                raise RuntimeError("benchmark request failed; inspect retained request result")
            if (type(result.prompt_tokens) is not int or type(result.completion_tokens) is not int or
                    result.prompt_tokens <= 0 or not 0 < result.completion_tokens <= kwargs["max_tokens"] or
                    result.prompt_tokens + kwargs["max_tokens"] + 32 > self.value["context_tokens"]):
                raise ValueError("reported usage/context envelope differs")
            return result
        except BaseException as error:
            self.record({"event": "failure", "ordinal": ordinal, "type": type(error).__name__})
            raise
        finally:
            self.active -= 1

    @contextmanager
    def installed(self):
        self.runner.stream_chat = self.stream_chat
        try:
            yield self
        finally:
            self.runner.stream_chat = self.original


def run_worker(url, model, output, dependency_receipt, dependency_sha256):
    output = Path(output)
    if digest(dependency_receipt) != dependency_sha256:
        raise ValueError("native dependency receipt pin differs")
    observed = native_preflight()
    if observed != json.loads(Path(dependency_receipt).read_bytes()):
        raise ValueError("native interpreter/NumPy/source changed after preflight")
    value = profile()
    upstream_pin = verify_upstream()
    runner = load_upstream()
    from betterbench.config import Config
    from betterbench.corpus import load_corpus
    from betterbench.html_report import render_html
    from betterbench.report import sample_gate
    cfg = Config(**value["config"])
    corpus = load_corpus(ROOT / "upstream/corpus/v1")
    if len(corpus) != 8 or sum(map(len, corpus.values())) != 29:
        raise ValueError("complete pinned corpus required")
    corpus = {name: [replace(prompt, max_tokens=value["output_tokens"]) for prompt in rows]
              for name, rows in corpus.items()}
    results = {"schema": 2, "betterbench_version": "0.6.1", "corpus_version": "1.0",
        "config": cfg.as_dict(), "selected_profile": value,
        "env": {"model": model, "categories": list(corpus), "notes": {
            "scope": "bounded selected-profile performance; correctness is graded separately",
            "upstream_manifest_sha256": upstream_pin,
            "profile_sha256": digest(ROOT / "profile.json"), "dependency_receipt_sha256": dependency_sha256}},
        "bounded_checkpoint": True, "default_benchmark_qualified": False, "qualification": False}
    bound = SelectedRequests(runner, value, output)
    with bound.installed():
        results["single_stream"] = asyncio.run(runner.single_stream(url, model, corpus, cfg))
        results["prefill"] = asyncio.run(runner.prefill_sweep(url, model, cfg, max_ctx=32768))
        results["concurrency"] = asyncio.run(runner.concurrency_sweep(url, model, corpus, cfg))
    if bound.count != request_count(value) or bound.peak_active > 2:
        raise ValueError("actual request count/concurrency differs")
    results["actual_requests"] = bound.count
    results["peak_client_concurrency"] = bound.peak_active
    results["sample_gate"] = sample_gate(results)
    raw, html = canonical(results), render_html(results).encode()
    # Reserve space for buffered stdout and the supervisor/summary terminals.
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + len(raw) + len(html) + (64 << 10) > value["max_artifact_bytes"]:
        raise ValueError("combined benchmark artifact cap reached")
    with (output / "results.json").open("xb") as stream:
        stream.write(raw)
    with (output / "results.html").open("xb") as stream:
        stream.write(html)
    return {"requests": bound.count, "peak_client_concurrency": bound.peak_active,
            "results_sha256": digest(output / "results.json"), "qualification": False}
