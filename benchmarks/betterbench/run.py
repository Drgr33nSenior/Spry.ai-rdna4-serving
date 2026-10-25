"""Finite local client supervisor; does not launch or stop a model server."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import selected


def supervise(argv, output, seconds):
    started = time.monotonic()
    terminal = {"schema": 1, "completed": False, "qualification": False, "timed_out": False}
    with (output / "client.stdout").open("xb") as stdout, (output / "client.stderr").open("xb") as stderr:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "BETTERBENCH_NO_UPDATE_CHECK": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2"})
        try:
            terminal["exit_code"] = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            terminal["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            terminal["exit_code"] = process.returncode
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
        finally:
            terminal["elapsed_seconds"] = time.monotonic() - started
            terminal["child_reaped"] = process.poll() is not None
            terminal["completed"] = terminal.get("exit_code") == 0 and not terminal["timed_out"]
            selected.exclusive(output / "terminal.json", terminal)
    return 0 if terminal["completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-preflight", action="store_true")
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--dependency-receipt", type=Path)
    parser.add_argument("--dependency-sha256")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.source_preflight:
        selected.exclusive(args.output, {"upstream_manifest_sha256": selected.verify_upstream(),
            "profile": selected.profile(), "coverage": "source-only", "native_compatible": False})
        return 0
    if args.native_preflight:
        selected.exclusive(args.output, selected.native_preflight())
        return 0
    if not all((args.endpoint, args.model, args.dependency_receipt, args.dependency_sha256)):
        parser.error("run requires endpoint, model and pinned native dependency receipt")
    selected.endpoint(args.endpoint)
    selected.verify_upstream()
    value = selected.profile()
    if args.worker:
        summary = selected.run_worker(args.endpoint, args.model, args.output,
            args.dependency_receipt, args.dependency_sha256)
        selected.exclusive(args.output / "summary.json", summary)
        return 0
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    selected.exclusive(args.output / "admission.json", {"schema": 1, "profile": value,
        "model": args.model, "dependency_receipt_sha256": args.dependency_sha256,
        "upstream_manifest_sha256": selected.digest(ROOT / "UPSTREAM.json"),
        "model_server_started": False, "qualification": False})
    argv = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
        "--output", str(args.output.resolve()), "--endpoint", args.endpoint, "--model", args.model,
        "--dependency-receipt", str(args.dependency_receipt.resolve()),
        "--dependency-sha256", args.dependency_sha256]
    return supervise(argv, args.output, value["campaign_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())
