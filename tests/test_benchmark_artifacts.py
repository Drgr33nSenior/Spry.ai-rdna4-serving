"""Raw-byte CPU transport fixtures; no model, remote process or GPU operations."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_artifacts.py"
SPEC = importlib.util.spec_from_file_location("spry_artifacts_cpu", SOURCE)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spry-artifacts-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "source"
        self.output.mkdir()
        self.binding = {"session_sha256": "1" * 64, "arm": "bf16", "spec_sha256": "2" * 64}
        for index, name in enumerate(artifacts.MEMBER_LIMITS):
            (self.output / name).write_bytes(b'{"retained_raw":' + str(index).encode() + b"}\n")

    def export(self):
        sink = io.BytesIO()
        result = artifacts.export_bundle(self.output, self.binding, sink)
        return result, sink.getvalue()

    def receive(self, value=None, stdout=None, destination="received"):
        if stdout is None:
            if value is None:
                _, stdout = self.export()
            else:
                stdout = artifacts.PREFIX + json.dumps(value, separators=(",", ":")).encode() + b"\n"
        return artifacts.receive_bundle(stdout, b"declared stderr\n", self.root / destination, self.binding)

    def test_export_receive_exact_raw_and_transport_is_not_qualification(self):
        _, raw = self.export()
        receipt = self.receive(stdout=b"unrelated bounded startup log\n" + raw)
        self.assertEqual(receipt["schema"], 2)
        self.assertTrue(receipt["transport_valid"])
        self.assertTrue(receipt["all_members_present"])
        self.assertFalse(receipt["qualification"])
        for name in artifacts.MEMBER_LIMITS:
            self.assertEqual((self.root / "received" / name).read_bytes(), (self.output / name).read_bytes())
            self.assertEqual((self.root / "received" / name).stat().st_mode & 0o777, 0o600)

    def test_missing_and_malformed_bytes_retained_without_claiming_complete(self):
        (self.output / "rank-0.json").unlink()
        (self.output / "rank-1.json").write_bytes(b"not-json\x00\xff{partial")
        value, _ = self.export()
        self.assertEqual(value["members"]["rank-0.json"], {"exists": False})
        receipt = self.receive(value)
        self.assertFalse(receipt["all_members_present"])
        self.assertFalse((self.root / "received/rank-0.json").exists())
        self.assertEqual((self.root / "received/rank-1.json").read_bytes(), b"not-json\x00\xff{partial")

    def test_export_extra_directory_symlink_hardlink_and_oversize_refuse(self):
        path = self.output / "unexpected"
        path.mkdir()
        with self.assertRaises(ValueError):
            self.export()
        path.rmdir()
        target = self.output / "rank-0.json"
        raw = target.read_bytes()
        target.unlink()
        target.symlink_to(self.output / "rank-1.json")
        with self.assertRaises(OSError):
            self.export()
        target.unlink()
        os.link(self.output / "rank-1.json", target)
        with self.assertRaises(ValueError):
            self.export()
        target.unlink()
        target.write_bytes(raw)
        with mock.patch.dict(artifacts.MEMBER_LIMITS, {"rank-0.json": 1}):
            with self.assertRaises(ValueError):
                self.export()

    def test_complete_maximum_payload_fits_declared_512mib_proof(self):
        for name, limit in artifacts.MEMBER_LIMITS.items():
            (self.output / name).write_bytes(b"x" * limit)
        value, raw = self.export()
        self.assertLessEqual(len(raw), artifacts.MAX_ENCODED_BYTES)
        self.assertLess(len(raw), 40 << 20)
        self.assertLess(artifacts.MAX_DURABLE_BYTES, 512 << 20)
        self.assertEqual(artifacts.MAX_SESSION_DURABLE_BYTES, int(467.25 * (1 << 20)))
        self.assertLess(artifacts.MAX_SESSION_DURABLE_BYTES, 512 << 20)
        receipt = self.receive(value)
        self.assertEqual(receipt["retained_payload_bytes"], artifacts.MAX_PAYLOAD_BYTES)

    def test_old_prefix_schema_or_five_file_bundle_refused(self):
        original, raw = self.export()
        with self.assertRaisesRegex(ValueError, "V2 artifact"):
            self.receive(stdout=raw.replace(artifacts.PREFIX, b"SPRY_BENCHMARK_BUNDLE_V1 ", 1))
        with self.assertRaisesRegex(ValueError, "schema/binding"):
            value = copy.deepcopy(original)
            value["schema"] = 1
            self.receive(value)
        with self.assertRaisesRegex(ValueError, "exact artifact members"):
            value = copy.deepcopy(original)
            value["members"] = {name: value["members"][name] for name in
                ("rank-0.json", "rank-1.json", "result.json", "rank-0-identity.json", "rank-1-identity.json")}
            self.receive(value)
        with self.assertRaisesRegex(ValueError, "V2 artifact"):
            self.receive(stdout=raw + raw.replace(artifacts.PREFIX, b"SPRY_BENCHMARK_BUNDLE_V1 ", 1))
        self.assertFalse((self.root / "received").exists())

    def test_duplicate_json_keys_names_wrong_binding_or_hash_refuse(self):
        original, raw = self.export()
        for mutate in (lambda item: item["binding"].update(arm="kv4"),
                       lambda item: item["members"].update({"escape": {"exists": False}}),
                       lambda item: item["members"]["rank-0.json"].update(sha256="f" * 64),
                       lambda item: item["members"]["rank-0.json"].update(bytes=True),
                       lambda item: item["members"]["rank-0.json"].update(base64="not base64!")):
            value = copy.deepcopy(original)
            mutate(value)
            with self.assertRaises((ValueError, TypeError)):
                self.receive(value)
            self.assertFalse((self.root / "received").exists())
        with self.assertRaises(ValueError):
            self.receive(stdout=raw + raw)
        with self.assertRaisesRegex(ValueError, "incomplete artifact"):
            self.receive(stdout=raw.rstrip(b"\n"))
        with self.assertRaises(ValueError):
            self.receive(stdout=artifacts.PREFIX + b'{"schema":1,"schema":1}\n')
        encoded = raw.replace(b'"members":{', b'"members":{"rank-0.json":{"exists":false},', 1)
        with self.assertRaises(ValueError):
            self.receive(stdout=encoded)

    def test_exclusive_destination_and_stdio_overflow_refuse(self):
        self.receive()
        before = (self.root / "received/result.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.receive()
        self.assertEqual((self.root / "received/result.json").read_bytes(), before)
        _, raw = self.export()
        with mock.patch.object(artifacts, "STREAM_LIMIT", 16):
            with self.assertRaises(ValueError):
                self.receive(stdout=raw)

    def test_bounded_pipe_capture_actual_owned_cpu_child(self):
        capture = artifacts.CapturePipes(self.root / "pipes")
        child = subprocess.Popen([sys.executable, "-I", "-B", "-S", "-c",
            "import os;os.write(1,b'raw stdout');os.write(2,b'raw stderr')"],
            stdout=capture.stdout, stderr=capture.stderr, start_new_session=True)
        self.assertEqual(child.wait(timeout=10), 0)
        capture.check()
        receipt = capture.finish()
        self.assertTrue(receipt["capture_complete"])
        self.assertTrue(receipt["reader_thread_absent"])
        self.assertFalse(receipt["remote_release_proven"])
        self.assertEqual((self.root / "pipes/stdout.raw").read_bytes(), b"raw stdout")
        with self.assertRaises(ProcessLookupError):
            os.killpg(child.pid, 0)
        with self.assertRaises(ValueError):
            capture.finish()

    def test_actual_capture_overflow_retains_cap_and_fails_not_truncated_pass(self):
        with mock.patch.object(artifacts, "STREAM_LIMIT", 1024):
            capture = artifacts.CapturePipes(self.root / "overflow")
            child = subprocess.Popen([sys.executable, "-I", "-B", "-S", "-c",
                "import os;os.write(1,b'x'*4096);os.write(2,b'y'*2048)"],
                stdout=capture.stdout, stderr=capture.stderr, start_new_session=True)
            self.assertEqual(child.wait(timeout=10), 0)
            receipt = capture.finish()
        self.assertFalse(receipt["capture_complete"])
        self.assertIn("overflow", receipt["failure"])
        for name in ("stdout", "stderr"):
            self.assertLessEqual((self.root / "overflow" / (name + ".raw")).stat().st_size, 1024)
        with self.assertRaises(ValueError):
            capture.check()
        with self.assertRaises(ProcessLookupError):
            os.killpg(child.pid, 0)

    def test_smaller_outer_transport_limit_is_explicit_and_enforced(self):
        with self.assertRaises(ValueError):
            artifacts.CapturePipes(self.root / "unbounded", stream_limit=artifacts.STREAM_LIMIT + 1)
        capture = artifacts.CapturePipes(self.root / "root-stdio", stream_limit=128)
        child = subprocess.Popen([sys.executable, "-I", "-B", "-S", "-c",
            "import os;os.write(1,b'x'*512)"], stdout=capture.stdout, stderr=capture.stderr,
            start_new_session=True)
        self.assertEqual(child.wait(timeout=10), 0)
        receipt = capture.finish()
        self.assertFalse(receipt["capture_complete"])
        self.assertEqual(receipt["streams"]["stdout"]["bytes"], 128)
        with self.assertRaises(ProcessLookupError):
            os.killpg(child.pid, 0)


class ArtifactStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spry-artifact-stage-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "workload"
        self.output.mkdir()
        self.binding = {"session_sha256": "1" * 64, "arm": "bf16", "spec_sha256": "2" * 64}

    def stage(self, output=None):
        return artifacts.stage_artifacts(output or self.output, self.binding)

    def test_binary_truncated_logs_partial_progress_and_results_survive_transport(self):
        log0 = b"\x00\xffnative stderr\r\n" * 160000
        log1 = b"\xffshort\x00\r\n"
        progress = b'{"phase":"prefill"}\n{"partial":\xff\x00'
        (self.output / "rank-0.log").write_bytes(log0)
        (self.output / "rank-1.log").write_bytes(log1)
        (self.output / "rank-0-progress.jsonl").write_bytes(progress)
        (self.output / "owned-workers.json").write_bytes(b"{partial-workers")
        (self.output / "rank-0.json").write_bytes(b"{partial-result")
        destination = self.stage()
        metadata = json.loads((destination / "diagnostics.json").read_bytes())
        self.assertEqual(set(metadata), {"schema", "binding", "logs", "errors"})
        self.assertEqual(metadata["schema"], 1)
        self.assertEqual(metadata["binding"], self.binding)
        self.assertEqual(metadata["errors"], [])
        for rank, raw in enumerate((log0, log1)):
            name = f"rank-{rank}.log.tail"
            tail = raw[-artifacts.MEMBER_LIMITS[name]:]
            self.assertEqual(metadata["logs"][str(rank)], {
                "exists": True, "missing": False, "original_bytes": len(raw),
                "full_sha256": hashlib.sha256(raw).hexdigest(), "tail_file": name,
                "tail_bytes": len(tail), "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "tail_offset": len(raw) - len(tail), "truncated": len(raw) > len(tail)})
            self.assertEqual((destination / name).read_bytes(), tail)
            self.assertEqual((self.output / f"rank-{rank}.log").read_bytes(), raw)
        sink = io.BytesIO()
        artifacts.export_bundle(destination, self.binding, sink)
        receipt = artifacts.receive_bundle(sink.getvalue(), b"", self.root / "received", self.binding)
        self.assertFalse(receipt["qualification"])
        self.assertFalse(receipt["all_members_present"])
        self.assertEqual((self.root / "received/rank-0-progress.jsonl").read_bytes(), progress)
        self.assertEqual((self.root / "received/owned-workers.json").read_bytes(), b"{partial-workers")
        self.assertEqual((self.root / "received/rank-0.json").read_bytes(), b"{partial-result")
        self.assertEqual((self.root / "received/diagnostics.json").read_bytes(),
                         (destination / "diagnostics.json").read_bytes())

    def test_missing_and_empty_logs_distinguished_and_stage_is_exclusive(self):
        (self.output / "rank-1.log").write_bytes(b"")
        destination = self.stage()
        metadata = json.loads((destination / "diagnostics.json").read_bytes())
        self.assertEqual(metadata["logs"]["0"], {"exists": False, "missing": True,
            "original_bytes": None, "full_sha256": None, "tail_file": "rank-0.log.tail",
            "tail_bytes": None, "tail_sha256": None, "tail_offset": None, "truncated": False})
        self.assertFalse((destination / "rank-0.log.tail").exists())
        self.assertTrue(metadata["logs"]["1"]["exists"])
        self.assertFalse(metadata["logs"]["1"]["missing"])
        self.assertEqual(metadata["logs"]["1"]["tail_bytes"], 0)
        self.assertEqual(metadata["logs"]["1"]["full_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertTrue((destination / "rank-1.log.tail").is_file())
        before = (destination / "diagnostics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual((destination / "diagnostics.json").read_bytes(), before)

    def test_full_log_limit_streams_in_bounded_blocks_and_retains_only_tail(self):
        log = self.output / "rank-0.log"
        with log.open("wb") as stream:
            stream.truncate(artifacts.RAW_LOG_LIMIT)
            stream.seek(artifacts.RAW_LOG_LIMIT - 3)
            stream.write(b"\x00\xff\x01")
        original_read, sizes = artifacts.os.read, []
        def recorded_read(fd, size):
            sizes.append(size)
            return original_read(fd, size)
        with mock.patch.object(artifacts.os, "read", side_effect=recorded_read):
            destination = self.stage()
        self.assertLessEqual(max(sizes), 65536)
        self.assertGreater(len(sizes), artifacts.RAW_LOG_LIMIT // 65536)
        expected = hashlib.sha256()
        with log.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                expected.update(block)
        record = json.loads((destination / "diagnostics.json").read_bytes())["logs"]["0"]
        self.assertEqual(record["original_bytes"], artifacts.RAW_LOG_LIMIT)
        self.assertEqual(record["full_sha256"], expected.hexdigest())
        self.assertEqual(record["tail_bytes"], 2 << 20)
        self.assertEqual(record["tail_offset"], (128 - 2) << 20)
        self.assertTrue(record["truncated"])
        self.assertEqual((destination / "rank-0.log.tail").stat().st_size, 2 << 20)

    def test_stage_links_nonregular_and_oversize_refuse_without_source_changes(self):
        for filename in ("rank-0.log", "rank-0-progress.jsonl", "owned-workers.json", "result.json"):
            for kind in ("symlink", "hardlink", "oversize", "fifo"):
                with self.subTest(filename=filename, kind=kind):
                    output = self.root / (filename + "-" + kind)
                    output.mkdir()
                    target = output / filename
                    original = output / "original"
                    original.write_bytes(b"protected")
                    if kind == "symlink":
                        target.symlink_to(original)
                    elif kind == "hardlink":
                        os.link(original, target)
                    elif kind == "fifo":
                        os.mkfifo(target)
                    else:
                        limit = artifacts.RAW_LOG_LIMIT if filename.endswith(".log") else artifacts.MEMBER_LIMITS[filename]
                        with target.open("wb") as stream:
                            stream.truncate(limit + 1)
                    with self.assertRaises((ValueError, OSError)):
                        self.stage(output)
                    self.assertEqual(original.read_bytes(), b"protected")

    def test_changed_source_during_staging_refuses(self):
        for filename in ("rank-0.log", "rank-0-progress.jsonl"):
            with self.subTest(filename=filename):
                output = self.root / (filename + "-changes")
                output.mkdir()
                target = output / filename
                target.write_bytes(b"native bytes")
                original_read, changed = artifacts.os.read, []
                def mutate_read(fd, size):
                    raw = original_read(fd, size)
                    if raw and not changed:
                        with target.open("ab") as stream:
                            stream.write(b"changed")
                        changed.append(True)
                    return raw
                with mock.patch.object(artifacts.os, "read", side_effect=mutate_read):
                    with self.assertRaisesRegex(ValueError, "changed/overflowed"):
                        self.stage(output)
                self.assertTrue(changed)
                self.assertFalse((output / "export/diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
