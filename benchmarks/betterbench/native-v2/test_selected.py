"""Offline source/payload regressions against the actual pinned runner/client."""
import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import selected


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = selected.load_upstream()
        from betterbench.config import Config
        from betterbench.corpus import load_corpus
        cls.Config = Config
        cls.corpus = load_corpus(ROOT / "upstream/corpus/v1")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.value = selected.profile()
        self.bodies = []
        self.fail = False
        self.usage = {"prompt_tokens": 32, "completion_tokens": 8}
        self.active = self.peak = 0
        self.lock = threading.Lock()
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with case.lock:
                    case.bodies.append(body)
                    case.active += 1
                    case.peak = max(case.peak, case.active)
                try:
                    if case.fail:
                        self.send_error(500, "retained test failure")
                        return
                    chunks = [
                        {"choices": [{"delta": {"content": "<think>check</think>"}}]},
                        {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                         "usage": case.usage},
                    ]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in chunks:
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(.001)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with case.lock:
                        case.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = "http://127.0.0.1:" + str(self.server.server_port) + "/v1"

    def receipts(self):
        return [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]

    def assert_payloads(self):
        self.assertTrue(self.bodies)
        for body in self.bodies:
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(body["reasoning_effort"], "low")
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertIn(body["max_tokens"], (16, 4096))

    def test_actual_bounded_runner_paths_send_exact_53_requests_at_c1_c2(self):
        # Mock only native admission. The runner, wire client, payload adapter,
        # result serialization and HTML renderer execute their actual source.
        receipt = self.output / "dependency-fixture.json"
        observed = {"explicit_source_only_test_fixture": True}
        selected.exclusive(receipt, observed)
        with patch.object(selected, "native_preflight", return_value=observed):
            summary = selected.run_worker(self.url, "mock", self.output, receipt, selected.digest(receipt))
        results = json.loads((self.output / "results.json").read_bytes())
        single, prefill, concurrent = (results[key] for key in ("single_stream", "prefill", "concurrency"))
        self.assertEqual(summary["requests"], 53)
        self.assertEqual(len(self.bodies), 53)
        self.assertEqual(results["peak_client_concurrency"], 2)
        self.assertLessEqual(self.peak, 2)
        self.assertEqual(len(single), 8)
        self.assertEqual([row["target_depth"] for row in prefill], [2000, 8000, 16000])
        self.assertEqual([row["level"] for row in concurrent], [1, 2])
        self.assertEqual(sum(row["event"] == "after" for row in self.receipts()), 53)
        self.assertTrue(results["bounded_checkpoint"])
        self.assertFalse(results["default_benchmark_qualified"])
        self.assertGreater((self.output / "results.html").stat().st_size, 1000)
        self.assert_payloads()

    def test_actual_paired_ab_path_also_preserves_selected_body(self):
        cfg = self.Config(warmup=1, ab_min_pairs=99, ab_max_pairs=2)
        corpus = {"code": self.corpus["code"][:1]}
        with selected.SelectedRequests(self.runner, self.value, self.output).installed() as bound:
            result = asyncio.run(self.runner.paired_ab(self.url, self.url, "mock", corpus, cfg, log=lambda *_: None))
        self.assertEqual(bound.count, 6)
        self.assertEqual(result["pairs"], 2)
        self.assert_payloads()

    def test_failed_response_stops_and_retains_partial_evidence_without_retry(self):
        self.fail = True
        cfg = self.Config(warmup=0, runs_per_category=3)
        original = self.runner.stream_chat
        with self.assertRaisesRegex(RuntimeError, "benchmark request failed"):
            with selected.SelectedRequests(self.runner, self.value, self.output).installed() as bound:
                asyncio.run(self.runner.single_stream(self.url, "mock", {"code": self.corpus["code"][:1]}, cfg))
        self.assertEqual(bound.count, 1)
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual([row["event"] for row in self.receipts()], ["before", "after", "failure"])
        self.assertIs(self.runner.stream_chat, original)

    def test_request_and_deadline_caps_refuse_before_another_send(self):
        value = deepcopy(self.value)
        value["max_requests"] = 1
        cfg = self.Config(warmup=0, runs_per_category=3)
        with self.assertRaisesRegex(ValueError, "ceiling reached"):
            with selected.SelectedRequests(self.runner, value, self.output).installed():
                asyncio.run(self.runner.single_stream(self.url, "mock", {"code": self.corpus["code"][:1]}, cfg))
        self.assertEqual(len(self.bodies), 1)
        bound = selected.SelectedRequests(self.runner, value, self.output, deadline=time.monotonic() - 1)
        with self.assertRaisesRegex(ValueError, "ceiling reached"):
            asyncio.run(bound.stream_chat(self.url, "mock", [], max_tokens=1, temperature=0))
        self.assertEqual(len(self.bodies), 1)

    def test_missing_usage_refuses_instead_of_using_chunk_count_as_tokens(self):
        self.usage = None
        cfg = self.Config(warmup=0, runs_per_category=3)
        with self.assertRaisesRegex(ValueError, "usage/context envelope differs"):
            with selected.SelectedRequests(self.runner, self.value, self.output).installed():
                asyncio.run(self.runner.single_stream(self.url, "mock", {"code": self.corpus["code"][:1]}, cfg))
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual([row["event"] for row in self.receipts()], ["before", "after", "failure"])


class SourceTests(unittest.TestCase):
    def test_exact_source_membership_and_profile_counts(self):
        self.assertEqual(len(selected.verify_upstream()), 64)
        self.assertEqual(selected.request_count(selected.profile()), 53)

    def test_external_endpoints_and_credentials_refuse(self):
        for url in ("https://127.0.0.1:1234/v1", "http://example.org:1234/v1",
                    "http://user:pass@127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1?q=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                selected.endpoint(url)

    def test_source_only_interpreter_cannot_claim_native_preflight(self):
        with patch.object(selected.platform, "python_version", return_value="3.11.1"):
            with self.assertRaisesRegex(ValueError, "3.14.4 required"):
                selected.native_preflight()

    def test_actual_supervisor_reaps_timed_out_child_and_preserves_terminal(self):
        spec = importlib.util.spec_from_file_location("betterbench_supervisor_test", ROOT / "run.py")
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = supervisor.supervise([sys.executable, "-I", "-B", "-c", "import time; time.sleep(30)"], output, .05)
            terminal = json.loads((output / "terminal.json").read_bytes())
            self.assertEqual(result, 1)
            self.assertTrue(terminal["timed_out"])
            self.assertTrue(terminal["child_reaped"])
            self.assertFalse(terminal["completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
