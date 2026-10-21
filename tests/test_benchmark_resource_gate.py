"""CPU-only resource-gate composition; no systemd unit, cgroup or GPU is started.

Named temporary files stand in for kernel proc/cgroup records. Source bytes and
UID/GID observations are explicit substitutes. The real validation and refusal
bodies run, with an exec spy proving that invalid evidence cannot launch bwrap.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_resource_gate.py"
SPEC = importlib.util.spec_from_file_location("spry_benchmark_resource_gate", SOURCE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class ResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spry-resource-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.proc = self.base / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.cgroup = self.base / "cgroup"
        self.root = "/home/developer/fp8-kv-benchmark.CPUfixture0123"
        self.sources = {self.root + "/review/benchmark_resource_gate.py": SOURCE.read_bytes(),
                        self.root + "/review/model_benchmark.py": b"declared CPU entry bytes\n",
                        "/usr/bin/bwrap": b"declared CPU executable bytes\n"}
        sha = lambda value: hashlib.sha256(value).hexdigest()
        self.manifest = {"schema": 1, "scope": gate.SCOPE, "root": self.root,
            "session_sha256": "1" * 64, "benchmark_spec_sha256": "2" * 64,
            "gate_path": self.root + "/review/benchmark_resource_gate.py",
            "gate_sha256": sha(self.sources[self.root + "/review/benchmark_resource_gate.py"]),
            "entry_path": self.root + "/review/model_benchmark.py",
            "entry_sha256": sha(self.sources[self.root + "/review/model_benchmark.py"]),
            "bwrap_sha256": sha(self.sources["/usr/bin/bwrap"]),
            "units": {}, "workload_argv": {}, "workload_argv_sha256": {}}
        for arm in ("bf16", "kv4"):
            self.manifest["units"][arm] = "spry-fp8-benchmark-" + "1" * 16 + "-" + arm + ".scope"
            argv = ["/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
                    "--ro-bind", self.manifest["entry_path"], "/mnt/review/model_benchmark.py",
                    "/mnt/work/build-venv/bin/python3", "-I", "-B", "/mnt/review/model_benchmark.py",
                    "--execute-reviewed-benchmark", "--spec", "/mnt/review/BENCHMARK-SPEC2.json",
                    "--spec-sha256", "2" * 64, "--output", "/mnt/output/benchmark", "--arm", arm]
            self.manifest["workload_argv"][arm] = argv
            self.manifest["workload_argv_sha256"][arm] = gate.argv_sha256(argv)
        self.execs = []
        self.emitted = []
        self.limit_calls = []
        self.observation_now = 100.0
        self.observation_waits = []
        self.prepare_cgroup("bf16")

    def observation_sleep(self, seconds):
        self.observation_waits.append(seconds)
        self.observation_now += seconds

    def prepare_cgroup(self, arm):
        expected = gate.USER_MANAGER + "/app.slice/" + self.manifest["units"][arm]
        self.directory = self.cgroup / expected.lstrip("/")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.proc / "self/cgroup").write_text("0::" + expected + "\n")
        (self.proc / "self/mountinfo").write_text("18 1 0:42 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup rw\n")
        (self.proc / "self/stat").write_text(str(os.getpid()) + " (declared CPU gate) " +
            " ".join(["R"] + ["0"] * 18 + ["123456"]) + "\n")
        for name in ("cgroup.controllers", "cgroup.subtree_control"):
            (self.directory.parent / name).write_text("cpu memory pids\n")
        values = {"cpu.max": "2400000 100000\n", "memory.max": str(gate.MEMORY_BYTES) + "\n",
                  "memory.swap.max": "0\n", "pids.max": "512\n",
                  "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
                  "pids.events": "max 0\n", "memory.current": "123456\n", "pids.current": "2\n"}
        values.update({"memory.peak": "234567\n", "cgroup.events": "populated 1\nfrozen 0\n",
                       "cgroup.procs": str(os.getpid()) + "\n",
                       "cgroup.threads": str(os.getpid()) + "\n"})
        for name, value in values.items():
            (self.directory / name).write_text(value)

    def run_gate(self, manifest=None, arm="bf16", **changes):
        options = dict(gate_path=self.manifest["gate_path"],
                       source_reader=lambda path, _limit: self.sources[path],
                       proc_root=self.proc, cgroup_root=self.cgroup,
                       credentials=(1000, 1000, 1000, 1000),
                       execute=lambda executable, argv: self.execs.append((executable, list(argv))),
                       set_limit=lambda number, value: self.limit_calls.append((number, value)),
                       emit=self.emitted.append,
                       observation_clock=lambda: self.observation_now,
                       observation_sleeper=self.observation_sleep)
        options.update(changes)
        return gate.run_gate(manifest or self.manifest, arm, **options)

    def refused(self, **changes):
        self.execs.clear()
        self.emitted.clear()
        self.limit_calls.clear()
        with self.assertRaises((ValueError, OSError, KeyError)):
            self.run_gate(**changes)
        self.assertEqual(self.execs, [])
        self.assertEqual(self.emitted, [])

    def test_actual_composition_executes_exact_argv_after_each_arm_gate(self):
        for arm in ("bf16", "kv4"):
            with self.subTest(arm=arm):
                self.prepare_cgroup(arm)
                result = self.run_gate(arm=arm)
                self.assertTrue(result["actual_cgroup_verified"])
                self.assertFalse(result["scratch_quota_verified"])
                self.assertEqual(result["cpu_max"], [2400000, 100000])
                self.assertEqual(self.execs[-1], ("/usr/bin/bwrap", self.manifest["workload_argv"][arm]))
                self.assertEqual(self.emitted[-1], result)
                self.assertEqual(self.limit_calls[-2:], [(gate.resource.RLIMIT_CORE, (0, 0)),
                                 (gate.resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))])

    def test_cache_modes_keep_actual_resource_gate_and_exact_native_tail(self):
        path = self.root + '/review/cache_gpu_entry.py'
        self.sources[path] = b'declared CPU cache entry, never executed\n'
        self.manifest.update(entry_path=path, entry_sha256=hashlib.sha256(self.sources[path]).hexdigest(),
                             artifacts_sha256='a' * 64)
        for arm in ('bf16', 'kv4'):
            argv = ['/usr/bin/bwrap', '--die-with-parent', '--new-session', '--unshare-all',
                    '--ro-bind', path, '/mnt/review/cache_gpu_entry.py',
                    '--size', '67108864', '--tmpfs', '/mnt/output',
                    '/mnt/work/build-venv/bin/python3', '-I', '-B', '-S', '/mnt/review/cache_gpu_entry.py',
                    '--execute-native-cache', '--spec-sha256', '2' * 64, '--arm', arm]
            self.manifest['workload_argv'][arm] = argv
            self.manifest['workload_argv_sha256'][arm] = gate.argv_sha256(argv)
        for mode in ('native-cache-regression-v1', 'native-cache-counterbalanced-v2'):
            self.manifest['experiment_mode'] = mode
            def child(argv):
                self.execs.append((argv[0], list(argv)))
                (self.directory / 'pids.current').write_text('1\n')
                return {'child_exit': 0, 'child_reaped': True}
            self.assertTrue(self.run_gate(supervisor=child)['complete'])
            self.assertTrue(self.emitted[-2]['actual_cgroup_verified'])
            self.assertEqual(self.execs[-1][1], self.manifest['workload_argv']['bf16'])
        self.manifest['experiment_mode'] = 'native-cache-unknown'
        self.refused()
        self.manifest['experiment_mode'] = 'native-cache-counterbalanced-v2'
        self.manifest['workload_argv']['bf16'][-1] = 'kv4'
        self.manifest['workload_argv_sha256']['bf16'] = gate.argv_sha256(self.manifest['workload_argv']['bf16'])
        self.refused()

    def test_wrong_unbounded_or_malformed_cpu_never_executes(self):
        for value in ("max 100000", "1200000 100000", "2400000 0", "2400000", "true 100000"):
            with self.subTest(value=value):
                (self.directory / "cpu.max").write_text(value + "\n")
                self.refused()

    def test_wrong_memory_swap_pids_and_missing_files_never_execute(self):
        for name, value in (("memory.max", "max"), ("memory.max", str(gate.MEMORY_BYTES - 1)),
                            ("memory.swap.max", "1"), ("pids.max", "max"), ("pids.max", "511")):
            with self.subTest(name=name, value=value):
                self.prepare_cgroup("bf16")
                (self.directory / name).write_text(value + "\n")
                self.refused()
        self.prepare_cgroup("bf16")
        (self.directory / "cpu.max").unlink()
        self.refused()

    def test_missing_controllers_and_own_scope_or_mount_mismatch_refuse(self):
        for name in ("cgroup.controllers", "cgroup.subtree_control"):
            with self.subTest(name=name):
                self.prepare_cgroup("bf16")
                (self.directory.parent / name).write_text("cpu memory\n")
                self.refused()
        self.prepare_cgroup("bf16")
        (self.proc / "self/cgroup").write_text("0::/user.slice/other.scope\n")
        self.refused()
        self.prepare_cgroup("bf16")
        (self.proc / "self/mountinfo").write_text("18 1 0:42 / /sys/fs/cgroup rw - tmpfs tmpfs rw\n")
        self.refused()

    def test_nonzero_oom_pid_events_and_invalid_usage_refuse(self):
        for name, value in (("memory.events", "max 0\noom 1\noom_kill 0\n"),
                            ("memory.events", "max 0\noom 0\noom_kill 1\n"),
                            ("pids.events", "max 1\n"), ("memory.current", "0\n"),
                            ("pids.current", "513\n")):
            with self.subTest(name=name, value=value):
                self.prepare_cgroup("bf16")
                (self.directory / name).write_text(value)
                self.refused()

    def test_source_gate_path_credentials_and_arm_substitution_refuse(self):
        for path in self.sources:
            with self.subTest(path=path):
                self.refused(source_reader=lambda name, limit, changed=path:
                             b"substituted\n" if name == changed else self.sources[name])
        self.refused(gate_path=self.manifest["entry_path"])
        self.refused(credentials=(0, 0, 0, 0))
        self.refused(arm="fp4")

    def test_root_session_unit_argv_and_unknown_fields_refuse(self):
        mutations = (("root", "/home/developer/unknown"), ("root", "/home/developer/fp8-kv-benchmark../escape"),
                     ("session_sha256", "3" * 64), ("scope", "unknown"), ("schema", True))
        for name, value in mutations:
            with self.subTest(name=name, value=value):
                candidate = copy.deepcopy(self.manifest)
                candidate[name] = value
                self.refused(manifest=candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["extra"] = True
        self.refused(manifest=candidate)
        candidate = copy.deepcopy(self.manifest)
        candidate["workload_argv"]["bf16"][-1] = "kv4"
        candidate["workload_argv_sha256"]["bf16"] = gate.argv_sha256(candidate["workload_argv"]["bf16"])
        self.refused(manifest=candidate)

    def test_control_symlink_and_source_hardlink_refuse(self):
        original = self.directory / "cpu.max"
        original.rename(self.directory / "cpu-real")
        original.symlink_to("cpu-real")
        self.refused()
        path = self.base / "source"
        path.write_bytes(b"source\n")
        os.link(path, self.base / "source-link")
        with self.assertRaises(ValueError):
            gate.bounded_file(path)

    def test_manifest_hash_duplicate_keys_and_projection(self):
        path = self.base / "manifest.json"
        raw = json.dumps(self.manifest, sort_keys=True).encode()
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        self.assertEqual(gate.read_manifest(path, sha), self.manifest)
        command = gate.scope_argv(path, sha, "kv4")
        self.assertIn("--property=CPUQuota=2400%", command)
        self.assertIn("--property=MemoryMax=" + str(gate.MEMORY_BYTES), command)
        self.assertIn("--property=MemorySwapMax=0", command)
        self.assertIn("--property=TasksMax=512", command)
        self.assertIn("--unit=" + self.manifest["units"]["kv4"], command)
        self.assertEqual(command[-2:], ("--arm", "kv4"))
        with self.assertRaises(ValueError):
            gate.read_manifest(path, "0" * 64)
        duplicate = b'{"schema":1,"schema":1}'
        path.write_bytes(duplicate)
        with self.assertRaises(ValueError):
            gate.read_manifest(path, hashlib.sha256(duplicate).hexdigest())

    def test_default_cli_refuses_before_runtime_or_exec(self):
        with self.assertRaisesRegex(ValueError, "NOT_RUN"):
            gate.main([])

    def test_failed_limit_setter_does_not_emit_or_execute(self):
        def denied(_number, _value):
            raise OSError("declared CPU setter failure")
        self.refused(set_limit=denied)

    def physical_manifest(self):
        value = copy.deepcopy(self.manifest)
        value["artifacts_sha256"] = "4" * 64
        for arm in ("bf16", "kv4"):
            value["workload_argv"][arm] += ["--physical-artifacts", "--session-sha256", "1" * 64,
                                             "--artifacts-sha256", "4" * 64]
            value["workload_argv_sha256"][arm] = gate.argv_sha256(value["workload_argv"][arm])
        return value

    def test_physical_composition_waits_reaps_then_observes_live_wrapper_scope(self):
        value = self.physical_manifest()
        def child(argv):
            self.assertEqual(argv, value["workload_argv"]["bf16"])
            self.assertEqual(self.emitted[-1]["phase"], "initial")
            (self.directory / "pids.current").write_text("1\n")
            return {"child_pid": 999999, "child_exit": 0, "child_reaped": True, "owner_signals": []}
        result = self.run_gate(value, supervisor=child)
        self.assertEqual(self.execs, [])
        self.assertEqual([item["phase"] for item in self.emitted], ["initial", "final"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["only_wrapper_remains"])
        self.assertFalse(result["cgroup_release_proven"])
        self.assertFalse(result["qualification"])
        self.assertEqual(result["gate_process"], {"pid": os.getpid(), "start_ticks": 123456})
        self.assertEqual(result["cgroup_identity"]["inode"], self.directory.stat().st_ino)
        self.assertEqual(len(result["cgroup_observation"]["snapshots"]), 2)
        self.assertTrue(result["cgroup_observation"]["quiescence"]["complete"])
        self.assertEqual(result["cgroup_observation"]["quiescence"]["consecutive"], 2)

    def test_transient_count_is_retained_then_two_exact_wrapper_samples_are_required(self):
        def child(_argv):
            return {"child_exit": 0, "child_reaped": True}
        def settle(seconds):
            self.observation_sleep(seconds)
            (self.directory / "pids.current").write_text("1\n")
        final = self.run_gate(self.physical_manifest(), supervisor=child, observation_sleeper=settle)
        snapshots = final["cgroup_observation"]["snapshots"]
        self.assertEqual([row["usage"]["pids.current"] for row in snapshots], [2, 1, 1])
        self.assertEqual([row["only_wrapper_remains"] for row in snapshots], [False, True, True])
        self.assertTrue(final["complete"])
        self.assertEqual(self.observation_waits, [0.2, 0.2])
        self.assertFalse(final["cgroup_release_proven"])

    def test_quiescence_consecutive_count_resets_after_membership_changes(self):
        remaining = iter([2, 1, 1])
        def child(_argv):
            (self.directory / "pids.current").write_text("1\n")
            return {"child_exit": 1, "child_reaped": True}
        def transition(seconds):
            self.observation_sleep(seconds)
            (self.directory / "pids.current").write_text(str(next(remaining)) + "\n")
        final = self.run_gate(self.physical_manifest(), supervisor=child, observation_sleeper=transition)
        self.assertEqual([row["usage"]["pids.current"] for row in final["cgroup_observation"]["snapshots"]],
                         [1, 2, 1, 1])
        self.assertEqual(final["child_exit"], 1)

    def test_expiry_after_one_wrapper_sample_cannot_pass(self):
        def child(_argv):
            (self.directory / "pids.current").write_text("1\n")
            return {"child_exit": 0, "child_reaped": True}
        def late(seconds):
            self.observation_sleep(gate.FINAL_QUIESCENCE_SECONDS)
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=child, observation_sleeper=late)
        final = self.emitted[-1]
        self.assertFalse(final["complete"])
        self.assertEqual(len(final["cgroup_observation"]["snapshots"]), 1)
        self.assertFalse(final["cgroup_observation"]["quiescence"]["complete"])

    def test_mid_observation_deadline_crossing_is_retained_and_refused(self):
        original = gate._final_cgroup_snapshot
        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            self.observation_now += gate.FINAL_QUIESCENCE_SECONDS
            return result
        def child(_argv):
            (self.directory / "pids.current").write_text("1\n")
            return {"child_exit": 0, "child_reaped": True}
        with mock.patch.object(gate, "_final_cgroup_snapshot", side_effect=slow):
            with self.assertRaisesRegex(ValueError, "final scoped"):
                self.run_gate(self.physical_manifest(), supervisor=child)
        final = self.emitted[-1]
        self.assertIn("deadline", final["failure"])
        self.assertEqual(len(final["cgroup_observation"]["snapshots"]), 1)
        self.assertFalse(final["complete"])

    def test_second_snapshot_error_or_identity_change_refuses_without_more_waits(self):
        for mutation in ("oom", "missing", "process", "membership", "cgroup", "limits"):
            with self.subTest(mutation=mutation):
                self.prepare_cgroup("bf16")
                self.observation_waits.clear()
                def child(_argv):
                    (self.directory / "pids.current").write_text("1\n")
                    return {"child_exit": 0, "child_reaped": True}
                def change(seconds):
                    self.observation_sleep(seconds)
                    if mutation == "oom":
                        (self.directory / "memory.events").write_text("max 0\noom 1\noom_kill 0\n")
                    elif mutation == "missing":
                        (self.directory / "cgroup.threads").unlink()
                    elif mutation == "process":
                        target = self.proc / "self/stat"
                        target.write_text(target.read_text().replace("123456", "123457"))
                    elif mutation == "membership":
                        (self.proc / "self/cgroup").write_text("0::/unexpected.scope\n")
                    elif mutation == "limits":
                        (self.directory / "pids.max").write_text("511\n")
                    else:
                        self.directory.rename(self.directory.with_name(self.directory.name + "-changed"))
                        self.prepare_cgroup("bf16")
                with self.assertRaisesRegex(ValueError, "final scoped"):
                    self.run_gate(self.physical_manifest(), supervisor=child, observation_sleeper=change)
                final = self.emitted[-1]
                self.assertFalse(final["complete"])
                self.assertEqual(len(final["cgroup_observation"]["snapshots"]), 2)
                self.assertFalse(final["cgroup_observation"]["snapshots"][-1]["complete"])
                self.assertEqual(self.observation_waits, [0.2])

    def test_snapshot_overflow_retains_hash_and_refuses_within_receipt_bound(self):
        def child(_argv):
            (self.directory / "pids.current").write_text("512\n")
            (self.directory / "cgroup.procs").write_text("\n".join(str(1000000000 + i) for i in range(512)))
            (self.directory / "cgroup.threads").write_text("\n".join(str(1000000000 + i) for i in range(512)))
            return {"child_exit": 0, "child_reaped": True}
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=child)
        final = self.emitted[-1]
        self.assertFalse(final["complete"])
        snapshot = final["cgroup_observation"]["snapshots"][0]
        self.assertGreater(snapshot["observed_bytes"], gate.FINAL_SAMPLE_BYTES)
        self.assertRegex(snapshot["observed_sha256"], r"^[0-9a-f]{64}$")
        self.assertLess(len(json.dumps(final).encode()), 65536)
        self.assertEqual(self.observation_waits, [])

    def test_persistent_survivor_keeps_bounded_history_and_cannot_outwait_owner(self):
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=lambda _: {"child_exit": 0, "child_reaped": True})
        final = self.emitted[-1]
        self.assertFalse(final["complete"])
        samples = final["cgroup_observation"]["snapshots"]
        self.assertGreater(len(samples), 1)
        self.assertLessEqual(len(samples), gate.FINAL_SAMPLE_LIMIT)
        self.assertTrue(all(row["usage"]["pids.current"] == 2 for row in samples))
        self.assertLessEqual(sum(self.observation_waits), gate.FINAL_QUIESCENCE_SECONDS)
        self.assertLess(len(json.dumps(final).encode()), 65536)
        self.assertFalse(final["cgroup_observation"]["quiescence"]["complete"])

    def test_stalled_fixture_clock_is_still_limited_by_snapshot_count(self):
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=lambda _: {"child_exit": 0, "child_reaped": True},
                          observation_sleeper=lambda _: None)
        final = self.emitted[-1]
        self.assertEqual(len(final["cgroup_observation"]["snapshots"]), gate.FINAL_SAMPLE_LIMIT)
        self.assertIn("sample limit", final["failure"])
        self.assertLess(len(json.dumps(final).encode()), 65536)

    def test_malformed_membership_refuses_without_settling_retry(self):
        for content in ("0\n", "1\n1\n", "2147483648\n", "1\nnot-a-pid\n", "\n"):
            with self.subTest(content=content):
                self.prepare_cgroup("bf16")
                self.observation_waits.clear()
                def child(_argv):
                    (self.directory / "pids.current").write_text("1\n")
                    (self.directory / "cgroup.threads").write_text(content)
                    return {"child_exit": 0, "child_reaped": True}
                with self.assertRaisesRegex(ValueError, "final scoped"):
                    self.run_gate(self.physical_manifest(), supervisor=child)
                self.assertIn("membership malformed", self.emitted[-1]["failure"])
                self.assertEqual(self.observation_waits, [])

    def test_interrupted_wait_does_not_accept_or_continue_observation(self):
        initial = gate.verify_cgroup(self.manifest, "bf16", proc_root=self.proc, cgroup_root=self.cgroup)
        initial["gate_process"] = gate.process_identity(proc_root=self.proc)
        observed = {}
        stop = KeyboardInterrupt("declared owner interruption")
        def interrupted(_seconds):
            raise stop
        with self.assertRaises(KeyboardInterrupt) as caught:
            gate.final_cgroup_observation(initial, child_exit=0, child_reaped=True, proc_root=self.proc,
                cgroup_root=self.cgroup, observation=observed, clock=lambda: self.observation_now,
                sleeper=interrupted)
        self.assertIs(caught.exception, stop)
        self.assertEqual(len(observed["snapshots"]), 1)
        self.assertFalse(observed["quiescence"]["complete"])

    def test_final_missing_changed_oom_surviving_process_or_unreaped_child_refuse(self):
        changes = (("memory.events", "max 0\noom 0\noom_kill 1\n"),
                   ("pids.events", "max 1\n"), ("pids.current", "2\n"),
                   ("cgroup.procs", str(os.getpid()) + "\n999999\n"),
                   ("cgroup.threads", str(os.getpid()) + "\n999999\n"),
                   ("cgroup.events", "populated 0\nfrozen 0\n"),
                   ("memory.peak", None), ("memory.max", "12345\n"))
        for name, content in changes:
            with self.subTest(name=name):
                self.prepare_cgroup("bf16")
                self.emitted.clear()
                def child(_argv):
                    (self.directory / "pids.current").write_text("1\n")
                    if content is None:
                        (self.directory / name).unlink()
                    else:
                        (self.directory / name).write_text(content)
                    return {"child_pid": 999999, "child_exit": 0, "child_reaped": True}
                with self.assertRaisesRegex(ValueError, "final scoped"):
                    self.run_gate(self.physical_manifest(), supervisor=child)
                self.assertFalse(self.emitted[-1]["complete"])
                self.assertIn("failure", self.emitted[-1])
        self.prepare_cgroup("bf16")
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=lambda _:
                          {"child_exit": 0, "child_reaped": False})

    def test_survivor_failure_preserves_counts_membership_and_events_before_refusal(self):
        def child(_argv):
            (self.directory / "pids.current").write_text("3\n")
            (self.directory / "cgroup.procs").write_text(str(os.getpid()) + "\n999999\n")
            (self.directory / "cgroup.threads").write_text(str(os.getpid()) + "\n999999\n999998\n")
            return {"child_pid": 999999, "child_exit": 0, "child_reaped": True}
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=child)
        final = self.emitted[-1]
        self.assertFalse(final["complete"])
        self.assertFalse(final["qualification"])
        self.assertIn("survives", final["failure"])
        observation = final["cgroup_observation"]
        self.assertEqual(observation["usage"]["pids.current"], 3)
        self.assertEqual(observation["processes"], [str(os.getpid()), "999999"])
        self.assertEqual(observation["threads"], [str(os.getpid()), "999999", "999998"])
        self.assertEqual(observation["memory_events"]["oom"], 0)
        self.assertEqual(observation["pids_events"]["max"], 0)
        self.assertNotIn("only_wrapper_remains", final)

    def test_partial_observation_survives_missing_file_and_thread_overflow(self):
        for missing, oversized in (("memory.peak", False), ("cgroup.threads", False),
                                   ("cgroup.threads", True)):
            with self.subTest(file=missing, oversized=oversized):
                self.prepare_cgroup("bf16")
                def child(_argv):
                    (self.directory / "pids.current").write_text("1\n")
                    if oversized:
                        (self.directory / missing).write_bytes(b"1" * 65537)
                    else:
                        (self.directory / missing).unlink()
                    return {"child_exit": 0, "child_reaped": True}
                with self.assertRaisesRegex(ValueError, "final scoped"):
                    self.run_gate(self.physical_manifest(), supervisor=child)
                final = self.emitted[-1]
                self.assertFalse(final["complete"])
                self.assertEqual(final["cgroup_observation"]["usage"]["memory.current"], 123456)
                self.assertEqual(final["cgroup_observation"]["memory_events"]["oom"], 0)

    def test_physical_manifest_requires_artifact_tail_and_supervisor(self):
        value = self.physical_manifest()
        with self.assertRaisesRegex(ValueError, "supervisor"):
            self.run_gate(value)
        value["workload_argv"]["bf16"][-1] = "5" * 64
        value["workload_argv_sha256"]["bf16"] = gate.argv_sha256(value["workload_argv"]["bf16"])
        with self.assertRaisesRegex(ValueError, "artifact source/session"):
            gate.validate_manifest(value)

    def test_wait_wrapper_actual_function_delegates_reaps_and_forwards_only_pidfd(self):
        # Explicit CPU substitutes: no Linux pidfd or physical child is opened.
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        calls = []
        class Child:
            pid = 999999
            returncode = None
            def wait(self, timeout=None):
                if timeout is None:
                    import signal
                    os.kill(os.getpid(), signal.SIGTERM)
                self.returncode = 7
                return 7
        child = Child()
        def popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return child
        sent = []
        result = gate.supervise_workload(["declared-cpu-child"], popen=popen,
            pidfd_open=lambda pid, flags: read_fd,
            send_signal=lambda fd, number, info, flags: sent.append((fd, number)))
        self.assertEqual(calls[0][0], ["declared-cpu-child"])
        self.assertTrue(result["child_reaped"])
        self.assertEqual(result["child_exit"], 7)
        self.assertEqual(sent, [(read_fd, gate.signal.SIGTERM)])
        self.assertEqual(result["owner_signals"], [gate.signal.SIGTERM])
        with self.assertRaises(OSError):
            os.fstat(read_fd)

    def test_final_cgroup_replacement_is_not_treated_as_release(self):
        def child(_argv):
            self.directory.rename(self.directory.with_name(self.directory.name + "-old"))
            self.prepare_cgroup("bf16")
            (self.directory / "pids.current").write_text("1\n")
            return {"child_exit": 0, "child_reaped": True}
        with self.assertRaisesRegex(ValueError, "final scoped"):
            self.run_gate(self.physical_manifest(), supervisor=child)
        self.assertIn("identity changed", self.emitted[-1]["failure"])

    def test_actual_physical_bwrap_projection_passes_exact_gate_tail(self):
        adapter_spec = importlib.util.spec_from_file_location("spry_resource_projection_adapter",
            SOURCE.with_name("benchmark_owner_adapter.py"))
        adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(adapter)
        value = self.physical_manifest()
        for arm in ("bf16", "kv4"):
            # Declared CPU filesystem substitute: macOS /home is not the native
            # Linux target. The actual projection and full gate validator run.
            with mock.patch.object(adapter.Path, "resolve", autospec=True, side_effect=lambda path: path):
                argv = adapter.bwrap_argv(arm, retained_root=self.base / "retained",
                    composition=self.base / "composition", review=Path(self.root) / "review",
                    model=self.base / "model", output=Path(self.root) / "arms" / arm,
                    entry=Path(value["entry_path"]), spec_sha256=value["benchmark_spec_sha256"],
                    bounded_scratch=True, physical_artifacts=True,
                    session_sha256=value["session_sha256"], artifacts_sha256=value["artifacts_sha256"])
            self.assertNotIn("--bind", argv)
            self.assertEqual(argv.count("--tmpfs"), 4)
            value["workload_argv"][arm] = list(argv)
            value["workload_argv_sha256"][arm] = gate.argv_sha256(argv)
        self.assertEqual(gate.validate_manifest(value), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
