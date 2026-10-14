"""Owned correctness-only TP2 entry. No runtime imports before explicit execution.

The reviewed outer launcher must pin this file, the spec, installed runtime,
model files, physical device mapping and package build/common-stack receipts.
This entry does not mint those receipts or qualify serving performance.
"""
import argparse
from array import array
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import stat
import time


def checked_file(path, expected, maximum=8 << 20):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute reviewed input required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError("input type/size")
        data = b""
        while len(data) <= maximum:
            block = os.read(fd, min(1048576, maximum + 1 - len(data)))
            if not block:
                break
            data += block
        after = os.fstat(fd)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stamp(before) != stamp(after) or stamp(after) != stamp(os.stat(path, follow_symlinks=False)):
            raise ValueError("input changed")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("input digest")
        return data
    finally:
        os.close(fd)


def write_result(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(raw) > 1 << 20:
        raise ValueError("result bound")
    with open(path, "xb") as stream:
        stream.write(raw + b"\n")


def preflight(spec):
    if spec["scope"] != "reviewed-private-tp2-model-smoke-only":
        raise ValueError("not reviewed smoke scope")
    if spec["gpu_ids"] != [0, 1] or not 60 <= spec["worker_seconds"] <= 1800:
        raise ValueError("fixed two ranks and finite worker deadline required")
    if spec["context_length"] != 4096 or not 128 <= spec["physical_slots"] <= 4096:
        raise ValueError("smoke context/slot bound")
    if not 8 <= len(spec["prompt_token_ids"]) <= 64 or any(type(x) is not int or x < 0 for x in spec["prompt_token_ids"]):
        raise ValueError("fixed bounded token fixture required")
    if not 1 <= spec["cut_len"] < len(spec["prompt_token_ids"]):
        raise ValueError("continuation split")
    config = checked_file(spec["config_path"], "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc")
    checked_file(spec["stack_receipt"], spec["stack_receipt_sha256"])
    package = Path(spec["package_root"])
    proof_path = package / "srt/layers/attention/ultraquant-bf16-build.json"
    proof = json.loads(checked_file(proof_path, spec["build_proof_sha256"]))
    if (proof.get("schema") != 2 or proof.get("serving_composition_complete") is not True
            or proof.get("qualification") != "build-reviewed-not-serving-qualified"
            or proof.get("stack_receipt_sha256") != spec["stack_receipt_sha256"]
            or proof.get("binary_sha256") != "d09bb0649fdbfabfff1ee865ebd2b4f2923a85ec4242af56c683668df85a9171"):
        raise ValueError("closed serving composition/build gate")
    return config


def run_rank(spec, rank, output):
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # RCCL creates 41,288,068-byte shared-memory files during TP initialization.
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (spec["worker_seconds"], spec["worker_seconds"]))
    # HIP may reserve large virtual address ranges; no invented AS/RSS bound.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    log = os.open(output / f"rank-{rank}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    result = {"rank": rank, "stages": [], "failure": None, "cleanup_errors": [],
              "owner_cleanup": False, "serving_qualified": False, "benchmark": False}
    cache = runner = None
    requests = []
    distributed = False
    started = time.monotonic()
    try:
        config = preflight(spec)
        import sys
        sys.path.insert(0, str(Path(spec["package_root"]).parent))
        import torch
        import torch.distributed as dist
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import bootstrap
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel, get_tp_group,
        )
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.common import release_kv_cache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import SpawnRanks, get_device, get_model, get_parallel, publish, spawn_world_rank
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.layers.attention.ultraquant_contract import bf16_source_request
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.kernels.ops.mamba.triton_ops import initialize_mamba_selective_state_update_backend

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        args = ServerArgs(**spec["server_args"])
        args.resolve_once()
        _set_envs_and_config(args)
        publish(args, role="scheduler", ranks=SpawnRanks(
            world_rank=spawn_world_rank(args, tp_rank=rank, pp_rank=0), gpu_id=spec["gpu_ids"][rank]))
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        model_config = ModelConfig.from_server_args(args)
        bootstrap.init_parallel_runtime(server_args=args, device=get_device().device, dist_port=spec["nccl_port"])
        distributed = True
        bootstrap.init_layer_runtime(model_config=model_config)
        if get_model().is_startup_weight_load_overlap:
            raise ValueError("smoke requires synchronous weight loading")
        request = dict(bf16_source_request(config, rank=rank,
            physical_slots=spec["physical_slots"], context_length=spec["context_length"]))
        request.update(stack_receipt=spec["stack_receipt"], stack_receipt_sha256=spec["stack_receipt_sha256"])
        runner = ModelRunner(model_config=model_config, mem_fraction_static=args.mem_fraction_static,
            gpu_id=spec["gpu_ids"][rank], nccl_port=spec["nccl_port"], server_args=args,
            _uq_bf16_request=request)
        runner.alloc_memory_pool()
        runner.init_attention_backends()
        # Bounded caller metadata only; no Q/K/V or weight tensors are logged.
        scale_records = []
        for module in runner.model.modules():
            if hasattr(module, "k_scale") and hasattr(module, "v_scale"):
                fields = {"layer_id": getattr(module, "layer_id", None)}
                for field in ("k_scale", "v_scale", "k_scale_float", "v_scale_float"):
                    value = getattr(module, field, None)
                    if isinstance(value, torch.Tensor):
                        fields[field] = {"type": type(value).__name__, "shape": list(value.shape),
                            "dtype": str(value.dtype), "device": str(value.device),
                            "scalar": value.item() if value.numel() == 1 else "non-scalar"}
                    else:
                        fields[field] = value
                scale_records.append(fields)
                if len(scale_records) > 16:
                    raise ValueError("unexpected full-attention scale metadata count")
        print(json.dumps({"stage": "attention-scale-metadata", "rank": rank,
                          "layers": scale_records}, allow_nan=False), flush=True)
        initialize_mamba_selective_state_update_backend(args)
        runner.init_cuda_graphs()
        cache = UnifiedRadixCache(CacheInitParams(disable=False,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, page_size=1,
            tp_cache_group=get_tp_group().cpu_group,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            tree_core_backend="python", enable_mamba_extra_buffer=False))
        owner = runner.token_to_kv_pool.full_kv_pool.owner
        def capacity():
            return [runner.token_to_kv_pool_allocator.available_size(),
                    len(runner.req_to_token_pool.free_slots),
                    runner.req_to_token_pool.mamba_allocator.available_size()]
        before = capacity()
        tokens = spec["prompt_token_ids"]
        if any(x >= model_config.vocab_size for x in tokens):
            raise ValueError("prompt token outside vocabulary")

        def new_request(name, ids):
            req = Req(rid=name, origin_input_text="owned fixed token smoke",
                origin_input_ids=array("q", ids), sampling_params=SamplingParams(temperature=0, max_new_tokens=3))
            requests.append(req)
            req.logprob_start_len = -1
            req.init_next_round_input(cache)
            req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
            req.set_extend_range(len(req.prefix_indices), len(req.full_untruncated_fill_ids))
            return req

        def extend(req):
            batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=runner.req_to_token_pool,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, tree_cache=cache,
                model_config=model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
            batch.prepare_for_extend()
            if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
                batch.prefill_input_ids_cpu = None
            return batch

        def forward(label, batch):
            print(json.dumps({"stage": label, "rank": rank}), flush=True)
            with torch.no_grad():
                fb = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
                logits = runner.forward(fb).logits_output
                values = logits.next_token_logits
                if tuple(values.shape) != (1, model_config.vocab_size) or not torch.isfinite(values).all().item():
                    raise ValueError("nonfinite or wrong-shaped logits")
                sampled = runner.sample(logits, fb)
                token = int(sampled.item())
                if not 0 <= token < model_config.vocab_size:
                    raise ValueError("sample outside vocabulary")
                peers = [None, None]
                dist.all_gather_object(peers, token, group=get_tp_group().cpu_group)
                if peers != [token, token]:
                    raise ValueError("TP output disagreement")
                result["stages"].append({"stage": label, "token": token,
                    "logits_shape": list(values.shape), "finite": True,
                    "logits_min": float(values.min().item()), "logits_max": float(values.max().item())})
                return sampled, token

        req = new_request("smoke-primary", tokens[:spec["cut_len"]])
        batch = extend(req)
        forward("initial-prefill", batch)
        cache.cache_unfinished_req(req)
        tail = tokens[spec["cut_len"]:]
        req.origin_input_ids.extend(tail)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(len(req.prefix_indices), len(tokens))
        batch = extend(req)
        sampled, token = forward("continuation", batch)
        req.output_ids.append(token)
        batch.input_ids = sampled.to(torch.int64)
        batch.prepare_for_decode()
        _, token = forward("decode", batch)
        req.output_ids.append(token)
        reuse_tokens = list(tokens) + [req.output_ids[0], tokens[-1]]
        release_kv_cache(req, cache, is_insert=True)
        reused = new_request("smoke-prefix-reuse", reuse_tokens)
        hit = len(reused.prefix_indices)
        if hit <= 0:
            raise ValueError("expected real prefix reuse")
        result["prefix_hit_tokens"] = hit
        batch = extend(reused)
        forward("prefix-reuse", batch)
        release_kv_cache(reused, cache, is_insert=False)
        cache.release_for_idle_bf16_flush()
        cache.reset()
        owner.refuse_reset()
        if capacity() != before:
            raise ValueError("pool ownership/capacity did not return to baseline")
        result.update(owner_cleanup=True, capacity_before=before, capacity_after=capacity())
        torch.cuda.synchronize()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        if cache is not None:
            for req in requests:
                try:
                    if req.kv.holds_kv or req.kv.holds_mamba:
                        release_kv_cache(req, cache, is_insert=False)
                except BaseException as exc:
                    result["cleanup_errors"].append(type(exc).__name__ + ": " + str(exc)[:512])
        if distributed:
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except BaseException as exc:
                result["cleanup_errors"].append(type(exc).__name__ + ": " + str(exc)[:512])
        result["elapsed_seconds"] = time.monotonic() - started
        write_result(output / f"rank-{rank}.json", result)
    raise SystemExit(0 if result["failure"] is None and not result["cleanup_errors"] and result["owner_cleanup"] else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-reviewed-smoke", action="store_true")
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_reviewed_smoke:
        raise SystemExit("NOT_RUN: explicit reviewed runtime/spec required")
    spec = json.loads(checked_file(args.spec, args.spec_sha256))
    preflight(spec)
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("fresh absolute owned output required")
    output.mkdir(mode=0o700)
    ctx = multiprocessing.get_context("spawn")
    children = []
    deadline = time.monotonic() + spec["worker_seconds"]
    timed_out = False
    parent_error = None
    cleanup_errors = []
    def terminate_parent(signum, frame):
        raise RuntimeError("outer supervisor requested termination")
    prior_term = signal.signal(signal.SIGTERM, terminate_parent)
    try:
        for rank in (0, 1):
            child = ctx.Process(target=run_rank, args=(spec, rank, output))
            child.start()
            children.append(child)
        write_result(output / "owned-workers.json", {"pids": [p.pid for p in children], "worker_seconds": spec["worker_seconds"]})
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
        timed_out = any(child.is_alive() for child in children)
    except BaseException as exc:
        parent_error = type(exc).__name__ + ": " + str(exc)[:2048]
    finally:
        # Repeated TERM must not interrupt closure of the separately owned sessions.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in children:
            if child.is_alive():
                # Signal only the still-owned direct child's own session/group.
                for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                    if not child.is_alive():
                        break
                    try:
                        if os.getpgid(child.pid) != child.pid:
                            # pre-setsid startup: signal the owned direct child only
                            child.terminate() if sig == signal.SIGTERM else child.kill()
                        else:
                            os.killpg(child.pid, sig)
                        child.join(wait)
                    except ProcessLookupError:
                        child.join(wait)
                    except OSError as exc:
                        cleanup_errors.append(type(exc).__name__ + ": " + str(exc)[:512])
        signal.signal(signal.SIGTERM, prior_term)
    records = [json.loads((output / f"rank-{rank}.json").read_bytes())
               if (output / f"rank-{rank}.json").is_file() else None for rank in (0, 1)]
    groups_absent = []
    for child in children:
        try:
            os.killpg(child.pid, 0)
            groups_absent.append(False)
        except ProcessLookupError:
            groups_absent.append(True)
        except OSError:
            groups_absent.append(False)
    passed = parent_error is None and not cleanup_errors and all(groups_absent) and not timed_out and len(children) == 2 and all(p.exitcode == 0 for p in children) and all(
        r is not None and r["failure"] is None and r["owner_cleanup"] and not r["cleanup_errors"] for r in records)
    write_result(output / "result.json", {"status": "MODEL_SMOKE_PASS" if passed else "MODEL_SMOKE_FAILED",
        "timed_out": timed_out, "exits": [p.exitcode for p in children], "ranks": records,
        "parent_error": parent_error, "cleanup_errors": cleanup_errors, "groups_observed_absent": groups_absent,
        "serving_quality_qualified": False, "benchmark": False, "independent_release_required": True})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
