"""Private target-only FP8 LM head (D3b, FP32 logits); import does not initialize a device.

Install from the target runner's init_cuda_graphs, before any target graph capture. The target's
LogitsProcessor computes logits through self._compute_lm_head; this adapter binds an instance-level
replacement on that one processor. Verification rows (8 and 16: one or two requests times the DFlash
block of eight) run the approved block-128 FP8 path with FP32 output. Every other call, including
prefill rows, a different head, a bias or an ineligible tensor, runs the class's current method: the
FP32-head startup witness's wrapper during startup, the native method after it restores.
While that witness is active, each FP8 call is reported to it (metadata only; owner-approved D3b
amendment), so startup certifies the replaced branch instead of silently bypassing the witness.
The head module, its BF16 weight and the drafter's borrowed reference are never modified, so the
drafter's own candidate call keeps its BF16 dense path.
The owning process must already hold the exact GPU allowance for that serving packet.
"""

N, K = 124160, 5120
ROWS = (8, 16)
NAME = "lm_head.target"
UNQUANTIZED_HEAD_METHODS = ("UnquantizedEmbeddingMethod",)


def tensor_bytes():
    """Exact tensor payload; allocator/graph-pool overhead is measured separately."""
    fp8, scales = N * K, 4 * (N // 128) * (K // 128)
    return {
        "modules": 1, "tp_size": 2, "n": N, "k": K, "fp8_rows": list(ROWS), "output_dtype": "float32",
        "shared_bf16_target_head_bytes": 2 * N * K,
        "added_fp8_weight": fp8, "added_fp32_scales": scales,
        "added_persistent_bytes_per_rank": fp8 + scales,
        "per_call": {str(m): {"activation_fp8": m * K, "activation_scales": 4 * m * (K // 128),
                              "output_fp32": 4 * m * N} for m in ROWS},
    }


class TargetHeadFP8:
    """Replacement for one LogitsProcessor's _compute_lm_head; exact original call otherwise."""

    def __init__(self, processor, head, torch, ops, device, emit):
        self.processor, self.head = processor, head
        self.emit, self.routed = emit, set()
        self.weight = head.weight  # BF16, shared with the drafter; read, never written.
        self.weight_pointer = self.weight.data_ptr()
        try:
            self.weight_version = self.weight._version
        except RuntimeError as exc:
            raise ValueError("D3b requires a versioned, immutable target head weight") from exc
        self.torch, self.ops, self.device = torch, ops, device
        self.n, self.k, self.name = N, K, NAME
        for row in range(0, N, 128):
            if not torch.isfinite(self.weight[row:row + 128]).all().item():
                raise ValueError("nonfinite D3b BF16 head weight")
        self.b8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=self.weight.device)
        self.scales = torch.empty((N // 128, K // 128), dtype=torch.float32, device=self.weight.device)
        ops.quantize_weight_preallocated(self.weight, self.b8, self.scales, expected_device=device)

    def _bound(self):
        if (self.processor.__dict__.get("_compute_lm_head") is not self or self.head.weight is not self.weight
                or self.weight.data_ptr() != self.weight_pointer or self.weight._version != self.weight_version):
            raise RuntimeError("D3b target head binding changed; reload into a fresh model and recapture")

    def _eligible(self, hidden_states, lm_head, embedding_bias):
        x = hidden_states
        return (lm_head is self.head and embedding_bias is None and x.ndim == 2 and x.shape[0] in ROWS
                and x.shape[1] == K and x.dtype == self.torch.bfloat16 and x.device == self.weight.device
                and x.is_contiguous() and not x.requires_grad)

    def __call__(self, hidden_states, lm_head, embedding_bias=None):
        if not self._eligible(hidden_states, lm_head, embedding_bias):
            # Resolved per call: never a stale startup wrapper once the witness has restored.
            return type(self.processor)._compute_lm_head(self.processor, hidden_states, lm_head, embedding_bias)
        self._bound()
        m = hidden_states.shape[0]
        if m not in self.routed:
            # Host-side, once per row count (at capture or first eager call): proves the FP8 route ran.
            self.routed.add(m)
            self.emit("D3B_TARGET_HEAD_ROUTE", {"rows": m, "output_dtype": "float32", "capturing": bool(
                self.torch.cuda.is_current_stream_capturing())})
        # Each invocation owns its mutable tensors; during capture they use the graph allocator.
        xq = self.torch.empty((m, K), dtype=self.torch.float8_e4m3fn, device=hidden_states.device)
        xs = self.torch.empty((m, K // 128), dtype=self.torch.float32, device=hidden_states.device)
        out = self.torch.empty((m, N), dtype=self.torch.float32, device=hidden_states.device)
        self.ops.quantize_preallocated(hidden_states, xq, xs, expected_device=self.device)
        self.ops.gemm_preallocated(xq, self.b8, xs, self.scales, out, expected_device=self.device)
        report = getattr(type(self.processor).__dict__.get("_compute_lm_head"), "d3b_observe", None)
        if report is not None:
            report(self.processor, hidden_states, self.head, out)
        return out


def _prepare(model, *, torch, ops, native_types, expected_device, expected_rank, emit):
    """Validate the target model, its head, its processor and TP layout before allocating anything."""
    head_type, processor_type = native_types
    if type(expected_device) is not int or expected_device not in (0, 1):
        raise ValueError("D3b requires explicit logical device 0 or 1")
    if type(expected_rank) is not int or expected_rank not in (0, 1):
        raise ValueError("D3b requires TP rank zero or one")
    if model.training:
        raise ValueError("D3b requires the loaded eval target model")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("D3b target installation must precede graph capture")
    head, processor = getattr(model, "lm_head", None), getattr(model, "logits_processor", None)
    if type(head) is not head_type or type(processor) is not processor_type:
        raise ValueError("D3b requires the target ParallelLMHead and its LogitsProcessor")
    existing = processor.__dict__.get("_compute_lm_head")
    if existing is not None:
        # tp_worker initializes every runner in model_runner_list; a runner sharing this model
        # reuses the one FP8 binding instead of quantizing again. Any other override is refused.
        if isinstance(existing, TargetHeadFP8) and existing.head is head and existing.processor is processor:
            return {"modules": 1, "name": NAME, "tp_rank": expected_rank, "device": expected_device, "reused": True,
                    "head_id": id(head), "processor_id": id(processor), "replacement_id": id(existing), **tensor_bytes()}
        raise RuntimeError("D3b target head conflicts with an existing logits processor override")
    if not processor.use_fp32_lm_head or processor.logit_scale is not None:
        raise ValueError("D3b requires the FP32 head path without a logit scale")
    if type(head.quant_method).__name__ not in UNQUANTIZED_HEAD_METHODS or hasattr(head, "set_lora"):
        raise ValueError("D3b requires the dense unquantized, non-LoRA target head")
    weight = head.weight
    if (tuple(weight.shape) != (N, K) or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or weight.requires_grad or weight.device.type != "cuda" or weight.device.index != expected_device):
        raise ValueError("D3b loaded BF16 head shard mismatch")
    indices = head.shard_indices
    if (head.tp_size != 2 or indices.num_org_elements != N or indices.org_vocab_start_index != expected_rank * N
            or indices.num_org_elements_padded != N or head.org_vocab_size != 2 * N):
        raise ValueError("D3b TP2 vocabulary shard layout mismatch")
    original = processor._compute_lm_head
    if getattr(original, "__func__", None) is not processor_type._compute_lm_head:
        raise ValueError("D3b requires the native bound _compute_lm_head")
    ops.runtime(expected_device)
    with torch.no_grad():
        replacement = TargetHeadFP8(processor, head, torch, ops, expected_device, emit)
    # Publish only when the FP8 copy is complete. A failure leaves the native method in place and
    # must abort startup; it is never swallowed.
    processor._compute_lm_head = replacement
    return {"modules": 1, "name": NAME, "tp_rank": expected_rank, "device": expected_device, "reused": False,
            "head_id": id(head), "processor_id": id(processor), "replacement_id": id(replacement), **tensor_bytes()}


def install(model, *, expected_device, expected_rank):
    """Install only inside a separately allowed candidate serving process."""
    import json
    import sys
    import torch
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from d3b_head_launches import ServingOps

    def emit(label, value):
        # Bounded lines in the retained worker log: one install per rank, one route per row count.
        sys.stderr.write(label + " " + json.dumps(dict(value, tp_rank=expected_rank), sort_keys=True) + "\n")
        sys.stderr.flush()

    inventory = _prepare(model, torch=torch, ops=ServingOps(), expected_device=expected_device, emit=emit,
                         expected_rank=expected_rank, native_types=(ParallelLMHead, LogitsProcessor))
    emit("D3B_TARGET_HEAD_INSTALLED", {k: inventory[k] for k in (
        "name", "device", "reused", "fp8_rows", "output_dtype", "added_persistent_bytes_per_rank")})
    return inventory
