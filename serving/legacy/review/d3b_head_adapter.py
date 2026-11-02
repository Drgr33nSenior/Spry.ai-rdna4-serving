"""Private DFlash2 drafter-only FP8 head (D3b); import does not initialize a device.

Install from the draft runner's init_cuda_graphs: after the DFlash worker points the drafter's
lm_head at the target's head and before the draft graph is captured. The target's head module is
never modified. The drafter receives a separate proxy module whose quantized method serves only the
drafter's candidate call. The current target LogitsProcessor owns a separate FP8 copy for verification
and retains its existing dense fallback. Both adapters read the same immutable BF16 head weight.
The owning process must already hold the exact GPU allowance for that serving packet.
"""

N, K = 124160, 5120
# Selector rows per draft graph bucket: batch * (block_size - 1); the anchor position is dropped.
BUCKET_ROWS = {1: 7, 2: 14}
ROWS = (7, 14)
NAME = "lm_head.draft"
UNQUANTIZED_HEAD_METHODS = ("UnquantizedEmbeddingMethod",)


def emit_route(label, value):
    """One worker-log line per captured row count; no tensor reads or device synchronization."""
    import json
    import sys

    sys.stderr.write(label + " " + json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    sys.stderr.flush()


def tensor_bytes():
    """Exact tensor payload; allocator/graph-pool overhead is measured separately."""
    fp8, scales = N * K, 4 * (N // 128) * (K // 128)
    return {
        "modules": 1, "tp_size": 2, "n": N, "k": K, "fp8_rows": list(ROWS),
        "shared_bf16_target_head_bytes": 2 * N * K,
        "added_fp8_weight": fp8, "added_fp32_scales": scales,
        "added_persistent_bytes_per_rank": fp8 + scales,
        "per_call": {str(m): {"activation_fp8": m * K, "activation_scales": 4 * m * (K // 128),
                              "output_bf16": 2 * m * N} for m in ROWS},
    }


class D3bDraftHeadMethod:
    """FP8 candidate-logit projection for the drafter only; exact dense fallback otherwise."""

    def __init__(self, head, torch, ops, device, num_org, rank, emit):
        self.head = head
        self.weight = head.weight  # The target's BF16 head; read, never written.
        self.weight_pointer = self.weight.data_ptr()
        try:
            self.weight_version = self.weight._version
        except RuntimeError as exc:
            raise ValueError("D3b requires a versioned, immutable target head weight") from exc
        self.torch, self.ops, self.device, self.num_org = torch, ops, device, num_org
        self.n, self.k, self.name = N, K, NAME
        self.graph_observer, self.proxy_id = None, None
        self.rank, self.emit, self.captured_rows = rank, emit, set()
        for row in range(0, N, 128):
            if not torch.isfinite(self.weight[row:row + 128]).all().item():
                raise ValueError("nonfinite D3b BF16 head weight")
        self.b8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=self.weight.device)
        self.scales = torch.empty((N // 128, K // 128), dtype=torch.float32, device=self.weight.device)
        ops.quantize_weight_preallocated(self.weight, self.b8, self.scales, expected_device=device)

    def _bound(self, layer):
        if (id(layer) != self.proxy_id or layer.quant_method is not self
                or self.head.weight is not self.weight
                or self.weight.data_ptr() != self.weight_pointer
                or self.weight._version != self.weight_version):
            raise RuntimeError("D3b head binding changed; reload into a fresh model and recapture")

    def _eligible(self, x, bias):
        return (bias is None and x.ndim == 2 and x.shape[0] in ROWS and x.shape[1] == K
                and x.dtype == self.torch.bfloat16 and x.device == self.weight.device
                and x.is_contiguous() and not x.requires_grad)

    def apply(self, layer, x, bias=None):
        self._bound(layer)
        if not self._eligible(x, bias):
            # Exactly the drafter's dense projection (models/dflash.py _project_candidate_logits).
            output = self.torch.matmul(x.to(self.weight.dtype), self.weight[:self.num_org].T)
            if self.graph_observer is not None:
                self.graph_observer.route_completed(self, layer, x, bias, False)
            return output
        m = x.shape[0]
        # Each invocation owns its mutable tensors; during capture they use the graph allocator.
        xq = self.torch.empty((m, K), dtype=self.torch.float8_e4m3fn, device=x.device)
        xs = self.torch.empty((m, K // 128), dtype=self.torch.float32, device=x.device)
        out = self.torch.empty((m, N), dtype=self.torch.bfloat16, device=x.device)
        self.ops.quantize_preallocated(x, xq, xs, expected_device=self.device)
        self.ops.gemm_preallocated(xq, self.b8, xs, self.scales, out, expected_device=self.device)
        if m not in self.captured_rows and self.torch.cuda.is_current_stream_capturing():
            # Dispatch has returned without error. Eager warmups never consume the capture witness.
            # This proves host dispatch during capture, not completion or graph replay on the GPU.
            self.emit("D3B_DRAFT_HEAD_ROUTE", {
                "schema": 1, "name": NAME, "tp_rank": self.rank, "rows": m,
                "device": out.device.index, "device_type": out.device.type,
                "output_dtype": str(out.dtype).removeprefix("torch."), "capturing": True,
                "route": "d3b_fp8_bf16_logits",
            })
            self.captured_rows.add(m)
        if self.graph_observer is not None:
            self.graph_observer.route_completed(self, layer, x, bias, True)
        return out

    def process_weights_after_loading(self, layer):
        raise RuntimeError("D3b is installed after load; weight reload requires a fresh model")


def proxy_type(torch):
    class D3bDraftHead(torch.nn.Module):
        """Drafter-only head: shares the target's shard indices, owns an FP8 copy, no parameters."""

        def __init__(self, head, method):
            super().__init__()
            object.__setattr__(self, "_target_head", head)  # Not registered as a child module.
            self.weight = method.b8  # FP8: dense head paths refuse it; quant_method serves the call.
            self.quant_method = method
            self.shard_indices = head.shard_indices
            self.org_vocab_size = head.org_vocab_size
            self.tp_size = head.tp_size

        def forward(self, *args, **kwargs):
            raise RuntimeError("the D3b drafter head is used only through quant_method.apply")

    return D3bDraftHead


def _prepare(model, *, torch, ops, native_types, expected_device, expected_rank, emit=emit_route):
    """Validate the drafter, the attached target head and TP layout before allocating anything."""
    draft_type, head_type = native_types
    if type(expected_device) is not int or expected_device not in (0, 1):
        raise ValueError("D3b requires explicit logical device 0 or 1")
    if type(expected_rank) is not int or expected_rank not in (0, 1):
        raise ValueError("D3b requires TP rank zero or one")
    if type(model) is not draft_type or model.training:
        raise ValueError("D3b requires the loaded eval DFlash2DraftModel")
    if model.block_size != 8 or getattr(model, "candidate_selector", None) is None:
        raise ValueError("D3b requires block size eight and the candidate selector")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("D3b installation must precede graph capture")
    head = model.lm_head
    if head is None or type(head) is not head_type:
        raise ValueError("D3b requires the target ParallelLMHead attached by the DFlash worker")
    if type(head.quant_method).__name__ not in UNQUANTIZED_HEAD_METHODS:
        raise ValueError("D3b requires the dense unquantized target head")
    weight = head.weight
    if (tuple(weight.shape) != (N, K) or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or weight.requires_grad or weight.device.type != "cuda" or weight.device.index != expected_device):
        raise ValueError("D3b loaded BF16 head shard mismatch")
    indices = head.shard_indices
    if (head.tp_size != 2 or indices.num_org_elements != N or indices.org_vocab_start_index != expected_rank * N
            or indices.num_org_elements_padded != N or head.org_vocab_size != 2 * N):
        raise ValueError("D3b TP2 vocabulary shard layout mismatch")
    ops.runtime(expected_device)
    with torch.no_grad():
        method = D3bDraftHeadMethod(head, torch, ops, expected_device, indices.num_org_elements,
                                   expected_rank, emit)
    proxy = proxy_type(torch)(head, method)
    method.proxy_id = id(proxy)
    # Publish only when the FP8 copy is complete. A failure leaves the target head attached and
    # must abort startup; it is never swallowed.
    model.lm_head = proxy
    return {"modules": 1, "name": NAME, "tp_rank": expected_rank, "device": expected_device,
            "target_head_id": id(head), "proxy_id": id(proxy), "method_id": id(method), **tensor_bytes()}


def install(model, *, expected_device, expected_rank):
    """Install only inside a separately allowed candidate serving process."""
    import torch
    from sglang.srt.models.dflash import DFlash2DraftModel
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from d3b_head_launches import ServingOps

    return _prepare(model, torch=torch, ops=ServingOps(), expected_device=expected_device,
                    expected_rank=expected_rank, native_types=(DFlash2DraftModel, ParallelLMHead))
