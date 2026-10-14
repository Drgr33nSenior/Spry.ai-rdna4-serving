"""Experimental UltraQuant RDNA4 contract; standard library only.

Added locally; not upstream SGLang functionality. See the sibling experiment's
codec specification. No model/runtime installation or execution occurs here.
"""

import hashlib
import json
from types import MappingProxyType

MODE = "ultraquant-rdna4-v1"
BF16_SOURCE_MODE = "ultraquant-qwen27b-bf16-source-v2"
SOURCE_REVISION = "0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
MAX_ROWS = 4096
MAX_SLOTS = 1048576
MAX_CONTEXT = 131072
# Bounded metadata (8 MiB indices), input/outputs at max head count, and kernel
# scratch allowance. This is a planning reserve, not a measured temporary peak.
WORKSPACE_RESERVE = 2 * 1024 * 1024 * 1024

# CPU planning only. This name is deliberately absent from the backend registry.
QWEN_PLANNING_MODE = "ultraquant-qwen27b-kv4-plan-v1"
QWEN_MODEL = "Qwen/Qwen3.8-27B-FP8"
QWEN_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
QWEN_CONFIG_SHA256 = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
QWEN_FULL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
QWEN_PROTECTED_LAYERS = (3, 7, 59, 63)
QWEN_STORAGE = "local-g32-e2m1-ue8m0-interleaved17-v1"
QWEN_ROTATION = "post-mrope-full256-qk-fp32-wht"


def validate_args(args):
    """Refuse conflicting requested settings before upstream auto adjustment."""
    names = ("attention_backend", "prefill_attention_backend", "decode_attention_backend")
    if getattr(args, "_ultraquant_bf16_requested", False) or BF16_SOURCE_MODE in (
        getattr(args, name, None) for name in names
    ):
        raise ValueError("BF16 source candidate is not an admitted executable backend")
    if getattr(args, "_ultraquant_qwen_planning_requested", False) or QWEN_PLANNING_MODE in (
        getattr(args, name, None) for name in names
    ):
        raise ValueError("Qwen KV4 planning mode cannot authorize native launch")
    if getattr(args, "_ultraquant_requested", False) and args.attention_backend != MODE:
        raise ValueError("requested UltraQuant backend was changed during argument resolution")
    if MODE not in [getattr(args, name, None) for name in names]:
        return
    if args.attention_backend != MODE or any(
        getattr(args, name, None) not in (None, MODE) for name in names[1:]
    ):
        raise ValueError("UltraQuant requires the same explicit prefill/decode backend")
    required = {
        "tp_size": 1, "pp_size": 1, "dp_size": 1, "nnodes": 1,
        "attn_cp_size": 1, "dcp_size": 1, "page_size": 1, "dtype": "float16",
        "kv_cache_dtype": "auto", "disable_overlap_schedule": True,
        "cuda_graph_backend_decode": "disabled", "cuda_graph_backend_prefill": "disabled",
        "disaggregation_mode": "null", "cpu_offload_gb": 0,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"UltraQuant requires {name}={expected}")
    refused = (
        "enable_torch_compile", "enable_mixed_chunk", "enable_hierarchical_cache",
        "enable_lmcache", "enable_memory_saver", "enable_pdmux", "enable_two_batch_overlap",
        "prefill_only_disable_kv_cache", "enable_dp_attention", "enable_dp_lm_head",
        "enable_page_major_kv_layout", "enable_unified_memory",
        "enable_hisparse", "enable_deterministic_inference", "enable_return_hidden_states",
        "speculative_algorithm", "quantization", "torchao_config", "modelopt_quant",
        "quantize_and_serve", "lora_paths", "enable_lora",
    )
    for name in refused:
        if getattr(args, name, None):
            raise ValueError(f"UltraQuant does not support {name}")
    if getattr(args, "cuda_graph_config", None):
        # The final resolved object is checked separately by the backend.
        if isinstance(args.cuda_graph_config, (dict, str)):
            raise ValueError("UltraQuant requires the two explicit disabled graph flags")
    for name, limit in (("max_total_tokens", MAX_SLOTS - 1), ("context_length", MAX_CONTEXT),
                        ("chunked_prefill_size", MAX_ROWS), ("max_running_requests", 128)):
        value = getattr(args, name, None)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"UltraQuant requires an explicit bounded {name}")


def validate_model(model):
    # An intentional architecture allowlist. The workstation Qwen3.5/3.8 models
    # are hybrid recurrent models and cannot enter this conventional KV path.
    if getattr(model.hf_config, "architectures", None) != ["LlamaForCausalLM"]:
        raise ValueError("UltraQuant v1 supports only conventional LlamaForCausalLM")
    if getattr(model, "is_encoder_decoder", False) or getattr(model.hf_config, "sliding_window", None):
        raise ValueError("UltraQuant does not support cross/sliding-window attention")
    if getattr(model.hf_config, "quantization_config", None):
        raise ValueError("UltraQuant v1 requires unquantized model weights")
    dim, qheads, kvheads = model.head_dim, model.num_attention_heads, model.get_num_kv_heads(1)
    if dim not in (32, 64, 128, 256) or model.v_head_dim != dim:
        raise ValueError("UltraQuant requires equal supported QK/V dimensions")
    if not 1 <= kvheads <= 64 or not kvheads <= qheads <= 128 or qheads % kvheads:
        raise ValueError("UltraQuant requires integral bounded grouped query attention")


def cell_bytes(layers, heads, dim):
    if type(layers) is not int or not 1 <= layers <= 256 or type(heads) is not int or not 1 <= heads <= 64:
        raise ValueError("invalid UltraQuant layer/head count")
    if type(dim) is not int or dim not in (32, 64, 128, 256):
        raise ValueError("unsupported UltraQuant head dimension")
    return layers * 2 * heads * (dim // 32) * 17


def capacity(available_bytes, per_token_bytes):
    # Include slot zero and the fixed reserve in addition to actual packed bytes.
    return max(0, min(MAX_SLOTS - 1, (available_bytes - WORKSPACE_RESERVE) // per_token_bytes - 1))


def causal_rows(sequences, prefixes, extends, locations, slots):
    """Build reader spans over packed physical slot indices, without cache data.

    sequences contains the scheduler's logical-to-physical map for each request.
    The return starts/lengths are one span per Q row. Visible lengths are causal,
    so initial and continuation prefill never attend future appended tokens.
    """
    if not (len(sequences) == len(prefixes) == len(extends)) or not sequences:
        raise ValueError("inconsistent request metadata")
    if type(slots) is not int or not 1 < slots <= MAX_SLOTS:
        raise ValueError("invalid cache capacity")
    indices, starts, lengths, writes = [], [], [], []
    for sequence, prefix, extend in zip(sequences, prefixes, extends):
        if type(prefix) is not int or type(extend) is not int or prefix < 0 or extend <= 0:
            raise ValueError("invalid prefix/extend lengths")
        if prefix + extend != len(sequence) or len(sequence) > MAX_CONTEXT:
            raise ValueError("sequence must equal prefix plus new tokens")
        if any(type(slot) is not int or not 0 < slot < slots for slot in sequence):
            raise ValueError("invalid physical cache slot")
        if len(set(sequence)) != len(sequence):
            raise ValueError("a request cannot alias its own token slots")
        starts.extend([len(indices)] * extend)
        lengths.extend(range(prefix + 1, prefix + extend + 1))
        indices.extend(sequence)
        writes.extend(sequence[prefix:])
    if len(starts) > MAX_ROWS or len(indices) > MAX_SLOTS:
        raise ValueError("metadata exceeds the experimental bound")
    if writes != locations or len(set(writes)) != len(writes):
        raise ValueError("writer locations must match new logical slots exactly")
    # A new write may not overwrite another request's reused prefix.
    reused = {slot for sequence, prefix in zip(sequences, prefixes) for slot in sequence[:prefix]}
    if reused.intersection(writes):
        raise ValueError("append aliases a reused prefix")
    return indices, starts, lengths


def qwen_planning_metadata(config_bytes, *, model, revision, rank, source_tree_sha256,
                           engine="sglang", source_revision=SOURCE_REVISION,
                           tp_size=2, input_dtype="bfloat16", output_dtype="bfloat16",
                           storage_format=QWEN_STORAGE, rotation_policy=QWEN_ROTATION,
                           page_size=1, graphs=False, overlap=False, mtp=False):
    """Validate the fixed FP8 model and return CPU planning metadata only.

    The BF16/FP32 compute policy is a future proposal, not an implemented kernel.
    Config bytes bind the entire pinned configuration, including FP8 weight scales.
    """
    if model != QWEN_MODEL or revision != QWEN_REVISION or source_revision != SOURCE_REVISION:
        raise ValueError("unknown Qwen model or preparation source identity")
    if engine != "sglang" or type(source_tree_sha256) is not str or len(source_tree_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_tree_sha256
    ):
        raise ValueError("Qwen requires the SGLang source-tree digest")
    if type(config_bytes) is not bytes or len(config_bytes) > 1024 * 1024:
        raise ValueError("Qwen requires bounded pinned config bytes")
    if hashlib.sha256(config_bytes).hexdigest() != QWEN_CONFIG_SHA256:
        raise ValueError("Qwen config checksum mismatch")
    if type(tp_size) is not int or tp_size != 2 or type(rank) is not int or rank not in (0, 1):
        raise ValueError("Qwen planning requires TP2 rank 0 or 1")
    if type(page_size) is not int or page_size != 1:
        raise ValueError("Qwen planning requires page size one")
    if any(type(value) is not bool or value for value in (graphs, overlap, mtp)):
        raise ValueError("Qwen planning does not support graphs, overlap or MTP")
    if (input_dtype, output_dtype, storage_format, rotation_policy) != (
        "bfloat16", "bfloat16", QWEN_STORAGE, QWEN_ROTATION
    ):
        raise ValueError("Qwen planning precision or rotation policy mismatch")
    config = json.loads(config_bytes)
    text = config["text_config"]
    layer_types = tuple("full_attention" if i in QWEN_FULL_LAYERS else "linear_attention"
                        for i in range(64))
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or (
        text.get("num_hidden_layers"), text.get("head_dim"), text.get("num_attention_heads"),
        text.get("num_key_value_heads"), text.get("dtype"), tuple(text.get("layer_types", ()))
    ) != (64, 256, 24, 4, "bfloat16", layer_types):
        raise ValueError("Qwen hybrid layer map or geometry mismatch")
    quant = config["quantization_config"]
    if (quant.get("quant_method"), quant.get("fmt"), quant.get("activation_scheme"),
        quant.get("weight_block_size")) != ("fp8", "e4m3", "dynamic", [128, 128]):
        raise ValueError("Qwen requires the official FP8 weight contract")
    rope = text["rope_parameters"]
    if (rope.get("partial_rotary_factor"), rope.get("mrope_interleaved"),
        rope.get("mrope_section")) != (0.25, True, [11, 11, 10]):
        raise ValueError("Qwen partial MRoPE configuration mismatch")
    prefix_identity = (
        QWEN_MODEL, QWEN_REVISION, QWEN_CONFIG_SHA256, engine, SOURCE_REVISION, source_tree_sha256,
        QWEN_PLANNING_MODE, QWEN_STORAGE, input_dtype, output_dtype, "fp32-wht-and-accumulation",
        "bf16-tile-unpack", QWEN_ROTATION, "v-unrotated", QWEN_PROTECTED_LAYERS, "protected-bf16-unrotated",
        tp_size, rank, 12, 2, page_size,
    )
    return MappingProxyType({
        "mode": QWEN_PLANNING_MODE, "native_launch_allowed": False, "qualified": False,
        "model": QWEN_MODEL, "revision": QWEN_REVISION, "config_sha256": QWEN_CONFIG_SHA256,
        "preparation_source_revision": SOURCE_REVISION, "tp_size": 2, "rank": rank,
        "source_tree_sha256": source_tree_sha256, "weight_precision": "fp8-e4m3-block128x128",
        "global_query_heads": 24, "global_kv_heads": 4, "query_heads": 12, "kv_heads": 2,
        "head_dim": 256, "full_layer_ids": QWEN_FULL_LAYERS, "gdn_layers": 48,
        "protected_layer_ids": QWEN_PROTECTED_LAYERS, "protected_compact_ids": (0, 1, 14, 15),
        "protection_status": "hybrid-proposal-unqualified", "input_dtype": input_dtype,
        "output_dtype": output_dtype, "storage_format": storage_format,
        "rotation_policy": rotation_policy, "v_rotation": "none", "page_size": 1,
        "prefix_identity": prefix_identity,
        "payload_bytes_per_token_per_rank": 12 * 2 * 272 + 4 * 2 * 2 * 256 * 2,
        "gdn_bytes_per_state_slot_per_rank_at_retained_defaults": 48 * (5120 * 3 * 2 + 24 * 128 * 128 * 4),
    })


def validate_qwen_prefix_identity(metadata, observed_identity):
    """Reject another model, policy, engine/source, partition or stale prefix."""
    if type(observed_identity) is not tuple or observed_identity != metadata["prefix_identity"]:
        raise ValueError("Qwen prefix identity mismatch")


def qwen_layer_route(*, physical_id=None, compact_id=None):
    """Resolve fixed full-attention IDs; protected layers have no WHT."""
    if physical_id is None and compact_id is None:
        raise ValueError("Qwen layer identity is required")
    if physical_id is not None and (type(physical_id) is not int or physical_id not in QWEN_FULL_LAYERS):
        raise ValueError("Qwen physical layer is not a full-attention layer")
    if compact_id is not None and (type(compact_id) is not int or not 0 <= compact_id < 16):
        raise ValueError("Qwen compact full-attention layer is invalid")
    if physical_id is None:
        physical_id = QWEN_FULL_LAYERS[compact_id]
    expected = QWEN_FULL_LAYERS.index(physical_id)
    if compact_id is not None and compact_id != expected:
        raise ValueError("conflicting Qwen physical and compact layer IDs")
    protected = physical_id in QWEN_PROTECTED_LAYERS
    return {"physical_id": physical_id, "compact_id": expected, "protected": protected,
            "storage": "bfloat16" if protected else QWEN_STORAGE,
            "rotation": "none" if protected else QWEN_ROTATION}


def qwen_writer_metadata(layer, loc, cache_k, cache_v, k_scale=1.0, v_scale=1.0, *,
                         layer_id_override=None, dcp_kv_mask=None):
    """Specify the hybrid full-pool call; no cache allocation or mutation.

    The historical hybrid caller passes layer=None. Pinned 0.5.21 passes the
    physical layer object and the same compact layer_id_override.
    Activation tensors and write locations remain opaque at this metadata boundary.
    Slot ownership is checked by causal_rows before a future writer can mutate data.
    """
    for scale in (k_scale, v_scale):
        if scale is not None and (type(scale) not in (int, float) or scale != 1):
            raise ValueError("Qwen writer supports only absent or unit scale arguments")
    if dcp_kv_mask is not None:
        raise ValueError("Qwen writer does not support dcp_kv_mask")
    if layer is None and layer_id_override is None:
        raise ValueError("hybrid Qwen writer requires layer_id_override")
    physical_id = None if layer is None else getattr(layer, "layer_id", None)
    if layer is not None and type(physical_id) is not int:
        raise ValueError("Qwen writer requires an integer physical layer_id")
    return qwen_layer_route(physical_id=physical_id,
                            compact_id=layer_id_override)


def qwen_payload_bytes(physical_token_slots):
    """Payload arithmetic only; slot zero is separate and page-one padding is zero."""
    if type(physical_token_slots) is not int or not 0 <= physical_token_slots < MAX_SLOTS:
        raise ValueError("invalid Qwen physical token-slot count")
    per_token = 12 * 2 * 272 + 4 * 2 * 2 * 256 * 2
    return {"token_payload": physical_token_slots * per_token, "reserved_slot_zero": per_token,
            "page_padding": 0, "total_payload_with_slot_zero": (physical_token_slots + 1) * per_token}

def bf16_write_slots(locations, slots, immutable=()):
    """Check the complete physical write set before any native mutation."""
    if type(slots) is not int or not 2 <= slots <= MAX_SLOTS:
        raise ValueError("invalid physical slot count")
    if not 1 <= len(locations) <= MAX_ROWS or any(
        type(x) is not int or not 0 < x < slots for x in locations
    ):
        raise ValueError("invalid physical write locations")
    if len(set(locations)) != len(locations) or set(locations).intersection(immutable):
        raise ValueError("duplicate write or immutable shared prefix")
    return tuple(locations)


def bf16_pool_policy(*, rank, physical_slots, page_size=1, graphs=False, overlap=False,
                     mtp=False, dcp=False, offload=False, cache_copy=False, unified=False):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("BF16 candidate requires TP2 rank zero or one")
    if type(physical_slots) is not int or not 1 <= physical_slots < MAX_SLOTS:
        raise ValueError("fixed physical slot count required")
    if type(page_size) is not int or page_size != 1 or any(
        type(x) is not bool or x for x in (graphs, overlap, mtp, dcp, offload, cache_copy, unified)
    ):
        raise ValueError("unsupported candidate page/scheduler/pool policy")
    return MappingProxyType({"rank": rank, "tp": 2, "q_heads": 12, "kv_heads": 2,
        "head_dim": 256, "slots": physical_slots + 1, "page_size": 1,
        "protected": (0, 1, 14, 15), "storage": QWEN_STORAGE,
        "rotation": QWEN_ROTATION, "compute": "bf16-in-out-unpack-fp32-accumulate",
        "payload": qwen_payload_bytes(physical_slots)["total_payload_with_slot_zero"],
        "workspace_reserve": WORKSPACE_RESERVE})


def bf16_source_request(config_bytes, *, rank, physical_slots, context_length):
    """Explicit internal source request; no public backend admission."""
    if type(config_bytes) is not bytes or hashlib.sha256(config_bytes).hexdigest() != QWEN_CONFIG_SHA256:
        raise ValueError("fixed official FP8 Qwen config required")
    if type(context_length) is not int or not 1 <= context_length <= MAX_CONTEXT:
        raise ValueError("unsupported BF16 context bound")
    return MappingProxyType({"config_bytes": config_bytes,
        "policy": bf16_pool_policy(rank=rank, physical_slots=physical_slots),
        "context_length": context_length})


def validate_bf16_source_args(args):
    # Do not change the v1 validator or register this mode. An explicit public
    # future-mode request still fails before runtime import.
    validate_args(args)
    expected = {"tp_size": 2, "pp_size": 1, "dp_size": 1, "nnodes": 1,
        "attn_cp_size": 1, "dcp_size": 1, "page_size": 1, "dtype": "bfloat16",
        "kv_cache_dtype": "auto", "disable_overlap_schedule": True,
        "cuda_graph_backend_decode": "disabled", "cuda_graph_backend_prefill": "disabled",
        "disaggregation_mode": "null", "cpu_offload_gb": 0}
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"BF16 source path requires {name}={value}")
    for name in ("enable_unified_memory", "enable_memory_saver", "enable_page_major_kv_layout",
                 "enable_hierarchical_cache", "enable_lmcache", "enable_two_batch_overlap",
                 "enable_pdmux", "enable_torch_compile", "speculative_algorithm"):
        if getattr(args, name, None):
            raise ValueError(f"BF16 source path does not support {name}")
    graphs = getattr(args, "cuda_graph_config", None)
    if graphs is not None and any(getattr(getattr(graphs, phase, None), "backend", None) != "disabled"
                                  for phase in ("decode", "prefill")):
        raise ValueError("BF16 source path requires resolved disabled graphs")
