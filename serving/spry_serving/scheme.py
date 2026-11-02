"""Inactive Quark scheme composition; no registration or native execution entry.

The caller must supply the authenticated Quark class, Torch and component ops.
Only the component's exact admitted shapes are callable. This source does not
admit a full serving workload or alter SGLang's installed AITER admission.
"""

from copy import deepcopy


def _p1_route(torch):
    """P1 prefill route when the package setting selects it (P1-NUMERICAL-CONTRACT-v1); else None."""
    import mtp_contract
    setting = mtp_contract.settings().get('p1_prefill')
    if setting is None:
        return None
    import p1_prefill
    return p1_prefill.install(torch, setting)


def build_scheme_class(parent, torch, ops, weight_spec, input_spec):
    """Compose the existing serialized parameter loader with isolated kernels."""
    expected_weight = deepcopy(weight_spec)
    expected_input = deepcopy(input_spec)
    route = _p1_route(torch)

    class IsolatedPackedMXFP4(parent):
        requires_weight_loader_v2 = True

        def __init__(self, weight_quant_spec, input_quant_spec,
                     is_checkpoint_mxfp4_serialized=True,
                     dequantization_config=None):
            if (weight_quant_spec != expected_weight
                    or input_quant_spec != expected_input
                    or is_checkpoint_mxfp4_serialized is not True
                    or dequantization_config is not None):
                raise ValueError('exact serialized dense MXFP4 specifications required')
            if torch.get_default_dtype() != torch.bfloat16:
                raise ValueError('BF16 target input/output required')
            # The class factory runs before Scheduler; native ModelRunner has
            # selected this rank's device when it constructs the scheme.
            ops.bind_runtime(torch.cuda.current_device())
            # This is a separate operator, not the AITER constructor or fallback.
            self.out_dtype = torch.bfloat16
            self.qscheme = 'per_group'
            self.weight_quant_spec = deepcopy(weight_quant_spec)
            self.input_quant_spec = deepcopy(input_quant_spec)
            self.is_checkpoint_mxfp4_serialized = True
            self.dequantization_config = None

        def create_weights(self, layer, output_partition_sizes,
                           input_size_per_partition, params_dtype,
                           weight_loader, **kwargs):
            if params_dtype != torch.bfloat16:
                raise ValueError('BF16 source activation contract')
            n = sum(output_partition_sizes)
            ops.validate_geometry(1, n, input_size_per_partition)
            # Preserve the actual v2 loader, packed axis and group-scale loader.
            super().create_weights(
                layer, output_partition_sizes, input_size_per_partition,
                params_dtype, weight_loader, **kwargs)
            layer._isolated_mxfp4_loaded = False

        def process_weights_after_loading(self, layer):
            layer._isolated_mxfp4_loaded = False
            n = self.output_size_per_partition
            k = self.input_size_per_partition
            for value, shape in ((layer.weight, (n, k // 2)),
                                 (layer.weight_scale, (n, k // 32))):
                if (tuple(value.shape) != shape or value.dtype != torch.uint8
                        or not value.is_contiguous()
                        or value.device.type != 'cuda'):
                    raise ValueError('serialized packed weight/scale layout')
            if layer.weight.device != layer.weight_scale.device:
                raise ValueError('weight and scale device ownership')
            # Load-time check only; no extra reads are added inside graph replay.
            if bool(torch.any(layer.weight_scale == 255).item()):
                raise ValueError('E8M0 NaN scale is not admitted')
            if route is not None and route.admits_shape(n, k):
                # P1: every admitted shard is expanded once here and its guard flags must be zero.
                route.validate_layer(layer, n, k)
            layer._isolated_mxfp4_loaded = True

        def apply_weights(self, layer, x, bias=None):
            if (not getattr(layer, '_isolated_mxfp4_loaded', False)
                    or not isinstance(x, torch.Tensor) or x.dim() != 2
                    or bias is not None):
                raise ValueError('loaded plain rank-two input; no tuple or bias route')
            m, k = x.shape
            n = self.output_size_per_partition
            ops.validate_geometry(m, n, k)
            if (k != self.input_size_per_partition or x.dtype != torch.bfloat16
                    or not x.is_contiguous() or x.device != layer.weight.device):
                raise ValueError('input shape/dtype/device/stride contract')
            if route is not None and route.admits(m, n, k):
                # P1: prefill rows M>=1024 on the FP8 route; decode/verify rows are unchanged.
                return route.apply(layer, x, n, k)
            # Torch capture owns these allocations. No global scratch is shared
            # across layers, requests or streams; lifetime needs a native gate.
            packed = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
            scales = torch.empty_strided((m, k // 32), (1, m),
                                        dtype=torch.uint8, device=x.device)
            output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            ops.quantize_preallocated(x, packed, scales, expected_device=x.device.index)
            ops.gemm_preallocated(packed, layer.weight, scales, layer.weight_scale,
                                  output, expected_device=x.device.index)
            return output

        def pack_rows(self, layers):
            """Load time: one row-concatenated copy of loaded same-input shards.

            The separate parameters are left untouched (the P1 prefill route pins their identity);
            the copy is read only by apply_packed. Returns None unless every shard is admitted.
            """
            if not all(getattr(layer, '_isolated_mxfp4_loaded', False) for layer in layers):
                return None
            k = self.input_size_per_partition
            first = layers[0].weight
            for layer in layers:
                for value in (layer.weight, layer.weight_scale):
                    if (value.dtype != torch.uint8 or not value.is_contiguous()
                            or value.device != first.device):
                        return None
                if (layer.weight.shape[1] != k // 2 or layer.weight_scale.shape[1] != k // 32
                        or layer.weight.shape[0] != layer.weight_scale.shape[0]):
                    return None
            widths = tuple(int(layer.weight.shape[0]) for layer in layers)
            ops.validate_geometry(16, sum(widths), k)
            with torch.no_grad():
                weight = torch.cat([layer.weight for layer in layers], dim=0).contiguous()
                scale = torch.cat([layer.weight_scale for layer in layers], dim=0).contiguous()
            return weight, scale, widths, k

        def apply_packed(self, packed, x):
            """Verify rows M8/M16: quantize the shared input once, one GEMM, column views per shard.

            Each output column is computed exactly as by the separate call: same quantized input bytes,
            the same m-selected tile configuration, and per-column K accumulation independent of N.
            """
            weight, scale, widths, k = packed
            if (not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[0] not in (8, 16)
                    or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous()
                    or x.device != weight.device):
                raise ValueError('packed verify input contract')
            m, n = x.shape[0], sum(widths)
            ops.validate_geometry(m, n, k)
            packed_x = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
            scales = torch.empty_strided((m, k // 32), (1, m), dtype=torch.uint8, device=x.device)
            output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            ops.quantize_preallocated(x, packed_x, scales, expected_device=x.device.index)
            ops.gemm_preallocated(packed_x, weight, scales, scale, output, expected_device=x.device.index)
            views, offset = [], 0
            for width in widths:
                views.append(output[:, offset:offset + width])
                offset += width
            return tuple(views)

    return IsolatedPackedMXFP4
