"""Bind the selected memory cleanup after loading and before graph capture."""
import logging


def run(runner):
    import mtp_contract

    settings = mtp_contract.settings()
    if (not mtp_contract.dense_mxfp4_enabled()
            or settings.get('d3b_target_head') is None
            or settings.get('d3b_draft_head_arm') != 'both'
            or settings.get('gdn_layout_both') is not True):
        raise ValueError('post-load hook requires the selected W5 adapter set')
    if runner.startup_weight_load is not None:
        raise RuntimeError('memory cleanup requires completed synchronous weight loading')
    if not runner.is_draft_worker:
        if hasattr(runner, 'd3b_target_head_inventory'):
            raise RuntimeError('D3b target head installation is single-use per serving process')
        layers = [m for m in runner.model.modules() if type(m).__name__ == 'Qwen3_5GatedDeltaNet']
        if len(layers) != 48:
            raise RuntimeError('packed MXFP4 GDN input projection must cover all 48 GDN layers')
        for layer in layers:
            if not layer.finalize_packed_mxfp4_in_proj():
                raise RuntimeError('packed MXFP4 GDN input projection rejected a layer')
        from d3b_target_head_adapter import install as install_target
        runner.d3b_target_head_inventory = install_target(
            runner.model, expected_device=runner.gpu_id, expected_rank=runner.tp_rank)
        if settings.get('instrumentation', 'on') != 'production':
            import torch
            from d3b_head_launches import ServingOps
            from spry_fp8_head import verify_shared_fp8_head
            ops = ServingOps()
            ops.runtime(runner.gpu_id)
            proof = verify_shared_fp8_head(runner.model.lm_head, torch, ops,
                                          expected_device=runner.gpu_id)
            runner._spry_memory_recovery = dict(schema=1, policy='shared-head-packed-views-v1',
                device=runner.gpu_id, head=proof,
                packed=[dict(qkvz=dict(layer.in_proj_qkvz._spry_packed_view),
                             ba=dict(layer.in_proj_ba._spry_packed_view)) for layer in layers],
                before_graph_capture=True, predicted_released_payload_bytes=1711668800)
        # Return released blocks to the driver without changing fixed pool sizes.
        import torch
        torch.cuda.empty_cache()
        logging.getLogger(type(runner).__module__).info(
            'Packed MXFP4 GDN input projection enabled for %d layers', len(layers))
    else:
        if hasattr(runner, 'd3b_draft_head_inventory'):
            raise RuntimeError('D3b head installation is single-use per serving process')
        from d3b_head_adapter import install as install_draft
        runner.d3b_draft_head_inventory = install_draft(
            runner.model, expected_device=runner.gpu_id, expected_rank=runner.tp_rank)
