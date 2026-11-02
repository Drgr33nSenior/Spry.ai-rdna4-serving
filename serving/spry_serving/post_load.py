"""Install the selected W5 model adapters before target or draft graph capture.

The worker authenticates this source before loading Scheduler. Importing it
performs no model imports, device access or adapter installation.
"""
import logging


def run(runner):
    """Install once on the loaded runner; startup failures propagate."""
    import mtp_contract

    settings = mtp_contract.settings()
    if (not mtp_contract.dense_mxfp4_enabled()
            or settings.get('d3b_target_head') is None
            or settings.get('d3b_draft_head_arm') != 'both'
            or settings.get('gdn_layout_both') is not True):
        raise ValueError('post-load hook requires the selected W5 adapter set')
    logger = logging.getLogger(type(runner).__module__)
    if not runner.is_draft_worker:
        if hasattr(runner, 'd3b_target_head_inventory'):
            raise RuntimeError('D3b target head installation is single-use per serving process')
        if runner.startup_weight_load is not None:
            raise RuntimeError('D3b target head requires completed synchronous target weight loading')
        from d3b_target_head_adapter import install as install_d3b_target_head
        runner.d3b_target_head_inventory = install_d3b_target_head(runner.model, expected_device=runner.gpu_id, expected_rank=runner.tp_rank)
        gdn_layers = [m for m in runner.model.modules() if type(m).__name__ == 'Qwen3_5GatedDeltaNet']
        gdn_packed = [m.finalize_packed_mxfp4_in_proj() for m in gdn_layers]
        if len(gdn_layers) != 48 or not all(gdn_packed):
            raise RuntimeError('packed MXFP4 GDN input projection must cover all 48 GDN layers')
        logger.info('Packed MXFP4 GDN input projection enabled for %d layers', len(gdn_layers))
    if runner.is_draft_worker:
        if hasattr(runner, 'd3b_draft_head_inventory'):
            raise RuntimeError('D3b head installation is single-use per serving process')
        from d3b_head_adapter import install as install_d3b_draft_head
        runner.d3b_draft_head_inventory = install_d3b_draft_head(runner.model, expected_device=runner.gpu_id, expected_rank=runner.tp_rank)
