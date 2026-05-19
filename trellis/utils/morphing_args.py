from copy import deepcopy
from typing import Iterable, Mapping, Optional


BASE_MORPHING_PARAMS = {
    "init_morphing_flag": False,
    # Sparse-structure sampling steps. This maps to the SS sampler's `steps`
    # argument unless `sparse_structure_sampler_params["steps"]` is explicitly
    # provided at the pipeline call site.
    "ss_steps": 25,
    "ss_mca_flag": True,
    "slat_mca_flag": True,
    "ss_tfsa_flag": True,
    "slat_tfsa_flag": True,
    "oc_flag": True,
}


# Legacy attention modify_attn_score(...) parameter surface.
#
# These keys are consumed by:
# - dense attention: trellis/modules/attention/full_attn.py
# - sparse attention: trellis/modules/sparse/attention/full_attn.py
#
# The actual enable switch stays in each method preset (`modify=True/False`).
# This dict only lists the parameters passed into modify_attn_score(...).
ATTENTION_MODIFY_DEFAULTS = {
    # Sparse attention did not consume modify in version 77. Keep it opt-in so
    # old CGAR parameters do not change SLAT/sparse attention behavior.
    "sparse_modify": False,
    # Penalize losing queries that crowd the same top-1 key.
    "modify_lambda_scale": 3,
    # Maximum number of conflict-reduction passes.
    "modify_max_passes": 12,
    # Stop once average top-1 key overload is at or below this value.
    "modify_stop_conflict": 0.5,
    # Kept for API completeness. The current modify_attn_score implementation
    # accepts this argument but does not use it internally.
}


ATTENTION_GATE_DEFAULTS = {
    # Sparse attention did not consume gate_attn in version 77. Keep it opt-in.
    "sparse_gate_attn": False,
    # Gate placement:
    # - "logits": gate in the native logits/probability path.
    # - "post": gate the attention output after xformers/flash/sdpa returns.
    "gate_mode": "logits",
    # Used by gate_mode="logits": suppress high-entropy, low-max-logit outputs.
    "gate_entropy_threshold": 6.0,
    "gate_max_logit_threshold": 1.0,
    # Used by gate_mode="post": q/k confidence threshold for output gating.
    "gate_qk_confidence_threshold": 1.0,
}


METHOD_PRESETS = {
    "CGAR": {
        **ATTENTION_MODIFY_DEFAULTS,
        **ATTENTION_GATE_DEFAULTS,
        "modify": True,
        "gate_attn": True,
        "sa_use": True,
        "modify_lambda_scale": 0.8,
    },
    "MCRF": {
        "ot_coherence_enabled": True,
        "ot_coherence_stage": "ss",
        "ot_anchor_patch_size": 4,
        "ot_max_anchors": 512,
        "ot_cost_pos_weight": 0.5,
        "ot_cost_feat_weight": 0.8,
        "ot_sinkhorn_eps": 0.05,
        "ot_sinkhorn_iters": 80,
        "ot_filter_k": 16,
        "ot_filter_sigma_pos": 2.0,
        "ot_filter_sigma_motion": 2.0,
        "ot_filter_lambda": 0.3,
        "ot_filter_use_confidence": True,
        "ot_filter_start_step_ratio": 1.0,
        "ot_filter_end_step_ratio": 0.0,
    },
    "CA_OC": {
        "ss_ca_oc_flag": True,
        "ss_ca_oc_grid_size": 16,
        "ss_ca_oc_desc_dim": 32,
    },
    "MAVF": {
        "mavf_enable": True,
        "mavf_lambda_feat": 1.0,
        "mavf_lambda_delta": 0.3,
        "mavf_eta_rank": 0.2,
        "mavf_sigma": 0.20,
        "mavf_eps": 0.15,
        "mavf_integration_bins": 128,
    },
    "TOPO_REPAIR": {
        "topo_repair_enable": True,
        "topo_repair_mu": 0.3,
        "topo_repair_uncert_weight": 1.0,
        "topo_repair_thin_weight": 0.5,
        "topo_repair_air_weight": 1.0,
        "topo_repair_target_weight": 1.0,
        "topo_repair_fg_weight": 0.5,
    },
}


METHOD_ALIASES = {
    "MODIFY_GATE": "CGAR",
    "OT": "MCRF",
    "TOPO": "TOPO_REPAIR",
    "TYPO_REPAIR": "TOPO_REPAIR",
}


def normalize_method_name(method: str) -> str:
    name = str(method).strip().upper().replace("-", "_")
    return METHOD_ALIASES.get(name, name)


def build_morphing_params(
    methods: Iterable[str],
    overrides: Optional[Mapping] = None,
    base: Optional[Mapping] = None,
) -> dict:
    params = deepcopy(dict(BASE_MORPHING_PARAMS if base is None else base))
    for method in methods:
        name = normalize_method_name(method)
        if name not in METHOD_PRESETS:
            raise ValueError(f"Unknown morphing method: {method}")
        params.update(deepcopy(METHOD_PRESETS[name]))
    if overrides:
        params.update(dict(overrides))
    return params
