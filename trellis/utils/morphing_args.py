from copy import deepcopy
from typing import Iterable, Mapping, Optional


BASE_MORPHING_PARAMS = {
    # 是否走初始化形变路径；一般保持 False，正常跑完整 morphing sampler。
    "init_morphing_flag": False,
    # 稀疏结构阶段采样步数；如果调用 pipeline 时显式传入
    # sparse_structure_sampler_params["steps"]，会优先使用调用处的值。
    "ss_steps": 25,
    # 形变主开关：
    # - ss_mca_flag：稀疏结构阶段融合 source/target 的 cross-attention 特征。
    # - slat_mca_flag：SLAT 阶段融合 source/target 特征。
    # - ss_tfsa_flag / slat_tfsa_flag：复用上一帧 self-attention 的 K/V cache，
    #   用于提升时序连续性。
    "ss_mca_flag": True,
    "slat_mca_flag": True,
    "ss_tfsa_flag": True,
    "slat_tfsa_flag": True,
    # cache 模式选择：
    # - "memory"：一次运行内把 attention/coords cache 放在内存里，速度更快，
    #   也避免生成大量 .pt 文件。
    # - "disk"：兼容旧的文件缓存逻辑。
    # save_coords_cache=False 时不会保存 coords_morphing*.pt。
    "tfsa_cache_mode": "memory",
    "coords_cache_mode": "memory",
    "save_coords_cache": False,
    # 同一组 source/target 图片跨帧重复使用时，缓存图像条件特征，避免重复编码。
    "cache_image_cond": True,
    # CA-OC 的粗方向校正开关；"CA_OC" 方法 preset 主要打开 ss_ca_oc_flag，
    # oc_flag 控制 coords 级别的方向比较与旋转修正。
    "oc_flag": True,
    # SLAT 阶段采样步数，以及 source/target 特征融合模式。
    # slat_fuse_mode 在 sample_slat_morphing 中使用。
    "slat_steps": 25,
    "slat_fuse_mode": "linear",
}


# CGAR / modify_attn_score(...) 相关参数。
#
# 使用位置：
# - dense attention: trellis/modules/attention/full_attn.py
# - sparse attention: trellis/modules/sparse/attention/full_attn.py
#
# 真正是否启用由各个方法 preset 中的 modify=True/False 控制。
# 这里主要列 modify_attn_score(...) 需要的超参。
ATTENTION_MODIFY_DEFAULTS = {
    # 是否在 sparse attention 中也启用 modify；默认 False，避免 CGAR 影响 SLAT/sparse
    # attention 的旧行为。
    "sparse_modify": False,
    # 对扎堆到同一个 top-1 key 的 loser query 施加惩罚的强度。
    "modify_lambda_scale": 3,
    # 最大冲突消解轮数。
    "modify_max_passes": 12,
    # 平均 top-1 key overload 低于该阈值时停止继续修正。
    "modify_stop_conflict": 0.5,
    # CGAR 执行频率；默认每个 sampler step 都执行，保持原始效果。
    # 如果只是做速度/效果消融，可以把 stride 调大。
    "modify_step_stride": 1,
    # CGAR 生效的 step 范围，按 sampler step_idx 计数。
    # start=0/end=None 表示全程生效。
    "modify_start_step": 0,
    "modify_end_step": None,
    # CGAR 实现选择：
    # - "fixed"：和 legacy 同一套 winner/loser 更新规则，但减少 CPU/GPU 同步。
    # - "legacy"：原始实现，主要用于对照检查。
    "modify_impl": "fixed",
    # modify_temperature 当前保留接口兼容，实际 modify_attn_score 内部暂未使用。
}


ATTENTION_GATE_DEFAULTS = {
    # 是否在 sparse attention 中启用 gate；默认 False，保持旧行为。
    "sparse_gate_attn": False,
    # gate 位置选择：
    # - "logits"：在 logits/probability 路径中 gate，和 native attention 路径绑定。
    # - "post"：在 xformers/flash/sdpa 输出后 gate，更便宜，适合没有执行 CGAR 的 step。
    "gate_mode": "logits",
    # gate_mode="logits" 使用：抑制高熵、低最大 logit 的不确定 attention 输出。
    "gate_entropy_threshold": 6.0,
    "gate_max_logit_threshold": 1.0,
    # gate_mode="post" 使用：基于 q/k confidence 的输出 gate 阈值。
    "gate_qk_confidence_threshold": 1.0,
}


# 方法 preset：build_morphing_params(methods) 会把这里的参数合并到 base params。
# 小参数大多在对应方法的实现位置中使用，下面只说明主要功能。
METHOD_PRESETS = {
    # CGAR：解决 QK top-key 扎堆竞争问题。默认作用在稀疏结构阶段的 dense
    # cross-attention logits 上；sa_use=False 表示默认不改 self-attention。
    "CGAR": {
        **ATTENTION_MODIFY_DEFAULTS,
        **ATTENTION_GATE_DEFAULTS,
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
    },
    # MCRF / OT：构建稀疏结构 anchor，并用 OT 约束运动一致性。
    # cost_* 影响 anchor 匹配代价；filter_* 控制后续局部平滑强度和邻域。
    "MCRF": {
        "ot_coherence_enabled": True,
        # 生效阶段选择；当前主要用于 sparse structure 阶段。
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
    # CA_OC：坐标/方向一致性修正。通过 grid descriptor 比较候选旋转，
    # 尽量保持 source/target 方向对齐。
    "CA_OC": {
        "ss_ca_oc_flag": True,
        "ss_ca_oc_grid_size": 16,
        "ss_ca_oc_desc_dim": 32,
    },
    # PF_DPLC：source/target 特征插值前的局部修正。
    # k/chunk_size 控制邻域搜索；lam/max_delta 控制修正强度；
    # collapse/residual 相关阈值用于识别容易局部塌缩的位置。
    "PF_DPLC": {
        "enable_pre_fusion_dplc": True,
        "dplc_k": 8,
        "dplc_lam": 0.15,
        "dplc_collapse_th": 0.75,
        "dplc_residual_quantile": 0.80,
        "dplc_max_delta_ratio": 0.15,
        "dplc_chunk_size": 1024,
        "dplc_debug": False,
        "dplc_print_stats": False,
    },
    # DPLC_EVOLUTION：在 PF_DPLC 基础上加入历史帧感知的局部融合进度调整。
    # 会使用 coords_cache_mode 中保存的上一些 sparse coords，在 birth/death 区域
    # 局部放慢或推进 alpha。
    "DPLC_EVOLUTION": {
        "enable_pre_fusion_dplc": True,
        "dplc_k": 8,
        "dplc_lam": 0.15,
        "dplc_collapse_th": 0.75,
        "dplc_residual_quantile": 0.80,
        "dplc_max_delta_ratio": 0.15,
        "dplc_chunk_size": 1024,
        "dplc_debug": False,
        "dplc_print_stats": False,
        "enable_dplc_evolution": True,
        # 历史窗口和 alpha shift 控制：
        # history_window 表示看前几帧；start 表示 EVO 开始明显生效的 alpha 区间；
        # strength/slow_strength/max_shift 限制局部融合进度的调整幅度。
        "dplc_evo_history_window": 3,
        "dplc_evo_start": 0.35,
        "dplc_evo_strength": 0.18,
        "dplc_evo_slow_strength": 0.08,
        "dplc_evo_max_shift": 0.20,
        # support gate 和可选平滑，主要用于压制孤立噪声 shift。
        "dplc_evo_min_neighbors": 1,
        "dplc_evo_smooth_iters": 0,
        "dplc_evo_smooth_weight": 0.0,
        "dplc_evo_trend_bonus": 0.0,
        "dplc_evo_support_floor": 0.65,
        "dplc_evo_support_power": 1.0,
        "dplc_evo_history_floor": 0.50,
        "dplc_evo_debug": False,
        "dplc_evo_debug_every_call": False,
        "dplc_evo_debug_block": 0,
    },
    # MAVF：motion-aware vector field 修正。参数主要在 sparse-structure
    # transformer block 的 _apply_mavf_v0_birth_field 中使用。
    "MAVF": {
        "mavf_enable": True,
        "mavf_lambda_feat": 1.0,
        "mavf_lambda_delta": 0.3,
        "mavf_eta_rank": 0.2,
        "mavf_sigma": 0.20,
        "mavf_eps": 0.15,
        "mavf_integration_bins": 128,
    },
}


METHOD_ALIASES = {
    "MODIFY_GATE": "CGAR",
    "OT": "MCRF",
    "DPLC": "PF_DPLC",
    "PF-DPLC": "PF_DPLC",
    "DPLC_EVO": "DPLC_EVOLUTION",
    "PF_DPLC_EVO": "DPLC_EVOLUTION",
    "PF-DPLC-EVO": "DPLC_EVOLUTION",
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
