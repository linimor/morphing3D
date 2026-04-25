import os
# os.environ['SPARSE_ATTN_BACKEND'] = 'naive'
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import *

SEED = 0
pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
pipeline.cuda()

src_img_path_list = []
tar_img_path_list = []

for tmp_name in [
    # ["typical_vehicle_pirate_ship.png", "typical_vehicle_excavator.png"]
    # ["bee.png", "red_tree.png"],
    ["Super_Big_Mech.png", "bee.png"],
    # ["0004.png", "3015.png"],
    # ["0003.png", "0004.png"],
    # ["Pigsy.png", "Sun_Wukong.png"],
    # ["Bull_Demon_King.png", "head.png"],
    # ["Big_Mesh_red.png", "Sun_Wukong.png"]
    ]:
    src_img_path_list.append(f"./assets/example_morphing/{tmp_name[0]}")
    tar_img_path_list.append(f"./assets/example_morphing/{tmp_name[1]}")

save_dir_path = "./outputs"

for idx in range(len(src_img_path_list)):
    src_img_path = src_img_path_list[idx]
    tar_img_path = tar_img_path_list[idx]
    src_img = Image.open(src_img_path)
    tar_img = Image.open(tar_img_path)
    src_name = os.path.basename(src_img_path).split(".")[0]
    tar_name = os.path.basename(tar_img_path).split(".")[0]
    src_save_path = os.path.join(save_dir_path, "cache", src_name)
    tar_save_path = os.path.join(save_dir_path, "cache", tar_name)
    src_save_cache_path = os.path.join(src_save_path, "cache")
    tar_save_cache_path = os.path.join(tar_save_path, "cache")
    os.makedirs(src_save_path, exist_ok=True)
    os.makedirs(tar_save_path, exist_ok=True)
    os.makedirs(src_save_cache_path, exist_ok=True)
    os.makedirs(tar_save_cache_path, exist_ok=True)
    morphing_params = {"save_cache_path": src_save_cache_path, 
                       "init_morphing_flag": False, 
                       "ss_mca_flag": False, 
                       "slat_mca_flag": False, 
                       "ss_tfsa_flag": False, 
                       "slat_tfsa_flag": False, 
                       "oc_flag": False}
    if not os.path.exists(f"{src_save_cache_path}/slat_init.pt"):
        run_morphing_cache(pipeline, src_img, tar_img, morphing_params, SEED, src_save_path, src_name)
    morphing_params = {"save_cache_path": tar_save_cache_path, 
                       "init_morphing_flag": False, 
                       "ss_mca_flag": False, 
                       "slat_mca_flag": False, 
                       "ss_tfsa_flag": False, 
                       "slat_tfsa_flag": False, 
                       "oc_flag": False}
    if not os.path.exists(f"{tar_save_cache_path}/slat_init.pt"):
        run_morphing_cache(pipeline, tar_img, src_img, morphing_params, SEED, tar_save_path, tar_name)

    name = src_name + "+" + tar_name + "_ot_modify_gate"

    # Morphing 基础控制：
    # - ss/slat_mca_flag：开启 SS/SLAT 阶段的 source-target cross-attention 插值。
    # - ss/slat_tfsa_flag：开启 SS/SLAT 阶段的 temporal feature self-attention 复用。
    # - modify：对 cross-attention logits 做冲突修正，减少多个 query 同时塌到同一个 key。
    # - gate_attn：对不确定性很高的 cross-attention residual 做抑制，避免错误残差强行写入。
    # - modify_lambda_scale：attention 冲突修正强度，越大冲突区域改得越明显。
    #
    # OT coherence 控制：
    # - ot_coherence_enabled：开启 training-free 的 SS residual coherent filter。
    # - ot_coherence_stage：保持为 "ss"，表示 OT 只作用在 SS MCA output 之后，不改 SLAT。
    # - ot_anchor_patch_size：把 SS active voxels 按 3D patch 聚合成 anchors。
    # - ot_max_anchors：限制最多使用多少 anchors，避免 Sinkhorn OT 太慢。
    # - ot_cost_pos_weight / ot_cost_feat_weight：控制 OT cost 中位置距离和几何特征距离的权重。
    # - ot_sinkhorn_eps / ot_sinkhorn_iters：控制 Sinkhorn 软匹配的熵正则强度和迭代次数。
    # - ot_filter_k：每个 SS token 平滑 residual 时使用的空间近邻 token 数。
    # - ot_filter_sigma_pos：空间距离权重带宽，用于 token-anchor 和 token-token 权重。
    # - ot_filter_sigma_motion：运动方向相似度带宽，用于约束 residual 平滑只在相似运动区域传播。
    # - ot_filter_lambda：coherent filter 最大强度，越大 residual 越平滑、整体运动越一致。
    # - ot_filter_use_confidence：根据 OT 置信度调节强度；匹配越不确定，平滑越强。
    # - ot_filter_start_step_ratio / ot_filter_end_step_ratio：控制 SS sampling 中 filter 强度随 step 衰减。
    # - ot_debug：保存/打印 OT motion field 调试信息到 outputs/debug_ot。
    morphing_params = {"morphing_num": 50, 
                       "src_load_cache_path": src_save_cache_path,
                        "tar_load_cache_path": tar_save_cache_path, 
                        "init_morphing_flag": False, 
                        "ss_mca_flag": True, 
                        "slat_mca_flag": True, 
                        "ss_tfsa_flag": True, 
                        "slat_tfsa_flag": True, 
                        "oc_flag": True,
                        "modify": True,
                        "gate_attn": True,
                        "sa_use": True,
                        "modify_lambda_scale": 0.8,
                        "ot_coherence_enabled": True,
                        "ot_coherence_stage": "ss",
                        "ot_anchor_patch_size": 4,
                        "ot_max_anchors": 512,
                        "ot_cost_pos_weight": 0.3,
                        "ot_cost_feat_weight": 1.0,
                        "ot_sinkhorn_eps": 0.05,
                        "ot_sinkhorn_iters": 80,
                        "ot_filter_k": 16,
                        "ot_filter_sigma_pos": 2.0,
                        "ot_filter_sigma_motion": 2.0,
                        "ot_filter_lambda": 0.3,
                        "ot_filter_use_confidence": True,
                        "ot_filter_start_step_ratio": 1.0,
                        "ot_filter_end_step_ratio": 0.0,
                        "ot_debug": False,
                        }
    save_path = os.path.join(save_dir_path, "3Dmorphing", name)
    os.makedirs(save_path, exist_ok=True)
    save_cache_path = os.path.join(save_path, "cache")
    os.makedirs(save_cache_path, exist_ok=True)
    morphing_params["save_cache_path"] = save_cache_path
    run_morphing(pipeline, src_img, tar_img, morphing_params, SEED, save_path, name)
