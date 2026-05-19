import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import run_morphing, run_morphing_cache


def ensure_cache(pipeline, img, other_img, cache_dir, root_dir, name, seed):
    cache_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "slat_init.pt").exists():
        return
    params = {
        "save_cache_path": str(cache_dir),
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "slat_mca_flag": False,
        "ss_tfsa_flag": False,
        "slat_tfsa_flag": False,
        "oc_flag": False,
    }
    run_morphing_cache(pipeline, img, other_img, params, seed, str(root_dir), name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--src", default="./assets/example_morphing/typical_humanoid_goblin.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_creature_dragon.png")
    parser.add_argument("--out-root", default="./outputs/3Dmorphing")
    parser.add_argument("--cache-root", default="./outputs/cache")
    parser.add_argument("--morphing-num", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modify-lambda-scale", type=float, default=3.0)
    parser.add_argument("--modify-max-passes", type=int, default=12)
    parser.add_argument("--modify-stop-conflict", type=float, default=0.0)
    args = parser.parse_args()

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    src_path = Path(args.src)
    tar_path = Path(args.tar)
    src_img = Image.open(src_path)
    tar_img = Image.open(tar_path)
    src_name = src_path.stem
    tar_name = tar_path.stem

    cache_root = Path(args.cache_root)
    src_root = cache_root / src_name
    tar_root = cache_root / tar_name
    src_cache = src_root / "cache"
    tar_cache = tar_root / "cache"
    ensure_cache(pipeline, src_img, tar_img, src_cache, src_root, src_name, args.seed)
    ensure_cache(pipeline, tar_img, src_img, tar_cache, tar_root, tar_name, args.seed)

    run_name = (
        f"{src_name}+{tar_name}_standard_modify"
        f"_lambda{args.modify_lambda_scale:g}"
        f"_passes{args.modify_max_passes}"
        f"_stop{args.modify_stop_conflict:g}"
    )
    save_path = Path(args.out_root) / run_name
    save_cache = save_path / "cache"
    save_path.mkdir(parents=True, exist_ok=True)
    save_cache.mkdir(parents=True, exist_ok=True)

    morphing_params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "save_cache_path": str(save_cache),
        "init_morphing_flag": False,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": False,
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
        "ss_ca_oc_flag": True,
        "delete_loaded_ca_oc_cache": True,
        "modify_lambda_scale": args.modify_lambda_scale,
        "modify_max_passes": args.modify_max_passes,
        "modify_stop_conflict": args.modify_stop_conflict,
        "ot_coherence_enabled": False,
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
        "ot_debug": False,
        "delete_loaded_tfsa_cache": True,
    }
    run_morphing(pipeline, src_img, tar_img, morphing_params, args.seed, str(save_path), run_name)
    print(f"[done] {save_path}")


if __name__ == "__main__":
    main()
