import os
from pathlib import Path

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import run_morphing


def main():
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    seed = 0
    src_name = "bee"
    tar_name = "red_tree"

    src_img = Image.open(f"./assets/example_morphing/{src_name}.png")
    tar_img = Image.open(f"./assets/example_morphing/{tar_name}.png")

    pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    pipeline.cuda()

    save_path = Path("./outputs/3Dmorphing/bee+red_tree_ss_ca_knn_full")
    save_cache_path = save_path / "cache"
    save_cache_path.mkdir(parents=True, exist_ok=True)

    morphing_params = {
        # Full morphing run.
        "morphing_num": 40,
        "src_load_cache_path": "./outputs/cache/bee/cache",
        "tar_load_cache_path": "./outputs/cache/red_tree/cache",
        "save_cache_path": str(save_cache_path),
        "init_morphing_flag": False,
        # Keep the original morphing path enabled.
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": True,
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
        "insert_num": 3,
        "noise_init": True,
        "noise_": None,
        # SS-only CA token-wise KNN alpha.
        "use_ca_knn_alpha": True,
        "ca_knn_k": 8,
        "ca_knn_temperature": 0.15,
        "ca_knn_projection_gain": 0.65,
        "ca_knn_smoothness": 0.35,
        "ca_knn_alpha_offset_scale": 1.0,
        "ca_knn_alpha_debug": False,
    }

    run_morphing(
        pipeline,
        src_img,
        tar_img,
        morphing_params,
        seed,
        str(save_path),
        "bee+red_tree_ss_ca_knn_full",
    )
    print(f"DONE {save_path}")


if __name__ == "__main__":
    main()
