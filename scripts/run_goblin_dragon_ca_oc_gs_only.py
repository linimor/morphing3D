import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import run_morphing_cache, seed_everything


def save_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T, H, W, 3] frames, got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(int(fps), 1)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def ensure_cache(pipeline, img, other_img, cache_dir: Path, root_dir: Path, name: str, seed: int) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--src", default="./assets/example_morphing/typical_humanoid_goblin.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_creature_dragon.png")
    parser.add_argument("--out-root", default="./outputs/3Dmorphing_ca_oc_test")
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

    cache_root = Path(args.cache_root)
    src_cache = cache_root / src_path.stem / "cache"
    tar_cache = cache_root / tar_path.stem / "cache"
    ensure_cache(pipeline, src_img, tar_img, src_cache, src_cache.parent, src_path.stem, args.seed)
    ensure_cache(pipeline, tar_img, src_img, tar_cache, tar_cache.parent, tar_path.stem, args.seed)

    run_name = f"{src_path.stem}+{tar_path.stem}_ss_ca_oc_modify_gate_gs_only"
    out_dir = Path(args.out_root) / run_name
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in cache_dir.glob("*"):
        f.unlink()

    params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "save_cache_path": str(cache_dir),
        "init_morphing_flag": False,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": False,
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
        "modify_lambda_scale": args.modify_lambda_scale,
        "modify_max_passes": args.modify_max_passes,
        "modify_stop_conflict": args.modify_stop_conflict,
        "ss_ca_oc_flag": True,
        "delete_loaded_ca_oc_cache": True,
        "delete_loaded_tfsa_cache": True,
        "ot_coherence_enabled": False,
    }

    seed_everything(args.seed)
    alpha_array = np.linspace(1, 0, args.morphing_num)
    gs_video_list = []
    for morphing_idx in range(1, args.morphing_num - 1):
        params["alpha"] = float(alpha_array[morphing_idx])
        params["morphing_idx"] = morphing_idx
        params["tfsa_cache_idx"] = morphing_idx - 1
        params["tfsa_alpha"] = 0.8
        params["return_intermediate"] = True
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=params,
            seed=args.seed,
            formats=["gaussian"],
        )
        gs_video_list.append(
            np.stack(render_utils.render_rot_video(outputs["gaussian"][0], bg_color=(1, 1, 1))["color"], axis=0)
        )
        for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
            for f in cache_dir.glob(f"{prefix}{params['tfsa_cache_idx']}_*"):
                f.unlink()
        print(f"[frame] {morphing_idx}/{args.morphing_num - 2}")

    gs_video = np.stack(gs_video_list, axis=0)
    view_idx = [0, 20, 40, 60, 80, 100]
    morphing_gs_video = np.concatenate(np.transpose(gs_video[:, view_idx, ...], (1, 0, 2, 3, 4)), axis=2)
    show_idx = np.arange(0, args.morphing_num - 2, max((args.morphing_num - 2) // 4, 1)).tolist()
    if len(show_idx) < 5 and args.morphing_num > 3:
        show_idx.append(args.morphing_num - 3)
    show_idx = sorted(set(min(i, gs_video.shape[0] - 1) for i in show_idx))
    show_gs_video = np.concatenate(gs_video[show_idx, ...], axis=2)

    save_mp4(out_dir / f"morphing_{run_name}.mp4", morphing_gs_video, fps=max((args.morphing_num - 2) // 2, 1))
    save_mp4(out_dir / f"show_{run_name}.mp4", show_gs_video, fps=30)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
