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
from trellis.utils.morphing_utils import seed_everything


def save_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T, H, W, 3] frames, got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(max(int(fps), 1)),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def clear_run_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.name in {"ss_diff_debug", "scaf_debug"}:
            for f in child.glob("*"):
                f.unlink()


def main() -> None:
    seed = 0
    morphing_num = 25
    src_name = "chong"
    tar_name = "hudie"
    out_dir = ROOT / "outputs/analysis/scaf_chong_hudie_m25/SCAF_default"
    cache_dir = out_dir
    clear_run_dir(out_dir)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(ROOT / "TRELLIS-image-large"))
    pipeline.cuda()

    src_img = Image.open(ROOT / f"assets/example_morphing/{src_name}.png")
    tar_img = Image.open(ROOT / f"assets/example_morphing/{tar_name}.png")
    src_cache = ROOT / f"outputs/cache/{src_name}/cache"
    tar_cache = ROOT / f"outputs/cache/{tar_name}/cache"
    if not (src_cache / "slat_init.pt").exists() or not (tar_cache / "slat_init.pt").exists():
        raise FileNotFoundError("Expected endpoint caches under outputs/cache/chong/cache and outputs/cache/hudie/cache")

    endpoint_occ_path = src_cache / f"ss_endpoint_occ16_to_{tar_name}.pt"
    params = {
        "morphing_num": morphing_num,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "ss_endpoint_occ16_path": str(endpoint_occ_path),
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
        "modify_lambda_scale": 0.8,
        "ot_coherence_enabled": False,
        "dump_ss_diff_debug": True,
        "dump_ss_token_debug": False,
        "scaf_enable": True,
        "scaf_integration_bins": 128,
        "scaf_debug": True,
        "scaf_debug_step_idx": 24,
        "scaf_debug_block_idx": 23,
        "mavf_enable": True,
        "enable_cmf_v0": False,
        "enable_ddpf_v0": False,
    }

    seed_everything(seed)
    alpha_array = np.linspace(1.0, 0.0, morphing_num)
    fixed_frames = []
    for morphing_idx in range(1, morphing_num - 1):
        params["alpha"] = float(alpha_array[morphing_idx])
        params["morphing_idx"] = morphing_idx
        params["tfsa_cache_idx"] = morphing_idx - 1
        params["tfsa_alpha"] = 0.8
        params["return_intermediate"] = True
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=params,
            seed=seed,
            formats=["gaussian"],
        )
        frame = render_utils.render_rot_video(
            outputs["gaussian"][0],
            resolution=512,
            bg_color=(1, 1, 1),
            num_frames=1,
        )["color"][0]
        fixed_frames.append(frame)

        for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
            for f in cache_dir.glob(f"{prefix}{params['tfsa_cache_idx']}_*"):
                f.unlink()
        print(f"[SCAF] frame {morphing_idx}/{morphing_num - 2} alpha={params['alpha']:.4f}")

    for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
        for f in cache_dir.glob(f"{prefix}{morphing_num - 2}_*"):
            f.unlink()

    fixed_video = np.stack(fixed_frames, axis=0)
    save_mp4(out_dir / "SCAF_default_fixed_view.mp4", fixed_video, fps=12)
    print(f"[SCAF] done: {out_dir}")


if __name__ == "__main__":
    main()
