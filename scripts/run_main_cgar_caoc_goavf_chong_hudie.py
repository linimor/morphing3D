import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_args import build_morphing_params
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


def clear_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for child in cache_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.name in {"goavf_qfield_debug", "ss_diff_debug"}:
            for path in child.glob("*"):
                if path.is_file():
                    path.unlink()


def ensure_endpoint_cache(pipeline, img, other_img, cache_path: Path, save_path: Path, name: str, seed: int) -> None:
    required = ("cond.pt", "coords.pt", "slat_init.pt")
    if all((cache_path / filename).exists() for filename in required):
        return
    cache_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    params = {
        "save_cache_path": str(cache_path),
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "slat_mca_flag": False,
        "ss_tfsa_flag": False,
        "slat_tfsa_flag": False,
        "oc_flag": False,
    }
    run_morphing_cache(pipeline, img, other_img, params, seed, str(save_path), name)


def validate_qfield(path: Path, variant: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing G-OAVF qfield file: {path}")
    data = torch.load(path, map_location="cpu")
    if "p" not in data or variant not in data:
        raise KeyError(f"Expected keys 'p' and '{variant}' in {path}")
    q = torch.as_tensor(data[variant])
    if q.ndim != 2 or q.shape[1] != 4096:
        raise ValueError(f"Expected {variant} qfield shape [T, 4096], got {tuple(q.shape)}")


def render_frame(sample, resolution: int) -> np.ndarray:
    return render_utils.render_rot_video(
        sample,
        resolution=resolution,
        bg_color=(1, 1, 1),
        num_frames=1,
    )["color"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main-flow CGAR + CA_OC + temporary G-OAVF qfield.")
    parser.add_argument("--src-name", default="chong")
    parser.add_argument("--tar-name", default="hudie")
    parser.add_argument("--morphing-num", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="B_local_residual")
    parser.add_argument("--activation", default="arctan")
    parser.add_argument("--activation-mu", type=float, default=0.45)
    parser.add_argument("--activation-sharpness", type=float, default=12.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--qfield-path",
        type=Path,
        default=ROOT / "outputs/analysis/goavf_contrast_residual_chong_hudie/contrast_q_fields.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_img_path = ROOT / f"assets/example_morphing/{args.src_name}.png"
    tar_img_path = ROOT / f"assets/example_morphing/{args.tar_name}.png"
    if not src_img_path.exists() or not tar_img_path.exists():
        raise FileNotFoundError(f"Missing input image: {src_img_path} or {tar_img_path}")
    validate_qfield(args.qfield_path, args.variant)

    src_img = Image.open(src_img_path)
    tar_img = Image.open(tar_img_path)
    src_save_path = ROOT / "outputs/cache" / args.src_name
    tar_save_path = ROOT / "outputs/cache" / args.tar_name
    src_cache = src_save_path / "cache"
    tar_cache = tar_save_path / "cache"

    activation_tag = (
        f"{args.activation}_mu{args.activation_mu:.2f}_s{args.activation_sharpness:.1f}"
        .replace(".", "p")
    )
    run_name = f"{args.src_name}+{args.tar_name}_CGAR_CA_OC_GOAVF_QFIELD_{args.variant}_{activation_tag}"
    out_dir = ROOT / "outputs/3Dmorphing" / run_name
    cache_dir = out_dir / "cache"
    fixed_video_path = out_dir / f"{run_name}_fixed_view.mp4"
    if fixed_video_path.exists() and not args.overwrite:
        print(f"Skip existing {fixed_video_path}")
        return

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(ROOT / "TRELLIS-image-large"))
    pipeline.cuda()
    seed_everything(args.seed)

    ensure_endpoint_cache(pipeline, src_img, tar_img, src_cache, src_save_path, args.src_name, args.seed)
    ensure_endpoint_cache(pipeline, tar_img, src_img, tar_cache, tar_save_path, args.tar_name, args.seed)
    clear_cache_dir(cache_dir)

    morphing_params = build_morphing_params(
        ["CGAR", "CA_OC", "GOAVF_QFIELD"],
        overrides={
            "morphing_num": args.morphing_num,
            "src_load_cache_path": str(src_cache),
            "tar_load_cache_path": str(tar_cache),
            "save_cache_path": str(cache_dir),
            "goavf_qfield_path": str(args.qfield_path),
            "goavf_qfield_variant": args.variant,
            "goavf_qfield_activation": args.activation,
            "goavf_qfield_activation_mu": args.activation_mu,
            "goavf_qfield_activation_sharpness": args.activation_sharpness,
            "oc_flag": False,
            "return_intermediate": True,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_array = np.linspace(1.0, 0.0, args.morphing_num)
    fixed_frames = []
    for morphing_idx in range(1, args.morphing_num - 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        morphing_params["tfsa_alpha"] = 0.8
        with torch.no_grad():
            outputs = pipeline.run_morphing(
                src_img=src_img,
                tar_img=tar_img,
                morphing_params=morphing_params,
                seed=args.seed,
                formats=["gaussian"],
            )
        fixed_frames.append(render_frame(outputs["gaussian"][0], args.resolution))
        for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
            for path in cache_dir.glob(f"{prefix}{morphing_params['tfsa_cache_idx']}_*"):
                path.unlink()
        print(f"[CGAR+CA_OC+G-OAVF] frame {morphing_idx}/{args.morphing_num - 2} alpha={morphing_params['alpha']:.4f}")

    fixed_video = np.stack(fixed_frames, axis=0)
    save_mp4(fixed_video_path, fixed_video, fps=12)
    print(f"[CGAR+CA_OC+G-OAVF] wrote: {fixed_video_path}")
    print(f"[CGAR+CA_OC+G-OAVF] cache/debug: {cache_dir}")


if __name__ == "__main__":
    main()
