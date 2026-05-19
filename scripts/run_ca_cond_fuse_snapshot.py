import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single src-target morphing snapshot and export Gaussian previews."
    )
    parser.add_argument("--src", default="./assets/example_morphing/typical_vehicle_pirate_ship.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_vehicle_excavator.png")
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--out-dir", default="./outputs/ca_cond_fuse")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=["ca-cond-fuse", "normal"],
        default="ca-cond-fuse",
        help="ca-cond-fuse fuses src/tar condition before CA; normal uses the existing morphing CA interpolation path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--interpolate-init-cache",
        action="store_true",
        help="Also interpolate cached SS/SLAT initial noise at alpha. Default keeps current example behavior: fresh seeded noise.",
    )
    return parser.parse_args()


def load_rgba(path):
    return Image.open(path).convert("RGBA")


def ensure_endpoint_cache(pipeline, img, ref_img, cache_dir, seed):
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = ["coords_zs_init.pt", "slat_init.pt", "coords.pt"]
    if all((cache_dir / filename).exists() for filename in required):
        return

    seed_everything(seed)
    params = {
        "save_cache_path": str(cache_dir),
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "slat_mca_flag": False,
        "ss_tfsa_flag": False,
        "slat_tfsa_flag": False,
        "oc_flag": False,
    }
    with torch.no_grad():
        pipeline.run_morphing(
            src_img=img,
            tar_img=ref_img,
            morphing_params=params,
            seed=seed,
            formats=["gaussian"],
        )


def render_view(sample, yaw, out_path, resolution):
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaw,
        np.deg2rad(20),
        2,
        40,
    )
    frames = render_utils.render_frames(
        sample,
        [extrinsics],
        [intrinsics],
        {"resolution": resolution, "bg_color": (1, 1, 1)},
        verbose=False,
    )["color"]
    imageio.imwrite(out_path, frames[0])
    return frames[0]


def make_contact_sheet(src_path, front_path, back_path, tar_path, out_path):
    labels = ["Source", "CA fuse front", "CA fuse back", "Target"]
    paths = [src_path, front_path, back_path, tar_path]
    thumb_size = 320
    header = 42
    gap = 18
    sheet = Image.new("RGB", (thumb_size * 4 + gap * 5, thumb_size + header + gap * 2), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, path) in enumerate(zip(labels, paths)):
        x = gap + idx * (thumb_size + gap)
        draw.text((x, 10), label, fill=(20, 20, 20), font=font)
        img = Image.open(path).convert("RGBA")
        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        canvas = Image.new("RGBA", (thumb_size, thumb_size), "white")
        canvas.alpha_composite(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
        sheet.paste(canvas.convert("RGB"), (x, header + gap))
    sheet.save(out_path)


def main():
    args = parse_args()
    if args.device != "cuda":
        raise ValueError("This TRELLIS setup expects CUDA rendering and sampling.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.src)
    tar_path = Path(args.tar)
    mode_tag = "ca_cond_fuse" if args.mode == "ca-cond-fuse" else "normal_pipeline"
    name = args.name or f"{src_path.stem}+{tar_path.stem}_{mode_tag}_alpha{args.alpha:.2f}".replace(".", "p")

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    src_img = load_rgba(src_path)
    tar_img = load_rgba(tar_path)

    cache_root = out_dir / "cache"
    src_cache = cache_root / src_path.stem
    tar_cache = cache_root / tar_path.stem
    ensure_endpoint_cache(pipeline, src_img, tar_img, src_cache, args.seed)
    ensure_endpoint_cache(pipeline, tar_img, src_img, tar_cache, args.seed)

    seed_everything(args.seed)
    morphing_params = {
        "save_cache_path": str(out_dir / f"{name}_cache"),
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "init_morphing_flag": bool(args.interpolate_init_cache),
        "alpha": args.alpha,
        "ss_mca_flag": args.mode == "normal",
        "slat_mca_flag": args.mode == "normal",
        "ss_mca_cond_fuse_flag": args.mode == "ca-cond-fuse",
        "slat_mca_cond_fuse_flag": args.mode == "ca-cond-fuse",
        "ss_mca_cond_fuse_alpha": args.alpha,
        "slat_mca_cond_fuse_alpha": args.alpha,
        "ss_tfsa_flag": False,
        "slat_tfsa_flag": False,
        "oc_flag": False,
        "modify": True,
        "gate_attn": True,
        "sa_use": True,
        "modify_lambda_scale": 0.8,
        "ot_coherence_enabled": False,
    }
    Path(morphing_params["save_cache_path"]).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=args.seed,
            formats=["gaussian"],
        )

    gaussian = outputs["gaussian"][0]
    ply_path = out_dir / f"{name}.ply"
    gaussian.save_ply(str(ply_path))

    front_path = out_dir / f"{name}_front.png"
    back_path = out_dir / f"{name}_back.png"
    render_view(gaussian, 0, front_path, args.resolution)
    render_view(gaussian, np.pi, back_path, args.resolution)

    src_copy = out_dir / f"{src_path.stem}_input.png"
    tar_copy = out_dir / f"{tar_path.stem}_input.png"
    src_img.save(src_copy)
    tar_img.save(tar_copy)

    sheet_path = out_dir / f"{name}_before_after.png"
    make_contact_sheet(src_copy, front_path, back_path, tar_copy, sheet_path)

    print(f"Gaussian PLY: {ply_path}")
    print(f"Front PNG: {front_path}")
    print(f"Back PNG: {back_path}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
