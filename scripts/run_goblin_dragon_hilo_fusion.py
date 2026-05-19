import csv
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import seed_everything, slat_interp as original_slat_interp
import trellis.modules.sparse.transformer.modulated as sparse_modulated


def coarse_lowpass(feats: torch.Tensor, coords: torch.Tensor, cell_size: int = 4) -> torch.Tensor:
    xyz = coords[:, 1:].long() if coords.shape[-1] == 4 else coords.long()
    cell = torch.div(xyz, int(cell_size), rounding_mode="floor")
    linear = cell[:, 0] * 4096 + cell[:, 1] * 64 + cell[:, 2]
    unique, inverse = torch.unique(linear, sorted=False, return_inverse=True)
    low = torch.zeros(unique.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
    count = torch.zeros(unique.shape[0], 1, device=feats.device, dtype=feats.dtype)
    low.index_add_(0, inverse, feats)
    count.index_add_(0, inverse, torch.ones(feats.shape[0], 1, device=feats.device, dtype=feats.dtype))
    return low.div(count.clamp(min=1.0))[inverse]


def make_hilo_interp(mode: str, cell_size: int):
    def hilo_interp(slat1, slat2, alpha: float, mapping_mode="order", interp_mode="linear", unique_flag=True, indices=None):
        if slat1.feats.shape != slat2.feats.shape:
            return original_slat_interp(slat1, slat2, alpha, mapping_mode, interp_mode, unique_flag, indices)
        low1 = coarse_lowpass(slat1.feats, slat1.coords, cell_size=cell_size)
        low2 = coarse_lowpass(slat2.feats, slat2.coords, cell_size=cell_size)
        high1 = slat1.feats - low1
        high2 = slat2.feats - low2
        if mode == "low_only":
            feats = alpha * low1 + (1.0 - alpha) * low2 + high1
        elif mode == "high_only":
            feats = low1 + alpha * high1 + (1.0 - alpha) * high2
        elif mode == "hilo":
            feats = alpha * low1 + (1.0 - alpha) * low2 + alpha * high1 + (1.0 - alpha) * high2
        else:
            feats = alpha * slat1.feats + (1.0 - alpha) * slat2.feats
        return slat1.replace(feats)
    return hilo_interp


def render_four_view(sample, resolution=384):
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitch = [np.deg2rad(20)] * 4
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    out = render_utils.render_frames(sample, extr, intr, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False)
    return np.concatenate(out["color"], axis=1)


def label_row(img, text):
    label_h = 54
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + label_h), "white")
    canvas.paste(Image.fromarray(img), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = None
    draw.text((16, 14), text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    out_dir = Path("outputs/diagnostics/goblin_dragon_hilo_fusion")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    pipeline.cuda()
    src_img = Image.open("./assets/example_morphing/typical_humanoid_goblin.png")
    tar_img = Image.open("./assets/example_morphing/typical_creature_dragon.png")

    common = {
        "morphing_num": 3,
        "src_load_cache_path": "./outputs/cache/typical_humanoid_goblin/cache",
        "tar_load_cache_path": "./outputs/cache/typical_creature_dragon/cache",
        "init_morphing_flag": True,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": False,
        "oc_flag": False,
        "alpha": 0.5,
        "morphing_idx": 1,
        "tfsa_cache_idx": 0,
        "tfsa_alpha": 0.8,
        "sa_use": False,
        "modify": False,
        "gate_attn": True,
        "gate_mode": "post",
        "ot_coherence_enabled": False,
    }
    variants = [
        ("linear", "linear baseline"),
        ("low_only", "only low-frequency fused, high-frequency kept from source branch"),
        ("high_only", "only high-frequency fused, low-frequency kept from source branch"),
    ]

    rows = []
    rendered_rows = []
    for mode, description in variants:
        sparse_modulated.slat_interp = make_hilo_interp(mode, cell_size=4)
        cache_dir = out_dir / mode / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        params = {**common, "save_cache_path": str(cache_dir)}
        seed_everything(0)
        with torch.no_grad():
            outputs = pipeline.run_morphing(
                src_img=src_img,
                tar_img=tar_img,
                morphing_params=params,
                seed=0,
                sparse_structure_sampler_params={"steps": 4},
                slat_sampler_params={"steps": 4},
                formats=["gaussian"],
            )
        img = render_four_view(outputs["gaussian"][0], resolution=384)
        imageio.imwrite(out_dir / f"{mode}_alpha0p5.png", img)
        rendered_rows.append(label_row(img, f"{mode}: {description}"))
        rows.append({"mode": mode, "description": description, "image": str(out_dir / f"{mode}_alpha0p5.png")})
        print(f"[rendered] {mode}")

    sparse_modulated.slat_interp = original_slat_interp
    sheet = np.concatenate(rendered_rows, axis=0)
    imageio.imwrite(out_dir / "comparison_alpha0p5.png", sheet)
    write_csv(out_dir / "summary.csv", rows)
    print(f"[done] {out_dir / 'comparison_alpha0p5.png'}")


if __name__ == "__main__":
    main()
