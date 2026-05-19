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


def pca_lowpass(feats: torch.Tensor, rank: int = 8) -> torch.Tensor:
    x = feats.float()
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    rank = max(1, min(int(rank), min(xc.shape) - 1))
    try:
        u, s, v = torch.pca_lowrank(xc, q=rank, center=False, niter=2)
        low = (u[:, :rank] * s[:rank]) @ v[:, :rank].T + mean
    except RuntimeError:
        u, s, v = torch.linalg.svd(xc, full_matrices=False)
        low = (u[:, :rank] * s[:rank]) @ v[:rank, :] + mean
    return low.to(dtype=feats.dtype)


def fft_grid_lowpass(feats: torch.Tensor, coords: torch.Tensor, grid_size: int = 16, keep_ratio: float = 0.25) -> torch.Tensor:
    x = feats.float()
    xyz = coords[:, 1:].long() if coords.shape[-1] == 4 else coords.long()
    xyz = xyz.clamp(0, 63)
    cell = torch.div(xyz * int(grid_size), 64, rounding_mode="floor").clamp(0, int(grid_size) - 1)
    linear = cell[:, 0] * grid_size * grid_size + cell[:, 1] * grid_size + cell[:, 2]

    grid = torch.zeros(grid_size ** 3, x.shape[1], device=x.device, dtype=torch.float32)
    count = torch.zeros(grid_size ** 3, 1, device=x.device, dtype=torch.float32)
    grid.index_add_(0, linear, x)
    count.index_add_(0, linear, torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32))
    grid = grid / count.clamp(min=1.0)
    grid = grid.reshape(grid_size, grid_size, grid_size, x.shape[1]).permute(3, 0, 1, 2).contiguous()

    freq = torch.fft.fftn(grid, dim=(-3, -2, -1))
    fx = torch.fft.fftfreq(grid_size, device=x.device).reshape(grid_size, 1, 1)
    fy = torch.fft.fftfreq(grid_size, device=x.device).reshape(1, grid_size, 1)
    fz = torch.fft.fftfreq(grid_size, device=x.device).reshape(1, 1, grid_size)
    radius = torch.sqrt(fx * fx + fy * fy + fz * fz)
    cutoff = float(keep_ratio) * 0.5
    mask = radius <= cutoff
    low_grid = torch.fft.ifftn(freq * mask.unsqueeze(0), dim=(-3, -2, -1)).real
    low_grid = low_grid.permute(1, 2, 3, 0).reshape(grid_size ** 3, x.shape[1])
    return low_grid[linear].to(dtype=feats.dtype)


def decompose(feats: torch.Tensor, coords: torch.Tensor, method: str, pca_rank: int, fft_grid: int, fft_keep: float):
    if method == "pca":
        low = pca_lowpass(feats, rank=pca_rank)
    elif method == "fft":
        low = fft_grid_lowpass(feats, coords, grid_size=fft_grid, keep_ratio=fft_keep)
    else:
        raise ValueError(f"Unknown decomposition method: {method}")
    return low, feats - low


def make_decomp_interp(method: str, part: str, pca_rank: int, fft_grid: int, fft_keep: float):
    def decomp_interp(slat1, slat2, alpha: float, mapping_mode="order", interp_mode="linear", unique_flag=True, indices=None):
        if slat1.feats.shape != slat2.feats.shape:
            return original_slat_interp(slat1, slat2, alpha, mapping_mode, interp_mode, unique_flag, indices)
        low1, high1 = decompose(slat1.feats, slat1.coords, method, pca_rank, fft_grid, fft_keep)
        low2, high2 = decompose(slat2.feats, slat2.coords, method, pca_rank, fft_grid, fft_keep)
        if part == "low_only":
            feats = alpha * low1 + (1.0 - alpha) * low2 + high1
        elif part == "high_only":
            feats = low1 + alpha * high1 + (1.0 - alpha) * high2
        elif part == "both":
            feats = alpha * low1 + (1.0 - alpha) * low2 + alpha * high1 + (1.0 - alpha) * high2
        else:
            feats = alpha * slat1.feats + (1.0 - alpha) * slat2.feats
        return slat1.replace(feats)
    return decomp_interp


def render_four_view(sample, resolution=320):
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitch = [np.deg2rad(20)] * 4
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    out = render_utils.render_frames(sample, extr, intr, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False)
    return np.concatenate(out["color"], axis=1)


def label_row(img, text):
    label_h = 50
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + label_h), "white")
    canvas.paste(Image.fromarray(img), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = None
    draw.text((14, 13), text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    out_dir = Path("outputs/diagnostics/goblin_dragon_decomp_fusion")
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
        ("linear", "linear", "none", "baseline"),
        ("pca_low", "pca", "low_only", "PCA distribution low fused, residual from source"),
        ("pca_high", "pca", "high_only", "PCA distribution high fused, low from source"),
        ("fft_low", "fft", "low_only", "3D FFT low fused, residual from source"),
        ("fft_high", "fft", "high_only", "3D FFT high fused, low from source"),
    ]

    rows = []
    rendered_rows = []
    for name, method, part, desc in variants:
        if name == "linear":
            sparse_modulated.slat_interp = original_slat_interp
        else:
            sparse_modulated.slat_interp = make_decomp_interp(method, part, pca_rank=8, fft_grid=16, fft_keep=0.25)
        cache_dir = out_dir / name / "cache"
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
        img = render_four_view(outputs["gaussian"][0], resolution=320)
        imageio.imwrite(out_dir / f"{name}_alpha0p5.png", img)
        rendered_rows.append(label_row(img, f"{name}: {desc}"))
        rows.append({"name": name, "method": method, "part": part, "description": desc, "image": str(out_dir / f"{name}_alpha0p5.png")})
        print(f"[rendered] {name}")

    sparse_modulated.slat_interp = original_slat_interp
    sheet = np.concatenate(rendered_rows, axis=0)
    imageio.imwrite(out_dir / "comparison_alpha0p5.png", sheet)
    write_csv(out_dir / "summary.csv", rows)
    print(f"[done] {out_dir / 'comparison_alpha0p5.png'}")


if __name__ == "__main__":
    main()
