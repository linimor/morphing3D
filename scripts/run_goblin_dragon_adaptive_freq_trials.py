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


EPS = 1e-6


def _relative_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm() / (a.float().norm() + b.float().norm()).clamp(min=EPS)


def pca_lowpass(feats: torch.Tensor, rank: int = 8) -> torch.Tensor:
    x = feats.float()
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    rank = max(1, min(int(rank), min(xc.shape) - 1))
    try:
        u, s, v = torch.pca_lowrank(xc, q=rank, center=False, niter=2)
        low = (u[:, :rank] * s[:rank]) @ v[:, :rank].T + mean
    except RuntimeError:
        u, s, vh = torch.linalg.svd(xc, full_matrices=False)
        low = (u[:, :rank] * s[:rank]) @ vh[:rank, :] + mean
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
    low_grid = torch.fft.ifftn(freq * (radius <= cutoff).unsqueeze(0), dim=(-3, -2, -1)).real
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


def _clamp01(value: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    return value.clamp(min=float(lo), max=float(hi))


def choose_target_weights(strategy: dict, alpha: float, low_delta: torch.Tensor, high_delta: torch.Tensor, pca_delta: torch.Tensor):
    target_progress = torch.tensor(1.0 - float(alpha), device=low_delta.device)
    if strategy["kind"] == "fixed":
        low_tgt = torch.tensor(strategy["low_tgt"], device=low_delta.device)
        high_tgt = torch.tensor(strategy["high_tgt"], device=low_delta.device)
    elif strategy["kind"] == "delta_soft":
        base = torch.tensor(strategy.get("base_tgt", 0.5), device=low_delta.device)
        temp = torch.tensor(strategy.get("temp", 0.35), device=low_delta.device)
        low_tgt = base * torch.exp(-strategy.get("low_strength", 1.0) * low_delta / temp)
        high_tgt = base * torch.exp(-strategy.get("high_strength", 0.35) * high_delta / temp)
        high_tgt = high_tgt + strategy.get("high_bonus", 0.15)
    elif strategy["kind"] == "ratio":
        total = (low_delta + high_delta).clamp(min=EPS)
        low_conflict = low_delta / total
        high_conflict = high_delta / total
        low_tgt = strategy.get("base_tgt", 0.5) - strategy.get("span", 0.25) * low_conflict
        high_tgt = strategy.get("base_tgt", 0.5) + strategy.get("span", 0.25) * (1.0 - high_conflict)
    elif strategy["kind"] == "pca_guided":
        temp = torch.tensor(strategy.get("temp", 0.25), device=low_delta.device)
        semantic_conflict = torch.exp(-pca_delta / temp)
        low_tgt = strategy.get("low_min", 0.20) + strategy.get("low_span", 0.35) * semantic_conflict
        high_tgt = strategy.get("high_base", 0.65) - strategy.get("high_penalty", 0.15) * pca_delta.clamp(max=1.0)
    elif strategy["kind"] == "hybrid":
        total = (low_delta + high_delta).clamp(min=EPS)
        spatial_low_conflict = low_delta / total
        semantic_conflict = (1.0 - torch.exp(-pca_delta / strategy.get("pca_temp", 0.25))).clamp(0.0, 1.0)
        low_suppress = strategy.get("low_suppress", 0.32) * spatial_low_conflict * semantic_conflict
        high_boost = strategy.get("high_boost", 0.12) * (1.0 - semantic_conflict)
        low_tgt = target_progress - low_suppress
        high_tgt = target_progress + high_boost
    else:
        raise ValueError(f"Unknown strategy kind: {strategy['kind']}")
    return _clamp01(torch.as_tensor(low_tgt, device=low_delta.device)), _clamp01(torch.as_tensor(high_tgt, device=low_delta.device))


def make_freq_interp(strategy: dict, stats: list):
    method = strategy.get("method", "fft")
    pca_rank = strategy.get("pca_rank", 8)
    fft_grid = strategy.get("fft_grid", 16)
    fft_keep = strategy.get("fft_keep", 0.25)

    def freq_interp(slat1, slat2, alpha: float, mapping_mode="order", interp_mode="linear", unique_flag=True, indices=None):
        if strategy["kind"] == "linear" or slat1.feats.shape != slat2.feats.shape:
            return original_slat_interp(slat1, slat2, alpha, mapping_mode, interp_mode, unique_flag, indices)

        low1, high1 = decompose(slat1.feats, slat1.coords, method, pca_rank, fft_grid, fft_keep)
        low2, high2 = decompose(slat2.feats, slat2.coords, method, pca_rank, fft_grid, fft_keep)

        if method == "pca":
            pca_delta = _relative_delta(low1, low2)
        else:
            pca1 = pca_lowpass(slat1.feats, rank=pca_rank)
            pca2 = pca_lowpass(slat2.feats, rank=pca_rank)
            pca_delta = _relative_delta(pca1, pca2)

        low_delta = _relative_delta(low1, low2)
        high_delta = _relative_delta(high1, high2)
        low_tgt, high_tgt = choose_target_weights(strategy, alpha, low_delta, high_delta, pca_delta)

        feats = (1.0 - low_tgt) * low1 + low_tgt * low2 + (1.0 - high_tgt) * high1 + high_tgt * high2
        stats.append({
            "low_delta": float(low_delta.detach().cpu()),
            "high_delta": float(high_delta.detach().cpu()),
            "pca_delta": float(pca_delta.detach().cpu()),
            "low_tgt": float(low_tgt.detach().cpu()),
            "high_tgt": float(high_tgt.detach().cpu()),
        })
        return slat1.replace(feats.to(dtype=slat1.feats.dtype))

    return freq_interp


def render_four_view(sample, resolution=256):
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitch = [np.deg2rad(20)] * 4
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    out = render_utils.render_frames(sample, extr, intr, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False)
    return np.concatenate(out["color"], axis=1)


def label_row(img, text):
    label_h = 42
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + label_h), "white")
    canvas.paste(Image.fromarray(img), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = None
    draw.text((12, 10), text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def summarize_stats(stats: list):
    if not stats:
        return {}
    keys = stats[0].keys()
    out = {}
    for key in keys:
        vals = np.asarray([row[key] for row in stats], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_min"] = float(vals.min())
        out[f"{key}_max"] = float(vals.max())
    return out


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    out_dir = Path("outputs/diagnostics/goblin_dragon_adaptive_freq_morph10")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    pipeline.cuda()
    src_img = Image.open("./assets/example_morphing/typical_humanoid_goblin.png")
    tar_img = Image.open("./assets/example_morphing/typical_creature_dragon.png")

    common = {
        "morphing_num": 10,
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
        "modify": True,
        "sparse_modify": True,
        "gate_attn": True,
        "sparse_gate_attn": True,
        "gate_mode": "post",
        "modify_mode": "sink_penalty",
        "modify_precheck": True,
        "modify_sink_penalty_type": "log",
        "modify_lambda_scale": 16.0,
        "modify_sink_threshold": 0.02,
        "modify_sink_top_count_weight": 0.5,
        "modify_sink_mass_weight": 0.5,
        "gate_qk_confidence_threshold": 1.0,
        "ot_coherence_enabled": False,
    }

    strategy = {
        "name": "hybrid_fft_pca",
        "kind": "hybrid",
        "method": "fft",
        "pca_temp": 0.25,
        "low_suppress": 0.32,
        "high_boost": 0.12,
        "label": "alpha-aware FFT split + PCA conflict gate",
    }

    rows = []
    rendered_rows = []
    alpha_array = np.linspace(1.0, 0.0, common["morphing_num"])
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for morphing_idx in range(1, common["morphing_num"] - 1):
        alpha = float(alpha_array[morphing_idx])
        name = f"frame_{morphing_idx:02d}_alpha_{alpha:.3f}".replace(".", "p")
        stats = []
        sparse_modulated.slat_interp = make_freq_interp(strategy, stats)
        params = {
            **common,
            "save_cache_path": str(cache_dir),
            "alpha": alpha,
            "morphing_idx": morphing_idx,
            "tfsa_cache_idx": morphing_idx - 1,
        }
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
        img = render_four_view(outputs["gaussian"][0], resolution=256)
        image_path = out_dir / f"{name}.png"
        imageio.imwrite(image_path, img)
        rendered_rows.append(label_row(img, f"{name}: {strategy['label']}"))
        row = {
            "name": name,
            "morphing_idx": morphing_idx,
            "morphing_num": common["morphing_num"],
            "alpha": alpha,
            "label": strategy["label"],
            "image": str(image_path),
            **strategy,
            **summarize_stats(stats),
        }
        rows.append(row)
        print(f"[rendered] {name}")

    sparse_modulated.slat_interp = original_slat_interp
    sheet = np.concatenate(rendered_rows, axis=0)
    imageio.imwrite(out_dir / "comparison_morph10.png", sheet)
    write_csv(out_dir / "summary.csv", rows)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"[done] {out_dir / 'comparison_morph10.png'}")


if __name__ == "__main__":
    main()
