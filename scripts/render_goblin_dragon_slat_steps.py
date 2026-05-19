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

from trellis.modules import sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import cal_eucdist_matrix, feature_interp, seed_everything


def render_one_view(sample, resolution=320):
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.0], [np.deg2rad(20)], 2, 40)
    out = render_utils.render_frames(
        sample,
        extr,
        intr,
        {"resolution": resolution, "bg_color": (1, 1, 1)},
        verbose=False,
    )
    return out["color"][0]


def add_label(img, text):
    label_h = 42
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + label_h), "white")
    canvas.paste(Image.fromarray(img), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = None
    draw.text((8, 10), text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    out_dir = Path("/tmp/morphany3d_diagnostics/goblin_dragon_slat_steps")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    pipeline.cuda()

    src_img = Image.open("./assets/example_morphing/typical_humanoid_goblin.png")
    tar_img = Image.open("./assets/example_morphing/typical_creature_dragon.png")
    src_cond = pipeline.get_cond([pipeline.preprocess_image(src_img)])
    tar_cond = pipeline.get_cond([pipeline.preprocess_image(tar_img)])

    params = {
        "morphing_num": 3,
        "src_load_cache_path": "./outputs/cache/typical_humanoid_goblin/cache",
        "tar_load_cache_path": "./outputs/cache/typical_creature_dragon/cache",
        "save_cache_path": str(out_dir / "cache"),
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
        "tar_cond": tar_cond["cond"],
        "sa_use": False,
        "modify": False,
        "gate_attn": True,
        "gate_mode": "post",
        "ot_coherence_enabled": False,
    }
    Path(params["save_cache_path"]).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        coords, _, _ = pipeline.sample_sparse_structure_morphing(
            src_cond,
            num_samples=1,
            sampler_params={"steps": 4},
            morphing_params=params,
        )

    flow_model = pipeline.models["slat_flow_model"]
    src_noise = torch.load(Path(params["src_load_cache_path"]) / "slat_init.pt").to(pipeline.device)
    src_coords = torch.load(Path(params["src_load_cache_path"]) / "coords.pt").to(pipeline.device)
    tar_noise = torch.load(Path(params["tar_load_cache_path"]) / "slat_init.pt").to(pipeline.device)
    tar_coords = torch.load(Path(params["tar_load_cache_path"]) / "coords.pt").to(pipeline.device)
    src_dist = cal_eucdist_matrix(coords[:, 1:].detach().float(), src_coords[:, 1:].float())
    tar_dist = cal_eucdist_matrix(coords[:, 1:].detach().float(), tar_coords[:, 1:].float())
    src_indices = torch.argmin(src_dist, dim=1)
    tar_indices = torch.argmin(tar_dist, dim=1)
    feat_noise = feature_interp(src_noise[src_indices], tar_noise[tar_indices], params["alpha"])
    noise = sp.SparseTensor(feats=feat_noise, coords=coords)

    sampler_params = {**pipeline.slat_sampler_params, "steps": 8}
    with torch.no_grad():
        ret = pipeline.slat_sampler.sample(
            flow_model,
            noise,
            **src_cond,
            **sampler_params,
            **params,
            verbose=True,
        )

    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
    frames = []
    rows = []
    prev_feats = None
    for idx, sample in enumerate(ret.pred_x_t, start=1):
        slat = sample.replace(sample.feats * std + mean)
        with torch.no_grad():
            outputs = pipeline.decode_slat(slat, formats=["gaussian"])
            img = render_one_view(outputs["gaussian"][0], resolution=320)
        feats = slat.feats.detach().float()
        delta = 0.0 if prev_feats is None else float((feats - prev_feats).norm(dim=-1).mean().item())
        prev_feats = feats
        row = {
            "step": idx,
            "feat_mean": float(feats.mean().item()),
            "feat_std": float(feats.std(unbiased=False).item()),
            "feat_norm_mean": float(feats.norm(dim=-1).mean().item()),
            "delta_from_prev": delta,
        }
        rows.append(row)
        labeled = add_label(img, f"SLAT step {idx:02d}/08  delta={delta:.3f}  norm={row['feat_norm_mean']:.3f}")
        imageio.imwrite(out_dir / f"slat_step_{idx:02d}.png", labeled)
        frames.append(labeled)
        print(f"[rendered] step={idx} delta={delta:.4f}")

    sheet = np.concatenate(frames, axis=1)
    imageio.imwrite(out_dir / "slat_steps_sheet.png", sheet)
    imageio.mimsave(out_dir / "slat_steps.gif", frames, fps=2)
    write_csv(out_dir / "slat_step_metrics.csv", rows)
    print(f"[done] {out_dir / 'slat_steps_sheet.png'}")


if __name__ == "__main__":
    main()
