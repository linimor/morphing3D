#!/usr/bin/env python3
"""Render alpha=0.5 multi-view previews for DPLC_EVO hyperparameter candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from tools.tune_dplc_evo import COARSE_CANDIDATES, candidate_by_name, local_candidates
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_args import build_morphing_params
from trellis.utils.morphing_utils import save_video, seed_everything
from trellis.utils import render_utils


BASE_METHODS = ["CGAR", "CA_OC", "DPLC_EVO"]


def save_contact_sheet(path: Path, frames: np.ndarray, view_indices: List[int]) -> None:
    selected = np.asarray(frames)[view_indices]
    if selected.dtype != np.uint8:
        if selected.size and float(np.nanmax(selected)) <= 1.0:
            selected = np.clip(selected * 255.0, 0, 255).astype(np.uint8)
        else:
            selected = np.clip(selected, 0, 255).astype(np.uint8)
    if selected.shape[-1] == 4:
        selected = selected[..., :3]
    sheet = np.concatenate(selected, axis=1)
    Image.fromarray(sheet).save(path)


def ensure_endpoint_cache(
    pipeline: TrellisImageTo3DPipeline,
    src_img: Image.Image,
    tar_img: Image.Image,
    src_name: str,
    tar_name: str,
    out_root: Path,
    seed: int,
    rebuild: bool,
) -> tuple[Path, Path]:
    src_cache = out_root / "endpoint_cache" / src_name / "cache"
    tar_cache = out_root / "endpoint_cache" / tar_name / "cache"
    src_cache.mkdir(parents=True, exist_ok=True)
    tar_cache.mkdir(parents=True, exist_ok=True)

    def build_cache(image_a: Image.Image, image_b: Image.Image, cache_dir: Path) -> None:
        required = ["coords.pt", "coords_zs.pt", "coords_zs_init.pt", "slat_init.pt"]
        if not rebuild and all((cache_dir / name).exists() for name in required):
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
            pipeline.run_morphing(src_img=image_a, tar_img=image_b, morphing_params=params, seed=seed)

    build_cache(src_img, tar_img, src_cache)
    build_cache(tar_img, src_img, tar_cache)
    return src_cache, tar_cache


def run_midpoint_candidate(
    pipeline: TrellisImageTo3DPipeline,
    src_img: Image.Image,
    tar_img: Image.Image,
    src_cache: Path,
    tar_cache: Path,
    candidate: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / "midpoint" / candidate["name"]
    cache_dir = out_dir / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    middle_idx = args.middle_index
    if middle_idx is None:
        middle_idx = args.morphing_num // 2
    middle_idx = max(1, min(int(middle_idx), args.morphing_num - 2))
    alpha_array = np.linspace(1.0, 0.0, args.morphing_num)

    overrides = {
        "morphing_num": args.morphing_num,
        "ss_steps": args.ss_steps,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "save_cache_path": str(cache_dir),
        "save_coords_cache": True,
        "rm_cache": False,
        "return_intermediate": True,
    }
    overrides.update(candidate["params"])
    morphing_params = build_morphing_params(BASE_METHODS, overrides=overrides)
    with (out_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(morphing_params, f, indent=2, sort_keys=True)

    outputs = None
    for morphing_idx in range(1, middle_idx + 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        morphing_params["tfsa_alpha"] = args.tfsa_alpha
        morphing_params["save_current_tfsa_cache"] = morphing_idx < middle_idx
        with torch.no_grad():
            outputs = pipeline.run_morphing(
                src_img=src_img,
                tar_img=tar_img,
                morphing_params=morphing_params,
                seed=args.seed,
            )

    if outputs is None:
        raise RuntimeError("No midpoint output was generated")

    color_video = np.stack(render_utils.render_rot_video(outputs["gaussian"][0], bg_color=(1, 1, 1))["color"], axis=0)
    normal_video = np.stack(render_utils.render_rot_video(outputs["mesh"][0], bg_color=(1, 1, 1))["normal"], axis=0)
    view_indices = [idx for idx in args.view_indices if 0 <= idx < color_video.shape[0]]
    if not view_indices:
        view_indices = [0, color_video.shape[0] // 2]

    alpha_tag = f"{float(alpha_array[middle_idx]):.3f}".replace(".", "p")
    color_sheet = out_dir / f"alpha{alpha_tag}_views.png"
    normal_sheet = out_dir / f"alpha{alpha_tag}_normal_views.png"
    save_contact_sheet(color_sheet, color_video, view_indices)
    save_contact_sheet(normal_sheet, normal_video, view_indices)

    if args.save_spin:
        save_video(str(out_dir / f"alpha{alpha_tag}_spin.mp4"), color_video, fps=30)
        save_video(str(out_dir / f"alpha{alpha_tag}_normal_spin.mp4"), normal_video, fps=30)

    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return {
        "name": candidate["name"],
        "alpha": float(alpha_array[middle_idx]),
        "middle_idx": middle_idx,
        "color_views": str(color_sheet),
        "normal_views": str(normal_sheet),
        **candidate["params"],
    }


def build_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.search == "coarse":
        return deepcopy(COARSE_CANDIDATES[: args.limit])
    if args.center_json:
        with open(args.center_json, "r", encoding="utf-8") as f:
            center = json.load(f)
    else:
        center = candidate_by_name(args.center_name)
    return local_candidates(center, args.limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DPLC_EVO alpha=0.5 candidate previews.")
    parser.add_argument("--src", default="./assets/example_morphing/typical_humanoid_goblin.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_creature_dragon.png")
    parser.add_argument("--model-path", default="./TRELLIS-image-large")
    parser.add_argument("--out-root", default="./outputs/dplc_evo_midpoint")
    parser.add_argument("--search", choices=["coarse", "local"], default="coarse")
    parser.add_argument("--center-name", default="latest_balanced_smooth")
    parser.add_argument("--center-json", default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--morphing-num", type=int, default=7)
    parser.add_argument("--middle-index", type=int, default=None)
    parser.add_argument("--ss-steps", type=int, default=15)
    parser.add_argument("--tfsa-alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--view-indices", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--save-spin", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--rebuild-endpoint-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    src_name = Path(args.src).stem
    tar_name = Path(args.tar).stem

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    src_cache, tar_cache = ensure_endpoint_cache(
        pipeline,
        src_img,
        tar_img,
        src_name,
        tar_name,
        out_root,
        args.seed,
        args.rebuild_endpoint_cache,
    )

    rows = []
    candidates = build_candidates(args)
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[DPLC_EVO midpoint] {idx}/{len(candidates)} {candidate['name']}")
        rows.append(run_midpoint_candidate(pipeline, src_img, tar_img, src_cache, tar_cache, candidate, args))
        with (out_root / "midpoint_summary.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, sort_keys=True)

    print(f"[DPLC_EVO midpoint] wrote previews under {out_root / 'midpoint'}")


if __name__ == "__main__":
    main()
