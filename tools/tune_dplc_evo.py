#!/usr/bin/env python3
"""Hyperparameter search helper for DPLC_EVO morphing.

The script keeps the source/target caches fixed, runs DPLC_EVO variants, then
scores each rendered video with SG-TGMCS. It is intended for coarse-to-fine
tuning: run a small coarse search, inspect the ranked CSV and previews, then
run a local search around the best candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from PIL import Image

from tools.eval_geometric_morphing_continuity import evaluate_video_to_dir

if TYPE_CHECKING:
    from trellis.pipelines import TrellisImageTo3DPipeline


BASE_METHODS = ["CGAR", "CA_OC", "DPLC_EVO"]


COARSE_CANDIDATES: List[Dict[str, Any]] = [
    {
        "name": "baseline_current",
        "params": {},
    },
    {
        "name": "latest_balanced_smooth",
        "params": {
            "dplc_lam": 0.12,
            "dplc_residual_quantile": 0.82,
            "dplc_max_delta_ratio": 0.12,
            "dplc_evo_start": 0.32,
            "dplc_evo_strength": 0.16,
            "dplc_evo_slow_strength": 0.06,
            "dplc_evo_max_shift": 0.18,
            "dplc_evo_min_neighbors": 1,
            "dplc_evo_smooth_iters": 1,
            "dplc_evo_smooth_weight": 0.30,
            "dplc_evo_trend_bonus": 0.10,
            "dplc_evo_support_floor": 0.70,
            "dplc_evo_history_floor": 0.55,
        },
    },
    {
        "name": "latest_conservative",
        "params": {
            "dplc_lam": 0.10,
            "dplc_residual_quantile": 0.85,
            "dplc_max_delta_ratio": 0.10,
            "dplc_evo_start": 0.40,
            "dplc_evo_strength": 0.12,
            "dplc_evo_slow_strength": 0.05,
            "dplc_evo_max_shift": 0.14,
            "dplc_evo_min_neighbors": 2,
            "dplc_evo_smooth_iters": 1,
            "dplc_evo_smooth_weight": 0.25,
            "dplc_evo_support_floor": 0.75,
            "dplc_evo_history_floor": 0.60,
        },
    },
    {
        "name": "latest_stronger_birth_death",
        "params": {
            "dplc_lam": 0.16,
            "dplc_residual_quantile": 0.78,
            "dplc_max_delta_ratio": 0.14,
            "dplc_evo_start": 0.28,
            "dplc_evo_strength": 0.22,
            "dplc_evo_slow_strength": 0.08,
            "dplc_evo_max_shift": 0.22,
            "dplc_evo_min_neighbors": 1,
            "dplc_evo_smooth_iters": 1,
            "dplc_evo_smooth_weight": 0.35,
            "dplc_evo_trend_bonus": 0.15,
            "dplc_evo_support_floor": 0.65,
            "dplc_evo_history_floor": 0.50,
        },
    },
    {
        "name": "latest_delayed_stable",
        "params": {
            "dplc_lam": 0.12,
            "dplc_residual_quantile": 0.84,
            "dplc_max_delta_ratio": 0.12,
            "dplc_evo_start": 0.45,
            "dplc_evo_strength": 0.18,
            "dplc_evo_slow_strength": 0.04,
            "dplc_evo_max_shift": 0.18,
            "dplc_evo_min_neighbors": 2,
            "dplc_evo_smooth_iters": 2,
            "dplc_evo_smooth_weight": 0.25,
            "dplc_evo_support_floor": 0.80,
            "dplc_evo_history_floor": 0.60,
        },
    },
    {
        "name": "latest_early_motion",
        "params": {
            "dplc_lam": 0.14,
            "dplc_residual_quantile": 0.80,
            "dplc_max_delta_ratio": 0.13,
            "dplc_evo_start": 0.22,
            "dplc_evo_strength": 0.15,
            "dplc_evo_slow_strength": 0.10,
            "dplc_evo_max_shift": 0.18,
            "dplc_evo_min_neighbors": 1,
            "dplc_evo_smooth_iters": 1,
            "dplc_evo_smooth_weight": 0.40,
            "dplc_evo_trend_bonus": 0.20,
            "dplc_evo_support_floor": 0.70,
            "dplc_evo_history_floor": 0.55,
        },
    },
]


LOCAL_KEYS = {
    "dplc_lam": [-0.03, 0.0, 0.03],
    "dplc_residual_quantile": [-0.03, 0.0, 0.03],
    "dplc_max_delta_ratio": [-0.03, 0.0, 0.03],
    "dplc_evo_start": [-0.08, 0.0, 0.08],
    "dplc_evo_strength": [-0.04, 0.0, 0.04],
    "dplc_evo_slow_strength": [-0.03, 0.0, 0.03],
    "dplc_evo_max_shift": [-0.04, 0.0, 0.04],
    "dplc_evo_smooth_weight": [-0.15, 0.0, 0.15],
    "dplc_evo_support_floor": [-0.10, 0.0, 0.10],
    "dplc_evo_history_floor": [-0.10, 0.0, 0.10],
}


def clamp_param(key: str, value: float) -> float:
    if key in {"dplc_residual_quantile", "dplc_evo_start", "dplc_evo_smooth_weight", "dplc_evo_support_floor", "dplc_evo_history_floor"}:
        return round(min(1.0, max(0.0, value)), 4)
    if key in {"dplc_lam", "dplc_max_delta_ratio", "dplc_evo_strength", "dplc_evo_slow_strength", "dplc_evo_max_shift"}:
        return round(max(0.0, value), 4)
    return round(value, 4)


def local_candidates(center: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    base = deepcopy(center)
    candidates = [{"name": "local_center", "params": base}]
    for key, deltas in LOCAL_KEYS.items():
        if key not in base:
            continue
        for delta in deltas:
            if delta == 0.0:
                continue
            params = deepcopy(base)
            params[key] = clamp_param(key, float(params[key]) + delta)
            suffix = "p" if delta > 0 else "m"
            candidates.append({"name": f"local_{key}_{suffix}{abs(delta):.3f}".replace(".", "p"), "params": params})
            if len(candidates) >= limit:
                return candidates
    return candidates[:limit]


def candidate_by_name(name: str) -> Dict[str, Any]:
    for candidate in COARSE_CANDIDATES:
        if candidate["name"] == name:
            return deepcopy(candidate["params"])
    raise ValueError(f"Unknown center candidate: {name}")


def build_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        video=None,
        video_dir=None,
        glob="*.mp4",
        out_dir=None,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        resize_width=args.resize_width,
        max_frames=args.max_eval_frames,
        save_debug=args.save_eval_debug,
        save_resampled_debug=False,
        eval_fps=args.eval_fps,
        resample_mode="hold",
        use_time_aware=True,
        spread_alpha=4.0,
        time_score_mode="sqrt_product",
        freeze_grace=0.12,
        repeat_grace=0.05,
        static_lambda=1.0,
        use_static_gate=True,
        tau_mode="percentile",
        tau_percentile=10.0,
        tau_median_ratio=0.1,
        tau_fixed=0.0,
        active_beta=0.35,
        lambda_smooth=1.0,
        bg_threshold=245,
        min_component_ratio=0.001,
        w_mask=0.25,
        w_contour=0.35,
        w_skeleton=0.20,
        w_radial=0.10,
        w_hu=0.10,
        run_tests=False,
    )


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (dict, list)):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def ensure_endpoint_cache(
    pipeline: "TrellisImageTo3DPipeline",
    src_img: Image.Image,
    tar_img: Image.Image,
    src_name: str,
    tar_name: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    from trellis.utils.morphing_utils import run_morphing_cache

    src_save_path = Path(args.out_root) / "endpoint_cache" / src_name
    tar_save_path = Path(args.out_root) / "endpoint_cache" / tar_name
    src_cache = src_save_path / "cache"
    tar_cache = tar_save_path / "cache"
    src_cache.mkdir(parents=True, exist_ok=True)
    tar_cache.mkdir(parents=True, exist_ok=True)
    cache_params = {
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "slat_mca_flag": False,
        "ss_tfsa_flag": False,
        "slat_tfsa_flag": False,
        "oc_flag": False,
    }
    if not (src_cache / "slat_init.pt").exists() or args.rebuild_endpoint_cache:
        params = dict(cache_params, save_cache_path=str(src_cache))
        run_morphing_cache(pipeline, src_img, tar_img, params, args.seed, str(src_save_path), src_name)
    if not (tar_cache / "slat_init.pt").exists() or args.rebuild_endpoint_cache:
        params = dict(cache_params, save_cache_path=str(tar_cache))
        run_morphing_cache(pipeline, tar_img, src_img, params, args.seed, str(tar_save_path), tar_name)
    return src_cache, tar_cache


def run_candidate(
    pipeline: "TrellisImageTo3DPipeline",
    src_img: Image.Image,
    tar_img: Image.Image,
    src_cache: Path,
    tar_cache: Path,
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    eval_args: argparse.Namespace,
) -> Dict[str, Any]:
    from trellis.utils.morphing_args import build_morphing_params
    from trellis.utils.morphing_utils import run_morphing

    run_name = candidate["name"]
    out_dir = Path(args.out_root) / args.search / run_name
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "morphing_num": args.morphing_num,
        "ss_steps": args.ss_steps,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "save_cache_path": str(cache_dir),
        "save_coords_cache": True,
    }
    overrides.update(candidate["params"])
    morphing_params = build_morphing_params(BASE_METHODS, overrides=overrides)
    with (out_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(morphing_params, f, indent=2, sort_keys=True)

    result_name = f"{args.src_name}+{args.tar_name}_{run_name}"
    run_morphing(pipeline, src_img, tar_img, morphing_params, args.seed, str(out_dir), result_name)

    video_path = out_dir / f"morphing_{result_name}.mp4"
    metrics = evaluate_video_to_dir(str(video_path), out_dir / "tgmcs", eval_args)
    row = {
        "name": run_name,
        "video": str(video_path),
        "T_GMCS_static": metrics.get("T_GMCS_static"),
        "T_GMCS": metrics.get("T_GMCS"),
        "F_static": metrics.get("F_static"),
        "freeze_ratio": metrics.get("freeze_ratio"),
        "repeated_frame_ratio": metrics.get("repeated_frame_ratio"),
        "G_MCS_resampled": metrics.get("G_MCS_resampled"),
    }
    row.update(candidate["params"])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DPLC_EVO hyperparameters with SG-TGMCS ranking.")
    parser.add_argument("--src", default="./assets/example_morphing/typical_humanoid_goblin.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_creature_dragon.png")
    parser.add_argument("--model-path", default="./TRELLIS-image-large")
    parser.add_argument("--out-root", default="./outputs/dplc_evo_tuning")
    parser.add_argument("--search", choices=["coarse", "local"], default="coarse")
    parser.add_argument("--center-name", default="latest_balanced_smooth")
    parser.add_argument("--center-json", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--morphing-num", type=int, default=15)
    parser.add_argument("--ss-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-eval-frames", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--eval-fps", type=float, default=24.0)
    parser.add_argument("--grid-rows", type=int, default=1)
    parser.add_argument("--grid-cols", type=int, default=1)
    parser.add_argument("--save-eval-debug", action="store_true")
    parser.add_argument("--rebuild-endpoint-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.src_name = Path(args.src).stem
    args.tar_name = Path(args.tar).stem
    return args


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if args.search == "coarse":
        candidates = deepcopy(COARSE_CANDIDATES[: args.limit])
    else:
        if args.center_json:
            with open(args.center_json, "r", encoding="utf-8") as f:
                center = json.load(f)
        else:
            center = candidate_by_name(args.center_name)
        candidates = local_candidates(center, args.limit)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    with (Path(args.out_root) / f"{args.search}_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, sort_keys=True)

    if args.dry_run:
        print(json.dumps(candidates, indent=2, sort_keys=True))
        return

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    src_cache, tar_cache = ensure_endpoint_cache(pipeline, src_img, tar_img, args.src_name, args.tar_name, args)
    eval_args = build_eval_args(args)

    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[DPLC_EVO tune] {idx}/{len(candidates)} {candidate['name']}")
        row = run_candidate(pipeline, src_img, tar_img, src_cache, tar_cache, candidate, args, eval_args)
        rows.append(row)
        rows_sorted = sorted(rows, key=lambda item: float(item.get("T_GMCS_static") or 0.0), reverse=True)
        write_rows(Path(args.out_root) / f"{args.search}_summary.csv", rows_sorted)
        print(f"  T_GMCS_static={float(row.get('T_GMCS_static') or 0.0):.6f}")

    best = sorted(rows, key=lambda item: float(item.get("T_GMCS_static") or 0.0), reverse=True)[0]
    with (Path(args.out_root) / f"{args.search}_best.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)
    print(f"[DPLC_EVO tune] best={best['name']} T_GMCS_static={float(best.get('T_GMCS_static') or 0.0):.6f}")


if __name__ == "__main__":
    main()
