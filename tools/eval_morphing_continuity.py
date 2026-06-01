#!/usr/bin/env python3
"""Training-free morphing continuity metric for rendered videos.

The metric uses only image-processing features: foreground masks, silhouette
edges, and foreground color histograms. It is designed to reward visible,
endpoint-directed change that is spread across the whole video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8
STATIC_EPS = 0.05
MIN_TOTAL_CHANGE = 1e-6


@dataclass(frozen=True)
class DistanceWeights:
    mask: float = 0.4
    edge: float = 0.4
    color: float = 0.2


@dataclass
class FrameFeature:
    mask: np.ndarray
    edge: np.ndarray
    hist: np.ndarray


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def read_video_frames(
    video_path: str, max_frames: Optional[int] = None, resize_width: Optional[int] = 256
) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_indices: Optional[set[int]] = None
    if max_frames is not None and max_frames > 0 and total > max_frames:
        idx = np.linspace(0, total - 1, max_frames)
        sample_indices = set(np.unique(np.rint(idx).astype(np.int64)).tolist())
        sample_indices.add(0)
        sample_indices.add(total - 1)

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if sample_indices is None or frame_idx in sample_indices:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if resize_width is not None and resize_width > 0 and rgb.shape[1] != resize_width:
                scale = resize_width / float(rgb.shape[1])
                new_h = max(1, int(round(rgb.shape[0] * scale)))
                rgb = cv2.resize(rgb, (resize_width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        frame_idx += 1

    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 frames, got {len(frames)} from {video_path}")
    return frames, fps


def split_grid_frames(frames: Sequence[np.ndarray], grid_rows: int, grid_cols: int) -> List[List[np.ndarray]]:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if grid_rows == 1 and grid_cols == 1:
        return [list(frames)]

    views: List[List[np.ndarray]] = [[] for _ in range(grid_rows * grid_cols)]
    for frame in frames:
        h, w = frame.shape[:2]
        cell_h = h // grid_rows
        cell_w = w // grid_cols
        if cell_h <= 0 or cell_w <= 0:
            raise ValueError(f"Frame is too small for {grid_rows}x{grid_cols} grid: {w}x{h}")
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0 = r * cell_h
                x0 = c * cell_w
                y1 = (r + 1) * cell_h if r < grid_rows - 1 else h
                x1 = (c + 1) * cell_w if c < grid_cols - 1 else w
                views[r * grid_cols + c].append(frame[y0:y1, x0:x1].copy())
    return views


def extract_foreground_mask(frame: np.ndarray) -> np.ndarray:
    """Extract foreground from a rendered RGB/RGBA frame with near-white background."""
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA frame, got shape {frame.shape}")

    rgb = frame[..., :3]
    if frame.shape[2] == 4:
        alpha_fg = frame[..., 3] > 8
    else:
        alpha_fg = np.ones(rgb.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    near_white = (min_rgb >= 235) & ((max_rgb - min_rgb) <= 25)
    bright_low_sat = (value >= 245) & (saturation <= 22)
    bg = near_white | bright_low_sat
    mask = (~bg) & alpha_fg

    mask_u8 = (mask.astype(np.uint8) * 255)
    h, w = mask_u8.shape
    k = max(3, int(round(min(h, w) * 0.01)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool)

    image_area = h * w
    min_area = max(16, int(0.0005 * image_area))
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if areas.size else 0
    keep = np.zeros(num_labels, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area >= max(min_area, int(0.02 * largest)):
            keep[label] = True
    return keep[labels]


def extract_edge(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel) > 0
    return edge


def compute_color_hist(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) < 10:
        return np.zeros(32 * 16 * 16, dtype=np.float32)
    hsv = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask.astype(np.uint8), [32, 16, 16], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def extract_features(frame: np.ndarray) -> FrameFeature:
    mask = extract_foreground_mask(frame)
    return FrameFeature(mask=mask, edge=extract_edge(mask), hist=compute_color_hist(frame, mask))


def mask_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return clamp01(1.0 - inter / float(union))


def edge_chamfer_distance(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    a = edge_a.astype(bool)
    b = edge_b.astype(bool)
    count_a = int(a.sum())
    count_b = int(b.sum())
    if count_a == 0 and count_b == 0:
        return 0.0
    if count_a == 0 or count_b == 0:
        return 1.0

    b_obstacles = np.where(b, 0, 255).astype(np.uint8)
    a_obstacles = np.where(a, 0, 255).astype(np.uint8)
    dist_to_b = cv2.distanceTransform(b_obstacles, cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform(a_obstacles, cv2.DIST_L2, 3)
    chamfer = 0.5 * (float(dist_to_b[a].mean()) + float(dist_to_a[b].mean()))
    diag = math.hypot(*a.shape[:2])
    return clamp01(chamfer / (diag + EPS))


def color_hist_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    if float(hist_a.sum()) <= 0 and float(hist_b.sum()) <= 0:
        return 0.0
    diff = hist_a - hist_b
    denom = hist_a + hist_b + EPS
    chi2 = 0.5 * float(np.sum((diff * diff) / denom))
    return clamp01(chi2)


def normalized_weights(weights: DistanceWeights) -> DistanceWeights:
    total = weights.mask + weights.edge + weights.color
    if total <= 0:
        raise ValueError("At least one feature weight must be positive")
    return DistanceWeights(weights.mask / total, weights.edge / total, weights.color / total)


def feature_distance(feat_a: FrameFeature, feat_b: FrameFeature, weights: DistanceWeights) -> float:
    w = normalized_weights(weights)
    return clamp01(
        w.mask * mask_distance(feat_a.mask, feat_b.mask)
        + w.edge * edge_chamfer_distance(feat_a.edge, feat_b.edge)
        + w.color * color_hist_distance(feat_a.hist, feat_b.hist)
    )


def choose_tau(adjacent_d: np.ndarray, tau: Optional[float], auto_tau: bool) -> Tuple[float, str]:
    if tau is not None:
        return max(0.0, float(tau)), "fixed"
    if auto_tau:
        positive = adjacent_d[np.isfinite(adjacent_d)]
        if positive.size == 0:
            return 0.0, "auto_empty"
        median_tau = 0.1 * float(np.median(positive))
        percentile_tau = float(np.percentile(positive, 10))
        if median_tau <= EPS:
            return max(0.0, percentile_tau), "auto_p10"
        if percentile_tau <= EPS:
            return max(0.0, median_tau), "auto_10pct_median"
        return max(0.0, min(median_tau, percentile_tau)), "auto_min_10pct_median_p10"
    return 0.0, "none"


def temporal_participation_from_distances(
    adjacent_d: Sequence[float],
    fps: float = 30.0,
    tau: float = 0.0,
    min_total_change: float = MIN_TOTAL_CHANGE,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    d = np.asarray(adjacent_d, dtype=np.float64)
    if d.size == 0:
        return 0.0, 0.0, d, d
    denoised = np.maximum(d - float(tau), 0.0)
    total = float(denoised.sum())
    if total <= min_total_change:
        return 0.0, 0.0, denoised, np.zeros_like(denoised)

    q = denoised / (total + EPS)
    m = float(d.size)
    p_frame = 1.0 / (m * float(np.sum(q * q)) + EPS)
    p_frame = clamp01(p_frame)

    safe_fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    delta_t = 1.0 / safe_fps
    rates = denoised / delta_t
    duration = m * delta_t
    numerator = float(np.sum(rates * delta_t) ** 2)
    denominator = float(duration * np.sum(rates * rates * delta_t) + EPS)
    p_time = clamp01(numerator / denominator)
    return p_frame, p_time, denoised, q


def monotonic_progress_score(progress: Sequence[float]) -> float:
    p = np.asarray(progress, dtype=np.float64)
    if p.size < 2:
        return 0.0
    delta = np.diff(p)
    backward = float(np.maximum(-delta, 0.0).sum())
    variation = float(np.abs(delta).sum())
    if variation <= EPS:
        return 0.0
    return clamp01(1.0 - backward / (variation + EPS))


def coverage_score(progress: Sequence[float], endpoint_dist: float, static_eps: float = STATIC_EPS) -> float:
    p = np.asarray(progress, dtype=np.float64)
    if p.size < 2:
        return 0.0
    c_raw = clamp01(float(p[-1] - p[0]))
    c_endpoint = clamp01(float(endpoint_dist) / (float(endpoint_dist) + static_eps + EPS))
    return c_raw * c_endpoint


def change_utilization_score(endpoint_dist: float, path_dist: float) -> float:
    return clamp01(float(endpoint_dist) / (float(path_dist) + EPS))


def compute_scores_from_distances(
    d_to_source: Sequence[float],
    d_to_target: Sequence[float],
    adjacent_d: Sequence[float],
    endpoint_dist: float,
    fps: float,
    gamma: float,
    tau: float,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    d_src = np.asarray(d_to_source, dtype=np.float64)
    d_tgt = np.asarray(d_to_target, dtype=np.float64)
    adj = np.asarray(adjacent_d, dtype=np.float64)
    progress = d_src / (d_src + d_tgt + EPS)

    p_frame, p_time, denoised, q = temporal_participation_from_distances(adj, fps=fps, tau=tau)
    monotonicity = monotonic_progress_score(progress)
    path_dist = float(adj.sum())
    coverage = coverage_score(progress, endpoint_dist)
    utilization = change_utilization_score(endpoint_dist, path_dist)
    mcs = clamp01(coverage * monotonicity * (p_time ** float(gamma)) * utilization)

    mean_adj = float(adj.mean()) if adj.size else 0.0
    std_adj = float(adj.std()) if adj.size else 0.0
    metrics = {
        "MCS": mcs,
        "temporal_participation": p_frame,
        "temporal_participation_time": p_time,
        "monotonicity": monotonicity,
        "coverage": coverage,
        "change_utilization": utilization,
        "feature_utilization": utilization,
        "endpoint_distance": float(endpoint_dist),
        "path_distance": path_dist,
        "mean_adjacent_distance": mean_adj,
        "std_adjacent_distance": std_adj,
        "coefficient_of_variation": float(std_adj / (mean_adj + EPS)) if mean_adj > 0 else 0.0,
    }

    frame_data: List[Dict[str, float]] = []
    for i in range(progress.size):
        frame_data.append(
            {
                "frame_idx": float(i),
                "p_t": float(progress[i]),
                "d_to_source": float(d_src[i]),
                "d_to_target": float(d_tgt[i]),
                "adjacent_d_t": float(adj[i]) if i < adj.size else float("nan"),
                "denoised_adjacent_d_t": float(denoised[i]) if i < denoised.size else float("nan"),
                "q_t": float(q[i]) if i < q.size else float("nan"),
            }
        )
    return metrics, frame_data


def compute_single_view_metrics(
    frames: Sequence[np.ndarray], fps: float, args: argparse.Namespace, view_index: int = 0
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[FrameFeature]]:
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for one view")

    weights = DistanceWeights(args.w_mask, args.w_edge, args.w_color)
    features = [extract_features(frame) for frame in frames]
    adjacent_d = np.asarray(
        [feature_distance(features[i], features[i + 1], weights) for i in range(len(features) - 1)],
        dtype=np.float64,
    )
    tau, tau_method = choose_tau(adjacent_d, args.tau, args.auto_tau)
    d_to_source = [feature_distance(features[i], features[0], weights) for i in range(len(features))]
    d_to_target = [feature_distance(features[i], features[-1], weights) for i in range(len(features))]
    endpoint_dist = feature_distance(features[0], features[-1], weights)

    metrics, frame_data = compute_scores_from_distances(
        d_to_source=d_to_source,
        d_to_target=d_to_target,
        adjacent_d=adjacent_d,
        endpoint_dist=endpoint_dist,
        fps=fps,
        gamma=args.gamma,
        tau=tau,
    )
    metrics.update(
        {
            "view_index": view_index,
            "num_frames": len(frames),
            "tau": tau,
            "tau_method": tau_method,
            "weights": {
                "mask": normalized_weights(weights).mask,
                "edge": normalized_weights(weights).edge,
                "color": normalized_weights(weights).color,
            },
        }
    )
    return metrics, frame_data, features


def mean_metric_dict(metric_dicts: Sequence[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(m[key]) for m in metric_dicts if key in m and isinstance(m[key], (int, float))]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def compute_video_metrics(
    frames: Sequence[np.ndarray], fps: float, args: argparse.Namespace
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[np.ndarray], List[FrameFeature]]:
    views = split_grid_frames(frames, args.grid_rows, args.grid_cols)
    per_view_metrics: List[Dict[str, Any]] = []
    per_view_frame_data: List[List[Dict[str, float]]] = []
    first_features: List[FrameFeature] = []

    for view_idx, view_frames in enumerate(views):
        metrics, frame_data, features = compute_single_view_metrics(view_frames, fps, args, view_index=view_idx)
        per_view_metrics.append(metrics)
        per_view_frame_data.append(frame_data)
        if view_idx == 0:
            first_features = features

    aggregate_keys = [
        "MCS",
        "temporal_participation",
        "temporal_participation_time",
        "monotonicity",
        "coverage",
        "change_utilization",
        "feature_utilization",
        "endpoint_distance",
        "path_distance",
        "mean_adjacent_distance",
        "std_adjacent_distance",
        "coefficient_of_variation",
        "tau",
    ]
    metrics = mean_metric_dict(per_view_metrics, aggregate_keys)
    metrics.update(
        {
            "video": args.video,
            "fps": float(fps),
            "num_frames": len(frames),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "gamma": float(args.gamma),
            "weights": per_view_metrics[0]["weights"],
        }
    )
    if len(per_view_metrics) > 1:
        metrics["per_view_metrics"] = per_view_metrics

    frame_data = per_view_frame_data[0]
    if len(per_view_frame_data) > 1:
        for idx, rows in enumerate(frame_data):
            for view_idx, view_rows in enumerate(per_view_frame_data):
                rows[f"view{view_idx}_p_t"] = view_rows[idx]["p_t"]
                rows[f"view{view_idx}_adjacent_d_t"] = view_rows[idx]["adjacent_d_t"]

    debug_frames = views[0]
    return metrics, frame_data, debug_frames, first_features


def save_metrics_json(metrics: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_frame_csv(frame_data: Sequence[Dict[str, float]], path: str | Path) -> None:
    if not frame_data:
        return
    fieldnames: List[str] = []
    for row in frame_data:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_data:
            writer.writerow(row)


def save_line_plot(values: Sequence[float], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(np.asarray(values, dtype=np.float64), marker="o", linewidth=1.5, markersize=3)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_contact_sheet(images: Sequence[np.ndarray], cols: int = 5, pad: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = int(math.ceil(len(images) / float(cols)))
    sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        if img.ndim == 2:
            img_rgb = np.repeat(img[..., None], 3, axis=2)
        else:
            img_rgb = img[..., :3]
        canvas[: img_rgb.shape[0], : img_rgb.shape[1]] = img_rgb
        sheet[y : y + h, x : x + w] = canvas
    return sheet


def save_debug_plots(
    out_dir: str | Path,
    frame_data: Sequence[Dict[str, float]],
    frames: Sequence[np.ndarray],
    features: Sequence[FrameFeature],
) -> None:
    out = Path(out_dir)
    p = [row["p_t"] for row in frame_data]
    d = [row["adjacent_d_t"] for row in frame_data[:-1]]
    q = [row["q_t"] for row in frame_data[:-1]]
    save_line_plot(p, "Endpoint-relative progress", "p_t", out / "progress_curve.png")
    save_line_plot(d, "Adjacent feature distance", "D(F_t, F_{t+1})", out / "adjacent_distance.png")

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(q)), q)
    plt.title("Temporal participation q_t")
    plt.xlabel("Interval index")
    plt.ylabel("q_t")
    plt.tight_layout()
    plt.savefig(out / "q_distribution.png", dpi=160)
    plt.close()

    sample_n = min(10, len(frames))
    sample_idx = np.unique(np.rint(np.linspace(0, len(frames) - 1, sample_n)).astype(int))
    mask_imgs: List[np.ndarray] = []
    edge_imgs: List[np.ndarray] = []
    for idx in sample_idx:
        frame = frames[int(idx)]
        mask = features[int(idx)].mask
        edge = features[int(idx)].edge
        mask_overlay = frame.copy()
        mask_overlay[~mask] = (0.65 * mask_overlay[~mask] + 0.35 * 255).astype(np.uint8)
        mask_imgs.append(mask_overlay)
        edge_img = np.zeros_like(frame)
        edge_img[edge] = np.array([255, 40, 40], dtype=np.uint8)
        edge_imgs.append(edge_img)

    cv2.imwrite(str(out / "masks_contact_sheet.jpg"), cv2.cvtColor(make_contact_sheet(mask_imgs), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "edges_contact_sheet.jpg"), cv2.cvtColor(make_contact_sheet(edge_imgs), cv2.COLOR_RGB2BGR))


def run_synthetic_tests() -> None:
    fps = 30.0
    n = 11
    uniform = np.ones(n - 1)
    p, p_time, _, _ = temporal_participation_from_distances(uniform, fps=fps, tau=0.0)
    assert abs(p - 1.0) < 1e-6, p
    assert abs(p_time - 1.0) < 1e-6, p_time

    single = np.zeros(n - 1)
    single[4] = 1.0
    p, p_time, _, _ = temporal_participation_from_distances(single, fps=fps, tau=0.0)
    expected = 1.0 / (n - 1)
    assert abs(p - expected) < 1e-6, (p, expected)
    assert abs(p_time - expected) < 1e-6, (p_time, expected)

    d_src = np.zeros(n)
    d_tgt = np.zeros(n)
    adj = np.zeros(n - 1)
    metrics, _ = compute_scores_from_distances(d_src, d_tgt, adj, endpoint_dist=0.0, fps=fps, gamma=1.0, tau=0.0)
    assert metrics["MCS"] < 1e-6, metrics["MCS"]

    progress = np.linspace(0.0, 1.0, n)
    d_src = progress
    d_tgt = 1.0 - progress
    jitter_adj = np.ones(n - 1) * 0.4
    metrics, _ = compute_scores_from_distances(
        d_src, d_tgt, jitter_adj, endpoint_dist=0.2, fps=fps, gamma=1.0, tau=0.0
    )
    assert metrics["change_utilization"] < 0.1, metrics["change_utilization"]
    print("Synthetic tests passed.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-free Morphing Continuity Score (MCS).")
    parser.add_argument("--video", required=False, help="Input video path.")
    parser.add_argument("--out_dir", required=False, help="Directory for metrics and debug outputs.")
    parser.add_argument("--grid_rows", type=int, default=1)
    parser.add_argument("--grid_cols", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--auto_tau", action="store_true")
    parser.add_argument("--w_mask", type=float, default=0.4)
    parser.add_argument("--w_edge", type=float, default=0.4)
    parser.add_argument("--w_color", type=float, default=0.2)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize_width", type=int, default=256)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--run_tests", action="store_true", help="Run synthetic math tests and exit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.run_tests:
        run_synthetic_tests()
        return
    if not args.video or not args.out_dir:
        parser.error("--video and --out_dir are required unless --run_tests is used")

    os.makedirs(args.out_dir, exist_ok=True)
    frames, fps = read_video_frames(args.video, max_frames=args.max_frames, resize_width=args.resize_width)
    metrics, frame_data, debug_frames, debug_features = compute_video_metrics(frames, fps, args)

    save_metrics_json(metrics, Path(args.out_dir) / "metrics.json")
    save_frame_csv(frame_data, Path(args.out_dir) / "frame_metrics.csv")
    if args.save_debug:
        save_debug_plots(args.out_dir, frame_data, debug_frames, debug_features)

    print(f"Video: {args.video}")
    print(f"Frames: {len(frames)}, FPS: {fps:.3f}")
    print(f"MCS: {metrics['MCS']:.6f}")
    print(f"Temporal Participation: {metrics['temporal_participation_time']:.6f}")
    print(f"Monotonicity: {metrics['monotonicity']:.6f}")
    print(f"Coverage: {metrics['coverage']:.6f}")
    print(f"Change Utilization: {metrics['change_utilization']:.6f}")


if __name__ == "__main__":
    main()
