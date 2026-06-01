#!/usr/bin/env python3
"""SG-TGMCS: Static-Gated Time-aware Geometric Morphing Continuity Score.

The final score is reported as ``T_GMCS_static`` in machine-readable outputs.
It evaluates rendered morphing videos with only classical image processing:
foreground masks, contours, skeletons, radial signatures, and Hu moments.
Appearance/color is diagnostic only and is not used in the main score.
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
NEARLY_STATIC_EPS = 1e-6


@dataclass(frozen=True)
class GeometryWeights:
    mask: float = 0.25
    contour: float = 0.35
    skeleton: float = 0.20
    radial: float = 0.10
    hu: float = 0.10


@dataclass
class GeometryFeature:
    mask: np.ndarray
    edge: np.ndarray
    skeleton: np.ndarray
    radial: np.ndarray
    hu: np.ndarray
    app_hist: np.ndarray
    skeleton_available: bool


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = 256,
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


def extract_foreground_mask(
    frame: np.ndarray,
    bg_threshold: int = 245,
    min_component_ratio: float = 0.001,
) -> np.ndarray:
    rgb = frame[..., :3]
    threshold = int(bg_threshold)
    strict_white = np.all(rgb > threshold, axis=2)
    relaxed_white = (rgb.min(axis=2) > max(0, threshold - 10)) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30)
    mask_u8 = ((~(strict_white | relaxed_white)).astype(np.uint8) * 255)

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

    min_area = max(8, int(float(min_component_ratio) * h * w))
    keep = np.zeros(num_labels, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep[label] = True
    return keep[labels]


def extract_silhouette_edge(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel) > 0


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    img = (mask.astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel > 0


def extract_skeleton(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=bool), True
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            thin = cv2.ximgproc.thinning((mask.astype(np.uint8) * 255))
            return thin > 0, True
        return morphological_skeleton(mask), True
    except Exception:
        return np.zeros_like(mask, dtype=bool), False


def extract_radial_signature(mask: np.ndarray, num_angles: int = 64) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.zeros(num_angles, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    h, w = mask.shape
    diag = math.hypot(h, w) + EPS
    max_r = math.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))
    radii = np.zeros(num_angles, dtype=np.float32)
    for i, theta in enumerate(np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False)):
        dx = math.cos(theta)
        dy = math.sin(theta)
        last_r = 0.0
        for r in np.linspace(0.0, max_r, max(32, int(max_r) + 1)):
            x = int(round(cx + dx * r))
            y = int(round(cy + dy * r))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask[y, x]:
                last_r = float(r)
        radii[i] = last_r / diag
    return radii


def extract_hu_moments(mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.zeros(7, dtype=np.float32)
    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).reshape(-1)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + EPS)
    return hu_log.astype(np.float32)


def extract_appearance_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) < 10:
        return np.zeros(32 * 16 * 16, dtype=np.float32)
    hsv = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask.astype(np.uint8), [32, 16, 16], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def extract_geometry_features(frame: np.ndarray, args: argparse.Namespace) -> GeometryFeature:
    mask = extract_foreground_mask(frame, args.bg_threshold, args.min_component_ratio)
    edge = extract_silhouette_edge(mask)
    skeleton, skeleton_available = extract_skeleton(mask)
    radial = extract_radial_signature(mask)
    hu = extract_hu_moments(mask)
    app_hist = extract_appearance_histogram(frame, mask)
    return GeometryFeature(mask, edge, skeleton, radial, hu, app_hist, skeleton_available)


def mask_iou_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return clamp01(1.0 - int(np.logical_and(a, b).sum()) / float(union))


def chamfer_distance_from_binary_maps(a_map: np.ndarray, b_map: np.ndarray) -> float:
    a = a_map.astype(bool)
    b = b_map.astype(bool)
    count_a = int(a.sum())
    count_b = int(b.sum())
    if count_a == 0 and count_b == 0:
        return 0.0
    if count_a == 0 or count_b == 0:
        return 1.0
    dist_to_b = cv2.distanceTransform(np.where(b, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform(np.where(a, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = float(dist_to_b[a].mean()) + float(dist_to_a[b].mean())
    diag = math.hypot(*a.shape[:2]) + EPS
    return clamp01(chamfer / diag)


def radial_distance(a: np.ndarray, b: np.ndarray) -> float:
    return clamp01(float(np.mean(np.abs(a - b))))


def hu_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.mean(np.abs(a - b)))
    return clamp01(d / (d + 1.0))


def normalized_weights(weights: GeometryWeights, skeleton_available: bool) -> GeometryWeights:
    values = {
        "mask": max(0.0, float(weights.mask)),
        "contour": max(0.0, float(weights.contour)),
        "skeleton": max(0.0, float(weights.skeleton)),
        "radial": max(0.0, float(weights.radial)),
        "hu": max(0.0, float(weights.hu)),
    }
    if not skeleton_available and values["skeleton"] > 0:
        skel = values["skeleton"]
        values["skeleton"] = 0.0
        base = values["mask"] + values["contour"]
        if base > 0:
            values["mask"] += skel * values["mask"] / base
            values["contour"] += skel * values["contour"] / base
        else:
            values["contour"] += skel
    total = sum(values.values())
    if total <= 0:
        raise ValueError("At least one geometry weight must be positive")
    return GeometryWeights(*(values[k] / total for k in ["mask", "contour", "skeleton", "radial", "hu"]))


def geometry_distance(feat_a: GeometryFeature, feat_b: GeometryFeature, weights: GeometryWeights) -> float:
    skel_ok = feat_a.skeleton_available and feat_b.skeleton_available
    w = normalized_weights(weights, skel_ok)
    d_skel = chamfer_distance_from_binary_maps(feat_a.skeleton, feat_b.skeleton) if skel_ok else 0.0
    return clamp01(
        w.mask * mask_iou_distance(feat_a.mask, feat_b.mask)
        + w.contour * chamfer_distance_from_binary_maps(feat_a.edge, feat_b.edge)
        + w.skeleton * d_skel
        + w.radial * radial_distance(feat_a.radial, feat_b.radial)
        + w.hu * hu_distance(feat_a.hu, feat_b.hu)
    )


def appearance_distance(feat_a: GeometryFeature, feat_b: GeometryFeature) -> float:
    ha = feat_a.app_hist
    hb = feat_b.app_hist
    if float(ha.sum()) <= 0 and float(hb.sum()) <= 0:
        return 0.0
    chi2 = 0.5 * float(np.sum(((ha - hb) ** 2) / (ha + hb + EPS)))
    return clamp01(chi2)


def compute_tau(d: Sequence[float], args: argparse.Namespace) -> float:
    arr = np.asarray(d, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if args.tau_mode == "percentile":
        return max(0.0, float(np.percentile(arr, args.tau_percentile)))
    if args.tau_mode == "median_ratio":
        return max(0.0, float(args.tau_median_ratio) * float(np.median(arr)))
    if args.tau_mode == "fixed":
        return max(0.0, float(args.tau_fixed))
    raise ValueError(f"Unknown tau_mode: {args.tau_mode}")


def compute_participation_metrics_from_distances(
    d_geo: Sequence[float],
    tau: float,
    active_beta: float = 0.35,
    lambda_smooth: float = 1.0,
) -> Dict[str, Any]:
    d = np.asarray(d_geo, dtype=np.float64)
    m = int(d.size)
    if m == 0:
        return {
            "G_MCS": 0.0,
            "A_geo": 0.0,
            "P_geo": 0.0,
            "S_geo": 0.0,
            "J_anti": 0.0,
            "d_hat": d,
            "q_geo": d,
            "active": np.zeros(0, dtype=bool),
            "q_geo_max": 0.0,
            "active_interval_count": 0,
            "active_interval_ratio": 0.0,
            "nearly_static": True,
        }
    d_hat = np.maximum(d - float(tau), 0.0)
    total = float(d_hat.sum())
    if total < NEARLY_STATIC_EPS:
        return {
            "G_MCS": 0.0,
            "A_geo": 0.0,
            "P_geo": 0.0,
            "S_geo": 0.0,
            "J_anti": 0.0,
            "d_hat": d_hat,
            "q_geo": np.zeros_like(d_hat),
            "active": np.zeros_like(d_hat, dtype=bool),
            "q_geo_max": 0.0,
            "active_interval_count": 0,
            "active_interval_ratio": 0.0,
            "nearly_static": True,
        }

    q = d_hat / (total + EPS)
    uniform_q = 1.0 / m
    active = q > float(active_beta) * uniform_q
    a_geo = clamp01(float(active.sum()) / float(m))
    p_geo = clamp01(1.0 / (m * float(np.sum(q * q)) + EPS))
    nonzero = d_hat[d_hat > 0]
    if nonzero.size < 3 or active.sum() < 3:
        s_geo = 0.0
    else:
        speed_change = float(np.median(np.abs(np.diff(d_hat))))
        typical_speed = float(np.median(nonzero))
        s_geo = clamp01(math.exp(-float(lambda_smooth) * speed_change / (typical_speed + EPS)))
    q_max = float(q.max()) if q.size else 0.0
    j_anti = clamp01((1.0 - q_max) / (1.0 - uniform_q + EPS)) if m > 1 else 0.0
    g_mcs = clamp01(a_geo * p_geo * s_geo * j_anti)
    return {
        "G_MCS": g_mcs,
        "A_geo": a_geo,
        "P_geo": p_geo,
        "S_geo": s_geo,
        "J_anti": j_anti,
        "d_hat": d_hat,
        "q_geo": q,
        "active": active,
        "q_geo_max": q_max,
        "active_interval_count": int(active.sum()),
        "active_interval_ratio": a_geo,
        "nearly_static": False,
    }


def resample_frames_zero_order_hold(
    frames: Sequence[np.ndarray],
    original_fps: float,
    eval_fps: float,
    mode: str = "hold",
) -> Tuple[List[np.ndarray], List[int], List[float], float]:
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to resample")
    if original_fps <= 0 or not np.isfinite(original_fps):
        original_fps = 30.0
    if eval_fps <= 0 or not np.isfinite(eval_fps):
        raise ValueError("eval_fps must be positive")
    duration = (len(frames) - 1) / float(original_fps)
    k = int(math.floor(duration * float(eval_fps))) + 1
    k = max(2, k)
    target_times = np.linspace(0.0, duration, k)
    indices: List[int] = []
    for t in target_times:
        if mode == "hold":
            idx = int(math.floor(t * original_fps + 1e-6))
        elif mode == "nearest":
            idx = int(round(t * original_fps))
        else:
            raise ValueError(f"Unknown resample mode: {mode}")
        indices.append(max(0, min(len(frames) - 1, idx)))
    indices[-1] = len(frames) - 1
    return [frames[i] for i in indices], indices, [float(t) for t in target_times], float(duration)


def compute_time_metrics_from_distances(
    d_geo: Sequence[float],
    tau: float,
    active_beta: float = 0.35,
    spread_alpha: float = 4.0,
    time_score_mode: str = "sqrt_product",
) -> Dict[str, Any]:
    d = np.asarray(d_geo, dtype=np.float64)
    m = int(d.size)
    if m == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "T_GMCS": 0.0,
            "A_time": 0.0,
            "P_time_geo": 0.0,
            "P_spread": 0.0,
            "S_step": 0.0,
            "J_anti_time": 0.0,
            "d_hat": empty,
            "q_time": empty,
            "active": np.zeros(0, dtype=bool),
            "cumulative_progress_z": empty,
            "ideal_progress": empty,
            "spread_mae": 0.0,
            "freeze_ratio": 1.0,
            "mean_step_resampled": 0.0,
            "std_step_resampled": 0.0,
            "cv_step_resampled": 0.0,
            "q_max_time": 0.0,
            "time_active_interval_count": 0,
            "time_active_interval_ratio": 0.0,
            "nearly_static": True,
        }

    d_hat = np.maximum(d - float(tau), 0.0)
    total = float(d_hat.sum())
    if total < NEARLY_STATIC_EPS:
        empty_like = np.zeros_like(d_hat)
        return {
            "T_GMCS": 0.0,
            "A_time": 0.0,
            "P_time_geo": 0.0,
            "P_spread": 0.0,
            "S_step": 0.0,
            "J_anti_time": 0.0,
            "d_hat": d_hat,
            "q_time": empty_like,
            "active": np.zeros_like(d_hat, dtype=bool),
            "cumulative_progress_z": empty_like,
            "ideal_progress": np.linspace(1.0 / m, 1.0, m),
            "spread_mae": 0.0,
            "freeze_ratio": 1.0,
            "mean_step_resampled": 0.0,
            "std_step_resampled": 0.0,
            "cv_step_resampled": 0.0,
            "q_max_time": 0.0,
            "time_active_interval_count": 0,
            "time_active_interval_ratio": 0.0,
            "nearly_static": True,
        }

    q = d_hat / (total + EPS)
    uniform_q = 1.0 / float(m)
    active = q > float(active_beta) * uniform_q
    a_time = clamp01(float(active.sum()) / float(m))
    p_time_geo = clamp01(1.0 / (m * float(np.sum(q * q)) + EPS))
    z = np.cumsum(d_hat) / (total + EPS)
    ideal = np.arange(1, m + 1, dtype=np.float64) / float(m)
    spread_mae = float(np.mean(np.abs(z - ideal)))
    p_spread = clamp01(math.exp(-float(spread_alpha) * spread_mae))
    mean_step = float(d_hat.mean())
    std_step = float(d_hat.std())
    cv_step = float(std_step / (mean_step + EPS)) if mean_step > 0 else 0.0
    s_step = clamp01(1.0 / (1.0 + cv_step))
    freeze_ratio = float(np.mean(d_hat <= EPS))
    q_max = float(q.max()) if q.size else 0.0
    j_anti = clamp01((1.0 - q_max) / (1.0 - uniform_q + EPS)) if m > 1 else 0.0

    if time_score_mode == "sqrt_product":
        t_gmcs = p_spread * math.sqrt(max(0.0, a_time * p_time_geo * s_step * j_anti))
    elif time_score_mode == "product":
        t_gmcs = a_time * p_time_geo * p_spread * s_step * j_anti
    else:
        raise ValueError(f"Unknown time_score_mode: {time_score_mode}")

    return {
        "T_GMCS": clamp01(t_gmcs),
        "A_time": a_time,
        "P_time_geo": p_time_geo,
        "P_spread": p_spread,
        "S_step": s_step,
        "J_anti_time": j_anti,
        "d_hat": d_hat,
        "q_time": q,
        "active": active,
        "cumulative_progress_z": z,
        "ideal_progress": ideal,
        "spread_mae": spread_mae,
        "freeze_ratio": freeze_ratio,
        "mean_step_resampled": mean_step,
        "std_step_resampled": std_step,
        "cv_step_resampled": cv_step,
        "q_max_time": q_max,
        "time_active_interval_count": int(active.sum()),
        "time_active_interval_ratio": a_time,
        "nearly_static": False,
    }


def compute_static_gate(
    t_gmcs: float,
    freeze_ratio: float,
    repeated_frame_ratio: float,
    freeze_grace: float = 0.12,
    repeat_grace: float = 0.05,
    static_lambda: float = 1.0,
    use_static_gate: bool = True,
) -> Dict[str, float]:
    freeze_grace = clamp01(float(freeze_grace))
    repeat_grace = clamp01(float(repeat_grace))
    static_lambda = max(0.0, float(static_lambda))
    freeze_ratio = clamp01(float(freeze_ratio))
    repeated_frame_ratio = clamp01(float(repeated_frame_ratio))

    freeze_den = max(EPS, 1.0 - freeze_grace)
    repeat_den = max(EPS, 1.0 - repeat_grace)
    freeze_excess = clamp01(max(0.0, (freeze_ratio - freeze_grace) / freeze_den))
    repeat_excess = clamp01(max(0.0, (repeated_frame_ratio - repeat_grace) / repeat_den))
    static_excess = clamp01(max(freeze_excess, repeat_excess))
    f_static = 1.0 if not use_static_gate else clamp01(math.exp(-static_lambda * static_excess))
    return {
        "T_GMCS_static": clamp01(float(t_gmcs) * f_static),
        "F_static": f_static,
        "static_excess": static_excess,
        "freeze_excess": freeze_excess,
        "repeat_excess": repeat_excess,
        "freeze_grace": freeze_grace,
        "repeat_grace": repeat_grace,
        "static_lambda": static_lambda,
    }


def progress_monotonicity(progress: Sequence[float]) -> float:
    p = np.asarray(progress, dtype=np.float64)
    if p.size < 2:
        return 0.0
    delta = np.diff(p)
    backward = float(np.maximum(-delta, 0.0).sum())
    variation = float(np.abs(delta).sum())
    if variation <= EPS:
        return 0.0
    return clamp01(1.0 - backward / (variation + EPS))


def compute_single_view_gmcs(
    frames: Sequence[np.ndarray],
    fps: float,
    args: argparse.Namespace,
    view_index: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[GeometryFeature]]:
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for one view")
    weights = GeometryWeights(args.w_mask, args.w_contour, args.w_skeleton, args.w_radial, args.w_hu)
    features = [extract_geometry_features(frame, args) for frame in frames]
    skeleton_available = all(f.skeleton_available for f in features)

    d_geo = np.asarray([geometry_distance(features[i], features[i + 1], weights) for i in range(len(features) - 1)])
    d_app = np.asarray([appearance_distance(features[i], features[i + 1]) for i in range(len(features) - 1)])
    tau = compute_tau(d_geo, args)
    part = compute_participation_metrics_from_distances(d_geo, tau, args.active_beta, args.lambda_smooth)

    d_src = np.asarray([geometry_distance(features[i], features[0], weights) for i in range(len(features))])
    d_tgt = np.asarray([geometry_distance(features[i], features[-1], weights) for i in range(len(features))])
    progress = d_src / (d_src + d_tgt + EPS)
    endpoint_geo = geometry_distance(features[0], features[-1], weights)
    geo_path = float(d_geo.sum())
    app_path = float(d_app.sum())
    mean_geo = float(d_geo.mean()) if d_geo.size else 0.0
    std_geo = float(d_geo.std()) if d_geo.size else 0.0

    norm_w = normalized_weights(weights, skeleton_available)
    metrics: Dict[str, Any] = {
        "view_index": view_index,
        "G_MCS": part["G_MCS"],
        "A_geo": part["A_geo"],
        "P_geo": part["P_geo"],
        "S_geo": part["S_geo"],
        "J_anti": part["J_anti"],
        "endpoint_geo_distance": float(endpoint_geo),
        "geo_path_length": geo_path,
        "geo_change_utilization": clamp01(endpoint_geo / (geo_path + EPS)),
        "app_path_length": app_path,
        "appearance_change_ratio": clamp01(app_path / (app_path + geo_path + EPS)),
        "geometry_to_appearance_ratio": float(geo_path / (app_path + EPS)),
        "mean_geo_step": mean_geo,
        "std_geo_step": std_geo,
        "cv_geo_step": float(std_geo / (mean_geo + EPS)) if mean_geo > 0 else 0.0,
        "q_geo_max": part["q_geo_max"],
        "active_interval_count": part["active_interval_count"],
        "active_interval_ratio": part["active_interval_ratio"],
        "tau": float(tau),
        "tau_mode": args.tau_mode,
        "weights": {
            "mask": norm_w.mask,
            "contour": norm_w.contour,
            "skeleton": norm_w.skeleton,
            "radial": norm_w.radial,
            "hu": norm_w.hu,
        },
        "skeleton_available": bool(skeleton_available),
        "nearly_static": bool(part["nearly_static"]),
        "progress_monotonicity": progress_monotonicity(progress),
    }

    frame_rows: List[Dict[str, float]] = []
    d_hat = part["d_hat"]
    q_geo = part["q_geo"]
    active = part["active"]
    for i in range(len(frames)):
        frame_rows.append(
            {
                "frame_idx": float(i),
                "d_geo_to_source": float(d_src[i]),
                "d_geo_to_target": float(d_tgt[i]),
                "progress_p_t": float(progress[i]),
                "d_geo_adjacent": float(d_geo[i]) if i < d_geo.size else float("nan"),
                "d_geo_denoised": float(d_hat[i]) if i < d_hat.size else float("nan"),
                "q_geo": float(q_geo[i]) if i < q_geo.size else float("nan"),
                "active_interval": float(active[i]) if i < active.size else float("nan"),
                "d_app_adjacent": float(d_app[i]) if i < d_app.size else float("nan"),
            }
        )
    return metrics, frame_rows, features


def mean_metric_dict(metric_dicts: Sequence[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(m[key]) for m in metric_dicts if key in m and isinstance(m[key], (int, float))]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def compute_video_gmcs(
    frames: Sequence[np.ndarray],
    fps: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[np.ndarray], List[GeometryFeature]]:
    views = split_grid_frames(frames, args.grid_rows, args.grid_cols)
    view_metrics: List[Dict[str, Any]] = []
    view_rows: List[List[Dict[str, float]]] = []
    first_features: List[GeometryFeature] = []
    for view_idx, view_frames in enumerate(views):
        metrics, rows, features = compute_single_view_gmcs(view_frames, fps, args, view_idx)
        view_metrics.append(metrics)
        view_rows.append(rows)
        if view_idx == 0:
            first_features = features

    keys = [
        "G_MCS",
        "A_geo",
        "P_geo",
        "S_geo",
        "J_anti",
        "endpoint_geo_distance",
        "geo_path_length",
        "geo_change_utilization",
        "app_path_length",
        "appearance_change_ratio",
        "geometry_to_appearance_ratio",
        "mean_geo_step",
        "std_geo_step",
        "cv_geo_step",
        "q_geo_max",
        "active_interval_count",
        "active_interval_ratio",
        "tau",
        "progress_monotonicity",
    ]
    metrics = mean_metric_dict(view_metrics, keys)
    metrics.update(
        {
            "video": args.video,
            "fps": float(fps),
            "num_frames": len(frames),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "tau_mode": args.tau_mode,
            "weights": view_metrics[0]["weights"],
            "skeleton_available": bool(all(m["skeleton_available"] for m in view_metrics)),
            "nearly_static": bool(all(m["nearly_static"] for m in view_metrics)),
        }
    )
    if len(view_metrics) > 1:
        metrics["per_view_metrics"] = view_metrics

    rows = view_rows[0]
    if len(view_rows) > 1:
        for idx, row in enumerate(rows):
            for view_idx, vrows in enumerate(view_rows):
                row[f"view{view_idx}_q_geo"] = vrows[idx]["q_geo"]
                row[f"view{view_idx}_d_geo_adjacent"] = vrows[idx]["d_geo_adjacent"]
    return metrics, rows, views[0], first_features


def _prefixed_native_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "G_MCS_native": metrics["G_MCS"],
        "A_geo_native": metrics["A_geo"],
        "P_geo_native": metrics["P_geo"],
        "S_geo_native": metrics["S_geo"],
        "J_anti_native": metrics["J_anti"],
    }


def _resampled_row_data(
    rows: Sequence[Dict[str, float]],
    source_indices: Sequence[int],
    timestamps: Sequence[float],
    time_part: Dict[str, Any],
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    d_hat = time_part["d_hat"]
    q_time = time_part["q_time"]
    active = time_part["active"]
    z = time_part["cumulative_progress_z"]
    ideal = time_part["ideal_progress"]
    for i, row in enumerate(rows):
        out.append(
            {
                "resampled_frame_idx": float(i),
                "source_frame_idx": float(source_indices[i]),
                "timestamp": float(timestamps[i]),
                "d_geo_to_source": row["d_geo_to_source"],
                "d_geo_to_target": row["d_geo_to_target"],
                "progress_p_t": row["progress_p_t"],
                "d_geo_adjacent": row["d_geo_adjacent"],
                "d_geo_denoised": float(d_hat[i]) if i < len(d_hat) else float("nan"),
                "q_time": float(q_time[i]) if i < len(q_time) else float("nan"),
                "active_interval": float(active[i]) if i < len(active) else float("nan"),
                "cumulative_progress_z": float(z[i]) if i < len(z) else float("nan"),
                "ideal_progress": float(ideal[i]) if i < len(ideal) else float("nan"),
                "d_app_adjacent": row["d_app_adjacent"],
            }
        )
    return out


def compute_time_aware_video_gmcs(
    frames: Sequence[np.ndarray],
    original_fps: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], List[np.ndarray], List[GeometryFeature]]:
    native_views = split_grid_frames(frames, args.grid_rows, args.grid_cols)
    resampled_frames, resampled_indices, target_times, duration = resample_frames_zero_order_hold(
        frames, original_fps, args.eval_fps, args.resample_mode
    )
    resampled_views = split_grid_frames(resampled_frames, args.grid_rows, args.grid_cols)

    per_view: List[Dict[str, Any]] = []
    first_time_rows: List[Dict[str, float]] = []
    first_native_rows: List[Dict[str, float]] = []
    first_features: List[GeometryFeature] = []

    for view_idx, (native_view, time_view) in enumerate(zip(native_views, resampled_views)):
        native_metrics, native_rows, _ = compute_single_view_gmcs(native_view, original_fps, args, view_idx)
        resampled_metrics, resampled_rows, resampled_features = compute_single_view_gmcs(
            time_view, args.eval_fps, args, view_idx
        )
        d_geo = [row["d_geo_adjacent"] for row in resampled_rows[:-1]]
        time_part = compute_time_metrics_from_distances(
            d_geo=d_geo,
            tau=resampled_metrics["tau"],
            active_beta=args.active_beta,
            spread_alpha=args.spread_alpha,
            time_score_mode=args.time_score_mode,
        )
        view_metrics: Dict[str, Any] = {
            "view_index": view_idx,
            "T_GMCS": time_part["T_GMCS"],
            "G_MCS_native": native_metrics["G_MCS"],
            "G_MCS_resampled": resampled_metrics["G_MCS"],
            "A_time": time_part["A_time"],
            "P_time_geo": time_part["P_time_geo"],
            "P_spread": time_part["P_spread"],
            "S_step": time_part["S_step"],
            "J_anti_time": time_part["J_anti_time"],
            "A_geo_native": native_metrics["A_geo"],
            "P_geo_native": native_metrics["P_geo"],
            "S_geo_native": native_metrics["S_geo"],
            "J_anti_native": native_metrics["J_anti"],
            "endpoint_geo_distance": resampled_metrics["endpoint_geo_distance"],
            "geo_path_length": resampled_metrics["geo_path_length"],
            "geo_change_utilization": resampled_metrics["geo_change_utilization"],
            "app_path_length": resampled_metrics["app_path_length"],
            "appearance_change_ratio": resampled_metrics["appearance_change_ratio"],
            "geometry_to_appearance_ratio": resampled_metrics["geometry_to_appearance_ratio"],
            "progress_monotonicity": resampled_metrics["progress_monotonicity"],
            "freeze_ratio": time_part["freeze_ratio"],
            "spread_mae": time_part["spread_mae"],
            "mean_step_resampled": time_part["mean_step_resampled"],
            "std_step_resampled": time_part["std_step_resampled"],
            "cv_step_resampled": time_part["cv_step_resampled"],
            "q_max_time": time_part["q_max_time"],
            "time_active_interval_count": time_part["time_active_interval_count"],
            "time_active_interval_ratio": time_part["time_active_interval_ratio"],
            "tau": resampled_metrics["tau"],
            "skeleton_available": resampled_metrics["skeleton_available"] and native_metrics["skeleton_available"],
            "nearly_static": time_part["nearly_static"],
        }
        view_metrics.update(
            compute_static_gate(
                time_part["T_GMCS"],
                time_part["freeze_ratio"],
                0.0,
                args.freeze_grace,
                args.repeat_grace,
                args.static_lambda,
                args.use_static_gate,
            )
        )
        per_view.append(view_metrics)
        if view_idx == 0:
            first_time_rows = _resampled_row_data(resampled_rows, resampled_indices, target_times, time_part)
            first_native_rows = native_rows
            first_features = resampled_features

    keys = [
        "T_GMCS",
        "G_MCS_native",
        "G_MCS_resampled",
        "A_time",
        "P_time_geo",
        "P_spread",
        "S_step",
        "J_anti_time",
        "A_geo_native",
        "P_geo_native",
        "S_geo_native",
        "J_anti_native",
        "endpoint_geo_distance",
        "geo_path_length",
        "geo_change_utilization",
        "app_path_length",
        "appearance_change_ratio",
        "geometry_to_appearance_ratio",
        "progress_monotonicity",
        "freeze_ratio",
        "spread_mae",
        "mean_step_resampled",
        "std_step_resampled",
        "cv_step_resampled",
        "q_max_time",
        "time_active_interval_count",
        "time_active_interval_ratio",
        "tau",
    ]
    metrics = mean_metric_dict(per_view, keys)
    unique_count = len(set(resampled_indices))
    repeated_frame_ratio = clamp01(1.0 - unique_count / float(len(resampled_frames)))
    gate = compute_static_gate(
        metrics["T_GMCS"],
        metrics["freeze_ratio"],
        repeated_frame_ratio,
        args.freeze_grace,
        args.repeat_grace,
        args.static_lambda,
        args.use_static_gate,
    )
    metrics.update(
        {
            "video": args.video,
            "original_fps": float(original_fps),
            "eval_fps": float(args.eval_fps),
            "duration": float(duration),
            "original_num_frames": len(frames),
            "resampled_num_frames": len(resampled_frames),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "tau_mode": args.tau_mode,
            "active_beta": float(args.active_beta),
            "spread_alpha": float(args.spread_alpha),
            "time_score_mode": args.time_score_mode,
            "resample_mode": args.resample_mode,
            "resampled_indices": [int(i) for i in resampled_indices],
            "unique_source_frame_count": int(unique_count),
            "repeated_frame_ratio": repeated_frame_ratio,
            "skeleton_available": bool(all(m["skeleton_available"] for m in per_view)),
            "nearly_static": bool(all(m["nearly_static"] for m in per_view)),
        }
    )
    metrics.update(gate)
    if len(per_view) > 1:
        metrics["per_view_metrics"] = per_view
    return metrics, first_time_rows, first_native_rows, resampled_views[0], first_features


def save_metrics_json(metrics: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_frame_metrics_csv(rows: Sequence[Dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_line(values: Sequence[float], title: str, ylabel: str, out_path: Path) -> None:
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
    rows: Sequence[Dict[str, float]],
    frames: Sequence[np.ndarray],
    features: Sequence[GeometryFeature],
) -> None:
    out = Path(out_dir)
    d_geo = [row["d_geo_adjacent"] for row in rows[:-1]]
    d_app = [row["d_app_adjacent"] for row in rows[:-1]]
    q = [row["q_geo"] for row in rows[:-1]]
    active = [row["active_interval"] for row in rows[:-1]]
    progress = [row["progress_p_t"] for row in rows]
    save_line(d_geo, "Adjacent geometry distance", "D_geo", out / "geo_adjacent_distance.png")
    save_line(progress, "Endpoint-relative geometry progress", "p_t", out / "progress_curve_geo.png")

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(q)), q)
    plt.title("Geometric participation q_geo")
    plt.xlabel("Interval index")
    plt.ylabel("q_geo")
    plt.tight_layout()
    plt.savefig(out / "q_geo_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.bar(np.arange(len(active)), active)
    plt.title("Active geometric intervals")
    plt.xlabel("Interval index")
    plt.ylabel("active")
    plt.tight_layout()
    plt.savefig(out / "active_intervals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(d_geo, label="geometry", marker="o", markersize=3)
    plt.plot(d_app, label="appearance", marker="o", markersize=3)
    plt.title("Appearance vs geometry adjacent distances")
    plt.xlabel("Interval index")
    plt.ylabel("distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "app_vs_geo_distance.png", dpi=160)
    plt.close()

    sample_n = min(10, len(frames))
    sample_idx = np.unique(np.rint(np.linspace(0, len(frames) - 1, sample_n)).astype(int))
    sampled_frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    skeletons: List[np.ndarray] = []
    for idx in sample_idx:
        frame = frames[int(idx)]
        feat = features[int(idx)]
        sampled_frames.append(frame)
        masks.append((feat.mask.astype(np.uint8) * 255))
        edges.append((feat.edge.astype(np.uint8) * 255))
        skeletons.append((feat.skeleton.astype(np.uint8) * 255))

    cv2.imwrite(str(out / "sampled_frames.jpg"), cv2.cvtColor(make_contact_sheet(sampled_frames), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "sampled_masks.jpg"), cv2.cvtColor(make_contact_sheet(masks), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "sampled_edges.jpg"), cv2.cvtColor(make_contact_sheet(edges), cv2.COLOR_RGB2BGR))
    if all(f.skeleton_available for f in features):
        cv2.imwrite(str(out / "sampled_skeletons.jpg"), cv2.cvtColor(make_contact_sheet(skeletons), cv2.COLOR_RGB2BGR))


def save_time_debug_plots(
    out_dir: str | Path,
    time_rows: Sequence[Dict[str, float]],
    native_rows: Sequence[Dict[str, float]],
    resampled_frames: Sequence[np.ndarray],
    resampled_features: Sequence[GeometryFeature],
    metrics: Optional[Dict[str, Any]] = None,
    save_resampled_debug: bool = False,
) -> None:
    out = Path(out_dir)
    native_d = [row["d_geo_adjacent"] for row in native_rows[:-1]]
    native_q = [row["q_geo"] for row in native_rows[:-1]]
    native_progress = [row["progress_p_t"] for row in native_rows]
    save_line(native_d, "Native adjacent geometry distance", "D_geo", out / "native_geo_adjacent_distance.png")
    save_line(native_progress, "Native geometry progress", "p_t", out / "native_progress_curve_geo.png")

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(native_q)), native_q)
    plt.title("Native geometric participation q_geo")
    plt.xlabel("Interval index")
    plt.ylabel("q_geo")
    plt.tight_layout()
    plt.savefig(out / "native_q_geo_distribution.png", dpi=160)
    plt.close()

    d_geo = [row["d_geo_adjacent"] for row in time_rows[:-1]]
    d_hat = [row["d_geo_denoised"] for row in time_rows[:-1]]
    q = [row["q_time"] for row in time_rows[:-1]]
    active = [row["active_interval"] for row in time_rows[:-1]]
    z = [row["cumulative_progress_z"] for row in time_rows[:-1]]
    ideal = [row["ideal_progress"] for row in time_rows[:-1]]
    source_idx = [row["source_frame_idx"] for row in time_rows]
    freeze = [1.0 if row["d_geo_denoised"] <= EPS else 0.0 for row in time_rows[:-1]]

    save_line(d_geo, "Time-aware adjacent geometry distance", "D_geo", out / "time_geo_adjacent_distance.png")
    save_line(d_hat, "Time-aware denoised geometry step", "d_hat", out / "time_denoised_geo_step.png")
    save_line(source_idx, "Resampled source frame indices", "source index", out / "time_source_frame_indices.png")

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(q)), q)
    plt.title("Time-aware geometric participation q")
    plt.xlabel("Interval index")
    plt.ylabel("q_time")
    plt.tight_layout()
    plt.savefig(out / "time_q_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.bar(np.arange(len(active)), active)
    plt.title("Time-aware active intervals")
    plt.xlabel("Interval index")
    plt.ylabel("active")
    plt.tight_layout()
    plt.savefig(out / "time_active_intervals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ideal, label="ideal", linewidth=1.5)
    plt.plot(z, label="actual", marker="o", markersize=3)
    plt.title("Cumulative geometry progress vs ideal")
    plt.xlabel("Interval index")
    plt.ylabel("normalized progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "time_cumulative_progress_vs_ideal.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.bar(np.arange(len(freeze)), freeze)
    plt.title("Freeze-jump pattern")
    plt.xlabel("Interval index")
    plt.ylabel("frozen")
    plt.tight_layout()
    plt.savefig(out / "time_freeze_jump_pattern.png", dpi=160)
    plt.close()

    if metrics is not None:
        names = ["freeze_ratio", "repeated_frame_ratio", "freeze_excess", "repeat_excess", "static_excess", "F_static"]
        values = [float(metrics.get(name, 0.0)) for name in names]
        plt.figure(figsize=(9, 4))
        plt.bar(np.arange(len(names)), values)
        plt.xticks(np.arange(len(names)), names, rotation=25, ha="right")
        plt.ylim(0, 1)
        plt.title("Static gate diagnostics")
        plt.tight_layout()
        plt.savefig(out / "static_gate_diagnostics.png", dpi=160)
        plt.close()

    if save_resampled_debug:
        sample_n = min(10, len(resampled_frames))
        sample_idx = np.unique(np.rint(np.linspace(0, len(resampled_frames) - 1, sample_n)).astype(int))
        sampled_frames: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        edges: List[np.ndarray] = []
        skeletons: List[np.ndarray] = []
        for idx in sample_idx:
            frame = resampled_frames[int(idx)]
            feat = resampled_features[int(idx)]
            sampled_frames.append(frame)
            masks.append((feat.mask.astype(np.uint8) * 255))
            edges.append((feat.edge.astype(np.uint8) * 255))
            skeletons.append((feat.skeleton.astype(np.uint8) * 255))
        cv2.imwrite(str(out / "resampled_frames.jpg"), cv2.cvtColor(make_contact_sheet(sampled_frames), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "resampled_masks.jpg"), cv2.cvtColor(make_contact_sheet(masks), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "resampled_edges.jpg"), cv2.cvtColor(make_contact_sheet(edges), cv2.COLOR_RGB2BGR))
        if all(f.skeleton_available for f in resampled_features):
            cv2.imwrite(
                str(out / "resampled_skeletons.jpg"),
                cv2.cvtColor(make_contact_sheet(skeletons), cv2.COLOR_RGB2BGR),
            )


def interpretation(metrics: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    if "T_GMCS" in metrics:
        if metrics.get("T_GMCS", 0.0) >= 0.3 and metrics.get("T_GMCS_static", metrics.get("T_GMCS", 0.0)) < 0.75 * metrics.get("T_GMCS", 0.0):
            notes.append("High T-GMCS but low T-GMCS_static: many static or duplicated frames may make the video look artificially stable")
        if metrics["T_GMCS"] >= 0.6:
            notes.append("High T-GMCS: temporally continuous geometric morphing")
        if metrics["G_MCS_native"] >= 0.4 and metrics["T_GMCS"] < 0.3:
            notes.append("Native G-MCS high but T-GMCS low: likely low-FPS or flipbook-like morphing")
        if metrics["freeze_ratio"] >= 0.5:
            notes.append("High freeze_ratio: many playback intervals are static")
        if metrics.get("repeated_frame_ratio", 0.0) >= 0.5:
            notes.append("High repeated_frame_ratio: low-FPS source frames are held for multiple evaluation frames")
        if metrics["P_spread"] < 0.5:
            notes.append("Low P_spread: morphing progress is released unevenly")
        if metrics["S_step"] < 0.5:
            notes.append("Low S_step: step sizes are unstable, likely freeze-jump behavior")
        if metrics["appearance_change_ratio"] >= 0.6:
            notes.append("High appearance_change_ratio: changes may be appearance-driven rather than geometry-driven")
        if metrics["geo_change_utilization"] < 0.25 and metrics["T_GMCS"] >= 0.5:
            notes.append("Low geo_change_utilization but high T-GMCS: possibly valid nonlinear cross-category morphing")
        if not notes:
            notes.append("mixed time-aware geometry behavior; inspect component scores and debug plots")
        return notes

    if metrics["G_MCS"] >= 0.6 and metrics["appearance_change_ratio"] < 0.5:
        notes.append("high G-MCS + low appearance ratio: good geometric morphing continuity")
    if metrics["G_MCS"] < 0.3 and metrics["appearance_change_ratio"] >= 0.5:
        notes.append("low G-MCS + high appearance ratio: likely appearance-driven / 2D-like transition")
    if metrics["A_geo"] < 0.35:
        notes.append("low A_geo: changes happen in too few intervals")
    if metrics["J_anti"] < 0.5:
        notes.append("low J_anti: abrupt jump detected")
    if metrics["S_geo"] < 0.5:
        notes.append("low S_geo: unstable local geometric changes")
    if metrics["geo_change_utilization"] < 0.25 and metrics["G_MCS"] >= 0.5:
        notes.append("low CU but high G-MCS: possibly valid nonlinear cross-category morphing path")
    if not notes:
        notes.append("mixed geometry/appearance behavior; inspect component scores and debug plots")
    return notes


def run_tests() -> None:
    ns = argparse.Namespace(active_beta=0.35, lambda_smooth=1.0)

    uniform = compute_participation_metrics_from_distances([1] * 9, 0.0, ns.active_beta, ns.lambda_smooth)
    assert uniform["A_geo"] > 0.99
    assert uniform["P_geo"] > 0.99
    assert uniform["J_anti"] > 0.99
    assert uniform["G_MCS"] > 0.99

    single = compute_participation_metrics_from_distances([0, 0, 0, 0, 9, 0, 0, 0, 0], 0.0, ns.active_beta, ns.lambda_smooth)
    assert abs(single["A_geo"] - 1 / 9) < 1e-6
    assert abs(single["P_geo"] - 1 / 9) < 1e-6
    assert single["J_anti"] < 1e-6
    assert single["G_MCS"] < 1e-6

    static = compute_participation_metrics_from_distances([0] * 9, 0.0, ns.active_beta, ns.lambda_smooth)
    assert static["G_MCS"] == 0.0
    assert static["nearly_static"] is True

    nonlinear = compute_participation_metrics_from_distances(
        [0.5, 0.8, 1.0, 1.2, 1.1, 1.0, 0.9, 0.7, 0.5], 0.0, ns.active_beta, ns.lambda_smooth
    )
    assert nonlinear["G_MCS"] > 0.5
    assert nonlinear["S_geo"] > 0.7

    jitter = compute_participation_metrics_from_distances(
        [0.1, 2.0, 0.1, 2.0, 0.1, 2.0, 0.1, 2.0, 0.1], 0.0, ns.active_beta, ns.lambda_smooth
    )
    assert jitter["S_geo"] < 0.5
    assert jitter["G_MCS"] < uniform["G_MCS"]

    t_uniform = compute_time_metrics_from_distances([1] * 12, 0.0, 0.35, 4.0, "sqrt_product")
    assert t_uniform["A_time"] > 0.99
    assert t_uniform["P_time_geo"] > 0.99
    assert t_uniform["P_spread"] > 0.99
    assert t_uniform["S_step"] > 0.99
    assert t_uniform["J_anti_time"] > 0.99
    assert t_uniform["T_GMCS"] > 0.99

    t_single = compute_time_metrics_from_distances([0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0], 0.0, 0.35, 4.0, "sqrt_product")
    assert t_single["A_time"] < 0.1
    assert t_single["P_time_geo"] < 0.1
    assert t_single["S_step"] < 0.3
    assert t_single["J_anti_time"] < 1e-6
    assert t_single["T_GMCS"] < 1e-6

    t_hold = compute_time_metrics_from_distances([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 0.0, 0.35, 4.0, "sqrt_product")
    assert abs(t_hold["A_time"] - 0.25) < 1e-6
    assert abs(t_hold["P_time_geo"] - 0.25) < 1e-6
    assert t_hold["S_step"] < t_uniform["S_step"]
    assert t_hold["T_GMCS"] < t_uniform["T_GMCS"]

    t_static = compute_time_metrics_from_distances([0] * 8, 0.0, 0.35, 4.0, "sqrt_product")
    assert t_static["T_GMCS"] == 0.0
    assert t_static["nearly_static"] is True

    t_nonlinear = compute_time_metrics_from_distances([0.5, 0.7, 0.9, 1.1, 1.2, 1.1, 0.9, 0.7, 0.5], 0.0, 0.35, 4.0, "sqrt_product")
    assert t_nonlinear["T_GMCS"] > 0.6
    assert t_nonlinear["P_spread"] > 0.8
    assert t_nonlinear["S_step"] > 0.7

    gate_good = compute_static_gate(0.8, 0.05, 0.0)
    assert abs(gate_good["F_static"] - 1.0) < 1e-6
    assert abs(gate_good["T_GMCS_static"] - 0.8) < 1e-6

    gate_tiny_freeze = compute_static_gate(0.4, 0.106, 0.0)
    assert abs(gate_tiny_freeze["F_static"] - 1.0) < 1e-6
    assert abs(gate_tiny_freeze["T_GMCS_static"] - 0.4) < 1e-6

    gate_flipbook = compute_static_gate(0.09, 0.75, 0.73)
    assert gate_flipbook["F_static"] < 1.0
    assert gate_flipbook["T_GMCS_static"] < 0.09

    gate_duplicates = compute_static_gate(0.5, 0.50, 0.0)
    assert gate_duplicates["F_static"] < 1.0
    assert gate_duplicates["T_GMCS_static"] < 0.5

    gate_static = compute_static_gate(0.0, 1.0, 1.0)
    assert gate_static["F_static"] < 0.5
    assert gate_static["T_GMCS_static"] == 0.0
    print("Synthetic G-MCS and T-GMCS tests passed.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SG-TGMCS: Static-Gated Time-aware Geometric Morphing Continuity Score. "
            "The main output field is T_GMCS_static."
        )
    )
    parser.add_argument("--video", required=False)
    parser.add_argument("--video_dir", required=False)
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--out_dir", required=False)
    parser.add_argument("--grid_rows", type=int, default=1)
    parser.add_argument("--grid_cols", type=int, default=1)
    parser.add_argument("--resize_width", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--save_resampled_debug", action="store_true")
    parser.add_argument("--eval_fps", type=float, default=24.0)
    parser.add_argument("--resample_mode", choices=["hold", "nearest"], default="hold")
    parser.add_argument("--use_time_aware", action="store_true", default=True)
    parser.add_argument("--spread_alpha", type=float, default=4.0)
    parser.add_argument("--time_score_mode", choices=["sqrt_product", "product"], default="sqrt_product")
    parser.add_argument("--freeze_grace", type=float, default=0.12)
    parser.add_argument("--repeat_grace", type=float, default=0.05)
    parser.add_argument("--static_lambda", type=float, default=1.0)
    parser.add_argument("--use_static_gate", action="store_true", default=True)
    parser.add_argument("--tau_mode", choices=["percentile", "median_ratio", "fixed"], default="percentile")
    parser.add_argument("--tau_percentile", type=float, default=10.0)
    parser.add_argument("--tau_median_ratio", type=float, default=0.1)
    parser.add_argument("--tau_fixed", type=float, default=0.0)
    parser.add_argument("--active_beta", type=float, default=0.35)
    parser.add_argument("--lambda_smooth", type=float, default=1.0)
    parser.add_argument("--bg_threshold", type=int, default=245)
    parser.add_argument("--min_component_ratio", type=float, default=0.001)
    parser.add_argument("--w_mask", type=float, default=0.25)
    parser.add_argument("--w_contour", type=float, default=0.35)
    parser.add_argument("--w_skeleton", type=float, default=0.20)
    parser.add_argument("--w_radial", type=float, default=0.10)
    parser.add_argument("--w_hu", type=float, default=0.10)
    parser.add_argument("--run_tests", action="store_true")
    return parser


SUMMARY_FIELDS = [
    "video",
    "original_fps",
    "original_num_frames",
    "resampled_num_frames",
    "T_GMCS_static",
    "F_static",
    "static_excess",
    "freeze_excess",
    "repeat_excess",
    "T_GMCS",
    "G_MCS_native",
    "G_MCS_resampled",
    "A_time",
    "P_time_geo",
    "P_spread",
    "S_step",
    "J_anti_time",
    "freeze_ratio",
    "repeated_frame_ratio",
    "appearance_change_ratio",
    "geo_change_utilization",
    "endpoint_geo_distance",
    "geo_path_length",
]


def safe_output_name(video_path: str) -> str:
    p = Path(video_path)
    return f"{p.parent.name}__{p.stem}".replace(os.sep, "_")


def evaluate_video_to_dir(video_path: str, out_dir: str | Path, args: argparse.Namespace) -> Dict[str, Any]:
    args.video = video_path
    os.makedirs(out_dir, exist_ok=True)
    frames, fps = read_video_frames(video_path, args.max_frames, args.resize_width)
    metrics, time_rows, native_rows, debug_frames, debug_features = compute_time_aware_video_gmcs(frames, fps, args)
    save_metrics_json(metrics, Path(out_dir) / "metrics.json")
    save_frame_metrics_csv(time_rows, Path(out_dir) / "frame_metrics.csv")
    save_frame_metrics_csv(native_rows, Path(out_dir) / "native_frame_metrics.csv")
    if args.save_debug:
        save_time_debug_plots(
            out_dir,
            time_rows,
            native_rows,
            debug_frames,
            debug_features,
            metrics=metrics,
            save_resampled_debug=args.save_resampled_debug,
        )
    return metrics


def save_summary_csv(rows: Sequence[Dict[str, Any]], path: str | Path, extra_fields: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fields = list(SUMMARY_FIELDS)
    if extra_fields:
        fields = extra_fields + fields
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (list, dict)):
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def print_time_summary(metrics: Dict[str, Any]) -> None:
    print(f"Video: {metrics['video']}")
    print(f"Original FPS: {metrics['original_fps']:.3f}")
    print(f"Eval FPS: {metrics['eval_fps']:.3f}")
    print(f"Original frames: {metrics['original_num_frames']}")
    print(f"Resampled frames: {metrics['resampled_num_frames']}")
    print(f"Duration: {metrics['duration']:.3f}")
    print("")
    print(f"T-GMCS-static: {metrics['T_GMCS_static']:.6f}")
    print(f"T-GMCS: {metrics['T_GMCS']:.6f}")
    print(f"Native G-MCS: {metrics['G_MCS_native']:.6f}")
    print(f"Resampled G-MCS: {metrics['G_MCS_resampled']:.6f}")
    print("")
    print(f"A_time: {metrics['A_time']:.6f}")
    print(f"P_time_geo: {metrics['P_time_geo']:.6f}")
    print(f"P_spread: {metrics['P_spread']:.6f}")
    print(f"S_step: {metrics['S_step']:.6f}")
    print(f"J_anti_time: {metrics['J_anti_time']:.6f}")
    print("")
    print(f"freeze_ratio: {metrics['freeze_ratio']:.6f}")
    print(f"repeated_frame_ratio: {metrics['repeated_frame_ratio']:.6f}")
    print(f"F_static: {metrics['F_static']:.6f}")
    print(f"static_excess: {metrics['static_excess']:.6f}")
    print(f"freeze_excess: {metrics['freeze_excess']:.6f}")
    print(f"repeat_excess: {metrics['repeat_excess']:.6f}")
    print(f"appearance_change_ratio: {metrics['appearance_change_ratio']:.6f}")
    print(f"geo_change_utilization: {metrics['geo_change_utilization']:.6f}")
    for note in interpretation(metrics):
        print(f"- {note}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.run_tests:
        run_tests()
        return
    if not args.out_dir:
        parser.error("--out_dir is required unless --run_tests is used")

    if args.video_dir:
        video_paths = sorted(str(p) for p in Path(args.video_dir).rglob(args.glob))
        if not video_paths:
            raise RuntimeError(f"No videos found in {args.video_dir} with glob {args.glob}")
        os.makedirs(args.out_dir, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for video in video_paths:
            out_dir = Path(args.out_dir) / safe_output_name(video)
            metrics = evaluate_video_to_dir(video, out_dir, args)
            rows.append(metrics)
            print(f"{metrics['T_GMCS']:.6f}\t{video}")
        save_summary_csv(rows, Path(args.out_dir) / "summary.csv")
        print(f"Summary: {Path(args.out_dir) / 'summary.csv'}")
        return

    if not args.video:
        parser.error("--video or --video_dir is required unless --run_tests is used")

    metrics = evaluate_video_to_dir(args.video, args.out_dir, args)
    print_time_summary(metrics)


if __name__ == "__main__":
    main()
