import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "naive")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trellis.utils.morphing_utils import (
    compute_token_boundary_proxy,
    coords_to_token_boundary_targets,
    extract_token_boundary_proxy_features,
)


def _load_sample(cache_dir: Path):
    z_s = torch.load(cache_dir / "coords_zs.pt", map_location="cpu")
    coords = torch.load(cache_dir / "coords.pt", map_location="cpu").int()
    feature_pack = extract_token_boundary_proxy_features(z_s, patch_size=1)
    targets = coords_to_token_boundary_targets(coords, token_grid_side=feature_pack["grid_side"])
    heuristic = compute_token_boundary_proxy(z_s, patch_size=1)
    return {
        "name": cache_dir.parent.name,
        "cache_dir": str(cache_dir),
        "norm_features": feature_pack["norm_features"][0].float(),
        "heuristic_score": heuristic["score"][0].float(),
        "heuristic_mask": heuristic["mask"][0].bool(),
        "soft_target": targets["soft_target"].float(),
        "hard_target": targets["hard_target"].bool(),
        "occupied_mask": targets["occupied_mask"].bool(),
        "feature_names": feature_pack["feature_names"],
    }


def _collect_samples():
    caches = sorted(Path("./outputs/cache").glob("*/cache"))
    samples = []
    for cache_dir in caches:
        if not (cache_dir / "coords.pt").exists() or not (cache_dir / "coords_zs.pt").exists():
            continue
        samples.append(_load_sample(cache_dir))
    return samples


def _binary_metrics(score: torch.Tensor, target: torch.Tensor, threshold: float):
    pred = score >= threshold
    tp = (pred & target).sum().item()
    fp = (pred & (~target)).sum().item()
    fn = ((~pred) & target).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def _best_threshold(score: torch.Tensor, target: torch.Tensor):
    best = None
    for thr in torch.linspace(0.1, 0.9, steps=33):
        metrics = _binary_metrics(score, target, float(thr.item()))
        if best is None or metrics["iou"] > best["iou"]:
            best = metrics
    return best


def _fit_calibrator(train_samples, steps: int = 400, lr: float = 0.1):
    feats = []
    soft_targets = []
    hard_targets = []
    for sample in train_samples:
        occ = sample["occupied_mask"]
        feats.append(sample["norm_features"][occ])
        soft_targets.append(sample["soft_target"][occ])
        hard_targets.append(sample["hard_target"][occ])

    x = torch.cat(feats, dim=0)
    y_soft = torch.cat(soft_targets, dim=0).clamp(0.0, 1.0)
    y_hard = torch.cat(hard_targets, dim=0)

    weights = torch.zeros(x.shape[-1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    pos_weight = ((~y_hard).sum().float() / y_hard.sum().float().clamp(min=1.0)).clamp(min=1.0, max=20.0)
    for _ in range(int(steps)):
        logits = x @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_soft, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_score = torch.sigmoid(x @ weights + bias)
        best = _best_threshold(train_score, y_hard)

    return {
        "weights": weights.detach().tolist(),
        "bias": float(bias.detach().item()),
        "threshold": float(best["threshold"]),
        "feature_names": train_samples[0]["feature_names"],
        "train_best": best,
    }


def _fit_ratio_predictor(train_samples, steps: int = 400, lr: float = 0.05):
    pooled = []
    targets = []
    for sample in train_samples:
        feat = sample["norm_features"]
        pooled_feat = torch.cat(
            [
                feat.mean(dim=0),
                feat.std(dim=0, unbiased=False),
                feat.max(dim=0).values,
            ],
            dim=0,
        )
        pooled.append(pooled_feat)
        targets.append(sample["hard_target"].float().mean())

    x = torch.stack(pooled, dim=0)
    y = torch.stack(targets, dim=0)
    weights = torch.zeros(x.shape[-1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    for _ in range(int(steps)):
        pred = torch.sigmoid(x @ weights + bias)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = torch.sigmoid(x @ weights + bias)
        mae = (pred - y).abs().mean().item()

    return {
        "weights": weights.detach().tolist(),
        "bias": float(bias.detach().item()),
        "mae": float(mae),
        "min_ratio": float(y.min().item()),
        "max_ratio": float(y.max().item()),
    }


def _fit_threshold_predictor(train_samples, calibrator, steps: int = 400, lr: float = 0.05):
    pooled = []
    targets = []
    for sample in train_samples:
        feat = sample["norm_features"]
        pooled_feat = torch.cat(
            [
                feat.mean(dim=0),
                feat.std(dim=0, unbiased=False),
                feat.max(dim=0).values,
            ],
            dim=0,
        )
        occ = sample["occupied_mask"]
        target = sample["hard_target"][occ]
        logits = sample["norm_features"][occ] @ torch.tensor(calibrator["weights"], dtype=feat.dtype) + float(calibrator["bias"])
        score = torch.sigmoid(logits)
        oracle_thr = _best_threshold(score, target)["threshold"]
        pooled.append(pooled_feat)
        targets.append(torch.tensor(oracle_thr, dtype=feat.dtype))

    x = torch.stack(pooled, dim=0)
    y = torch.stack(targets, dim=0)
    weights = torch.zeros(x.shape[-1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    for _ in range(int(steps)):
        pred = torch.sigmoid(x @ weights + bias)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = torch.sigmoid(x @ weights + bias)
        mae = (pred - y).abs().mean().item()

    return {
        "weights": weights.detach().tolist(),
        "bias": float(bias.detach().item()),
        "mae": float(mae),
        "min_threshold": float(y.min().item()),
        "max_threshold": float(y.max().item()),
    }


def _evaluate(samples, calibrator):
    per_sample = []
    heuristic_scores = []
    calibrated_scores = []
    targets = []
    adaptive_preds = []

    for sample in samples:
        occ = sample["occupied_mask"]
        target = sample["hard_target"][occ]
        heuristic_score = sample["heuristic_score"][occ]
        logits = sample["norm_features"][occ] @ torch.tensor(calibrator["weights"], dtype=sample["norm_features"].dtype) + float(calibrator["bias"])
        calibrated_score = torch.sigmoid(logits)
        ratio_cfg = calibrator.get("ratio_predictor")
        adaptive_topk = None
        adaptive_ratio = None
        adaptive_metrics = None
        adaptive_threshold_metrics = None
        if ratio_cfg is not None:
            feat = sample["norm_features"]
            pooled_feat = torch.cat(
                [
                    feat.mean(dim=0),
                    feat.std(dim=0, unbiased=False),
                    feat.max(dim=0).values,
                ],
                dim=0,
            )
            adaptive_ratio = torch.sigmoid(
                pooled_feat @ torch.tensor(ratio_cfg["weights"], dtype=feat.dtype) + float(ratio_cfg["bias"])
            ).item()
            adaptive_ratio = min(max(adaptive_ratio, ratio_cfg["min_ratio"]), ratio_cfg["max_ratio"])
            keep = max(1, int(round(target.numel() * adaptive_ratio)))
            topk_idx = torch.topk(calibrated_score, k=keep, largest=True).indices
            adaptive_topk = torch.zeros_like(calibrated_score, dtype=torch.bool)
            adaptive_topk[topk_idx] = True
            tp = (adaptive_topk & target).sum().item()
            fp = (adaptive_topk & (~target)).sum().item()
            fn = ((~adaptive_topk) & target).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            iou = tp / max(tp + fp + fn, 1)
            adaptive_metrics = {
                "predicted_ratio": float(adaptive_ratio),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "iou": float(iou),
            }
        threshold_cfg = calibrator.get("threshold_predictor")
        if threshold_cfg is not None:
            feat = sample["norm_features"]
            pooled_feat = torch.cat(
                [
                    feat.mean(dim=0),
                    feat.std(dim=0, unbiased=False),
                    feat.max(dim=0).values,
                ],
                dim=0,
            )
            adaptive_threshold = torch.sigmoid(
                pooled_feat @ torch.tensor(threshold_cfg["weights"], dtype=feat.dtype) + float(threshold_cfg["bias"])
            ).item()
            adaptive_threshold = min(max(adaptive_threshold, threshold_cfg["min_threshold"]), threshold_cfg["max_threshold"])
            adaptive_threshold_metrics = _binary_metrics(calibrated_score, target, adaptive_threshold)
            adaptive_threshold_metrics["predicted_threshold"] = float(adaptive_threshold)

        heuristic_best = _best_threshold(heuristic_score, target)
        calibrated_fixed = _binary_metrics(calibrated_score, target, calibrator["threshold"])
        calibrated_best = _best_threshold(calibrated_score, target)
        per_sample.append(
            {
                "name": sample["name"],
                "occupied_token_count": int(target.numel()),
                "positive_token_count": int(target.sum().item()),
                "heuristic_best": heuristic_best,
                "calibrated_fixed": calibrated_fixed,
                "calibrated_best": calibrated_best,
                "calibrated_adaptive_topk": adaptive_metrics,
                "calibrated_adaptive_threshold": adaptive_threshold_metrics,
            }
        )
        heuristic_scores.append(heuristic_score)
        calibrated_scores.append(calibrated_score)
        targets.append(target)
        if adaptive_topk is not None:
            adaptive_preds.append((adaptive_topk, target, adaptive_ratio))

    heuristic_scores = torch.cat(heuristic_scores, dim=0)
    calibrated_scores = torch.cat(calibrated_scores, dim=0)
    targets = torch.cat(targets, dim=0)
    overall = {
        "heuristic_best": _best_threshold(heuristic_scores, targets),
        "calibrated_fixed": _binary_metrics(calibrated_scores, targets, calibrator["threshold"]),
        "calibrated_best": _best_threshold(calibrated_scores, targets),
    }
    if adaptive_preds:
        pred = torch.cat([p.float() for p, _, _ in adaptive_preds], dim=0).bool()
        target = torch.cat([t for _, t, _ in adaptive_preds], dim=0)
        tp = (pred & target).sum().item()
        fp = (pred & (~target)).sum().item()
        fn = ((~pred) & target).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = tp / max(tp + fp + fn, 1)
        overall["calibrated_adaptive_topk"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "mean_predicted_ratio": float(sum(r for _, _, r in adaptive_preds) / len(adaptive_preds)),
        }
    threshold_metrics = []
    for sample in per_sample:
        if sample["calibrated_adaptive_threshold"] is not None:
            threshold_metrics.append(sample["calibrated_adaptive_threshold"])
    if threshold_metrics:
        overall["calibrated_adaptive_threshold"] = {
            "precision": float(sum(m["precision"] for m in threshold_metrics) / len(threshold_metrics)),
            "recall": float(sum(m["recall"] for m in threshold_metrics) / len(threshold_metrics)),
            "f1": float(sum(m["f1"] for m in threshold_metrics) / len(threshold_metrics)),
            "iou": float(sum(m["iou"] for m in threshold_metrics) / len(threshold_metrics)),
            "mean_predicted_threshold": float(sum(m["predicted_threshold"] for m in threshold_metrics) / len(threshold_metrics)),
        }
    return {
        "overall": overall,
        "per_sample": per_sample,
    }


def main():
    out_dir = Path("./outputs/boundary_proxy_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples()
    holdout_names = {"bee", "red_tree"}
    train_samples = [s for s in samples if s["name"] not in holdout_names]
    eval_samples = [s for s in samples if s["name"] in holdout_names]

    calibrator = _fit_calibrator(train_samples)
    calibrator["ratio_predictor"] = _fit_ratio_predictor(train_samples)
    calibrator["threshold_predictor"] = _fit_threshold_predictor(train_samples, calibrator)
    report = {
        "train_names": [s["name"] for s in train_samples],
        "eval_names": [s["name"] for s in eval_samples],
        "calibrator": calibrator,
        "evaluation": _evaluate(eval_samples, calibrator),
    }

    with open(out_dir / "boundary_proxy_calibrator.json", "w", encoding="utf-8") as f:
        json.dump(calibrator, f, indent=2)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"DONE {out_dir}")


if __name__ == "__main__":
    main()
