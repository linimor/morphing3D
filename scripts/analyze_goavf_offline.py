import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GRID = 16
N = GRID ** 3
TOKEN_VOXELS = 4 ** 3


def token_index(coords16: np.ndarray) -> np.ndarray:
    coords16 = np.asarray(coords16, dtype=np.int64)
    return coords16[:, 0] * GRID * GRID + coords16[:, 1] * GRID + coords16[:, 2]


def index_to_coord(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    return np.stack([idx // (GRID * GRID), (idx // GRID) % GRID, idx % GRID], axis=1)


def neighbors(idx: int):
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    if x > 0:
        yield idx - GRID * GRID
    if x + 1 < GRID:
        yield idx + GRID * GRID
    if y > 0:
        yield idx - GRID
    if y + 1 < GRID:
        yield idx + GRID
    if z > 0:
        yield idx - 1
    if z + 1 < GRID:
        yield idx + 1


def debug_sort_key(path: Path) -> int:
    m = re.search(r"_morph(-?\d+)_step", path.name)
    return int(m.group(1)) if m else 10 ** 9


def load_endpoint(path: Path) -> dict:
    data = torch.load(path, map_location="cpu")
    coords = data["ss_token_coords"].cpu().numpy().astype(np.int64)
    occ_s = data["occ_s_16"].cpu().numpy().astype(bool)
    occ_t = data["occ_t_16"].cpu().numpy().astype(bool)
    raw = {}
    for prefix, key in (("src", "src_coords_raw"), ("tar", "tar_coords_raw")):
        val = data.get(key)
        raw[prefix] = torch.as_tensor(val).cpu().numpy().astype(np.int64) if val is not None else None
    return {"coords": coords, "occ_s": occ_s, "occ_t": occ_t, "raw": raw}


def raw_stats(raw_coords: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.full((N, 3), np.nan, dtype=np.float64)
    count = np.zeros(N, dtype=np.int64)
    fill = np.zeros(N, dtype=np.float64)
    if raw_coords is None or raw_coords.size == 0:
        return centroid, count, fill
    xyz = raw_coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    tid = token_index(token)
    sums = np.zeros((N, 3), dtype=np.float64)
    np.add.at(sums, tid, xyz.astype(np.float64) / 4.0)
    np.add.at(count, tid, 1)
    has = count > 0
    centroid[has] = sums[has] / count[has, None]
    fill = count.astype(np.float64) / float(TOKEN_VOXELS)
    return centroid, count, fill


def standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sig = x.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return (x - mu) / sig


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x)
    k = max(1, min(int(k), n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            mask = labels == c
            centers[c] = x[mask].mean(axis=0) if mask.any() else x[rng.integers(0, n)]
    return labels, centers


def knn_edges(x: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    n = len(x)
    k = max(1, min(k, max(n - 1, 1)))
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argsort(d2, axis=1)[:, :k]
    edges = {}
    for i in range(n):
        for j in nbr[i]:
            key = tuple(sorted((int(i), int(j))))
            edges[key] = min(edges.get(key, float("inf")), float(math.sqrt(d2[i, j])))
    return [(a, b, d) for (a, b), d in sorted(edges.items())]


def load_scaf_debug(debug_dir: Path) -> list[dict]:
    frames = []
    for path in sorted(debug_dir.glob("*.pt"), key=debug_sort_key):
        data = torch.load(path, map_location="cpu")
        frames.append(
            {
                "frame": debug_sort_key(path),
                "alpha": float(data["alpha"]),
                "p": float(data["p"]),
                "birth_idx": data["birth_idx"].cpu().numpy().astype(np.int64),
                "q_birth": data["q_birth"].float().cpu().numpy().astype(np.float64),
                "a_raw": data["a_raw"].float().cpu().numpy().astype(np.float64),
                "residual_norm": data["residual_norm"].float().cpu().numpy().astype(np.float64),
                "edge_pairs": data["edge_pairs"].cpu().numpy().astype(np.int64),
            }
        )
    return frames


def robust01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [5, 95]) if x.size else (0.0, 1.0)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def gaussian_integral_q(phase: np.ndarray, bandwidth: np.ndarray, p: float, bins: int = 128) -> np.ndarray:
    u = np.linspace(0.0, 1.0, bins, dtype=np.float64)
    bw = np.maximum(bandwidth, 1e-3)
    v = np.exp(-((u[None, :] - phase[:, None]) ** 2) / (2.0 * bw[:, None] ** 2))
    seg = 0.5 * (v[:, 1:] + v[:, :-1]) * (u[1:] - u[:-1])[None, :]
    cum = np.concatenate([np.zeros((len(phase), 1)), np.cumsum(seg, axis=1)], axis=1)
    total = np.maximum(cum[:, -1], 1e-8)
    idx = np.searchsorted(u, p, side="left")
    if idx <= 0:
        integ = np.zeros(len(phase))
    elif idx >= bins:
        integ = total
    else:
        u0, u1 = u[idx - 1], u[idx]
        frac = np.clip((p - u0) / max(u1 - u0, 1e-8), 0.0, 1.0)
        vp = v[:, idx - 1] + frac * (v[:, idx] - v[:, idx - 1])
        integ = cum[:, idx - 1] + 0.5 * (v[:, idx - 1] + vp) * (p - u0)
    return np.clip(integ / total, 0.0, 1.0)


def build_overlapped_anchors(desc: np.ndarray, token_ids: np.ndarray, args) -> dict:
    k = args.anchor_count or max(6, min(64, int(round(math.sqrt(len(desc))))))
    labels, centers = kmeans(desc, k, seed=args.seed)
    d2 = ((desc[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argsort(d2, axis=1)[:, : max(1, args.anchor_overlap)]
    membership = []
    token_anchor = defaultdict(list)
    anchor_tokens = defaultdict(list)
    for local_i, anchors in enumerate(nearest):
        dist = np.sqrt(np.maximum(d2[local_i, anchors], 0.0))
        raw_w = np.exp(-dist / max(np.median(dist) + 1e-6, 1e-3))
        raw_w = raw_w / max(raw_w.sum(), 1e-8)
        tid = int(token_ids[local_i])
        for aid, w in zip(anchors, raw_w):
            rec = (tid, int(aid), float(w), int(local_i))
            membership.append(rec)
            token_anchor[tid].append(rec)
            anchor_tokens[int(aid)].append(rec)
    return {
        "labels": labels,
        "centers": centers,
        "membership": membership,
        "token_anchor": token_anchor,
        "anchor_tokens": anchor_tokens,
        "anchor_count": k,
    }


def classify_anchor_support(anchor_tokens, occ_s, occ_t, coords, token_to_anchor_primary):
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    rows = []
    for aid, recs in anchor_tokens.items():
        tids = np.array([r[0] for r in recs], dtype=np.int64)
        unique = np.unique(tids)
        sr = float(shared[unique].mean()) if unique.size else 0.0
        br = float(birth[unique].mean()) if unique.size else 0.0
        adj_shared = any(birth[t] and any(shared[nb] for nb in neighbors(int(t))) for t in unique)
        typ = "mixed"
        if sr >= 0.65:
            typ = "shared-core"
        elif sr >= 0.15 and br >= 0.15:
            typ = "transition"
        elif br >= 0.5 and adj_shared:
            typ = "birth-supported"
        elif br >= 0.5:
            typ = "birth-free"
        rows.append({"anchor_id": int(aid), "shared_ratio": sr, "birth_ratio": br, "adjacent_to_shared": bool(adj_shared), "anchor_type": typ})
    support_types = {"shared-core", "transition", "birth-supported"}
    supported = {r["anchor_id"] for r in rows if r["anchor_type"] in support_types}
    return rows, supported


def anchor_graph(anchor_count, centers, anchor_tokens, support_rows, residual_anchor):
    edges = {}
    center_edges = knn_edges(centers, k=min(6, max(anchor_count - 1, 1)))
    for a, b, d in center_edges:
        edges[(a, b)] = edges.get((a, b), 0.0) + math.exp(-d)
    token_sets = {a: {r[0] for r in recs} for a, recs in anchor_tokens.items()}
    for a in range(anchor_count):
        for b in range(a + 1, anchor_count):
            ov = len(token_sets.get(a, set()) & token_sets.get(b, set()))
            if ov:
                edges[(a, b)] = edges.get((a, b), 0.0) + ov / max(len(token_sets.get(a, set()) | token_sets.get(b, set())), 1)
            res_sim = 1.0 / (1.0 + abs(residual_anchor[a] - residual_anchor[b]))
            edges[(a, b)] = edges.get((a, b), 0.0) + 0.15 * res_sim
    row_by_id = {r["anchor_id"]: r for r in support_rows}
    for (a, b), w in list(edges.items()):
        ta = row_by_id.get(a, {}).get("anchor_type", "mixed")
        tb = row_by_id.get(b, {}).get("anchor_type", "mixed")
        if (ta in {"shared-core", "transition"} and tb.startswith("birth")) or (tb in {"shared-core", "transition"} and ta.startswith("birth")):
            edges[(a, b)] = w + 0.5
    return edges


def laplacian(anchor_count: int, edges: dict[tuple[int, int], float]) -> np.ndarray:
    L = np.zeros((anchor_count, anchor_count), dtype=np.float64)
    for (a, b), w in edges.items():
        L[a, a] += w
        L[b, b] += w
        L[a, b] -= w
        L[b, a] -= w
    return L


def solve_anchor_u(r, edges, overlap_edges, support_rows, args):
    n = len(r)
    L = laplacian(n, edges)
    Lov = laplacian(n, overlap_edges)
    A = np.eye(n) * args.drive_weight + args.smooth_weight * L + args.curvature_weight * (L @ L) + args.overlap_weight * Lov
    b = args.drive_weight * r
    row_by_id = {row["anchor_id"]: row for row in support_rows}
    support_ids = [row["anchor_id"] for row in support_rows if row["anchor_type"] in {"shared-core", "transition", "birth-supported"}]
    for row in support_rows:
        aid = row["anchor_id"]
        if row["anchor_type"] != "birth-free" or not support_ids:
            continue
        best = min(support_ids, key=lambda sid: np.linalg.norm(np.array([aid]) - np.array([sid])))
        A[aid, aid] += args.support_weight
        A[aid, best] -= args.support_weight
    A += np.eye(n) * 1e-5
    u = np.linalg.solve(A, b)
    return np.clip(u, 0.0, 1.0), L


def edge_q_metrics(q: np.ndarray, birth: np.ndarray) -> dict:
    diffs = []
    for tid in np.flatnonzero(birth):
        for nb in neighbors(int(tid)):
            if nb > tid and birth[nb]:
                diffs.append(abs(q[tid] - q[nb]))
    if not diffs:
        return {"edge_q_diff_mean": None, "edge_q_diff_max": None}
    diffs = np.asarray(diffs)
    return {"edge_q_diff_mean": float(diffs.mean()), "edge_q_diff_max": float(diffs.max())}


def main():
    parser = argparse.ArgumentParser(description="Offline G-OAVF q-field analysis.")
    parser.add_argument("--endpoint", type=Path, default=ROOT / "outputs/cache/chong/cache/ss_endpoint_occ16_to_hudie.pt")
    parser.add_argument("--scaf-debug-dir", type=Path, default=ROOT / "outputs/analysis/scaf_chong_hudie_m25/SCAF_default/scaf_debug")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/goavf_offline_chong_hudie")
    parser.add_argument("--anchor-count", type=int, default=0)
    parser.add_argument("--anchor-overlap", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drive-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=0.35)
    parser.add_argument("--curvature-weight", type=float, default=0.08)
    parser.add_argument("--overlap-weight", type=float, default=0.25)
    parser.add_argument("--support-weight", type=float, default=0.20)
    parser.add_argument("--delta-scale", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ep = load_endpoint(args.endpoint)
    coords = ep["coords"]
    occ_s, occ_t = ep["occ_s"], ep["occ_t"]
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    active = shared | birth
    active_ids = np.flatnonzero(active)

    _, _, tar_fill = raw_stats(ep["raw"]["tar"])
    tar_centroid, _, _ = raw_stats(ep["raw"]["tar"])
    centroid = np.where(np.isfinite(tar_centroid), tar_centroid, coords.astype(np.float64))
    frames = load_scaf_debug(args.scaf_debug_dir)
    if not frames:
        raise FileNotFoundError(f"No SCAF debug files under {args.scaf_debug_dir}")
    residual = np.zeros(N, dtype=np.float64)
    residual[frames[len(frames) // 2]["birth_idx"]] = frames[len(frames) // 2]["residual_norm"]
    residual01 = robust01(residual)

    type_onehot = np.stack([shared.astype(float), birth.astype(float)], axis=1)
    desc = np.concatenate([coords / 15.0, centroid / 15.0, type_onehot, tar_fill[:, None], residual01[:, None]], axis=1)
    desc_active = standardize(desc[active_ids])
    anchors = build_overlapped_anchors(desc_active, active_ids, args)
    primary = np.full(N, -1, dtype=np.int64)
    primary[active_ids] = anchors["labels"]
    support_rows, supported_ids = classify_anchor_support(anchors["anchor_tokens"], occ_s, occ_t, coords, primary)

    residual_anchor = np.zeros(anchors["anchor_count"], dtype=np.float64)
    anchor_centroid = np.zeros((anchors["anchor_count"], 3), dtype=np.float64)
    for aid in range(anchors["anchor_count"]):
        recs = anchors["anchor_tokens"].get(aid, [])
        tids = np.array([r[0] for r in recs], dtype=np.int64)
        w = np.array([r[2] for r in recs], dtype=np.float64)
        if tids.size:
            residual_anchor[aid] = np.average(residual01[tids], weights=w)
            anchor_centroid[aid] = np.average(centroid[tids], axis=0, weights=w)

    edges = anchor_graph(anchors["anchor_count"], anchor_centroid, anchors["anchor_tokens"], support_rows, residual_anchor)
    overlap_edges = {}
    token_to_anchors = anchors["token_anchor"]
    for tid, recs in token_to_anchors.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = sorted((recs[i][1], recs[j][1]))
                overlap_edges[(a, b)] = overlap_edges.get((a, b), 0.0) + recs[i][2] * recs[j][2]

    q_goavf = []
    q_scaf = []
    metrics = []
    for frame in frames:
        p = frame["p"]
        q_s = np.zeros(N, dtype=np.float64)
        q_s[shared] = p
        q_s[frame["birth_idx"]] = frame["q_birth"]
        q_scaf.append(q_s)

        local_q_by_pair = {}
        anchor_drive = np.zeros(anchors["anchor_count"], dtype=np.float64)
        anchor_wsum = np.zeros(anchors["anchor_count"], dtype=np.float64)
        for aid, recs in anchors["anchor_tokens"].items():
            tids = np.array([r[0] for r in recs], dtype=np.int64)
            weights = np.array([r[2] for r in recs], dtype=np.float64)
            spatial = np.linalg.norm(centroid[tids] - anchor_centroid[aid], axis=1)
            spatial_cost = robust01(spatial)
            change_cost = residual01[tids]
            support_cost = np.array([0.0 if shared[t] else min(1.0, min((np.linalg.norm(coords[t] - coords[s]) for s in np.flatnonzero(shared)), default=6.0) / 6.0) for t in tids])
            local_cost = 0.40 * spatial_cost + 0.30 * change_cost + 0.20 * np.abs(change_cost - residual_anchor[aid]) + 0.10 * support_cost
            order = np.argsort(local_cost)
            phase = np.empty_like(local_cost)
            phase[order] = np.arange(len(local_cost), dtype=np.float64) / max(len(local_cost) - 1, 1)
            bw = np.full_like(phase, max(np.std(phase), 0.08))
            local_q = gaussian_integral_q(phase, bw, p)
            drive = np.average(local_q, weights=weights)
            anchor_drive[aid] = drive
            anchor_wsum[aid] = weights.sum()
            for rec, lq in zip(recs, local_q):
                local_q_by_pair[(rec[0], rec[1])] = float(lq)

        u, L = solve_anchor_u(anchor_drive, edges, overlap_edges, support_rows, args)
        q_g = np.zeros(N, dtype=np.float64)
        q_g[shared] = p
        for tid, recs in token_to_anchors.items():
            val = 0.0
            wsum = 0.0
            for _, aid, w, _ in recs:
                delta = args.delta_scale * (local_q_by_pair[(tid, aid)] - anchor_drive[aid])
                val += w * (u[aid] + delta)
                wsum += w
            q_g[tid] = np.clip(val / max(wsum, 1e-8), 0.0, 1.0)
        q_goavf.append(q_g)

        curv = L @ u
        free_ids = {r["anchor_id"] for r in support_rows if r["anchor_type"] == "birth-free"}
        free_birth = np.array([birth[i] and primary[i] in free_ids for i in range(N)])
        supported_birth = birth & ~free_birth
        unsupported_outer_activation = float(q_g[free_birth].mean()) if free_birth.any() else 0.0
        supported_activation = float(q_g[supported_birth].mean()) if supported_birth.any() else 0.0
        row = {
            "frame": frame["frame"],
            "alpha": frame["alpha"],
            "p": p,
            **{f"scaf_{k}": v for k, v in edge_q_metrics(q_s, birth).items()},
            **{f"goavf_{k}": v for k, v in edge_q_metrics(q_g, birth).items()},
            "graph_curvature_mean": float(np.abs(curv).mean()),
            "graph_curvature_max": float(np.abs(curv).max()),
            "unsupported_outer_activation": unsupported_outer_activation,
            "supported_birth_activation": supported_activation,
            "unsupported_minus_supported": unsupported_outer_activation - supported_activation,
        }
        metrics.append(row)

    q_goavf = np.stack(q_goavf)
    q_scaf = np.stack(q_scaf)
    torch.save(
        {
            "frames": [f["frame"] for f in frames],
            "alpha": torch.tensor([f["alpha"] for f in frames]),
            "p": torch.tensor([f["p"] for f in frames]),
            "q_goavf": torch.from_numpy(q_goavf).float(),
            "q_scaf": torch.from_numpy(q_scaf).float(),
            "shared_mask": torch.from_numpy(shared),
            "birth_mask": torch.from_numpy(birth),
            "anchor_rows": support_rows,
            "anchor_edges": [{"source": a, "target": b, "weight": w} for (a, b), w in sorted(edges.items())],
        },
        args.out_dir / "q_field_per_alpha.pt",
    )

    write_visual(args.out_dir / "q_field_visualization.png", coords, birth, shared, frames, q_scaf, q_goavf, support_rows, primary)
    mean_metrics = summarize_metrics(metrics)
    risk = {
        "metrics_per_alpha": metrics,
        "summary": mean_metrics,
        "hollow_ring_risk": bool(
            mean_metrics["unsupported_outer_activation_mean"] > mean_metrics["supported_birth_activation_mean"] + 0.10
            or mean_metrics["unsupported_minus_supported_max"] > 0.20
        ),
        "interpretation": "High unsupported_outer_activation before supported anchors is a birth-free early activation signal.",
    }
    (args.out_dir / "hollow_ring_risk_report.json").write_text(json.dumps(risk, indent=2), encoding="utf-8")
    print(f"G-OAVF offline analysis written to: {args.out_dir}")
    print(json.dumps(mean_metrics, indent=2))


def summarize_metrics(metrics):
    keys = [
        "scaf_edge_q_diff_mean",
        "scaf_edge_q_diff_max",
        "goavf_edge_q_diff_mean",
        "goavf_edge_q_diff_max",
        "graph_curvature_mean",
        "graph_curvature_max",
        "unsupported_outer_activation",
        "supported_birth_activation",
        "unsupported_minus_supported",
    ]
    out = {}
    for key in keys:
        vals = [m[key] for m in metrics if m.get(key) is not None]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_max"] = float(np.max(vals)) if vals else None
    return out


def write_visual(path, coords, birth, shared, frames, q_scaf, q_goavf, support_rows, primary):
    target_idx = min(range(len(frames)), key=lambda i: abs(frames[i]["p"] - 0.5))
    fig = plt.figure(figsize=(12, 5), dpi=170)
    for panel, (name, q) in enumerate((("SCAF", q_scaf[target_idx]), ("G-OAVF", q_goavf[target_idx])), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ids = np.flatnonzero(shared | birth)
        pts = coords[ids]
        colors = q[ids]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, cmap="viridis", s=14, vmin=0, vmax=1, alpha=0.85)
        ax.set_title(f"{name} q field, frame {frames[target_idx]['frame']}, p={frames[target_idx]['p']:.2f}")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_zlim(0, 15)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=42)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
