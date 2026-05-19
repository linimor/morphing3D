import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GRID = 16
RAW_GRID = 64
TOKEN_VOXELS = 4 ** 3
N = GRID ** 3
DELTAS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


def token_index(coords16: np.ndarray) -> np.ndarray:
    coords16 = np.asarray(coords16, dtype=np.int64)
    return coords16[:, 0] * GRID * GRID + coords16[:, 1] * GRID + coords16[:, 2]


def index_to_coord(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    return np.stack([x, y, z], axis=1)


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


def coords_file_sort_key(path: Path) -> int:
    m = re.search(r"coords_morphing(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 10 ** 9


def debug_file_sort_key(path: Path) -> int:
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
    return {"coords": coords, "occ_s": occ_s, "occ_t": occ_t, "raw": raw, "path": str(path)}


def raw_token_stats(raw_coords: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_residual_norm(debug_dir: Path | None, token_count: int) -> np.ndarray:
    out = np.zeros(token_count, dtype=np.float64)
    if debug_dir is None or not debug_dir.exists():
        return out
    files = sorted(debug_dir.glob("*.pt"), key=debug_file_sort_key)
    if not files:
        return out
    mid = files[len(files) // 2]
    data = torch.load(mid, map_location="cpu")
    birth_idx = data.get("birth_idx")
    residual_norm = data.get("residual_norm")
    if torch.is_tensor(birth_idx) and torch.is_tensor(residual_norm):
        out[birth_idx.cpu().numpy().astype(np.int64)] = residual_norm.cpu().numpy().astype(np.float64)
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sig = x.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return (x - mu) / sig


def build_knn_graph(desc: np.ndarray, k: int) -> tuple[list[list[int]], list[dict]]:
    n = desc.shape[0]
    k = max(1, min(int(k), max(n - 1, 1)))
    d2 = ((desc[:, None, :] - desc[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    adj = [set() for _ in range(n)]
    edges = {}
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
            key = tuple(sorted((i, j)))
            edges[key] = min(float(math.sqrt(d2[i, j])), edges.get(key, float("inf")))
    return [sorted(v) for v in adj], [
        {"source": int(i), "target": int(j), "distance": float(d)} for (i, j), d in sorted(edges.items())
    ]


def kmeans(desc: np.ndarray, cluster_count: int, seed: int = 0, iters: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = desc.shape[0]
    cluster_count = max(1, min(cluster_count, n))
    centers = desc[rng.choice(n, size=cluster_count, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((desc[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(cluster_count):
            mask = labels == c
            if mask.any():
                centers[c] = desc[mask].mean(axis=0)
            else:
                centers[c] = desc[rng.integers(0, n)]
    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[int(v)] for v in labels], dtype=np.int64)


def component_clusters(knn_adj: list[list[int]]) -> np.ndarray:
    n = len(knn_adj)
    labels = np.full(n, -1, dtype=np.int64)
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        q = deque([i])
        labels[i] = cur
        while q:
            node = q.popleft()
            for nb in knn_adj[node]:
                if labels[nb] < 0:
                    labels[nb] = cur
                    q.append(nb)
        cur += 1
    return labels


def load_frame_occ(path: Path) -> np.ndarray:
    coords = torch.load(path, map_location="cpu")
    if isinstance(coords, dict):
        for key in ("coords", "coords_morphing", "active_coords"):
            if key in coords:
                coords = coords[key]
                break
    coords = torch.as_tensor(coords).cpu().numpy()
    xyz = coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    occ = np.zeros(N, dtype=bool)
    occ[token_index(token)] = True
    return occ


def find_burst(method_dir: Path, birth: np.ndarray) -> dict:
    rows = []
    for path in sorted(method_dir.glob("coords_morphing*.pt"), key=coords_file_sort_key):
        frame = coords_file_sort_key(path)
        occ = load_frame_occ(path)
        active_birth = occ & birth
        rows.append({"frame": frame, "path": path, "occ": occ, "active_birth": active_birth})
    if len(rows) < 2:
        return {"rows": rows, "burst": None}
    jumps = []
    denom = max(int(birth.sum()), 1)
    for prev, cur in zip(rows, rows[1:]):
        new_birth = cur["active_birth"] & ~prev["active_birth"]
        jump = float(new_birth.sum() / denom)
        jumps.append((jump, prev, cur, new_birth))
    jump, prev, cur, new_birth = max(jumps, key=lambda x: x[0])
    return {
        "rows": rows,
        "burst": {
            "jump": jump,
            "frame_before": int(prev["frame"]),
            "frame_after": int(cur["frame"]),
            "new_birth": new_birth,
            "active_birth_after": cur["active_birth"],
        },
    }


def classify_anchors(anchor_rows: list[dict], anchor_edges: set[tuple[int, int]]) -> None:
    transition_ids = set()
    shared_like_ids = set()
    for row in anchor_rows:
        if row["shared_ratio"] >= 0.65:
            row["anchor_type"] = "shared-core"
            shared_like_ids.add(row["anchor_id"])
        elif row["shared_ratio"] >= 0.15 and row["birth_ratio"] >= 0.15:
            row["anchor_type"] = "transition"
            transition_ids.add(row["anchor_id"])
        else:
            row["anchor_type"] = "pending"
    support_ids = shared_like_ids | transition_ids
    changed = True
    while changed:
        changed = False
        for row in anchor_rows:
            if row["anchor_type"] != "pending" or row["birth_ratio"] < 0.5:
                continue
            aid = row["anchor_id"]
            touches = any((min(aid, s), max(aid, s)) in anchor_edges for s in support_ids)
            if row["adjacent_to_shared"] or touches:
                row["anchor_type"] = "birth-supported"
                support_ids.add(aid)
                changed = True
    for row in anchor_rows:
        if row["anchor_type"] == "pending":
            row["anchor_type"] = "birth-free" if row["birth_ratio"] >= 0.5 else "mixed-other"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_anchors(path: Path, anchor_rows: list[dict], token_to_anchor: np.ndarray, cluster_token_ids: np.ndarray) -> None:
    colors = {
        "shared-core": "#4c78a8",
        "transition": "#f58518",
        "birth-supported": "#54a24b",
        "birth-free": "#e45756",
        "mixed-other": "#b279a2",
    }
    fig = plt.figure(figsize=(8, 7), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    anchor_type = {r["anchor_id"]: r["anchor_type"] for r in anchor_rows}
    for typ, color in colors.items():
        aids = {aid for aid, t in anchor_type.items() if t == typ}
        mask = np.array([token_to_anchor[int(t)] in aids for t in cluster_token_ids], dtype=bool)
        pts = index_to_coord(cluster_token_ids[mask])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, c=color, label=typ, alpha=0.8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=42)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-guided anchor analysis for SS shared/birth structure.")
    parser.add_argument("--endpoint", type=Path, default=ROOT / "outputs/cache/chong/cache/ss_endpoint_occ16_to_hudie.pt")
    parser.add_argument("--method-dir", type=Path, default=ROOT / "outputs/analysis/scaf_chong_hudie_m25/SCAF_default")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "outputs/analysis/scaf_chong_hudie_m25/SCAF_default/scaf_debug")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/cluster_guided_anchors_chong_hudie")
    parser.add_argument("--knn-k", type=int, default=12)
    parser.add_argument("--clusters", type=int, default=0, help="0 means sqrt token-count heuristic.")
    parser.add_argument("--cluster-method", choices=("kmeans", "components"), default="kmeans")
    parser.add_argument("--early-frame-ratio", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = load_endpoint(args.endpoint)
    coords = endpoint["coords"]
    occ_s = endpoint["occ_s"]
    occ_t = endpoint["occ_t"]
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    death = occ_s & ~occ_t
    target = occ_t

    src_centroid, src_count, src_fill = raw_token_stats(endpoint["raw"]["src"])
    tar_centroid, tar_count, tar_fill = raw_token_stats(endpoint["raw"]["tar"])
    coord_fallback = coords.astype(np.float64)
    tar_centroid_filled = np.where(np.isfinite(tar_centroid), tar_centroid, coord_fallback)
    src_centroid_filled = np.where(np.isfinite(src_centroid), src_centroid, coord_fallback)
    residual_norm = load_residual_norm(args.debug_dir, len(coords))

    cluster_mask = shared | birth
    cluster_token_ids = np.flatnonzero(cluster_mask)
    type_onehot = np.stack([shared.astype(float), birth.astype(float), death.astype(float)], axis=1)
    desc = np.concatenate(
        [
            coords.astype(np.float64) / float(GRID - 1),
            tar_centroid_filled / float(GRID - 1),
            type_onehot,
            tar_fill[:, None],
            residual_norm[:, None],
        ],
        axis=1,
    )
    desc_active = standardize(desc[cluster_token_ids])
    knn_adj, knn_edges = build_knn_graph(desc_active, args.knn_k)

    if args.cluster_method == "components":
        labels = component_clusters(knn_adj)
    else:
        cluster_count = args.clusters or max(4, min(64, int(round(math.sqrt(len(cluster_token_ids))))))
        labels = kmeans(desc_active, cluster_count=cluster_count)

    token_to_anchor = np.full(N, -1, dtype=np.int64)
    token_to_anchor[cluster_token_ids] = labels
    anchor_count = int(labels.max() + 1) if labels.size else 0

    anchor_edges = set()
    for e in knn_edges:
        a = int(labels[e["source"]])
        b = int(labels[e["target"]])
        if a != b:
            anchor_edges.add((min(a, b), max(a, b)))

    death_ids = np.flatnonzero(death)
    if death_ids.size and anchor_count:
        anchor_centers = np.stack([
            coords[cluster_token_ids[labels == aid]].mean(axis=0) for aid in range(anchor_count)
        ])
        d2 = ((coords[death_ids, None, :] - anchor_centers[None, :, :]) ** 2).sum(axis=2)
        death_anchor = d2.argmin(axis=1)
    else:
        death_anchor = np.empty(0, dtype=np.int64)

    anchor_rows = []
    for aid in range(anchor_count):
        token_ids = cluster_token_ids[labels == aid]
        assigned_death = death_ids[death_anchor == aid] if death_ids.size else np.empty(0, dtype=np.int64)
        stat_ids = np.concatenate([token_ids, assigned_death])
        total = max(len(stat_ids), 1)
        adjacent_to_shared = False
        for tid in token_ids:
            if birth[tid] and any(shared[nb] for nb in neighbors(int(tid))):
                adjacent_to_shared = True
                break
        row = {
            "anchor_id": aid,
            "token_count": int(len(stat_ids)),
            "clustered_token_count": int(len(token_ids)),
            "death_assigned_count": int(len(assigned_death)),
            "shared_ratio": float(shared[stat_ids].sum() / total),
            "birth_ratio": float(birth[stat_ids].sum() / total),
            "death_ratio": float(death[stat_ids].sum() / total),
            "target_fill_ratio_mean": float(tar_fill[token_ids].mean()) if len(token_ids) else 0.0,
            "centroid_x": float(coords[token_ids, 0].mean()) if len(token_ids) else 0.0,
            "centroid_y": float(coords[token_ids, 1].mean()) if len(token_ids) else 0.0,
            "centroid_z": float(coords[token_ids, 2].mean()) if len(token_ids) else 0.0,
            "raw_centroid_x": float(tar_centroid_filled[token_ids, 0].mean()) if len(token_ids) else 0.0,
            "raw_centroid_y": float(tar_centroid_filled[token_ids, 1].mean()) if len(token_ids) else 0.0,
            "raw_centroid_z": float(tar_centroid_filled[token_ids, 2].mean()) if len(token_ids) else 0.0,
            "adjacent_to_shared": bool(adjacent_to_shared),
            "anchor_type": "pending",
        }
        anchor_rows.append(row)

    classify_anchors(anchor_rows, anchor_edges)
    write_csv(args.out_dir / "anchor_summary.csv", anchor_rows)

    anchor_graph = {
        "endpoint": endpoint["path"],
        "method_dir": str(args.method_dir),
        "cluster_method": args.cluster_method,
        "knn_k": args.knn_k,
        "anchor_count": anchor_count,
        "nodes": anchor_rows,
        "edges": [
            {"source": int(a), "target": int(b)} for a, b in sorted(anchor_edges)
        ],
    }
    (args.out_dir / "anchor_graph.json").write_text(json.dumps(anchor_graph, indent=2), encoding="utf-8")
    plot_anchors(args.out_dir / "anchor_types_3d.png", anchor_rows, token_to_anchor, cluster_token_ids)

    burst_info = find_burst(args.method_dir, birth)
    burst = burst_info["burst"]
    report = {
        "method_dir": str(args.method_dir),
        "endpoint": endpoint["path"],
        "birth_count": int(birth.sum()),
        "shared_count": int(shared.sum()),
        "death_count": int(death.sum()),
        "anchor_count": anchor_count,
    }
    if burst is not None:
        new_birth_ids = np.flatnonzero(burst["new_birth"])
        active_after_ids = np.flatnonzero(burst["active_birth_after"])
        by_anchor = summarize_birth_by_anchor(new_birth_ids, active_after_ids, token_to_anchor, anchor_rows)
        early_limit = max(1, int(round(len(burst_info["rows"]) * args.early_frame_ratio)))
        early_frames = {r["frame"] for r in burst_info["rows"][:early_limit]}
        early_birth_free = summarize_early_birth_free(burst_info["rows"], token_to_anchor, anchor_rows, birth, early_frames)
        report.update(
            {
                "max_burst_frame_before": burst["frame_before"],
                "max_burst_frame_after": burst["frame_after"],
                "birth_activation_max_jump": burst["jump"],
                "new_birth_token_count": int(new_birth_ids.size),
                "new_birth_by_anchor_type": by_anchor["new_birth_by_type"],
                "new_birth_by_anchor": by_anchor["new_birth_by_anchor"],
                "active_birth_after_by_anchor_type": by_anchor["active_after_by_type"],
                "birth_free_new_birth_ratio": by_anchor["new_birth_by_type"].get("birth-free", 0) / max(int(new_birth_ids.size), 1),
                "early_frame_ratio": args.early_frame_ratio,
                "early_frames": sorted(int(v) for v in early_frames),
                "birth_free_early_active_count": early_birth_free["birth_free_early_active_count"],
                "birth_free_early_active_ratio_of_birth_free": early_birth_free["birth_free_early_active_ratio_of_birth_free"],
                "birth_free_early_active_ratio_of_all_birth": early_birth_free["birth_free_early_active_ratio_of_all_birth"],
                "birth_free_early_activation_suspected": bool(
                    early_birth_free["birth_free_early_active_ratio_of_birth_free"] >= 0.25
                    or by_anchor["new_birth_by_type"].get("birth-free", 0) / max(int(new_birth_ids.size), 1) >= 0.25
                ),
            }
        )
    else:
        report["note"] = "Not enough coords_morphing frames to compute burst."
    (args.out_dir / "burst_anchor_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"cluster-guided anchor analysis written to: {args.out_dir}")
    if burst is not None:
        print(
            f"max burst {report['max_burst_frame_before']}->{report['max_burst_frame_after']}, "
            f"jump={report['birth_activation_max_jump']:.4f}, "
            f"birth-free new ratio={report['birth_free_new_birth_ratio']:.4f}, "
            f"early suspected={report['birth_free_early_activation_suspected']}"
        )


def summarize_birth_by_anchor(new_birth_ids, active_after_ids, token_to_anchor, anchor_rows):
    type_by_anchor = {r["anchor_id"]: r["anchor_type"] for r in anchor_rows}
    new_by_type = defaultdict(int)
    active_by_type = defaultdict(int)
    new_by_anchor = defaultdict(int)
    for tid in new_birth_ids:
        aid = int(token_to_anchor[tid])
        typ = type_by_anchor.get(aid, "unassigned")
        new_by_type[typ] += 1
        new_by_anchor[str(aid)] += 1
    for tid in active_after_ids:
        aid = int(token_to_anchor[tid])
        typ = type_by_anchor.get(aid, "unassigned")
        active_by_type[typ] += 1
    return {
        "new_birth_by_type": dict(sorted(new_by_type.items())),
        "active_after_by_type": dict(sorted(active_by_type.items())),
        "new_birth_by_anchor": dict(sorted(new_by_anchor.items(), key=lambda kv: int(kv[0]))),
    }


def summarize_early_birth_free(rows, token_to_anchor, anchor_rows, birth, early_frames):
    birth_free_anchors = {r["anchor_id"] for r in anchor_rows if r["anchor_type"] == "birth-free"}
    birth_free_tokens = np.flatnonzero(np.array([token_to_anchor[i] in birth_free_anchors for i in range(N)]) & birth)
    early_active = np.zeros(N, dtype=bool)
    for row in rows:
        if row["frame"] in early_frames:
            early_active |= row["active_birth"]
    birth_free_early = early_active[birth_free_tokens]
    return {
        "birth_free_early_active_count": int(birth_free_early.sum()),
        "birth_free_early_active_ratio_of_birth_free": float(birth_free_early.mean()) if birth_free_tokens.size else 0.0,
        "birth_free_early_active_ratio_of_all_birth": float(birth_free_early.sum() / max(int(birth.sum()), 1)),
    }


if __name__ == "__main__":
    main()
