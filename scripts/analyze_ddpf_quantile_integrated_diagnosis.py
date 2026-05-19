import csv
import json
import math
import re
from collections import deque
from itertools import permutations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/analysis/ddpf_quantile_integrated_diagnosis_chong_hudie_m25"

METHODS = {
    "baseline": ROOT / "outputs/analysis/ss_decode_occ_compare_chong_to_hudie_m25/baseline",
    "A": ROOT / "outputs/analysis/ddpf_quantile_lambda_tau_scan_chong_hudie_m25/A_lambda0p15_tau0p10",
    "B": ROOT / "outputs/analysis/ddpf_quantile_lambda_tau_scan_chong_hudie_m25/B_lambda0p20_tau0p10",
    "C": ROOT / "outputs/analysis/ddpf_quantile_lambda_tau_scan_chong_hudie_m25/C_lambda0p30_tau0p10",
    "D": ROOT / "outputs/analysis/ddpf_quantile_lambda_tau_scan_chong_hudie_m25/D_lambda0p20_tau0p14",
}

GRID = 16
N = GRID**3
NEIGHBOR_DELTAS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


def load_endpoint(method_dir):
    path = method_dir / "ss_diff_debug/ss_endpoint_occ16.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing endpoint occ file: {path}")
    data = torch.load(path, map_location="cpu")
    coords = data["ss_token_coords"].cpu().numpy().astype(np.int64)
    occ_s = data["occ_s_16"].cpu().numpy().astype(bool)
    occ_t = data["occ_t_16"].cpu().numpy().astype(bool)
    return {"path": str(path), "coords": coords, "occ_s": occ_s, "occ_t": occ_t}


def coord_to_index(coords):
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * GRID * GRID + coords[:, 1] * GRID + coords[:, 2]


def index_to_coord(idx):
    idx = np.asarray(idx, dtype=np.int64)
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    return np.stack([x, y, z], axis=1)


def neighbors_of_index(idx):
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    out = []
    if x > 0:
        out.append(idx - GRID * GRID)
    if x + 1 < GRID:
        out.append(idx + GRID * GRID)
    if y > 0:
        out.append(idx - GRID)
    if y + 1 < GRID:
        out.append(idx + GRID)
    if z > 0:
        out.append(idx - 1)
    if z + 1 < GRID:
        out.append(idx + 1)
    return out


def distance_to_mask(mask):
    dist = np.full(N, np.inf, dtype=np.float64)
    q = deque()
    for idx in np.flatnonzero(mask):
        dist[idx] = 0.0
        q.append(int(idx))
    while q:
        idx = q.popleft()
        nd = dist[idx] + 1.0
        for nb in neighbors_of_index(idx):
            if nd < dist[nb]:
                dist[nb] = nd
                q.append(nb)
    return dist


def phi_smoothness_for_endpoint(endpoint, label):
    occ_s = endpoint["occ_s"]
    occ_t = endpoint["occ_t"]
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    dist = distance_to_mask(shared)
    birth_idx = np.flatnonzero(birth)
    order = np.lexsort((birth_idx, dist[birth_idx]))
    phi = np.full(N, np.nan, dtype=np.float64)
    if len(birth_idx) == 1:
        phi[birth_idx[order]] = 0.0
    elif len(birth_idx) > 1:
        phi[birth_idx[order]] = np.arange(len(birth_idx), dtype=np.float64) / float(len(birth_idx) - 1)

    diffs = []
    edge_count = 0
    for idx in birth_idx:
        for nb in neighbors_of_index(int(idx)):
            if nb > idx and birth[nb]:
                edge_count += 1
                diffs.append(abs(phi[idx] - phi[nb]))
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) == 0:
        stats = {
            "edge_phi_diff_mean": None,
            "edge_phi_diff_max": None,
            "edge_phi_diff_p95": None,
            "large_phi_jump_edge_count": 0,
            "large_phi_jump_edge_ratio": None,
        }
    else:
        stats = {
            "edge_phi_diff_mean": float(diffs.mean()),
            "edge_phi_diff_max": float(diffs.max()),
            "edge_phi_diff_p95": float(np.percentile(diffs, 95)),
            "large_phi_jump_edge_count": int((diffs > 0.2).sum()),
            "large_phi_jump_edge_ratio": float((diffs > 0.2).mean()),
        }
    stats.update(
        {
            "label": label,
            "endpoint_path": endpoint["path"],
            "shared_count": int(shared.sum()),
            "birth_count": int(birth.sum()),
            "death_count": int((occ_s & ~occ_t).sum()),
            "birth_graph_edge_count": int(edge_count),
            "phi_source": "recomputed_quantile_distance_to_shared",
        }
    )
    return stats


def coords_file_sort_key(path):
    m = re.search(r"coords_morphing(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 10**9


def load_frame_occ(path):
    coords = torch.load(path, map_location="cpu")
    if isinstance(coords, dict):
        for key in ("coords", "coords_morphing", "active_coords"):
            if key in coords:
                coords = coords[key]
                break
    coords = torch.as_tensor(coords).cpu().numpy()
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"unexpected coords shape in {path}: {coords.shape}")
    xyz = coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    occ = np.zeros(N, dtype=bool)
    occ[coord_to_index(token)] = True
    return occ


def analyze_components(active_birth, active_shared):
    active_birth_idx = np.flatnonzero(active_birth)
    active_shared_set = set(np.flatnonzero(active_shared).tolist())
    visited = np.zeros(N, dtype=bool)
    component_count = 0
    isolated_component_count = 0
    connected_birth_count = 0
    boundary_contact_count = 0
    min_dist = None

    # Multi-source BFS distance from active_shared through the full grid.
    if active_birth_idx.size and active_shared_set:
        dist = np.full(N, -1, dtype=np.int32)
        q = deque()
        for idx in active_shared_set:
            dist[idx] = 0
            q.append(idx)
        while q:
            idx = q.popleft()
            for nb in neighbors_of_index(idx):
                if dist[nb] < 0:
                    dist[nb] = dist[idx] + 1
                    q.append(nb)
        vals = dist[active_birth_idx]
        vals = vals[vals >= 0]
        min_dist = int(vals.min()) if vals.size else None

    for idx in active_birth_idx:
        idx = int(idx)
        if visited[idx]:
            continue
        component_count += 1
        q = deque([idx])
        visited[idx] = True
        component = []
        touches_shared = False
        contacts = 0
        while q:
            cur = q.popleft()
            component.append(cur)
            for nb in neighbors_of_index(cur):
                if active_shared[nb]:
                    touches_shared = True
                    contacts += 1
                if active_birth[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        boundary_contact_count += contacts
        if touches_shared:
            connected_birth_count += len(component)
        else:
            isolated_component_count += 1

    return {
        "active_birth_component_count": int(component_count),
        "connected_to_active_shared_birth_count": int(connected_birth_count),
        "connected_to_active_shared_ratio": float(connected_birth_count / active_birth_idx.size)
        if active_birth_idx.size
        else 0.0,
        "isolated_birth_component_count": int(isolated_component_count),
        "boundary_contact_count": int(boundary_contact_count),
        "min_graph_distance_active_birth_to_active_shared": min_dist,
    }


def analyze_method(method, method_dir, endpoint):
    occ_s = endpoint["occ_s"]
    occ_t = endpoint["occ_t"]
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    death = occ_s & ~occ_t
    files = sorted(method_dir.glob("coords_morphing*.pt"), key=coords_file_sort_key)
    rows = []
    for path in files:
        frame_idx = coords_file_sort_key(path)
        occ = load_frame_occ(path)
        active_shared = occ & shared
        active_birth = occ & birth
        row = {
            "method": method,
            "frame_idx": frame_idx,
            "alpha": frame_idx / 25.0,
            "total_occ_count": int(occ.sum()),
            "active_birth_count": int(active_birth.sum()),
            "active_shared_count": int(active_shared.sum()),
            "active_birth_ratio": float(active_birth.sum() / max(int(occ.sum()), 1)),
            "shared_keep_ratio": float(active_shared.sum() / max(int(shared.sum()), 1)),
            "birth_activation_ratio": float(active_birth.sum() / max(int(birth.sum()), 1)),
            "death_remaining_ratio": float((occ & death).sum() / max(int(death.sum()), 1)),
        }
        row.update(analyze_components(active_birth, active_shared))
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def burst_for_rows(rows):
    rows = sorted(rows, key=lambda r: r["frame_idx"])
    jumps = []
    for before, after in zip(rows, rows[1:]):
        jumps.append((after["birth_activation_ratio"] - before["birth_activation_ratio"], before, after))
    if not jumps:
        return None
    jump, before, after = max(jumps, key=lambda x: x[0])
    fields = [
        ("alpha", "alpha"),
        ("birth_ratio", "birth_activation_ratio"),
        ("connected_to_active_shared_ratio", "connected_to_active_shared_ratio"),
        ("isolated_birth_component_count", "isolated_birth_component_count"),
        ("boundary_contact_count", "boundary_contact_count"),
        ("min_graph_distance", "min_graph_distance_active_birth_to_active_shared"),
    ]
    out = {
        "jump": float(jump),
        "frame_before": int(before["frame_idx"]),
        "frame_after": int(after["frame_idx"]),
    }
    for name, key in fields:
        out[f"{name}_before"] = before[key]
        out[f"{name}_after"] = after[key]
    connected_high = after["connected_to_active_shared_ratio"] >= 0.75
    isolated_low = after["isolated_birth_component_count"] <= max(1, before["isolated_birth_component_count"] + 1)
    contact_ok = after["boundary_contact_count"] >= max(1, before["boundary_contact_count"] // 2)
    out["ss_continuous_at_burst"] = bool(connected_high and isolated_low and contact_ok)
    if out["ss_continuous_at_burst"]:
        out["interpretation"] = "birth jump remains attached to active_shared; SS occupancy growth looks spatially continuous"
    else:
        out["interpretation"] = "birth jump shows weaker shared attachment or more isolated components; SS alpha field may still be spatially discontinuous"
    return out


def pca_axes(occ):
    coords = index_to_coord(np.flatnonzero(occ)).astype(np.float64)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = coords.T @ coords / max(len(coords) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def contact_edges(a, b):
    count = 0
    for idx in np.flatnonzero(a):
        for nb in neighbors_of_index(int(idx)):
            if b[nb]:
                count += 1
    return int(count)


def chamfer_like(a, b):
    ca = index_to_coord(np.flatnonzero(a)).astype(np.float64)
    cb = index_to_coord(np.flatnonzero(b)).astype(np.float64)
    if len(ca) == 0 or len(cb) == 0:
        return None
    d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    return float((np.sqrt(d2.min(axis=1)).mean() + np.sqrt(d2.min(axis=0)).mean()) / 2.0)


def cube_rotations():
    rots = []
    for perm in permutations(range(3)):
        pmat = np.zeros((3, 3), dtype=int)
        for out_axis, in_axis in enumerate(perm):
            pmat[out_axis, in_axis] = 1
        for signs in product([-1, 1], repeat=3):
            mat = pmat * np.asarray(signs)[:, None]
            if round(np.linalg.det(mat)) == 1:
                rots.append(mat)
    return rots


def rotate_occ(occ, mat):
    coords = index_to_coord(np.flatnonzero(occ))
    centered2 = 2 * coords - (GRID - 1)
    rotated2 = centered2 @ mat.T
    rotated = ((rotated2 + (GRID - 1)) // 2).astype(np.int64)
    out = np.zeros(N, dtype=bool)
    out[coord_to_index(rotated)] = True
    return out


def orientation_report(endpoint):
    occ_s = endpoint["occ_s"]
    occ_t = endpoint["occ_t"]
    _, axes_s = pca_axes(occ_s)
    _, axes_t = pca_axes(occ_t)
    pca_main_axis_abs_cosine = float(abs(np.dot(axes_s[:, 0], axes_t[:, 0])))

    original = {
        "shared_iou": iou(occ_t, occ_s),
        "boundary_contact": contact_edges(occ_t, occ_s),
        "chamfer_like": chamfer_like(occ_t, occ_s),
    }
    scores = []
    for idx, mat in enumerate(cube_rotations()):
        rot_t = rotate_occ(occ_t, mat)
        score = {
            "rotation_index": idx,
            "matrix": mat.tolist(),
            "shared_iou": iou(rot_t, occ_s),
            "boundary_contact": contact_edges(rot_t, occ_s),
            "chamfer_like": chamfer_like(rot_t, occ_s),
        }
        scores.append(score)
    best = max(scores, key=lambda x: (x["shared_iou"], x["boundary_contact"], -x["chamfer_like"]))
    gain = best["shared_iou"] - original["shared_iou"]
    return {
        "endpoint_path": endpoint["path"],
        "pca_main_axis_abs_cosine": pca_main_axis_abs_cosine,
        "original_direction": original,
        "best_rotation_direction": best,
        "best_rotation_index": int(best["rotation_index"]),
        "orientation_gain": float(gain),
        "orientation_mismatch_suspected": bool(gain >= 0.08 or best["shared_iou"] >= original["shared_iou"] * 1.5),
        "rotation_score_basis": "best by shared_iou, then boundary_contact, then lower chamfer_like",
    }


def plot_metric(all_rows, methods, metric, ylabel, out_name):
    plt.figure(figsize=(8, 5), dpi=150)
    for method in methods:
        rows = sorted(all_rows[method], key=lambda r: r["frame_idx"])
        plt.plot([r["alpha"] for r in rows], [r[metric] for r in rows], marker="o", linewidth=1.8, label=method)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name)
    plt.close()


def make_contact_sheet_b(endpoint, rows):
    burst = burst_for_rows(rows)
    if not burst:
        return
    frames = [burst["frame_before"], burst["frame_after"]]
    method_dir = METHODS["B"]
    shared = endpoint["occ_s"] & endpoint["occ_t"]
    birth = endpoint["occ_t"] & ~endpoint["occ_s"]
    fig = plt.figure(figsize=(12, 5), dpi=150)
    for i, frame in enumerate(frames, start=1):
        occ = load_frame_occ(method_dir / f"coords_morphing{frame}.pt")
        active_shared = occ & shared
        active_birth = occ & birth
        connected = np.zeros(N, dtype=bool)
        visited = np.zeros(N, dtype=bool)
        for idx in np.flatnonzero(active_birth):
            idx = int(idx)
            if visited[idx]:
                continue
            q = deque([idx])
            visited[idx] = True
            comp = []
            touches = False
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in neighbors_of_index(cur):
                    if active_shared[nb]:
                        touches = True
                    if active_birth[nb] and not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            if touches:
                connected[comp] = True
        isolated = active_birth & ~connected
        ax = fig.add_subplot(1, 2, i, projection="3d")
        for mask, color, label, size in [
            (active_shared, "#9aa0a6", "active shared", 14),
            (connected, "#1f77b4", "birth connected", 22),
            (isolated, "#d62728", "birth isolated", 30),
        ]:
            pts = index_to_coord(np.flatnonzero(mask))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, label=label, alpha=0.85)
        row = next(r for r in rows if r["frame_idx"] == frame)
        ax.set_title(
            f"B frame {frame}, alpha={row['alpha']:.2f}\n"
            f"birth={row['birth_activation_ratio']:.2f}, conn={row['connected_to_active_shared_ratio']:.2f}, "
            f"iso={row['isolated_birth_component_count']}"
        )
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_zlim(0, 15)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=38)
        ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "diagnosis_contact_sheet_B_burst_before_after.png")
    plt.close()


def summarize_method(rows):
    if not rows:
        return {}
    burst = burst_for_rows(rows)
    return {
        "frame_count": len(rows),
        "max_birth_activation_ratio": max(r["birth_activation_ratio"] for r in rows),
        "mean_connected_to_active_shared_ratio": float(np.mean([r["connected_to_active_shared_ratio"] for r in rows])),
        "min_connected_to_active_shared_ratio": min(r["connected_to_active_shared_ratio"] for r in rows),
        "mean_isolated_birth_component_count": float(np.mean([r["isolated_birth_component_count"] for r in rows])),
        "max_isolated_birth_component_count": max(r["isolated_birth_component_count"] for r in rows),
        "mean_boundary_contact_count": float(np.mean([r["boundary_contact_count"] for r in rows])),
        "max_birth_jump": burst["jump"] if burst else None,
        "burst_frame_before": burst["frame_before"] if burst else None,
        "burst_frame_after": burst["frame_after"] if burst else None,
        "burst_connected_ratio_after": burst["connected_to_active_shared_ratio_after"] if burst else None,
        "burst_isolated_components_after": burst["isolated_birth_component_count_after"] if burst else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoints = {name: load_endpoint(path) for name, path in METHODS.items()}

    phi_stats = {
        "primary_endpoint_for_interpretation": "B",
        "large_jump_threshold": 0.2,
        "methods": {name: phi_smoothness_for_endpoint(ep, name) for name, ep in endpoints.items()},
    }
    (OUT_DIR / "phi_smoothness_quantile.json").write_text(json.dumps(phi_stats, indent=2), encoding="utf-8")

    all_rows = {}
    for method, method_dir in METHODS.items():
        rows = analyze_method(method, method_dir, endpoints[method])
        all_rows[method] = rows
        if method in {"baseline", "A", "B"}:
            write_csv(OUT_DIR / f"connectivity_{method}.csv", rows)

    summary_rows = []
    for method, rows in all_rows.items():
        summary = summarize_method(rows)
        summary_rows.append({"method": method, **summary})
    write_csv(OUT_DIR / "connectivity_summary_all_methods.csv", summary_rows)

    burst = {method: burst_for_rows(all_rows[method]) for method in ("baseline", "A", "B")}
    burst["interpretation_rules"] = {
        "ss_continuous": "high connected_to_active_shared_ratio, low isolated components, and boundary_contact not dropping at max birth jump",
        "ss_discontinuous": "isolated components increase or connected ratio drops at max birth jump",
    }
    (OUT_DIR / "burst_diagnosis.json").write_text(json.dumps(burst, indent=2), encoding="utf-8")

    orient = {
        "baseline_endpoint": orientation_report(endpoints["baseline"]),
        "B_endpoint": orientation_report(endpoints["B"]),
        "note": "No features are rotated; this only compares endpoint occupancy masks.",
    }
    (OUT_DIR / "orientation_alignment_report.json").write_text(json.dumps(orient, indent=2), encoding="utf-8")

    plot_metric(all_rows, ["baseline", "A", "B"], "birth_activation_ratio", "birth activation ratio", "birth_activation_vs_alpha_baseline_A_B.png")
    plot_metric(all_rows, ["baseline", "A", "B"], "connected_to_active_shared_ratio", "connected to active shared ratio", "connected_to_shared_ratio_vs_alpha_baseline_A_B.png")
    plot_metric(all_rows, ["baseline", "A", "B"], "isolated_birth_component_count", "isolated birth components", "isolated_birth_components_vs_alpha_baseline_A_B.png")
    plot_metric(all_rows, ["baseline", "A", "B"], "boundary_contact_count", "boundary contact count", "boundary_contact_vs_alpha_baseline_A_B.png")
    plot_metric(all_rows, ["baseline", "A", "B"], "shared_keep_ratio", "shared keep ratio", "shared_keep_vs_alpha_baseline_A_B.png")
    make_contact_sheet_b(endpoints["B"], all_rows["B"])

    b_burst = burst["B"]
    a_sum = summarize_method(all_rows["A"])
    b_sum = summarize_method(all_rows["B"])
    phi_b = phi_stats["methods"]["B"]
    orient_b = orient["B_endpoint"]
    phi_obvious_jump = (phi_b["large_phi_jump_edge_ratio"] or 0.0) > 0.05 or (phi_b["edge_phi_diff_p95"] or 0.0) > 0.2
    b_connected = bool(b_burst and b_burst["ss_continuous_at_burst"])
    better = "B" if (
        b_sum["mean_connected_to_active_shared_ratio"],
        -b_sum["mean_isolated_birth_component_count"],
        b_sum["mean_boundary_contact_count"],
    ) > (
        a_sum["mean_connected_to_active_shared_ratio"],
        -a_sum["mean_isolated_birth_component_count"],
        a_sum["mean_boundary_contact_count"],
    ) else "A"
    ss_continuous = b_connected and not phi_obvious_jump
    print("\nDDPF quantile integrated diagnosis")
    print(f"output_dir: {OUT_DIR}")
    print(
        "1. quantile phi obvious spatial jumps: "
        f"{'YES' if phi_obvious_jump else 'NO'} "
        f"(B p95={phi_b['edge_phi_diff_p95']:.4f}, max={phi_b['edge_phi_diff_max']:.4f}, "
        f"large_edge_ratio={phi_b['large_phi_jump_edge_ratio']:.4f})"
    )
    print(
        "2. B max burst remains connected to shared: "
        f"{'YES' if b_connected else 'NO'} "
        f"(frames {b_burst['frame_before']}->{b_burst['frame_after']}, "
        f"conn_after={b_burst['connected_to_active_shared_ratio_after']:.4f}, "
        f"iso_after={b_burst['isolated_birth_component_count_after']}, "
        f"contact_after={b_burst['boundary_contact_count_after']})"
    )
    print(
        "3. better active-birth continuity between A/B: "
        f"{better} "
        f"(A mean_conn={a_sum['mean_connected_to_active_shared_ratio']:.4f}, mean_iso={a_sum['mean_isolated_birth_component_count']:.2f}; "
        f"B mean_conn={b_sum['mean_connected_to_active_shared_ratio']:.4f}, mean_iso={b_sum['mean_isolated_birth_component_count']:.2f})"
    )
    print(
        "4. orientation mismatch suspected: "
        f"{'YES' if orient_b['orientation_mismatch_suspected'] else 'NO'} "
        f"(B original_iou={orient_b['original_direction']['shared_iou']:.4f}, "
        f"best_iou={orient_b['best_rotation_direction']['shared_iou']:.4f}, "
        f"gain={orient_b['orientation_gain']:.4f}, pca_cos={orient_b['pca_main_axis_abs_cosine']:.4f})"
    )
    if ss_continuous:
        print("5. SS looks continuous enough; if visual output is still unstable, next inspect SLAT / decoder / render side instead of further changing SS DDPF.")
    else:
        print("5. SS-side discontinuity is still plausible; inspect burst connectivity and phi smoothness before moving to SLAT / decoder / render side.")


if __name__ == "__main__":
    main()
