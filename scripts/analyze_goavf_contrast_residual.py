import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GRID = 16
N = GRID ** 3


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


def index_to_coord(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    return np.stack([idx // (GRID * GRID), (idx // GRID) % GRID, idx % GRID], axis=1)


def load_endpoint(path: Path) -> dict:
    d = torch.load(path, map_location="cpu")
    coords = d["ss_token_coords"].cpu().numpy().astype(np.int64)
    occ_s = d["occ_s_16"].cpu().numpy().astype(bool)
    occ_t = d["occ_t_16"].cpu().numpy().astype(bool)
    return {
        "coords": coords,
        "shared": occ_s & occ_t,
        "birth": occ_t & ~occ_s,
        "death": occ_s & ~occ_t,
    }


def edge_pairs(mask: np.ndarray) -> np.ndarray:
    edges = []
    for idx in np.flatnonzero(mask):
        for nb in neighbors(int(idx)):
            if nb > idx and mask[nb]:
                edges.append((int(idx), int(nb)))
    return np.asarray(edges, dtype=np.int64)


def graph_smooth(q: np.ndarray, edges: np.ndarray, mask: np.ndarray, steps: int = 4, rho: float = 0.45) -> np.ndarray:
    out = q.copy()
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for _ in range(steps):
        nxt = out.copy()
        for idx in np.flatnonzero(mask):
            nbs = adj[int(idx)]
            if nbs:
                nxt[idx] = (1.0 - rho) * out[idx] + rho * float(np.mean(out[nbs]))
        out = nxt
    return out


def edge_metrics(q: np.ndarray, edges: np.ndarray) -> dict:
    if len(edges) == 0:
        return {"edge_q_diff_mean": None, "edge_q_diff_max": None}
    d = np.abs(q[edges[:, 0]] - q[edges[:, 1]])
    return {"edge_q_diff_mean": float(d.mean()), "edge_q_diff_max": float(d.max())}


def curvature_metrics(q: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> dict:
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    vals = []
    for idx in np.flatnonzero(mask):
        nbs = adj[int(idx)]
        if nbs:
            vals.append(abs(q[idx] - float(np.mean(q[nbs]))))
    if not vals:
        return {"graph_curvature_mean": None, "graph_curvature_max": None}
    vals = np.asarray(vals, dtype=np.float64)
    return {"graph_curvature_mean": float(vals.mean()), "graph_curvature_max": float(vals.max())}


def distance_from_shared(shared: np.ndarray) -> np.ndarray:
    dist = np.full(N, 999, dtype=np.int32)
    q = list(np.flatnonzero(shared).astype(int))
    for idx in q:
        dist[idx] = 0
    head = 0
    while head < len(q):
        cur = q[head]
        head += 1
        for nb in neighbors(cur):
            if dist[nb] > dist[cur] + 1:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def compute_gate(birth: np.ndarray, shared: np.ndarray, edges: np.ndarray) -> np.ndarray:
    gate = np.zeros(N, dtype=np.float64)
    dist = distance_from_shared(shared)
    # Weak near shared boundary, stronger deeper in birth/diff.
    gate[birth] = np.clip((dist[birth].astype(np.float64) - 1.0) / 4.0, 0.0, 1.0)
    adj_birth_count = np.zeros(N, dtype=np.float64)
    for idx in np.flatnonzero(birth):
        nbs = list(neighbors(int(idx)))
        adj_birth_count[idx] = sum(bool(birth[nb]) for nb in nbs) / max(len(nbs), 1)
    gate[birth] *= np.clip(adj_birth_count[birth], 0.0, 1.0)
    return gate


def safe_beta(base_q: np.ndarray, residual: np.ndarray, edges: np.ndarray, factor: float = 1.5) -> float:
    if len(edges) == 0:
        return 0.0
    base_d = np.abs(base_q[edges[:, 0]] - base_q[edges[:, 1]])
    res_d = np.abs(residual[edges[:, 0]] - residual[edges[:, 1]])
    budget = factor * float(base_d.max()) - float(base_d.max())
    denom = float(res_d.max())
    if budget <= 0.0 or denom <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, budget / denom))


def unsupported_metrics(q: np.ndarray, birth: np.ndarray, anchor_rows: list[dict], q_data: dict) -> dict:
    # q_field_per_alpha does not persist token-anchor membership. Use anchor type priors if available:
    # if there are birth-free anchors, report global late/outer proxy from low local support.
    shared = q_data["shared_mask"].numpy().astype(bool)
    dist = distance_from_shared(shared)
    outer_birth = birth & (dist >= 3)
    supported_birth = birth & (dist < 3)
    unsupported = float(q[outer_birth].mean()) if outer_birth.any() else 0.0
    supported = float(q[supported_birth].mean()) if supported_birth.any() else 0.0
    return {
        "unsupported_outer_activation": unsupported,
        "supported_birth_activation": supported,
        "unsupported_minus_supported": unsupported - supported,
        "hollow_ring_risk": bool(unsupported > supported + 0.10),
    }


def birth_stats(q: np.ndarray, birth: np.ndarray) -> dict:
    vals = q[birth]
    return {
        "q_std_in_birth": float(vals.std()) if vals.size else 0.0,
        "q_range_in_birth": float(vals.max() - vals.min()) if vals.size else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def visualize(path: Path, coords: np.ndarray, birth: np.ndarray, shared: np.ndarray, frames, fields: dict[str, np.ndarray]) -> None:
    mid = min(range(len(frames)), key=lambda i: abs(float(frames[i]) - 0.5))
    ids = np.flatnonzero(shared | birth)
    pts = coords[ids]
    fig = plt.figure(figsize=(15, 5), dpi=170)
    for panel, name in enumerate(("A_goavf", "B_local_residual", "C_gated_residual"), start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        q = fields[name][mid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=q[ids], cmap="viridis", s=13, vmin=0, vmax=1, alpha=0.85)
        ax.set_title(f"{name}, p={float(frames[mid]):.2f}")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_zlim(0, 15)
        ax.view_init(elev=24, azim=42)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for version in sorted({r["version"] for r in rows}):
        subset = [r for r in rows if r["version"] == version]
        out[version] = {}
        for key in [
            "edge_q_diff_mean",
            "edge_q_diff_max",
            "graph_curvature_mean",
            "graph_curvature_max",
            "q_std_in_birth",
            "q_range_in_birth",
            "unsupported_outer_activation",
            "unsupported_minus_supported",
        ]:
            vals = [r[key] for r in subset if r[key] is not None]
            out[version][f"{key}_mean"] = float(np.mean(vals)) if vals else None
            out[version][f"{key}_max"] = float(np.max(vals)) if vals else None
        out[version]["hollow_ring_risk_any"] = bool(any(r["hollow_ring_risk"] for r in subset))
        out[version]["beta_mean"] = float(np.mean([r["beta"] for r in subset]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="G-OAVF contrast-preserving residual q-field analysis.")
    parser.add_argument("--q-field", type=Path, default=ROOT / "outputs/analysis/goavf_offline_chong_hudie/q_field_per_alpha.pt")
    parser.add_argument("--endpoint", type=Path, default=ROOT / "outputs/cache/chong/cache/ss_endpoint_occ16_to_hudie.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/goavf_contrast_residual_chong_hudie")
    parser.add_argument("--smooth-steps", type=int, default=4)
    parser.add_argument("--smooth-rho", type=float, default=0.45)
    parser.add_argument("--edge-budget-factor", type=float, default=1.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_data = torch.load(args.q_field, map_location="cpu")
    ep = load_endpoint(args.endpoint)
    shared = ep["shared"]
    birth = ep["birth"]
    active = shared | birth
    edges = edge_pairs(birth)
    active_edges = edge_pairs(active)
    q_scaf = q_data["q_scaf"].numpy().astype(np.float64)
    q_goavf = q_data["q_goavf"].numpy().astype(np.float64)
    p_vals = q_data["p"].numpy().astype(np.float64)
    anchor_rows = q_data.get("anchor_rows", [])

    gate = compute_gate(birth, shared, edges)
    fields = {"A_goavf": q_goavf.copy(), "B_local_residual": [], "C_gated_residual": []}
    rows = []
    betas = {"B_local_residual": [], "C_gated_residual": []}

    for frame_i, p in enumerate(p_vals):
        base = q_goavf[frame_i]
        smooth_scaf = graph_smooth(q_scaf[frame_i], edges, birth, steps=args.smooth_steps, rho=args.smooth_rho)
        residual = np.zeros(N, dtype=np.float64)
        residual[birth] = q_scaf[frame_i, birth] - smooth_scaf[birth]

        beta_b = safe_beta(base, residual, edges, factor=args.edge_budget_factor)
        q_b = np.clip(base + beta_b * residual, 0.0, 1.0)

        gated_residual = residual * gate
        beta_c = safe_beta(base, gated_residual, edges, factor=args.edge_budget_factor)
        q_c = np.clip(base + beta_c * gated_residual, 0.0, 1.0)

        fields["B_local_residual"].append(q_b)
        fields["C_gated_residual"].append(q_c)
        betas["B_local_residual"].append(beta_b)
        betas["C_gated_residual"].append(beta_c)

        for version, q, beta in (
            ("A_goavf", base, 0.0),
            ("B_local_residual", q_b, beta_b),
            ("C_gated_residual", q_c, beta_c),
        ):
            row = {
                "version": version,
                "frame_index": int(q_data["frames"][frame_i]),
                "p": float(p),
                "beta": float(beta),
            }
            row.update(edge_metrics(q, edges))
            row.update(curvature_metrics(q, active_edges, active))
            row.update(birth_stats(q, birth))
            row.update(unsupported_metrics(q, birth, anchor_rows, q_data))
            rows.append(row)

    fields["B_local_residual"] = np.stack(fields["B_local_residual"])
    fields["C_gated_residual"] = np.stack(fields["C_gated_residual"])
    torch.save(
        {
            "frames": q_data["frames"],
            "p": q_data["p"],
            "q_scaf": torch.from_numpy(q_scaf).float(),
            "A_goavf": torch.from_numpy(fields["A_goavf"]).float(),
            "B_local_residual": torch.from_numpy(fields["B_local_residual"]).float(),
            "C_gated_residual": torch.from_numpy(fields["C_gated_residual"]).float(),
            "gate": torch.from_numpy(gate).float(),
            "birth_mask": torch.from_numpy(birth),
            "shared_mask": torch.from_numpy(shared),
            "beta_B": torch.tensor(betas["B_local_residual"]).float(),
            "beta_C": torch.tensor(betas["C_gated_residual"]).float(),
        },
        args.out_dir / "contrast_q_fields.pt",
    )
    visualize(args.out_dir / "q_field_visualization.png", ep["coords"], birth, shared, p_vals, fields)
    write_csv(args.out_dir / "contrast_metrics_per_alpha.csv", rows)
    summary = summarize(rows)
    scaf_edge = [edge_metrics(q, edges)["edge_q_diff_max"] for q in q_scaf]
    summary["scaf_reference"] = {
        "edge_q_diff_max_mean": float(np.mean(scaf_edge)),
        "edge_q_diff_max_max": float(np.max(scaf_edge)),
        "q_std_in_birth_mean": float(np.mean([q[birth].std() for q in q_scaf])),
        "q_range_in_birth_mean": float(np.mean([q[birth].max() - q[birth].min() for q in q_scaf])),
    }
    summary["criteria"] = {
        "B_continuity_better_than_scaf": bool(summary["B_local_residual"]["edge_q_diff_max_mean"] < summary["scaf_reference"]["edge_q_diff_max_mean"]),
        "C_continuity_better_than_scaf": bool(summary["C_gated_residual"]["edge_q_diff_max_mean"] < summary["scaf_reference"]["edge_q_diff_max_mean"]),
        "B_birth_contrast_above_A": bool(summary["B_local_residual"]["q_std_in_birth_mean"] > summary["A_goavf"]["q_std_in_birth_mean"]),
        "C_birth_contrast_above_A": bool(summary["C_gated_residual"]["q_std_in_birth_mean"] > summary["A_goavf"]["q_std_in_birth_mean"]),
        "B_no_hollow_ring_risk": not summary["B_local_residual"]["hollow_ring_risk_any"],
        "C_no_hollow_ring_risk": not summary["C_gated_residual"]["hollow_ring_risk_any"],
    }
    (args.out_dir / "contrast_residual_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"contrast residual analysis written to: {args.out_dir}")
    print(json.dumps(summary["criteria"], indent=2))


if __name__ == "__main__":
    main()
