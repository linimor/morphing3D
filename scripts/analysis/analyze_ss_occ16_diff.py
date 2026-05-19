#!/usr/bin/env python3
import argparse
import json
import os
from collections import deque
from typing import Dict, List, Tuple

import torch


NEIGHBOR_OFFSETS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _as_bool_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    x = torch.as_tensor(x).detach().cpu()
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {tuple(x.shape)}")
    return x > 0


def _load_occ_pt(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    required = ["ss_token_coords", "occ_s_16", "occ_t_16", "src_coords_raw", "tar_coords_raw"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required keys in {path}: {missing}")

    ss_token_coords = torch.as_tensor(data["ss_token_coords"]).detach().cpu().long()
    if ss_token_coords.ndim != 2 or ss_token_coords.shape[-1] != 3:
        raise ValueError(f"ss_token_coords must be [N, 3], got {tuple(ss_token_coords.shape)}")

    occ_s_16 = _as_bool_1d(data["occ_s_16"], "occ_s_16")
    occ_t_16 = _as_bool_1d(data["occ_t_16"], "occ_t_16")
    if occ_s_16.shape[0] != ss_token_coords.shape[0] or occ_t_16.shape[0] != ss_token_coords.shape[0]:
        raise ValueError(
            "occ_s_16/occ_t_16 length must match ss_token_coords length: "
            f"{occ_s_16.shape[0]}, {occ_t_16.shape[0]}, {ss_token_coords.shape[0]}"
        )

    src_coords_raw = torch.as_tensor(data["src_coords_raw"]).detach().cpu()
    tar_coords_raw = torch.as_tensor(data["tar_coords_raw"]).detach().cpu()
    if src_coords_raw.ndim == 2 and src_coords_raw.shape[-1] == 4:
        src_coords_raw = src_coords_raw[:, 1:]
    if tar_coords_raw.ndim == 2 and tar_coords_raw.shape[-1] == 4:
        tar_coords_raw = tar_coords_raw[:, 1:]

    return {
        "ss_token_coords": ss_token_coords,
        "occ_s_16": occ_s_16,
        "occ_t_16": occ_t_16,
        "src_coords_raw": src_coords_raw,
        "tar_coords_raw": tar_coords_raw,
    }


def _build_graph(ss_token_coords: torch.Tensor) -> List[List[int]]:
    coord_to_index = {tuple(coord.tolist()): idx for idx, coord in enumerate(ss_token_coords)}
    adjacency: List[List[int]] = [[] for _ in range(ss_token_coords.shape[0])]
    for idx, coord in enumerate(ss_token_coords.tolist()):
        x, y, z = coord
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor_idx = coord_to_index.get((x + dx, y + dy, z + dz))
            if neighbor_idx is not None:
                adjacency[idx].append(neighbor_idx)
    return adjacency


def _connected_components(mask: torch.Tensor, adjacency: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
    mask = mask.bool().cpu()
    comp_id = torch.full((mask.shape[0],), -1, dtype=torch.long)
    sizes: List[int] = []
    current_id = 0

    for start in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        if comp_id[start].item() != -1:
            continue
        queue = deque([start])
        comp_id[start] = current_id
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if mask[neighbor] and comp_id[neighbor].item() == -1:
                    comp_id[neighbor] = current_id
                    queue.append(neighbor)
        sizes.append(size)
        current_id += 1

    return comp_id, sizes


def _boundary_count(mask: torch.Tensor, shared_mask: torch.Tensor, adjacency: List[List[int]]) -> int:
    count = 0
    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        if any(shared_mask[neighbor].item() for neighbor in adjacency[idx]):
            count += 1
    return count


def _components_without_shared_boundary(
    comp_id: torch.Tensor,
    comp_sizes: List[int],
    shared_mask: torch.Tensor,
    adjacency: List[List[int]],
) -> int:
    if not comp_sizes:
        return 0
    has_boundary = [False] * len(comp_sizes)
    for idx in torch.nonzero(comp_id >= 0, as_tuple=False).flatten().tolist():
        cid = int(comp_id[idx].item())
        if any(shared_mask[neighbor].item() for neighbor in adjacency[idx]):
            has_boundary[cid] = True
    return sum(1 for value in has_boundary if not value)


def _distance_to_shared(
    target_mask: torch.Tensor,
    shared_mask: torch.Tensor,
    adjacency: List[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    n = target_mask.shape[0]
    dist = torch.full((n,), -1, dtype=torch.long)
    queue = deque()
    for idx in torch.nonzero(shared_mask, as_tuple=False).flatten().tolist():
        dist[idx] = 0
        queue.append(idx)

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if dist[neighbor].item() == -1:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)

    target_dist = torch.full((n,), -1, dtype=torch.long)
    target_dist[target_mask] = dist[target_mask]
    unreachable = int(((target_mask) & (target_dist < 0)).sum().item())
    return target_dist, dist, unreachable


def _normalize_phi(dist: torch.Tensor, mask: torch.Tensor, invert: bool = False) -> torch.Tensor:
    phi = torch.zeros(mask.shape[0], dtype=torch.float32)
    valid = mask & (dist >= 0)
    if not valid.any():
        return phi
    values = dist[valid].float()
    min_v = values.min()
    max_v = values.max()
    if torch.isclose(max_v, min_v):
        normalized = torch.ones_like(values)
    else:
        normalized = (values - min_v) / (max_v - min_v)
    if invert:
        normalized = 1.0 - normalized
    phi[valid] = normalized
    return phi


def _top_sizes(sizes: List[int], k: int = 10) -> List[int]:
    return sorted((int(size) for size in sizes), reverse=True)[:k]


def _ratio(count: int, denom: int) -> float:
    return float(count / denom) if denom > 0 else 0.0


def _try_plot(
    output_dir: str,
    ss_token_coords: torch.Tensor,
    shared_mask: torch.Tensor,
    birth_mask: torch.Tensor,
    death_mask: torch.Tensor,
    birth_phi_geo: torch.Tensor,
    death_phi_geo: torch.Tensor,
    birth_sizes: List[int],
    death_sizes: List[int],
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable; skipped plots: {exc}"

    coords = ss_token_coords.float().numpy()
    shared = shared_mask.numpy()
    birth = birth_mask.numpy()
    death = death_mask.numpy()
    source_only = death
    target_only = birth

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    background = ~(shared | birth | death)
    if background.any():
        ax.scatter(coords[background, 0], coords[background, 1], coords[background, 2], s=4, c="#d0d0d0", alpha=0.15, label="empty")
    if shared.any():
        ax.scatter(coords[shared, 0], coords[shared, 1], coords[shared, 2], s=10, c="#2ca02c", label="shared")
    if source_only.any():
        ax.scatter(coords[source_only, 0], coords[source_only, 1], coords[source_only, 2], s=12, c="#d62728", label="death/source-only")
    if target_only.any():
        ax.scatter(coords[target_only, 0], coords[target_only, 1], coords[target_only, 2], s=12, c="#1f77b4", label="birth/target-only")
    ax.set_title("SS Occ16 Classes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "occ_class_scatter_3d.png"), dpi=180)
    plt.close(fig)

    def scatter_phi(mask: torch.Tensor, phi: torch.Tensor, filename: str, title: str) -> None:
        mask_np = mask.numpy()
        fig_phi = plt.figure(figsize=(8, 7))
        ax_phi = fig_phi.add_subplot(111, projection="3d")
        if mask_np.any():
            vals = phi.numpy()[mask_np]
            plot = ax_phi.scatter(coords[mask_np, 0], coords[mask_np, 1], coords[mask_np, 2], s=14, c=vals, cmap="viridis")
            fig_phi.colorbar(plot, ax=ax_phi, shrink=0.6)
        ax_phi.set_title(title)
        ax_phi.set_xlabel("x")
        ax_phi.set_ylabel("y")
        ax_phi.set_zlabel("z")
        fig_phi.tight_layout()
        fig_phi.savefig(os.path.join(output_dir, filename), dpi=180)
        plt.close(fig_phi)

    scatter_phi(birth_mask, birth_phi_geo, "birth_phi_scatter_3d.png", "Birth Phi Geo")
    scatter_phi(death_mask, death_phi_geo, "death_phi_scatter_3d.png", "Death Phi Geo")

    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    if birth_sizes:
        ax_hist.hist(birth_sizes, bins=min(30, max(1, len(birth_sizes))), alpha=0.6, label="birth")
    if death_sizes:
        ax_hist.hist(death_sizes, bins=min(30, max(1, len(death_sizes))), alpha=0.6, label="death")
    ax_hist.set_title("Component Size Histogram")
    ax_hist.set_xlabel("component size")
    ax_hist.set_ylabel("count")
    ax_hist.legend()
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(output_dir, "component_size_hist.png"), dpi=180)
    plt.close(fig_hist)
    return ""


def analyze(occ_pt: str, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    data = _load_occ_pt(occ_pt)
    ss_token_coords = data["ss_token_coords"]
    occ_s_16 = data["occ_s_16"]
    occ_t_16 = data["occ_t_16"]

    adjacency = _build_graph(ss_token_coords)
    shared_mask = occ_s_16 & occ_t_16
    birth_mask = occ_t_16 & (~occ_s_16)
    death_mask = occ_s_16 & (~occ_t_16)

    birth_comp_id, birth_sizes = _connected_components(birth_mask, adjacency)
    death_comp_id, death_sizes = _connected_components(death_mask, adjacency)
    shared_comp_id, shared_sizes = _connected_components(shared_mask, adjacency)

    birth_boundary_count = _boundary_count(birth_mask, shared_mask, adjacency)
    death_boundary_count = _boundary_count(death_mask, shared_mask, adjacency)
    birth_components_without_boundary = _components_without_shared_boundary(birth_comp_id, birth_sizes, shared_mask, adjacency)
    death_components_without_boundary = _components_without_shared_boundary(death_comp_id, death_sizes, shared_mask, adjacency)

    warnings: List[str] = []
    if int(shared_mask.sum().item()) == 0:
        warnings.append("No shared tokens; graph distance to shared is unreachable for all birth/death tokens.")

    birth_dist_to_shared, _, birth_unreachable = _distance_to_shared(birth_mask, shared_mask, adjacency)
    death_dist_to_shared, _, death_unreachable = _distance_to_shared(death_mask, shared_mask, adjacency)
    birth_phi_geo = _normalize_phi(birth_dist_to_shared, birth_mask, invert=False)
    death_phi_geo = _normalize_phi(death_dist_to_shared, death_mask, invert=True)

    source_count = int(occ_s_16.sum().item())
    target_count = int(occ_t_16.sum().item())
    shared_count = int(shared_mask.sum().item())
    birth_count = int(birth_mask.sum().item())
    death_count = int(death_mask.sum().item())
    union_count = int((occ_s_16 | occ_t_16).sum().item())

    stats = {
        "occ_pt": occ_pt,
        "source_occupied_token_count": source_count,
        "target_occupied_token_count": target_count,
        "shared_token_count": shared_count,
        "birth_token_count": birth_count,
        "death_token_count": death_count,
        "ratios": {
            "birth_over_union": _ratio(birth_count, union_count),
            "death_over_union": _ratio(death_count, union_count),
            "shared_over_union": _ratio(shared_count, union_count),
            "birth_over_target": _ratio(birth_count, target_count),
            "death_over_source": _ratio(death_count, source_count),
        },
        "birth_connected_components": {
            "count": len(birth_sizes),
            "top10_sizes": _top_sizes(birth_sizes),
        },
        "death_connected_components": {
            "count": len(death_sizes),
            "top10_sizes": _top_sizes(death_sizes),
        },
        "shared_connected_components": {
            "count": len(shared_sizes),
            "top10_sizes": _top_sizes(shared_sizes),
        },
        "birth_boundary_token_count_adjacent_to_shared": birth_boundary_count,
        "death_boundary_token_count_adjacent_to_shared": death_boundary_count,
        "birth_components_without_shared_boundary_count": birth_components_without_boundary,
        "death_components_without_shared_boundary_count": death_components_without_boundary,
        "birth_unreachable_to_shared_count": birth_unreachable,
        "death_unreachable_to_shared_count": death_unreachable,
        "raw_coord_shapes_after_spatial_column_selection": {
            "src_coords_raw": list(data["src_coords_raw"].shape),
            "tar_coords_raw": list(data["tar_coords_raw"].shape),
        },
        "warnings": warnings,
    }

    torch.save(
        {
            "ss_token_coords": ss_token_coords,
            "occ_s_16": occ_s_16,
            "occ_t_16": occ_t_16,
            "shared_mask": shared_mask,
            "birth_mask": birth_mask,
            "death_mask": death_mask,
            "birth_comp_id": birth_comp_id,
            "death_comp_id": death_comp_id,
            "shared_comp_id": shared_comp_id,
            "birth_dist_to_shared": birth_dist_to_shared,
            "death_dist_to_shared": death_dist_to_shared,
            "birth_phi_geo": birth_phi_geo,
            "death_phi_geo": death_phi_geo,
        },
        os.path.join(output_dir, "diff_maps.pt"),
    )

    plot_warning = _try_plot(
        output_dir,
        ss_token_coords,
        shared_mask,
        birth_mask,
        death_mask,
        birth_phi_geo,
        death_phi_geo,
        birth_sizes,
        death_sizes,
    )
    if plot_warning:
        stats["warnings"].append(plot_warning)

    with open(os.path.join(output_dir, "diff_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 16^3 sparse-structure endpoint occupancy differences.")
    parser.add_argument("--occ_pt", required=True, help="Path to ss_endpoint_occ16.pt")
    parser.add_argument("--output_dir", required=True, help="Directory for diff_stats.json, diff_maps.pt, and plots")
    args = parser.parse_args()

    stats = analyze(args.occ_pt, args.output_dir)
    print("SS occ16 diff analysis")
    print(f"  source occupied: {stats['source_occupied_token_count']}")
    print(f"  target occupied: {stats['target_occupied_token_count']}")
    print(f"  shared: {stats['shared_token_count']}")
    print(f"  birth: {stats['birth_token_count']}")
    print(f"  death: {stats['death_token_count']}")
    print(f"  birth components: {stats['birth_connected_components']['count']} top10={stats['birth_connected_components']['top10_sizes']}")
    print(f"  death components: {stats['death_connected_components']['count']} top10={stats['death_connected_components']['top10_sizes']}")
    print(f"  shared components: {stats['shared_connected_components']['count']} top10={stats['shared_connected_components']['top10_sizes']}")
    print(f"  birth boundary tokens: {stats['birth_boundary_token_count_adjacent_to_shared']}")
    print(f"  death boundary tokens: {stats['death_boundary_token_count_adjacent_to_shared']}")
    print(f"  output_dir: {args.output_dir}")
    for warning in stats.get("warnings", []):
        print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
