import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F

from trellis import models
from trellis.modules.spatial import patchify


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe SS tokens ([1, 4096, 1024]) against decoded occupancy geometry."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./TRELLIS-image-large",
        help="Path to the pretrained TRELLIS pipeline directory.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to a cache directory that contains coords_zs.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <cache>/ss_token_probe.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--pair-samples", type=int, default=50000)
    parser.add_argument("--anchor-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_ss_models(pretrained_dir: str, device: torch.device):
    with open(os.path.join(pretrained_dir, "pipeline.json"), "r", encoding="utf-8") as f:
        pipeline_cfg = json.load(f)["args"]["models"]

    flow_model = models.from_pretrained(
        os.path.join(pretrained_dir, pipeline_cfg["sparse_structure_flow_model"])
    ).to(device)
    decoder = models.from_pretrained(
        os.path.join(pretrained_dir, pipeline_cfg["sparse_structure_decoder"])
    ).to(device)
    flow_model.eval()
    decoder.eval()
    return flow_model, decoder


@torch.no_grad()
def encode_ss_tokens(z_s: torch.Tensor, flow_model) -> torch.Tensor:
    x = patchify(z_s, flow_model.patch_size)
    x = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
    x = flow_model.input_layer(x)
    if hasattr(flow_model, "pos_emb") and flow_model.pos_emb is not None:
        x = x + flow_model.pos_emb[None]
    x = F.layer_norm(x, x.shape[-1:])
    return x


@torch.no_grad()
def decode_ss_voxels(z_s: torch.Tensor, decoder) -> torch.Tensor:
    voxels = decoder(z_s) > 0
    return voxels.float()


def make_token_coords(resolution: int, device: torch.device) -> torch.Tensor:
    coords = torch.meshgrid(
        *[torch.arange(resolution, device=device) for _ in range(3)],
        indexing="ij",
    )
    return torch.stack(coords, dim=-1).reshape(-1, 3).float()


def blockify_occupancy(occ: torch.Tensor, token_resolution: int) -> torch.Tensor:
    up_ratio = occ.shape[0] // token_resolution
    if occ.shape[0] % token_resolution != 0:
        raise ValueError(
            f"Decoded occupancy resolution {occ.shape[0]} is not divisible by token resolution {token_resolution}."
        )
    blocks = occ.view(
        token_resolution,
        up_ratio,
        token_resolution,
        up_ratio,
        token_resolution,
        up_ratio,
    )
    blocks = blocks.permute(0, 2, 4, 1, 3, 5).contiguous()
    return blocks


def boundary_density(blocks: torch.Tensor) -> torch.Tensor:
    dx = (blocks[:, :, :, 1:, :, :] - blocks[:, :, :, :-1, :, :]).abs().mean(dim=(3, 4, 5))
    dy = (blocks[:, :, :, :, 1:, :] - blocks[:, :, :, :, :-1, :]).abs().mean(dim=(3, 4, 5))
    dz = (blocks[:, :, :, :, :, 1:] - blocks[:, :, :, :, :, :-1]).abs().mean(dim=(3, 4, 5))
    return (dx + dy + dz) / 3.0


def compute_geometry_features(voxels: torch.Tensor, token_resolution: int):
    occ = voxels[0, 0].float()
    blocks = blockify_occupancy(occ, token_resolution)
    occ_ratio = blocks.mean(dim=(3, 4, 5)).reshape(-1)
    surf_ratio = boundary_density(blocks).reshape(-1)

    occupied_xyz = torch.nonzero(occ > 0.5, as_tuple=False).float()
    if occupied_xyz.numel() == 0:
        shape_centroid = torch.full((3,), float("nan"), device=occ.device)
    else:
        shape_centroid = occupied_xyz.mean(dim=0)

    block_scale = occ.shape[0] / token_resolution
    token_coords = make_token_coords(token_resolution, occ.device)
    token_centers = (token_coords + 0.5) * block_scale - 0.5
    dist_to_shape_center = torch.norm(token_centers - shape_centroid[None], dim=-1)

    return {
        "token_coords": token_coords,
        "token_centers": token_centers,
        "occ_ratio": occ_ratio,
        "surf_ratio": surf_ratio,
        "dist_to_shape_center": dist_to_shape_center,
        "block_scale": block_scale,
        "decoded_resolution": int(occ.shape[0]),
    }


def safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() < 1e-12:
        return float("nan")
    return float((x * y).sum().item() / denom.item())


def sample_pairwise_metrics(token_feats: torch.Tensor, geom: dict, num_pairs: int, seed: int):
    generator = torch.Generator(device=token_feats.device)
    generator.manual_seed(seed)
    n = token_feats.shape[0]
    i = torch.randint(0, n, (num_pairs,), generator=generator, device=token_feats.device)
    j = torch.randint(0, n - 1, (num_pairs,), generator=generator, device=token_feats.device)
    j = j + (j >= i).long()

    feats = F.normalize(token_feats.float(), dim=-1)
    cos = (feats[i] * feats[j]).sum(dim=-1)
    l2 = torch.norm(token_feats[i] - token_feats[j], dim=-1)
    norm_gap = (token_feats[i].norm(dim=-1) - token_feats[j].norm(dim=-1)).abs()

    occ_gap = (geom["occ_ratio"][i] - geom["occ_ratio"][j]).abs()
    surf_gap = (geom["surf_ratio"][i] - geom["surf_ratio"][j]).abs()
    center_gap = (geom["dist_to_shape_center"][i] - geom["dist_to_shape_center"][j]).abs()
    spatial_dist = torch.norm(geom["token_coords"][i] - geom["token_coords"][j], dim=-1)

    return {
        "pairs_used": int(num_pairs),
        "corr_cos_vs_occ_gap": safe_corr(cos, occ_gap),
        "corr_cos_vs_surf_gap": safe_corr(cos, surf_gap),
        "corr_cos_vs_center_gap": safe_corr(cos, center_gap),
        "corr_cos_vs_spatial_dist": safe_corr(cos, spatial_dist),
        "corr_l2_vs_occ_gap": safe_corr(l2, occ_gap),
        "corr_l2_vs_surf_gap": safe_corr(l2, surf_gap),
        "corr_norm_gap_vs_occ_gap": safe_corr(norm_gap, occ_gap),
        "corr_norm_gap_vs_surf_gap": safe_corr(norm_gap, surf_gap),
    }


def knn_metrics(token_feats: torch.Tensor, geom: dict, k: int, seed: int):
    n = token_feats.shape[0]
    feats = F.normalize(token_feats.float(), dim=-1)
    sim = feats @ feats.t()
    sim.fill_diagonal_(-2.0)
    nn_idx = sim.topk(k=k, dim=-1).indices

    generator = torch.Generator(device=token_feats.device)
    generator.manual_seed(seed + 1)
    rand_idx = torch.randint(0, n, (n, k), generator=generator, device=token_feats.device)
    self_idx = torch.arange(n, device=token_feats.device)[:, None]
    rand_idx = (rand_idx + (rand_idx >= self_idx).long()) % n

    occ = geom["occ_ratio"]
    surf = geom["surf_ratio"]
    center = geom["dist_to_shape_center"]
    coords = geom["token_coords"]

    def mean_abs_gap(values: torch.Tensor, indices: torch.Tensor):
        base = values[:, None]
        return (base - values[indices]).abs().mean(dim=-1)

    def mean_spatial_gap(indices: torch.Tensor):
        base = coords[:, None, :]
        return torch.norm(base - coords[indices], dim=-1).mean(dim=-1)

    knn_occ_gap = mean_abs_gap(occ, nn_idx)
    knn_surf_gap = mean_abs_gap(surf, nn_idx)
    knn_center_gap = mean_abs_gap(center, nn_idx)
    knn_spatial_gap = mean_spatial_gap(nn_idx)

    rand_occ_gap = mean_abs_gap(occ, rand_idx)
    rand_surf_gap = mean_abs_gap(surf, rand_idx)
    rand_center_gap = mean_abs_gap(center, rand_idx)
    rand_spatial_gap = mean_spatial_gap(rand_idx)

    per_token = {
        "knn_occ_gap": knn_occ_gap,
        "knn_surf_gap": knn_surf_gap,
        "knn_center_gap": knn_center_gap,
        "knn_spatial_gap": knn_spatial_gap,
        "rand_occ_gap": rand_occ_gap,
        "rand_surf_gap": rand_surf_gap,
        "rand_center_gap": rand_center_gap,
        "rand_spatial_gap": rand_spatial_gap,
    }

    summary = {
        "k": int(k),
        "mean_knn_occ_gap": float(knn_occ_gap.mean().item()),
        "mean_rand_occ_gap": float(rand_occ_gap.mean().item()),
        "mean_knn_surf_gap": float(knn_surf_gap.mean().item()),
        "mean_rand_surf_gap": float(rand_surf_gap.mean().item()),
        "mean_knn_center_gap": float(knn_center_gap.mean().item()),
        "mean_rand_center_gap": float(rand_center_gap.mean().item()),
        "mean_knn_spatial_gap": float(knn_spatial_gap.mean().item()),
        "mean_rand_spatial_gap": float(rand_spatial_gap.mean().item()),
    }
    return summary, per_token, nn_idx


def select_anchor_indices(token_feats: torch.Tensor, occ_ratio: torch.Tensor, anchor_count: int):
    norms = token_feats.norm(dim=-1)
    occupied = torch.nonzero(occ_ratio > 0, as_tuple=False).flatten()
    if occupied.numel() == 0:
        occupied = torch.arange(token_feats.shape[0], device=token_feats.device)
    top_count = min(anchor_count, occupied.numel())
    top_local = norms[occupied].topk(k=top_count).indices
    return occupied[top_local]


def write_token_csv(path: Path, token_feats: torch.Tensor, geom: dict, knn_per_token: dict):
    fieldnames = [
        "token_idx",
        "grid_x",
        "grid_y",
        "grid_z",
        "token_norm",
        "occ_ratio",
        "surf_ratio",
        "dist_to_shape_center",
        "knn_occ_gap",
        "rand_occ_gap",
        "knn_surf_gap",
        "rand_surf_gap",
        "knn_spatial_gap",
        "rand_spatial_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        norms = token_feats.norm(dim=-1).cpu()
        coords = geom["token_coords"].cpu().int()
        for idx in range(token_feats.shape[0]):
            writer.writerow(
                {
                    "token_idx": idx,
                    "grid_x": int(coords[idx, 0].item()),
                    "grid_y": int(coords[idx, 1].item()),
                    "grid_z": int(coords[idx, 2].item()),
                    "token_norm": float(norms[idx].item()),
                    "occ_ratio": float(geom["occ_ratio"][idx].item()),
                    "surf_ratio": float(geom["surf_ratio"][idx].item()),
                    "dist_to_shape_center": float(geom["dist_to_shape_center"][idx].item()),
                    "knn_occ_gap": float(knn_per_token["knn_occ_gap"][idx].item()),
                    "rand_occ_gap": float(knn_per_token["rand_occ_gap"][idx].item()),
                    "knn_surf_gap": float(knn_per_token["knn_surf_gap"][idx].item()),
                    "rand_surf_gap": float(knn_per_token["rand_surf_gap"][idx].item()),
                    "knn_spatial_gap": float(knn_per_token["knn_spatial_gap"][idx].item()),
                    "rand_spatial_gap": float(knn_per_token["rand_spatial_gap"][idx].item()),
                }
            )


def write_anchor_csv(path: Path, anchors: torch.Tensor, nn_idx: torch.Tensor, token_feats: torch.Tensor, geom: dict):
    fieldnames = [
        "anchor_token_idx",
        "anchor_norm",
        "anchor_grid_x",
        "anchor_grid_y",
        "anchor_grid_z",
        "anchor_occ_ratio",
        "anchor_surf_ratio",
        "neighbor_mean_grid_x",
        "neighbor_mean_grid_y",
        "neighbor_mean_grid_z",
        "neighbor_std_grid",
        "neighbor_mean_occ_ratio",
        "neighbor_std_occ_ratio",
        "neighbor_mean_surf_ratio",
        "neighbor_std_surf_ratio",
        "neighbor_mean_cos",
    ]
    feats = F.normalize(token_feats.float(), dim=-1)
    coords = geom["token_coords"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anchor in anchors.tolist():
            neighbors = nn_idx[anchor]
            neighbor_coords = coords[neighbors]
            neighbor_occ = geom["occ_ratio"][neighbors]
            neighbor_surf = geom["surf_ratio"][neighbors]
            neighbor_cos = (feats[anchor][None] * feats[neighbors]).sum(dim=-1)
            coord_std = torch.norm(neighbor_coords - neighbor_coords.mean(dim=0, keepdim=True), dim=-1).mean()
            writer.writerow(
                {
                    "anchor_token_idx": anchor,
                    "anchor_norm": float(token_feats[anchor].norm().item()),
                    "anchor_grid_x": float(coords[anchor, 0].item()),
                    "anchor_grid_y": float(coords[anchor, 1].item()),
                    "anchor_grid_z": float(coords[anchor, 2].item()),
                    "anchor_occ_ratio": float(geom["occ_ratio"][anchor].item()),
                    "anchor_surf_ratio": float(geom["surf_ratio"][anchor].item()),
                    "neighbor_mean_grid_x": float(neighbor_coords[:, 0].mean().item()),
                    "neighbor_mean_grid_y": float(neighbor_coords[:, 1].mean().item()),
                    "neighbor_mean_grid_z": float(neighbor_coords[:, 2].mean().item()),
                    "neighbor_std_grid": float(coord_std.item()),
                    "neighbor_mean_occ_ratio": float(neighbor_occ.mean().item()),
                    "neighbor_std_occ_ratio": float(neighbor_occ.std(unbiased=False).item()),
                    "neighbor_mean_surf_ratio": float(neighbor_surf.mean().item()),
                    "neighbor_std_surf_ratio": float(neighbor_surf.std(unbiased=False).item()),
                    "neighbor_mean_cos": float(neighbor_cos.mean().item()),
                }
            )


def main():
    args = parse_args()
    set_seed(args.seed)

    cache_dir = Path(args.cache)
    z_s_path = cache_dir / "coords_zs.pt"
    if not z_s_path.exists():
        raise FileNotFoundError(f"Missing SS cache: {z_s_path}")

    output_dir = Path(args.output_dir) if args.output_dir else cache_dir / "ss_token_probe"
    ensure_dir(output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    z_s = torch.load(z_s_path, map_location=device)
    flow_model, decoder = load_ss_models(args.pretrained, device)

    with torch.no_grad():
        token_feats = encode_ss_tokens(z_s, flow_model)[0].float()
        voxels = decode_ss_voxels(z_s, decoder)

    token_resolution = flow_model.resolution // flow_model.patch_size
    geom = compute_geometry_features(voxels, token_resolution=token_resolution)
    pairwise = sample_pairwise_metrics(token_feats, geom, args.pair_samples, args.seed)
    knn_summary, knn_per_token, nn_idx = knn_metrics(token_feats, geom, args.knn_k, args.seed)
    anchors = select_anchor_indices(token_feats, geom["occ_ratio"], args.anchor_count)

    write_token_csv(output_dir / "token_metrics.csv", token_feats, geom, knn_per_token)
    write_anchor_csv(output_dir / "knn_anchor_summary.csv", anchors, nn_idx, token_feats, geom)

    summary = {
        "cache": str(cache_dir),
        "pretrained": args.pretrained,
        "device": str(device),
        "z_s_shape": list(z_s.shape),
        "token_shape": list(token_feats.shape),
        "decoded_resolution": geom["decoded_resolution"],
        "token_resolution": token_resolution,
        "num_nonempty_token_blocks": int((geom["occ_ratio"] > 0).sum().item()),
        "mean_token_norm": float(token_feats.norm(dim=-1).mean().item()),
        "std_token_norm": float(token_feats.norm(dim=-1).std(unbiased=False).item()),
        "mean_occ_ratio": float(geom["occ_ratio"].mean().item()),
        "mean_surf_ratio": float(geom["surf_ratio"].mean().item()),
        "pairwise": pairwise,
        "knn": knn_summary,
        "artifacts": {
            "token_metrics_csv": str(output_dir / "token_metrics.csv"),
            "knn_anchor_summary_csv": str(output_dir / "knn_anchor_summary.csv"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
