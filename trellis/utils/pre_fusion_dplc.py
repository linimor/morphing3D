from typing import Any, Dict, Optional, Tuple, Union

import torch


def _empty_stats(reason: str) -> Dict[str, Any]:
    return {
        "skipped": True,
        "reason": reason,
        "risk_rate": 0.0,
        "mean_collapse_ratio": 0.0,
        "mean_residual_error": 0.0,
        "residual_threshold": 0.0,
        "mean_delta_norm": 0.0,
        "max_delta_norm": 0.0,
        "direction_error_float32": 0.0,
        "direction_error": 0.0,
    }


def _xyz_coords(coords: torch.Tensor, n_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
    if coords is None or not torch.is_tensor(coords):
        return None
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if coords.ndim != 2 or coords.shape[0] != n_tokens or coords.shape[1] not in (3, 4):
        return None
    if coords.shape[1] == 4:
        coords = coords[:, 1:4]
    coords = coords.to(device=device, dtype=torch.float32)
    if not torch.isfinite(coords).all():
        return None
    return coords


def _chunked_knn(coords: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    n_tokens = coords.shape[0]
    k_eff = min(max(int(k), 1), n_tokens - 1)
    chunk_size = max(int(chunk_size), 1)
    all_indices = torch.arange(n_tokens, device=coords.device)
    knn_chunks = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        dist = torch.cdist(coords[start:end], coords)
        dist[torch.arange(end - start, device=coords.device), all_indices[start:end]] = float("inf")
        knn_chunks.append(dist.topk(k_eff, largest=False, dim=1).indices)

    return torch.cat(knn_chunks, dim=0)


def _as_float_alpha(alpha: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    if torch.is_tensor(alpha):
        return alpha.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(float(alpha), device=device, dtype=torch.float32)


def _pre_fusion_dplc_2d(
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    coords: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    k: int,
    lam: float,
    collapse_th: float,
    residual_quantile: float,
    max_delta_ratio: float,
    chunk_size: int,
    eps: float,
    knn_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    n_tokens = h_src.shape[0]
    xyz = _xyz_coords(coords, n_tokens, h_src.device)
    if xyz is None:
        return h_src, h_tar, _empty_stats("coords_unavailable_or_mismatched")
    if n_tokens <= 1:
        return h_src, h_tar, _empty_stats("not_enough_tokens")

    orig_dtype = h_src.dtype
    h_src_f = h_src.float()
    h_tar_f = h_tar.float()
    alpha_f = _as_float_alpha(alpha, h_src.device)

    h0 = (1.0 - alpha_f) * h_src_f + alpha_f * h_tar_f
    direction = h_tar_f - h_src_f

    if knn_idx is None:
        knn_idx = _chunked_knn(xyz, k=k, chunk_size=chunk_size)
    else:
        knn_idx = knn_idx.to(device=h_src.device, dtype=torch.long)
    h_nei = h0[knn_idx].mean(dim=1)
    corr = h_nei - h0

    direction_norm_sq = direction.square().sum(dim=-1, keepdim=True)
    corr_parallel = (corr * direction).sum(dim=-1, keepdim=True) / (direction_norm_sq + eps) * direction
    corr_perp = corr - corr_parallel

    h0_norm = h0.norm(dim=-1, keepdim=True)
    src_norm = h_src_f.norm(dim=-1, keepdim=True)
    tar_norm = h_tar_f.norm(dim=-1, keepdim=True)
    collapse_ratio = h0_norm / ((1.0 - alpha_f) * src_norm + alpha_f * tar_norm + eps)

    residual_error = (h0 - h_nei).norm(dim=-1, keepdim=True)
    q = max(0.0, min(float(residual_quantile), 1.0))
    residual_th = torch.quantile(residual_error.flatten(), q)
    risk = (collapse_ratio < float(collapse_th)) | (residual_error > residual_th)
    risk_f = risk.to(dtype=torch.float32)

    delta = float(lam) * risk_f * corr_perp
    delta_norm = delta.norm(dim=-1, keepdim=True)
    max_norm = float(max_delta_ratio) * (h0_norm + eps)
    scale = torch.minimum(torch.ones_like(delta_norm), max_norm / (delta_norm + eps))
    delta = delta * scale

    h_src_new_f = h_src_f + delta
    h_tar_new_f = h_tar_f + delta
    direction_error_float32 = ((h_tar_new_f - h_src_new_f) - direction).abs().max()
    h_src_new = h_src_new_f.to(dtype=orig_dtype)
    h_tar_new = h_tar_new_f.to(dtype=orig_dtype)
    direction_error = ((h_tar_new - h_src_new) - (h_tar - h_src)).abs().max()

    stats = {
        "skipped": False,
        "risk_rate": float(risk_f.mean().detach().cpu().item()),
        "mean_collapse_ratio": float(collapse_ratio.mean().detach().cpu().item()),
        "mean_residual_error": float(residual_error.mean().detach().cpu().item()),
        "residual_threshold": float(residual_th.detach().cpu().item()),
        "mean_delta_norm": float(delta.norm(dim=-1).mean().detach().cpu().item()),
        "max_delta_norm": float(delta.norm(dim=-1).max().detach().cpu().item()),
        "direction_error_float32": float(direction_error_float32.detach().cpu().item()),
        "direction_error": float(direction_error.detach().cpu().item()),
    }
    return h_src_new, h_tar_new, stats


def pre_fusion_dplc(
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    coords: Optional[torch.Tensor],
    alpha: Union[float, torch.Tensor],
    enabled: bool = True,
    k: int = 8,
    lam: float = 0.10,
    collapse_th: float = 0.75,
    residual_quantile: float = 0.80,
    max_delta_ratio: float = 0.15,
    chunk_size: int = 1024,
    eps: float = 1e-6,
    return_stats: bool = False,
    knn_idx: Optional[torch.Tensor] = None,
):
    if not enabled:
        result = (h_src, h_tar, _empty_stats("disabled")) if return_stats else (h_src, h_tar)
        return result
    if not torch.is_tensor(h_src) or not torch.is_tensor(h_tar) or h_src.shape != h_tar.shape:
        result = (h_src, h_tar, _empty_stats("shape_mismatch")) if return_stats else (h_src, h_tar)
        return result
    if coords is None:
        result = (h_src, h_tar, _empty_stats("coords_none")) if return_stats else (h_src, h_tar)
        return result

    squeezed = False
    if h_src.ndim == 3 and h_src.shape[0] == 1:
        h_src_2d = h_src.squeeze(0)
        h_tar_2d = h_tar.squeeze(0)
        squeezed = True
    elif h_src.ndim == 2:
        h_src_2d = h_src
        h_tar_2d = h_tar
    else:
        result = (h_src, h_tar, _empty_stats("unsupported_feature_shape")) if return_stats else (h_src, h_tar)
        return result

    try:
        h_src_new, h_tar_new, stats = _pre_fusion_dplc_2d(
            h_src_2d,
            h_tar_2d,
            coords,
            alpha,
            k=k,
            lam=lam,
            collapse_th=collapse_th,
            residual_quantile=residual_quantile,
            max_delta_ratio=max_delta_ratio,
            chunk_size=chunk_size,
            eps=eps,
            knn_idx=knn_idx,
        )
    except Exception as exc:
        stats = _empty_stats(f"exception:{type(exc).__name__}")
        h_src_new, h_tar_new = h_src_2d, h_tar_2d

    if squeezed:
        h_src_new = h_src_new.unsqueeze(0)
        h_tar_new = h_tar_new.unsqueeze(0)

    if return_stats:
        return h_src_new, h_tar_new, stats
    return h_src_new, h_tar_new
