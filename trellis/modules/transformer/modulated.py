from typing import *
import os
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet
from ...utils.morphing_utils import *
from ...utils.ot_coherence import (
    get_ot_filter_lambda,
    get_ss_token_positions,
    ot_motion_coherent_filter,
)


def _ss_ca_oc_cache_path(kwargs: dict, cache_idx: int, step_idx: int, block_idx: int) -> Optional[str]:
    save_cache_path = kwargs.get("save_cache_path", None)
    if save_cache_path is None:
        return None
    return os.path.join(
        save_cache_path,
        f"ss_ca_oc_morphing{cache_idx}_step{step_idx}_block{block_idx}.pt",
    )


def _load_ss_endpoint_occ16(kwargs: dict, tag: str) -> Optional[dict]:
    endpoint_occ = kwargs.get("ss_endpoint_occ16", None)
    if endpoint_occ is not None:
        return endpoint_occ

    occ_path = kwargs.get("ss_endpoint_occ16_path", None)
    if occ_path is None:
        return None

    if not os.path.exists(occ_path):
        print(f"[{tag}] warning: endpoint occupancy file not found: {occ_path}; skip")
        return None
    return torch.load(occ_path, map_location="cpu")


def _ss_rot90_tokens(x: torch.Tensor, k: int, grid_size: int) -> torch.Tensor:
    if k % 4 == 0:
        return x
    if x.ndim != 3:
        raise ValueError(f"Expected [B, L, C] SS tokens, got {tuple(x.shape)}")
    batch, token_count, channels = x.shape
    grid_size = int(grid_size)
    if grid_size <= 0 or token_count != grid_size ** 3:
        inferred = round(token_count ** (1.0 / 3.0))
        if inferred ** 3 != token_count:
            raise ValueError(f"Cannot rotate SS CA tokens with L={token_count}")
        grid_size = inferred
    return torch.rot90(
        x.reshape(batch, grid_size, grid_size, grid_size, channels),
        k=int(k) % 4,
        dims=(1, 2),
    ).reshape(batch, token_count, channels)


def _ss_ca_oc_descriptor(x: torch.Tensor, desc_dim: int) -> torch.Tensor:
    desc_dim = max(1, min(int(desc_dim), x.shape[-1]))
    if desc_dim == x.shape[-1]:
        return x
    pad = (-x.shape[-1]) % desc_dim
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], desc_dim, -1).mean(dim=-1)


def _ss_ca_oc_best_rotation(
    current: torch.Tensor,
    ref: Any,
    grid_size: int,
    branch: str,
    kwargs: dict,
    step_idx: int,
    block_idx: int,
) -> int:
    if isinstance(ref, dict):
        ref = ref.get(branch, None)
    if not torch.is_tensor(ref) or ref.ndim != current.ndim or tuple(ref.shape[:-1]) != tuple(current.shape[:-1]):
        return 0

    ref = ref.to(device=current.device, dtype=current.dtype)
    ref_norm = F.normalize(ref.float(), dim=-1, eps=1e-6)
    desc_dim = int(ref.shape[-1])
    losses = []
    for k in range(4):
        cand = _ss_rot90_tokens(current, k, grid_size)
        cand = _ss_ca_oc_descriptor(cand, desc_dim)
        cand_norm = F.normalize(cand.float(), dim=-1, eps=1e-6)
        losses.append((cand_norm - ref_norm).square().mean())
    best_k = int(torch.argmin(torch.stack(losses)).item())
    del ref
    return best_k


def _ss_ca_oc_align_pair(
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    kwargs: dict,
    step_idx: int,
    block_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not kwargs.get("ss_ca_oc_flag", False):
        return h_src, h_tar

    morphing_idx = kwargs.get("morphing_idx", None)
    if morphing_idx is None:
        return h_src, h_tar

    grid_size = int(kwargs.get("ss_ca_oc_grid_size", 16))
    prev_idx = kwargs.get("tfsa_cache_idx", None)
    src_k = 0
    tar_k = 0
    cache_path = _ss_ca_oc_cache_path(kwargs, prev_idx, step_idx, block_idx) if prev_idx is not None else None

    if cache_path is not None and os.path.exists(cache_path):
        ref = torch.load(cache_path, map_location=h_tar.device)
        src_k = _ss_ca_oc_best_rotation(h_src, ref, grid_size, "src", kwargs, step_idx, block_idx)
        tar_k = _ss_ca_oc_best_rotation(h_tar, ref, grid_size, "tar", kwargs, step_idx, block_idx)
        if kwargs.get("delete_loaded_ca_oc_cache", kwargs.get("delete_loaded_tfsa_cache", False)):
            try:
                os.remove(cache_path)
            except FileNotFoundError:
                pass
        del ref

    h_src = _ss_rot90_tokens(h_src, src_k, grid_size)
    h_tar = _ss_rot90_tokens(h_tar, tar_k, grid_size)
    cur_path = _ss_ca_oc_cache_path(kwargs, morphing_idx, step_idx, block_idx)
    if cur_path is not None and not os.path.exists(cur_path):
        desc_dim = int(kwargs.get("ss_ca_oc_desc_dim", 32))
        torch.save({
            "src": _ss_ca_oc_descriptor(h_src.detach(), desc_dim).cpu().half(),
            "tar": _ss_ca_oc_descriptor(h_tar.detach(), desc_dim).cpu().half(),
        }, cur_path)
    return h_src, h_tar


def _dump_ddpf_v0_debug(debug: dict, kwargs: dict, step_idx: int, block_idx: int) -> None:
    return


def _apply_ddpf_v0_birth_correction(
    h: torch.Tensor,
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    kwargs: dict,
    step_idx: int,
    block_idx: int,
) -> torch.Tensor:
    if not kwargs.get("enable_ddpf_v0", False):
        return h

    endpoint_occ = _load_ss_endpoint_occ16(kwargs, "DDPF V0")
    if endpoint_occ is None:
        return h

    try:
        if h.ndim != 3 or h_src.shape != h.shape or h_tar.shape != h.shape:
            print(
                f"[DDPF V0] warning: expected h/h_src/h_tar [B, L, C] with matching shapes, "
                f"got {tuple(h.shape)}, {tuple(h_src.shape)}, {tuple(h_tar.shape)}; skip correction"
            )
            return h

        ss_token_coords = torch.as_tensor(endpoint_occ["ss_token_coords"]).detach()
        occ_s_16 = torch.as_tensor(endpoint_occ["occ_s_16"]).detach()
        occ_t_16 = torch.as_tensor(endpoint_occ["occ_t_16"]).detach()

        token_count = h.shape[1]
        if ss_token_coords.shape != (token_count, 3):
            print(f"[DDPF V0] warning: ss_token_coords shape {tuple(ss_token_coords.shape)} does not match L={token_count}; skip correction")
            return h
        if occ_s_16.shape[0] != token_count or occ_t_16.shape[0] != token_count:
            print(
                f"[DDPF V0] warning: occ shapes {tuple(occ_s_16.shape)}, {tuple(occ_t_16.shape)} "
                f"do not match L={token_count}; skip correction"
            )
            return h
        if not torch.isfinite(ss_token_coords.float()).all() or not torch.isfinite(occ_s_16.float()).all() or not torch.isfinite(occ_t_16.float()).all():
            print("[DDPF V0] warning: endpoint occupancy contains NaN/Inf; skip correction")
            return h

        coords = ss_token_coords.to(device=h.device, dtype=torch.float32)
        occ_s = (occ_s_16 > 0).to(device=h.device)
        occ_t = (occ_t_16 > 0).to(device=h.device)
        shared_mask = occ_s & occ_t
        birth_mask = occ_t & (~occ_s)
        birth_count = int(birth_mask.sum().item())
        shared_count = int(shared_mask.sum().item())
        if birth_count == 0:
            phi_mode = str(kwargs.get("ddpf_phi_mode", "distance"))
            _dump_ddpf_v0_debug(
                {
                    "alpha": float(kwargs.get("alpha", 0.5)),
                    "p": 1.0 - float(kwargs.get("alpha", 0.5)),
                    "ddpf_phi_mode": phi_mode,
                    "birth_count": birth_count,
                    "shared_count": shared_count,
                    "birth_component_count": 0,
                    "birth_component_top_sizes": [],
                    "phi_birth": torch.empty(0, device=h.device, dtype=h.dtype),
                    "alpha_local_birth": torch.empty(0, device=h.device, dtype=h.dtype),
                    "alpha_local_min": None,
                    "alpha_local_mean": None,
                    "alpha_local_max": None,
                    "correction_norm": 0.0,
                    "phi_min": None,
                    "phi_max": None,
                },
                kwargs,
                step_idx,
                block_idx,
            )
            return h
        if shared_count == 0:
            print("[DDPF V0] warning: no shared tokens available; skip correction")
            return h

        alpha = float(kwargs.get("alpha", 0.5))
        p = 1.0 - alpha
        tau = max(float(kwargs.get("ddpf_tau", 0.10)), 1e-6)
        lambda_ddpf = float(kwargs.get("ddpf_lambda", 0.30))
        margin0 = float(kwargs.get("ddpf_margin0", 0.05))
        margin_k = float(kwargs.get("ddpf_margin_k", 0.40))
        phi_mode = str(kwargs.get("ddpf_phi_mode", "distance"))
        front_pad_arg = kwargs.get("ddpf_front_pad", None)
        front_pad = 3.0 * tau if front_pad_arg is None else float(front_pad_arg)
        endpoint_eps = float(kwargs.get("ddpf_endpoint_eps", 1e-4))
        F_front = p + front_pad * (p ** 2)

        birth_coords = coords[birth_mask]
        shared_coords = coords[shared_mask]
        d_i = torch.cdist(birth_coords[None], shared_coords[None]).amin(dim=-1).squeeze(0)
        d_min = d_i.min()
        d_max = d_i.max()
        if torch.isclose(d_max, d_min, atol=1e-6, rtol=1e-6):
            phi_distance = torch.zeros_like(d_i)
        else:
            phi_distance = (d_i - d_min) / (d_max - d_min)

        birth_coords_int = birth_coords.round().long().detach().cpu()
        coord_to_birth_idx = {tuple(coord.tolist()): idx for idx, coord in enumerate(birth_coords_int)}
        visited = set()
        birth_components = []
        for coord_tuple, start_idx in coord_to_birth_idx.items():
            if start_idx in visited:
                continue
            stack = [start_idx]
            visited.add(start_idx)
            component = []
            while stack:
                cur_idx = stack.pop()
                component.append(cur_idx)
                x, y, z = birth_coords_int[cur_idx].tolist()
                for nb in (
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y + 1, z),
                    (x, y - 1, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
                ):
                    nb_idx = coord_to_birth_idx.get(nb, None)
                    if nb_idx is not None and nb_idx not in visited:
                        visited.add(nb_idx)
                        stack.append(nb_idx)
            birth_components.append(component)
        birth_component_count = len(birth_components)
        birth_component_top_sizes = sorted((len(component) for component in birth_components), reverse=True)[:10]

        if phi_mode == "distance":
            phi = phi_distance
        elif phi_mode == "quantile":
            order = torch.argsort(d_i)
            rank = torch.empty_like(d_i)
            denom = max(birth_count - 1, 1)
            rank[order] = torch.arange(birth_count, device=h.device, dtype=d_i.dtype) / float(denom)
            phi = rank
        elif phi_mode == "component_quantile":
            phi = phi_distance.clone()
            for component in birth_components:
                if len(component) >= 3:
                    comp_idx = torch.as_tensor(component, device=h.device, dtype=torch.long)
                    comp_d = d_i[comp_idx]
                    comp_order = torch.argsort(comp_d)
                    comp_phi = torch.empty_like(comp_d)
                    comp_phi[comp_order] = torch.arange(len(component), device=h.device, dtype=d_i.dtype) / float(max(len(component) - 1, 1))
                    phi[comp_idx] = comp_phi
        else:
            print(f"[DDPF V0] warning: unsupported ddpf_phi_mode={phi_mode}; using distance")
            phi_mode = "distance"
            phi = phi_distance

        alpha_tensor = torch.as_tensor(alpha, device=h.device, dtype=h.dtype)
        phi_h = phi.to(dtype=h.dtype)
        alpha_local = torch.sigmoid((phi_h - torch.as_tensor(F_front, device=h.device, dtype=h.dtype)) / torch.as_tensor(tau, device=h.device, dtype=h.dtype))
        margin = torch.as_tensor(margin0, device=h.device, dtype=h.dtype) + torch.as_tensor(margin_k, device=h.device, dtype=h.dtype) * phi_h
        alpha_local = torch.maximum(alpha_local, alpha_tensor - margin)
        alpha_local = alpha_local.clamp(0.0, 1.0)
        if alpha >= 1.0 - endpoint_eps:
            alpha_local = torch.ones_like(alpha_local)
        elif alpha <= endpoint_eps:
            alpha_local = torch.zeros_like(alpha_local)
        alpha_local_view = alpha_local.reshape(1, birth_count, 1)

        correction = lambda_ddpf * (alpha_local_view - alpha_tensor) * (h_src[:, birth_mask, :] - h_tar[:, birth_mask, :])
        h_out = h.clone()
        h_out[:, birth_mask, :] = h[:, birth_mask, :] + correction
        correction_norm = float(correction.float().norm().item())

        _dump_ddpf_v0_debug(
            {
                "alpha": alpha,
                "p": p,
                "F": F_front,
                "front_pad": front_pad,
                "ddpf_endpoint_eps": endpoint_eps,
                "ddpf_phi_mode": phi_mode,
                "birth_count": birth_count,
                "shared_count": shared_count,
                "birth_component_count": birth_component_count,
                "birth_component_top_sizes": birth_component_top_sizes,
                "phi_birth": phi_h,
                "alpha_local_birth": alpha_local,
                "alpha_local_min": float(alpha_local.float().min().item()),
                "alpha_local_mean": float(alpha_local.float().mean().item()),
                "alpha_local_max": float(alpha_local.float().max().item()),
                "correction_norm": correction_norm,
                "phi_min": float(phi.float().min().item()),
                "phi_max": float(phi.float().max().item()),
            },
            kwargs,
            step_idx,
            block_idx,
        )
        return h_out
    except Exception as exc:
        print(f"[DDPF V0] warning: failed to apply correction: {exc}")
        return h


def _dump_cmf_v0_debug(debug: dict, kwargs: dict, step_idx: int, block_idx: int) -> None:
    return


def _apply_cmf_v0_birth_field(
    h: torch.Tensor,
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    kwargs: dict,
    step_idx: int,
    block_idx: int,
) -> torch.Tensor:
    if not kwargs.get("enable_cmf_v0", False):
        return h

    endpoint_occ = _load_ss_endpoint_occ16(kwargs, "CMF V0")
    if endpoint_occ is None:
        return h
    save_cache_path = kwargs.get("save_cache_path", None)

    try:
        if h.ndim != 3 or h_src.shape != h.shape or h_tar.shape != h.shape:
            print(
                f"[CMF V0] warning: expected h/h_src/h_tar [B, L, C] with matching shapes, "
                f"got {tuple(h.shape)}, {tuple(h_src.shape)}, {tuple(h_tar.shape)}; skip CMF"
            )
            return h

        ss_token_coords = torch.as_tensor(endpoint_occ["ss_token_coords"]).detach()
        occ_s_16 = torch.as_tensor(endpoint_occ["occ_s_16"]).detach()
        occ_t_16 = torch.as_tensor(endpoint_occ["occ_t_16"]).detach()

        token_count = h.shape[1]
        if ss_token_coords.shape != (token_count, 3):
            print(f"[CMF V0] warning: ss_token_coords shape {tuple(ss_token_coords.shape)} does not match L={token_count}; skip CMF")
            return h
        if occ_s_16.shape[0] != token_count or occ_t_16.shape[0] != token_count:
            print(f"[CMF V0] warning: occ shapes do not match L={token_count}; skip CMF")
            return h

        coords_int = ss_token_coords.to(device=h.device).round().long()
        occ_s = (occ_s_16 > 0).to(device=h.device)
        occ_t = (occ_t_16 > 0).to(device=h.device)
        shared_mask = occ_s & occ_t
        birth_mask = occ_t & (~occ_s)
        death_mask = occ_s & (~occ_t)
        birth_idx = torch.nonzero(birth_mask, as_tuple=False).flatten()
        birth_count = int(birth_idx.numel())
        shared_count = int(shared_mask.sum().item())
        if birth_count == 0:
            return h

        beta = float(kwargs.get("cmf_beta", 0.35))
        lambda_iso = float(kwargs.get("cmf_lambda_iso", 1.0))
        lambda_var = float(kwargs.get("cmf_lambda_var", 0.5))
        smooth_steps = int(kwargs.get("cmf_smooth_steps", 5))
        smooth_rho = float(kwargs.get("cmf_smooth_rho", 0.4))
        smooth_steps = max(smooth_steps, 0)
        smooth_rho = min(max(smooth_rho, 0.0), 1.0)
        alpha = float(kwargs.get("alpha", 0.5))
        p = min(max(1.0 - alpha, 0.0), 1.0)

        coord_to_token = {tuple(c.tolist()): i for i, c in enumerate(coords_int.detach().cpu())}
        birth_pos = {int(idx.item()): j for j, idx in enumerate(birth_idx)}
        birth_neighbors = [[] for _ in range(birth_count)]
        target_neighbor_count = torch.zeros(birth_count, device=h.device, dtype=h.dtype)
        edge_pairs = []

        occ_t_cpu = occ_t.detach().cpu()
        for j, token_i in enumerate(birth_idx.detach().cpu().tolist()):
            x, y, z = coords_int[token_i].detach().cpu().tolist()
            for nb_coord in (
                (x + 1, y, z),
                (x - 1, y, z),
                (x, y + 1, z),
                (x, y - 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ):
                nb_token = coord_to_token.get(nb_coord, None)
                if nb_token is None:
                    continue
                if bool(occ_t_cpu[nb_token].item()):
                    target_neighbor_count[j] += 1.0
                nb_birth_j = birth_pos.get(nb_token, None)
                if nb_birth_j is not None:
                    birth_neighbors[j].append(nb_birth_j)
                    if j < nb_birth_j:
                        edge_pairs.append((j, nb_birth_j))

        isolation = 1.0 - (target_neighbor_count / 6.0)
        h_tar_birth = h_tar[:, birth_idx, :].float()
        feat_var = torch.zeros(birth_count, device=h.device, dtype=torch.float32)
        for j, nbs in enumerate(birth_neighbors):
            if len(nbs) == 0:
                continue
            nb_idx = torch.as_tensor(nbs, device=h.device, dtype=torch.long)
            dist = (h_tar_birth[:, nb_idx, :] - h_tar_birth[:, j:j + 1, :]).norm(dim=-1)
            feat_var[j] = dist.mean()

        cost = lambda_iso * isolation.float() + lambda_var * feat_var
        cost_std = cost.std(unbiased=False).clamp_min(1e-6)
        cost_z = (cost - cost.mean()) / cost_std
        g_raw = (-cost_z).clamp(-1.0, 1.0).to(dtype=h.dtype)
        g = g_raw.clone()
        for _ in range(smooth_steps):
            g_next = g.clone()
            for j, nbs in enumerate(birth_neighbors):
                if len(nbs) == 0:
                    continue
                nb_idx = torch.as_tensor(nbs, device=h.device, dtype=torch.long)
                nb_mean = g[nb_idx].mean()
                g_next[j] = (1.0 - smooth_rho) * g[j] + smooth_rho * nb_mean
            g = g_next.clamp(-1.0, 1.0)

        q_birth = (torch.as_tensor(p, device=h.device, dtype=h.dtype) + beta * p * (1.0 - p) * g).clamp(0.0, 1.0)
        alpha_local_birth = 1.0 - q_birth

        h_out = h.clone()
        alpha_view = alpha_local_birth.reshape(1, birth_count, 1)
        h_out[:, birth_idx, :] = alpha_view * h_src[:, birth_idx, :] + (1.0 - alpha_view) * h_tar[:, birth_idx, :]

        if edge_pairs:
            e0 = torch.as_tensor([a for a, _ in edge_pairs], device=h.device, dtype=torch.long)
            e1 = torch.as_tensor([b for _, b in edge_pairs], device=h.device, dtype=torch.long)
            edge_q_diff = (q_birth[e0] - q_birth[e1]).abs().float()
            edge_q_diff_mean = float(edge_q_diff.mean().item())
            edge_q_diff_p95 = float(torch.quantile(edge_q_diff, 0.95).item())
            edge_q_diff_max = float(edge_q_diff.max().item())
        else:
            edge_q_diff_mean = None
            edge_q_diff_p95 = None
            edge_q_diff_max = None

        delta_mean = None
        delta_max = None
        morphing_idx = int(kwargs.get("morphing_idx", -1))
        if morphing_idx > 1 and save_cache_path is not None:
            prev_dir = os.path.join(save_cache_path, "cmf_v0_debug")
            prev_name = (
                f"cmf_v0_debug_morph{morphing_idx - 1}_step{int(kwargs.get('step_idx', step_idx if step_idx is not None else -1))}"
                f"_block{int(kwargs.get('block_idx', block_idx if block_idx is not None else -1))}"
            )
            try:
                prev_files = [name for name in os.listdir(prev_dir) if name.startswith(prev_name) and name.endswith(".pt")]
                if prev_files:
                    prev = torch.load(os.path.join(prev_dir, sorted(prev_files)[-1]), map_location=h.device)
                    prev_alpha_local = prev.get("alpha_local_birth", None)
                    if torch.is_tensor(prev_alpha_local) and prev_alpha_local.numel() == alpha_local_birth.numel():
                        delta = (alpha_local_birth.float() - prev_alpha_local.to(device=h.device).float()).abs()
                        delta_mean = float(delta.mean().item())
                        delta_max = float(delta.max().item())
            except Exception:
                delta_mean = None
                delta_max = None

        _dump_cmf_v0_debug(
            {
                "method": "CMF-V0",
                "alpha": alpha,
                "p": p,
                "cmf_beta": beta,
                "cmf_lambda_iso": lambda_iso,
                "cmf_lambda_var": lambda_var,
                "cmf_smooth_steps": smooth_steps,
                "cmf_smooth_rho": smooth_rho,
                "birth_count": birth_count,
                "shared_count": shared_count,
                "death_count": int(death_mask.sum().item()),
                "birth_edge_count": len(edge_pairs),
                "birth_idx": birth_idx,
                "birth_coords": coords_int[birth_idx],
                "isolation": isolation.detach(),
                "feat_var": feat_var.detach(),
                "cost": cost.detach(),
                "g_raw": g_raw,
                "g_smooth": g,
                "q_birth": q_birth,
                "alpha_local_birth": alpha_local_birth,
                "edge_q_diff_mean": edge_q_diff_mean,
                "edge_q_diff_p95": edge_q_diff_p95,
                "edge_q_diff_max": edge_q_diff_max,
                "delta_alpha_local_mean_between_frames": delta_mean,
                "delta_alpha_local_max_between_frames": delta_max,
            },
            kwargs,
            step_idx,
            block_idx,
        )
        return h_out
    except Exception as exc:
        print(f"[CMF V0] warning: failed to apply CMF: {exc}")
        return h


def _dump_mavf_v0_debug(debug: dict, kwargs: dict, step_idx: int, block_idx: int) -> None:
    return


def _apply_mavf_v0_birth_field(
    h: torch.Tensor,
    h_src: torch.Tensor,
    h_tar: torch.Tensor,
    kwargs: dict,
    step_idx: int,
    block_idx: int,
) -> torch.Tensor:
    if not kwargs.get("mavf_enable", False):
        return h

    endpoint_occ = _load_ss_endpoint_occ16(kwargs, "MAVF V0")
    if endpoint_occ is None:
        return h

    try:
        if h.ndim != 3 or h_src.shape != h.shape or h_tar.shape != h.shape:
            print(
                f"[MAVF V0] warning: expected h/h_src/h_tar [B, L, C] with matching shapes, "
                f"got {tuple(h.shape)}, {tuple(h_src.shape)}, {tuple(h_tar.shape)}; skip MAVF"
            )
            return h

        ss_token_coords = torch.as_tensor(endpoint_occ["ss_token_coords"]).detach()
        occ_s_16 = torch.as_tensor(endpoint_occ["occ_s_16"]).detach()
        occ_t_16 = torch.as_tensor(endpoint_occ["occ_t_16"]).detach()

        token_count = h.shape[1]
        if ss_token_coords.shape != (token_count, 3):
            print(f"[MAVF V0] warning: ss_token_coords shape {tuple(ss_token_coords.shape)} does not match L={token_count}; skip MAVF")
            return h
        if occ_s_16.shape[0] != token_count or occ_t_16.shape[0] != token_count:
            print(f"[MAVF V0] warning: occ shapes do not match L={token_count}; skip MAVF")
            return h

        coords_int = ss_token_coords.to(device=h.device).round().long()
        occ_s = (occ_s_16 > 0).to(device=h.device)
        occ_t = (occ_t_16 > 0).to(device=h.device)
        shared_mask = occ_s & occ_t
        birth_mask = occ_t & (~occ_s)
        birth_idx = torch.nonzero(birth_mask, as_tuple=False).flatten()
        birth_count = int(birth_idx.numel())
        if birth_count == 0:
            return h

        lambda_feat = float(kwargs.get("mavf_lambda_feat", 1.0))
        lambda_delta = float(kwargs.get("mavf_lambda_delta", 0.3))
        eta_rank = min(max(float(kwargs.get("mavf_eta_rank", 0.2)), 0.0), 1.0)
        sigma = max(float(kwargs.get("mavf_sigma", 0.20)), 1e-6)
        eps = max(float(kwargs.get("mavf_eps", 0.15)), 0.0)
        integration_bins = max(int(kwargs.get("mavf_integration_bins", 128)), 8)
        alpha = float(kwargs.get("alpha", 0.5))
        p = min(max(1.0 - alpha, 0.0), 1.0)

        coord_to_token = {tuple(c.tolist()): i for i, c in enumerate(coords_int.detach().cpu())}
        birth_pos = {int(idx.item()): j for j, idx in enumerate(birth_idx)}
        shared_cpu = shared_mask.detach().cpu()
        h_tar_birth = h_tar[:, birth_idx, :].float()
        h_src_birth = h_src[:, birth_idx, :].float()
        h_tar_desc = h_tar_birth.mean(dim=0)
        delta_norm = (h_tar_birth - h_src_birth).norm(dim=-1).mean(dim=0)

        adjacency = [[] for _ in range(birth_count)]
        edge_pairs = []
        roots = []
        for j, token_i in enumerate(birth_idx.detach().cpu().tolist()):
            x, y, z = coords_int[token_i].detach().cpu().tolist()
            touches_shared = False
            for nb_coord in (
                (x + 1, y, z),
                (x - 1, y, z),
                (x, y + 1, z),
                (x, y - 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ):
                nb_token = coord_to_token.get(nb_coord, None)
                if nb_token is None:
                    continue
                if bool(shared_cpu[nb_token].item()):
                    touches_shared = True
                nb_birth_j = birth_pos.get(nb_token, None)
                if nb_birth_j is not None:
                    adjacency[j].append(nb_birth_j)
                    if j < nb_birth_j:
                        edge_pairs.append((j, nb_birth_j))
            if touches_shared:
                roots.append(j)
        if not roots:
            roots = list(range(birth_count))

        edge_costs = {}
        for i, j in edge_pairs:
            cos_ij = F.cosine_similarity(h_tar_desc[i:i + 1], h_tar_desc[j:j + 1], dim=-1, eps=1e-6)[0]
            feat_cost = 1.0 - float(cos_ij.clamp(-1.0, 1.0).item())
            delta_cost = float((delta_norm[i] - delta_norm[j]).abs().item())
            cost = 1.0 + lambda_feat * feat_cost + lambda_delta * delta_cost
            edge_costs[(i, j)] = cost
            edge_costs[(j, i)] = cost

        inf = float("inf")
        a_raw_list = [inf for _ in range(birth_count)]
        heap = []
        for r in roots:
            a_raw_list[r] = 0.0
            heapq.heappush(heap, (0.0, r))
        while heap:
            cur_cost, cur = heapq.heappop(heap)
            if cur_cost > a_raw_list[cur]:
                continue
            for nb in adjacency[cur]:
                nxt_cost = cur_cost + edge_costs.get((cur, nb), 1.0)
                if nxt_cost < a_raw_list[nb]:
                    a_raw_list[nb] = nxt_cost
                    heapq.heappush(heap, (nxt_cost, nb))
        max_finite = max((v for v in a_raw_list if v < inf), default=0.0)
        a_raw_list = [max_finite if v == inf else v for v in a_raw_list]
        a_raw = torch.as_tensor(a_raw_list, device=h.device, dtype=torch.float32)
        a_min = a_raw.min()
        a_max = a_raw.max()
        if torch.isclose(a_max, a_min, atol=1e-6, rtol=1e-6):
            a_norm = torch.zeros_like(a_raw)
        else:
            a_norm = (a_raw - a_min) / (a_max - a_min)

        order = torch.argsort(a_raw)
        a_rank = torch.empty_like(a_raw)
        a_rank[order] = torch.arange(birth_count, device=h.device, dtype=torch.float32) / float(max(birth_count - 1, 1))
        a_final = ((1.0 - eta_rank) * a_norm + eta_rank * a_rank).clamp(0.0, 1.0)

        u_grid = torch.linspace(0.0, 1.0, integration_bins, device=h.device, dtype=torch.float32)
        t_grid = 3.0 * u_grid.square() - 2.0 * u_grid.square() * u_grid
        v_grid = eps + torch.exp(-((t_grid.unsqueeze(0) - a_final.unsqueeze(1)).square()) / (2.0 * sigma * sigma))
        seg = 0.5 * (v_grid[:, 1:] + v_grid[:, :-1]) * (u_grid[1:] - u_grid[:-1]).unsqueeze(0)
        cum = torch.cat([torch.zeros(birth_count, 1, device=h.device), torch.cumsum(seg, dim=1)], dim=1)
        total = cum[:, -1].clamp_min(1e-8)
        p_tensor = torch.as_tensor(p, device=h.device, dtype=torch.float32)
        idx_hi = int(torch.searchsorted(u_grid, p_tensor, right=False).item())
        if idx_hi <= 0:
            integ_p = torch.zeros_like(total)
            v_at_p = v_grid[:, 0]
        elif idx_hi >= integration_bins:
            integ_p = total
            v_at_p = v_grid[:, -1]
        else:
            u0 = u_grid[idx_hi - 1]
            u1 = u_grid[idx_hi]
            frac = ((p_tensor - u0) / (u1 - u0).clamp_min(1e-8)).clamp(0.0, 1.0)
            v0 = v_grid[:, idx_hi - 1]
            v1 = v_grid[:, idx_hi]
            v_at_p = v0 + frac * (v1 - v0)
            partial = 0.5 * (v0 + v_at_p) * (p_tensor - u0)
            integ_p = cum[:, idx_hi - 1] + partial
        q_birth = (integ_p / total).clamp(0.0, 1.0).to(dtype=h.dtype)
        h_out = h.clone()
        q_view = q_birth.reshape(1, birth_count, 1)
        h_out[:, birth_idx, :] = (1.0 - q_view) * h_src[:, birth_idx, :] + q_view * h_tar[:, birth_idx, :]
        return h_out
    except Exception as exc:
        print(f"[MAVF V0] warning: failed to apply MAVF: {exc}")
        return h


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, step_idx: int, block_idx: int, **kwargs):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        if len(kwargs) > 0:
            if kwargs["ss_tfsa_flag"]:
                h_cur, score_cur = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, return_score=True, **kwargs)
                h_prev, score_prev = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, cache_idx=-1, return_score=True, **kwargs)
                # score_diff = score_prev - score_cur
                # lambda_ = 5.0
                # delta_alpha = 0.3 * torch.tanh(lambda_ * score_diff)
                # delta_alpha = delta_alpha.unsqueeze(-1)  # [B, Lq, 1]
                # tfsa_alpha = torch.clamp(kwargs["tfsa_alpha"] - delta_alpha, 0.0, 1.0)
                h = feature_interp(h_cur, h_prev, kwargs["tfsa_alpha"], interp_mode="linear")
            else:
                h = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, **kwargs)
        else:
            h = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx)

        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)

        if len(kwargs) > 0:
            if kwargs.get("ss_mca_cond_fuse_flag", False):
                alpha = kwargs.get("ss_mca_cond_fuse_alpha", kwargs.get("alpha", 0.5))
                fused_context = feature_interp(context, kwargs["tar_cond"], alpha, interp_mode="linear")
                h = self.cross_attn(x=h, context=fused_context, step_idx=step_idx, block_idx=block_idx, **kwargs)
            elif kwargs["ss_mca_flag"]:
                attn_kwargs = {
                    "modify": kwargs.get("modify", False),
                    "gate_attn": kwargs.get("gate_attn", False),
                    "modify_lambda_scale": kwargs.get("modify_lambda_scale", 0.3),
                    "modify_max_passes": kwargs.get("modify_max_passes", 4),
                    "modify_stop_conflict": kwargs.get("modify_stop_conflict", 0.5),
                    "modify_temperature": kwargs.get("modify_temperature", 1.0),
                }
                h_src = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx, **attn_kwargs)
                h_tar = self.cross_attn(x=h, context=kwargs["tar_cond"], step_idx=step_idx, block_idx=block_idx, **attn_kwargs)
                h_src, h_tar = _ss_ca_oc_align_pair(h_src, h_tar, kwargs, step_idx, block_idx)
                # print(score_src.shape) #[1, 4096]
                # src_score = score_src          # [B, Lq]
                # tar_score = score_tar          # [B, Lq]
                # score_diff = tar_score - src_score   # [B, Lq]
                # lambda_ = 5.0
                # delta_alpha = 0.3 * torch.tanh(lambda_ * score_diff)   # [B, Lq]
                # delta_alpha = delta_alpha.unsqueeze(-1)  # [B, Lq, 1]
                # alpha = torch.clamp(kwargs["alpha"] - delta_alpha, 0.0, 1.0)
                h = feature_interp(h_src, h_tar, kwargs["alpha"], interp_mode="linear")
                if kwargs.get("mavf_enable", False):
                    h = _apply_mavf_v0_birth_field(h, h_src, h_tar, kwargs, step_idx, block_idx)
                elif kwargs.get("enable_cmf_v0", False):
                    h = _apply_cmf_v0_birth_field(h, h_src, h_tar, kwargs, step_idx, block_idx)
                else:
                    h = _apply_ddpf_v0_birth_correction(h, h_src, h_tar, kwargs, step_idx, block_idx)
                if kwargs.get("ot_coherence_enabled", False) and kwargs.get("ot_coherence_stage", "ss") == "ss":
                    motion_field = kwargs.get("ot_motion_field", None)
                    if motion_field is not None:
                        try:
                            alpha = float(kwargs["alpha"])
                            src_center = motion_field["src_center"].to(device=h.device, dtype=h.dtype)
                            disp = motion_field["disp"].to(device=h.device, dtype=h.dtype)
                            anchor_pos = src_center + (1.0 - alpha) * disp
                            lam = get_ot_filter_lambda(
                                kwargs.get("ot_filter_lambda", 0.3),
                                step_idx,
                                kwargs.get("ss_num_steps", kwargs.get("steps", None)),
                                kwargs.get("ot_filter_start_step_ratio", 1.0),
                                kwargs.get("ot_filter_end_step_ratio", 0.0),
                            )
                            token_pos = get_ss_token_positions(
                                h,
                                ss_coords=kwargs.get("ss_coords", None),
                                grid_size=kwargs.get("ss_token_grid_size", None),
                            )
                            h = ot_motion_coherent_filter(
                                out=h,
                                token_pos=token_pos,
                                anchor_pos=anchor_pos,
                                anchor_motion=disp,
                                anchor_conf=motion_field.get("conf", None),
                                anchor_mask=motion_field.get("mask", None),
                                k_neighbors=kwargs.get("ot_filter_k", 16),
                                sigma_pos=kwargs.get("ot_filter_sigma_pos", 2.0),
                                sigma_motion=kwargs.get("ot_filter_sigma_motion", 2.0),
                                lambda0=lam,
                                use_confidence=kwargs.get("ot_filter_use_confidence", True),
                            )
                        except Exception:
                            pass
            else:
                h = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx, **kwargs)
        else:
            h = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx)

        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, step_idx: int, block_idx: int, **kwargs):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, step_idx, block_idx, **kwargs, use_reentrant=False)
        else:
            return self._forward(x, mod, context, step_idx, block_idx, **kwargs)
        
