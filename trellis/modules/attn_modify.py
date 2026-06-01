from typing import *

import torch


def modify_attn_score(
    attn_score: torch.Tensor,
    lambda_scale: float = 3,
    max_passes: int = 12,
    stop_conflict: float = 0.5,
    temperature: float = 1.0,
    impl: str = "legacy",
) -> torch.Tensor:
    """Reduce many-query-to-one-key conflicts in raw attention logits."""
    if impl == "fixed":
        return modify_attn_score_fixed(
            attn_score,
            lambda_scale=lambda_scale,
            max_passes=max_passes,
            stop_conflict=stop_conflict,
            temperature=temperature,
        )
    score = attn_score.float().clone()
    dtype = attn_score.dtype
    bsz, heads, query_len, key_len = score.shape
    flat = score.reshape(bsz * heads, query_len, key_len)

    for _ in range(max(int(max_passes), 0)):
        best_val, best_key = flat.max(dim=-1)
        counts = torch.zeros(flat.shape[0], key_len, device=flat.device, dtype=flat.dtype)
        counts.scatter_add_(1, best_key, torch.ones_like(best_val))
        overload = torch.relu(counts - 1.0)
        if float(overload.sum(dim=1).mean()) <= float(stop_conflict):
            break

        winner = torch.full_like(counts, -torch.inf)
        winner.scatter_reduce_(1, best_key, best_val, reduce="amax", include_self=True)
        winner_for_query = winner.gather(1, best_key)
        crowd_for_query = overload.gather(1, best_key)
        loser = (counts.gather(1, best_key) > 1) & (best_val < winner_for_query) & (best_val > 0)
        if not bool(loser.any()):
            break

        loser_idx = loser.nonzero(as_tuple=False)
        m_idx, q_idx = loser_idx[:, 0], loser_idx[:, 1]
        k_idx = best_key[m_idx, q_idx]
        cur = flat[m_idx, q_idx, k_idx]
        penalty = float(lambda_scale) * (1.0 + crowd_for_query[m_idx, q_idx]) * winner_for_query[m_idx, q_idx]
        flat[m_idx, q_idx, k_idx] = torch.clamp(cur - penalty, min=0.0)

    return flat.reshape(bsz, heads, query_len, key_len).to(dtype)


def modify_attn_score_fixed(
    attn_score: torch.Tensor,
    lambda_scale: float = 3,
    max_passes: int = 12,
    stop_conflict: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    CGAR conflict reduction with the legacy update rule, but without per-pass
    CPU synchronization or nonzero-based writes.
    """
    max_passes = max(int(max_passes), 0)
    if max_passes == 0:
        return attn_score

    dtype = attn_score.dtype
    score = attn_score.float().clone()
    bsz, heads, query_len, key_len = score.shape
    flat = score.reshape(bsz * heads, query_len, key_len)

    for _ in range(max_passes):
        best_val, best_key = flat.max(dim=-1)
        counts = torch.zeros(flat.shape[0], key_len, device=flat.device, dtype=flat.dtype)
        counts.scatter_add_(1, best_key, torch.ones_like(best_val))
        overload = torch.relu(counts - 1.0)
        keep_correcting = overload.sum(dim=1).mean() > float(stop_conflict)

        winner = torch.full_like(counts, -torch.inf)
        winner.scatter_reduce_(1, best_key, best_val, reduce="amax", include_self=True)
        winner_for_query = winner.gather(1, best_key)
        crowd_for_query = overload.gather(1, best_key)
        loser = keep_correcting & (counts.gather(1, best_key) > 1) & (best_val < winner_for_query) & (best_val > 0)

        penalty = float(lambda_scale) * (1.0 + crowd_for_query) * winner_for_query
        new_val = torch.where(loser, torch.clamp(best_val - penalty, min=0.0), best_val)
        flat.scatter_(2, best_key.unsqueeze(-1), new_val.unsqueeze(-1))

    return flat.reshape(bsz, heads, query_len, key_len).to(dtype)


def qk_sink_precheck(
    q: torch.Tensor,
    k: torch.Tensor,
    q_seqlen: List[int],
    kv_seqlen: List[int],
    *,
    threshold: float = 2.5,
    min_query_tokens: int = 64,
    min_key_tokens: int = 64,
) -> bool:
    """
    Cheap q/k-only attn-sink risk check.

    This does not compute q @ k.T. It catches keys that are likely to become
    sinks because their norm is unusually large or they align with the mean
    query direction for many queries in the same block.
    """
    if threshold <= 0:
        return True

    q_start = 0
    kv_start = 0
    eps = 1e-6
    with torch.no_grad():
        for lq, lkv in zip(q_seqlen, kv_seqlen):
            if lq < min_query_tokens or lkv < min_key_tokens:
                q_start += lq
                kv_start += lkv
                continue

            q_i = q[q_start:q_start + lq].float()     # [Lq, H, C]
            k_i = k[kv_start:kv_start + lkv].float()  # [Lk, H, C]

            k_norm = k_i.norm(dim=-1).transpose(0, 1)  # [H, Lk]
            norm_z = (k_norm.max(dim=-1).values - k_norm.mean(dim=-1)) / (k_norm.std(dim=-1) + eps)

            q_mean = torch.nn.functional.normalize(q_i.mean(dim=0), dim=-1)  # [H, C]
            k_dir = torch.nn.functional.normalize(k_i.permute(1, 0, 2), dim=-1)  # [H, Lk, C]
            align = (k_dir * q_mean[:, None, :]).sum(dim=-1)
            align_z = (align.max(dim=-1).values - align.mean(dim=-1)) / (align.std(dim=-1) + eps)

            risk = torch.maximum(norm_z, align_z).max()
            if bool(risk > threshold):
                return True

            q_start += lq
            kv_start += lkv

    return False


def apply_sink_penalty(
    logits: torch.Tensor,
    *,
    lambda_scale: float = 0.8,
    threshold: float = 0.15,
    top_count_weight: float = 0.5,
    mass_weight: float = 0.5,
    penalty_type: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize keys with abnormal top-1 count or column attention mass."""
    dtype = logits.dtype
    work = logits.float()
    heads, query_len, key_len = work.shape
    if query_len == 0 or key_len == 0:
        attn_weight = torch.softmax(work, dim=-1).to(dtype)
        return logits, attn_weight

    top_key = work.argmax(dim=-1)  # [H, Lq]
    top_count = torch.zeros(heads, key_len, device=work.device, dtype=work.dtype)
    top_count.scatter_add_(1, top_key, torch.ones_like(top_key, dtype=work.dtype))
    top_ratio = top_count / max(float(query_len), 1.0)

    attn_weight = torch.softmax(work, dim=-1)
    col_mass = attn_weight.sum(dim=-2) / max(float(query_len), 1.0)
    sink_score = float(top_count_weight) * top_ratio + float(mass_weight) * col_mass
    if penalty_type == "log":
        eps = 1e-6
        penalty = float(lambda_scale) * torch.relu(torch.log((sink_score + eps) / max(float(threshold), eps)))
    else:
        penalty = float(lambda_scale) * torch.relu(sink_score - float(threshold))
    if bool((penalty > 0).any()):
        work = work - penalty[:, None, :]
        attn_weight = torch.softmax(work, dim=-1)

    return work.to(dtype), attn_weight.to(dtype)


def apply_sink_cap(
    logits: torch.Tensor,
    *,
    cap: float = 0.2,
    max_iters: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cap per-key average attention mass and renormalize rows."""
    dtype = logits.dtype
    probs = torch.softmax(logits.float(), dim=-1)
    query_len = probs.shape[-2]
    eps = 1e-8
    for _ in range(max(int(max_iters), 0)):
        col_mass = probs.sum(dim=-2) / max(float(query_len), 1.0)
        scale = torch.where(col_mass > float(cap), float(cap) / (col_mass + eps), torch.ones_like(col_mass))
        if not bool((scale < 1).any()):
            break
        probs = probs * scale[:, None, :]
        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
    pseudo_logits = torch.log(probs + eps)
    return pseudo_logits.to(dtype), probs.to(dtype)


def apply_qk_output_gate(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_seqlen: List[int],
    kv_seqlen: List[int],
    *,
    confidence_threshold: float = 1.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Post-attention gate that does not need logits/probs.

    For each query/head, q_norm * max(k_norm) / sqrt(dim) is an upper bound on
    the largest possible qk logit in that block. If even this upper bound is
    small, the attention result is likely low-confidence, so the output is
    gated after the fused attention kernel returns.
    """
    if confidence_threshold <= 0:
        return out

    head_dim = q.shape[-1]
    attn_scale = (1.0 / (head_dim ** 0.5)) if scale is None else scale
    gated = out
    q_start = 0
    kv_start = 0
    for lq, lkv in zip(q_seqlen, kv_seqlen):
        q_i = q[q_start:q_start + lq].float()     # [Lq, H, C]
        k_i = k[kv_start:kv_start + lkv].float()  # [Lk, H, C]
        q_norm = q_i.norm(dim=-1)                 # [Lq, H]
        max_k_norm = k_i.norm(dim=-1).max(dim=0).values  # [H]
        confidence = q_norm * max_k_norm[None, :] * float(attn_scale)
        gate = confidence >= float(confidence_threshold)
        gated = gated.clone()
        gated[q_start:q_start + lq] = gated[q_start:q_start + lq] * gate[..., None].to(gated.dtype)
        q_start += lq
        kv_start += lkv

    return gated
