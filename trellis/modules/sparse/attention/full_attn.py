from typing import *
import torch
from .. import SparseTensor
from .. import DEBUG, ATTN
from ...attn_modify import apply_qk_output_gate, apply_sink_cap, apply_sink_penalty, modify_attn_score, qk_sink_precheck

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_scaled_dot_product_attention',
]


import math
import torch

def _native_varlen_attention_with_probs(
    q: torch.Tensor,                  # [Tq, H, C]
    k: torch.Tensor,                  # [Tk, H, C]
    v: torch.Tensor,                  # [Tk, H, Co]
    q_seqlen: List[int],
    kv_seqlen: List[int],
    *,
    scale: Optional[float] = None,
    return_probs: bool = True,
    modify: bool = False,
    gate_attn: bool = False,
    gate_mode: str = "logits",
    modify_mode: str = "legacy",
    modify_lambda_scale: float = 0.3,
    modify_max_passes: int = 4,
    modify_stop_conflict: float = 0.5,
    modify_temperature: float = 1.0,
    modify_sink_threshold: float = 0.15,
    modify_sink_top_count_weight: float = 0.5,
    modify_sink_mass_weight: float = 0.5,
    modify_sink_penalty_type: str = "linear",
    modify_sink_cap_iters: int = 4,
    gate_entropy_threshold: float = 6.0,
    gate_max_logit_threshold: float = 1.0,
) -> Tuple[torch.Tensor, Optional[List]]:
    """
    Native variable-length full attention.

    Args:
        q: Flattened queries, shape [Tq, H, C]
        k: Flattened keys,    shape [Tk, H, C]
        v: Flattened values,  shape [Tk, H, Co]
        q_seqlen: per-batch query lengths
        kv_seqlen: per-batch key/value lengths
        scale: attention scale. If None, use 1 / sqrt(C)
        return_probs: whether to also return softmax attention maps

    Returns:
        out:
            shape [Tq, H, Co]
        probs_list:
            list of attention probabilities, each with shape [H, Lq_i, Lk_i]
            or None if return_probs=False
    """
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.shape[1] == k.shape[1] == v.shape[1], "head count mismatch"
    assert len(q_seqlen) == len(kv_seqlen), "batch block count mismatch"
    assert sum(q_seqlen) == q.shape[0], "sum(q_seqlen) != Tq"
    assert sum(kv_seqlen) == k.shape[0] == v.shape[0], "sum(kv_seqlen) != Tk"

    head_dim = q.shape[-1]
    attn_scale = (1.0 / math.sqrt(head_dim)) if scale is None else scale

    outs: List[torch.Tensor] = []
    probs_list: List = []

    q_start = 0
    kv_start = 0

    for lq, lkv in zip(q_seqlen, kv_seqlen):
        # q_i: [Lq, H, C] -> [H, Lq, C]
        q_i = q[q_start:q_start + lq].permute(1, 0, 2).contiguous()
        k_i = k[kv_start:kv_start + lkv].permute(1, 0, 2).contiguous()
        v_i = v[kv_start:kv_start + lkv].permute(1, 0, 2).contiguous()

        # scores: [H, Lq, Lk]
        scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * attn_scale
        if modify:
            if modify_mode == "sink_cap":
                scores, probs = apply_sink_cap(
                    scores,
                    cap=modify_sink_threshold,
                    max_iters=modify_sink_cap_iters,
                )
            elif modify_mode == "sink_penalty":
                scores, probs = apply_sink_penalty(
                    scores,
                    lambda_scale=modify_lambda_scale,
                    threshold=modify_sink_threshold,
                    top_count_weight=modify_sink_top_count_weight,
                    mass_weight=modify_sink_mass_weight,
                    penalty_type=modify_sink_penalty_type,
                )
            else:
                scores = modify_attn_score(
                    scores.unsqueeze(0),
                    lambda_scale=modify_lambda_scale,
                    max_passes=modify_max_passes,
                    stop_conflict=modify_stop_conflict,
                    temperature=modify_temperature,
                ).squeeze(0)
                probs = torch.softmax(scores, dim=-1)
        else:
            probs = torch.softmax(scores, dim=-1)

        # out_i: [H, Lq, Co]
        out_i = torch.matmul(probs, v_i)
        if gate_attn and gate_mode == "logits":
            entropy = -(probs.float() * torch.log(probs.float() + 1e-8)).sum(dim=-1, keepdim=True)
            max_logits = scores.float().max(dim=-1, keepdim=True).values
            gate = ~((entropy > float(gate_entropy_threshold)) & (max_logits < float(gate_max_logit_threshold)))
            out_i = out_i * gate.to(out_i.dtype)

        # -> [Lq, H, Co]
        out_i = out_i.permute(1, 0, 2).contiguous()

        outs.append(out_i)
        if return_probs:
            probs_list.append(probs)

        q_start += lq
        kv_start += lkv

    out = torch.cat(outs, dim=0)  # [Tq, H, Co]
    return out, (probs_list if return_probs else None)

@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: torch.Tensor, v: torch.Tensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: SparseTensor, v: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    return_score = kwargs.get("return_score", False)
    modify = bool(kwargs.get("modify", False))
    gate_attn = bool(kwargs.get("gate_attn", False))
    gate_mode = str(kwargs.get("gate_mode", "logits"))
    modify_mode = str(kwargs.get("modify_mode", "legacy"))
    modify_precheck = bool(kwargs.get("modify_precheck", False))
    modify_precheck_threshold = float(kwargs.get("modify_precheck_threshold", 2.5))
    modify_precheck_min_query_tokens = int(kwargs.get("modify_precheck_min_query_tokens", 64))
    modify_precheck_min_key_tokens = int(kwargs.get("modify_precheck_min_key_tokens", 64))
    modify_lambda_scale = float(kwargs.get("modify_lambda_scale", 0.3))
    modify_max_passes = int(kwargs.get("modify_max_passes", 4))
    modify_stop_conflict = float(kwargs.get("modify_stop_conflict", 0.5))
    modify_temperature = float(kwargs.get("modify_temperature", 1.0))
    modify_sink_threshold = float(kwargs.get("modify_sink_threshold", 0.15))
    modify_sink_top_count_weight = float(kwargs.get("modify_sink_top_count_weight", 0.5))
    modify_sink_mass_weight = float(kwargs.get("modify_sink_mass_weight", 0.5))
    modify_sink_penalty_type = str(kwargs.get("modify_sink_penalty_type", "linear"))
    modify_sink_cap_iters = int(kwargs.get("modify_sink_cap_iters", 4))
    gate_entropy_threshold = float(kwargs.get("gate_entropy_threshold", 6.0))
    gate_max_logit_threshold = float(kwargs.get("gate_max_logit_threshold", 1.0))
    gate_qk_confidence_threshold = float(kwargs.get("gate_qk_confidence_threshold", 1.0))
    probs_list=None
    num_all_args = len(args) # + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                assert (s.coords[s.layout[i]] == i).all(), f"SparseScaledDotProductSelfAttention: batch index mismatch"
        if num_all_args in [2, 3]:
            assert q.shape[:2] == [1, sum(q_seqlen)], f"SparseScaledDotProductSelfAttention: q shape mismatch"
        if num_all_args == 3:
            assert k.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: k shape mismatch"
            assert v.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: v shape mismatch"
    run_native = return_score or modify or (gate_attn and gate_mode == "logits")
    if modify and modify_precheck:
        if num_all_args == 1:
            q_precheck, k_precheck = qkv[:, 0], qkv[:, 1]
        elif num_all_args == 2:
            q_precheck, k_precheck = q, kv[:, 0]
        else:
            q_precheck, k_precheck = q, k
        run_native = return_score or (gate_attn and gate_mode == "logits") or qk_sink_precheck(
            q_precheck,
            k_precheck,
            q_seqlen,
            kv_seqlen,
            threshold=modify_precheck_threshold,
            min_query_tokens=modify_precheck_min_query_tokens,
            min_key_tokens=modify_precheck_min_key_tokens,
        )
        modify = run_native and modify

    if run_native:
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        out, probs_list = _native_varlen_attention_with_probs(
            q=q,
            k=k,
            v=v,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            return_probs=return_score,
            modify=modify,
            gate_attn=gate_attn,
            gate_mode=gate_mode,
            modify_mode=modify_mode,
            modify_lambda_scale=modify_lambda_scale,
            modify_max_passes=modify_max_passes,
            modify_stop_conflict=modify_stop_conflict,
            modify_temperature=modify_temperature,
            modify_sink_threshold=modify_sink_threshold,
            modify_sink_top_count_weight=modify_sink_top_count_weight,
            modify_sink_mass_weight=modify_sink_mass_weight,
            modify_sink_penalty_type=modify_sink_penalty_type,
            modify_sink_cap_iters=modify_sink_cap_iters,
            gate_entropy_threshold=gate_entropy_threshold,
            gate_max_logit_threshold=gate_max_logit_threshold,
        )
    else:
        if ATTN == 'xformers':
            if num_all_args == 1:
                q, k, v = qkv.unbind(dim=1)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=1)
            q_flat, k_flat = q, k
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
            out = xops.memory_efficient_attention(q, k, v, mask)[0]
            if gate_attn and gate_mode == "post":
                out = apply_qk_output_gate(
                    out,
                    q_flat,
                    k_flat,
                    q_seqlen,
                    kv_seqlen,
                    confidence_threshold=gate_qk_confidence_threshold,
                )
        elif ATTN == 'flash_attn':
            cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
            if num_all_args in [2, 3]:
                cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            if num_all_args == 1:
                out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
                if gate_attn and gate_mode == "post":
                    q_gate, k_gate, _ = qkv.unbind(dim=1)
                    out = apply_qk_output_gate(
                        out,
                        q_gate,
                        k_gate,
                        q_seqlen,
                        kv_seqlen,
                        confidence_threshold=gate_qk_confidence_threshold,
                    )
            elif num_all_args == 2:
                out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
                if gate_attn and gate_mode == "post":
                    k_gate, _ = kv.unbind(dim=1)
                    out = apply_qk_output_gate(
                        out,
                        q,
                        k_gate,
                        q_seqlen,
                        kv_seqlen,
                        confidence_threshold=gate_qk_confidence_threshold,
                    )
            elif num_all_args == 3:
                out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
                if gate_attn and gate_mode == "post":
                    out = apply_qk_output_gate(
                        out,
                        q,
                        k,
                        q_seqlen,
                        kv_seqlen,
                        confidence_threshold=gate_qk_confidence_threshold,
                    )
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
    
    if s is not None:
        if probs_list is None:
            return s.replace(out)
        else:
            return s.replace(out), probs_list
    else:
        if probs_list is None:
            return out.reshape(N, L, H, -1)
        else:
            return out.reshape(N, L, H, -1), probs_list
