from typing import *
import math

import torch

from . import BACKEND
from ..attn_modify import apply_qk_output_gate, modify_attn_score

if BACKEND == "xformers":
    import xformers.ops as xops
elif BACKEND == "flash_attn":
    import flash_attn
elif BACKEND == "sdpa":
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == "naive":
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = ["scaled_dot_product_attention", "modify_attn_score"]


def _score_from_attention(attn_weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    score_per_head = attn_weight.max(dim=-1).values.permute(0, 2, 1)
    head_weight = torch.softmax(out.norm(dim=-1), dim=-1)
    return (score_per_head * head_weight).sum(dim=-1)


def _naive_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    return_score: bool = False,
    modify: bool = False,
    gate_attn: bool = False,
    gate_mode: str = "logits",
    modify_lambda_scale: float = 0.3,
    modify_max_passes: int = 4,
    modify_stop_conflict: float = 0.5,
    modify_temperature: float = 1.0,
) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if modify:
        logits = modify_attn_score(
            logits,
            lambda_scale=modify_lambda_scale,
            max_passes=modify_max_passes,
            stop_conflict=modify_stop_conflict,
            temperature=modify_temperature,
        )
    attn_weight = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn_weight, v)

    if gate_attn and gate_mode == "logits":
        entropy = -(attn_weight * torch.log(attn_weight + 1e-8)).sum(dim=-1, keepdim=True)
        max_logits = logits.max(dim=-1, keepdim=True).values
        gate = ~((entropy > 6.0) & (max_logits < 1.0))
        out = out * gate.to(out.dtype)

    out = out.permute(0, 2, 1, 3)
    if return_score:
        return out, _score_from_attention(attn_weight, out)
    return out


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor: ...


@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor: ...


@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def scaled_dot_product_attention(*args, **kwargs):
    arg_names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_args = len(args)
    assert num_args in arg_names, f"Invalid number of arguments: {num_args}"

    return_score = bool(kwargs.get("return_score", False))
    modify = bool(kwargs.get("modify", False))
    gate_attn = bool(kwargs.get("gate_attn", False))
    gate_mode = str(kwargs.get("gate_mode", "logits"))
    modify_lambda_scale = float(kwargs.get("modify_lambda_scale", 0.3))
    modify_max_passes = int(kwargs.get("modify_max_passes", 4))
    modify_stop_conflict = float(kwargs.get("modify_stop_conflict", 0.5))
    modify_temperature = float(kwargs.get("modify_temperature", 1.0))
    gate_qk_confidence_threshold = float(kwargs.get("gate_qk_confidence_threshold", 1.0))

    if num_args == 1:
        qkv = args[0]
        assert qkv.ndim == 5 and qkv.shape[2] == 3, f"Expected [N, L, 3, H, C], got {tuple(qkv.shape)}"
        q, k, v = qkv.unbind(dim=2)
    elif num_args == 2:
        q, kv = args
        assert q.ndim == 4 and kv.ndim == 5 and kv.shape[2] == 2
        k, v = kv.unbind(dim=2)
    else:
        q, k, v = args
        assert q.ndim == k.ndim == v.ndim == 4

    if return_score or modify or (gate_attn and gate_mode == "logits") or BACKEND == "naive":
        return _naive_sdpa(
            q,
            k,
            v,
            return_score=return_score,
            modify=modify,
            gate_attn=gate_attn,
            gate_mode=gate_mode,
            modify_lambda_scale=modify_lambda_scale,
            modify_max_passes=modify_max_passes,
            modify_stop_conflict=modify_stop_conflict,
            modify_temperature=modify_temperature,
        )

    if BACKEND == "xformers":
        out = xops.memory_efficient_attention(q, k, v)
        if gate_attn and gate_mode == "post":
            n, lq = q.shape[:2]
            lkv = k.shape[1]
            out = apply_qk_output_gate(
                out.reshape(n * lq, *out.shape[2:]),
                q.reshape(n * lq, *q.shape[2:]),
                k.reshape(n * lkv, *k.shape[2:]),
                [lq] * n,
                [lkv] * n,
                confidence_threshold=gate_qk_confidence_threshold,
            ).reshape_as(out)
        return out
    if BACKEND == "flash_attn":
        if num_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(args[0])
        if num_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, args[1])
        if num_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
        if gate_attn and gate_mode == "post":
            n, lq = q.shape[:2]
            lkv = k.shape[1]
            out = apply_qk_output_gate(
                out.reshape(n * lq, *out.shape[2:]),
                q.reshape(n * lq, *q.shape[2:]),
                k.reshape(n * lkv, *k.shape[2:]),
                [lq] * n,
                [lkv] * n,
                confidence_threshold=gate_qk_confidence_threshold,
            ).reshape_as(out)
        return out
    if BACKEND == "sdpa":
        out = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        out = out.permute(0, 2, 1, 3)
        if gate_attn and gate_mode == "post":
            n, lq = q.shape[:2]
            lkv = k.shape[1]
            out = apply_qk_output_gate(
                out.reshape(n * lq, *out.shape[2:]),
                q.reshape(n * lq, *q.shape[2:]),
                k.reshape(n * lkv, *k.shape[2:]),
                [lq] * n,
                [lkv] * n,
                confidence_threshold=gate_qk_confidence_threshold,
            ).reshape_as(out)
        return out
    raise ValueError(f"Unknown attention backend: {BACKEND}")
