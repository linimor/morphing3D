import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from trellis.modules.sparse import SparseTensor
from trellis.modules.sparse.attention import modules as sparse_attn_modules
from trellis.modules.sparse.attention import full_attn as sparse_full_attn
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import seed_everything


PROBE_STATE = {
    "enabled": False,
    "variant": None,
    "records": [],
    "context": {},
    "sample_q": 768,
    "sample_k": 2048,
}


def _tensor_stats(x):
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _gini(x):
    x = x.float().flatten()
    if x.numel() == 0:
        return 0.0
    total = x.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    x = torch.sort(x).values
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float(((2 * idx - n - 1) * x).sum().div(n * total).item())


def _pick_even_indices(length, max_items, device):
    if length <= max_items:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, max_items, device=device).round().long().unique()


def _extract_qk(args):
    if len(args) == 1:
        qkv = args[0]
        if not isinstance(qkv, SparseTensor):
            return None
        q = qkv.feats[:, 0]
        k = qkv.feats[:, 1]
        return q, k, "self"

    if len(args) == 2:
        q, kv = args
        if isinstance(q, SparseTensor):
            q_feats = q.feats
        elif torch.is_tensor(q):
            q_feats = q.reshape(-1, q.shape[-2], q.shape[-1])
        else:
            return None

        if isinstance(kv, SparseTensor):
            k_feats = kv.feats[:, 0]
        elif torch.is_tensor(kv):
            k_feats = kv.reshape(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])[:, 0]
        else:
            return None
        return q_feats, k_feats, "cross"

    if len(args) == 3:
        q, k, _ = args
        if isinstance(q, SparseTensor):
            q_feats = q.feats
        elif torch.is_tensor(q):
            q_feats = q.reshape(-1, q.shape[-2], q.shape[-1])
        else:
            return None

        if isinstance(k, SparseTensor):
            k_feats = k.feats
        elif torch.is_tensor(k):
            k_feats = k.reshape(-1, k.shape[-2], k.shape[-1])
        else:
            return None
        return q_feats, k_feats, "cross"

    return None


@torch.no_grad()
def _record_attention_stats(args):
    extracted = _extract_qk(args)
    if extracted is None:
        return

    q, k, inferred_type = extracted
    if q.ndim != 3 or k.ndim != 3:
        return

    ctx = dict(PROBE_STATE["context"])
    attn_type = ctx.get("attn_type", inferred_type)
    max_q = int(PROBE_STATE["sample_q"])
    max_k = int(PROBE_STATE["sample_k"])

    q_idx = _pick_even_indices(q.shape[0], max_q, q.device)
    k_idx = _pick_even_indices(k.shape[0], max_k, k.device)
    q_s = q.index_select(0, q_idx).float()
    k_s = k.index_select(0, k_idx).float()

    # [T, H, C] -> [H, T, C]
    qh = q_s.permute(1, 0, 2).contiguous()
    kh = k_s.permute(1, 0, 2).contiguous()
    logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(qh.shape[-1])
    probs = torch.softmax(logits, dim=-1)

    row_sum = probs.sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    key_mass = probs.sum(dim=1)
    key_mass_frac = key_mass / max(float(q_s.shape[0]), 1.0)
    argmax_key = probs.argmax(dim=-1)
    max_argmax_count = []
    unique_argmax_ratio = []
    for h in range(argmax_key.shape[0]):
        counts = torch.bincount(argmax_key[h], minlength=k_s.shape[0]).float()
        max_argmax_count.append(float(counts.max().item()))
        unique_argmax_ratio.append(float((counts > 0).sum().item() / max(argmax_key.shape[1], 1)))

    record = {
        "variant": PROBE_STATE["variant"],
        "attn_type": attn_type,
        "step_idx": ctx.get("step_idx"),
        "block_idx": ctx.get("block_idx"),
        "alpha": ctx.get("alpha"),
        "morphing_idx": ctx.get("morphing_idx"),
        "modify": bool(ctx.get("modify", False)),
        "gate_attn": bool(ctx.get("gate_attn", False)),
        "ot_coherence_enabled": bool(ctx.get("ot_coherence_enabled", False)),
        "q_tokens": int(q.shape[0]),
        "k_tokens": int(k.shape[0]),
        "sample_q": int(q_s.shape[0]),
        "sample_k": int(k_s.shape[0]),
        "row_sum_mean": _tensor_stats(row_sum)["mean"],
        "row_sum_min": _tensor_stats(row_sum)["min"],
        "row_sum_max": _tensor_stats(row_sum)["max"],
        "max_prob_mean": _tensor_stats(max_prob)["mean"],
        "max_prob_p95": float(torch.quantile(max_prob.flatten(), 0.95).item()),
        "entropy_mean": _tensor_stats(entropy)["mean"],
        "entropy_min": _tensor_stats(entropy)["min"],
        "key_mass_frac_mean": _tensor_stats(key_mass_frac)["mean"],
        "key_mass_frac_std": _tensor_stats(key_mass_frac)["std"],
        "key_mass_frac_p95": float(torch.quantile(key_mass_frac.flatten(), 0.95).item()),
        "key_mass_frac_max": _tensor_stats(key_mass_frac)["max"],
        "key_mass_gini": _gini(key_mass_frac),
        "argmax_max_count_mean": float(np.mean(max_argmax_count)),
        "argmax_max_count_max": float(np.max(max_argmax_count)),
        "unique_argmax_ratio_mean": float(np.mean(unique_argmax_ratio)),
    }
    PROBE_STATE["records"].append(record)


def install_probe():
    original_attention = sparse_attn_modules.sparse_scaled_dot_product_attention
    original_forward = sparse_attn_modules.SparseMultiHeadAttention.forward

    def attention_wrapper(*args, **kwargs):
        if PROBE_STATE["enabled"]:
            _record_attention_stats(args)
        return original_attention(*args, **kwargs)

    def forward_wrapper(self, x, context=None, step_idx=0, block_idx=0, cache_idx=0, **kwargs):
        prev = PROBE_STATE["context"]
        if PROBE_STATE["enabled"]:
            PROBE_STATE["context"] = {
                "attn_type": self._type,
                "step_idx": int(step_idx),
                "block_idx": int(block_idx),
                "cache_idx": int(cache_idx),
                "alpha": float(kwargs.get("alpha", -1.0)),
                "morphing_idx": int(kwargs.get("morphing_idx", -1)),
                "modify": bool(kwargs.get("modify", False)),
                "gate_attn": bool(kwargs.get("gate_attn", False)),
                "ot_coherence_enabled": bool(kwargs.get("ot_coherence_enabled", False)),
            }
        try:
            return original_forward(self, x, context=context, step_idx=step_idx, block_idx=block_idx, cache_idx=cache_idx, **kwargs)
        finally:
            PROBE_STATE["context"] = prev

    sparse_attn_modules.sparse_scaled_dot_product_attention = attention_wrapper
    sparse_full_attn.sparse_scaled_dot_product_attention = attention_wrapper
    sparse_attn_modules.SparseMultiHeadAttention.forward = forward_wrapper


def variant_params(name):
    params = {
        "modify": False,
        "gate_attn": False,
        "ot_coherence_enabled": False,
    }
    if name == "modify":
        params["modify"] = True
    elif name == "gate":
        params["gate_attn"] = True
    elif name == "ot":
        params["ot_coherence_enabled"] = True
    elif name == "ot_modify_gate":
        params.update({"modify": True, "gate_attn": True, "ot_coherence_enabled": True})
    elif name != "base":
        raise ValueError(f"Unknown variant: {name}")
    return params


def summarize(records):
    groups = {}
    for rec in records:
        key = (rec["variant"], rec["attn_type"])
        groups.setdefault(key, []).append(rec)

    summary = []
    for (variant, attn_type), items in sorted(groups.items()):
        row = {
            "variant": variant,
            "attn_type": attn_type,
            "calls": len(items),
        }
        for field in [
            "max_prob_mean",
            "max_prob_p95",
            "entropy_mean",
            "key_mass_frac_max",
            "key_mass_frac_p95",
            "key_mass_gini",
            "argmax_max_count_mean",
            "argmax_max_count_max",
            "unique_argmax_ratio_mean",
        ]:
            row[field] = float(np.mean([x[field] for x in items]))
        summary.append(row)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Short-run slat attention mass probe for Godzilla -> bee morphing.")
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--src", default="./assets/example_morphing/Godzilla.png")
    parser.add_argument("--tar", default="./assets/example_morphing/bee.png")
    parser.add_argument("--src-cache", default="./outputs/cache/Godzilla/cache")
    parser.add_argument("--tar-cache", default="./outputs/cache/bee/cache")
    parser.add_argument("--out-dir", default="./outputs/diagnostics/slat_attn_godzilla_bee")
    parser.add_argument("--variants", nargs="+", default=["base", "modify", "gate", "ot", "ot_modify_gate"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.67, 0.33])
    parser.add_argument("--ss-steps", type=int, default=4)
    parser.add_argument("--slat-steps", type=int, default=4)
    parser.add_argument("--sample-q", type=int, default=768)
    parser.add_argument("--sample-k", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    PROBE_STATE["sample_q"] = args.sample_q
    PROBE_STATE["sample_k"] = args.sample_k
    install_probe()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    base_params = {
        "morphing_num": len(args.alphas) + 2,
        "src_load_cache_path": args.src_cache,
        "tar_load_cache_path": args.tar_cache,
        "init_morphing_flag": True,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": False,
        "sa_use": False,
        "modify_lambda_scale": 0.8,
        "ot_coherence_stage": "ss",
        "ot_anchor_patch_size": 4,
        "ot_max_anchors": 512,
        "ot_cost_pos_weight": 0.5,
        "ot_cost_feat_weight": 0.8,
        "ot_sinkhorn_eps": 0.05,
        "ot_sinkhorn_iters": 80,
        "ot_filter_k": 16,
        "ot_filter_sigma_pos": 2.0,
        "ot_filter_sigma_motion": 2.0,
        "ot_filter_lambda": 0.3,
        "ot_filter_use_confidence": True,
        "ot_filter_start_step_ratio": 1.0,
        "ot_filter_end_step_ratio": 0.0,
        "ot_debug": False,
        "return_intermediate": True,
    }

    run_meta = {
        "src": args.src,
        "tar": args.tar,
        "variants": args.variants,
        "alphas": args.alphas,
        "ss_steps": args.ss_steps,
        "slat_steps": args.slat_steps,
        "sample_q": args.sample_q,
        "sample_k": args.sample_k,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for variant in args.variants:
        variant_cache = out_dir / variant / "cache"
        if variant_cache.exists():
            shutil.rmtree(variant_cache)
        variant_cache.mkdir(parents=True, exist_ok=True)
        params = dict(base_params)
        params.update(variant_params(variant))
        params["save_cache_path"] = str(variant_cache)
        params.pop("_ot_motion_field_cache", None)

        PROBE_STATE["variant"] = variant
        for idx, alpha in enumerate(args.alphas, start=1):
            seed_everything(args.seed)
            params["alpha"] = float(alpha)
            params["morphing_idx"] = idx
            params["tfsa_cache_idx"] = idx - 1
            params["tfsa_alpha"] = 0.8
            print(f"[probe] variant={variant} alpha={alpha:.2f} ss_steps={args.ss_steps} slat_steps={args.slat_steps}")
            PROBE_STATE["enabled"] = True
            with torch.no_grad():
                pipeline.run_morphing(
                    src_img=src_img,
                    tar_img=tar_img,
                    morphing_params=params,
                    seed=args.seed,
                    sparse_structure_sampler_params={"steps": args.ss_steps},
                    slat_sampler_params={"steps": args.slat_steps},
                    formats=[],
                )
            PROBE_STATE["enabled"] = False

    summary = summarize(PROBE_STATE["records"])
    payload = {
        "meta": run_meta,
        "summary": summary,
        "records": PROBE_STATE["records"],
        "notes": [
            "Statistics are sampled from slat SparseMultiHeadAttention q/k tensors before the original attention kernel runs.",
            "key_mass_frac is attention mass received by each sampled key divided by sampled query count; high max/gini indicates sink-like concentration.",
            "modify/gate flags are recorded from morphing_params; this probe does not change core attention behavior.",
        ],
    }
    (out_dir / "slat_attn_mass_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_dir / "slat_attn_mass_summary.csv", summary)
    write_csv(out_dir / "slat_attn_mass_records.csv", PROBE_STATE["records"])

    print(f"[probe] wrote {out_dir / 'slat_attn_mass_probe.json'}")
    for row in summary:
        print(
            f"[summary] {row['variant']:>14} {row['attn_type']:>5} calls={row['calls']:3d} "
            f"mass_max={row['key_mass_frac_max']:.4f} gini={row['key_mass_gini']:.4f} "
            f"argmax_max={row['argmax_max_count_max']:.1f} entropy={row['entropy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
