import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from trellis.modules.attn_modify import apply_sink_cap, apply_sink_penalty
from trellis.modules.sparse import SparseTensor
from trellis.modules.sparse.attention import full_attn as sparse_full_attn
from trellis.modules.sparse.attention import modules as sparse_attn_modules
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import seed_everything


STATE = {
    "enabled": False,
    "variant": None,
    "records": [],
    "context": {},
    "sample_q": 512,
    "sample_k": 1024,
}


def pick_even(length, max_items, device):
    if length <= max_items:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, max_items, device=device).round().long().unique()


def gini(x):
    x = x.float().flatten()
    total = x.sum()
    if x.numel() == 0 or float(total.item()) <= 0:
        return 0.0
    x = torch.sort(x).values
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float(((2 * idx - n - 1) * x).sum().div(n * total).item())


def mass_stats(probs):
    max_prob = probs.max(dim=-1).values
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    key_mass_frac = probs.sum(dim=1) / max(float(probs.shape[1]), 1.0)
    argmax_key = probs.argmax(dim=-1)
    max_counts = []
    unique_ratios = []
    for h in range(argmax_key.shape[0]):
        counts = torch.bincount(argmax_key[h], minlength=probs.shape[-1]).float()
        max_counts.append(float(counts.max().item()))
        unique_ratios.append(float((counts > 0).sum().item() / max(argmax_key.shape[1], 1)))
    return {
        "max_prob_mean": float(max_prob.mean().item()),
        "max_prob_p95": float(torch.quantile(max_prob.flatten(), 0.95).item()),
        "entropy_mean": float(entropy.mean().item()),
        "key_mass_frac_max": float(key_mass_frac.max().item()),
        "key_mass_frac_p95": float(torch.quantile(key_mass_frac.flatten(), 0.95).item()),
        "key_mass_gini": gini(key_mass_frac),
        "argmax_max_count_mean": float(np.mean(max_counts)),
        "argmax_max_count_max": float(np.max(max_counts)),
        "unique_argmax_ratio_mean": float(np.mean(unique_ratios)),
    }


def extract_qk(args):
    if len(args) == 1 and isinstance(args[0], SparseTensor):
        qkv = args[0].feats
        return qkv[:, 0], qkv[:, 1], "self"
    if len(args) == 2:
        q, kv = args
        if isinstance(q, SparseTensor):
            q = q.feats
        elif torch.is_tensor(q):
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
        else:
            return None
        if isinstance(kv, SparseTensor):
            k = kv.feats[:, 0]
        elif torch.is_tensor(kv):
            k = kv.reshape(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])[:, 0]
        else:
            return None
        return q, k, "cross"
    if len(args) == 3:
        q, k, _ = args
        q = q.feats if isinstance(q, SparseTensor) else q.reshape(-1, q.shape[-2], q.shape[-1])
        k = k.feats if isinstance(k, SparseTensor) else k.reshape(-1, k.shape[-2], k.shape[-1])
        return q, k, "cross"
    return None


@torch.no_grad()
def record_attention(args, kwargs):
    extracted = extract_qk(args)
    if extracted is None:
        return
    q, k, inferred_type = extracted
    if q.ndim != 3 or k.ndim != 3:
        return

    q_idx = pick_even(q.shape[0], int(STATE["sample_q"]), q.device)
    k_idx = pick_even(k.shape[0], int(STATE["sample_k"]), k.device)
    q_s = q.index_select(0, q_idx).float().permute(1, 0, 2).contiguous()
    k_s = k.index_select(0, k_idx).float().permute(1, 0, 2).contiguous()
    logits = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(q_s.shape[-1])
    before = mass_stats(torch.softmax(logits, dim=-1))

    after = None
    if bool(kwargs.get("modify", False)) and kwargs.get("modify_mode", "legacy") in ["sink_penalty", "sink_cap"]:
        if kwargs.get("modify_mode") == "sink_cap":
            _, probs_after = apply_sink_cap(
                logits,
                cap=float(kwargs.get("modify_sink_threshold", 0.2)),
                max_iters=int(kwargs.get("modify_sink_cap_iters", 4)),
            )
        else:
            _, probs_after = apply_sink_penalty(
                logits,
                lambda_scale=float(kwargs.get("modify_lambda_scale", 0.8)),
                threshold=float(kwargs.get("modify_sink_threshold", 0.15)),
                top_count_weight=float(kwargs.get("modify_sink_top_count_weight", 0.5)),
                mass_weight=float(kwargs.get("modify_sink_mass_weight", 0.5)),
                penalty_type=str(kwargs.get("modify_sink_penalty_type", "linear")),
            )
        after = mass_stats(probs_after.float())

    ctx = STATE["context"]
    rec = {
        "variant": STATE["variant"],
        "attn_type": ctx.get("attn_type", inferred_type),
        "step_idx": ctx.get("step_idx"),
        "block_idx": ctx.get("block_idx"),
        "phase": "before",
        "q_tokens": int(q.shape[0]),
        "k_tokens": int(k.shape[0]),
        "sample_q": int(q_s.shape[1]),
        "sample_k": int(k_s.shape[1]),
        **before,
    }
    STATE["records"].append(rec)
    if after is not None:
        rec_after = dict(rec)
        rec_after.update({"phase": "after", **after})
        STATE["records"].append(rec_after)


def install_probe():
    original_attention = sparse_attn_modules.sparse_scaled_dot_product_attention
    original_forward = sparse_attn_modules.SparseMultiHeadAttention.forward

    def attention_wrapper(*args, **kwargs):
        if STATE["enabled"]:
            record_attention(args, kwargs)
        return original_attention(*args, **kwargs)

    def forward_wrapper(self, x, context=None, step_idx=0, block_idx=0, cache_idx=0, **kwargs):
        prev = STATE["context"]
        if STATE["enabled"]:
            STATE["context"] = {
                "attn_type": self._type,
                "step_idx": int(step_idx),
                "block_idx": int(block_idx),
                "cache_idx": int(cache_idx),
            }
        try:
            return original_forward(self, x, context=context, step_idx=step_idx, block_idx=block_idx, cache_idx=cache_idx, **kwargs)
        finally:
            STATE["context"] = prev

    sparse_attn_modules.sparse_scaled_dot_product_attention = attention_wrapper
    sparse_full_attn.sparse_scaled_dot_product_attention = attention_wrapper
    sparse_attn_modules.SparseMultiHeadAttention.forward = forward_wrapper


def summarize(records):
    grouped = {}
    for rec in records:
        key = (rec["variant"], rec["attn_type"], rec["phase"])
        grouped.setdefault(key, []).append(rec)
    rows = []
    for (variant, attn_type, phase), items in sorted(grouped.items()):
        row = {"variant": variant, "attn_type": attn_type, "phase": phase, "calls": len(items)}
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
            row[field] = float(np.mean([r[field] for r in items]))
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_four_view(sample, resolution=384):
    yaws = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    pitch = [math.radians(20)] * 4
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    out = render_utils.render_frames(sample, extr, intr, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False)
    return np.concatenate(out["color"], axis=1)


def main():
    out_dir = Path("outputs/diagnostics/goblin_dragon_attn_mass_render")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    install_probe()

    pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    pipeline.cuda()

    src_img = Image.open("./assets/example_morphing/typical_humanoid_goblin.png")
    tar_img = Image.open("./assets/example_morphing/typical_creature_dragon.png")

    common = {
        "morphing_num": 3,
        "src_load_cache_path": "./outputs/cache/typical_humanoid_goblin/cache",
        "tar_load_cache_path": "./outputs/cache/typical_creature_dragon/cache",
        "init_morphing_flag": True,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": False,
        "alpha": 0.5,
        "morphing_idx": 1,
        "tfsa_cache_idx": 0,
        "tfsa_alpha": 0.8,
        "sa_use": False,
        "ot_coherence_enabled": False,
        "gate_attn": True,
        "gate_mode": "post",
        "gate_qk_confidence_threshold": 1.0,
    }
    variants = {
        "fast_post_gate": {
            "modify": False,
        },
        "slow_sink_penalty": {
            "modify": True,
            "modify_mode": "sink_penalty",
            "modify_precheck": False,
            "modify_lambda_scale": 0.8,
            "modify_sink_threshold": 0.15,
        },
    }

    rendered = []
    for name, extra in variants.items():
        cache_dir = out_dir / name / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        params = {**common, **extra, "save_cache_path": str(cache_dir)}
        STATE["variant"] = name
        STATE["enabled"] = True
        seed_everything(0)
        with torch.no_grad():
            outputs = pipeline.run_morphing(
                src_img=src_img,
                tar_img=tar_img,
                morphing_params=params,
                seed=0,
                sparse_structure_sampler_params={"steps": 4},
                slat_sampler_params={"steps": 4},
                formats=["gaussian"],
            )
        STATE["enabled"] = False
        img = render_four_view(outputs["gaussian"][0], resolution=384)
        imageio.imwrite(out_dir / f"{name}_alpha0p5.png", img)
        rendered.append(img)
        print(f"[done] {name} image={out_dir / f'{name}_alpha0p5.png'}")

    comparison = np.concatenate(rendered, axis=0)
    imageio.imwrite(out_dir / "comparison_alpha0p5.png", comparison)

    summary = summarize(STATE["records"])
    write_csv(out_dir / "attn_mass_summary.csv", summary)
    write_csv(out_dir / "attn_mass_records.csv", STATE["records"])
    (out_dir / "attn_mass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in summary:
        print(
            f"[summary] {row['variant']} {row['attn_type']} {row['phase']} calls={row['calls']} "
            f"mass_max={row['key_mass_frac_max']:.4f} gini={row['key_mass_gini']:.4f} "
            f"argmax_max={row['argmax_max_count_max']:.1f} entropy={row['entropy_mean']:.4f}"
        )
    print(f"[done] comparison={out_dir / 'comparison_alpha0p5.png'}")


if __name__ == "__main__":
    main()
