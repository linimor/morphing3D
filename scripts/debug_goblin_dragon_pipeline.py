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

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from trellis.modules.attention import full_attn as dense_full_attn
from trellis.modules.attention import modules as dense_attn_modules
from trellis.modules.sparse import SparseTensor
from trellis.modules.sparse.attention import modules as sparse_attn_modules
from trellis.modules.sparse.attention import full_attn as sparse_full_attn
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import seed_everything


DEBUG_STATE = {
    "enabled": False,
    "records": [],
    "context": {},
    "sample_q": 512,
    "sample_k": 1536,
    "run_id": None,
    "alpha": None,
}


def _stats(x):
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _gini(x):
    x = x.float().flatten()
    total = x.sum()
    if x.numel() == 0 or float(total.item()) <= 0.0:
        return 0.0
    x = torch.sort(x).values
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float(((2 * idx - n - 1) * x).sum().div(n * total).item())


def _pick(length, max_items, device):
    if length <= max_items:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, max_items, device=device).round().long().unique()


def _qk_dense(args):
    if len(args) == 1:
        qkv = args[0]
        if not torch.is_tensor(qkv) or qkv.ndim != 5:
            return None
        q, k, _ = qkv.unbind(dim=2)
        return q.reshape(-1, q.shape[-2], q.shape[-1]), k.reshape(-1, k.shape[-2], k.shape[-1]), "self"
    if len(args) == 2:
        q, kv = args
        if not (torch.is_tensor(q) and torch.is_tensor(kv)):
            return None
        k = kv[:, :, 0]
        return q.reshape(-1, q.shape[-2], q.shape[-1]), k.reshape(-1, k.shape[-2], k.shape[-1]), "cross"
    if len(args) == 3:
        q, k, _ = args
        if not (torch.is_tensor(q) and torch.is_tensor(k)):
            return None
        return q.reshape(-1, q.shape[-2], q.shape[-1]), k.reshape(-1, k.shape[-2], k.shape[-1]), "cross"
    return None


def _qk_sparse(args):
    if len(args) == 1:
        qkv = args[0]
        if not isinstance(qkv, SparseTensor):
            return None
        return qkv.feats[:, 0], qkv.feats[:, 1], "self"
    if len(args) == 2:
        q, kv = args
        q_feats = q.feats if isinstance(q, SparseTensor) else q.reshape(-1, q.shape[-2], q.shape[-1])
        if isinstance(kv, SparseTensor):
            k_feats = kv.feats[:, 0]
        elif torch.is_tensor(kv):
            k_feats = kv.reshape(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])[:, 0]
        else:
            return None
        return q_feats, k_feats, "cross"
    if len(args) == 3:
        q, k, _ = args
        q_feats = q.feats if isinstance(q, SparseTensor) else q.reshape(-1, q.shape[-2], q.shape[-1])
        k_feats = k.feats if isinstance(k, SparseTensor) else k.reshape(-1, k.shape[-2], k.shape[-1])
        return q_feats, k_feats, "cross"
    return None


@torch.no_grad()
def _record_attn(stage, args):
    extracted = _qk_dense(args) if stage == "ss" else _qk_sparse(args)
    if extracted is None:
        return
    q, k, inferred_type = extracted
    if q.ndim != 3 or k.ndim != 3:
        return
    ctx = dict(DEBUG_STATE["context"])
    attn_type = ctx.get("attn_type", inferred_type)

    q_idx = _pick(q.shape[0], int(DEBUG_STATE["sample_q"]), q.device)
    k_idx = _pick(k.shape[0], int(DEBUG_STATE["sample_k"]), k.device)
    q_s = q.index_select(0, q_idx).float()
    k_s = k.index_select(0, k_idx).float()
    qh = q_s.permute(1, 0, 2).contiguous()
    kh = k_s.permute(1, 0, 2).contiguous()
    logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(qh.shape[-1])
    if stage == "ss" and bool(ctx.get("modify", False)):
        logits = dense_full_attn.modify_attn_score(
            logits,
            lambda_scale=float(ctx.get("modify_lambda_scale", 0.3)),
            max_passes=int(ctx.get("modify_max_passes", 4)),
            stop_conflict=float(ctx.get("modify_stop_conflict", 0.5)),
            temperature=float(ctx.get("modify_temperature", 1.0)),
        )
    probs = torch.softmax(logits, dim=-1)

    max_prob = probs.max(dim=-1).values
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    key_mass_frac = probs.sum(dim=1) / max(float(q_s.shape[0]), 1.0)
    argmax_key = probs.argmax(dim=-1)
    argmax_counts = []
    unique_ratios = []
    for h in range(argmax_key.shape[0]):
        counts = torch.bincount(argmax_key[h], minlength=k_s.shape[0]).float()
        argmax_counts.append(float(counts.max().item()))
        unique_ratios.append(float((counts > 0).sum().item() / max(argmax_key.shape[1], 1)))

    DEBUG_STATE["records"].append({
        "record_type": "attention",
        "run_id": DEBUG_STATE["run_id"],
        "alpha": DEBUG_STATE["alpha"],
        "stage": stage,
        "attn_type": attn_type,
        "step_idx": ctx.get("step_idx"),
        "block_idx": ctx.get("block_idx"),
        "modify": bool(ctx.get("modify", False)),
        "gate_attn": bool(ctx.get("gate_attn", False)),
        "ot_coherence_enabled": bool(ctx.get("ot_coherence_enabled", False)),
        "q_tokens": int(q.shape[0]),
        "k_tokens": int(k.shape[0]),
        "sample_q": int(q_s.shape[0]),
        "sample_k": int(k_s.shape[0]),
        "max_prob_mean": _stats(max_prob)["mean"],
        "max_prob_p95": float(torch.quantile(max_prob.flatten(), 0.95).item()),
        "entropy_mean": _stats(entropy)["mean"],
        "entropy_min": _stats(entropy)["min"],
        "key_mass_frac_mean": _stats(key_mass_frac)["mean"],
        "key_mass_frac_p95": float(torch.quantile(key_mass_frac.flatten(), 0.95).item()),
        "key_mass_frac_max": _stats(key_mass_frac)["max"],
        "key_mass_gini": _gini(key_mass_frac),
        "argmax_max_count_mean": float(np.mean(argmax_counts)),
        "argmax_max_count_max": float(np.max(argmax_counts)),
        "unique_argmax_ratio_mean": float(np.mean(unique_ratios)),
    })


def install_attention_probe():
    dense_original_attention = dense_attn_modules.scaled_dot_product_attention
    dense_original_forward = dense_attn_modules.MultiHeadAttention.forward
    sparse_original_attention = sparse_attn_modules.sparse_scaled_dot_product_attention
    sparse_original_forward = sparse_attn_modules.SparseMultiHeadAttention.forward

    def dense_attention_wrapper(*args, **kwargs):
        if DEBUG_STATE["enabled"]:
            _record_attn("ss", args)
        return dense_original_attention(*args, **kwargs)

    def dense_forward_wrapper(self, x, context=None, indices=None, step_idx=0, block_idx=0, cache_idx=0, **kwargs):
        prev = DEBUG_STATE["context"]
        if DEBUG_STATE["enabled"]:
            DEBUG_STATE["context"] = {
                "attn_type": self._type,
                "step_idx": int(step_idx),
                "block_idx": int(block_idx),
                "cache_idx": int(cache_idx),
                "modify": bool(kwargs.get("modify", False)),
                "gate_attn": bool(kwargs.get("gate_attn", False)),
                "modify_lambda_scale": float(kwargs.get("modify_lambda_scale", 0.3)),
                "modify_max_passes": int(kwargs.get("modify_max_passes", 4)),
                "modify_stop_conflict": float(kwargs.get("modify_stop_conflict", 0.5)),
                "modify_temperature": float(kwargs.get("modify_temperature", 1.0)),
                "ot_coherence_enabled": bool(kwargs.get("ot_coherence_enabled", False)),
            }
        try:
            return dense_original_forward(self, x, context=context, indices=indices, step_idx=step_idx, block_idx=block_idx, cache_idx=cache_idx, **kwargs)
        finally:
            DEBUG_STATE["context"] = prev

    def sparse_attention_wrapper(*args, **kwargs):
        if DEBUG_STATE["enabled"]:
            _record_attn("slat", args)
        return sparse_original_attention(*args, **kwargs)

    def sparse_forward_wrapper(self, x, context=None, step_idx=0, block_idx=0, cache_idx=0, **kwargs):
        prev = DEBUG_STATE["context"]
        if DEBUG_STATE["enabled"]:
            DEBUG_STATE["context"] = {
                "attn_type": self._type,
                "step_idx": int(step_idx),
                "block_idx": int(block_idx),
                "cache_idx": int(cache_idx),
                "modify": bool(kwargs.get("modify", False)),
                "gate_attn": bool(kwargs.get("gate_attn", False)),
                "modify_lambda_scale": float(kwargs.get("modify_lambda_scale", 0.3)),
                "modify_max_passes": int(kwargs.get("modify_max_passes", 4)),
                "modify_stop_conflict": float(kwargs.get("modify_stop_conflict", 0.5)),
                "modify_temperature": float(kwargs.get("modify_temperature", 1.0)),
                "ot_coherence_enabled": bool(kwargs.get("ot_coherence_enabled", False)),
            }
        try:
            return sparse_original_forward(self, x, context=context, step_idx=step_idx, block_idx=block_idx, cache_idx=cache_idx, **kwargs)
        finally:
            DEBUG_STATE["context"] = prev

    dense_attn_modules.scaled_dot_product_attention = dense_attention_wrapper
    dense_attn_modules.MultiHeadAttention.forward = dense_forward_wrapper
    sparse_attn_modules.sparse_scaled_dot_product_attention = sparse_attention_wrapper
    sparse_full_attn.sparse_scaled_dot_product_attention = sparse_attention_wrapper
    sparse_attn_modules.SparseMultiHeadAttention.forward = sparse_forward_wrapper


def coords_xyz(coords):
    if coords is None:
        return torch.empty(0, 3)
    coords = coords.detach().cpu()
    return coords[:, 1:4].float() if coords.shape[-1] == 4 else coords[:, :3].float()


def occupancy_iou(a, b, res=64):
    a = coords_xyz(a).long().clamp(0, res - 1)
    b = coords_xyz(b).long().clamp(0, res - 1)
    occ_a = torch.zeros((res, res, res), dtype=torch.bool)
    occ_b = torch.zeros((res, res, res), dtype=torch.bool)
    if a.numel():
        occ_a[a[:, 0], a[:, 1], a[:, 2]] = True
    if b.numel():
        occ_b[b[:, 0], b[:, 1], b[:, 2]] = True
    inter = (occ_a & occ_b).sum().item()
    union = (occ_a | occ_b).sum().item()
    return float(inter / union) if union else 0.0


def nearest_stats(a, b, max_points=4096):
    a = coords_xyz(a)
    b = coords_xyz(b)
    if a.numel() == 0 or b.numel() == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    if a.shape[0] > max_points:
        a = a[_pick(a.shape[0], max_points, a.device)]
    if b.shape[0] > max_points:
        b = b[_pick(b.shape[0], max_points, b.device)]
    d = torch.cdist(a, b).min(dim=1).values
    return {"mean": float(d.mean().item()), "p95": float(torch.quantile(d, 0.95).item()), "max": float(d.max().item())}


def record_voxel(run_id, alpha, coords, src_coords, tar_coords, stage_extra=None):
    src_nn = nearest_stats(coords, src_coords)
    tar_nn = nearest_stats(coords, tar_coords)
    rec = {
        "record_type": "voxel",
        "run_id": run_id,
        "alpha": alpha,
        "stage": "ss_output",
        "coords_count": int(coords.shape[0]),
        "src_count": int(src_coords.shape[0]),
        "tar_count": int(tar_coords.shape[0]),
        "iou_src": occupancy_iou(coords, src_coords),
        "iou_tar": occupancy_iou(coords, tar_coords),
        "nn_to_src_mean": src_nn["mean"],
        "nn_to_src_p95": src_nn["p95"],
        "nn_to_src_max": src_nn["max"],
        "nn_to_tar_mean": tar_nn["mean"],
        "nn_to_tar_p95": tar_nn["p95"],
        "nn_to_tar_max": tar_nn["max"],
    }
    if stage_extra:
        rec.update(stage_extra)
    DEBUG_STATE["records"].append(rec)


def install_pipeline_probe(pipeline, src_coords, tar_coords):
    original_ss = pipeline.sample_sparse_structure_morphing
    original_slat = pipeline.sample_slat_morphing

    def ss_wrapper(cond, num_samples=1, sampler_params={}, morphing_params={}):
        coords, voxels, z_s = original_ss(cond, num_samples=num_samples, sampler_params=sampler_params, morphing_params=morphing_params)
        if DEBUG_STATE["enabled"]:
            record_voxel(
                DEBUG_STATE["run_id"],
                DEBUG_STATE["alpha"],
                coords,
                src_coords,
                tar_coords,
                {
                    "ss_latent_mean": float(z_s.float().mean().item()),
                    "ss_latent_std": float(z_s.float().std(unbiased=False).item()),
                    "voxel_occupancy": int(voxels.sum().item()),
                },
            )
        return coords, voxels, z_s

    def slat_wrapper(cond, coords, sampler_params={}, morphing_params={}):
        slat = original_slat(cond, coords, sampler_params=sampler_params, morphing_params=morphing_params)
        if DEBUG_STATE["enabled"]:
            feats = slat.feats.detach().float()
            DEBUG_STATE["records"].append({
                "record_type": "slat_output",
                "run_id": DEBUG_STATE["run_id"],
                "alpha": DEBUG_STATE["alpha"],
                "stage": "slat_output",
                "tokens": int(feats.shape[0]),
                "feat_dim": int(feats.shape[1]),
                "feat_mean": float(feats.mean().item()),
                "feat_std": float(feats.std(unbiased=False).item()),
                "feat_abs_mean": float(feats.abs().mean().item()),
                "feat_norm_mean": float(feats.norm(dim=-1).mean().item()),
                "feat_norm_max": float(feats.norm(dim=-1).max().item()),
            })
        return slat

    pipeline.sample_sparse_structure_morphing = ss_wrapper
    pipeline.sample_slat_morphing = slat_wrapper


def render_multiview(sample, kind="color", resolution=384):
    yaws = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    pitch = [math.radians(20)] * len(yaws)
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    out = render_utils.render_frames(sample, extr, intr, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False)
    return np.concatenate(out[kind], axis=1)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_tfsa_cache(cache_dir, morphing_idx=None):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
    if morphing_idx is None:
        patterns = ["ss_sa_morphing*.pt", "slat_sa_morphing*.pt", "feat_coords_morphing*.pt"]
    else:
        patterns = [
            f"ss_sa_morphing{morphing_idx}_*.pt",
            f"slat_sa_morphing{morphing_idx}_*.pt",
            f"feat_coords_morphing{morphing_idx}.pt",
        ]
    for pattern in patterns:
        for path in cache_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def summarize_attention(records):
    groups = {}
    for r in records:
        if r.get("record_type") != "attention":
            continue
        key = (r["stage"], r["attn_type"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (stage, attn_type), items in sorted(groups.items()):
        row = {"stage": stage, "attn_type": attn_type, "calls": len(items)}
        for field in [
            "max_prob_mean",
            "max_prob_p95",
            "entropy_mean",
            "key_mass_frac_max",
            "key_mass_gini",
            "argmax_max_count_max",
            "unique_argmax_ratio_mean",
        ]:
            row[field] = float(np.mean([x[field] for x in items]))
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Debug Goblin -> Dragon morphing across SS and SLAT stages.")
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--src", default="./assets/example_morphing/typical_humanoid_goblin.png")
    parser.add_argument("--tar", default="./assets/example_morphing/typical_creature_dragon.png")
    parser.add_argument("--src-cache", default="./outputs/cache/typical_humanoid_goblin/cache")
    parser.add_argument("--tar-cache", default="./outputs/cache/typical_creature_dragon/cache")
    parser.add_argument("--out-dir", default="./outputs/diagnostics/goblin_dragon_pipeline_debug")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.75, 0.50, 0.25])
    parser.add_argument("--ss-steps", type=int, default=8)
    parser.add_argument("--slat-steps", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sample-q", type=int, default=512)
    parser.add_argument("--sample-k", type=int, default=1536)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--modify-lambda-scale", type=float, default=0.8)
    parser.add_argument("--modify-max-passes", type=int, default=4)
    parser.add_argument("--modify-stop-conflict", type=float, default=0.5)
    parser.add_argument("--modify-temperature", type=float, default=1.0)
    args = parser.parse_args()

    DEBUG_STATE["sample_q"] = args.sample_q
    DEBUG_STATE["sample_k"] = args.sample_k
    install_attention_probe()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_coords = torch.load(Path(args.src_cache) / "coords.pt", map_location="cpu")
    tar_coords = torch.load(Path(args.tar_cache) / "coords.pt", map_location="cpu")

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    install_pipeline_probe(pipeline, src_coords, tar_coords)

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
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
        "modify_lambda_scale": args.modify_lambda_scale,
        "modify_max_passes": args.modify_max_passes,
        "modify_stop_conflict": args.modify_stop_conflict,
        "modify_temperature": args.modify_temperature,
        "ot_coherence_enabled": True,
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
        "ot_debug": True,
        "return_intermediate": True,
    }
    meta = {
        "src": args.src,
        "tar": args.tar,
        "runs": args.runs,
        "alphas": args.alphas,
        "ss_steps": args.ss_steps,
        "slat_steps": args.slat_steps,
        "seed_start": args.seed_start,
        "sample_q": args.sample_q,
        "sample_k": args.sample_k,
        "modify_lambda_scale": args.modify_lambda_scale,
        "modify_max_passes": args.modify_max_passes,
        "modify_stop_conflict": args.modify_stop_conflict,
        "modify_temperature": args.modify_temperature,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for run in range(args.runs):
        seed = args.seed_start + run
        run_id = f"run_{run:02d}_seed_{seed}"
        run_dir = out_dir / run_id
        cache_dir = run_dir / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        normal_frames = []
        params = dict(base_params)
        params["save_cache_path"] = str(cache_dir)
        params.pop("_ot_motion_field_cache", None)
        for midx, alpha in enumerate(args.alphas, start=1):
            seed_everything(seed)
            params["alpha"] = float(alpha)
            params["morphing_idx"] = midx
            params["tfsa_cache_idx"] = midx - 1
            params["tfsa_alpha"] = 0.8
            DEBUG_STATE["enabled"] = True
            DEBUG_STATE["run_id"] = run_id
            DEBUG_STATE["alpha"] = float(alpha)
            print(f"[debug] {run_id} alpha={alpha:.2f} ss_steps={args.ss_steps} slat_steps={args.slat_steps}")
            with torch.no_grad():
                outputs = pipeline.run_morphing(
                    src_img=src_img,
                    tar_img=tar_img,
                    morphing_params=params,
                    seed=seed,
                    sparse_structure_sampler_params={"steps": args.ss_steps},
                    slat_sampler_params={"steps": args.slat_steps},
                    formats=["mesh", "gaussian"],
                )
            DEBUG_STATE["enabled"] = False
            frames.append(render_multiview(outputs["gaussian"][0], "color", args.resolution))
            normal_frames.append(render_multiview(outputs["mesh"][0], "normal", args.resolution))
            if midx > 1:
                cleanup_tfsa_cache(cache_dir, midx - 1)

        imageio.mimsave(run_dir / f"{run_id}_morph_multiview.mp4", frames, fps=2)
        imageio.mimsave(run_dir / f"{run_id}_normal_multiview.mp4", normal_frames, fps=2)
        cleanup_tfsa_cache(cache_dir)
        partial_records = [r for r in DEBUG_STATE["records"] if r.get("run_id") == run_id]
        write_csv(run_dir / f"{run_id}_records.csv", partial_records)
        (run_dir / f"{run_id}_records.json").write_text(json.dumps(partial_records, indent=2), encoding="utf-8")

    attention_summary = summarize_attention(DEBUG_STATE["records"])
    payload = {
        "meta": meta,
        "attention_summary": attention_summary,
        "records": DEBUG_STATE["records"],
        "notes": [
            "SS attention is recorded from dense transformer attention q/k samples.",
            "SLAT attention is recorded from sparse transformer attention q/k samples.",
            "Voxel metrics compare decoded SS coords against cached source/target endpoint coords.",
            "argmax_max_count close to sample_q indicates many query tokens select the same key token.",
        ],
    }
    (out_dir / "goblin_dragon_debug.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_dir / "goblin_dragon_records.csv", DEBUG_STATE["records"])
    write_csv(out_dir / "goblin_dragon_attention_summary.csv", attention_summary)
    write_csv(out_dir / "goblin_dragon_voxel_records.csv", [r for r in DEBUG_STATE["records"] if r.get("record_type") == "voxel"])
    write_csv(out_dir / "goblin_dragon_slat_output_records.csv", [r for r in DEBUG_STATE["records"] if r.get("record_type") == "slat_output"])
    print(f"[debug] wrote diagnostics to {out_dir}")
    for row in attention_summary:
        print(
            f"[summary] {row['stage']:>4} {row['attn_type']:>5} calls={row['calls']:4d} "
            f"mass_max={row['key_mass_frac_max']:.4f} gini={row['key_mass_gini']:.4f} "
            f"argmax_max={row['argmax_max_count_max']:.1f} entropy={row['entropy_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
