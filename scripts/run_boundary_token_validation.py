import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "naive")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trellis.utils.morphing_utils import compute_token_boundary_proxy, summarize_boundary_voxels


def _write_svg(out_path: Path, src_name: str, tar_name: str, src_summary: dict, tar_summary: dict, run_summary: dict):
    def _fmt(summary: dict):
        exact = summary["exact"]
        proxy = summary["proxy"]
        return (
            f"exact boundary ratio={exact['boundary_voxel_ratio']:.3f}, "
            f"boundary comps={exact['boundary_connectivity']['component_count']}, "
            f"largest={exact['boundary_connectivity']['largest_component_ratio']:.3f}; "
            f"proxy token ratio={proxy['boundary_token_ratio']:.3f}, "
            f"thr={proxy['threshold']:.3f}"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="860" viewBox="0 0 1280 860">
  <rect width="1280" height="860" fill="#f6f7f3"/>
  <rect x="40" y="40" width="1200" height="780" rx="24" fill="#ffffff" stroke="#182028" stroke-width="2"/>
  <text x="80" y="110" font-family="monospace" font-size="34" fill="#182028">Boundary Token Feasibility: {src_name} -&gt; {tar_name}</text>
  <text x="80" y="150" font-family="monospace" font-size="19" fill="#425466">Path A gives exact boundary voxels after occupancy decode. Path B gives a decoder-free proxy on the SS token lattice.</text>

  <rect x="80" y="200" width="500" height="220" rx="18" fill="#eef6ff" stroke="#3a6ea5" stroke-width="2"/>
  <text x="110" y="245" font-family="monospace" font-size="26" fill="#16324f">A. Exact boundary after decode</text>
  <text x="110" y="285" font-family="monospace" font-size="18" fill="#16324f">z_s -&gt; sparse_structure_decoder -&gt; occupancy logits -&gt; voxels</text>
  <text x="110" y="320" font-family="monospace" font-size="18" fill="#16324f">boundary rule: occupied voxel with at least one 6-neighbor empty</text>
  <text x="110" y="355" font-family="monospace" font-size="18" fill="#16324f">adds connectivity check on boundary and occupied volumes</text>
  <text x="110" y="390" font-family="monospace" font-size="18" fill="#16324f">used for diagnostics and next-frame guard tuning</text>

  <rect x="700" y="200" width="500" height="220" rx="18" fill="#fff3e8" stroke="#d97706" stroke-width="2"/>
  <text x="730" y="245" font-family="monospace" font-size="26" fill="#7c2d12">B. Proxy boundary without decode</text>
  <text x="730" y="285" font-family="monospace" font-size="18" fill="#7c2d12">z_s -&gt; patchify into SS token grid -&gt; 6-neighbor latent contrast score</text>
  <text x="730" y="320" font-family="monospace" font-size="18" fill="#7c2d12">high local contrast token == likely geometric boundary token</text>
  <text x="730" y="355" font-family="monospace" font-size="18" fill="#7c2d12">threshold = max(quantile, mean + std scale)</text>
  <text x="730" y="390" font-family="monospace" font-size="18" fill="#7c2d12">used directly inside CA/SA KNN alpha fusion as a local guard</text>

  <line x1="580" y1="310" x2="700" y2="310" stroke="#182028" stroke-width="3" marker-end="url(#arrow)"/>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#182028"/>
    </marker>
  </defs>

  <rect x="80" y="470" width="1120" height="120" rx="18" fill="#182028"/>
  <text x="110" y="515" font-family="monospace" font-size="24" fill="#f8fafc">Source summary</text>
  <text x="110" y="550" font-family="monospace" font-size="18" fill="#dbe4ee">{_fmt(src_summary)}</text>
  <text x="110" y="585" font-family="monospace" font-size="18" fill="#dbe4ee">Target summary: {_fmt(tar_summary)}</text>

  <rect x="80" y="630" width="1120" height="140" rx="18" fill="#ecfdf3" stroke="#15803d" stroke-width="2"/>
  <text x="110" y="675" font-family="monospace" font-size="24" fill="#14532d">Morphing run summary</text>
  <text x="110" y="710" font-family="monospace" font-size="18" fill="#14532d">frames with diagnostics={run_summary['frame_count']}, mean exact boundary ratio={run_summary['mean_boundary_ratio']:.3f}, mean proxy ratio={run_summary['mean_proxy_ratio']:.3f}</text>
  <text x="110" y="745" font-family="monospace" font-size="18" fill="#14532d">max boundary comps={run_summary['max_boundary_components']}, worst largest-component-ratio={run_summary['min_largest_component_ratio']:.3f}, final alpha damp={run_summary['final_boundary_alpha_damp']:.3f}</text>
  <text x="110" y="780" font-family="monospace" font-size="16" fill="#14532d">note: this environment could not finish full morphing because optional TRELLIS runtime deps such as kaolin are missing.</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def _coords_to_voxels(coords: torch.Tensor, res: int = 64) -> torch.Tensor:
    voxels = torch.zeros((1, 1, res, res, res), dtype=torch.bool)
    if coords.numel() == 0:
        return voxels
    xyz = coords[:, 1:4].long().clamp_(0, res - 1)
    voxels[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return voxels


def _load_decode_summary(cache_dir: Path):
    z_s = torch.load(cache_dir / "coords_zs.pt", map_location="cpu")
    coords = torch.load(cache_dir / "coords.pt", map_location="cpu").int()
    voxels = _coords_to_voxels(coords)
    exact = summarize_boundary_voxels(voxels)["stats"][0]
    proxy = compute_token_boundary_proxy(z_s, patch_size=1)["stats"][0]
    return {
        "exact": exact,
        "proxy": proxy,
    }


def main():
    seed = 0
    src_name = "bee"
    tar_name = "red_tree"

    out_dir = Path("./outputs/boundary_token_validation") / f"{src_name}+{tar_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_cache = Path(f"./outputs/cache/{src_name}/cache")
    tar_cache = Path(f"./outputs/cache/{tar_name}/cache")
    src_summary = _load_decode_summary(src_cache)
    tar_summary = _load_decode_summary(tar_cache)

    run_summary = {
        "frame_count": 0,
        "mean_boundary_ratio": 0.0,
        "mean_proxy_ratio": 0.0,
        "max_boundary_components": 0,
        "min_largest_component_ratio": 0.0,
        "final_boundary_alpha_damp": 0.0,
        "status": "blocked_by_missing_runtime_dependencies",
        "blocked_seed": seed,
    }

    summary = {
        "source": src_summary,
        "target": tar_summary,
        "run": run_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_svg(
        out_dir / "boundary_token_feasibility.svg",
        src_name,
        tar_name,
        src_summary,
        tar_summary,
        run_summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"DONE {out_dir}")


if __name__ == "__main__":
    main()
