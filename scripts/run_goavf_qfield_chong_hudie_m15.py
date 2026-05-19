import csv
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils.morphing_utils import seed_everything


GRID = 16
N = GRID ** 3


def save_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T, H, W, 3] frames, got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(max(int(fps), 1)),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def clear_run_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.name in {"goavf_qfield_debug", "ss_diff_debug"}:
            for f in child.glob("*"):
                f.unlink()


def token_index(coords16: np.ndarray) -> np.ndarray:
    coords16 = np.asarray(coords16, dtype=np.int64)
    return coords16[:, 0] * GRID * GRID + coords16[:, 1] * GRID + coords16[:, 2]


def neighbors(idx: int):
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    if x > 0:
        yield idx - GRID * GRID
    if x + 1 < GRID:
        yield idx + GRID * GRID
    if y > 0:
        yield idx - GRID
    if y + 1 < GRID:
        yield idx + GRID
    if z > 0:
        yield idx - 1
    if z + 1 < GRID:
        yield idx + 1


def load_endpoint(path: Path) -> dict:
    data = torch.load(path, map_location="cpu")
    occ_s = data["occ_s_16"].cpu().numpy().astype(bool)
    occ_t = data["occ_t_16"].cpu().numpy().astype(bool)
    return {
        "shared": occ_s & occ_t,
        "birth": occ_t & ~occ_s,
        "death": occ_s & ~occ_t,
    }


def load_frame_occ(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    coords = torch.load(path, map_location="cpu")
    if isinstance(coords, dict):
        for key in ("coords", "coords_morphing", "active_coords"):
            if key in coords:
                coords = coords[key]
                break
    coords = torch.as_tensor(coords).cpu().numpy()
    if coords.ndim != 2 or coords.shape[1] < 3:
        return None
    xyz = coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    occ = np.zeros(N, dtype=bool)
    occ[token_index(token)] = True
    return occ


def component_metrics(active_birth: np.ndarray, active_shared: np.ndarray) -> dict:
    visited = np.zeros(N, dtype=bool)
    active_birth_idx = np.flatnonzero(active_birth)
    component_count = 0
    isolated_count = 0
    connected_birth = 0
    for start in active_birth_idx:
        start = int(start)
        if visited[start]:
            continue
        component_count += 1
        q = deque([start])
        visited[start] = True
        comp = []
        touches_shared = False
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors(cur):
                if active_shared[nb]:
                    touches_shared = True
                if active_birth[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        if touches_shared:
            connected_birth += len(comp)
        else:
            isolated_count += 1
    return {
        "active_birth_component_count": int(component_count),
        "isolated_birth_components": int(isolated_count),
        "connected_to_active_shared_ratio": float(connected_birth / active_birth_idx.size) if active_birth_idx.size else 0.0,
        "hollow_ring_risk": bool(isolated_count > 0 and (connected_birth / active_birth_idx.size if active_birth_idx.size else 1.0) < 0.75),
    }


def load_debug(debug_dir: Path, morphing_idx: int) -> dict:
    files = sorted(debug_dir.glob(f"goavf_qfield_debug_morph{morphing_idx}_step*_block*.pt"))
    if not files:
        return {}
    data = torch.load(files[-1], map_location="cpu")
    out = {}
    for key in ("edge_q_diff_mean", "edge_q_diff_max", "graph_curvature_mean", "graph_curvature_max"):
        val = data.get(key)
        if torch.is_tensor(val):
            val = val.item()
        out[key] = None if val is None else float(val)
    for key in ("q_final", "alpha_local"):
        val = data.get(key)
        if torch.is_tensor(val):
            out[f"{key}_path"] = str(files[-1])
    return out


def frame_metrics(out_dir: Path, endpoint: dict, morphing_idx: int, alpha: float) -> dict:
    coords_path = out_dir / f"coords_morphing{morphing_idx}.pt"
    occ = load_frame_occ(coords_path)
    row = {
        "morphing_idx": morphing_idx,
        "alpha": alpha,
        "p": 1.0 - alpha,
        "coords_path": str(coords_path) if coords_path.exists() else None,
        "ss_decoded_occupancy_coords": str(coords_path) if coords_path.exists() else None,
    }
    if occ is not None:
        shared = endpoint["shared"]
        birth = endpoint["birth"]
        death = endpoint["death"]
        active_shared = occ & shared
        active_birth = occ & birth
        row.update(
            {
                "birth_activation": float(active_birth.sum() / max(int(birth.sum()), 1)),
                "shared_keep": float(active_shared.sum() / max(int(shared.sum()), 1)),
                "death_remaining": float((occ & death).sum() / max(int(death.sum()), 1)),
            }
        )
        row.update(component_metrics(active_birth, active_shared))
    else:
        row.update(
            {
                "birth_activation": None,
                "shared_keep": None,
                "death_remaining": None,
                "active_birth_component_count": None,
                "isolated_birth_components": None,
                "connected_to_active_shared_ratio": None,
                "hollow_ring_risk": None,
            }
        )
    row.update(load_debug(out_dir / "goavf_qfield_debug", morphing_idx))
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_variant(pipeline, src_img, tar_img, base_params: dict, endpoint: dict, out_root: Path, variant: str) -> None:
    seed = 0
    morphing_num = 15
    out_dir = out_root / variant
    clear_run_dir(out_dir)
    params = dict(base_params)
    params.update(
        {
            "save_cache_path": str(out_dir),
            "goavf_qfield_variant": variant,
        }
    )

    alpha_array = np.linspace(1.0, 0.0, morphing_num)
    fixed_frames = []
    metrics = []
    for morphing_idx in range(1, morphing_num - 1):
        params["alpha"] = float(alpha_array[morphing_idx])
        params["morphing_idx"] = morphing_idx
        params["tfsa_cache_idx"] = morphing_idx - 1
        params["tfsa_alpha"] = 0.8
        params["return_intermediate"] = True
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=params,
            seed=seed,
            formats=["gaussian"],
        )
        frame = render_utils.render_rot_video(
            outputs["gaussian"][0],
            resolution=512,
            bg_color=(1, 1, 1),
            num_frames=1,
        )["color"][0]
        fixed_frames.append(frame)
        metrics.append(frame_metrics(out_dir, endpoint, morphing_idx, params["alpha"]))

        for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
            for f in out_dir.glob(f"{prefix}{params['tfsa_cache_idx']}_*"):
                f.unlink()
        print(f"[G-OAVF {variant}] frame {morphing_idx}/{morphing_num - 2} alpha={params['alpha']:.4f}")

    for prefix in ("ss_sa_morphing", "slat_sa_morphing", "ss_ca_oc_morphing"):
        for f in out_dir.glob(f"{prefix}{morphing_num - 2}_*"):
            f.unlink()
    save_mp4(out_dir / f"{variant}_fixed_view.mp4", np.stack(fixed_frames, axis=0), fps=12)
    write_csv(out_dir / "per_frame_metrics.csv", metrics)
    (out_dir / "per_frame_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    seed = 0
    src_name = "chong"
    tar_name = "hudie"
    out_root = ROOT / "outputs/analysis/goavf_qfield_chong_hudie_m15"
    src_cache = ROOT / f"outputs/cache/{src_name}/cache"
    tar_cache = ROOT / f"outputs/cache/{tar_name}/cache"
    endpoint_occ_path = src_cache / f"ss_endpoint_occ16_to_{tar_name}.pt"
    qfield_path = ROOT / "outputs/analysis/goavf_contrast_residual_chong_hudie/contrast_q_fields.pt"
    if not qfield_path.exists():
        raise FileNotFoundError(f"Missing qfield file: {qfield_path}")

    endpoint = load_endpoint(endpoint_occ_path)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(ROOT / "TRELLIS-image-large"))
    pipeline.cuda()
    src_img = Image.open(ROOT / f"assets/example_morphing/{src_name}.png")
    tar_img = Image.open(ROOT / f"assets/example_morphing/{tar_name}.png")
    base_params = {
        "morphing_num": 15,
        "src_load_cache_path": str(src_cache),
        "tar_load_cache_path": str(tar_cache),
        "ss_endpoint_occ16_path": str(endpoint_occ_path),
        "init_morphing_flag": False,
        "ss_mca_flag": True,
        "slat_mca_flag": True,
        "ss_tfsa_flag": True,
        "slat_tfsa_flag": True,
        "oc_flag": False,
        "modify": True,
        "gate_attn": True,
        "sa_use": False,
        "modify_lambda_scale": 0.8,
        "ot_coherence_enabled": False,
        "dump_ss_diff_debug": True,
        "dump_ss_token_debug": False,
        "goavf_qfield_enable": True,
        "goavf_qfield_path": str(qfield_path),
        "goavf_qfield_debug": True,
        "goavf_qfield_debug_step_idx": 24,
        "goavf_qfield_debug_block_idx": 23,
        "scaf_enable": False,
        "mavf_enable": False,
        "enable_cmf_v0": False,
        "enable_ddpf_v0": False,
    }

    seed_everything(seed)
    for variant in ("A_goavf", "B_local_residual"):
        run_variant(pipeline, src_img, tar_img, base_params, endpoint, out_root, variant)
    print(f"[G-OAVF qfield] done: {out_root}")


if __name__ == "__main__":
    main()
