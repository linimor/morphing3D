import argparse
import csv
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GRID = 16
N = GRID ** 3


def coords_file_sort_key(path: Path) -> int:
    m = re.search(r"coords_morphing(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 10 ** 9


def debug_file_sort_key(path: Path) -> int:
    m = re.search(r"_morph(-?\d+)_step", path.name)
    return int(m.group(1)) if m else 10 ** 9


def coord_to_index(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * GRID * GRID + coords[:, 1] * GRID + coords[:, 2]


def neighbors_of_index(idx: int):
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


def load_endpoint(method_dir: Path) -> dict:
    path = method_dir / "ss_diff_debug" / "ss_endpoint_occ16.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing endpoint occ file: {path}")
    data = torch.load(path, map_location="cpu")
    return {
        "coords": data["ss_token_coords"].cpu().numpy().astype(np.int64),
        "occ_s": data["occ_s_16"].cpu().numpy().astype(bool),
        "occ_t": data["occ_t_16"].cpu().numpy().astype(bool),
    }


def load_frame_occ(path: Path) -> np.ndarray:
    coords = torch.load(path, map_location="cpu")
    if isinstance(coords, dict):
        for key in ("coords", "coords_morphing", "active_coords"):
            if key in coords:
                coords = coords[key]
                break
    coords = torch.as_tensor(coords).cpu().numpy()
    xyz = coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    occ = np.zeros(N, dtype=bool)
    occ[coord_to_index(token)] = True
    return occ


def analyze_components(active_birth: np.ndarray, active_shared: np.ndarray) -> dict:
    active_birth_idx = np.flatnonzero(active_birth)
    visited = np.zeros(N, dtype=bool)
    component_count = 0
    isolated_component_count = 0
    connected_birth_count = 0

    for start in active_birth_idx:
        start = int(start)
        if visited[start]:
            continue
        component_count += 1
        q = deque([start])
        visited[start] = True
        component = []
        touches_shared = False
        while q:
            cur = q.popleft()
            component.append(cur)
            for nb in neighbors_of_index(cur):
                if active_shared[nb]:
                    touches_shared = True
                if active_birth[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        if touches_shared:
            connected_birth_count += len(component)
        else:
            isolated_component_count += 1

    return {
        "active_birth_component_count": int(component_count),
        "connected_to_active_shared_ratio": float(connected_birth_count / active_birth_idx.size) if active_birth_idx.size else 0.0,
        "isolated_birth_component_count": int(isolated_component_count),
    }


def analyze_coords(method: str, method_dir: Path, endpoint: dict) -> list[dict]:
    occ_s = endpoint["occ_s"]
    occ_t = endpoint["occ_t"]
    shared = occ_s & occ_t
    birth = occ_t & ~occ_s
    rows = []
    for path in sorted(method_dir.glob("coords_morphing*.pt"), key=coords_file_sort_key):
        frame_idx = coords_file_sort_key(path)
        occ = load_frame_occ(path)
        active_shared = occ & shared
        active_birth = occ & birth
        row = {
            "method": method,
            "frame_idx": frame_idx,
            "active_birth_count": int(active_birth.sum()),
            "active_shared_count": int(active_shared.sum()),
            "birth_activation_ratio": float(active_birth.sum() / max(int(birth.sum()), 1)),
            "shared_keep_ratio": float(active_shared.sum() / max(int(shared.sum()), 1)),
        }
        row.update(analyze_components(active_birth, active_shared))
        rows.append(row)
    return rows


def summarize_coords(rows: list[dict]) -> dict:
    if not rows:
        return {
            "birth_activation_max_jump": None,
            "birth_activation_mean_jump": None,
            "connected_to_active_shared_ratio": None,
            "isolated_birth_component_count": None,
            "shared_keep_mean": None,
        }
    rows = sorted(rows, key=lambda r: r["frame_idx"])
    jumps = [
        abs(after["birth_activation_ratio"] - before["birth_activation_ratio"])
        for before, after in zip(rows, rows[1:])
    ]
    return {
        "birth_activation_max_jump": float(max(jumps)) if jumps else 0.0,
        "birth_activation_mean_jump": float(np.mean(jumps)) if jumps else 0.0,
        "connected_to_active_shared_ratio": float(np.mean([r["connected_to_active_shared_ratio"] for r in rows])),
        "isolated_birth_component_count": float(np.mean([r["isolated_birth_component_count"] for r in rows])),
        "shared_keep_mean": float(np.mean([r["shared_keep_ratio"] for r in rows])),
    }


def analyze_debug(method_dir: Path, debug_dir_name: str) -> dict:
    debug_dir = method_dir / debug_dir_name
    files = sorted(debug_dir.glob("*.pt"), key=debug_file_sort_key)
    rows = []
    for path in files:
        data = torch.load(path, map_location="cpu")
        rows.append(
            {
                "frame_idx": debug_file_sort_key(path),
                "alpha": float(data.get("alpha", np.nan)),
                "p": float(data.get("p", np.nan)),
                "q_birth_std": _float_or_none(data.get("q_birth_std")),
                "edge_q_diff_p95": _float_or_none(data.get("edge_q_diff_p95")),
                "edge_q_diff_max": _float_or_none(data.get("edge_q_diff_max")),
            }
        )
    valid_q = [r["q_birth_std"] for r in rows if r["q_birth_std"] is not None]
    valid_p95 = [r["edge_q_diff_p95"] for r in rows if r["edge_q_diff_p95"] is not None]
    valid_max = [r["edge_q_diff_max"] for r in rows if r["edge_q_diff_max"] is not None]
    return {
        "debug_frame_count": len(rows),
        "q_birth_std_mean": float(np.mean(valid_q)) if valid_q else None,
        "edge_q_diff_p95_mean": float(np.mean(valid_p95)) if valid_p95 else None,
        "edge_q_diff_max": float(max(valid_max)) if valid_max else None,
        "debug_rows": rows,
    }


def _float_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.item()
    return float(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MAVF-V0 and SCAF chong->hudie m25 outputs.")
    parser.add_argument("--mavf-dir", type=Path, default=ROOT / "outputs/analysis/mavf_v0_chong_hudie_m25/MAVF_V0_default")
    parser.add_argument("--scaf-dir", type=Path, default=ROOT / "outputs/analysis/scaf_chong_hudie_m25/SCAF_default")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/scaf_vs_mavf_chong_hudie_m25")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "MAVF-V0": (args.mavf_dir, "mavf_v0_debug"),
        "SCAF": (args.scaf_dir, "scaf_debug"),
    }

    summary_rows = []
    all_coords_rows = []
    debug_rows_by_method = {}
    for method, (method_dir, debug_dir_name) in configs.items():
        endpoint = load_endpoint(method_dir)
        coords_rows = analyze_coords(method, method_dir, endpoint)
        all_coords_rows.extend(coords_rows)
        coord_summary = summarize_coords(coords_rows)
        debug_summary = analyze_debug(method_dir, debug_dir_name)
        debug_rows_by_method[method] = debug_summary.pop("debug_rows")
        summary_rows.append({"method": method, **debug_summary, **coord_summary})

    write_csv(args.out_dir / "coords_metrics.csv", all_coords_rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    for method, rows in debug_rows_by_method.items():
        write_csv(args.out_dir / f"debug_metrics_{method.replace('-', '_')}.csv", rows)

    summary = {row["method"]: row for row in summary_rows}
    verdict = {
        "q_birth_std_larger": _compare(summary, "q_birth_std_mean"),
        "edge_q_diff_p95_larger": _compare(summary, "edge_q_diff_p95_mean"),
        "birth_activation_max_jump_delta": _delta(summary, "birth_activation_max_jump"),
        "isolated_birth_component_count_delta": _delta(summary, "isolated_birth_component_count"),
        "interpretation": (
            "If SCAF increases q/edge variation but video growth is still not gradual, inspect decoder thresholding or downstream topology behavior."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps({"summary": summary_rows, "verdict": verdict}, indent=2), encoding="utf-8")

    print(f"SCAF vs MAVF summary: {args.out_dir}")
    for row in summary_rows:
        print(
            f"{row['method']}: q_std={row['q_birth_std_mean']} "
            f"edge_p95={row['edge_q_diff_p95_mean']} max_jump={row['birth_activation_max_jump']} "
            f"conn={row['connected_to_active_shared_ratio']} iso={row['isolated_birth_component_count']} "
            f"shared_keep={row['shared_keep_mean']}"
        )


def _compare(summary: dict, key: str):
    mavf = summary.get("MAVF-V0", {}).get(key)
    scaf = summary.get("SCAF", {}).get(key)
    if mavf is None or scaf is None:
        return None
    return bool(scaf > mavf)


def _delta(summary: dict, key: str):
    mavf = summary.get("MAVF-V0", {}).get(key)
    scaf = summary.get("SCAF", {}).get(key)
    if mavf is None or scaf is None:
        return None
    return float(scaf - mavf)


if __name__ == "__main__":
    main()
