#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import torch


COLUMN_INTERPRETATIONS = {
    "A_raw_1_4": [1, 2, 3],
    "B_raw_0_1_2": [0, 1, 2],
    "C_raw_0_2_3": [0, 2, 3],
}


def _tensor_basic_stats(x: torch.Tensor) -> Dict:
    x_cpu = torch.as_tensor(x).detach().cpu()
    unique = torch.unique(x_cpu)
    return {
        "shape": list(x_cpu.shape),
        "unique_values": unique.tolist(),
        "min": float(x_cpu.min().item()) if x_cpu.numel() else None,
        "max": float(x_cpu.max().item()) if x_cpu.numel() else None,
        "sum": float(x_cpu.sum().item()) if x_cpu.numel() else 0.0,
        "positive_count": int((x_cpu > 0).sum().item()),
    }


def _coord_min_max(coords: torch.Tensor) -> Dict:
    coords = coords.detach().cpu()
    if coords.numel() == 0:
        return {"min": None, "max": None}
    return {
        "min": coords.min(dim=0).values.tolist(),
        "max": coords.max(dim=0).values.tolist(),
    }


def _downsample_to_16(coords: torch.Tensor) -> torch.Tensor:
    return torch.div(coords.long(), 4, rounding_mode="floor").clamp(0, 15)


def _unique_token_count(coords_16: torch.Tensor) -> int:
    if coords_16.numel() == 0:
        return 0
    return int(torch.unique(coords_16.cpu(), dim=0).shape[0])


def _build_coord_to_index(ss_token_coords: torch.Tensor) -> Dict[Tuple[int, int, int], int]:
    token_coords = ss_token_coords.detach().cpu().long()
    return {tuple(coord.tolist()): idx for idx, coord in enumerate(token_coords)}


def _scatter_occ(coords_16: torch.Tensor, ss_token_coords: torch.Tensor) -> torch.Tensor:
    coord_to_index = _build_coord_to_index(ss_token_coords)
    occ = torch.zeros(ss_token_coords.shape[0], dtype=torch.bool)
    for coord in coords_16.detach().cpu().long():
        idx = coord_to_index.get(tuple(coord.tolist()))
        if idx is not None:
            occ[idx] = True
    return occ


def _raw_coord_interpretations(raw: torch.Tensor) -> Dict[str, Dict]:
    raw = torch.as_tensor(raw).detach().cpu()
    result: Dict[str, Dict] = {"shape": list(raw.shape)}
    if raw.ndim != 2:
        result["error"] = f"expected 2D raw coords, got {tuple(raw.shape)}"
        return result

    interpretations: Dict[str, Dict] = {}
    if raw.shape[-1] == 4:
        for name, cols in COLUMN_INTERPRETATIONS.items():
            coords = raw[:, cols]
            coords_16 = _downsample_to_16(coords)
            interpretations[name] = {
                "columns": cols,
                "coord_min_max": _coord_min_max(coords),
                "coords16_min_max": _coord_min_max(coords_16),
                "unique_token_count": _unique_token_count(coords_16),
            }
    elif raw.shape[-1] == 3:
        coords = raw
        coords_16 = _downsample_to_16(coords)
        interpretations["raw_0_1_2"] = {
            "columns": [0, 1, 2],
            "coord_min_max": _coord_min_max(coords),
            "coords16_min_max": _coord_min_max(coords_16),
            "unique_token_count": _unique_token_count(coords_16),
        }
    else:
        result["error"] = f"expected raw coords [N, 3] or [N, 4], got {tuple(raw.shape)}"
        return result

    result["interpretations"] = interpretations
    return result


def _choose_default_spatial_coords(raw: torch.Tensor) -> Tuple[torch.Tensor, str]:
    raw = torch.as_tensor(raw).detach().cpu()
    if raw.ndim != 2:
        raise ValueError(f"expected 2D raw coords, got {tuple(raw.shape)}")
    if raw.shape[-1] == 4:
        return raw[:, 1:4], "A_raw_1_4"
    if raw.shape[-1] == 3:
        return raw, "raw_0_1_2"
    raise ValueError(f"expected raw coords [N, 3] or [N, 4], got {tuple(raw.shape)}")


def audit(occ_pt: str, output_dir: str) -> Dict:
    data = torch.load(occ_pt, map_location="cpu")
    required = ["ss_token_coords", "occ_s_16", "occ_t_16", "src_coords_raw", "tar_coords_raw"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required keys in {occ_pt}: {missing}")

    ss_token_coords = torch.as_tensor(data["ss_token_coords"]).detach().cpu().long()
    occ_s_16 = torch.as_tensor(data["occ_s_16"]).detach().cpu()
    occ_t_16 = torch.as_tensor(data["occ_t_16"]).detach().cpu()
    src_coords_raw = torch.as_tensor(data["src_coords_raw"]).detach().cpu()
    tar_coords_raw = torch.as_tensor(data["tar_coords_raw"]).detach().cpu()

    src_spatial, src_default_interp = _choose_default_spatial_coords(src_coords_raw)
    tar_spatial, tar_default_interp = _choose_default_spatial_coords(tar_coords_raw)
    occ_s_rebuild = _scatter_occ(_downsample_to_16(src_spatial), ss_token_coords)
    occ_t_rebuild = _scatter_occ(_downsample_to_16(tar_spatial), ss_token_coords)

    occ_s_bool = occ_s_16 > 0
    occ_t_bool = occ_t_16 > 0
    diff_s = int((occ_s_rebuild != occ_s_bool).sum().item())
    diff_t = int((occ_t_rebuild != occ_t_bool).sum().item())

    report = {
        "occ_pt": occ_pt,
        "ss_token_coords": {
            "shape": list(ss_token_coords.shape),
            "min_max": _coord_min_max(ss_token_coords),
        },
        "occ_s_16": _tensor_basic_stats(occ_s_16),
        "occ_t_16": _tensor_basic_stats(occ_t_16),
        "src_coords_raw": _raw_coord_interpretations(src_coords_raw),
        "tar_coords_raw": _raw_coord_interpretations(tar_coords_raw),
        "rebuild_comparison": {
            "default_source_interpretation": src_default_interp,
            "default_target_interpretation": tar_default_interp,
            "occ_s_rebuild_positive_count": int(occ_s_rebuild.sum().item()),
            "occ_t_rebuild_positive_count": int(occ_t_rebuild.sum().item()),
            "diff_s": diff_s,
            "diff_t": diff_t,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _print_report(report: Dict) -> None:
    print("SS endpoint occ16 audit")
    print(f"  occ_pt: {report['occ_pt']}")
    print(f"  ss_token_coords.shape: {report['ss_token_coords']['shape']}")
    print(f"  occ_s_16.shape: {report['occ_s_16']['shape']}")
    print(f"  occ_t_16.shape: {report['occ_t_16']['shape']}")
    for name in ["occ_s_16", "occ_t_16"]:
        stats = report[name]
        print(
            f"  {name}: unique={stats['unique_values']} "
            f"min={stats['min']} max={stats['max']} sum={stats['sum']} "
            f"positive_count={stats['positive_count']}"
        )
    for name in ["src_coords_raw", "tar_coords_raw"]:
        raw = report[name]
        print(f"  {name}.shape: {raw['shape']}")
        for interp_name, interp in raw.get("interpretations", {}).items():
            print(
                f"    {interp_name} cols={interp['columns']} "
                f"min={interp['coord_min_max']['min']} max={interp['coord_min_max']['max']} "
                f"coords16_min={interp['coords16_min_max']['min']} "
                f"coords16_max={interp['coords16_min_max']['max']} "
                f"unique_tokens={interp['unique_token_count']}"
            )
    comp = report["rebuild_comparison"]
    print(
        "  rebuild comparison: "
        f"source_interp={comp['default_source_interpretation']} "
        f"target_interp={comp['default_target_interpretation']} "
        f"occ_s_rebuild_positive={comp['occ_s_rebuild_positive_count']} "
        f"occ_t_rebuild_positive={comp['occ_t_rebuild_positive_count']} "
        f"diff_s={comp['diff_s']} diff_t={comp['diff_t']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ss_endpoint_occ16.pt consistency against raw endpoint coords.")
    parser.add_argument("--occ_pt", required=True, help="Path to ss_endpoint_occ16.pt")
    parser.add_argument("--output_dir", required=True, help="Directory for audit_report.json")
    args = parser.parse_args()

    report = audit(args.occ_pt, args.output_dir)
    _print_report(report)
    print(f"  wrote: {os.path.join(args.output_dir, 'audit_report.json')}")


if __name__ == "__main__":
    main()
