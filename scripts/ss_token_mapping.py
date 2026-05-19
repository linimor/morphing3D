import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


GRID16 = 16
VOXEL_SCALE = 4
GRID64 = GRID16 * VOXEL_SCALE
TOKEN_COUNT = GRID16 ** 3


def ss_token_coords_16(device: torch.device | str | None = None) -> torch.Tensor:
    """Canonical SS token order used by trellis_image_to_3d._ss_token_coords_16."""
    if device is None:
        device = torch.device("cpu")
    coords = torch.meshgrid(
        *[torch.arange(GRID16, device=device) for _ in range(3)],
        indexing="ij",
    )
    return torch.stack(coords, dim=-1).reshape(-1, 3)


def token_id_to_coord16(token_id: int | torch.Tensor) -> torch.Tensor:
    token = torch.as_tensor(token_id, dtype=torch.long)
    if torch.any((token < 0) | (token >= TOKEN_COUNT)):
        raise ValueError(f"token_id must be in [0, {TOKEN_COUNT - 1}]")
    x = token // (GRID16 * GRID16)
    y = (token // GRID16) % GRID16
    z = token % GRID16
    return torch.stack((x, y, z), dim=-1)


def coord16_to_token_id(coord16: Iterable[int] | torch.Tensor) -> torch.Tensor:
    coord = torch.as_tensor(coord16, dtype=torch.long)
    if coord.shape[-1] != 3:
        raise ValueError(f"coord16 must have last dim 3, got {tuple(coord.shape)}")
    if torch.any((coord < 0) | (coord >= GRID16)):
        raise ValueError(f"coord16 values must be in [0, {GRID16 - 1}]")
    return coord[..., 0] * GRID16 * GRID16 + coord[..., 1] * GRID16 + coord[..., 2]


def token_id_to_decode_block64(token_id: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive 64^3 voxel block bounds [lo, hi] covered by one 16^3 SS token."""
    coord16 = token_id_to_coord16(token_id)
    lo = coord16 * VOXEL_SCALE
    hi = lo + (VOXEL_SCALE - 1)
    return lo, hi


def token_id_to_decode_center64(token_id: int | torch.Tensor) -> torch.Tensor:
    coord16 = token_id_to_coord16(token_id).to(torch.float32)
    return coord16 * VOXEL_SCALE + (VOXEL_SCALE - 1) / 2.0


def decode_coord64_to_coord16(coord64: Iterable[int] | torch.Tensor) -> torch.Tensor:
    coord = torch.as_tensor(coord64, dtype=torch.long)
    if coord.shape[-1] != 3:
        raise ValueError(f"coord64 must have last dim 3, got {tuple(coord.shape)}")
    return torch.div(coord, VOXEL_SCALE, rounding_mode="floor").clamp(0, GRID16 - 1)


def decode_coord64_to_token_id(coord64: Iterable[int] | torch.Tensor) -> torch.Tensor:
    return coord16_to_token_id(decode_coord64_to_coord16(coord64))


def q_vector_to_grid16(q: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q)
    if q.shape[-1] != TOKEN_COUNT:
        raise ValueError(f"q last dim must be {TOKEN_COUNT}, got {tuple(q.shape)}")
    return q.reshape(*q.shape[:-1], GRID16, GRID16, GRID16)


def grid16_to_q_vector(grid: torch.Tensor) -> torch.Tensor:
    grid = torch.as_tensor(grid)
    if grid.shape[-3:] != (GRID16, GRID16, GRID16):
        raise ValueError(f"grid last dims must be {(GRID16, GRID16, GRID16)}, got {tuple(grid.shape)}")
    return grid.reshape(*grid.shape[:-3], TOKEN_COUNT)


def validate_endpoint(endpoint_path: Path) -> dict:
    data = torch.load(endpoint_path, map_location="cpu")
    stored = torch.as_tensor(data["ss_token_coords"]).long()
    canonical = ss_token_coords_16()
    if stored.shape != canonical.shape:
        return {
            "ok": False,
            "reason": f"shape mismatch: stored={list(stored.shape)}, canonical={list(canonical.shape)}",
        }
    diff = stored != canonical
    mismatch_rows = torch.nonzero(diff.any(dim=1), as_tuple=False).flatten()
    report = {
        "ok": int(mismatch_rows.numel()) == 0,
        "stored_shape": list(stored.shape),
        "canonical_shape": list(canonical.shape),
        "mismatch_count": int(mismatch_rows.numel()),
        "mapping_formula": "token_id = x * 256 + y * 16 + z",
        "decode64_block_formula": "voxel64 in [coord16 * 4, coord16 * 4 + 3]",
    }
    if mismatch_rows.numel() > 0:
        sample = mismatch_rows[:10]
        report["first_mismatches"] = [
            {
                "token_id": int(i),
                "stored": stored[i].tolist(),
                "canonical": canonical[i].tolist(),
            }
            for i in sample
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable SS token-id <-> 3D coordinate mapping.")
    parser.add_argument("--token", type=int, help="Map a 1D SS token id to 16^3 coord and 64^3 decode block.")
    parser.add_argument("--coord16", nargs=3, type=int, metavar=("X", "Y", "Z"), help="Map 16^3 coord to token id.")
    parser.add_argument("--coord64", nargs=3, type=int, metavar=("X", "Y", "Z"), help="Map decoded 64^3 coord to SS token id.")
    parser.add_argument("--validate-endpoint", type=Path, help="Validate endpoint ss_token_coords against canonical order.")
    args = parser.parse_args()

    out = {}
    if args.token is not None:
        lo, hi = token_id_to_decode_block64(args.token)
        out["token"] = {
            "token_id": args.token,
            "coord16": token_id_to_coord16(args.token).tolist(),
            "decode_center64": token_id_to_decode_center64(args.token).tolist(),
            "decode_block64_lo": lo.tolist(),
            "decode_block64_hi": hi.tolist(),
        }
    if args.coord16 is not None:
        out["coord16"] = {
            "coord16": args.coord16,
            "token_id": int(coord16_to_token_id(args.coord16).item()),
        }
    if args.coord64 is not None:
        coord16 = decode_coord64_to_coord16(args.coord64)
        out["coord64"] = {
            "coord64": args.coord64,
            "coord16": coord16.tolist(),
            "token_id": int(coord16_to_token_id(coord16).item()),
        }
    if args.validate_endpoint is not None:
        out["validate_endpoint"] = validate_endpoint(args.validate_endpoint)
    if not out:
        parser.error("provide at least one of --token, --coord16, --coord64, --validate-endpoint")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
