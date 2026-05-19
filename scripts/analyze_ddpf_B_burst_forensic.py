import csv
import json
import re
from collections import deque
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
B_DIR = ROOT / "outputs/analysis/ddpf_quantile_lambda_tau_scan_chong_hudie_m25/B_lambda0p20_tau0p10"
OUT_DIR = ROOT / "outputs/analysis/ddpf_B_burst_forensic_chong_hudie_m25"
GRID = 16
N = GRID**3
FRAMES = [17, 18, 19, 20]


def coord_to_index(coords):
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * GRID * GRID + coords[:, 1] * GRID + coords[:, 2]


def index_to_coord(idx):
    idx = np.asarray(idx, dtype=np.int64)
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    return np.stack([x, y, z], axis=1)


def neighbors_of_index(idx):
    x = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    z = idx % GRID
    out = []
    if x > 0:
        out.append(idx - GRID * GRID)
    if x + 1 < GRID:
        out.append(idx + GRID * GRID)
    if y > 0:
        out.append(idx - GRID)
    if y + 1 < GRID:
        out.append(idx + GRID)
    if z > 0:
        out.append(idx - 1)
    if z + 1 < GRID:
        out.append(idx + 1)
    return out


def load_endpoint():
    path = B_DIR / "ss_diff_debug/ss_endpoint_occ16.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing endpoint occ file: {path}")
    data = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "occ_s": data["occ_s_16"].cpu().numpy().astype(bool),
        "occ_t": data["occ_t_16"].cpu().numpy().astype(bool),
    }


def load_frame_occ(frame):
    path = B_DIR / f"coords_morphing{frame}.pt"
    coords = torch.load(path, map_location="cpu")
    coords = torch.as_tensor(coords).cpu().numpy()
    xyz = coords[:, -3:].astype(np.int64)
    token = np.clip(xyz // 4, 0, GRID - 1)
    occ = np.zeros(N, dtype=bool)
    occ[coord_to_index(token)] = True
    return occ


def distance_to_mask(mask):
    if not mask.any():
        return np.full(N, -1, dtype=np.int32)
    dist = np.full(N, -1, dtype=np.int32)
    q = deque()
    for idx in np.flatnonzero(mask):
        dist[int(idx)] = 0
        q.append(int(idx))
    while q:
        idx = q.popleft()
        for nb in neighbors_of_index(idx):
            if dist[nb] < 0:
                dist[nb] = dist[idx] + 1
                q.append(nb)
    return dist


def components(mask):
    visited = np.zeros(N, dtype=bool)
    comps = []
    for start in np.flatnonzero(mask):
        start = int(start)
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = []
        while q:
            idx = q.popleft()
            comp.append(idx)
            for nb in neighbors_of_index(idx):
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        comps.append(comp)
    return comps


def connectivity_metrics(occ, shared_mask, birth_mask, death_mask):
    active_shared = occ & shared_mask
    active_birth = occ & birth_mask
    comps = components(active_birth)
    connected_count = 0
    isolated_count = 0
    boundary_contact = 0
    for comp in comps:
        touches = False
        contacts = 0
        for idx in comp:
            for nb in neighbors_of_index(idx):
                if active_shared[nb]:
                    touches = True
                    contacts += 1
        boundary_contact += contacts
        if touches:
            connected_count += len(comp)
        else:
            isolated_count += 1
    active_birth_count = int(active_birth.sum())
    return {
        "total_occ_count": int(occ.sum()),
        "active_birth_count": active_birth_count,
        "active_shared_count": int(active_shared.sum()),
        "active_birth_ratio": float(active_birth_count / max(int(occ.sum()), 1)),
        "shared_keep_ratio": float(active_shared.sum() / max(int(shared_mask.sum()), 1)),
        "birth_activation_ratio": float(active_birth_count / max(int(birth_mask.sum()), 1)),
        "death_remaining_ratio": float((occ & death_mask).sum() / max(int(death_mask.sum()), 1)),
        "active_birth_component_count": len(comps),
        "connected_to_active_shared_birth_count": connected_count,
        "connected_to_active_shared_ratio": float(connected_count / max(active_birth_count, 1)),
        "isolated_birth_component_count": isolated_count,
        "boundary_contact_count": boundary_contact,
    }


def analyze_new_birth(active_birth_18, active_birth_19, active_shared_19):
    new_mask = active_birth_19 & ~active_birth_18
    old_mask = active_birth_18
    dist_shared = distance_to_mask(active_shared_19)
    dist_old_birth = distance_to_mask(old_mask)
    comps = components(new_mask)
    rows = []
    connected_tokens = 0
    isolated_tokens = 0
    isolated_components = 0
    for cid, comp in enumerate(comps):
        comp_mask = np.zeros(N, dtype=bool)
        comp_mask[comp] = True
        contact_edges = 0
        touches_shared = False
        for idx in comp:
            for nb in neighbors_of_index(idx):
                if active_shared_19[nb]:
                    touches_shared = True
                    contact_edges += 1
        if touches_shared:
            connected_tokens += len(comp)
        else:
            isolated_tokens += len(comp)
            isolated_components += 1
        ds = dist_shared[comp]
        dob = dist_old_birth[comp]
        coords = index_to_coord(comp)
        rows.append(
            {
                "component_id": cid,
                "size": len(comp),
                "touches_active_shared_19": bool(touches_shared),
                "active_shared_contact_edges": int(contact_edges),
                "min_graph_distance_to_active_shared_19": int(ds[ds >= 0].min()) if np.any(ds >= 0) else None,
                "min_graph_distance_to_active_birth_18": int(dob[dob >= 0].min()) if np.any(dob >= 0) else None,
                "centroid_x": float(coords[:, 0].mean()),
                "centroid_y": float(coords[:, 1].mean()),
                "centroid_z": float(coords[:, 2].mean()),
                "tokens": " ".join(f"{x},{y},{z}" for x, y, z in coords.tolist()),
            }
        )
    count = int(new_mask.sum())
    return {
        "mask": new_mask,
        "component_rows": rows,
        "summary": {
            "new_active_birth_count": count,
            "new_active_birth_component_count": len(comps),
            "new_component_sizes": [r["size"] for r in rows],
            "new_component_touching_active_shared_count": int(sum(r["touches_active_shared_19"] for r in rows)),
            "new_component_isolated_count": int(isolated_components),
            "new_active_birth_connected_to_active_shared_count": int(connected_tokens),
            "new_active_birth_connected_to_active_shared_ratio": float(connected_tokens / max(count, 1)),
            "new_active_birth_isolated_token_count": int(isolated_tokens),
            "new_active_birth_isolated_token_ratio": float(isolated_tokens / max(count, 1)),
            "new_active_birth_isolated_component_ratio": float(isolated_components / max(len(comps), 1)),
        },
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scatter3d(path, title, masks):
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for mask, color, label, size, alpha in masks:
        pts = index_to_coord(np.flatnonzero(mask))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=size, label=label, alpha=alpha)
    ax.set_title(title)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=38)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_ss_plots(frame_occ, shared_mask, birth_mask, new_mask):
    active_shared_18 = frame_occ[18] & shared_mask
    active_birth_18 = frame_occ[18] & birth_mask
    active_shared_19 = frame_occ[19] & shared_mask
    active_birth_19 = frame_occ[19] & birth_mask
    old_birth_19 = active_birth_19 & active_birth_18

    connected_new = np.zeros(N, dtype=bool)
    isolated_new = np.zeros(N, dtype=bool)
    for comp in components(new_mask):
        touches = any(active_shared_19[nb] for idx in comp for nb in neighbors_of_index(idx))
        if touches:
            connected_new[comp] = True
        else:
            isolated_new[comp] = True

    scatter3d(
        OUT_DIR / "frame18_active_birth_vs_shared.png",
        "B frame 18 active birth vs shared",
        [
            (active_shared_18, "#8e8e8e", "active shared", 12, 0.65),
            (active_birth_18, "#1f77b4", "active birth", 24, 0.9),
        ],
    )
    scatter3d(
        OUT_DIR / "frame19_active_birth_vs_shared.png",
        "B frame 19 active birth vs shared",
        [
            (active_shared_19, "#8e8e8e", "active shared", 12, 0.65),
            (old_birth_19, "#1f77b4", "old active birth", 22, 0.9),
            (connected_new, "#2ca02c", "new birth connected", 32, 0.95),
            (isolated_new, "#d62728", "new birth isolated", 38, 0.95),
        ],
    )
    scatter3d(
        OUT_DIR / "frame19_new_birth_components.png",
        "B frame 19 new birth components",
        [
            (active_shared_19, "#8e8e8e", "active shared", 10, 0.45),
            (old_birth_19, "#1f77b4", "old active birth", 16, 0.45),
            (connected_new, "#2ca02c", "new birth connected", 38, 0.95),
            (isolated_new, "#d62728", "new birth isolated", 44, 0.95),
        ],
    )


def read_video_frame(video_path, frame_number_1_based):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    # The coords files are numbered 1..24, while OpenCV seek is zero-based.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_1_based - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read video frame {frame_number_1_based}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def contact_sheet(metrics):
    video = B_DIR / "B_lambda0p20_tau0p10_fixed_view.mp4"
    tiles = []
    for frame in FRAMES:
        img = read_video_frame(video, frame).resize((384, 384))
        canvas = Image.new("RGB", (384, 462), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        row = metrics[frame]
        lines = [
            f"frame {frame}  alpha={frame / 25.0:.2f}",
            f"birth={row['birth_activation_ratio']:.3f}  conn={row['connected_to_active_shared_ratio']:.3f}",
            f"isolated_components={row['isolated_birth_component_count']}  contact={row['boundary_contact_count']}",
        ]
        y = 394
        for line in lines:
            draw.text((10, y), line, fill=(0, 0, 0))
            y += 20
        tiles.append(canvas)
    sheet = Image.new("RGB", (384 * 2, 462 * 2), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 384, (i // 2) * 462))
    sheet.save(OUT_DIR / "B_burst_frame17_20_contact_sheet.png")


def find_existing_logits():
    pats = ["*logit*.pt", "*voxel*.pt"]
    found = []
    for pat in pats:
        found.extend(B_DIR.rglob(pat))
    frame_re = re.compile(r"(17|18|19|20|0017|0018|0019|0020)")
    return sorted({str(p) for p in found if frame_re.search(p.name)})


def find_existing_slat():
    keywords = re.compile(r"slat|sparse|latent|feature|sample", re.I)
    return sorted(str(p) for p in B_DIR.rglob("*") if p.is_file() and keywords.search(p.name))


def unavailable_reports():
    logits = find_existing_logits()
    slat = find_existing_slat()
    if logits:
        status = "found_but_not_analyzed"
        note = "Candidate logits files were found but their schema is not part of this offline coords-only analysis script."
    else:
        status = "unavailable"
        note = "No voxel logits dump for frames 17-20 was found under the B experiment directory."
    (OUT_DIR / "ss_logits_threshold_report.json").write_text(
        json.dumps(
            {
                "status": status,
                "candidate_files": logits,
                "analysis_performed": False,
                "reason": note,
                "needed_minimal_dump": [
                    "voxel_logits_morphing0017.pt",
                    "voxel_logits_morphing0018.pt",
                    "voxel_logits_morphing0019.pt",
                    "voxel_logits_morphing0020.pt",
                ],
                "required_tensor": "raw 64^3 SS decoder voxel logits before thresholding; no sampling/coords changes",
                "planned_metrics_when_available": [
                    "4x4x4 block max/mean/positive voxel ratio per 16^3 token",
                    "new birth frame18 vs frame19 block max logit",
                    "near-threshold ratio abs(logit)<0.05",
                    "strong jump ratio delta>0.2",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if slat:
        sstatus = "found_but_not_analyzed"
        snote = "Candidate SLAT-like files were found but no frame-indexed SLAT feature dump with coords/features for frames 18-19 was identified."
    else:
        sstatus = "unavailable"
        snote = "No frame-indexed SLAT coords/features or final sample dump was found under the B experiment directory."
    (OUT_DIR / "slat_burst_diagnosis.json").write_text(
        json.dumps(
            {
                "status": sstatus,
                "candidate_files": slat,
                "analysis_performed": False,
                "reason": snote,
                "needed_minimal_dump": {
                    "frames": [18, 19],
                    "data": [
                        "SLAT sparse coords with mapping to 16^3 SS token",
                        "SLAT feature tensor for each sparse point",
                        "optional rendered intermediate/final sample frame id mapping",
                    ],
                },
                "planned_metrics_when_available": [
                    "SLAT point density in active shared / old birth / new birth",
                    "new birth feature norm and frame18->19 norm delta",
                    "sparse point cloud continuity in new birth region",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = load_endpoint()
    shared_mask = endpoint["occ_s"] & endpoint["occ_t"]
    birth_mask = endpoint["occ_t"] & ~endpoint["occ_s"]
    death_mask = endpoint["occ_s"] & ~endpoint["occ_t"]

    frame_occ = {frame: load_frame_occ(frame) for frame in FRAMES}
    active_birth_18 = frame_occ[18] & birth_mask
    active_birth_19 = frame_occ[19] & birth_mask
    active_shared_19 = frame_occ[19] & shared_mask

    metrics = {
        frame: connectivity_metrics(frame_occ[frame], shared_mask, birth_mask, death_mask)
        for frame in FRAMES
    }
    new_birth = analyze_new_birth(active_birth_18, active_birth_19, active_shared_19)
    write_csv(OUT_DIR / "ss_burst_components_frame18_19.csv", new_birth["component_rows"])

    anatomy = {
        "experiment": "B_lambda0p20_tau0p10",
        "settings": {"ddpf_phi_mode": "quantile", "ddpf_lambda": 0.20, "ddpf_tau": 0.10},
        "endpoint_occ_path": endpoint["path"],
        "frame_pair": {"before": 18, "after": 19},
        "mask_counts": {
            "shared_mask": int(shared_mask.sum()),
            "birth_mask": int(birth_mask.sum()),
            "death_mask": int(death_mask.sum()),
        },
        "frame_metrics": {
            str(frame): {"alpha": frame / 25.0, **metrics[frame]}
            for frame in FRAMES
        },
        "new_active_birth_frame18_19": new_birth["summary"],
        "component_csv": str(OUT_DIR / "ss_burst_components_frame18_19.csv"),
        "interpretation": {},
    }
    s = new_birth["summary"]
    boundary_like = s["new_active_birth_connected_to_active_shared_ratio"] >= 0.75
    isolated_heavy = s["new_active_birth_isolated_token_ratio"] >= 0.25 or s["new_component_isolated_count"] >= 5
    anatomy["interpretation"] = {
        "new_birth_mainly_boundary_expansion": bool(boundary_like),
        "new_birth_has_nontrivial_isolated_fragments": bool(isolated_heavy),
        "short_read": (
            "new birth is mostly boundary-attached, but isolated fragments are nontrivial"
            if boundary_like and isolated_heavy
            else "new birth is mostly boundary-attached"
            if boundary_like
            else "new birth is substantially fragmented away from active shared"
        ),
    }
    (OUT_DIR / "ss_burst_token_anatomy.json").write_text(json.dumps(anatomy, indent=2), encoding="utf-8")

    make_ss_plots(frame_occ, shared_mask, birth_mask, new_birth["mask"])
    contact_sheet(metrics)
    unavailable_reports()

    print("\nB burst forensic analysis")
    print(f"output_dir: {OUT_DIR}")
    print(
        "1. frame18->19 new birth: "
        f"{s['new_active_birth_count']} tokens, {s['new_active_birth_component_count']} components, "
        f"connected_ratio={s['new_active_birth_connected_to_active_shared_ratio']:.4f}, "
        f"isolated_token_ratio={s['new_active_birth_isolated_token_ratio']:.4f}, "
        f"isolated_components={s['new_component_isolated_count']}"
    )
    print(
        "   verdict: "
        f"{'mostly boundary expansion' if boundary_like else 'fragmented / isolated growth'}"
        f"{' with nontrivial isolated fragments' if isolated_heavy else ''}"
    )
    print(
        "2. isolated components enough to explain visual burst: "
        f"{'PARTIAL' if isolated_heavy else 'UNLIKELY'}"
    )
    print("3. SS logits threshold cliff: UNKNOWN (no frame17-20 raw voxel logits dump found)")
    print("4. evidence for SLAT / decoder / render side: INCONCLUSIVE (no frame-indexed SLAT feature dump found)")
    if isolated_heavy:
        print("5. next recommendation: A graph-smoothed phi plus C SLAT diagnosis; keep B SS-DPPF logic unchanged until logits are dumped.")
    else:
        print("5. next recommendation: C SLAT diagnosis; D keep SS-DPPF unchanged unless logits later show threshold cliff.")


if __name__ == "__main__":
    main()
