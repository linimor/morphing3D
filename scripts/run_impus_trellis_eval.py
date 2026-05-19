import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torchvision import transforms

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TRELLIS 3D models from IMPUS image sequences, render videos, and evaluate FID/PPL/PDV."
    )
    parser.add_argument("--impus-dir", default="./IMPUS")
    parser.add_argument("--model", default="./TRELLIS-image-large")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Reuse existing rendered frames and videos; do not load TRELLIS or regenerate models.",
    )
    return parser.parse_args()


def list_sequences(inputs_dir):
    return sorted([p for p in inputs_dir.iterdir() if p.is_dir()])


def list_frames(seq_dir):
    return sorted(
        [
            p
            for p in seq_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )


def save_mesh_obj(mesh, path):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def render_front(sample, resolution):
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        0.0,
        np.deg2rad(20),
        2,
        40,
    )
    return render_utils.render_frames(
        sample,
        [extrinsics],
        [intrinsics],
        {"resolution": resolution, "bg_color": (1, 1, 1)},
        verbose=False,
    )["color"][0]


def image_for_metric(path_or_array, size=299):
    if isinstance(path_or_array, np.ndarray):
        image = Image.fromarray(path_or_array).convert("RGB")
    else:
        image = Image.open(path_or_array).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


@torch.no_grad()
def inception_features(images, model, device, batch_size=16):
    feats = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size]).to(device)
        pred = model(batch)[0]
        if pred.ndim == 4:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        feats.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(features_a, features_b):
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)
    eps = 1e-6
    covmean, _ = linalg.sqrtm((sigma_a + np.eye(sigma_a.shape[0]) * eps).dot(sigma_b + np.eye(sigma_b.shape[0]) * eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


@torch.no_grad()
def perceptual_distances(rendered_frames, lpips_model, device):
    if len(rendered_frames) < 2:
        return []
    values = []
    for prev, cur in zip(rendered_frames[:-1], rendered_frames[1:]):
        a = image_for_metric(prev, size=256).mul(2).sub(1).unsqueeze(0).to(device)
        b = image_for_metric(cur, size=256).mul(2).sub(1).unsqueeze(0).to(device)
        values.append(float(lpips_model(a, b).item()))
    return values


def evaluate_sequence(input_paths, rendered_frames, inception, lpips_model, device):
    input_images = [image_for_metric(p) for p in input_paths]
    rendered_images = [image_for_metric(frame) for frame in rendered_frames]

    input_features = inception_features(input_images, inception, device)
    rendered_features = inception_features(rendered_images, inception, device)
    fid = frechet_distance(input_features, rendered_features)

    perceptual_steps = perceptual_distances(rendered_frames, lpips_model, device)
    ppl = float(np.sum(perceptual_steps)) if perceptual_steps else 0.0
    pdv = float(np.var(perceptual_steps)) if perceptual_steps else 0.0
    return {"FID": fid, "PPL": ppl, "PDV": pdv}


def generate_sequence(pipeline, seq_dir, out_dir, args):
    frame_paths = list_frames(seq_dir)
    models_dir = out_dir / "models"
    render_dir = out_dir / "rendered_frames"
    models_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered_frames = []
    for idx, frame_path in enumerate(frame_paths):
        stem = frame_path.stem
        ply_path = models_dir / f"{stem}.ply"
        obj_path = models_dir / f"{stem}.obj"
        render_path = render_dir / f"{idx:03d}_{stem}.png"

        if not args.overwrite and ply_path.exists() and obj_path.exists() and render_path.exists():
            rendered_frames.append(imageio.imread(render_path))
            continue

        image = Image.open(frame_path).convert("RGBA")
        torch.manual_seed(args.seed)
        with torch.no_grad():
            outputs = pipeline.run(image, seed=args.seed, formats=["mesh", "gaussian"])

        gaussian = outputs["gaussian"][0]
        mesh = outputs["mesh"][0]
        gaussian.save_ply(str(ply_path))
        save_mesh_obj(mesh, obj_path)

        rendered = render_front(gaussian, args.resolution)
        imageio.imwrite(render_path, rendered)
        rendered_frames.append(rendered)

    video_path = out_dir / f"{seq_dir.name}_trellis.mp4"
    if args.overwrite or not video_path.exists():
        imageio.mimsave(video_path, rendered_frames, fps=args.fps)
    return frame_paths, rendered_frames, video_path


def load_existing_sequence(seq_dir, out_dir):
    frame_paths = list_frames(seq_dir)
    render_paths = sorted((out_dir / "rendered_frames").glob("*.png"))
    if len(render_paths) != len(frame_paths):
        raise FileNotFoundError(
            f"Expected {len(frame_paths)} rendered frames under {out_dir / 'rendered_frames'}, "
            f"found {len(render_paths)}."
        )
    rendered_frames = [imageio.imread(path) for path in render_paths]
    video_path = out_dir / f"{seq_dir.name}_trellis.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing existing video: {video_path}")
    return frame_paths, rendered_frames, video_path


def write_metrics(out_root, metrics):
    json_path = out_root / "metrics.json"
    csv_path = out_root / "metrics.csv"
    payload = {
        "metric_definitions": {
            "FID": "Frechet distance between InceptionV3 features of IMPUS input frames and TRELLIS rendered frames.",
            "PPL": "Sum of LPIPS distances between consecutive TRELLIS rendered frames.",
            "PDV": "Variance of LPIPS distances between consecutive TRELLIS rendered frames.",
        },
        "results": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "video", "FID", "PPL", "PDV"])
        writer.writeheader()
        for item in metrics:
            writer.writerow(item)


def main():
    args = parse_args()
    impus_dir = Path(args.impus_dir)
    inputs_dir = impus_dir / "Inputs"
    out_root = Path(args.out_dir) if args.out_dir else impus_dir / "TRELLIS_results"
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This TRELLIS setup expects CUDA for sampling and rendering.")

    pipeline = None
    if not args.metrics_only:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
        pipeline.cuda()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    metrics = []
    for seq_dir in list_sequences(inputs_dir):
        seq_out = out_root / seq_dir.name
        seq_out.mkdir(parents=True, exist_ok=True)
        if args.metrics_only:
            input_paths, rendered_frames, video_path = load_existing_sequence(seq_dir, seq_out)
        else:
            input_paths, rendered_frames, video_path = generate_sequence(pipeline, seq_dir, seq_out, args)
        scores = evaluate_sequence(input_paths, rendered_frames, inception, lpips_model, device)
        metrics.append(
            {
                "sequence": seq_dir.name,
                "video": str(video_path.relative_to(impus_dir)),
                **scores,
            }
        )
        write_metrics(out_root, metrics)
        print(f"{seq_dir.name}: FID={scores['FID']:.6f}, PPL={scores['PPL']:.6f}, PDV={scores['PDV']:.6f}")

    write_metrics(out_root, metrics)
    print(f"Saved videos and metrics under {out_root}")


if __name__ == "__main__":
    main()
