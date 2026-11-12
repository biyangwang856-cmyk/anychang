"""Detect changes across the common footprint of two georeferenced RGB GeoTIFFs."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
UPSTREAM = "190b07a2eb4c389a5dc3f72802c67f428cdfd6b7"
MIRROR = "487029d784104fb1a9523f7045226a27f8aa1c0d"
WEIGHTS = {
    "vit_h": ("sam_vit_h_4b8939.pth", "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"),
    "vit_b": ("sam_vit_b_01ec64.pth", "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("t1", type=Path, help="earlier RGB or RGBA GeoTIFF")
    p.add_argument("t2", type=Path, help="later RGB or RGBA GeoTIFF")
    p.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    p.add_argument("--resolution", type=float, help="output metres per pixel (default: auto)")
    p.add_argument("--alignment", choices=("auto", "geo"), default="auto")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--backbone", choices=WEIGHTS, default="vit_h")
    p.add_argument("--weights", type=Path, help="optional local copy of the official SAM checkpoint")
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "anychange-ortho")
    p.add_argument("--angle", type=float, default=145, help="AnyChange angle in degrees")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--points-per-batch", type=int, default=16)
    p.add_argument("--min-area", type=float, default=4, help="minimum vector patch area in square metres")
    p.add_argument("--prepare-only", action="store_true", help="export aligned imagery and quality grid")
    args = p.parse_args()
    if args.resolution is not None and (not math.isfinite(args.resolution) or args.resolution <= 0):
        p.error("--resolution must be positive")
    if not 0 < args.angle < 180 or not math.isfinite(args.min_area) or args.min_area < 0:
        p.error("--angle must be between 0 and 180 and --min-area must be nonnegative")
    if not 64 <= args.tile_size <= 1024 or not 0 < args.stride <= args.tile_size:
        p.error("use tile size 64..1024 and stride 1..tile-size")
    if not 1 <= args.points_per_side <= 128 or not 1 <= args.points_per_batch <= 128:
        p.error("SAM point settings must be in 1..128")
    for path in (args.t1, args.t2):
        if not path.is_file():
            p.error(f"input does not exist: {path}")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("--out must be a new or empty directory")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_weights(args):
    filename, expected = WEIGHTS[args.backbone]
    if args.weights:
        path = args.weights.expanduser()
        if sha256(path) != expected:
            raise ValueError(f"Checkpoint digest differs from official {filename}")
        return path
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = args.cache_dir / filename
    if path.is_file() and sha256(path) == expected:
        return path
    urls = (
        f"https://dl.fbaipublicfiles.com/segment_anything/{filename}",
        f"https://huggingface.co/ybelkada/segment-anything/resolve/{MIRROR}/checkpoints/{filename}",
    )
    errors = []
    for url in urls:
        partial = None
        try:
            print(f"Downloading {filename} from {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "AnyChange-Ortho/1.0"})
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as target:
                partial = Path(target.name)
                with urllib.request.urlopen(request, timeout=60) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                        digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("downloaded checkpoint digest does not match")
            partial.replace(path)
            return path
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not fetch {filename}. Supply a verified local copy with --weights. {' / '.join(errors)}")


def windows(length, size, stride):
    last = max(0, length - size)
    return sorted(set(range(0, last + 1, stride)) | {last})


def run_inference(args, output, checkpoint, device):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import torch

    sys.path.insert(0, str(ROOT / "vendor"))
    from torchange.models.segment_any_change.anychange import AnyChange, rle_to_mask

    started = time.perf_counter()
    model = AnyChange(args.backbone, str(checkpoint), device=device)
    model.make_mask_generator(points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
                              stability_score_thresh=.95)
    model.set_hyperparameters(change_confidence_threshold=args.angle, bitemporal_match=True, auto_threshold=False)
    no_data = -9999.
    tile = args.tile_size
    with ExitStack() as stack, torch.inference_mode():
        quality, early, late = [stack.enter_context(rasterio.open(output / f)) for f in
                                ("quality.tif", "t1_aligned.tif", "t2_aligned.tif")]
        common = quality.read(1) > 0
        height, width = common.shape
        score = np.full((height, width), no_data, dtype="float32")
        jobs = []
        for y in windows(height, tile, args.stride):
            for x in windows(width, tile, args.stride):
                roi = np.s_[y:y + tile, x:x + tile]
                if common[roi].any():
                    jobs.append((x, y))
        if not jobs:
            raise ValueError("No joint valid observations")
        for index, (x, y) in enumerate(jobs, 1):
            roi = np.s_[y:y + tile, x:x + tile]
            valid = common[roi]
            h, w = valid.shape
            frames = []
            for ds in (early, late):
                rgb = ds.read((1, 2, 3), window=Window(x, y, w, h)).transpose(1, 2, 0)
                frame = np.empty((tile, tile, 3), dtype="uint8")
                frame[:] = (124, 116, 104)
                frame[:h, :w][valid] = rgb[valid]
                frames.append(frame)
            try:
                prediction = model(*frames)
            finally:
                model.clear_cached_embedding()
            scores = prediction["scores"].cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite model score in window ({x}, {y})")
            merged = score[roi]
            merged[valid] = np.maximum(merged[valid], -1.)
            for encoded, value in zip(prediction["masks"], scores):
                patch = rle_to_mask(encoded)[:h, :w] & valid
                merged[patch] = np.maximum(merged[patch], np.clip(value, -1., 1.))
            if index == 1 or index % 25 == 0 or index == len(jobs):
                print(f"AnyChange {index}/{len(jobs)} windows ({time.perf_counter()-started:.1f}s)", flush=True)
        if not np.array_equal(score != no_data, common):
            raise RuntimeError("Change score coverage differs from joint valid observations")
        profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32",
                       transform=quality.transform, crs=quality.crs, nodata=no_data, tiled=True,
                       compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
        with rasterio.open(output / "anychange_score.tif", "w", **profile) as ds:
            ds.write(score, 1)
            ds.set_band_description(1, "Maximum retained negative cosine score; -1=no proposal")
    return dict(model="AnyChange", backbone=args.backbone, upstream_commit=UPSTREAM,
                checkpoint_sha256=WEIGHTS[args.backbone][1], device=str(device), windows=len(jobs),
                tile_px=tile, stride_px=args.stride, angle_deg=args.angle,
                points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
                stability=.95, seconds=round(time.perf_counter() - started, 2))


def main():
    args = parse_args()
    from prepare_pair import prepare_pair

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = prepare_pair(args.t1, args.t2, args.out, args.resolution, args.alignment)
    (args.out / "alignment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {metadata['common_pixels']:,} common pixels at {metadata['resolution_m']:g} m/px", flush=True)
    if args.prepare_only:
        return
    # Some lab images ship legacy generated ONNX protobuf modules.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import torch
    from export_results import export_results

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(default_device if args.device == "auto" else args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("--device must select CPU or CUDA")
    if device.type == "cuda" and (not torch.cuda.is_available() or
                                   device.index is not None and device.index >= torch.cuda.device_count()):
        raise ValueError("The selected CUDA device is unavailable. Verify PyTorch and the GPU driver.")
    if device.type == "cpu":
        print("CPU selected. Full ViT-H scenes may take a long time.", flush=True)
    checkpoint = get_weights(args)
    metadata["inference"] = run_inference(args, args.out, checkpoint, device)
    export_results(args.out, metadata, math.cos(math.radians(args.angle)), args.min_area)
    print(f"Results: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"AnyChange-Ortho: {exc}", file=sys.stderr)
        sys.exit(1)
