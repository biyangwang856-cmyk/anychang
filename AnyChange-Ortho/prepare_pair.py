"""Prepare two georeferenced RGB orthophotos for change detection."""

from contextlib import ExitStack
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window
from scipy.spatial import cKDTree


MAX_GRID_PIXELS = 40_000_000
BLOCK_SIZE = 512
RGB = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
RGBA = RGB + (ColorInterp.alpha,)


def validate_input(src):
    if src.driver != "GTiff":
        raise ValueError(f"{src.name}: expected a GeoTIFF")
    affine = src.transform
    if (
        not src.crs
        or not np.isfinite(tuple(affine)[:6]).all()
        or affine == Affine.identity()
        or not np.isfinite(affine.determinant)
        or affine.determinant == 0
    ):
        raise ValueError(f"{src.name}: a valid CRS and georeferencing affine are required")
    if src.count not in (3, 4) or any(dtype != "uint8" for dtype in src.dtypes):
        raise ValueError(
            f"{src.name}: expected an 8-bit RGB or RGBA GeoTIFF. "
            "Export three RGB bands with consistent color scaling across both dates, "
            "and preserve the NoData mask or alpha."
        )
    if any(color not in (expected, ColorInterp.undefined)
           for color, expected in zip(src.colorinterp[:3], RGB)):
        raise ValueError(f"{src.name}: the first three bands must be red, green, and blue")
    if src.count == 4 and src.colorinterp[3] not in (ColorInterp.alpha, ColorInterp.undefined):
        raise ValueError(f"{src.name}: the fourth band must be alpha")


def intersection(first, second):
    west = max(first[0], second[0])
    south = max(first[1], second[1])
    east = min(first[2], second[2])
    north = min(first[3], second[3])
    if not np.isfinite((west, south, east, north)).all() or east <= west or north <= south:
        raise ValueError("The two orthophotos have no common geographic extent")
    return west, south, east, north


def projected_bounds(src, crs):
    return transform_bounds(src.crs, crs, *src.bounds, densify_pts=41)


def select_crs(t1, t2):
    if t1.crs == t2.crs and t1.crs.is_projected and math.isclose(t1.crs.linear_units_factor[1], 1):
        return t1.crs
    wgs84 = CRS.from_epsg(4326)
    bounds1, bounds2 = (projected_bounds(src, wgs84) for src in (t1, t2))
    if any(bounds[0] > bounds[2] for bounds in (bounds1, bounds2)):
        raise ValueError("The geographic bounds cross the antimeridian. Supply both inputs in one projected meter CRS.")
    west, south, east, north = intersection(bounds1, bounds2)
    longitude, latitude = (west + east) / 2, (south + north) / 2
    if north - south > 8 or east - west > 12:
        raise ValueError("Common extent is too broad to choose a local metric CRS. Supply both inputs in one projected meter CRS.")
    if latitude >= 84:
        return CRS.from_epsg(3413)
    if latitude <= -80:
        return CRS.from_epsg(3031)
    zone = min(60, max(1, math.floor((longitude + 180) / 6) + 1))
    return CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def pixel_size(src, crs, center):
    cx, cy = transform(crs, src.crs, [center[0]], [center[1]])
    x, y = ~src.transform * (cx[0], cy[0])
    points = [src.transform * p for p in ((x, y), (x + 1, y), (x, y + 1))]
    xs, ys = transform(src.crs, crs, *zip(*points))
    size = max(math.hypot(xs[i] - xs[0], ys[i] - ys[0]) for i in (1, 2))
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"{src.name}: could not estimate a valid ground sample distance")
    return size


def target_grid(t1, t2, resolution):
    crs = select_crs(t1, t2)
    west, south, east, north = intersection(projected_bounds(t1, crs), projected_bounds(t2, crs))
    center = ((west + east) / 2, (south + north) / 2)
    source_sizes = [pixel_size(src, crs, center) for src in (t1, t2)]
    if resolution is None:
        resolution = max(.15, round(max(source_sizes), 9))
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be a positive finite value in meters per pixel")
    cols, rows = (east - west) / resolution, (north - south) / resolution
    if not np.isfinite((cols, rows)).all():
        raise ValueError("Target grid is too large. Increase the --resolution value.")
    width, height = math.ceil(cols - 1e-7), math.ceil(rows - 1e-7)
    if width < 1 or height < 1:
        raise ValueError("The common extent is smaller than one pixel at this resolution")
    if width * height > MAX_GRID_PIXELS:
        larger = math.sqrt((east - west) * (north - south) / MAX_GRID_PIXELS) * 1.05
        raise ValueError(
            f"Target grid is {width:,} × {height:,} pixels. "
            f"Increase --resolution to about {larger:.3g} m/pixel or crop the input pair."
        )
    return crs, from_origin(west, north, resolution, resolution), width, height, source_sizes


def tiff_options(crs, affine, width, height, count):
    opts = {
        "driver": "GTiff", "crs": crs, "transform": affine,
        "width": width, "height": height, "count": count,
        "dtype": "uint8", "tiled": True, "compress": "deflate",
        "blockxsize": 256, "blockysize": 256, "predictor": 2,
    }
    if count == 4:
        opts.update(photometric="RGB", interleave="pixel")
    return opts


def masked_source(src, path, stack):
    """Put RGB, dataset mask and explicit alpha into a single GDAL warp source."""
    masks = src.mask_flag_enums[:3]
    if src.count == 3 and all(MaskFlags.all_valid in flags for flags in masks):
        return src, True
    if src.count == 4 and src.colorinterp[3] == ColorInterp.alpha and all(
        MaskFlags.alpha in flags for flags in masks
    ):
        return src, False
    with rasterio.open(path, "w", **tiff_options(src.crs, src.transform, src.width, src.height, 4)) as dst:
        dst.colorinterp = RGBA
        for y in range(0, src.height, BLOCK_SIZE):
            for x in range(0, src.width, BLOCK_SIZE):
                win = Window(x, y, min(BLOCK_SIZE, src.width - x), min(BLOCK_SIZE, src.height - y))
                rgb = src.read((1, 2, 3), window=win)
                valid = (src.read_masks((1, 2, 3), window=win) != 0).all(axis=0)
                alpha = src.read(4, window=win) if src.count == 4 else np.full(valid.shape, 255, dtype="uint8")
                alpha[~valid] = 0
                dst.write(rgb, (1, 2, 3), window=win)
                dst.write(alpha, 4, window=win)
    return stack.enter_context(rasterio.open(path)), False


def warp_rgba(src, path, stack, crs, affine, width, height, source_size):
    normalized, add_alpha = masked_source(src, path, stack)
    resampling = Resampling.average if affine.a >= source_size * 1.25 else Resampling.bilinear
    with WarpedVRT(
        normalized, crs=crs, transform=affine, width=width, height=height,
        resampling=resampling, add_alpha=add_alpha, warp_mem_limit=128,
        src_alpha=0 if add_alpha else 4,
    ) as vrt:
        return np.ascontiguousarray(vrt.read((1, 2, 3, 4)).transpose(1, 2, 0))


def feature_matches(first, second, max_displacement):
    sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.022)
    matcher = cv2.BFMatcher()
    matches = []
    height, width = first.shape[:2]
    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            window = np.s_[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            alpha = (first[window][:, :, 3] >= 250) & (second[window][:, :, 3] >= 250)
            if np.count_nonzero(alpha) < 256:
                continue
            valid = alpha.astype("uint8") * 255
            detections = [
                sift.detectAndCompute(cv2.cvtColor(frame[window][:, :, :3], cv2.COLOR_RGB2GRAY), valid)
                for frame in (first, second)
            ]
            (ka, da), (kb, db) = detections
            if da is None or db is None or len(da) < 2 or len(db) < 2:
                continue
            reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
            for pair in matcher.knnMatch(da, db, k=2):
                if len(pair) != 2:
                    continue
                nearest, runner_up = pair
                if nearest.distance >= .74 * runner_up.distance or reverse[nearest.trainIdx] != nearest.queryIdx:
                    continue
                ax, ay = ka[nearest.queryIdx].pt
                bx, by = kb[nearest.trainIdx].pt
                if math.hypot(ax - bx, ay - by) <= max_displacement:
                    matches.append((ax + x, ay + y, bx + x, by + y))
    return np.asarray(matches, dtype="float32").reshape(-1, 4)


def quantiles(values, resolution):
    return [float(q) for q in np.quantile(values * resolution, [.5, .9, .95])]


def georeferenced(first, second, reason, details=None):
    common = (first[:, :, 3] >= 250) & (second[:, :, 3] >= 250)
    return second, common.astype("uint8") * 2, {
        **(details or {}), "applied": "geo", "reason": reason, "local_correction": False,
    }


def register_pair(first, second, resolution, alignment):
    if alignment == "geo":
        return georeferenced(first, second, "requested")
    height, width = first.shape[:2]
    if max(height, width) >= 32767:
        return georeferenced(first, second, "grid exceeds OpenCV remap dimension limit")
    max_displacement = min(24., max(4., 2.5 / resolution))
    pairs = feature_matches(first, second, max_displacement)
    details = {"match_count": len(pairs), "maximum_match_shift_px": max_displacement}
    if len(pairs) < 48:
        return georeferenced(first, second, "insufficient repeatable matches", details)
    shuffled = np.random.default_rng(20261112).permutation(len(pairs))
    held, fit = np.split(shuffled, [len(pairs) // 4])
    src, tgt = pairs[fit, :2], pairs[fit, 2:]
    model, inliers = cv2.estimateAffinePartial2D(
        src, tgt, method=cv2.RANSAC, ransacReprojThreshold=2.5,
    )
    if model is None or inliers is None or inliers.mean() < .55:
        return georeferenced(first, second, "unstable global transform", details)
    held_src, held_tgt = pairs[held, :2], pairs[held, 2:]
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")
    shifted_corners = cv2.transform(corners[:, None, :], model)[:, 0, :]
    if np.linalg.norm(shifted_corners - corners, axis=1).max() > max_displacement:
        return georeferenced(first, second, "global transform exceeds permitted shift", details)
    base_held = cv2.transform(held_src[:, None, :], model)[:, 0, :]
    grid_error = np.linalg.norm(held_src - held_tgt, axis=1)
    base_error = np.linalg.norm(base_held - held_tgt, axis=1)
    details.update(
        fit=int(len(fit)), holdout=int(len(held)), global_inlier_ratio=float(inliers.mean()),
        grid_error_m=quantiles(grid_error, resolution), affine_error_m=quantiles(base_error, resolution),
    )
    if np.median(base_error) > np.median(grid_error) + .05 or (
        np.quantile(base_error, .9) > np.quantile(grid_error, .9) * 1.03 + .05
    ):
        return georeferenced(first, second, "global transform worsens held-out alignment", details)

    # Estimate a smooth local residual from fit points, then evaluate on held-out points.
    residual = tgt - cv2.transform(src[:, None, :], model)[:, 0, :]
    reliable = np.linalg.norm(residual, axis=1) < 3
    final_error = base_error
    local = None
    if np.count_nonzero(reliable) >= 30:
        nx, ny = math.ceil(width / 64) + 1, math.ceil(height / 64) + 1
        gx, gy = np.meshgrid(np.linspace(0, width - 1, nx), np.linspace(0, height - 1, ny))
        distances, neighbors = cKDTree(src[reliable]).query(
            np.column_stack((gx.ravel(), gy.ravel())), k=30,
        )
        shifts = np.median(residual[reliable][neighbors], axis=1)
        shifts *= np.clip((300 - distances[:, -1]) / 180, 0, 1)[:, None]
        local_grid = shifts.reshape(ny, nx, 2).astype("float32")
        ix = np.clip(np.rint(held_src[:, 0]).astype(int), 0, width - 1)
        iy = np.clip(np.rint(held_src[:, 1]).astype(int), 0, height - 1)
        local_candidate = cv2.resize(local_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        local_error = np.linalg.norm(base_held + local_candidate[iy, ix] - held_tgt, axis=1)
        if (
            np.median(local_error) < np.median(base_error) - .015
            and np.quantile(local_error, .9) <= np.quantile(base_error, .9) * 1.05
        ):
            local, final_error = local_candidate, local_error
        else:
            del local_candidate
    if np.median(final_error) >= np.median(grid_error) - .08:
        details.update(final_error_m=details["grid_error_m"], local_correction=False)
        return second, quality_blocks(first, second, held_src, grid_error, resolution), {
            **details, "applied": "geo", "reason": "held-out shift correction has negligible benefit",
        }
    if np.quantile(final_error, .9) > np.quantile(grid_error, .9) * 1.05 + .05:
        return georeferenced(first, second, "proposed correction worsens high residuals", details)

    if local is None:
        aligned = cv2.warpAffine(
            second, model, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        xs, ys = np.arange(width, dtype="float32")[None, :], np.arange(height, dtype="float32")[:, None]
        for axis in (0, 1):
            local[:, :, axis] += model[axis, 0] * xs + model[axis, 1] * ys + model[axis, 2]
        aligned = cv2.remap(second, local, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    quality = quality_blocks(first, aligned, held_src, final_error, resolution)
    details.update(
        applied="global+local" if local is not None else "global",
        local_correction=local is not None, final_error_m=quantiles(final_error, resolution),
        affine=model.tolist(),
    )
    return aligned, quality, details


def quality_blocks(first, second, held_src, error, resolution):
    common = (first[:, :, 3] >= 250) & (second[:, :, 3] >= 250)
    quality = np.zeros(common.shape, dtype="uint8")
    limit_px = max(1.5, .45 / resolution)
    for y in range(0, common.shape[0], BLOCK_SIZE):
        for x in range(0, common.shape[1], BLOCK_SIZE):
            window = np.s_[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            sel = (
                (held_src[:, 0] >= x) & (held_src[:, 0] < x + BLOCK_SIZE)
                & (held_src[:, 1] >= y) & (held_src[:, 1] < y + BLOCK_SIZE)
            )
            values = error[sel]
            status = 2 if len(values) < 8 else (3 if np.quantile(values, .9) > limit_px else 1)
            quality[window][common[window]] = status
    return quality


def write_raster(path, array, crs, affine):
    rgb = array.ndim == 3
    opts = tiff_options(crs, affine, array.shape[1], array.shape[0], 4 if rgb else 1)
    if not rgb:
        opts["nodata"] = 0
    with rasterio.open(path, "w", **opts) as dst:
        if rgb:
            dst.colorinterp = RGBA
            dst.write(array.transpose(2, 0, 1))
        else:
            dst.set_band_description(1, "0=outside common coverage, 1=supported, 2=sparse, 3=high residual")
            dst.write(array, 1)


def prepare_pair(t1: Path, t2: Path, out: Path, resolution: float | None, alignment: str) -> dict:
    """Write a shared meter grid and return its metadata."""
    if alignment == "geo-only":
        alignment = "geo"
    if alignment not in ("auto", "geo"):
        raise ValueError("Alignment must be auto or geo")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    output_names = ("t1_aligned.tif", "t2_aligned.tif", "quality.tif")
    if any(Path(path).resolve() in {(out / name).resolve() for name in output_names} for path in (t1, t2)):
        raise ValueError("The output folder would overwrite an input orthophoto")
    with TemporaryDirectory(prefix=".prepare-", dir=out) as temp_dir:
        temp = Path(temp_dir)
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(p)) for p in (t1, t2)]
            for src in sources:
                validate_input(src)
            crs, affine, width, height, source_sizes = target_grid(*sources, resolution)
            records = [
                {"path": str(path), "crs": src.crs.to_string(), "shape": list(src.shape),
                 "estimated_pixel_m": size}
                for path, src, size in zip((t1, t2), sources, source_sizes)
            ]
            first, second = [
                warp_rgba(src, temp / f"source{index}.tif", stack, crs, affine, width, height, size)
                for index, (src, size) in enumerate(zip(sources, source_sizes), start=1)
            ]
        before_pixels = int(np.count_nonzero((first[:, :, 3] >= 250) & (second[:, :, 3] >= 250)))
        if not before_pixels:
            raise ValueError("The two orthophotos have no common valid pixels at this resolution")
        aligned, quality, registration = register_pair(first, second, affine.a, alignment)
        if np.count_nonzero(quality) < .95 * before_pixels:
            aligned, quality, registration = georeferenced(
                first, second, "proposed correction loses more than 5% of common coverage",
                {"rejected_correction": registration},
            )
        second = aligned
        common = quality != 0
        common_pixels = int(np.count_nonzero(common))
        for frame in (first, second):
            frame[~common] = 0
            frame[common, 3] = 255
        for name, array in zip(output_names, (first, second, quality)):
            write_raster(temp / name, array, crs, affine)
        for name in output_names:
            (temp / name).replace(out / name)
    registration["requested"] = alignment
    registration["common_pixels_before"] = before_pixels
    registration["quality_p90_threshold_m"] = max(.45, 1.5 * affine.a)
    return {
        "crs": crs.to_string(), "epsg": crs.to_epsg(), "resolution_m": affine.a,
        "transform": list(affine)[:6], "width": width, "height": height,
        "common_pixels": common_pixels, "common_area_m2": common_pixels * affine.a ** 2,
        "quality_pixels": {str(i): int(np.count_nonzero(quality == i)) for i in (1, 2, 3)},
        "registration": registration, "inputs": {"t1": records[0], "t2": records[1]},
    }
