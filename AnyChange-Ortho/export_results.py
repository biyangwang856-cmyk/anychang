"""Export change candidates, GIS rasters, and offline review pages."""

import base64
import csv
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import transform, transform_geom
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.geometry.polygon import orient
from shapely.validation import make_valid


NO_DATA = -9999.0
NO_PROPOSAL = -1.0
MASK_NO_DATA = 255
PREVIEW_MAX_SIDE = 2000
BACKGROUND = (23, 28, 35)
OVERLAY_RGB = (38, 213, 223)
FIELDS = (
    "id", "status", "area_m2", "pixel_count", "center_x_m", "center_y_m", "crs",
    "longitude", "latitude", "max_score", "registration_caution_pct",
)
OUTPUT_FILES = (
    "change_mask.tif", "change_candidates.geojson", "change_candidates.csv",
    "summary.json", "viewer.html", "report.html",
)


def check_grid(reference, dataset):
    if (dataset.shape, dataset.crs, dataset.transform) != (
        reference.shape, reference.crs, reference.transform
    ):
        raise ValueError(f"{Path(dataset.name).name} does not match the change score grid")


def check_inputs(score_ds, quality_ds, t1, t2):
    for dataset in (quality_ds, t1, t2):
        check_grid(score_ds, dataset)
    crs = score_ds.crs
    if crs is None or not crs.is_projected or not math.isclose(crs.linear_units_factor[1], 1.0):
        raise ValueError("Change scores must use a projected metre-based CRS")
    if score_ds.count != 1 or score_ds.dtypes[0] != "float32" or score_ds.nodata != NO_DATA:
        raise ValueError("anychange_score.tif must be a float32 single-band raster with NoData -9999")
    if quality_ds.count != 1 or quality_ds.dtypes[0] != "uint8":
        raise ValueError("quality.tif must be a uint8 single-band raster")
    for dataset in (t1, t2):
        if dataset.count < 3 or dataset.dtypes[:3] != ("uint8",) * 3:
            raise ValueError(f"{Path(dataset.name).name} must contain uint8 RGB bands")
    affine = score_ds.transform
    area = abs(affine.a * affine.e - affine.b * affine.d)
    if (not math.isfinite(area) or area <= 0 or
            not all(math.isfinite(value) for value in score_ds.bounds)):
        raise ValueError("Invalid change score grid transform")
    return area


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def polygonal(geometry):
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    parts = [orient(part, sign=1.0) for part in polygon_parts(geometry) if not part.is_empty]
    if not parts:
        raise ValueError("Polygonization did not produce an area geometry")
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def candidate_features(raw, score, quality, affine, crs, pixel_area, min_area):
    if not raw.any():
        return [], 0, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype("uint8"), connectivity=4)
    pixel_counts = stats[:, cv2.CC_STAT_AREA]
    retained = pixel_counts * pixel_area >= min_area
    retained[0] = False
    if not retained.any():
        return [], count - 1, 0

    selected = retained[labels]
    maxima = np.full(count, NO_PROPOSAL, dtype="float32")
    np.maximum.at(maxima, labels[selected], score[selected])
    caution_counts = np.bincount(labels[selected & (quality != 1)], minlength=count)
    records = []
    local_geometries = []
    for geometry, value in shapes(labels, mask=selected, connectivity=4, transform=affine):
        component = int(value)
        polygon = polygonal(shape(geometry))
        if not polygon.is_valid or not math.isclose(
            polygon.area, int(pixel_counts[component]) * pixel_area, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"Unexpected area or invalid geometry in component {component}")
        local_geometry = mapping(polygon)
        local_geometries.append((local_geometry, 1))
        anchor = polygon.representative_point()
        longitude, latitude = transform(crs, "EPSG:4326", [anchor.x], [anchor.y])
        wgs84 = polygonal(shape(transform_geom(crs, "EPSG:4326", local_geometry)))
        west, south, east, north = wgs84.bounds
        if not wgs84.is_valid or not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError(f"Invalid WGS84 geometry in component {component}")
        attributes = {
            "status": "model_candidate",
            "area_m2": round(int(pixel_counts[component]) * pixel_area, 6),
            "pixel_count": int(pixel_counts[component]),
            "center_x_m": round(anchor.x, 3),
            "center_y_m": round(anchor.y, 3),
            "crs": crs.to_string(),
            "longitude": round(longitude[0], 8),
            "latitude": round(latitude[0], 8),
            "max_score": round(float(maxima[component]), 6),
            "registration_caution_pct": round(100 * int(caution_counts[component]) / int(pixel_counts[component]), 2),
        }
        records.append((component, {"type": "Feature", "geometry": mapping(wgs84), "properties": attributes}))

    reconstructed = rasterize(local_geometries, out_shape=raw.shape, transform=affine, dtype="uint8")
    if not np.array_equal(reconstructed.astype(bool), selected) or len(records) != int(retained.sum()):
        raise ValueError("Vector polygons do not reproduce the retained raster components")
    records.sort(key=lambda item: (-item[1]["properties"]["pixel_count"], item[0]))
    features = []
    for index, (_, feature) in enumerate(records, 1):
        feature["properties"] = {"id": f"A{index:04d}", **feature["properties"]}
        features.append(feature)
    return features, count - 1, int(selected.sum())


def data_uri(image, fmt):
    stream = BytesIO()
    options = {"quality": 92, "optimize": True} if fmt == "JPEG" else {"optimize": True}
    image.save(stream, format=fmt, **options)
    mime = Image.MIME[fmt]
    return f"data:{mime};base64,{base64.b64encode(stream.getvalue()).decode('ascii')}"


def preview(t1, t2, common, raw):
    rows = np.flatnonzero(common.any(axis=1))
    cols = np.flatnonzero(common.any(axis=0))
    if rows.size and cols.size:
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
    else:
        top, left = 0, 0
        bottom, right = common.shape
    region = np.s_[top:bottom, left:right]
    width, height = right - left, bottom - top
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    window = Window(left, top, width, height)
    common_preview = cv2.resize(common[region].astype("uint8") * 255, size, interpolation=cv2.INTER_AREA) >= 128
    frames = []
    for dataset in (t1, t2):
        rgb = dataset.read((1, 2, 3), window=window, out_shape=(3, size[1], size[0]), resampling=Resampling.average)
        rgb[:, ~common_preview] = np.array(BACKGROUND)[:, None]
        frames.append(Image.fromarray(np.moveaxis(rgb, 0, 2)))
    raw_preview = cv2.resize(raw[region].astype("uint8") * 255, size, interpolation=cv2.INTER_AREA) >= 16
    rgba = np.zeros((size[1], size[0], 4), dtype="uint8")
    rgba[raw_preview & common_preview] = (*OVERLAY_RGB, 154)
    return frames[0], frames[1], Image.fromarray(rgba)


def write_viewer(path, t1, t2, layer):
    template = Path(__file__).with_name("viewer_template.html").read_text(encoding="utf-8")
    fields = {
        "WIDTH": str(t1.width), "HEIGHT": str(t1.height),
        "T1": data_uri(t1, "JPEG"), "T2": data_uri(t2, "JPEG"), "OVERLAY": data_uri(layer, "PNG"),
    }
    page = re.sub(r"@@([A-Z0-9_]+)@@", lambda match: fields[match[1]], template)
    path.write_text(page, encoding="utf-8")


def write_report(path, summary, t1, t2, layer, features):
    def shrink(image):
        image = image.copy()
        image.thumbnail((1100, 900), Image.Resampling.LANCZOS)
        return image

    result = Image.alpha_composite(t2.convert("RGBA"), layer).convert("RGB")
    inputs = summary.get("inputs", {})

    def image_label(key, label):
        source = inputs.get(key, {}).get("path")
        if source:
            filename = Path(str(source).replace("\\", "/")).name
            label += f" · {filename}"
        return escape(label)

    previews = [(image_label("t1", "前期正射"), shrink(t1)),
                (image_label("t2", "后期正射及候选"), shrink(result))]
    panels = "".join(f'<figure><img src="{data_uri(image, "JPEG")}" alt="{label}">'
                     f"<figcaption>{label}</figcaption></figure>" for label, image in previews)
    measures = (
        ("坐标参考系", summary["crs"]),
        ("地面像元面积", f'{summary["pixel_area_m2"]:g} m²'),
        ("共同有效面积", f'{summary["overlap_m2"]:,.2f} m²'),
        ("阈值后像元面积", f'{summary["raw_area_m2"]:,.2f} m²'),
        ("阈值后连通域", f'{summary["raw_count"]:,}'),
        ("矢量候选", f'{summary["candidate_count"]:,} 个 / {summary["candidate_area_m2"]:,.2f} m²'),
        ("模型分数阈值", f'{summary["threshold"]:.6g}（严格大于）'),
        ("矢量最小面积", f'{summary["min_area_m2"]:g} m²'),
    )
    stats = "".join(f"<tr><th scope='row'>{label}</th><td>{escape(value)}</td></tr>" for label, value in measures)
    rows = []
    for feature in features[:20]:
        props = feature["properties"]
        rows.append(
            f'<tr><td>{props["id"]}</td><td>{props["area_m2"]:,.2f}</td>'
            f'<td>{props["center_x_m"]:,.2f}</td><td>{props["center_y_m"]:,.2f}</td>'
            f'<td>{props["registration_caution_pct"]:.2f}%</td></tr>'
        )
    listing = (
        '<h2>面积较大的候选</h2><table><thead><tr><th>编号</th><th>面积 (m²)</th>'
        '<th>X (m)</th><th>Y (m)</th><th>配准警示面积占比</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        if rows else '<p>当前分数和面积条件下没有矢量候选。</p>'
    )
    page = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>双时相正射变化检测报告</title>
<style>
  body {{ margin: 0; color: #182333; background: #fff; font: 16px/1.55 system-ui, sans-serif }}
  main {{ max-width: 1000px; margin: auto; padding: 24px }}
  h1 {{ margin: 0 0 8px; font-size: 28px }}
  h2 {{ margin: 18px 0 6px; font-size: 19px }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-variant-numeric: tabular-nums }}
  th,td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #cbd5e1 }}
  th {{ background: #f1f5f9 }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; break-inside: avoid }}
  figure {{ margin: 0 }}
  img {{ width: 100%; display: block }}
  figcaption {{ text-align: center; overflow-wrap: anywhere }}
  tr {{ break-inside: avoid }}
  @page {{ size: A4; margin: 15mm }}
  @media print {{ main {{ padding: 0 }} .links {{ display: none }} body {{ font-size: 10.5pt }} }}
</style>
<main><h1>双时相正射变化检测报告</h1>
<p>AnyChange 对两幅配准正射的共同有效范围生成变化候选。图中青色为超过分数阈值的像元。</p>
<div class="pair">{panels}</div>
<h2>检测统计</h2><table><tbody>{stats}</tbody></table>
{listing}
<p>表中位置为候选面内部代表点。配准警示占比高的区域需优先比对原图。统计面积表示模型候选范围，需通过影像复核确认变化。</p>
<p class="links"><a href="viewer.html">离线卷帘</a> · <a href="change_candidates.csv">完整候选清单</a> · <a href="summary.json">运行参数与统计</a></p>
</main></html>'''
    path.write_text(page, encoding="utf-8")


def export_results(out: Path, metadata: dict, threshold: float, min_area: float):
    """Write output files and return the JSON-compatible detection summary."""
    out = Path(out)
    threshold, min_area = float(threshold), float(min_area)
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a negative-cosine score in [-1, 1]")
    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be nonnegative square metres")

    with TemporaryDirectory(prefix=".export-", dir=out) as temporary:
        staging = Path(temporary)
        with (
            rasterio.open(out / "anychange_score.tif") as score_ds,
            rasterio.open(out / "quality.tif") as quality_ds,
            rasterio.open(out / "t1_aligned.tif") as t1,
            rasterio.open(out / "t2_aligned.tif") as t2,
        ):
            pixel_area = check_inputs(score_ds, quality_ds, t1, t2)
            score = score_ds.read(1)
            quality = quality_ds.read(1)
            if quality.max() > 3:
                raise ValueError("quality.tif contains values outside 0, 1, 2, 3")
            common = quality > 0
            if not np.array_equal(score != NO_DATA, common):
                raise ValueError("Score NoData pixels do not match the common coverage grid")
            if t1.colorinterp[-1] == t2.colorinterp[-1] == ColorInterp.alpha:
                joint_alpha = (t1.read(t1.count) >= 250) & (t2.read(t2.count) >= 250)
                if not np.array_equal(joint_alpha, common):
                    raise ValueError("Aligned orthophoto alpha bands do not match common coverage")
                del joint_alpha
            valid_score = score[common]
            if not np.isfinite(valid_score).all() or (valid_score < -1).any() or (valid_score > 1).any():
                raise ValueError("Scores inside common coverage must be finite and in [-1, 1]")
            del valid_score
            common_pixels = int(common.sum())
            if "common_pixels" in metadata and metadata["common_pixels"] != common_pixels:
                raise ValueError("Recorded overlap differs from the quality raster")
            raw = common & (score > threshold) & (score != NO_PROPOSAL)
            features, raw_count, candidate_pixels = candidate_features(
                raw, score, quality, score_ds.transform, score_ds.crs, pixel_area, min_area
            )
            raw_pixels = int(raw.sum())
            summary = {
                **metadata,
                "crs": score_ds.crs.to_string(),
                "bounds": list(score_ds.bounds),
                "width": score_ds.width,
                "height": score_ds.height,
                "pixel_area_m2": pixel_area,
                "common_pixels": common_pixels,
                "overlap_m2": round(common_pixels * pixel_area, 6),
                "threshold": threshold,
                "min_area_m2": min_area,
                "raw_count": raw_count,
                "raw_pixels": raw_pixels,
                "raw_area_m2": round(raw_pixels * pixel_area, 6),
                "candidate_count": len(features),
                "candidate_pixels": candidate_pixels,
                "candidate_area_m2": round(candidate_pixels * pixel_area, 6),
                "registration_caution_m2": round(int((raw & (quality != 1)).sum()) * pixel_area, 6),
                "mask_codes": {"0": "no candidate", "1": "model candidate", "255": "outside common coverage"},
                "files": list(OUTPUT_FILES),
            }
            with rasterio.open(staging / "change_mask.tif", "w", driver="GTiff", count=1,
                               height=score_ds.height, width=score_ds.width, crs=score_ds.crs,
                               transform=score_ds.transform, dtype="uint8", nodata=MASK_NO_DATA,
                               tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=2) as dst:
                dst.write(np.where(common, raw, MASK_NO_DATA).astype("uint8"), 1)
                dst.set_band_description(1, "0 no candidate; 1 model candidate; 255 outside common coverage")
            frames = preview(t1, t2, common, raw)

        geojson = {"type": "FeatureCollection", "features": features}
        (staging / "change_candidates.geojson").write_text(
            json.dumps(geojson, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8"
        )
        with (staging / "change_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(feature["properties"] for feature in features)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        write_viewer(staging / "viewer.html", *frames)
        write_report(staging / "report.html", summary, *frames, features)
        for filename in OUTPUT_FILES:
            (staging / filename).replace(out / filename)
    return summary
