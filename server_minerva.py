#!/usr/bin/env python3
"""Minimal Minerva-style viewer backend for H&E + group overlays."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from utils.minerva_groups import (
    GROUP_SPECS,
    build_group_flags,
    build_group_meta,
    compute_marker_thresholds,
    resolve_component_sources,
)

app = FastAPI(title="Minerva-style CRC Viewer")
G: dict = {}

TILE_SIZE = 254
OVERLAP = 1


def load_features(path: Path) -> pd.DataFrame:
    """Load cell-level features from CSV/XLSX/Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def resolve_coord_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Prefer transformed coordinates (Xt, Yt); fallback to raw X, Y."""
    if {"Xt", "Yt"}.issubset(df.columns):
        return "Xt", "Yt"
    if {"X", "Y"}.issubset(df.columns):
        return "X", "Y"
    raise ValueError("Features file must include Xt/Yt or X/Y columns")


def load_mx_to_he_affine(index_json: Path) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """Load inverse affine (MX->HE) from index.json warp_matrix (HE->MX)."""
    if not index_json.exists():
        return None, {"enabled": False, "reason": f"index_json not found: {index_json}"}

    with index_json.open() as handle:
        payload = json.load(handle)

    warp = payload.get("warp_matrix")
    if warp is None:
        return None, {"enabled": False, "reason": "warp_matrix missing in index_json"}

    matrix = np.asarray(warp, dtype=np.float64)
    if matrix.shape != (2, 3):
        return None, {"enabled": False, "reason": f"warp_matrix shape must be (2,3), got {matrix.shape}"}

    a = matrix[:, :2]
    t = matrix[:, 2]
    try:
        a_inv = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return None, {"enabled": False, "reason": "warp_matrix linear part is non-invertible"}

    affine = np.eye(3, dtype=np.float64)
    affine[:2, :2] = a_inv
    affine[:2, 2] = -(a_inv @ t)
    return affine, {
        "enabled": True,
        "registration_mode": payload.get("registration_mode"),
        "registration_qc_decision": payload.get("registration_qc_decision"),
        "source": str(index_json),
    }


def apply_affine_points(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    mx_to_he: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 3x3 affine to coordinate vectors."""
    ones = np.ones_like(x_vals, dtype=np.float64)
    pts = np.stack([x_vals.astype(np.float64), y_vals.astype(np.float64), ones], axis=1)
    warped = (mx_to_he @ pts.T).T
    return warped[:, 0], warped[:, 1]


def dzi_level_info(img_w: int, img_h: int, tile_size: int = TILE_SIZE) -> tuple[dict[int, dict], int]:
    """Return Deep Zoom level metadata and max level index."""
    max_dim = max(img_w, img_h)
    max_level = math.ceil(math.log2(max_dim)) if max_dim > 0 else 0
    levels: dict[int, dict] = {}
    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        width = max(1, math.ceil(img_w / scale))
        height = max(1, math.ceil(img_h / scale))
        levels[level] = {
            "w": width,
            "h": height,
            "cols": math.ceil(width / tile_size),
            "rows": math.ceil(height / tile_size),
        }
    return levels, max_level


def choose_render_level(
    series,
    render_level_arg: int | None,
    max_render_dim: int,
    min_render_level: int,
) -> int:
    """Select TIFF pyramid level for preloaded render image."""
    num_levels = len(series.levels)
    if num_levels <= 1:
        return 0

    min_idx = max(1, min(int(min_render_level), num_levels - 1))
    max_idx = num_levels - 1

    if render_level_arg is not None:
        requested = int(render_level_arg)
        return max(min_idx, min(requested, max_idx))

    for idx in range(min_idx, num_levels):
        lvl = series.levels[idx]
        lvl_axes = lvl.axes.upper()
        lvl_h = int(lvl.shape[lvl_axes.index("Y")])
        lvl_w = int(lvl.shape[lvl_axes.index("X")])
        if max(lvl_w, lvl_h) <= int(max_render_dim):
            return idx

    # If no level is small enough, still avoid level 0 and use the coarsest.
    return max_idx


@app.on_event("startup")
async def startup() -> None:
    """Load image store, features, and group metadata once at startup."""
    args = G["args"]

    features_path = Path(args.features)
    if not features_path.exists():
        raise RuntimeError(f"Features file not found: {features_path}")
    df = load_features(features_path)
    print(f"Loaded features: {len(df):,} rows")

    x_col, y_col = resolve_coord_cols(df)
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col]).copy()

    thresholds = compute_marker_thresholds(df, percentile=args.marker_percentile)
    flags = build_group_flags(df, thresholds)
    component_sources = resolve_component_sources(thresholds)
    df = pd.concat([df, flags], axis=1)

    tif = tifffile.TiffFile(args.image)
    G["tif"] = tif  # keep handle alive for zarr-backed reads
    series = tif.series[0]
    axes = series.axes.upper()
    G["axes"] = axes
    full_img_h = int(series.shape[axes.index("Y")])
    full_img_w = int(series.shape[axes.index("X")])

    # Pick a practical render level for smooth interaction with huge WSIs.
    render_level = choose_render_level(
        series=series,
        render_level_arg=args.render_level,
        max_render_dim=args.max_render_dim,
        min_render_level=args.min_render_level,
    )

    render_series = series.levels[render_level]
    render_axes = render_series.axes.upper()
    render_arr = np.array(render_series.asarray())
    if render_arr.ndim == 3 and "C" in render_axes and render_axes.index("C") < render_axes.index("Y"):
        render_arr = np.moveaxis(render_arr, 0, -1)
    if render_arr.ndim == 2:
        render_arr = np.stack([render_arr] * 3, axis=-1)
    elif render_arr.shape[-1] == 1:
        render_arr = np.repeat(render_arr, 3, axis=-1)
    elif render_arr.shape[-1] > 3:
        render_arr = render_arr[..., :3]
    if render_arr.dtype != np.uint8:
        lo = float(np.percentile(render_arr, 1))
        hi = float(np.percentile(render_arr, 99))
        if hi > lo:
            render_arr = ((render_arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        else:
            render_arr = np.zeros_like(render_arr, dtype=np.uint8)

    G["render_rgb"] = render_arr
    render_h, render_w = render_arr.shape[:2]
    G["img_h"] = render_h
    G["img_w"] = render_w
    G["full_img_h"] = full_img_h
    G["full_img_w"] = full_img_w
    G["render_level"] = render_level
    G["scale_full_to_render"] = (render_w / max(1, full_img_w), render_h / max(1, full_img_h))

    transform_meta: dict = {"enabled": False}
    view_x = df[x_col].to_numpy(dtype=np.float64, copy=True)
    view_y = df[y_col].to_numpy(dtype=np.float64, copy=True)
    if args.index_json:
        mx_to_he, transform_meta = load_mx_to_he_affine(Path(args.index_json))
        if mx_to_he is not None:
            view_x, view_y = apply_affine_points(view_x, view_y, mx_to_he)
        else:
            print(f"Coordinate transform disabled: {transform_meta.get('reason')}")

    # Map coordinates from full-resolution HE space into render level space.
    sx, sy = G["scale_full_to_render"]
    view_x = view_x * sx
    view_y = view_y * sy

    df["_x_view"] = view_x
    df["_y_view"] = view_y
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_x_view", "_y_view"]).copy()

    # Keep a small margin for slightly out-of-bounds transformed points.
    margin = float(args.coord_margin)
    in_bounds = (
        (df["_x_view"] >= -margin)
        & (df["_x_view"] <= G["img_w"] + margin)
        & (df["_y_view"] >= -margin)
        & (df["_y_view"] <= G["img_h"] + margin)
    )
    df = df.loc[in_bounds].copy()

    levels, max_level = dzi_level_info(G["img_w"], G["img_h"])
    G["dzi_levels"] = levels
    G["dzi_max_level"] = max_level

    group_points: dict[str, np.ndarray] = {}
    group_component_points: dict[str, dict[str, np.ndarray]] = {}
    group_components_meta: dict[str, list[dict]] = {}
    coords = df[["_x_view", "_y_view"]].to_numpy(dtype=np.float32)
    for group_id in GROUP_SPECS:
        mask = df[f"grp_{group_id}"].to_numpy(dtype=bool, copy=False)
        group_points[group_id] = coords[mask]

    groups_meta = build_group_meta(df, component_sources=component_sources)
    marker_masks: dict[str, np.ndarray] = {}
    marker_points: dict[str, np.ndarray] = {}
    for group in groups_meta:
        gid = group["id"]
        comps = group.get("components", [])
        group_components_meta[gid] = comps
        comp_map: dict[str, np.ndarray] = {}
        for comp in comps:
            source = comp.get("source_marker")
            if source is None:
                comp_map[comp["id"]] = np.empty((0, 2), dtype=np.float32)
                continue
            if source not in marker_masks:
                values = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=np.float32, copy=False)
                marker_masks[source] = np.isfinite(values) & (values > float(thresholds[source]))
            if source not in marker_points:
                marker_points[source] = coords[marker_masks[source]]
            comp_map[comp["id"]] = marker_points[source]
        group_component_points[gid] = comp_map
    G["df"] = df
    G["coord_cols"] = ("_x_view", "_y_view")
    G["thresholds"] = thresholds
    G["groups_meta"] = groups_meta
    G["group_ids"] = {group["id"] for group in groups_meta}
    G["group_points"] = group_points
    G["group_component_points"] = group_component_points
    G["group_components_meta"] = group_components_meta
    G["transform_meta"] = transform_meta

    print(
        f"Using coordinate columns: {x_col}, {y_col} "
        f"-> render coords: _x_view/_y_view"
    )
    if transform_meta.get("enabled"):
        print(
            "Applied MX->HE affine transform from "
            f"{transform_meta.get('source')} "
            f"(mode={transform_meta.get('registration_mode')})"
        )
    print(f"Marker percentile cutoff: p{args.marker_percentile:g}")
    print("Available groups:")
    for item in groups_meta:
        print(f"  - {item['id']}: {item['count']:,} cells")

    print(
        f"Image loaded (render level {render_level}): {G['img_w']}x{G['img_h']} "
        f"from full {full_img_w}x{full_img_h}; DZI max_level={max_level}"
    )

    try:
        # Use the coarsest native pyramid level for low-zoom tiles to avoid
        # expensive full-resolution reads at startup/home zoom.
        thumb_series = series.levels[-1] if len(series.levels) > 1 else series
        thumb_axes = thumb_series.axes.upper()
        thumb_arr = np.array(thumb_series.asarray())

        if thumb_arr.ndim == 3 and "C" in thumb_axes and thumb_axes.index("C") < thumb_axes.index("Y"):
            thumb_arr = np.moveaxis(thumb_arr, 0, -1)
        if thumb_arr.ndim == 2:
            thumb_arr = np.stack([thumb_arr] * 3, axis=-1)
        elif thumb_arr.shape[-1] == 1:
            thumb_arr = np.repeat(thumb_arr, 3, axis=-1)
        elif thumb_arr.shape[-1] > 3:
            thumb_arr = thumb_arr[..., :3]

        if thumb_arr.dtype != np.uint8:
            lo = float(np.percentile(thumb_arr, 1))
            hi = float(np.percentile(thumb_arr, 99))
            if hi > lo:
                thumb_arr = ((thumb_arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
            else:
                thumb_arr = np.zeros_like(thumb_arr, dtype=np.uint8)

        G["thumb_rgb"] = thumb_arr
        thumb_h, thumb_w = thumb_arr.shape[:2]
        thumb_downsample = max(G["img_w"] / max(1, thumb_w), G["img_h"] / max(1, thumb_h))
        thumb_dzi_level = max(
            0,
            int(round(G["dzi_max_level"] - math.log2(max(thumb_downsample, 1.0)))),
        )
        # Serve a couple extra coarse levels from thumbnail to prevent giant reads.
        G["lowres_max_level"] = min(G["dzi_max_level"], thumb_dzi_level + 2)

        thumb_img = Image.fromarray(thumb_arr)
        buffer = io.BytesIO()
        thumb_img.save(buffer, "JPEG", quality=80)
        G["thumbnail_bytes"] = buffer.getvalue()
        print(
            f"Low-res tile fallback enabled: thumb={thumb_w}x{thumb_h}, "
            f"DZI levels <= {G['lowres_max_level']} served from thumbnail."
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Thumbnail generation failed: {exc}")
        G["thumb_rgb"] = None
        G["lowres_max_level"] = -1
        G["thumbnail_bytes"] = None


def read_region(y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Read a render-level RGB region (padded with black out of bounds)."""
    render = G["render_rgb"]
    img_h = render.shape[0]
    img_w = render.shape[1]

    y0c = max(0, y0)
    x0c = max(0, x0)
    y1c = min(img_h, y0 + h)
    x1c = min(img_w, x0 + w)
    if y0c >= y1c or x0c >= x1c:
        return np.zeros((h, w, 3), dtype=np.uint8)

    arr = render[y0c:y1c, x0c:x1c]

    out_h, out_w = arr.shape[:2]
    if (out_h, out_w) != (h, w):
        out = np.zeros((h, w, 3), dtype=np.uint8)
        dy = y0c - y0
        dx = x0c - x0
        out[dy : dy + out_h, dx : dx + out_w] = arr
        arr = out
    return arr


def to_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    """Encode PIL image as JPEG bytes."""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def dzi_xml(width: int, height: int, fmt: str) -> str:
    """Return DZI descriptor XML."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="{fmt}" Overlap="{OVERLAP}" TileSize="{TILE_SIZE}">'
        f'<Size Width="{width}" Height="{height}"/></Image>'
    )


@app.get("/")
async def root() -> Response:
    """Serve the minerva-style frontend."""
    path = Path(__file__).parent / "viewer_minerva.html"
    if not path.exists():
        return Response("viewer_minerva.html not found", status_code=404)
    return FileResponse(path)


@app.get("/meta")
async def meta_route() -> JSONResponse:
    """Return frontend metadata."""
    return JSONResponse(
        {
            "img_w": G["img_w"],
            "img_h": G["img_h"],
            "n_cells": int(len(G["df"])),
            "coord_cols": list(G["coord_cols"]),
            "marker_percentile": G["args"].marker_percentile,
            "groups": G["groups_meta"],
            "thresholds": G["thresholds"],
            "transform": G.get("transform_meta", {"enabled": False}),
        }
    )


@app.get("/thumbnail.jpg")
async def thumbnail() -> Response:
    """Serve startup thumbnail fallback."""
    data = G.get("thumbnail_bytes")
    if not data:
        return Response(b"", status_code=404)
    return Response(data, media_type="image/jpeg")


@app.get("/dzi/he.dzi")
async def he_dzi() -> Response:
    """DZI descriptor for H&E image."""
    return Response(dzi_xml(G["img_w"], G["img_h"], "jpg"), media_type="application/xml")


@app.get("/dzi/he_files/{level}/{col_row}.jpg")
async def he_tile(level: int, col_row: str) -> Response:
    """Serve one DZI H&E tile."""
    try:
        col, row = map(int, col_row.split("_"))
    except ValueError as exc:
        raise HTTPException(400, "Invalid tile coordinates") from exc

    level_info = G["dzi_levels"].get(level)
    if not level_info:
        raise HTTPException(404, "Level not found")

    x0 = max(0, col * TILE_SIZE - OVERLAP)
    y0 = max(0, row * TILE_SIZE - OVERLAP)
    x1 = min(level_info["w"], (col + 1) * TILE_SIZE + OVERLAP)
    y1 = min(level_info["h"], (row + 1) * TILE_SIZE + OVERLAP)
    tile_w = x1 - x0
    tile_h = y1 - y0

    # Coarse DZI levels are served from a low-res thumbnail to avoid loading
    # huge full-resolution regions for home/overview tiles.
    thumb = G.get("thumb_rgb")
    if thumb is not None and level <= G.get("lowres_max_level", -1):
        th, tw = thumb.shape[:2]
        sx = tw / level_info["w"]
        sy = th / level_info["h"]
        tx0 = max(0, min(tw, int(math.floor(x0 * sx))))
        ty0 = max(0, min(th, int(math.floor(y0 * sy))))
        tx1 = max(tx0 + 1, min(tw, int(math.ceil(x1 * sx))))
        ty1 = max(ty0 + 1, min(th, int(math.ceil(y1 * sy))))
        region = thumb[ty0:ty1, tx0:tx1]
        tile = Image.fromarray(region).resize((tile_w, tile_h), Image.BILINEAR)
        return Response(to_jpeg(tile), media_type="image/jpeg")

    scale = G["img_w"] / level_info["w"]
    region = read_region(
        int(y0 * scale),
        int(x0 * scale),
        max(1, int(tile_h * scale)),
        max(1, int(tile_w * scale)),
    )
    tile = Image.fromarray(region).resize((tile_w, tile_h), Image.BILINEAR)
    return Response(to_jpeg(tile), media_type="image/jpeg")


@app.get("/cells")
async def cells(
    group: str = Query(..., description="One of: he, immune, tissue, cancer, proliferative, vasculature"),
    x0: float = Query(...),
    y0: float = Query(...),
    x1: float = Query(...),
    y1: float = Query(...),
    max_cells: int = Query(8000, ge=100, le=50000),
) -> JSONResponse:
    """Return cells in view for the selected overlay group."""
    if group == "he":
        return JSONResponse({"cells": [], "total_in_view": 0, "sampled": False})
    if group not in G["group_ids"]:
        raise HTTPException(400, f"Unknown group: {group}")

    components_meta = G.get("group_components_meta", {}).get(group, [])
    component_points = G.get("group_component_points", {}).get(group)
    if component_points and components_meta:
        component_slices: list[tuple[dict, np.ndarray, int]] = []
        total = 0
        for comp in components_meta:
            if not comp.get("available"):
                continue
            points = component_points.get(comp["id"])
            if points is None or points.size == 0:
                continue
            in_view = (
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
            subset = points[in_view]
            comp_total = int(subset.shape[0])
            component_slices.append((comp, subset, comp_total))
            total += comp_total

        sampled = total > max_cells
        records: list[dict] = []
        component_counts: list[dict] = []
        if total > 0:
            remaining = max_cells if sampled else total
            for idx, (comp, subset, comp_total) in enumerate(component_slices):
                if remaining <= 0:
                    break

                if sampled:
                    if idx == len(component_slices) - 1:
                        target = min(comp_total, remaining)
                    else:
                        share = int(round((comp_total / total) * max_cells))
                        target = min(comp_total, remaining, max(1, share))
                    if target <= 0:
                        continue
                    step = max(1, comp_total // target)
                    draw_pts = subset[::step][:target]
                else:
                    draw_pts = subset

                color = comp["color"]
                records.extend(
                    {
                        "x": float(px),
                        "y": float(py),
                        "r": int(color[0]),
                        "g": int(color[1]),
                        "b": int(color[2]),
                        "a": int(color[3]),
                    }
                    for px, py in draw_pts
                )
                rendered = int(draw_pts.shape[0])
                remaining -= rendered
                component_counts.append(
                    {
                        "id": comp["id"],
                        "label": comp["label"],
                        "source_marker": comp.get("source_marker"),
                        "total_in_view": comp_total,
                        "rendered": rendered,
                    }
                )

        if len(records) > max_cells:
            records = records[:max_cells]
            sampled = True

        return JSONResponse(
            {
                "cells": records,
                "total_in_view": total,
                "sampled": sampled,
                "component_counts": component_counts,
            }
        )

    points = G.get("group_points", {}).get(group)
    sampled = False
    if points is not None:
        in_view = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        subset_pts = points[in_view]
        total = int(subset_pts.shape[0])
        if total > max_cells:
            # Fast deterministic downsampling to keep response + draw times stable.
            step = max(1, total // max_cells)
            subset_pts = subset_pts[::step][:max_cells]
            sampled = True
        subset = subset_pts
    else:
        x_col, y_col = G["coord_cols"]
        mask = (
            (G["df"][x_col] >= x0)
            & (G["df"][x_col] <= x1)
            & (G["df"][y_col] >= y0)
            & (G["df"][y_col] <= y1)
            & G["df"][f"grp_{group}"]
        )
        total = int(mask.sum())
        subset_df = G["df"].loc[mask, [x_col, y_col]]
        subset = subset_df.to_numpy(dtype=np.float32, copy=False)
        if len(subset) > max_cells:
            subset = subset[:max_cells]
            sampled = True

    color = GROUP_SPECS[group].color
    records = [
        {"x": float(x), "y": float(y), "r": color[0], "g": color[1], "b": color[2], "a": color[3]}
        for x, y in subset
    ]
    return JSONResponse({"cells": records, "total_in_view": total, "sampled": sampled})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal Minerva-style H&E + group viewer")
    parser.add_argument("--image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--features", required=True, help="Path to cell-level features (CSV/XLSX/Parquet)")
    parser.add_argument(
        "--index-json",
        default=None,
        help="Optional patchify index.json containing HE->MX warp_matrix; used to map MX coords to HE view",
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--marker-percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff used for marker-positive grouping",
    )
    parser.add_argument(
        "--render-level",
        type=int,
        default=None,
        help="Explicit TIFF pyramid level to preload (0=full-res, higher=coarser)",
    )
    parser.add_argument(
        "--min-render-level",
        type=int,
        default=1,
        help="Minimum pyramid level to preload when auto-selecting (default skips full-res level 0)",
    )
    parser.add_argument(
        "--max-render-dim",
        type=int,
        default=7000,
        help="Auto-select first pyramid level with max(width,height) <= this value",
    )
    parser.add_argument(
        "--coord-margin",
        type=float,
        default=512.0,
        help="Keep transformed points within image bounds +/- margin (pixels)",
    )
    args = parser.parse_args()
    G["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
