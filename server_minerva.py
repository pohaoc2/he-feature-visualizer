#!/usr/bin/env python3
"""Minimal Minerva-style viewer backend for H&E + group overlays."""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import uvicorn
import zarr
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from utils.minerva_groups import GROUP_SPECS, build_group_flags, build_group_meta, compute_marker_thresholds

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
    df = pd.concat([df, flags], axis=1)
    groups_meta = build_group_meta(df)

    print(f"Using coordinate columns: {x_col}, {y_col}")
    print(f"Marker percentile cutoff: p{args.marker_percentile:g}")
    print("Available groups:")
    for item in groups_meta:
        print(f"  - {item['id']}: {item['count']:,} cells")

    G["df"] = df
    G["coord_cols"] = (x_col, y_col)
    G["thresholds"] = thresholds
    G["groups_meta"] = groups_meta
    G["group_ids"] = {group["id"] for group in groups_meta}

    tif = tifffile.TiffFile(args.image)
    series = tif.series[0]
    axes = series.axes.upper()
    G["axes"] = axes

    raw_store = zarr.open(series.aszarr(), mode="r")
    G["store"] = raw_store if isinstance(raw_store, zarr.Array) else raw_store["0"]

    G["img_h"] = int(df[y_col].max()) + 1 if args.image_h_from_features else series.shape[axes.index("Y")]
    G["img_w"] = int(df[x_col].max()) + 1 if args.image_w_from_features else series.shape[axes.index("X")]

    levels, max_level = dzi_level_info(G["img_w"], G["img_h"])
    G["dzi_levels"] = levels
    G["dzi_max_level"] = max_level

    print(f"Image loaded: {G['img_w']}x{G['img_h']} axes={axes} DZI max_level={max_level}")

    store = G["store"]
    cy = G["img_h"] // 2
    cx = G["img_w"] // 2
    slices = []
    for ax in axes:
        if ax == "C":
            slices.append(slice(None))
        elif ax == "Y":
            slices.append(slice(cy, cy + 256))
        elif ax == "X":
            slices.append(slice(cx, cx + 256))
        else:
            slices.append(0)

    try:
        sample = np.array(store[tuple(slices)])
        G["display_min"] = float(np.percentile(sample, 1))
        G["display_max"] = float(np.percentile(sample, 99))
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Could not sample display range: {exc}")
        G["display_min"] = 0.0
        G["display_max"] = 255.0

    try:
        target_h, target_w = 512, 512
        step_y = max(1, G["img_h"] // target_h)
        step_x = max(1, G["img_w"] // target_w)
        thumb_slices = []
        for ax in axes:
            if ax == "C":
                thumb_slices.append(slice(None))
            elif ax == "Y":
                thumb_slices.append(slice(0, G["img_h"], step_y))
            elif ax == "X":
                thumb_slices.append(slice(0, G["img_w"], step_x))
            else:
                thumb_slices.append(0)
        thumb_arr = np.array(store[tuple(thumb_slices)])
        if thumb_arr.ndim == 3 and axes.index("C") < axes.index("Y"):
            thumb_arr = np.moveaxis(thumb_arr, 0, -1)
        if thumb_arr.ndim == 2:
            thumb_arr = np.stack([thumb_arr] * 3, axis=-1)
        elif thumb_arr.shape[-1] == 1:
            thumb_arr = np.repeat(thumb_arr, 3, axis=-1)
        elif thumb_arr.shape[-1] > 3:
            thumb_arr = thumb_arr[..., :3]
        if thumb_arr.dtype != np.uint8:
            lo = G["display_min"]
            hi = G["display_max"] if G["display_max"] > lo else float(thumb_arr.max())
            if hi > lo:
                thumb_arr = (
                    (thumb_arr.astype(np.float32) - lo) / (hi - lo) * 255
                ).clip(0, 255).astype(np.uint8)
            else:
                thumb_arr = np.zeros_like(thumb_arr, dtype=np.uint8)
        thumb_img = Image.fromarray(thumb_arr)
        buffer = io.BytesIO()
        thumb_img.save(buffer, "JPEG", quality=80)
        G["thumbnail_bytes"] = buffer.getvalue()
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Thumbnail generation failed: {exc}")
        G["thumbnail_bytes"] = None


def read_region(y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Read a full-resolution RGB region (padded with black out of bounds)."""
    store = G["store"]
    axes = G["axes"]
    img_h = G["img_h"]
    img_w = G["img_w"]

    y0c = max(0, y0)
    x0c = max(0, x0)
    y1c = min(img_h, y0 + h)
    x1c = min(img_w, x0 + w)
    if y0c >= y1c or x0c >= x1c:
        return np.zeros((h, w, 3), dtype=np.uint8)

    slices = []
    for ax in axes:
        if ax == "C":
            slices.append(slice(None))
        elif ax == "Y":
            slices.append(slice(y0c, y1c))
        elif ax == "X":
            slices.append(slice(x0c, x1c))
        else:
            slices.append(0)

    arr = np.array(store[tuple(slices)])
    if arr.ndim == 3 and axes.index("C") < axes.index("Y"):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        lo = G["display_min"]
        hi = G["display_max"] if G["display_max"] > lo else float(arr.max())
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

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

    x_col, y_col = G["coord_cols"]
    mask = (
        (G["df"][x_col] >= x0)
        & (G["df"][x_col] <= x1)
        & (G["df"][y_col] >= y0)
        & (G["df"][y_col] <= y1)
        & G["df"][f"grp_{group}"]
    )
    total = int(mask.sum())
    subset = G["df"].loc[mask, [x_col, y_col]]
    sampled = False
    if len(subset) > max_cells:
        subset = subset.sample(max_cells, random_state=42)
        sampled = True

    color = GROUP_SPECS[group].color
    records = [
        {"x": float(x), "y": float(y), "r": color[0], "g": color[1], "b": color[2], "a": color[3]}
        for x, y in zip(subset[x_col].to_numpy(), subset[y_col].to_numpy())
    ]
    return JSONResponse({"cells": records, "total_in_view": total, "sampled": sampled})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal Minerva-style H&E + group viewer")
    parser.add_argument("--image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--features", required=True, help="Path to cell-level features (CSV/XLSX/Parquet)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--marker-percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff used for marker-positive grouping",
    )
    parser.add_argument(
        "--image-w-from-features",
        action="store_true",
        help="Infer image width from max feature x-coordinate instead of TIFF metadata",
    )
    parser.add_argument(
        "--image-h-from-features",
        action="store_true",
        help="Infer image height from max feature y-coordinate instead of TIFF metadata",
    )
    args = parser.parse_args()
    G["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
