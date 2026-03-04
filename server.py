#!/usr/bin/env python3
"""server.py — FastAPI DZI tile server for CRC H&E viewer."""

import argparse, io, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn
from PIL import Image

app = FastAPI(title="CRC Viewer")
G: dict = {}

TILE_SIZE = 254
OVERLAP   = 1

# Pre-baked 1x1 transparent PNG — instant response for out-of-range tiles (no PIL, no I/O)
TRANSPARENT_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def dzi_level_info(img_w, img_h, tile_size=TILE_SIZE):
    max_dim   = max(img_w, img_h)
    max_level = math.ceil(math.log2(max_dim)) if max_dim > 0 else 0
    levels = {}
    for lv in range(max_level + 1):
        scale = 2 ** (max_level - lv)
        w = max(1, math.ceil(img_w / scale))
        h = max(1, math.ceil(img_h / scale))
        levels[lv] = {"w": w, "h": h,
                      "cols": math.ceil(w / tile_size),
                      "rows": math.ceil(h / tile_size)}
    return levels, max_level


@app.on_event("startup")
async def startup():
    args  = G["args"]
    cache = Path(args.cache)

    with open(cache / "meta.json") as f:
        G["meta"] = json.load(f)

    G["df"] = pd.read_parquet(cache / "features.parquet")
    print(f"Loaded {len(G['df']):,} cells.")

    tif    = tifffile.TiffFile(args.image)
    series = tif.series[0]
    axes   = series.axes.upper()
    G["axes"] = axes

    raw_store = zarr.open(series.aszarr(), mode="r")
    if isinstance(raw_store, zarr.Array):
        G["store"] = raw_store
    else:
        G["store"] = raw_store["0"]   # zarr v3: Group keys are strings

    G["img_h"] = G["meta"]["img_h"]
    G["img_w"] = G["meta"]["img_w"]

    levels, max_level = dzi_level_info(G["img_w"], G["img_h"])
    G["dzi_levels"]      = levels
    G["dzi_max_level"]   = max_level
    G["min_heatmap_lv"]  = max_level - 4   # heatmap tiles exist from here upward
    print(f"Image {G['img_w']}x{G['img_h']}, axes={axes}, DZI max_level={max_level}")
    print(f"Heatmap tiles cover DZI levels {G['min_heatmap_lv']}..{max_level}")

    # Sample central patch to determine display range
    store = G["store"]
    cy = G["img_h"] // 2;  cx = G["img_w"] // 2
    sl = []
    for ax in axes:
        if ax == "C":   sl.append(slice(None))
        elif ax == "Y": sl.append(slice(cy, cy + 256))
        elif ax == "X": sl.append(slice(cx, cx + 256))
        else:           sl.append(0)
    try:
        sample = np.array(store[tuple(sl)])
        p1  = float(np.percentile(sample, 1))
        p99 = float(np.percentile(sample, 99))
        print(f"Pixel sample — dtype={sample.dtype}, shape={sample.shape}, "
              f"min={int(sample.min())}, max={int(sample.max())}, "
              f"p1={p1:.0f}, p99={p99:.0f}")
        G["display_min"] = p1
        G["display_max"] = p99
    except Exception as e:
        print(f"Could not sample pixel range: {e}")
        G["display_min"] = 0
        G["display_max"] = 255

    # Generate overview thumbnail (used for low zoom levels < 8)
    # Read a coarse downsample of the full image
    try:
        thumb_h, thumb_w = 512, 512
        sy = max(1, G["img_h"] // thumb_h)
        sx = max(1, G["img_w"] // thumb_w)
        sl_thumb = []
        for ax in axes:
            if ax == "C":   sl_thumb.append(slice(None))
            elif ax == "Y": sl_thumb.append(slice(0, G["img_h"], sy))
            elif ax == "X": sl_thumb.append(slice(0, G["img_w"], sx))
            else:           sl_thumb.append(0)
        thumb_arr = np.array(G["store"][tuple(sl_thumb)])
        if thumb_arr.ndim == 3 and axes.index("C") < axes.index("Y"):
            thumb_arr = np.moveaxis(thumb_arr, 0, -1)
        if thumb_arr.ndim == 2:
            thumb_arr = np.stack([thumb_arr]*3, axis=-1)
        elif thumb_arr.shape[-1] == 1:
            thumb_arr = np.repeat(thumb_arr, 3, axis=-1)
        elif thumb_arr.shape[-1] > 3:
            thumb_arr = thumb_arr[..., :3]
        if thumb_arr.dtype != np.uint8:
            lo = G.get("display_min", 0); hi = G.get("display_max", float(thumb_arr.max()))
            if hi > lo:
                thumb_arr = ((thumb_arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0,255).astype(np.uint8)
        thumb_img = Image.fromarray(thumb_arr)
        thumb_bytes = io.BytesIO()
        thumb_img.save(thumb_bytes, "JPEG", quality=80)
        G["thumbnail_bytes"] = thumb_bytes.getvalue()
        G["thumbnail_w"] = thumb_arr.shape[1]
        G["thumbnail_h"] = thumb_arr.shape[0]
        print(f"Thumbnail generated: {thumb_arr.shape[1]}x{thumb_arr.shape[0]}")
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")
        G["thumbnail_bytes"] = None


# ---------------------------------------------------------------------------
# Image reading
# ---------------------------------------------------------------------------

def read_region(y0, x0, h, w) -> np.ndarray:
    store = G["store"]
    axes  = G["axes"]
    img_h = G["img_h"]
    img_w = G["img_w"]

    y0c = max(0, y0);  x0c = max(0, x0)
    y1c = min(img_h, y0 + h)
    x1c = min(img_w, x0 + w)

    if y0c >= y1c or x0c >= x1c:
        return np.zeros((h, w, 3), dtype=np.uint8)

    sl = []
    for ax in axes:
        if ax == "C":   sl.append(slice(None))
        elif ax == "Y": sl.append(slice(y0c, y1c))
        elif ax == "X": sl.append(slice(x0c, x1c))
        else:           sl.append(0)

    try:
        arr = np.array(store[tuple(sl)])
    except Exception as e:
        print(f"read error: {e}")
        return np.zeros((h, w, 3), dtype=np.uint8)

    # (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and axes.index("C") < axes.index("Y"):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        lo = G.get("display_min", 0)
        hi = G.get("display_max", float(arr.max()))
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    rh, rw = arr.shape[:2]
    if (rh, rw) != (h, w):
        out = np.zeros((h, w, 3), dtype=np.uint8)
        dy = y0c - y0;  dx = x0c - x0
        out[dy:dy + rh, dx:dx + rw] = arr
        arr = out

    return arr


def to_jpeg(img: Image.Image, q=85) -> bytes:
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); return buf.getvalue()

def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

def dzi_xml(w, h, fmt):
    return (f'<?xml version="1.0" encoding="utf-8"?>'
            f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
            f' Format="{fmt}" Overlap="{OVERLAP}" TileSize="{TILE_SIZE}">'
            f'<Size Width="{w}" Height="{h}"/></Image>')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    p = Path(__file__).parent / "viewer.html"
    return FileResponse(p) if p.exists() else Response("viewer.html not found", status_code=404)

@app.get("/meta")
async def meta_route():
    return JSONResponse(G["meta"])

@app.get("/debug/patch.png")
async def debug_patch():
    cy = G["img_h"] // 2;  cx = G["img_w"] // 2
    region = read_region(cy - 256, cx - 256, 512, 512)
    return Response(to_png(Image.fromarray(region)), media_type="image/png")

@app.get("/debug/info")
async def debug_info():
    return JSONResponse({
        "axes": G["axes"], "img_w": G["img_w"], "img_h": G["img_h"],
        "store_shape": list(G["store"].shape), "store_dtype": str(G["store"].dtype),
        "display_min": G.get("display_min"), "display_max": G.get("display_max"),
        "dzi_max_level": G["dzi_max_level"], "min_heatmap_lv": G["min_heatmap_lv"],
    })

# ── Thumbnail (for low-zoom overview) ──

@app.get("/thumbnail.jpg")
async def thumbnail():
    tb = G.get("thumbnail_bytes")
    if tb:
        return Response(tb, media_type="image/jpeg")
    return Response(b"", status_code=404)

# ── H&E DZI ──

@app.get("/dzi/he.dzi")
async def he_dzi():
    return Response(dzi_xml(G["img_w"], G["img_h"], "jpg"), media_type="application/xml")

@app.get("/dzi/he_files/{level}/{col_row}.jpg")
async def he_tile(level: int, col_row: str):
    try:
        col, row = map(int, col_row.split("_"))
    except ValueError:
        raise HTTPException(400)

    lv = G["dzi_levels"].get(level)
    if not lv:
        raise HTTPException(404)

    ts, ov = TILE_SIZE, OVERLAP
    x0 = max(0, col * ts - ov);  y0 = max(0, row * ts - ov)
    x1 = min(lv["w"], (col + 1) * ts + ov)
    y1 = min(lv["h"], (row + 1) * ts + ov)
    tw = x1 - x0;  th = y1 - y0

    scale = G["img_w"] / lv["w"]
    region = read_region(int(y0 * scale), int(x0 * scale),
                         max(1, int(th * scale)), max(1, int(tw * scale)))
    tile = Image.fromarray(region).resize((tw, th), Image.BILINEAR)
    return Response(to_jpeg(tile), media_type="image/jpeg")

# ── Heatmap DZI ──

@app.get("/dzi/heatmap_{mode}.dzi")
async def heatmap_dzi(mode: str):
    return Response(dzi_xml(G["img_w"], G["img_h"], "png"), media_type="application/xml")

@app.get("/dzi/heatmap_{mode}_files/{level}/{col_row}.png")
async def heatmap_tile(mode: str, level: int, col_row: str):
    try:
        col, row = map(int, col_row.split("_"))
    except ValueError:
        raise HTTPException(400)

    # Tiles only pre-generated for levels min_heatmap_lv..max_level
    # Return instant transparent PNG for anything below that range
    if level < G["min_heatmap_lv"]:
        return Response(TRANSPARENT_PNG, media_type="image/png")

    our_z = level - G["min_heatmap_lv"]   # 0..4
    our_z = max(0, min(4, our_z))

    cache = Path(G["args"].cache)
    p = cache / "tiles" / "heatmap" / mode / str(our_z) / str(col) / f"{row}.png"
    if p.exists():
        return FileResponse(p, media_type="image/png")

    return Response(TRANSPARENT_PNG, media_type="image/png")

# ── Cell dots ──

@app.get("/cells")
async def cells(
    mode: str = Query(...),
    x0: float = Query(...), y0: float = Query(...),
    x1: float = Query(...), y1: float = Query(...),
    max_cells: int = Query(8000),
):
    df   = G["df"]
    mask = (df["Xt"] >= x0) & (df["Xt"] <= x1) & \
           (df["Yt"] >= y0) & (df["Yt"] <= y1)
    if mode == "vasculature": mask &= df["is_vessel"]
    elif mode == "immune":    mask &= df["is_immune"]

    sub   = df[mask]
    total = int(mask.sum())
    if len(sub) > max_cells:
        sub = sub.sample(max_cells, random_state=42)

    m  = G["meta"]
    cc = m["cell_colors"];  ic = m["immune_colors"];  vc = m["vasc_color"]

    records = []
    for _, r in sub.iterrows():
        if mode == "cells":         c = cc.get(r.get("cell_type", "other"), cc["other"])
        elif mode == "vasculature": c = vc
        else:                       c = ic.get(r.get("immune_sub", "other"), ic["other"])
        records.append({"x": r["Xt"], "y": r["Yt"],
                        "r": c[0], "g": c[1], "b": c[2], "a": c[3]})

    return JSONResponse({"cells": records, "total_in_view": total})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--cache",  default="cache")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--host",   default="127.0.0.1")
    args = parser.parse_args()
    G["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")