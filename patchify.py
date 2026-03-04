#!/usr/bin/env python3
"""
patchify.py — Crop H&E into 256×256 patches and matching feature rasters.
Drops mostly-empty patches (by tissue coverage). Saves under processed/ for
one-patch-at-a-time viewing on limited hardware.

Usage:
    python patchify.py --image data/CRC02-HE.ome.tif \\
                      --features-csv data/CRC02.csv \\
                      --out processed/ \\
                      --stride 256 \\
                      --features cell_mask vasculature immune
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import zarr

# Reuse cell-type and marker logic from preprocess
from preprocess import (
    CELL_COLORS,
    IMMUNE_COLORS,
    VASC_COLOR,
    compute_thresholds,
    make_colors,
)
from tissue_mask import tissue_fraction_rgb

PATCH_SIZE = 256
MARKERS = ["Keratin", "CD45", "aSMA", "CD31", "CD8a", "CD68", "FOXP3", "CD4", "CD20"]
DOT_RADIUS = 2  # pixels for each cell dot


def get_image_reader(tif, image_path, cache_meta_path=None):
    """Return (img_w, img_h, read_region_fn). read_region_fn(y0, x0, h, w) -> (H,W,3) uint8.
    Caller must keep tif open for the lifetime of read_region."""
    series = tif.series[0]
    axes = series.axes.upper()
    shape = series.shape
    img_h = shape[axes.index("Y")]
    img_w = shape[axes.index("X")]

    if cache_meta_path and Path(cache_meta_path).exists():
        with open(cache_meta_path) as f:
            meta = json.load(f)
        img_w, img_h = meta["img_w"], meta["img_h"]

    raw_store = zarr.open(series.aszarr(), mode="r")
    store = raw_store if isinstance(raw_store, zarr.Array) else raw_store["0"]

    def _read_region_raw(y0, x0, h, w):
        """Read patch in store dtype (e.g. uint16); same layout as read_region but no uint8 norm."""
        y0c = max(0, int(y0))
        x0c = max(0, int(x0))
        y1c = min(img_h, y0c + h)
        x1c = min(img_w, x0c + w)
        if y0c >= y1c or x0c >= x1c:
            return np.zeros((h, w, 3), dtype=store.dtype)
        sl = []
        for ax in axes:
            if ax == "C":
                sl.append(slice(None))
            elif ax == "Y":
                sl.append(slice(y0c, y1c))
            elif ax == "X":
                sl.append(slice(x0c, x1c))
            else:
                sl.append(0)
        arr = np.array(store[tuple(sl)])
        if arr.ndim == 3 and axes.index("C") < axes.index("Y"):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        rh, rw = arr.shape[:2]
        if (rh, rw) != (h, w):
            out = np.zeros((h, w, 3), dtype=arr.dtype)
            dy, dx = y0c - int(y0), x0c - int(x0)
            out[dy : dy + rh, dx : dx + rw] = arr
            arr = out
        return arr

    def read_region(y0, x0, h, w):
        """Read patch as uint8 RGB, normalized per-patch (p1/p99) so contrast is preserved."""
        raw = _read_region_raw(y0, x0, h, w)
        if raw.dtype == np.uint8:
            return raw
        p1, p99 = np.percentile(raw, (1, 99))
        if p99 > p1:
            out = ((raw.astype(np.float32) - p1) / (p99 - p1) * 255).clip(0, 255).astype(np.uint8)
        else:
            out = np.zeros((h, w, 3), dtype=np.uint8)
        return out

    return img_w, img_h, read_region


def tissue_fraction(rgb_patch, bg_threshold):
    """Fraction of pixels that are tissue (darker than bg_threshold)."""
    gray = np.dot(rgb_patch.astype(np.float32) / 255.0, [0.299, 0.587, 0.114])
    return (gray < (bg_threshold / 255.0)).mean()


def draw_cell_dots(colors, lx, ly, size=PATCH_SIZE, radius=DOT_RADIUS):
    """Draw dots at (lx, ly) with colors (N,4) into 256×256 RGBA.
    Image indexing: out[row, col] = out[y, x]; lx = local x (col), ly = local y (row)."""
    out = np.zeros((size, size, 4), dtype=np.uint8)
    r = radius
    for k in range(len(lx)):
        col = int(np.round(lx[k]))  # x in patch
        row = int(np.round(ly[k]))  # y in patch
        c0 = max(0, col - r)
        c1 = min(size, col + r + 1)
        r0 = max(0, row - r)
        r1 = min(size, row + r + 1)
        rgba = np.asarray(colors[k], dtype=np.uint8).reshape(1, 1, 4)
        out[r0:r1, c0:c1, :] = rgba
    return out


def main():
    parser = argparse.ArgumentParser(description="Patch H&E and feature rasters for lightweight viewing.")
    parser.add_argument("--image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--features-csv", required=True, help="Path to cell features CSV (with Xt, Yt)")
    parser.add_argument("--out", default="processed", help="Output directory (e.g. processed/)")
    parser.add_argument("--stride", type=int, default=256, help="Patch stride (default 256 = no overlap)")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["cell_mask", "vasculature", "immune"],
        choices=["cell_mask", "vasculature", "immune"],
        help="Feature layers to generate",
    )
    parser.add_argument("--tissue-min", type=float, default=0.1, help="Min tissue fraction to keep patch (default 0.1)")
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=220.0,
        help="Grayscale below this = tissue (default 220)",
    )
    parser.add_argument("--cache-meta", default=None, help="Path to cache/meta.json for dimensions and thresholds")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("Loading features …")
    path = args.features_csv
    df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    print(f"  {len(df):,} cells.")

    # Image reader (windowed); keep tif open for the run
    print("Opening image (windowed) …")
    tif = tifffile.TiffFile(args.image)
    img_w, img_h, read_region = get_image_reader(tif, args.image, args.cache_meta)
    print(f"  {img_w}×{img_h}")

    # Coordinate columns: prefer Xt,Yt (image pixel space), else X,Y
    x_col = "Xt" if "Xt" in df.columns else "X"
    y_col = "Yt" if "Yt" in df.columns else "Y"
    if x_col not in df.columns or y_col not in df.columns:
        raise SystemExit(f"CSV must have 'Xt'/'Yt' or 'X'/'Y' for coordinates. Found: {list(df.columns)}")
    x_vals = df[x_col].values.astype(np.float64)
    y_vals = df[y_col].values.astype(np.float64)
    # If coordinates look normalized (in 0..1), scale to image size
    x_max, y_max = float(np.nanmax(x_vals)), float(np.nanmax(y_vals))
    if x_max <= 1.0 and y_max <= 1.0 and (x_max > 0 or y_max > 0):
        print("  Scaling normalized coordinates (0–1) to image pixels.")
        df = df.copy()
        df[x_col] = x_vals * img_w
        df[y_col] = y_vals * img_h
        x_vals = df[x_col].values
        y_vals = df[y_col].values
    print(f"  Coordinates: {x_col}=[{np.nanmin(x_vals):.1f}, {np.nanmax(x_vals):.1f}], "
          f"{y_col}=[{np.nanmin(y_vals):.1f}, {np.nanmax(y_vals):.1f}]")

    # Thresholds
    if args.cache_meta and Path(args.cache_meta).exists():
        with open(args.cache_meta) as f:
            thresh = json.load(f)["thresholds"]
        print("  Using thresholds from cache-meta.")
    else:
        thresh = compute_thresholds(df, MARKERS)
        print("  Thresholds (95th):", {k: f"{v:.1f}" for k, v in thresh.items()})

    stride = args.stride
    n_cols = max(0, (img_w - PATCH_SIZE) // stride + 1) if img_w >= PATCH_SIZE else 0
    n_rows = max(0, (img_h - PATCH_SIZE) // stride + 1) if img_h >= PATCH_SIZE else 0
    total = n_rows * n_cols
    print(f"Patch grid: stride={stride} → {n_rows}×{n_cols} = {total} candidate patches")

    # Create output dirs
    (out_dir / "he").mkdir(parents=True, exist_ok=True)
    if args.features:
        (out_dir / "overlay_cells").mkdir(parents=True, exist_ok=True)
    for f in args.features:
        (out_dir / f).mkdir(parents=True, exist_ok=True)

    index = []
    kept = 0
    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * stride
            x0 = j * stride
            if y0 + PATCH_SIZE > img_h or x0 + PATCH_SIZE > img_w:
                continue

            # Read H&E patch
            patch_rgb = read_region(y0, x0, PATCH_SIZE, PATCH_SIZE)
            if tissue_fraction_rgb(patch_rgb) < args.tissue_min:
                continue

            # Cells in this patch (x_col, y_col in image pixel space)
            mask = (
                (df[x_col] >= x0)
                & (df[x_col] < x0 + PATCH_SIZE)
                & (df[y_col] >= y0)
                & (df[y_col] < y0 + PATCH_SIZE)
            )
            sub = df.loc[mask]
            n_cells = len(sub)

            # Only keep patches that have at least one cell when we're writing feature layers
            if args.features and n_cells == 0:
                continue

            kept += 1
            if kept <= 3:
                print(f"  [patch {i}_{j}] cells in patch: {n_cells}")

            # Save H&E
            Image.fromarray(patch_rgb).save(out_dir / "he" / f"{i}_{j}.png")

            lx = (sub[x_col].values - x0).astype(np.float32)
            ly = (sub[y_col].values - y0).astype(np.float32)

            if args.features:
                # Overlay = all cell types (same as cell_mask content)
                colors_cells = make_colors(sub, "cells", thresh)
                overlay = draw_cell_dots(colors_cells, lx, ly)
                Image.fromarray(overlay, "RGBA").save(out_dir / "overlay_cells" / f"{i}_{j}.png")

            if "cell_mask" in args.features:
                colors_cells = make_colors(sub, "cells", thresh)
                out = draw_cell_dots(colors_cells, lx, ly)
                Image.fromarray(out, "RGBA").save(out_dir / "cell_mask" / f"{i}_{j}.png")
            if "vasculature" in args.features:
                colors_v = make_colors(sub, "vasculature", thresh)
                out = draw_cell_dots(colors_v, lx, ly)
                Image.fromarray(out, "RGBA").save(out_dir / "vasculature" / f"{i}_{j}.png")
            if "immune" in args.features:
                colors_i = make_colors(sub, "immune", thresh)
                out = draw_cell_dots(colors_i, lx, ly)
                Image.fromarray(out, "RGBA").save(out_dir / "immune" / f"{i}_{j}.png")

            index.append({"i": i, "j": j, "x0": x0, "y0": y0, "x1": x0 + PATCH_SIZE, "y1": y0 + PATCH_SIZE})

        if (i + 1) % 50 == 0 or i == n_rows - 1:
            print(f"  Row {i+1}/{n_rows} … kept {kept} so far")

    # Save index
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "patches": index,
                "stride": stride,
                "tissue_min": args.tissue_min,
                "img_w": img_w,
                "img_h": img_h,
                "features": args.features,
            },
            f,
            indent=2,
        )
    print(f"Kept {kept}/{total} patches. Index: {index_path}")
    print("Done. Run patch viewer: python server_patches.py --processed processed/ --port 8000")


if __name__ == "__main__":
    main()
