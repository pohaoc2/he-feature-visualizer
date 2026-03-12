#!/usr/bin/env python3
"""
preprocess.py — Run once before starting the viewer server.

Usage:
    python preprocess.py --features data/CRC02.csv \
                         --image    data/CRC02-HE.ome.tif \
                         --out      cache/
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import tifffile

# ── Config ──────────────────────────────────────────────────────
TILE_SIZE    = 254        # Must match server.py
OVERLAP      = 1
HEATMAP_ZOOM = [0, 1, 2, 3, 4]   # our zoom levels (0 = most zoomed out)
SIGMA        = 6          # gaussian blur in tile pixels

CELL_COLORS = {
    "tumor":   (255, 100, 150, 180),
    "immune":  ( 80, 220,  80, 180),
    "stromal": (255, 160,  50, 180),
    "other":   (180, 180, 180, 100),
}
VASC_COLOR = (220, 50, 50, 200)
IMMUNE_COLORS = {
    "CD8a":  ( 60, 120, 255, 200),
    "CD68":  ( 60, 200,  80, 200),
    "FOXP3": (160,  60, 220, 200),
    "CD4":   (240, 210,  40, 200),
    "CD20":  ( 40, 210, 210, 200),
    "other": (200, 200, 200, 120),
}


# ── DZI level info ───────────────────────────────────────────────

def dzi_level_info(img_w, img_h, tile_size=TILE_SIZE):
    """Returns dict of {level: {w, h, cols, rows}} and max_level."""
    max_dim   = max(img_w, img_h)
    max_level = math.ceil(math.log2(max_dim))
    levels = {}
    for lv in range(max_level + 1):
        scale = 2 ** (max_level - lv)
        w = max(1, math.ceil(img_w / scale))
        h = max(1, math.ceil(img_h / scale))
        levels[lv] = {
            "w": w, "h": h,
            "cols": math.ceil(w / tile_size),
            "rows": math.ceil(h / tile_size),
        }
    return levels, max_level


# ── Cell type assignment ─────────────────────────────────────────

def compute_thresholds(df, markers, pct=95):
    return {m: float(np.percentile(df[m].dropna(), pct))
            for m in markers if m in df.columns}

def assign_cell_type(df, thresh):
    ct = pd.Series("other", index=df.index)
    for col, t, label in [
        ("aSMA",    thresh.get("aSMA",    np.inf), "stromal"),
        ("CD45",    thresh.get("CD45",    np.inf), "immune"),
        ("Keratin", thresh.get("Keratin", np.inf), "tumor"),
    ]:
        if col in df.columns:
            ct[df[col] > t] = label
    return ct

def assign_immune_sub(df, thresh):
    sub = pd.Series("other", index=df.index)
    for marker in ["CD20", "CD4", "FOXP3", "CD68", "CD8a"]:
        if marker in df.columns:
            sub[df[marker] > thresh.get(marker, np.inf)] = marker
    return sub

def make_colors(df, mode, thresh):
    """Return (N,4) uint8 RGBA array."""
    c = np.zeros((len(df), 4), dtype=np.uint8)
    if mode == "cells":
        ct = assign_cell_type(df, thresh)
        for name, rgba in CELL_COLORS.items():
            c[ct == name] = rgba
    elif mode == "vasculature":
        mask = df.get("CD31", pd.Series(0, index=df.index)) > thresh.get("CD31", np.inf)
        c[mask]  = VASC_COLOR
        c[~mask] = (0, 0, 0, 0)
    elif mode == "immune":
        cd45 = df.get("CD45", pd.Series(0, index=df.index)) > thresh.get("CD45", np.inf)
        sub  = assign_immune_sub(df, thresh)
        for name, rgba in IMMUNE_COLORS.items():
            c[cd45 & (sub == name)] = rgba
        c[~cd45] = (0, 0, 0, 0)
    return c


# ── Heatmap tile rendering ────────────────────────────────────────

def render_tile(xs, ys, colors, lv_info, col, row,
                tile_size=TILE_SIZE, overlap=OVERLAP, sigma=SIGMA):
    """
    Render one heatmap tile.
    xs, ys: cell coordinates in full-resolution image pixels.
    lv_info: {w, h, cols, rows} for this DZI level.
    """
    lw, lh = lv_info["w"], lv_info["h"]

    # Tile bounds in level pixels (with overlap)
    x0 = max(0, col * tile_size - overlap)
    y0 = max(0, row * tile_size - overlap)
    x1 = min(lw, (col + 1) * tile_size + overlap)
    y1 = min(lh, (row + 1) * tile_size + overlap)
    tw = x1 - x0;  th = y1 - y0

    if tw <= 0 or th <= 0:
        return Image.new("RGBA", (tile_size, tile_size), (0,0,0,0))

    # Scale factor: full-res px → level px
    scale_x = lw / (xs.max() - xs.min() + 1e-9) if xs.max() > xs.min() else 1
    scale_y = lh / (ys.max() - ys.min() + 1e-9) if ys.max() > ys.min() else 1
    # Better: scale by image dimensions stored externally — passed as global below
    # (handled by caller passing pre-scaled coordinates)

    # Filter cells within this tile (with margin)
    margin = 3 / scale_x  # a few pixels margin in image space
    mask = (xs >= (x0 - margin) / scale_x + xs.min()) & \
           (xs <= (x1 + margin) / scale_x + xs.min()) & \
           (ys >= (y0 - margin) / scale_y + ys.min()) & \
           (ys <= (y1 + margin) / scale_y + ys.min())

    if not mask.any():
        return Image.new("RGBA", (tw, th), (0,0,0,0))

    lxs = xs[mask]
    lys = ys[mask]
    lc  = colors[mask]

    # Map image coords → tile pixel coords
    px = ((lxs * scale_x - x0)).astype(int).clip(0, tw - 1)
    py = ((lys * scale_y - y0)).astype(int).clip(0, th - 1)

    grids = []
    for ch in range(4):
        g = np.zeros((th, tw), dtype=np.float32)
        np.add.at(g, (py, px), lc[:, ch].astype(np.float32))
        g = gaussian_filter(g, sigma=sigma)
        grids.append(g)

    alpha = grids[3]
    if alpha.max() > 0:
        alpha = (alpha / alpha.max() * 200).clip(0, 255)
    grids[3] = alpha

    data = np.stack(grids, axis=-1).astype(np.uint8)
    return Image.fromarray(data, "RGBA")


def generate_heatmap_tiles(df, thresh, img_w, img_h, out_dir,
                            modes=("cells", "vasculature", "immune")):
    levels, max_level = dzi_level_info(img_w, img_h)

    # Normalise cell coordinates to [0..1] then scale to each level
    xs_norm = df["Xt"].values / img_w
    ys_norm = df["Yt"].values / img_h

    for mode in modes:
        print(f"  [{mode}] computing cell colors …")
        colors = make_colors(df, mode, thresh)

        for our_z in HEATMAP_ZOOM:
            # Map our zoom 0–4 to DZI level (max_level - 4 + our_z)
            dzi_lv = max_level - 4 + our_z
            dzi_lv = max(0, min(max_level, dzi_lv))
            lv     = levels[dzi_lv]

            xs_lv = xs_norm * lv["w"]
            ys_lv = ys_norm * lv["h"]

            print(f"  [{mode}] z={our_z} → DZI {dzi_lv}  "
                  f"({lv['w']}x{lv['h']}, {lv['cols']}x{lv['rows']} tiles)")

            n_total = lv["cols"] * lv["rows"]
            n_done  = 0

            for col in range(lv["cols"]):
                for row in range(lv["rows"]):
                    # Tile bounds in level pixels
                    x0 = max(0, col * TILE_SIZE - OVERLAP)
                    y0 = max(0, row * TILE_SIZE - OVERLAP)
                    x1 = min(lv["w"], (col + 1) * TILE_SIZE + OVERLAP)
                    y1 = min(lv["h"], (row + 1) * TILE_SIZE + OVERLAP)
                    tw = x1 - x0;  th = y1 - y0

                    if tw <= 0 or th <= 0:
                        continue

                    # Filter cells in tile bounds (with small margin)
                    margin = max(2, SIGMA * 2)
                    mask = ((xs_lv >= x0 - margin) & (xs_lv < x1 + margin) &
                            (ys_lv >= y0 - margin) & (ys_lv < y1 + margin))

                    tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))

                    if mask.any():
                        lxs = xs_lv[mask]
                        lys = ys_lv[mask]
                        lc  = colors[mask]

                        px = (lxs - x0).astype(int).clip(0, tw - 1)
                        py = (lys - y0).astype(int).clip(0, th - 1)

                        grids = []
                        for ch in range(4):
                            g = np.zeros((th, tw), dtype=np.float32)
                            np.add.at(g, (py, px), lc[:, ch].astype(np.float32))
                            g = gaussian_filter(g, sigma=SIGMA)
                            grids.append(g)

                        alpha = grids[3]
                        if alpha.max() > 0:
                            alpha = (alpha / alpha.max() * 200).clip(0, 255)
                        grids[3] = alpha
                        data = np.stack(grids, axis=-1).astype(np.uint8)
                        tile = Image.fromarray(data, "RGBA")

                    tile_dir = out_dir / "tiles" / "heatmap" / mode / str(our_z) / str(col)
                    tile_dir.mkdir(parents=True, exist_ok=True)
                    tile.save(tile_dir / f"{row}.png")
                    n_done += 1

            print(f"    → {n_done}/{n_total} tiles saved.")

        print(f"  [{mode}] done.\n")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--image",    required=True)
    parser.add_argument("--out",      default="cache")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("Loading features …")
    if args.features.endswith(".xlsx"):
        df = pd.read_excel(args.features)
    else:
        df = pd.read_csv(args.features)
    print(f"  {len(df):,} cells loaded.")

    # Image dimensions
    print("Reading image metadata …")
    with tifffile.TiffFile(args.image) as tif:
        series = tif.series[0]
        axes   = series.axes.upper()
        shape  = series.shape
        print(f"  axes={axes} shape={shape}")
        img_h = shape[axes.index("Y")]
        img_w = shape[axes.index("X")]
    print(f"  {img_w}w x {img_h}h")

    # Thresholds
    markers = ["Keratin", "CD45", "aSMA", "CD31", "CD8a", "CD68", "FOXP3", "CD4", "CD20"]
    thresh  = compute_thresholds(df, markers)
    print("Thresholds (95th pct):", {k: f"{v:.1f}" for k, v in thresh.items()})

    # Augment + save parquet
    print("Saving features.parquet …")
    df["cell_type"]  = assign_cell_type(df, thresh)
    df["immune_sub"] = assign_immune_sub(df, thresh)
    df["is_vessel"]  = (df.get("CD31", 0) > thresh.get("CD31", np.inf)).astype(bool)
    df["is_immune"]  = (df.get("CD45", 0) > thresh.get("CD45", np.inf)).astype(bool)
    df.to_parquet(out_dir / "features.parquet", index=False)

    # Save meta
    _, max_level = dzi_level_info(img_w, img_h)
    meta = {
        "img_w": img_w, "img_h": img_h,
        "thresholds": thresh,
        "cell_colors":   {k: list(v) for k, v in CELL_COLORS.items()},
        "immune_colors": {k: list(v) for k, v in IMMUNE_COLORS.items()},
        "vasc_color":    list(VASC_COLOR),
        "zoom_levels":   HEATMAP_ZOOM,
        "tile_size":     TILE_SIZE,
        "dzi_max_level": max_level,
        "n_cells":       len(df),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved cache/meta.json")

    # Generate heatmap tiles
    print("\nGenerating heatmap tiles …")
    generate_heatmap_tiles(df, thresh, img_w, img_h, out_dir)

    print("✓ Preprocessing complete. Run: python server.py --image data/CRC02-HE.ome.tif --cache cache/")


if __name__ == "__main__":
    main()
