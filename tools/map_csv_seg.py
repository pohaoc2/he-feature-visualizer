#!/usr/bin/env python3
"""
map_csv_seg.py — QC tool: overlay CSV cell centroids on nucleus segmentation mask.

Supports two mask types (auto-detected by unique value count):
  - Instance mask (e.g. WD-76845-097.mask.tif, uint32): each nonzero pixel is a
    cell ID.  Reports how many CSV centroids land on a unique cell ID, and how
    many mask cells have a matching CSV centroid.
  - Binary mask (e.g. WD-76845-097.ome.seg.tif, uint16 0/65535): reports fraction
    of CSV centroids that fall on foreground pixels.

Both masks are assumed to be in MX pixel space. CSV Xt/Yt are divided by
--csv-mpp to convert from µm to MX px.

The overlay PNG shows:
  - Mask rendered in grayscale (binary) or pseudo-color (instance)
  - Green dots: CSV centroids that hit a mask cell
  - Red dots:   CSV centroids outside any mask cell

Usage
-----
# Instance mask (recommended)
python -m tools.map_csv_seg \\
    --seg  data/WD-76845-097.mask.tif \\
    --csv  data/WD-76845-097.csv \\
    --save-png data/csv_mask_overlay.png

# Binary mask
python -m tools.map_csv_seg \\
    --seg  data/WD-76845-097.ome.seg.tif \\
    --csv  data/WD-76845-097.csv \\
    --save-png data/csv_seg_overlay.png
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd
import tifffile

from utils.ome import get_image_dims, open_zarr_store, read_overview_chw

# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------


def load_mask_overview(
    mask_path: pathlib.Path, ds: int
) -> tuple[np.ndarray, int, int, bool]:
    """Load mask at 1/ds resolution.

    Returns
    -------
    mask_ov : (H, W) array — raw downsampled values (uint32 or uint8)
    img_w, img_h : full-res dimensions
    is_instance : True if dtype is uint32 (instance segmentation with per-cell IDs)
    """
    with tifffile.TiffFile(str(mask_path)) as tif:
        store = open_zarr_store(tif)
        img_w, img_h, axes = get_image_dims(tif)
        # Detect instance mask from full-res dtype before downsampling
        is_instance = tif.series[0].dtype == np.uint32
        chw = read_overview_chw(store, axes, img_h, img_w, ds)

    mask_ov = chw[0]  # (H, W)
    return mask_ov, img_w, img_h, is_instance


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csv_centroids(csv_path: pathlib.Path, csv_mpp: float = 0.65) -> np.ndarray:
    """Load (Xt, Yt) from CSV as (N, 2) float64 in MX px.

    Divides by csv_mpp to convert from µm to MX px.
    """
    df = pd.read_csv(csv_path)
    if "Xt" not in df.columns or "Yt" not in df.columns:
        raise ValueError(
            f"CSV must have 'Xt' and 'Yt' columns; found: {list(df.columns)[:10]}"
        )
    pts = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
    if csv_mpp != 1.0:
        pts = pts / csv_mpp
    return pts


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_centroids_to_mask(
    centroids_mx: np.ndarray,
    mask_ov: np.ndarray,
    ds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up mask value at each centroid position (overview resolution).

    Returns
    -------
    mask_vals : (N,) int   — mask value at each centroid (0 = background)
    inside    : (N,) bool  — True if centroid hits a nonzero mask pixel
    """
    h_ov, w_ov = mask_ov.shape
    cx = np.clip((centroids_mx[:, 0] / ds).astype(int), 0, w_ov - 1)
    cy = np.clip((centroids_mx[:, 1] / ds).astype(int), 0, h_ov - 1)
    mask_vals = mask_ov[cy, cx].astype(np.int64)
    inside = mask_vals > 0
    return mask_vals, inside


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def _instance_to_gray(mask_ov: np.ndarray) -> np.ndarray:
    """Convert instance mask to uint8 grayscale for overlay background."""
    fg = (mask_ov > 0).astype(np.uint8) * 80  # dim gray for cell footprints
    return fg


def make_overlay(
    mask_ov: np.ndarray,
    centroids_mx: np.ndarray,
    inside: np.ndarray,
    ds: int,
    dot_radius: int,
    max_dots: int,
) -> np.ndarray:
    """Build BGR overlay image with centroids drawn on the mask."""
    is_instance = mask_ov.max() > 255

    if is_instance:
        gray = _instance_to_gray(mask_ov)
    else:
        gray = (mask_ov > 0).astype(np.uint8) * 200

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h_ov, w_ov = mask_ov.shape

    cx = np.clip((centroids_mx[:, 0] / ds).astype(int), 0, w_ov - 1)
    cy = np.clip((centroids_mx[:, 1] / ds).astype(int), 0, h_ov - 1)

    n = len(centroids_mx)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(n, max_dots), replace=False)

    for i in idx:
        color = (0, 220, 0) if inside[i] else (0, 0, 220)  # BGR green / red
        cv2.circle(overlay, (int(cx[i]), int(cy[i])), dot_radius, color, -1)

    return overlay


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    mask_path = pathlib.Path(args.seg)
    csv_path = pathlib.Path(args.csv)
    ds = args.downsample

    print(f"Loading mask: {mask_path}  (downsample={ds})")
    mask_ov, img_w, img_h, is_instance = load_mask_overview(mask_path, ds)
    mask_type = "instance" if is_instance else "binary"
    print(f"  Mask type    : {mask_type}")
    print(f"  Overview shape: {mask_ov.shape}  (full-res: {img_w}×{img_h})")

    n_fg_px = int((mask_ov > 0).sum())
    pct_fg = 100 * n_fg_px / mask_ov.size
    print(f"  Foreground px: {n_fg_px:,} / {mask_ov.size:,}  ({pct_fg:.1f}%)")

    if is_instance:
        n_cells_mask = int(np.unique(mask_ov[mask_ov > 0]).shape[0])
        print(f"  Unique cell IDs in overview: {n_cells_mask:,}")

    print(f"\nLoading CSV: {csv_path}  (csv-mpp={args.csv_mpp})")
    centroids = load_csv_centroids(csv_path, csv_mpp=args.csv_mpp)
    print(f"  {len(centroids):,} CSV cells loaded.")

    mask_vals, inside = match_centroids_to_mask(centroids, mask_ov, ds)
    n_inside = int(inside.sum())
    pct_inside = 100 * n_inside / max(1, len(centroids))

    print(f"\nResults:")
    print(
        f"  CSV centroids hitting mask : {n_inside:,} / {len(centroids):,}  ({pct_inside:.1f}%)"
    )
    print(f"  CSV centroids outside mask : {len(centroids) - n_inside:,}")

    if is_instance:
        n_matched_ids = int(np.unique(mask_vals[inside]).shape[0])
        print(f"  Unique mask cell IDs hit   : {n_matched_ids:,}")

    if args.save_png:
        out_path = pathlib.Path(args.save_png)
        print(f"\nSaving overlay PNG: {out_path}")
        overlay = make_overlay(
            mask_ov=mask_ov,
            centroids_mx=centroids,
            inside=inside,
            ds=ds,
            dot_radius=args.dot_radius,
            max_dots=args.max_dots,
        )
        cv2.imwrite(str(out_path), overlay)
        n_shown = min(len(centroids), args.max_dots)
        print(f"  ({n_shown:,} of {len(centroids):,} dots shown)")
        print(f"  Green = hits mask cell, Red = misses")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QC: overlay CSV cell centroids on nucleus segmentation mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seg",
        required=True,
        help="Segmentation mask OME-TIFF (instance uint32 or binary uint16).",
    )
    p.add_argument(
        "--csv", required=True, help="Cell features CSV with Xt, Yt columns."
    )
    p.add_argument("--save-png", default=None, help="Path to save overlay PNG.")
    p.add_argument(
        "--downsample", type=int, default=8, help="Overview downsample factor."
    )
    p.add_argument(
        "--dot-radius", type=int, default=2, help="Dot radius for centroids."
    )
    p.add_argument(
        "--csv-mpp",
        type=float,
        default=0.65,
        help="µm/px of CSV coordinate space. Divides Xt/Yt by this value to "
        "convert from µm to MX px (default: 0.65 for WD-76845-097).",
    )
    p.add_argument(
        "--max-dots", type=int, default=50000, help="Max number of dots to show."
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
