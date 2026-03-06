"""Visualize a cell-mask TIF alongside an OME-TIFF as side-by-side PNGs.

Two modes
---------
overview  — whole-slide at a reduced resolution (fast, good for orientation)
crop      — full-resolution crop of a small region (shows cell-level detail)

Usage
-----
# Overview at 64x downsample
python viz_mask.py overview --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif

# Full-res crop — auto-detect densest cell region
python viz_mask.py crop --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif

# Full-res crop at a specific location
python viz_mask.py crop --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif \
    --row 15488 --col 20928 --crop-size 1024

# Custom output path and downsample
python viz_mask.py overview --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif \
    --out data/overview.png --downsample 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def colorize_mask(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Map a uint32 label mask to an RGB image with random per-cell colours."""
    rng = np.random.default_rng(seed)
    max_id = int(mask.max())
    lut = rng.integers(30, 256, size=(max_id + 1, 3), dtype=np.uint8)
    lut[0] = 0  # background = black
    return lut[mask]


def contrast_stretch(arr: np.ndarray) -> np.ndarray:
    """Linearly stretch an array to uint8 [0, 255]."""
    vmin, vmax = int(arr.min()), int(arr.max())
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    return (
        ((arr.astype(np.float32) - vmin) / (vmax - vmin) * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    """Compose two (H, W, 3) arrays horizontally with a black gap."""
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return canvas


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def visualize_overview(
    mask_path: Path,
    ome_path: Path,
    out_path: Path,
    downsample: int = 64,
) -> None:
    """Side-by-side overview PNG at `downsample`x resolution."""
    t0 = time.perf_counter()

    print(f"[overview] reading mask at 1/{downsample} resolution ...")
    with tifffile.TiffFile(str(mask_path)) as tif:
        mask_small = tif.pages[0].asarray()[::downsample, ::downsample]

    n_cells = int((np.unique(mask_small) != 0).sum())
    print(f"[overview] mask: {mask_small.shape}, {n_cells} cell IDs")
    mask_rgb = colorize_mask(mask_small)

    print(f"[overview] reading OME at 1/{downsample} resolution ...")
    with tifffile.TiffFile(str(ome_path)) as tif:
        ome_small = tif.pages[0].asarray()[::downsample, ::downsample]

    vmin, vmax = int(ome_small.min()), int(ome_small.max())
    print(f"[overview] OME: {ome_small.shape}, range=[{vmin}, {vmax}]")
    ome_rgb = np.stack([contrast_stretch(ome_small)] * 3, axis=-1)

    img = Image.fromarray(side_by_side(mask_rgb, ome_rgb))
    img.save(str(out_path))
    print(
        f"[overview] saved {out_path} ({img.size[0]}×{img.size[1]}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )


# ---------------------------------------------------------------------------
# Full-res crop
# ---------------------------------------------------------------------------


def find_dense_region(
    mask_path: Path,
    downsample: int = 64,
    search_window: int = 16,
) -> tuple[int, int]:
    """Return the full-res (row, col) center of the densest cell region."""
    from scipy.ndimage import uniform_filter

    with tifffile.TiffFile(str(mask_path)) as tif:
        mask_ds = tif.pages[0].asarray()[::downsample, ::downsample]

    density = uniform_filter((mask_ds > 0).astype(np.float32), size=search_window)
    peak_ds = np.unravel_index(density.argmax(), density.shape)
    return int(peak_ds[0] * downsample), int(peak_ds[1] * downsample)


def read_crop(path: Path, row: int, col: int, size: int) -> np.ndarray:
    """Read a (size × size) region from a single-page TIFF at full resolution."""
    with tifffile.TiffFile(str(path)) as tif:
        h_full, w_full = tif.pages[0].shape
        y0 = max(0, row - size // 2)
        x0 = max(0, col - size // 2)
        y1 = min(h_full, y0 + size)
        x1 = min(w_full, x0 + size)
        return tif.pages[0].asarray()[y0:y1, x0:x1]


def visualize_crop(
    mask_path: Path,
    ome_path: Path,
    out_path: Path,
    row: int | None = None,
    col: int | None = None,
    crop_size: int = 1024,
    search_downsample: int = 64,
) -> None:
    """Full-resolution side-by-side crop PNG.

    If `row`/`col` are not given, the densest cell region is auto-detected.
    """
    t0 = time.perf_counter()

    if row is None or col is None:
        print(
            f"[crop] auto-detecting densest region (search at {search_downsample}x) ..."
        )
        row, col = find_dense_region(mask_path, downsample=search_downsample)
        print(f"[crop] center: row={row}, col={col}")

    print(f"[crop] reading {crop_size}×{crop_size} crop from mask ...")
    mask_crop = read_crop(mask_path, row, col, crop_size)

    print(f"[crop] reading {crop_size}×{crop_size} crop from OME ...")
    ome_crop = read_crop(ome_path, row, col, crop_size)

    n_cells = int((np.unique(mask_crop) != 0).sum())
    print(
        f"[crop] mask: {mask_crop.shape}, {n_cells} cells, "
        f"{(mask_crop > 0).mean() * 100:.1f}% non-zero"
    )
    print(f"[crop] OME:  {ome_crop.shape}, unique values: {np.unique(ome_crop)}")

    mask_rgb = colorize_mask(mask_crop)
    ome_rgb = np.stack([contrast_stretch(ome_crop)] * 3, axis=-1)

    img = Image.fromarray(side_by_side(mask_rgb, ome_rgb, gap=4))
    img.save(str(out_path))
    print(
        f"[crop] saved {out_path} ({img.size[0]}×{img.size[1]}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mask", required=True, type=Path, help="Path to the uint32 mask TIF"
    )
    p.add_argument("--ome", required=True, type=Path, help="Path to the OME-TIFF")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize mask + OME side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ov = sub.add_parser("overview", help="Whole-slide overview at reduced resolution")
    _add_common(p_ov)
    p_ov.add_argument(
        "--downsample", type=int, default=64, help="Downsample factor (default: 64)"
    )

    p_cr = sub.add_parser("crop", help="Full-resolution crop of a small region")
    _add_common(p_cr)
    p_cr.add_argument(
        "--row",
        type=int,
        default=None,
        help="Center row in full-res pixels (default: auto-detect)",
    )
    p_cr.add_argument(
        "--col",
        type=int,
        default=None,
        help="Center col in full-res pixels (default: auto-detect)",
    )
    p_cr.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Crop side length in pixels (default: 1024)",
    )
    p_cr.add_argument(
        "--search-downsample",
        type=int,
        default=64,
        help="Downsample for auto-detection search (default: 64)",
    )

    args = parser.parse_args()

    if args.cmd == "overview":
        out = args.out or args.mask.parent / f"{args.mask.stem}_overview.png"
        visualize_overview(args.mask, args.ome, out, downsample=args.downsample)

    elif args.cmd == "crop":
        out = args.out or args.mask.parent / f"{args.mask.stem}_crop_fullres.png"
        visualize_crop(
            args.mask,
            args.ome,
            out,
            row=args.row,
            col=args.col,
            crop_size=args.crop_size,
            search_downsample=args.search_downsample,
        )


if __name__ == "__main__":
    main()
