#!/usr/bin/env python3
"""
patchify.py -- Stage 1: Extract 256x256 patches from paired OME-TIFFs.

Extracts H&E patches as PNG, selected multiplex channels as .npy arrays,
and an index.json manifest.  H&E and multiplex images are assumed to be
pre-aligned; coordinate mapping uses the MPP ratio only (no affine warp).

Downstream stages
-----------------
  Stage 2 -- Cell segmentation (CellViT) on H&E patches.
  Stage 3 -- Multiplex marker quantification per cell.
  Stage 4 -- Spatial analysis and visualisation.

Importable API
--------------
tissue_mask_hsv      -- CLAM-style HSV tissue detection
tissue_fraction      -- Scalar tissue coverage of an RGB patch
build_tissue_mask    -- Build overview tissue mask from zarr store
get_patch_grid       -- Enumerate (i, j) patch coordinates
get_tissue_patches   -- Filter patches by tissue coverage
read_he_patch        -- Windowed read of H&E zarr store -> uint8 RGB
read_mask_patch      -- Windowed read of segmentation mask -> uint32
read_multiplex_patch -- Windowed read of multiplex zarr store -> uint16
load_channel_indices -- Resolve channel names from a metadata CSV

CLI
---
python patchify.py --he-image PATH --multiplex-image PATH --metadata-csv PATH
                   [--out processed/] [--patch-size 256] [--stride 256]
                   [--tissue-min 0.1] [--channels CD31 Ki67 ...]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import zarr

from utils.channels import resolve_channel_indices
from utils.normalize import percentile_to_uint8
from utils.ome import (
    get_image_dims,
    get_ome_mpp,
    open_zarr_store,
    read_overview_chw,
)
from stages.patchify_lib import masking as _masking
from stages.patchify_lib import readers as _readers

# ---------------------------------------------------------------------------
# Public API -- re-exported from sub-modules
# ---------------------------------------------------------------------------

tissue_mask_hsv = _masking.tissue_mask_hsv
tissue_fraction = _masking.tissue_fraction
build_tissue_mask = _masking.build_tissue_mask

get_tissue_patches = _readers.get_tissue_patches
get_patch_grid = _readers.get_patch_grid
_clip_and_read = _readers._clip_and_read
read_he_patch = _readers.read_he_patch
read_mask_patch = _readers.read_mask_patch
read_multiplex_patch = _readers.read_multiplex_patch

# Backward-compatible alias
load_channel_indices = resolve_channel_indices


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_PY_CTX: dict = {}


def _init_py_worker(
    he_path: str, mx_path: str, seg_path: str | None, ctx_rest: dict
) -> None:
    """Open zarr stores once per worker process, store in module global."""
    global _PY_CTX
    import tifffile as _tifffile
    from utils.ome import open_zarr_store as _open_zarr_store

    he_tif = _tifffile.TiffFile(he_path)
    mx_tif = _tifffile.TiffFile(mx_path)
    _PY_CTX = dict(ctx_rest)
    _PY_CTX["he_store"] = _open_zarr_store(he_tif)
    _PY_CTX["mx_store"] = _open_zarr_store(mx_tif)
    if seg_path is not None:
        seg_tif = _tifffile.TiffFile(seg_path)
        _PY_CTX["seg_store"] = _open_zarr_store(seg_tif)
    else:
        _PY_CTX["seg_store"] = None


def _py_patch_worker(task: tuple) -> tuple:
    """Process one (idx, x0, y0) patch; return (idx, entry)."""
    idx, x0, y0 = task
    ctx = _PY_CTX
    he_store = ctx["he_store"]
    mx_store = ctx["mx_store"]
    seg_store = ctx["seg_store"]
    he_axes, he_w, he_h = ctx["he_axes"], ctx["he_w"], ctx["he_h"]
    mx_axes, mx_w, mx_h = ctx["mx_axes"], ctx["mx_w"], ctx["mx_h"]
    seg_axes, seg_w, seg_h = ctx["seg_axes"], ctx["seg_w"], ctx["seg_h"]
    mpp_scale: float = ctx["mpp_scale"]
    channel_indices: list[int] = ctx["channel_indices"]
    patch_size: int = ctx["patch_size"]
    min_mx_overlap: float = ctx["min_mx_overlap"]
    out_dir: Path = ctx["out_dir"]

    # H&E patch
    he_patch = read_he_patch(he_store, he_axes, he_w, he_h, y0, x0, patch_size)
    Image.fromarray(he_patch).save(out_dir / "he" / f"{x0}_{y0}.png")

    # Optional segmentation mask (already in H&E pixel space)
    has_seg = False
    if seg_store is not None:
        seg_patch = read_mask_patch(
            seg_store, seg_axes, seg_w, seg_h, y0, x0, patch_size
        )
        np.save(out_dir / "masks" / f"{x0}_{y0}.npy", seg_patch)
        has_seg = True

    # Multiplex patch -- map H&E coordinates to MX space via MPP ratio
    mx_x0 = int(round(x0 * mpp_scale))
    mx_y0 = int(round(y0 * mpp_scale))
    mx_size = int(round(patch_size * mpp_scale))
    mx_size = max(1, mx_size)

    # Compute overlap fraction (fraction of MX patch that falls inside MX bounds)
    x_end = mx_x0 + mx_size
    y_end = mx_y0 + mx_size
    if mx_x0 < 0 or mx_y0 < 0 or x_end > mx_w or y_end > mx_h:
        # Clamp and compute fractional coverage
        x_valid = max(0, min(x_end, mx_w) - max(0, mx_x0))
        y_valid = max(0, min(y_end, mx_h) - max(0, mx_y0))
        mx_overlap = float(x_valid * y_valid) / float(mx_size * mx_size + 1e-9)
    else:
        mx_overlap = 1.0

    has_mx = mx_overlap >= min_mx_overlap
    if has_mx:
        mx_patch = read_multiplex_patch(
            mx_store,
            mx_axes,
            mx_w,
            mx_h,
            y0=mx_y0,
            x0=mx_x0,
            size_y=mx_size,
            size_x=mx_size,
            channel_indices=channel_indices,
        )
        np.save(out_dir / "multiplex" / f"{x0}_{y0}.npy", mx_patch)

    entry: dict = {
        "x0": x0,
        "y0": y0,
        "has_multiplex": has_mx,
        "multiplex_overlap_fraction": float(mx_overlap),
    }
    if seg_store is not None:
        entry["has_mask"] = has_seg
    return idx, entry


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 -- Extract H&E and multiplex patches from OME-TIFFs."
    )
    parser.add_argument("--he-image", required=True)
    parser.add_argument("--multiplex-image", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument(
        "--he-source-image",
        default=None,
        help="Optional full-slide H&E source path for crop metadata.",
    )
    parser.add_argument(
        "--multiplex-source-image",
        default=None,
        help="Optional full-slide multiplex source path for crop metadata.",
    )
    parser.add_argument(
        "--he-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Optional top-left (x y) of H&E crop in full-slide H&E px.",
    )
    parser.add_argument(
        "--mx-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Optional top-left (x y) of multiplex crop in full-slide MX px.",
    )
    parser.add_argument("--out", default="processed")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tissue-min", type=float, default=0.1)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[
            "Hoechst",
            "AF1",
            "CD31",
            "CD45",
            "CD68",
            "Argo550",
            "CD4",
            "FOXP3",
            "CD8a",
            "CD45RO",
            "CD20",
            "PD-L1",
            "CD3e",
            "CD163",
            "E-cadherin",
            "PD-1",
            "Ki67",
            "Pan-CK",
            "SMA",
        ],
        metavar="NAME",
        help="Multiplex channel names to extract.",
    )
    parser.add_argument(
        "--overview-downsample",
        type=int,
        default=64,
        help="Stride for H&E overview sampling (default 64).",
    )
    parser.add_argument(
        "--mask-image",
        default=None,
        help="Optional cell segmentation mask OME-TIFF in H&E pixel space.",
    )
    parser.add_argument(
        "--min-multiplex-overlap",
        type=float,
        default=1.0,
        help="Minimum MX overlap fraction to save multiplex patch (0-1, default: 1.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for patch extraction (default: 4).",
    )
    args = parser.parse_args()
    if not (0.0 <= args.min_multiplex_overlap <= 1.0):
        parser.error("--min-multiplex-overlap must be between 0.0 and 1.0.")

    out_dir = Path(args.out)
    ds = args.overview_downsample
    patch_size = args.patch_size
    min_mx_overlap = float(args.min_multiplex_overlap)
    (out_dir / "he").mkdir(parents=True, exist_ok=True)
    (out_dir / "multiplex").mkdir(parents=True, exist_ok=True)
    if args.mask_image:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print("Resolving channel indices ...")
    channel_indices, resolved_names = load_channel_indices(
        args.metadata_csv, args.channels
    )

    print("Opening H&E image ...")
    he_tif = tifffile.TiffFile(args.he_image)
    mx_tif = None
    seg_tif = None
    try:
        he_w, he_h, he_axes = get_image_dims(he_tif)
        he_store = open_zarr_store(he_tif)
        he_mpp_x, _ = get_ome_mpp(he_tif)
        print(f"  {he_w} x {he_h}  axes={he_axes}  mpp={he_mpp_x}")

        print("Opening multiplex image ...")
        mx_tif = tifffile.TiffFile(args.multiplex_image)
        mx_w, mx_h, mx_axes = get_image_dims(mx_tif)
        mx_store = open_zarr_store(mx_tif)
        mx_mpp_x, _ = get_ome_mpp(mx_tif)
        print(f"  {mx_w} x {mx_h}  axes={mx_axes}  mpp={mx_mpp_x}")

        # MPP scale: factor to map H&E full-res coords -> MX full-res coords
        if he_mpp_x and mx_mpp_x and mx_mpp_x > 0:
            mpp_scale = he_mpp_x / mx_mpp_x
        else:
            mpp_scale = mx_w / he_w if he_w > 0 else 1.0
        print(f"  MPP scale H&E -> MX: {mpp_scale:.4f}")

        print(f"Building tissue mask (downsample={ds}) ...")
        mask = build_tissue_mask(he_store, he_axes, he_w, he_h, downsample=ds)
        print(f"  Tissue fraction: {mask.mean():.2%}")

        seg_store = seg_axes = seg_w = seg_h = None
        if args.mask_image:
            print("Opening cell mask image ...")
            seg_tif = tifffile.TiffFile(args.mask_image)
            seg_w, seg_h, seg_axes = get_image_dims(seg_tif)
            seg_store = open_zarr_store(seg_tif)
            print(f"  {seg_w} x {seg_h}  axes={seg_axes}")

        print("Selecting tissue patches ...")
        coords = get_tissue_patches(
            mask, he_w, he_h, patch_size, args.stride, args.tissue_min, ds
        )
        print(f"  {len(coords)} patches selected")

        print(f"Extracting patches (workers={args.workers}) ...")
        print(f"  Multiplex save threshold: overlap >= {min_mx_overlap:.3f}")

        ctx_rest = {
            "he_axes": he_axes,
            "he_w": he_w,
            "he_h": he_h,
            "mx_axes": mx_axes,
            "mx_w": mx_w,
            "mx_h": mx_h,
            "seg_axes": seg_axes,
            "seg_w": seg_w,
            "seg_h": seg_h,
            "mpp_scale": mpp_scale,
            "channel_indices": channel_indices,
            "patch_size": patch_size,
            "min_mx_overlap": min_mx_overlap,
            "out_dir": out_dir,
        }

        n_total = len(coords)
        results: list[tuple | None] = [None] * n_total
        done = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_py_worker,
            initargs=(
                str(args.he_image),
                str(args.multiplex_image),
                str(args.mask_image) if args.mask_image else None,
                ctx_rest,
            ),
        ) as executor:
            futures = {
                executor.submit(_py_patch_worker, (idx, x0, y0)): idx
                for idx, (x0, y0) in enumerate(coords)
            }
            for future in as_completed(futures):
                orig_idx, entry = future.result()
                results[orig_idx] = entry
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{n_total} ...")

        index = list(results)

        n_mx = sum(p["has_multiplex"] for p in index)
        print(f"  Done. {n_mx}/{len(index)} patches have multiplex.")
        if seg_store is not None:
            n_seg = sum(p.get("has_mask", False) for p in index)
            print(f"        {n_seg}/{len(index)} patches have cell mask.")

        # Build H&E overview for visualisation
        he_chw = read_overview_chw(he_store, he_axes, he_h, he_w, ds)
        he_chw = (
            he_chw[:3] if he_chw.shape[0] >= 3 else np.repeat(he_chw[:1], 3, axis=0)
        )
        if he_chw.dtype != np.uint8:
            he_chw = percentile_to_uint8(he_chw)

        meta: dict = {
            "patches": index,
            "patch_size": patch_size,
            "stride": args.stride,
            "tissue_min": args.tissue_min,
            "img_w": he_w,
            "img_h": he_h,
            "he_mpp": he_mpp_x,
            "mx_mpp": mx_mpp_x,
            "mpp_scale": mpp_scale,
            "channels": resolved_names,
            "channel_indices": [int(i) for i in channel_indices],
        }
        if args.mask_image:
            meta["mask_image"] = str(args.mask_image)
        if args.he_crop_origin is not None:
            meta["he_crop_origin"] = [
                float(args.he_crop_origin[0]),
                float(args.he_crop_origin[1]),
            ]
        if args.mx_crop_origin is not None:
            meta["mx_crop_origin"] = [
                float(args.mx_crop_origin[0]),
                float(args.mx_crop_origin[1]),
            ]

        if (
            args.he_source_image
            or args.multiplex_source_image
            or args.he_crop_origin is not None
            or args.mx_crop_origin is not None
        ):
            crop_region: dict = {
                "he_size": [int(he_w), int(he_h)],
                "mx_size": [int(mx_w), int(mx_h)],
            }
            if args.he_source_image:
                crop_region["he_source_image"] = str(args.he_source_image)
            if args.multiplex_source_image:
                crop_region["multiplex_source_image"] = str(args.multiplex_source_image)
            if args.he_crop_origin is not None:
                crop_region["he_origin"] = [
                    float(args.he_crop_origin[0]),
                    float(args.he_crop_origin[1]),
                ]
            if args.mx_crop_origin is not None:
                crop_region["mx_origin"] = [
                    float(args.mx_crop_origin[0]),
                    float(args.mx_crop_origin[1]),
                ]
            meta["crop_region"] = crop_region

        with open(out_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Index written to {out_dir / 'index.json'}")
        print("Stage 1 complete.")

    finally:
        he_tif.close()
        if mx_tif is not None:
            mx_tif.close()
        if seg_tif is not None:
            seg_tif.close()


if __name__ == "__main__":
    main()
