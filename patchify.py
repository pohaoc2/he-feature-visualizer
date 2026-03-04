#!/usr/bin/env python3
"""
patchify.py -- Stage 1 of a multi-stage histopathology analysis pipeline.

Stage 1 (this file): Extract 256x256 patches from paired OME-TIFFs (H&E and
multiplex immunofluorescence) using CLAM-style tissue detection.  Saves H&E
patches as PNG, selected multiplex channels as .npy arrays, and an index.json
manifest.

Planned downstream stages:
  Stage 2 -- Cell segmentation (e.g. CellViT) run on the H&E patches.
  Stage 3 -- Feature extraction / marker quantification per cell using the
             multiplex channel arrays.
  Stage 4 -- Spatial analysis and visualisation in the interactive viewer.

Importable API
--------------
tissue_mask_hsv      -- CLAM-style HSV tissue detection via cv2
tissue_fraction      -- Scalar tissue coverage of an RGB patch
get_patch_grid       -- Enumerate (i, j) patch coordinates
read_he_patch        -- Windowed read of an H&E zarr store -> uint8 RGB
read_multiplex_patch -- Windowed read of a multiplex zarr store -> uint16 array
load_channel_indices -- Resolve channel names from a metadata CSV

CLI
---
python patchify.py --he-image PATH --multiplex-image PATH --metadata-csv PATH
                   [--out processed/] [--patch-size 256] [--stride 256]
                   [--tissue-min 0.1] [--channels CD31 Ki67 CD45 PCNA]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import zarr


# ---------------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------------

def tissue_mask_hsv(rgb: np.ndarray, mthresh: int = 7, close: int = 4) -> np.ndarray:
    """CLAM-style tissue detection using cv2 HSV operations.

    Parameters
    ----------
    rgb:     uint8 (H, W, 3) RGB image.
    mthresh: kernel size for cv2.medianBlur (must be odd; default 7).
    close:   side length of rectangular structuring element for morphological
             closing (default 4).

    Steps
    -----
    1. Convert RGB -> HSV via cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).
    2. Extract saturation channel (index 1), ensure uint8.
    3. Apply median blur: cv2.medianBlur(sat, mthresh).
    4. Otsu threshold: cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU).
    5. Morphological closing: cv2.morphologyEx with cv2.MORPH_CLOSE,
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)).

    Returns
    -------
    bool ndarray (H, W) -- True where tissue is detected.
    """
    # Ensure mthresh is odd (cv2 requirement for medianBlur)
    if mthresh % 2 == 0:
        mthresh += 1

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.uint8)
    blurred = cv2.medianBlur(sat, mthresh)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def tissue_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels classified as tissue by tissue_mask_hsv with defaults."""
    mask = tissue_mask_hsv(rgb)
    return float(np.mean(mask))


# ---------------------------------------------------------------------------
# Patch grid
# ---------------------------------------------------------------------------

def get_patch_grid(img_w: int, img_h: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Return list of (i, j) patch indices that are fully within the image.

    Patch top-left pixel coordinates: x0 = j * stride, y0 = i * stride.
    Only patches satisfying x0 + patch_size <= img_w and
    y0 + patch_size <= img_h are included.
    """
    coords = []
    i = 0
    while True:
        y0 = i * stride
        if y0 + patch_size > img_h:
            break
        j = 0
        while True:
            x0 = j * stride
            if x0 + patch_size > img_w:
                break
            coords.append((i, j))
            j += 1
        i += 1
    return coords


# ---------------------------------------------------------------------------
# Windowed patch readers
# ---------------------------------------------------------------------------

def _open_zarr_store(tif):
    """Return a zarr Array from a TiffFile, handling Array vs Group."""
    raw = zarr.open(tif.series[0].aszarr(), mode="r")
    if isinstance(raw, zarr.Array):
        return raw
    return raw["0"]


def _clip_and_read(store, axes: str, img_w: int, img_h: int,
                   y0: int, x0: int, size: int):
    """Read a clipped region from the store and return (arr, dy, dx, rh, rw).

    dy, dx: offsets within the output patch where valid data begins (for zero-padding).
    rh, rw: height and width of the clipped read region.
    """
    # Resolve a raw ZarrTiffStore (from tif.aszarr()) to a subscriptable zarr Array.
    if not isinstance(store, zarr.Array):
        raw = zarr.open(store, mode="r")
        store = raw if isinstance(raw, zarr.Array) else raw["0"]

    y0c = max(0, int(y0))
    x0c = max(0, int(x0))
    y1c = min(img_h, y0c + size)
    x1c = min(img_w, x0c + size)

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
    dy = y0c - int(y0)
    dx = x0c - int(x0)
    rh = y1c - y0c
    rw = x1c - x0c
    return arr, dy, dx, rh, rw


def read_he_patch(zarr_store, axes: str, img_w: int, img_h: int,
                  y0: int, x0: int, size: int) -> np.ndarray:
    """Read H&E patch as uint8 RGB (size, size, 3).

    Handle axes permutations (CYX, YXC, YX, etc.).
    If store dtype != uint8: percentile normalize (p1/p99) to uint8.
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(zarr_store, axes, img_w, img_h, y0, x0, size)

    # Bring channel axis last (-> YXC) if it exists and is first
    if arr.ndim == 3:
        c_pos = axes.index("C") if "C" in axes else -1
        y_pos = axes.index("Y") if "Y" in axes else -1
        if c_pos != -1 and y_pos != -1 and c_pos < y_pos:
            arr = np.moveaxis(arr, 0, -1)

    # Normalise to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

    # Dtype normalisation
    if arr.dtype != np.uint8:
        p1 = float(np.percentile(arr, 1))
        p99 = float(np.percentile(arr, 99))
        if p99 > p1:
            arr = ((arr.astype(np.float32) - p1) / (p99 - p1) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # Zero-pad to (size, size, 3)
    if arr.shape[0] != size or arr.shape[1] != size:
        out = np.zeros((size, size, 3), dtype=np.uint8)
        out[dy: dy + rh, dx: dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_multiplex_patch(zarr_store, axes: str, img_w: int, img_h: int,
                         y0: int, x0: int, size: int,
                         channel_indices: list[int]) -> np.ndarray:
    """Read multiplex patch for specific channel indices.

    Returns (C, size, size) uint16 where C = len(channel_indices).
    Handle axes permutations (CYX, YXC, etc.).
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(zarr_store, axes, img_w, img_h, y0, x0, size)

    # Build the set of active axes (those whose slices were slice objects, not scalars)
    active_axes = [ax for ax in axes if ax in ("C", "Y", "X")]

    # Bring C to front -> (C, rh, rw)
    if "C" in active_axes:
        c_idx = active_axes.index("C")
        if c_idx != 0:
            arr = np.moveaxis(arr, c_idx, 0)
        # arr is now (all_channels, rh, rw); select the requested channels
        arr = arr[channel_indices]
    else:
        # No channel axis -- replicate single plane for each requested channel
        arr = np.stack([arr] * len(channel_indices), axis=0)

    # Cast to uint16
    arr = arr.astype(np.uint16)

    # Zero-pad to (C_sel, size, size)
    n_ch = arr.shape[0]
    if arr.shape[1] != size or arr.shape[2] != size:
        out = np.zeros((n_ch, size, size), dtype=np.uint16)
        out[:, dy: dy + rh, dx: dx + rw] = arr[:, :rh, :rw]
        return out

    return arr


# ---------------------------------------------------------------------------
# Channel metadata parsing
# ---------------------------------------------------------------------------

def load_channel_indices(metadata_csv: str, channel_names: list[str]) -> tuple[list[int], list[str]]:
    """Parse channel metadata CSV to find integer indices for requested channel names.

    CSV has columns including 'Channel ID' (format 'Channel:0:N') and 'Target Name'.
    Match channel_names against Target Name column (case-insensitive).
    Returns (indices, resolved_names) -- indices are the integer N from 'Channel:0:N'.
    Raise ValueError if any name not found.
    """
    df = pd.read_csv(metadata_csv)
    df.columns = [c.strip() for c in df.columns]

    if "Channel ID" not in df.columns:
        raise ValueError(
            f"metadata CSV must have a 'Channel ID' column. Found: {list(df.columns)}"
        )
    if "Target Name" not in df.columns:
        raise ValueError(
            f"metadata CSV must have a 'Target Name' column. Found: {list(df.columns)}"
        )

    # Build mapping: lower-case target name -> (integer index, original name)
    seen: dict[str, tuple[int, str]] = {}
    for _, row in df.iterrows():
        channel_id = str(row["Channel ID"]).strip()
        target = str(row["Target Name"]).strip()
        target_lower = target.lower()
        if target_lower in seen:
            continue  # keep first occurrence
        parts = channel_id.split(":")
        try:
            idx = int(parts[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Cannot parse integer index from Channel ID '{channel_id}'"
            ) from exc
        seen[target_lower] = (idx, target)

    indices: list[int] = []
    resolved: list[str] = []
    missing: list[str] = []
    for name in channel_names:
        key = name.lower()
        if key not in seen:
            missing.append(name)
        else:
            idx, orig = seen[key]
            indices.append(idx)
            resolved.append(orig)

    if missing:
        available = sorted(seen.keys())
        raise ValueError(
            f"Channel(s) not found in metadata CSV: {missing}\n"
            f"Available target names: {available}"
        )

    return indices, resolved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _get_image_dims(tif):
    """Return (img_w, img_h, axes_upper) from an open TiffFile."""
    series = tif.series[0]
    axes = series.axes.upper()
    shape = series.shape
    img_h = shape[axes.index("Y")]
    img_w = shape[axes.index("X")]
    return img_w, img_h, axes


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 -- Extract H&E and multiplex patches from OME-TIFFs "
            "with CLAM-style tissue detection."
        )
    )
    parser.add_argument("--he-image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--multiplex-image", required=True, help="Path to multiplex OME-TIFF")
    parser.add_argument(
        "--metadata-csv", required=True,
        help="Channel metadata CSV with 'Channel ID' and 'Target Name' columns",
    )
    parser.add_argument("--out", default="processed", help="Output directory (default: processed/)")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size in pixels (default: 256)")
    parser.add_argument("--stride", type=int, default=256, help="Patch stride (default: 256, no overlap)")
    parser.add_argument(
        "--tissue-min", type=float, default=0.1,
        help="Minimum tissue fraction to keep a patch (default: 0.1)",
    )
    parser.add_argument(
        "--channels", nargs="+", default=["CD31", "Ki67", "CD45", "PCNA"],
        metavar="NAME",
        help="Multiplex channel Target Names to extract (default: CD31 Ki67 CD45 PCNA)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    he_dir = out_dir / "he"
    mx_dir = out_dir / "multiplex"
    he_dir.mkdir(parents=True, exist_ok=True)
    mx_dir.mkdir(parents=True, exist_ok=True)

    patch_size = args.patch_size
    stride = args.stride

    # --- Resolve channel indices from metadata CSV ---
    print("Resolving channel indices from metadata CSV ...")
    channel_indices, resolved_names = load_channel_indices(args.metadata_csv, args.channels)
    print(f"  Channels: {list(zip(resolved_names, channel_indices))}")

    # --- Open H&E image ---
    print("Opening H&E image (windowed) ...")
    he_tif = tifffile.TiffFile(args.he_image)
    he_w, he_h, he_axes = _get_image_dims(he_tif)
    he_store = _open_zarr_store(he_tif)
    print(f"  H&E: {he_w} x {he_h}, axes={he_axes}, dtype={he_store.dtype}")

    # --- Open multiplex image ---
    print("Opening multiplex image (windowed) ...")
    mx_tif = tifffile.TiffFile(args.multiplex_image)
    mx_w, mx_h, mx_axes = _get_image_dims(mx_tif)
    mx_store = _open_zarr_store(mx_tif)
    print(f"  Multiplex: {mx_w} x {mx_h}, axes={mx_axes}, dtype={mx_store.dtype}")

    # Use H&E dimensions for the patch grid
    grid = get_patch_grid(he_w, he_h, patch_size, stride)
    total = len(grid)
    n_rows = (he_h - patch_size) // stride + 1 if he_h >= patch_size else 0
    n_cols = (he_w - patch_size) // stride + 1 if he_w >= patch_size else 0
    print(f"Patch grid: stride={stride}, patch_size={patch_size} -> {n_rows}x{n_cols} = {total} candidates")

    index: list[dict] = []
    kept = 0
    last_row = -1

    for _idx_g, (i, j) in enumerate(grid):
        y0 = i * stride
        x0 = j * stride

        # Progress: print at start of each new block of 50 rows
        if i != last_row and i % 50 == 0:
            print(f"  Row {i} ... kept {kept} so far")
            last_row = i

        # Read H&E patch and apply tissue filter
        he_patch = read_he_patch(he_store, he_axes, he_w, he_h, y0, x0, patch_size)
        frac = tissue_fraction(he_patch)
        if frac < args.tissue_min:
            continue

        # Read matching multiplex patch
        mx_patch = read_multiplex_patch(
            mx_store, mx_axes, mx_w, mx_h,
            y0, x0, patch_size, channel_indices,
        )

        # Save H&E as PNG
        Image.fromarray(he_patch).save(he_dir / f"{i}_{j}.png")

        # Save multiplex as .npy (C, H, W) uint16
        np.save(mx_dir / f"{i}_{j}.npy", mx_patch)

        index.append({
            "i": i, "j": j,
            "x0": x0, "y0": y0,
            "x1": x0 + patch_size,
            "y1": y0 + patch_size,
        })
        kept += 1

    print(f"  Done. Kept {kept}/{total} patches.")

    # --- Write index.json ---
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "patches": index,
                "stride": stride,
                "patch_size": patch_size,
                "tissue_min": args.tissue_min,
                "img_w": he_w,
                "img_h": he_h,
                "channels": resolved_names,
            },
            f,
            indent=2,
        )

    print(f"Index written to {index_path}")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
