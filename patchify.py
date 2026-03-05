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
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import zarr


# ---------------------------------------------------------------------------
# OME physical pixel size (mpp)
# ---------------------------------------------------------------------------

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _safe_float(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def get_ome_mpp(tif) -> tuple[float | None, float | None]:
    """Return (mpp_x, mpp_y) in µm/pixel from OME-XML, or (None, None) if missing."""
    ome_xml = getattr(tif, "ome_metadata", None)
    if not ome_xml and hasattr(tif, "pages") and len(tif.pages) > 0:
        ome_xml = getattr(tif.pages[0], "description", None)
    if not ome_xml:
        return None, None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None, None
    pixels = root.find(".//ome:Pixels", OME_NS)
    if pixels is None:
        return None, None
    psx = _safe_float(pixels.get("PhysicalSizeX"))
    psy = _safe_float(pixels.get("PhysicalSizeY"))
    return psx, psy


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


def build_tissue_mask(
    store, axes: str, img_w: int, img_h: int, downsample: int = 64
) -> np.ndarray:
    """Build a boolean tissue mask from a downsampled H&E overview.

    Downloads only ~(img_h/downsample * img_w/downsample * 3) bytes.

    Parameters
    ----------
    store:      zarr Array opened from tifffile series.
    axes:       Axes string (e.g. 'CYX' or 'YXC').
    img_w/h:    Full-resolution image dimensions.
    downsample: Stride for overview sampling (default 64).

    Returns
    -------
    bool ndarray of shape exactly (img_h // downsample, img_w // downsample).
    """
    axes = axes.upper()
    if "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"axes must contain both 'Y' and 'X'; got {axes!r}"
        )
    c_first = "C" in axes and axes.index("C") < axes.index("Y")

    # Truncate dimensions to exact multiples of downsample so that
    # ceil(trunc / downsample) == trunc // downsample == img_h // downsample.
    img_h_trunc = (img_h // downsample) * downsample
    img_w_trunc = (img_w // downsample) * downsample

    if c_first:
        raw = np.array(store[:, :img_h_trunc:downsample, :img_w_trunc:downsample])  # (C, H//ds, W//ds)
        overview = np.moveaxis(raw, 0, -1)                                            # (H//ds, W//ds, C)
    else:
        overview = np.array(store[:img_h_trunc:downsample, :img_w_trunc:downsample, :])

    if overview.shape[-1] > 3:
        overview = overview[..., :3]
    if overview.dtype != np.uint8:
        p1 = float(np.percentile(overview, 1))
        p99 = float(np.percentile(overview, 99))
        if p99 > p1:
            overview = ((overview.astype(np.float32) - p1) / (p99 - p1) * 255).clip(0, 255).astype(np.uint8)
        else:
            overview = np.zeros_like(overview, dtype=np.uint8)

    return tissue_mask_hsv(overview)


def get_tissue_patches(
    mask: np.ndarray,
    img_w: int, img_h: int,
    patch_size: int, stride: int,
    tissue_min: float,
    downsample: int,
) -> list[tuple[int, int]]:
    """Return list of (x0, y0) level-0 patch coords that meet tissue threshold.

    Only patches satisfying x0+patch_size <= img_w and y0+patch_size <= img_h
    are considered (no padding).
    """
    kept = []
    y0 = 0
    while y0 + patch_size <= img_h:
        x0 = 0
        while x0 + patch_size <= img_w:
            my0 = y0 // downsample
            mx0 = x0 // downsample
            my1 = max(my0 + 1, (y0 + patch_size) // downsample)
            mx1 = max(mx0 + 1, (x0 + patch_size) // downsample)
            my1 = min(my1, mask.shape[0])
            mx1 = min(mx1, mask.shape[1])
            region = mask[my0:my1, mx0:mx1]
            if region.size > 0 and float(region.mean()) >= tissue_min:
                kept.append((x0, y0))
            x0 += stride
        y0 += stride
    return kept


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
                   y0: int, x0: int, size_y: int, size_x: int):
    """Read a clipped region from the store and return (arr, dy, dx, rh, rw).

    dy, dx: offsets within the output patch where valid data begins (for zero-padding).
    rh, rw: height and width of the clipped read region.
    """
    # Resolve a raw ZarrTiffStore (from tif.aszarr()) to a subscriptable zarr Array.
    if not isinstance(store, zarr.Array):
        raw = zarr.open(store, mode="r")
        store = raw if isinstance(raw, zarr.Array) else raw["0"]

    y0c = max(0, min(int(y0), img_h))
    x0c = max(0, min(int(x0), img_w))
    y1c = min(img_h, y0c + size_y)
    x1c = min(img_w, x0c + size_x)

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
    arr, dy, dx, rh, rw = _clip_and_read(zarr_store, axes, img_w, img_h, y0, x0, size, size)

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
        if rh > 0 and rw > 0:
            out[dy: dy + rh, dx: dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_multiplex_patch(zarr_store, axes: str, img_w: int, img_h: int,
                         y0: int, x0: int, size_y: int, size_x: int,
                         channel_indices: list[int]) -> np.ndarray:
    """Read multiplex patch for specific channel indices.

    Returns (C, size_y, size_x) uint16 where C = len(channel_indices).
    Handle axes permutations (CYX, YXC, etc.).
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(zarr_store, axes, img_w, img_h, y0, x0, size_y, size_x)

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

    # Zero-pad to (C_sel, size_y, size_x)
    n_ch = arr.shape[0]
    if arr.shape[1] != size_y or arr.shape[2] != size_x:
        out = np.zeros((n_ch, size_y, size_x), dtype=np.uint16)
        if rh > 0 and rw > 0:
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


def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale a 2D array to uint8 using p1/p99 percentile normalization."""
    p1  = float(np.percentile(arr, 1))
    p99 = float(np.percentile(arr, 99))
    if p99 > p1:
        return ((arr.astype(np.float32) - p1) / (p99 - p1) * 255).clip(0, 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def visualize(
    he_overview: np.ndarray,
    mx_overview: np.ndarray,
    he_coords: list[tuple[int, int]],
    mx_coords: list[tuple[int, int]],
    vis_channels: list[int],
    out_dir: Path,
    patch_size_ov: int = 4,
    gap: int = 10,
) -> None:
    """Save side-by-side vis_patches.jpg to out_dir.

    Parameters
    ----------
    he_overview:    (H_ov, W_ov, 3) uint8 H&E overview.
    mx_overview:    (C, H_mx_ov, W_mx_ov) uint16 multiplex overview.
    he_coords:      (x0, y0) in overview pixel coords for H&E panel.
    mx_coords:      Corresponding coords in multiplex overview pixels.
    vis_channels:   3 channel indices for multiplex RGB composite.
    patch_size_ov:  Box size in overview pixels (default 4).
    gap:            White gap between panels in pixels (default 10).
    """
    import PIL.ImageDraw

    he_panel = Image.fromarray(he_overview.astype(np.uint8), "RGB").copy()
    draw_he  = PIL.ImageDraw.Draw(he_panel)
    for x0, y0 in he_coords:
        draw_he.rectangle([x0, y0, x0 + patch_size_ov - 1, y0 + patch_size_ov - 1],
                          outline=(0, 200, 0), width=1)

    c0 = _norm_to_uint8(mx_overview[vis_channels[0]])
    c1 = _norm_to_uint8(mx_overview[vis_channels[1]])
    c2 = _norm_to_uint8(mx_overview[vis_channels[2]])
    mx_panel = Image.fromarray(np.stack([c0, c1, c2], axis=-1), "RGB").copy()
    draw_mx  = PIL.ImageDraw.Draw(mx_panel)
    for x0, y0 in mx_coords:
        draw_mx.rectangle([x0, y0, x0 + patch_size_ov - 1, y0 + patch_size_ov - 1],
                          outline=(0, 200, 0), width=1)

    h   = max(he_panel.height, mx_panel.height)
    w   = he_panel.width + gap + mx_panel.width
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(he_panel, (0, 0))
    out.paste(mx_panel, (he_panel.width + gap, 0))
    out.save(str(out_dir / "vis_patches.jpg"), quality=90)
    print(f"Saved vis_patches.jpg ({w}x{h} px, {len(he_coords)} patches shown)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 -- Extract H&E and multiplex patches from OME-TIFFs."
    )
    parser.add_argument("--he-image",           required=True)
    parser.add_argument("--multiplex-image",     required=True)
    parser.add_argument("--metadata-csv",        required=True)
    parser.add_argument("--out",                 default="processed")
    parser.add_argument("--patch-size",          type=int,   default=256)
    parser.add_argument("--stride",              type=int,   default=256)
    parser.add_argument("--tissue-min",          type=float, default=0.1)
    parser.add_argument("--channels",            nargs="+",  default=["CD31", "Ki67", "CD45", "PCNA"])
    parser.add_argument("--overview-downsample", type=int,   default=64,
                        help="Stride for H&E overview sampling (default 64)")
    parser.add_argument("--vis-channels",        type=int,   nargs=3, default=[0, 10, 20],
                        help="3 multiplex channel indices for RGB composite in vis")
    args = parser.parse_args()

    out_dir    = Path(args.out)
    ds         = args.overview_downsample
    patch_size = args.patch_size
    (out_dir / "he").mkdir(parents=True, exist_ok=True)
    (out_dir / "multiplex").mkdir(parents=True, exist_ok=True)

    print("Resolving channel indices ...")
    channel_indices, resolved_names = load_channel_indices(args.metadata_csv, args.channels)

    print("Opening H&E image ...")
    he_tif   = tifffile.TiffFile(args.he_image)
    he_w, he_h, he_axes = _get_image_dims(he_tif)
    he_store = _open_zarr_store(he_tif)
    he_mpp_x, _ = get_ome_mpp(he_tif)
    print(f"  {he_w} x {he_h}  axes={he_axes}  mpp={he_mpp_x}")

    print("Opening multiplex image ...")
    mx_tif   = tifffile.TiffFile(args.multiplex_image)
    mx_w, mx_h, mx_axes = _get_image_dims(mx_tif)
    mx_store = _open_zarr_store(mx_tif)
    mx_mpp_x, _ = get_ome_mpp(mx_tif)
    print(f"  {mx_w} x {mx_h}  axes={mx_axes}  mpp={mx_mpp_x}")

    scale = (he_mpp_x / mx_mpp_x) if (he_mpp_x and mx_mpp_x) else (he_w / mx_w)
    print(f"  scale H&E -> multiplex: {scale:.4f}")

    print(f"Building tissue mask (downsample={ds}) ...")
    mask = build_tissue_mask(he_store, he_axes, he_w, he_h, downsample=ds)
    print(f"  Tissue fraction: {mask.mean():.2%}")

    # Keep overview for visualization (already computed inside build_tissue_mask,
    # but we re-read it here since build_tissue_mask doesn't return it)
    he_axes_up = he_axes.upper()
    c_first    = "C" in he_axes_up and he_axes_up.index("C") < he_axes_up.index("Y")
    img_h_trunc = (he_h // ds) * ds
    img_w_trunc = (he_w // ds) * ds
    if c_first:
        he_overview = np.moveaxis(
            np.array(he_store[:, :img_h_trunc:ds, :img_w_trunc:ds]), 0, -1
        ).astype(np.uint8)
    else:
        he_overview = np.array(he_store[:img_h_trunc:ds, :img_w_trunc:ds, :]).astype(np.uint8)
    if he_overview.shape[-1] > 3:
        he_overview = he_overview[..., :3]

    print("Selecting tissue patches ...")
    coords = get_tissue_patches(mask, he_w, he_h, patch_size, args.stride, args.tissue_min, ds)
    print(f"  {len(coords)} patches selected")

    print("Extracting patches ...")
    index: list[dict]            = []
    he_vis_coords: list[tuple[int, int]] = []
    mx_vis_coords: list[tuple[int, int]] = []

    for idx, (x0, y0) in enumerate(coords):
        if idx % 500 == 0:
            print(f"  {idx}/{len(coords)} ...")

        he_patch = read_he_patch(he_store, he_axes, he_w, he_h, y0, x0, patch_size)
        Image.fromarray(he_patch).save(out_dir / "he" / f"{x0}_{y0}.png")

        x0_mx   = round(x0 * scale)
        y0_mx   = round(y0 * scale)
        size_mx = max(1, round(patch_size * scale))
        has_mx  = (x0_mx + size_mx <= mx_w) and (y0_mx + size_mx <= mx_h)

        if has_mx:
            mx_patch = read_multiplex_patch(
                mx_store, mx_axes, mx_w, mx_h,
                y0_mx, x0_mx, size_mx, size_mx, channel_indices,
            )
            if mx_patch.shape[1] != patch_size or mx_patch.shape[2] != patch_size:
                resized = np.zeros((mx_patch.shape[0], patch_size, patch_size), dtype=mx_patch.dtype)
                for c in range(mx_patch.shape[0]):
                    resized[c] = cv2.resize(mx_patch[c], (patch_size, patch_size),
                                            interpolation=cv2.INTER_LINEAR)
                mx_patch = resized
            np.save(out_dir / "multiplex" / f"{x0}_{y0}.npy", mx_patch)
            mx_vis_coords.append((x0_mx // ds, y0_mx // ds))

        he_vis_coords.append((x0 // ds, y0 // ds))
        index.append({"x0": x0, "y0": y0, "has_multiplex": has_mx})

    n_mx = sum(p["has_multiplex"] for p in index)
    print(f"  Done. {n_mx}/{len(index)} patches have multiplex.")

    print("Generating vis_patches.jpg ...")
    mx_axes_up = mx_axes.upper()
    mx_c_first = "C" in mx_axes_up and mx_axes_up.index("C") < mx_axes_up.index("Y")
    mx_h_trunc = (mx_h // ds) * ds
    mx_w_trunc = (mx_w // ds) * ds
    if mx_c_first:
        mx_overview = np.array(mx_store[:, :mx_h_trunc:ds, :mx_w_trunc:ds])
    else:
        mx_overview = np.moveaxis(np.array(mx_store[:mx_h_trunc:ds, :mx_w_trunc:ds, :]), -1, 0)
    vis_ch = [min(c, mx_overview.shape[0] - 1) for c in args.vis_channels]
    visualize(he_overview, mx_overview, he_vis_coords, mx_vis_coords, vis_ch, out_dir)

    with open(out_dir / "index.json", "w") as f:
        json.dump({
            "patches": index,
            "patch_size": patch_size,
            "stride": args.stride,
            "tissue_min": args.tissue_min,
            "img_w": he_w, "img_h": he_h,
            "he_mpp": he_mpp_x, "mx_mpp": mx_mpp_x,
            "scale_he_to_mx": scale,
            "channels": resolved_names,
        }, f, indent=2)

    print(f"Index written to {out_dir / 'index.json'}")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
