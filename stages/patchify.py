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
        raise ValueError(f"axes must contain both 'Y' and 'X'; got {axes!r}")
    c_first = "C" in axes and axes.index("C") < axes.index("Y")

    # Truncate dimensions to exact multiples of downsample so that
    # ceil(trunc / downsample) == trunc // downsample == img_h // downsample.
    img_h_trunc = (img_h // downsample) * downsample
    img_w_trunc = (img_w // downsample) * downsample

    if c_first:
        raw = np.array(
            store[:, :img_h_trunc:downsample, :img_w_trunc:downsample]
        )  # (C, H//ds, W//ds)
        overview = np.moveaxis(raw, 0, -1)  # (H//ds, W//ds, C)
    else:
        overview = np.array(store[:img_h_trunc:downsample, :img_w_trunc:downsample, :])

    if overview.shape[-1] > 3:
        overview = overview[..., :3]
    if overview.dtype != np.uint8:
        overview = percentile_to_uint8(overview)

    return tissue_mask_hsv(overview)


def build_mx_tissue_mask(store, axes: str, mx_h: int, mx_w: int, ds: int) -> np.ndarray:
    """Build a binary tissue mask from the MX DNA channel (ch0) at overview resolution.

    Returns bool ndarray (mx_h // ds, mx_w // ds).
    """
    chw = read_overview_chw(store, axes, mx_h, mx_w, ds)  # (C, H, W)
    dna_u8 = percentile_to_uint8(chw[0])
    if dna_u8.max() == 0:
        return np.zeros((mx_h // ds, mx_w // ds), dtype=bool)
    _, binary = cv2.threshold(dna_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def register_he_mx_affine(  # pylint: disable=unused-argument
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Compute affine warp M_full (2×3, float32) mapping H&E full-res → MX full-res.

    Uses ECC maximisation on binary tissue masks.  Falls back to mpp-scale
    identity if ECC fails.
    """
    he_ov_h, he_ov_w = he_mask.shape
    mx_ov_h, mx_ov_w = mx_mask.shape

    # Resize MX mask to H&E overview size (float32 [0,1])
    mx_resized = cv2.resize(
        mx_mask.astype(np.float32), (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    )
    he_f32 = he_mask.astype(np.float32)

    # Gaussian blur for smoother ECC convergence
    he_f32 = cv2.GaussianBlur(he_f32, (5, 5), 0)
    mx_resized = cv2.GaussianBlur(mx_resized, (5, 5), 0)

    # ECC: find m_ov that warps mx_resized to align with he_f32
    m_ov = np.eye(
        2, 3, dtype=np.float32
    )  # start at identity (images already same size)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, m_ov = cv2.findTransformECC(
            he_f32, mx_resized, m_ov, cv2.MOTION_AFFINE, criteria
        )
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: ECC registration failed ({e}). Falling back to mpp scale.")
        scale = he_w / mx_w  # approx 2.0; equivalent to he_mpp/mx_mpp
        return np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32)

    # Convert m_ov (overview space) → m_full (H&E full-res → MX full-res)
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    m_full = np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, m_ov[0, 2] * ds / rx],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, m_ov[1, 2] * ds / ry],
        ],
        dtype=np.float32,
    )
    return m_full


def transform_he_to_mx_point(m_full: np.ndarray, x0: int, y0: int) -> tuple[int, int]:
    """Apply the 2×3 affine m_full to (x0, y0) and return rounded (x_mx, y_mx)."""
    pt = np.array([x0, y0, 1.0], dtype=np.float64)
    result = m_full.astype(np.float64) @ pt
    return int(round(result[0])), int(round(result[1]))


def get_tissue_patches(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    patch_size: int,
    stride: int,
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


def get_patch_grid(
    img_w: int, img_h: int, patch_size: int, stride: int
) -> list[tuple[int, int]]:
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


def _clip_and_read(
    store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size_y: int, size_x: int
):
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


def read_he_patch(
    zarr_store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size: int
) -> np.ndarray:
    """Read H&E patch as uint8 RGB (size, size, 3).

    Handle axes permutations (CYX, YXC, YX, etc.).
    If store dtype != uint8: percentile normalize (p1/p99) to uint8.
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

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

    if arr.dtype != np.uint8:
        arr = percentile_to_uint8(arr)
    else:
        arr = arr.astype(np.uint8)

    # Zero-pad to (size, size, 3)
    if arr.shape[0] != size or arr.shape[1] != size:
        out = np.zeros((size, size, 3), dtype=np.uint8)
        if rh > 0 and rw > 0:
            out[dy : dy + rh, dx : dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_mask_patch(
    zarr_store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size: int
) -> np.ndarray:
    """Read a cell-segmentation mask patch as uint32 (size, size).

    The mask is assumed to be in H&E pixel space (same resolution, same
    registration) so the caller passes the same (x0, y0) used for the H&E
    patch — no coordinate transform is required.

    Pixel values are integer label IDs (0 = background).  If the stored
    dtype is narrower than uint32 it is safely upcast; float masks are
    rounded and cast.

    Handle axes permutations (YX, CYX with C=1, XY, etc.).
    Clip to image bounds, zero-pad if the patch extends beyond the edge.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

    # Collapse a redundant channel axis (C=1 segmentation masks)
    if arr.ndim == 3:
        # Bring C first if needed
        active = [ax for ax in axes if ax in ("C", "Y", "X")]
        if "C" in active and active.index("C") > 0:
            perm = [active.index(a) for a in ("C", "Y", "X") if a in active]
            arr = arr.transpose(perm)
        if arr.shape[0] == 1:
            arr = arr[0]  # (1, H, W) → (H, W)
        else:
            arr = arr[0]  # take first channel if C > 1

    # Upcast to uint32 (handles uint8, uint16, int32, float)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr).astype(np.uint32)
    else:
        arr = arr.astype(np.uint32)

    # Zero-pad to (size, size)
    if arr.shape[0] != size or arr.shape[1] != size:
        out = np.zeros((size, size), dtype=np.uint32)
        if rh > 0 and rw > 0:
            out[dy : dy + rh, dx : dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_multiplex_patch(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    y0: int,
    x0: int,
    size_y: int,
    size_x: int,
    channel_indices: list[int],
) -> np.ndarray:
    """Read multiplex patch for specific channel indices.

    Returns (C, size_y, size_x) uint16 where C = len(channel_indices).
    Handle axes permutations (CYX, YXC, etc.).
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size_y, size_x
    )

    # active_axes: CYX axes in the order they appear after scalar-collapsing non-CYX dims
    active_axes = [ax for ax in axes if ax in ("C", "Y", "X")]

    # Transpose to canonical (C, Y, X) order — handles any permutation of CYX axes
    if "C" in active_axes:
        target = [ax for ax in ("C", "Y", "X") if ax in active_axes]
        if active_axes != target:
            perm = [active_axes.index(ax) for ax in target]
            arr = arr.transpose(perm)
        arr = arr[channel_indices]  # select requested channels -> (C_sel, Y, X)
    else:
        # No channel axis — ensure spatial dims are (Y, X) order
        target = [ax for ax in ("Y", "X") if ax in active_axes]
        if active_axes != target:
            perm = [active_axes.index(ax) for ax in target]
            arr = arr.transpose(perm)
        arr = np.stack([arr] * len(channel_indices), axis=0)

    # Cast to uint16
    arr = arr.astype(np.uint16)

    # Zero-pad to (C_sel, size_y, size_x)
    n_ch = arr.shape[0]
    if arr.shape[1] != size_y or arr.shape[2] != size_x:
        out = np.zeros((n_ch, size_y, size_x), dtype=np.uint16)
        if rh > 0 and rw > 0:
            out[:, dy : dy + rh, dx : dx + rw] = arr[:, :rh, :rw]
        return out

    return arr


# ---------------------------------------------------------------------------
# Channel metadata parsing
# ---------------------------------------------------------------------------

# Re-export for backward compatibility with tests
load_channel_indices = resolve_channel_indices


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
    parser.add_argument("--out", default="processed")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tissue-min", type=float, default=0.1)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[
            "Keratin",
            "NaKATPase",
            "CDX2",
            "CD45",
            "CD3",
            "CD4",
            "CD8a",
            "CD20",
            "CD45RO",
            "CD68",
            "CD163",
            "FOXP3",
            "PD1",
            "aSMA",
            "CD31",
            "Desmin",
            "Collagen",
            "Ki67",
            "PCNA",
            "Vimentin",
            "Ecadherin",
        ],
        metavar="NAME",
        help="Multiplex channel names to extract (default: full Stage 3 marker panel, 21 channels).",
    )
    parser.add_argument(
        "--overview-downsample",
        type=int,
        default=64,
        help="Stride for H&E overview sampling (default 64)",
    )
    parser.add_argument(
        "--vis-channels",
        type=int,
        nargs=3,
        default=[0, 10, 20],
        help="3 multiplex channel indices for RGB composite in vis",
    )
    parser.add_argument(
        "--register",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ECC affine registration between H&E and MX tissue masks (default: on)",
    )
    parser.add_argument(
        "--mask-image",
        default=None,
        help="Optional cell segmentation mask OME-TIFF in H&E pixel space. "
        "Patches saved to processed/masks/{x0}_{y0}.npy as uint32 label IDs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    ds = args.overview_downsample
    patch_size = args.patch_size
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

        scale = (he_mpp_x / mx_mpp_x) if (he_mpp_x and mx_mpp_x) else (he_w / mx_w)
        print(f"  scale H&E -> multiplex: {scale:.4f}")
        print(f"Building tissue mask (downsample={ds}) ...")
        mask = build_tissue_mask(he_store, he_axes, he_w, he_h, downsample=ds)
        print(f"  Tissue fraction: {mask.mean():.2%}")

        if args.register:
            print("Computing affine registration via ECC on tissue masks ...")
            mx_mask = build_mx_tissue_mask(mx_store, mx_axes, mx_h, mx_w, ds)
            m_full = register_he_mx_affine(mask, mx_mask, ds, he_h, he_w, mx_h, mx_w)
            print(f"  Warp matrix:\n{m_full}")
        else:
            m_full = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)

        he_chw = read_overview_chw(he_store, he_axes, he_h, he_w, ds)  # (C, H, W)
        he_chw = (
            he_chw[:3] if he_chw.shape[0] >= 3 else np.repeat(he_chw[:1], 3, axis=0)
        )
        if he_chw.dtype != np.uint8:
            he_chw = percentile_to_uint8(he_chw)

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

        print("Extracting patches ...")
        index: list[dict] = []
        he_vis_coords: list[tuple[int, int]] = []
        mx_vis_coords: list[tuple[int, int]] = []

        for idx, (x0, y0) in enumerate(coords[:10]):
            if idx % 500 == 0:
                print(f"  {idx}/{len(coords)} ...")

            he_patch = read_he_patch(he_store, he_axes, he_w, he_h, y0, x0, patch_size)
            Image.fromarray(he_patch).save(out_dir / "he" / f"{x0}_{y0}.png")

            has_seg = False
            if seg_store is not None:
                seg_patch = read_mask_patch(
                    seg_store, seg_axes, seg_w, seg_h, y0, x0, patch_size
                )
                np.save(out_dir / "masks" / f"{x0}_{y0}.npy", seg_patch)
                has_seg = True

            x0_mx, y0_mx = transform_he_to_mx_point(m_full, x0, y0)
            size_mx = max(1, round(patch_size * scale))
            has_mx = (x0_mx + size_mx <= mx_w) and (y0_mx + size_mx <= mx_h)

            if has_mx:
                mx_patch = read_multiplex_patch(
                    mx_store,
                    mx_axes,
                    mx_w,
                    mx_h,
                    y0_mx,
                    x0_mx,
                    size_mx,
                    size_mx,
                    channel_indices,
                )
                if mx_patch.shape[1] != patch_size or mx_patch.shape[2] != patch_size:
                    resized = np.zeros(
                        (mx_patch.shape[0], patch_size, patch_size),
                        dtype=mx_patch.dtype,
                    )
                    for c in range(mx_patch.shape[0]):
                        resized[c] = cv2.resize(
                            mx_patch[c],
                            (patch_size, patch_size),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    mx_patch = resized
                np.save(out_dir / "multiplex" / f"{x0}_{y0}.npy", mx_patch)
                mx_vis_coords.append((x0_mx // ds, y0_mx // ds))

            he_vis_coords.append((x0 // ds, y0 // ds))
            entry: dict = {"x0": x0, "y0": y0, "has_multiplex": has_mx}
            if seg_store is not None:
                entry["has_mask"] = has_seg
            index.append(entry)

        n_mx = sum(p["has_multiplex"] for p in index)
        print(f"  Done. {n_mx}/{len(index)} patches have multiplex.")
        if seg_store is not None:
            n_seg = sum(p.get("has_mask", False) for p in index)
            print(f"        {n_seg}/{len(index)} patches have cell mask.")

        with open(out_dir / "index.json", "w", encoding="utf-8") as f:
            meta: dict = {
                "patches": index,
                "patch_size": patch_size,
                "stride": args.stride,
                "tissue_min": args.tissue_min,
                "img_w": he_w,
                "img_h": he_h,
                "he_mpp": he_mpp_x,
                "mx_mpp": mx_mpp_x,
                "scale_he_to_mx": scale,
                "channels": resolved_names,
                "warp_matrix": m_full.tolist(),
                "registration": args.register,
            }
            if args.mask_image:
                meta["mask_image"] = str(args.mask_image)
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
