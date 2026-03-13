"""Visualize TIFF pairs and extract matched crops.

Commands
--------
overview   -- mask/OME overview PNG at reduced resolution
crop       -- mask/OME full-resolution crop PNG
crop-dna   -- mask + MX DNA full-resolution crop PNG with overlap panel
pair-crop  -- save matched native-resolution TIFF crops from two images
pair-viz   -- render two TIFFs side by side as a PNG

Usage
-----
# Existing mask + OME overview
python viz_mask.py overview --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif

# Existing mask + OME crop
python viz_mask.py crop --mask data/WD-76845-097.tif --ome data/WD-76845-097.ome.tif

# Mask + MX DNA crop (three panels: mask, DNA, overlap)
python -m tools.viz_mask crop-dna \
    --mask data/WD-76845-097.mask.tif \
    --ome data/WD-76845-097.ome.tif \
    --dna-channel 0

# Generic matched crop: auto-detect a shared 1024x1024 region in image1
python viz_mask.py pair-crop \
    --image1 data/WD-76845-096.ome.tif \
    --image2 data/WD-76845-097.ome.tif

# Generic matched crop: explicit region in image1
python viz_mask.py pair-crop \
    --image1 data/WD-76845-096.ome.tif \
    --image2 data/WD-76845-097.ome.tif \
    --crop-region 49152,30208,1024

# Generic side-by-side visualization
python viz_mask.py pair-viz \
    --image1 data/WD-76845-096.ome.tif \
    --image2 data/WD-76845-097.ome.tif \
    --save-path data/paired_overview.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from stages.patchify_lib.masking import tissue_mask_hsv
from stages.patchify_lib.registration import register_he_mx_affine
from utils.normalize import percentile_to_uint8
from utils.ome import get_image_dims, get_ome_mpp, open_zarr_store, read_overview_chw

DEFAULT_PAIR_CROP_SIZE = 1024
DEFAULT_OVERVIEW_DOWNSAMPLE = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_channel_axis(axes: str) -> str | None:
    """Return preferred channel-like axis label from an axes string."""
    axes_up = axes.upper()
    for ax in ("C", "I", "S"):
        if ax in axes_up:
            return ax
    return None


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


def side_by_side_many(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """Compose N (H, W, 3) arrays horizontally with a black gap."""
    if not images:
        raise ValueError("images must contain at least one panel")
    if len(images) == 1:
        return images[0]

    h = max(img.shape[0] for img in images)
    w = sum(img.shape[1] for img in images) + gap * (len(images) - 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    x = 0
    for img in images:
        canvas[: img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap
    return canvas


def parse_crop_region(
    crop_region: str | None,
    default_size: int = DEFAULT_PAIR_CROP_SIZE,
) -> tuple[int, int, int, int] | None:
    """Parse x,y[,size] or x,y,width,height into an image1 crop box."""
    if crop_region is None:
        return None

    parts = [int(tok.strip()) for tok in crop_region.split(",") if tok.strip()]
    if len(parts) == 2:
        x0, y0 = parts
        return x0, y0, default_size, default_size
    if len(parts) == 3:
        x0, y0, size = parts
        return x0, y0, size, size
    if len(parts) == 4:
        x0, y0, width, height = parts
        return x0, y0, width, height

    raise ValueError("crop_region must be 'x,y', 'x,y,size', or 'x,y,width,height'")


def _normalize_channel_u8(channel: np.ndarray) -> np.ndarray:
    """Convert one image channel to uint8 for visualization."""
    if channel.dtype == np.uint8:
        return channel.astype(np.uint8, copy=False)
    return percentile_to_uint8(channel)


def _render_chw_as_rgb(chw: np.ndarray) -> np.ndarray:
    """Render a CHW array as uint8 RGB for quick visualization."""
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {chw.shape!r}")

    if chw.shape[0] == 1 or chw.shape[0] > 4:
        gray = _normalize_channel_u8(chw[0])
        return np.stack([gray] * 3, axis=-1)

    rgb = chw[:3] if chw.shape[0] >= 3 else np.repeat(chw[:1], 3, axis=0)
    rgb_u8 = np.stack([_normalize_channel_u8(ch) for ch in rgb], axis=0)
    return np.moveaxis(rgb_u8, 0, -1)


def _build_binary_tissue_mask(chw: np.ndarray) -> np.ndarray:
    """Build a coarse tissue mask from a CHW overview."""
    if chw.shape[0] >= 3 and chw.shape[0] <= 4:
        return tissue_mask_hsv(_render_chw_as_rgb(chw))

    gray = _normalize_channel_u8(chw[0])
    if gray.max() == 0:
        return np.zeros(gray.shape, dtype=bool)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5, sigmaY=2.5)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
    return binary.astype(bool)


def _transform_points_affine(m_full: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points."""
    pts = np.asarray(points_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)
    return (m_full.astype(np.float64) @ hom.T).T


def _map_box_between_images(
    m_full: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Map an image1 crop box to an axis-aligned image2 crop box."""
    x0, y0, width, height = box
    corners = np.array(
        [
            [x0, y0],
            [x0 + width - 1, y0],
            [x0, y0 + height - 1],
            [x0 + width - 1, y0 + height - 1],
        ],
        dtype=np.float64,
    )
    mapped = _transform_points_affine(m_full, corners)
    xs = mapped[:, 0]
    ys = mapped[:, 1]
    mx_x0 = int(np.floor(xs.min()))
    mx_y0 = int(np.floor(ys.min()))
    mx_x1 = int(np.ceil(xs.max())) + 1
    mx_y1 = int(np.ceil(ys.max())) + 1
    return mx_x0, mx_y0, mx_x1 - mx_x0, mx_y1 - mx_y0


def _full_affine_to_overview(
    m_full: np.ndarray,
    image1_mask: np.ndarray,
    image2_mask: np.ndarray,
    image1_h: int,
    image1_w: int,
    image2_h: int,
    image2_w: int,
) -> np.ndarray:
    """Convert a full-resolution image1->image2 affine into overview space."""
    image1_ov_h, image1_ov_w = image1_mask.shape
    image2_ov_h, image2_ov_w = image2_mask.shape

    image1_sx = image1_w / max(1, image1_ov_w)
    image1_sy = image1_h / max(1, image1_ov_h)
    image2_inv_sx = image2_ov_w / max(1, image2_w)
    image2_inv_sy = image2_ov_h / max(1, image2_h)

    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    return np.array(
        [
            [
                a * image1_sx * image2_inv_sx,
                b * image1_sy * image2_inv_sx,
                tx * image2_inv_sx,
            ],
            [
                c * image1_sx * image2_inv_sy,
                d * image1_sy * image2_inv_sy,
                ty * image2_inv_sy,
            ],
        ],
        dtype=np.float32,
    )


def _mask_fraction_for_box(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> float:
    """Return mean mask coverage for a full-resolution crop box."""
    x0, y0, width, height = box
    x1 = x0 + width
    y1 = y0 + height
    mx0 = max(0, min(mask.shape[1], int(np.floor(x0 * mask.shape[1] / img_w))))
    mx1 = max(0, min(mask.shape[1], int(np.ceil(x1 * mask.shape[1] / img_w))))
    my0 = max(0, min(mask.shape[0], int(np.floor(y0 * mask.shape[0] / img_h))))
    my1 = max(0, min(mask.shape[0], int(np.ceil(y1 * mask.shape[0] / img_h))))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    return float(mask[my0:my1, mx0:mx1].mean())


def _validate_box_within_image(
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    label: str,
) -> None:
    """Raise when a crop box leaves image bounds."""
    x0, y0, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} crop box must have positive width and height")
    if x0 < 0 or y0 < 0 or x0 + width > img_w or y0 + height > img_h:
        raise ValueError(
            f"{label} crop box {(x0, y0, width, height)} exceeds image bounds "
            f"{img_w}x{img_h}"
        )


def _read_window(
    store,
    axes: str,
    y0: int,
    x0: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, str]:
    """Read a full-resolution crop and return it in CYX or YX order."""
    axes_up = axes.upper()
    ch_axis = _resolve_channel_axis(axes_up)
    sl: list[int | slice] = []
    for ax in axes_up:
        if ch_axis is not None and ax == ch_axis:
            sl.append(slice(None))
        elif ax == "Y":
            sl.append(slice(y0, y0 + height))
        elif ax == "X":
            sl.append(slice(x0, x0 + width))
        else:
            sl.append(0)

    arr = np.array(store[tuple(sl)])
    active: list[str] = []
    for ax in axes_up:
        if ch_axis is not None and ax == ch_axis:
            active.append("C")
        elif ax in ("Y", "X"):
            active.append(ax)
    target = [ax for ax in ("C", "Y", "X") if ax in active]
    if active != target:
        perm = [active.index(ax) for ax in target]
        arr = arr.transpose(perm)

    crop_axes = "".join(target) if target else "YX"
    return arr, crop_axes


def _tiff_suffix(path: Path) -> str:
    """Return a TIFF-like suffix, preserving .ome.tif when present."""
    lower = path.name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return path.name[-len(suffix) :]
    return path.suffix or ".tif"


def _stem_without_tiff_suffix(path: Path) -> str:
    """Return a filename stem without TIFF-like suffixes."""
    lower = path.name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _resolve_pair_crop_paths(
    image1_path: Path,
    image2_path: Path,
    image1_box: tuple[int, int, int, int],
    image2_box: tuple[int, int, int, int],
    save_path: Path | None,
) -> tuple[Path, Path]:
    """Resolve output paths for a pair-crop invocation."""
    if save_path is None:
        x1, y1, w1, h1 = image1_box
        x2, y2, w2, h2 = image2_box
        out1 = image1_path.parent / (
            f"{_stem_without_tiff_suffix(image1_path)}_crop_x{x1}_y{y1}_{w1}x{h1}"
            f"{_tiff_suffix(image1_path)}"
        )
        out2 = image2_path.parent / (
            f"{_stem_without_tiff_suffix(image2_path)}_crop_x{x2}_y{y2}_{w2}x{h2}"
            f"{_tiff_suffix(image2_path)}"
        )
        return out1, out2

    if save_path.exists() and save_path.is_dir():
        x1, y1, w1, h1 = image1_box
        x2, y2, w2, h2 = image2_box
        out1 = save_path / (
            f"{_stem_without_tiff_suffix(image1_path)}_crop_x{x1}_y{y1}_{w1}x{h1}"
            f"{_tiff_suffix(image1_path)}"
        )
        out2 = save_path / (
            f"{_stem_without_tiff_suffix(image2_path)}_crop_x{x2}_y{y2}_{w2}x{h2}"
            f"{_tiff_suffix(image2_path)}"
        )
        return out1, out2

    prefix = _stem_without_tiff_suffix(save_path)
    out1 = save_path.parent / f"{prefix}_image1{_tiff_suffix(image1_path)}"
    out2 = save_path.parent / f"{prefix}_image2{_tiff_suffix(image2_path)}"
    return out1, out2


def _save_tiff_crop(
    path: Path,
    arr: np.ndarray,
    axes: str,
    mpp_x: float | None = None,
    mpp_y: float | None = None,
) -> None:
    """Write a TIFF crop, preserving dtype/axes and optional physical pixel size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, float | str] = {"axes": axes}
    if mpp_x is not None and mpp_y is not None:
        metadata.update(
            {
                "PhysicalSizeX": float(mpp_x),
                "PhysicalSizeY": float(mpp_y),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
            }
        )
    tifffile.imwrite(str(path), arr, ome=True, metadata=metadata)


def _auto_detect_shared_crop(
    image1_mask: np.ndarray,
    image2_mask: np.ndarray,
    m_full: np.ndarray,
    image1_w: int,
    image1_h: int,
    image2_w: int,
    image2_h: int,
    crop_size: int,
    stride: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], dict[str, float]]:
    """Find a shared tissue-dense crop box across two images."""
    m_ov = _full_affine_to_overview(
        m_full,
        image1_mask,
        image2_mask,
        image1_h,
        image1_w,
        image2_h,
        image2_w,
    )
    image2_in_image1 = (
        cv2.warpAffine(
            image2_mask.astype(np.float32),
            m_ov,
            (image1_mask.shape[1], image1_mask.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0.5
    )
    shared_mask = np.logical_and(image1_mask, image2_in_image1)

    best_score = float("-inf")
    best_image1_box = None
    best_image2_box = None
    best_stats: dict[str, float] | None = None

    for y0 in range(0, image1_h - crop_size + 1, stride):
        for x0 in range(0, image1_w - crop_size + 1, stride):
            image1_box = (x0, y0, crop_size, crop_size)
            image1_frac = _mask_fraction_for_box(
                image1_mask, image1_box, image1_w, image1_h
            )
            shared_frac = _mask_fraction_for_box(
                shared_mask, image1_box, image1_w, image1_h
            )
            if image1_frac < 0.6 or shared_frac < 0.5:
                continue

            image2_box = _map_box_between_images(m_full, image1_box)
            if (
                image2_box[0] < 0
                or image2_box[1] < 0
                or image2_box[0] + image2_box[2] > image2_w
                or image2_box[1] + image2_box[3] > image2_h
            ):
                continue

            image2_frac = _mask_fraction_for_box(
                image2_mask, image2_box, image2_w, image2_h
            )
            if image2_frac < 0.6:
                continue

            score = image1_frac + shared_frac + image2_frac
            if score > best_score:
                best_score = score
                best_image1_box = image1_box
                best_image2_box = image2_box
                best_stats = {
                    "image1_tissue_fraction": image1_frac,
                    "shared_tissue_fraction": shared_frac,
                    "image2_tissue_fraction": image2_frac,
                }

    if best_image1_box is None or best_image2_box is None or best_stats is None:
        raise ValueError(
            "Could not find a shared crop region that fits inside both images"
        )

    return best_image1_box, best_image2_box, best_stats


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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(side_by_side(mask_rgb, ome_rgb))
    img.save(str(out_path))
    print(
        f"[overview] saved {out_path} ({img.size[0]}x{img.size[1]}) "
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
    """Read a (size x size) region from a single-page TIFF at full resolution."""
    with tifffile.TiffFile(str(path)) as tif:
        h_full, w_full = tif.pages[0].shape
        y0 = max(0, row - size // 2)
        x0 = max(0, col - size // 2)
        y1 = min(h_full, y0 + size)
        x1 = min(w_full, x0 + size)
        return tif.pages[0].asarray()[y0:y1, x0:x1]


def _read_channel_crop(
    path: Path, row: int, col: int, size: int, channel_index: int = 0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Read a centered crop from a TIFF/OME-TIFF, selecting one channel when present."""
    with tifffile.TiffFile(str(path)) as tif:
        img_w, img_h, axes = get_image_dims(tif)
        store = open_zarr_store(tif)

        y0 = max(0, row - size // 2)
        x0 = max(0, col - size // 2)
        y1 = min(img_h, y0 + size)
        x1 = min(img_w, x0 + size)
        crop, crop_axes = _read_window(store, axes, y0, x0, y1 - y0, x1 - x0)

    if "C" in crop_axes:
        if channel_index < 0 or channel_index >= crop.shape[0]:
            raise ValueError(
                f"Requested channel {channel_index} but image has {crop.shape[0]} channels"
            )
        return np.asarray(crop[channel_index]), (y0, x0)

    if channel_index != 0:
        raise ValueError(
            "Requested non-zero channel from a single-channel image; use --dna-channel 0"
        )
    return np.asarray(crop), (y0, x0)


def _make_dna_mask_overlap(
    dna_u8: np.ndarray, mask_crop: np.ndarray, alpha: float = 0.35
) -> np.ndarray:
    """Overlay mask foreground on top of DNA grayscale for quick spatial QC."""
    dna_rgb = np.stack([dna_u8] * 3, axis=-1).astype(np.float32)
    mask_fg = mask_crop > 0

    tint_rgb = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    dna_rgb[mask_fg] = (1.0 - alpha) * dna_rgb[mask_fg] + alpha * tint_rgb

    mask_edges = cv2.Canny((mask_fg.astype(np.uint8) * 255), 32, 96) > 0
    dna_rgb[mask_edges] = np.array([0.0, 255.0, 255.0], dtype=np.float32)
    return dna_rgb.clip(0, 255).astype(np.uint8)


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

    print(f"[crop] reading {crop_size}x{crop_size} crop from mask ...")
    mask_crop = read_crop(mask_path, row, col, crop_size)

    print(f"[crop] reading {crop_size}x{crop_size} crop from OME ...")
    ome_crop = read_crop(ome_path, row, col, crop_size)

    n_cells = int((np.unique(mask_crop) != 0).sum())
    print(
        f"[crop] mask: {mask_crop.shape}, {n_cells} cells, "
        f"{(mask_crop > 0).mean() * 100:.1f}% non-zero"
    )
    print(f"[crop] OME:  {ome_crop.shape}, unique values: {np.unique(ome_crop)}")

    mask_rgb = colorize_mask(mask_crop)
    ome_rgb = np.stack([contrast_stretch(ome_crop)] * 3, axis=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(side_by_side(mask_rgb, ome_rgb, gap=4))
    img.save(str(out_path))
    print(
        f"[crop] saved {out_path} ({img.size[0]}x{img.size[1]}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )


def visualize_crop_dna(
    mask_path: Path,
    ome_path: Path,
    out_path: Path,
    row: int | None = None,
    col: int | None = None,
    crop_size: int = 1024,
    search_downsample: int = 64,
    dna_channel: int = 0,
    overlap_alpha: float = 0.35,
) -> None:
    """Three-panel full-resolution crop: mask, MX DNA channel, and overlap."""
    t0 = time.perf_counter()

    if row is None or col is None:
        print(
            f"[crop-dna] auto-detecting densest region (search at {search_downsample}x) ..."
        )
        row, col = find_dense_region(mask_path, downsample=search_downsample)
        print(f"[crop-dna] center: row={row}, col={col}")

    print(f"[crop-dna] reading {crop_size}x{crop_size} crop from mask ...")
    mask_crop, (mask_y0, mask_x0) = _read_channel_crop(mask_path, row, col, crop_size, 0)
    print(
        f"[crop-dna] mask origin x={mask_x0} y={mask_y0}, shape={mask_crop.shape}, "
        f"non-zero={(mask_crop > 0).mean() * 100:.1f}%"
    )

    print(
        f"[crop-dna] reading {crop_size}x{crop_size} crop from OME DNA channel {dna_channel} ..."
    )
    dna_crop, (dna_y0, dna_x0) = _read_channel_crop(
        ome_path, row, col, crop_size, dna_channel
    )
    print(f"[crop-dna] DNA origin x={dna_x0} y={dna_y0}, shape={dna_crop.shape}")

    if mask_crop.shape != dna_crop.shape:
        raise ValueError(
            "Mask crop and DNA crop shapes differ. "
            f"mask={mask_crop.shape}, dna={dna_crop.shape}. "
            "Ensure both images are in the same pixel space."
        )

    n_cells = int((np.unique(mask_crop) != 0).sum())
    dna_u8 = percentile_to_uint8(dna_crop)
    print(
        f"[crop-dna] mask unique non-zero IDs={n_cells}, "
        f"DNA range=[{int(dna_crop.min())}, {int(dna_crop.max())}]"
    )

    mask_rgb = colorize_mask(mask_crop)
    dna_rgb = np.stack([dna_u8] * 3, axis=-1)
    overlap_rgb = _make_dna_mask_overlap(dna_u8, mask_crop, alpha=overlap_alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(side_by_side_many([mask_rgb, dna_rgb, overlap_rgb], gap=4))
    img.save(str(out_path))
    print(
        f"[crop-dna] saved {out_path} ({img.size[0]}x{img.size[1]}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )


# ---------------------------------------------------------------------------
# Generic TIFF pair crop / visualization
# ---------------------------------------------------------------------------


def crop_tiff_pair(
    image1_path: Path,
    image2_path: Path,
    crop_region: str | None = None,
    save_path: Path | None = None,
    crop_size: int = DEFAULT_PAIR_CROP_SIZE,
    downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
    stride: int | None = None,
) -> tuple[Path, Path]:
    """Crop a matched region from two TIFFs and save both native-resolution crops.

    `crop_region` is interpreted in image1 full-resolution pixels as:
      - x,y
      - x,y,size
      - x,y,width,height
    """
    t0 = time.perf_counter()
    image1_box = parse_crop_region(crop_region, default_size=crop_size)
    stride = stride or max(1, crop_size // 2)

    with tifffile.TiffFile(str(image1_path)) as tif1, tifffile.TiffFile(
        str(image2_path)
    ) as tif2:
        image1_w, image1_h, image1_axes = get_image_dims(tif1)
        image2_w, image2_h, image2_axes = get_image_dims(tif2)
        store1 = open_zarr_store(tif1)
        store2 = open_zarr_store(tif2)
        mpp1_x, mpp1_y = get_ome_mpp(tif1)
        mpp2_x, mpp2_y = get_ome_mpp(tif2)

        # If one image lacks OME mpp but both images share the same pixel grid,
        # propagate mpp from the other image so crops remain spatially calibrated.
        if (image1_w, image1_h) == (image2_w, image2_h):
            if mpp1_x is None and mpp2_x is not None:
                mpp1_x = mpp2_x
            if mpp1_y is None and mpp2_y is not None:
                mpp1_y = mpp2_y
            if mpp2_x is None and mpp1_x is not None:
                mpp2_x = mpp1_x
            if mpp2_y is None and mpp1_y is not None:
                mpp2_y = mpp1_y

        if image1_box is not None and (image1_w, image1_h) == (image2_w, image2_h):
            m_full = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            stats = {
                "image1_tissue_fraction": float("nan"),
                "shared_tissue_fraction": float("nan"),
                "image2_tissue_fraction": float("nan"),
            }
        else:
            print(f"[pair-crop] building tissue masks at 1/{downsample} resolution ...")
            image1_chw = read_overview_chw(
                store1, image1_axes, image1_h, image1_w, downsample
            )
            image2_chw = read_overview_chw(
                store2, image2_axes, image2_h, image2_w, downsample
            )
            image1_mask = _build_binary_tissue_mask(image1_chw)
            image2_mask = _build_binary_tissue_mask(image2_chw)

            print("[pair-crop] estimating image1 -> image2 affine transform ...")
            m_full = register_he_mx_affine(
                image1_mask,
                image2_mask,
                downsample,
                image1_h,
                image1_w,
                image2_h,
                image2_w,
            )
            print(f"[pair-crop] affine:\n{m_full}")
            if mpp1_x is not None and mpp2_x is not None:
                print(f"[pair-crop] mpp scale image1 -> image2: {mpp1_x / mpp2_x:.4f}")

            if image1_box is None:
                image1_box, image2_box, stats = _auto_detect_shared_crop(
                    image1_mask,
                    image2_mask,
                    m_full,
                    image1_w,
                    image1_h,
                    image2_w,
                    image2_h,
                    crop_size,
                    stride,
                )
            else:
                stats = {
                    "image1_tissue_fraction": _mask_fraction_for_box(
                        image1_mask, image1_box, image1_w, image1_h
                    ),
                    "shared_tissue_fraction": float("nan"),
                    "image2_tissue_fraction": float("nan"),
                }

        if image1_box is None:
            raise ValueError("image1 crop box could not be determined")

        _validate_box_within_image(image1_box, image1_w, image1_h, "image1")
        image2_box = _map_box_between_images(m_full, image1_box)
        _validate_box_within_image(image2_box, image2_w, image2_h, "image2")

        crop1, crop1_axes = _read_window(
            store1,
            image1_axes,
            image1_box[1],
            image1_box[0],
            image1_box[3],
            image1_box[2],
        )
        crop2, crop2_axes = _read_window(
            store2,
            image2_axes,
            image2_box[1],
            image2_box[0],
            image2_box[3],
            image2_box[2],
        )

    out1, out2 = _resolve_pair_crop_paths(
        image1_path, image2_path, image1_box, image2_box, save_path
    )
    _save_tiff_crop(out1, crop1, crop1_axes, mpp_x=mpp1_x, mpp_y=mpp1_y)
    _save_tiff_crop(out2, crop2, crop2_axes, mpp_x=mpp2_x, mpp_y=mpp2_y)

    print(
        f"[pair-crop] image1 box: x={image1_box[0]} y={image1_box[1]} "
        f"w={image1_box[2]} h={image1_box[3]}"
    )
    print(
        f"[pair-crop] image2 box: x={image2_box[0]} y={image2_box[1]} "
        f"w={image2_box[2]} h={image2_box[3]}"
    )
    print(
        "[pair-crop] tissue fractions: "
        f"image1={stats['image1_tissue_fraction']:.3f} "
        f"shared={stats['shared_tissue_fraction']:.3f} "
        f"image2={stats['image2_tissue_fraction']:.3f}"
    )
    print(f"[pair-crop] saved {out1} and {out2} " f"in {time.perf_counter() - t0:.1f}s")
    return out1, out2


def visualize_tiff_pair(
    image1_path: Path,
    image2_path: Path,
    save_path: Path | None = None,
    downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
    show: bool = False,
) -> Path | None:
    """Render two TIFFs side by side as a PNG, optionally opening a viewer."""
    t0 = time.perf_counter()

    with tifffile.TiffFile(str(image1_path)) as tif1:
        image1_w, image1_h, image1_axes = get_image_dims(tif1)
        chw1 = read_overview_chw(
            open_zarr_store(tif1), image1_axes, image1_h, image1_w, downsample
        )

    with tifffile.TiffFile(str(image2_path)) as tif2:
        image2_w, image2_h, image2_axes = get_image_dims(tif2)
        chw2 = read_overview_chw(
            open_zarr_store(tif2), image2_axes, image2_h, image2_w, downsample
        )

    rgb1 = _render_chw_as_rgb(chw1)
    rgb2 = _render_chw_as_rgb(chw2)
    img = Image.fromarray(side_by_side(rgb1, rgb2))

    if save_path is None:
        save_path = image1_path.parent / (
            f"{_stem_without_tiff_suffix(image1_path)}_vs_"
            f"{_stem_without_tiff_suffix(image2_path)}.png"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path))
    print(
        f"[pair-viz] saved {save_path} ({img.size[0]}x{img.size[1]}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    if show:
        img.show()

    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mask", required=True, type=Path, help="Path to the uint32 mask TIF"
    )
    p.add_argument("--ome", required=True, type=Path, help="Path to the OME-TIFF")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")


def _add_pair_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image1", required=True, type=Path, help="Reference TIFF path")
    p.add_argument("--image2", required=True, type=Path, help="Matched TIFF path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TIFF pairs and extract matched crops.",
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

    p_crop_dna = sub.add_parser(
        "crop-dna",
        help="Full-resolution crop with mask, MX DNA channel, and overlap panel",
    )
    _add_common(p_crop_dna)
    p_crop_dna.add_argument(
        "--row",
        type=int,
        default=None,
        help="Center row in full-res pixels (default: auto-detect)",
    )
    p_crop_dna.add_argument(
        "--col",
        type=int,
        default=None,
        help="Center col in full-res pixels (default: auto-detect)",
    )
    p_crop_dna.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Crop side length in pixels (default: 1024)",
    )
    p_crop_dna.add_argument(
        "--search-downsample",
        type=int,
        default=64,
        help="Downsample for auto-detection search (default: 64)",
    )
    p_crop_dna.add_argument(
        "--dna-channel",
        type=int,
        default=0,
        help="OME channel index to treat as DNA (default: 0)",
    )
    p_crop_dna.add_argument(
        "--overlap-alpha",
        type=float,
        default=0.35,
        help="Blend weight for mask tint in overlap panel (default: 0.35)",
    )

    p_pair_crop = sub.add_parser(
        "pair-crop", help="Save matched native-resolution TIFF crops from two images"
    )
    _add_pair_common(p_pair_crop)
    p_pair_crop.add_argument(
        "--crop-region",
        "--crop_region",
        default=None,
        help=(
            "Optional image1 crop box: x,y | x,y,size | x,y,width,height. "
            "If omitted, a shared tissue-dense region is auto-detected."
        ),
    )
    p_pair_crop.add_argument(
        "--save-path",
        "--save_path",
        type=Path,
        default=None,
        help=(
            "Optional output directory or filename prefix. "
            "If omitted, crops are saved next to each source image."
        ),
    )
    p_pair_crop.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_PAIR_CROP_SIZE,
        help=f"Default auto-detected crop size in image1 pixels (default: {DEFAULT_PAIR_CROP_SIZE})",
    )
    p_pair_crop.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_OVERVIEW_DOWNSAMPLE,
        help=f"Downsample for tissue-mask registration (default: {DEFAULT_OVERVIEW_DOWNSAMPLE})",
    )
    p_pair_crop.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Optional search stride for auto-detection (default: crop_size/2)",
    )

    p_pair_viz = sub.add_parser(
        "pair-viz", help="Render two TIFFs side by side as a PNG"
    )
    _add_pair_common(p_pair_viz)
    p_pair_viz.add_argument(
        "--save-path",
        "--save_path",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to <image1>_vs_<image2>.png",
    )
    p_pair_viz.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_OVERVIEW_DOWNSAMPLE,
        help=f"Downsample factor for the overview render (default: {DEFAULT_OVERVIEW_DOWNSAMPLE})",
    )
    p_pair_viz.add_argument(
        "--show",
        action="store_true",
        help="Open the rendered side-by-side image after saving it",
    )

    args = parser.parse_args()

    if args.cmd == "overview":
        out = args.out or args.mask.parent / f"{args.mask.stem}_overview.png"
        visualize_overview(args.mask, args.ome, out, downsample=args.downsample)
        return

    if args.cmd == "crop":
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
        return

    if args.cmd == "crop-dna":
        out = args.out or args.mask.parent / f"{args.mask.stem}_crop_dna_overlap.png"
        visualize_crop_dna(
            args.mask,
            args.ome,
            out,
            row=args.row,
            col=args.col,
            crop_size=args.crop_size,
            search_downsample=args.search_downsample,
            dna_channel=args.dna_channel,
            overlap_alpha=args.overlap_alpha,
        )
        return

    if args.cmd == "pair-crop":
        crop_tiff_pair(
            args.image1,
            args.image2,
            crop_region=args.crop_region,
            save_path=args.save_path,
            crop_size=args.crop_size,
            downsample=args.downsample,
            stride=args.stride,
        )
        return

    if args.cmd == "pair-viz":
        visualize_tiff_pair(
            args.image1,
            args.image2,
            save_path=args.save_path,
            downsample=args.downsample,
            show=args.show,
        )


if __name__ == "__main__":
    main()
