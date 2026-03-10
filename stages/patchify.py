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
from stages.patchify_lib import masking as _masking
from stages.patchify_lib import qc as _qc
from stages.patchify_lib import readers as _readers
from stages.patchify_lib import registration as _registration

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


def _read_channel_overview(
    store, axes: str, img_h: int, img_w: int, ds: int, channel_index: int
) -> np.ndarray:
    """Read one channel at overview resolution and return array as (H, W)."""
    ax = axes.upper()
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds
    sl = []
    for a in ax:
        if a == "C":
            sl.append(channel_index)
        elif a == "Y":
            sl.append(slice(0, h_t, ds))
        elif a == "X":
            sl.append(slice(0, w_t, ds))
        else:
            sl.append(0)
    arr = np.array(store[tuple(sl)])
    active = [a for a in ax if a in ("Y", "X")]
    target = [a for a in ("Y", "X") if a in active]
    if active != target:
        arr = arr.transpose([active.index(a) for a in target])
    return arr


def build_mx_tissue_mask(
    store, axes: str, mx_h: int, mx_w: int, ds: int, channel_index: int = 0
) -> np.ndarray:
    """Build a binary tissue mask from the MX DNA channel (ch0) at overview resolution.

    Returns bool ndarray (mx_h // ds, mx_w // ds).
    """
    ch = _read_channel_overview(store, axes, mx_h, mx_w, ds, channel_index)
    dna_u8 = percentile_to_uint8(ch)
    if dna_u8.max() == 0:
        return np.zeros((mx_h // ds, mx_w // ds), dtype=bool)
    # Build a smooth tissue envelope from punctate nuclei signal.
    blur = cv2.GaussianBlur(dna_u8, (0, 0), sigmaX=2.5, sigmaY=2.5)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
    return binary.astype(bool)


def _read_he_gray_overview(
    store, axes: str, img_h: int, img_w: int, ds: int
) -> np.ndarray:
    """Read H&E at overview resolution and return normalized float32 grayscale (H, W)."""
    chw = read_overview_chw(
        store, axes, img_h, img_w, ds
    )  # (C, H, W) uint8 or original dtype
    if chw.shape[0] == 1:
        # Single-channel (e.g. SYX) — already grayscale
        gray = chw[0]
        if gray.dtype != np.uint8:
            gray = percentile_to_uint8(gray)
    else:
        rgb = np.moveaxis(chw[:3], 0, -1)  # (H, W, 3)
        if rgb.dtype != np.uint8:
            rgb = percentile_to_uint8(rgb)
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


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


PASS_AFFINE = "PASS_AFFINE"
FAIL_GLOBAL_NEEDS_LANDMARKS = "FAIL_GLOBAL_NEEDS_LANDMARKS"
FAIL_LOCAL_NEEDS_DEFORMABLE = "FAIL_LOCAL_NEEDS_DEFORMABLE"
REG_MODE_AFFINE = "affine"
REG_MODE_DEFORMABLE = "deformable"


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return centroid (x, y) for nonzero mask pixels, or image center if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return float(w) * 0.5, float(h) * 0.5
    return float(xs.mean()), float(ys.mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU of two boolean masks."""
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / (union + 1e-9))


def _full_affine_to_overview(
    m_full: np.ndarray,
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Convert full-resolution H&E->MX affine to overview-space affine."""
    he_ov_h, he_ov_w = he_mask.shape
    mx_ov_h, mx_ov_w = mx_mask.shape

    he_sx = he_w / max(1, he_ov_w)
    he_sy = he_h / max(1, he_ov_h)
    mx_inv_sx = mx_ov_w / max(1, mx_w)
    mx_inv_sy = mx_ov_h / max(1, mx_h)

    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    return np.array(
        [
            [a * he_sx * mx_inv_sx, b * he_sy * mx_inv_sx, tx * mx_inv_sx],
            [c * he_sx * mx_inv_sy, d * he_sy * mx_inv_sy, ty * mx_inv_sy],
        ],
        dtype=np.float32,
    )


def _warp_mx_mask_to_he_template(
    he_mask: np.ndarray, mx_mask: np.ndarray, m_ov: np.ndarray
) -> np.ndarray:
    """Warp MX overview mask into H&E overview template space using H&E->MX affine."""
    he_h, he_w = he_mask.shape
    warped = cv2.warpAffine(
        mx_mask.astype(np.float32),
        m_ov.astype(np.float32),
        (he_w, he_h),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0.5


def compute_channel_drift_metrics(
    mx_chw: np.ndarray, ref_channel: int = 0, min_response_for_qc: float = 0.2
) -> dict[str, object]:
    """Estimate per-channel drift to a reference channel via phase correlation."""
    if mx_chw.ndim != 3:
        raise ValueError(f"mx_chw must be (C,H,W); got shape {mx_chw.shape}")

    c, _, _ = mx_chw.shape
    if c == 0:
        return {
            "ref_channel": ref_channel,
            "n_channels": 0,
            "per_channel": [],
            "median_drift_px": 0.0,
            "max_drift_px": 0.0,
        }

    ref = percentile_to_uint8(mx_chw[ref_channel]).astype(np.float32)
    per_channel = []
    drifts: list[float] = []
    usable_drifts: list[float] = []

    for ch_idx in range(c):
        if ch_idx == ref_channel:
            continue
        moving = percentile_to_uint8(mx_chw[ch_idx]).astype(np.float32)
        try:
            (dx, dy), response = cv2.phaseCorrelate(ref, moving)
        except cv2.error:  # pylint: disable=catching-non-exception
            dx, dy, response = 0.0, 0.0, 0.0
        drift = float(np.hypot(dx, dy))
        drifts.append(drift)
        if response >= min_response_for_qc:
            usable_drifts.append(drift)
        per_channel.append(
            {
                "channel": int(ch_idx),
                "dx": float(dx),
                "dy": float(dy),
                "drift_px": drift,
                "response": float(response),
            }
        )

    summary_drifts = usable_drifts if usable_drifts else drifts
    median = float(np.median(summary_drifts)) if summary_drifts else 0.0
    max_drift = float(np.max(summary_drifts)) if summary_drifts else 0.0
    return {
        "ref_channel": int(ref_channel),
        "n_channels": int(c),
        "considered_channels": int(len(summary_drifts)),
        "min_response_for_qc": float(min_response_for_qc),
        "per_channel": per_channel,
        "median_drift_px": median,
        "max_drift_px": max_drift,
    }


def channel_drift_passes(
    metrics: dict[str, object], median_thresh: float = 1.5, max_thresh: float = 4.0
) -> bool:
    """Return True when drift metrics meet default QC thresholds."""
    median_drift = float(metrics.get("median_drift_px", 0.0))
    max_drift = float(metrics.get("max_drift_px", 0.0))
    return median_drift <= median_thresh and max_drift <= max_thresh


def compute_global_qc_metrics(
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    m_full: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> dict[str, float]:
    """Compute global registration QC metrics from overview masks."""
    m_ov = _full_affine_to_overview(m_full, he_mask, mx_mask, he_h, he_w, mx_h, mx_w)
    he_bin = he_mask > 0
    mx_warped = _warp_mx_mask_to_he_template(he_mask, mx_mask, m_ov)

    mask_iou = _iou(he_bin, mx_warped)
    cx_he, cy_he = _mask_centroid(he_bin)
    cx_mx, cy_mx = _mask_centroid(mx_warped)
    diag = float(np.hypot(*he_bin.shape[::-1]))
    centroid_offset_pct = (
        100.0 * float(np.hypot(cx_mx - cx_he, cy_mx - cy_he)) / (diag + 1e-9)
    )

    a, b, _ = map(float, m_full[0])
    c, d, _ = map(float, m_full[1])
    observed_sx = float(np.hypot(a, c))
    observed_sy = float(np.hypot(b, d))
    expected_sx = mx_w / max(1, he_w)
    expected_sy = mx_h / max(1, he_h)
    err_x = abs(observed_sx - expected_sx) / (abs(expected_sx) + 1e-9)
    err_y = abs(observed_sy - expected_sy) / (abs(expected_sy) + 1e-9)
    scale_error_pct = 100.0 * max(err_x, err_y)

    return {
        "mask_iou": float(mask_iou),
        "centroid_offset_pct": float(centroid_offset_pct),
        "scale_error_pct": float(scale_error_pct),
        "observed_scale_x": float(observed_sx),
        "observed_scale_y": float(observed_sy),
        "expected_scale_x": float(expected_sx),
        "expected_scale_y": float(expected_sy),
    }


def global_qc_passes(
    metrics: dict[str, float],
    iou_thresh: float = 0.75,
    centroid_thresh_pct: float = 3.0,
    scale_err_thresh_pct: float = 10.0,
) -> bool:
    """Return True when global registration QC metrics pass thresholds."""
    return (
        float(metrics.get("mask_iou", 0.0)) >= iou_thresh
        and float(metrics.get("centroid_offset_pct", 1e9)) <= centroid_thresh_pct
        and float(metrics.get("scale_error_pct", 1e9)) <= scale_err_thresh_pct
    )


def compute_patch_qc_metrics(
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    m_full: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
    sample_count: int = 50,
    patch_size_ov: int = 24,
    seed: int = 0,
) -> dict[str, float]:
    """Compute patch-level QC by local IoU gain vs scale-only baseline."""
    he_bin = he_mask > 0
    he_ov_h, he_ov_w = he_bin.shape

    m_full_base = np.array(
        [[mx_w / max(1, he_w), 0.0, 0.0], [0.0, mx_h / max(1, he_h), 0.0]],
        dtype=np.float32,
    )
    m_ov_reg = _full_affine_to_overview(
        m_full, he_mask, mx_mask, he_h, he_w, mx_h, mx_w
    )
    m_ov_base = _full_affine_to_overview(
        m_full_base, he_mask, mx_mask, he_h, he_w, mx_h, mx_w
    )
    mx_reg = _warp_mx_mask_to_he_template(he_mask, mx_mask, m_ov_reg)
    mx_base = _warp_mx_mask_to_he_template(he_mask, mx_mask, m_ov_base)

    valid_reg = cv2.warpAffine(
        np.ones(mx_mask.shape, dtype=np.float32),
        m_ov_reg.astype(np.float32),
        (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Boundary regions are more sensitive to local misalignment than interiors.
    edge = (
        cv2.morphologyEx(
            he_bin.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        > 0
    )
    ys, xs = np.nonzero(edge)
    if len(xs) < max(10, sample_count // 2):
        ys, xs = np.nonzero(he_bin)
    if len(xs) == 0:
        return {
            "sample_count": 0.0,
            "improved_fraction": 0.0,
            "median_gain": 0.0,
            "inside_fraction_pass_rate": 0.0,
            "mean_gain": 0.0,
        }

    rng = np.random.default_rng(seed)
    n = min(sample_count, len(xs))
    pick = rng.choice(len(xs), size=n, replace=False)
    half = max(1, patch_size_ov // 2)

    gains = []
    inside = []
    for idx in pick:
        cy = int(ys[idx])
        cx = int(xs[idx])
        y0 = max(0, cy - half)
        y1 = min(he_ov_h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(he_ov_w, cx + half)
        he_local = he_bin[y0:y1, x0:x1]
        reg_local = mx_reg[y0:y1, x0:x1]
        base_local = mx_base[y0:y1, x0:x1]
        iou_reg = _iou(he_local, reg_local)
        iou_base = _iou(he_local, base_local)
        gains.append(float(iou_reg - iou_base))
        inside.append(float(valid_reg[y0:y1, x0:x1].mean()))

    gains_arr = np.array(gains, dtype=np.float64)
    inside_arr = np.array(inside, dtype=np.float64)
    return {
        "sample_count": float(len(gains)),
        "improved_fraction": float(np.mean(gains_arr > 1e-9)),
        "median_gain": float(np.median(gains_arr)),
        "inside_fraction_pass_rate": float(np.mean(inside_arr >= 0.85)),
        "mean_gain": float(np.mean(gains_arr)),
    }


def patch_qc_passes(
    metrics: dict[str, float],
    improved_fraction_thresh: float = 0.8,
    median_gain_thresh: float = 0.0,
    inside_pass_thresh: float = 0.95,
) -> bool:
    """Return True when patch-level QC metrics pass thresholds."""
    return (
        float(metrics.get("improved_fraction", 0.0)) >= improved_fraction_thresh
        and float(metrics.get("median_gain", 0.0)) > median_gain_thresh
        and float(metrics.get("inside_fraction_pass_rate", 0.0)) >= inside_pass_thresh
    )


def decide_registration_path(
    global_metrics: dict[str, float], patch_metrics: dict[str, float]
) -> str:
    """Select the next registration path based on QC outcomes."""
    if not global_qc_passes(global_metrics):
        return FAIL_GLOBAL_NEEDS_LANDMARKS
    if not patch_qc_passes(patch_metrics):
        return FAIL_LOCAL_NEEDS_DEFORMABLE
    return PASS_AFFINE


def _apply_inverse_flow(
    image: np.ndarray, flow_dx: np.ndarray, flow_dy: np.ndarray
) -> np.ndarray:
    """Sample image at (x-flow_dx, y-flow_dy), returning image in template space."""
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = grid_x - flow_dx.astype(np.float32)
    map_y = grid_y - flow_dy.astype(np.float32)
    return cv2.remap(
        image.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def estimate_deformable_field(
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    m_full: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> dict[str, object]:
    """Estimate an overview-space dense field that refines affine registration."""
    m_ov = _full_affine_to_overview(m_full, he_mask, mx_mask, he_h, he_w, mx_h, mx_w)
    he_f = cv2.GaussianBlur((he_mask > 0).astype(np.float32), (0, 0), sigmaX=1.5)
    mx_aff = cv2.warpAffine(
        (mx_mask > 0).astype(np.float32),
        m_ov.astype(np.float32),
        (he_mask.shape[1], he_mask.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mx_f = cv2.GaussianBlur(mx_aff.astype(np.float32), (0, 0), sigmaX=1.5)

    flow = cv2.calcOpticalFlowFarneback(
        mx_f,
        he_f,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=41,
        iterations=6,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.0)

    # Guardrail: avoid unrealistic deformations from low-texture areas.
    max_disp = 0.08 * float(min(he_mask.shape))
    mag = np.hypot(flow[..., 0], flow[..., 1])
    scale = np.ones_like(mag, dtype=np.float32)
    valid = mag > (max_disp + 1e-6)
    scale[valid] = max_disp / mag[valid]
    flow[..., 0] *= scale
    flow[..., 1] *= scale

    mx_aff_bin = mx_aff > 0.5
    mx_def = _apply_inverse_flow(mx_aff.astype(np.float32), flow[..., 0], flow[..., 1])
    mx_def_bin = mx_def > 0.5
    iou_aff = _iou(he_mask > 0, mx_aff_bin)
    iou_def = _iou(he_mask > 0, mx_def_bin)

    return {
        "flow_dx_ov": flow[..., 0].astype(np.float32),
        "flow_dy_ov": flow[..., 1].astype(np.float32),
        "iou_affine": float(iou_aff),
        "iou_deformable": float(iou_def),
        "mean_disp_ov": float(np.mean(np.hypot(flow[..., 0], flow[..., 1]))),
        "max_disp_ov": float(np.max(np.hypot(flow[..., 0], flow[..., 1]))),
    }


def compute_deformable_patch_qc_metrics(
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    m_full: np.ndarray,
    flow_dx_ov: np.ndarray,
    flow_dy_ov: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
    sample_count: int = 50,
    patch_size_ov: int = 24,
    seed: int = 0,
) -> dict[str, float]:
    """Patch-level gain metrics comparing deformable result to affine baseline."""
    he_bin = he_mask > 0
    he_ov_h, he_ov_w = he_bin.shape

    m_ov = _full_affine_to_overview(m_full, he_mask, mx_mask, he_h, he_w, mx_h, mx_w)
    mx_aff = cv2.warpAffine(
        (mx_mask > 0).astype(np.float32),
        m_ov.astype(np.float32),
        (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mx_aff_bin = mx_aff > 0.5
    mx_def = _apply_inverse_flow(mx_aff, flow_dx_ov, flow_dy_ov)
    mx_def_bin = mx_def > 0.5

    valid_aff = cv2.warpAffine(
        np.ones(mx_mask.shape, dtype=np.float32),
        m_ov.astype(np.float32),
        (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_def = _apply_inverse_flow(valid_aff, flow_dx_ov, flow_dy_ov)

    edge = (
        cv2.morphologyEx(
            he_bin.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        > 0
    )
    ys, xs = np.nonzero(edge)
    if len(xs) < max(10, sample_count // 2):
        ys, xs = np.nonzero(he_bin)
    if len(xs) == 0:
        return {
            "sample_count": 0.0,
            "improved_fraction": 0.0,
            "median_gain": 0.0,
            "inside_fraction_pass_rate": 0.0,
            "mean_gain": 0.0,
        }

    rng = np.random.default_rng(seed)
    n = min(sample_count, len(xs))
    pick = rng.choice(len(xs), size=n, replace=False)
    half = max(1, patch_size_ov // 2)

    gains = []
    inside = []
    for idx in pick:
        cy = int(ys[idx])
        cx = int(xs[idx])
        y0 = max(0, cy - half)
        y1 = min(he_ov_h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(he_ov_w, cx + half)
        he_local = he_bin[y0:y1, x0:x1]
        cand_local = mx_def_bin[y0:y1, x0:x1]
        base_local = mx_aff_bin[y0:y1, x0:x1]
        gains.append(float(_iou(he_local, cand_local) - _iou(he_local, base_local)))
        inside.append(float(valid_def[y0:y1, x0:x1].mean()))

    gains_arr = np.array(gains, dtype=np.float64)
    inside_arr = np.array(inside, dtype=np.float64)
    return {
        "sample_count": float(len(gains)),
        "improved_fraction": float(np.mean(gains_arr > 1e-9)),
        "median_gain": float(np.median(gains_arr)),
        "inside_fraction_pass_rate": float(np.mean(inside_arr >= 0.85)),
        "mean_gain": float(np.mean(gains_arr)),
    }


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
    # Resolve a raw ZarrTiffStore (from tif.aszarr()) to a subscriptable array.
    # Newer zarr releases may expose array classes that are not exactly zarr.Array,
    # so prefer capability checks before attempting zarr.open().
    if not (
        hasattr(store, "__getitem__")
        and hasattr(store, "shape")
        and hasattr(store, "ndim")
    ):
        raw = zarr.open(store, mode="r")
        if hasattr(raw, "__getitem__") and hasattr(raw, "shape"):
            store = raw
        else:
            store = raw["0"]

    y0i = int(y0)
    x0i = int(x0)
    y1i = y0i + int(size_y)
    x1i = x0i + int(size_x)

    y0c = max(0, min(y0i, img_h))
    x0c = max(0, min(x0i, img_w))
    y1c = max(0, min(y1i, img_h))
    x1c = max(0, min(x1i, img_w))

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
    dy = y0c - y0i
    dx = x0c - x0i
    rh = max(0, y1c - y0c)
    rw = max(0, x1c - x0c)
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


def _transform_points_affine(m: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points and return Nx2 transformed points."""
    pts = np.asarray(points_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)
    return (m.astype(np.float64) @ hom.T).T


def _he_patch_to_mx_local_affine(
    m_full: np.ndarray, he_x0: int, he_y0: int
) -> np.ndarray:
    """Return affine mapping patch-local (u,v) -> MX full-res (x,y)."""
    m = m_full.astype(np.float64)
    a, b, tx = m[0]
    c, d, ty = m[1]
    return np.array(
        [
            [a, b, a * he_x0 + b * he_y0 + tx],
            [c, d, c * he_x0 + d * he_y0 + ty],
        ],
        dtype=np.float64,
    )


def read_multiplex_patch_affine(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    channel_indices: list[int],
) -> tuple[np.ndarray, bool]:
    """Read an affine-aligned MX patch in H&E patch frame.

    Returns:
      - patch: (C, patch_size, patch_size) uint16
      - inside: True if all transformed patch corners lie inside MX bounds
    """
    n_ch = len(channel_indices)
    out = np.zeros((n_ch, patch_size, patch_size), dtype=np.uint16)

    m_local = _he_patch_to_mx_local_affine(m_full, he_x0, he_y0)
    corners = np.array(
        [
            [0.0, 0.0],
            [patch_size - 1.0, 0.0],
            [0.0, patch_size - 1.0],
            [patch_size - 1.0, patch_size - 1.0],
        ],
        dtype=np.float64,
    )
    corners_mx = _transform_points_affine(m_local, corners)
    xs = corners_mx[:, 0]
    ys = corners_mx[:, 1]
    inside = bool(
        np.all(xs >= 0.0)
        and np.all(ys >= 0.0)
        and np.all(xs < float(img_w))
        and np.all(ys < float(img_h))
    )

    x_min = int(np.floor(xs.min()))
    x_max = int(np.ceil(xs.max())) + 1
    y_min = int(np.floor(ys.min()))
    y_max = int(np.ceil(ys.max())) + 1
    src_w = max(1, x_max - x_min)
    src_h = max(1, y_max - y_min)

    src = read_multiplex_patch(
        zarr_store,
        axes,
        img_w,
        img_h,
        y0=y_min,
        x0=x_min,
        size_y=src_h,
        size_x=src_w,
        channel_indices=channel_indices,
    )

    # Patch-local destination -> source-bbox-local mapping.
    m_patch = m_local.copy()
    m_patch[0, 2] -= x_min
    m_patch[1, 2] -= y_min
    m_patch = m_patch.astype(np.float32)

    for c in range(n_ch):
        warped = cv2.warpAffine(
            src[c].astype(np.float32),
            m_patch,
            (patch_size, patch_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[c] = np.clip(np.rint(warped), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return out, inside


def read_multiplex_patch_affine_deform(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    channel_indices: list[int],
    flow_dx_ov: np.ndarray,
    flow_dy_ov: np.ndarray,
    he_full_w: int,
    he_full_h: int,
) -> tuple[np.ndarray, bool]:
    """Read a deformable-refined MX patch in H&E patch frame.

    The dense field is defined in H&E overview coordinates and applied as an
    inverse displacement before mapping through the affine matrix.
    """
    n_ch = len(channel_indices)
    out = np.zeros((n_ch, patch_size, patch_size), dtype=np.uint16)

    # Destination grid in H&E full-res coordinates.
    uu, vv = np.meshgrid(
        np.arange(patch_size, dtype=np.float32),
        np.arange(patch_size, dtype=np.float32),
    )
    x_he = uu + float(he_x0)
    y_he = vv + float(he_y0)

    # Sample overview displacement and convert to full-res H&E pixels.
    h_ov, w_ov = flow_dx_ov.shape
    he_sx = he_full_w / float(max(1, w_ov))
    he_sy = he_full_h / float(max(1, h_ov))
    x_ov = x_he / he_sx
    y_ov = y_he / he_sy
    dx_ov = cv2.remap(
        flow_dx_ov.astype(np.float32),
        x_ov.astype(np.float32),
        y_ov.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    dy_ov = cv2.remap(
        flow_dy_ov.astype(np.float32),
        x_ov.astype(np.float32),
        y_ov.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    x_corr = x_he - dx_ov * he_sx
    y_corr = y_he - dy_ov * he_sy

    # Affine map corrected H&E coordinates into MX full-res coordinates.
    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    map_mx_x = a * x_corr + b * y_corr + tx
    map_mx_y = c * x_corr + d * y_corr + ty

    inside = bool(
        np.all(map_mx_x >= 0.0)
        and np.all(map_mx_y >= 0.0)
        and np.all(map_mx_x < float(img_w))
        and np.all(map_mx_y < float(img_h))
    )

    x_min = int(np.floor(float(map_mx_x.min())))
    x_max = int(np.ceil(float(map_mx_x.max()))) + 1
    y_min = int(np.floor(float(map_mx_y.min())))
    y_max = int(np.ceil(float(map_mx_y.max()))) + 1
    src_w = max(1, x_max - x_min)
    src_h = max(1, y_max - y_min)

    src = read_multiplex_patch(
        zarr_store,
        axes,
        img_w,
        img_h,
        y0=y_min,
        x0=x_min,
        size_y=src_h,
        size_x=src_w,
        channel_indices=channel_indices,
    )

    local_x = (map_mx_x - float(x_min)).astype(np.float32)
    local_y = (map_mx_y - float(y_min)).astype(np.float32)
    for ch in range(n_ch):
        warped = cv2.remap(
            src[ch].astype(np.float32),
            local_x,
            local_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[ch] = np.clip(np.rint(warped), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return out, inside


# ---------------------------------------------------------------------------
# Channel metadata parsing
# ---------------------------------------------------------------------------

# Re-export for backward compatibility with tests
load_channel_indices = resolve_channel_indices

# Canonical implementations have been split into patchify_lib modules.
# Re-export names here to preserve public API and test compatibility.
tissue_mask_hsv = _masking.tissue_mask_hsv
tissue_fraction = _masking.tissue_fraction
build_tissue_mask = _masking.build_tissue_mask
_read_channel_overview = _masking._read_channel_overview
build_mx_tissue_mask = _masking.build_mx_tissue_mask

PASS_AFFINE = _qc.PASS_AFFINE
FAIL_GLOBAL_NEEDS_LANDMARKS = _qc.FAIL_GLOBAL_NEEDS_LANDMARKS
FAIL_LOCAL_NEEDS_DEFORMABLE = _qc.FAIL_LOCAL_NEEDS_DEFORMABLE
_mask_centroid = _qc._mask_centroid
_iou = _qc._iou
_full_affine_to_overview = _qc._full_affine_to_overview
_warp_mx_mask_to_he_template = _qc._warp_mx_mask_to_he_template
compute_channel_drift_metrics = _qc.compute_channel_drift_metrics
channel_drift_passes = _qc.channel_drift_passes
compute_global_qc_metrics = _qc.compute_global_qc_metrics
global_qc_passes = _qc.global_qc_passes
compute_patch_qc_metrics = _qc.compute_patch_qc_metrics
patch_qc_passes = _qc.patch_qc_passes
decide_registration_path = _qc.decide_registration_path

REG_MODE_AFFINE = _registration.REG_MODE_AFFINE
REG_MODE_DEFORMABLE = _registration.REG_MODE_DEFORMABLE
register_he_mx_affine = _registration.register_he_mx_affine
refine_affine_fine_scale = _registration.refine_affine_fine_scale
register_he_mx_affine_intensity = _registration.register_he_mx_affine_intensity
register_he_mx_orb = _registration.register_he_mx_orb
_apply_inverse_flow = _registration._apply_inverse_flow
estimate_deformable_field = _registration.estimate_deformable_field
estimate_deformable_field_intensity = _registration.estimate_deformable_field_intensity
compute_deformable_patch_qc_metrics = _registration.compute_deformable_patch_qc_metrics
transform_he_to_mx_point = _registration.transform_he_to_mx_point
_transform_points_affine = _registration._transform_points_affine
_he_patch_to_mx_local_affine = _registration._he_patch_to_mx_local_affine

get_tissue_patches = _readers.get_tissue_patches
get_patch_grid = _readers.get_patch_grid
_clip_and_read = _readers._clip_and_read
read_he_patch = _readers.read_he_patch
read_mask_patch = _readers.read_mask_patch
read_multiplex_patch = _readers.read_multiplex_patch
read_multiplex_patch_affine = _readers.read_multiplex_patch_affine
read_multiplex_patch_affine_deform = _readers.read_multiplex_patch_affine_deform
multiplex_patch_overlap_fraction_affine = (
    _readers.multiplex_patch_overlap_fraction_affine
)
multiplex_patch_overlap_fraction_deform = (
    _readers.multiplex_patch_overlap_fraction_deform
)


def _evaluate_registration_qc(
    m_full: np.ndarray,
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
    ds: int,
    coords: list,
    patch_size: int,
) -> dict:
    """Evaluate global + patch-level QC for a given warp matrix.

    Returns a dict with keys: 'global_qc', 'patch_qc', 'decision'.
    """
    global_qc = compute_global_qc_metrics(
        he_mask, mx_mask, m_full, he_h, he_w, mx_h, mx_w
    )
    patch_qc = compute_patch_qc_metrics(
        he_mask,
        mx_mask,
        m_full,
        he_h,
        he_w,
        mx_h,
        mx_w,
        sample_count=50,
        patch_size_ov=max(8, patch_size // max(1, ds)),
        seed=0,
    )
    decision = decide_registration_path(global_qc, patch_qc)
    return {
        "global_qc": global_qc,
        "patch_qc": patch_qc,
        "decision": decision,
    }


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
            "pan-cytokeratin",
            "na/k atpase",
            "cdx-2",
            "cd45",
            "cd3",
            "cd4",
            "cd8a",
            "cd20",
            "cd45ro",
            "cd68",
            "cd163",
            "foxp3",
            "pd-1",
            "aortic smooth muscle actin",
            "cd31",
            "desmin",
            "collagen",
            "antigen ki67",
            "pcna",
            "vimentin",
            "e-cadherin",
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
    parser.add_argument(
        "--force-deformable",
        action="store_true",
        help="Force deformable registration when local QC suggests deformable "
        "refinement, even if auto-gating would keep affine.",
    )
    parser.add_argument(
        "--min-multiplex-overlap",
        type=float,
        default=1.0,
        help="Minimum MX overlap fraction required to save multiplex patch "
        "(0.0-1.0, default: 1.0 for full-coverage only).",
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

        scale = (he_mpp_x / mx_mpp_x) if (he_mpp_x and mx_mpp_x) else (mx_w / he_w)
        print(f"  scale H&E -> multiplex: {scale:.4f}")
        print(f"Building tissue mask (downsample={ds}) ...")
        mask = build_tissue_mask(he_store, he_axes, he_w, he_h, downsample=ds)
        print(f"  Tissue fraction: {mask.mean():.2%}")
        mx_mask = build_mx_tissue_mask(mx_store, mx_axes, mx_h, mx_w, ds)

        # --- Channel drift QC (MX internal, run once) ---
        print("Computing channel drift QC ...")
        drift_channel_indices = sorted(set([0, *channel_indices]))
        drift_stack = np.stack(
            [
                _read_channel_overview(mx_store, mx_axes, mx_h, mx_w, ds, c).astype(
                    np.float32
                )
                for c in drift_channel_indices
            ],
            axis=0,
        )
        ref_local = drift_channel_indices.index(0)
        drift_metrics = compute_channel_drift_metrics(
            drift_stack, ref_channel=ref_local
        )
        for row in drift_metrics["per_channel"]:
            row["channel"] = int(drift_channel_indices[int(row["channel"])])
        drift_metrics["evaluated_channel_indices"] = drift_channel_indices
        drift_pass = channel_drift_passes(drift_metrics)

        # --- Registration cascade A -> B -> C ---
        registration_method = "fallback_scale"
        registration_mode = REG_MODE_AFFINE
        deform_state = None
        deform_patch_qc = None
        # coords not yet computed; pass empty list (QC samples from mask edges)
        cascade_coords: list = []

        if not args.register:
            m_full = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
            registration_method = "fallback_scale"
            final_qc = _evaluate_registration_qc(
                m_full,
                mask,
                mx_mask,
                he_h,
                he_w,
                mx_h,
                mx_w,
                ds,
                cascade_coords,
                patch_size,
            )
            decision = final_qc["decision"]
            global_qc = final_qc["global_qc"]
            patch_qc = final_qc["patch_qc"]
        else:
            scale_fallback = he_w / mx_w
            m_full = np.array(
                [[1 / scale_fallback, 0, 0], [0, 1 / scale_fallback, 0]],
                dtype=np.float32,
            )

            # Approach A: phase-corr-initialized ECC on tissue masks, then
            # fine-scale (ds=8) refinement to resolve sub-pixel translation.
            ds_fine = 8
            print("  [A] Phase-corr ECC on tissue masks (coarse) ...")
            m_A_coarse = register_he_mx_affine(
                mask,
                mx_mask,
                ds,
                he_h,
                he_w,
                mx_h,
                mx_w,
                fallback_scale=(1.0 / scale) if scale > 0 else None,
            )
            print(f"  [A] Fine-scale refinement at ds={ds_fine} ...")
            he_mask_fine = build_tissue_mask(he_store, he_axes, he_w, he_h, ds_fine)
            mx_mask_fine = build_mx_tissue_mask(mx_store, mx_axes, mx_h, mx_w, ds_fine)
            m_A = refine_affine_fine_scale(
                he_mask_fine,
                mx_mask_fine,
                m_A_coarse,
                ds_fine,
                he_h,
                he_w,
                mx_h,
                mx_w,
            )
            qc_A = _evaluate_registration_qc(
                m_A,
                mask,
                mx_mask,
                he_h,
                he_w,
                mx_h,
                mx_w,
                ds,
                cascade_coords,
                patch_size,
            )
            if qc_A["decision"] == PASS_AFFINE:
                print(f"  [A] PASS (iou={qc_A['global_qc']['mask_iou']:.3f})")
                m_full = m_A
                registration_method = "affine_centroid"
            else:
                print(f"  [A] FAIL ({qc_A['decision']}) -> trying B ...")

                # Approach B: intensity-based ECC (HE gray vs DNA channel)
                he_gray_ov = _read_he_gray_overview(he_store, he_axes, he_h, he_w, ds)
                mx_dna_ov_f32 = _read_channel_overview(
                    mx_store, mx_axes, mx_h, mx_w, ds, channel_index=0
                ).astype(np.float32)
                mx_dna_max = float(mx_dna_ov_f32.max())
                if mx_dna_max > 0:
                    mx_dna_ov_f32 /= mx_dna_max

                print("  [B] Intensity-based ECC (HE gray vs DNA channel) ...")
                m_B = register_he_mx_affine_intensity(
                    he_gray_ov,
                    mx_dna_ov_f32,
                    mask,
                    mx_mask,
                    ds,
                    he_h,
                    he_w,
                    mx_h,
                    mx_w,
                    fallback_m_full=m_A,
                    fallback_scale=(1.0 / scale) if scale > 0 else None,
                )
                qc_B = _evaluate_registration_qc(
                    m_B,
                    mask,
                    mx_mask,
                    he_h,
                    he_w,
                    mx_h,
                    mx_w,
                    ds,
                    cascade_coords,
                    patch_size,
                )
                if qc_B["decision"] == PASS_AFFINE:
                    print(f"  [B] PASS (iou={qc_B['global_qc']['mask_iou']:.3f})")
                    m_full = m_B
                    registration_method = "affine_intensity"
                else:
                    print(f"  [B] FAIL ({qc_B['decision']}) -> trying C ...")

                    # Approach C: ORB keypoints + RANSAC
                    he_ov_h, he_ov_w = mask.shape
                    # Reuse he_gray_ov from approach B
                    he_gray_u8 = (he_gray_ov * 255).astype(np.uint8)
                    mx_dna_u8 = cv2.resize(
                        (np.clip(mx_dna_ov_f32, 0, 1) * 255).astype(np.uint8),
                        (he_ov_w, he_ov_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    print("  [C] ORB feature matching + RANSAC ...")
                    m_C = register_he_mx_orb(
                        he_gray_u8,
                        mx_dna_u8,
                        ds,
                        he_h,
                        he_w,
                        mx_h,
                        mx_w,
                    )
                    if m_C is not None:
                        qc_C = _evaluate_registration_qc(
                            m_C,
                            mask,
                            mx_mask,
                            he_h,
                            he_w,
                            mx_h,
                            mx_w,
                            ds,
                            cascade_coords,
                            patch_size,
                        )
                        print(
                            f"  [C] iou={qc_C['global_qc']['mask_iou']:.3f} "
                            f"decision={qc_C['decision']}"
                        )
                        m_full = m_C
                        registration_method = "orb"
                    else:
                        print(
                            "  [C] ORB returned no valid transform -> using best of A/B by IoU"
                        )
                        iou_A = qc_A["global_qc"]["mask_iou"]
                        iou_B = qc_B["global_qc"]["mask_iou"]
                        if iou_A >= iou_B:
                            m_full = m_A
                            registration_method = "affine_centroid"
                        else:
                            m_full = m_B
                            registration_method = "affine_intensity"

            # Final QC using selected m_full
            final_qc = _evaluate_registration_qc(
                m_full,
                mask,
                mx_mask,
                he_h,
                he_w,
                mx_h,
                mx_w,
                ds,
                cascade_coords,
                patch_size,
            )
            decision = final_qc["decision"]
            global_qc = final_qc["global_qc"]
            patch_qc = final_qc["patch_qc"]

        # --- Deformable refinement when affine patch QC fails ---
        # Uses fine-scale (ds_fine) binary mask Farneback rather than ds=64,
        # giving 8× sharper tissue boundary signal to detect local deformation.
        deform_attempted = False
        if decision == FAIL_LOCAL_NEEDS_DEFORMABLE and args.register:
            print(
                f"  [D] Affine patch QC failed -> fine-scale deformable (ds={ds_fine}) ..."
            )
            deform_attempted = True

            # he_mask_fine / mx_mask_fine were computed above for approach A.
            deform_state = estimate_deformable_field_intensity(
                None,
                None,  # intensity images reserved for future use
                he_mask_fine,
                mx_mask_fine,
                m_full,
                he_h,
                he_w,
                mx_h,
                mx_w,
            )
            print(
                f"  [D] iou_affine={deform_state['iou_affine']:.4f} "
                f"iou_deformable={deform_state['iou_deformable']:.4f} "
                f"mean_disp={deform_state['mean_disp_ov']:.3f}px"
            )

            # compute_deformable_patch_qc_metrics uses masks at ds=64; the flow
            # is at ds=ds_fine so we need to rescale to ds=64 overview units.
            # Flow is in overview-pixel units: 1 ds=8 ov px = ds_fine full-res px.
            # After resizing to ds=64 grid, divide by (ds/ds_fine) so that
            # `dx_ov * he_sx` still gives the correct full-res displacement.
            h_ov64, w_ov64 = mask.shape
            scale_ratio = float(ds) / float(ds_fine)  # e.g. 64/8 = 8
            flow_dx_64 = (
                cv2.resize(
                    deform_state["flow_dx_ov"],
                    (w_ov64, h_ov64),
                    interpolation=cv2.INTER_LINEAR,
                )
                / scale_ratio
            )
            flow_dy_64 = (
                cv2.resize(
                    deform_state["flow_dy_ov"],
                    (w_ov64, h_ov64),
                    interpolation=cv2.INTER_LINEAR,
                )
                / scale_ratio
            )
            deform_patch_qc = compute_deformable_patch_qc_metrics(
                mask,
                mx_mask,
                m_full,
                flow_dx_64,
                flow_dy_64,
                he_h,
                he_w,
                mx_h,
                mx_w,
            )
            # Accept deformable when forced, or if either the patch QC improves
            # significantly OR the overview IoU improves by >=1%
            # (the coarse patch_qc can miss full-res improvement when the
            # rescaled flow is <1 overview pixel).
            iou_gain = deform_state["iou_deformable"] - deform_state["iou_affine"]
            patch_ok = float(deform_patch_qc.get("improved_fraction", 0)) >= 0.5
            iou_ok = iou_gain >= 0.01
            force_ok = bool(args.force_deformable)
            if force_ok or patch_ok or iou_ok:
                # Store the rescaled flow so patch extraction uses ds=64 units
                # (read_multiplex_patch_affine_deform auto-scales by w_ov).
                deform_state = dict(deform_state)
                deform_state["flow_dx_ov"] = flow_dx_64
                deform_state["flow_dy_ov"] = flow_dy_64
                registration_mode = REG_MODE_DEFORMABLE
                decision = PASS_AFFINE  # deformable is good enough
                if force_ok:
                    print("  [D] Forcing deformable mode (--force-deformable).")
                print(
                    f"  [D] Deformable PASS "
                    f"(improved_fraction={deform_patch_qc['improved_fraction']:.2f}, "
                    f"iou_gain={iou_gain:+.4f})"
                )
            else:
                deform_state = None  # don't apply deformable if not helpful
                print(
                    f"  [D] Deformable not beneficial "
                    f"(improved_fraction={deform_patch_qc.get('improved_fraction', 0):.2f}, "
                    f"iou_gain={iou_gain:+.4f}), keeping affine"
                )

        print(
            f"  Registration: method={registration_method} "
            f"iou={global_qc['mask_iou']:.4f} "
            f"decision={decision}"
        )
        print(f"  Warp matrix:\n{m_full}")

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
        print(f"  Multiplex save threshold: overlap >= {min_mx_overlap:.3f}")

        for idx, (x0, y0) in enumerate(coords[:]):
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

            if registration_mode == REG_MODE_DEFORMABLE and deform_state is not None:
                mx_patch, inside_mx = read_multiplex_patch_affine_deform(
                    mx_store,
                    mx_axes,
                    mx_w,
                    mx_h,
                    he_x0=x0,
                    he_y0=y0,
                    patch_size=patch_size,
                    m_full=m_full,
                    channel_indices=channel_indices,
                    flow_dx_ov=deform_state["flow_dx_ov"],
                    flow_dy_ov=deform_state["flow_dy_ov"],
                    he_full_w=he_w,
                    he_full_h=he_h,
                )
                mx_overlap = multiplex_patch_overlap_fraction_deform(
                    he_x0=x0,
                    he_y0=y0,
                    patch_size=patch_size,
                    m_full=m_full,
                    flow_dx_ov=deform_state["flow_dx_ov"],
                    flow_dy_ov=deform_state["flow_dy_ov"],
                    he_full_w=he_w,
                    he_full_h=he_h,
                    mx_w=mx_w,
                    mx_h=mx_h,
                )
            else:
                mx_patch, inside_mx = read_multiplex_patch_affine(
                    mx_store,
                    mx_axes,
                    mx_w,
                    mx_h,
                    he_x0=x0,
                    he_y0=y0,
                    patch_size=patch_size,
                    m_full=m_full,
                    channel_indices=channel_indices,
                )
                mx_overlap = multiplex_patch_overlap_fraction_affine(
                    he_x0=x0,
                    he_y0=y0,
                    patch_size=patch_size,
                    m_full=m_full,
                    mx_w=mx_w,
                    mx_h=mx_h,
                )
            if min_mx_overlap >= 1.0 - 1e-9:
                has_mx = bool(inside_mx)
            else:
                has_mx = bool(mx_overlap >= min_mx_overlap)
            if has_mx:
                np.save(out_dir / "multiplex" / f"{x0}_{y0}.npy", mx_patch)

            x0_mx, y0_mx = transform_he_to_mx_point(m_full, x0, y0)
            mx_vis_coords.append((x0_mx // ds, y0_mx // ds))

            he_vis_coords.append((x0 // ds, y0 // ds))
            entry: dict = {
                "x0": x0,
                "y0": y0,
                "has_multiplex": has_mx,
                "multiplex_overlap_fraction": float(mx_overlap),
            }
            if seg_store is not None:
                entry["has_mask"] = has_seg
            index.append(entry)

        reg_dir = out_dir / "registration"
        reg_dir.mkdir(parents=True, exist_ok=True)
        with open(reg_dir / "affine.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "warp_matrix": m_full.tolist(),
                    "registration_enabled": bool(args.register),
                    "overview_downsample": int(ds),
                },
                f,
                indent=2,
            )
        if deform_state is not None:
            np.savez_compressed(
                reg_dir / "deform_field.npz",
                flow_dx_ov=deform_state["flow_dx_ov"],
                flow_dy_ov=deform_state["flow_dy_ov"],
            )
        with open(reg_dir / "qc_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "channel_drift": {
                        **drift_metrics,
                        "pass": drift_pass,
                        "thresholds": {"median_px": 1.5, "max_px": 4.0},
                    },
                    "global_qc": {
                        **global_qc,
                        "pass": global_qc_passes(global_qc),
                        "thresholds": {
                            "mask_iou": 0.75,
                            "centroid_offset_pct": 3.0,
                            "scale_error_pct": 10.0,
                        },
                    },
                    "patch_qc": {
                        **patch_qc,
                        "pass": patch_qc_passes(patch_qc),
                        "thresholds": {
                            "improved_fraction": 0.8,
                            "median_gain": 0.0,
                            "inside_fraction_pass_rate": 0.95,
                        },
                    },
                    "deformable": {
                        "attempted": deform_attempted,
                        "used": registration_mode == REG_MODE_DEFORMABLE,
                        "metrics": (
                            None
                            if deform_state is None
                            else {
                                k: v
                                for k, v in deform_state.items()
                                if not isinstance(v, np.ndarray)
                            }
                        ),
                        "patch_qc": deform_patch_qc,
                    },
                    "decision": decision,
                    "registration_mode": registration_mode,
                    "registration_method": registration_method,
                },
                f,
                indent=2,
            )
        with open(reg_dir / "final_transform.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": registration_mode,
                    "warp_matrix": m_full.tolist(),
                    "decision_initial": decision,
                    "deform_field": (
                        "deform_field.npz"
                        if (
                            deform_state is not None
                            and registration_mode == REG_MODE_DEFORMABLE
                        )
                        else None
                    ),
                    "registration_method": registration_method,
                },
                f,
                indent=2,
            )

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
                "registration_qc_decision": decision,
                "registration_mode": registration_mode,
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
