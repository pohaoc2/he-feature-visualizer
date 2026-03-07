"""Mask construction utilities for H&E and multiplex overviews."""

import cv2
import numpy as np

from utils.normalize import percentile_to_uint8


def tissue_mask_hsv(rgb: np.ndarray, mthresh: int = 7, close: int = 4) -> np.ndarray:
    """CLAM-style tissue detection using cv2 HSV operations."""
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
    """Build a boolean tissue mask from a downsampled H&E overview."""
    axes = axes.upper()
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"axes must contain both 'Y' and 'X'; got {axes!r}")
    c_first = "C" in axes and axes.index("C") < axes.index("Y")

    img_h_trunc = (img_h // downsample) * downsample
    img_w_trunc = (img_w // downsample) * downsample

    if c_first:
        raw = np.array(store[:, :img_h_trunc:downsample, :img_w_trunc:downsample])
        overview = np.moveaxis(raw, 0, -1)
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
    """Build a binary tissue mask from MX DNA channel at overview resolution."""
    ch = _read_channel_overview(store, axes, mx_h, mx_w, ds, channel_index)
    dna_u8 = percentile_to_uint8(ch)
    if dna_u8.max() == 0:
        return np.zeros((mx_h // ds, mx_w // ds), dtype=bool)
    blur = cv2.GaussianBlur(dna_u8, (0, 0), sigmaX=2.5, sigmaY=2.5)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
    return binary.astype(bool)
