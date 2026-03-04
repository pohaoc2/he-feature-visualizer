from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening


def _rgb_to_s_v(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB uint8 image to saturation and value (HSV) in [0,1].
    Returns (S, V) as float32 arrays shaped (H,W).
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[-1] != 3:
        raise ValueError("rgb must have shape (H,W,3)")
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    v = mx
    s = np.zeros_like(v, dtype=np.float32)
    nonzero = mx > 1e-6
    s[nonzero] = (delta[nonzero] / mx[nonzero]).astype(np.float32)
    return s, v.astype(np.float32)


def tissue_mask_rgb(
    rgb: np.ndarray,
    *,
    sat_thresh: float = 0.05,
    val_thresh: float = 0.98,
    open_size: int = 3,
    close_size: int = 5,
) -> np.ndarray:
    """
    CLAM-style (clean-room) tissue detection for H&E-like RGB.

    Heuristic:
    - Tissue tends to have non-trivial saturation (not gray/white)
    - Background tends to be high value (bright)

    Returns boolean mask (H,W).
    """
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    s, v = _rgb_to_s_v(rgb)
    m = (s >= float(sat_thresh)) & (v <= float(val_thresh))

    if open_size and open_size > 1:
        st = np.ones((open_size, open_size), dtype=bool)
        m = binary_opening(m, structure=st)
    if close_size and close_size > 1:
        st = np.ones((close_size, close_size), dtype=bool)
        m = binary_closing(m, structure=st)
    m = binary_fill_holes(m)
    return m.astype(bool)


def tissue_fraction_rgb(rgb: np.ndarray, **kwargs) -> float:
    m = tissue_mask_rgb(rgb, **kwargs)
    return float(np.mean(m))

