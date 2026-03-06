"""Percentile-based intensity normalization for image arrays."""

from __future__ import annotations

import numpy as np


def percentile_norm(
    arr: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Clip *arr* to [p_low, p_high] percentiles and normalize to [0, 1] float32.

    Returns zeros for uniform input (p_high == p_low).
    """
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def percentile_to_uint8(
    arr: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Clip *arr* to [p_low, p_high] percentiles and scale to uint8 [0, 255]."""
    return (percentile_norm(arr, p_low, p_high) * 255).astype(np.uint8)
