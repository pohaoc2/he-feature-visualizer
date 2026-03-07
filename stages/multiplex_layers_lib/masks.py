"""Vessel-mask extraction and cleanup helpers."""

import cv2
import numpy as np


def binarize_otsu(arr: np.ndarray) -> np.ndarray:
    """Otsu threshold on (H, W) float32 [0,1] -> bool mask."""
    scaled = (arr * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def refine_vasculature_with_sma(
    cd31_mask: np.ndarray,
    sma_mask: np.ndarray,
    adjacency_px: int = 2,
) -> np.ndarray:
    """Refine a CD31 vessel mask with nearby SMA support."""
    if adjacency_px < 0:
        raise ValueError("adjacency_px must be >= 0")

    if not np.any(sma_mask):
        return cd31_mask.copy()

    k = 2 * adjacency_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near_cd31 = cv2.dilate(cd31_mask.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )
    return cd31_mask | (sma_mask & near_cd31)


def _validate_odd_kernel_size(name: str, value: int) -> None:
    """Validate odd kernel size convention where 0 means disabled."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    if value != 0 and value % 2 == 0:
        raise ValueError(f"{name} must be odd (or 0 to disable), got {value}")


def remove_small_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` pixels."""
    if min_area < 0:
        raise ValueError("min_area must be >= 0")

    mask_bool = mask.astype(bool)
    if min_area <= 1 or not np.any(mask_bool):
        return mask_bool.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    kept = np.zeros_like(mask_bool, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == label] = True
    return kept


def cleanup_vasculature_mask(
    vessel_mask: np.ndarray,
    open_kernel_size: int = 0,
    close_kernel_size: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """Apply optional morphology cleanup and component-size filtering."""
    _validate_odd_kernel_size("open_kernel_size", open_kernel_size)
    _validate_odd_kernel_size("close_kernel_size", close_kernel_size)

    cleaned = vessel_mask.astype(np.uint8)
    if open_kernel_size > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return remove_small_components(cleaned.astype(bool), min_area=min_area)


def apply_vessel_mask_quality_fallback(
    candidate_mask: np.ndarray,
    cd31_fallback_mask: np.ndarray,
    noisy_max_fraction: float = 0.98,
) -> tuple[np.ndarray, str]:
    """Stabilize vessel masks for empty/noisy edge cases."""
    if candidate_mask.shape != cd31_fallback_mask.shape:
        raise ValueError(
            f"candidate_mask shape {candidate_mask.shape} must match "
            f"cd31_fallback_mask shape {cd31_fallback_mask.shape}"
        )
    if not (0.0 < noisy_max_fraction <= 1.0):
        raise ValueError("noisy_max_fraction must be in (0, 1]")

    candidate = candidate_mask.astype(bool)
    fallback = cd31_fallback_mask.astype(bool)

    if not np.any(candidate):
        if np.any(fallback):
            return fallback.copy(), "empty_fallback"
        return candidate.copy(), "empty"

    coverage = float(np.mean(candidate))
    if coverage >= noisy_max_fraction:
        return fallback.copy(), "noisy_fallback"

    return candidate.copy(), "ok"
