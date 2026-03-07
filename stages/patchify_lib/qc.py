"""Registration QC metrics and decision gates."""

import cv2
import numpy as np

from utils.normalize import percentile_to_uint8

PASS_AFFINE = "PASS_AFFINE"
FAIL_GLOBAL_NEEDS_LANDMARKS = "FAIL_GLOBAL_NEEDS_LANDMARKS"
FAIL_LOCAL_NEEDS_DEFORMABLE = "FAIL_LOCAL_NEEDS_DEFORMABLE"


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
        gains.append(float(_iou(he_local, reg_local) - _iou(he_local, base_local)))
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
