"""Affine and deformable registration helpers."""

import cv2
import numpy as np

from .qc import _full_affine_to_overview, _iou, _warp_mx_mask_to_he_template

REG_MODE_AFFINE = "affine"
REG_MODE_DEFORMABLE = "deformable"


def _mask_centroid_or_none(mask: np.ndarray) -> tuple[float, float] | None:
    """Return centroid (x, y) for nonzero mask pixels, or None when empty."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _centroid_init_m_ov(he_f32: np.ndarray, mx_resized: np.ndarray) -> np.ndarray:
    """Return 2x3 identity+translation init for ECC based on tissue centroids.

    Both arrays must be in the same (HE overview) pixel space.
    Falls back to identity if either mask is empty.
    """
    he_ctr = _mask_centroid_or_none(he_f32 > 0.5)
    mx_ctr = _mask_centroid_or_none(mx_resized > 0.5)
    m = np.eye(2, 3, dtype=np.float32)
    if he_ctr is not None and mx_ctr is not None:
        m[0, 2] = float(he_ctr[0] - mx_ctr[0])
        m[1, 2] = float(he_ctr[1] - mx_ctr[1])
    return m


def register_he_mx_affine(  # pylint: disable=unused-argument
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Compute affine warp M_full (2x3) mapping H&E full-res -> MX full-res."""
    he_ov_h, he_ov_w = he_mask.shape
    mx_ov_h, mx_ov_w = mx_mask.shape

    mx_resized = cv2.resize(
        mx_mask.astype(np.float32), (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    )
    he_f32 = he_mask.astype(np.float32)
    he_f32 = cv2.GaussianBlur(he_f32, (5, 5), 0)
    mx_resized = cv2.GaussianBlur(mx_resized, (5, 5), 0)

    m_ov = _centroid_init_m_ov(he_f32, mx_resized)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, m_ov = cv2.findTransformECC(
            he_f32, mx_resized, m_ov, cv2.MOTION_AFFINE, criteria
        )
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: ECC registration failed ({e}). Falling back to mpp scale.")
        scale = he_w / mx_w
        return np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32)

    # ECC occasionally leaves a small global translation bias on sparse masks.
    # Re-center warped MX tissue centroid to the H&E centroid (capped).
    mx_warped = cv2.warpAffine(
        mx_resized,
        m_ov.astype(np.float32),
        (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    he_ctr = _mask_centroid_or_none(he_f32 > 0.5)
    mx_ctr = _mask_centroid_or_none(mx_warped > 0.5)
    if he_ctr is not None and mx_ctr is not None:
        dx = float(mx_ctr[0] - he_ctr[0])
        dy = float(mx_ctr[1] - he_ctr[1])
        max_recentre_px = 12.0
        shift_mag = float(np.hypot(dx, dy))
        if shift_mag > max_recentre_px and shift_mag > 1e-9:
            scale = max_recentre_px / shift_mag
            dx *= scale
            dy *= scale
        m_ov[0, 2] += dx
        m_ov[1, 2] += dy

    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h

    # ECC runs on `mx_resized`, where MX overview is resized onto HE overview grid.
    # OpenCV resize uses pixel-center mapping:
    #   x_resized = rx * (x_mx_ov + 0.5) - 0.5
    #   y_resized = ry * (y_mx_ov + 0.5) - 0.5
    # Convert ECC translation terms back from resized-MX coords to original
    # MX overview coords before lifting to full-resolution pixels.
    tx_ov = (float(m_ov[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov[1, 2]) + 0.5) / ry - 0.5
    m_full = np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, tx_ov * ds],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, ty_ov * ds],
        ],
        dtype=np.float32,
    )
    return m_full


def register_he_mx_affine_intensity(
    he_gray_ov: np.ndarray,
    mx_dna_ov: np.ndarray,
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Affine registration using actual image intensities instead of binary masks.

    Parameters
    ----------
    he_gray_ov : float32 (he_ov_h, he_ov_w) normalized HE grayscale at overview res.
    mx_dna_ov  : float32 (mx_ov_h, mx_ov_w) normalized MX DNA channel at overview res.
    he_mask    : bool (he_ov_h, he_ov_w) tissue mask for centroid init.
    mx_mask    : bool (mx_ov_h, mx_ov_w) tissue mask for centroid init.
    ds         : overview downsample factor.
    he_h/w, mx_h/w : full-resolution image dimensions.

    Returns
    -------
    m_full : float32 (2, 3) affine mapping H&E full-res -> MX full-res.
    """
    he_ov_h, he_ov_w = he_gray_ov.shape
    mx_ov_h, mx_ov_w = mx_dna_ov.shape

    # Resize MX DNA to HE overview grid (same as mask-based ECC)
    mx_resized = cv2.resize(
        mx_dna_ov, (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    )
    # Resize MX mask for centroid init
    mx_mask_resized = cv2.resize(
        mx_mask.astype(np.float32), (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    ) > 0.5

    # Mild blur to smooth noise without removing texture
    he_f = cv2.GaussianBlur(he_gray_ov.astype(np.float32), (3, 3), 0)
    mx_f = cv2.GaussianBlur(mx_resized.astype(np.float32), (3, 3), 0)

    # Centroid init from tissue masks
    m_ov = _centroid_init_m_ov(
        he_mask.astype(np.float32), mx_mask_resized.astype(np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, m_ov = cv2.findTransformECC(he_f, mx_f, m_ov, cv2.MOTION_AFFINE, criteria)
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: intensity ECC failed ({e}). Returning centroid-init scale-only.")
        scale = he_w / mx_w
        return np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32)

    # Re-centre cap (same logic as register_he_mx_affine)
    mx_warped = cv2.warpAffine(
        mx_f, m_ov.astype(np.float32), (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    he_ctr = _mask_centroid_or_none(he_mask > 0)
    mx_ctr = _mask_centroid_or_none(mx_warped > mx_warped.mean() + 1e-6)
    if he_ctr is not None and mx_ctr is not None:
        dx = float(mx_ctr[0] - he_ctr[0])
        dy = float(mx_ctr[1] - he_ctr[1])
        shift_mag = float(np.hypot(dx, dy))
        max_recentre_px = 12.0
        if shift_mag > max_recentre_px and shift_mag > 1e-9:
            scale = max_recentre_px / shift_mag
            dx *= scale
            dy *= scale
        m_ov[0, 2] += dx
        m_ov[1, 2] += dy

    # Convert overview → full-res (same pixel-center formula as register_he_mx_affine)
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    tx_ov = (float(m_ov[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov[1, 2]) + 0.5) / ry - 0.5
    return np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, tx_ov * ds],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, ty_ov * ds],
        ],
        dtype=np.float32,
    )


def register_he_mx_orb(
    he_gray_u8: np.ndarray,
    mx_dna_u8: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
    n_features: int = 2000,
    ratio_thresh: float = 0.75,
    min_inliers: int = 8,
) -> np.ndarray | None:
    """Affine registration via ORB keypoints + RANSAC.

    Parameters
    ----------
    he_gray_u8 : uint8 (he_ov_h, he_ov_w) H&E grayscale at overview resolution.
    mx_dna_u8  : uint8 (he_ov_h, he_ov_w) MX DNA channel resized to HE overview grid.
    ds         : overview downsample factor.
    he_h/w, mx_h/w : full-resolution dimensions.
    n_features : ORB feature budget.
    ratio_thresh : Lowe ratio test threshold.
    min_inliers : minimum RANSAC inliers to accept the result.

    Returns
    -------
    m_full : float32 (2, 3) mapping H&E full-res -> MX full-res, or None if failed.
    """
    he_ov_h, he_ov_w = he_gray_u8.shape
    mx_ov_h = int(round(he_ov_h * mx_h / he_h))
    mx_ov_w = int(round(he_ov_w * mx_w / he_w))

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(he_gray_u8, None)
    kp2, des2 = orb.detectAndCompute(mx_dna_u8, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  WARNING: ORB found too few keypoints.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:  # pylint: disable=catching-non-exception
        return None

    good = [
        m for m, n in raw_matches
        if len([m, n]) == 2 and m.distance < ratio_thresh * n.distance
    ]
    if len(good) < min_inliers:
        print(f"  WARNING: ORB ratio test left only {len(good)} matches (need {min_inliers}).")
        return None

    pts_he = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_mx = np.float32([kp2[m.trainIdx].pt for m in good])

    # estimateAffine2D: find M s.t. M * pts_mx ≈ pts_he (HE-overview space)
    m_ov, inliers = cv2.estimateAffine2D(
        pts_mx, pts_he, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if m_ov is None or n_inliers < min_inliers:
        print(f"  WARNING: ORB RANSAC failed or too few inliers ({n_inliers}).")
        return None

    m_ov = m_ov.astype(np.float32)

    # Convert HE-overview-space affine (maps MX-resized -> HE) to full-res H&E -> MX
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    tx_ov = (float(m_ov[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov[1, 2]) + 0.5) / ry - 0.5
    return np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, tx_ov * ds],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, ty_ov * ds],
        ],
        dtype=np.float32,
    )


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
    """Apply the 2x3 affine m_full to (x0, y0) and return rounded (x_mx, y_mx)."""
    pt = np.array([x0, y0, 1.0], dtype=np.float64)
    result = m_full.astype(np.float64) @ pt
    return int(round(result[0])), int(round(result[1]))


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
