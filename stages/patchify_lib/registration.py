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


def _phase_corr_init_m_ov(
    he_f32: np.ndarray,
    mx_resized: np.ndarray,
    min_response: float = 0.02,
) -> np.ndarray:
    """Return 2x3 ECC init using phase correlation for pixel-accurate translation.

    Phase correlation finds the translational shift via cross-power spectrum in
    O(N log N) and is far more precise than centroid differencing for the
    forward-map convention used by findTransformECC (warpMatrix maps mx → he).

    Falls back to centroid init when the phase-correlation response is too low
    (e.g., nearly empty masks or near-uniform images).
    """
    try:
        (dx, dy), response = cv2.phaseCorrelate(he_f32, mx_resized)
    except cv2.error:  # pylint: disable=catching-non-exception
        return _centroid_init_m_ov(he_f32, mx_resized)

    if float(response) < min_response:
        return _centroid_init_m_ov(he_f32, mx_resized)

    m = np.eye(2, 3, dtype=np.float32)
    m[0, 2] = float(dx)
    m[1, 2] = float(dy)
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

    m_ov = _phase_corr_init_m_ov(he_f32, mx_resized)
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


def refine_affine_fine_scale(
    he_mask_fine: np.ndarray,
    mx_mask_fine: np.ndarray,
    m_full_coarse: np.ndarray,
    ds_fine: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Refine a coarse affine by running ECC at a finer overview downsample.

    The coarse m_full (from ds=64) captures rotation and scale well but
    cannot resolve sub-pixel translation errors due to the coarse grid.
    Running ECC at ds_fine (e.g. 8) with the coarse result as initialiser
    gives 8x more precision per overview pixel and resolves translation to
    within ~ds_fine HE pixels.

    Parameters
    ----------
    he_mask_fine : bool (he_h//ds_fine, he_w//ds_fine) tissue mask.
    mx_mask_fine : bool (mx_h//ds_fine, mx_w//ds_fine) tissue mask.
    m_full_coarse : float32 (2, 3) coarse H&E full-res -> MX full-res affine.
    ds_fine : int  downsample factor used to build the fine masks.
    he_h/w, mx_h/w : full-resolution image dimensions.

    Returns
    -------
    m_full_fine : float32 (2, 3) refined H&E full-res -> MX full-res affine,
                  or m_full_coarse unchanged if ECC fails.
    """
    he_ov_h, he_ov_w = he_mask_fine.shape
    mx_ov_h, mx_ov_w = mx_mask_fine.shape

    mx_resized = cv2.resize(
        mx_mask_fine.astype(np.float32), (he_ov_w, he_ov_h),
        interpolation=cv2.INTER_LINEAR,
    )
    he_f32 = cv2.GaussianBlur(he_mask_fine.astype(np.float32), (5, 5), 0)
    mx_f32 = cv2.GaussianBlur(mx_resized, (5, 5), 0)

    # ECC runs on mx_resized (MX mask resized to HE overview dimensions).
    # The warp matrix must map HE_ov coords -> mx_resized coords.
    # m_full maps H&E full-res -> MX full-res; the linear part carries through
    # unchanged in overview space (ds cancels), but the output axis must be
    # scaled by rx,ry to land in the resized-MX (= HE-sized) pixel grid.
    # Concretely:
    #   x_mx_ov  ≈  a * x_he_ov + b * y_he_ov + tx / ds_fine
    #   x_resized = (x_mx_ov + 0.5) * rx - 0.5
    #             ≈  a*rx * x_he_ov + b*rx * y_he_ov + tx*rx/ds_fine
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    a, b, tx = float(m_full_coarse[0, 0]), float(m_full_coarse[0, 1]), float(m_full_coarse[0, 2])
    c, d, ty = float(m_full_coarse[1, 0]), float(m_full_coarse[1, 1]), float(m_full_coarse[1, 2])
    m_ov_init = np.array(
        [
            [a * rx, b * rx, tx * rx / ds_fine],
            [c * ry, d * ry, ty * ry / ds_fine],
        ],
        dtype=np.float32,
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        _, m_ov_fine = cv2.findTransformECC(
            he_f32, mx_f32, m_ov_init.astype(np.float32),
            cv2.MOTION_AFFINE, criteria,
        )
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: fine-scale ECC failed ({e}). Keeping coarse result.")
        return m_full_coarse

    tx_ov = (float(m_ov_fine[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov_fine[1, 2]) + 0.5) / ry - 0.5
    return np.array(
        [
            [m_ov_fine[0, 0] / rx, m_ov_fine[0, 1] / rx, tx_ov * ds_fine],
            [m_ov_fine[1, 0] / ry, m_ov_fine[1, 1] / ry, ty_ov * ds_fine],
        ],
        dtype=np.float32,
    )


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
    fallback_m_full: np.ndarray | None = None,
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

    # Phase-corr init from tissue masks (more precise than centroid for translation)
    m_ov = _phase_corr_init_m_ov(
        he_mask.astype(np.float32), mx_mask_resized.astype(np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, m_ov = cv2.findTransformECC(he_f, mx_f, m_ov, cv2.MOTION_AFFINE, criteria)
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: intensity ECC failed ({e}). Returning fallback transform.")
        if fallback_m_full is not None:
            return fallback_m_full
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
        match_pair[0]
        for match_pair in raw_matches
        if len(match_pair) == 2
        and match_pair[0].distance < ratio_thresh * match_pair[1].distance
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
    he_f32 = cv2.GaussianBlur((he_mask > 0).astype(np.float32), (0, 0), sigmaX=1.5)
    mx_aff = cv2.warpAffine(
        (mx_mask > 0).astype(np.float32),
        m_ov.astype(np.float32),
        (he_mask.shape[1], he_mask.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mx_f32 = cv2.GaussianBlur(mx_aff.astype(np.float32), (0, 0), sigmaX=1.5)

    # NOTE: calcOpticalFlowFarneback in OpenCV 4.13 is broken with float32 input
    # (returns exactly zero). Convert to uint8 as a workaround.
    he_u8 = np.clip(he_f32 * 255, 0, 255).astype(np.uint8)
    mx_u8 = np.clip(mx_f32 * 255, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        mx_u8,
        he_u8,
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
    he_f = he_f32
    mx_f = mx_f32

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


def estimate_deformable_field_intensity(
    he_gray_ov: np.ndarray,
    mx_dna_ov: np.ndarray,
    he_mask_fine: np.ndarray,
    mx_mask_fine: np.ndarray,
    m_full: np.ndarray,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> dict[str, object]:
    """Estimate dense flow using fine-scale tissue masks (he_mask_fine / mx_mask_fine).

    Runs Farneback on binary tissue masks at a fine downsample (e.g. ds=8)
    rather than ds=64, providing 8× more spatial resolution for the flow.
    At ds=8 tissue boundaries (glands, vessels) are much sharper than at ds=64,
    giving Farneback better gradient signal to detect local deformation.

    Parameters
    ----------
    he_gray_ov : unused (reserved for future intensity-based flow).
    mx_dna_ov  : unused (reserved for future intensity-based flow).
    he_mask_fine : bool (he_ov_h, he_ov_w)  H&E tissue mask at fine scale.
    mx_mask_fine : bool (mx_ov_h, mx_ov_w)  MX tissue mask at fine scale.
    m_full     : float32 (2, 3)  H&E full-res -> MX full-res affine.
    he_h/w, mx_h/w : full-resolution dimensions.

    Returns
    -------
    dict with keys: flow_dx_ov, flow_dy_ov, iou_affine, iou_deformable,
                    mean_disp_ov, max_disp_ov.
                    Flow is stored in fine-scale overview-pixel units and is
                    automatically rescaled when applied via
                    read_multiplex_patch_affine_deform (uses he_full_w / w_ov).
    """
    he_ov_h, he_ov_w = he_mask_fine.shape
    mx_ov_h, mx_ov_w = mx_mask_fine.shape

    # Build m_ov: maps HE fine-overview coords -> MX fine-overview coords.
    he_sx = he_w / max(1, he_ov_w)
    he_sy = he_h / max(1, he_ov_h)
    mx_inv_sx = mx_ov_w / max(1, mx_w)
    mx_inv_sy = mx_ov_h / max(1, mx_h)
    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    m_ov = np.array(
        [
            [a * he_sx * mx_inv_sx, b * he_sy * mx_inv_sx, tx * mx_inv_sx],
            [c * he_sx * mx_inv_sy, d * he_sy * mx_inv_sy, ty * mx_inv_sy],
        ],
        dtype=np.float32,
    )

    # Warp MX binary mask to H&E fine overview space.
    he_f = cv2.GaussianBlur((he_mask_fine > 0).astype(np.float32), (0, 0), sigmaX=1.5)
    mx_aff = cv2.warpAffine(
        (mx_mask_fine > 0).astype(np.float32),
        m_ov,
        (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mx_f32 = cv2.GaussianBlur(mx_aff.astype(np.float32), (0, 0), sigmaX=1.5)
    mx_f = mx_f32

    # NOTE: calcOpticalFlowFarneback in OpenCV 4.13 is broken with float32 input.
    he_u8 = np.clip(he_f * 255, 0, 255).astype(np.uint8)
    mx_u8 = np.clip(mx_f32 * 255, 0, 255).astype(np.uint8)

    # Farneback at fine scale: sharper tissue boundaries → more gradient signal.
    flow = cv2.calcOpticalFlowFarneback(
        mx_u8,
        he_u8,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=41,
        iterations=6,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.5)

    max_disp = 0.08 * float(min(he_ov_h, he_ov_w))
    mag = np.hypot(flow[..., 0], flow[..., 1])
    scale_arr = np.ones_like(mag, dtype=np.float32)
    valid = mag > (max_disp + 1e-6)
    scale_arr[valid] = max_disp / mag[valid]
    flow[..., 0] *= scale_arr
    flow[..., 1] *= scale_arr

    mx_aff_bin = mx_aff > 0.5
    mx_def = _apply_inverse_flow(mx_aff.astype(np.float32), flow[..., 0], flow[..., 1])
    mx_def_bin = mx_def > 0.5
    iou_aff = _iou(he_mask_fine > 0, mx_aff_bin)
    iou_def = _iou(he_mask_fine > 0, mx_def_bin)

    mag_final = np.hypot(flow[..., 0], flow[..., 1])
    return {
        "flow_dx_ov": flow[..., 0].astype(np.float32),
        "flow_dy_ov": flow[..., 1].astype(np.float32),
        "iou_affine": float(iou_aff),
        "iou_deformable": float(iou_def),
        "mean_disp_ov": float(np.mean(mag_final)),
        "max_disp_ov": float(np.max(mag_final)),
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
