"""
refine_registration.py — Stage 2.5: CellViT-guided ICP + TPS registration refinement.

Uses CellViT-detected H&E nuclei and CSV cell centroids (ground-truth MX space)
as landmark correspondences to fit an ICP-refined affine followed by a
thin-plate-spline (TPS) warp.  All matching is done in H&E pixel space for
interpretability.  Produces pixel-aligned H&E+multiplex patch pairs for
multimodal model training.

Pipeline:
  Stage 1 → processed_crop/
  Stage 2 (CellViT) → processed_crop/cellvit/*.json
  Stage 2.5 (this) → writes processed_crop/index_icp_tps.json + updates multiplex/*.npy

Algorithm
---------
1. Build global H&E centroid cloud from CellViT JSON files.
2. Load CSV centroids (Xt, Yt in µm) → MX px (÷ csv_mpp) → H&E px (inv m_full).
3. ICP in H&E space: iteratively align CellViT cloud toward CSV-in-HE cloud.
4. Post-ICP mutual nearest-neighbour matching + RANSAC affine outlier filter.
5. Fit TPS (H&E px → MX px) on RANSAC inliers with scipy RBFInterpolator.
6. Re-extract multiplex patches with TPS warp (cv2.remap).
7. Write updated index file with ICP matrix, TPS control points and metadata.

CLI
---
python -m stages.refine_registration \\
  --processed processed_crop/ \\
  --he-image data/WD-76845-096-crop.ome.tif \\
  --multiplex-image data/WD-76845-097-crop.ome.tif \\
  --csv data/WD-76845-097.csv \\
  --csv-mpp 0.65 \\
  --distance-gate 20 \\
  --max-tps-points 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.interpolate
import tifffile

from utils.ome import get_image_dims, get_ome_mpp, open_zarr_store

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Build global H&E cell geometry from CellViT JSON files
# ---------------------------------------------------------------------------


def load_he_cells(
    cellvit_dir: pathlib.Path,
    patches: list[dict],
    coord_scale: float,
    min_contour_vertices: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Load H&E centroids and contours from CellViT JSONs.

    Returns
    -------
    he_pts_he : (N, 2) float64
        Cell centroids in H&E full-resolution pixel coordinates.
    he_pts_mx : (N, 2) float64
        Same centroids scaled into MX pixel coordinates.
    he_contours_he : list[(K_i, 2) float64 | None], length N
        Per-cell contour vertices in H&E coordinates. ``None`` when contour is
        missing/invalid for that cell.
    """
    he_centroids: list[tuple[float, float]] = []
    he_contours_he: list[np.ndarray | None] = []

    for patch_meta in patches:
        x0 = int(patch_meta["x0"])
        y0 = int(patch_meta["y0"])
        patch_id = f"{x0}_{y0}"
        json_path = cellvit_dir / f"{patch_id}.json"
        if not json_path.exists():
            continue
        with json_path.open() as fh:
            data = json.load(fh)

        for cell in data.get("cells", []):
            centroid = cell.get("centroid")
            if centroid is None or len(centroid) < 2:
                continue

            lx = float(centroid[0])
            ly = float(centroid[1])
            gx_he = float(x0) + lx
            gy_he = float(y0) + ly
            he_centroids.append((gx_he, gy_he))

            contour = cell.get("contour")
            contour_he: np.ndarray | None = None
            if (
                isinstance(contour, list)
                and len(contour) >= min_contour_vertices
            ):
                arr = np.asarray(contour, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    arr = arr[:, :2]
                    arr[:, 0] += float(x0)
                    arr[:, 1] += float(y0)
                    if len(arr) >= min_contour_vertices:
                        contour_he = arr
            he_contours_he.append(contour_he)

    if not he_centroids:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )

    he_pts_he = np.asarray(he_centroids, dtype=np.float64)
    he_pts_mx = he_pts_he * coord_scale
    return he_pts_he, he_pts_mx, he_contours_he


def load_he_centroids(
    cellvit_dir: pathlib.Path,
    patches: list[dict],
    coord_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible centroid loader (without contour output)."""
    he_pts_he, he_pts_mx, _ = load_he_cells(
        cellvit_dir=cellvit_dir,
        patches=patches,
        coord_scale=coord_scale,
    )
    return he_pts_he, he_pts_mx


# ---------------------------------------------------------------------------
# Step 2: Load MX centroid cloud
# ---------------------------------------------------------------------------


def load_mx_centroids(
    csv_path: pathlib.Path,
    csv_mpp: float = 1.0,
    crop_origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, scipy.spatial.KDTree]:
    """Load (Xt, Yt) MX centroids from CSV and build KDTree.

    If csv_mpp != 1.0, coordinates are divided by csv_mpp to convert from µm to MX px.
    If crop_origin (ox, oy) is given, it is subtracted after mpp conversion to shift
    full-slide MX px coords into crop-image MX px space.
    """
    df = pd.read_csv(csv_path)
    if "Xt" not in df.columns or "Yt" not in df.columns:
        raise ValueError(
            f"CSV must have 'Xt' and 'Yt' columns; found: {list(df.columns)[:10]}"
        )
    mx_pts = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
    if csv_mpp != 1.0:
        mx_pts = mx_pts / csv_mpp
    if crop_origin is not None:
        mx_pts = mx_pts - np.array(crop_origin, dtype=np.float64)
    kdtree = scipy.spatial.KDTree(mx_pts)
    return mx_pts, kdtree


# ---------------------------------------------------------------------------
# Step 2 (new): Convert CSV centroids µm → MX px → H&E px
# ---------------------------------------------------------------------------


def csv_to_he_coords(
    csv_path: pathlib.Path,
    m_full: np.ndarray,
    csv_mpp: float = 1.0,
    crop_origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CSV cell centroids (µm) to both MX px and H&E px.

    Parameters
    ----------
    csv_path    : Path to CSV with 'Xt', 'Yt' columns (in µm).
    m_full      : 2×3 affine mapping H&E px → MX px.
    csv_mpp     : µm per pixel of the CSV coordinate space (divides Xt/Yt).
    crop_origin : (ox, oy) offset in full-slide MX px to subtract after mpp
                  conversion, for when multiplex-image is a crop.

    Returns
    -------
    mx_pts : (N, 2) float64  — centroids in MX pixel space
    he_pts : (N, 2) float64  — centroids in H&E pixel space via inv(m_full)
    """
    df = pd.read_csv(csv_path)
    if "Xt" not in df.columns or "Yt" not in df.columns:
        raise ValueError(
            f"CSV must have 'Xt' and 'Yt' columns; found: {list(df.columns)[:10]}"
        )
    mx_pts = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
    if csv_mpp != 1.0:
        mx_pts = mx_pts / csv_mpp
    if crop_origin is not None:
        mx_pts = mx_pts - np.array(crop_origin, dtype=np.float64)

    # MX px → H&E px via inv(m_full)
    m3 = np.vstack([m_full, [0.0, 0.0, 1.0]])  # 3×3
    m3_inv = np.linalg.inv(m3)  # 3×3 inverse
    m_inv = m3_inv[:2, :]  # 2×3: MX px → H&E px

    ones = np.ones((len(mx_pts), 1), dtype=np.float64)
    he_pts = (m_inv @ np.hstack([mx_pts, ones]).T).T  # (N, 2)

    return mx_pts, he_pts


# ---------------------------------------------------------------------------
# Crop-origin resolution
# ---------------------------------------------------------------------------


def _coerce_origin_pair(value: object) -> tuple[float, float] | None:
    """Normalize origin-like payload to ``(x, y)`` float tuple."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    if isinstance(value, dict):
        if "x0" in value and "y0" in value:
            try:
                return float(value["x0"]), float(value["y0"])
            except (TypeError, ValueError):
                return None
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except (TypeError, ValueError):
                return None

    return None


def resolve_mx_crop_origin(
    index_payload: dict,
    cli_origin: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Resolve MX crop origin with precedence: CLI -> index.json metadata."""
    if cli_origin is not None:
        return float(cli_origin[0]), float(cli_origin[1])

    origin = _coerce_origin_pair(index_payload.get("mx_crop_origin"))
    if origin is not None:
        return origin

    crop_region = index_payload.get("crop_region")
    if isinstance(crop_region, dict):
        for key in ("mx_origin", "mx_crop_origin", "mx"):
            origin = _coerce_origin_pair(crop_region.get(key))
            if origin is not None:
                return origin

    return None


# ---------------------------------------------------------------------------
# Patch ROI helpers (HE space)
# ---------------------------------------------------------------------------


def patch_roi_bbox_he(
    patches: list[dict],
    patch_size_default: int = 256,
) -> tuple[float, float, float, float]:
    """Bounding box of all Stage-1 patches in H&E pixel space.

    Returns (x_min, x_max, y_min, y_max), where max bounds are exclusive.
    """
    if not patches:
        raise ValueError("Cannot compute patch ROI bbox: index has no patches.")

    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf

    for patch_meta in patches:
        x0 = float(patch_meta["x0"])
        y0 = float(patch_meta["y0"])
        patch_size = float(patch_meta.get("patch_size", patch_size_default))
        x_min = min(x_min, x0)
        y_min = min(y_min, y0)
        x_max = max(x_max, x0 + patch_size)
        y_max = max(y_max, y0 + patch_size)

    return float(x_min), float(x_max), float(y_min), float(y_max)


def filter_csv_to_patch_roi(
    mx_pts: np.ndarray,
    csv_in_he: np.ndarray,
    roi_bbox_he: tuple[float, float, float, float],
    margin_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep CSV points whose HE-projected location is inside patch ROI + margin."""
    x_min, x_max, y_min, y_max = roi_bbox_he
    x0 = x_min - float(margin_px)
    x1 = x_max + float(margin_px)
    y0 = y_min - float(margin_px)
    y1 = y_max + float(margin_px)

    keep = (
        (csv_in_he[:, 0] >= x0)
        & (csv_in_he[:, 0] <= x1)
        & (csv_in_he[:, 1] >= y0)
        & (csv_in_he[:, 1] <= y1)
    )
    return mx_pts[keep], csv_in_he[keep], keep


# ---------------------------------------------------------------------------
# Step 3 (new): ICP in H&E space
# ---------------------------------------------------------------------------


def affine_icp(
    src_he: np.ndarray,
    dst_he: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
    distance_gate: float | None = None,
) -> tuple[np.ndarray, int, int]:
    """Point-to-point affine ICP aligning CellViT cloud toward CSV-in-H&E cloud.

    Each iteration:
      1. Nearest-neighbour match src_curr → dst_he (optional distance gate).
      2. Fit affine via cv2.estimateAffine2D (RANSAC).
      3. Apply step transform to src_curr; accumulate into M_icp.
      4. Converge when ‖M_step[:,:2] − I‖_F + ‖M_step[:,2]‖ < tol.

    Parameters
    ----------
    src_he        : (N, 2) CellViT centroids in H&E px (aligned toward dst).
    dst_he        : (M, 2) CSV centroids in H&E px (fixed target).
    max_iter      : Maximum number of ICP iterations.
    tol           : Convergence threshold on the step-transform magnitude.
    distance_gate : Maximum H&E px distance for NN matching per iteration.

    Returns
    -------
    M_icp     : (2, 3) cumulative affine  (H&E → corrected H&E)
    n_matches : Number of matched pairs in the final iteration.
    n_iters   : Number of iterations performed.
    """
    dst_kdtree = scipy.spatial.KDTree(dst_he)
    M_cum = np.eye(2, 3, dtype=np.float64)  # starts as identity
    src_curr = src_he.astype(np.float64).copy()
    n_matches = 0
    n_iters = 0

    for i in range(max_iter):
        dists, fwd_idx = dst_kdtree.query(src_curr, workers=-1)

        if distance_gate is not None:
            mask = dists <= distance_gate
            if mask.sum() < 4:
                log.warning(
                    "ICP iter %d: only %d pts within gate (need >= 4); stopping.",
                    i,
                    mask.sum(),
                )
                break
            src_m = src_curr[mask]
            dst_m = dst_he[fwd_idx[mask]]
        else:
            src_m = src_curr
            dst_m = dst_he[fwd_idx]

        n_matches = len(src_m)

        M_step, _ = cv2.estimateAffine2D(
            src_m.astype(np.float32),
            dst_m.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        if M_step is None:
            log.warning("ICP iter %d: estimateAffine2D returned None; stopping.", i)
            break

        M_step = M_step.astype(np.float64)

        # Apply step to current cloud
        ones = np.ones((len(src_curr), 1), dtype=np.float64)
        src_curr = (M_step @ np.hstack([src_curr, ones]).T).T

        # Accumulate: M_cum = M_step ∘ M_cum
        M_step3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
        M_cum3 = np.vstack([M_cum, [0.0, 0.0, 1.0]])
        M_cum = (M_step3 @ M_cum3)[:2, :]

        delta = np.linalg.norm(M_step[:, :2] - np.eye(2)) + np.linalg.norm(M_step[:, 2])
        n_iters = i + 1
        if delta < tol:
            log.info("  ICP converged at iteration %d (delta=%.2e).", n_iters, delta)
            break
    else:
        log.warning("ICP did not converge within %d iterations.", max_iter)

    return M_cum, n_matches, n_iters


# ---------------------------------------------------------------------------
# Step 3: Affine application helper + new HE-space matching
# ---------------------------------------------------------------------------


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine to (N, 2) points. Returns (N, 2)."""
    a, b, tx = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    c, d, ty = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    x = a * pts[:, 0] + b * pts[:, 1] + tx
    y = c * pts[:, 0] + d * pts[:, 1] + ty
    return np.column_stack([x, y])


def match_centroids_he(
    icp_he: np.ndarray,
    he_pts_he: np.ndarray,
    csv_in_he: np.ndarray,
    mx_pts: np.ndarray,
    distance_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ICP-aligned CellViT cloud to CSV-in-H&E, returning TPS src/dst.

    Matching is done in H&E space (interpretable). Mutual nearest-neighbour
    check is applied to reduce false matches.

    Parameters
    ----------
    icp_he     : (N, 2) ICP-transformed CellViT centroids in H&E px.
    he_pts_he  : (N, 2) original CellViT centroids in H&E px (TPS source).
    csv_in_he  : (M, 2) CSV centroids in H&E px (matching target).
    mx_pts     : (M, 2) CSV centroids in MX px (TPS destination).
    distance_gate : Max H&E px distance for a valid match.

    Returns
    -------
    src_he : (K, 2) original CellViT H&E coords for matched pairs.
    dst_mx : (K, 2) matched CSV MX coords for matched pairs.
    """
    csv_kdtree = scipy.spatial.KDTree(csv_in_he)

    # Forward: each ICP-aligned CellViT → nearest CSV-in-HE
    dists, fwd_idx = csv_kdtree.query(icp_he, workers=-1)
    valid_mask = dists <= distance_gate

    if valid_mask.sum() == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Mutual NN check
    icp_kdtree = scipy.spatial.KDTree(icp_he[valid_mask])
    matched_csv_he = csv_in_he[fwd_idx[valid_mask]]
    _, back_idx = icp_kdtree.query(matched_csv_he, workers=-1)

    valid_indices = np.where(valid_mask)[0]
    is_mutual = back_idx == np.arange(valid_mask.sum())

    src_he = he_pts_he[valid_indices[is_mutual]]
    dst_mx = mx_pts[fwd_idx[valid_indices[is_mutual]]]

    return src_he, dst_mx


def _point_to_cell_distance(
    point_he: np.ndarray,
    cell_center_he: np.ndarray,
    contour_he: np.ndarray | None,
) -> float:
    """Distance from a CSV point to a CellViT cell representation in H&E space.

    If contour is available, distance is 0 when point is inside contour and
    otherwise Euclidean distance to the contour boundary. If contour is missing,
    falls back to centroid Euclidean distance.
    """
    if contour_he is None or len(contour_he) < 3:
        return float(np.linalg.norm(point_he - cell_center_he))

    signed_dist = cv2.pointPolygonTest(
        contour_he.astype(np.float32),
        (float(point_he[0]), float(point_he[1])),
        True,
    )
    if signed_dist >= 0:
        return 0.0
    return float(-signed_dist)


def match_cells_he_contour(
    m_icp: np.ndarray,
    he_pts_he: np.ndarray,
    he_contours_he: list[np.ndarray | None],
    csv_in_he: np.ndarray,
    mx_pts: np.ndarray,
    distance_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Contour-aware mutual nearest-neighbour matching in H&E space.

    Matching score for each (cell, CSV) pair:
      - 0 if CSV point is inside transformed CellViT contour;
      - otherwise distance to transformed contour boundary;
      - fallback to centroid distance when contour is missing.

    The function applies mutual nearest-neighbour selection under this distance
    metric and returns source/destination pairs for downstream RANSAC/TPS.
    """
    n_he = len(he_pts_he)
    n_csv = len(csv_in_he)
    if n_he == 0 or n_csv == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    if len(he_contours_he) != n_he:
        raise ValueError(
            f"len(he_contours_he) ({len(he_contours_he)}) != len(he_pts_he) ({n_he})"
        )

    icp_he = apply_affine(m_icp, he_pts_he)

    # Transform all available contours with the same ICP affine.
    icp_contours_he: list[np.ndarray | None] = []
    for contour in he_contours_he:
        if contour is None or len(contour) < 3:
            icp_contours_he.append(None)
            continue
        icp_contours_he.append(apply_affine(m_icp, contour))

    # Pairwise distance matrix under contour-aware metric.
    # Typical sizes here are manageable (e.g., ~1.5k x ~400).
    dist = np.empty((n_he, n_csv), dtype=np.float32)
    for i in range(n_he):
        c_i = icp_he[i]
        contour_i = icp_contours_he[i]
        if contour_i is None:
            dist[i, :] = np.linalg.norm(csv_in_he - c_i, axis=1)
            continue
        for j in range(n_csv):
            dist[i, j] = _point_to_cell_distance(csv_in_he[j], c_i, contour_i)

    # Forward NN: each HE cell picks one CSV.
    fwd_j = np.argmin(dist, axis=1)
    fwd_d = dist[np.arange(n_he), fwd_j]
    valid_he = fwd_d <= float(distance_gate)
    if not np.any(valid_he):
        return np.empty((0, 2)), np.empty((0, 2))

    # Reverse NN: each CSV picks one HE cell.
    rev_i = np.argmin(dist, axis=0)

    src_idx: list[int] = []
    dst_idx: list[int] = []
    for i in np.where(valid_he)[0]:
        j = int(fwd_j[i])
        if rev_i[j] == i and dist[i, j] <= float(distance_gate):
            src_idx.append(int(i))
            dst_idx.append(j)

    if not src_idx:
        return np.empty((0, 2)), np.empty((0, 2))

    src_he = he_pts_he[np.asarray(src_idx, dtype=np.int64)]
    dst_mx = mx_pts[np.asarray(dst_idx, dtype=np.int64)]
    return src_he, dst_mx


# ---------------------------------------------------------------------------
# (kept for backward-compat) old MX-space matching — no longer called by main
# ---------------------------------------------------------------------------


def match_centroids(
    he_pts_he: np.ndarray,
    mx_pts: np.ndarray,
    kdtree: scipy.spatial.KDTree,
    m_full: np.ndarray,
    distance_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(Legacy) Match H&E centroids to MX centroids via m_full + KDTree.

    Uses mutual nearest-neighbour check in MX pixel space.

    Returns
    -------
    src_he : (M, 2) matched H&E centroids in H&E full-res px
    dst_mx : (M, 2) matched MX centroids in MX px
    """
    he_approx_mx = apply_affine(m_full, he_pts_he)

    dists, fwd_idx = kdtree.query(he_approx_mx, workers=-1)
    valid_mask = dists <= distance_gate

    if valid_mask.sum() == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    he_kdtree = scipy.spatial.KDTree(he_approx_mx[valid_mask])
    matched_mx_pts = mx_pts[fwd_idx[valid_mask]]
    _, back_idx = he_kdtree.query(matched_mx_pts, workers=-1)

    valid_indices = np.where(valid_mask)[0]
    is_mutual = back_idx == np.arange(valid_mask.sum())

    src_he = he_pts_he[valid_indices[is_mutual]]
    dst_mx = matched_mx_pts[is_mutual]

    return src_he, dst_mx


# ---------------------------------------------------------------------------
# Step 4: RANSAC affine filtering
# ---------------------------------------------------------------------------


def ransac_filter(
    src_he: np.ndarray,
    dst_mx: np.ndarray,
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter matches using cv2.estimateAffine2D RANSAC. Returns inlier pairs."""
    if len(src_he) < 4:
        log.warning(
            "Too few matches (%d) for RANSAC; skipping RANSAC filter.", len(src_he)
        )
        return src_he, dst_mx

    src_f = src_he.astype(np.float32)
    dst_f = dst_mx.astype(np.float32)
    _, inlier_mask = cv2.estimateAffine2D(
        src_f,
        dst_f,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if inlier_mask is None:
        log.warning("RANSAC returned no transform; returning all matches.")
        return src_he, dst_mx

    mask = inlier_mask.ravel().astype(bool)
    return src_he[mask], dst_mx[mask]


# ---------------------------------------------------------------------------
# Step 5: Fit TPS
# ---------------------------------------------------------------------------


def subsample_uniform(
    src: np.ndarray,
    dst: np.ndarray,
    max_pts: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatially-uniform subsample of control points to at most max_pts."""
    if len(src) <= max_pts:
        return src, dst

    if rng is None:
        rng = np.random.default_rng(42)

    # Divide bounding box into a grid and pick one point per cell
    x_min, y_min = src[:, 0].min(), src[:, 1].min()
    x_max, y_max = src[:, 0].max(), src[:, 1].max()
    n_cells = max_pts
    cell_w = max(1.0, (x_max - x_min) / np.sqrt(n_cells))
    cell_h = max(1.0, (y_max - y_min) / np.sqrt(n_cells))

    cell_i = ((src[:, 0] - x_min) / cell_w).astype(int)
    cell_j = ((src[:, 1] - y_min) / cell_h).astype(int)
    cell_key = cell_i * 100000 + cell_j  # unique cell ID

    selected: list[int] = []
    seen: set[int] = set()
    # Shuffle for fair per-cell selection
    order = rng.permutation(len(src))
    for idx in order:
        k = int(cell_key[idx])
        if k not in seen:
            seen.add(k)
            selected.append(int(idx))
        if len(selected) >= max_pts:
            break

    sel = np.array(selected)
    return src[sel], dst[sel]


def fit_tps(
    src_he: np.ndarray,
    dst_mx: np.ndarray,
    max_tps_points: int = 2000,
) -> tuple[scipy.interpolate.RBFInterpolator, scipy.interpolate.RBFInterpolator]:
    """Fit two TPS interpolators (x and y components).

    Parameters
    ----------
    src_he : (K, 2) H&E full-res control point source coords
    dst_mx : (K, 2) corresponding MX pixel target coords

    Returns
    -------
    tps_x, tps_y : RBFInterpolator objects (thin_plate_spline kernel)
    """
    src_sub, dst_sub = subsample_uniform(src_he, dst_mx, max_tps_points)
    log.info("  Fitting TPS on %d control points.", len(src_sub))

    tps_x = scipy.interpolate.RBFInterpolator(
        src_sub, dst_sub[:, 0], kernel="thin_plate_spline", degree=1
    )
    tps_y = scipy.interpolate.RBFInterpolator(
        src_sub, dst_sub[:, 1], kernel="thin_plate_spline", degree=1
    )
    return tps_x, tps_y


# ---------------------------------------------------------------------------
# Step 6: Re-extract multiplex patches with TPS
# ---------------------------------------------------------------------------


def _make_patch_pixel_grid(x0: int, y0: int, patch_size: int) -> np.ndarray:
    """Return (patch_size*patch_size, 2) grid of H&E full-res pixel coords."""
    uu, vv = np.meshgrid(
        np.arange(patch_size, dtype=np.float64),
        np.arange(patch_size, dtype=np.float64),
    )
    x_he = uu.ravel() + float(x0)
    y_he = vv.ravel() + float(y0)
    return np.column_stack([x_he, y_he])


def read_multiplex_patch_tps(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    he_x0: int,
    he_y0: int,
    patch_size: int,
    tps_x: scipy.interpolate.RBFInterpolator,
    tps_y: scipy.interpolate.RBFInterpolator,
    channel_indices: list[int],
) -> tuple[np.ndarray, bool]:
    """Read MX patch in H&E patch frame using TPS warp.

    Each H&E pixel position is mapped to MX coordinates via the TPS, then
    sampled with bilinear interpolation (cv2.remap).

    Returns
    -------
    out     : (C, patch_size, patch_size) uint16
    inside  : bool — True if all mapped coords are within MX bounds
    """
    n_ch = len(channel_indices)
    out = np.zeros((n_ch, patch_size, patch_size), dtype=np.uint16)

    grid_he = _make_patch_pixel_grid(he_x0, he_y0, patch_size)

    map_mx_x = tps_x(grid_he).reshape(patch_size, patch_size).astype(np.float32)
    map_mx_y = tps_y(grid_he).reshape(patch_size, patch_size).astype(np.float32)

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

    # Clamp to image bounds
    x_min_c = max(0, x_min)
    y_min_c = max(0, y_min)
    x_max_c = min(img_w, x_max)
    y_max_c = min(img_h, y_max)
    src_w = max(1, x_max_c - x_min_c)
    src_h = max(1, y_max_c - y_min_c)

    # Read MX region
    ax_up = axes.upper()
    sl: list[int | slice] = []
    for ax in ax_up:
        if ax == "C":
            sl.append(slice(None))
        elif ax == "Y":
            sl.append(slice(y_min_c, y_min_c + src_h))
        elif ax == "X":
            sl.append(slice(x_min_c, x_min_c + src_w))
        else:
            sl.append(0)

    src_raw = np.array(zarr_store[tuple(sl)])

    # Transpose to (C, H, W)
    active = [ax for ax in ax_up if ax in ("C", "Y", "X")]
    if "C" in active:
        target = [ax for ax in ("C", "Y", "X") if ax in active]
        if active != target:
            perm = [active.index(ax) for ax in target]
            src_raw = src_raw.transpose(perm)
        src_raw = src_raw[channel_indices]
    else:
        target_yx = [ax for ax in ("Y", "X") if ax in active]
        if active != target_yx:
            perm = [active.index(ax) for ax in target_yx]
            src_raw = src_raw.transpose(perm)
        src_raw = np.stack([src_raw] * n_ch, axis=0)

    # Build local coordinate maps
    local_x = (map_mx_x - float(x_min_c)).astype(np.float32)
    local_y = (map_mx_y - float(y_min_c)).astype(np.float32)

    for ch in range(n_ch):
        warped = cv2.remap(
            src_raw[ch].astype(np.float32),
            local_x,
            local_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[ch] = np.clip(np.rint(warped), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return out, inside


def reextract_multiplex_patches(
    mx_tif_path: pathlib.Path,
    processed_dir: pathlib.Path,
    patches: list[dict],
    channel_indices: list[int],
    tps_x: scipy.interpolate.RBFInterpolator,
    tps_y: scipy.interpolate.RBFInterpolator,
    patch_size_default: int = 256,
) -> None:
    """Re-extract all multiplex patches using TPS and overwrite .npy files."""
    mx_dir = processed_dir / "multiplex"

    with tifffile.TiffFile(str(mx_tif_path)) as tif:
        mx_store = open_zarr_store(tif)
        mx_w, mx_h, mx_axes = get_image_dims(tif)

    n = len(patches)
    log.info("Re-extracting %d multiplex patches with TPS ...", n)
    for i, patch_meta in enumerate(patches):
        x0 = int(patch_meta["x0"])
        y0 = int(patch_meta["y0"])
        patch_size = int(patch_meta.get("patch_size", patch_size_default))
        patch_id = f"{x0}_{y0}"

        arr, _inside = read_multiplex_patch_tps(
            zarr_store=mx_store,
            axes=mx_axes,
            img_w=mx_w,
            img_h=mx_h,
            he_x0=x0,
            he_y0=y0,
            patch_size=patch_size,
            tps_x=tps_x,
            tps_y=tps_y,
            channel_indices=channel_indices,
        )

        npy_path = mx_dir / f"{patch_id}.npy"
        np.save(str(npy_path), arr)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            log.info("  %d / %d patches re-extracted.", i + 1, n)


# ---------------------------------------------------------------------------
# Step 7: Update index.json
# ---------------------------------------------------------------------------


def update_index(
    processed_dir: pathlib.Path,
    tps_control_he: np.ndarray,
    tps_control_mx: np.ndarray,
    n_matches: int,
    inlier_fraction: float,
    he_total: int = 0,
    csv_total: int = 0,
    he_match_rate: float = 0.0,
    csv_match_rate: float = 0.0,
    he_inlier_rate: float = 0.0,
    csv_inlier_rate: float = 0.0,
    csv_total_before_roi: int = 0,
    csv_roi_margin_px: float = 0.0,
    csv_roi_bbox_he: tuple[float, float, float, float] | None = None,
    icp_matrix: np.ndarray | None = None,
    icp_n_iters: int = 0,
    icp_n_matches: int = 0,
    cellvit_match_repr: str = "contour",
    he_cells_with_contour: int = 0,
    index_in_name: str = "index.json",
    index_out_name: str = "index.json",
) -> pathlib.Path:
    """Write updated index file with ICP matrix, TPS control points and metadata."""
    index_in_path = processed_dir / index_in_name
    index_out_path = processed_dir / index_out_name
    with index_in_path.open() as fh:
        index = json.load(fh)

    index["registration_mode"] = "icp_tps"
    index["tps_n_matches"] = int(n_matches)
    index["tps_inlier_fraction"] = float(inlier_fraction)
    index["tps_control_he"] = tps_control_he.tolist()
    index["tps_control_mx"] = tps_control_mx.tolist()
    if icp_matrix is not None:
        index["icp_matrix"] = icp_matrix.tolist()
    index["icp_n_iters"] = int(icp_n_iters)
    index["icp_n_matches"] = int(icp_n_matches)
    index["cellvit_match_repr"] = str(cellvit_match_repr)
    index["he_cells_with_contour"] = int(he_cells_with_contour)
    index["he_total_centroids"] = int(he_total)
    index["csv_total_centroids"] = int(csv_total)
    index["he_match_rate"] = float(he_match_rate)
    index["csv_match_rate"] = float(csv_match_rate)
    index["he_inlier_rate"] = float(he_inlier_rate)
    index["csv_inlier_rate"] = float(csv_inlier_rate)
    index["csv_total_before_roi"] = int(csv_total_before_roi)
    index["csv_roi_margin_px"] = float(csv_roi_margin_px)
    if csv_roi_bbox_he is not None:
        x_min, x_max, y_min, y_max = csv_roi_bbox_he
        index["csv_roi_bbox_he"] = {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        }

    with index_out_path.open("w") as fh:
        json.dump(index, fh, indent=2)
    log.info(
        "Wrote %s with ICP+TPS metadata (%d control points).",
        index_out_path,
        len(tps_control_he),
    )
    return index_out_path


# ---------------------------------------------------------------------------
# TPS residual evaluation
# ---------------------------------------------------------------------------


def compute_tps_residual(
    tps_x: scipy.interpolate.RBFInterpolator,
    tps_y: scipy.interpolate.RBFInterpolator,
    src_he: np.ndarray,
    dst_mx: np.ndarray,
) -> float:
    """Mean L2 residual of TPS fit on inlier control points (MX px)."""
    pred_x = tps_x(src_he)
    pred_y = tps_y(src_he)
    pred = np.column_stack([pred_x, pred_y])
    residual = np.linalg.norm(pred - dst_mx, axis=1).mean()
    return float(residual)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    processed_dir = pathlib.Path(args.processed)
    cellvit_dir = processed_dir / "cellvit"
    index_path = processed_dir / "index.json"

    # Load index.json
    log.info("Loading index.json from %s ...", index_path)
    with index_path.open() as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    m_full_list = index.get("warp_matrix")
    if m_full_list is None:
        raise ValueError("index.json has no 'warp_matrix' field.")
    m_full = np.array(m_full_list, dtype=np.float64)
    log.info("  m_full = %s", m_full.tolist())

    # Resolve channel indices
    channel_indices = index.get("channel_indices")
    if channel_indices is None:
        channel_names = index.get("channels", [])
        if channel_names and args.metadata_csv:
            from utils.channels import resolve_channel_indices

            channel_indices, _ = resolve_channel_indices(
                args.metadata_csv, channel_names
            )
            log.info(
                "  Resolved %d channel indices from metadata CSV.", len(channel_indices)
            )
        else:
            raise ValueError(
                "index.json has no 'channel_indices' and no --metadata-csv provided "
                "to resolve channel names. Pass --metadata-csv."
            )

    # Step 1: Build H&E cell cloud (centroids + contours) from CellViT JSONs
    log.info("Step 1: Loading H&E cell geometry from CellViT JSONs ...")
    he_pts_he, _, he_contours_he = load_he_cells(
        cellvit_dir=cellvit_dir,
        patches=patches,
        coord_scale=1.0,  # coord_scale unused; we work in HE space throughout
        min_contour_vertices=args.contour_min_vertices,
    )
    n_with_contour = sum(1 for c in he_contours_he if c is not None)
    log.info(
        "  %d H&E cells loaded (%d with valid contours, %d centroid-only fallback).",
        len(he_pts_he),
        n_with_contour,
        len(he_pts_he) - n_with_contour,
    )

    if len(he_pts_he) == 0:
        raise RuntimeError(
            f"No CellViT centroids found in {cellvit_dir}. "
            "Run Stage 2 (CellViT) before Stage 2.5."
        )

    # Step 2: Load CSV centroids → MX px → H&E px via inv(m_full)
    log.info(
        "Step 2: Loading CSV centroids from %s and converting to H&E px ...", args.csv
    )
    cli_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None
    crop_origin = resolve_mx_crop_origin(index, cli_origin=cli_origin)
    if crop_origin is not None:
        if cli_origin is not None:
            log.info(
                "  Applying crop origin offset from CLI: (%.1f, %.1f) MX px",
                *crop_origin,
            )
        else:
            log.info(
                "  Using crop origin from index.json: (%.1f, %.1f) MX px",
                *crop_origin,
            )
    else:
        log.info("  No crop origin offset applied.")
    mx_pts, csv_in_he = csv_to_he_coords(
        pathlib.Path(args.csv),
        m_full=m_full,
        csv_mpp=args.csv_mpp,
        crop_origin=crop_origin,
    )
    csv_total_before_roi = len(mx_pts)
    roi_bbox_he = patch_roi_bbox_he(
        patches, patch_size_default=index.get("patch_size", 256)
    )

    if args.no_csv_roi_filter:
        csv_roi_margin_px = 0.0
        log.info(
            "  %d CSV centroids loaded (ROI filter disabled).",
            csv_total_before_roi,
        )
    else:
        if args.csv_roi_margin is None:
            csv_roi_margin_px = 0.5 * float(index.get("patch_size", 256))
        else:
            csv_roi_margin_px = float(args.csv_roi_margin)
        mx_pts, csv_in_he, _keep = filter_csv_to_patch_roi(
            mx_pts,
            csv_in_he,
            roi_bbox_he=roi_bbox_he,
            margin_px=csv_roi_margin_px,
        )
        log.info(
            "  %d / %d CSV centroids kept inside patch ROI (margin=%.1f px).",
            len(mx_pts),
            csv_total_before_roi,
            csv_roi_margin_px,
        )

    if len(mx_pts) < 4:
        raise RuntimeError(
            f"Only {len(mx_pts)} CSV points remain after ROI filtering "
            "(need >= 4). Try increasing --csv-roi-margin or disable ROI filtering."
        )

    # Step 3: ICP in H&E space to correct global drift in m_full
    log.info(
        "Step 3: ICP alignment in H&E space (max_iter=%d, tol=%.1e, gate=%g H&E px) ...",
        args.icp_max_iter,
        args.icp_tol,
        args.distance_gate,
    )
    M_icp, icp_n_matches, icp_n_iters = affine_icp(
        src_he=he_pts_he,
        dst_he=csv_in_he,
        max_iter=args.icp_max_iter,
        tol=args.icp_tol,
        distance_gate=args.distance_gate,
    )
    log.info(
        "  ICP done: %d iters, %d final matches, M_icp=%s",
        icp_n_iters,
        icp_n_matches,
        M_icp.tolist(),
    )

    # Step 4: Post-ICP matching in H&E space + RANSAC
    log.info(
        "Step 4: Post-ICP mutual NN matching + RANSAC (gate=%g H&E px, repr=%s) ...",
        args.distance_gate,
        args.cellvit_match_repr,
    )
    if args.cellvit_match_repr == "contour":
        src_he, dst_mx = match_cells_he_contour(
            m_icp=M_icp,
            he_pts_he=he_pts_he,
            he_contours_he=he_contours_he,
            csv_in_he=csv_in_he,
            mx_pts=mx_pts,
            distance_gate=args.distance_gate,
        )
    else:
        icp_he = apply_affine(M_icp, he_pts_he)
        src_he, dst_mx = match_centroids_he(
            icp_he=icp_he,
            he_pts_he=he_pts_he,
            csv_in_he=csv_in_he,
            mx_pts=mx_pts,
            distance_gate=args.distance_gate,
        )
    n_initial = len(src_he)
    log.info("  %d mutual nearest-neighbour matches (H&E space).", n_initial)

    if n_initial < 4:
        raise RuntimeError(
            f"Only {n_initial} post-ICP matches found (need >= 4). "
            "Try increasing --distance-gate or checking ICP convergence."
        )

    src_he_inlier, dst_mx_inlier = ransac_filter(src_he, dst_mx, ransac_thresh=5.0)
    n_inliers = len(src_he_inlier)
    inlier_frac = n_inliers / max(1, n_initial)
    he_total = len(he_pts_he)
    csv_total = len(csv_in_he)
    he_match_rate = n_initial / max(1, he_total)
    csv_match_rate = n_initial / max(1, csv_total)
    he_inlier_rate = n_inliers / max(1, he_total)
    csv_inlier_rate = n_inliers / max(1, csv_total)
    log.info(
        "  %d inliers / %d total (%.1f%%)", n_inliers, n_initial, 100 * inlier_frac
    )
    log.info(
        "  Match rates: HE %.1f%% (%d/%d), CSV %.1f%% (%d/%d)",
        100 * he_match_rate,
        n_initial,
        he_total,
        100 * csv_match_rate,
        n_initial,
        csv_total,
    )
    log.info(
        "  Inlier rates: HE %.1f%% (%d/%d), CSV %.1f%% (%d/%d)",
        100 * he_inlier_rate,
        n_inliers,
        he_total,
        100 * csv_inlier_rate,
        n_inliers,
        csv_total,
    )

    if n_inliers < 4:
        raise RuntimeError(
            f"Only {n_inliers} RANSAC inliers found (need >= 4). "
            "Matching quality too low for TPS."
        )

    # Step 5: Fit TPS (H&E px → MX px)
    log.info(
        "Step 5: Fitting TPS H&E px → MX px (max_tps_points=%d) ...",
        args.max_tps_points,
    )
    tps_x, tps_y = fit_tps(
        src_he_inlier, dst_mx_inlier, max_tps_points=args.max_tps_points
    )

    src_sub, dst_sub = subsample_uniform(
        src_he_inlier, dst_mx_inlier, args.max_tps_points
    )
    residual = compute_tps_residual(tps_x, tps_y, src_he_inlier, dst_mx_inlier)
    log.info("  TPS residual (mean L2 on inliers): %.3f MX px", residual)

    # Step 6: Re-extract multiplex patches
    if not args.skip_reextract:
        log.info("Step 6: Re-extracting multiplex patches ...")
        reextract_multiplex_patches(
            mx_tif_path=pathlib.Path(args.multiplex_image),
            processed_dir=processed_dir,
            patches=patches,
            channel_indices=channel_indices,
            tps_x=tps_x,
            tps_y=tps_y,
            patch_size_default=index.get("patch_size", 256),
        )
    else:
        log.info("Step 6: Skipped (--skip-reextract).")

    # Step 7: Write updated index file
    log.info("Step 7: Writing updated index file ...")
    index_out_path = update_index(
        processed_dir=processed_dir,
        tps_control_he=src_sub,
        tps_control_mx=dst_sub,
        n_matches=n_inliers,
        inlier_fraction=inlier_frac,
        he_total=he_total,
        csv_total=csv_total,
        he_match_rate=he_match_rate,
        csv_match_rate=csv_match_rate,
        he_inlier_rate=he_inlier_rate,
        csv_inlier_rate=csv_inlier_rate,
        csv_total_before_roi=csv_total_before_roi,
        csv_roi_margin_px=csv_roi_margin_px,
        csv_roi_bbox_he=None if args.no_csv_roi_filter else roi_bbox_he,
        icp_matrix=M_icp,
        icp_n_iters=icp_n_iters,
        icp_n_matches=icp_n_matches,
        cellvit_match_repr=args.cellvit_match_repr,
        he_cells_with_contour=n_with_contour,
        index_out_name=args.index_out,
    )

    print(
        f"\nStage 2.5 complete.\n"
        f"  ICP iters       : {icp_n_iters}\n"
        f"  ICP matches     : {icp_n_matches}\n"
        f"  Initial matches : {n_initial}\n"
        f"  RANSAC inliers  : {n_inliers} ({100*inlier_frac:.1f}%)\n"
        f"  HE match rate   : {100*he_match_rate:.1f}% ({n_initial}/{he_total})\n"
        f"  CSV match rate  : {100*csv_match_rate:.1f}% ({n_initial}/{csv_total})\n"
        f"  HE inlier rate  : {100*he_inlier_rate:.1f}% ({n_inliers}/{he_total})\n"
        f"  CSV inlier rate : {100*csv_inlier_rate:.1f}% ({n_inliers}/{csv_total})\n"
        f"  Match repr      : {args.cellvit_match_repr} "
        f"(contours {n_with_contour}/{he_total})\n"
        f"  CSV kept in ROI : {csv_total}/{csv_total_before_roi}\n"
        f"  TPS control pts : {len(src_sub)}\n"
        f"  TPS residual    : {residual:.3f} MX px\n"
        f"  Index output    : {index_out_path}\n"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2.5: CellViT-guided ICP + TPS registration refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed", required=True, help="Processed output directory.")
    p.add_argument("--he-image", required=True, help="H&E OME-TIFF path.")
    p.add_argument("--multiplex-image", required=True, help="Multiplex OME-TIFF path.")
    p.add_argument("--csv", required=True, help="MX cell CSV with Xt,Yt columns (µm).")
    p.add_argument(
        "--metadata-csv",
        default=None,
        help="Multiplex channel metadata CSV. Required when channel_indices is not "
        "stored in index.json.",
    )
    p.add_argument(
        "--csv-mpp",
        type=float,
        default=0.65,
        help="µm/px of CSV coordinate space (divides Xt/Yt to convert µm → MX px).",
    )
    p.add_argument(
        "--mx-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Top-left corner of the crop image in full-slide MX px (x y). "
        "Subtracted from CSV coords after mpp conversion. If omitted, "
        "attempts to read mx_crop_origin from index.json.",
    )
    p.add_argument(
        "--distance-gate",
        type=float,
        default=20.0,
        help="Max H&E px distance for ICP and post-ICP CellViT/CSV matching.",
    )
    p.add_argument(
        "--cellvit-match-repr",
        choices=["contour", "centroid"],
        default="contour",
        help="CellViT representation used in post-ICP matching.",
    )
    p.add_argument(
        "--contour-min-vertices",
        type=int,
        default=3,
        help="Minimum contour vertices required to treat a CellViT contour as valid.",
    )
    p.add_argument(
        "--csv-roi-margin",
        type=float,
        default=None,
        help="Margin (H&E px) added around the Stage-1 patch ROI when filtering "
        "CSV points. Default: 0.5 * patch_size.",
    )
    p.add_argument(
        "--no-csv-roi-filter",
        action="store_true",
        help="Disable CSV filtering to Stage-1 patch ROI.",
    )
    p.add_argument(
        "--icp-max-iter",
        type=int,
        default=50,
        help="Maximum ICP iterations.",
    )
    p.add_argument(
        "--icp-tol",
        type=float,
        default=1e-4,
        help="ICP convergence threshold on step-transform magnitude.",
    )
    p.add_argument(
        "--max-tps-points",
        type=int,
        default=2000,
        help="Max TPS control points (spatially subsampled).",
    )
    p.add_argument(
        "--skip-reextract",
        action="store_true",
        help="Skip re-extraction of multiplex patches (dry-run / index update only).",
    )
    p.add_argument(
        "--index-out",
        default="index_icp_tps.json",
        help="Output index filename written under --processed.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
