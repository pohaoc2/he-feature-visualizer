"""
Tests for stages/refine_registration.py — Stage 2.5.

All tests use synthetic data — no real TIFF or CSV files are required.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.spatial
import scipy.interpolate

from stages.refine_registration import (
    load_he_cells,
    affine_icp,
    apply_affine,
    compute_tps_residual,
    csv_to_he_coords,
    filter_csv_to_patch_roi,
    fit_tps,
    load_he_centroids,
    load_mx_centroids,
    match_cells_he_contour,
    match_centroids,
    match_centroids_he,
    patch_roi_bbox_he,
    ransac_filter,
    resolve_mx_crop_origin,
    subsample_uniform,
    update_index,
    _make_patch_pixel_grid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_m_full() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _scale_m_full(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# apply_affine
# ---------------------------------------------------------------------------


def test_apply_affine_identity():
    m = _identity_m_full()
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = apply_affine(m, pts)
    np.testing.assert_allclose(out, pts)


def test_apply_affine_scale():
    m = _scale_m_full(0.5, 0.5)
    pts = np.array([[100.0, 200.0]])
    out = apply_affine(m, pts)
    np.testing.assert_allclose(out, [[50.0, 100.0]])


def test_apply_affine_translation():
    m = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    pts = np.array([[10.0, 10.0]])
    out = apply_affine(m, pts)
    np.testing.assert_allclose(out, [[15.0, 7.0]])


# ---------------------------------------------------------------------------
# load_he_centroids
# ---------------------------------------------------------------------------


def test_load_he_centroids_basic(tmp_path):
    """CellViT JSON → H&E centroids in HE and MX space."""
    patches = [{"x0": 0, "y0": 0}, {"x0": 256, "y0": 0}]
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()

    # Patch (0,0): two cells
    (cellvit_dir / "0_0.json").write_text(
        json.dumps({"cells": [{"centroid": [10.0, 20.0]}, {"centroid": [30.0, 40.0]}]})
    )
    # Patch (256,0): one cell
    (cellvit_dir / "256_0.json").write_text(
        json.dumps({"cells": [{"centroid": [5.0, 15.0]}]})
    )

    he_pts_he, he_pts_mx = load_he_centroids(cellvit_dir, patches, coord_scale=0.5)

    assert he_pts_he.shape == (3, 2)
    assert he_pts_mx.shape == (3, 2)
    # First cell: global HE = (0+10, 0+20) = (10, 20); MX = (5, 10)
    np.testing.assert_allclose(he_pts_he[0], [10.0, 20.0])
    np.testing.assert_allclose(he_pts_mx[0], [5.0, 10.0])
    # Third cell: global HE = (256+5, 0+15) = (261, 15); MX = (130.5, 7.5)
    np.testing.assert_allclose(he_pts_he[2], [261.0, 15.0])
    np.testing.assert_allclose(he_pts_mx[2], [130.5, 7.5])


def test_load_he_centroids_missing_json(tmp_path):
    """Missing CellViT JSON files are silently skipped."""
    patches = [{"x0": 0, "y0": 0}]
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    # No JSON files created

    he_pts_he, he_pts_mx = load_he_centroids(cellvit_dir, patches, coord_scale=0.5)
    assert he_pts_he.shape == (0, 2)
    assert he_pts_mx.shape == (0, 2)


def test_load_he_centroids_empty_cells(tmp_path):
    """JSON with empty cells list returns empty arrays."""
    patches = [{"x0": 0, "y0": 0}]
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    (cellvit_dir / "0_0.json").write_text(json.dumps({"cells": []}))

    he_pts_he, _ = load_he_centroids(cellvit_dir, patches, coord_scale=0.5)
    assert len(he_pts_he) == 0


def test_load_he_cells_includes_contours(tmp_path):
    """CellViT contour vertices are loaded and shifted to global H&E coords."""
    patches = [{"x0": 100, "y0": 200}]
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    (cellvit_dir / "100_200.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "centroid": [10.0, 20.0],
                        "contour": [[8, 18], [12, 18], [12, 22], [8, 22]],
                    }
                ]
            }
        )
    )

    he_pts_he, he_pts_mx, contours = load_he_cells(
        cellvit_dir, patches, coord_scale=0.5
    )
    assert he_pts_he.shape == (1, 2)
    np.testing.assert_allclose(he_pts_he[0], [110.0, 220.0])
    np.testing.assert_allclose(he_pts_mx[0], [55.0, 110.0])
    assert len(contours) == 1
    assert contours[0] is not None
    np.testing.assert_allclose(contours[0][0], [108.0, 218.0])


# ---------------------------------------------------------------------------
# load_mx_centroids
# ---------------------------------------------------------------------------


def test_load_mx_centroids_basic(tmp_path):
    csv_path = tmp_path / "cells.csv"
    df = pd.DataFrame({"Xt": [10.0, 20.0, 30.0], "Yt": [5.0, 15.0, 25.0]})
    df.to_csv(csv_path, index=False)

    mx_pts, kdtree = load_mx_centroids(csv_path)
    assert mx_pts.shape == (3, 2)
    assert isinstance(kdtree, scipy.spatial.KDTree)
    np.testing.assert_allclose(mx_pts[0], [10.0, 5.0])


def test_load_mx_centroids_missing_columns(tmp_path):
    csv_path = tmp_path / "cells.csv"
    pd.DataFrame({"X": [1], "Y": [2]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Xt"):
        load_mx_centroids(csv_path)


# ---------------------------------------------------------------------------
# match_centroids
# ---------------------------------------------------------------------------


def _make_grid_points(n: int, scale: float = 1.0) -> np.ndarray:
    """Create a regular grid of n*n points."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([i * scale, j * scale])
            if len(pts) >= n:
                return np.array(pts[:n], dtype=np.float64)
    return np.array(pts[:n], dtype=np.float64)


def test_match_centroids_identity():
    """With identity m_full and perfect correspondence, all points should match."""
    n = 20
    he_pts_he = _make_grid_points(n, scale=10.0)
    mx_pts = he_pts_he.copy()  # identity mapping
    kdtree = scipy.spatial.KDTree(mx_pts)
    m_full = _identity_m_full()

    src_he, dst_mx = match_centroids(
        he_pts_he, mx_pts, kdtree, m_full, distance_gate=1.0
    )
    assert len(src_he) == n
    np.testing.assert_allclose(src_he, dst_mx)


def test_match_centroids_scale_half():
    """With scale=0.5 m_full, scaled H&E pts should match MX pts at half resolution."""
    n = 20
    he_pts_he = _make_grid_points(n, scale=10.0)
    mx_pts = he_pts_he * 0.5  # ground truth MX
    kdtree = scipy.spatial.KDTree(mx_pts)
    m_full = _scale_m_full(0.5, 0.5)

    src_he, _ = match_centroids(
        he_pts_he, mx_pts, kdtree, m_full, distance_gate=0.5
    )
    assert len(src_he) == n


def test_match_centroids_too_far():
    """Points outside distance gate should not be matched."""
    he_pts_he = np.array([[0.0, 0.0], [100.0, 100.0]])
    mx_pts = np.array([[50.0, 50.0], [200.0, 200.0]])  # all far away
    kdtree = scipy.spatial.KDTree(mx_pts)
    m_full = _identity_m_full()

    src_he, _ = match_centroids(
        he_pts_he, mx_pts, kdtree, m_full, distance_gate=1.0
    )
    assert len(src_he) == 0


def test_match_cells_he_contour_inside_point_matches():
    """CSV point inside contour should match with distance 0."""
    m_icp = _identity_m_full()
    he_pts_he = np.array([[10.0, 10.0]], dtype=np.float64)
    he_contours = [
        np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float64)
    ]
    csv_in_he = np.array([[11.0, 11.0]], dtype=np.float64)
    mx_pts = np.array([[101.0, 201.0]], dtype=np.float64)

    src_he, dst_mx = match_cells_he_contour(
        m_icp=m_icp,
        he_pts_he=he_pts_he,
        he_contours_he=he_contours,
        csv_in_he=csv_in_he,
        mx_pts=mx_pts,
        distance_gate=1.0,
    )
    assert len(src_he) == 1
    np.testing.assert_allclose(src_he[0], [10.0, 10.0])
    np.testing.assert_allclose(dst_mx[0], [101.0, 201.0])


def test_match_cells_he_contour_falls_back_to_centroid_distance():
    """Missing contour falls back to centroid-distance matching."""
    m_icp = _identity_m_full()
    he_pts_he = np.array([[0.0, 0.0], [20.0, 20.0]], dtype=np.float64)
    he_contours = [None, None]
    csv_in_he = np.array([[0.5, 0.5], [30.0, 30.0]], dtype=np.float64)
    mx_pts = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64)

    src_he, dst_mx = match_cells_he_contour(
        m_icp=m_icp,
        he_pts_he=he_pts_he,
        he_contours_he=he_contours,
        csv_in_he=csv_in_he,
        mx_pts=mx_pts,
        distance_gate=2.0,
    )
    assert len(src_he) == 1
    np.testing.assert_allclose(src_he[0], [0.0, 0.0])
    np.testing.assert_allclose(dst_mx[0], [100.0, 100.0])


# ---------------------------------------------------------------------------
# ransac_filter
# ---------------------------------------------------------------------------


def test_ransac_filter_keeps_inliers():
    """Pure affine point set should have high inlier count."""
    rng = np.random.default_rng(0)
    n = 50
    src = rng.uniform(0, 1000, (n, 2))
    # Ground truth: scale 0.5 + translation (100, 200)
    m_true = np.array([[0.5, 0.0, 100.0], [0.0, 0.5, 200.0]])
    dst = apply_affine(m_true, src) + rng.normal(0, 0.5, (n, 2))

    # Add 5 outliers
    dst[-5:] = rng.uniform(0, 5000, (5, 2))

    src_in, _ = ransac_filter(src, dst, ransac_thresh=5.0)
    assert len(src_in) >= n - 5  # most original inliers kept


def test_ransac_filter_too_few_points():
    """With < 4 points, no filtering is applied."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dst = src + 0.1
    src_out, _ = ransac_filter(src, dst)
    assert len(src_out) == len(src)


# ---------------------------------------------------------------------------
# subsample_uniform
# ---------------------------------------------------------------------------


def test_subsample_uniform_no_op():
    """When n <= max_pts, returns all points unchanged."""
    src = np.arange(20).reshape(10, 2).astype(float)
    dst = src * 2
    s, _ = subsample_uniform(src, dst, max_pts=20)
    assert len(s) == 10


def test_subsample_uniform_reduces():
    """Returns at most max_pts points."""
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (500, 2))
    dst = src + 1
    s, d = subsample_uniform(src, dst, max_pts=100)
    assert len(s) <= 100
    assert len(s) == len(d)


# ---------------------------------------------------------------------------
# fit_tps + compute_tps_residual
# ---------------------------------------------------------------------------


def test_fit_tps_identity():
    """TPS fitted on identity mapping should have near-zero residual."""
    rng = np.random.default_rng(42)
    n = 50
    src = rng.uniform(0, 500, (n, 2))
    dst = src.copy()  # identity

    tps_x, tps_y = fit_tps(src, dst, max_tps_points=n)
    residual = compute_tps_residual(tps_x, tps_y, src, dst)
    assert residual < 1.0, f"Expected near-zero residual, got {residual:.4f}"


def test_fit_tps_affine():
    """TPS should fit a pure affine transform exactly."""
    rng = np.random.default_rng(7)
    n = 80
    src = rng.uniform(0, 1000, (n, 2))
    m = np.array([[0.5, 0.0, 50.0], [0.0, 0.5, 100.0]])
    dst = apply_affine(m, src)

    tps_x, tps_y = fit_tps(src, dst, max_tps_points=n)
    residual = compute_tps_residual(tps_x, tps_y, src, dst)
    assert residual < 1.0, f"TPS affine residual too high: {residual:.4f}"


# ---------------------------------------------------------------------------
# _make_patch_pixel_grid
# ---------------------------------------------------------------------------


def test_make_patch_pixel_grid_shape():
    grid = _make_patch_pixel_grid(x0=10, y0=20, patch_size=4)
    assert grid.shape == (16, 2)


def test_make_patch_pixel_grid_values():
    grid = _make_patch_pixel_grid(x0=0, y0=0, patch_size=2)
    # (col, row) order; x = col + x0, y = row + y0
    # row=0: x=0,y=0 and x=1,y=0; row=1: x=0,y=1 and x=1,y=1
    expected_x = [0.0, 1.0, 0.0, 1.0]
    expected_y = [0.0, 0.0, 1.0, 1.0]
    np.testing.assert_allclose(grid[:, 0], expected_x)
    np.testing.assert_allclose(grid[:, 1], expected_y)


# ---------------------------------------------------------------------------
# update_index
# ---------------------------------------------------------------------------


def test_update_index(tmp_path):
    index_path = tmp_path / "index.json"
    initial = {
        "warp_matrix": [[0.5, 0.0, -100.0], [0.0, 0.5, 200.0]],
        "registration_mode": "deformable",
        "patches": [{"x0": 0, "y0": 0, "patch_size": 256}],
    }
    index_path.write_text(json.dumps(initial))

    ctrl_he = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    ctrl_mx = ctrl_he * 0.5

    update_index(
        processed_dir=tmp_path,
        tps_control_he=ctrl_he,
        tps_control_mx=ctrl_mx,
        n_matches=42,
        inlier_fraction=0.85,
    )

    result = json.loads(index_path.read_text())
    assert result["registration_mode"] == "icp_tps"
    assert result["tps_n_matches"] == 42
    assert abs(result["tps_inlier_fraction"] - 0.85) < 1e-6
    assert len(result["tps_control_he"]) == 3
    assert len(result["tps_control_mx"]) == 3
    # Original fields preserved
    assert "warp_matrix" in result
    assert "patches" in result


def test_update_index_writes_new_file_when_requested(tmp_path):
    """When index_out_name differs, input index.json must remain unchanged."""
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "warp_matrix": [[0.5, 0.0, -100.0], [0.0, 0.5, 200.0]],
                "registration_mode": "affine",
                "patches": [{"x0": 0, "y0": 0, "patch_size": 256}],
            }
        )
    )

    ctrl_he = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    ctrl_mx = ctrl_he * 0.5
    out_path = update_index(
        processed_dir=tmp_path,
        tps_control_he=ctrl_he,
        tps_control_mx=ctrl_mx,
        n_matches=42,
        inlier_fraction=0.85,
        index_out_name="index_icp_tps.json",
    )

    original = json.loads(index_path.read_text())
    modified = json.loads((tmp_path / "index_icp_tps.json").read_text())

    assert out_path == (tmp_path / "index_icp_tps.json")
    assert original["registration_mode"] == "affine"
    assert "tps_n_matches" not in original
    assert modified["registration_mode"] == "icp_tps"
    assert modified["tps_n_matches"] == 42


# ---------------------------------------------------------------------------
# csv_to_he_coords
# ---------------------------------------------------------------------------


def test_csv_to_he_coords_identity(tmp_path):
    """With identity m_full and csv_mpp=1, MX px == CSV µm and HE px == MX px."""
    csv_path = tmp_path / "cells.csv"
    pd.DataFrame({"Xt": [100.0, 200.0], "Yt": [50.0, 150.0]}).to_csv(
        csv_path, index=False
    )

    m_full = _identity_m_full()
    mx_pts, he_pts = csv_to_he_coords(csv_path, m_full, csv_mpp=1.0)

    np.testing.assert_allclose(mx_pts, [[100.0, 50.0], [200.0, 150.0]])
    # inv(identity) is identity → he == mx
    np.testing.assert_allclose(he_pts, mx_pts, atol=1e-9)


def test_csv_to_he_coords_scale_mpp(tmp_path):
    """csv_mpp divides Xt/Yt before inverse transform."""
    csv_path = tmp_path / "cells.csv"
    pd.DataFrame({"Xt": [1.3, 2.6], "Yt": [0.65, 1.3]}).to_csv(csv_path, index=False)

    m_full = _identity_m_full()
    mx_pts, he_pts = csv_to_he_coords(csv_path, m_full, csv_mpp=0.65)

    # 1.3 / 0.65 = 2.0, 2.6 / 0.65 = 4.0 ...
    np.testing.assert_allclose(mx_pts[:, 0], [2.0, 4.0], atol=1e-9)
    np.testing.assert_allclose(he_pts, mx_pts, atol=1e-9)


def test_csv_to_he_coords_inverse_scale(tmp_path):
    """inv(scale(0.5)) should double the MX px to get HE px."""
    csv_path = tmp_path / "cells.csv"
    pd.DataFrame({"Xt": [100.0], "Yt": [200.0]}).to_csv(csv_path, index=False)

    m_full = _scale_m_full(0.5, 0.5)
    mx_pts, he_pts = csv_to_he_coords(csv_path, m_full, csv_mpp=1.0)

    # mx = [100, 200]; inv(scale 0.5) = scale 2 → he = [200, 400]
    np.testing.assert_allclose(mx_pts, [[100.0, 200.0]], atol=1e-9)
    np.testing.assert_allclose(he_pts, [[200.0, 400.0]], atol=1e-9)


def test_csv_to_he_coords_crop_origin(tmp_path):
    """crop_origin is subtracted from MX px before inverse transform."""
    csv_path = tmp_path / "cells.csv"
    pd.DataFrame({"Xt": [500.0], "Yt": [300.0]}).to_csv(csv_path, index=False)

    m_full = _identity_m_full()
    mx_pts, he_pts = csv_to_he_coords(
        csv_path, m_full, csv_mpp=1.0, crop_origin=(200.0, 100.0)
    )

    # After origin subtraction: mx = [300, 200]
    np.testing.assert_allclose(mx_pts, [[300.0, 200.0]], atol=1e-9)
    np.testing.assert_allclose(he_pts, [[300.0, 200.0]], atol=1e-9)


def test_csv_to_he_coords_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"X": [1.0], "Y": [2.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Xt"):
        csv_to_he_coords(csv_path, _identity_m_full())


def test_resolve_mx_crop_origin_prefers_cli():
    index = {"mx_crop_origin": [10.0, 20.0]}
    got = resolve_mx_crop_origin(index, cli_origin=(30.0, 40.0))
    assert got == (30.0, 40.0)


def test_resolve_mx_crop_origin_reads_top_level():
    index = {"mx_crop_origin": [123.0, 456.0]}
    got = resolve_mx_crop_origin(index)
    assert got == (123.0, 456.0)


def test_resolve_mx_crop_origin_reads_crop_region():
    index = {"crop_region": {"mx_origin": [11.0, 22.0]}}
    got = resolve_mx_crop_origin(index)
    assert got == (11.0, 22.0)


def test_resolve_mx_crop_origin_none_when_absent():
    index = {"patches": []}
    got = resolve_mx_crop_origin(index)
    assert got is None


def test_patch_roi_bbox_he_uses_patch_sizes():
    patches = [
        {"x0": 10, "y0": 20, "patch_size": 100},
        {"x0": 80, "y0": 40, "patch_size": 200},
    ]
    x_min, x_max, y_min, y_max = patch_roi_bbox_he(patches, patch_size_default=256)
    assert (x_min, x_max, y_min, y_max) == (10.0, 280.0, 20.0, 240.0)


def test_filter_csv_to_patch_roi_applies_margin():
    mx_pts = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        dtype=np.float64,
    )
    csv_in_he = np.array(
        [[10.0, 10.0], [50.0, 50.0], [101.0, 101.0], [140.0, 140.0]],
        dtype=np.float64,
    )
    roi = (20.0, 100.0, 20.0, 100.0)

    mx_keep0, he_keep0, mask0 = filter_csv_to_patch_roi(
        mx_pts, csv_in_he, roi_bbox_he=roi, margin_px=0.0
    )
    assert mask0.tolist() == [False, True, False, False]
    assert len(mx_keep0) == 1
    np.testing.assert_allclose(he_keep0[0], [50.0, 50.0])

    mx_keep1, he_keep1, mask1 = filter_csv_to_patch_roi(
        mx_pts, csv_in_he, roi_bbox_he=roi, margin_px=5.0
    )
    assert mask1.tolist() == [False, True, True, False]
    assert len(mx_keep1) == 2
    np.testing.assert_allclose(he_keep1[1], [101.0, 101.0])


# ---------------------------------------------------------------------------
# affine_icp
# ---------------------------------------------------------------------------


def test_affine_icp_identity():
    """When src == dst, ICP should converge immediately with identity transform."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, (200, 2))
    M_icp, _, _ = affine_icp(pts, pts, max_iter=50, tol=1e-4)
    # Transform should be close to identity
    np.testing.assert_allclose(M_icp[:, :2], np.eye(2), atol=0.05)
    np.testing.assert_allclose(M_icp[:, 2], [0.0, 0.0], atol=5.0)


def test_affine_icp_pure_translation():
    """ICP should recover a known translation."""
    rng = np.random.default_rng(2)
    src = rng.uniform(100, 900, (300, 2))
    tx, ty = 30.0, -20.0
    dst = src + np.array([tx, ty])  # dst = src + translation

    M_icp, _, _ = affine_icp(src, dst, max_iter=50, tol=1e-4)
    src_warped = apply_affine(M_icp, src)

    # After ICP, src_warped should be close to dst
    np.testing.assert_allclose(src_warped, dst, atol=1.0)


def test_affine_icp_distance_gate_too_small():
    """If distance gate excludes all points, ICP should stop early without error."""
    src = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 0.0], [300.0, 200.0]])
    dst = src + 1000.0  # very far away

    M_icp, _, _ = affine_icp(
        src, dst, max_iter=10, tol=1e-4, distance_gate=1.0
    )
    # Should return without raising; n_matches may be 0
    assert M_icp.shape == (2, 3)


def test_affine_icp_output_shape():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 500, (50, 2))
    dst = rng.uniform(0, 500, (80, 2))
    M_icp, n_matches, n_iters = affine_icp(src, dst, max_iter=5)
    assert M_icp.shape == (2, 3)
    assert isinstance(n_matches, int)
    assert isinstance(n_iters, int)


# ---------------------------------------------------------------------------
# match_centroids_he
# ---------------------------------------------------------------------------


def test_match_centroids_he_exact():
    """When ICP-transformed CellViT perfectly overlaps CSV-in-HE, all should match."""
    rng = np.random.default_rng(4)
    n = 50
    he_pts = rng.uniform(0, 1000, (n, 2))
    csv_in_he = he_pts.copy()  # perfect overlap
    mx_pts = he_pts * 0.5  # dummy MX coords

    src_he, dst_mx = match_centroids_he(
        icp_he=he_pts,
        he_pts_he=he_pts,
        csv_in_he=csv_in_he,
        mx_pts=mx_pts,
        distance_gate=1.0,
    )
    assert len(src_he) == n
    np.testing.assert_allclose(dst_mx, mx_pts, atol=1e-9)


def test_match_centroids_he_empty_when_far():
    """No matches when all pairs exceed distance gate."""
    he_pts = np.array([[0.0, 0.0], [100.0, 0.0]])
    csv_in_he = he_pts + 1000.0
    mx_pts = csv_in_he * 0.5

    src_he, dst_mx = match_centroids_he(
        icp_he=he_pts,
        he_pts_he=he_pts,
        csv_in_he=csv_in_he,
        mx_pts=mx_pts,
        distance_gate=5.0,
    )
    assert len(src_he) == 0
    assert len(dst_mx) == 0


def test_update_index_with_icp(tmp_path):
    """update_index should store icp_matrix and icp_n_iters fields."""
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"warp_matrix": [[0.5, 0, 0], [0, 0.5, 0]], "patches": []})
    )

    ctrl_he = np.array([[0.0, 0.0], [100.0, 100.0]])
    ctrl_mx = ctrl_he * 0.5
    icp_m = np.array([[1.0, 0.01, 2.0], [-0.01, 1.0, -3.0]])

    update_index(
        processed_dir=tmp_path,
        tps_control_he=ctrl_he,
        tps_control_mx=ctrl_mx,
        n_matches=20,
        inlier_fraction=0.9,
        he_total=100,
        csv_total=500,
        he_match_rate=0.2,
        csv_match_rate=0.04,
        he_inlier_rate=0.15,
        csv_inlier_rate=0.03,
        csv_total_before_roi=700,
        csv_roi_margin_px=128.0,
        csv_roi_bbox_he=(0.0, 256.0, 0.0, 256.0),
        icp_matrix=icp_m,
        icp_n_iters=7,
        icp_n_matches=150,
    )

    result = json.loads(index_path.read_text())
    assert result["registration_mode"] == "icp_tps"
    assert result["icp_n_iters"] == 7
    assert result["icp_n_matches"] == 150
    assert len(result["icp_matrix"]) == 2
    assert len(result["icp_matrix"][0]) == 3
    assert result["he_total_centroids"] == 100
    assert result["csv_total_centroids"] == 500
    assert result["he_match_rate"] == pytest.approx(0.2)
    assert result["csv_match_rate"] == pytest.approx(0.04)
    assert result["csv_total_before_roi"] == 700
    assert result["csv_roi_margin_px"] == pytest.approx(128.0)
    assert result["csv_roi_bbox_he"]["x_max"] == pytest.approx(256.0)
