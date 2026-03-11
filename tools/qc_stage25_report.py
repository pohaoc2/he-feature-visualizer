#!/usr/bin/env python3
"""
qc_stage25_report.py -- Comprehensive Stage 2.5 sanity-check report.

Checks:
1) HE/MX crop consistency under stage-1 and stage-2.5 affine mapping.
2) CSV compatibility with MX DNA / optional MX mask.
3) Transformed-CSV support in transformed MX DNA (HE frame).
4) Transformed MX DNA proximity to HE signal (cross-modality proxies).
5) Low match-rate diagnosis (data-size vs matching/geometry limits).

The script is designed for crop-sized TIFF inputs to keep runtime low.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
import tifffile

from stages.refine_registration import (
    apply_affine,
    csv_to_he_coords,
    filter_csv_to_patch_roi,
    load_he_centroids,
    match_centroids_he,
    patch_roi_bbox_he,
    ransac_filter,
    resolve_mx_crop_origin,
)


def _normalize_to_cyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Convert TIFF series array to (C, Y, X) by dropping non-C/Y/X dims."""
    axes_u = axes.upper()
    if len(axes_u) != arr.ndim:
        raise ValueError(f"axes length ({len(axes_u)}) != arr.ndim ({arr.ndim})")

    slicer: list[int | slice] = []
    kept_axes: list[str] = []
    for ax in axes_u:
        if ax in ("C", "Y", "X"):
            slicer.append(slice(None))
            kept_axes.append(ax)
        else:
            slicer.append(0)
    arr2 = arr[tuple(slicer)]

    if "C" in kept_axes:
        target_axes = [ax for ax in ("C", "Y", "X") if ax in kept_axes]
        if kept_axes != target_axes:
            perm = [kept_axes.index(ax) for ax in target_axes]
            arr2 = arr2.transpose(perm)
    else:
        target_axes = [ax for ax in ("Y", "X") if ax in kept_axes]
        if kept_axes != target_axes:
            perm = [kept_axes.index(ax) for ax in target_axes]
            arr2 = arr2.transpose(perm)
        arr2 = np.expand_dims(arr2, axis=0)

    if arr2.ndim != 3:
        raise ValueError(f"Expected CYX after normalization, got shape {arr2.shape}")
    return arr2


def _read_cyx(path: pathlib.Path) -> np.ndarray:
    """Read TIFF first series as CYX."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = series.axes
    return _normalize_to_cyx(np.asarray(arr), axes)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile normalize image to uint8."""
    img_f = img.astype(np.float32)
    lo = np.percentile(img_f, 1.0)
    hi = np.percentile(img_f, 99.0)
    if hi <= lo:
        return np.zeros_like(img_f, dtype=np.uint8)
    norm = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(norm * 255.0).astype(np.uint8)


def _inv_affine_2x3(m: np.ndarray) -> np.ndarray:
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(m3)[:2, :]


def _compose_affine(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return m = m1 @ m2 for two 2x3 affine transforms."""
    a = np.vstack([m1, [0.0, 0.0, 1.0]])
    b = np.vstack([m2, [0.0, 0.0, 1.0]])
    return (a @ b)[:2, :]


def _sample_at_points(img: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(pts_xy[:, 0]).astype(int), 0, img.shape[1] - 1)
    y = np.clip(np.rint(pts_xy[:, 1]).astype(int), 0, img.shape[0] - 1)
    return img[y, x]


def _median_p90(x: np.ndarray) -> tuple[float, float]:
    return float(np.median(x)), float(np.quantile(x, 0.9))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel().astype(np.float64)
    bb = b.ravel().astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _nmi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    h2, _, _ = np.histogram2d(
        a.ravel(),
        b.ravel(),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    pxy = h2 / max(float(h2.sum()), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    if not np.any(nz):
        return 0.0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    denom = max(np.sqrt(hx * hy), 1e-12)
    return float(mi / denom)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _as_float_or(value: Any, fallback: float) -> float:
    """Best-effort float conversion with fallback for None/invalid values."""
    try:
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Comprehensive Stage 2.5 sanity-check report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed", required=True, help="Processed directory with index.json.")
    p.add_argument("--he-image", required=True, help="HE crop OME-TIFF.")
    p.add_argument("--multiplex-image", required=True, help="MX crop OME-TIFF.")
    p.add_argument("--csv", required=True, help="CSV with Xt,Yt (um).")
    p.add_argument("--mask", default=None, help="Optional full-slide MX instance mask TIFF.")
    p.add_argument("--index", default=None, help="Optional index path. Default: <processed>/index.json")
    p.add_argument("--csv-mpp", type=float, default=0.65, help="CSV um/px scale.")
    p.add_argument(
        "--mx-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Crop origin in full-slide MX px. If omitted, infer from index.",
    )
    p.add_argument(
        "--dna-channel",
        type=int,
        default=0,
        help="MX DNA channel index.",
    )
    p.add_argument(
        "--distance-gates",
        type=float,
        nargs="+",
        default=[12.0, 20.0, 28.0],
        help="Distance gates for match-rate diagnosis.",
    )
    p.add_argument(
        "--roi-margin",
        type=float,
        default=None,
        help="Override ROI margin in H&E px for CSV filtering.",
    )
    p.add_argument(
        "--report-json",
        default=None,
        help="Optional output JSON report path.",
    )
    args = p.parse_args()

    processed = pathlib.Path(args.processed)
    index_path = pathlib.Path(args.index) if args.index else (processed / "index.json")
    he_path = pathlib.Path(args.he_image)
    mx_path = pathlib.Path(args.multiplex_image)
    csv_path = pathlib.Path(args.csv)
    mask_path = pathlib.Path(args.mask) if args.mask else None

    index = json.loads(index_path.read_text())
    m_full = np.array(index["warp_matrix"], dtype=np.float64)
    m_icp = np.array(index.get("icp_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dtype=np.float64)
    m_stage25_aff = _compose_affine(m_full, m_icp)

    cli_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None
    resolved_origin = resolve_mx_crop_origin(index, cli_origin=cli_origin)
    mx_crop_origin = np.array(resolved_origin, dtype=np.float64) if resolved_origin is not None else None

    he_cyx = _read_cyx(he_path)
    mx_cyx = _read_cyx(mx_path)
    he_img = _to_uint8(he_cyx[:3] if he_cyx.shape[0] >= 3 else np.repeat(he_cyx[:1], 3, axis=0))
    mx_dna = mx_cyx[int(args.dna_channel)].astype(np.float32)
    _, he_h, he_w = he_img.shape
    _, mx_h, mx_w = mx_cyx.shape

    csv_um = pd.read_csv(csv_path, usecols=["Xt", "Yt"])[["Xt", "Yt"]].to_numpy(dtype=np.float64)
    csv_full_mx = csv_um / float(args.csv_mpp)

    report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Q1: Crop consistency
    # ------------------------------------------------------------------
    he_mpp = _as_float_or(index.get("he_mpp"), 0.325)
    mx_mpp = _as_float_or(index.get("mx_mpp"), 0.65)
    he_corners = np.array(
        [[0.0, 0.0], [he_w - 1.0, 0.0], [he_w - 1.0, he_h - 1.0], [0.0, he_h - 1.0]],
        dtype=np.float64,
    )
    mx_corners_stage1 = apply_affine(m_full, he_corners)
    mx_corners_stage25 = apply_affine(m_stage25_aff, he_corners)
    inside1 = (
        (mx_corners_stage1[:, 0] >= 0)
        & (mx_corners_stage1[:, 0] < mx_w)
        & (mx_corners_stage1[:, 1] >= 0)
        & (mx_corners_stage1[:, 1] < mx_h)
    )
    inside25 = (
        (mx_corners_stage25[:, 0] >= 0)
        & (mx_corners_stage25[:, 0] < mx_w)
        & (mx_corners_stage25[:, 1] >= 0)
        & (mx_corners_stage25[:, 1] < mx_h)
    )
    report["q1_crop_consistency"] = {
        "he_size_px": [he_w, he_h],
        "mx_size_px": [mx_w, mx_h],
        "he_size_um": [he_w * he_mpp, he_h * he_mpp],
        "mx_size_um": [mx_w * mx_mpp, mx_h * mx_mpp],
        "mx_crop_origin": None if mx_crop_origin is None else mx_crop_origin.tolist(),
        "stage1_corners_inside_mx": int(np.sum(inside1)),
        "stage25_affine_corners_inside_mx": int(np.sum(inside25)),
        "stage1_corners_mx": mx_corners_stage1.tolist(),
        "stage25_affine_corners_mx": mx_corners_stage25.tolist(),
    }

    # ------------------------------------------------------------------
    # Q2: CSV vs MX DNA / mask
    # ------------------------------------------------------------------
    if mx_crop_origin is not None:
        csv_local_mx = csv_full_mx - mx_crop_origin
    else:
        csv_local_mx = csv_full_mx.copy()
    in_mx_crop = (
        (csv_local_mx[:, 0] >= 0)
        & (csv_local_mx[:, 0] < mx_w)
        & (csv_local_mx[:, 1] >= 0)
        & (csv_local_mx[:, 1] < mx_h)
    )
    csv_local_in_crop = csv_local_mx[in_mx_crop]

    if len(csv_local_in_crop) > 0:
        vals_csv = _sample_at_points(mx_dna, csv_local_in_crop)
        rng = np.random.default_rng(0)
        rx = rng.integers(0, mx_w, size=len(vals_csv))
        ry = rng.integers(0, mx_h, size=len(vals_csv))
        vals_rand = mx_dna[ry, rx]
        csv_med, csv_p90 = _median_p90(vals_csv)
        rnd_med, rnd_p90 = _median_p90(vals_rand)
    else:
        vals_csv = np.array([], dtype=np.float32)
        vals_rand = np.array([], dtype=np.float32)
        csv_med = csv_p90 = rnd_med = rnd_p90 = float("nan")

    q2: dict[str, Any] = {
        "csv_total": int(len(csv_full_mx)),
        "csv_inside_mx_crop": int(np.sum(in_mx_crop)),
        "mx_dna_at_csv_median": csv_med,
        "mx_dna_at_csv_p90": csv_p90,
        "mx_dna_at_random_median": rnd_med,
        "mx_dna_at_random_p90": rnd_p90,
    }
    if mask_path is not None:
        mask = tifffile.memmap(str(mask_path))
        mh, mw = mask.shape[-2], mask.shape[-1]
        xg = np.clip(np.rint(csv_full_mx[:, 0]).astype(int), 0, mw - 1)
        yg = np.clip(np.rint(csv_full_mx[:, 1]).astype(int), 0, mh - 1)
        hit_global = mask[yg, xg] > 0
        q2["csv_on_mask_global_rate"] = float(np.mean(hit_global))

        if mx_crop_origin is not None and len(csv_local_in_crop) > 0:
            csv_full_in_crop = csv_local_in_crop + mx_crop_origin
            xc = np.clip(np.rint(csv_full_in_crop[:, 0]).astype(int), 0, mw - 1)
            yc = np.clip(np.rint(csv_full_in_crop[:, 1]).astype(int), 0, mh - 1)
            hit_crop = mask[yc, xc] > 0
            q2["csv_on_mask_crop_rate"] = float(np.mean(hit_crop))
        else:
            q2["csv_on_mask_crop_rate"] = None
    report["q2_csv_vs_mx"] = q2

    # ------------------------------------------------------------------
    # Q3: transformed CSV vs transformed MX DNA
    # ------------------------------------------------------------------
    if mx_crop_origin is None:
        report["q3_transformed_csv_vs_transformed_mx_dna"] = {
            "skipped": True,
            "reason": "mx_crop_origin unavailable; cannot reliably map full-slide CSV to crop coordinates.",
        }
    else:
        mx_stage1, he_stage1 = csv_to_he_coords(
            csv_path=csv_path,
            m_full=m_full,
            csv_mpp=float(args.csv_mpp),
            crop_origin=tuple(mx_crop_origin.tolist()),
        )
        mx_stage25, he_stage25 = csv_to_he_coords(
            csv_path=csv_path,
            m_full=m_stage25_aff,
            csv_mpp=float(args.csv_mpp),
            crop_origin=tuple(mx_crop_origin.tolist()),
        )
        roi_bbox = patch_roi_bbox_he(index.get("patches", []), patch_size_default=index.get("patch_size", 256))
        if args.roi_margin is not None:
            roi_margin = float(args.roi_margin)
        else:
            roi_margin = float(index.get("csv_roi_margin_px", 0.5 * float(index.get("patch_size", 256))))

        _, he_stage1_roi, _ = filter_csv_to_patch_roi(mx_stage1, he_stage1, roi_bbox_he=roi_bbox, margin_px=roi_margin)
        _, he_stage25_roi, _ = filter_csv_to_patch_roi(mx_stage25, he_stage25, roi_bbox_he=roi_bbox, margin_px=roi_margin)

        dna_he_stage1 = cv2.warpAffine(
            mx_dna,
            m_full.astype(np.float64),
            (he_w, he_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        dna_he_stage25 = cv2.warpAffine(
            mx_dna,
            m_stage25_aff.astype(np.float64),
            (he_w, he_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        vals1 = _sample_at_points(dna_he_stage1, he_stage1_roi) if len(he_stage1_roi) else np.array([], dtype=np.float32)
        vals25 = _sample_at_points(dna_he_stage25, he_stage25_roi) if len(he_stage25_roi) else np.array([], dtype=np.float32)

        rng = np.random.default_rng(1)
        x0, x1, y0, y1 = roi_bbox
        x_lo = int(max(0, np.floor(x0 - roi_margin)))
        x_hi = int(min(he_w, np.ceil(x1 + roi_margin)))
        y_lo = int(max(0, np.floor(y0 - roi_margin)))
        y_hi = int(min(he_h, np.ceil(y1 + roi_margin)))
        n_rand = max(len(vals1), 1)
        rx = rng.integers(x_lo, max(x_lo + 1, x_hi), size=n_rand)
        ry = rng.integers(y_lo, max(y_lo + 1, y_hi), size=n_rand)
        rand1 = dna_he_stage1[ry, rx]
        rand25 = dna_he_stage25[ry, rx]

        report["q3_transformed_csv_vs_transformed_mx_dna"] = {
            "csv_stage1_roi_count": int(len(vals1)),
            "csv_stage25_affine_roi_count": int(len(vals25)),
            "stage1_median_at_csv": float(np.median(vals1)) if len(vals1) else float("nan"),
            "stage1_p90_at_csv": float(np.quantile(vals1, 0.9)) if len(vals1) else float("nan"),
            "stage1_median_at_random": float(np.median(rand1)),
            "stage1_p90_at_random": float(np.quantile(rand1, 0.9)),
            "stage25_affine_median_at_csv": float(np.median(vals25)) if len(vals25) else float("nan"),
            "stage25_affine_p90_at_csv": float(np.quantile(vals25, 0.9)) if len(vals25) else float("nan"),
            "stage25_affine_median_at_random": float(np.median(rand25)),
            "stage25_affine_p90_at_random": float(np.quantile(rand25, 0.9)),
        }

    # ------------------------------------------------------------------
    # Q4: transformed MX DNA close to HE (cross-modality proxies)
    # ------------------------------------------------------------------
    he_u8 = he_img.astype(np.uint8)
    he_nuc = (255.0 - he_u8[2].astype(np.float32)) if he_u8.shape[0] >= 3 else (255.0 - he_u8[0].astype(np.float32))
    dna_he_stage1 = cv2.warpAffine(
        mx_dna,
        m_full.astype(np.float64),
        (he_w, he_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    dna_he_stage25 = cv2.warpAffine(
        mx_dna,
        m_stage25_aff.astype(np.float64),
        (he_w, he_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    he_n = _normalize01(he_nuc)
    d1_n = _normalize01(dna_he_stage1)
    d25_n = _normalize01(dna_he_stage25)
    tissue = he_u8.mean(axis=0).astype(np.float32) < 245.0

    edge_he = np.hypot(cv2.Sobel(he_n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(he_n, cv2.CV_32F, 0, 1, ksize=3))
    edge_1 = np.hypot(cv2.Sobel(d1_n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d1_n, cv2.CV_32F, 0, 1, ksize=3))
    edge_25 = np.hypot(cv2.Sobel(d25_n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d25_n, cv2.CV_32F, 0, 1, ksize=3))

    report["q4_transformed_mx_close_to_he"] = {
        "stage1_intensity_corr": _pearson(he_n[tissue], d1_n[tissue]),
        "stage1_nmi": _nmi(he_n[tissue], d1_n[tissue]),
        "stage1_edge_corr": _pearson(edge_he[tissue], edge_1[tissue]),
        "stage25_affine_intensity_corr": _pearson(he_n[tissue], d25_n[tissue]),
        "stage25_affine_nmi": _nmi(he_n[tissue], d25_n[tissue]),
        "stage25_affine_edge_corr": _pearson(edge_he[tissue], edge_25[tissue]),
    }

    # ------------------------------------------------------------------
    # Q5: low match-rate diagnosis
    # ------------------------------------------------------------------
    he_pts, _ = load_he_centroids(
        cellvit_dir=processed / "cellvit",
        patches=index.get("patches", []),
        coord_scale=1.0,
    )
    if mx_crop_origin is None:
        report["q5_low_match_rate_diagnosis"] = {
            "skipped": True,
            "reason": "mx_crop_origin unavailable.",
        }
    elif len(he_pts) == 0:
        report["q5_low_match_rate_diagnosis"] = {
            "skipped": True,
            "reason": "No CellViT centroids found (run Stage 2 first).",
        }
    else:
        mx_pts, csv_he = csv_to_he_coords(
            csv_path=csv_path,
            m_full=m_full,
            csv_mpp=float(args.csv_mpp),
            crop_origin=tuple(mx_crop_origin.tolist()),
        )
        roi_bbox = patch_roi_bbox_he(index.get("patches", []), patch_size_default=index.get("patch_size", 256))
        if args.roi_margin is not None:
            roi_margin = float(args.roi_margin)
        else:
            roi_margin = float(index.get("csv_roi_margin_px", 0.5 * float(index.get("patch_size", 256))))
        mx_roi, csv_he_roi, _ = filter_csv_to_patch_roi(mx_pts, csv_he, roi_bbox_he=roi_bbox, margin_px=roi_margin)

        kdt = scipy.spatial.KDTree(csv_he_roi)
        d_pre, _ = kdt.query(he_pts, workers=-1)
        d_post, _ = kdt.query(apply_affine(m_icp, he_pts), workers=-1)

        gates_report: dict[str, Any] = {}
        for gate in args.distance_gates:
            gate_f = float(gate)
            pre_fwd = int(np.sum(d_pre <= gate_f))
            post_fwd = int(np.sum(d_post <= gate_f))
            pre_src, pre_dst = match_centroids_he(he_pts, he_pts, csv_he_roi, mx_roi, distance_gate=gate_f)
            post_src, post_dst = match_centroids_he(apply_affine(m_icp, he_pts), he_pts, csv_he_roi, mx_roi, distance_gate=gate_f)
            pre_in, _ = ransac_filter(pre_src, pre_dst, ransac_thresh=5.0)
            post_in, _ = ransac_filter(post_src, post_dst, ransac_thresh=5.0)
            gates_report[str(int(round(gate_f)))] = {
                "pre_forward": pre_fwd,
                "pre_mutual": int(len(pre_src)),
                "pre_inlier": int(len(pre_in)),
                "post_forward": post_fwd,
                "post_mutual": int(len(post_src)),
                "post_inlier": int(len(post_in)),
            }

        ratio = float(len(csv_he_roi) / max(1, len(he_pts)))
        diagnosis: list[str] = []
        if 0.7 <= ratio <= 1.3:
            diagnosis.append("CSV/HE counts are balanced in ROI; low match is not mainly due to count imbalance.")
        elif ratio > 1.3:
            diagnosis.append("CSV points substantially outnumber HE points in ROI; count imbalance likely contributes.")
        else:
            diagnosis.append("HE points outnumber CSV points in ROI; CSV sparsity likely contributes.")

        gate20 = gates_report.get("20")
        if gate20 is not None:
            inlier_rate_he = gate20["post_inlier"] / max(1, len(he_pts))
            if inlier_rate_he < 0.15:
                diagnosis.append("Low post-ICP inlier rate suggests geometric mismatch/outliers dominate.")
            else:
                diagnosis.append("Post-ICP inlier rate is moderate; residual mismatch may be local/non-rigid.")

        report["q5_low_match_rate_diagnosis"] = {
            "he_count": int(len(he_pts)),
            "csv_count_roi": int(len(csv_he_roi)),
            "csv_he_ratio": ratio,
            "nearest_distance_pre_mean": float(np.mean(d_pre)),
            "nearest_distance_pre_median": float(np.median(d_pre)),
            "nearest_distance_pre_p95": float(np.quantile(d_pre, 0.95)),
            "nearest_distance_post_mean": float(np.mean(d_post)),
            "nearest_distance_post_median": float(np.median(d_post)),
            "nearest_distance_post_p95": float(np.quantile(d_post, 0.95)),
            "by_gate": gates_report,
            "diagnosis": diagnosis,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("\n=== Stage 2.5 QC Report ===")
    print(f"processed: {processed}")
    print(f"index:     {index_path}")
    print(f"he image:  {he_path}")
    print(f"mx image:  {mx_path}")
    print(f"csv:       {csv_path}")
    if mask_path is not None:
        print(f"mask:      {mask_path}")
    print()

    q1 = report["q1_crop_consistency"]
    print("[Q1] Crop consistency")
    print(f"  HE px: {q1['he_size_px'][0]}x{q1['he_size_px'][1]}, MX px: {q1['mx_size_px'][0]}x{q1['mx_size_px'][1]}")
    print(f"  HE um: {q1['he_size_um'][0]:.1f}x{q1['he_size_um'][1]:.1f}, MX um: {q1['mx_size_um'][0]:.1f}x{q1['mx_size_um'][1]:.1f}")
    print(f"  Corners inside MX (stage1/stage25-aff): {q1['stage1_corners_inside_mx']}/4, {q1['stage25_affine_corners_inside_mx']}/4")

    q2p = report["q2_csv_vs_mx"]
    print("[Q2] CSV vs MX")
    print(f"  CSV total: {q2p['csv_total']}, inside MX crop: {q2p['csv_inside_mx_crop']}")
    print(
        "  DNA @CSV median/p90: "
        f"{q2p['mx_dna_at_csv_median']:.1f}/{q2p['mx_dna_at_csv_p90']:.1f} "
        f"vs random {q2p['mx_dna_at_random_median']:.1f}/{q2p['mx_dna_at_random_p90']:.1f}"
    )
    if "csv_on_mask_global_rate" in q2p:
        g = q2p["csv_on_mask_global_rate"]
        c = q2p["csv_on_mask_crop_rate"]
        print(f"  CSV-on-mask global: {100.0*g:.2f}%")
        if c is not None:
            print(f"  CSV-on-mask crop:   {100.0*c:.2f}%")

    q3 = report["q3_transformed_csv_vs_transformed_mx_dna"]
    print("[Q3] Transformed CSV vs transformed MX DNA")
    if q3.get("skipped"):
        print(f"  Skipped: {q3['reason']}")
    else:
        print(
            "  Stage1 median/p90 @CSV: "
            f"{q3['stage1_median_at_csv']:.1f}/{q3['stage1_p90_at_csv']:.1f}, "
            "random: "
            f"{q3['stage1_median_at_random']:.1f}/{q3['stage1_p90_at_random']:.1f}"
        )
        print(
            "  Stage25-aff median/p90 @CSV: "
            f"{q3['stage25_affine_median_at_csv']:.1f}/{q3['stage25_affine_p90_at_csv']:.1f}, "
            "random: "
            f"{q3['stage25_affine_median_at_random']:.1f}/{q3['stage25_affine_p90_at_random']:.1f}"
        )

    q4 = report["q4_transformed_mx_close_to_he"]
    print("[Q4] Transformed MX close to HE (proxy metrics)")
    print(
        "  Stage1 corr/NMI/edge: "
        f"{q4['stage1_intensity_corr']:.4f}/{q4['stage1_nmi']:.4f}/{q4['stage1_edge_corr']:.4f}"
    )
    print(
        "  Stage25-aff corr/NMI/edge: "
        f"{q4['stage25_affine_intensity_corr']:.4f}/{q4['stage25_affine_nmi']:.4f}/{q4['stage25_affine_edge_corr']:.4f}"
    )

    q5 = report["q5_low_match_rate_diagnosis"]
    print("[Q5] Low match-rate diagnosis")
    if q5.get("skipped"):
        print(f"  Skipped: {q5['reason']}")
    else:
        print(
            f"  Counts HE/CSV(ROI): {q5['he_count']}/{q5['csv_count_roi']} "
            f"(ratio={q5['csv_he_ratio']:.3f})"
        )
        print(
            f"  NN dist pre mean/med/p95: {q5['nearest_distance_pre_mean']:.2f}/"
            f"{q5['nearest_distance_pre_median']:.2f}/{q5['nearest_distance_pre_p95']:.2f}"
        )
        print(
            f"  NN dist post mean/med/p95: {q5['nearest_distance_post_mean']:.2f}/"
            f"{q5['nearest_distance_post_median']:.2f}/{q5['nearest_distance_post_p95']:.2f}"
        )
        for gate, stats in q5["by_gate"].items():
            print(
                f"  gate={gate}px pre fwd/mut/in={stats['pre_forward']}/{stats['pre_mutual']}/{stats['pre_inlier']} "
                f"post={stats['post_forward']}/{stats['post_mutual']}/{stats['post_inlier']}"
            )
        for line in q5["diagnosis"]:
            print(f"  - {line}")

    if args.report_json:
        out_json = pathlib.Path(args.report_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_to_jsonable(report), indent=2))
        print(f"\nWrote JSON report: {out_json}")


if __name__ == "__main__":
    main()
