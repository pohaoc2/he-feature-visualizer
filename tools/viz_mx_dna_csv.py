#!/usr/bin/env python3
"""
viz_mx_dna_csv.py — 6-panel MX DNA + CSV centroid visualization.

Panels:
  a) MX DNA overview
  b) CSV centroids (MX space)
  c) Overlap: MX DNA + CSV centroids, with red central-region square
  d) Zoom-in (central MX region, default 1024x1024 full-res px)
  e) Transformed overlap (MX DNA + CSV warped to H&E via inv(warp_matrix))
  f) Zoom-in of transformed overlap (central H&E region)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tifffile

from stages.refine_registration import resolve_mx_crop_origin
from utils.normalize import percentile_to_uint8
from utils.ome import get_image_dims, open_zarr_store


def _read_channel_overview(
    store, axes: str, img_h: int, img_w: int, ds: int, c_idx: int
) -> np.ndarray:
    """Read one channel at overview resolution as (H, W)."""
    ax = axes.upper()
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds
    sl: list[int | slice] = []
    for a in ax:
        if a == "C":
            sl.append(c_idx)
        elif a == "Y":
            sl.append(slice(0, h_t, ds))
        elif a == "X":
            sl.append(slice(0, w_t, ds))
        else:
            sl.append(0)
    arr = np.asarray(store[tuple(sl)])
    active = [a for a in ax if a in ("Y", "X")]
    target = [a for a in ("Y", "X") if a in active]
    if active != target:
        arr = arr.transpose([active.index(a) for a in target])
    return arr


def _load_csv_points_mx(csv_path: Path, csv_mpp: float) -> np.ndarray:
    """Load CSV centroids in MX px."""
    cols = set(pd.read_csv(csv_path, nrows=0).columns.tolist())

    if {"Xt_mx_px", "Yt_mx_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_mx_px", "Yt_mx_px"])
        return df[["Xt_mx_px", "Yt_mx_px"]].to_numpy(dtype=np.float64)

    if {"Xt", "Yt"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt", "Yt"])
        pts_um = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
        return pts_um / float(csv_mpp)

    raise ValueError(
        "CSV must include Xt/Yt (um) or Xt_mx_px/Yt_mx_px columns for MX-space plotting."
    )


def _sample_points(
    pts: np.ndarray, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled points and selected indices."""
    n = len(pts)
    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
        return pts, idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx], idx


def _central_box(img_w: int, img_h: int, size: int) -> tuple[int, int, int, int]:
    """Return central box [x0, y0, x1, y1] in full-res pixels."""
    size = int(max(1, size))
    x0 = max(0, (img_w - size) // 2)
    y0 = max(0, (img_h - size) // 2)
    x1 = min(img_w, x0 + size)
    y1 = min(img_h, y0 + size)
    return x0, y0, x1, y1


def _invert_affine_2x3(m: np.ndarray) -> np.ndarray:
    """Return inverse of a 2x3 affine matrix as 2x3 float64."""
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    m3_inv = np.linalg.inv(m3)
    return m3_inv[:2, :]


def _apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points."""
    if len(pts) == 0:
        return pts.copy()
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (m @ np.hstack([pts.astype(np.float64), ones]).T).T


def _affine_he_to_mx_overview(
    m_full: np.ndarray,
    sx_he: float,
    sy_he: float,
    sx_mx: float,
    sy_mx: float,
) -> np.ndarray:
    """Convert full-res HE->MX affine into overview-space HE->MX affine."""
    a = m_full[:, :2].astype(np.float64)
    t = m_full[:, 2].astype(np.float64)
    s_src = np.diag([sx_he, sy_he]).astype(np.float64)
    s_dst_inv = np.diag([1.0 / sx_mx, 1.0 / sy_mx]).astype(np.float64)
    a_ov = s_dst_inv @ a @ s_src
    t_ov = s_dst_inv @ t
    return np.column_stack([a_ov, t_ov])


def _get_index_mx_size(index_payload: dict) -> tuple[int, int] | None:
    """Return expected crop MX size from index metadata when available."""
    crop_region = index_payload.get("crop_region")
    if not isinstance(crop_region, dict):
        return None
    mx_size = crop_region.get("mx_size")
    if isinstance(mx_size, (list, tuple)) and len(mx_size) >= 2:
        try:
            return int(mx_size[0]), int(mx_size[1])
        except (TypeError, ValueError):
            return None
    return None


def _get_index_he_size(index_payload: dict) -> tuple[int, int] | None:
    """Return expected HE size from index metadata when available."""
    img_w = index_payload.get("img_w")
    img_h = index_payload.get("img_h")
    if img_w is not None and img_h is not None:
        try:
            return int(img_w), int(img_h)
        except (TypeError, ValueError):
            pass

    crop_region = index_payload.get("crop_region")
    if isinstance(crop_region, dict):
        he_size = crop_region.get("he_size")
        if isinstance(he_size, (list, tuple)) and len(he_size) >= 2:
            try:
                return int(he_size[0]), int(he_size[1])
            except (TypeError, ValueError):
                return None
    return None


def _set_axes_bounds(ax, width: int, height: int) -> None:
    """Show pixel axes and clamp view to image bounds."""
    w = max(1, int(width))
    h = max(1, int(height))
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)  # image coordinates: y increases downward
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.tick_params(axis="both", labelsize=8)


def main() -> None:
    p = argparse.ArgumentParser(
        description="6-panel MX DNA + CSV centroid visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multiplex-image", required=True, help="Multiplex OME-TIFF path.")
    p.add_argument("--csv", required=True, help="CSV path with centroid columns.")
    p.add_argument("--out-png", required=True, help="Output PNG path.")
    p.add_argument("--dna-channel", type=int, default=0, help="DNA channel index.")
    p.add_argument("--downsample", type=int, default=8, help="Overview downsample.")
    p.add_argument("--csv-mpp", type=float, default=0.65, help="CSV um/px scale.")
    p.add_argument(
        "--mx-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Optional crop origin (full-slide MX px) to subtract from CSV points.",
    )
    p.add_argument(
        "--index",
        default=None,
        help="Optional index.json to infer mx_crop_origin when --mx-crop-origin is not set.",
    )
    p.add_argument(
        "--force-index-crop-origin",
        action="store_true",
        help="Force applying mx_crop_origin from --index even when MX image size "
        "does not match index crop size metadata.",
    )
    p.add_argument(
        "--zoom-size",
        type=int,
        default=1024,
        help="Central zoom size in full-res MX px.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=300000,
        help="Max centroids to plot (sampled) for readability.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Centroid marker size.",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=1500,
        help="Max centroid trajectories to draw in transformed panels e/f.",
    )
    p.add_argument(
        "--trajectory-lw",
        type=float,
        default=0.45,
        help="Line width for centroid trajectories in transformed panels e/f.",
    )
    p.add_argument(
        "--trajectory-alpha",
        type=float,
        default=0.35,
        help="Alpha for centroid trajectories in transformed panels e/f.",
    )
    p.add_argument(
        "--trajectory-color",
        default="yellow",
        help="Color for centroid trajectories in transformed panels e/f.",
    )
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = p.parse_args()

    ds = max(1, int(args.downsample))
    mx_path = Path(args.multiplex_image)
    csv_path = Path(args.csv)
    out_path = Path(args.out_png)

    with tifffile.TiffFile(mx_path) as tif:
        mx_w, mx_h, mx_axes = get_image_dims(tif)
        mx_store = open_zarr_store(tif)
        dna_ov = _read_channel_overview(
            mx_store,
            mx_axes,
            mx_h,
            mx_w,
            ds,
            c_idx=int(args.dna_channel),
        )

    payload = None
    if args.index:
        with open(args.index, encoding="utf-8") as f:
            payload = json.load(f)
    cli_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None

    crop_origin: tuple[float, float] | None = None
    if cli_origin is not None:
        crop_origin = resolve_mx_crop_origin(payload or {}, cli_origin=cli_origin)
    elif payload:
        inferred_origin = resolve_mx_crop_origin(payload, cli_origin=None)
        expected_mx_size = _get_index_mx_size(payload)
        if inferred_origin is not None:
            if expected_mx_size is None:
                crop_origin = inferred_origin
            else:
                exp_w, exp_h = expected_mx_size
                size_matches = (mx_w == exp_w) and (mx_h == exp_h)
                if size_matches or args.force_index_crop_origin:
                    crop_origin = inferred_origin
                else:
                    print(
                        "WARNING: index.json has mx_crop_origin for crop size "
                        f"{exp_w}x{exp_h}, but multiplex image is {mx_w}x{mx_h}. "
                        "Skipping crop-origin shift (likely full-slide MX input). "
                        "Use --force-index-crop-origin to override."
                    )

    dna_u8 = percentile_to_uint8(dna_ov.astype(np.float32))
    ov_h, ov_w = dna_u8.shape
    sx = mx_w / float(max(1, ov_w))
    sy = mx_h / float(max(1, ov_h))

    pts_mx = _load_csv_points_mx(csv_path, csv_mpp=float(args.csv_mpp))
    if crop_origin is not None:
        pts_mx = pts_mx - np.array(crop_origin, dtype=np.float64)

    in_bounds = (
        (pts_mx[:, 0] >= 0)
        & (pts_mx[:, 0] < mx_w)
        & (pts_mx[:, 1] >= 0)
        & (pts_mx[:, 1] < mx_h)
    )
    pts_in = pts_mx[in_bounds]
    pts_plot, _ = _sample_points(pts_in, max_points=int(args.max_points), seed=args.seed)
    pts_ov = np.column_stack([pts_plot[:, 0] / sx, pts_plot[:, 1] / sy])

    x0, y0, x1, y1 = _central_box(mx_w, mx_h, int(args.zoom_size))
    x0_ov = int(np.floor(x0 / sx))
    x1_ov = int(np.ceil(x1 / sx))
    y0_ov = int(np.floor(y0 / sy))
    y1_ov = int(np.ceil(y1 / sy))
    x0_ov = max(0, min(x0_ov, ov_w - 1))
    y0_ov = max(0, min(y0_ov, ov_h - 1))
    x1_ov = max(x0_ov + 1, min(x1_ov, ov_w))
    y1_ov = max(y0_ov + 1, min(y1_ov, ov_h))

    zoom_bg = dna_u8[y0_ov:y1_ov, x0_ov:x1_ov]
    in_zoom = (
        (pts_plot[:, 0] >= x0)
        & (pts_plot[:, 0] < x1)
        & (pts_plot[:, 1] >= y0)
        & (pts_plot[:, 1] < y1)
    )
    pts_zoom = pts_plot[in_zoom]
    pts_zoom_ov = np.column_stack([pts_zoom[:, 0] / sx - x0_ov, pts_zoom[:, 1] / sy - y0_ov])

    # Optional transformed view: warp MX DNA + CSV from MX -> HE using inv(warp_matrix)
    he_w = he_h = None
    dna_he_u8 = None
    pts_plot_he = np.empty((0, 2), dtype=np.float64)
    pts_plot_he_in = np.empty((0, 2), dtype=np.float64)
    pts_plot_he_orig_ov = np.empty((0, 2), dtype=np.float64)
    pts_plot_he_in_ov = np.empty((0, 2), dtype=np.float64)
    pts_zoom_he_ov = np.empty((0, 2), dtype=np.float64)
    pts_zoom_he_orig_ov = np.empty((0, 2), dtype=np.float64)
    traj_start_ov = np.empty((0, 2), dtype=np.float64)
    traj_end_ov = np.empty((0, 2), dtype=np.float64)
    traj_zoom_start_ov = np.empty((0, 2), dtype=np.float64)
    traj_zoom_end_ov = np.empty((0, 2), dtype=np.float64)
    tf_x0_ov = tf_y0_ov = tf_x1_ov = tf_y1_ov = 0
    transformed_ready = False

    if payload and isinstance(payload.get("warp_matrix"), list):
        try:
            m_full = np.asarray(payload["warp_matrix"], dtype=np.float64)
            if m_full.shape != (2, 3):
                raise ValueError(f"warp_matrix shape is {m_full.shape}, expected (2,3)")

            he_size = _get_index_he_size(payload)
            if he_size is not None:
                he_w, he_h = he_size
                he_ov_w = max(1, he_w // ds)
                he_ov_h = max(1, he_h // ds)
                sx_he = he_w / float(he_ov_w)
                sy_he = he_h / float(he_ov_h)

                m_he_to_mx_ov = _affine_he_to_mx_overview(
                    m_full,
                    sx_he=sx_he,
                    sy_he=sy_he,
                    sx_mx=sx,
                    sy_mx=sy,
                )
                m_mx_to_he_ov = _invert_affine_2x3(m_he_to_mx_ov)
                dna_he_u8 = cv2.warpAffine(
                    dna_u8,
                    m_mx_to_he_ov.astype(np.float32),
                    (he_ov_w, he_ov_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                m_mx_to_he = _invert_affine_2x3(m_full)
                pts_plot_he = _apply_affine(m_mx_to_he, pts_plot)
                in_he = (
                    (pts_plot_he[:, 0] >= 0)
                    & (pts_plot_he[:, 0] < he_w)
                    & (pts_plot_he[:, 1] >= 0)
                    & (pts_plot_he[:, 1] < he_h)
                )
                pts_plot_he_in = pts_plot_he[in_he]

                # Trajectories: scale-only MX->HE baseline -> affine-warped MX->HE.
                # This visualizes centroid displacement induced by warp_matrix.
                pts_plot_he_base = np.column_stack(
                    [
                        pts_plot[:, 0] * (he_w / float(mx_w)),
                        pts_plot[:, 1] * (he_h / float(mx_h)),
                    ]
                )[in_he]
                pts_plot_he_orig_in = pts_plot_he_base
                pts_plot_he_orig_ov = np.column_stack(
                    [pts_plot_he_orig_in[:, 0] / sx_he, pts_plot_he_orig_in[:, 1] / sy_he]
                )
                pts_plot_he_in_ov = np.column_stack(
                    [pts_plot_he_in[:, 0] / sx_he, pts_plot_he_in[:, 1] / sy_he]
                )

                traj_start_he = pts_plot_he_orig_in
                traj_end_he = pts_plot_he_in
                max_traj = max(0, int(args.max_trajectories))
                if max_traj > 0 and len(traj_end_he) > max_traj:
                    rng = np.random.default_rng(args.seed + 17)
                    t_idx = rng.choice(len(traj_end_he), size=max_traj, replace=False)
                    traj_start_he = traj_start_he[t_idx]
                    traj_end_he = traj_end_he[t_idx]

                if len(traj_end_he) > 0:
                    traj_start_ov = np.column_stack(
                        [traj_start_he[:, 0] / sx_he, traj_start_he[:, 1] / sy_he]
                    )
                    traj_end_ov = np.column_stack(
                        [traj_end_he[:, 0] / sx_he, traj_end_he[:, 1] / sy_he]
                    )

                tf_x0, tf_y0, tf_x1, tf_y1 = _central_box(
                    he_w, he_h, int(args.zoom_size)
                )
                tf_x0_ov = int(np.floor(tf_x0 / sx_he))
                tf_x1_ov = int(np.ceil(tf_x1 / sx_he))
                tf_y0_ov = int(np.floor(tf_y0 / sy_he))
                tf_y1_ov = int(np.ceil(tf_y1 / sy_he))
                tf_x0_ov = max(0, min(tf_x0_ov, he_ov_w - 1))
                tf_y0_ov = max(0, min(tf_y0_ov, he_ov_h - 1))
                tf_x1_ov = max(tf_x0_ov + 1, min(tf_x1_ov, he_ov_w))
                tf_y1_ov = max(tf_y0_ov + 1, min(tf_y1_ov, he_ov_h))

                in_tf_zoom = (
                    (pts_plot_he_in[:, 0] >= tf_x0)
                    & (pts_plot_he_in[:, 0] < tf_x1)
                    & (pts_plot_he_in[:, 1] >= tf_y0)
                    & (pts_plot_he_in[:, 1] < tf_y1)
                )
                pts_zoom_he = pts_plot_he_in[in_tf_zoom]
                pts_zoom_he_orig = pts_plot_he_orig_in[in_tf_zoom]
                pts_zoom_he_ov = np.column_stack(
                    [
                        pts_zoom_he[:, 0] / sx_he - tf_x0_ov,
                        pts_zoom_he[:, 1] / sy_he - tf_y0_ov,
                    ]
                )
                pts_zoom_he_orig_ov = np.column_stack(
                    [
                        pts_zoom_he_orig[:, 0] / sx_he - tf_x0_ov,
                        pts_zoom_he_orig[:, 1] / sy_he - tf_y0_ov,
                    ]
                )

                if len(traj_end_ov) > 0:
                    traj_end_in_tf_zoom = (
                        (traj_end_ov[:, 0] >= tf_x0_ov)
                        & (traj_end_ov[:, 0] < tf_x1_ov)
                        & (traj_end_ov[:, 1] >= tf_y0_ov)
                        & (traj_end_ov[:, 1] < tf_y1_ov)
                    )
                    traj_start_in_tf_zoom = (
                        (traj_start_ov[:, 0] >= tf_x0_ov)
                        & (traj_start_ov[:, 0] < tf_x1_ov)
                        & (traj_start_ov[:, 1] >= tf_y0_ov)
                        & (traj_start_ov[:, 1] < tf_y1_ov)
                    )
                    traj_in_tf_zoom = traj_end_in_tf_zoom & traj_start_in_tf_zoom
                    traj_zoom_start_ov = np.column_stack(
                        [
                            traj_start_ov[traj_in_tf_zoom, 0] - tf_x0_ov,
                            traj_start_ov[traj_in_tf_zoom, 1] - tf_y0_ov,
                        ]
                    )
                    traj_zoom_end_ov = np.column_stack(
                        [
                            traj_end_ov[traj_in_tf_zoom, 0] - tf_x0_ov,
                            traj_end_ov[traj_in_tf_zoom, 1] - tf_y0_ov,
                        ]
                    )
                transformed_ready = True
            else:
                print(
                    "WARNING: index has warp_matrix but no HE size metadata "
                    "(img_w/img_h or crop_region.he_size); skipping panels e/f."
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"WARNING: failed to build transformed view (e/f): {exc}")

    print(f"MX dimensions: {mx_w} x {mx_h} px")
    print(f"Overview: {ov_w} x {ov_h} (downsample={ds}, sx={sx:.3f}, sy={sy:.3f})")
    print(f"CSV points total: {len(pts_mx):,}")
    print(f"CSV points in bounds: {len(pts_in):,}")
    print(f"CSV points plotted: {len(pts_plot):,}")
    print(f"Central zoom full-res: x=[{x0},{x1}) y=[{y0},{y1})")
    if crop_origin is not None:
        print(f"Applied mx_crop_origin: ({crop_origin[0]:.1f}, {crop_origin[1]:.1f})")
    if transformed_ready and he_w is not None and he_h is not None:
        print(f"Transformed HE canvas: {he_w} x {he_h} px")
        print(
            f"Transformed points in HE bounds: {len(pts_plot_he_in):,} / {len(pts_plot):,}"
        )

    # Figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=160)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    ax_e, ax_f = axes[2, 0], axes[2, 1]

    # a) MX DNA
    ax_a.imshow(dna_u8, cmap="gray", vmin=0, vmax=255)
    ax_a.set_title("a) MX DNA")
    bounds_a = (ov_w, ov_h)

    # b) centroids only
    blank = np.zeros((ov_h, ov_w), dtype=np.uint8)
    ax_b.imshow(blank, cmap="gray", vmin=0, vmax=255)
    if len(pts_ov) > 0:
        ax_b.scatter(
            pts_ov[:, 0],
            pts_ov[:, 1],
            s=float(args.point_size),
            c="cyan",
            alpha=0.65,
            linewidths=0,
        )
    ax_b.set_title("b) CSV centroids")
    bounds_b = (ov_w, ov_h)

    # c) overlap + red central square
    ax_c.imshow(dna_u8, cmap="gray", vmin=0, vmax=255)
    if len(pts_ov) > 0:
        ax_c.scatter(
            pts_ov[:, 0],
            pts_ov[:, 1],
            s=float(args.point_size),
            c="cyan",
            alpha=0.65,
            linewidths=0,
        )
    rect = Rectangle(
        (x0_ov, y0_ov),
        x1_ov - x0_ov,
        y1_ov - y0_ov,
        linewidth=1.8,
        edgecolor="red",
        facecolor="none",
    )
    ax_c.add_patch(rect)
    ax_c.set_title(
        f"c) Overlap (MX DNA + CSV) with central {int(args.zoom_size)}x{int(args.zoom_size)} box"
    )
    bounds_c = (ov_w, ov_h)

    # d) zoom center region
    ax_d.imshow(zoom_bg, cmap="gray", vmin=0, vmax=255)
    if len(pts_zoom_ov) > 0:
        ax_d.scatter(
            pts_zoom_ov[:, 0],
            pts_zoom_ov[:, 1],
            s=max(2.0, float(args.point_size) * 1.2),
            c="cyan",
            alpha=0.75,
            linewidths=0,
        )
    ax_d.set_title(f"d) Central zoom ({int(args.zoom_size)}x{int(args.zoom_size)})")
    bounds_d = (zoom_bg.shape[1], zoom_bg.shape[0])

    # e) transformed overlap (MX -> HE using inverse warp_matrix)
    if transformed_ready and dna_he_u8 is not None:
        ax_e.imshow(dna_he_u8, cmap="gray", vmin=0, vmax=255)
        if len(pts_plot_he_orig_ov) > 0:
            ax_e.scatter(
                pts_plot_he_orig_ov[:, 0],
                pts_plot_he_orig_ov[:, 1],
                s=max(1.5, float(args.point_size) * 0.9),
                c="orange",
                alpha=0.5,
                linewidths=0,
            )
        if len(traj_end_ov) > 0 and float(args.trajectory_lw) > 0:
            seg = np.stack([traj_start_ov, traj_end_ov], axis=1)
            lc = LineCollection(
                seg,
                colors=args.trajectory_color,
                linewidths=float(args.trajectory_lw),
                alpha=float(args.trajectory_alpha),
            )
            ax_e.add_collection(lc)
        if len(pts_plot_he_in_ov) > 0:
            ax_e.scatter(
                pts_plot_he_in_ov[:, 0],
                pts_plot_he_in_ov[:, 1],
                s=float(args.point_size),
                c="cyan",
                alpha=0.65,
                linewidths=0,
            )
        rect_tf = Rectangle(
            (tf_x0_ov, tf_y0_ov),
            tf_x1_ov - tf_x0_ov,
            tf_y1_ov - tf_y0_ov,
            linewidth=1.8,
            edgecolor="red",
            facecolor="none",
        )
        ax_e.add_patch(rect_tf)
        ax_e.set_title("e) Original (orange) -> transformed (cyan)")
        bounds_e = (dna_he_u8.shape[1], dna_he_u8.shape[0])
    else:
        ax_e.imshow(np.zeros_like(dna_u8), cmap="gray", vmin=0, vmax=255)
        ax_e.set_title("e) Transformed overlap unavailable (need --index warp_matrix)")
        bounds_e = (dna_u8.shape[1], dna_u8.shape[0])

    # f) transformed zoom
    if transformed_ready and dna_he_u8 is not None:
        tf_zoom_bg = dna_he_u8[tf_y0_ov:tf_y1_ov, tf_x0_ov:tf_x1_ov]
        ax_f.imshow(tf_zoom_bg, cmap="gray", vmin=0, vmax=255)
        if len(pts_zoom_he_orig_ov) > 0:
            ax_f.scatter(
                pts_zoom_he_orig_ov[:, 0],
                pts_zoom_he_orig_ov[:, 1],
                s=max(1.8, float(args.point_size)),
                c="orange",
                alpha=0.55,
                linewidths=0,
            )
        if len(traj_zoom_end_ov) > 0 and float(args.trajectory_lw) > 0:
            seg_zoom = np.stack([traj_zoom_start_ov, traj_zoom_end_ov], axis=1)
            lc_zoom = LineCollection(
                seg_zoom,
                colors=args.trajectory_color,
                linewidths=max(0.35, float(args.trajectory_lw) * 1.1),
                alpha=min(1.0, float(args.trajectory_alpha) + 0.1),
            )
            ax_f.add_collection(lc_zoom)
        if len(pts_zoom_he_ov) > 0:
            ax_f.scatter(
                pts_zoom_he_ov[:, 0],
                pts_zoom_he_ov[:, 1],
                s=max(2.0, float(args.point_size) * 1.2),
                c="cyan",
                alpha=0.75,
                linewidths=0,
            )
        ax_f.set_title("f) Transformed central zoom")
        bounds_f = (tf_zoom_bg.shape[1], tf_zoom_bg.shape[0])
    else:
        ax_f.imshow(np.zeros_like(zoom_bg), cmap="gray", vmin=0, vmax=255)
        ax_f.set_title("f) Transformed zoom unavailable")
        bounds_f = (zoom_bg.shape[1], zoom_bg.shape[0])

    for ax, (w, h) in (
        (ax_a, bounds_a),
        (ax_b, bounds_b),
        (ax_c, bounds_c),
        (ax_d, bounds_d),
        (ax_e, bounds_e),
        (ax_f, bounds_f),
    ):
        _set_axes_bounds(ax, w, h)
        ax.set_facecolor("black")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
