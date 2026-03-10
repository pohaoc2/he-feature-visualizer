#!/usr/bin/env python3
"""
Create HE + centroid visualizations from transformed CSV coordinates.

Outputs:
1) HE overview image
2) Centroid scatter mapped to HE coordinates
3) HE + centroid overlap (overview)
4) Dense-region overlap zoom (512 x 512 by default)

Also saves a summary panel:
- 2x2 when only HE/centroid visualizations are requested
- 2x3 when MX mask visualizations are enabled
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

# Ensure matplotlib has writable cache directories in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from utils.normalize import percentile_to_uint8
from utils.ome import get_image_dims

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _to_chw(arr: np.ndarray, axes: str) -> np.ndarray:
    ax = axes.upper()
    active = [a for a in ax if a in ("C", "Y", "X")]
    if "C" in active:
        target = [a for a in ("C", "Y", "X") if a in active]
        if active != target:
            arr = arr.transpose([active.index(a) for a in target])
        return arr

    target = [a for a in ("Y", "X") if a in active]
    if active != target:
        arr = arr.transpose([active.index(a) for a in target])
    return arr[np.newaxis, ...]


def _read_he_overview(
    he_path: Path,
    downsample: int,
) -> tuple[np.ndarray, int, int, float, float, int]:
    with tifffile.TiffFile(str(he_path)) as tif:
        img_w, img_h, axes = get_image_dims(tif)
        series = tif.series[0]
        ax = axes.upper()
        x_idx = ax.index("X")
        y_idx = ax.index("Y")

        best_level_idx = 0
        best_score = float("inf")
        for i, level in enumerate(series.levels):
            lvl_w = level.shape[x_idx]
            lvl_h = level.shape[y_idx]
            ds_x = img_w / float(lvl_w)
            ds_y = img_h / float(lvl_h)
            score = abs(ds_x - downsample) + abs(ds_y - downsample)
            if score < best_score:
                best_score = score
                best_level_idx = i

        arr = series.levels[best_level_idx].asarray()
        he_chw = _to_chw(arr, axes)

    if he_chw.shape[0] >= 3:
        he_chw = he_chw[:3]
    else:
        he_chw = np.repeat(he_chw[:1], 3, axis=0)

    if he_chw.dtype != np.uint8:
        he_chw = percentile_to_uint8(he_chw)

    he_rgb = np.moveaxis(he_chw, 0, -1)
    ov_h, ov_w = he_rgb.shape[:2]
    scale_x = img_w / float(ov_w)
    scale_y = img_h / float(ov_h)
    return he_rgb, img_w, img_h, scale_x, scale_y, best_level_idx


def _read_he_patch(he_path: Path, x0: int, y0: int, size: int) -> np.ndarray:
    # PIL supports tiled TIFF region reads via crop() without loading full slide.
    Image.MAX_IMAGE_PIXELS = None
    x1 = x0 + size
    y1 = y0 + size
    channels: list[np.ndarray] = []
    with Image.open(str(he_path)) as img:
        n_frames = int(getattr(img, "n_frames", 1))
        n_take = min(3, n_frames)
        for i in range(n_take):
            img.seek(i)
            patch = img.crop((x0, y0, x1, y1))
            channels.append(np.array(patch))

    if len(channels) == 0:
        raise ValueError(f"No readable frames found in {he_path}")
    if len(channels) == 1:
        channels = [channels[0], channels[0], channels[0]]
    if len(channels) == 2:
        channels = [channels[0], channels[1], channels[1]]
    return np.stack(channels[:3], axis=-1)


def _sample_points(points_xy: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points_xy) <= max_points:
        return points_xy
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points_xy), size=max_points, replace=False)
    return points_xy[idx]


def _infer_scale_he_to_mx(
    index_payload: dict | None,
    he_mpp: float,
    mx_mpp: float,
    scale_he_to_mx: float | None,
) -> float:
    if scale_he_to_mx is not None:
        return float(scale_he_to_mx)
    if index_payload and index_payload.get("scale_he_to_mx") is not None:
        return float(index_payload["scale_he_to_mx"])
    return float(he_mpp) / float(mx_mpp)


def _load_points_he(
    csv_path: Path,
    scale_he_to_mx: float,
    mx_mpp: float,
) -> tuple[np.ndarray, str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = set(header)

    if {"Xt_he_px", "Yt_he_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_he_px", "Yt_he_px"])
        pts_he = df[["Xt_he_px", "Yt_he_px"]].to_numpy(dtype=np.float64)
        return pts_he, "Xt_he_px/Yt_he_px (from transformed CSV)"

    if {"Xt_mx_px", "Yt_mx_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_mx_px", "Yt_mx_px"])
        pts_mx = df[["Xt_mx_px", "Yt_mx_px"]].to_numpy(dtype=np.float64)
        pts_he = pts_mx / float(scale_he_to_mx)
        return pts_he, "Xt_mx_px/Yt_mx_px converted with HE->MX scale"

    if {"Xt", "Yt"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt", "Yt"])
        pts_um = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
        pts_mx = pts_um / float(mx_mpp)
        pts_he = pts_mx / float(scale_he_to_mx)
        return pts_he, "Xt/Yt (um) -> MX px via mx_mpp -> HE px via scale"

    raise ValueError("CSV must have Xt_he/Yt_he, or Xt_mx/Yt_mx, or Xt/Yt columns.")


def _load_points_mx(
    csv_path: Path,
    scale_he_to_mx: float,
    mx_mpp: float,
) -> tuple[np.ndarray, str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = set(header)

    if {"Xt_mx_px", "Yt_mx_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_mx_px", "Yt_mx_px"])
        pts_mx = df[["Xt_mx_px", "Yt_mx_px"]].to_numpy(dtype=np.float64)
        return pts_mx, "Xt_mx_px/Yt_mx_px (from transformed CSV)"

    if {"Xt", "Yt"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt", "Yt"])
        pts_um = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
        pts_mx = pts_um / float(mx_mpp)
        return pts_mx, "Xt/Yt (um) -> MX px via mx_mpp"

    if {"Xt_he_px", "Yt_he_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_he_px", "Yt_he_px"])
        pts_he = df[["Xt_he_px", "Yt_he_px"]].to_numpy(dtype=np.float64)
        pts_mx = pts_he * float(scale_he_to_mx)
        return pts_mx, "Xt_he_px/Yt_he_px -> MX px via HE->MX scale"

    raise ValueError("CSV must have Xt_mx/Yt_mx, Xt/Yt, or Xt_he/Yt_he columns.")


def _find_dense_window(
    points_he: np.ndarray,
    img_w: int,
    img_h: int,
    window_size: int,
) -> tuple[int, int, int]:
    inside = (
        (points_he[:, 0] >= 0)
        & (points_he[:, 0] < img_w)
        & (points_he[:, 1] >= 0)
        & (points_he[:, 1] < img_h)
    )
    pts = points_he[inside]
    if len(pts) == 0:
        return 0, 0, 0

    x = np.floor(pts[:, 0]).astype(np.int64)
    y = np.floor(pts[:, 1]).astype(np.int64)

    bins_x = int(np.ceil(img_w / window_size))
    bins_y = int(np.ceil(img_h / window_size))

    bx = np.clip(x // window_size, 0, bins_x - 1)
    by = np.clip(y // window_size, 0, bins_y - 1)

    counts = np.zeros((bins_y, bins_x), dtype=np.int64)
    np.add.at(counts, (by, bx), 1)

    iy, ix = np.unravel_index(np.argmax(counts), counts.shape)
    x0 = int(ix * window_size)
    y0 = int(iy * window_size)

    x0 = min(max(0, x0), max(0, img_w - window_size))
    y0 = min(max(0, y0), max(0, img_h - window_size))

    return x0, y0, int(counts[iy, ix])


def _prepare_mask_visualization(
    mask_path: Path,
    csv_path: Path,
    zoom_size: int,
    mx_mpp: float,
    scale_he_to_mx: float,
    max_points_overview: int,
    mx_downsample: int,
    seed: int,
) -> dict:
    mask_mm = tifffile.memmap(str(mask_path))
    while mask_mm.ndim > 2:
        mask_mm = mask_mm[0]
    if mask_mm.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask_mm.shape}")

    mx_h, mx_w = int(mask_mm.shape[0]), int(mask_mm.shape[1])
    points_mx, conversion_mode_mx = _load_points_mx(
        csv_path=csv_path,
        scale_he_to_mx=scale_he_to_mx,
        mx_mpp=mx_mpp,
    )

    in_bounds = (
        (points_mx[:, 0] >= 0)
        & (points_mx[:, 0] < mx_w)
        & (points_mx[:, 1] >= 0)
        & (points_mx[:, 1] < mx_h)
    )
    pts_in = points_mx[in_bounds]
    xi = np.floor(pts_in[:, 0]).astype(np.int64)
    yi = np.floor(pts_in[:, 1]).astype(np.int64)
    hit_in = mask_mm[yi, xi] > 0

    pts_hit = pts_in[hit_in]
    pts_miss = pts_in[~hit_in]
    dense_points = pts_hit if len(pts_hit) else pts_in

    mx_zoom_x0, mx_zoom_y0, dense_count_mx = _find_dense_window(
        points_he=dense_points,
        img_w=mx_w,
        img_h=mx_h,
        window_size=zoom_size,
    )

    downsample = max(1, int(mx_downsample))
    mask_ov = mask_mm[::downsample, ::downsample]
    mask_ov_vis = np.where(mask_ov > 0, 230, 20).astype(np.uint8)
    mask_ov_rgb = np.repeat(mask_ov_vis[..., np.newaxis], 3, axis=2)
    mx_ov_h, mx_ov_w = mask_ov_rgb.shape[:2]
    mx_ov_scale_x = mx_w / float(mx_ov_w)
    mx_ov_scale_y = mx_h / float(mx_ov_h)

    pts_hit_plot = _sample_points(pts_hit, max_points_overview, seed=seed + 101)
    pts_miss_plot = _sample_points(pts_miss, max_points_overview // 6, seed=seed + 102)
    pts_hit_ov = np.column_stack(
        [pts_hit_plot[:, 0] / mx_ov_scale_x, pts_hit_plot[:, 1] / mx_ov_scale_y]
    )
    pts_miss_ov = np.column_stack(
        [pts_miss_plot[:, 0] / mx_ov_scale_x, pts_miss_plot[:, 1] / mx_ov_scale_y]
    )

    mask_patch = mask_mm[
        mx_zoom_y0 : mx_zoom_y0 + zoom_size, mx_zoom_x0 : mx_zoom_x0 + zoom_size
    ]
    mask_patch_vis = np.where(mask_patch > 0, 230, 20).astype(np.uint8)
    mask_patch_rgb = np.repeat(mask_patch_vis[..., np.newaxis], 3, axis=2)

    hit_zoom_sel = (
        (pts_hit[:, 0] >= mx_zoom_x0)
        & (pts_hit[:, 0] < mx_zoom_x0 + zoom_size)
        & (pts_hit[:, 1] >= mx_zoom_y0)
        & (pts_hit[:, 1] < mx_zoom_y0 + zoom_size)
    )
    miss_zoom_sel = (
        (pts_miss[:, 0] >= mx_zoom_x0)
        & (pts_miss[:, 0] < mx_zoom_x0 + zoom_size)
        & (pts_miss[:, 1] >= mx_zoom_y0)
        & (pts_miss[:, 1] < mx_zoom_y0 + zoom_size)
    )
    pts_zoom_hit_local = pts_hit[hit_zoom_sel].copy()
    pts_zoom_hit_local[:, 0] -= mx_zoom_x0
    pts_zoom_hit_local[:, 1] -= mx_zoom_y0
    pts_zoom_miss_local = pts_miss[miss_zoom_sel].copy()
    pts_zoom_miss_local[:, 0] -= mx_zoom_x0
    pts_zoom_miss_local[:, 1] -= mx_zoom_y0

    return {
        "conversion_mode_mx": conversion_mode_mx,
        "points_total_mx": len(points_mx),
        "points_in_bounds_mx": int(in_bounds.sum()),
        "points_hit_mx": int(hit_in.sum()),
        "mx_w": mx_w,
        "mx_h": mx_h,
        "mx_zoom_x0": mx_zoom_x0,
        "mx_zoom_y0": mx_zoom_y0,
        "dense_count_mx": dense_count_mx,
        "mx_ov_scale_x": mx_ov_scale_x,
        "mx_ov_scale_y": mx_ov_scale_y,
        "mask_ov_h": mx_ov_h,
        "mask_ov_w": mx_ov_w,
        "mask_ov_rgb": mask_ov_rgb,
        "pts_hit_ov": pts_hit_ov,
        "pts_miss_ov": pts_miss_ov,
        "mask_patch_rgb": mask_patch_rgb,
        "pts_zoom_hit_local": pts_zoom_hit_local,
        "pts_zoom_miss_local": pts_zoom_miss_local,
    }


def _build_plots(
    he_rgb: np.ndarray,
    points_he: np.ndarray,
    img_w: int,
    img_h: int,
    ov_scale_x: float,
    ov_scale_y: float,
    zoom_x0: int,
    zoom_y0: int,
    zoom_size: int,
    he_patch_rgb: np.ndarray,
    out_prefix: Path,
    max_points_overview: int,
    seed: int,
    conversion_mode: str,
    save_individual: bool,
    summary_path: Path | None,
    mask_data: dict | None = None,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    inside = (
        (points_he[:, 0] >= 0)
        & (points_he[:, 0] < img_w)
        & (points_he[:, 1] >= 0)
        & (points_he[:, 1] < img_h)
    )
    pts_in = points_he[inside]
    pts_out = points_he[~inside]

    pts_in_plot = _sample_points(pts_in, max_points_overview, seed=seed)
    pts_out_plot = _sample_points(pts_out, max_points_overview // 10, seed=seed + 1)

    pts_in_ov = np.column_stack(
        [pts_in_plot[:, 0] / ov_scale_x, pts_in_plot[:, 1] / ov_scale_y]
    )
    pts_out_ov = np.column_stack(
        [pts_out_plot[:, 0] / ov_scale_x, pts_out_plot[:, 1] / ov_scale_y]
    )
    h_ov, w_ov = he_rgb.shape[0], he_rgb.shape[1]

    zoom_inside = (
        (pts_in[:, 0] >= zoom_x0)
        & (pts_in[:, 0] < zoom_x0 + zoom_size)
        & (pts_in[:, 1] >= zoom_y0)
        & (pts_in[:, 1] < zoom_y0 + zoom_size)
    )
    pts_zoom = pts_in[zoom_inside]
    pts_zoom_local = pts_zoom.copy()
    pts_zoom_local[:, 0] -= zoom_x0
    pts_zoom_local[:, 1] -= zoom_y0

    he_only_png = out_prefix.with_name(f"{out_prefix.name}.he_overview.png")
    centroids_png = out_prefix.with_name(f"{out_prefix.name}.centroids_he.png")
    overlap_png = out_prefix.with_name(f"{out_prefix.name}.overlap_overview.png")
    zoom_png = out_prefix.with_name(f"{out_prefix.name}.overlap_zoom_{zoom_size}.png")
    mask_overlap_png = out_prefix.with_name(
        f"{out_prefix.name}.mask_centroid_overlap.png"
    )
    mask_zoom_png = out_prefix.with_name(
        f"{out_prefix.name}.mask_centroid_zoom_{zoom_size}.png"
    )

    if summary_path is not None:
        summary_png = Path(summary_path)
    else:
        suffix = "summary_6panel.png" if mask_data is not None else "summary_4panel.png"
        summary_png = out_prefix.with_name(f"{out_prefix.name}.{suffix}")
    summary_png.parent.mkdir(parents=True, exist_ok=True)

    def draw_he_overview(ax: plt.Axes, title: str) -> None:
        ax.imshow(he_rgb)
        ax.add_patch(
            Rectangle(
                (zoom_x0 / ov_scale_x, zoom_y0 / ov_scale_y),
                zoom_size / ov_scale_x,
                zoom_size / ov_scale_y,
                linewidth=1.4,
                edgecolor="#ff0000",
                facecolor="none",
            )
        )
        ax.set_xlim(0, w_ov)
        ax.set_ylim(h_ov, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        ax.set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")

    def draw_mask_overview(ax: plt.Axes, title: str) -> None:
        if mask_data is None:
            return
        ax.imshow(mask_data["mask_ov_rgb"])
        if len(mask_data["pts_miss_ov"]):
            ax.scatter(
                mask_data["pts_miss_ov"][:, 0],
                mask_data["pts_miss_ov"][:, 1],
                s=0.35,
                c="#ff6b6b",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
                label="outside mask",
            )
        if len(mask_data["pts_hit_ov"]):
            ax.scatter(
                mask_data["pts_hit_ov"][:, 0],
                mask_data["pts_hit_ov"][:, 1],
                s=0.35,
                c="#00e5ff",
                alpha=0.5,
                linewidths=0,
                rasterized=True,
                label="inside mask",
            )
        ax.add_patch(
            Rectangle(
                (
                    mask_data["mx_zoom_x0"] / mask_data["mx_ov_scale_x"],
                    mask_data["mx_zoom_y0"] / mask_data["mx_ov_scale_y"],
                ),
                zoom_size / mask_data["mx_ov_scale_x"],
                zoom_size / mask_data["mx_ov_scale_y"],
                linewidth=1.4,
                edgecolor="#ff0000",
                facecolor="none",
            )
        )
        ax.set_xlim(0, mask_data["mask_ov_w"])
        ax.set_ylim(mask_data["mask_ov_h"], 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(f"X (MX px / {mask_data['mx_ov_scale_x']:.2f})")
        ax.set_ylabel(f"Y (MX px / {mask_data['mx_ov_scale_y']:.2f})")
        ax.legend(loc="upper right", markerscale=8, frameon=True)

    def draw_mask_zoom(ax: plt.Axes, title: str) -> None:
        if mask_data is None:
            return
        ax.imshow(mask_data["mask_patch_rgb"])
        if len(mask_data["pts_zoom_miss_local"]):
            ax.scatter(
                mask_data["pts_zoom_miss_local"][:, 0],
                mask_data["pts_zoom_miss_local"][:, 1],
                s=8.0,
                c="#ff6b6b",
                alpha=0.6,
                linewidths=0,
                rasterized=True,
            )
        if len(mask_data["pts_zoom_hit_local"]):
            ax.scatter(
                mask_data["pts_zoom_hit_local"][:, 0],
                mask_data["pts_zoom_hit_local"][:, 1],
                s=8.0,
                c="#00e5ff",
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
        ax.set_xlim(0, zoom_size)
        ax.set_ylim(zoom_size, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("X (MX px)")
        ax.set_ylabel("Y (MX px)")

    if save_individual:
        fig1, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
        draw_he_overview(ax1, "1) HE image (overview)")
        fig1.savefig(he_only_png, dpi=220)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 8), constrained_layout=True)
        if len(pts_out_ov):
            ax2.scatter(
                pts_out_ov[:, 0],
                pts_out_ov[:, 1],
                s=0.3,
                c="#bdbdbd",
                alpha=0.25,
                linewidths=0,
                rasterized=True,
                label="outside HE bounds",
            )
        ax2.scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.3,
            c="#e15759",
            alpha=0.45,
            linewidths=0,
            rasterized=True,
            label="mapped centroids (HE coords)",
        )
        ax2.set_xlim(0, w_ov)
        ax2.set_ylim(h_ov, 0)
        ax2.set_aspect("equal", adjustable="box")
        ax2.grid(True, alpha=0.15, linewidth=0.5)
        ax2.set_title("2) Transformed centroids mapped to HE coordinates")
        ax2.set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        ax2.set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")
        ax2.legend(loc="upper right", markerscale=8, frameon=True)
        fig2.savefig(centroids_png, dpi=220)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax3.imshow(he_rgb)
        ax3.scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.25,
            c="#00ffff",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax3.add_patch(
            Rectangle(
                (zoom_x0 / ov_scale_x, zoom_y0 / ov_scale_y),
                zoom_size / ov_scale_x,
                zoom_size / ov_scale_y,
                linewidth=1.4,
                edgecolor="#ff0000",
                facecolor="none",
            )
        )
        ax3.set_xlim(0, w_ov)
        ax3.set_ylim(h_ov, 0)
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_title("3) HE + mapped centroids overlap (overview)")
        ax3.set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        ax3.set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")
        fig3.savefig(overlap_png, dpi=220)
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax4.imshow(he_patch_rgb)
        if len(pts_zoom_local):
            ax4.scatter(
                pts_zoom_local[:, 0],
                pts_zoom_local[:, 1],
                s=6.0,
                c="#39ff14",
                alpha=0.75,
                linewidths=0,
                rasterized=True,
            )
        ax4.set_xlim(0, zoom_size)
        ax4.set_ylim(zoom_size, 0)
        ax4.set_aspect("equal", adjustable="box")
        ax4.set_title(
            f"4) Overlap zoom ({zoom_size} x {zoom_size})\n"
            f"HE window x=[{zoom_x0},{zoom_x0 + zoom_size}), "
            f"y=[{zoom_y0},{zoom_y0 + zoom_size})"
        )
        ax4.set_xlabel("X (HE px)")
        ax4.set_ylabel("Y (HE px)")
        fig4.savefig(zoom_png, dpi=220)
        plt.close(fig4)

        if mask_data is not None:
            fig5, ax5 = plt.subplots(figsize=(8, 8), constrained_layout=True)
            draw_mask_overview(ax5, "5) MX mask + centroids overlap (overview)")
            fig5.savefig(mask_overlap_png, dpi=220)
            plt.close(fig5)

            fig6, ax6 = plt.subplots(figsize=(8, 8), constrained_layout=True)
            draw_mask_zoom(
                ax6, f"6) MX mask + centroids zoom ({zoom_size} x {zoom_size})"
            )
            fig6.savefig(mask_zoom_png, dpi=220)
            plt.close(fig6)

    if mask_data is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
        axes = np.asarray(axes)
        draw_he_overview(axes[0, 0], "1) HE image (overview)")

        if len(pts_out_ov):
            axes[0, 1].scatter(
                pts_out_ov[:, 0],
                pts_out_ov[:, 1],
                s=0.25,
                c="#bdbdbd",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
        axes[0, 1].scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.25,
            c="#e15759",
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        axes[0, 1].set_xlim(0, w_ov)
        axes[0, 1].set_ylim(h_ov, 0)
        axes[0, 1].set_aspect("equal", adjustable="box")
        axes[0, 1].grid(True, alpha=0.15, linewidth=0.5)
        axes[0, 1].set_title("2) Transformed centroids mapped to HE")
        axes[0, 1].set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        axes[0, 1].set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")

        axes[1, 0].imshow(he_rgb)
        axes[1, 0].scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.2,
            c="#00ffff",
            alpha=0.3,
            linewidths=0,
            rasterized=True,
        )
        axes[1, 0].add_patch(
            Rectangle(
                (zoom_x0 / ov_scale_x, zoom_y0 / ov_scale_y),
                zoom_size / ov_scale_x,
                zoom_size / ov_scale_y,
                linewidth=1.4,
                edgecolor="#ff0000",
                facecolor="none",
            )
        )
        axes[1, 0].set_xlim(0, w_ov)
        axes[1, 0].set_ylim(h_ov, 0)
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].set_title("3) HE + centroid overlap")
        axes[1, 0].set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        axes[1, 0].set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")

        axes[1, 1].imshow(he_patch_rgb)
        if len(pts_zoom_local):
            axes[1, 1].scatter(
                pts_zoom_local[:, 0],
                pts_zoom_local[:, 1],
                s=5.0,
                c="#39ff14",
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
        axes[1, 1].set_xlim(0, zoom_size)
        axes[1, 1].set_ylim(zoom_size, 0)
        axes[1, 1].set_aspect("equal", adjustable="box")
        axes[1, 1].set_title(f"4) Dense-region zoom ({zoom_size} x {zoom_size})")
        axes[1, 1].set_xlabel("X (HE px)")
        axes[1, 1].set_ylabel("Y (HE px)")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        axes = np.asarray(axes)
        draw_he_overview(axes[0, 0], "1) HE image (overview)")

        if len(pts_out_ov):
            axes[0, 1].scatter(
                pts_out_ov[:, 0],
                pts_out_ov[:, 1],
                s=0.25,
                c="#bdbdbd",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
        axes[0, 1].scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.25,
            c="#e15759",
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        axes[0, 1].set_xlim(0, w_ov)
        axes[0, 1].set_ylim(h_ov, 0)
        axes[0, 1].set_aspect("equal", adjustable="box")
        axes[0, 1].grid(True, alpha=0.15, linewidth=0.5)
        axes[0, 1].set_title("2) Transformed centroids mapped to HE")
        axes[0, 1].set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        axes[0, 1].set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")

        axes[0, 2].imshow(he_rgb)
        axes[0, 2].scatter(
            pts_in_ov[:, 0],
            pts_in_ov[:, 1],
            s=0.2,
            c="#00ffff",
            alpha=0.3,
            linewidths=0,
            rasterized=True,
        )
        axes[0, 2].add_patch(
            Rectangle(
                (zoom_x0 / ov_scale_x, zoom_y0 / ov_scale_y),
                zoom_size / ov_scale_x,
                zoom_size / ov_scale_y,
                linewidth=1.4,
                edgecolor="#ff0000",
                facecolor="none",
            )
        )
        axes[0, 2].set_xlim(0, w_ov)
        axes[0, 2].set_ylim(h_ov, 0)
        axes[0, 2].set_aspect("equal", adjustable="box")
        axes[0, 2].set_title("3) HE + centroid overlap")
        axes[0, 2].set_xlabel(f"X (HE px / {ov_scale_x:.2f})")
        axes[0, 2].set_ylabel(f"Y (HE px / {ov_scale_y:.2f})")

        axes[1, 0].imshow(he_patch_rgb)
        if len(pts_zoom_local):
            axes[1, 0].scatter(
                pts_zoom_local[:, 0],
                pts_zoom_local[:, 1],
                s=5.0,
                c="#39ff14",
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
        axes[1, 0].set_xlim(0, zoom_size)
        axes[1, 0].set_ylim(zoom_size, 0)
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].set_title(f"4) Dense-region zoom ({zoom_size} x {zoom_size})")
        axes[1, 0].set_xlabel("X (HE px)")
        axes[1, 0].set_ylabel("Y (HE px)")

        draw_mask_overview(axes[1, 1], "5) MX mask + centroids overlap")
        draw_mask_zoom(
            axes[1, 2], f"6) MX mask + centroids zoom ({zoom_size} x {zoom_size})"
        )

    title = "HE / transformed centroid visualization"
    subtitle = f"HE conversion: {conversion_mode}"
    if mask_data is not None:
        subtitle += f" | MX conversion: {mask_data['conversion_mode_mx']}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    fig.savefig(summary_png, dpi=220)
    plt.close(fig)

    if save_individual:
        print(f"Saved: {he_only_png}")
        print(f"Saved: {centroids_png}")
        print(f"Saved: {overlap_png}")
        print(f"Saved: {zoom_png}")
        if mask_data is not None:
            print(f"Saved: {mask_overlap_png}")
            print(f"Saved: {mask_zoom_png}")
    print(f"Saved: {summary_png}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize HE image + centroids, with optional MX mask overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--he-image", required=True, help="Path to HE OME-TIFF.")
    p.add_argument("--csv", required=True, help="Path to transformed centroid CSV.")
    p.add_argument(
        "--mask-image",
        default=None,
        help="Optional MX mask TIFF (instance/binary) for 5th and 6th visualizations.",
    )
    p.add_argument("--index", default=None, help="Optional index.json path for scale.")
    p.add_argument(
        "--out-prefix", default="data/he_centroid_viz", help="Output file prefix."
    )
    p.add_argument("--downsample", type=int, default=64, help="HE overview downsample.")
    p.add_argument(
        "--zoom-size",
        type=int,
        default=512,
        help="Dense-region zoom window size in HE pixels.",
    )
    p.add_argument("--he-mpp", type=float, default=0.325, help="HE mpp (um/px).")
    p.add_argument("--mx-mpp", type=float, default=0.65, help="MX mpp (um/px).")
    p.add_argument(
        "--scale-he-to-mx",
        type=float,
        default=None,
        help="HE->MX scale (if omitted, infer from index.json or mpp ratio).",
    )
    p.add_argument(
        "--max-points-overview",
        type=int,
        default=300000,
        help="Max points to draw in overview scatter panels.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Random seed for point sampling."
    )
    p.add_argument(
        "--mx-downsample",
        type=int,
        default=32,
        help="Downsample factor for MX mask overview.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate only the summary figure (4-panel without mask, 6-panel with mask).",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        help="Optional explicit output path for the summary PNG.",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    he_path = Path(args.he_image)
    csv_path = Path(args.csv)
    out_prefix = Path(args.out_prefix)
    mask_path = Path(args.mask_image) if args.mask_image else None

    payload = None
    if args.index:
        with open(args.index, encoding="utf-8") as f:
            payload = json.load(f)

    scale_he_to_mx = _infer_scale_he_to_mx(
        index_payload=payload,
        he_mpp=float(args.he_mpp),
        mx_mpp=float(args.mx_mpp),
        scale_he_to_mx=args.scale_he_to_mx,
    )

    he_rgb, img_w, img_h, ov_scale_x, ov_scale_y, level_idx = _read_he_overview(
        he_path,
        downsample=int(args.downsample),
    )
    points_he, conversion_mode = _load_points_he(
        csv_path=csv_path,
        scale_he_to_mx=scale_he_to_mx,
        mx_mpp=float(args.mx_mpp),
    )

    zoom_x0, zoom_y0, dense_count = _find_dense_window(
        points_he=points_he,
        img_w=img_w,
        img_h=img_h,
        window_size=int(args.zoom_size),
    )
    he_patch_rgb = _read_he_patch(
        he_path=he_path,
        x0=zoom_x0,
        y0=zoom_y0,
        size=int(args.zoom_size),
    )

    inside = (
        (points_he[:, 0] >= 0)
        & (points_he[:, 0] < img_w)
        & (points_he[:, 1] >= 0)
        & (points_he[:, 1] < img_h)
    )
    print(f"HE dimensions: {img_w} x {img_h} px")
    print(
        f"Overview level: {level_idx} "
        f"(effective HE->overview scale x={ov_scale_x:.2f}, y={ov_scale_y:.2f})"
    )
    print(f"Total centroids: {len(points_he):,}")
    print(
        f"Centroids inside HE bounds: {int(inside.sum()):,} ({inside.mean() * 100:.2f}%)"
    )
    print(f"Coordinate conversion: {conversion_mode}")
    print(f"Scale HE->MX used: {scale_he_to_mx:.6f}")
    print(
        f"Densest {args.zoom_size}x{args.zoom_size} window: "
        f"x0={zoom_x0}, y0={zoom_y0}, cells={dense_count:,}"
    )

    mask_data = None
    if mask_path is not None:
        mask_data = _prepare_mask_visualization(
            mask_path=mask_path,
            csv_path=csv_path,
            zoom_size=int(args.zoom_size),
            mx_mpp=float(args.mx_mpp),
            scale_he_to_mx=scale_he_to_mx,
            max_points_overview=int(args.max_points_overview),
            mx_downsample=int(args.mx_downsample),
            seed=int(args.seed),
        )
        hit_frac = (
            100.0
            * mask_data["points_hit_mx"]
            / max(1, mask_data["points_in_bounds_mx"])
        )
        print(f"MX mask dimensions: {mask_data['mx_w']} x {mask_data['mx_h']} px")
        print(
            f"MX centroids in bounds: {mask_data['points_in_bounds_mx']:,} / "
            f"{mask_data['points_total_mx']:,}"
        )
        print(
            f"MX centroids hitting mask: {mask_data['points_hit_mx']:,} "
            f"({hit_frac:.2f}%)"
        )
        print(
            f"MX densest {args.zoom_size}x{args.zoom_size} window: "
            f"x0={mask_data['mx_zoom_x0']}, y0={mask_data['mx_zoom_y0']}, "
            f"cells={mask_data['dense_count_mx']:,}"
        )

    _build_plots(
        he_rgb=he_rgb,
        points_he=points_he,
        img_w=img_w,
        img_h=img_h,
        ov_scale_x=ov_scale_x,
        ov_scale_y=ov_scale_y,
        zoom_x0=zoom_x0,
        zoom_y0=zoom_y0,
        zoom_size=int(args.zoom_size),
        he_patch_rgb=he_patch_rgb,
        out_prefix=out_prefix,
        max_points_overview=int(args.max_points_overview),
        seed=int(args.seed),
        conversion_mode=conversion_mode,
        save_individual=not bool(args.summary_only),
        summary_path=Path(args.summary_path) if args.summary_path else None,
        mask_data=mask_data,
    )


if __name__ == "__main__":
    main(_parse_args())
