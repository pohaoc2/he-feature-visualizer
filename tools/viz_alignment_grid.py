#!/usr/bin/env python3
"""
viz_alignment_grid.py — 3×3 alignment debug grid.

All panels use H&E coordinate space.

Columns:
  1. H&E side        — overview | CellViT contours | H&E + CellViT
  2. MX warped→H&E   — MX DNA warped via m_full | CSV→H&E | warped MX + CSV
  3. Cross-modal HE  — transparent H&E+warped-MX | CellViT contours + CSV | patch zoom

Transform note
--------------
m_full (2×3) maps H&E crop px → MX crop px.
To warp MX into H&E overview space we use cv2.warpAffine with WARP_INVERSE_MAP
so that for each output H&E overview pixel the matrix samples the correct MX
overview pixel.  At overview scale (downsample=ds) only the translation column
is divided by ds; the linear part is unchanged.

Usage
-----
python -m tools.viz_alignment_grid \\
    --processed       processed_crop/ \\
    --he-image        data/WD-76845-096-crop.ome.tif \\
    --multiplex-image data/WD-76845-097-crop.ome.tif \\
    --csv             data/WD-76845-097.csv \\
    --csv-mpp         0.65 \\
    --mx-crop-origin  6360 432 \\
    --downsample      8 \\
    --out-dir         debug/
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
import pandas as pd
import tifffile
import zarr

from stages.refine_registration import resolve_mx_crop_origin
from utils.normalize import percentile_to_uint8
from utils.ome import get_image_dims

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _open_zarr(tif: tifffile.TiffFile):
    raw = zarr.open(tif.series[0].aszarr(), mode="r")
    return raw if isinstance(raw, zarr.Array) else raw["0"]


def _load_he_overview(he_path: pathlib.Path, ds: int) -> np.ndarray:
    """(H, W, 3) uint8 BGR."""
    with tifffile.TiffFile(str(he_path)) as tif:
        img_w, img_h, axes = get_image_dims(tif)
        store = _open_zarr(tif)
        ax = axes.upper()
        sl: list = []
        for a in ax:
            if a == "C":
                sl.append(slice(None))
            elif a == "Y":
                sl.append(slice(0, (img_h // ds) * ds, ds))
            elif a == "X":
                sl.append(slice(0, (img_w // ds) * ds, ds))
            else:
                sl.append(0)
        data = np.asarray(store[tuple(sl)])
        if "C" in ax and ax.index("C") != 0:
            data = np.moveaxis(data, ax.index("C"), 0)
    if data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
    elif data.shape[0] >= 3:
        rgb = data[:3].transpose(1, 2, 0)
    else:
        g = data[0]
        rgb = np.stack([g, g, g], axis=-1)
    if rgb.dtype != np.uint8:
        rgb = percentile_to_uint8(rgb.astype(np.float32))
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _load_mx_dna_overview(mx_path: pathlib.Path, ds: int) -> np.ndarray:
    """(H, W) uint8 grayscale — channel 0 only, at NATIVE MX resolution."""
    with tifffile.TiffFile(str(mx_path)) as tif:
        img_w, img_h, axes = get_image_dims(tif)
        store = _open_zarr(tif)
        ax = axes.upper()
        sl: list = []
        for a in ax:
            if a == "C":
                sl.append(0)
            elif a == "Y":
                sl.append(slice(0, (img_h // ds) * ds, ds))
            elif a == "X":
                sl.append(slice(0, (img_w // ds) * ds, ds))
            else:
                sl.append(0)
        ch0 = np.asarray(store[tuple(sl)])
    if ch0.ndim > 2:
        ch0 = ch0.squeeze()
    return percentile_to_uint8(ch0.astype(np.float32))


def _load_seg_crop(
    seg_path: pathlib.Path,
    crop_origin: tuple[float, float] | None,
    mx_w: int,
    mx_h: int,
    level: int = 1,
) -> np.ndarray:
    """(H, W) uint8 binary mask cropped from seg.tif pyramid level.

    Uses a pyramid level instead of stride-based downsampling so that
    thin cell-instance boundaries are not skipped by the step.
    The crop coordinates are scaled to the requested level.
    """
    ox0 = int(crop_origin[0]) if crop_origin else 0
    oy0 = int(crop_origin[1]) if crop_origin else 0
    with tifffile.TiffFile(str(seg_path)) as tif:
        n_levels = len(tif.series[0].levels)
        lvl = min(level, n_levels - 1)
        lvl_shape = tif.series[0].levels[lvl].shape  # (H, W) or (C, H, W)

        # Scale factor from level 0 to requested level
        full_h = tif.series[0].levels[0].shape[-2]
        full_w = tif.series[0].levels[0].shape[-1]
        scale = lvl_shape[-2] / full_h  # e.g. 0.5 for level 1

        # Crop bounds scaled to this level
        ox = int(ox0 * scale)
        oy = int(oy0 * scale)
        cw = int(mx_w * scale)
        ch = int(mx_h * scale)

        raw = zarr.open(tif.series[0].aszarr(), mode="r")
        store_lvl = raw[str(lvl)]
        ax = tif.series[0].axes.upper()
        sl: list = []
        for a in ax:
            if a == "Y":
                sl.append(slice(oy, oy + ch))
            elif a == "X":
                sl.append(slice(ox, ox + cw))
            elif a == "C":
                sl.append(0)
            else:
                sl.append(0)
        seg = np.asarray(store_lvl[tuple(sl)])

    if seg.ndim > 2:
        seg = seg.squeeze()
    # Binary: 0 → 0, nonzero → 255
    return (seg > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# CellViT loaders
# ---------------------------------------------------------------------------


def load_cellvit_geometry(
    cellvit_dir: pathlib.Path,
    patches: list[dict],
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """CellViT centroids + contours in H&E full-res px.

    Returns
    -------
    pts_he : (N, 2) float64
        Global CellViT centroids in H&E space.
    contours_he : list[(K_i, 2) float64 | None]
        Contour vertices per cell in H&E space. ``None`` means contour missing.
    """
    pts: list[tuple[float, float]] = []
    contours: list[np.ndarray | None] = []
    for p in patches:
        x0, y0 = int(p["x0"]), int(p["y0"])
        jp = cellvit_dir / f"{x0}_{y0}.json"
        if not jp.exists():
            continue
        for c in json.load(jp.open()).get("cells", []):
            centroid = c.get("centroid")
            if centroid is None or len(centroid) < 2:
                continue

            lx, ly = float(centroid[0]), float(centroid[1])
            pts.append((x0 + lx, y0 + ly))

            contour = c.get("contour")
            contour_he: np.ndarray | None = None
            if isinstance(contour, list) and len(contour) >= 3:
                arr = np.asarray(contour, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    arr = arr[:, :2]
                    arr[:, 0] += float(x0)
                    arr[:, 1] += float(y0)
                    if len(arr) >= 3:
                        contour_he = arr
            contours.append(contour_he)
    if not pts:
        return np.empty((0, 2), dtype=np.float64), []
    return np.asarray(pts, dtype=np.float64), contours


def load_csv_pts_mx(
    csv_path: pathlib.Path,
    csv_mpp: float,
    crop_origin: tuple[float, float] | None,
) -> np.ndarray:
    """(N, 2) CSV centroids in MX crop px."""
    cols = set(pd.read_csv(csv_path, nrows=0).columns.tolist())
    if {"Xt_mx_px", "Yt_mx_px"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt_mx_px", "Yt_mx_px"])
        pts = df[["Xt_mx_px", "Yt_mx_px"]].to_numpy(dtype=np.float64)
    elif {"Xt", "Yt"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["Xt", "Yt"])
        pts = df[["Xt", "Yt"]].to_numpy(dtype=np.float64) / csv_mpp
    else:
        raise ValueError(
            "CSV must include Xt/Yt (um) or Xt_mx_px/Yt_mx_px columns."
        )
    if crop_origin is not None:
        pts -= np.array(crop_origin)
    return pts


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def _inv_m_full(m_full: np.ndarray) -> np.ndarray:
    m3 = np.eye(3, dtype=np.float64)
    m3[:2] = m_full
    return np.linalg.inv(m3)


def _to_mat3(m_2x3: np.ndarray) -> np.ndarray:
    m3 = np.eye(3, dtype=np.float64)
    m3[:2, :] = m_2x3.astype(np.float64)
    return m3


def _compose_affine_with_icp(m_stage1: np.ndarray, m_icp: np.ndarray) -> np.ndarray:
    """Compose Stage-1 H&E->MX affine with Stage-2.5 ICP H&E correction."""
    return (_to_mat3(m_stage1) @ _to_mat3(m_icp))[:2, :]


def mx_to_he(pts_mx: np.ndarray, m_full_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts_mx), 1))
    return (m_full_inv @ np.hstack([pts_mx, ones]).T).T[:, :2]


def warp_mx_to_he(
    mx_gray: np.ndarray,
    m_full: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
) -> np.ndarray:
    """Warp native MX overview (H_mx, W_mx) → H&E overview (he_h, he_w).

    m_full (2×3) maps H&E full-res px → MX full-res px.
    At overview scale: same linear part, translation / ds.
    WARP_INVERSE_MAP: for each output (H&E) pixel, M gives the source (MX) pixel.
    mx_gray must be at native MX overview resolution (NOT pre-resized to H&E).
    """
    m_ov = m_full.astype(np.float64).copy()
    m_ov[:, 2] /= ds  # scale translation to overview px; linear part unchanged
    return cv2.warpAffine(
        mx_gray,
        m_ov,
        (he_w, he_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _dots(
    img: np.ndarray,
    pts_ov: np.ndarray,
    color: tuple,
    radius: int = 2,
    thickness: int = -1,
    outline_color: tuple[int, int, int] | None = None,
    outline_thickness: int = 1,
    max_pts: int = 20_000,
) -> tuple[np.ndarray, int]:
    out = img.copy()
    if len(pts_ov) == 0:
        return out, 0
    h, w = out.shape[:2]
    in_b = (
        (pts_ov[:, 0] >= 0)
        & (pts_ov[:, 0] < w)
        & (pts_ov[:, 1] >= 0)
        & (pts_ov[:, 1] < h)
    )
    pts_in = pts_ov[in_b]
    if len(pts_in) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts_in), max_pts, replace=False)
        pts_in = pts_in[idx]
    for x, y in pts_in.astype(int):
        if outline_color is not None:
            if thickness <= 0:
                cv2.circle(
                    out,
                    (x, y),
                    radius + max(1, outline_thickness),
                    outline_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    out,
                    (x, y),
                    radius + max(1, outline_thickness),
                    outline_color,
                    thickness + max(1, outline_thickness),
                    lineType=cv2.LINE_AA,
                )
        cv2.circle(out, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
    return out, int(in_b.sum())


def _draw_contours(
    img: np.ndarray,
    contours_ov: list[np.ndarray | None],
    color: tuple[int, int, int],
    thickness: int = 2,
    max_contours: int = 20_000,
) -> tuple[np.ndarray, int]:
    """Draw CellViT contours (already in overview pixel space)."""
    out = img.copy()
    if not contours_ov:
        return out, 0
    h, w = out.shape[:2]
    n_drawn = 0
    for contour in contours_ov:
        if contour is None or len(contour) < 3:
            continue
        if n_drawn >= max_contours:
            break
        c = np.asarray(contour, dtype=np.float64)
        x_min, y_min = float(c[:, 0].min()), float(c[:, 1].min())
        x_max, y_max = float(c[:, 0].max()), float(c[:, 1].max())
        if x_max < 0 or y_max < 0 or x_min >= w or y_min >= h:
            continue
        pts = np.rint(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            out,
            [pts],
            isClosed=True,
            color=color,
            thickness=max(1, int(thickness)),
            lineType=cv2.LINE_AA,
        )
        n_drawn += 1
    return out, n_drawn


def _label(
    img: np.ndarray,
    text: str,
    scale: float = 1.05,
    bar_frac: float = 0.09,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(54, int(bar_frac * h))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    y = min(h - 8, int(bar_h * 0.72))
    cv2.putText(
        out,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return out


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h))


def _append_legend_bar(grid: np.ndarray, blend_alpha: float) -> np.ndarray:
    """Append a global legend bar to the bottom of the composite grid."""
    h, w = grid.shape[:2]
    bar_h = max(120, int(h * 0.10))
    legend = np.full((bar_h, w, 3), 26, dtype=np.uint8)
    cv2.rectangle(legend, (0, 0), (w - 1, bar_h - 1), (80, 80, 80), 2)

    title = "Legend"
    cv2.putText(
        legend,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    def draw_item(
        x: int,
        y: int,
        label: str,
        marker: str,
        color: tuple[int, int, int],
        outline: tuple[int, int, int] | None = None,
    ) -> int:
        if marker == "dot":
            if outline is not None:
                cv2.circle(legend, (x, y), 10, outline, -1, lineType=cv2.LINE_AA)
            cv2.circle(legend, (x, y), 7, color, -1, lineType=cv2.LINE_AA)
        elif marker == "contour":
            poly = np.array(
                [[x - 10, y + 6], [x - 2, y - 9], [x + 8, y - 5], [x + 10, y + 7]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            if outline is not None:
                cv2.polylines(legend, [poly], True, outline, 4, lineType=cv2.LINE_AA)
            cv2.polylines(legend, [poly], True, color, 2, lineType=cv2.LINE_AA)
        elif marker == "ring":
            cv2.circle(legend, (x, y), 10, (0, 0, 0), 5, lineType=cv2.LINE_AA)
            cv2.circle(legend, (x, y), 10, color, 2, lineType=cv2.LINE_AA)
        elif marker == "box":
            cv2.rectangle(legend, (x - 10, y - 8), (x + 10, y + 8), color, 2)
        cv2.putText(
            legend,
            label,
            (x + 18, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        txt_w, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0]
        return x + 18 + txt_w + 42

    y1 = 66
    x = 24
    x = draw_item(
        x,
        y1,
        "CellViT contour (C1/C3)",
        "contour",
        (0, 255, 255),
        (0, 0, 0),
    )
    x = draw_item(x, y1, "CSV centroid", "ring", (255, 255, 0))
    _ = draw_item(x, y1, "Zoom patch ROI", "box", (0, 0, 255))

    line2 = (
        f"C3R1 uses transparent overlay: H&E(RGB) + warped MX(HOT), "
        f"alpha={blend_alpha:.2f}"
    )
    cv2.putText(
        legend,
        line2,
        (24, min(bar_h - 14, 104)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([grid, legend])


def _draw_c3_overlay_legend(panel: np.ndarray, blend_alpha: float) -> np.ndarray:
    """Draw a local legend box directly on c3r1."""
    out = panel.copy()
    h, w = out.shape[:2]
    box_w = min(w - 20, max(380, int(w * 0.33)))
    box_h = min(h - 20, max(116, int(h * 0.14)))
    x0 = 12
    y0 = h - box_h - 12
    x1 = x0 + box_w
    y1 = y0 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (230, 230, 230), 2)

    ty = y0 + 28
    cv2.putText(
        out,
        "c3r1 legend",
        (x0 + 12, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # H&E indicator
    cv2.rectangle(out, (x0 + 14, ty + 12), (x0 + 36, ty + 28), (210, 210, 210), -1)
    cv2.putText(
        out,
        "H&E image",
        (x0 + 48, ty + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    # MX(HOT) indicator
    cv2.rectangle(out, (x0 + 200, ty + 12), (x0 + 222, ty + 28), (0, 140, 255), -1)
    cv2.putText(
        out,
        "MX warped (HOT)",
        (x0 + 234, ty + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"blend = (1-a)*H&E + a*MX,  a={blend_alpha:.2f}",
        (x0 + 14, y1 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Zoom patch helper
# ---------------------------------------------------------------------------


def _make_zoom_panel(
    patch_id: str | None,
    he_bgr: np.ndarray,
    cv_pts_he_ov: np.ndarray,
    cv_contours_he_ov: list[np.ndarray | None],
    csv_pts_he_ov: np.ndarray,
    ds: int,
    title_scale: float,
    target_h: int,
    target_w: int,
    patch_size: int,
    cellvit_contour_lw: int,
    csv_radius: int,
    csv_thickness: int,
) -> np.ndarray:
    """Zoomed H&E region around selected patch with CellViT contour + CSV overlays."""
    if not patch_id:
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return _label(blank, "c3r3) zoom patch unavailable", scale=title_scale)

    x0_he, y0_he = map(int, patch_id.split("_"))
    x0_ov = int(np.floor(x0_he / ds))
    y0_ov = int(np.floor(y0_he / ds))
    p_ov = max(2, int(np.ceil(patch_size / ds)))
    x1_ov = x0_ov + p_ov
    y1_ov = y0_ov + p_ov

    h, w = he_bgr.shape[:2]
    cx = (x0_ov + x1_ov) // 2
    cy = (y0_ov + y1_ov) // 2
    half_w = max(150, p_ov * 2)
    half_h = max(150, p_ov * 2)

    zx0 = max(0, cx - half_w)
    zy0 = max(0, cy - half_h)
    zx1 = min(w, cx + half_w)
    zy1 = min(h, cy + half_h)
    if zx1 <= zx0 or zy1 <= zy0:
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return _label(blank, f"c3r3) invalid zoom for patch {patch_id}", scale=title_scale)

    zoom = he_bgr[zy0:zy1, zx0:zx1].copy()
    patch_tl = (max(0, x0_ov - zx0), max(0, y0_ov - zy0))
    patch_br = (min(zx1 - zx0 - 1, x1_ov - zx0), min(zy1 - zy0 - 1, y1_ov - zy0))
    cv2.rectangle(zoom, patch_tl, patch_br, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    cv_local = cv_pts_he_ov - np.array([zx0, zy0], dtype=np.float64)
    cv_contours_local: list[np.ndarray | None] = []
    for contour in cv_contours_he_ov:
        if contour is None:
            cv_contours_local.append(None)
            continue
        cv_contours_local.append(contour - np.array([zx0, zy0], dtype=np.float64))
    csv_local = csv_pts_he_ov - np.array([zx0, zy0], dtype=np.float64)
    zoom, _ = _draw_contours(
        zoom,
        cv_contours_local,
        color=(0, 255, 0),
        thickness=max(1, cellvit_contour_lw),
    )
    zoom, _ = _dots(
        zoom,
        csv_local,
        color=(255, 255, 0),
        radius=max(3, csv_radius),
        thickness=max(1, csv_thickness),
        outline_color=(0, 0, 0),
        outline_thickness=max(1, csv_thickness),
    )

    cv_in_patch = (
        (cv_pts_he_ov[:, 0] >= x0_ov)
        & (cv_pts_he_ov[:, 0] < x1_ov)
        & (cv_pts_he_ov[:, 1] >= y0_ov)
        & (cv_pts_he_ov[:, 1] < y1_ov)
    )
    csv_in_patch = (
        (csv_pts_he_ov[:, 0] >= x0_ov)
        & (csv_pts_he_ov[:, 0] < x1_ov)
        & (csv_pts_he_ov[:, 1] >= y0_ov)
        & (csv_pts_he_ov[:, 1] < y1_ov)
    )

    zoom = _resize_to(zoom, target_h, target_w)
    return _label(
        zoom,
        f"c3r3) zoom patch {patch_id} | CellViT cells={int(cv_in_patch.sum())} CSV={int(csv_in_patch.sum())}",
        scale=title_scale,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = args.downsample
    cli_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None

    processed_dir = pathlib.Path(args.processed)
    index_path = pathlib.Path(args.index_path) if args.index_path else (processed_dir / "index.json")
    with index_path.open() as fh:
        index = json.load(fh)
    crop_origin = resolve_mx_crop_origin(index, cli_origin=cli_origin)
    patches = index.get("patches", [])
    m_full = np.array(index["warp_matrix"], dtype=np.float64)
    if args.icp_index:
        with pathlib.Path(args.icp_index).open() as fh:
            icp_payload = json.load(fh)
        if "icp_matrix" not in icp_payload:
            raise ValueError(f"{args.icp_index} has no icp_matrix.")
        m_icp = np.array(icp_payload["icp_matrix"], dtype=np.float64)
        m_full = _compose_affine_with_icp(m_full, m_icp)
        print(f"Composed Stage-2.5 affine using icp_matrix from: {args.icp_index}")
    m_inv = _inv_m_full(m_full)
    patch_size = int(index.get("patch_size", 512))
    title_scale = float(args.title_scale)
    cellvit_contour_lw = int(args.cellvit_radius)
    csv_radius = int(args.csv_radius)
    csv_thickness = int(args.csv_thickness)

    print("m_full (H&E crop px → MX crop px):")
    print(f"  [[{m_full[0,0]:.4f}, {m_full[0,1]:.4f}, {m_full[0,2]:.1f}],")
    print(f"   [{m_full[1,0]:.4f}, {m_full[1,1]:.4f}, {m_full[1,2]:.1f}]]")
    if cli_origin is not None:
        print(f"Using mx_crop_origin from CLI: ({crop_origin[0]:.1f}, {crop_origin[1]:.1f})")
    elif crop_origin is not None:
        print(
            f"Using mx_crop_origin from index.json: ({crop_origin[0]:.1f}, {crop_origin[1]:.1f})"
        )
    else:
        print("WARNING: mx_crop_origin unavailable; CSV points will be treated as crop-local.")

    # -----------------------------------------------------------------------
    # Load overviews
    # -----------------------------------------------------------------------
    print("Loading H&E overview ...")
    he_bgr = _load_he_overview(pathlib.Path(args.he_image), ds)
    ov_h, ov_w = he_bgr.shape[:2]
    print(f"  H&E overview: {ov_w}x{ov_h} px")

    print("Loading MX DNA overview (native resolution) ...")
    mx_gray_native = _load_mx_dna_overview(pathlib.Path(args.multiplex_image), ds)
    mx_h_nat, mx_w_nat = mx_gray_native.shape
    print(f"  MX DNA native: {mx_w_nat}x{mx_h_nat} px")

    # Warp MX (native resolution) → H&E overview space
    # IMPORTANT: pass native MX (not pre-resized) so m_ov coordinates are correct
    mx_gray_warped = warp_mx_to_he(mx_gray_native, m_full, ds, ov_h, ov_w)
    mx_bgr_warped = cv2.applyColorMap(mx_gray_warped, cv2.COLORMAP_HOT)
    print(f"  MX warped→H&E: {mx_bgr_warped.shape[1]}x{mx_bgr_warped.shape[0]} px")

    # -----------------------------------------------------------------------
    # Load CellViT geometry + CSV centroids
    # -----------------------------------------------------------------------
    print("Loading CellViT cells (centroids + contours) ...")
    cv_pts_he, cv_contours_he = load_cellvit_geometry(processed_dir / "cellvit", patches)
    cv_pts_ov = cv_pts_he / ds
    cv_contours_ov: list[np.ndarray | None] = []
    for contour in cv_contours_he:
        if contour is None:
            cv_contours_ov.append(None)
        else:
            cv_contours_ov.append(contour / ds)
    n_cv_contours = sum(1 for c in cv_contours_ov if c is not None)
    print(
        f"  CellViT: {len(cv_pts_he):,} cells, {n_cv_contours:,} valid contours."
    )

    print("Loading CSV centroids ...")
    csv_pts_mx_all = load_csv_pts_mx(pathlib.Path(args.csv), args.csv_mpp, crop_origin)
    # Filter to MX crop bounds
    mx_full_w = mx_w_nat * ds
    mx_full_h = mx_h_nat * ds
    in_crop = (
        (csv_pts_mx_all[:, 0] >= 0)
        & (csv_pts_mx_all[:, 0] < mx_full_w)
        & (csv_pts_mx_all[:, 1] >= 0)
        & (csv_pts_mx_all[:, 1] < mx_full_h)
    )
    csv_pts_mx_crop = csv_pts_mx_all[in_crop]
    csv_pts_he_crop = mx_to_he(csv_pts_mx_crop, m_inv)  # H&E full-res px
    csv_pts_he_ov = csv_pts_he_crop / ds  # H&E overview px
    print(f"  CSV: {len(csv_pts_mx_crop):,} / {len(csv_pts_mx_all):,} in crop")
    he_in_bounds = (
        (csv_pts_he_crop[:, 0] >= 0)
        & (csv_pts_he_crop[:, 0] < (ov_w * ds))
        & (csv_pts_he_crop[:, 1] >= 0)
        & (csv_pts_he_crop[:, 1] < (ov_h * ds))
    )
    print(f"  CSV projected to H&E FOV: {int(he_in_bounds.sum()):,} / {len(csv_pts_he_crop):,}")

    # -----------------------------------------------------------------------
    # Choose best zoom patch
    # -----------------------------------------------------------------------
    best_pid, best_score = None, -1
    for p in patches:
        x0, y0 = int(p["x0"]), int(p["y0"])
        pid = f"{x0}_{y0}"
        if not (processed_dir / "he" / f"{pid}.png").exists():
            continue
        jp = processed_dir / "cellvit" / f"{pid}.json"
        n_cv = len(json.load(jp.open()).get("cells", [])) if jp.exists() else 0
        csv_local = csv_pts_he_crop - np.array([x0, y0])
        n_csv = int(
            (
                (csv_local[:, 0] >= 0)
                & (csv_local[:, 0] < patch_size)
                & (csv_local[:, 1] >= 0)
                & (csv_local[:, 1] < patch_size)
            ).sum()
        )
        if n_cv + n_csv > best_score:
            best_score, best_pid = n_cv + n_csv, pid
    print(f"  Zoom patch: {best_pid} (score={best_score})")

    # -----------------------------------------------------------------------
    # Column 1: H&E side (all in H&E overview space)
    # -----------------------------------------------------------------------
    c1r1 = _label(he_bgr, "c1r1) H&E overview", scale=title_scale)

    c1r2, n_cv = _draw_contours(
        _blank_bgr(he_bgr),
        cv_contours_ov,
        color=(0, 255, 255),
        thickness=max(1, cellvit_contour_lw),
    )
    c1r2 = _label(c1r2, f"c1r2) CellViT contours ({n_cv:,})", scale=title_scale)

    c1r3, _ = _draw_contours(
        he_bgr,
        cv_contours_ov,
        color=(0, 255, 255),
        thickness=max(1, cellvit_contour_lw),
    )
    c1r3 = _label(c1r3, "c1r3) H&E + CellViT contours (cyan)", scale=title_scale)

    # -----------------------------------------------------------------------
    # Column 2: MX warped to H&E space
    # -----------------------------------------------------------------------
    c2r1 = _label(mx_bgr_warped, "c2r1) MX DNA warped->H&E (m_full)", scale=title_scale)

    c2r2, n_csv_he = _dots(
        _blank_bgr(he_bgr),
        csv_pts_he_ov,
        (255, 255, 0),
        radius=max(3, csv_radius),
        thickness=max(1, csv_thickness),
        outline_color=(255, 255, 255),
        outline_thickness=max(1, csv_thickness),
    )
    c2r2 = _label(c2r2, f"c2r2) CSV centroids -> H&E ({n_csv_he:,})", scale=title_scale)

    c2r3, _ = _dots(
        mx_bgr_warped,
        csv_pts_he_ov,
        (255, 255, 0),
        radius=max(3, csv_radius),
        thickness=max(1, csv_thickness),
        outline_color=(0, 0, 0),
        outline_thickness=max(1, csv_thickness),
    )
    c2r3 = _label(c2r3, "c2r3) warped MX + CSV->H&E", scale=title_scale)

    # -----------------------------------------------------------------------
    # Column 3: Cross-modal in H&E space
    # -----------------------------------------------------------------------
    mx_u8 = percentile_to_uint8(mx_gray_warped.astype(np.float32))
    mx_color_warped = cv2.applyColorMap(mx_u8, cv2.COLORMAP_HOT)
    blend_alpha = float(np.clip(args.blend_alpha, 0.05, 0.95))
    # Match debug_match_he_mul.py style: linear blend between RGB H&E and HOT MX.
    blend = (
        he_bgr.astype(np.float32) * (1.0 - blend_alpha)
        + mx_color_warped.astype(np.float32) * blend_alpha
    ).clip(0, 255).astype(np.uint8)
    c3r1 = _label(
        blend,
        f"c3r1) Overlay H&E + warped MX (HOT), alpha={blend_alpha:.2f}",
        scale=title_scale,
    )
    c3r1 = _draw_c3_overlay_legend(c3r1, blend_alpha=blend_alpha)

    c3r2 = he_bgr.copy()
    c3r2, _ = _draw_contours(
        c3r2,
        cv_contours_ov,
        color=(0, 255, 0),
        thickness=max(1, cellvit_contour_lw),
    )
    c3r2, _ = _dots(
        c3r2,
        csv_pts_he_ov,
        (255, 255, 0),
        radius=max(3, csv_radius),
        thickness=max(1, csv_thickness),
        outline_color=(0, 0, 0),
        outline_thickness=max(1, csv_thickness),
    )
    c3r2 = _label(c3r2, "c3r2) CellViT contours=green, CSV->H&E=cyan", scale=title_scale)

    c3r3 = _make_zoom_panel(
        best_pid,
        he_bgr=he_bgr,
        cv_pts_he_ov=cv_pts_ov,
        cv_contours_he_ov=cv_contours_ov,
        csv_pts_he_ov=csv_pts_he_ov,
        ds=ds,
        title_scale=title_scale,
        target_h=ov_h,
        target_w=ov_w,
        patch_size=patch_size,
        cellvit_contour_lw=max(1, cellvit_contour_lw),
        csv_radius=max(3, csv_radius),
        csv_thickness=max(1, csv_thickness),
    )

    if best_pid:
        zx0, zy0 = map(int, best_pid.split("_"))
        zx1 = zx0 + patch_size
        zy1 = zy0 + patch_size
        rect_tl = (int(zx0 / ds), int(zy0 / ds))
        rect_br = (int(zx1 / ds), int(zy1 / ds))
        for panel in (c1r1, c2r1, c3r1, c3r2):
            cv2.rectangle(panel, rect_tl, rect_br, (0, 0, 255), 3)
        cv2.putText(
            c3r2,
            "zoom patch",
            (rect_tl[0] + 6, max(28, rect_tl[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Assemble grid
    # -----------------------------------------------------------------------
    panels = [
        [c1r1, c2r1, c3r1],
        [c1r2, c2r2, c3r2],
        [c1r3, c2r3, c3r3],
    ]
    rows = [np.concatenate(row, axis=1) for row in panels]
    grid = np.concatenate(rows, axis=0)
    if not args.no_legend:
        grid = _append_legend_bar(grid, blend_alpha=blend_alpha)

    out_path = out_dir / "alignment_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"\nSaved: {out_path}  ({grid.shape[1]}x{grid.shape[0]} px)")

    names = [
        ["c1r1_he.png", "c2r1_mx_warped.png", "c3r1_blend.png"],
        ["c1r2_cellvit.png", "c2r2_csv_he.png", "c3r2_cv_csv_he.png"],
        ["c1r3_he_cellvit.png", "c2r3_mx_csv.png", "c3r3_zoom.png"],
    ]
    for row_p, row_n in zip(panels, names):
        for panel, name in zip(row_p, row_n):
            cv2.imwrite(str(out_dir / name), panel)
    print(f"Saved 9 individual panels to {out_dir}")


def _blank_bgr(ref: np.ndarray) -> np.ndarray:
    return np.zeros_like(ref)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3×3 alignment debug grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed", required=True)
    p.add_argument(
        "--index-path",
        default=None,
        help="Optional index JSON path. Default: <processed>/index.json",
    )
    p.add_argument(
        "--icp-index",
        default=None,
        help="Optional Stage-2.5 index with icp_matrix; if set, compose warp_matrix with icp_matrix.",
    )
    p.add_argument("--he-image", required=True)
    p.add_argument("--multiplex-image", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--csv-mpp", type=float, default=0.65)
    p.add_argument(
        "--mx-crop-origin", type=float, nargs=2, default=None, metavar=("OX", "OY")
    )
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument(
        "--cellvit-radius",
        type=int,
        default=3,
        help="CellViT contour line width in pixels.",
    )
    p.add_argument("--csv-radius", type=int, default=6)
    p.add_argument("--csv-thickness", type=int, default=2)
    p.add_argument("--title-scale", type=float, default=1.05)
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=0.68,
        help="Alpha for c3r1 transparent overlay of warped MX over H&E.",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable global legend bar in alignment_grid.png.",
    )
    p.add_argument("--out-dir", default="debug/")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
