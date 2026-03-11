"""
viz_registration_debug.py — ICP point-assignment debug visualization.

Generates a 3-panel figure comparing original and ICP-modified CSV point
assignments against CellViT detections in H&E space.

Panels
------
a  Cropped HE region + CellViT contours + original CSV-in-HE points.
b  Cropped HE region + CellViT contours + ICP-modified CSV points.
c  Zoomed patch region of panel b.

IMPORTANT — coordinate spaces
------------------------------
- CellViT centroids are in H&E crop pixel space (0 … img_w × 0 … img_h).
- CSV (Xt, Yt) are in µm for the FULL SLIDE.  After ÷csv_mpp → full-slide MX
  px.  If the multiplex image is a CROP, pass --mx-crop-origin OX OY (top-left
  of the crop in full-slide MX px), or store mx_crop_origin in index.json.
  Without this shift into crop MX px space before inv(m_full), CSV centroids
  will project far outside the crop and point overlays will look empty.

CLI
---
python -m tools.viz_registration_debug \\
  --processed crop/ \\
  --he-image data/WD-76845-096-crop.ome.tif \\
  --csv data/WD-76845-097.csv \\
  --csv-mpp 0.65 \\
  --mx-crop-origin OX OY \\
  --crop-margin 128 \\
  --zoom-patch 0_1024 \\
  --out-png crop/registration_debug.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from utils.ome import get_image_dims, open_zarr_store, read_overview_chw
from stages.refine_registration import (
    affine_icp,
    apply_affine,
    csv_to_he_coords,
    load_he_centroids,
    resolve_mx_crop_origin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _he_overview_rgb(he_tif_path: pathlib.Path, ds: int) -> tuple[np.ndarray, int, int]:
    """Read H&E overview as uint8 RGB (H, W, 3).  Returns rgb, img_w, img_h."""
    with tifffile.TiffFile(str(he_tif_path)) as tif:
        store = open_zarr_store(tif)
        img_w, img_h, axes = get_image_dims(tif)
        chw = read_overview_chw(store, axes, img_h, img_w, ds)

    # chw: (C, H, W). H&E RGB is stored as 3 channels.
    if chw.shape[0] >= 3:
        rgb = np.stack([chw[0], chw[1], chw[2]], axis=-1)
    else:
        g = chw[0]
        rgb = np.stack([g, g, g], axis=-1)

    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (
            ((rgb.astype(np.float32) - lo) / (hi - lo) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
    else:
        rgb = np.zeros_like(rgb, dtype=np.uint8)

    return rgb, img_w, img_h


def _subsample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[rng.choice(len(pts), n, replace=False)]


def _filter_bbox(
    pts: np.ndarray, x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    """Keep only points inside [x0, x1] × [y0, y1]."""
    if len(pts) == 0:
        return pts
    mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return pts[mask]


def _extract_cells(data: object) -> list[dict]:
    """Return list of CellViT cell dicts from common JSON layouts."""
    if isinstance(data, dict):
        for key in ("cells", "nuclei", "detections", "instances"):
            value = data.get(key)
            if isinstance(value, list):
                return [cell for cell in value if isinstance(cell, dict)]
        if all(isinstance(v, dict) for v in data.values()):
            return [v for v in data.values() if isinstance(v, dict)]
        return []
    if isinstance(data, list):
        return [cell for cell in data if isinstance(cell, dict)]
    return []


def _extract_contour(cell: dict) -> np.ndarray | None:
    """Parse one cell contour as (N, 2) float32, or None when unavailable."""
    contour = None
    for key in ("contour", "contours", "polygon", "boundary", "coords", "points"):
        if key in cell:
            contour = cell[key]
            break
    if contour is None:
        return None

    pts = np.asarray(contour, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pass
    elif pts.ndim == 3 and pts.shape[-1] == 2:
        pts = pts.reshape(-1, 2)
    else:
        return None

    if len(pts) < 3:
        return None

    # Close the contour for line rendering.
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def load_he_contours(
    cellvit_dir: pathlib.Path, patches: list[dict]
) -> list[np.ndarray]:
    """Load CellViT contours and convert local patch coords to global H&E px."""
    contours: list[np.ndarray] = []
    for patch_meta in patches:
        x0 = int(patch_meta["x0"])
        y0 = int(patch_meta["y0"])
        patch_id = f"{x0}_{y0}"
        json_path = cellvit_dir / f"{patch_id}.json"
        if not json_path.exists():
            continue

        with json_path.open() as fh:
            data = json.load(fh)
        for cell in _extract_cells(data):
            pts = _extract_contour(cell)
            if pts is None:
                continue
            pts = pts.copy()
            pts[:, 0] += float(x0)
            pts[:, 1] += float(y0)
            contours.append(pts)

    return contours


def _filter_contours_bbox(
    contours: list[np.ndarray], x0: float, x1: float, y0: float, y1: float
) -> list[np.ndarray]:
    """Keep contours whose bounding boxes intersect the ROI."""
    keep: list[np.ndarray] = []
    for pts in contours:
        c_x0 = float(np.min(pts[:, 0]))
        c_x1 = float(np.max(pts[:, 0]))
        c_y0 = float(np.min(pts[:, 1]))
        c_y1 = float(np.max(pts[:, 1]))
        if c_x1 < x0 or c_x0 > x1 or c_y1 < y0 or c_y0 > y1:
            continue
        keep.append(pts)
    return keep


def _add_contours(
    ax: plt.Axes,
    contours: list[np.ndarray],
    color: str,
    linewidth: float,
    alpha: float,
) -> int:
    """Render contour polylines on an axes. Returns number of contours drawn."""
    if not contours:
        return 0
    line_collection = LineCollection(
        contours, colors=color, linewidths=linewidth, alpha=alpha
    )
    ax.add_collection(line_collection)
    return len(contours)


def _transform_contours(m: np.ndarray, contours: list[np.ndarray]) -> list[np.ndarray]:
    """Apply affine transform to each contour."""
    transformed: list[np.ndarray] = []
    for pts in contours:
        transformed.append(apply_affine(m, pts))
    return transformed


def _in_bbox(pts: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Return boolean mask for points inside ROI."""
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (pts[:, 0] >= x0)
        & (pts[:, 0] <= x1)
        & (pts[:, 1] >= y0)
        & (pts[:, 1] <= y1)
    )


def _trajectory_segments(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    max_segments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build line segments showing src->dst trajectories near a ROI."""
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    src_in = _in_bbox(src_pts, x0, x1, y0, y1)
    dst_in = _in_bbox(dst_pts, x0, x1, y0, y1)
    keep = np.where(src_in | dst_in)[0]
    if len(keep) == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if len(keep) > max_segments:
        keep = rng.choice(keep, size=max_segments, replace=False)

    return np.stack([src_pts[keep], dst_pts[keep]], axis=1).astype(np.float64)


def _invert_affine_2x3(m: np.ndarray) -> np.ndarray:
    """Return inverse of a 2x3 affine matrix as 2x3."""
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    inv3 = np.linalg.inv(m3)
    return inv3[:2, :]


def _select_zoom_patch(
    patches: list[dict],
    img_w: int,
    img_h: int,
    default_patch_size: int,
    patch_key: str | None,
) -> tuple[int, int, int]:
    """Choose zoom patch by explicit key or nearest-to-center fallback."""
    if not patches:
        return 0, 0, default_patch_size

    if patch_key:
        try:
            sx, sy = patch_key.split("_", 1)
            x_key = int(sx)
            y_key = int(sy)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"--zoom-patch must be 'x0_y0', got: {patch_key!r}") from exc
        for p in patches:
            if int(p["x0"]) == x_key and int(p["y0"]) == y_key:
                return x_key, y_key, int(p.get("patch_size", default_patch_size))
        raise ValueError(f"--zoom-patch {patch_key!r} not found in index.json patches.")

    cx = 0.5 * float(img_w)
    cy = 0.5 * float(img_h)
    best = None
    best_d2 = float("inf")
    for p in patches:
        x0 = int(p["x0"])
        y0 = int(p["y0"])
        ps = int(p.get("patch_size", default_patch_size))
        pcx = x0 + 0.5 * ps
        pcy = y0 + 0.5 * ps
        d2 = (pcx - cx) ** 2 + (pcy - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (x0, y0, ps)

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    processed_dir = pathlib.Path(args.processed)
    cellvit_dir = processed_dir / "cellvit"
    index_path = processed_dir / "index.json"

    # Load index.json
    with index_path.open() as fh:
        index = json.load(fh)
    patches = index["patches"]
    m_full = np.array(index["warp_matrix"], dtype=np.float64)
    patch_size = index.get("patch_size", 256)
    print(f"Loaded index.json: {len(patches)} patches, patch_size={patch_size}")

    # H&E overview — two resolutions
    he_tif_path = pathlib.Path(args.he_image)
    ds_lo = max(1, args.overview_ds)  # for panels a/b (full FOV)
    ds_hi = max(1, args.roi_overview_ds)  # for panels c/d (zoomed, higher res)

    rgb_lo, img_w, img_h = _he_overview_rgb(he_tif_path, ds_lo)
    rgb_hi, _, _ = _he_overview_rgb(he_tif_path, ds_hi)
    print(
        f"H&E overview lo={rgb_lo.shape} ds={ds_lo},  hi={rgb_hi.shape} ds={ds_hi}  "
        f"(full-res {img_w}×{img_h})"
    )

    # CellViT centroids + contours (H&E crop full-res px)
    he_pts_he, _ = load_he_centroids(cellvit_dir, patches, coord_scale=1.0)
    he_contours = load_he_contours(cellvit_dir, patches)
    print(f"CellViT centroids: {len(he_pts_he)}")
    print(f"CellViT contours: {len(he_contours)}")
    if len(he_pts_he) > 0:
        print(
            f"  HE range  x=[{he_pts_he[:,0].min():.0f}, {he_pts_he[:,0].max():.0f}]"
            f"  y=[{he_pts_he[:,1].min():.0f}, {he_pts_he[:,1].max():.0f}]"
        )

    # CSV centroids → MX px + H&E px
    cli_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None
    crop_origin = resolve_mx_crop_origin(index, cli_origin=cli_origin)
    if crop_origin is None:
        print(
            "WARNING: --mx-crop-origin not provided. CSV coords may be in full-slide "
            "MX space and will project outside the crop. Overlays may look empty."
        )
    elif cli_origin is None:
        print(
            f"Using mx_crop_origin from index.json: ({crop_origin[0]:.1f}, {crop_origin[1]:.1f})"
        )
    mx_pts, csv_in_he = csv_to_he_coords(
        pathlib.Path(args.csv),
        m_full=m_full,
        csv_mpp=args.csv_mpp,
        crop_origin=crop_origin,
    )
    print(f"CSV centroids: {len(mx_pts)}")
    if len(csv_in_he) > 0:
        print(
            f"  CSV-in-HE range  x=[{csv_in_he[:,0].min():.0f}, {csv_in_he[:,0].max():.0f}]"
            f"  y=[{csv_in_he[:,1].min():.0f}, {csv_in_he[:,1].max():.0f}]"
        )
    csv_in_he_in_bounds = _filter_bbox(csv_in_he, 0.0, float(img_w), 0.0, float(img_h))
    print(
        f"  CSV-in-HE inside H&E bounds: {len(csv_in_he_in_bounds)} / {len(csv_in_he)}"
    )

    # ICP: align CellViT → CSV-in-HE (lightweight, for viz only)
    M_icp = np.eye(2, 3, dtype=np.float64)
    icp_target = csv_in_he_in_bounds if len(csv_in_he_in_bounds) >= 4 else csv_in_he
    if len(he_pts_he) >= 4 and len(icp_target) >= 4:
        print(
            f"Running ICP (max_iter={args.icp_max_iter}, gate={args.distance_gate} H&E px) ..."
        )
        M_icp, icp_n, icp_iters = affine_icp(
            src_he=he_pts_he,
            dst_he=icp_target,
            max_iter=args.icp_max_iter,
            tol=args.icp_tol,
            distance_gate=args.distance_gate,
        )
        print(f"  ICP done: {icp_iters} iters, {icp_n} final matches")
    else:
        print("Skipping ICP: too few centroids.")

    # Build ICP-modified CSV points in original HE frame:
    # M_icp maps CellViT -> CSV-in-HE, so inverse maps CSV -> CellViT.
    M_icp_inv = _invert_affine_2x3(M_icp)
    csv_icp_he = apply_affine(M_icp_inv, csv_in_he) if len(csv_in_he) > 0 else csv_in_he

    zoom_x0, zoom_y0, zoom_ps = _select_zoom_patch(
        patches=patches,
        img_w=img_w,
        img_h=img_h,
        default_patch_size=patch_size,
        patch_key=args.zoom_patch,
    )
    crop_margin = max(0, int(args.crop_margin))
    view_x0 = max(0, zoom_x0 - crop_margin)
    view_y0 = max(0, zoom_y0 - crop_margin)
    view_x1 = min(img_w, zoom_x0 + zoom_ps + crop_margin)
    view_y1 = min(img_h, zoom_y0 + zoom_ps + crop_margin)
    margin = max(0, int(args.zoom_margin))
    roi_x0 = max(0, zoom_x0 - margin)
    roi_y0 = max(0, zoom_y0 - margin)
    roi_x1 = min(img_w, zoom_x0 + zoom_ps + margin)
    roi_y1 = min(img_h, zoom_y0 + zoom_ps + margin)
    print(
        f"Zoom patch: x0={zoom_x0}, y0={zoom_y0}, size={zoom_ps}, "
        f"ROI x=[{roi_x0}, {roi_x1}] y=[{roi_y0}, {roi_y1}]"
    )
    print(
        f"Cropped A/B ROI: x=[{view_x0}, {view_x1}] y=[{view_y0}, {view_y1}]"
    )

    rng = np.random.default_rng(0)
    N_MAX = 10000
    extent_lo = [0, img_w, img_h, 0]
    extent_hi = [0, img_w, img_h, 0]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(22, 7), dpi=150)
    fig.suptitle("ICP Point Assignment Debug", fontsize=13)

    for ax in (ax_a, ax_b):
        ax.imshow(rgb_lo, extent=extent_lo, origin="upper", aspect="equal")
        ax.set_xlabel("H&E x (px)")
        ax.set_ylabel("H&E y (px)")
        ax.set_xlim(view_x0, view_x1)
        ax.set_ylim(view_y1, view_y0)

    # a) original CSV points
    ax_a.set_title("a) Cropped HE + CellViT + original CSV")
    if len(he_contours) > 0:
        he_contours_view = _filter_contours_bbox(
            he_contours, view_x0, view_x1, view_y0, view_y1
        )
        _add_contours(
            ax_a,
            he_contours_view,
            color="cyan",
            linewidth=args.contour_lw,
            alpha=args.contour_alpha,
        )
    if len(he_pts_he) > 0:
        pts_cv = _subsample(
            _filter_bbox(he_pts_he, view_x0, view_x1, view_y0, view_y1), N_MAX, rng
        )
        ax_a.scatter(
            pts_cv[:, 0],
            pts_cv[:, 1],
            s=float(args.cellvit_point_size),
            c="lime",
            alpha=0.6,
            linewidths=0,
            label=f"CellViT ({len(pts_cv)})",
        )
    if len(csv_in_he) > 0:
        pts_csv = _subsample(
            _filter_bbox(csv_in_he, view_x0, view_x1, view_y0, view_y1), N_MAX, rng
        )
        ax_a.scatter(
            pts_csv[:, 0],
            pts_csv[:, 1],
            s=float(args.csv_point_size),
            c="red",
            alpha=0.55,
            linewidths=0,
            label=f"CSV original ({len(pts_csv)})",
        )
    ax_a.legend(loc="upper right", markerscale=3, fontsize=7)

    # b) ICP-modified CSV points
    ax_b.set_title("b) Cropped HE + CellViT + transformed CSV (ICP)")
    if len(he_contours) > 0:
        _add_contours(
            ax_b,
            he_contours_view,
            color="cyan",
            linewidth=args.contour_lw,
            alpha=args.contour_alpha,
        )
    if len(he_pts_he) > 0:
        pts_cv2 = _subsample(
            _filter_bbox(he_pts_he, view_x0, view_x1, view_y0, view_y1), N_MAX, rng
        )
        ax_b.scatter(
            pts_cv2[:, 0],
            pts_cv2[:, 1],
            s=float(args.cellvit_point_size),
            c="lime",
            alpha=0.6,
            linewidths=0,
            label=f"CellViT ({len(pts_cv2)})",
        )
    if len(csv_icp_he) > 0:
        pts_csv_icp = _subsample(
            _filter_bbox(csv_icp_he, view_x0, view_x1, view_y0, view_y1), N_MAX, rng
        )
        ax_b.scatter(
            pts_csv_icp[:, 0],
            pts_csv_icp[:, 1],
            s=float(args.csv_point_size),
            c="orange",
            alpha=0.55,
            linewidths=0,
            label=f"CSV ICP-modified ({len(pts_csv_icp)})",
        )
    traj_segments = _trajectory_segments(
        src_pts=csv_in_he,
        dst_pts=csv_icp_he,
        x0=view_x0,
        x1=view_x1,
        y0=view_y0,
        y1=view_y1,
        max_segments=max(1, int(args.max_trajectories)),
        rng=rng,
    )
    if len(traj_segments) > 0:
        traj_collection = LineCollection(
            traj_segments,
            colors="yellow",
            linewidths=float(args.trajectory_lw),
            alpha=0.28,
        )
        ax_b.add_collection(traj_collection)
    zoom_rect = mpatches.Rectangle(
        (zoom_x0, zoom_y0),
        zoom_ps,
        zoom_ps,
        linewidth=float(args.box_lw),
        edgecolor="yellow",
        facecolor="none",
    )
    ax_b.add_patch(zoom_rect)
    ax_b.legend(loc="upper right", markerscale=3, fontsize=7)

    # c) zoom patch from panel b
    ax_c.imshow(rgb_hi, extent=extent_hi, origin="upper", aspect="equal")
    ax_c.set_xlim(roi_x0, roi_x1)
    ax_c.set_ylim(roi_y1, roi_y0)
    ax_c.set_xlabel("H&E x (px)")
    ax_c.set_ylabel("H&E y (px)")
    ax_c.set_title("c) Zoom patch of panel b")

    if len(he_contours) > 0:
        roi_contours = _filter_contours_bbox(he_contours, roi_x0, roi_x1, roi_y0, roi_y1)
        _add_contours(
            ax_c,
            roi_contours,
            color="cyan",
            linewidth=args.contour_lw,
            alpha=args.contour_alpha,
        )
    if len(he_pts_he) > 0:
        pts_cv_zoom = _subsample(
            _filter_bbox(he_pts_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        ax_c.scatter(
            pts_cv_zoom[:, 0],
            pts_cv_zoom[:, 1],
            s=float(args.zoom_point_size),
            c="lime",
            alpha=0.7,
            linewidths=0,
            label=f"CellViT ({len(pts_cv_zoom)})",
        )
    if len(csv_icp_he) > 0:
        pts_csv_zoom = _subsample(
            _filter_bbox(csv_icp_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        ax_c.scatter(
            pts_csv_zoom[:, 0],
            pts_csv_zoom[:, 1],
            s=float(args.zoom_point_size),
            c="orange",
            alpha=0.65,
            linewidths=0,
            label=f"CSV ICP-modified ({len(pts_csv_zoom)})",
        )
    patch_rect_zoom = mpatches.Rectangle(
        (zoom_x0, zoom_y0),
        zoom_ps,
        zoom_ps,
        linewidth=float(args.box_lw),
        edgecolor="yellow",
        facecolor="none",
    )
    ax_c.add_patch(patch_rect_zoom)
    ax_c.legend(loc="upper right", markerscale=2, fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = pathlib.Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-panel ICP point-assignment debug visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--processed", required=True, help="Stage 1 output directory (has index.json)."
    )
    p.add_argument("--he-image", required=True, help="H&E OME-TIFF path.")
    p.add_argument("--csv", required=True, help="MX cell CSV with Xt,Yt columns (µm).")
    p.add_argument(
        "--csv-mpp", type=float, default=0.65, help="µm/px of CSV coordinate space."
    )
    p.add_argument(
        "--mx-crop-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("OX", "OY"),
        help="Top-left of crop in full-slide MX px. If omitted, attempts to "
        "read mx_crop_origin from index.json.",
    )
    p.add_argument("--out-png", required=True, help="Output PNG path.")
    p.add_argument(
        "--zoom-patch",
        default=None,
        help="Optional zoom patch key in x0_y0 format. Default: patch nearest image center.",
    )
    p.add_argument(
        "--zoom-margin",
        type=int,
        default=0,
        help="Extra margin (pixels) around zoom patch for panel c.",
    )
    p.add_argument(
        "--crop-margin",
        type=int,
        default=128,
        help="Extra margin (pixels) around the selected patch for panels a/b.",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=2500,
        help="Maximum CSV trajectory segments (original -> transformed) drawn in panel b.",
    )
    p.add_argument(
        "--cellvit-point-size",
        type=float,
        default=9.0,
        help="Marker size for CellViT centroids in panels a/b.",
    )
    p.add_argument(
        "--csv-point-size",
        type=float,
        default=9.0,
        help="Marker size for CSV centroids in panels a/b.",
    )
    p.add_argument(
        "--zoom-point-size",
        type=float,
        default=14.0,
        help="Marker size for centroids in zoom panel c.",
    )
    p.add_argument(
        "--trajectory-lw",
        type=float,
        default=1.0,
        help="Line width for CSV trajectory segments in panel b.",
    )
    p.add_argument(
        "--box-lw",
        type=float,
        default=1.8,
        help="Line width for zoom patch rectangles in panels b/c.",
    )
    p.add_argument(
        "--overview-ds",
        type=int,
        default=2,
        help="Downsample factor for H&E background in panels a/b (smaller = higher resolution).",
    )
    p.add_argument(
        "--roi-overview-ds",
        type=int,
        default=1,
        help="Downsample factor for H&E background in panel c (smaller = higher resolution).",
    )
    p.add_argument(
        "--contour-lw",
        type=float,
        default=1.4,
        help="Line width for CellViT contour overlays.",
    )
    p.add_argument(
        "--contour-alpha",
        type=float,
        default=0.65,
        help="Alpha for CellViT contour overlays.",
    )
    p.add_argument(
        "--distance-gate", type=float, default=20.0, help="ICP distance gate (H&E px)."
    )
    p.add_argument("--icp-max-iter", type=int, default=50, help="Max ICP iterations.")
    p.add_argument(
        "--icp-tol", type=float, default=1e-4, help="ICP convergence tolerance."
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
