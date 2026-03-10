"""
viz_registration_debug.py — 4-panel registration debug visualization.

Generates a figure comparing CellViT detections, CSV centroids in H&E space,
and their alignment after ICP.  Run after Stage 2 (CellViT), before Stage 2.5.

Panels
------
a  H&E overview (high-res) with Stage 1 patch grid boxes.
b  CellViT centroids (all patches) overlaid on H&E.
c  CSV-in-HE (blue, pre-ICP) vs ICP-aligned CellViT (orange), zoomed to patch
   region with H&E px axes.
d  Overlap: CellViT (green) vs CSV-in-HE (red), same zoom and axes.

IMPORTANT — coordinate spaces
------------------------------
- CellViT centroids are in H&E crop pixel space (0 … img_w × 0 … img_h).
- CSV (Xt, Yt) are in µm for the FULL SLIDE.  After ÷csv_mpp → full-slide MX
  px.  If the multiplex image is a CROP, you MUST pass --mx-crop-origin OX OY
  (top-left of the crop in full-slide MX px) so that CSV coords are shifted into
  crop MX px space before inv(m_full) is applied.  Without this, CSV centroids
  will project far outside the crop and panels c/d will look empty.

CLI
---
python -m tools.viz_registration_debug \\
  --processed crop/ \\
  --he-image data/WD-76845-096-crop.ome.tif \\
  --csv data/WD-76845-097.csv \\
  --csv-mpp 0.65 \\
  --mx-crop-origin OX OY \\   # required for crops
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
    crop_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None
    if crop_origin is None:
        print(
            "WARNING: --mx-crop-origin not provided. CSV coords may be in full-slide "
            "MX space and will project outside the crop. Panels c/d may look empty."
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

    icp_cellvit_he = apply_affine(M_icp, he_pts_he) if len(he_pts_he) > 0 else he_pts_he
    icp_contours = _transform_contours(M_icp, he_contours) if he_contours else []

    # Patch bounding box in H&E px
    px0 = [p["x0"] for p in patches]
    py0 = [p["y0"] for p in patches]
    patch_x_min = min(px0)
    patch_y_min = min(py0)
    patch_x_max = max(px0) + patch_size
    patch_y_max = max(py0) + patch_size
    margin = patch_size // 2
    roi_x0 = max(0, patch_x_min - margin)
    roi_x1 = min(img_w, patch_x_max + margin)
    roi_y0 = max(0, patch_y_min - margin)
    roi_y1 = min(img_h, patch_y_max + margin)
    print(f"Patch ROI (H&E px): x=[{roi_x0}, {roi_x1}]  y=[{roi_y0}, {roi_y1}]")

    rng = np.random.default_rng(0)
    N_MAX = 8000  # max scatter points per cloud

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes_grid = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    fig.suptitle("Registration Debug: Stage 2.5 ICP+TPS", fontsize=13)

    ax_a, ax_b = axes_grid[0, 0], axes_grid[0, 1]
    ax_c, ax_d = axes_grid[1, 0], axes_grid[1, 1]

    # Extent maps overview pixels → H&E full-res px for panels a/b
    extent_lo = [0, img_w, img_h, 0]  # [left, right, bottom, top] in H&E px

    # ── Panel a: H&E overview + patch grid ──────────────────────────────────
    ax_a.imshow(rgb_lo, extent=extent_lo, origin="upper", aspect="equal")
    ax_a.set_title("a) H&E overview + patch grid")
    for p in patches:
        rect = mpatches.Rectangle(
            (p["x0"], p["y0"]),
            patch_size,
            patch_size,
            linewidth=1.0,
            edgecolor="lime",
            facecolor="none",
        )
        ax_a.add_patch(rect)
    ax_a.set_xlabel("H&E x (px)")
    ax_a.set_ylabel("H&E y (px)")
    ax_a.invert_yaxis()

    # ── Panel b: CellViT centroids + contours on H&E ────────────────────────
    ax_b.imshow(rgb_lo, extent=extent_lo, origin="upper", aspect="equal")
    ax_b.set_title(
        f"b) CellViT centroids + contours (points={len(he_pts_he)}, contours={len(he_contours)})"
    )
    _add_contours(
        ax_b,
        he_contours,
        color="deepskyblue",
        linewidth=args.contour_lw,
        alpha=args.contour_alpha,
    )
    if len(he_pts_he) > 0:
        pts = _subsample(he_pts_he, N_MAX, rng)
        ax_b.scatter(pts[:, 0], pts[:, 1], s=1, c="yellow", alpha=0.7, linewidths=0)
    ax_b.set_xlabel("H&E x (px)")
    ax_b.set_ylabel("H&E y (px)")
    ax_b.invert_yaxis()

    # Extent for high-res ROI panels (maps hi-res overview → H&E px)
    extent_hi = [0, img_w, img_h, 0]

    def _setup_roi_ax(ax: plt.Axes, title: str) -> None:
        """Configure ROI axis with H&E px limits and labels."""
        ax.imshow(rgb_hi, extent=extent_hi, origin="upper", aspect="equal")
        ax.set_xlim(roi_x0, roi_x1)
        ax.set_ylim(roi_y1, roi_y0)  # y inverted (image origin top-left)
        ax.set_title(title)
        ax.set_xlabel("H&E x (px)")
        ax.set_ylabel("H&E y (px)")
        # Draw patch outlines for reference
        for p in patches:
            rect = mpatches.Rectangle(
                (p["x0"], p["y0"]),
                patch_size,
                patch_size,
                linewidth=0.6,
                edgecolor="lime",
                facecolor="none",
                alpha=0.5,
            )
            ax.add_patch(rect)

    # ── Panel c: CSV-in-HE (blue) vs ICP-aligned CellViT (orange) ───────────
    _setup_roi_ax(
        ax_c, "c) CSV-in-HE (blue) vs ICP-aligned CellViT (orange + contours)"
    )

    if len(csv_in_he) > 0:
        pts_csv = _subsample(
            _filter_bbox(csv_in_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        n_visible = len(pts_csv)
        ax_c.scatter(
            pts_csv[:, 0],
            pts_csv[:, 1],
            s=4,
            c="deepskyblue",
            alpha=0.6,
            linewidths=0,
            label=f"CSV-in-HE (visible={n_visible})",
        )
        if n_visible == 0:
            ax_c.text(
                0.5,
                0.5,
                "CSV-in-HE: 0 pts in ROI\n(check --mx-crop-origin)",
                transform=ax_c.transAxes,
                ha="center",
                va="center",
                color="deepskyblue",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="black", alpha=0.6),
            )

    if len(icp_cellvit_he) > 0:
        pts_icp = _subsample(
            _filter_bbox(icp_cellvit_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        ax_c.scatter(
            pts_icp[:, 0],
            pts_icp[:, 1],
            s=4,
            c="orange",
            alpha=0.6,
            linewidths=0,
            label=f"ICP CellViT (visible={len(pts_icp)})",
        )
    roi_icp_contours = _filter_contours_bbox(
        icp_contours, roi_x0, roi_x1, roi_y0, roi_y1
    )
    _add_contours(
        ax_c,
        roi_icp_contours,
        color="orange",
        linewidth=args.contour_lw,
        alpha=args.contour_alpha,
    )

    ax_c.legend(loc="upper right", markerscale=3, fontsize=7)

    # ── Panel d: CellViT + contours (green/cyan) vs CSV-in-HE (red) ────────
    _setup_roi_ax(
        ax_d, "d) Overlap: CellViT (green + cyan contours) vs CSV-in-HE (red)"
    )

    roi_contours = _filter_contours_bbox(he_contours, roi_x0, roi_x1, roi_y0, roi_y1)
    _add_contours(
        ax_d,
        roi_contours,
        color="cyan",
        linewidth=args.contour_lw,
        alpha=args.contour_alpha,
    )

    if len(he_pts_he) > 0:
        pts_cv = _subsample(
            _filter_bbox(he_pts_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        ax_d.scatter(
            pts_cv[:, 0],
            pts_cv[:, 1],
            s=4,
            c="lime",
            alpha=0.7,
            linewidths=0,
            label=f"CellViT (visible={len(pts_cv)})",
        )

    if len(csv_in_he) > 0:
        pts_csv2 = _subsample(
            _filter_bbox(csv_in_he, roi_x0, roi_x1, roi_y0, roi_y1), N_MAX, rng
        )
        ax_d.scatter(
            pts_csv2[:, 0],
            pts_csv2[:, 1],
            s=4,
            c="red",
            alpha=0.6,
            linewidths=0,
            label=f"CSV-in-HE (visible={len(pts_csv2)})",
        )
        if len(pts_csv2) == 0:
            ax_d.text(
                0.5,
                0.45,
                "CSV-in-HE: 0 pts in ROI\n(check --mx-crop-origin)",
                transform=ax_d.transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="black", alpha=0.6),
            )

    ax_d.legend(loc="upper right", markerscale=3, fontsize=7)

    plt.tight_layout()
    out_png = pathlib.Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-panel registration debug visualization (run after Stage 2).",
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
        help="Top-left of crop in full-slide MX px (required for crop images).",
    )
    p.add_argument("--out-png", required=True, help="Output PNG path.")
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
        help="Downsample factor for H&E background in panels c/d (smaller = higher resolution).",
    )
    p.add_argument(
        "--contour-lw",
        type=float,
        default=0.9,
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
