#!/usr/bin/env python3
"""
viz_alignment_grid.py — 3×3 (+1 col) alignment debug grid.

All of columns 1-3 use H&E coordinate space.  Column 4 uses MX crop space
(same as seg.tif / CSV native coords) as a ground-truth reference.

Columns:
  1. H&E side        — overview | CellViT centroids | H&E + CellViT
  2. MX warped→H&E   — MX DNA warped via m_full | CSV→H&E | warped MX + CSV
  3. Cross-modal HE  — H&E+warped-MX blend | CellViT(green)+CSV(red) | patch zoom
  4. MX/seg native   — seg.tif crop | seg + CSV in MX space | seg + MX DNA

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
    --seg             data/WD-76845-097.ome.seg.tif \\
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
from PIL import Image

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
# Centroid loaders
# ---------------------------------------------------------------------------


def load_cellvit_pts(cellvit_dir: pathlib.Path, patches: list[dict]) -> np.ndarray:
    """(N, 2) CellViT centroids in H&E full-res px."""
    pts: list[tuple[float, float]] = []
    for p in patches:
        x0, y0 = int(p["x0"]), int(p["y0"])
        jp = cellvit_dir / f"{x0}_{y0}.json"
        if not jp.exists():
            continue
        for c in json.load(jp.open()).get("cells", []):
            lx, ly = c.get("centroid", [0, 0])
            pts.append((x0 + float(lx), y0 + float(ly)))
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


def load_csv_pts_mx(
    csv_path: pathlib.Path,
    csv_mpp: float,
    crop_origin: tuple[float, float] | None,
) -> np.ndarray:
    """(N, 2) CSV centroids in MX crop px."""
    df = pd.read_csv(csv_path)
    pts = df[["Xt", "Yt"]].to_numpy(dtype=np.float64) / csv_mpp
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
        cv2.circle(out, (x, y), radius, color, -1)
    return out, int(in_b.sum())


def _label(img: np.ndarray, text: str, scale: float = 0.5) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        out,
        text,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h))


# ---------------------------------------------------------------------------
# Zoom patch helper
# ---------------------------------------------------------------------------


def _make_zoom_panel(
    patch_id: str,
    processed_dir: pathlib.Path,
    csv_pts_mx_crop: np.ndarray,
    m_full: np.ndarray,
    target_h: int,
    target_w: int,
    patch_size: int,
) -> np.ndarray:
    """H&E patch + CellViT (green) + CSV projected to H&E patch space (red)."""
    x0, y0 = map(int, patch_id.split("_"))
    he_path = processed_dir / "he" / f"{patch_id}.png"
    if not he_path.exists():
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return _label(blank, f"no patch {patch_id}")
    he_rgb = np.array(Image.open(he_path).convert("RGB"))
    out = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2BGR)

    # CellViT (green dots)
    cells = []
    cv_path = processed_dir / "cellvit" / f"{patch_id}.json"
    if cv_path.exists():
        cells = json.load(cv_path.open()).get("cells", [])
        for c in cells:
            lx, ly = c.get("centroid", [0, 0])
            cv2.circle(out, (int(lx), int(ly)), 4, (0, 220, 0), -1)

    # CSV → H&E patch local (red dots)
    m_inv = _inv_m_full(m_full)
    csv_he = mx_to_he(csv_pts_mx_crop, m_inv)
    csv_local = csv_he - np.array([x0, y0])
    in_p = (
        (csv_local[:, 0] >= 0)
        & (csv_local[:, 0] < patch_size)
        & (csv_local[:, 1] >= 0)
        & (csv_local[:, 1] < patch_size)
    )
    for x, y in csv_local[in_p].astype(int):
        cv2.circle(out, (x, y), 4, (0, 0, 220), -1)

    out = _label(out, f"patch {patch_id}  CV={len(cells)} CSV={in_p.sum()}", scale=0.4)
    return _resize_to(out, target_h, target_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = args.downsample
    crop_origin = tuple(args.mx_crop_origin) if args.mx_crop_origin else None

    processed_dir = pathlib.Path(args.processed)
    with (processed_dir / "index.json").open() as fh:
        index = json.load(fh)
    patches = index.get("patches", [])
    m_full = np.array(index["warp_matrix"], dtype=np.float64)
    m_inv = _inv_m_full(m_full)
    patch_size = int(index.get("patch_size", 512))

    print("m_full (H&E crop px → MX crop px):")
    print(f"  [[{m_full[0,0]:.4f}, {m_full[0,1]:.4f}, {m_full[0,2]:.1f}],")
    print(f"   [{m_full[1,0]:.4f}, {m_full[1,1]:.4f}, {m_full[1,2]:.1f}]]")

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

    # Native MX as hot-colormap (for column 4)
    mx_bgr_native = _resize_to(
        cv2.applyColorMap(mx_gray_native, cv2.COLORMAP_HOT), ov_h, ov_w
    )

    # -----------------------------------------------------------------------
    # Load seg.tif crop
    # -----------------------------------------------------------------------
    seg_gray = None
    if args.seg:
        print("Loading seg.tif crop ...")
        with tifffile.TiffFile(str(args.seg)) as tif:
            seg_w, seg_h, _ = get_image_dims(tif)
        # MX crop full-res dimensions (same coord space as seg)
        mx_full_w = mx_w_nat * ds
        mx_full_h = mx_h_nat * ds
        seg_gray = _load_seg_crop(
            pathlib.Path(args.seg), crop_origin, mx_full_w, mx_full_h, level=1
        )
        seg_gray = _resize_to(seg_gray, ov_h, ov_w)
        n_seg = (seg_gray > 0).sum()
        print(
            f"  seg crop overview: {seg_gray.shape[1]}x{seg_gray.shape[0]}, "
            f"nonzero overview px: {n_seg:,}"
        )

    # -----------------------------------------------------------------------
    # Load centroids
    # -----------------------------------------------------------------------
    print("Loading CellViT centroids ...")
    cv_pts_he = load_cellvit_pts(processed_dir / "cellvit", patches)
    cv_pts_ov = cv_pts_he / ds
    print(f"  CellViT: {len(cv_pts_he):,} cells")

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
    csv_pts_mx_ov = csv_pts_mx_crop / ds  # MX crop overview px
    csv_pts_he_crop = mx_to_he(csv_pts_mx_crop, m_inv)  # H&E full-res px
    csv_pts_he_ov = csv_pts_he_crop / ds  # H&E overview px
    print(f"  CSV: {len(csv_pts_mx_crop):,} / {len(csv_pts_mx_all):,} in crop")

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

    r = args.dot_radius

    # -----------------------------------------------------------------------
    # Column 1: H&E side (all in H&E overview space)
    # -----------------------------------------------------------------------
    c1r1 = _label(he_bgr, "C1 H&E overview")

    c1r2, n_cv = _dots(_blank_bgr(he_bgr), cv_pts_ov, (0, 220, 220), r)
    c1r2 = _label(c1r2, f"C1 CellViT centroids ({n_cv:,})")

    c1r3, _ = _dots(he_bgr, cv_pts_ov, (0, 220, 220), r)
    c1r3 = _label(c1r3, "C1 H&E + CellViT (cyan)")

    # -----------------------------------------------------------------------
    # Column 2: MX warped to H&E space
    # -----------------------------------------------------------------------
    c2r1 = _label(mx_bgr_warped, "C2 MX DNA warped→H&E (m_full)")

    c2r2, n_csv_he = _dots(_blank_bgr(he_bgr), csv_pts_he_ov, (220, 220, 0), r)
    c2r2 = _label(c2r2, f"C2 CSV centroids→H&E ({n_csv_he:,})")

    c2r3, _ = _dots(mx_bgr_warped, csv_pts_he_ov, (220, 220, 0), r)
    c2r3 = _label(c2r3, "C2 warped MX + CSV→H&E (yellow)")

    # -----------------------------------------------------------------------
    # Column 3: Cross-modal in H&E space
    # -----------------------------------------------------------------------
    he_gray = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2GRAY)
    blend = np.zeros((ov_h, ov_w, 3), dtype=np.uint8)
    blend[:, :, 0] = he_gray  # B channel = H&E
    blend[:, :, 1] = mx_gray_warped  # G channel = warped MX DNA
    c3r1 = _label(blend, "C3 H&E(blue) + warped-MX(green)")

    c3r2 = he_bgr.copy()
    c3r2, _ = _dots(c3r2, cv_pts_ov, (0, 220, 0), r)  # green = CellViT
    c3r2, _ = _dots(c3r2, csv_pts_he_ov, (0, 0, 220), r)  # red   = CSV→H&E
    c3r2 = _label(c3r2, "C3 CellViT=green  CSV->HE=red")

    c3r3 = _make_zoom_panel(
        best_pid, processed_dir, csv_pts_mx_crop, m_full, ov_h, ov_w, patch_size
    )

    # -----------------------------------------------------------------------
    # Column 4: seg.tif crop in MX native space
    # -----------------------------------------------------------------------
    if seg_gray is not None:
        seg_bgr = cv2.cvtColor(seg_gray, cv2.COLOR_GRAY2BGR)

        c4r1 = _label(seg_bgr.copy(), "C4 seg.tif crop (MX space)")

        c4r2, n_seg_csv = _dots(seg_bgr.copy(), csv_pts_mx_ov, (0, 220, 220), r)
        c4r2 = _label(c4r2, f"C4 seg + CSV centroids (cyan, {n_seg_csv:,})")

        c4r3 = cv2.addWeighted(mx_bgr_native, 0.7, seg_bgr, 0.3, 0)
        c4r3 = _label(c4r3, "C4 MX DNA + seg overlay")
    else:
        blank = np.zeros((ov_h, ov_w, 3), dtype=np.uint8)
        c4r1 = _label(blank, "C4 seg: no --seg provided")
        c4r2 = blank.copy()
        c4r3 = blank.copy()

    # -----------------------------------------------------------------------
    # Assemble grid
    # -----------------------------------------------------------------------
    panels = [
        [c1r1, c2r1, c3r1, c4r1],
        [c1r2, c2r2, c3r2, c4r2],
        [c1r3, c2r3, c3r3, c4r3],
    ]
    rows = [np.concatenate(row, axis=1) for row in panels]
    grid = np.concatenate(rows, axis=0)

    out_path = out_dir / "alignment_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"\nSaved: {out_path}  ({grid.shape[1]}x{grid.shape[0]} px)")

    names = [
        ["c1r1_he.png", "c2r1_mx_warped.png", "c3r1_blend.png", "c4r1_seg.png"],
        [
            "c1r2_cellvit.png",
            "c2r2_csv_he.png",
            "c3r2_cv_csv_he.png",
            "c4r2_seg_csv.png",
        ],
        ["c1r3_he_cellvit.png", "c2r3_mx_csv.png", "c3r3_zoom.png", "c4r3_mx_seg.png"],
    ]
    for row_p, row_n in zip(panels, names):
        for panel, name in zip(row_p, row_n):
            cv2.imwrite(str(out_dir / name), panel)
    print(f"Saved 12 individual panels to {out_dir}")


def _blank_bgr(ref: np.ndarray) -> np.ndarray:
    return np.zeros_like(ref)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3×3+1col alignment debug grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed", required=True)
    p.add_argument("--he-image", required=True)
    p.add_argument("--multiplex-image", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument(
        "--seg", default=None, help="Full-slide seg TIFF (YX uint16) for column 4."
    )
    p.add_argument("--csv-mpp", type=float, default=0.65)
    p.add_argument(
        "--mx-crop-origin", type=float, nargs=2, default=None, metavar=("OX", "OY")
    )
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--dot-radius", type=int, default=2)
    p.add_argument("--out-dir", default="debug/")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
