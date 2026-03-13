#!/usr/bin/env python3
"""
debug_match_he_mul.py — Interactive H&E / multiplex alignment viewer.

Loads both images at 1/downsample resolution and shows two rows:
  Row 1 (overview):
    1. H&E overview
    2. Selected multiplex channel (DNA channel shown in gray; others in hot colormap)
    3. Alpha-blended overlay
  Row 2 (auto zoom):
    4. H&E zoom at an automatically selected region
    5. DNA-channel zoom (gray)
    6. H&E + DNA overlay zoom
  Row 3 (optional, when seg.tif is provided/found):
    7. H&E zoom (same region as row 2)
    8. Segmentation IDs (same region, colorized by nucleus ID)
    9. H&E + colorized segmentation overlay

Controls:
  - Slider "Channel"  : cycle through all multiplex channels (0..N-1)
  - Slider "Alpha"    : adjust overlay transparency
  - ◀ Prev / Next ▶   : step one channel at a time

Usage:
    python3 -m tools.debug_match_he_mul \
        --he-image data/WD-76845-096.ome.tif \
        --multiplex-image data/WD-76845-097.ome.tif \
        --metadata-csv data/WD-76845-097-metadata.csv \
        [--index-json processed_center/index.json] \
        [--seg-image data/WD-76845-097.ome.seg.tif] \
        [--downsample 64] [--alpha 0.5] [--zoom-size 256] [--zoom-downsample 1]
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib

if "--save-png" in sys.argv:
    matplotlib.use("Agg")

# matplotlib.pyplot must be imported after use() is set — keep below the guard.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
import numpy as np
import tifffile

from utils.channels import load_channel_metadata
from utils.normalize import percentile_to_uint8
from utils.ome import get_ome_mpp, open_zarr_store, read_overview_chw

# pylint: enable=wrong-import-position

PANEL_TITLE_FONTSIZE = 9
SUPTITLE_FONTSIZE = 8


def _resolve_channel_axis(axes: str) -> str | None:
    """Return preferred channel-like axis among C/I/S, if present."""
    ax_up = axes.upper()
    for ax in ("C", "I", "S"):
        if ax in ax_up:
            return ax
    return None


def _read_channel_overview(
    store, axes: str, img_h: int, img_w: int, ds: int, c_idx: int
) -> np.ndarray:
    """Read a single channel at overview resolution, returning (H, W).

    Uses an integer index on the channel-like axis (C/I/S), so zarr reads only
    that one channel.
    """
    ax = axes.upper()
    ch_axis = _resolve_channel_axis(ax)
    if ch_axis is None:
        raise ValueError(
            f"Cannot select channel index {c_idx}: axes '{axes}' has no C/I/S axis."
        )
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds
    sl = []
    for a in ax:
        if a == ch_axis:
            sl.append(c_idx)
        elif a == "Y":
            sl.append(slice(0, h_t, ds))
        elif a == "X":
            sl.append(slice(0, w_t, ds))
        else:
            sl.append(0)
    arr = np.array(store[tuple(sl)])
    active = [a for a in ax if a in ("Y", "X")]
    target = [a for a in ("Y", "X") if a in active]
    if active != target:
        arr = arr.transpose([active.index(a) for a in target])
    return arr


def _apply_inverse_flow(
    image: np.ndarray, flow_dx: np.ndarray, flow_dy: np.ndarray
) -> np.ndarray:
    """Sample image at (x-flow_dx, y-flow_dy) in destination template space."""
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


def _apply_inverse_flow_nearest(
    image: np.ndarray, flow_dx: np.ndarray, flow_dy: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour remap variant for discrete label images."""
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
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _colorize_instance_ids(mask: np.ndarray) -> np.ndarray:
    """Colorize instance IDs (0 background) into RGB without huge LUT allocation."""
    ids = np.asarray(mask)
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    unique_ids = np.unique(ids)
    unique_ids = unique_ids[unique_ids != 0]
    for uid in unique_ids:
        u = int(uid)
        color = np.array(
            [
                (u * 37) % 200 + 30,
                (u * 57) % 200 + 30,
                (u * 97) % 200 + 30,
            ],
            dtype=np.uint8,
        )
        out[ids == uid] = color
    return out


def _is_dna_like_channel(name: str) -> bool:
    """Heuristic: identify DNA-like channels by marker name."""
    key = name.lower()
    return ("dna" in key) or ("hoechst" in key) or ("dapi" in key)


def _auto_zoom_center_from_dna(
    dna_overview: np.ndarray, search_window: int = 64
) -> tuple[int, int]:
    """Find a dense foreground region in DNA overview and return (row, col)."""
    u8 = percentile_to_uint8(dna_overview)
    h, w = u8.shape
    if h == 0 or w == 0:
        return 0, 0
    if int(u8.max()) == 0:
        return h // 2, w // 2

    blur = cv2.GaussianBlur(u8, (0, 0), sigmaX=2.0, sigmaY=2.0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(search_window))
    density = cv2.blur((binary > 0).astype(np.float32), (k, k))
    peak = np.unravel_index(np.argmax(density), density.shape)
    return int(peak[0]), int(peak[1])


def _center_crop_box(
    row: int, col: int, size: int, img_h: int, img_w: int
) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) for a clamped center crop."""
    size = max(16, min(int(size), img_h, img_w))
    y0 = max(0, int(row) - size // 2)
    x0 = max(0, int(col) - size // 2)
    y1 = min(img_h, y0 + size)
    x1 = min(img_w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return y0, y1, x0, x1


def _read_window(
    store,
    axes: str,
    y0: int,
    x0: int,
    height: int,
    width: int,
    step: int = 1,
    channel_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Read a TIFF window and return it in CYX or YX axis order."""
    axes_up = axes.upper()
    ch_axis = _resolve_channel_axis(axes_up)
    sl: list[int | slice] = []
    for ax in axes_up:
        if ch_axis is not None and ax == ch_axis:
            if channel_index is None:
                sl.append(slice(None))
            else:
                sl.append(int(channel_index))
        elif ax == "Y":
            sl.append(slice(y0, y0 + height, step))
        elif ax == "X":
            sl.append(slice(x0, x0 + width, step))
        else:
            sl.append(0)
    arr = np.array(store[tuple(sl)])

    active: list[str] = []
    for ax in axes_up:
        if ch_axis is not None and ax == ch_axis:
            if channel_index is None:
                active.append("C")
        elif ax in ("Y", "X"):
            active.append(ax)
    target = [ax for ax in ("C", "Y", "X") if ax in active]
    if active != target:
        arr = arr.transpose([active.index(ax) for ax in target])
    crop_axes = "".join(target) if target else "YX"
    return arr, crop_axes


def _transform_points_affine(m_full: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points."""
    pts = np.asarray(points_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)
    return (m_full.astype(np.float64) @ hom.T).T


def _map_he_box_to_mx_box(
    m_full: np.ndarray, he_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Map a H&E full-resolution box into an axis-aligned MX full-resolution box."""
    x0, y0, width, height = he_box
    corners = np.array(
        [
            [x0, y0],
            [x0 + width - 1, y0],
            [x0, y0 + height - 1],
            [x0 + width - 1, y0 + height - 1],
        ],
        dtype=np.float64,
    )
    mapped = _transform_points_affine(m_full, corners)
    xs = mapped[:, 0]
    ys = mapped[:, 1]
    mx_x0 = int(np.floor(xs.min()))
    mx_y0 = int(np.floor(ys.min()))
    mx_x1 = int(np.ceil(xs.max())) + 1
    mx_y1 = int(np.ceil(ys.max())) + 1
    return mx_x0, mx_y0, mx_x1 - mx_x0, mx_y1 - mx_y0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interactive H&E / multiplex overlay for alignment debugging."
    )
    parser.add_argument("--he-image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument(
        "--multiplex-image", required=True, help="Path to multiplex OME-TIFF"
    )
    parser.add_argument(
        "--metadata-csv", required=True, help="Path to channel metadata CSV"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=64,
        help="Stride for overview sampling (default 64)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Initial overlay alpha 0–1 (default 0.5)",
    )
    parser.add_argument(
        "--index-json",
        default=None,
        help="Path to processed/index.json; if given, applies warp_matrix "
        "from ECC registration instead of naive resize",
    )
    parser.add_argument(
        "--seg-image",
        default=None,
        help="Optional segmentation TIFF/OME-TIFF aligned to multiplex space "
        "(default: auto-detect <multiplex>.ome.seg.tif if present)",
    )
    parser.add_argument(
        "--save-png",
        default=None,
        metavar="PATH",
        help="Save a static figure (2x3, or 3x3 when seg row is enabled) and exit "
        "without launching the interactive viewer",
    )
    parser.add_argument(
        "--dna-channel",
        type=int,
        default=None,
        help="Optional loaded-channel index treated as DNA (default: auto-detect by name)",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=256,
        help="Auto-zoom crop size in overview pixels (default: 256)",
    )
    parser.add_argument(
        "--zoom-search-window",
        type=int,
        default=64,
        help="Local window size used to pick the auto-zoom center (default: 64)",
    )
    parser.add_argument(
        "--zoom-downsample",
        type=int,
        default=1,
        help="Downsample factor for zoom-row rendering (1 = highest resolution, default: 1)",
    )
    args = parser.parse_args()
    ds = args.downsample

    seg_path: Path | None = None
    if args.seg_image:
        seg_path = Path(args.seg_image)
        if not seg_path.exists():
            raise FileNotFoundError(f"--seg-image not found: {seg_path}")
    else:
        mx_path = Path(args.multiplex_image)
        auto_candidates = []
        if mx_path.name.endswith(".ome.tif"):
            auto_candidates.append(
                mx_path.with_name(mx_path.name.replace(".ome.tif", ".ome.seg.tif"))
            )
        if mx_path.name.endswith(".ome.tiff"):
            auto_candidates.append(
                mx_path.with_name(mx_path.name.replace(".ome.tiff", ".ome.seg.tiff"))
            )
        auto_candidates.append(mx_path.with_suffix(".seg.tif"))
        auto_candidates.append(mx_path.with_suffix(".seg.tiff"))
        for cand in auto_candidates:
            if cand.exists():
                seg_path = cand
                break
    if seg_path is not None:
        print(f"Seg image: {seg_path}")
    else:
        print("Seg image: not provided/found (seg row disabled)")

    # ------------------------------------------------------------------
    # Load H&E overview
    # ------------------------------------------------------------------
    print("Loading H&E overview ...")
    he_tif = tifffile.TiffFile(args.he_image)
    he_s = he_tif.series[0]
    he_ax = he_s.axes.upper()
    he_h = he_s.shape[he_ax.index("Y")]
    he_w = he_s.shape[he_ax.index("X")]
    he_store = open_zarr_store(he_tif)
    he_chw = read_overview_chw(he_store, he_ax, he_h, he_w, ds)
    he_chw = he_chw[:3] if he_chw.shape[0] >= 3 else np.repeat(he_chw[:1], 3, axis=0)
    if he_chw.dtype != np.uint8:
        he_chw = percentile_to_uint8(he_chw)
    he_rgb = np.moveaxis(he_chw.astype(np.uint8), 0, -1)
    h_he, w_he = he_rgb.shape[:2]
    he_mpp, _ = get_ome_mpp(he_tif)
    print(f"  H&E: {he_w}×{he_h} px  mpp={he_mpp}  overview: {w_he}×{h_he}")

    # ------------------------------------------------------------------
    # Resolve which channels to load
    # ------------------------------------------------------------------
    meta = {}
    registered = False
    m_full = None
    m_disp = None
    flow_dx = None
    flow_dy = None
    deformable_mode = False
    if args.index_json:
        with open(args.index_json, encoding="utf-8") as f:
            meta = json.load(f)
        if "warp_matrix" in meta:
            m_full = np.array(meta["warp_matrix"], dtype=np.float64)
            m_disp = m_full.copy()
            m_disp[0, 2] /= ds
            m_disp[1, 2] /= ds
            registered = True
            print(f"  ECC warp matrix loaded from {args.index_json}")
            print(f"  m_disp (overview space):\n{m_disp}")
        deformable_mode = meta.get("registration_mode") == "deformable"
        if deformable_mode:
            reg_dir = Path(args.index_json).resolve().parent / "registration"
            flow_path = reg_dir / "deform_field.npz"
            if flow_path.exists():
                dd = np.load(flow_path)
                flow_dx = dd["flow_dx_ov"].astype(np.float32)
                flow_dy = dd["flow_dy_ov"].astype(np.float32)
                print(f"  Deformable field loaded from {flow_path}")
            else:
                print(
                    f"  WARNING: registration_mode=deformable but {flow_path} not found"
                )

    metadata = load_channel_metadata(args.metadata_csv)
    all_ch_names = dict(metadata.values())
    name_to_idx = {v.lower(): k for k, v in all_ch_names.items()}

    if meta.get("channels"):
        # Only load the channels used during patchify
        selected_names = meta["channels"]
        selected_indices = [
            name_to_idx[n.lower()] for n in selected_names if n.lower() in name_to_idx
        ]
        selected_names = [n for n in selected_names if n.lower() in name_to_idx]
    else:
        selected_indices = []  # empty → load all
        selected_names = []

    # ------------------------------------------------------------------
    # Load multiplex overview — only selected channels (or all if none specified)
    # ------------------------------------------------------------------
    print("Opening multiplex image ...")
    mx_tif = tifffile.TiffFile(args.multiplex_image)
    mx_s = mx_tif.series[0]
    mx_ax = mx_s.axes.upper()
    mx_h = mx_s.shape[mx_ax.index("Y")]
    mx_w = mx_s.shape[mx_ax.index("X")]
    mx_store = open_zarr_store(mx_tif)
    mx_mpp, _ = get_ome_mpp(mx_tif)
    scale = (he_mpp / mx_mpp) if (he_mpp and mx_mpp) else (mx_w / he_w)

    if selected_indices:
        print(f"  Loading {len(selected_indices)} channels: {selected_names}")
        raw_list = [
            _read_channel_overview(mx_store, mx_ax, mx_h, mx_w, ds, c).astype(
                np.float32
            )
            for c in selected_indices
        ]
        mx_chw_h = raw_list[0].shape[0]
        mx_chw_w = raw_list[0].shape[1]
        n_ch = len(selected_indices)

        def ch_label(pos: int) -> str:
            return selected_names[pos]

    else:
        print("  Loading all channels (no channel list in index.json) ...")
        mx_chw_all = read_overview_chw(mx_store, mx_ax, mx_h, mx_w, ds)
        raw_list = [
            mx_chw_all[c].astype(np.float32) for c in range(mx_chw_all.shape[0])
        ]
        mx_chw_h, mx_chw_w = mx_chw_all.shape[1], mx_chw_all.shape[2]
        n_ch = len(raw_list)

        def ch_label(pos: int) -> str:
            return all_ch_names.get(pos, f"ch{pos}")

    channel_raw_index = selected_indices if selected_indices else list(range(n_ch))

    print(
        f"  Multiplex: {mx_w}×{mx_h} px  mpp={mx_mpp}"
        f"  overview: {mx_chw_w}×{mx_chw_h}  channels loaded={n_ch}"
    )
    print(f"  Scale HE→MX: {scale:.4f}")

    if args.dna_channel is not None:
        if args.dna_channel < 0 or args.dna_channel >= n_ch:
            raise ValueError(
                f"--dna-channel must be in [0, {n_ch - 1}] for loaded channels"
            )
        dna_idx = int(args.dna_channel)
    else:
        dna_idx = next(
            (i for i in range(n_ch) if _is_dna_like_channel(ch_label(i))),
            0,
        )
    dna_raw_idx = int(channel_raw_index[dna_idx])
    print(f"  DNA channel: ch{dna_idx} ({ch_label(dna_idx)})")

    if m_full is None:
        # Fallback mapping when no index-json warp is supplied.
        m_full = np.array(
            [
                [mx_w / max(1.0, float(he_w)), 0.0, 0.0],
                [0.0, mx_h / max(1.0, float(he_h)), 0.0],
            ],
            dtype=np.float64,
        )

    # Warp or resize each channel into H&E overview space
    action = "Warping" if registered else "Resizing"
    action2 = " + deformable remap" if (registered and flow_dx is not None) else ""
    print(f"  {action}{action2} {n_ch} channels to {w_he}×{h_he} ...")
    mx_full = np.zeros((n_ch, h_he, w_he), dtype=np.float32)
    flow_dx_disp = flow_dy_disp = None
    if flow_dx is not None:
        fh, fw = flow_dx.shape
        if (fh, fw) != (h_he, w_he):
            sx = w_he / float(max(1, fw))
            sy = h_he / float(max(1, fh))
            flow_dx_disp = (
                cv2.resize(flow_dx, (w_he, h_he), interpolation=cv2.INTER_LINEAR) * sx
            )
            flow_dy_disp = (
                cv2.resize(flow_dy, (w_he, h_he), interpolation=cv2.INTER_LINEAR) * sy
            )
        else:
            flow_dx_disp = flow_dx
            flow_dy_disp = flow_dy
    for i, ch in enumerate(raw_list):
        if registered:
            warped = cv2.warpAffine(
                ch,
                m_disp,
                (w_he, h_he),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
            if flow_dx_disp is not None:
                warped = _apply_inverse_flow(warped, flow_dx_disp, flow_dy_disp)
            mx_full[i] = warped
        else:
            mx_full[i] = cv2.resize(ch, (w_he, h_he), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Colourmap helper
    # ------------------------------------------------------------------
    cmap_hot = matplotlib.colormaps["hot"]

    def channel_to_rgb(c_idx: int, force_gray: bool = False) -> np.ndarray:
        """Render one multiplex channel as RGB (DNA channel in gray, others in hot)."""
        ch = mx_full[c_idx]
        u8 = percentile_to_uint8(ch)
        if force_gray or c_idx == dna_idx:
            return np.stack([u8, u8, u8], axis=-1).astype(np.uint8)
        rgba = cmap_hot(u8.astype(np.float32) / 255.0)  # (H, W, 4)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    def blend(he: np.ndarray, mx_rgb: np.ndarray, a: float) -> np.ndarray:
        return (
            (he.astype(np.float32) * (1 - a) + mx_rgb.astype(np.float32) * a)
            .clip(0, 255)
            .astype(np.uint8)
        )

    def mx_title(c_idx: int) -> str:
        cmap_name = "gray" if c_idx == dna_idx else "hot"
        return f"MX ch{c_idx}: {ch_label(c_idx)} ({cmap_name})"

    zoom_row, zoom_col = _auto_zoom_center_from_dna(
        mx_full[dna_idx], search_window=args.zoom_search_window
    )
    y0z, y1z, x0z, x1z = _center_crop_box(
        zoom_row, zoom_col, args.zoom_size, h_he, w_he
    )
    zoom_h = y1z - y0z
    zoom_w = x1z - x0z
    print(
        "  Auto-zoom region "
        f"(overview): center=(col={zoom_col}, row={zoom_row}) "
        f"box=(x={x0z}, y={y0z}, w={zoom_w}, h={zoom_h})"
    )

    zoom_step = max(1, int(args.zoom_downsample))
    x0_full = int(x0z * ds)
    y0_full = int(y0z * ds)
    x1_full = int(min(he_w, x1z * ds))
    y1_full = int(min(he_h, y1z * ds))
    he_box_full = (
        x0_full,
        y0_full,
        max(1, x1_full - x0_full),
        max(1, y1_full - y0_full),
    )

    he_crop_raw, he_crop_axes = _read_window(
        he_store,
        he_ax,
        y0_full,
        x0_full,
        he_box_full[3],
        he_box_full[2],
        step=zoom_step,
        channel_index=None,
    )
    if he_crop_axes == "YX":
        he_chw_zoom = np.repeat(he_crop_raw[np.newaxis, ...], 3, axis=0)
    else:
        he_chw_zoom = (
            he_crop_raw[:3]
            if he_crop_raw.shape[0] >= 3
            else np.repeat(he_crop_raw[:1], 3, axis=0)
        )
    if he_chw_zoom.dtype != np.uint8:
        he_chw_zoom = np.stack([percentile_to_uint8(ch) for ch in he_chw_zoom], axis=0)
    he_zoom = np.moveaxis(he_chw_zoom.astype(np.uint8), 0, -1)

    mx_box = _map_he_box_to_mx_box(m_full, he_box_full)
    mx_x0 = max(0, mx_box[0])
    mx_y0 = max(0, mx_box[1])
    mx_x1 = min(mx_w, mx_box[0] + mx_box[2])
    mx_y1 = min(mx_h, mx_box[1] + mx_box[3])
    mx_w_read = max(0, mx_x1 - mx_x0)
    mx_h_read = max(0, mx_y1 - mx_y0)
    if mx_w_read > 0 and mx_h_read > 0:
        mx_dna_crop, _ = _read_window(
            mx_store,
            mx_ax,
            mx_y0,
            mx_x0,
            mx_h_read,
            mx_w_read,
            step=1,
            channel_index=dna_raw_idx,
        )
        mx_dna_crop = mx_dna_crop.astype(np.float32)
    else:
        mx_dna_crop = np.zeros((1, 1), dtype=np.float32)

    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    tx_local = a * x0_full + b * y0_full + tx - mx_x0
    ty_local = c * x0_full + d * y0_full + ty - mx_y0
    m_local = np.array(
        [
            [a * zoom_step, b * zoom_step, tx_local],
            [c * zoom_step, d * zoom_step, ty_local],
        ],
        dtype=np.float32,
    )

    zh, zw = he_zoom.shape[:2]
    dna_zoom = cv2.warpAffine(
        mx_dna_crop,
        m_local,
        (zw, zh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if flow_dx is not None and flow_dy is not None:
        fh, fw = flow_dx.shape
        yy, xx = np.meshgrid(
            np.arange(zh, dtype=np.float32),
            np.arange(zw, dtype=np.float32),
            indexing="ij",
        )
        x_full_grid = float(x0_full) + xx * float(zoom_step)
        y_full_grid = float(y0_full) + yy * float(zoom_step)
        map_fx = x_full_grid * (fw / max(1.0, float(he_w)))
        map_fy = y_full_grid * (fh / max(1.0, float(he_h)))

        flow_dx_patch = cv2.remap(
            flow_dx.astype(np.float32),
            map_fx,
            map_fy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        flow_dy_patch = cv2.remap(
            flow_dy.astype(np.float32),
            map_fx,
            map_fy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sx_disp = he_w / max(1.0, float(fw * zoom_step))
        sy_disp = he_h / max(1.0, float(fh * zoom_step))
        dna_zoom = _apply_inverse_flow(
            dna_zoom,
            flow_dx_patch * float(sx_disp),
            flow_dy_patch * float(sy_disp),
        )

    dna_zoom_u8 = percentile_to_uint8(dna_zoom.astype(np.float32))
    dna_zoom_rgb = np.stack([dna_zoom_u8, dna_zoom_u8, dna_zoom_u8], axis=-1)
    print(
        "  Zoom row source "
        f"(full-res): x={x0_full}, y={y0_full}, w={he_box_full[2]}, h={he_box_full[3]} "
        f"| zoom-downsample={zoom_step} -> display {zw}x{zh}"
    )

    seg_zoom_rgb = None
    seg_overlay_rgb = None
    if seg_path is not None:
        with tifffile.TiffFile(str(seg_path)) as seg_tif:
            seg_s = seg_tif.series[0]
            seg_ax = seg_s.axes.upper()
            seg_h = seg_s.shape[seg_ax.index("Y")]
            seg_w = seg_s.shape[seg_ax.index("X")]
            seg_store = open_zarr_store(seg_tif)

            sx = seg_w / max(1.0, float(mx_w))
            sy = seg_h / max(1.0, float(mx_h))
            m_full_seg = m_full.astype(np.float64).copy()
            m_full_seg[0, :] *= sx
            m_full_seg[1, :] *= sy

            seg_box = _map_he_box_to_mx_box(m_full_seg, he_box_full)
            seg_x0 = max(0, seg_box[0])
            seg_y0 = max(0, seg_box[1])
            seg_x1 = min(seg_w, seg_box[0] + seg_box[2])
            seg_y1 = min(seg_h, seg_box[1] + seg_box[3])
            seg_w_read = max(0, seg_x1 - seg_x0)
            seg_h_read = max(0, seg_y1 - seg_y0)

            if seg_w_read > 0 and seg_h_read > 0:
                seg_crop, seg_axes = _read_window(
                    seg_store,
                    seg_ax,
                    seg_y0,
                    seg_x0,
                    seg_h_read,
                    seg_w_read,
                    step=1,
                    channel_index=0 if _resolve_channel_axis(seg_ax) is not None else None,
                )
                if seg_axes == "CYX":
                    seg_crop = seg_crop[0]
                seg_crop = np.asarray(seg_crop)
            else:
                seg_crop = np.zeros((1, 1), dtype=np.float32)

            a_s, b_s, tx_s = map(float, m_full_seg[0])
            c_s, d_s, ty_s = map(float, m_full_seg[1])
            tx_local_s = a_s * x0_full + b_s * y0_full + tx_s - seg_x0
            ty_local_s = c_s * x0_full + d_s * y0_full + ty_s - seg_y0
            m_local_seg = np.array(
                [
                    [a_s * zoom_step, b_s * zoom_step, tx_local_s],
                    [c_s * zoom_step, d_s * zoom_step, ty_local_s],
                ],
                dtype=np.float32,
            )
            seg_zoom = cv2.warpAffine(
                seg_crop.astype(np.float32),
                m_local_seg,
                (zw, zh),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            if flow_dx is not None and flow_dy is not None:
                seg_zoom = _apply_inverse_flow_nearest(
                    seg_zoom,
                    flow_dx_patch * float(sx_disp),
                    flow_dy_patch * float(sy_disp),
                )

            seg_zoom_ids = np.rint(seg_zoom).astype(np.int64, copy=False)
            seg_zoom_rgb = _colorize_instance_ids(seg_zoom_ids)
            seg_overlay_rgb = blend(he_zoom, seg_zoom_rgb, args.alpha)
            n_seg_ids = int(np.unique(seg_zoom_ids[seg_zoom_ids > 0]).shape[0])
            print(
                f"  Seg zoom source: {seg_path} | region x={seg_x0} y={seg_y0} "
                f"w={seg_w_read} h={seg_h_read} | unique IDs={n_seg_ids}"
            )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    has_seg_row = seg_zoom_rgb is not None and seg_overlay_rgb is not None
    n_rows = 3 if has_seg_row else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 2:
        (ax_he, ax_mx, ax_ov), (ax_he_z, ax_mx_z, ax_ov_z) = axes
        ax_he_seg = ax_seg = ax_ov_seg = None
    else:
        (ax_he, ax_mx, ax_ov), (ax_he_z, ax_mx_z, ax_ov_z), (
            ax_he_seg,
            ax_seg,
            ax_ov_seg,
        ) = axes
    fig.patch.set_facecolor("#111")
    plt.subplots_adjust(
        left=0.03, right=0.97, top=0.92, bottom=0.18, wspace=0.05, hspace=0.08
    )

    # Initial render
    init_ch = 0
    init_mx = channel_to_rgb(init_ch)
    init_ov = blend(he_rgb, init_mx, args.alpha)
    init_zoom_ov = blend(he_zoom, dna_zoom_rgb, args.alpha)

    ax_he.imshow(he_rgb)
    ax_he.set_title(
        "H&E overview", color="white", fontsize=PANEL_TITLE_FONTSIZE, pad=6
    )
    ax_he.set_xticks([])
    ax_he.set_yticks([])
    ax_he.set_facecolor("black")

    im_mx = ax_mx.imshow(init_mx)
    ax_mx.set_title(mx_title(init_ch), color="white", fontsize=PANEL_TITLE_FONTSIZE, pad=6)
    ax_mx.set_xticks([])
    ax_mx.set_yticks([])
    ax_mx.set_facecolor("black")
    ax_mx.add_patch(
        mpatches.Rectangle(
            (x0z, y0z),
            zoom_w,
            zoom_h,
            fill=False,
            edgecolor="#00ffff",
            linewidth=1.5,
        )
    )

    im_ov = ax_ov.imshow(init_ov)
    ax_ov.set_title(
        f"Overlay  α={args.alpha:.2f}  ch{init_ch}: {ch_label(init_ch)}",
        color="white",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=6,
    )
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    ax_ov.set_facecolor("black")
    ax_ov.add_patch(
        mpatches.Rectangle(
            (x0z, y0z),
            zoom_w,
            zoom_h,
            fill=False,
            edgecolor="#00ffff",
            linewidth=1.5,
        )
    )

    ax_he_z.imshow(he_zoom)
    ax_he_z.set_title(
        f"H&E auto-zoom (high-res ds={zoom_step})",
        color="white",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=6,
    )
    ax_he_z.set_xticks([])
    ax_he_z.set_yticks([])
    ax_he_z.set_facecolor("black")

    ax_mx_z.imshow(dna_zoom_rgb)
    ax_mx_z.set_title(
        f"DNA auto-zoom ch{dna_idx}: {ch_label(dna_idx)} (gray)",
        color="white",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=6,
    )
    ax_mx_z.set_xticks([])
    ax_mx_z.set_yticks([])
    ax_mx_z.set_facecolor("black")

    im_ov_z = ax_ov_z.imshow(init_zoom_ov)
    ax_ov_z.set_title(
        f"Zoom overlay (H&E + DNA, high-res)  α={args.alpha:.2f}",
        color="white",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=6,
    )
    ax_ov_z.set_xticks([])
    ax_ov_z.set_yticks([])
    ax_ov_z.set_facecolor("black")

    im_ov_seg = None
    if (
        has_seg_row
        and ax_he_seg is not None
        and ax_seg is not None
        and ax_ov_seg is not None
    ):
        ax_he_seg.imshow(he_zoom)
        ax_he_seg.set_title(
            f"H&E auto-zoom (same region, ds={zoom_step})",
            color="white",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=6,
        )
        ax_he_seg.set_xticks([])
        ax_he_seg.set_yticks([])
        ax_he_seg.set_facecolor("black")

        ax_seg.imshow(seg_zoom_rgb)
        ax_seg.set_title(
            "Seg IDs auto-zoom (colorized)",
            color="white",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=6,
        )
        ax_seg.set_xticks([])
        ax_seg.set_yticks([])
        ax_seg.set_facecolor("black")

        im_ov_seg = ax_ov_seg.imshow(seg_overlay_rgb)
        ax_ov_seg.set_title(
            f"Seg overlap (H&E + seg IDs)  α={args.alpha:.2f}",
            color="white",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=6,
        )
        ax_ov_seg.set_xticks([])
        ax_ov_seg.set_yticks([])
        ax_ov_seg.set_facecolor("black")

    ax_he.add_patch(
        mpatches.Rectangle(
            (x0z, y0z),
            zoom_w,
            zoom_h,
            fill=False,
            edgecolor="#00ffff",
            linewidth=1.5,
        )
    )

    if registered and flow_dx is not None:
        reg_label = "ECC + deformable"
    elif registered:
        reg_label = "ECC-registered"
    else:
        reg_label = "naive resize (no registration)"
    fig.suptitle(
        f"H&E vs Multiplex  |  downsample={ds}x  |  {n_ch} channels  "
        f"|  H&E {w_he}x{h_he}  MX {mx_chw_w}x{mx_chw_h}  "
        f"|  DNA ch{dna_idx}: {ch_label(dna_idx)}  "
        f"|  zoom x={x0z} y={y0z} w={zoom_w} h={zoom_h}  "
        f"|  zoom-ds={zoom_step}  "
        f"|  seg-row={'on' if has_seg_row else 'off'}  "
        f"|  {reg_label}",
        color="white",
        fontsize=SUPTITLE_FONTSIZE,
        y=0.97,
    )

    # ------------------------------------------------------------------
    # Static save mode — write PNG and exit without GUI
    # ------------------------------------------------------------------
    if args.save_png:
        fig.savefig(args.save_png, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved static overlay to {args.save_png}")
        he_tif.close()
        mx_tif.close()
        plt.close(fig)
        return

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------
    # Channel slider
    ax_sl_ch = plt.axes([0.13, 0.09, 0.58, 0.04], facecolor="#2a2a2a")
    slider_ch = mwidgets.Slider(
        ax_sl_ch,
        "Channel",
        0,
        n_ch - 1,
        valinit=init_ch,
        valstep=1,
        color="#ff8800",
    )
    slider_ch.label.set_color("white")
    slider_ch.valtext.set_color("white")

    # Alpha slider
    ax_sl_al = plt.axes([0.13, 0.04, 0.58, 0.04], facecolor="#2a2a2a")
    slider_alpha = mwidgets.Slider(
        ax_sl_al,
        "Alpha",
        0.0,
        1.0,
        valinit=args.alpha,
        color="#4499ff",
    )
    slider_alpha.label.set_color("white")
    slider_alpha.valtext.set_color("white")

    # Prev / Next buttons
    ax_prev = plt.axes([0.74, 0.075, 0.06, 0.055])
    ax_next = plt.axes([0.81, 0.075, 0.06, 0.055])
    btn_prev = mwidgets.Button(ax_prev, "◀ Prev", color="#333", hovercolor="#555")
    btn_next = mwidgets.Button(ax_next, "Next ▶", color="#333", hovercolor="#555")
    btn_prev.label.set_color("white")
    btn_prev.label.set_fontsize(9)
    btn_next.label.set_color("white")
    btn_next.label.set_fontsize(9)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _redraw(c: int, a: float) -> None:
        mx_rgb = channel_to_rgb(c)
        ov = blend(he_rgb, mx_rgb, a)
        ov_zoom = blend(he_zoom, dna_zoom_rgb, a)
        ov_seg = blend(he_zoom, seg_zoom_rgb, a) if has_seg_row else None
        im_mx.set_data(mx_rgb)
        im_ov.set_data(ov)
        im_ov_z.set_data(ov_zoom)
        if has_seg_row and im_ov_seg is not None and ov_seg is not None:
            im_ov_seg.set_data(ov_seg)
        ax_mx.set_title(mx_title(c), color="white", fontsize=PANEL_TITLE_FONTSIZE, pad=6)
        ax_ov.set_title(
            f"Overlay  α={a:.2f}  ch{c}: {ch_label(c)}",
            color="white",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=6,
        )
        ax_ov_z.set_title(
            f"Zoom overlay (H&E + DNA, high-res)  α={a:.2f}",
            color="white",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=6,
        )
        if has_seg_row and ax_ov_seg is not None:
            ax_ov_seg.set_title(
                f"Seg overlap (H&E + seg IDs)  α={a:.2f}",
                color="white",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=6,
            )
        fig.canvas.draw_idle()

    def on_slider(_val):
        _redraw(int(round(slider_ch.val)), slider_alpha.val)

    def on_prev(_):
        slider_ch.set_val(max(0, int(slider_ch.val) - 1))

    def on_next(_):
        slider_ch.set_val(min(n_ch - 1, int(slider_ch.val) + 1))

    slider_ch.on_changed(on_slider)
    slider_alpha.on_changed(on_slider)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    print("Ready — use the slider or ◀/▶ buttons to browse channels.")
    plt.show()

    he_tif.close()
    mx_tif.close()


if __name__ == "__main__":
    main()
