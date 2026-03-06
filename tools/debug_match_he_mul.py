#!/usr/bin/env python3
"""
debug_match_he_mul.py — Interactive H&E / multiplex alignment viewer.

Loads both images at 1/downsample resolution, shows three panels:
  1. H&E overview
  2. Selected multiplex channel (hot colormap)
  3. Alpha-blended overlay

Controls:
  - Slider "Channel"  : cycle through all multiplex channels (0..N-1)
  - Slider "Alpha"    : adjust overlay transparency
  - ◀ Prev / Next ▶   : step one channel at a time

Usage:
    python debug_match_he_mul.py \\
        --he-image data/CRC02-HE.ome.tif \\
        --multiplex-image data/CRC02.ome.tif \\
        --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \\
        [--downsample 64] [--alpha 0.5]
"""

import argparse
import json
import sys

import cv2
import matplotlib

if "--save-png" in sys.argv:
    matplotlib.use("Agg")

# matplotlib.pyplot must be imported after use() is set — keep below the guard.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import tifffile

from utils.channels import load_channel_metadata
from utils.normalize import percentile_to_uint8
from utils.ome import get_ome_mpp, open_zarr_store, read_overview_chw
# pylint: enable=wrong-import-position


def _read_channel_overview(
    store, axes: str, img_h: int, img_w: int, ds: int, c_idx: int
) -> np.ndarray:
    """Read a single channel at overview resolution, returning (H, W).

    Uses an integer index on the C axis so zarr reads only that one channel.
    """
    ax = axes.upper()
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds
    sl = []
    for a in ax:
        if a == "C":
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
        "--save-png",
        default=None,
        metavar="PATH",
        help="Save a static 3-panel PNG (H&E | MX ch0 | overlay) and exit "
        "without launching the interactive viewer",
    )
    args = parser.parse_args()
    ds = args.downsample

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
    m_disp = None
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
    scale = (he_mpp / mx_mpp) if (he_mpp and mx_mpp) else (he_w / mx_w)

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

    print(
        f"  Multiplex: {mx_w}×{mx_h} px  mpp={mx_mpp}"
        f"  overview: {mx_chw_w}×{mx_chw_h}  channels loaded={n_ch}"
    )
    print(f"  Scale HE→MX: {scale:.4f}")

    # Warp or resize each channel into H&E overview space
    action = "Warping" if registered else "Resizing"
    print(f"  {action} {n_ch} channels to {w_he}×{h_he} ...")
    mx_full = np.zeros((n_ch, h_he, w_he), dtype=np.float32)
    for i, ch in enumerate(raw_list):
        if registered:
            mx_full[i] = cv2.warpAffine(
                ch,
                m_disp,
                (w_he, h_he),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        else:
            mx_full[i] = cv2.resize(ch, (w_he, h_he), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Colourmap helper
    # ------------------------------------------------------------------
    cmap_hot = matplotlib.colormaps["hot"]

    def channel_to_rgb(c_idx: int) -> np.ndarray:
        """Normalise channel c_idx and apply 'hot' colormap -> uint8 (H, W, 3)."""
        ch = mx_full[c_idx]
        u8 = percentile_to_uint8(ch)
        rgba = cmap_hot(u8.astype(np.float32) / 255.0)  # (H, W, 4)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    def blend(he: np.ndarray, mx_rgb: np.ndarray, a: float) -> np.ndarray:
        return (
            (he.astype(np.float32) * (1 - a) + mx_rgb.astype(np.float32) * a)
            .clip(0, 255)
            .astype(np.uint8)
        )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, (ax_he, ax_mx, ax_ov) = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor("#111")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.91, bottom=0.18, wspace=0.05)

    # Initial render
    init_ch = 0
    init_mx = channel_to_rgb(init_ch)
    init_ov = blend(he_rgb, init_mx, args.alpha)

    ax_he.imshow(he_rgb)
    ax_he.set_title("H&E overview", color="white", fontsize=11, pad=6)
    ax_he.set_xticks([])
    ax_he.set_yticks([])
    ax_he.set_facecolor("black")

    im_mx = ax_mx.imshow(init_mx)
    ax_mx.set_title(
        f"MX ch{init_ch}: {ch_label(init_ch)}", color="white", fontsize=11, pad=6
    )
    ax_mx.set_xticks([])
    ax_mx.set_yticks([])
    ax_mx.set_facecolor("black")

    im_ov = ax_ov.imshow(init_ov)
    ax_ov.set_title(
        f"Overlay  α={args.alpha:.2f}  ch{init_ch}: {ch_label(init_ch)}",
        color="white",
        fontsize=11,
        pad=6,
    )
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    ax_ov.set_facecolor("black")

    reg_label = "ECC-registered" if registered else "naive resize (no registration)"
    fig.suptitle(
        f"H&E vs Multiplex  |  downsample={ds}x  |  {n_ch} channels  "
        f"|  H&E {w_he}x{h_he}  MX {mx_chw_w}x{mx_chw_h}  |  {reg_label}",
        color="white",
        fontsize=10,
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
        im_mx.set_data(mx_rgb)
        im_ov.set_data(ov)
        ax_mx.set_title(f"MX ch{c}: {ch_label(c)}", color="white", fontsize=11, pad=6)
        ax_ov.set_title(
            f"Overlay  α={a:.2f}  ch{c}: {ch_label(c)}",
            color="white",
            fontsize=11,
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
