#!/usr/bin/env python3
"""
visualize_pipeline.py -- Summary figure for the histopathology patch pipeline.

Shows seven panels for a single selected patch:
  1. Original location  -- thumbnail of full H&E with red rectangle
  2. H&E patch          -- 256x256 RGB PNG
  3. Multiplex (CD31)   -- channel 0 of .npy with 'hot' colormap
  4. Cell segmentation  -- H&E with CellViT contours overlaid
  5. Cell mask          -- colorized label IDs from masks/*.npy
  6. Cell type          -- cell_types RGBA composited on black
  7. Cell state         -- cell_states RGBA composited on black

CLI:
  python visualize_pipeline.py --processed processed/ [--patch 58624_4096]
                               [--he-image data/CRC02-HE.ome.tif]
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from utils.normalize import percentile_norm, percentile_to_uint8

# ---------------------------------------------------------------------------
# Color legends (must match assign_cells.py CELL_TYPE_COLORS / CELL_STATE_COLORS)
# ---------------------------------------------------------------------------
TYPE_LEGEND = [
    ("tumor", (220, 50, 50)),
    ("immune", (50, 100, 220)),
    ("stromal", (50, 180, 50)),
    ("other", (150, 150, 150)),
]
STATE_LEGEND = [
    ("proliferating", (0, 255, 0)),
    ("emt", (255, 165, 0)),
    ("apoptotic", (139, 0, 139)),
    ("quiescent", (100, 149, 237)),
    ("healthy", (144, 238, 144)),
    ("other", (80, 80, 80)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gray_placeholder(text: str = "", size: int = 256) -> np.ndarray:
    """Return a (size, size, 3) uint8 gray image with centered white text."""
    img = np.full((size, size, 3), 60, dtype=np.uint8)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = max(0, (size - tw) // 2)
        ty = max(th, (size + th) // 2)
        cv2.putText(
            img,
            text,
            (tx, ty),
            font,
            font_scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
    return img


def _composite_rgba_on_black(rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA uint8 image onto a black background -> RGB uint8."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    out = (alpha * rgb).clip(0, 255).astype(np.uint8)
    return out


def _apply_colormap(norm: np.ndarray, cmap_name: str = "hot") -> np.ndarray:
    """Apply a matplotlib colormap to a [0,1] 2-D array -> uint8 (H, W, 3)."""
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm)  # (H, W, 4) float64
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def build_original_location(
    he_image_path: str | None,
    x0: int,
    y0: int,
    patch_size: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Return a (256, 256, 3) uint8 thumbnail with a red rectangle marking the patch."""

    if he_image_path is not None and Path(he_image_path).exists():
        try:
            tif = tifffile.TiffFile(he_image_path)
            series = tif.series[0]
            # Try level 3 then 4 for a small thumbnail
            chosen_level = None
            for level_idx in (3, 4, 2, 1):
                if level_idx < len(series.levels):
                    chosen_level = level_idx
                    break
            if chosen_level is None:
                chosen_level = len(series.levels) - 1

            level = series.levels[chosen_level]
            axes = level.axes.upper()

            # Read the thumbnail
            store = level.aszarr()
            import zarr

            raw = zarr.open(store, mode="r")
            arr = raw if isinstance(raw, zarr.Array) else raw["0"]
            arr = np.array(arr)

            # Move channel axis to last if needed (CYX -> YXC)
            if arr.ndim == 3:
                c_pos = axes.index("C") if "C" in axes else -1
                y_pos = axes.index("Y") if "Y" in axes else -1
                if c_pos != -1 and y_pos != -1 and c_pos < y_pos:
                    arr = np.moveaxis(arr, 0, -1)

            # Ensure RGB
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] > 3:
                arr = arr[..., :3]

            # Normalize dtype
            if arr.dtype != np.uint8:
                arr = percentile_to_uint8(arr)

            # Resize to 256x256 first, then draw rectangle in output space
            thumbnail = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)

            # Map patch coords to 256x256 space and enforce minimum 6px visibility
            rx = int(x0 / img_w * 256)
            ry = int(y0 / img_h * 256)
            rw = max(6, int(patch_size / img_w * 256))
            rh = max(6, int(patch_size / img_h * 256))
            cv2.rectangle(thumbnail, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            return thumbnail

        except Exception as exc:
            print(f"  Warning: could not read he-image for thumbnail: {exc}")

    # Placeholder: 256x256 gray with proportional red square
    thumb = np.full((256, 256, 3), 60, dtype=np.uint8)
    if img_w > 0 and img_h > 0:
        rx = int(x0 / img_w * 256)
        ry = int(y0 / img_h * 256)
        rw = max(1, int(patch_size / img_w * 256))
        rh = max(1, int(patch_size / img_h * 256))
        cv2.rectangle(thumb, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
    return thumb


def build_he_panel(he_png_path: Path) -> np.ndarray:
    if he_png_path.exists():
        img = Image.open(he_png_path).convert("RGB")
        return np.array(img)
    return _gray_placeholder("no H&E")


def build_multiplex_panel(
    npy_path: Path, channel_idx: int = 0, use_max_proj: bool = False
) -> np.ndarray:
    if npy_path.exists():
        arr = np.load(npy_path)  # (C, H, W) uint16
        if arr.ndim == 3:
            if use_max_proj:
                ch = arr.max(axis=0).astype(np.float32)
            elif arr.shape[0] > channel_idx:
                ch = arr[channel_idx].astype(np.float32)
            else:
                ch = arr[0].astype(np.float32)
        else:
            ch = arr.astype(np.float32)
        norm = percentile_norm(ch)
        return _apply_colormap(norm, "hot")
    return _gray_placeholder("no multiplex")


def build_cellseg_panel(
    he_png_path: Path, cellvit_json_path: Path
) -> tuple[np.ndarray, bool]:
    """Return (image, had_segmentation)."""
    base = build_he_panel(he_png_path)

    if not cellvit_json_path.exists():
        return base, False

    try:
        with open(cellvit_json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not load cellvit JSON: {exc}")
        return base, False

    overlay = base.copy()

    # CellViT JSON format: may be a list of cells or a dict with a 'cells' key
    cells = data
    if isinstance(data, dict):
        # Try common keys
        for key in ("cells", "nuclei", "detections", "instances"):
            if key in data:
                cells = data[key]
                break
        else:
            # Maybe it has integer keys (instance map format)
            cells = list(data.values()) if data else []

    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            # Look for contour/polygon under common key names
            contour = None
            for key in (
                "contour",
                "contours",
                "polygon",
                "boundary",
                "coords",
                "points",
            ):
                if key in cell:
                    contour = cell[key]
                    break
            if contour is None:
                continue
            # Convert to numpy int32 array shaped (N, 1, 2) for cv2.polylines
            pts = np.array(contour, dtype=np.int32)
            if pts.ndim == 1 and len(pts) == 2:
                # Single point -- skip
                continue
            if pts.ndim == 2 and pts.shape[1] == 2:
                pts = pts.reshape((-1, 1, 2))
            elif pts.ndim == 3 and pts.shape[2] == 2:
                pass  # already (N, 1, 2) or similar
            else:
                continue
            cv2.polylines(
                overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=1
            )

    return overlay, True


def build_overlay_panel(png_path: Path, label: str) -> np.ndarray:
    if png_path.exists():
        try:
            img = Image.open(png_path).convert("RGBA")
            rgba = np.array(img)
            return _composite_rgba_on_black(rgba)
        except Exception as exc:
            print(f"  Warning: could not load {label} PNG: {exc}")
    return _gray_placeholder(f"no {label}")


def _extract_2d_mask(arr: np.ndarray) -> np.ndarray | None:
    """Best-effort conversion of a loaded mask array to 2-D."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[:, :, 0]
        if arr.shape[0] <= 4:
            return arr[0]
        if arr.shape[-1] <= 4:
            return arr[:, :, 0]
    return None


def _colorize_label_mask(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Map integer label IDs to deterministic RGB colors (0 -> black)."""
    label_ids, inverse = np.unique(mask, return_inverse=True)
    colors = np.zeros((label_ids.shape[0], 3), dtype=np.uint8)
    non_bg = label_ids != 0
    if np.any(non_bg):
        rng = np.random.default_rng(seed)
        colors[non_bg] = rng.integers(
            30, 256, size=(int(non_bg.sum()), 3), dtype=np.uint8
        )
    return colors[inverse].reshape(mask.shape + (3,))


def build_cell_mask_panel(mask_npy_path: Path) -> tuple[np.ndarray, bool]:
    """Return (colorized cell mask RGB, had_mask_file)."""
    if not mask_npy_path.exists():
        return _gray_placeholder("no cell mask"), False

    try:
        arr = np.load(mask_npy_path)
    except Exception as exc:
        print(f"  Warning: could not load cell mask .npy: {exc}")
        return _gray_placeholder("bad cell mask"), False

    mask = _extract_2d_mask(arr)
    if mask is None:
        print(f"  Warning: unsupported cell mask shape {arr.shape} in {mask_npy_path}")
        return _gray_placeholder("bad cell mask"), False

    if np.issubdtype(mask.dtype, np.floating):
        mask_u32 = np.round(mask).astype(np.uint32)
    else:
        mask_u32 = mask.astype(np.uint32, copy=False)

    return _colorize_label_mask(mask_u32), True


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def _show_panel(
    ax,
    img: np.ndarray,
    title: str,
    cmap=None,
    legend: list[tuple[str, tuple]] | None = None,
):
    """Display an image on an axes with black background and white title.
    legend: list of (label, (R,G,B)) tuples drawn as colored patches inside the axes."""
    ax.set_facecolor("black")
    if cmap is not None:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_title(title, color="white", fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if legend:
        handles = [
            mpatches.Patch(
                facecolor=tuple(c / 255 for c in rgb), label=label, edgecolor="none"
            )
            for label, rgb in legend
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=6,
            framealpha=0.55,
            facecolor="#111111",
            labelcolor="white",
            edgecolor="none",
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=1.0,
        )


def make_summary_figure(
    processed_dir: Path,
    patch_key: str,
    he_image_path: str | None,
    mx_channel: int = 0,
    mx_max_proj: bool = False,
) -> Path:
    """Build and save the 7-panel summary figure for the given patch.

    Returns the path to the saved PNG.
    """
    # Load index.json
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found at {index_path}")

    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)

    patches = index_data.get("patches", [])
    img_w = index_data.get("img_w", 0)
    img_h = index_data.get("img_h", 0)
    patch_size = index_data.get("patch_size", 256)

    # Find the selected patch
    x0_str, y0_str = patch_key.split("_")
    x0_sel, y0_sel = int(x0_str), int(y0_str)
    patch_meta = None
    for p in patches:
        if p["x0"] == x0_sel and p["y0"] == y0_sel:
            patch_meta = p
            break
    if patch_meta is None:
        raise ValueError(f"Patch {patch_key} not found in index.json")

    x0 = patch_meta["x0"]
    y0 = patch_meta["y0"]

    # Channel name for multiplex panel title
    channels = index_data.get("channels", [])
    if mx_max_proj:
        mx_ch_name = "max-proj"
    elif mx_channel < len(channels):
        mx_ch_name = channels[mx_channel]
    else:
        mx_ch_name = f"ch{mx_channel}"

    # File paths
    he_png = processed_dir / "he" / f"{patch_key}.png"
    mx_npy = processed_dir / "multiplex" / f"{patch_key}.npy"
    cellvit_json = processed_dir / "cellvit" / f"{patch_key}.json"
    mask_npy = processed_dir / "masks" / f"{patch_key}.npy"
    cell_types_png = processed_dir / "cell_types" / f"{patch_key}.png"
    cell_states_png = processed_dir / "cell_states" / f"{patch_key}.png"
    summary_json = processed_dir / "cell_summary.json"

    print(f"Building figure for patch {patch_key} (x0={x0}, y0={y0}) ...")

    # Per-patch summary from cell_summary.json
    patch_summary_text = ""
    if summary_json.exists():
        with open(summary_json, encoding="utf-8") as f:
            summary = json.load(f)
        ps = summary.get("per_patch", {}).get(patch_key, {})
        if ps:
            n = ps.get("n_cells", 0)
            type_counts = ps.get("cell_types", {})
            state_counts = ps.get("cell_states", {})
            type_str = "  ".join(
                f"{k}:{v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
            )
            state_str = "  ".join(
                f"{k}:{v}" for k, v in sorted(state_counts.items(), key=lambda x: -x[1])
            )
            patch_summary_text = (
                f"patch {patch_key}  |  {n} cells"
                f"  |  types: {type_str}  |  states: {state_str}"
            )
            print(f"\nPer-patch summary:\n  {patch_summary_text}")

    # Build panels
    panel_loc = build_original_location(he_image_path, x0, y0, patch_size, img_w, img_h)
    panel_he = build_he_panel(he_png)
    panel_mx = build_multiplex_panel(
        mx_npy, channel_idx=mx_channel, use_max_proj=mx_max_proj
    )
    panel_seg, had_seg = build_cellseg_panel(he_png, cellvit_json)
    panel_cell_mask, had_mask = build_cell_mask_panel(mask_npy)
    panel_cell_types = build_overlay_panel(cell_types_png, "cell type")
    panel_cell_states = build_overlay_panel(cell_states_png, "cell state")

    # Assemble figure
    fig, axes = plt.subplots(1, 7, figsize=(21, 4))
    fig.patch.set_facecolor("black")

    titles = [
        "Original location",
        "H&E patch",
        f"Multiplex ({mx_ch_name})",
        "Cell segmentation" if had_seg else "Cell segmentation\n(no data)",
        "Cell mask" if had_mask else "Cell mask\n(no data)",
        "Cell type",
        "Cell state",
    ]
    panels = [
        panel_loc,
        panel_he,
        panel_mx,
        panel_seg,
        panel_cell_mask,
        panel_cell_types,
        panel_cell_states,
    ]

    legends = [None, None, None, None, None, TYPE_LEGEND, STATE_LEGEND]
    for ax, img, title, leg in zip(axes, panels, titles, legends):
        _show_panel(ax, img, title, legend=leg)

    label = patch_summary_text if patch_summary_text else f"patch {patch_key}"
    fig.text(
        0.5,
        0.01,
        label,
        ha="center",
        va="bottom",
        color="white",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = processed_dir / f"pipeline_summary_{patch_key}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved figure to {out_path}")

    plt.show()
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_grid_figure(
    processed_dir: Path,
    patch_keys: list[str],
    he_image_path: str | None,
    mx_channel: int = 0,
    mx_max_proj: bool = False,
) -> Path:
    """Build a (N_patches × 7) grid figure and save it.

    Each row is one patch with the same 7 panels as make_summary_figure.
    Returns the path to the saved PNG.
    """
    n = len(patch_keys)
    fig, axes_grid = plt.subplots(n, 7, figsize=(16, 2 * n))
    fig.patch.set_facecolor("black")
    if n == 1:
        axes_grid = [axes_grid]  # make iterable

    # Load shared index data once
    with open(processed_dir / "index.json", encoding="utf-8") as f:
        index_data = json.load(f)
    patches_meta = {f"{p['x0']}_{p['y0']}": p for p in index_data.get("patches", [])}
    img_w = index_data.get("img_w", 0)
    img_h = index_data.get("img_h", 0)
    patch_size = index_data.get("patch_size", 256)
    channels = index_data.get("channels", [])
    if mx_max_proj:
        mx_ch_name = "max-proj"
    elif mx_channel < len(channels):
        mx_ch_name = channels[mx_channel]
    else:
        mx_ch_name = f"ch{mx_channel}"

    summary_json = processed_dir / "cell_summary.json"
    summary_per_patch = {}
    if summary_json.exists():
        with open(summary_json, encoding="utf-8") as f:
            summary_per_patch = json.load(f).get("per_patch", {})

    col_titles = [
        "Original location",
        "H&E patch",
        f"Multiplex ({mx_ch_name})",
        "Cell segmentation",
        "Cell mask",
        "Cell type",
        "Cell state",
    ]
    legends = [None, None, None, None, None, TYPE_LEGEND, STATE_LEGEND]

    for row_idx, patch_key in enumerate(patch_keys):
        ax_row = axes_grid[row_idx]
        meta = patches_meta.get(patch_key)
        if meta is None:
            for ax in ax_row:
                _show_panel(
                    ax, _gray_placeholder(f"patch {patch_key} not in index"), ""
                )
            continue

        x0, y0 = meta["x0"], meta["y0"]
        he_png = processed_dir / "he" / f"{patch_key}.png"
        mx_npy = processed_dir / "multiplex" / f"{patch_key}.npy"
        cellvit_json = processed_dir / "cellvit" / f"{patch_key}.json"
        mask_npy = processed_dir / "masks" / f"{patch_key}.npy"
        cell_types_png = processed_dir / "cell_types" / f"{patch_key}.png"
        cell_states_png = processed_dir / "cell_states" / f"{patch_key}.png"

        # Show the original-location panel for every row so each selected patch
        # has its own red rectangle in the first column.
        panel_loc = build_original_location(
            he_image_path, x0, y0, patch_size, img_w, img_h
        )

        panel_he = build_he_panel(he_png)
        panel_mx = build_multiplex_panel(
            mx_npy, channel_idx=mx_channel, use_max_proj=mx_max_proj
        )
        panel_seg, had_seg = build_cellseg_panel(he_png, cellvit_json)
        panel_cell_mask, had_mask = build_cell_mask_panel(mask_npy)
        panel_cell_types = build_overlay_panel(cell_types_png, "cell type")
        panel_cell_states = build_overlay_panel(cell_states_png, "cell state")
        panels = [
            panel_loc,
            panel_he,
            panel_mx,
            panel_seg,
            panel_cell_mask,
            panel_cell_types,
            panel_cell_states,
        ]

        # Legends and titles only on first row
        row_legends = legends if row_idx == 0 else [None] * 7

        # Per-patch cell summary as row label
        ps = summary_per_patch.get(patch_key, {})
        n_cells = ps.get("n_cells", "?")
        type_str = " ".join(
            f"{k[0].upper()}:{v}"
            for k, v in sorted(ps.get("cell_types", {}).items(), key=lambda x: -x[1])
        )
        row_label = f"patch {patch_key}  {n_cells} cells  {type_str}"

        for col_idx, (ax, img, leg) in enumerate(zip(ax_row, panels, row_legends)):
            title = col_titles[col_idx] if row_idx == 0 else ""
            if col_idx == 3 and not had_seg:
                title = "Cell segmentation\n(no data)" if row_idx == 0 else ""
            if col_idx == 4 and not had_mask:
                title = "Cell mask\n(no data)" if row_idx == 0 else ""
            _show_panel(ax, img, title, legend=leg)

        # Row label on the leftmost axis y-label
        ax_row[0].set_ylabel(
            row_label,
            color="white",
            fontsize=7,
            rotation=0,
            labelpad=4,
            ha="right",
            va="center",
        )

    fig.tight_layout(rect=[0, 0.01, 1, 1])
    out_path = processed_dir / f"pipeline_grid_{'_'.join(patch_keys[:3])}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved grid figure to {out_path}")
    plt.show()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate pipeline summary figure(s) for one or more patches."
    )
    parser.add_argument(
        "--processed",
        default="processed/",
        help="Path to the processed/ directory (default: processed/)",
    )
    parser.add_argument(
        "--patch",
        default=None,
        help="Patch key in 'x0_y0' format, e.g. '58624_4096'."
        " Defaults to first patch in index.json.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help="Randomly select N patches and show them as a grid (e.g. --random 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random (for reproducibility).",
    )
    parser.add_argument(
        "--he-image",
        default=None,
        help="Optional path to the full H&E OME-TIFF for the 'Original location' panel.",
    )
    parser.add_argument(
        "--mx-channel",
        type=int,
        default=0,
        metavar="IDX",
        help="Multiplex channel index to display (0-based, range 0–3, default: 0).",
    )
    parser.add_argument(
        "--mx-max-proj",
        action="store_true",
        help="Show max-projection across all multiplex channels instead of a single channel.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found at {index_path}")
    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)
    all_patches = index_data.get("patches", [])
    if not all_patches:
        raise ValueError("index.json contains no patches.")

    if args.random > 0:
        if args.seed is not None:
            random.seed(args.seed)
        chosen = random.sample(all_patches, min(args.random, len(all_patches)))
        patch_keys = [f"{p['x0']}_{p['y0']}" for p in chosen]
        print(f"Randomly selected {len(patch_keys)} patches: {patch_keys}")
        make_grid_figure(
            processed_dir,
            patch_keys,
            args.he_image,
            mx_channel=args.mx_channel,
            mx_max_proj=args.mx_max_proj,
        )
    else:
        patch_key = args.patch
        if patch_key is None:
            first = all_patches[0]
            patch_key = f"{first['x0']}_{first['y0']}"
            print(f"No --patch specified; using first patch: {patch_key}")
        make_summary_figure(
            processed_dir,
            patch_key,
            args.he_image,
            mx_channel=args.mx_channel,
            mx_max_proj=args.mx_max_proj,
        )


if __name__ == "__main__":
    main()
