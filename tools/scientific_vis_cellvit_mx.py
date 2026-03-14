#!/usr/bin/env python3
"""Publication-style visualization for MX + CellViT + inferred type/state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep matplotlib cache writable in sandboxed environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path("/tmp/matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from PIL import Image

from utils.marker_aliases import canonicalize_marker_name, normalize_marker_name
from utils.normalize import percentile_norm

DEFAULT_SCI_VIS_ROOT = Path(
    "/home/pohaoc2/.claude/plugins/marketplaces/"
    "claude-scientific-skills/scientific-skills/scientific-visualization"
)

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "healthy": (50, 180, 50, 200),
    "other": (150, 150, 150, 150),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (0, 255, 0, 200),
    "quiescent": (100, 149, 237, 200),
    "dead": (139, 0, 139, 200),
    "other": (80, 80, 80, 150),
}


def _load_patch_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        cells = data.get("cells", [])
        if isinstance(cells, list):
            return cells
    if isinstance(data, list):
        return data
    return []


def _composite_rgba_on_rgb(
    base_rgb: np.ndarray, overlay_rgba: np.ndarray
) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[-1] != 4:
        return base_rgb
    ov = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    out = (alpha * ov + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return out


def _draw_cellvit_contours(he_rgb: np.ndarray, cells: list[dict]) -> np.ndarray:
    out = he_rgb.copy()
    for cell in cells:
        contour = cell.get("contour", [])
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    return out


def _resolve_mx_channel(
    index_json: Path,
    mx_channel: int,
    mx_marker: str | None,
) -> tuple[int, str]:
    resolved_idx = int(mx_channel)
    resolved_name = f"ch{resolved_idx}"

    if not index_json.exists():
        return resolved_idx, resolved_name

    with index_json.open(encoding="utf-8") as fh:
        index_data = json.load(fh)
    channels = [str(x) for x in index_data.get("channels", [])]
    if not channels:
        return resolved_idx, resolved_name

    if mx_marker:
        target_norm = normalize_marker_name(mx_marker)
        target_canon = canonicalize_marker_name(mx_marker)
        for i, name in enumerate(channels):
            name_norm = normalize_marker_name(name)
            name_canon = canonicalize_marker_name(name)
            if (
                name_norm == target_norm
                or name_canon == target_canon
                or target_norm in name_norm
            ):
                resolved_idx = i
                resolved_name = name
                break
        else:
            raise ValueError(
                f"Marker '{mx_marker}' not found in index channels: {channels}"
            )
    else:
        if resolved_idx < 0 or resolved_idx >= len(channels):
            raise ValueError(
                f"--mx-channel {resolved_idx} out of range [0, {len(channels) - 1}]"
            )
        resolved_name = channels[resolved_idx]

    return resolved_idx, resolved_name


def _scientific_style(
    sci_vis_root: Path,
):
    scripts_dir = sci_vis_root / "scripts"
    if not scripts_dir.exists():
        return None, None, f"scientific-vis scripts not found at: {scripts_dir}"

    sys.path.insert(0, str(scripts_dir))
    try:
        from figure_export import save_publication_figure
        from style_presets import apply_publication_style, set_color_palette

        apply_publication_style("default")
        set_color_palette("okabe_ito")
        return save_publication_figure, scripts_dir, None
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, None, f"failed to load scientific-vis scripts: {exc}"


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _build_legends(ax: plt.Axes) -> None:
    ax.axis("off")
    type_handles = []
    for name in ("cancer", "immune", "healthy"):
        rgba = CELL_TYPE_COLORS[name]
        type_handles.append(
            mpatches.Patch(
                color=np.array(rgba[:3], dtype=float) / 255.0,
                label=name,
                alpha=0.8,
            )
        )

    state_handles = []
    for name in ("proliferative", "quiescent", "dead"):
        rgba = CELL_STATE_COLORS[name]
        state_handles.append(
            mpatches.Patch(
                color=np.array(rgba[:3], dtype=float) / 255.0,
                label=name,
                alpha=0.8,
            )
        )

    leg1 = ax.legend(
        handles=type_handles,
        title="Cell Type",
        loc="upper left",
        frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=state_handles,
        title="Cell State",
        loc="lower left",
        frameon=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-style figure for one patch showing "
            "H&E, MX channel, CellViT contours, cell type, and cell state."
        )
    )
    parser.add_argument("--processed", required=True, help="Processed directory.")
    parser.add_argument(
        "--patch",
        required=True,
        help="Patch key as x0_y0, e.g. 256_256.",
    )
    parser.add_argument(
        "--mx-channel",
        type=int,
        default=0,
        help="MX channel index if --mx-marker is not used.",
    )
    parser.add_argument(
        "--mx-marker",
        default=None,
        help="Preferred marker name (resolved from index.json channels).",
    )
    parser.add_argument(
        "--scientific-vis-root",
        default=str(DEFAULT_SCI_VIS_ROOT),
        help="Path to scientific-visualization skill directory.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix without extension. Default: <processed>/scientific_vis_<patch>.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated formats (e.g. pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    patch = str(args.patch)
    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else processed_dir / f"scientific_vis_{patch}"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    he_path = processed_dir / "he" / f"{patch}.png"
    mx_path = processed_dir / "multiplex" / f"{patch}.npy"
    cellvit_path = processed_dir / "cellvit" / f"{patch}.json"
    type_path = processed_dir / "cell_types" / f"{patch}.png"
    state_path = processed_dir / "cell_states" / f"{patch}.png"
    index_path = processed_dir / "index.json"

    required = [he_path, mx_path, cellvit_path, type_path, state_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    mx_idx, mx_name = _resolve_mx_channel(index_path, args.mx_channel, args.mx_marker)

    he_rgb = np.array(Image.open(he_path).convert("RGB"))
    mx_arr = np.load(mx_path)
    if mx_arr.ndim != 3:
        raise ValueError(f"Expected multiplex patch shape (C,H,W), got {mx_arr.shape}")
    if mx_idx < 0 or mx_idx >= mx_arr.shape[0]:
        raise ValueError(
            f"Resolved MX channel index {mx_idx} out of range [0, {mx_arr.shape[0] - 1}]"
        )
    mx_img = percentile_norm(mx_arr[mx_idx].astype(np.float32))

    cells = _load_patch_json(cellvit_path)
    cellvit_overlay = _draw_cellvit_contours(he_rgb, cells)
    type_overlay = np.array(Image.open(type_path).convert("RGBA"))
    state_overlay = np.array(Image.open(state_path).convert("RGBA"))

    type_on_he = _composite_rgba_on_rgb(he_rgb, type_overlay)
    state_on_he = _composite_rgba_on_rgb(he_rgb, state_overlay)

    save_publication_figure, _, style_warning = _scientific_style(
        Path(args.scientific_vis_root)
    )
    if style_warning:
        print(f"Warning: {style_warning}")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    ax = axes.ravel()

    ax[0].imshow(he_rgb)
    ax[0].set_title("H&E")
    ax[0].axis("off")
    _panel_label(ax[0], "A")

    ax[1].imshow(mx_img, cmap="viridis")
    ax[1].set_title(f"MX channel: {mx_name}")
    ax[1].axis("off")
    _panel_label(ax[1], "B")

    ax[2].imshow(cellvit_overlay)
    ax[2].set_title(f"CellViT contours (n={len(cells)})")
    ax[2].axis("off")
    _panel_label(ax[2], "C")

    ax[3].imshow(type_on_he)
    ax[3].set_title("Inferred cell type")
    ax[3].axis("off")
    _panel_label(ax[3], "D")

    ax[4].imshow(state_on_he)
    ax[4].set_title("Inferred cell state")
    ax[4].axis("off")
    _panel_label(ax[4], "E")

    _build_legends(ax[5])
    _panel_label(ax[5], "F")

    fig.suptitle(f"Patch {patch}: MX + CellViT + Type/State", fontsize=10, y=1.02)

    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    if not formats:
        formats = ["png"]

    if save_publication_figure is not None:
        save_publication_figure(fig, out_prefix, formats=formats, dpi=int(args.dpi))
    else:
        for fmt in formats:
            out_path = out_prefix.with_suffix(f".{fmt}")
            fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
            print(f"Saved: {out_path}")

    plt.close(fig)
    print(f"Done. patch={patch}, mx_channel={mx_idx}, mx_marker={mx_name}")


if __name__ == "__main__":
    main()
