"""Visualise cell type / cell state / vasculature tiles for selected patches.

3×3 matrix: rows = selected patches, cols = Cell Type | Cell State | Vasculature.
Black title row with coloured labels, matching select_marker_patch_examples.py style.

Usage:
    python -m tools.paper_figures.vis_cell_feature_tiles \\
        --config-json paper/figures/crc33_marker_wsi_zoom_combined_cd45_cd31_panck_sma.json \\
        [--patch-ids "10240_10240 10240_10496 0_6400"] \\
        [--out paper/figures/crc33_cell_feature_tiles.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .select_marker_patch_examples import (
    DEFAULT_STAGE1_FIGURE_DIR,
    _draw_marker_header,
    _draw_scale_bar,
    _load_font,
    _paste_center,
    _scale_bar_font,
)
from utils.colormaps import CELL_STATE_COLORS, CELL_TYPE_COLORS


LEGACY_CELL_STATE_RGB: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (240, 190, 0): CELL_STATE_COLORS["proliferative"][:3],
    (120, 120, 120): CELL_STATE_COLORS["nonprolif"][:3],
    (110, 60, 20): CELL_STATE_COLORS["dead"][:3],
}

# Column definitions: label shown in header, source subdir, render mode, marker names + colors.
COLUMNS: list[dict[str, Any]] = [
    {
        "label": "Cell Type",
        "subdir": "cell_types/union",
        "mode": "crop_rgb",
        "markers": ["cancer", "immune", "healthy"],
        "colors": [
            list(CELL_TYPE_COLORS["cancer"][:3]),
            list(CELL_TYPE_COLORS["immune"][:3]),
            list(CELL_TYPE_COLORS["healthy"][:3]),
        ],
        "scale_bar_color": (255, 255, 255),
    },
    {
        "label": "Cell State",
        "subdir": "cell_states/union",
        "mode": "state_union_rgb",
        "markers": ["prolif", "nonprolif", "dead"],
        "colors": [
            list(CELL_STATE_COLORS["proliferative"][:3]),
            list(CELL_STATE_COLORS["nonprolif"][:3]),
            list(CELL_STATE_COLORS["dead"][:3]),
        ],
        "scale_bar_color": (255, 255, 255),
    },
    {
        "label": "Vasculature",
        "subdir": "vasculature",
        "mode": "rgba_on_black",
        "markers": ["CD31"],
        "colors": [[255, 40, 40]],
        "scale_bar_color": (255, 255, 255),
    },
]


def _load_patch_ids_from_selection_json(
    selection_json: Path,
    group_id: str,
    n: int,
) -> list[str]:
    with selection_json.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    group = manifest.get("groups", {}).get(group_id, {})
    return [str(sel["patch_id"]) for sel in group.get("selections", [])[:n]]


def _crop_union_tile(arr: np.ndarray, tile_size: int) -> Image.Image:
    cropped = arr[:, :tile_size, :]
    if cropped.shape[1] < tile_size:
        pad = np.zeros((cropped.shape[0], tile_size - cropped.shape[1], 3), dtype=np.uint8)
        cropped = np.concatenate([cropped, pad], axis=1)
    return Image.fromarray(cropped)


def _remap_legacy_state_palette(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    for legacy_rgb, target_rgb in LEGACY_CELL_STATE_RGB.items():
        mask = np.all(out == np.array(legacy_rgb, dtype=np.uint8), axis=-1)
        if np.any(mask):
            out[mask] = np.array(target_rgb, dtype=np.uint8)
    return out


def _composite_rgba_on_black(overlay: Image.Image) -> Image.Image:
    base = Image.new("RGBA", overlay.size, (0, 0, 0, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _load_tile(processed: Path, patch_id: str, col: dict[str, Any], tile_size: int) -> Image.Image:
    img_path = processed / col["subdir"] / f"{patch_id}.png"
    img = Image.open(img_path)

    if col["mode"] == "crop_rgb":
        return _crop_union_tile(np.array(img.convert("RGB")), tile_size)

    if col["mode"] == "state_union_rgb":
        arr = _remap_legacy_state_palette(np.array(img.convert("RGB")))
        return _crop_union_tile(arr, tile_size)

    if col["mode"] == "rgba_on_black":
        return _composite_rgba_on_black(img.convert("RGBA"))

    return img.convert("RGB")


def make_cell_feature_tiles(
    processed: Path,
    patch_ids: list[str],
    out: Path | None,
    tile_size: int = 256,
    panel_gap: int = 6,
    header_height: int = 32,
    mpp: float = 0.325,
    scale_bar_um: float = 20.0,
) -> Image.Image:
    n = len(patch_ids)
    tile = int(tile_size)
    gap = int(panel_gap)
    header_h = int(header_height)
    row_h = tile + gap

    width = 3 * tile + 2 * gap
    height = header_h + n * tile + max(0, n - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = _load_font(max(14, int(tile / 14)), bold=True)
    scale_font = _scale_bar_font(tile)

    x_cols = [i * (tile + gap) for i in range(3)]

    for col_idx, col in enumerate(COLUMNS):
        _draw_marker_header(
            draw,
            x_cols[col_idx],
            0,
            tile,
            header_h,
            col["markers"],
            col["colors"],
            header_font,
        )

    for row_idx, patch_id in enumerate(patch_ids):
        y = header_h + row_idx * row_h
        show_label = row_idx == 0
        for col_idx, col in enumerate(COLUMNS):
            tile_img = _load_tile(processed, patch_id, col, tile)
            _paste_center(canvas, tile_img, x_cols[col_idx], y, tile)
            _draw_scale_bar(
                draw,
                x_cols[col_idx],
                y,
                tile,
                tile,
                mpp,
                scale_bar_um,
                color=col["scale_bar_color"],
                font=scale_font,
                show_label=show_label,
            )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=DEFAULT_STAGE1_FIGURE_DIR / "crc33_marker_wsi_zoom_combined_cd45_cd31_panck_sma.json",
        help="Combined figure metadata JSON (sets defaults for processed, zoom_style, etc.).",
    )
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Selection manifest JSON; falls back to selection_json in --config-json.",
    )
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--patch-ids",
        default=None,
        help="Space-separated patch IDs (e.g. '10240_10240 10240_10496 0_6400'). "
             "Overrides --selection-json.",
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--panel-gap", type=int, default=None)
    parser.add_argument("--header-height", type=int, default=None)
    parser.add_argument("--scale-bar-um", type=float, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_STAGE1_FIGURE_DIR / "crc33_cell_feature_tiles.png",
    )
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config_json.exists():
        with args.config_json.open(encoding="utf-8") as fh:
            config = json.load(fh)

    zoom_style = config.get("zoom_style", {})
    processed = args.processed or Path(config.get("processed", "processed_crc33"))
    group_id = args.group_id or str(config.get("group_id", "cd45_cd31_panck_sma"))
    n = args.n if args.n is not None else int(config.get("n_examples", 3))
    tile_size = args.tile_size if args.tile_size is not None else int(zoom_style.get("tile_size_px", 256))
    panel_gap = args.panel_gap if args.panel_gap is not None else int(zoom_style.get("panel_gap_px", 6))
    header_height = args.header_height if args.header_height is not None else int(zoom_style.get("header_height_px", 32))
    scale_bar_um = args.scale_bar_um if args.scale_bar_um is not None else float(zoom_style.get("scale_bar_um", 20.0))

    if args.patch_ids:
        patch_ids = args.patch_ids.split()[:n]
    else:
        selection_json = args.selection_json
        if selection_json is None:
            sel_path = config.get("selection_json")
            selection_json = Path(sel_path) if sel_path else None
        if selection_json is None or not selection_json.exists():
            raise FileNotFoundError(
                f"No selection JSON found at {selection_json}. "
                "Use --patch-ids or --selection-json."
            )
        patch_ids = _load_patch_ids_from_selection_json(selection_json, group_id, n)

    mpp = 0.325
    index_path = processed / "index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as fh:
            idx = json.load(fh)
        mpp = float(idx.get("he_mpp") or idx.get("mx_mpp") or 0.325)

    canvas = make_cell_feature_tiles(
        processed,
        patch_ids,
        args.out,
        tile_size=tile_size,
        panel_gap=panel_gap,
        header_height=header_height,
        mpp=mpp,
        scale_bar_um=scale_bar_um,
    )
    print(f"Saved: {args.out}  ({canvas.width}x{canvas.height})")
    print(f"Patches: {patch_ids}")


if __name__ == "__main__":
    main()
