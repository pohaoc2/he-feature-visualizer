"""Build a combined WSI overview + marker-rich zoom-tile paper figure."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from . import select_marker_patch_examples as zoom
from . import vis_cell_feature_tiles as cell_features
from . import vis_he_mx_side_by_side as overview
from utils.channels import resolve_channel_indices
from utils.ome import get_image_dims, get_ome_mpp, open_zarr_store


WSI_MARKERS = ["CD45", "CD31", "Pan-CK", "SMA"]
WSI_COLORS = (
    np.array(
        [
            [160, 32, 240],
            [255, 220, 0],
            [0, 220, 90],
            [255, 40, 40],
        ],
        dtype=np.float32,
    )
    / 255.0
)


def _resolve_zoom_groups(markers_csv: Path) -> list[dict]:
    marker_map = zoom._read_marker_indices(markers_csv)
    resolved_groups = []
    for group in zoom.FIGURE_GROUPS:
        indices = []
        resolved_names = []
        for marker in group["markers"]:
            idx, resolved_name = zoom._resolve_marker(marker_map, marker)
            indices.append(idx)
            resolved_names.append(resolved_name)
        background_index, background_name = zoom._resolve_marker(marker_map, "Hoechst")
        resolved_groups.append(
            {
                **group,
                "indices": indices,
                "resolved_markers": resolved_names,
                "background_index": background_index,
                "background_marker": background_name,
            }
        )
    return resolved_groups


def _draw_wsi_scale_bar(
    rgb: np.ndarray,
    mpp: float,
    effective_ds: float,
    scale_bar_um: float,
    font_size: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if mpp <= 0 or scale_bar_um <= 0 or effective_ds <= 0:
        return rgb
    h, w = rgb.shape[:2]
    bar_px = int(round(scale_bar_um / (mpp * effective_ds)))
    bar_px = max(8, min(w - 16, bar_px))
    bar_h = max(3, h // 100)
    margin = max(6, h // 60)
    x1, y1 = w - margin, h - margin
    x0, y0 = x1 - bar_px, y1 - bar_h
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=color)
    font = zoom._load_font(font_size, bold=True)
    label = _format_distance_label(scale_bar_um)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    mid_x = (x0 + x1) / 2.0
    draw.text(
        (
            int(round(mid_x - text_w / 2.0 - bbox[0])),
            int(round(y0 - text_h - bbox[1] - 6)),
        ),
        label,
        font=font,
        fill=color,
    )
    return np.asarray(img)


def _format_distance_label(scale_bar_um: float) -> str:
    if scale_bar_um >= 1000.0:
        return f"{scale_bar_um / 1000.0:g} mm"
    return f"{scale_bar_um:g} µm"


def _stack_vertical(top: Image.Image, bottom: Image.Image, gap: int = 8) -> Image.Image:
    width = max(top.width, bottom.width)
    canvas = Image.new("RGBA", (width, top.height + gap + bottom.height), (0, 0, 0, 0))
    top_x = (width - top.width) // 2
    bottom_x = (width - bottom.width) // 2
    canvas.paste(top, (top_x, 0))
    canvas.paste(bottom, (bottom_x, top.height + gap))
    return canvas


def _load_selected_patch_ids(selection_json: Path, group_id: str, n: int) -> list[str]:
    with selection_json.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    selections = payload.get("groups", {}).get(group_id, {}).get("selections", [])
    return [str(sel["patch_id"]) for sel in selections[:n]]


def _load_processed_mpp(processed_dir: Path, default: float = 0.325) -> float:
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        return float(default)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)
    return float(index.get("he_mpp") or index.get("mx_mpp") or default)


def _draw_transition_arrow(
    width: int,
    height: int,
    header_height: int,
    label: str = "KMeans",
) -> Image.Image:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = zoom._load_font(max(16, width // 5), bold=True)
    body_y = header_height + max(24, (height - header_height) // 2)
    x0 = max(10, width // 8)
    x1 = min(width - 12, width * 5 // 6)
    shaft_w = max(4, width // 18)
    head_len = max(12, width // 5)
    draw.line((x0, body_y, x1 - head_len, body_y), fill=(0, 0, 0), width=shaft_w)
    draw.polygon(
        [
            (x1 - head_len, body_y - max(10, shaft_w * 2)),
            (x1, body_y),
            (x1 - head_len, body_y + max(10, shaft_w * 2)),
        ],
        fill=(0, 0, 0),
    )
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((width - text_w) // 2 - bbox[0], body_y - text_h - 18 - bbox[1]),
        label,
        font=font,
        fill=(0, 0, 0),
    )
    return img


def _render_rotated_wsi_pair(
    he_path: Path,
    mx_path: Path,
    markers_csv: Path,
    selection_json: Path,
    group_id: str,
    downsample: int,
    crop_box_min_px: int,
    crop_box_width: int,
    scale_bar_um: float,
    scale_font_size: int,
) -> Image.Image:
    indices, _ = resolve_channel_indices(str(markers_csv), WSI_MARKERS)
    mx_channel_indices = list(indices)

    with tifffile.TiffFile(str(he_path)) as tif:
        he_base_w, he_base_h, _ = overview._series_dims(tif.series[0])
        he_chw, _, he_effective_ds = overview._read_fast_overview_chw(tif, downsample)
        he_mpp_x, _ = get_ome_mpp(tif)
    he_mpp = float(he_mpp_x) if he_mpp_x is not None else 0.0
    he_rgb = overview._render_he_rgb(he_chw)

    with tifffile.TiffFile(str(mx_path)) as tif:
        mx_base_w, mx_base_h, _ = overview._series_dims(tif.series[0])
        bg_idx, _ = resolve_channel_indices(str(markers_csv), ["Hoechst"])
        read_indices = [int(bg_idx[0]), *mx_channel_indices]
        mx_chw, _, mx_effective_ds = overview._read_fast_overview_chw(
            tif,
            downsample,
            channel_indices=read_indices,
        )
        mx_mpp_x, _ = get_ome_mpp(tif)
    mx_mpp = float(mx_mpp_x) if mx_mpp_x is not None else 0.0
    mx_rgb = overview._composite_markers_rgb(
        mx_chw[1:],
        list(range(len(mx_channel_indices))),
        WSI_COLORS,
        black_percentile=70.0,
        white_percentile=99.8,
        gamma=0.8,
        background=mx_chw[0],
        background_black_percentile=65.0,
        background_dilate_px=2,
    )

    boxes = overview._load_selection_boxes(selection_json, group_ids={group_id})
    he_rgb = overview._draw_crop_boxes(
        he_rgb,
        boxes,
        base_w=he_base_w,
        base_h=he_base_h,
        x_key="he_x0",
        y_key="he_y0",
        size_key="he_size",
        min_box_px=crop_box_min_px,
        line_width=crop_box_width,
    )
    mx_rgb = overview._draw_crop_boxes(
        mx_rgb,
        boxes,
        base_w=mx_base_w,
        base_h=mx_base_h,
        x_key="mx_x0",
        y_key="mx_y0",
        size_key="mx_size",
        min_box_px=crop_box_min_px,
        line_width=crop_box_width,
    )
    he_rot = Image.fromarray(he_rgb).transpose(Image.Transpose.ROTATE_270)
    mx_rot = Image.fromarray(mx_rgb).transpose(Image.Transpose.ROTATE_270)
    he_rot = Image.fromarray(
        _draw_wsi_scale_bar(
            np.asarray(he_rot),
            he_mpp,
            he_effective_ds,
            scale_bar_um,
            scale_font_size,
            (0, 0, 0),
        )
    )
    mx_rot = Image.fromarray(
        _draw_wsi_scale_bar(
            np.asarray(mx_rot),
            mx_mpp,
            mx_effective_ds,
            scale_bar_um,
            scale_font_size,
            (255, 255, 255),
        )
    )
    return _stack_vertical(he_rot, mx_rot, gap=8)


def _render_zoom_panel(
    processed_dir: Path,
    mx_ome: Path,
    markers_csv: Path,
    selection_json: Path,
    group_id: str,
    n: int,
    tile_size: int,
    panel_gap: int,
    header_height: int,
    scale_bar_um: float,
    mx_black_percentile: float,
    mx_median_size: int,
) -> Image.Image:
    patch_records, _, mpp = zoom._load_patch_records(processed_dir)
    row_records, _, _ = zoom._load_rows_from_selection_json(selection_json, patch_records, n)
    row_records = [rec for rec in row_records if rec["selected_for"] == group_id]
    if not row_records:
        raise ValueError(f"No selected rows found for group {group_id!r}")
    resolved_groups = _resolve_zoom_groups(markers_csv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "zoom_panel.png"
        with tifffile.TiffFile(str(mx_ome)) as tif:
            img_w, img_h, axes = get_image_dims(tif)
            store = open_zarr_store(tif)
            canvas = zoom._make_figure(
                processed_dir,
                row_records[:n],
                resolved_groups,
                tmp_path,
                title="",
                store=store,
                axes=axes,
                img_h=img_h,
                img_w=img_w,
                tile_size=tile_size,
                panel_gap=panel_gap,
                header_height=header_height,
                mpp=mpp,
                scale_bar_um=scale_bar_um,
                mx_black_percentile=mx_black_percentile,
                mx_median_size=mx_median_size,
            )
        return canvas


def _fit_height(img: Image.Image, target_h: int) -> Image.Image:
    scale = target_h / img.height
    new_w = max(1, int(round(img.width * scale)))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he", type=Path, default=Path("data/he_crc33.ome.tif"))
    parser.add_argument("--mx", type=Path, default=Path("data/mx_crc33.ome.tiff"))
    parser.add_argument("--processed", type=Path, default=Path("processed_crc33"))
    parser.add_argument("--markers-csv", type=Path, default=Path("data/markers.csv"))
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=zoom.DEFAULT_STAGE1_FIGURE_DIR / "crc33_marker_high_patch_examples_selections.json",
    )
    parser.add_argument("--out-dir", type=Path, default=zoom.DEFAULT_STAGE1_FIGURE_DIR)
    parser.add_argument(
        "--group-id",
        default="cd45_cd31_panck_sma",
        choices=[zoom.GROUP_CD45["id"], zoom.GROUP_CD68["id"]],
        help="Which three selected examples to show and box on the WSI overview.",
    )
    parser.add_argument(
        "--prefix",
        default="crc33_marker_wsi_zoom_combined",
        help="Output filename prefix.",
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--overview-downsample", type=int, default=64)
    parser.add_argument("--crop-box-min-px", type=int, default=12)
    parser.add_argument("--crop-box-width", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--panel-gap", type=int, default=6)
    parser.add_argument("--header-height", type=int, default=32)
    parser.add_argument("--scale-bar-um", type=float, default=20.0)
    parser.add_argument("--wsi-scale-bar-um", type=float, default=5000.0)
    parser.add_argument("--mx-render-black-percentile", type=float, default=5.0)
    parser.add_argument("--mx-render-median-size", type=int, default=3)
    args = parser.parse_args()

    scale_font_size = max(28, int(args.tile_size / 9))
    wsi = _render_rotated_wsi_pair(
        args.he,
        args.mx,
        args.markers_csv,
        args.selection_json,
        args.group_id,
        args.overview_downsample,
        args.crop_box_min_px,
        args.crop_box_width,
        scale_bar_um=args.wsi_scale_bar_um,
        scale_font_size=scale_font_size,
    )
    zoom_panel = _render_zoom_panel(
        args.processed,
        args.mx,
        args.markers_csv,
        args.selection_json,
        args.group_id,
        args.n,
        args.tile_size,
        args.panel_gap,
        args.header_height,
        args.scale_bar_um,
        args.mx_render_black_percentile,
        args.mx_render_median_size,
    )

    patch_ids = _load_selected_patch_ids(args.selection_json, args.group_id, args.n)
    cell_feature_panel = cell_features.make_cell_feature_tiles(
        args.processed,
        patch_ids,
        out=None,
        tile_size=args.tile_size,
        panel_gap=args.panel_gap,
        header_height=args.header_height,
        mpp=_load_processed_mpp(args.processed),
        scale_bar_um=args.scale_bar_um,
    )
    stage1_tiles_dir = args.out_dir / "tiles"
    exported_cell_feature_tiles = cell_features.export_rendered_tiles(
        args.processed,
        patch_ids,
        stage1_tiles_dir,
        tile_size=args.tile_size,
        group_id=args.group_id,
    )
    exported_metabolic_tiles = zoom.export_metabolic_tiles(
        args.processed,
        patch_ids,
        stage1_tiles_dir,
        group_id=args.group_id,
    )
    metabolic_panel = zoom.make_metabolic_gradient_tiles(
        args.processed,
        patch_ids,
        output_path=None,
        tile_size=args.tile_size,
        panel_gap=args.panel_gap,
        header_height=args.header_height,
        mpp=_load_processed_mpp(args.processed),
        scale_bar_um=args.scale_bar_um,
    )

    panel_w, panel_h = zoom_panel.size
    tile_rows_h = panel_h - int(args.header_height)
    section_gap = int(args.panel_gap)
    left = _fit_height(wsi, tile_rows_h)
    arrow_w = max(72, int(args.tile_size * 0.45))
    arrow_panel = _draw_transition_arrow(
        arrow_w,
        panel_h,
        int(args.header_height),
        label="KMeans",
    )
    pde_arrow_panel = _draw_transition_arrow(
        arrow_w,
        panel_h,
        int(args.header_height),
        label="PDE",
    )
    combined = Image.new(
        "RGBA",
        (
            left.width
            + section_gap
            + panel_w
            + section_gap
            + arrow_panel.width
            + section_gap
            + cell_feature_panel.width
            + section_gap
            + pde_arrow_panel.width
            + section_gap
            + metabolic_panel.width,
            panel_h,
        ),
        (0, 0, 0, 0),
    )
    combined.paste(left, (0, int(args.header_height)))
    x = left.width + section_gap
    combined.paste(zoom_panel, (x, 0))
    x += panel_w + section_gap
    combined.paste(arrow_panel, (x, 0))
    x += arrow_panel.width + section_gap
    combined.paste(cell_feature_panel, (x, 0))
    x += cell_feature_panel.width + section_gap
    combined.paste(pde_arrow_panel, (x, 0))
    x += pde_arrow_panel.width + section_gap
    combined.paste(metabolic_panel, (x, 0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"{args.prefix}_{args.group_id}.png"
    combined.save(out_png)

    metadata = {
        "output_png": str(out_png),
        "he": str(args.he),
        "mx": str(args.mx),
        "processed": str(args.processed),
        "markers_csv": str(args.markers_csv),
        "selection_json": str(args.selection_json),
        "group_id": args.group_id,
        "n_examples": int(args.n),
        "layout": {
            "left_section": "H&E/MX WSI overview pair, rotated 90 degrees clockwise and stacked vertically",
            "middle_section": "three selected H&E and MX zoom tiles",
            "transition": "KMeans arrow",
            "right_section": "three selected cell type/state/CD31 tiles",
            "pde_transition": "PDE arrow",
            "far_right_section": "three selected oxygen/glucose tiles",
            "zoom_panel_size_px": [panel_w, panel_h],
            "cell_feature_panel_size_px": list(cell_feature_panel.size),
            "arrow_panel_size_px": list(arrow_panel.size),
            "pde_arrow_panel_size_px": list(pde_arrow_panel.size),
            "metabolic_panel_size_px": list(metabolic_panel.size),
            "wsi_fitted_size_px": [left.width, left.height],
            "section_gap_px": section_gap,
            "wsi_original_rotated_size_px": list(wsi.size),
            "combined_size_px": list(combined.size),
        },
        "wsi_overlay": {
            "crop_box_count": int(args.n),
            "crop_box_min_px": int(args.crop_box_min_px),
            "crop_box_width": int(args.crop_box_width),
            "overview_downsample": int(args.overview_downsample),
            "scale_bar_um": float(args.wsi_scale_bar_um),
        },
        "zoom_style": {
            "tile_size_px": int(args.tile_size),
            "panel_gap_px": int(args.panel_gap),
            "header_height_px": int(args.header_height),
            "scale_bar_um": float(args.scale_bar_um),
            "mx_render_black_percentile": float(args.mx_render_black_percentile),
            "mx_render_median_size": int(args.mx_render_median_size),
        },
        "cell_feature_tiles": {
            "patch_ids": patch_ids,
            "columns": ["cell type", "cell state", "CD31"],
            "exported_tiles": exported_cell_feature_tiles,
            "export_dir": str(stage1_tiles_dir),
        },
        "metabolic_tiles": {
            "patch_ids": patch_ids,
            "columns": ["oxygen", "glucose"],
            "exported_tiles": exported_metabolic_tiles,
            "export_dir": str(stage1_tiles_dir),
        },
    }
    out_json = out_png.with_suffix(".json")
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")

    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
