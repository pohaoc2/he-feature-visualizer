"""Select and render marker-rich raw-MX patch examples.

The figures have columns:
  H&E | CD45/CD31/Pan-CK/SMA raw MX crop | CD68/CD163/Ki67/PD-L1 raw MX crop

Patch coordinates come from ``processed_crc33/index.json``. Multiplex scoring
and rendering are read directly from ``data/mx_crc33.ome.tiff``; this tool does
not use the processed ``multiplex/*.npy`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .vis_he_mx_side_by_side import _composite_markers_rgb
from utils.cellvit_io import load_patch_json
from utils.marker_aliases import canonicalize_marker_name, normalize_marker_name
from utils.ome import get_image_dims, open_zarr_store


DEFAULT_STAGE1_FIGURE_DIR = Path("paper/figures/stage1")
TILE_SCALE_FONT_DIVISOR = 11
TILE_SCALE_FONT_MIN_SIZE = 22
CELLVIT_CONTOUR_OUTLINE_WIDTH = 4
CELLVIT_CONTOUR_INNER_WIDTH = 2
METABOLIC_COLORBAR_WIDTH = 18
METABOLIC_COLORBAR_PAD = 8


GROUP_CD45 = {
    "id": "cd45_cd31_panck_sma",
    "label": "High CD45 / CD31 / Pan-CK / SMA",
    "markers": ["CD45", "CD31", "Pan-CK", "SMA"],
    "colors_rgb": [
        [160, 32, 240],
        [255, 220, 0],
        [0, 220, 90],
        [255, 40, 40],
    ],
}

GROUP_CD68 = {
    "id": "cd68_cd163_ki67_pdl1",
    "label": "High CD68 / CD163 / Ki67 / PD-L1",
    "markers": ["CD68", "CD163", "Ki67", "PD-L1"],
    "colors_rgb": [
        [255, 128, 0],
        [0, 210, 210],
        [255, 60, 220],
        [80, 150, 255],
    ],
}

METABOLIC_COLUMNS = [
    {
        "label": "Oxygen",
        "subdir": "oxygen",
        "markers": ["Oxygen"],
        "colors_rgb": [[0, 255, 255]],
        "scale_bar_color": (0, 0, 0),
    },
    {
        "label": "Glucose",
        "subdir": "glucose",
        "markers": ["Glucose"],
        "colors_rgb": [[255, 242, 30]],
        "scale_bar_color": (0, 0, 0),
    },
]

FIGURE_GROUPS = [GROUP_CD45, GROUP_CD68]
SELECTION_GROUPS = [GROUP_CD68, GROUP_CD45]
CHANNEL_AXES = ("C", "I", "S")


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    arial_name = "Arialbd.ttf" if bold else "Arial.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/msttcorefonts") / arial_name,
        Path("/usr/share/fonts/truetype/liberation")
        / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_marker_indices(markers_csv: Path) -> dict[str, tuple[int, str]]:
    with markers_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty marker CSV: {markers_csv}")
        columns = {c.strip(): c for c in reader.fieldnames}
        if "Channel_Number" not in columns or "Marker_Name" not in columns:
            raise ValueError(
                "Expected marker CSV columns Channel_Number and Marker_Name; "
                f"found {reader.fieldnames}"
            )

        by_key: dict[str, tuple[int, str]] = {}
        for row in reader:
            name = str(row[columns["Marker_Name"]]).strip()
            if not name:
                continue
            channel_number = int(row[columns["Channel_Number"]])
            idx = channel_number - 1
            keys = {
                normalize_marker_name(name),
                normalize_marker_name(canonicalize_marker_name(name)),
            }
            for key in keys:
                by_key.setdefault(key, (idx, name))
        return by_key


def _resolve_marker(
    marker_map: dict[str, tuple[int, str]],
    marker: str,
) -> tuple[int, str]:
    keys = [
        normalize_marker_name(marker),
        normalize_marker_name(canonicalize_marker_name(marker)),
    ]
    for key in keys:
        if key in marker_map:
            return marker_map[key]
    available = sorted({name for _, name in marker_map.values()})
    raise ValueError(f"Marker {marker!r} not found. Available markers: {available}")


def _channel_axis(axes: str) -> str | None:
    axes_up = axes.upper()
    for ax in CHANNEL_AXES:
        if ax in axes_up:
            return ax
    return None


def _read_raw_crop_chw(
    store,
    axes: str,
    img_h: int,
    img_w: int,
    x0: int,
    y0: int,
    size: int,
    channel_indices: list[int],
) -> np.ndarray:
    """Read a raw OME crop as CHW, selecting only requested channels."""
    axes_up = axes.upper()
    ch_axis = _channel_axis(axes_up)
    y0i, x0i = int(y0), int(x0)
    y1i, x1i = y0i + int(size), x0i + int(size)
    y0c, x0c = max(0, y0i), max(0, x0i)
    y1c, x1c = min(int(img_h), y1i), min(int(img_w), x1i)
    rh, rw = max(0, y1c - y0c), max(0, x1c - x0c)
    if rh == 0 or rw == 0:
        return np.zeros((len(channel_indices), size, size), dtype=np.uint16)

    uniq = sorted(set(int(i) for i in channel_indices))
    sl: list[int | slice | list[int]] = []
    active: list[str] = []
    for ax in axes_up:
        if ch_axis is not None and ax == ch_axis:
            sl.append(uniq)
            active.append("C")
        elif ax == "Y":
            sl.append(slice(y0c, y1c))
            active.append("Y")
        elif ax == "X":
            sl.append(slice(x0c, x1c))
            active.append("X")
        else:
            sl.append(0)

    arr = np.array(store[tuple(sl)])
    if "C" in active:
        target = [ax for ax in ("C", "Y", "X") if ax in active]
        if active != target:
            arr = arr.transpose([active.index(ax) for ax in target])
        pos = {idx: j for j, idx in enumerate(uniq)}
        arr = np.stack([arr[pos[int(idx)]] for idx in channel_indices], axis=0)
    else:
        target = [ax for ax in ("Y", "X") if ax in active]
        if active != target:
            arr = arr.transpose([active.index(ax) for ax in target])
        arr = np.stack([arr] * len(channel_indices), axis=0)

    if arr.shape[1] == size and arr.shape[2] == size:
        return arr

    out = np.zeros((arr.shape[0], size, size), dtype=arr.dtype)
    dy, dx = y0c - y0i, x0c - x0i
    out[:, dy : dy + rh, dx : dx + rw] = arr[:, :rh, :rw]
    return out


def _top_fraction_mean(arr: np.ndarray, fraction: float) -> float:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(round(flat.size * fraction)))
    if k >= flat.size:
        return float(flat.mean())
    top = np.partition(flat, flat.size - k)[-k:]
    return float(top.mean())


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    n = values.shape[0]
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = ((i + 1 + j) / 2.0) / n
        i = j
    return ranks


def _load_patch_records(processed_dir: Path) -> tuple[list[dict[str, Any]], int, float]:
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing patch index: {index_path}")
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patch_size = int(index.get("patch_size", 256))
    mpp = float(index.get("mx_mpp") or index.get("he_mpp") or 0.325)
    scale_he_to_mx = float(index.get("scale_he_to_mx", 1.0) or 1.0)
    he_dir = processed_dir / "he"
    records = []
    for patch in index.get("patches", []):
        if not patch.get("has_multiplex", True):
            continue
        x0 = int(patch["x0"])
        y0 = int(patch["y0"])
        patch_id = f"{x0}_{y0}"
        if not (he_dir / f"{patch_id}.png").exists():
            continue
        records.append(
            {
                "patch_id": patch_id,
                "x0": x0,
                "y0": y0,
                "mx_x0": int(round(x0 * scale_he_to_mx)),
                "mx_y0": int(round(y0 * scale_he_to_mx)),
                "mx_size": max(1, int(round(patch_size * scale_he_to_mx))),
            }
        )
    if not records:
        raise RuntimeError(f"No paired H&E/raw-MX patch records under {processed_dir}")
    return records, patch_size, mpp


def _score_patches(
    patch_records: list[dict[str, Any]],
    store,
    axes: str,
    img_h: int,
    img_w: int,
    marker_indices: dict[str, int],
    top_fraction: float,
) -> list[dict[str, Any]]:
    markers = list(marker_indices)
    indices = [marker_indices[marker] for marker in markers]
    scored = [dict(rec) for rec in patch_records]
    stats_by_marker: dict[str, list[float]] = {marker: [] for marker in markers}

    for n_done, rec in enumerate(scored, start=1):
        chw = _read_raw_crop_chw(
            store,
            axes,
            img_h,
            img_w,
            x0=int(rec["mx_x0"]),
            y0=int(rec["mx_y0"]),
            size=int(rec["mx_size"]),
            channel_indices=indices,
        )
        for marker, channel in zip(markers, chw):
            stats_by_marker[marker].append(_top_fraction_mean(channel, top_fraction))
        if n_done % 500 == 0:
            print(f"Scored {n_done:,}/{len(scored):,} raw MX crops")

    for marker, values_list in stats_by_marker.items():
        values = np.asarray(values_list, dtype=np.float64)
        ranks = _percentile_ranks(values)
        for rec, value, rank in zip(scored, values, ranks):
            rec[f"{marker}_top_mean"] = float(value)
            rec[f"{marker}_rank"] = float(rank)

    for group in SELECTION_GROUPS:
        rank_cols = [f"{marker}_rank" for marker in group["markers"]]
        for rec in scored:
            ranks = [float(rec[col]) for col in rank_cols]
            rec[f"{group['id']}_score"] = float(np.mean(ranks))
            rec[f"{group['id']}_min_rank"] = float(np.min(ranks))

    return scored


def _select_top(
    records: list[dict[str, Any]],
    group: dict[str, Any],
    n: int,
    used: set[str],
) -> list[dict[str, Any]]:
    group_id = str(group["id"])
    ordered = sorted(
        records,
        key=lambda rec: (
            -float(rec[f"{group_id}_score"]),
            -float(rec[f"{group_id}_min_rank"]),
            str(rec["patch_id"]),
        ),
    )
    selected = []
    for rec in ordered:
        if str(rec["patch_id"]) in used:
            continue
        selected.append(rec)
        used.add(str(rec["patch_id"]))
        if len(selected) == n:
            return selected
    raise RuntimeError(f"Only found {len(selected)} selectable patches for {group_id}")


def _render_group_mx(
    rec: dict[str, Any],
    store,
    axes: str,
    img_h: int,
    img_w: int,
    marker_indices: list[int],
    colors_rgb: list[list[int]],
    black_percentile: float,
    median_size: int,
) -> np.ndarray:
    chw = _read_raw_crop_chw(
        store,
        axes,
        img_h,
        img_w,
        x0=int(rec["mx_x0"]),
        y0=int(rec["mx_y0"]),
        size=int(rec["mx_size"]),
        channel_indices=marker_indices,
    )
    colors = np.asarray(colors_rgb, dtype=np.float32) / 255.0
    rgb = _composite_markers_rgb(
        chw,
        list(range(len(marker_indices))),
        colors,
        black_percentile=black_percentile,
        white_percentile=99.8,
        gamma=0.8,
        background=None,
    )
    if median_size >= 3:
        if median_size % 2 == 0:
            median_size += 1
        rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.MedianFilter(median_size)))
    return rgb


def _paste_center(canvas: Image.Image, tile: Image.Image, x: int, y: int, size: int) -> None:
    tile = tile.convert("RGB")
    if tile.size != (size, size):
        tile.thumbnail((size, size), Image.Resampling.LANCZOS)
    x_off = x + (size - tile.width) // 2
    y_off = y + (size - tile.height) // 2
    canvas.paste(tile, (x_off, y_off))


def _draw_text_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (30, 30, 30),
    line_gap: int = 4,
) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    markers: list[str],
    colors: list[list[int]],
    font: ImageFont.ImageFont,
) -> None:
    cursor_x = x
    for marker, color in zip(markers, colors):
        draw.rectangle([cursor_x, y + 2, cursor_x + 9, y + 11], fill=tuple(color))
        draw.text((cursor_x + 13, y), marker, font=font, fill=(50, 50, 50))
        text_w = draw.textbbox((cursor_x + 13, y), marker, font=font)[2] - cursor_x
        cursor_x += max(58, text_w + 8)


def _header_label_centers(x: int, width: int, n_labels: int) -> list[float]:
    if n_labels <= 0:
        return []
    slot_width = float(width) / float(n_labels)
    return [float(x) + slot_width * (idx + 0.5) for idx in range(n_labels)]


def _draw_marker_header(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    markers: list[str],
    colors: list[list[int]],
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle([x, y, x + width - 1, y + height - 1], fill=(0, 0, 0))
    centers = _header_label_centers(x, width, len(markers))
    for center, marker, color in zip(centers, markers, colors):
        draw.text(
            (int(round(center)), int(round(y + height / 2.0))),
            marker,
            font=font,
            fill=tuple(color),
            anchor="mm",
        )


def _scale_bar_font(tile_size: int) -> ImageFont.ImageFont:
    return _load_font(
        max(TILE_SCALE_FONT_MIN_SIZE, int(tile_size / TILE_SCALE_FONT_DIVISOR)),
        bold=True,
    )


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    tile_size: int,
    raw_size_px: int,
    mpp: float,
    scale_bar_um: float,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    show_label: bool,
) -> None:
    if mpp <= 0 or scale_bar_um <= 0:
        return
    length = int(round((scale_bar_um / mpp) * (tile_size / float(raw_size_px))))
    length = max(8, min(tile_size - 16, length))
    bar_h = max(5, int(round(tile_size / 35)))
    margin = max(8, int(round(tile_size / 26)))
    x1 = x + tile_size - margin
    x0 = x1 - length
    y1 = y + tile_size - margin
    y0 = y1 - bar_h
    draw.rectangle([x0, y0, x1, y1], fill=color)
    if show_label:
        label = f"{scale_bar_um:g} µm"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        mid_x = (x0 + x1) / 2.0
        text_x = int(round(mid_x - text_w / 2.0 - bbox[0]))
        text_y = int(round(y0 - text_h - 5 - bbox[1]))
        draw.text((text_x, text_y), label, font=font, fill=color)


def _overlay_cellvit_contours(
    processed_dir: Path,
    patch_id: str,
    he_img: Image.Image,
) -> Image.Image:
    cellvit_path = processed_dir / "cellvit" / f"{patch_id}.json"
    if not cellvit_path.exists():
        return he_img.convert("RGB")

    cells = load_patch_json(cellvit_path)
    if not cells:
        return he_img.convert("RGB")

    img = he_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    scale_x = img.width / 256.0 if img.width > 0 else 1.0
    scale_y = img.height / 256.0 if img.height > 0 else 1.0
    for cell in cells:
        contour = cell.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        points = [
            (float(pt[0]) * scale_x, float(pt[1]) * scale_y)
            for pt in contour
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(points) < 3:
            continue
        closed = points + [points[0]]
        draw.line(closed, fill=(0, 0, 0), width=CELLVIT_CONTOUR_OUTLINE_WIDTH)
        draw.line(closed, fill=(255, 255, 255), width=CELLVIT_CONTOUR_INNER_WIDTH)
    return img


def _draw_cellvit_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    tile_size: int,
    font: ImageFont.ImageFont,
) -> None:
    pad = max(8, tile_size // 32)
    sample_w = max(18, tile_size // 10)
    sample_x0 = x + pad + 8
    sample_y = y + pad + 10
    label = "CellViT"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = sample_w + 16 + text_w + 16
    box_h = max(24, text_h + 14)
    draw.rectangle(
        [x + pad, y + pad, x + pad + box_w, y + pad + box_h],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
    )
    draw.line(
        [(sample_x0, sample_y), (sample_x0 + sample_w, sample_y)],
        fill=(0, 0, 0),
        width=CELLVIT_CONTOUR_OUTLINE_WIDTH,
    )
    draw.line(
        [(sample_x0, sample_y), (sample_x0 + sample_w, sample_y)],
        fill=(255, 255, 255),
        width=CELLVIT_CONTOUR_INNER_WIDTH,
    )
    draw.text(
        (sample_x0 + sample_w + 10, y + pad + box_h // 2),
        label,
        font=font,
        fill=(0, 0, 0),
        anchor="lm",
    )


def _draw_vertical_colorbar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    color_rgb: list[int],
) -> None:
    top = np.array(color_rgb, dtype=np.float32)
    bottom = np.zeros(3, dtype=np.float32)
    for offset in range(max(1, height)):
        t = 1.0 - (offset / max(1, height - 1))
        rgb = np.clip(bottom + t * (top - bottom), 0.0, 255.0).astype(np.uint8)
        draw.line([(x, y + offset), (x + width - 1, y + offset)], fill=tuple(rgb.tolist()))
    draw.rectangle([x, y, x + width - 1, y + height - 1], outline=(0, 0, 0))


def _flatten_rgba(tile: Image.Image) -> Image.Image:
    rgba = tile.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, (0, 0), rgba.split()[3])
    return rgb


def _load_processed_feature_tile(processed_dir: Path, patch_id: str, subdir: str) -> Image.Image:
    tile = Image.open(processed_dir / subdir / f"{patch_id}.png")
    if tile.mode == "RGBA":
        return _flatten_rgba(tile)
    return tile.convert("RGB")


def make_metabolic_gradient_tiles(
    processed_dir: Path,
    patch_ids: list[str],
    output_path: Path | None,
    tile_size: int = 256,
    panel_gap: int = 6,
    header_height: int = 32,
    mpp: float = 0.325,
    scale_bar_um: float = 20.0,
) -> Image.Image:
    tile = int(tile_size)
    gap = int(panel_gap)
    header_h = int(header_height)
    n_rows = len(patch_ids)
    n_cols = len(METABOLIC_COLUMNS)
    row_h = tile + gap
    column_w = tile + METABOLIC_COLORBAR_PAD + METABOLIC_COLORBAR_WIDTH
    width = n_cols * column_w + max(0, n_cols - 1) * gap
    height = header_h + n_rows * tile + max(0, n_rows - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(max(14, int(tile / 14)), bold=True)
    scale_font = _scale_bar_font(tile)
    x_cols = [idx * (column_w + gap) for idx in range(n_cols)]

    for x_col, column in zip(x_cols, METABOLIC_COLUMNS):
        _draw_marker_header(
            draw,
            x_col,
            0,
            column_w,
            header_h,
            column["markers"],
            column["colors_rgb"],
            header_font,
        )
        _draw_vertical_colorbar(
            draw,
            x_col + tile + METABOLIC_COLORBAR_PAD,
            header_h + 4,
            METABOLIC_COLORBAR_WIDTH,
            max(1, height - header_h - 8),
            column["colors_rgb"][0],
        )

    for row_idx, patch_id in enumerate(patch_ids):
        y = header_h + row_idx * row_h
        show_label = row_idx == 0
        for x_col, column in zip(x_cols, METABOLIC_COLUMNS):
            tile_img = _load_processed_feature_tile(
                processed_dir,
                patch_id,
                str(column["subdir"]),
            )
            _paste_center(canvas, tile_img, x_col, y, tile)
            _draw_scale_bar(
                draw,
                x_col,
                y,
                tile,
                tile,
                mpp,
                scale_bar_um,
                color=tuple(column["scale_bar_color"]),
                font=scale_font,
                show_label=show_label,
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return canvas


def _make_figure(
    processed_dir: Path,
    selected_rows: list[dict[str, Any]],
    resolved_groups: list[dict[str, Any]],
    output_path: Path,
    title: str,
    store,
    axes: str,
    img_h: int,
    img_w: int,
    tile_size: int,
    panel_gap: int,
    header_height: int,
    mpp: float,
    scale_bar_um: float,
    mx_black_percentile: float,
    mx_median_size: int,
) -> Image.Image:
    tile = int(tile_size)
    gap = int(panel_gap)
    header_h = int(header_height)
    row_h = tile + gap
    width = 3 * tile + 2 * gap
    height = header_h + len(selected_rows) * tile + max(0, len(selected_rows) - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(max(14, int(tile / 14)), bold=True)
    scale_font = _scale_bar_font(tile)
    legend_font = _load_font(max(12, int(tile / 20)), bold=True)

    x_cols = [0, tile + gap, 2 * (tile + gap)]
    _draw_marker_header(
        draw,
        x_cols[0],
        0,
        tile,
        header_h,
        ["H&E"],
        [[255, 255, 255]],
        header_font,
    )
    for x, group in zip(x_cols[1:], resolved_groups):
        _draw_marker_header(
            draw,
            x,
            0,
            tile,
            header_h,
            group["markers"],
            group["colors_rgb"],
            header_font,
        )

    he_dir = processed_dir / "he"
    for row_idx, rec in enumerate(selected_rows):
        y = header_h + row_idx * row_h
        patch_id = str(rec["patch_id"])

        he_img = _overlay_cellvit_contours(
            processed_dir,
            patch_id,
            Image.open(he_dir / f"{patch_id}.png"),
        )
        mx_imgs = [
            Image.fromarray(
                _render_group_mx(
                    rec,
                    store,
                    axes,
                    img_h,
                    img_w,
                    group["indices"],
                    group["colors_rgb"],
                    mx_black_percentile,
                    mx_median_size,
                )
            )
            for group in resolved_groups
        ]

        _paste_center(canvas, he_img, x_cols[0], y, tile)
        if row_idx == 0:
            _draw_cellvit_legend(draw, x_cols[0], y, tile, legend_font)
        _paste_center(canvas, mx_imgs[0], x_cols[1], y, tile)
        _paste_center(canvas, mx_imgs[1], x_cols[2], y, tile)
        raw_size = int(rec.get("mx_size", 256))
        show_label = row_idx == 0
        _draw_scale_bar(
            draw,
            x_cols[0],
            y,
            tile,
            raw_size,
            mpp,
            scale_bar_um,
            color=(0, 0, 0),
            font=scale_font,
            show_label=show_label,
        )
        for x_col in x_cols[1:]:
            _draw_scale_bar(
                draw,
                x_col,
                y,
                tile,
                raw_size,
                mpp,
                scale_bar_um,
                color=(255, 255, 255),
                font=scale_font,
                show_label=show_label,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas


def _selection_payload(
    rec: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any]:
    group_id = str(group["id"])
    markers = list(group["markers"])
    return {
        "patch_id": str(rec["patch_id"]),
        "he_x0": int(rec["x0"]),
        "he_y0": int(rec["y0"]),
        "mx_x0": int(rec["mx_x0"]),
        "mx_y0": int(rec["mx_y0"]),
        "mx_size": int(rec["mx_size"]),
        "score": float(rec[f"{group_id}_score"]),
        "min_marker_rank": float(rec[f"{group_id}_min_rank"]),
        "marker_top_1pct_mean": {
            marker: float(rec[f"{marker}_top_mean"]) for marker in markers
        },
        "marker_percentile_rank": {
            marker: float(rec[f"{marker}_rank"]) for marker in markers
        },
    }


def _load_rows_from_selection_json(
    selection_json: Path,
    patch_records: list[dict[str, Any]],
    n: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    with selection_json.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    patch_by_id = {str(rec["patch_id"]): rec for rec in patch_records}
    selections_by_group: dict[str, list[dict[str, Any]]] = {}
    row_records: list[dict[str, Any]] = []

    for group in [GROUP_CD68, GROUP_CD45]:
        group_id = str(group["id"])
        group_payload = manifest.get("groups", {}).get(group_id, {})
        loaded: list[dict[str, Any]] = []
        for selection in group_payload.get("selections", [])[:n]:
            patch_id = str(selection["patch_id"])
            base = dict(patch_by_id.get(patch_id, {}))
            if not base:
                x_str, y_str = patch_id.split("_", maxsplit=1)
                x0, y0 = int(x_str), int(y_str)
                base = {
                    "patch_id": patch_id,
                    "x0": x0,
                    "y0": y0,
                    "mx_x0": int(selection.get("mx_x0", x0)),
                    "mx_y0": int(selection.get("mx_y0", y0)),
                    "mx_size": int(selection.get("mx_size", 256)),
                }
            rec = {
                **base,
                f"{group_id}_score": float(selection["score"]),
                f"{group_id}_min_rank": float(selection["min_marker_rank"]),
            }
            for marker in group["markers"]:
                rec[f"{marker}_top_mean"] = float(
                    selection.get("marker_top_1pct_mean", {}).get(marker, 0.0)
                )
                rec[f"{marker}_rank"] = float(
                    selection.get("marker_percentile_rank", {}).get(marker, 0.0)
                )
            loaded.append(rec)
        selections_by_group[group_id] = loaded

    for group in [GROUP_CD45, GROUP_CD68]:
        group_id = str(group["id"])
        for rec in selections_by_group[group_id]:
            row_records.append(
                {
                    **rec,
                    "selected_for": group_id,
                    "selected_for_label": group["label"],
                    "selected_score": float(rec[f"{group_id}_score"]),
                    "selected_min_rank": float(rec[f"{group_id}_min_rank"]),
                }
            )

    return row_records, selections_by_group, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("processed_crc33"))
    parser.add_argument("--markers-csv", type=Path, default=Path("data/markers.csv"))
    parser.add_argument("--mx-ome", type=Path, default=Path("data/mx_crc33.ome.tiff"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE1_FIGURE_DIR)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Existing selection manifest to reuse. Default: <out-dir>/<prefix>_selections.json",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recompute scores from raw MX crops instead of reusing --selection-json.",
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.01,
        help="Fraction of brightest raw-MX pixels used for each marker score.",
    )
    parser.add_argument(
        "--prefix",
        default="crc33_marker_high_patch_examples",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--mx-render-black-percentile",
        type=float,
        default=5.0,
        help="Raw MX render black floor percentile (default: 5; no background gate).",
    )
    parser.add_argument(
        "--mx-render-median-size",
        type=int,
        default=3,
        help="Median filter size for rendered MX RGB panels; <3 disables (default: 3).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Displayed panel size in output pixels. Increase for higher-res exports.",
    )
    parser.add_argument(
        "--panel-gap",
        type=int,
        default=6,
        help="Equal pixel gap between image panels (default: 6).",
    )
    parser.add_argument(
        "--header-height",
        type=int,
        default=32,
        help="Header band height in output pixels (default: 32).",
    )
    parser.add_argument(
        "--scale-bar-um",
        type=float,
        default=20.0,
        help="Scale bar length in microns; <=0 disables (default: 20).",
    )
    args = parser.parse_args()

    marker_map = _read_marker_indices(args.markers_csv)
    resolved_groups = []
    all_marker_indices: dict[str, int] = {}
    for group in FIGURE_GROUPS:
        indices = []
        resolved_names = []
        for marker in group["markers"]:
            idx, resolved_name = _resolve_marker(marker_map, marker)
            indices.append(idx)
            resolved_names.append(resolved_name)
            all_marker_indices[marker] = idx
        background_index, background_name = _resolve_marker(marker_map, "Hoechst")
        resolved_groups.append(
            {
                **group,
                "indices": indices,
                "resolved_markers": resolved_names,
                "background_index": background_index,
                "background_marker": background_name,
            }
        )

    patch_records, _, mpp = _load_patch_records(args.processed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"{args.prefix}_selections.json"
    selection_json = args.selection_json or manifest_path
    loaded_manifest: dict[str, Any] | None = None

    with tifffile.TiffFile(str(args.mx_ome)) as tif:
        img_w, img_h, axes = get_image_dims(tif)
        store = open_zarr_store(tif)
        if selection_json.exists() and not args.rescore:
            row_records, selections_by_group, loaded_manifest = (
                _load_rows_from_selection_json(selection_json, patch_records, int(args.n))
            )
            print(f"Loaded selections from: {selection_json}")
        else:
            records = _score_patches(
                patch_records,
                store,
                axes,
                img_h,
                img_w,
                all_marker_indices,
                top_fraction=float(args.top_fraction),
            )

            used: set[str] = set()
            selections_by_group = {}
            for group in SELECTION_GROUPS:
                selected = _select_top(records, group, n=int(args.n), used=used)
                selections_by_group[group["id"]] = selected

            row_records = []
            for group in [GROUP_CD45, GROUP_CD68]:
                for rec in selections_by_group[group["id"]]:
                    group_id = str(group["id"])
                    row_records.append(
                        {
                            **rec,
                            "selected_for": group_id,
                            "selected_for_label": group["label"],
                            "selected_score": float(rec[f"{group_id}_score"]),
                            "selected_min_rank": float(rec[f"{group_id}_min_rank"]),
                        }
                    )

        subgroup_outputs: dict[str, str] = {}
        subgroup_metabolic_outputs: dict[str, str] = {}
        for group in [GROUP_CD45, GROUP_CD68]:
            group_id = str(group["id"])
            rows = [rec for rec in row_records if rec["selected_for"] == group_id]
            subgroup_png = args.out_dir / f"{args.prefix}_{group_id}.png"
            _make_figure(
                args.processed,
                rows,
                resolved_groups,
                subgroup_png,
                title=f"CRC33 raw MX crops selected for {group['label']}",
                store=store,
                axes=axes,
                img_h=img_h,
                img_w=img_w,
                tile_size=int(args.tile_size),
                panel_gap=int(args.panel_gap),
                header_height=int(args.header_height),
                mpp=mpp,
                scale_bar_um=float(args.scale_bar_um),
                mx_black_percentile=float(args.mx_render_black_percentile),
                mx_median_size=int(args.mx_render_median_size),
            )
            subgroup_outputs[group_id] = str(subgroup_png)

            subgroup_metabolic_png = (
                args.out_dir / f"{args.prefix}_{group_id}_oxygen_glucose.png"
            )
            make_metabolic_gradient_tiles(
                args.processed,
                [str(rec["patch_id"]) for rec in rows],
                subgroup_metabolic_png,
                tile_size=int(args.tile_size),
                panel_gap=int(args.panel_gap),
                header_height=int(args.header_height),
                mpp=mpp,
                scale_bar_um=float(args.scale_bar_um),
            )
            subgroup_metabolic_outputs[group_id] = str(subgroup_metabolic_png)

    score_source = (
        f"loaded from existing selection JSON: {selection_json}"
        if loaded_manifest is not None and not args.rescore
        else "computed from raw OME-TIFF crops"
    )
    manifest = {
        "processed_dir": str(args.processed),
        "markers_csv": str(args.markers_csv),
        "mx_ome": str(args.mx_ome),
        "top_fraction": float(args.top_fraction),
        "score_source": score_source,
        "mx_render_source": (
            "raw OME-TIFF crops, not processed multiplex/*.npy; no Hoechst/cell "
            "background gate is applied during rendering"
        ),
        "mx_render_normalization": {
            "black_percentile": float(args.mx_render_black_percentile),
            "white_percentile": 99.8,
            "gamma": 0.8,
            "background_gate": "none",
            "median_filter_size": int(args.mx_render_median_size),
        },
        "figure_style": {
            "tile_size_px": int(args.tile_size),
            "panel_gap_px": int(args.panel_gap),
            "header_height_px": int(args.header_height),
            "scale_bar_um": float(args.scale_bar_um),
            "mpp": mpp,
            "title": "none",
            "row_descriptions": "none",
            "column_titles": "H&E centered text; MX marker labels in black header bands",
        },
        "score_definition": (
            loaded_manifest.get("score_definition")
            if loaded_manifest is not None
            else (
                "For each marker, crop the raw MX OME-TIFF at the patch coordinate, "
                "compute the mean of the brightest top_fraction pixels, and convert "
                "that marker statistic to a percentile rank over all paired patches. "
                "A group score is the mean percentile rank across the group's four "
                "markers; min_marker_rank records the weakest marker percentile."
            )
        ),
        "outputs": {
            "subgroup_png": subgroup_outputs,
            "subgroup_metabolic_png": subgroup_metabolic_outputs,
        },
        "figure_columns": [
            "H&E",
            resolved_groups[0]["label"],
            resolved_groups[1]["label"],
        ],
        "groups": {},
    }
    for group in [GROUP_CD68, GROUP_CD45]:
        resolved = next(g for g in resolved_groups if g["id"] == group["id"])
        manifest["groups"][group["id"]] = {
            "label": group["label"],
            "requested_markers": list(group["markers"]),
            "resolved_markers": list(resolved["resolved_markers"]),
            "channel_indices_zero_based": list(resolved["indices"]),
            "background_gate": "none",
            "selections": [
                _selection_payload(rec, group)
                for rec in selections_by_group[group["id"]]
            ],
        }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    for path in subgroup_outputs.values():
        print(f"Saved: {path}")
    for path in subgroup_metabolic_outputs.values():
        print(f"Saved: {path}")
    print(f"Saved: {manifest_path}")
    for group in [GROUP_CD68, GROUP_CD45]:
        patch_ids = [rec["patch_id"] for rec in selections_by_group[group["id"]]]
        print(f"{group['id']}: {patch_ids}")


if __name__ == "__main__":
    main()
