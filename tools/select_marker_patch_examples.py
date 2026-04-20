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

from tools.vis_he_mx_side_by_side import _composite_markers_rgb
from utils.marker_aliases import canonicalize_marker_name, normalize_marker_name
from utils.ome import get_image_dims, open_zarr_store


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

FIGURE_GROUPS = [GROUP_CD45, GROUP_CD68]
SELECTION_GROUPS = [GROUP_CD68, GROUP_CD45]
CHANNEL_AXES = ("C", "I", "S")


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/local/share/fonts") / name,
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


def _load_patch_records(processed_dir: Path) -> tuple[list[dict[str, Any]], int]:
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing patch index: {index_path}")
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patch_size = int(index.get("patch_size", 256))
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
    return records, patch_size


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
    mx_black_percentile: float,
    mx_median_size: int,
) -> Image.Image:
    tile = 256
    gap = 14
    label_w = 190
    header_h = 78
    row_h = tile + 18
    footer_h = 18
    width = label_w + 3 * tile + 2 * gap
    height = header_h + len(selected_rows) * row_h + footer_h

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(15, bold=True)
    header_font = _load_font(12, bold=True)
    small_font = _load_font(9)
    label_font = _load_font(10)

    draw.text((12, 10), title, font=title_font, fill=(25, 25, 25))
    x_cols = [label_w, label_w + tile + gap, label_w + 2 * (tile + gap)]
    headers = ["H&E", resolved_groups[0]["label"], resolved_groups[1]["label"]]
    for x, header in zip(x_cols, headers):
        draw.text((x, 36), header, font=header_font, fill=(25, 25, 25))
    for x, group in zip(x_cols[1:], resolved_groups):
        _draw_legend(draw, x, 57, group["markers"], group["colors_rgb"], small_font)

    he_dir = processed_dir / "he"
    for row_idx, rec in enumerate(selected_rows):
        y = header_h + row_idx * row_h
        patch_id = str(rec["patch_id"])
        source = str(rec["selected_for_label"])
        score = float(rec["selected_score"])
        min_rank = float(rec["selected_min_rank"])
        _draw_text_lines(
            draw,
            (12, y + 8),
            [
                f"Row {row_idx + 1}",
                patch_id,
                source,
                f"score={score:.3f}",
                f"min rank={min_rank:.3f}",
            ],
            label_font,
        )

        he_img = Image.open(he_dir / f"{patch_id}.png").convert("RGB")
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
        _paste_center(canvas, mx_imgs[0], x_cols[1], y, tile)
        _paste_center(canvas, mx_imgs[1], x_cols[2], y, tile)

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
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
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

    patch_records, _ = _load_patch_records(args.processed)
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
                mx_black_percentile=float(args.mx_render_black_percentile),
                mx_median_size=int(args.mx_render_median_size),
            )
            subgroup_outputs[group_id] = str(subgroup_png)

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
        "outputs": {"subgroup_png": subgroup_outputs},
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
    print(f"Saved: {manifest_path}")
    for group in [GROUP_CD68, GROUP_CD45]:
        patch_ids = [rec["patch_id"] for rec in selections_by_group[group["id"]]]
        print(f"{group['id']}: {patch_ids}")


if __name__ == "__main__":
    main()
