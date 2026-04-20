"""Side-by-side H&E and multiplex overview with selected marker colors.

Example:
  python -m tools.vis_he_mx_side_by_side \\
    --he data/he_crc33.ome.tif \\
    --mx data/mx_crc33.ome.tiff \\
    --markers-csv data/markers.csv \\
    --save-png paper/figures/he_mx_overview.png \\
    --downsample 64
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw

from utils.channels import resolve_channel_indices
from utils.ome import read_overview_chw


def _series_dims(series: tifffile.TiffPageSeries) -> tuple[int, int, str]:
    axes = series.axes.upper()
    shape = series.shape
    return int(shape[axes.index("X")]), int(shape[axes.index("Y")]), axes


def _open_series_store(series: tifffile.TiffPageSeries):
    try:
        raw = zarr.open(series.aszarr(), mode="r")
        if isinstance(raw, zarr.Array):
            return raw
        return raw["0"]
    except TypeError as exc:
        if "ZarrTiffStore" not in str(exc):
            raise
        warnings.warn(
            "tifffile.aszarr is unavailable; falling back to a memmapped TIFF read.",
            RuntimeWarning,
            stacklevel=2,
        )
        return series.asarray(out="memmap")


def _choose_pyramid_level(
    tif: tifffile.TiffFile, requested_downsample: int
) -> tuple[tifffile.TiffPageSeries, int, int, float]:
    """Return series level, index, residual stride, and effective downsample."""
    if requested_downsample < 1:
        raise ValueError("--downsample must be >= 1")

    base = tif.series[0]
    levels = list(getattr(base, "levels", []) or [base])
    base_w, base_h, _ = _series_dims(base)

    best_idx = 0
    best_factor = 1.0
    for idx, level in enumerate(levels):
        level_w, level_h, _ = _series_dims(level)
        factor = min(base_w / float(level_w), base_h / float(level_h))
        if factor <= requested_downsample and factor >= best_factor:
            best_idx = idx
            best_factor = factor

    residual = max(1, int(round(requested_downsample / best_factor)))
    effective = best_factor * residual
    return levels[best_idx], best_idx, residual, effective


def _read_fast_overview_chw(
    tif: tifffile.TiffFile,
    requested_downsample: int,
    channel_indices: list[int] | None = None,
) -> tuple[np.ndarray, int, float]:
    level, level_idx, residual, effective = _choose_pyramid_level(
        tif, requested_downsample
    )
    level_w, level_h, axes = _series_dims(level)
    chw = read_overview_chw(
        _open_series_store(level),
        axes,
        level_h,
        level_w,
        residual,
        channel_indices=channel_indices,
    )
    return chw, level_idx, effective


def _preserve_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to uint8 while keeping channel balance intact."""
    if rgb.dtype == np.uint8:
        return rgb

    arr = rgb.astype(np.float32, copy=False)
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.8))
    if hi <= lo:
        return np.zeros(rgb.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _marker_norm(
    arr: np.ndarray,
    black_percentile: float = 70.0,
    white_percentile: float = 99.8,
    gamma: float = 0.8,
) -> np.ndarray:
    """Normalize one MX marker with a high black floor for dark backgrounds."""
    arr_f = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(arr_f, black_percentile))
    hi = float(np.percentile(arr_f, white_percentile))
    if hi <= lo:
        hi = float(arr_f.max())
    if hi <= lo:
        return np.zeros(arr_f.shape, dtype=np.float32)
    norm = np.clip((arr_f - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma, dtype=np.float32)
    return norm.astype(np.float32)


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def _background_mask(
    background: np.ndarray,
    black_percentile: float = 65.0,
    dilate_px: int = 2,
) -> np.ndarray:
    threshold = float(np.percentile(background.astype(np.float32), black_percentile))
    if threshold >= float(np.max(background)):
        return np.ones(background.shape, dtype=bool)
    return _dilate_binary(background > threshold, dilate_px)


def _render_he_rgb(chw: np.ndarray) -> np.ndarray:
    """CHW -> uint8 RGB, preserving H&E white balance."""
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {chw.shape!r}")
    if chw.shape[0] >= 3:
        rgb = np.moveaxis(chw[:3], 0, -1)
    else:
        rgb = np.repeat(chw[0][..., np.newaxis], 3, axis=-1)
    return _preserve_rgb_uint8(rgb)


def _composite_markers_rgb(
    chw: np.ndarray,
    channel_indices: list[int],
    colors_rgb: np.ndarray,
    black_percentile: float = 70.0,
    white_percentile: float = 99.8,
    gamma: float = 0.8,
    background: np.ndarray | None = None,
    background_black_percentile: float = 65.0,
    background_dilate_px: int = 2,
) -> np.ndarray:
    """Blend normalized marker intensities onto RGB using additive color weights."""
    h, w = chw.shape[1], chw.shape[2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.float32).reshape(-1, 3)
    if len(channel_indices) != len(colors_rgb):
        raise ValueError("channel_indices and colors length mismatch")
    for idx, col in zip(channel_indices, colors_rgb):
        n = _marker_norm(
            chw[idx],
            black_percentile=black_percentile,
            white_percentile=white_percentile,
            gamma=gamma,
        )
        acc += n[..., np.newaxis] * col
    if background is not None:
        mask = _background_mask(
            background,
            black_percentile=background_black_percentile,
            dilate_px=background_dilate_px,
        )
        acc *= mask[..., np.newaxis]
    acc = np.clip(acc, 0.0, 1.0)
    return (acc * 255.0 + 0.5).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return canvas


def _load_selection_boxes(
    selection_json: Path,
    group_ids: set[str] | None = None,
) -> list[dict[str, int]]:
    if not selection_json.exists():
        warnings.warn(
            f"Selection JSON not found; no crop boxes drawn: {selection_json}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    with selection_json.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    boxes: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for group_id, group in payload.get("groups", {}).items():
        if group_ids is not None and str(group_id) not in group_ids:
            continue
        for sel in group.get("selections", []):
            he_x0 = int(sel.get("he_x0", str(sel["patch_id"]).split("_")[0]))
            he_y0 = int(sel.get("he_y0", str(sel["patch_id"]).split("_")[1]))
            mx_x0 = int(sel.get("mx_x0", he_x0))
            mx_y0 = int(sel.get("mx_y0", he_y0))
            mx_size = int(sel.get("mx_size", 256))
            he_size = int(sel.get("he_size", mx_size))
            key = (he_x0, he_y0, he_size, mx_x0, mx_y0, mx_size)
            if key in seen:
                continue
            seen.add(key)
            boxes.append(
                {
                    "group_id": str(group_id),
                    "he_x0": he_x0,
                    "he_y0": he_y0,
                    "he_size": he_size,
                    "mx_x0": mx_x0,
                    "mx_y0": mx_y0,
                    "mx_size": mx_size,
                }
            )
    return boxes


def _draw_crop_boxes(
    rgb: np.ndarray,
    boxes: list[dict[str, int]],
    *,
    base_w: int,
    base_h: int,
    x_key: str,
    y_key: str,
    size_key: str,
    min_box_px: int,
    line_width: int,
) -> np.ndarray:
    if not boxes:
        return rgb

    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    out_w, out_h = img.size

    def draw_callout_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        shaft_w_outer = max(8, int(min(out_w, out_h) / 48))
        shaft_w_inner = max(4, shaft_w_outer - 4)
        head_len = max(18, int(min(out_w, out_h) / 16))
        head_w_outer = max(12, shaft_w_outer * 2)
        head_w_inner = max(7, head_w_outer - 5)
        base_x = end[0] - ux * head_len
        base_y = end[1] - uy * head_len
        outer_head = [
            end,
            (base_x + px * head_w_outer, base_y + py * head_w_outer),
            (base_x - px * head_w_outer, base_y - py * head_w_outer),
        ]
        inner_head = [
            end,
            (base_x + px * head_w_inner, base_y + py * head_w_inner),
            (base_x - px * head_w_inner, base_y - py * head_w_inner),
        ]
        draw.line([start, end], fill=(0, 0, 0), width=shaft_w_outer)
        draw.polygon(outer_head, fill=(0, 0, 0))
        draw.line([start, end], fill=(255, 255, 255), width=shaft_w_inner)
        draw.polygon(inner_head, fill=(255, 255, 255))

    for box in boxes:
        x0 = float(box[x_key])
        y0 = float(box[y_key])
        size = float(box[size_key])
        cx = (x0 + size / 2.0) * out_w / float(base_w)
        cy = (y0 + size / 2.0) * out_h / float(base_h)
        box_w = size * out_w / float(base_w)
        box_h = size * out_h / float(base_h)
        side = max(float(min_box_px), box_w, box_h)
        left = int(round(cx - side / 2.0))
        top = int(round(cy - side / 2.0))
        right = int(round(cx + side / 2.0))
        bottom = int(round(cy + side / 2.0))
        left = max(0, min(out_w - 1, left))
        right = max(0, min(out_w - 1, right))
        top = max(0, min(out_h - 1, top))
        bottom = max(0, min(out_h - 1, bottom))
        box_rect = [left, top, right, bottom]
        draw.rectangle(box_rect, outline=(255, 255, 255), width=max(4, int(line_width) + 3))
        draw.rectangle(box_rect, outline=(255, 0, 0), width=max(2, int(line_width) + 1))

        span = max(36, int(min(out_w, out_h) / 5))
        target_x = left if cx >= out_w / 2.0 else right
        target_y = top if cy >= out_h / 2.0 else bottom
        start_x = target_x + (-span if cx >= out_w / 2.0 else span)
        start_y = target_y + (-span if cy >= out_h / 2.0 else span)
        start_x = float(max(12, min(out_w - 12, start_x)))
        start_y = float(max(12, min(out_h - 12, start_y)))
        end_x = float(max(6, min(out_w - 6, target_x)))
        end_y = float(max(6, min(out_h - 6, target_y)))
        draw_callout_arrow((start_x, start_y), (end_x, end_y))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--he", required=True, type=Path, help="H&E OME-TIFF path")
    p.add_argument("--mx", required=True, type=Path, help="Multiplex OME-TIFF path")
    p.add_argument(
        "--markers-csv",
        required=True,
        type=Path,
        help="Channel metadata CSV (Channel_Number, Marker_Name)",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=64,
        help="Target overview downsample (larger = smaller output)",
    )
    p.add_argument(
        "--mx-black-percentile",
        type=float,
        default=70.0,
        help="MX values at or below this percentile render black (default: 70)",
    )
    p.add_argument(
        "--mx-white-percentile",
        type=float,
        default=99.8,
        help="MX values at this percentile render full intensity (default: 99.8)",
    )
    p.add_argument(
        "--mx-gamma",
        type=float,
        default=0.8,
        help="MX display gamma after percentile scaling (default: 0.8)",
    )
    p.add_argument(
        "--mx-background-marker",
        default="Hoechst",
        help="Marker used to mask MX non-tissue background; empty string disables",
    )
    p.add_argument(
        "--mx-background-percentile",
        type=float,
        default=65.0,
        help="Background-marker percentile used as tissue gate (default: 65)",
    )
    p.add_argument(
        "--mx-background-dilate",
        type=int,
        default=2,
        help="Dilate MX tissue gate by this many overview pixels (default: 2)",
    )
    p.add_argument(
        "--save-png",
        "--out",
        type=Path,
        default=None,
        dest="save_png",
        help="Output PNG path",
    )
    p.add_argument(
        "--selection-json",
        type=Path,
        default=Path("paper/figures/crc33_marker_high_patch_examples_selections.json"),
        help="Selection manifest whose crop regions are boxed on both WSI panels.",
    )
    p.add_argument(
        "--selection-group",
        action="append",
        default=None,
        help="Only draw boxes for this selection group. May be supplied more than once.",
    )
    p.add_argument(
        "--no-crop-boxes",
        action="store_true",
        help="Do not draw crop-region boxes from --selection-json.",
    )
    p.add_argument(
        "--crop-box-min-px",
        type=int,
        default=12,
        help="Minimum displayed crop-box side length in overview pixels.",
    )
    p.add_argument(
        "--crop-box-width",
        type=int,
        default=2,
        help="Crop-box outline width in overview pixels.",
    )
    args = p.parse_args()

    names = ["CD45", "CD31", "Pan-CK", "SMA"]
    indices, resolved = resolve_channel_indices(str(args.markers_csv), names)
    mx_channel_indices = list(indices)
    background_index: int | None = None
    background_resolved: str | None = None
    background_marker = str(args.mx_background_marker).strip()
    if background_marker:
        try:
            bg_indices, bg_resolved = resolve_channel_indices(
                str(args.markers_csv), [background_marker]
            )
            background_index = int(bg_indices[0])
            background_resolved = bg_resolved[0]
        except ValueError as exc:
            warnings.warn(
                f"Cannot resolve MX background marker {background_marker!r}; "
                f"continuing without background masking. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    # Colors aligned with *names* order: CD45 purple, CD31 yellow, PanCK green, SMA red
    colors = np.array(
        [
            [160, 32, 240],  # purple (approx orchid/darkorchid)
            [255, 220, 0],  # yellow
            [0, 220, 90],  # green
            [255, 40, 40],  # red
        ],
        dtype=np.float32,
    ) / 255.0

    with tifffile.TiffFile(str(args.he)) as tif:
        he_base_w, he_base_h, _ = _series_dims(tif.series[0])
        he_chw, he_level, he_effective_ds = _read_fast_overview_chw(
            tif, args.downsample
        )
    he_rgb = _render_he_rgb(he_chw)

    with tifffile.TiffFile(str(args.mx)) as tif:
        mx_base_w, mx_base_h, _ = _series_dims(tif.series[0])
        read_indices = (
            [background_index] + mx_channel_indices
            if background_index is not None
            else mx_channel_indices
        )
        mx_chw, mx_level, mx_effective_ds = _read_fast_overview_chw(
            tif,
            args.downsample,
            channel_indices=read_indices,
        )
    if background_index is not None:
        mx_background = mx_chw[0]
        mx_marker_chw = mx_chw[1:]
    else:
        mx_background = None
        mx_marker_chw = mx_chw

    mx_rgb = _composite_markers_rgb(
        mx_marker_chw,
        list(range(len(mx_channel_indices))),
        colors,
        black_percentile=args.mx_black_percentile,
        white_percentile=args.mx_white_percentile,
        gamma=args.mx_gamma,
        background=mx_background,
        background_black_percentile=args.mx_background_percentile,
        background_dilate_px=args.mx_background_dilate,
    )

    crop_groups = (
        {str(group_id) for group_id in args.selection_group}
        if args.selection_group
        else None
    )
    crop_boxes = (
        []
        if args.no_crop_boxes
        else _load_selection_boxes(args.selection_json, group_ids=crop_groups)
    )
    he_rgb = _draw_crop_boxes(
        he_rgb,
        crop_boxes,
        base_w=he_base_w,
        base_h=he_base_h,
        x_key="he_x0",
        y_key="he_y0",
        size_key="he_size",
        min_box_px=args.crop_box_min_px,
        line_width=args.crop_box_width,
    )
    mx_rgb = _draw_crop_boxes(
        mx_rgb,
        crop_boxes,
        base_w=mx_base_w,
        base_h=mx_base_h,
        x_key="mx_x0",
        y_key="mx_y0",
        size_key="mx_size",
        min_box_px=args.crop_box_min_px,
        line_width=args.crop_box_width,
    )
    combo = side_by_side(he_rgb, mx_rgb)

    out = args.save_png
    if out is None:
        out = Path("paper/figures") / f"{args.he.stem}_vs_{args.mx.stem}_markers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combo).save(str(out))

    legend = "  ".join(f"{n}:{r}" for n, r in zip(names, resolved))
    print(f"Saved {out}  ({combo.shape[1]}x{combo.shape[0]})")
    print(f"Channels - {legend}")
    if background_resolved is not None:
        print(f"MX background gate - {background_resolved}")
    if crop_boxes:
        print(f"Crop boxes - {len(crop_boxes)} from {args.selection_json}")
    print(
        f"Downsample target={args.downsample}; "
        f"HE level={he_level} effective~{he_effective_ds:.2f}; "
        f"MX level={mx_level} effective~{mx_effective_ds:.2f}"
    )


if __name__ == "__main__":
    main()
